"""
block_detector.py
==================
캡처된 프레임에서 현재 대기 중인 3개의 블록(피스)을 인식하여
각각 2차원 0/1 배열로 변환한다.

인식 절차
---------
1. config.tray.slot_rects 에 정의된 3개 영역을 각각 크롭한다.
2. ROI 가장자리(배경)의 채도 분포(median/MAD)로 Adaptive Threshold 를 계산하고,
   HSV 채도(saturation) 기준으로 배경과 블록 픽셀을 구분하는 마스크를 만든다.
3. 마스크의 바운딩 박스를 구해 블록이 존재하는지(빈 슬롯 여부) 판단한다.
4. (Grid Detection) 보드 셀 크기 x tray.piece_cell_scale 로 추정한 셀 픽셀
   크기를 이용해 바운딩 박스를 NxM 그리드로 나누고, 각 서브셀의 안쪽(margin)
   영역의 마스크 비율/평균 색상으로 0/1 (Occupied) 을 결정한다.
5. (Template Matching) 위에서 얻은 occupancy grid 를 shape_templates 에
   미리 정의된 모든 블록 모양과 비교(Jaccard 유사도)하여 가장 유사한
   템플릿을 선택하고, 그 템플릿의 Shape Matrix 를 Solver 입력으로 사용한다.
   Bounding Box 를 그대로 trim 한 결과를 사용하지 않으므로, 노이즈로 인한
   픽셀 한두 개의 오차가 Shape 자체를 바꾸지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import AppConfig, CONFIG
from logger import get_logger
from shape_templates import match_template
from utils import COLS, PieceCells, ROWS

logger = get_logger("block_detector")


@dataclass
class DetectedPiece:
    grid: np.ndarray            # 0/1 2차원 배열 (Template Matching 결과, Solver 입력용)
    cells: PieceCells            # 정규화된 (dr, dc) 셀 목록 (Template Matching 결과)
    bbox: Tuple[int, int, int, int]  # 슬롯 내부 기준 바운딩 박스 (x, y, w, h)
    empty: bool                  # 슬롯이 비어있는지 여부

    # --- 디버깅용 추가 정보 ---
    debug_grid: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int8))
    # bbox 를 grid_rows x grid_cols 로 나눈 (Template Matching 이전) 0/1 배열 (Grid Detection 결과)
    debug_cells: List[Tuple[int, int, Tuple[int, int, int], bool]] = field(default_factory=list)
    # (row, col, (R, G, B) 평균색, occupied) 목록 (debug_grid 와 동일한 좌표계)
    threshold_used: float = 0.0  # 이 슬롯에 사용된 Adaptive Saturation Threshold
    template_name: str = ""      # Template Matching 에서 선택된 템플릿 이름
    template_similarity: float = 0.0  # 선택된 템플릿과의 Jaccard 유사도 (0~1)


class BlockDetector:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or CONFIG

    # ------------------------------------------------------------------
    def _board_cell_size(self) -> Tuple[float, float]:
        _, _, w, h = self.config.board.board_rect
        if w <= 0 or h <= 0:
            # 캘리브레이션 전: 적당한 기본값
            return 40.0, 40.0
        return w / COLS, h / ROWS

    # ------------------------------------------------------------------
    def _adaptive_threshold(self, sat: np.ndarray) -> float:
        """ROI 가장자리(배경으로 추정되는 영역)의 채도 분포를 기반으로
        이 ROI 에 맞는 채도 임계값을 계산한다.

        고정 임계값 대신 ROI 마다 배경 밝기에 맞춰 임계값을 조정하면,
        트레이 배경색이 약간 달라지거나 캡처 노이즈로 밝기가 흔들려도
        Occupied 판정이 흔들리지 않는다.

        평균/표준편차 대신 중앙값(median) + MAD(중앙값 절대편차) 를 사용해,
        블록이 ROI 가장자리 일부를 침범하더라도(배경 픽셀이 과반수인 한)
        임계값이 한쪽으로 크게 치우치지 않도록 한다.
        """
        fallback = self.config.tray.empty_saturation_threshold
        h, w = sat.shape[:2]
        if h < 6 or w < 6:
            return fallback

        bh, bw = max(1, h // 6), max(1, w // 6)
        border = np.concatenate(
            [
                sat[:bh, :].flatten(),
                sat[-bh:, :].flatten(),
                sat[:, :bw].flatten(),
                sat[:, -bw:].flatten(),
            ]
        ).astype(np.float64)

        bg_median = float(np.median(border))
        mad = float(np.median(np.abs(border - bg_median)))
        threshold = bg_median + max(10.0, mad * 4.0)
        return min(threshold, 250.0)

    # ------------------------------------------------------------------
    def _foreground_mask(self, slot_img: np.ndarray) -> Tuple[np.ndarray, float]:
        """슬롯 이미지에서 배경이 아닌(채도가 높은) 픽셀의 이진 마스크와,
        이 ROI 에 사용된 Adaptive Threshold 값을 반환한다.

        Threshold 근처의 작은 흔들림(프레임 노이즈)이 셀 판정을 매번
        뒤집지 않도록, 마스크 생성 전 Median Blur 로 노이즈를 줄이고
        Morphology Open/Close 를 더 강하게 적용한다.
        """
        blurred = slot_img
        if slot_img.shape[0] >= 5 and slot_img.shape[1] >= 5:
            blurred = cv2.medianBlur(slot_img, 5)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        threshold = self._adaptive_threshold(sat)
        mask = (sat > threshold).astype(np.uint8) * 255

        # 노이즈 제거: Open(작은 점 제거) -> Close(작은 구멍 메움, 2회)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask, threshold

    # ------------------------------------------------------------------
    def detect_slot(self, frame: np.ndarray, slot_rect: Tuple[int, int, int, int]) -> DetectedPiece:
        """단일 트레이 슬롯에서 블록 모양을 인식한다."""
        x, y, w, h = slot_rect
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        if w <= 0 or h <= 0:
            return DetectedPiece(grid=np.zeros((1, 1), dtype=np.int8), cells=tuple(), bbox=(0, 0, 0, 0), empty=True)

        slot_img = frame[y : y + h, x : x + w]
        mask, threshold_used = self._foreground_mask(slot_img)

        coords = cv2.findNonZero(mask)
        if coords is None:
            return DetectedPiece(
                grid=np.zeros((1, 1), dtype=np.int8), cells=tuple(), bbox=(0, 0, 0, 0), empty=True,
                threshold_used=threshold_used,
            )

        bx, by, bw, bh = cv2.boundingRect(coords)
        if bw < 4 or bh < 4:
            return DetectedPiece(
                grid=np.zeros((1, 1), dtype=np.int8), cells=tuple(), bbox=(0, 0, 0, 0), empty=True,
                threshold_used=threshold_used,
            )

        cell_w, cell_h = self._board_cell_size()
        piece_cell_w = max(1.0, cell_w * self.config.tray.piece_cell_scale)
        piece_cell_h = max(1.0, cell_h * self.config.tray.piece_cell_scale)

        max_size = self.config.tray.max_piece_size
        grid_cols = int(round(bw / piece_cell_w))
        grid_rows = int(round(bh / piece_cell_h))
        grid_cols = max(1, min(max_size, grid_cols))
        grid_rows = max(1, min(max_size, grid_rows))

        sub_w = bw / grid_cols
        sub_h = bh / grid_rows

        debug_grid = np.zeros((grid_rows, grid_cols), dtype=np.int8)
        debug_cells: List[Tuple[int, int, Tuple[int, int, int], bool]] = []
        bbox_mask = mask[by : by + bh, bx : bx + bw]
        bbox_img = slot_img[by : by + bh, bx : bx + bw]

        # 서브셀 가장자리는 인접한 채워진 셀의 픽셀이 번져 들어오기 쉬우므로,
        # 각 서브셀의 안쪽(margin 만큼 줄인) 영역만으로 채움 비율/평균 색상을
        # 계산한다. 이렇게 하지 않으면 L자/T자 등 빈 칸이 있는 모양이 꽉 찬
        # 사각형으로 인식되는 문제가 발생한다.
        margin_ratio = 0.2
        fill_ratio_threshold = 0.35
        for r in range(grid_rows):
            for c in range(grid_cols):
                sx0 = int(c * sub_w)
                sy0 = int(r * sub_h)
                sx1 = int((c + 1) * sub_w)
                sy1 = int((r + 1) * sub_h)

                mx = int((sx1 - sx0) * margin_ratio)
                my = int((sy1 - sy0) * margin_ratio)
                inner_mask = bbox_mask[sy0 + my : sy1 - my, sx0 + mx : sx1 - mx]
                inner_img = bbox_img[sy0 + my : sy1 - my, sx0 + mx : sx1 - mx]
                if inner_mask.size == 0:
                    inner_mask = bbox_mask[sy0:sy1, sx0:sx1]
                    inner_img = bbox_img[sy0:sy1, sx0:sx1]

                occupied = False
                if inner_mask.size > 0:
                    fill_ratio = np.count_nonzero(inner_mask) / inner_mask.size
                    occupied = fill_ratio >= fill_ratio_threshold

                if inner_img.size > 0:
                    mean_bgr = inner_img.reshape(-1, 3).mean(axis=0)
                    rgb = (int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0]))
                else:
                    rgb = (0, 0, 0)

                debug_grid[r, c] = 1 if occupied else 0
                debug_cells.append((r, c, rgb, occupied))

        trimmed = self._trim_grid(debug_grid)
        if trimmed.sum() == 0:
            return DetectedPiece(
                grid=np.zeros((1, 1), dtype=np.int8), cells=tuple(), bbox=(0, 0, 0, 0), empty=True,
                debug_grid=debug_grid, debug_cells=debug_cells, threshold_used=threshold_used,
            )

        # Bounding Box 기반 추정 대신, occupancy grid 를 표준 Shape Template 들과
        # 비교하여 가장 유사한 모양을 Solver 입력으로 사용한다 (Template Matching).
        template, similarity = match_template(trimmed)
        return DetectedPiece(
            grid=template.grid, cells=template.cells, bbox=(bx, by, bw, bh), empty=False,
            debug_grid=debug_grid, debug_cells=debug_cells, threshold_used=threshold_used,
            template_name=template.name, template_similarity=similarity,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _trim_grid(grid: np.ndarray) -> np.ndarray:
        """grid 에서 occupied(1) 셀들의 최소 바운딩 박스만 남기고 잘라낸다.

        margin 적용으로 가장자리 행/열이 전부 비어 있게 되는 경우,
        결과 모양(Shape)이 실제 블록 모양과 정확히 일치하도록 한다.
        """
        if grid.sum() == 0:
            return grid
        rows = np.any(grid, axis=1)
        cols = np.any(grid, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        return grid[r0 : r1 + 1, c0 : c1 + 1]

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[DetectedPiece]:
        """config.tray.slot_rects 에 정의된 모든 슬롯을 인식한다."""
        results = []
        for slot_rect in self.config.tray.slot_rects:
            results.append(self.detect_slot(frame, slot_rect))
        return results

    # ------------------------------------------------------------------
    def detect_pieces_cells(self, frame: np.ndarray) -> List[PieceCells]:
        """솔버에 바로 넣을 수 있는 PieceCells 리스트(빈 슬롯은 빈 tuple)."""
        return [d.cells for d in self.detect(frame)]

    # ------------------------------------------------------------------
    def auto_detect_tray_slots(
        self, frame: np.ndarray, search_rect: Tuple[int, int, int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        search_rect (보통 보드 아래쪽 영역) 안에서 채도가 높은 덩어리(blob)를
        최대 3개 찾아 슬롯 영역으로 추정한다. 좌->우 순서로 정렬하여 반환한다.
        """
        x, y, w, h = search_rect
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        if w <= 0 or h <= 0:
            return []

        region = frame[y : y + h, x : x + w]
        mask, _ = self._foreground_mask(region)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw < 10 or bh < 10:
                continue
            boxes.append((bx, by, bw, bh))

        # 면적 기준 상위 3개 선택 후 좌->우 정렬
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[: self.config.tray.slot_count]
        boxes = sorted(boxes, key=lambda b: b[0])

        slots = []
        for bx, by, bw, bh in boxes:
            # 약간의 여백을 둔다
            pad_x = int(bw * 0.15)
            pad_y = int(bh * 0.15)
            slots.append(
                (
                    x + max(0, bx - pad_x),
                    y + max(0, by - pad_y),
                    bw + pad_x * 2,
                    bh + pad_y * 2,
                )
            )

        # 슬롯 수가 부족하면 빈 영역으로 채움
        while len(slots) < self.config.tray.slot_count:
            slots.append((0, 0, 0, 0))

        return slots

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_cell_grid(img: np.ndarray, ox: int, oy: int, bw: int, bh: int, debug_grid: np.ndarray) -> None:
        """img 위 (ox, oy) ~ (ox+bw, oy+bh) 영역을 debug_grid 모양대로 나누어
        Occupied 셀(노란색 반투명)/빈 셀(회색 외곽선)을 그린다 (in-place)."""
        grid_rows, grid_cols = debug_grid.shape
        if grid_rows == 0 or grid_cols == 0 or bw <= 0 or bh <= 0:
            return

        sub_w = bw / grid_cols
        sub_h = bh / grid_rows
        for r in range(grid_rows):
            for c in range(grid_cols):
                cx0 = int(ox + c * sub_w)
                cy0 = int(oy + r * sub_h)
                cx1 = int(ox + (c + 1) * sub_w)
                cy1 = int(oy + (r + 1) * sub_h)
                if debug_grid[r, c]:
                    cell_overlay = img.copy()
                    cv2.rectangle(cell_overlay, (cx0, cy0), (cx1, cy1), (0, 255, 255), -1)
                    cv2.addWeighted(cell_overlay, 0.4, img, 0.6, 0, img)
                    cv2.rectangle(img, (cx0, cy0), (cx1, cy1), (0, 255, 255), 1)
                else:
                    cv2.rectangle(img, (cx0, cy0), (cx1, cy1), (120, 120, 120), 1)

    # ------------------------------------------------------------------
    def draw_debug_overlay(self, frame: np.ndarray, detections: List[DetectedPiece]) -> np.ndarray:
        """슬롯 사각형 + 인식된 블록의 셀 단위 Grid(■=채움, □=빈칸) +
        Solver 에 전달되는 Shape Matrix 텍스트를 그린다."""
        out = frame.copy()
        for slot_rect, det in zip(self.config.tray.slot_rects, detections):
            x, y, w, h = slot_rect
            if w <= 0 or h <= 0:
                continue
            color = (0, 255, 0) if not det.empty else (0, 0, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            if det.empty or det.debug_grid.size == 0:
                continue

            bx, by, bw, bh = det.bbox
            ox, oy = x + bx, y + by
            cv2.rectangle(out, (ox, oy), (ox + bw, oy + bh), (255, 0, 255), 1)
            self._draw_cell_grid(out, ox, oy, bw, bh, det.debug_grid)

            # Template Matching 결과 + Solver 로 전달되는 Shape Matrix
            cv2.putText(
                out,
                f"{det.template_name} ({det.template_similarity:.2f})",
                (x, y + h + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            for i, row in enumerate(det.grid.tolist()):
                cv2.putText(
                    out,
                    str(row),
                    (x, y + h + 28 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        return out

    # ------------------------------------------------------------------
    def render_roi_debug(self, frame: np.ndarray, slot_rect: Tuple[int, int, int, int], det: DetectedPiece,
                          target_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
        """단일 블록 ROI를 잘라내어 셀 그리드를 그린 뒤 target_size 로 확대한 이미지를 반환한다.

        GUI 의 "Block N ROI" 미리보기 창에 사용한다.
        """
        x, y, w, h = slot_rect
        if w <= 0 or h <= 0:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        if w <= 0 or h <= 0:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        roi = frame[y : y + h, x : x + w].copy()
        if not det.empty and det.debug_grid.size > 0:
            bx, by, bw, bh = det.bbox
            cv2.rectangle(roi, (bx, by), (bx + bw, by + bh), (255, 0, 255), 1)
            self._draw_cell_grid(roi, bx, by, bw, bh, det.debug_grid)

        return cv2.resize(roi, target_size, interpolation=cv2.INTER_NEAREST)

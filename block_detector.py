"""
block_detector.py
==================
캡처된 프레임에서 현재 대기 중인 3개의 블록(피스)을 인식하여
각각 2차원 0/1 배열로 변환한다.

회전은 지원하지 않으므로, 인식된 모양 그대로 사용한다.

인식 절차
---------
1. config.tray.slot_rects 에 정의된 3개 영역을 각각 크롭한다.
2. HSV 채도(saturation) 기준으로 배경과 블록 픽셀을 구분하는 마스크를 만든다.
3. 마스크의 바운딩 박스를 구해 블록이 존재하는지(빈 슬롯 여부) 판단한다.
4. 보드 셀 크기 x tray.piece_cell_scale 로 추정한 셀 픽셀 크기를 이용해
   바운딩 박스를 NxM 그리드로 나누고, 각 서브셀의 마스크 비율로 0/1 을 결정한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import AppConfig, CONFIG
from logger import get_logger
from utils import COLS, PieceCells, ROWS, piece_grid_to_cells

logger = get_logger("block_detector")


@dataclass
class DetectedPiece:
    grid: np.ndarray            # 0/1 2차원 배열 (최소 바운딩 박스 크기)
    cells: PieceCells            # 정규화된 (dr, dc) 셀 목록
    bbox: Tuple[int, int, int, int]  # 슬롯 내부 기준 바운딩 박스 (x, y, w, h)
    empty: bool                  # 슬롯이 비어있는지 여부


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
    def _foreground_mask(self, slot_img: np.ndarray) -> np.ndarray:
        """슬롯 이미지에서 배경이 아닌(채도가 높은) 픽셀의 이진 마스크를 반환.

        Threshold 근처의 작은 흔들림(프레임 노이즈)이 셀 판정을 매번
        뒤집지 않도록, 마스크 생성 전 Median Blur 로 노이즈를 줄이고
        Morphology Open/Close 를 더 강하게 적용한다.
        """
        blurred = slot_img
        if slot_img.shape[0] >= 5 and slot_img.shape[1] >= 5:
            blurred = cv2.medianBlur(slot_img, 5)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        threshold = self.config.tray.empty_saturation_threshold
        mask = (sat > threshold).astype(np.uint8) * 255

        # 노이즈 제거: Open(작은 점 제거) -> Close(작은 구멍 메움, 2회)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

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
        mask = self._foreground_mask(slot_img)

        coords = cv2.findNonZero(mask)
        if coords is None:
            return DetectedPiece(grid=np.zeros((1, 1), dtype=np.int8), cells=tuple(), bbox=(0, 0, 0, 0), empty=True)

        bx, by, bw, bh = cv2.boundingRect(coords)
        if bw < 4 or bh < 4:
            return DetectedPiece(grid=np.zeros((1, 1), dtype=np.int8), cells=tuple(), bbox=(0, 0, 0, 0), empty=True)

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

        grid = np.zeros((grid_rows, grid_cols), dtype=np.int8)
        bbox_mask = mask[by : by + bh, bx : bx + bw]

        # 서브셀 가장자리는 인접한 채워진 셀의 픽셀이 번져 들어오기 쉬우므로,
        # 각 서브셀의 안쪽(margin 만큼 줄인) 영역만으로 채움 비율을 계산한다.
        # 이렇게 하지 않으면 L자/T자 등 빈 칸이 있는 모양이 꽉 찬 사각형으로
        # 인식되는 문제가 발생한다.
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
                inner = bbox_mask[sy0 + my : sy1 - my, sx0 + mx : sx1 - mx]
                if inner.size == 0:
                    inner = bbox_mask[sy0:sy1, sx0:sx1]
                if inner.size == 0:
                    continue

                fill_ratio = np.count_nonzero(inner) / inner.size
                if fill_ratio >= fill_ratio_threshold:
                    grid[r, c] = 1

        grid = self._trim_grid(grid)
        if grid.sum() == 0:
            return DetectedPiece(grid=np.zeros((1, 1), dtype=np.int8), cells=tuple(), bbox=(0, 0, 0, 0), empty=True)

        cells = piece_grid_to_cells(grid.tolist())
        return DetectedPiece(grid=grid, cells=cells, bbox=(bx, by, bw, bh), empty=False)

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
        mask = self._foreground_mask(region)

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
    def draw_debug_overlay(self, frame: np.ndarray, detections: List[DetectedPiece]) -> np.ndarray:
        """슬롯 사각형 + 인식된 블록의 셀 단위 Grid(■=채움, □=빈칸)를 그린다."""
        out = frame.copy()
        for slot_rect, det in zip(self.config.tray.slot_rects, detections):
            x, y, w, h = slot_rect
            if w <= 0 or h <= 0:
                continue
            color = (0, 255, 0) if not det.empty else (0, 0, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            if det.empty:
                continue

            bx, by, bw, bh = det.bbox
            ox, oy = x + bx, y + by
            cv2.rectangle(out, (ox, oy), (ox + bw, oy + bh), (255, 0, 255), 1)

            grid_rows, grid_cols = det.grid.shape
            if grid_rows == 0 or grid_cols == 0 or bw <= 0 or bh <= 0:
                continue

            sub_w = bw / grid_cols
            sub_h = bh / grid_rows
            for r in range(grid_rows):
                for c in range(grid_cols):
                    cx0 = int(ox + c * sub_w)
                    cy0 = int(oy + r * sub_h)
                    cx1 = int(ox + (c + 1) * sub_w)
                    cy1 = int(oy + (r + 1) * sub_h)
                    if det.grid[r, c]:
                        cell_overlay = out.copy()
                        cv2.rectangle(cell_overlay, (cx0, cy0), (cx1, cy1), (0, 255, 255), -1)
                        cv2.addWeighted(cell_overlay, 0.4, out, 0.6, 0, out)
                        cv2.rectangle(out, (cx0, cy0), (cx1, cy1), (0, 255, 255), 1)
                    else:
                        cv2.rectangle(out, (cx0, cy0), (cx1, cy1), (120, 120, 120), 1)
        return out

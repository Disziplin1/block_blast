"""
board_detector.py
==================
캡처된 프레임에서 8x8 게임 보드를 인식하여 0/1 배열로 변환한다.

크게 두 가지 기능을 제공한다.

1. 자동 캘리브레이션 (calibrate)
   - 화면에서 보드로 추정되는 정사각형 영역을 자동 탐지하여
     config.board.board_rect 를 갱신한다.
   - 보드가 비어 있는 상태를 기준으로 체커보드 배경의 채도(saturation)
     기준값을 측정해 둔다 (이후 '블록 있음/없음' 판정 정확도 향상).

2. 보드 인식 (detect)
   - board_rect 영역을 8x8 셀로 분할하고, 각 셀의 채도/명도를 분석하여
     0(빈칸) / 1(블록) 배열을 생성한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import AppConfig, CONFIG
from logger import get_logger
from utils import COLS, ROWS

logger = get_logger("board_detector")


@dataclass
class BoardRectCandidate:
    rect: Tuple[int, int, int, int]  # x, y, w, h
    score: float


class BoardDetector:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or CONFIG

    # ------------------------------------------------------------------
    # 자동 보드 영역 탐지
    # ------------------------------------------------------------------
    def auto_detect_board_rect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        프레임에서 정사각형에 가까운 가장 큰 윤곽선을 보드 영역으로 추정한다.

        Block Blast 의 보드는 화면에서 가로/세로 비율이 1:1에 가까운
        큰 사각형 영역으로 나타난다는 가정을 사용한다.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        h_frame, w_frame = frame.shape[:2]
        frame_area = h_frame * w_frame

        candidates: List[BoardRectCandidate] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 50 or h < 50:
                continue
            area = w * h
            if area < frame_area * 0.05 or area > frame_area * 0.95:
                continue
            aspect = w / float(h)
            if not (0.85 <= aspect <= 1.15):
                continue
            # 정사각형에 가깝고 면적이 클수록 높은 점수
            squareness = 1.0 - abs(1.0 - aspect)
            score = squareness * area
            candidates.append(BoardRectCandidate((x, y, w, h), score))

        if not candidates:
            logger.warning("auto_detect_board_rect: no candidate found")
            return None

        best = max(candidates, key=lambda c: c.score)
        logger.info("auto_detect_board_rect: found rect=%s score=%.1f", best.rect, best.score)
        return best.rect

    # ------------------------------------------------------------------
    # 캘리브레이션
    # ------------------------------------------------------------------
    def calibrate(self, frame: np.ndarray, board_rect: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """
        보드 영역을 확정하고, 현재(비어 있다고 가정하는) 보드의 체커보드
        채도 기준값을 측정하여 config 에 저장한다.

        board_rect 를 명시적으로 주면(GUI에서 사용자가 드래그로 지정한 경우)
        자동 탐지를 건너뛴다.
        """
        if board_rect is None:
            board_rect = self.auto_detect_board_rect(frame)
            if board_rect is None:
                return False

        self.config.board.board_rect = board_rect

        cells = self._iter_cells(frame, board_rect)
        even_sats: List[float] = []
        odd_sats: List[float] = []
        for r, c, cell_img in cells:
            sat = self._cell_saturation(cell_img)
            if (r + c) % 2 == 0:
                even_sats.append(sat)
            else:
                odd_sats.append(sat)

        baseline_even = float(np.mean(even_sats)) if even_sats else 0.0
        baseline_odd = float(np.mean(odd_sats)) if odd_sats else 0.0

        self.config.board.empty_saturation_baseline = (baseline_even, baseline_odd)
        self.config.board.calibrated = True

        logger.info(
            "Board calibrated: rect=%s baseline=(%.1f, %.1f)",
            board_rect,
            baseline_even,
            baseline_odd,
        )
        return True

    # ------------------------------------------------------------------
    # 셀 분할 / 분석 헬퍼
    # ------------------------------------------------------------------
    def _iter_cells(
        self, frame: np.ndarray, board_rect: Tuple[int, int, int, int]
    ):
        """board_rect 내부를 ROWSxCOLS 로 분할하여 (r, c, cell_image) 를 yield."""
        x, y, w, h = board_rect
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        if w <= 0 or h <= 0:
            return

        board_img = frame[y : y + h, x : x + w]
        cell_w = w / COLS
        cell_h = h / ROWS

        # 그리드 라인을 피하기 위해 셀 내부 60% 영역만 사용
        margin_ratio = 0.2

        for r in range(ROWS):
            for c in range(COLS):
                cx0 = int(c * cell_w)
                cy0 = int(r * cell_h)
                cx1 = int((c + 1) * cell_w)
                cy1 = int((r + 1) * cell_h)

                mx = int((cx1 - cx0) * margin_ratio)
                my = int((cy1 - cy0) * margin_ratio)

                inner = board_img[cy0 + my : cy1 - my, cx0 + mx : cx1 - mx]
                if inner.size == 0:
                    inner = board_img[cy0:cy1, cx0:cx1]
                yield r, c, inner

    @staticmethod
    def _cell_saturation(cell_img: np.ndarray) -> float:
        if cell_img.size == 0:
            return 0.0
        hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 1]))

    @staticmethod
    def _cell_value_std(cell_img: np.ndarray) -> float:
        if cell_img.size == 0:
            return 0.0
        hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
        return float(np.std(hsv[:, :, 2]))

    # ------------------------------------------------------------------
    # 보드 인식
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        현재 프레임에서 8x8 보드 상태를 인식한다.

        Returns
        -------
        np.ndarray, shape=(8,8), dtype=int8, 값 0(빈칸)/1(블록)
        """
        board_rect = self.config.board.board_rect
        if board_rect == (0, 0, 0, 0):
            logger.warning("detect(): board_rect not calibrated yet")
            return np.zeros((ROWS, COLS), dtype=np.int8)

        grid = np.zeros((ROWS, COLS), dtype=np.int8)
        baseline_even, baseline_odd = self.config.board.empty_saturation_baseline
        threshold = self.config.board.empty_saturation_threshold

        for r, c, cell_img in self._iter_cells(frame, board_rect):
            sat = self._cell_saturation(cell_img)

            if self.config.board.calibrated:
                baseline = baseline_even if (r + c) % 2 == 0 else baseline_odd
                is_filled = sat > baseline + threshold
            else:
                is_filled = sat > threshold

            grid[r, c] = 1 if is_filled else 0

        return grid

    # ------------------------------------------------------------------
    def draw_debug_overlay(self, frame: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """디버그용: 인식된 보드 그리드를 프레임 위에 그려서 반환한다."""
        out = frame.copy()
        board_rect = self.config.board.board_rect
        if board_rect == (0, 0, 0, 0):
            return out

        x, y, w, h = board_rect
        cell_w = w / COLS
        cell_h = h / ROWS

        for r in range(ROWS):
            for c in range(COLS):
                cx0 = int(x + c * cell_w)
                cy0 = int(y + r * cell_h)
                cx1 = int(x + (c + 1) * cell_w)
                cy1 = int(y + (r + 1) * cell_h)
                color = (0, 0, 255) if grid[r, c] else (0, 255, 0)
                cv2.rectangle(out, (cx0, cy0), (cx1, cy1), color, 1)

        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 0), 2)
        return out

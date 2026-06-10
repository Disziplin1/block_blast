"""
overlay.py
==========
추천 결과를 화면에 시각적으로 표시하는 오버레이 렌더링 모듈.

두 가지 렌더링 백엔드를 제공한다.

1. OverlayWindow (PyQt5)
   - 실제 화면(아이폰 미러링 창) 위에 떠 있는 투명/클릭통과 창.
   - 추천 위치에 반투명 박스, 순위 번호, 화살표, 이유를 그린다.

2. render_overlay_cv2
   - Debug 탭/스크린샷 저장용으로, OpenCV 이미지 위에 동일한 정보를 그린다.

좌표계
------
board_rect, tray.slot_rects 는 모두 "캡처된 프레임 기준 로컬 좌표"이다.
OverlayWindow 는 capture.region 의 화면 좌표(left, top)에 위치하므로,
프레임 로컬 좌표 = 위젯 로컬 좌표로 그대로 사용할 수 있다.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import AppConfig, CONFIG, OverlayConfig
from solver import MoveRecommendation, SolveResult
from utils import COLS, PieceCells, ROWS, piece_size


CellRect = Tuple[int, int, int, int]  # x, y, w, h (프레임 로컬 좌표)


# ---------------------------------------------------------------------------
# 좌표 계산 헬퍼 (백엔드 공용)
# ---------------------------------------------------------------------------
def board_cell_rect(board_rect: Tuple[int, int, int, int], row: int, col: int) -> CellRect:
    bx, by, bw, bh = board_rect
    cell_w = bw / COLS
    cell_h = bh / ROWS
    x = int(bx + col * cell_w)
    y = int(by + row * cell_h)
    w = int(cell_w)
    h = int(cell_h)
    return x, y, w, h


def placement_cell_rects(
    board_rect: Tuple[int, int, int, int],
    anchor_row: int,
    anchor_col: int,
    cells: PieceCells,
) -> List[CellRect]:
    return [board_cell_rect(board_rect, anchor_row + dr, anchor_col + dc) for dr, dc in cells]


def placement_bounding_rect(
    board_rect: Tuple[int, int, int, int],
    anchor_row: int,
    anchor_col: int,
    cells: PieceCells,
) -> CellRect:
    h, w = piece_size(cells)
    x, y, _, _ = board_cell_rect(board_rect, anchor_row, anchor_col)
    bx, by, bw, bh = board_rect
    cell_w = bw / COLS
    cell_h = bh / ROWS
    return x, y, int(cell_w * w), int(cell_h * h)


def slot_center(slot_rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x, y, w, h = slot_rect
    return x + w // 2, y + h // 2


def _heatmap_color(value: float) -> Tuple[int, int, int]:
    """0(나쁨,어두움) ~ 1(좋음,밝음) 점수를 BGR 색상으로 변환 (어두운 빨강 -> 밝은 초록)."""
    value = max(0.0, min(1.0, value))
    b = 0
    g = int(255 * value)
    r = int(255 * (1.0 - value))
    return (b, g, r)


# ---------------------------------------------------------------------------
# OpenCV 백엔드 (Debug / 스크린샷)
# ---------------------------------------------------------------------------
RANK_COLORS_BGR = {
    1: (0, 255, 0),
    2: (0, 200, 255),
    3: (0, 120, 255),
}


def render_overlay_cv2(
    frame: np.ndarray,
    solve_result: SolveResult,
    pieces: Sequence[PieceCells],
    config: Optional[AppConfig] = None,
) -> np.ndarray:
    """OpenCV BGR 이미지 위에 추천 결과를 그려서 반환한다 (원본은 변경하지 않음)."""
    cfg = config or CONFIG
    out = frame.copy()
    overlay = frame.copy()
    board_rect = cfg.board.board_rect

    # 위험 지역(Risk Zone) - 빨간색 (#5)
    if cfg.overlay.show_risk_zones and solve_result.risk_map is not None:
        risk_overlay = out.copy()
        for cell in solve_result.risk_map.cells:
            if cell.severity <= 0:
                continue
            x, y, w, h = board_cell_rect(board_rect, cell.row, cell.col)
            cv2.rectangle(risk_overlay, (x, y), (x + w, y + h), cfg.overlay.risk_color, -1)
            risk_alpha = cfg.overlay.risk_max_alpha * cell.severity
            cv2.addWeighted(risk_overlay, risk_alpha, out, 1 - risk_alpha, 0, out)
            risk_overlay = out.copy()

    # 추천 위치 Heat Map (#7)
    if cfg.overlay.show_heatmap:
        heat = solve_result.heatmaps.get(cfg.overlay.heatmap_piece_index)
        if heat is not None:
            heat_overlay = out.copy()
            for r in range(ROWS):
                for c in range(COLS):
                    val = heat[r, c]
                    if np.isnan(val):
                        continue
                    x, y, w, h = board_cell_rect(board_rect, r, c)
                    color = _heatmap_color(float(val))
                    cv2.rectangle(heat_overlay, (x, y), (x + w, y + h), color, -1)
            heat_alpha = cfg.overlay.heatmap_max_alpha
            cv2.addWeighted(heat_overlay, heat_alpha, out, 1 - heat_alpha, 0, out)

    recommendations = solve_result.recommendations
    if cfg.overlay.highlight_rank:
        recommendations = [r for r in recommendations if r.rank == cfg.overlay.highlight_rank]

    for rec in recommendations:
        color = RANK_COLORS_BGR.get(rec.rank, (200, 200, 200))
        cells = pieces[rec.piece_index] if rec.piece_index < len(pieces) else tuple()
        rects = placement_cell_rects(board_rect, rec.anchor_row, rec.anchor_col, cells)

        for (x, y, w, h) in rects:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)

        bx, by, bw, bh = placement_bounding_rect(board_rect, rec.anchor_row, rec.anchor_col, cells)
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), color, cfg.overlay.line_width)

        # 순위 번호
        cv2.putText(
            out,
            str(rec.rank),
            (bx + 4, by + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            cfg.overlay.font_scale,
            color,
            2,
            cv2.LINE_AA,
        )

        # 화살표: 트레이 슬롯 -> 배치 위치
        if cfg.overlay.show_arrows and rec.piece_index < len(cfg.tray.slot_rects):
            slot_rect = cfg.tray.slot_rects[rec.piece_index]
            if slot_rect != (0, 0, 0, 0):
                start = slot_center(slot_rect)
                end = (bx + bw // 2, by + bh // 2)
                cv2.arrowedLine(out, start, end, color, 2, tipLength=0.05)

        # 추천 이유 (첫 1~2개만)
        if cfg.overlay.show_reasons and rec.reasons:
            text = ", ".join(rec.reasons[:2])
            cv2.putText(
                out,
                f"#{rec.rank} {text}",
                (bx, by + bh + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

    alpha = cfg.overlay.alpha
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


# ---------------------------------------------------------------------------
# PyQt5 백엔드 (실시간 화면 오버레이)
# ---------------------------------------------------------------------------
try:
    from PyQt5 import QtCore, QtGui, QtWidgets

    class OverlayWindow(QtWidgets.QWidget):
        """
        화면 위에 떠 있는 투명/클릭통과 오버레이 창.

        - Qt.FramelessWindowHint + Qt.WindowStaysOnTopHint + Qt.Tool: 항상 위, 작업표시줄 미표시
        - Qt.WA_TranslucentBackground: 배경 투명
        - Qt.WA_TransparentForMouseEvents: 클릭/터치 이벤트를 그대로 통과시킴 (입력 이벤트 생성 안 함)
        """

        def __init__(self, config: Optional[AppConfig] = None, parent=None):
            super().__init__(parent)
            self.config = config or CONFIG
            self._solve_result: Optional[SolveResult] = None
            self._pieces: List[PieceCells] = []

            self.setWindowFlags(
                QtCore.Qt.FramelessWindowHint
                | QtCore.Qt.WindowStaysOnTopHint
                | QtCore.Qt.Tool
                | QtCore.Qt.X11BypassWindowManagerHint
            )
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
            self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)

        # ------------------------------------------------------------
        def sync_geometry(self) -> None:
            """capture.region 에 맞춰 오버레이 창의 화면상 위치/크기를 갱신한다."""
            x, y, w, h = self.config.capture.region
            self.setGeometry(x, y, w, h)

        # ------------------------------------------------------------
        def update_recommendation(self, solve_result: SolveResult, pieces: Sequence[PieceCells]) -> None:
            self._solve_result = solve_result
            self._pieces = list(pieces)
            self.update()  # repaint 요청

        # ------------------------------------------------------------
        def clear(self) -> None:
            self._solve_result = None
            self._pieces = []
            self.update()

        # ------------------------------------------------------------
        def paintEvent(self, event) -> None:  # noqa: N802 (Qt 콜백 명명 규칙)
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

            if self._solve_result is None:
                return

            cfg = self.config
            board_rect = cfg.board.board_rect

            color_map = {
                1: cfg.overlay.box_color_1,
                2: cfg.overlay.box_color_2,
                3: cfg.overlay.box_color_3,
            }

            # 위험 지역(Risk Zone) - 빨간색 (#5)
            if cfg.overlay.show_risk_zones and self._solve_result.risk_map is not None:
                painter.setPen(QtCore.Qt.NoPen)
                risk_qcolor = QtGui.QColor(*cfg.overlay.risk_color)
                for cell in self._solve_result.risk_map.cells:
                    if cell.severity <= 0:
                        continue
                    x, y, w, h = board_cell_rect(board_rect, cell.row, cell.col)
                    fill = QtGui.QColor(risk_qcolor)
                    fill.setAlphaF(cfg.overlay.risk_max_alpha * cell.severity)
                    painter.setBrush(QtGui.QBrush(fill))
                    painter.drawRect(x, y, w, h)

            # 추천 위치 Heat Map (#7)
            if cfg.overlay.show_heatmap:
                heat = self._solve_result.heatmaps.get(cfg.overlay.heatmap_piece_index)
                if heat is not None:
                    painter.setPen(QtCore.Qt.NoPen)
                    for r in range(ROWS):
                        for c in range(COLS):
                            val = heat[r, c]
                            if np.isnan(val):
                                continue
                            x, y, w, h = board_cell_rect(board_rect, r, c)
                            br, bg, bb = _heatmap_color(float(val))
                            fill = QtGui.QColor(bb, bg, br)  # BGR -> RGB
                            fill.setAlphaF(cfg.overlay.heatmap_max_alpha)
                            painter.setBrush(QtGui.QBrush(fill))
                            painter.drawRect(x, y, w, h)

            recommendations = self._solve_result.recommendations
            if cfg.overlay.highlight_rank:
                recommendations = [r for r in recommendations if r.rank == cfg.overlay.highlight_rank]

            for rec in recommendations:
                rgb = color_map.get(rec.rank, (200, 200, 200))
                qcolor = QtGui.QColor(*rgb)
                cells = self._pieces[rec.piece_index] if rec.piece_index < len(self._pieces) else tuple()

                rects = placement_cell_rects(board_rect, rec.anchor_row, rec.anchor_col, cells)

                # 반투명 채우기
                fill_color = QtGui.QColor(qcolor)
                fill_color.setAlphaF(cfg.overlay.alpha)
                painter.setBrush(QtGui.QBrush(fill_color))
                painter.setPen(QtCore.Qt.NoPen)
                for (x, y, w, h) in rects:
                    painter.drawRect(x, y, w, h)

                # 외곽선 + 점선
                pen = QtGui.QPen(qcolor, cfg.overlay.line_width)
                pen.setStyle(QtCore.Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(QtCore.Qt.NoBrush)
                bx, by, bw, bh = placement_bounding_rect(board_rect, rec.anchor_row, rec.anchor_col, cells)
                painter.drawRect(bx, by, bw, bh)

                # 순위 번호
                if cfg.overlay.show_numbers:
                    painter.setPen(QtGui.QPen(qcolor))
                    font = painter.font()
                    font.setPointSize(14)
                    font.setBold(True)
                    painter.setFont(font)
                    painter.drawText(bx + 4, by + 18, str(rec.rank))

                # 화살표: 트레이 슬롯 -> 배치 위치
                if cfg.overlay.show_arrows and rec.piece_index < len(cfg.tray.slot_rects):
                    slot_rect = cfg.tray.slot_rects[rec.piece_index]
                    if slot_rect != (0, 0, 0, 0):
                        sx, sy = slot_center(slot_rect)
                        ex, ey = bx + bw // 2, by + bh // 2
                        pen = QtGui.QPen(qcolor, 2)
                        pen.setStyle(QtCore.Qt.SolidLine)
                        painter.setPen(pen)
                        painter.drawLine(sx, sy, ex, ey)
                        _draw_arrowhead(painter, sx, sy, ex, ey, qcolor)

                # 추천 이유
                if cfg.overlay.show_reasons and rec.reasons:
                    text = f"#{rec.rank} " + ", ".join(rec.reasons[:2])
                    painter.setPen(QtGui.QPen(qcolor))
                    font = painter.font()
                    font.setPointSize(9)
                    font.setBold(False)
                    painter.setFont(font)
                    painter.drawText(bx, by + bh + 16, text)

            painter.end()

    def _draw_arrowhead(painter, x0: int, y0: int, x1: int, y1: int, color) -> None:
        import math

        angle = math.atan2(y1 - y0, x1 - x0)
        length = 10
        for offset in (math.pi / 6, -math.pi / 6):
            ax = x1 - length * math.cos(angle - offset)
            ay = y1 - length * math.sin(angle - offset)
            painter.drawLine(x1, y1, int(ax), int(ay))

    PYQT_AVAILABLE = True

except ImportError:  # pragma: no cover - PyQt5 미설치 환경 대비
    PYQT_AVAILABLE = False

    class OverlayWindow:  # type: ignore
        """PyQt5 가 설치되지 않은 환경을 위한 더미 클래스."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyQt5 is not installed. OverlayWindow is unavailable.")

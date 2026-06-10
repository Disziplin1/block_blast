"""
risk_analysis.py
=================
보드의 위험 지역(Dead Area, 구멍, 고립 공간, 막힘 패턴 등)을 셀 단위로
탐지하여, Overlay 에서 빨간색 계열로 표시할 수 있는 위험도 맵을 생성한다.

또한 "추천 위치 Heat Map" (#7) 을 위해, 특정 블록을 보드의 각 위치에
놓았을 때의 평가 점수를 8x8 격자로 계산하는 함수도 제공한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import CONFIG, HeuristicWeights
from heuristic import find_empty_regions, is_empty
from search import apply_placement, valid_placements
from utils import COLS, PieceCells, ROWS


# ---------------------------------------------------------------------------
# 위험 셀 분석
# ---------------------------------------------------------------------------
RISK_TYPE_WEIGHTS: Dict[str, float] = {
    "hole1": 1.0,            # 1칸 구멍
    "hole2": 0.7,            # 2칸 구멍
    "isolated": 1.0,         # 사방이 막힌 고립 셀
    "dead_area": 0.8,        # 작은(<=2칸) 영역에 속한 죽은 공간
    "corner_blocked": 0.4,   # 코너가 막혀있음
    "cross_blocked": 0.6,    # 십자 형태 막힘 (3면 이상 막힘)
    "corridor": 0.5,         # 폭 1칸짜리 긴 통로
    "fragment": 0.3,         # 작은 파편 영역(3~4칸)에 속한 셀
}


@dataclass
class RiskCell:
    row: int
    col: int
    risk_types: List[str]
    severity: float  # 0.0 (안전) ~ 1.0 (매우 위험)


@dataclass
class RiskMap:
    grid: np.ndarray                     # shape (ROWS, COLS), float, 0~1
    cells: List[RiskCell]                # severity > 0 인 셀들만
    region_count: int
    largest_region: int
    dead_area: int
    fragmentation_index: float           # 0~1, 1에 가까울수록 심하게 파편화


def _corridor_cells(board_bb: int) -> set:
    """long_corridor_count 와 동일한 기준으로, 통로를 구성하는 셀 좌표 집합을 반환."""
    cells = set()
    for r in range(ROWS):
        run: List[Tuple[int, int]] = []
        for c in range(COLS):
            blocked_above = not is_empty(board_bb, r - 1, c)
            blocked_below = not is_empty(board_bb, r + 1, c)
            if is_empty(board_bb, r, c) and blocked_above and blocked_below:
                run.append((r, c))
                if len(run) >= 4:
                    cells.update(run)
            else:
                run = []
    for c in range(COLS):
        run = []
        for r in range(ROWS):
            blocked_left = not is_empty(board_bb, r, c - 1)
            blocked_right = not is_empty(board_bb, r, c + 1)
            if is_empty(board_bb, r, c) and blocked_left and blocked_right:
                run.append((r, c))
                if len(run) >= 4:
                    cells.update(run)
            else:
                run = []
    return cells


def _cross_blocked_cells(board_bb: int) -> set:
    cells = set()
    for r in range(ROWS):
        for c in range(COLS):
            if not is_empty(board_bb, r, c):
                continue
            blocked = sum(
                1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= r + dr < ROWS and 0 <= c + dc < COLS and not is_empty(board_bb, r + dr, c + dc)
            )
            if blocked >= 3:
                cells.add((r, c))
    return cells


def analyze_risk(board_bb: int) -> RiskMap:
    """보드 상태로부터 셀별 위험도(0~1) 맵과 요약 통계를 계산한다."""
    grid = np.zeros((ROWS, COLS), dtype=np.float32)
    cell_types: Dict[Tuple[int, int], List[str]] = {}

    region_info = find_empty_regions(board_bb)

    for region in region_info.regions:
        size = len(region)
        if size == 1:
            for (r, c) in region:
                cell_types.setdefault((r, c), []).append("hole1")
        elif size == 2:
            for (r, c) in region:
                cell_types.setdefault((r, c), []).append("hole2")
        elif size <= 2:
            for (r, c) in region:
                cell_types.setdefault((r, c), []).append("dead_area")
        elif size <= 4:
            for (r, c) in region:
                cell_types.setdefault((r, c), []).append("fragment")

    # 고립 셀 (4방향 모두 막힘)
    for r in range(ROWS):
        for c in range(COLS):
            if not is_empty(board_bb, r, c):
                continue
            free_neighbors = sum(
                1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= r + dr < ROWS and 0 <= c + dc < COLS and is_empty(board_bb, r + dr, c + dc)
            )
            if free_neighbors == 0:
                cell_types.setdefault((r, c), []).append("isolated")

    # 십자 형태 막힘
    for (r, c) in _cross_blocked_cells(board_bb):
        cell_types.setdefault((r, c), []).append("cross_blocked")

    # 좁고 긴 통로
    for (r, c) in _corridor_cells(board_bb):
        cell_types.setdefault((r, c), []).append("corridor")

    # 코너가 막혀있는 경우, 코너에 인접한 빈 셀에 약한 페널티 표시
    corners = [(0, 0), (0, COLS - 1), (ROWS - 1, 0), (ROWS - 1, COLS - 1)]
    corner_neighbors = {
        (0, 0): [(0, 1), (1, 0)],
        (0, COLS - 1): [(0, COLS - 2), (1, COLS - 1)],
        (ROWS - 1, 0): [(ROWS - 2, 0), (ROWS - 1, 1)],
        (ROWS - 1, COLS - 1): [(ROWS - 2, COLS - 1), (ROWS - 1, COLS - 2)],
    }
    for corner in corners:
        if not is_empty(board_bb, *corner):
            for (r, c) in corner_neighbors[corner]:
                if is_empty(board_bb, r, c):
                    cell_types.setdefault((r, c), []).append("corner_blocked")

    cells: List[RiskCell] = []
    for (r, c), types in cell_types.items():
        severity = min(1.0, sum(RISK_TYPE_WEIGHTS.get(t, 0.2) for t in types))
        grid[r, c] = severity
        cells.append(RiskCell(row=r, col=c, risk_types=types, severity=severity))

    fragmentation_index = (
        min(1.0, max(0, region_info.region_count - 1) / 8.0) if region_info.region_count > 0 else 0.0
    )

    return RiskMap(
        grid=grid,
        cells=cells,
        region_count=region_info.region_count,
        largest_region=region_info.largest_size,
        dead_area=region_info.dead_count,
        fragmentation_index=fragmentation_index,
    )


# ---------------------------------------------------------------------------
# 추천 위치 Heat Map
# ---------------------------------------------------------------------------
def compute_placement_heatmap(
    board_bb: int,
    cells: PieceCells,
    weights: Optional[HeuristicWeights] = None,
    remaining_pieces: Optional[Sequence[PieceCells]] = None,
) -> np.ndarray:
    """
    주어진 블록(cells)을 보드의 각 가능한 위치(anchor)에 놓았을 때의
    평가 점수를 8x8 배열로 계산하여 0~1 로 정규화한다.

    배치 불가능한 anchor 는 NaN 으로 표시된다.
    """
    from heuristic import evaluate

    weights = weights or CONFIG.weights
    remaining_pieces = remaining_pieces or []

    heat = np.full((ROWS, COLS), np.nan, dtype=np.float32)
    if not cells:
        return heat

    placements = valid_placements(board_bb, 0, cells)
    if not placements:
        return heat

    scores = []
    for placement in placements:
        step = apply_placement(board_bb, placement, combo=0)
        score = evaluate(step.board_bb, step.score_gain, step.combo, remaining_pieces, weights)
        scores.append((placement, score))

    values = np.array([s for _, s in scores], dtype=np.float32)
    vmin, vmax = float(values.min()), float(values.max())
    span = vmax - vmin if vmax > vmin else 1.0

    for placement, score in scores:
        norm = (score - vmin) / span
        heat[placement.anchor_row, placement.anchor_col] = norm

    return heat

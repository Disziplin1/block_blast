"""
heuristic.py
============
보드 상태를 평가하는 휴리스틱 함수 모음.

핵심 원칙
---------
"현재 점수를 최대화" 하지 않고, "최종 점수(=오래 생존)를 최대화" 하기 위해
보드의 구조적 건강도(빈 공간의 형태, 단절 여부, 향후 배치 가능성 등)를
종합적으로 평가한다.

이 모듈은 search.py 의 best_first_moves 에 heuristic_fn 으로 전달되는
evaluate() 와, GUI/오버레이에 표시할 "추천 이유"를 생성하는
evaluate_with_breakdown() 을 제공한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from config import HeuristicWeights, CONFIG
from search import valid_placements
from utils import (
    COL_MASKS,
    COLS,
    FULL_BOARD,
    PieceCells,
    ROW_MASKS,
    ROWS,
    TOTAL_CELLS,
    cell_index,
    popcount,
)


# ---------------------------------------------------------------------------
# 기본 변환 / 좌표 헬퍼
# ---------------------------------------------------------------------------
def empty_cells(board_bb: int) -> List[Tuple[int, int]]:
    """비어 있는 셀들의 (row, col) 좌표 목록."""
    cells = []
    for idx in range(TOTAL_CELLS):
        if not (board_bb >> idx) & 1:
            r, c = divmod(idx, COLS)
            cells.append((r, c))
    return cells


def is_empty(board_bb: int, r: int, c: int) -> bool:
    if not (0 <= r < ROWS and 0 <= c < COLS):
        return False
    return not (board_bb >> cell_index(r, c)) & 1


# ---------------------------------------------------------------------------
# 연결된 빈 영역(Connected Empty Regions) 분석
# ---------------------------------------------------------------------------
@dataclass
class RegionInfo:
    regions: List[List[Tuple[int, int]]]  # 각 영역의 셀 목록
    sizes: List[int]
    largest_size: int
    region_count: int
    hole1_count: int  # 크기 1인 영역 (사방이 막힌 단일 빈칸)
    hole2_count: int  # 크기 2인 영역
    dead_count: int   # 어떤 표준 블록도 들어갈 수 없을 만큼 작거나 좁은 영역의 셀 수


def find_empty_regions(board_bb: int) -> RegionInfo:
    """4방향 연결 기준으로 빈 셀들을 그룹화한다."""
    visited = set()
    regions: List[List[Tuple[int, int]]] = []

    for r in range(ROWS):
        for c in range(COLS):
            if not is_empty(board_bb, r, c) or (r, c) in visited:
                continue
            # BFS
            stack = [(r, c)]
            visited.add((r, c))
            region = []
            while stack:
                cr, cc = stack.pop()
                region.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if is_empty(board_bb, nr, nc) and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        stack.append((nr, nc))
            regions.append(region)

    sizes = [len(reg) for reg in regions]
    largest = max(sizes) if sizes else 0
    hole1 = sum(1 for s in sizes if s == 1)
    hole2 = sum(1 for s in sizes if s == 2)
    # "죽은 공간": 크기가 3 이하인 영역은 대부분의 큰 블록이 들어갈 수 없음
    dead = sum(s for s in sizes if s <= 2)

    return RegionInfo(
        regions=regions,
        sizes=sizes,
        largest_size=largest,
        region_count=len(regions),
        hole1_count=hole1,
        hole2_count=hole2,
        dead_count=dead,
    )


def largest_empty_rectangle(board_bb: int) -> int:
    """가장 큰 직사각형 형태의 빈 공간 넓이(셀 수)를 계산한다."""
    grid = [[0 if is_empty(board_bb, r, c) else 1 for c in range(COLS)] for r in range(ROWS)]
    heights = [0] * COLS
    best = 0
    for r in range(ROWS):
        for c in range(COLS):
            heights[c] = 0 if grid[r][c] == 1 else heights[c] + 1
        # largest rectangle in histogram
        stack: List[int] = []
        for i in range(COLS + 1):
            h = heights[i] if i < COLS else 0
            while stack and heights[stack[-1]] >= h:
                top = stack.pop()
                width = i if not stack else i - stack[-1] - 1
                best = max(best, heights[top] * width)
            stack.append(i)
    return best


# ---------------------------------------------------------------------------
# 라인(행/열) 상태 분석
# ---------------------------------------------------------------------------
def line_fill_counts(board_bb: int) -> Tuple[List[int], List[int]]:
    """각 행/열에 채워진 셀 수를 반환한다."""
    row_counts = [popcount(board_bb & ROW_MASKS[r]) for r in range(ROWS)]
    col_counts = [popcount(board_bb & COL_MASKS[c]) for c in range(COLS)]
    return row_counts, col_counts


def near_complete_lines(board_bb: int, threshold: int = 1) -> int:
    """완성까지 threshold 칸 이하 남은 행/열의 개수 (콤보 가능성 지표)."""
    row_counts, col_counts = line_fill_counts(board_bb)
    count = 0
    for rc in row_counts:
        if 0 < (COLS - rc) <= threshold:
            count += 1
    for cc in col_counts:
        if 0 < (ROWS - cc) <= threshold:
            count += 1
    return count


def line_clear_efficiency(board_bb: int) -> Tuple[float, float]:
    """행/열 평균 채움 비율 (1에 가까울수록 클리어에 근접)."""
    row_counts, col_counts = line_fill_counts(board_bb)
    row_eff = sum(row_counts) / (ROWS * COLS)
    col_eff = sum(col_counts) / (ROWS * COLS)
    return row_eff, col_eff


# ---------------------------------------------------------------------------
# 배치 가능성 / 생존성
# ---------------------------------------------------------------------------
def placement_count(board_bb: int, cells: PieceCells) -> int:
    if not cells:
        return 0
    return len(valid_placements(board_bb, 0, cells))


def average_placeability(board_bb: int, piece_library: Sequence[PieceCells]) -> float:
    """표준 블록 라이브러리 전체에 대해 평균 배치 가능 위치 수."""
    if not piece_library:
        return 0.0
    total = sum(placement_count(board_bb, p) for p in piece_library)
    return total / len(piece_library)


def can_fit_big_pieces(board_bb: int) -> bool:
    """3x3, 1x5, 5x1 등 큰 블록이 들어갈 자리가 있는지."""
    big_pieces = [
        tuple((r, c) for r in range(3) for c in range(3)),  # 3x3
        tuple((0, c) for c in range(5)),                    # 1x5
        tuple((r, 0) for r in range(5)),                    # 5x1
    ]
    return any(placement_count(board_bb, p) > 0 for p in big_pieces)


# ---------------------------------------------------------------------------
# 외곽 / 코너 / 십자 막힘
# ---------------------------------------------------------------------------
def border_block_count(board_bb: int) -> int:
    """보드 가장자리 셀 중 채워진 셀 수 (너무 많으면 중앙이 고립되기 쉬움)."""
    count = 0
    for r in range(ROWS):
        for c in range(COLS):
            if r in (0, ROWS - 1) or c in (0, COLS - 1):
                if not is_empty(board_bb, r, c):
                    count += 1
    return count


def corner_block_count(board_bb: int) -> int:
    corners = [(0, 0), (0, COLS - 1), (ROWS - 1, 0), (ROWS - 1, COLS - 1)]
    return sum(1 for r, c in corners if not is_empty(board_bb, r, c))


def cross_block_pattern_count(board_bb: int) -> int:
    """
    빈 셀 주변 4방향이 모두 막혀있고, 대각선 중 일부도 막혀 십자 형태로
    고립을 유발하는 패턴 수 (이동 자유도를 크게 줄이는 형태).
    """
    count = 0
    for r in range(ROWS):
        for c in range(COLS):
            if not is_empty(board_bb, r, c):
                continue
            blocked_orthogonal = sum(
                1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if not is_empty(board_bb, r + dr, c + dc)
                and 0 <= r + dr < ROWS
                and 0 <= c + dc < COLS
            )
            if blocked_orthogonal >= 3:
                count += 1
    return count


def long_corridor_count(board_bb: int) -> int:
    """폭 1칸짜리 긴 빈 통로(길이 4 이상)의 개수."""
    count = 0
    # 가로 통로
    for r in range(ROWS):
        run = 0
        for c in range(COLS):
            if is_empty(board_bb, r, c) and is_empty(board_bb, r - 1, c) is False and is_empty(board_bb, r + 1, c) is False:
                run += 1
                if run >= 4:
                    count += 1
                    run = 0
            else:
                run = 0
    # 세로 통로
    for c in range(COLS):
        run = 0
        for r in range(ROWS):
            if is_empty(board_bb, r, c) and is_empty(board_bb, r, c - 1) is False and is_empty(board_bb, r, c + 1) is False:
                run += 1
                if run >= 4:
                    count += 1
                    run = 0
            else:
                run = 0
    return count


# ---------------------------------------------------------------------------
# 대칭성
# ---------------------------------------------------------------------------
def symmetry_score(board_bb: int) -> float:
    """좌우/상하 대칭 정도 (0~1, 1이면 완전 대칭)."""
    total = TOTAL_CELLS
    matches_h = 0
    matches_v = 0
    for r in range(ROWS):
        for c in range(COLS):
            a = not is_empty(board_bb, r, c)
            b_h = not is_empty(board_bb, r, COLS - 1 - c)
            b_v = not is_empty(board_bb, ROWS - 1 - r, c)
            if a == b_h:
                matches_h += 1
            if a == b_v:
                matches_v += 1
    return ((matches_h / total) + (matches_v / total)) / 2.0


# ---------------------------------------------------------------------------
# 중앙 공간 / 확장 가능 공간 / 연결성
# ---------------------------------------------------------------------------
def center_empty_count(board_bb: int) -> int:
    """중앙 4x4 영역의 빈 셀 수."""
    count = 0
    for r in range(2, 6):
        for c in range(2, 6):
            if is_empty(board_bb, r, c):
                count += 1
    return count


def expandable_space_count(board_bb: int) -> int:
    """빈 셀이면서 인접한 빈 셀이 2개 이상인 셀 수 (확장 가능한 공간)."""
    count = 0
    for r in range(ROWS):
        for c in range(COLS):
            if not is_empty(board_bb, r, c):
                continue
            neighbors = sum(
                1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if is_empty(board_bb, r + dr, c + dc)
            )
            if neighbors >= 2:
                count += 1
    return count


def isolated_cell_count(board_bb: int) -> int:
    """빈 셀인데 4방향이 모두 막혀 있는 (보드 경계 포함) 셀 수."""
    count = 0
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
                count += 1
    return count


# ---------------------------------------------------------------------------
# 종합 평가
# ---------------------------------------------------------------------------
@dataclass
class EvaluationBreakdown:
    total: float
    score_gain: float
    components: Dict[str, float]
    reasons: List[str]


def evaluate_with_breakdown(
    board_bb: int,
    score_gain: int,
    combo: int,
    remaining_pieces: Sequence[PieceCells],
    weights: HeuristicWeights = None,
) -> EvaluationBreakdown:
    """보드 상태를 평가하고, 점수 구성 요소 및 사람이 읽을 수 있는 이유를 반환한다."""
    w = weights or CONFIG.weights
    components: Dict[str, float] = {}
    reasons: List[str] = []

    filled = popcount(board_bb)
    empty = TOTAL_CELLS - filled

    # --- 기본 점수 ---
    components["score_gain"] = score_gain * w.score_gain
    if combo > 0:
        components["combo_bonus"] = combo * w.combo_bonus
        reasons.append(f"콤보 {combo}단계 유지")

    # --- 빈 공간 관련 ---
    components["empty_cell_bonus"] = empty * w.empty_cell_bonus

    region_info = find_empty_regions(board_bb)
    components["large_empty_region_bonus"] = region_info.largest_size * w.large_empty_region_bonus
    if region_info.largest_size >= 9:
        reasons.append("큰 빈 공간 유지")

    rect_area = largest_empty_rectangle(board_bb)
    components["rectangular_region_bonus"] = rect_area * w.rectangular_region_bonus
    if rect_area >= 9:
        reasons.append("직사각형 공간 확보")

    expandable = expandable_space_count(board_bb)
    components["expandable_space_bonus"] = expandable * w.expandable_space_bonus

    center = center_empty_count(board_bb)
    components["center_space_bonus"] = center * w.center_space_bonus
    if center >= 8:
        reasons.append("중앙 공간 확보")

    # --- 연결성 / 대칭성 ---
    # 영역 수가 적을수록(=덜 파편화) 좋음 -> 역수 형태로 보너스
    connectivity = (1.0 / region_info.region_count) if region_info.region_count > 0 else 1.0
    components["connectivity_bonus"] = connectivity * w.connectivity_bonus * 10

    sym = symmetry_score(board_bb)
    components["symmetry_bonus"] = sym * w.symmetry_bonus * 10

    # --- 클리어 가능성 ---
    near_complete = near_complete_lines(board_bb, threshold=1)
    components["chain_clear_potential_bonus"] = near_complete * w.chain_clear_potential_bonus
    if near_complete > 0:
        reasons.append(f"{near_complete}개 라인 클리어 임박")

    row_eff, col_eff = line_clear_efficiency(board_bb)
    components["line_clear_efficiency_bonus"] = row_eff * w.line_clear_efficiency_bonus * 10
    components["column_clear_efficiency_bonus"] = col_eff * w.column_clear_efficiency_bonus * 10

    # --- 다음 블록 대응력 ---
    placeable_next = sum(1 for p in remaining_pieces if p and placement_count(board_bb, p) > 0)
    total_next = sum(1 for p in remaining_pieces if p)
    if total_next > 0:
        fit_ratio = placeable_next / total_next
        components["next_piece_fit_bonus"] = fit_ratio * w.next_piece_fit_bonus
        if fit_ratio == 1.0:
            reasons.append("다음 블록 모두 배치 가능")

    move_freedom = sum(placement_count(board_bb, p) for p in remaining_pieces if p)
    components["move_freedom_bonus"] = move_freedom * w.move_freedom_bonus

    from utils import PIECE_LIBRARY
    avg_place = average_placeability(board_bb, PIECE_LIBRARY)
    components["avg_placeability_bonus"] = avg_place * w.avg_placeability_bonus

    # --- 생존 가능성 ---
    can_survive = move_freedom > 0
    components["survivability_bonus"] = (w.survivability_bonus if can_survive else -w.survivability_bonus * 5)
    if can_survive:
        if avg_place > 15:
            reasons.append("생존 가능성 높음")
    else:
        reasons.append("게임 오버 위험")

    if can_fit_big_pieces(board_bb):
        reasons.append("큰 블록 대응 가능")
    else:
        components["big_piece_unplaceable_penalty"] = -w.big_piece_unplaceable_penalty

    # --- 공간 활용도 ---
    utilization = filled / TOTAL_CELLS
    # 너무 꽉 차지도, 너무 비지도 않은 중간 상태를 약간 선호 (0.5 부근)
    components["space_utilization_bonus"] = (1.0 - abs(utilization - 0.5) * 2) * w.space_utilization_bonus * 10

    # --- 페널티 (구멍, 고립, 단편화) ---
    isolated = isolated_cell_count(board_bb)
    components["isolated_cell_penalty"] = -isolated * w.isolated_cell_penalty
    if isolated > 0:
        reasons.append(f"고립 셀 {isolated}개 발생")
    elif isolated == 0:
        reasons.append("고립 셀 생성 없음")

    components["hole1_penalty"] = -region_info.hole1_count * w.hole1_penalty
    components["hole2_penalty"] = -region_info.hole2_count * w.hole2_penalty
    if region_info.hole1_count > 0:
        reasons.append(f"1칸 구멍 {region_info.hole1_count}개")
    if region_info.hole2_count > 0:
        reasons.append(f"2칸 구멍 {region_info.hole2_count}개")

    components["dead_space_penalty"] = -region_info.dead_count * w.dead_space_penalty

    fragmentation = max(0, region_info.region_count - 1)
    components["fragmentation_penalty"] = -fragmentation * w.fragmentation_penalty
    if fragmentation >= 3:
        reasons.append("공간 파편화 심함")

    border_blocks = border_block_count(board_bb)
    components["border_block_penalty"] = -border_blocks * w.border_block_penalty * 0.1

    corners = corner_block_count(board_bb)
    components["corner_block_penalty"] = -corners * w.corner_block_penalty
    if corners >= 3:
        reasons.append("코너 다수 막힘")

    cross_blocks = cross_block_pattern_count(board_bb)
    components["cross_block_penalty"] = -cross_blocks * w.cross_block_penalty
    if cross_blocks > 0:
        reasons.append("십자 형태 막힘 발생")

    corridors = long_corridor_count(board_bb)
    components["long_corridor_penalty"] = -corridors * w.long_corridor_penalty
    if corridors > 0:
        reasons.append("좁고 긴 통로 생성")

    if move_freedom < 5:
        components["move_restriction_penalty"] = -(5 - move_freedom) * w.move_restriction_penalty
        reasons.append("미래 이동 제한 위험")

    if not reasons:
        reasons.append("안정적인 보드 상태 유지")

    total = sum(components.values())
    return EvaluationBreakdown(total=total, score_gain=score_gain, components=components, reasons=reasons)


def evaluate(
    board_bb: int,
    score_gain: int,
    combo: int,
    remaining_pieces: Sequence[PieceCells],
    weights: HeuristicWeights = None,
) -> float:
    """search.py 의 heuristic_fn 시그니처에 맞춘 단순 평가 함수."""
    return evaluate_with_breakdown(board_bb, score_gain, combo, remaining_pieces, weights).total

"""
search.py
=========
비트보드 기반 배치 탐색 엔진.

- 주어진 보드와 블록(피스)에 대해 가능한 모든 배치를 계산
- 배치 -> 라인 클리어 -> 점수/콤보 갱신을 시뮬레이션
- 3개 블록의 모든 순서(순열) x 모든 위치 조합을 DFS/백트래킹으로 탐색
- (선택) beam_width 로 가지치기, time_budget_sec 으로 시간 제한
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from utils import (
    COL_MASKS,
    COLS,
    FULL_BOARD,
    PieceCells,
    ROW_MASKS,
    ROWS,
    piece_size,
    popcount,
    quick_evaluate,
)

# ---------------------------------------------------------------------------
# 점수 규칙 (Block Blast 근사치)
# ---------------------------------------------------------------------------
SCORE_PER_CELL = 1
SCORE_PER_LINE = 10
MULTI_LINE_EXTRA = 10          # 동시에 여러 줄을 지울 때 줄당 추가 보너스
COMBO_BONUS_PER_LEVEL = 5      # 콤보 레벨(연속 클리어 횟수)당 추가 보너스


@dataclass
class Placement:
    """하나의 블록을 보드에 배치하는 행위를 나타낸다."""
    piece_index: int            # 입력 pieces 리스트에서의 원래 인덱스 (0,1,2)
    anchor_row: int
    anchor_col: int
    mask: int                   # 배치될 셀들의 비트마스크
    cell_count: int


@dataclass
class StepResult:
    """배치 + 라인 클리어 한 스텝의 결과."""
    board_bb: int
    score_gain: int
    lines_cleared: int
    rows_cleared: List[int]
    cols_cleared: List[int]
    combo: int


@dataclass
class SearchResult:
    """최종 탐색 결과 한 개 (하나의 피스 순서 + 배치 조합 경로)."""
    order: Tuple[int, ...]                  # 사용된 piece_index 순서
    placements: Tuple[Placement, ...]       # 각 단계의 배치 정보
    final_board_bb: int
    total_score_gain: int
    final_combo: int
    lines_cleared_total: int
    heuristic_score: float = 0.0

    @property
    def first_move(self) -> Placement:
        return self.placements[0]


# ---------------------------------------------------------------------------
# 배치 가능 위치 탐색
# ---------------------------------------------------------------------------
def valid_placements(board_bb: int, piece_index: int, cells: PieceCells) -> List[Placement]:
    """
    주어진 보드(board_bb)에 piece(cells)를 놓을 수 있는 모든 위치를 반환한다.
    piece 는 회전을 지원하지 않으므로 그대로의 형태만 검사한다.
    """
    if not cells:
        return []

    h, w = piece_size(cells)
    if h > ROWS or w > COLS:
        return []

    placements: List[Placement] = []
    cell_count = len(cells)

    for anchor_r in range(ROWS - h + 1):
        for anchor_c in range(COLS - w + 1):
            mask = 0
            for dr, dc in cells:
                r, c = anchor_r + dr, anchor_c + dc
                mask |= 1 << (r * COLS + c)
            if board_bb & mask == 0:
                placements.append(
                    Placement(
                        piece_index=piece_index,
                        anchor_row=anchor_r,
                        anchor_col=anchor_c,
                        mask=mask,
                        cell_count=cell_count,
                    )
                )
    return placements


# ---------------------------------------------------------------------------
# 배치 적용 + 라인 클리어
# ---------------------------------------------------------------------------
def apply_placement(board_bb: int, placement: Placement, combo: int) -> StepResult:
    """블록을 배치하고, 가득 찬 행/열을 제거한 뒤 점수/콤보를 계산한다."""
    new_board = board_bb | placement.mask

    rows_cleared: List[int] = []
    cols_cleared: List[int] = []

    for r in range(ROWS):
        if new_board & ROW_MASKS[r] == ROW_MASKS[r]:
            rows_cleared.append(r)
    for c in range(COLS):
        if new_board & COL_MASKS[c] == COL_MASKS[c]:
            cols_cleared.append(c)

    lines_cleared = len(rows_cleared) + len(cols_cleared)

    if lines_cleared > 0:
        clear_mask = 0
        for r in rows_cleared:
            clear_mask |= ROW_MASKS[r]
        for c in cols_cleared:
            clear_mask |= COL_MASKS[c]
        new_board &= ~clear_mask & FULL_BOARD
        new_combo = combo + 1
    else:
        new_combo = 0

    score_gain = placement.cell_count * SCORE_PER_CELL
    if lines_cleared > 0:
        score_gain += lines_cleared * SCORE_PER_LINE
        if lines_cleared > 1:
            score_gain += (lines_cleared - 1) * MULTI_LINE_EXTRA
        score_gain += new_combo * COMBO_BONUS_PER_LEVEL

    return StepResult(
        board_bb=new_board,
        score_gain=score_gain,
        lines_cleared=lines_cleared,
        rows_cleared=rows_cleared,
        cols_cleared=cols_cleared,
        combo=new_combo,
    )


def can_place_anywhere(board_bb: int, cells: PieceCells) -> bool:
    """이 블록을 보드 어딘가에 놓을 수 있는지 빠르게 확인한다."""
    return len(valid_placements(board_bb, 0, cells)) > 0


def is_game_over(board_bb: int, pieces: Sequence[PieceCells]) -> bool:
    """남은 모든 블록 중 어느 것도 놓을 곳이 없으면 게임 오버."""
    return all(not can_place_anywhere(board_bb, p) for p in pieces if p)


# ---------------------------------------------------------------------------
# 전체 탐색 (모든 순서 x 모든 위치, DFS/백트래킹)
# ---------------------------------------------------------------------------
HeuristicFn = Callable[[int, int, int, Sequence[PieceCells]], float]


def search_all(
    board_bb: int,
    pieces: Sequence[PieceCells],
    beam_width: int = 0,
    time_budget_sec: float = 0.5,
) -> List[SearchResult]:
    """
    가능한 모든 (피스 순서 x 배치 위치) 조합을 DFS로 탐색한다.

    Parameters
    ----------
    board_bb: 현재 보드 상태 (비트보드)
    pieces: 현재 트레이의 블록들 (회전 없음). 빈 슬롯은 빈 tuple() 로 표시.
    beam_width: 0 이면 가지치기 없음. >0 이면 각 깊이에서 저비용 휴리스틱
                (utils.quick_evaluate) 상위 N개만 확장하여 탐색 폭발을 억제한다.
    time_budget_sec: 탐색 시간 제한. 초과 시 현재까지의 결과를 반환.

    Returns
    -------
    모든(또는 시간/가지치기 제한 내) 탐색된 leaf SearchResult 리스트.
    leaf 가 없을 경우(어느 순서로도 배치 불가) 빈 리스트 반환.
    """
    active = [(i, p) for i, p in enumerate(pieces) if p]
    if not active:
        return []

    start_time = time.perf_counter()
    results: List[SearchResult] = []

    for order in itertools.permutations(active):
        if time.perf_counter() - start_time > time_budget_sec:
            break
        order_indices = tuple(idx for idx, _ in order)
        order_cells = [cells for _, cells in order]

        # 깊이별 프론티어: (board_bb, score_gain_sum, combo, lines_cleared_sum, placements_so_far)
        frontier: List[Tuple[int, int, int, int, Tuple[Placement, ...]]] = [
            (board_bb, 0, 0, 0, tuple())
        ]

        for depth, (orig_idx, cells) in enumerate(zip(order_indices, order_cells)):
            next_frontier: List[Tuple[int, int, int, int, Tuple[Placement, ...]]] = []

            for cur_bb, cur_score, cur_combo, cur_lines, cur_placements in frontier:
                if time.perf_counter() - start_time > time_budget_sec:
                    break
                placements = valid_placements(cur_bb, orig_idx, cells)
                for placement in placements:
                    step = apply_placement(cur_bb, placement, cur_combo)
                    next_frontier.append(
                        (
                            step.board_bb,
                            cur_score + step.score_gain,
                            step.combo,
                            cur_lines + step.lines_cleared,
                            cur_placements + (placement,),
                        )
                    )

            if not next_frontier:
                # 이 순서로는 depth 단계에서 막힘 -> 이 순열은 스킵
                frontier = []
                break

            if beam_width and len(next_frontier) > beam_width:
                next_frontier.sort(
                    key=lambda item: quick_evaluate(item[0], item[1], item[2]),
                    reverse=True,
                )
                next_frontier = next_frontier[:beam_width]

            frontier = next_frontier

        for bb, score_sum, combo, lines_total, placements in frontier:
            results.append(
                SearchResult(
                    order=order_indices,
                    placements=placements,
                    final_board_bb=bb,
                    total_score_gain=score_sum,
                    final_combo=combo,
                    lines_cleared_total=lines_total,
                )
            )

    return results


def best_first_moves(
    board_bb: int,
    pieces: Sequence[PieceCells],
    heuristic_fn: Optional[HeuristicFn] = None,
    beam_width: int = 0,
    time_budget_sec: float = 0.5,
    top_k: int = 3,
) -> List[SearchResult]:
    """
    search_all 결과 중, 첫 번째 배치(first_move)가 동일한 결과들을 그룹화하여
    각 그룹에서 가장 좋은 leaf 를 대표값으로 삼고, 휴리스틱 기준 상위 top_k 를 반환한다.

    -> "지금 어떤 블록을, 어디에 놓아야 최선인가" 를 알기 위함.
    """
    all_results = search_all(board_bb, pieces, beam_width, time_budget_sec)
    if not all_results:
        return []

    # 1단계: 저비용 quick_evaluate 로 첫 수(piece_index, anchor_row, anchor_col)
    # 기준 그룹화하여 각 그룹의 대표(미래까지 고려했을 때 가장 유망한) leaf 선택.
    # 정밀한 heuristic_fn 은 비용이 크므로 그룹별 대표에 대해서만 호출한다.
    best_per_first_move: dict = {}
    for r in all_results:
        fm = r.first_move
        key = (fm.piece_index, fm.anchor_row, fm.anchor_col)
        q = quick_evaluate(r.final_board_bb, r.total_score_gain, r.final_combo)
        if key not in best_per_first_move or q > best_per_first_move[key][0]:
            best_per_first_move[key] = (q, r)

    candidates = [r for _, r in best_per_first_move.values()]

    # 2단계: 그룹 대표들에 대해서만 정밀 휴리스틱 평가
    for r in candidates:
        r.heuristic_score = (
            heuristic_fn(r.final_board_bb, r.total_score_gain, r.final_combo, pieces)
            if heuristic_fn is not None
            else float(r.total_score_gain)
        )

    ranked = sorted(candidates, key=lambda r: r.heuristic_score, reverse=True)
    return ranked[:top_k]

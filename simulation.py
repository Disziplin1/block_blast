"""
simulation.py
==============
미래 턴을 내다보기 위한 Monte Carlo 시뮬레이션.

다음에 어떤 블록이 나올지는 알 수 없으므로, utils.PIECE_LIBRARY 에서
무작위로 블록을 뽑아 여러 번(시뮬레이션 횟수) 게임을 진행시켜
평균 생존 턴 수, 평균 점수, 평균 콤보, 게임 종료 확률을 추정한다.

탐색 비용을 줄이기 위해 시뮬레이션 내부에서는 매 턴마다 전체 search.search_all
을 돌리지 않고, "그리디 + 1-step 휴리스틱" 전략으로 빠르게 진행한다.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

from config import CONFIG, HeuristicWeights
from search import Placement, apply_placement, can_place_anywhere, valid_placements
from utils import PieceCells, quick_evaluate, random_pieces


@dataclass
class SimulationResult:
    avg_survival_turns: float
    avg_score: float
    avg_combo: float
    end_probability: float
    trials: int


def _greedy_play_turn(
    board_bb: int,
    pieces: Sequence[PieceCells],
    combo: int,
    weights: HeuristicWeights,
    rng: random.Random,
) -> Optional[tuple]:
    """
    한 턴(3개 블록)을 그리디하게 진행한다.

    각 단계에서 남은 블록들 중, 저비용 1-step 휴리스틱(quick_evaluate)이
    가장 높은 (블록, 배치) 조합을 선택하여 적용한다. 시뮬레이션은 수백~수천
    번 반복되므로 정밀한 heuristic.evaluate() 대신 빠른 근사치를 사용한다.

    Returns
    -------
    (new_board_bb, score_gain_total, new_combo) 또는
    어느 블록도 놓을 수 없으면 None (게임 오버).
    """
    remaining = list(pieces)
    cur_board = board_bb
    cur_combo = combo
    total_score = 0

    while any(p for p in remaining):
        best = None  # (value, idx, placement, step_result)
        for idx, cells in enumerate(remaining):
            if not cells:
                continue
            placements = valid_placements(cur_board, idx, cells)
            if not placements:
                continue
            for placement in placements:
                step = apply_placement(cur_board, placement, cur_combo)
                value = quick_evaluate(step.board_bb, step.score_gain, step.combo)
                if best is None or value > best[0]:
                    best = (value, idx, placement, step)

        if best is None:
            # 남은 블록 중 어느 것도 놓을 수 없음 -> 게임 오버
            return None

        _, idx, placement, step = best
        cur_board = step.board_bb
        cur_combo = step.combo
        total_score += step.score_gain
        remaining[idx] = tuple()

    return cur_board, total_score, cur_combo


def monte_carlo_evaluate(
    board_bb: int,
    current_remaining_pieces: Sequence[PieceCells],
    trials: Optional[int] = None,
    depth: Optional[int] = None,
    weights: Optional[HeuristicWeights] = None,
    time_budget_sec: Optional[float] = None,
    rng: Optional[random.Random] = None,
) -> SimulationResult:
    """
    board_bb 상태에서 시작하여, 무작위 미래 블록들로 `depth` 턴을 진행하는
    시뮬레이션을 `trials` 회 수행한다.

    current_remaining_pieces: 이번 탐색에서 아직 배치하지 않은(시뮬레이션 시작 시점에
    실제로 손에 들고 있는) 블록들. 첫 턴은 이 블록들로 시작하고, 이후 턴부터는
    무작위 블록 3개를 새로 뽑는다.
    """
    trials = trials if trials is not None else CONFIG.solver.monte_carlo_trials
    depth = depth if depth is not None else CONFIG.solver.monte_carlo_depth
    weights = weights or CONFIG.weights
    time_budget = time_budget_sec if time_budget_sec is not None else CONFIG.solver.time_budget_sec
    rng = rng or random.Random()

    survival_turns_list: List[int] = []
    score_list: List[float] = []
    combo_list: List[int] = []
    ended_count = 0

    start = time.perf_counter()
    actual_trials = 0

    for _ in range(trials):
        if time.perf_counter() - start > time_budget:
            break
        actual_trials += 1

        cur_board = board_bb
        cur_combo = 0
        total_score = 0.0
        survived_turns = 0
        ended = False

        first_pieces = [p for p in current_remaining_pieces if p]
        turn_pieces = first_pieces if first_pieces else random_pieces(3, rng)

        for turn in range(depth):
            if not any(turn_pieces):
                turn_pieces = random_pieces(3, rng)

            if all(not can_place_anywhere(cur_board, p) for p in turn_pieces if p):
                ended = True
                break

            result = _greedy_play_turn(cur_board, turn_pieces, cur_combo, weights, rng)
            if result is None:
                ended = True
                break

            cur_board, gained, cur_combo = result
            total_score += gained
            survived_turns += 1
            turn_pieces = random_pieces(3, rng)

        survival_turns_list.append(survived_turns)
        score_list.append(total_score)
        combo_list.append(cur_combo)
        if ended:
            ended_count += 1

    if actual_trials == 0:
        return SimulationResult(0.0, 0.0, 0.0, 1.0, 0)

    return SimulationResult(
        avg_survival_turns=sum(survival_turns_list) / actual_trials,
        avg_score=sum(score_list) / actual_trials,
        avg_combo=sum(combo_list) / actual_trials,
        end_probability=ended_count / actual_trials,
        trials=actual_trials,
    )

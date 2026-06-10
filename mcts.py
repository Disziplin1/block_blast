"""
mcts.py
=======
Beam Search(search.best_first_moves)와 병행하여 사용하는
Monte Carlo Tree Search 모듈 (#1: 다중 턴(3~8턴) 미래 시뮬레이션).

구조
----
- 트리 부분: 현재 트레이에 들어있는 블록들(보통 3개)을 "어떤 순서로,
  어디에" 놓을지를 UCB1 기반으로 탐색한다. 트레이가 모두 소진되면
  하나의 "리프"가 된다.
- 롤아웃 부분: 리프 이후부터는 매 턴마다 새로운 무작위 블록 3개를
  받는다고 가정하고, quick_evaluate 기반 그리디 정책으로
  최대 `max_turns` 턴까지 시뮬레이션하여 보상을 추정한다
  (simulation._greedy_play_turn 재사용).

루트의 각 자식(=현재 트레이의 첫 번째 배치 후보)에 대한 방문 횟수와
평균 보상을 통해, Beam Search 와는 독립적인 "장기 생존" 관점의
추천을 제공한다.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from config import CONFIG, HeuristicWeights
from search import Placement, apply_placement, can_place_anywhere, valid_placements
from simulation import _greedy_play_turn
from utils import PieceCells, quick_evaluate, random_pieces


@dataclass
class MCTSAction:
    slot: int            # 트레이 내 인덱스
    placement: Placement


@dataclass
class MCTSNode:
    board_bb: int
    tray: Tuple[PieceCells, ...]   # 사용한 슬롯은 () 로 표시된 트레이 상태
    combo: int
    score_so_far: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    untried: Optional[List[MCTSAction]] = None
    children: Dict[Tuple[int, int, int], "MCTSNode"] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


@dataclass
class MCTSChildStats:
    slot: int
    placement: Placement
    visits: int
    mean_value: float


@dataclass
class MCTSResult:
    best_action: Optional[MCTSAction]
    root_children: List[MCTSChildStats]
    iterations: int
    elapsed_sec: float


def _legal_actions(board_bb: int, tray: Sequence[PieceCells]) -> List[MCTSAction]:
    actions: List[MCTSAction] = []
    for slot, cells in enumerate(tray):
        if not cells:
            continue
        for placement in valid_placements(board_bb, slot, cells):
            actions.append(MCTSAction(slot=slot, placement=placement))
    return actions


def _action_key(action: MCTSAction) -> Tuple[int, int, int]:
    return (action.slot, action.placement.anchor_row, action.placement.anchor_col)


def _ucb_score(parent_visits: int, child: MCTSNode, c: float) -> float:
    if child.visits == 0:
        return float("inf")
    return child.mean_value + c * math.sqrt(math.log(max(1, parent_visits)) / child.visits)


def _rollout_value(
    board_bb: int,
    combo: int,
    score_so_far: float,
    max_turns: int,
    weights: HeuristicWeights,
    rng: random.Random,
) -> float:
    """
    트레이 소진 이후, 무작위 블록으로 max_turns 턴까지 그리디 진행하며
    누적 점수 + 최종 보드의 quick_evaluate 점수를 보상으로 반환한다.
    조기 게임오버 시 패널티를 부여한다.
    """
    cur_board = board_bb
    cur_combo = combo
    total_score = score_so_far
    survived = 0

    for _ in range(max_turns):
        turn_pieces = random_pieces(3, rng)
        if all(not can_place_anywhere(cur_board, p) for p in turn_pieces if p):
            break
        result = _greedy_play_turn(cur_board, turn_pieces, cur_combo, weights, rng)
        if result is None:
            break
        cur_board, gained, cur_combo = result
        total_score += gained
        survived += 1

    survival_bonus = survived * 8.0
    end_penalty = 0.0 if survived >= max_turns else (max_turns - survived) * 15.0
    return total_score + quick_evaluate(cur_board, 0, cur_combo) + survival_bonus - end_penalty


def mcts_search(
    board_bb: int,
    piece_cells_list: Sequence[PieceCells],
    weights: Optional[HeuristicWeights] = None,
    iterations: Optional[int] = None,
    max_turns: Optional[int] = None,
    time_budget_sec: Optional[float] = None,
    exploration: Optional[float] = None,
    rng: Optional[random.Random] = None,
) -> MCTSResult:
    """
    현재 트레이(piece_cells_list, 보통 3개)를 비우는 모든 순서/위치 조합을
    UCB1 트리로 탐색하고, 트레이를 비운 이후에는 무작위 미래 턴을
    `max_turns` 까지 롤아웃하여 장기 생존 관점의 가치를 추정한다.

    Returns
    -------
    MCTSResult: 루트(현재 트레이)에서의 최선의 첫 행동과, 각 자식 행동의
    방문 횟수/평균 가치 통계.
    """
    cfg = CONFIG.solver
    weights = weights or CONFIG.weights
    iterations = iterations if iterations is not None else cfg.mcts_iterations
    max_turns = max_turns if max_turns is not None else cfg.mcts_max_turns
    time_budget = time_budget_sec if time_budget_sec is not None else cfg.mcts_time_budget_sec
    c = exploration if exploration is not None else cfg.mcts_exploration
    rng = rng or random.Random()

    root_tray = tuple(piece_cells_list)
    root = MCTSNode(board_bb=board_bb, tray=root_tray, combo=0)
    root.untried = _legal_actions(root.board_bb, root.tray)

    start = time.perf_counter()
    iters_done = 0

    if not root.untried:
        return MCTSResult(best_action=None, root_children=[], iterations=0, elapsed_sec=0.0)

    for _ in range(iterations):
        if time.perf_counter() - start > time_budget:
            break
        iters_done += 1

        # --- Selection + Expansion ---
        path: List[MCTSNode] = [root]
        node = root

        while True:
            if node.untried is None:
                node.untried = _legal_actions(node.board_bb, node.tray)

            if node.untried:
                action = node.untried.pop(rng.randrange(len(node.untried)))
                step = apply_placement(node.board_bb, action.placement, node.combo)
                new_tray = list(node.tray)
                new_tray[action.slot] = tuple()
                if not any(new_tray):
                    new_tray = list(random_pieces(3, rng))
                child = MCTSNode(
                    board_bb=step.board_bb,
                    tray=tuple(new_tray),
                    combo=step.combo,
                    score_so_far=node.score_so_far + step.score_gain,
                )
                node.children[_action_key(action)] = child
                path.append(child)
                node = child
                break

            legal = _legal_actions(node.board_bb, node.tray)
            if not legal:
                break  # 게임 오버 노드 (확장 불가)

            best_key = None
            best_score = -float("inf")
            for action in legal:
                key = _action_key(action)
                child = node.children.get(key)
                if child is None:
                    best_key = (action, key)
                    break
                score = _ucb_score(node.visits, child, c)
                if score > best_score:
                    best_score = score
                    best_key = (action, key)

            action, key = best_key
            if key not in node.children:
                step = apply_placement(node.board_bb, action.placement, node.combo)
                new_tray = list(node.tray)
                new_tray[action.slot] = tuple()
                if not any(new_tray):
                    new_tray = list(random_pieces(3, rng))
                child = MCTSNode(
                    board_bb=step.board_bb,
                    tray=tuple(new_tray),
                    combo=step.combo,
                    score_so_far=node.score_so_far + step.score_gain,
                )
                node.children[key] = child
                path.append(child)
                node = child
                break

            path.append(node.children[key])
            node = node.children[key]

        # --- Simulation (rollout) ---
        value = _rollout_value(node.board_bb, node.combo, node.score_so_far, max_turns, weights, rng)

        # --- Backpropagation ---
        for n in path:
            n.visits += 1
            n.value_sum += value

    elapsed = time.perf_counter() - start

    root_children: List[MCTSChildStats] = []
    best_action: Optional[MCTSAction] = None
    best_visits = -1
    for action in _legal_actions(root.board_bb, root_tray):
        key = _action_key(action)
        child = root.children.get(key)
        visits = child.visits if child else 0
        mean_value = child.mean_value if child else 0.0
        root_children.append(
            MCTSChildStats(slot=action.slot, placement=action.placement, visits=visits, mean_value=mean_value)
        )
        if visits > best_visits:
            best_visits = visits
            best_action = action

    return MCTSResult(
        best_action=best_action,
        root_children=root_children,
        iterations=iters_done,
        elapsed_sec=elapsed,
    )

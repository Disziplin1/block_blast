"""
solver.py
=========
search.py + heuristic.py + simulation.py 를 결합하여,
"지금 어떤 블록을 어디에 두어야 하는가"에 대한 최종 추천을 생성하는
최상위 오케스트레이션 모듈.

GUI/오버레이는 이 모듈의 solve() 함수만 호출하면 된다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from config import AppConfig, CONFIG, HeuristicWeights
from heuristic import (
    average_placeability,
    evaluate,
    evaluate_with_breakdown,
    expandable_space_count,
    find_empty_regions,
    largest_empty_rectangle,
)
from mcts import MCTSResult, mcts_search
from risk_analysis import RiskMap, analyze_risk, compute_placement_heatmap
from search import SearchResult, best_first_moves, can_place_anywhere
from simulation import SimulationResult, monte_carlo_evaluate
from utils import (
    PIECE_LIBRARY,
    PieceCells,
    grid_to_bitboard,
    piece_grid_to_cells,
    popcount,
    TOTAL_CELLS,
)


RISK_LEVELS = ["Safe", "Good", "Risky", "Danger", "Critical"]
SURVIVAL_STARS = ["★☆☆☆☆", "★★☆☆☆", "★★★☆☆", "★★★★☆", "★★★★★"]


@dataclass
class MoveRecommendation:
    rank: int
    piece_index: int            # 트레이에서의 블록 인덱스 (0,1,2)
    anchor_row: int
    anchor_col: int
    cell_count: int

    # 우측 패널 표시 항목
    expected_score: float
    expected_combo: int
    lines_cleared: int
    survival_rating: int         # 1~5
    survival_stars: str
    board_stability: float
    future_space: int
    risk_level: str
    dead_area: int
    largest_empty_region: int
    placement_freedom: int

    # 공간 품질 지표 (#6)
    flexibility_score: float = 0.0   # 표준 블록 라이브러리 평균 배치 가능 수
    mobility_score: int = 0          # 남은 트레이 블록들의 총 배치 가능 수
    expansion_score: int = 0         # 확장 가능한 빈 공간 셀 수
    fragmentation_index: float = 0.0 # 0(양호)~1(심각)

    confidence: float = 0.0          # 0~100, 추천 신뢰도

    reasons: List[str] = field(default_factory=list)
    heuristic_score: float = 0.0
    monte_carlo: Optional[SimulationResult] = None

    # 시퀀스 전체 (디버그/오버레이 화살표용): 이 추천을 따를 경우의 전체 배치 순서
    full_sequence: Optional[SearchResult] = None


@dataclass
class SolveResult:
    recommendations: List[MoveRecommendation]
    board_empty_cells: int
    overall_placement_freedom: int
    evaluation_time_sec: float
    game_over_risk: bool

    # 위험 지역 분석 (#5) - 현재(이동 전) 보드 기준
    risk_map: Optional[RiskMap] = None

    # 추천 위치 Heat Map (#7) - piece_index -> 8x8 정규화 점수 배열 (NaN=배치불가)
    heatmaps: Dict[int, np.ndarray] = field(default_factory=dict)

    # 다중 턴 MCTS 결과 (#1, Beam Search 와 병행)
    mcts_result: Optional[MCTSResult] = None


def _risk_level_from_end_prob(end_prob: float) -> str:
    if end_prob < 0.05:
        return "Safe"
    if end_prob < 0.20:
        return "Good"
    if end_prob < 0.40:
        return "Risky"
    if end_prob < 0.70:
        return "Danger"
    return "Critical"


def _survival_rating_from_ratio(ratio: float) -> int:
    if ratio >= 0.9:
        return 5
    if ratio >= 0.7:
        return 4
    if ratio >= 0.5:
        return 3
    if ratio >= 0.3:
        return 2
    return 1


def _normalize_pieces(pieces: Sequence) -> List[PieceCells]:
    """np.ndarray / list-of-list / PieceCells(tuple of tuples) 입력을 모두 PieceCells로 변환."""
    normalized: List[PieceCells] = []
    for p in pieces:
        if p is None:
            normalized.append(tuple())
            continue
        if isinstance(p, tuple) and (len(p) == 0 or isinstance(p[0], tuple)):
            normalized.append(p)  # 이미 PieceCells
            continue
        # numpy array 또는 list of list
        arr = np.asarray(p)
        if arr.size == 0:
            normalized.append(tuple())
            continue
        normalized.append(piece_grid_to_cells(arr.tolist()))
    return normalized


def solve(
    board: np.ndarray,
    pieces: Sequence,
    config: Optional[AppConfig] = None,
    run_monte_carlo: bool = True,
) -> SolveResult:
    """
    현재 보드(8x8 0/1 배열)와 트레이의 블록들(0/1 2차원 배열 최대 3개)을 받아
    추천 결과(SolveResult)를 반환한다.
    """
    cfg = config or CONFIG
    weights = cfg.weights
    start_time = time.perf_counter()

    board_bb = grid_to_bitboard(np.asarray(board, dtype=np.int8))
    piece_cells_list = _normalize_pieces(pieces)

    overall_freedom = sum(
        len(_valid_count_safe(board_bb, p)) for p in piece_cells_list if p
    )
    empty = TOTAL_CELLS - popcount(board_bb)
    game_over_risk = all(not can_place_anywhere(board_bb, p) for p in piece_cells_list if p)

    def heuristic_fn(bb, score_gain, combo, remaining):
        return evaluate(bb, score_gain, combo, remaining, weights)

    raw_results = best_first_moves(
        board_bb,
        piece_cells_list,
        heuristic_fn=heuristic_fn,
        beam_width=cfg.solver.beam_width,
        time_budget_sec=cfg.solver.time_budget_sec,
        top_k=cfg.solver.top_k,
    )

    # 현재(이동 전) 보드의 위험 지역 분석 (#5)
    current_risk_map = analyze_risk(board_bb)

    # 트레이의 각 블록에 대한 추천 위치 Heat Map (#7)
    heatmaps: Dict[int, np.ndarray] = {}
    for idx, cells in enumerate(piece_cells_list):
        if cells:
            heatmaps[idx] = compute_placement_heatmap(board_bb, cells, weights, piece_cells_list)

    # 다중 턴(3~8턴) MCTS 탐색 (#1) - Beam Search 와 병행하여 장기 생존 관점 추천 보강
    mcts_result: Optional[MCTSResult] = None
    mcts_best_key = None
    if cfg.solver.use_mcts:
        mcts_result = mcts_search(board_bb, piece_cells_list, weights)
        if mcts_result.best_action is not None:
            ba = mcts_result.best_action
            mcts_best_key = (ba.slot, ba.placement.anchor_row, ba.placement.anchor_col)

    recommendations: List[MoveRecommendation] = []

    for rank, result in enumerate(raw_results, start=1):
        breakdown = evaluate_with_breakdown(
            result.final_board_bb,
            result.total_score_gain,
            result.final_combo,
            piece_cells_list,
            weights,
        )

        region_info = find_empty_regions(result.final_board_bb)
        rect_area = largest_empty_rectangle(result.final_board_bb)
        avg_place = average_placeability(result.final_board_bb, PIECE_LIBRARY)
        mobility = sum(len(_valid_count_safe(result.final_board_bb, p)) for p in piece_cells_list if p)
        expansion = expandable_space_count(result.final_board_bb)
        result_risk_map = analyze_risk(result.final_board_bb)

        mc_result = None
        survival_rating = 3
        risk_level = "Good"
        if run_monte_carlo:
            mc_result = monte_carlo_evaluate(
                result.final_board_bb,
                [],  # 다음 턴부터는 새 블록 3개를 무작위로 가정
                trials=cfg.solver.monte_carlo_trials,
                depth=cfg.solver.monte_carlo_depth,
                weights=weights,
                time_budget_sec=max(0.05, cfg.solver.time_budget_sec / max(1, cfg.solver.top_k)),
            )
            ratio = mc_result.avg_survival_turns / max(1, cfg.solver.monte_carlo_depth)
            survival_rating = _survival_rating_from_ratio(ratio)
            risk_level = _risk_level_from_end_prob(mc_result.end_probability)
        else:
            # Monte Carlo 미실행 시, 휴리스틱 기반 근사
            move_freedom = avg_place
            if move_freedom > 20:
                survival_rating = 5
            elif move_freedom > 12:
                survival_rating = 4
            elif move_freedom > 6:
                survival_rating = 3
            elif move_freedom > 2:
                survival_rating = 2
            else:
                survival_rating = 1
            risk_level = RISK_LEVELS[5 - survival_rating] if survival_rating <= 5 else "Critical"

        reasons = list(breakdown.reasons)
        if result_risk_map.dead_area < current_risk_map.dead_area:
            reasons.append("Dead Area 감소")
        if mobility > overall_freedom:
            reasons.append("Future Flexibility Increased")
        if result_risk_map.fragmentation_index < current_risk_map.fragmentation_index:
            reasons.append("Risk Reduced")
        if mcts_best_key is not None:
            first_key = (result.first_move.piece_index, result.first_move.anchor_row, result.first_move.anchor_col)
            if first_key == mcts_best_key:
                reasons.append("MCTS 장기 생존 추천")

        first = result.first_move
        rec = MoveRecommendation(
            rank=rank,
            piece_index=first.piece_index,
            anchor_row=first.anchor_row,
            anchor_col=first.anchor_col,
            cell_count=first.cell_count,
            expected_score=result.total_score_gain,
            expected_combo=result.final_combo,
            lines_cleared=result.lines_cleared_total,
            survival_rating=survival_rating,
            survival_stars=SURVIVAL_STARS[survival_rating - 1],
            board_stability=breakdown.total,
            future_space=region_info.largest_size,
            risk_level=risk_level,
            dead_area=region_info.dead_count,
            largest_empty_region=max(region_info.largest_size, rect_area),
            placement_freedom=int(avg_place),
            flexibility_score=avg_place,
            mobility_score=mobility,
            expansion_score=expansion,
            fragmentation_index=result_risk_map.fragmentation_index,
            reasons=reasons,
            heuristic_score=result.heuristic_score,
            monte_carlo=mc_result,
            full_sequence=result,
        )
        recommendations.append(rec)

    _assign_confidence(recommendations)

    eval_time = time.perf_counter() - start_time

    return SolveResult(
        recommendations=recommendations,
        board_empty_cells=empty,
        overall_placement_freedom=overall_freedom,
        evaluation_time_sec=eval_time,
        game_over_risk=game_over_risk,
        risk_map=current_risk_map,
        heatmaps=heatmaps,
        mcts_result=mcts_result,
    )


def _assign_confidence(recommendations: List[MoveRecommendation]) -> None:
    """
    추천들의 휴리스틱 점수 분포와 생존 확률을 바탕으로 0~100 신뢰도를 부여한다.

    1위와 다른 후보들의 점수 차이가 클수록(=확실한 최선이 존재할수록),
    그리고 Monte Carlo 종료 확률이 낮을수록 신뢰도가 높아진다.
    """
    if not recommendations:
        return

    scores = [r.heuristic_score for r in recommendations]
    score_min, score_max = min(scores), max(scores)
    score_range = score_max - score_min if score_max > score_min else 1.0

    for rec in recommendations:
        relative = (rec.heuristic_score - score_min) / score_range
        survival_component = 1.0
        if rec.monte_carlo is not None:
            survival_component = 1.0 - rec.monte_carlo.end_probability
        confidence = (0.5 + 0.5 * relative) * (0.5 + 0.5 * survival_component)
        rec.confidence = round(min(1.0, max(0.0, confidence)) * 100, 1)


def _valid_count_safe(board_bb: int, cells: PieceCells):
    from search import valid_placements

    return valid_placements(board_bb, 0, cells)

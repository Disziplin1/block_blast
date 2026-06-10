"""
tests/test_upgrade.py
======================
Final Upgrade(Phase A~G) 에서 추가된 모듈들에 대한 스모크 테스트.
pytest 없이 직접 실행 가능.
"""

import sys
import os
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import CONFIG
from utils import random_pieces, grid_to_bitboard, bitboard_to_grid
from search import Placement, apply_placement
from risk_analysis import analyze_risk, compute_placement_heatmap
from mcts import mcts_search
from solver import solve, _normalize_pieces
from data_logger import DataLogger
from stats import compute_statistics, ReplayPlayer
from tuning import genetic_algorithm, build_piece_probabilities, weighted_random_pieces
from rl_env import BlockBlastEnv, encode_observation


def _sample_board():
    board = np.zeros((8, 8), dtype=np.int8)
    board[7, :6] = 1
    return board


def test_risk_analysis_and_heatmap():
    board = _sample_board()
    bb = grid_to_bitboard(board)
    risk_map = analyze_risk(bb)
    assert risk_map.grid.shape == (8, 8)
    assert 0.0 <= risk_map.fragmentation_index <= 1.0

    pieces = _normalize_pieces(random_pieces(3))
    heat = compute_placement_heatmap(bb, pieces[0], CONFIG.weights, pieces)
    assert heat.shape == (8, 8)
    print("OK: risk_analysis + heatmap")


def test_solver_phase_a_fields():
    board = _sample_board()
    pieces = random_pieces(3)
    result = solve(board, pieces, run_monte_carlo=True)

    assert result.risk_map is not None
    assert isinstance(result.heatmaps, dict)
    for rec in result.recommendations:
        assert 0.0 <= rec.confidence <= 100.0
        assert 0.0 <= rec.fragmentation_index <= 1.0
        assert rec.lines_cleared >= 0
    print("OK: solver Phase A fields (risk_map, heatmaps, confidence, fragmentation)")


def test_mcts_search():
    board = np.zeros((8, 8), dtype=np.int8)
    bb = grid_to_bitboard(board)
    pieces = _normalize_pieces(random_pieces(3))

    t0 = time.perf_counter()
    result = mcts_search(bb, pieces, CONFIG.weights, iterations=30, time_budget_sec=0.1)
    elapsed = time.perf_counter() - t0

    assert result.best_action is not None
    assert result.iterations > 0
    print(f"OK: mcts_search ran {result.iterations} iterations in {elapsed:.3f}s")


def test_data_logger_and_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        old_dir = CONFIG.logging.data_dir
        CONFIG.logging.data_dir = tmpdir
        try:
            board = _sample_board()
            pieces = random_pieces(3)
            result = solve(board, pieces, run_monte_carlo=True)

            with DataLogger(CONFIG) as dl:
                row_id = dl.log_recommendation(board, pieces, result)
                assert row_id is not None
                assert dl.count() == 1

                summary = compute_statistics(dl)
                assert summary.total_recommendations == 1

                replay = ReplayPlayer(dl)
                assert len(replay) == 1
                assert replay.board_grid().shape == (8, 8)

                json_path = dl.export_json()
                csv_path = dl.export_csv()
                assert os.path.exists(json_path)
                assert os.path.exists(csv_path)

            print("OK: data_logger + stats + replay")
        finally:
            CONFIG.logging.data_dir = old_dir


def test_tuning_and_block_probability():
    import random as _random

    result = genetic_algorithm(
        population_size=4, generations=1, episodes_per_eval=1, max_turns=5, rng=_random.Random(1)
    )
    assert result.best_fitness >= 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        old_dir = CONFIG.logging.data_dir
        CONFIG.logging.data_dir = tmpdir
        try:
            with DataLogger(CONFIG) as dl:
                probs = build_piece_probabilities(dl)
                assert abs(sum(probs.values()) - 1.0) < 1e-6
                pieces = weighted_random_pieces(probs, 3, _random.Random(2))
                assert len(pieces) == 3
        finally:
            CONFIG.logging.data_dir = old_dir

    print("OK: genetic_algorithm + block probability model")


def test_rl_env():
    env = BlockBlastEnv(np.random.RandomState(0))
    obs = env.reset()
    assert obs.shape == (139,)

    mask = env.action_mask()
    assert mask.dtype == np.bool_
    valid = np.flatnonzero(mask)
    assert len(valid) > 0

    action = int(valid[0])
    obs2, reward, done, info = env.step(action)
    assert obs2.shape == (139,)
    print(f"OK: rl_env step reward={reward} done={done}")


if __name__ == "__main__":
    test_risk_analysis_and_heatmap()
    test_solver_phase_a_fields()
    test_mcts_search()
    test_data_logger_and_stats()
    test_tuning_and_block_probability()
    test_rl_env()
    print("\nAll Final Upgrade smoke tests passed.")

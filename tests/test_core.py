"""
tests/test_core.py
===================
Phase 1 핵심 알고리즘(보드/탐색/휴리스틱/시뮬레이션/솔버) 동작 확인용
간단한 스모크 테스트. pytest 없이 직접 실행 가능.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils import grid_to_bitboard, bitboard_to_grid, piece_grid_to_cells, PIECE_LIBRARY
from search import valid_placements, apply_placement, best_first_moves
from heuristic import evaluate, evaluate_with_breakdown, find_empty_regions
from simulation import monte_carlo_evaluate
from solver import solve
from config import CONFIG


def test_grid_bitboard_roundtrip():
    grid = np.zeros((8, 8), dtype=np.int8)
    grid[0, 0] = 1
    grid[7, 7] = 1
    grid[3, 4] = 1
    bb = grid_to_bitboard(grid)
    back = bitboard_to_grid(bb)
    assert np.array_equal(grid, back)
    print("OK: grid<->bitboard roundtrip")


def test_valid_placements_empty_board():
    bb = 0
    cells = piece_grid_to_cells([[1, 1], [1, 0]])
    placements = valid_placements(bb, 0, cells)
    # L-shape (3 cells, 2x2 bounding box) on empty 8x8 -> 7*7 = 49 positions
    assert len(placements) == 49, len(placements)
    print(f"OK: valid_placements on empty board = {len(placements)}")


def test_line_clear():
    # 마지막 행(7번째 row)을 7칸 채우고, 1x1 블록으로 마지막 칸을 채워 클리어 유발
    grid = np.zeros((8, 8), dtype=np.int8)
    grid[7, 0:7] = 1
    bb = grid_to_bitboard(grid)
    cells = piece_grid_to_cells([[1]])
    placements = valid_placements(bb, 0, cells)
    # anchor at (7,7) should clear the row
    target = [p for p in placements if p.anchor_row == 7 and p.anchor_col == 7]
    assert len(target) == 1
    step = apply_placement(bb, target[0], combo=0)
    assert step.lines_cleared == 1
    assert step.rows_cleared == [7]
    # row should now be empty
    new_grid = bitboard_to_grid(step.board_bb)
    assert new_grid[7, :].sum() == 0
    print(f"OK: line clear works, score_gain={step.score_gain}")


def test_solver_basic():
    # 빈 보드 + 임의의 3개 블록
    board = np.zeros((8, 8), dtype=np.int8)
    pieces = [
        [[1, 1], [1, 1]],   # 2x2
        [[1, 1, 1]],        # 1x3
        [[1], [1], [1]],    # 3x1
    ]

    t0 = time.perf_counter()
    result = solve(board, pieces, run_monte_carlo=False)
    elapsed = time.perf_counter() - t0

    assert len(result.recommendations) > 0
    top = result.recommendations[0]
    print(f"OK: solver (no MC) returned {len(result.recommendations)} recs in {elapsed:.3f}s")
    print(f"  Top move: piece={top.piece_index} at ({top.anchor_row},{top.anchor_col})"
          f" score={top.expected_score} risk={top.risk_level} stars={top.survival_stars}")
    print(f"  Reasons: {top.reasons}")


def test_solver_with_monte_carlo():
    board = np.zeros((8, 8), dtype=np.int8)
    # 보드를 절반 정도 채워서 좀 더 현실적인 상황 만들기
    rng = np.random.default_rng(42)
    for r in range(8):
        for c in range(8):
            if rng.random() < 0.3:
                board[r, c] = 1
    # 줄이 완전히 차지 않도록 일부 칸 비우기
    board[0, :] = 0

    pieces = [
        [[1, 1], [1, 1]],
        [[1, 0], [1, 0], [1, 1]],
        [[1, 1, 1, 1]],
    ]

    t0 = time.perf_counter()
    result = solve(board, pieces, run_monte_carlo=True)
    elapsed = time.perf_counter() - t0

    print(f"OK: solver (with MC) returned {len(result.recommendations)} recs in {elapsed:.3f}s")
    for rec in result.recommendations:
        print(f"  #{rec.rank}: piece={rec.piece_index} pos=({rec.anchor_row},{rec.anchor_col}) "
              f"score={rec.expected_score:.1f} combo={rec.expected_combo} "
              f"survival={rec.survival_stars} risk={rec.risk_level} "
              f"future_space={rec.future_space} dead_area={rec.dead_area}")
        if rec.monte_carlo:
            mc = rec.monte_carlo
            print(f"      MC: avg_turns={mc.avg_survival_turns:.2f} avg_score={mc.avg_score:.1f} "
                  f"end_prob={mc.end_probability:.2f} trials={mc.trials}")
        print(f"      Reasons: {rec.reasons}")
    assert elapsed < 5.0


def test_game_over_detection():
    # 보드를 거의 가득 채워서 1x1만 들어갈 자리 1칸 남기기 -> 큰 블록은 배치 불가
    board = np.ones((8, 8), dtype=np.int8)
    board[0, 0] = 0
    pieces = [
        [[1, 1, 1, 1, 1]],  # 1x5 - 못 들어감
        [[1, 1], [1, 1]],   # 2x2 - 못 들어감
        [[1]],              # 1x1 - 들어감
    ]
    result = solve(board, pieces, run_monte_carlo=False)
    assert not result.game_over_risk  # 1x1은 들어갈 수 있음
    print("OK: game_over_risk correctly False when at least one piece fits")

    pieces2 = [
        [[1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1]],
    ]
    result2 = solve(board, pieces2, run_monte_carlo=False)
    assert result2.game_over_risk
    print("OK: game_over_risk correctly True when nothing fits")


if __name__ == "__main__":
    test_grid_bitboard_roundtrip()
    test_valid_placements_empty_board()
    test_line_clear()
    test_solver_basic()
    test_solver_with_monte_carlo()
    test_game_over_detection()
    print("\nAll Phase 1 smoke tests passed.")

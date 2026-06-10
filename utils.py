"""
utils.py
========
보드/블록 표현 변환, 비트보드 연산, 공용 자료구조 등
여러 모듈에서 공통으로 사용하는 유틸리티 함수 모음.

보드 표현
---------
- "Grid" 표현: numpy.ndarray, shape=(ROWS, COLS), dtype=int8, 값은 0(빈칸)/1(블록)
- "BitBoard" 표현: 64bit Python int. bit index = row * COLS + col (0 = LSB)
  비트가 1이면 해당 셀이 채워져 있음.

블록(Piece) 표현
----------------
- "Grid" 표현: List[List[int]] 또는 numpy.ndarray, 0/1 값
- "Cells" 표현: 정규화된 (dr, dc) 오프셋 튜플들의 정렬된 tuple.
  정규화: 모든 셀의 최소 행/열이 0이 되도록 평행이동.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

ROWS = 8
COLS = 8
TOTAL_CELLS = ROWS * COLS
FULL_BOARD = (1 << TOTAL_CELLS) - 1

# 각 행을 채우는 비트마스크 (row r -> bits [r*COLS, (r+1)*COLS) )
ROW_MASKS: Tuple[int, ...] = tuple(
    ((1 << COLS) - 1) << (r * COLS) for r in range(ROWS)
)

# 각 열을 채우는 비트마스크 (col c -> bit (r*COLS + c) for all r)
COL_MASKS: Tuple[int, ...] = tuple(
    sum(1 << (r * COLS + c) for r in range(ROWS)) for c in range(COLS)
)


# ---------------------------------------------------------------------------
# Grid <-> BitBoard 변환
# ---------------------------------------------------------------------------
def grid_to_bitboard(grid: np.ndarray) -> int:
    """8x8 numpy 배열(0/1)을 64bit 정수로 변환한다."""
    bb = 0
    flat = grid.flatten()
    for idx, v in enumerate(flat):
        if v:
            bb |= 1 << idx
    return bb


def bitboard_to_grid(bb: int) -> np.ndarray:
    """64bit 정수를 8x8 numpy 배열(0/1)로 변환한다."""
    grid = np.zeros((ROWS, COLS), dtype=np.int8)
    for idx in range(TOTAL_CELLS):
        if bb & (1 << idx):
            r, c = divmod(idx, COLS)
            grid[r, c] = 1
    return grid


def cell_index(r: int, c: int) -> int:
    return r * COLS + c


def popcount(bb: int) -> int:
    """1비트 개수 (Python 3.8+ 호환을 위해 bin().count 사용)."""
    return bin(bb).count("1")


def quick_evaluate(board_bb: int, score_gain: int, combo: int) -> float:
    """
    탐색 중 가지치기(beam pruning)에 사용하는 저비용 평가 함수.

    완전한 heuristic.evaluate() 는 보드 전체를 여러 번 스캔하므로
    탐색 중 수천 번 호출하기엔 너무 느리다. 이 함수는 popcount /
    라인 채움 수준만 확인하여 O(ROWS+COLS) 로 동작한다.
    """
    filled = popcount(board_bb)
    empty = TOTAL_CELLS - filled

    near_complete = 0
    for r in range(ROWS):
        c = popcount(board_bb & ROW_MASKS[r])
        if 0 < COLS - c <= 1:
            near_complete += 1
    for c_ in range(COLS):
        cc = popcount(board_bb & COL_MASKS[c_])
        if 0 < ROWS - cc <= 1:
            near_complete += 1

    return score_gain * 1.0 + empty * 0.5 + near_complete * 5.0 + combo * 8.0


# ---------------------------------------------------------------------------
# Piece (블록) 표현
# ---------------------------------------------------------------------------
PieceCells = Tuple[Tuple[int, int], ...]


def normalize_piece(cells: Iterable[Tuple[int, int]]) -> PieceCells:
    """(dr, dc) 좌표 목록을 최소 행/열이 0이 되도록 정규화하고 정렬한다."""
    cells = list(cells)
    if not cells:
        return tuple()
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    return tuple(norm)


def piece_grid_to_cells(piece_grid: Sequence[Sequence[int]]) -> PieceCells:
    """2차원 배열(0/1) 형태의 블록을 (dr, dc) 셀 목록으로 변환한다."""
    cells = []
    for r, row in enumerate(piece_grid):
        for c, v in enumerate(row):
            if v:
                cells.append((r, c))
    return normalize_piece(cells)


def piece_cells_to_grid(cells: PieceCells) -> np.ndarray:
    """(dr, dc) 셀 목록을 최소 크기의 0/1 numpy 배열로 변환한다."""
    if not cells:
        return np.zeros((1, 1), dtype=np.int8)
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    grid = np.zeros((max_r + 1, max_c + 1), dtype=np.int8)
    for r, c in cells:
        grid[r, c] = 1
    return grid


def piece_size(cells: PieceCells) -> Tuple[int, int]:
    """블록의 (height, width)를 반환한다."""
    if not cells:
        return (0, 0)
    return (max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1)


def piece_cell_count(cells: PieceCells) -> int:
    return len(cells)


# ---------------------------------------------------------------------------
# 표준 Block Blast 블록 라이브러리 (Monte Carlo 랜덤 생성용)
# 회전은 지원하지 않으므로, 게임에서 등장 가능한 모든 회전 변형을
# 별도 항목으로 포함한다.
# ---------------------------------------------------------------------------
def _shapes_from_strings(*shapes: str) -> List[PieceCells]:
    result = []
    for shape in shapes:
        rows = [r for r in shape.strip("\n").split("\n")]
        grid = [[1 if ch == "#" else 0 for ch in row] for row in rows]
        result.append(piece_grid_to_cells(grid))
    return result


PIECE_LIBRARY: List[PieceCells] = _shapes_from_strings(
    # 1x1
    "#",
    # 1x2 / 2x1
    "##",
    "#\n#",
    # 1x3 / 3x1
    "###",
    "#\n#\n#",
    # 1x4 / 4x1
    "####",
    "#\n#\n#\n#",
    # 1x5 / 5x1
    "#####",
    "#\n#\n#\n#\n#",
    # 2x2 정사각형
    "##\n##",
    # 3x3 정사각형
    "###\n###\n###",
    # L 모양 (4가지 회전)
    "#.\n#.\n##",
    "##\n#.\n#.",
    "##\n.#\n.#",
    ".#\n.#\n##",
    # J 모양 (4가지 회전, L의 거울상)
    ".#\n.#\n##",
    "##\n.#\n.#",
    "##\n#.\n#.",
    "#.\n#.\n##",
    # T 모양 (4가지 회전)
    "###\n.#.",
    ".#\n##\n.#",
    ".#.\n###",
    "#.\n##\n#.",
    # S / Z 모양
    ".##\n##.",
    "##.\n.##",
    "#.\n##\n.#",
    ".#\n##\n#.",
    # 큰 L (3칸짜리 변)
    "#..\n#..\n###",
    "###\n#..\n#..",
    "###\n..#\n..#",
    "..#\n..#\n###",
    # 십자 모양 (plus)
    ".#.\n###\n.#.",
)


def random_piece(rng=None) -> PieceCells:
    """PIECE_LIBRARY 에서 무작위 블록을 하나 선택한다."""
    import random as _random

    r = rng or _random
    return r.choice(PIECE_LIBRARY)


def random_pieces(n: int, rng=None) -> List[PieceCells]:
    return [random_piece(rng) for _ in range(n)]

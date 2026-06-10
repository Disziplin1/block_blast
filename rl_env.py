"""
rl_env.py
=========
강화학습(PPO/DQN/A2C 등) 적용을 위한 State/Action/Reward 환경 인터페이스
스캐폴딩 (#6).

이 모듈은 별도의 RL 프레임워크(예: gymnasium)에 대한 의존성 없이,
동일한 패턴(reset/step/observation/action_space)을 따르는 최소한의
환경(BlockBlastEnv)을 제공한다. 추후 실제 학습 루프에서는 이
인터페이스를 gymnasium.Env 등으로 감싸기만 하면 된다.

State (관측값)
--------------
- board: 8x8 0/1 배열
- tray: 최대 3개의 블록(각각 0/1 그리드, 가변 크기 -> 5x5 로 패딩)
encode_observation() 으로 위 정보를 고정 길이 벡터로 인코딩한다.

Action (행동)
-------------
이산 행동 공간: action_id = piece_slot * 64 + (row * 8 + col)
  - piece_slot: 0~2 (트레이 내 블록 인덱스)
  - row, col: 배치 anchor 위치 (0~7)
총 3 * 64 = 192 개의 행동.
유효하지 않은 행동(배치 불가)은 action_mask() 로 걸러낸다.

Reward (보상)
-------------
- 기본: 이번 스텝에서 얻은 score_gain
- 라인 클리어/콤보에 추가 보너스
- 게임 오버 시 큰 패널티
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from search import Placement, apply_placement, can_place_anywhere, valid_placements
from utils import (
    COLS,
    PieceCells,
    ROWS,
    bitboard_to_grid,
    grid_to_bitboard,
    piece_cells_to_grid,
    random_pieces,
)


NUM_ACTIONS = 3 * ROWS * COLS  # 192
PIECE_GRID_SIZE = 5             # 블록 그리드를 5x5 로 패딩 (PIECE_LIBRARY 최대 크기 커버)


@dataclass
class StepInfo:
    lines_cleared: int
    score_gain: int
    combo: int
    game_over: bool


def decode_action(action_id: int) -> Tuple[int, int, int]:
    """action_id -> (piece_slot, row, col)."""
    cell = ROWS * COLS
    slot = action_id // cell
    rem = action_id % cell
    row = rem // COLS
    col = rem % COLS
    return slot, row, col


def encode_action(slot: int, row: int, col: int) -> int:
    return slot * ROWS * COLS + row * COLS + col


def _pad_piece_grid(cells: PieceCells) -> np.ndarray:
    grid = np.zeros((PIECE_GRID_SIZE, PIECE_GRID_SIZE), dtype=np.float32)
    if not cells:
        return grid
    piece_grid = np.array(piece_cells_to_grid(cells), dtype=np.float32)
    h, w = piece_grid.shape
    h = min(h, PIECE_GRID_SIZE)
    w = min(w, PIECE_GRID_SIZE)
    grid[:h, :w] = piece_grid[:h, :w]
    return grid


def encode_observation(board_bb: int, tray: Sequence[PieceCells]) -> np.ndarray:
    """
    (board 64 + tray 3 * 5x5=25 = 64 + 75 = 139) 길이의 1차원 float32 벡터.
    신경망 입력으로 바로 사용할 수 있다.
    """
    board = bitboard_to_grid(board_bb).astype(np.float32).flatten()
    pieces_flat = []
    for i in range(3):
        cells = tray[i] if i < len(tray) else tuple()
        pieces_flat.append(_pad_piece_grid(cells).flatten())
    return np.concatenate([board] + pieces_flat)


class BlockBlastEnv:
    """
    최소한의 State/Action/Reward 인터페이스를 제공하는 Block Blast 환경.

    사용 예시
    ---------
    >>> env = BlockBlastEnv()
    >>> obs = env.reset()
    >>> mask = env.action_mask()
    >>> action = int(np.argmax(mask))  # 임의의 유효 행동
    >>> obs, reward, done, info = env.step(action)
    """

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self._np_rng = rng or np.random.RandomState()
        self._py_rng = __import__("random").Random(int(self._np_rng.randint(0, 2**31 - 1)))
        self.board_bb: int = 0
        self.tray: List[PieceCells] = []
        self.combo: int = 0
        self.score: float = 0.0
        self.done: bool = False

    # ------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.board_bb = 0
        self.tray = random_pieces(3, self._py_rng)
        self.combo = 0
        self.score = 0.0
        self.done = False
        return encode_observation(self.board_bb, self.tray)

    # ------------------------------------------------------------
    def action_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        for slot, cells in enumerate(self.tray):
            if not cells:
                continue
            for placement in valid_placements(self.board_bb, slot, cells):
                mask[encode_action(slot, placement.anchor_row, placement.anchor_col)] = True
        return mask

    # ------------------------------------------------------------
    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, StepInfo]:
        if self.done:
            raise RuntimeError("step() called after episode finished. Call reset() first.")

        slot, row, col = decode_action(action_id)
        if slot >= len(self.tray) or not self.tray[slot]:
            return self._invalid_action_result()

        cells = self.tray[slot]
        placement = self._find_placement(cells, slot, row, col)
        if placement is None:
            return self._invalid_action_result()

        step = apply_placement(self.board_bb, placement, self.combo)
        self.board_bb = step.board_bb
        self.combo = step.combo
        self.score += step.score_gain
        self.tray[slot] = tuple()

        reward = float(step.score_gain)
        if step.lines_cleared > 0:
            reward += 5.0 * step.lines_cleared + 5.0 * (step.combo - 1 if step.combo > 1 else 0)

        if not any(self.tray):
            self.tray = random_pieces(3, self._py_rng)

        game_over = all(not can_place_anywhere(self.board_bb, p) for p in self.tray if p)
        if game_over:
            reward -= 50.0
            self.done = True

        info = StepInfo(
            lines_cleared=step.lines_cleared,
            score_gain=step.score_gain,
            combo=step.combo,
            game_over=game_over,
        )
        return encode_observation(self.board_bb, self.tray), reward, self.done, info

    # ------------------------------------------------------------
    def _find_placement(self, cells: PieceCells, slot: int, row: int, col: int) -> Optional[Placement]:
        for placement in valid_placements(self.board_bb, slot, cells):
            if placement.anchor_row == row and placement.anchor_col == col:
                return placement
        return None

    def _invalid_action_result(self) -> Tuple[np.ndarray, float, bool, StepInfo]:
        """유효하지 않은 행동을 선택한 경우, 큰 패널티를 주고 에피소드를 종료한다."""
        self.done = True
        info = StepInfo(lines_cleared=0, score_gain=0, combo=self.combo, game_over=True)
        return encode_observation(self.board_bb, self.tray), -100.0, True, info

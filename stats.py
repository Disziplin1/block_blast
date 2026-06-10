"""
stats.py
========
data_logger.DataLogger 에 누적된 추천/플레이 로그로부터:

1. 통계 집계 (#4 Statistics): 최고/평균 점수, 평균 생존 턴, 평균 콤보,
   평균 클리어 라인 수, 추천 수락률(accuracy), 평균 탐색 시간,
   평균 신뢰도, 평균 위험도.
2. 리플레이 (#3 Replay): 기록된 (board, pieces, recommendation) 시퀀스를
   순서대로 재생할 수 있는 ReplayPlayer.

를 제공한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from data_logger import DataLogger
from search import Placement, apply_placement
from utils import grid_to_bitboard, piece_grid_to_cells


RISK_LEVEL_SCORE = {
    "Safe": 1,
    "Good": 2,
    "Risky": 3,
    "Danger": 4,
    "Critical": 5,
}


@dataclass
class StatsSummary:
    total_recommendations: int
    best_score: float
    avg_score: float
    avg_survival_turns: float
    avg_combo: float
    avg_lines_cleared: float
    avg_confidence: float
    avg_risk_score: float          # 1(Safe)~5(Critical) 평균
    avg_search_time_sec: float
    recommendation_accuracy: float  # 0~1, 다음 기록의 보드가 추천을 따랐는지 비율


def _board_after_recommendation(row: Dict[str, Any]) -> Optional[np.ndarray]:
    """row 의 board + 추천된 첫 배치를 적용한 결과 보드를 반환한다 (라인 클리어 포함)."""
    board = np.array(json.loads(row["board_json"]), dtype=np.int8)
    pieces = json.loads(row["pieces_json"])

    piece_index = row["piece_index"]
    if piece_index is None or piece_index >= len(pieces) or pieces[piece_index] is None:
        return None

    cells = piece_grid_to_cells(pieces[piece_index]) if _looks_like_grid(pieces[piece_index]) else tuple(
        tuple(c) for c in pieces[piece_index]
    )
    if not cells:
        return None

    board_bb = grid_to_bitboard(board)
    mask = 0
    for dr, dc in cells:
        r = row["anchor_row"] + dr
        c = row["anchor_col"] + dc
        if not (0 <= r < 8 and 0 <= c < 8):
            return None
        mask |= 1 << (r * 8 + c)

    placement = Placement(piece_index=piece_index, anchor_row=row["anchor_row"], anchor_col=row["anchor_col"], mask=mask, cell_count=len(cells))
    step = apply_placement(board_bb, placement, combo=0)

    from utils import bitboard_to_grid

    return bitboard_to_grid(step.board_bb)


def _looks_like_grid(piece_data) -> bool:
    """piece_data 가 (행,열) 셀 오프셋 목록이 아니라 0/1 2차원 그리드인지 추정."""
    arr = np.asarray(piece_data)
    return arr.ndim == 2 and arr.shape[1] != 2


def compute_statistics(logger: DataLogger) -> StatsSummary:
    rows = logger.fetch_all()
    if not rows:
        return StatsSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    scores = [r["score_gain"] for r in rows]
    survivals = [r["survival_turns"] for r in rows if r["survival_turns"] is not None]
    combos = [r["combo"] for r in rows]
    lines = [r["lines_cleared"] for r in rows]
    confidences = [r["confidence"] for r in rows]
    risks = [RISK_LEVEL_SCORE.get(r["risk_level"], 3) for r in rows]
    search_times = [r["search_time_sec"] for r in rows if r["search_time_sec"] is not None]

    followed = 0
    comparable = 0
    for cur_row, next_row in zip(rows, rows[1:]):
        predicted = _board_after_recommendation(cur_row)
        if predicted is None:
            continue
        actual = np.array(json.loads(next_row["board_json"]), dtype=np.int8)
        comparable += 1
        if np.array_equal(predicted, actual):
            followed += 1

    return StatsSummary(
        total_recommendations=len(rows),
        best_score=max(scores),
        avg_score=sum(scores) / len(scores),
        avg_survival_turns=(sum(survivals) / len(survivals)) if survivals else 0.0,
        avg_combo=sum(combos) / len(combos),
        avg_lines_cleared=sum(lines) / len(lines),
        avg_confidence=sum(confidences) / len(confidences),
        avg_risk_score=sum(risks) / len(risks),
        avg_search_time_sec=(sum(search_times) / len(search_times)) if search_times else 0.0,
        recommendation_accuracy=(followed / comparable) if comparable else 0.0,
    )


class ReplayPlayer:
    """기록된 로그를 순서대로 재생(playback)할 수 있는 간단한 플레이어."""

    def __init__(self, logger: DataLogger):
        self.rows = logger.fetch_all()
        self.index = 0

    def __len__(self) -> int:
        return len(self.rows)

    def reset(self) -> None:
        self.index = 0

    def current(self) -> Optional[Dict[str, Any]]:
        if 0 <= self.index < len(self.rows):
            return self.rows[self.index]
        return None

    def step(self) -> Optional[Dict[str, Any]]:
        """다음 기록으로 이동 후 반환. 끝에 도달하면 None."""
        if self.index + 1 >= len(self.rows):
            return None
        self.index += 1
        return self.current()

    def board_grid(self, row: Optional[Dict[str, Any]] = None) -> np.ndarray:
        row = row or self.current()
        return np.array(json.loads(row["board_json"]), dtype=np.int8)

    def pieces_grids(self, row: Optional[Dict[str, Any]] = None) -> List[Any]:
        row = row or self.current()
        return json.loads(row["pieces_json"])

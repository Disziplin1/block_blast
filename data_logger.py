"""
data_logger.py
================
자동 학습/데이터 축적을 위한 추천 로깅 시스템 (#2).

매 추천(파이프라인 실행)마다 보드 상태, 트레이 블록, 선택된(1순위)
이동, 점수/콤보/클리어 라인 수, 남은 공간/미래 공간, 생존 턴, 위험도,
신뢰도, 평가 점수, 타임스탬프를 SQLite DB 에 기록한다.

이렇게 모인 데이터는 이후 Phase E(자기-튜닝, 블록 확률 모델)와
Phase F(RL 환경)의 학습 데이터로 재사용된다.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from config import AppConfig, CONFIG
from solver import SolveResult


SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    board_json TEXT NOT NULL,
    pieces_json TEXT NOT NULL,
    piece_index INTEGER,
    anchor_row INTEGER,
    anchor_col INTEGER,
    score_gain REAL,
    combo INTEGER,
    lines_cleared INTEGER,
    remaining_space INTEGER,
    future_space INTEGER,
    survival_turns REAL,
    risk_level TEXT,
    confidence REAL,
    evaluation_score REAL,
    search_time_sec REAL
);
"""


class DataLogger:
    """추천 결과를 SQLite DB(및 필요 시 JSON)로 기록하는 로거."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or CONFIG
        self.data_dir = Path(self.config.logging.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / self.config.logging.db_filename
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------
    def log_recommendation(
        self,
        board: np.ndarray,
        pieces: Sequence[Any],
        solve_result: SolveResult,
        rank: int = 1,
    ) -> Optional[int]:
        """solve_result 의 rank 위 추천을 한 행으로 기록하고 row id 를 반환한다."""
        rec = next((r for r in solve_result.recommendations if r.rank == rank), None)
        if rec is None:
            return None

        survival_turns = rec.monte_carlo.avg_survival_turns if rec.monte_carlo else None

        pieces_serializable = []
        for p in pieces:
            if p is None:
                pieces_serializable.append(None)
            else:
                pieces_serializable.append(np.asarray(p).tolist())

        row = (
            time.time(),
            json.dumps(np.asarray(board).tolist()),
            json.dumps(pieces_serializable),
            rec.piece_index,
            rec.anchor_row,
            rec.anchor_col,
            float(rec.expected_score),
            int(rec.expected_combo),
            int(rec.lines_cleared),
            int(solve_result.board_empty_cells),
            int(rec.future_space),
            survival_turns,
            rec.risk_level,
            float(rec.confidence),
            float(rec.heuristic_score),
            float(solve_result.evaluation_time_sec),
        )

        cur = self._conn.execute(
            """INSERT INTO recommendations
               (timestamp, board_json, pieces_json, piece_index, anchor_row, anchor_col,
                score_gain, combo, lines_cleared, remaining_space, future_space,
                survival_turns, risk_level, confidence, evaluation_score, search_time_sec)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            row,
        )
        self._conn.commit()
        return cur.lastrowid

    # ------------------------------------------------------------
    def fetch_all(self) -> List[Dict[str, Any]]:
        cur = self._conn.execute("SELECT * FROM recommendations ORDER BY id")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM recommendations")
        return int(cur.fetchone()[0])

    # ------------------------------------------------------------
    def export_json(self, out_path: Optional[str] = None) -> str:
        """전체 기록을 JSON 파일로 내보내고, 경로를 반환한다."""
        out_path = out_path or str(self.data_dir / "play_log.json")
        rows = self.fetch_all()
        Path(out_path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    def export_csv(self, out_path: Optional[str] = None) -> str:
        """전체 기록을 CSV 파일로 내보내고, 경로를 반환한다."""
        import csv

        out_path = out_path or str(self.data_dir / "play_log.csv")
        rows = self.fetch_all()
        if not rows:
            Path(out_path).write_text("", encoding="utf-8")
            return out_path

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return out_path

    # ------------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DataLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

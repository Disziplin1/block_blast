"""
logger.py
=========
프로젝트 전역에서 사용하는 로깅 유틸리티.

파이프라인 단계별 (Frame, Detect, Recognize, Search, Evaluate, Recommend,
Overlay) 소요 시간을 기록하기 위한 StageTimer 도 함께 제공한다.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from typing import Dict, Iterator, Optional

from config import CONFIG


_LOGGERS: Dict[str, logging.Logger] = {}


def get_logger(name: str = "block_blast_ai") -> logging.Logger:
    """이름별 싱글턴 로거를 반환한다."""
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, CONFIG.logging.log_level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if CONFIG.logging.log_to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if CONFIG.logging.log_to_file:
        os.makedirs(CONFIG.logging.log_dir, exist_ok=True)
        fh = RotatingFileHandler(
            os.path.join(CONFIG.logging.log_dir, f"{name}.log"),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger


class StageTimer:
    """
    파이프라인 단계별 소요 시간을 측정/기록하는 헬퍼.

    사용 예:
        timer = StageTimer()
        with timer.stage("Capture"):
            frame = capture()
        with timer.stage("Detect"):
            board = detect(frame)
        timer.log_summary()
    """

    def __init__(self, logger_name: str = "pipeline"):
        self.logger = get_logger(logger_name)
        self.timings: Dict[str, float] = {}
        self._start_total: Optional[float] = None

    def begin_frame(self) -> None:
        self._start_total = time.perf_counter()
        self.timings.clear()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.timings[name] = elapsed

    def total_ms(self) -> float:
        if self._start_total is None:
            return sum(self.timings.values())
        return (time.perf_counter() - self._start_total) * 1000.0

    def log_summary(self, level: int = logging.DEBUG) -> None:
        parts = [f"{k}={v:.1f}ms" for k, v in self.timings.items()]
        parts.append(f"Total={self.total_ms():.1f}ms")
        self.logger.log(level, " | ".join(parts))

    def as_dict(self) -> Dict[str, float]:
        d = dict(self.timings)
        d["Total"] = self.total_ms()
        return d

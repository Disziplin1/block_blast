"""
capture.py
==========
mss 기반 실시간 화면 캡처 모듈.

별도 스레드에서 지정된 화면 영역(아이폰 미러링 창 등)을 지정 FPS로
지속적으로 캡처하여 가장 최근 프레임을 보관한다. 메인/워커 스레드는
get_latest_frame() 으로 락을 짧게 잡고 최신 프레임의 복사본을 가져간다.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np

from config import AppConfig, CONFIG
from logger import get_logger

logger = get_logger("capture")


class ScreenCapture:
    """백그라운드 스레드에서 화면을 캡처하는 클래스."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or CONFIG
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._fps: float = 0.0

    # ------------------------------------------------------------------
    def _region_dict(self) -> Dict[str, int]:
        x, y, w, h = self.config.capture.region
        return {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}

    def set_region(self, x: int, y: int, w: int, h: int) -> None:
        self.config.capture.region = (x, y, w, h)

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="ScreenCapture")
        self._thread.start()
        logger.info("ScreenCapture started region=%s", self.config.capture.region)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("ScreenCapture stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    def _run(self) -> None:
        with mss.mss() as sct:
            frame_count = 0
            fps_timer = time.perf_counter()

            while self._running:
                loop_start = time.perf_counter()
                target_fps = max(1, self.config.capture.target_fps)
                target_dt = 1.0 / target_fps

                region = self._region_dict()
                try:
                    raw = sct.grab(region)
                except Exception as exc:  # pragma: no cover - 환경 의존적 예외
                    logger.error("capture error: %s", exc)
                    time.sleep(0.5)
                    continue

                frame = np.asarray(raw)  # BGRA, shape (h, w, 4)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                downscale = self.config.capture.downscale
                if downscale and downscale != 1.0:
                    frame = cv2.resize(
                        frame,
                        None,
                        fx=downscale,
                        fy=downscale,
                        interpolation=cv2.INTER_AREA,
                    )

                with self._lock:
                    self._latest_frame = frame
                    self._latest_ts = time.perf_counter()

                frame_count += 1
                now = time.perf_counter()
                if now - fps_timer >= 1.0:
                    self._fps = frame_count / (now - fps_timer)
                    frame_count = 0
                    fps_timer = now

                elapsed = time.perf_counter() - loop_start
                sleep_time = target_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    # ------------------------------------------------------------------
    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """가장 최근 프레임의 복사본과 타임스탬프(perf_counter)를 반환한다."""
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._latest_ts

    @property
    def fps(self) -> float:
        return self._fps

    # ------------------------------------------------------------------
    @staticmethod
    def list_monitors() -> List[Dict[str, int]]:
        """연결된 모니터(가상 화면 포함) 목록을 반환한다."""
        with mss.mss() as sct:
            return list(sct.monitors)

    @staticmethod
    def grab_once(region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """캘리브레이션 등에서 1회성으로 화면을 캡처할 때 사용."""
        with mss.mss() as sct:
            if region is None:
                mon = sct.monitors[CONFIG.capture.monitor_index]
            else:
                x, y, w, h = region
                mon = {"left": x, "top": y, "width": w, "height": h}
            raw = sct.grab(mon)
            frame = np.asarray(raw)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

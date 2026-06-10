"""
gui.py
======
PyQt5 기반 메인 GUI.

기능
----
- Start / Stop / Pause / Calibration / Debug / Save / Load 버튼
- 캡처 영역 설정 (스핀박스 + 화면 드래그 선택)
- 우측 상태 패널: FPS, 탐색 시간, 추천별 점수/생존도/위험도/이유 등
- Debug 모드: 인식된 보드/블록 + 추천 오버레이를 그린 프레임 미리보기
- 실시간 파이프라인은 별도 QThread(PipelineWorker) 에서 실행되어 GUI 가 멈추지 않음
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from block_detector import BlockDetector
from board_detector import BoardDetector
from capture import ScreenCapture
from config import AppConfig, CONFIG
from data_logger import DataLogger
from logger import StageTimer, get_logger
from overlay import OverlayWindow, PYQT_AVAILABLE, render_overlay_cv2
from heuristic import evaluate_with_breakdown
from solver import SolveResult, _normalize_pieces, solve
from utils import PieceCells

logger = get_logger("gui")


# ---------------------------------------------------------------------------
# 화면 영역 드래그 선택용 투명 위젯
# ---------------------------------------------------------------------------
class RegionSelector(QtWidgets.QWidget):
    """전체 가상 화면을 덮는 반투명 창에서 사용자가 드래그한 사각형 영역을 선택."""

    region_selected = QtCore.pyqtSignal(int, int, int, int)  # x, y, w, h (전역 좌표)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setCursor(QtCore.Qt.CrossCursor)

        # 모든 모니터를 포함하는 가상 데스크탑 전체 영역
        screen = QtWidgets.QApplication.primaryScreen()
        geo = screen.virtualGeometry()
        self.setGeometry(geo)
        self._origin_global = geo.topLeft()

        self._start: Optional[QtCore.QPoint] = None
        self._end: Optional[QtCore.QPoint] = None

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 80))
        if self._start and self._end:
            rect = QtCore.QRect(self._start, self._end).normalized()
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0), 2))
            painter.setBrush(QtGui.QColor(0, 255, 0, 40))
            painter.drawRect(rect)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._start = event.pos()
        self._end = event.pos()
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        self._end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._start is None:
            self.close()
            return
        rect = QtCore.QRect(self._start, event.pos()).normalized()
        global_x = self.geometry().x() + rect.x()
        global_y = self.geometry().y() + rect.y()
        if rect.width() > 5 and rect.height() > 5:
            self.region_selected.emit(global_x, global_y, rect.width(), rect.height())
        self.close()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()


# ---------------------------------------------------------------------------
# 백그라운드 파이프라인 워커
# ---------------------------------------------------------------------------
class PipelineResult:
    def __init__(
        self,
        frame: np.ndarray,
        board: np.ndarray,
        pieces: List[PieceCells],
        solve_result: SolveResult,
        timings: dict,
        fps: float,
    ):
        self.frame = frame
        self.board = board
        self.pieces = pieces
        self.solve_result = solve_result
        self.timings = timings
        self.fps = fps


class PipelineWorker(QtCore.QThread):
    """캡처 -> 인식 -> 탐색/평가 -> 추천 파이프라인을 별도 스레드에서 반복 실행."""

    result_ready = QtCore.pyqtSignal(object)  # PipelineResult
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(
        self,
        capture: ScreenCapture,
        board_detector: BoardDetector,
        block_detector: BlockDetector,
        config: AppConfig,
        parent=None,
    ):
        super().__init__(parent)
        self.capture = capture
        self.board_detector = board_detector
        self.block_detector = block_detector
        self.config = config
        self._running = False
        self._paused = False
        self._mutex = QtCore.QMutex()
        self._data_logger: Optional[DataLogger] = None
        if self.config.logging.data_logging_enabled:
            self._data_logger = DataLogger(self.config)

    def start_pipeline(self) -> None:
        self._running = True
        self._paused = False
        if not self.isRunning():
            self.start()

    def stop_pipeline(self) -> None:
        self._running = False
        self.wait(2000)

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def run(self) -> None:  # noqa: D401 - QThread entrypoint
        timer = StageTimer("pipeline")
        min_interval = max(0.0, self.config.gui.update_interval_ms / 1000.0)
        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            loop_start = time.perf_counter()
            timer.begin_frame()
            try:
                with timer.stage("Frame"):
                    frame, ts = self.capture.get_latest_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                with timer.stage("Detect"):
                    board = self.board_detector.detect(frame)

                with timer.stage("Recognize"):
                    pieces = self.block_detector.detect_pieces_cells(frame)

                with timer.stage("Search+Evaluate"):
                    solve_result = solve(board, pieces, self.config)

                with timer.stage("Recommend"):
                    pass  # solve_result.recommendations 자체가 추천 결과

                if self._data_logger is not None:
                    try:
                        self._data_logger.log_recommendation(board, pieces, solve_result)
                    except Exception:  # pragma: no cover - 로깅 실패는 파이프라인을 막지 않음
                        logger.exception("Data logging failed")

                result = PipelineResult(
                    frame=frame,
                    board=board,
                    pieces=pieces,
                    solve_result=solve_result,
                    timings=timer.as_dict(),
                    fps=self.capture.fps,
                )
                self.result_ready.emit(result)

            except Exception as exc:  # pragma: no cover - 방어적 처리
                logger.exception("Pipeline error: %s", exc)
                self.error_occurred.emit(str(exc))
                time.sleep(0.2)
                continue

            elapsed = time.perf_counter() - loop_start
            sleep_time = min_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


# ---------------------------------------------------------------------------
# 메인 윈도우
# ---------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = CONFIG
        self.setWindowTitle(self.config.gui.window_title)
        self.resize(*self.config.gui.window_size)

        self.capture = ScreenCapture(self.config)
        self.board_detector = BoardDetector(self.config)
        self.block_detector = BlockDetector(self.config)
        self.worker = PipelineWorker(self.capture, self.board_detector, self.block_detector, self.config)
        self.worker.result_ready.connect(self.on_result)
        self.worker.error_occurred.connect(self.on_error)

        self.overlay_window: Optional[OverlayWindow] = None
        if PYQT_AVAILABLE:
            self.overlay_window = OverlayWindow(self.config)

        self._debug_mode = False
        self._last_result: Optional[PipelineResult] = None

        self._build_ui()
        self._fit_to_screen()

    # ------------------------------------------------------------------
    def _fit_to_screen(self) -> None:
        """창이 화면 해상도보다 크게 시작해 화면 밖으로 나가는 것을 방지하고,
        세로 크기 조절이 가능하도록 화면 가용 영역에 맞춰 크기/위치를 조정한다."""
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        w = min(self.width(), avail.width())
        h = min(self.height(), avail.height())
        self.resize(w, h)
        x = avail.x() + max(0, (avail.width() - w) // 2)
        y = avail.y() + max(0, (avail.height() - h) // 2)
        self.move(x, y)

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # ---- 좌측: 컨트롤 + 디버그 미리보기 ----
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        # 컨트롤 버튼 툴바
        toolbar = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_calibration = QtWidgets.QPushButton("Calibration")
        self.btn_debug = QtWidgets.QPushButton("Debug")
        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_load = QtWidgets.QPushButton("Load")

        self.btn_debug.setCheckable(True)
        self.btn_pause.setCheckable(True)

        for btn in (
            self.btn_start,
            self.btn_stop,
            self.btn_pause,
            self.btn_calibration,
            self.btn_debug,
            self.btn_save,
            self.btn_load,
        ):
            toolbar.addWidget(btn)
        left_layout.addLayout(toolbar)

        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_pause.toggled.connect(self.on_pause_toggled)
        self.btn_calibration.clicked.connect(self.on_calibration)
        self.btn_debug.toggled.connect(self.on_debug_toggled)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_load.clicked.connect(self.on_load)

        # 캡처 영역 설정
        region_group = QtWidgets.QGroupBox("Capture Region")
        region_layout = QtWidgets.QHBoxLayout(region_group)
        self.spin_x = QtWidgets.QSpinBox()
        self.spin_y = QtWidgets.QSpinBox()
        self.spin_w = QtWidgets.QSpinBox()
        self.spin_h = QtWidgets.QSpinBox()
        for spin in (self.spin_x, self.spin_y, self.spin_w, self.spin_h):
            spin.setRange(0, 10000)
        x, y, w, h = self.config.capture.region
        self.spin_x.setValue(x)
        self.spin_y.setValue(y)
        self.spin_w.setValue(w)
        self.spin_h.setValue(h)

        region_layout.addWidget(QtWidgets.QLabel("X"))
        region_layout.addWidget(self.spin_x)
        region_layout.addWidget(QtWidgets.QLabel("Y"))
        region_layout.addWidget(self.spin_y)
        region_layout.addWidget(QtWidgets.QLabel("W"))
        region_layout.addWidget(self.spin_w)
        region_layout.addWidget(QtWidgets.QLabel("H"))
        region_layout.addWidget(self.spin_h)

        self.btn_apply_region = QtWidgets.QPushButton("Apply")
        self.btn_pick_region = QtWidgets.QPushButton("Pick on Screen")
        region_layout.addWidget(self.btn_apply_region)
        region_layout.addWidget(self.btn_pick_region)

        self.btn_apply_region.clicked.connect(self.on_apply_region)
        self.btn_pick_region.clicked.connect(self.on_pick_region)

        left_layout.addWidget(region_group)

        # 디버그 미리보기
        self.preview_label = QtWidgets.QLabel("Debug preview (toggle 'Debug' and 'Start')")
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #222; color: #aaa;")
        left_layout.addWidget(self.preview_label, stretch=1)

        # 로그 영역
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(120)
        left_layout.addWidget(self.log_view)

        main_layout.addWidget(left_widget, stretch=3)

        # ---- 우측: 상태 패널 ----
        right_widget = QtWidgets.QWidget()
        right_widget.setMaximumWidth(self.config.gui.panel_width)
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        self.status_labels = {}
        status_fields = [
            "FPS",
            "Evaluation Time",
            "Search Time",
            "Empty Cells",
            "Placement Freedom",
            "Game Over Risk",
            "Best Score",
            "Combo",
            "Confidence",
            "Risk",
            "Survival",
            "Largest Empty Area",
            "Dead Area",
            "Flexibility",
            "Mobility",
            "Fragmentation",
            "Future Score",
        ]
        status_group = QtWidgets.QGroupBox("Status")
        status_layout = QtWidgets.QFormLayout(status_group)
        for field in status_fields:
            label = QtWidgets.QLabel("-")
            self.status_labels[field] = label
            status_layout.addRow(field + ":", label)
        right_layout.addWidget(status_group)

        # 추천 1~3
        self.rec_groups = []
        self.rec_labels = []
        for rank in range(1, self.config.solver.top_k + 1):
            group = QtWidgets.QGroupBox(f"Recommendation #{rank}")
            form = QtWidgets.QFormLayout(group)
            labels = {}
            for field in [
                "Best Move",
                "Expected Score",
                "Expected Combo",
                "Survival Rating",
                "Board Stability",
                "Future Space",
                "Risk Level",
                "Dead Area",
                "Largest Empty Region",
                "Placement Freedom",
            ]:
                lbl = QtWidgets.QLabel("-")
                lbl.setWordWrap(True)
                labels[field] = lbl
                form.addRow(field + ":", lbl)
            reasons_label = QtWidgets.QLabel("-")
            reasons_label.setWordWrap(True)
            labels["Reasons"] = reasons_label
            form.addRow("Reasons:", reasons_label)

            self.rec_groups.append(group)
            self.rec_labels.append(labels)
            right_layout.addWidget(group)

        # Debug 모드: 휴리스틱 항목별 점수 + MCTS 트리 통계
        debug_group = QtWidgets.QGroupBox("Debug Info")
        debug_layout = QtWidgets.QVBoxLayout(debug_group)
        self.debug_view = QtWidgets.QPlainTextEdit()
        self.debug_view.setReadOnly(True)
        self.debug_view.setMaximumHeight(160)
        debug_layout.addWidget(self.debug_view)
        right_layout.addWidget(debug_group)

        right_layout.addStretch(1)

        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidget(right_widget)
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        right_scroll.setMaximumWidth(self.config.gui.panel_width + 24)
        main_layout.addWidget(right_scroll, stretch=1)

    # ------------------------------------------------------------------
    # 버튼 핸들러
    # ------------------------------------------------------------------
    def on_start(self) -> None:
        self.capture.start()
        self.worker.start_pipeline()
        if self.overlay_window is not None:
            self.overlay_window.sync_geometry()
            self.overlay_window.show()
        self._log("Pipeline started")

    def on_stop(self) -> None:
        self.worker.stop_pipeline()
        self.capture.stop()
        if self.overlay_window is not None:
            self.overlay_window.clear()
            self.overlay_window.hide()
        self._log("Pipeline stopped")

    def on_pause_toggled(self, checked: bool) -> None:
        self.worker.set_paused(checked)
        self._log("Paused" if checked else "Resumed")

    def on_calibration(self) -> None:
        frame, _ = self.capture.get_latest_frame()
        if frame is None:
            frame = ScreenCapture.grab_once(self.config.capture.region)
        ok = self.board_detector.calibrate(frame)
        if ok:
            x, y, w, h = self.config.board.board_rect
            search_y = y + h
            search_h = max(0, frame.shape[0] - search_y)
            search_rect = (x, search_y, w, search_h)
            slots = self.block_detector.auto_detect_tray_slots(frame, search_rect)
            self.config.tray.slot_rects = slots
            self._log(f"Calibration done. board_rect={self.config.board.board_rect}, slots={slots}")
        else:
            self._log("Calibration failed: could not detect board")

    def on_debug_toggled(self, checked: bool) -> None:
        self._debug_mode = checked

    def on_save(self) -> None:
        self.config.save()
        self._log("Config saved")

    def on_load(self) -> None:
        self.config.load_into()
        x, y, w, h = self.config.capture.region
        self.spin_x.setValue(x)
        self.spin_y.setValue(y)
        self.spin_w.setValue(w)
        self.spin_h.setValue(h)
        self._log("Config loaded")

    def on_apply_region(self) -> None:
        x = self.spin_x.value()
        y = self.spin_y.value()
        w = self.spin_w.value()
        h = self.spin_h.value()
        self.capture.set_region(x, y, w, h)
        if self.overlay_window is not None:
            self.overlay_window.sync_geometry()
        self._log(f"Capture region set to ({x},{y},{w},{h})")

    def on_pick_region(self) -> None:
        self._region_selector = RegionSelector()
        self._region_selector.region_selected.connect(self._on_region_picked)
        self._region_selector.showFullScreen()

    def _on_region_picked(self, x: int, y: int, w: int, h: int) -> None:
        self.spin_x.setValue(x)
        self.spin_y.setValue(y)
        self.spin_w.setValue(w)
        self.spin_h.setValue(h)
        self.on_apply_region()

    # ------------------------------------------------------------------
    # 파이프라인 결과 처리
    # ------------------------------------------------------------------
    def on_result(self, result: PipelineResult) -> None:
        self._last_result = result
        self._update_status_panel(result)
        self._update_recommendation_panels(result)

        if self.overlay_window is not None:
            self.overlay_window.update_recommendation(result.solve_result, result.pieces)

        if self._debug_mode:
            self._update_preview(result)
            self._update_debug_panel(result)
        else:
            self.debug_view.setPlainText("")

    def on_error(self, message: str) -> None:
        self._log(f"ERROR: {message}")

    # ------------------------------------------------------------------
    def _update_status_panel(self, result: PipelineResult) -> None:
        sr = result.solve_result
        self.status_labels["FPS"].setText(f"{result.fps:.1f}")
        self.status_labels["Evaluation Time"].setText(f"{sr.evaluation_time_sec * 1000:.1f} ms")
        self.status_labels["Search Time"].setText(f"{sr.evaluation_time_sec * 1000:.1f} ms")
        self.status_labels["Empty Cells"].setText(str(sr.board_empty_cells))
        self.status_labels["Placement Freedom"].setText(str(sr.overall_placement_freedom))
        self.status_labels["Game Over Risk"].setText("YES" if sr.game_over_risk else "no")

        if sr.recommendations:
            top = sr.recommendations[0]
            self.status_labels["Best Score"].setText(f"{top.expected_score:.1f}")
            self.status_labels["Combo"].setText(str(top.expected_combo))
            self.status_labels["Confidence"].setText(f"{top.confidence:.1f}%")
            self.status_labels["Risk"].setText(top.risk_level)
            self.status_labels["Survival"].setText(top.survival_stars)
            self.status_labels["Largest Empty Area"].setText(str(top.largest_empty_region))
            self.status_labels["Dead Area"].setText(str(top.dead_area))
            self.status_labels["Flexibility"].setText(f"{top.flexibility_score:.1f}")
            self.status_labels["Mobility"].setText(str(top.mobility_score))
            self.status_labels["Fragmentation"].setText(f"{top.fragmentation_index:.2f}")
            self.status_labels["Future Score"].setText(f"{top.heuristic_score:.1f}")
        else:
            for field in (
                "Best Score", "Combo", "Confidence", "Risk", "Survival",
                "Largest Empty Area", "Dead Area", "Flexibility", "Mobility",
                "Fragmentation", "Future Score",
            ):
                self.status_labels[field].setText("-")

    def _update_recommendation_panels(self, result: PipelineResult) -> None:
        sr = result.solve_result
        for i, labels in enumerate(self.rec_labels):
            if i < len(sr.recommendations):
                rec = sr.recommendations[i]
                labels["Best Move"].setText(
                    f"piece {rec.piece_index} -> ({rec.anchor_row}, {rec.anchor_col})"
                )
                labels["Expected Score"].setText(f"{rec.expected_score:.1f}")
                labels["Expected Combo"].setText(str(rec.expected_combo))
                labels["Survival Rating"].setText(rec.survival_stars)
                labels["Board Stability"].setText(f"{rec.board_stability:.2f}")
                labels["Future Space"].setText(str(rec.future_space))
                labels["Risk Level"].setText(rec.risk_level)
                labels["Dead Area"].setText(str(rec.dead_area))
                labels["Largest Empty Region"].setText(str(rec.largest_empty_region))
                labels["Placement Freedom"].setText(str(rec.placement_freedom))
                labels["Reasons"].setText(", ".join(rec.reasons))
            else:
                for lbl in labels.values():
                    lbl.setText("-")

    def _update_debug_panel(self, result: PipelineResult) -> None:
        sr = result.solve_result
        lines: List[str] = []

        if sr.recommendations:
            top = sr.recommendations[0]
            seq = top.full_sequence
            if seq is not None:
                pieces = _normalize_pieces(result.pieces)
                breakdown = evaluate_with_breakdown(
                    seq.final_board_bb, seq.total_score_gain, seq.final_combo, pieces, self.config.weights
                )
                lines.append("=== Heuristic Breakdown (#1) ===")
                lines.append(f"total: {breakdown.total:.2f}")
                for name, value in breakdown.components.items():
                    lines.append(f"  {name}: {value:+.2f}")

        if sr.mcts_result is not None:
            lines.append("")
            lines.append(f"=== MCTS (iters={sr.mcts_result.iterations}, {sr.mcts_result.elapsed_sec*1000:.1f} ms) ===")
            top_children = sorted(sr.mcts_result.root_children, key=lambda c: c.visits, reverse=True)[:5]
            for c in top_children:
                lines.append(
                    f"  piece={c.slot} pos=({c.placement.anchor_row},{c.placement.anchor_col}) "
                    f"visits={c.visits} value={c.mean_value:.2f}"
                )

        self.debug_view.setPlainText("\n".join(lines))

    def _update_preview(self, result: PipelineResult) -> None:
        frame = result.frame
        frame = self.board_detector.draw_debug_overlay(frame, result.board)
        frame = render_overlay_cv2(frame, result.solve_result, result.pieces, self.config)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self.preview_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled)

    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        self.log_view.appendPlainText(message)
        logger.info(message)

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        self.worker.stop_pipeline()
        self.capture.stop()
        if self.overlay_window is not None:
            self.overlay_window.close()
        event.accept()


def _set_dpi_awareness() -> None:
    """Windows 디스플레이 배율(DPI scaling)이 100%가 아닐 때, mss 가 캡처하는
    물리 픽셀 좌표와 Qt 의 화면 좌표가 어긋나는 것을 방지한다."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def main() -> None:
    _set_dpi_awareness()
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

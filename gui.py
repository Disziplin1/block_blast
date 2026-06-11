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

import hashlib
import sys
import time
from collections import Counter, deque
from typing import List, Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from block_detector import BlockDetector, DetectedPiece
from board_detector import BoardDetector
from capture import ScreenCapture
from config import AppConfig, CONFIG
from data_logger import DataLogger
from logger import StageTimer, get_logger
from overlay import OverlayWindow, PYQT_AVAILABLE, render_overlay_cv2
from heuristic import evaluate_with_breakdown
from solver import SolveResult, _normalize_pieces, solve
from utils import PieceCells, piece_cells_to_grid

logger = get_logger("gui")


# ---------------------------------------------------------------------------
# Vision 디버깅 헬퍼 (#3, #7)
# ---------------------------------------------------------------------------
def _piece_shape_str(cells: PieceCells) -> str:
    """PieceCells 를 콘솔 출력용 Shape Matrix 문자열로 변환한다."""
    if not cells:
        return "(empty)"
    return str(piece_cells_to_grid(cells).tolist())


def _stability_pct(history: "deque") -> float:
    """최근 프레임 히스토리에서 직전 프레임과 값이 같았던 비율(%)을 계산한다."""
    if len(history) < 2:
        return 100.0
    same = sum(1 for i in range(1, len(history)) if history[i] == history[i - 1])
    return (same / (len(history) - 1)) * 100.0


def _format_cell_debug(det: DetectedPiece) -> str:
    """블록 1개의 셀별 평균 RGB / Occupied 판정 결과를 콘솔 출력용 문자열로 변환한다."""
    if det.empty or not det.debug_cells:
        return "  (empty)"
    return "\n".join(f"  Cell({r},{c})=RGB{rgb}={occ}" for r, c, rgb, occ in det.debug_cells)


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
        detections: Optional[List[DetectedPiece]] = None,
        board_hash: str = "",
        block_shapes: Optional[List[str]] = None,
        search_executed: bool = True,
        board_stable_pct: float = 100.0,
        blocks_stable_pct: float = 100.0,
        roi_images: Optional[List[np.ndarray]] = None,
        block_stable: Optional[List[bool]] = None,
        block_confidence: Optional[List[float]] = None,
        template_names: Optional[List[str]] = None,
        template_similarities: Optional[List[float]] = None,
    ):
        self.frame = frame
        self.board = board
        self.pieces = pieces
        self.solve_result = solve_result
        self.timings = timings
        self.fps = fps
        self.detections = detections or []
        self.board_hash = board_hash
        self.block_shapes = block_shapes or []
        self.search_executed = search_executed
        self.board_stable_pct = board_stable_pct
        self.blocks_stable_pct = blocks_stable_pct
        self.roi_images = roi_images or []
        self.block_stable = block_stable or []
        self.block_confidence = block_confidence or []
        self.template_names = template_names or []
        self.template_similarities = template_similarities or []


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

        # Vision 안정성 디버깅용 상태 (#2, #3, #7)
        self._prev_board: Optional[np.ndarray] = None
        self._prev_pieces: Optional[List[PieceCells]] = None
        self._prev_solve_result: Optional[SolveResult] = None
        self._board_hash_history: deque = deque(maxlen=30)
        self._blocks_hash_history: deque = deque(maxlen=30)

        # 블록별 Shape 안정성 추적: 최근 10프레임 Majority Vote + 5프레임 이상
        # 동일 Majority 가 유지되면 Stable Shape 로 확정한다. Solver 에는
        # Stable Shape(없으면 현재 Majority Shape)만 전달한다.
        slot_count = self.config.tray.slot_count
        self._block_shape_history: List[deque] = [deque(maxlen=10) for _ in range(slot_count)]
        self._block_prev_majority: List[Optional[PieceCells]] = [None] * slot_count
        self._block_majority_consecutive: List[int] = [0] * slot_count
        self._block_stable_shape: List[Optional[PieceCells]] = [None] * slot_count

        self._debug_mode = False

    def set_debug_mode(self, enabled: bool) -> None:
        self._debug_mode = enabled

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
                    detections = self.block_detector.detect(frame)

                # 블록별 Shape 안정화: 최근 10프레임 Majority Vote 를 적용해
                # 프레임마다 Shape 가 흔들려도(L L L T L L L) 최종 Shape 는
                # 다수결(L)로 고정한다. Majority Shape 가 5프레임 이상 연속
                # 유지되면 Stable Shape 로 확정하고, Solver 에는 Stable Shape
                # (없으면 현재 Majority Shape)만 전달한다.
                pieces: List[PieceCells] = []
                block_stable: List[bool] = []
                block_confidence: List[float] = []
                template_names: List[str] = []
                template_similarities: List[float] = []
                for i, det in enumerate(detections):
                    hist = self._block_shape_history[i]
                    hist.append(det.cells)
                    majority_shape, majority_count = Counter(hist).most_common(1)[0]

                    if majority_shape == self._block_prev_majority[i]:
                        self._block_majority_consecutive[i] += 1
                    else:
                        self._block_majority_consecutive[i] = 1
                        self._block_prev_majority[i] = majority_shape

                    is_stable = self._block_majority_consecutive[i] >= 5
                    if is_stable:
                        self._block_stable_shape[i] = majority_shape

                    chosen = (
                        self._block_stable_shape[i]
                        if self._block_stable_shape[i] is not None
                        else majority_shape
                    )
                    pieces.append(chosen)
                    block_stable.append(is_stable)
                    block_confidence.append(majority_count / len(hist) * 100.0)
                    template_names.append(det.template_name)
                    template_similarities.append(det.template_similarity)

                # 보드/블록 해시 계산 (Solver 에 전달되는 Stable Shape 기준) (#3, #7)
                board_hash = hashlib.md5(board.tobytes()).hexdigest()[:8]
                blocks_hash = hashlib.md5(str(pieces).encode("utf-8")).hexdigest()[:8]
                self._board_hash_history.append(board_hash)
                self._blocks_hash_history.append(blocks_hash)
                board_stable_pct = _stability_pct(self._board_hash_history)
                blocks_stable_pct = _stability_pct(self._blocks_hash_history)

                # 화면이 이전 프레임과 동일하면 Solver 를 다시 실행하지 않는다 (#2)
                board_unchanged = (
                    self._prev_board is not None and np.array_equal(board, self._prev_board)
                )
                pieces_unchanged = pieces == self._prev_pieces
                if board_unchanged and pieces_unchanged and self._prev_solve_result is not None:
                    solve_result = self._prev_solve_result
                    search_executed = False
                else:
                    with timer.stage("Search+Evaluate"):
                        solve_result = solve(board, pieces, self.config)
                    search_executed = True
                    self._prev_board = board.copy()
                    self._prev_pieces = pieces
                    self._prev_solve_result = solve_result

                with timer.stage("Recommend"):
                    pass  # solve_result.recommendations 자체가 추천 결과

                if self._data_logger is not None:
                    try:
                        self._data_logger.log_recommendation(board, pieces, solve_result)
                    except Exception:  # pragma: no cover - 로깅 실패는 파이프라인을 막지 않음
                        logger.exception("Data logging failed")

                block_shapes = [_piece_shape_str(p) for p in pieces]
                confidence = solve_result.recommendations[0].confidence if solve_result.recommendations else 0.0
                logger.info(
                    "\nBoard Hash:\n%s\nBlock1:\n%s\nBlock2:\n%s\nBlock3:\n%s\n"
                    "Search Executed:\n%s\nConfidence:\n%.1f%%\n"
                    "Board Stable: %.0f%% | Blocks Stable: %.0f%%",
                    board_hash,
                    block_shapes[0] if len(block_shapes) > 0 else "(empty)",
                    block_shapes[1] if len(block_shapes) > 1 else "(empty)",
                    block_shapes[2] if len(block_shapes) > 2 else "(empty)",
                    "YES" if search_executed else "NO",
                    confidence,
                    board_stable_pct,
                    blocks_stable_pct,
                )

                roi_images: List[np.ndarray] = []
                if self._debug_mode:
                    for i, det in enumerate(detections, start=1):
                        logger.info(
                            "Block%d Cells (threshold=%.1f, template=%s sim=%.2f, stable=%s, confidence=%.0f%%):\n%s",
                            i, det.threshold_used, det.template_name, det.template_similarity,
                            block_stable[i - 1] if i - 1 < len(block_stable) else False,
                            block_confidence[i - 1] if i - 1 < len(block_confidence) else 0.0,
                            _format_cell_debug(det),
                        )
                        roi_images.append(self.block_detector.render_roi_debug(frame, self.config.tray.slot_rects[i - 1], det))

                result = PipelineResult(
                    frame=frame,
                    board=board,
                    pieces=pieces,
                    solve_result=solve_result,
                    timings=timer.as_dict(),
                    fps=self.capture.fps,
                    detections=detections,
                    board_hash=board_hash,
                    block_shapes=block_shapes,
                    search_executed=search_executed,
                    board_stable_pct=board_stable_pct,
                    blocks_stable_pct=blocks_stable_pct,
                    roi_images=roi_images,
                    block_stable=block_stable,
                    block_confidence=block_confidence,
                    template_names=template_names,
                    template_similarities=template_similarities,
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

        toolbar.addWidget(QtWidgets.QLabel("Show:"))
        self.combo_highlight = QtWidgets.QComboBox()
        self.combo_highlight.addItem("All", 0)
        for rank in range(1, self.config.solver.top_k + 1):
            self.combo_highlight.addItem(str(rank), rank)
        toolbar.addWidget(self.combo_highlight)

        left_layout.addLayout(toolbar)

        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_pause.toggled.connect(self.on_pause_toggled)
        self.btn_calibration.clicked.connect(self.on_calibration)
        self.btn_debug.toggled.connect(self.on_debug_toggled)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_load.clicked.connect(self.on_load)
        self.combo_highlight.currentIndexChanged.connect(self.on_highlight_changed)

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

        # Debug 모드: 블록별 ROI 미리보기 (Cell Grid + Shape Matrix) (#1, #2, #6)
        roi_group = QtWidgets.QGroupBox("Block ROI (Debug)")
        roi_layout = QtWidgets.QHBoxLayout(roi_group)
        self.roi_labels: List[QtWidgets.QLabel] = []
        self.roi_info_labels: List[QtWidgets.QLabel] = []
        for i in range(self.config.tray.slot_count):
            col = QtWidgets.QVBoxLayout()
            title = QtWidgets.QLabel(f"Block{i + 1} ROI")
            col.addWidget(title)
            img_label = QtWidgets.QLabel("-")
            img_label.setFixedSize(120, 120)
            img_label.setAlignment(QtCore.Qt.AlignCenter)
            img_label.setStyleSheet("background-color: #222; color: #888;")
            col.addWidget(img_label)
            info_label = QtWidgets.QLabel("-")
            info_label.setWordWrap(True)
            col.addWidget(info_label)
            self.roi_labels.append(img_label)
            self.roi_info_labels.append(info_label)
            roi_layout.addLayout(col)
        right_layout.addWidget(roi_group)

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
        self.worker.set_debug_mode(checked)
        if not checked:
            for label in self.roi_labels:
                label.clear()
            for label in self.roi_info_labels:
                label.setText("-")

    def on_highlight_changed(self, index: int) -> None:
        rank = self.combo_highlight.itemData(index)
        self.config.overlay.highlight_rank = int(rank)
        if self._last_result is not None and self.overlay_window is not None:
            self.overlay_window.update_recommendation(
                self._last_result.solve_result, self._last_result.pieces
            )
        self._log(f"Overlay highlight set to: {'All' if rank == 0 else rank}")

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
            self._update_block_rois(result)
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

        # Vision 안정성 정보 (#3, #5, #7)
        lines.append("=== Vision Stability ===")
        lines.append(f"Board Hash: {result.board_hash}")
        for i, shape in enumerate(result.block_shapes, start=1):
            lines.append(f"Block{i}: {shape}")
            if i - 1 < len(result.template_names):
                lines.append(
                    f"  Template: {result.template_names[i - 1]} "
                    f"(sim={result.template_similarities[i - 1]:.2f})"
                )
            if i - 1 < len(result.block_stable):
                stable = "Stable" if result.block_stable[i - 1] else "Unstable"
                conf = result.block_confidence[i - 1]
                lines.append(f"  -> {stable} (Confidence: {conf:.0f}%)")
        lines.append(f"Search Executed: {'YES' if result.search_executed else 'NO'}")
        lines.append(f"Board Stable: {result.board_stable_pct:.0f}%")
        lines.append(f"Blocks Stable: {result.blocks_stable_pct:.0f}%")
        lines.append("")

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
        frame = self.block_detector.draw_debug_overlay(frame, result.detections)
        frame = render_overlay_cv2(frame, result.solve_result, result.pieces, self.config)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self.preview_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled)

    def _update_block_rois(self, result: PipelineResult) -> None:
        """Block1/2/3 ROI(셀 그리드 + Shape Matrix)를 별도 미리보기로 표시한다 (#1, #2, #6)."""
        for i, img_label in enumerate(self.roi_labels):
            if i >= len(result.roi_images):
                img_label.clear()
                self.roi_info_labels[i].setText("-")
                continue

            roi = result.roi_images[i]
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg)
            img_label.setPixmap(pix.scaled(
                img_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            ))

            shape = result.block_shapes[i] if i < len(result.block_shapes) else "(empty)"
            stable = result.block_stable[i] if i < len(result.block_stable) else False
            conf = result.block_confidence[i] if i < len(result.block_confidence) else 0.0
            det = result.detections[i] if i < len(result.detections) else None
            threshold = det.threshold_used if det is not None else 0.0
            template_name = result.template_names[i] if i < len(result.template_names) else "-"
            similarity = result.template_similarities[i] if i < len(result.template_similarities) else 0.0
            self.roi_info_labels[i].setText(
                f"Template: {template_name} ({similarity:.2f})\n"
                f"Solver Shape: {shape}\n"
                f"{'Stable' if stable else 'Unstable'} (Conf: {conf:.0f}%)\n"
                f"Threshold: {threshold:.1f}"
            )

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

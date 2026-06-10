"""
config.py
=========
전역 설정값을 관리하는 모듈.

모든 모듈은 이 파일을 통해 설정값을 읽고 쓴다.
GUI의 'Save' / 'Load' 버튼은 이 설정을 JSON 파일로 직렬화/역직렬화한다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Dict, Any


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


# ---------------------------------------------------------------------------
# 보드 / 캡처 관련 설정
# ---------------------------------------------------------------------------
@dataclass
class CaptureConfig:
    monitor_index: int = 1          # mss monitor index (0 = 전체 가상 화면)
    region: Tuple[int, int, int, int] = (0, 0, 800, 1200)  # (left, top, width, height)
    target_fps: int = 20
    downscale: float = 1.0          # 캡처 후 리사이즈 비율 (성능 최적화용)


@dataclass
class BoardConfig:
    rows: int = 8
    cols: int = 8
    # 보드 영역 (캡처 좌표계 기준, calibration 으로 채워짐)
    board_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)
    cell_fill_threshold: float = 0.45  # 셀 내부 채움 비율이 이 값을 넘으면 '블록 있음'
    grid_line_color_tolerance: int = 20
    # 채도(saturation) 기준 빈칸 판정 임계값 (0~255). 캘리브레이션이 없을 때 사용.
    empty_saturation_threshold: float = 40.0
    # 캘리브레이션으로 측정한 빈 보드의 체커보드 두 칸(짝/홀) 채도 평균
    empty_saturation_baseline: Tuple[float, float] = (0.0, 0.0)
    calibrated: bool = False


@dataclass
class TrayConfig:
    """현재 대기 중인 3개의 블록이 표시되는 트레이 영역."""
    slot_count: int = 3
    slot_rects: List[Tuple[int, int, int, int]] = field(
        default_factory=lambda: [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)]
    )
    max_piece_size: int = 5  # 블록의 최대 가로/세로 셀 수
    # 트레이에 표시되는 블록 1칸의 픽셀 크기 = 보드 1칸 크기 * piece_cell_scale
    piece_cell_scale: float = 0.6
    # 트레이 배경과 블록을 구분하는 채도(saturation) 임계값
    empty_saturation_threshold: float = 35.0


# ---------------------------------------------------------------------------
# AI / Solver 관련 설정
# ---------------------------------------------------------------------------
@dataclass
class SolverConfig:
    # 탐색 깊이: 현재 트레이의 블록 개수 (보통 3)
    search_depth: int = 3
    # 평가 시 상위 N개 1차 후보만 남겨 가지치기 (0 = 가지치기 없음).
    # 8x8 보드에서 0.2~0.5초 내 평가를 위해 기본값을 사용한다.
    beam_width: int = 30
    # Monte Carlo 시뮬레이션 횟수
    monte_carlo_trials: int = 200
    # 시뮬레이션 시 미래 몇 턴까지 내다볼지
    monte_carlo_depth: int = 4
    # 추천 후보 개수 (①②③)
    top_k: int = 3
    # 평가 시간 제한 (초). 초과 시 현재까지 탐색한 결과 중 최선을 반환
    time_budget_sec: float = 0.5

    # --- MCTS (#1: Beam Search 와 병행하는 다중 턴(3~8턴) 미래 시뮬레이션) ---
    use_mcts: bool = False
    mcts_iterations: int = 150
    mcts_max_turns: int = 5          # 미래 몇 "턴"(트레이 1세트 = 보통 3블록)까지 시뮬레이션할지
    mcts_time_budget_sec: float = 0.15
    mcts_exploration: float = 1.4    # UCB1 탐색 상수 (c)


@dataclass
class HeuristicWeights:
    """
    heuristic.py 의 평가 함수에서 사용하는 가중치.
    양수 = 좋은 상태일수록 점수 증가
    음수 = 나쁜 상태일수록 점수 감소 (페널티)
    """
    # 긍정 요소
    score_gain: float = 1.0
    combo_bonus: float = 8.0
    lines_cleared_bonus: float = 12.0
    empty_cell_bonus: float = 0.5
    next_piece_fit_bonus: float = 6.0
    center_space_bonus: float = 1.5
    large_empty_region_bonus: float = 2.0
    rectangular_region_bonus: float = 1.5
    expandable_space_bonus: float = 1.2
    chain_clear_potential_bonus: float = 5.0
    survivability_bonus: float = 10.0
    move_freedom_bonus: float = 1.0
    future_turn_bonus: float = 3.0
    line_clear_efficiency_bonus: float = 1.0
    column_clear_efficiency_bonus: float = 1.0
    connectivity_bonus: float = 1.0
    symmetry_bonus: float = 0.3
    avg_placeability_bonus: float = 1.0
    space_utilization_bonus: float = 0.5

    # 부정 요소 (penalty, 양수로 정의 후 평가시 차감)
    isolated_cell_penalty: float = 4.0
    hole1_penalty: float = 3.0
    hole2_penalty: float = 2.0
    dead_space_penalty: float = 6.0
    fragmentation_penalty: float = 2.0
    border_block_penalty: float = 1.0
    corner_block_penalty: float = 1.5
    cross_block_penalty: float = 1.5
    long_corridor_penalty: float = 1.0
    big_piece_unplaceable_penalty: float = 5.0
    move_restriction_penalty: float = 2.0
    fragmentation_growth_penalty: float = 2.0
    narrow_space_penalty: float = 1.0
    placement_freedom_loss_penalty: float = 1.5


@dataclass
class OverlayConfig:
    box_color_1: Tuple[int, int, int] = (0, 255, 0)     # 1순위: 초록
    box_color_2: Tuple[int, int, int] = (0, 200, 255)   # 2순위: 노랑/주황
    box_color_3: Tuple[int, int, int] = (0, 120, 255)   # 3순위: 주황
    alpha: float = 0.35
    show_arrows: bool = True
    show_numbers: bool = True
    show_reasons: bool = True
    font_scale: float = 0.6
    line_width: int = 2

    # 위험 지역 분석 (#5)
    show_risk_zones: bool = True
    risk_color: Tuple[int, int, int] = (0, 0, 255)   # 빨강 (BGR/RGB 공용 사용처에 따라 다름)
    risk_max_alpha: float = 0.45

    # 추천 위치 Heat Map (#7)
    show_heatmap: bool = False
    heatmap_piece_index: int = 0
    heatmap_max_alpha: float = 0.5

    # 특정 순위(①②③)만 오버레이에 표시 (0 = 전체 표시)
    highlight_rank: int = 0


@dataclass
class GUIConfig:
    window_title: str = "Block Blast AI Assistant"
    window_size: Tuple[int, int] = (1280, 800)
    panel_width: int = 320
    update_interval_ms: int = 50


@dataclass
class LoggingConfig:
    log_dir: str = os.path.join(os.path.dirname(__file__), "logs")
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True

    # 자동 학습용 데이터 로깅 (#2)
    data_logging_enabled: bool = False
    data_dir: str = os.path.join(os.path.dirname(__file__), "data")
    db_filename: str = "play_log.db"


@dataclass
class AppConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    board: BoardConfig = field(default_factory=BoardConfig)
    tray: TrayConfig = field(default_factory=TrayConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    weights: HeuristicWeights = field(default_factory=HeuristicWeights)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AppConfig":
        cfg = AppConfig()
        for section_name, section_obj in vars(cfg).items():
            if section_name not in data:
                continue
            section_data = data[section_name]
            for key, value in section_data.items():
                if hasattr(section_obj, key):
                    # 튜플 필드는 JSON 에서 list 로 들어오므로 변환
                    current = getattr(section_obj, key)
                    if isinstance(current, tuple) and isinstance(value, list):
                        value = tuple(value)
                    elif key == "slot_rects" and isinstance(value, list):
                        value = [tuple(v) for v in value]
                    setattr(section_obj, key, value)
        return cfg

    def save(self, path: str = CONFIG_PATH) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: str = CONFIG_PATH) -> "AppConfig":
        if not os.path.exists(path):
            cfg = AppConfig()
            cfg.save(path)
            return cfg
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return AppConfig.from_dict(data)

    def load_into(self, path: str = CONFIG_PATH) -> None:
        """파일에서 설정을 읽어 이 인스턴스(self)의 필드를 제자리에서 갱신한다.

        CONFIG 는 여러 모듈에서 공유하는 싱글턴이므로, 새 인스턴스로
        교체하는 대신 필드를 덮어써야 모든 모듈이 갱신된 값을 본다.
        """
        loaded = AppConfig.load(path)
        for section_name in vars(self):
            setattr(self, section_name, getattr(loaded, section_name))


# 전역 싱글턴 설정 인스턴스 (모듈 임포트 시 1회 로드)
CONFIG = AppConfig.load()


if __name__ == "__main__":
    # 기본 설정 파일 생성/검증용
    CONFIG.save()
    print(f"Config saved to {CONFIG_PATH}")

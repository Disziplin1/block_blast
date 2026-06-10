# Ultimate Block Blast AI Assistant

Block Blast 게임을 위한 **실시간 추천 전용** AI Assistant입니다.
화면(아이폰 미러링 등)을 실시간으로 분석하여, 사람이 직접 둘 최적의
블록 배치를 제안합니다. **자동 입력(클릭/터치/드래그)은 절대 수행하지
않으며**, 추천만 표시합니다.

목표는 "현재 점수"가 아니라 **최종 점수(=최대한 오래 생존)** 를
최대화하는 것입니다.

## 폴더 구조

```
block_blast_ai/
├── main.py            # 진입점
├── capture.py         # mss 기반 실시간 화면 캡처
├── board_detector.py  # 8x8 보드 인식 + 캘리브레이션
├── block_detector.py  # 트레이의 3개 블록 인식
├── solver.py          # 탐색+휴리스틱+시뮬레이션+MCTS+위험분석 통합 오케스트레이션
├── search.py          # 비트보드 기반 배치 탐색 (DFS/백트래킹 + beam)
├── heuristic.py        # 보드 평가 휴리스틱
├── simulation.py       # Monte Carlo 미래 예측 (1턴 그리디)
├── mcts.py              # 다중 턴(3~8턴) MCTS (Beam Search 와 병행)
├── risk_analysis.py     # 위험 지역(Dead Area/구멍/고립) 분석 + 추천 Heat Map
├── data_logger.py        # 추천/플레이 기록 SQLite 로깅 (자동 학습용 데이터)
├── stats.py               # 통계 집계 + 리플레이(Replay) 시스템
├── tuning.py              # 휴리스틱 가중치 자동 튜닝(GA) + 블록 확률 모델
├── rl_env.py              # 강화학습용 State/Action/Reward 환경 스캐폴딩
├── overlay.py          # 추천 오버레이 (PyQt5 / OpenCV, 위험지역/Heat Map 포함)
├── gui.py               # PyQt5 메인 GUI (확장 대시보드 + Debug 모드)
├── config.py            # 전역 설정 (config.json 으로 저장/로드)
├── utils.py              # 보드/블록 변환, 비트보드 유틸, 블록 라이브러리
├── logger.py             # 로깅 + 파이프라인 단계별 타이머
├── tests/
│   ├── test_core.py      # Phase 1 핵심 알고리즘 스모크 테스트
│   └── test_upgrade.py   # Final Upgrade(MCTS/위험분석/로깅/튜닝/RL) 스모크 테스트
├── requirements.txt
└── config.json           # 실행 시 자동 생성되는 설정 파일
```

## 빠른 시작 (Windows, 명령어 입력 없이)

1. `setup.bat` 을 더블클릭 — 최초 1회만 실행 (가상환경 생성 + 패키지 설치)
2. `run.bat` 을 더블클릭 — GUI 실행

이후에는 `run.bat` 만 더블클릭하면 됩니다. 오류가 발생하면 창이 바로
닫히지 않고 메시지가 표시되니 그대로 캡처해서 공유해주세요.

## 설치 방법 (수동, 명령어 직접 입력)

Python 3.12 기준입니다.

```powershell
cd block_blast_ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 실행 방법

### 1. 핵심 알고리즘만 빠르게 테스트 (화면 캡처/GUI 없음)

```powershell
python main.py --selftest
```

### 2. 전체 GUI 실행

```powershell
python main.py
```

### 3. 사용 순서

1. **아이폰 화면 미러링**을 PC에 띄웁니다.
2. GUI에서 `Pick on Screen` 버튼으로 미러링 창 영역을 드래그하여
   캡처 영역(Capture Region)을 지정하고 `Apply` 를 누릅니다.
3. `Calibration` 버튼을 눌러 보드(8x8)와 트레이(3개 블록 슬롯) 위치를
   자동 인식합니다. (보드가 비어 있는 상태에서 캘리브레이션하면 정확도가
   높아집니다.)
4. `Start` 를 눌러 실시간 분석을 시작합니다.
   - 화면 위에 추천 위치가 초록(①) / 노랑(②) / 주황(③) 박스와 화살표로
     표시됩니다.
   - 우측 패널에서 각 추천의 예상 점수, 콤보, 생존도(★), 위험도,
     추천 이유 등을 확인할 수 있습니다.
5. `Debug` 를 켜면 좌측에 인식된 보드/블록/오버레이가 그려진 미리보기를
   확인할 수 있습니다.
6. `Pause` 로 일시정지, `Stop` 으로 분석을 중단합니다.
7. `Save` / `Load` 로 현재 설정(캘리브레이션 결과 포함)을 `config.json` 에
   저장하거나 불러올 수 있습니다.

## 설정 (config.json)

`config.py` 의 `AppConfig` 가 모든 설정을 정의하며, 최초 실행 시
`config.json` 이 자동 생성됩니다. 주요 항목:

- `capture`: 캡처 영역, 목표 FPS, 다운스케일 비율
- `board`: 보드 영역, 캘리브레이션 결과(체커보드 채도 기준값)
- `tray`: 트레이(블록 대기열) 슬롯 영역
- `solver`: 탐색 빔 너비, Monte Carlo 횟수/깊이, 시간 예산, 추천 개수(top_k)
- `weights`: 휴리스틱 평가 가중치 (생존성 중심으로 튜닝 가능)
- `overlay`: 오버레이 색상/투명도/표시 옵션
- `gui`: 창 크기, 갱신 주기

## 알고리즘 개요

1. **탐색 (search.py)**: 현재 트레이의 최대 3개 블록에 대해 가능한
   모든 순서(순열) x 모든 배치 위치를 비트보드 연산으로 탐색합니다.
   8x8 보드에서 0.2~0.5초 내 평가를 위해 `beam_width` 로 가지치기합니다.
2. **휴리스틱 (heuristic.py)**: 점수, 콤보, 라인 클리어, 빈 공간의
   크기/형태/연결성, 고립 셀/구멍, 다음 블록 대응력, 생존 가능성 등
   30여 개 요소를 종합 평가합니다.
3. **시뮬레이션 (simulation.py)**: 각 후보 배치 이후, 무작위로 등장할
   미래 블록들에 대해 Monte Carlo 시뮬레이션을 수행하여 평균 생존 턴,
   평균 점수, 게임 종료 확률을 추정합니다.
4. **솔버 (solver.py)**: 위 결과를 종합하여 ①②③ 순위의 추천과
   생존도(★1~5), 위험도(Safe/Good/Risky/Danger/Critical), 추천 이유를
   생성합니다.

## Final Upgrade 기능 (Phase A~G)

### #5 위험 지역 분석 / #7 추천 Heat Map
`risk_analysis.py` 가 보드의 구멍(hole1/hole2), 고립 셀, 막힌 코너/십자
패턴, 좁은 통로, 작은 파편 영역을 셀 단위로 분석하여 `RiskMap` 을
생성한다. `cfg.overlay.show_risk_zones=True` 이면 오버레이에 빨간색
음영으로 표시된다. 또한 트레이의 각 블록에 대해 모든 배치 위치의
평가 점수를 0~1로 정규화한 `heatmap` 을 계산하며,
`cfg.overlay.show_heatmap=True` + `heatmap_piece_index` 로 표시할 블록을
선택할 수 있다.

### #1 다중 턴 MCTS
`mcts.py` 는 현재 트레이를 비우는 모든 순서/위치를 UCB1 트리로
탐색하고, 트레이 소진 이후에는 무작위 미래 턴을 그리디하게 롤아웃하여
장기 생존 가치를 추정한다. `cfg.solver.use_mcts=True` 로 활성화하며,
`mcts_iterations`, `mcts_max_turns`, `mcts_time_budget_sec` 으로 탐색
범위를 조절한다. MCTS 가 가장 많이 방문한 첫 수와 일치하는 추천에는
"MCTS 장기 생존 추천" 이유가 추가된다.

### #6 신뢰도(Confidence)
각 추천의 `confidence` (0~100%) 는 휴리스틱 점수의 상대적 우위와
Monte Carlo 종료 확률을 결합하여 계산된다.

### #2 자동 데이터 로깅
`cfg.logging.data_logging_enabled=True` 로 설정하면, 매 파이프라인
프레임마다 `data_logger.DataLogger` 가 보드/블록/추천 이동/점수/콤보/
클리어 라인/남은 공간/미래 공간/생존 턴/위험도/신뢰도/평가 점수/
탐색 시간/타임스탬프를 `data/play_log.db` (SQLite) 에 기록한다.
`DataLogger.export_json()` / `export_csv()` 로 내보낼 수 있다.

### #3 리플레이 / #4 통계
`stats.py` 의 `ReplayPlayer` 로 기록된 보드/블록/추천을 순서대로
재생할 수 있고, `compute_statistics()` 로 최고/평균 점수, 평균 생존
턴, 평균 콤보, 평균 클리어 라인, 추천 수락률(다음 기록의 보드가
추천된 배치와 일치하는 비율), 평균 탐색 시간, 평균 신뢰도, 평균
위험도를 집계한다.

### #5 자기 튜닝 / #10 블록 확률 모델
`tuning.py::genetic_algorithm()` 은 그리디 자가 플레이 시뮬레이션의
생존 턴 + 점수를 적합도로 사용하는 유전 알고리즘으로
`HeuristicWeights` 를 탐색한다. `build_piece_probabilities()` 는 누적된
로그에서 블록 모양별 등장 빈도를 추정하여, `weighted_random_pieces()`
로 더 현실적인 미래 블록 샘플링에 사용할 수 있다.

### #6(RL) 강화학습 스캐폴딩
`rl_env.py::BlockBlastEnv` 은 `reset()/step(action)/action_mask()` 를
제공하는 최소한의 Gym 스타일 환경이다. 관측값은 보드(64) + 트레이 3개
블록(각 5x5=25, 총 75) 을 이어붙인 길이 139 벡터이며, 행동은
`piece_slot * 64 + row * 8 + col` 로 인코딩된 192개의 이산 행동이다.
추후 PPO/DQN/A2C 등 RL 알고리즘을 이 인터페이스 위에 구현할 수 있다.

### #8 GUI 대시보드 / Debug 모드
GUI 우측 패널에 FPS, 탐색 시간, 최고 점수, 콤보, 신뢰도, 위험도,
생존도, 최대 빈 공간, Dead Area, Flexibility, Mobility, Fragmentation,
Future Score 가 표시된다. `Debug` 토글 시 1순위 추천에 대한 휴리스틱
항목별 점수와 (활성화된 경우) MCTS 루트의 상위 후보 방문 횟수/가치가
표시된다.

## 테스트

```powershell
python tests\test_core.py
python tests\test_upgrade.py
```

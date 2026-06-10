"""
tuning.py
=========
휴리스틱 가중치(HeuristicWeights) 자동 튜닝 (#5: Self-tuning) 및
플레이 데이터 기반 블록 등장 확률 모델 (#10: Block Probability Model).

1. 자동 튜닝
   - 유전 알고리즘(Genetic Algorithm)으로 HeuristicWeights 의 각 항목을
     탐색하여, "그리디 자가 플레이" 시뮬레이션에서의 평균 생존 턴 수 +
     평균 점수를 최대화하는 가중치 조합을 찾는다.
   - 시간 예산을 고려해 population/generations/episodes 를 작게 유지한다
     (기본값으로도 실행 가능하지만, 실제 활용 시에는 오프라인에서
     충분한 시간을 두고 실행하는 것을 권장한다).

2. 블록 확률 모델
   - data_logger 에 누적된 로그에서 등장한 블록 모양의 빈도를 세어,
     PIECE_LIBRARY 각 모양에 대한 등장 확률 분포를 추정한다.
   - 이 분포로 가중 무작위 샘플링을 수행하는 weighted_random_pieces 를
     제공하여, simulation/MCTS 의 "다음 블록 예측"을 더 현실적으로
     만들 수 있다.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, fields, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import HeuristicWeights
from data_logger import DataLogger
from heuristic import evaluate
from search import apply_placement, can_place_anywhere, valid_placements
from utils import PIECE_LIBRARY, PieceCells, normalize_piece, random_pieces


# ---------------------------------------------------------------------------
# 가중치 <-> 벡터 변환
# ---------------------------------------------------------------------------
WEIGHT_FIELDS: List[str] = [f.name for f in fields(HeuristicWeights)]


def weights_to_vector(weights: HeuristicWeights) -> np.ndarray:
    return np.array([getattr(weights, name) for name in WEIGHT_FIELDS], dtype=np.float64)


def vector_to_weights(vec: np.ndarray) -> HeuristicWeights:
    kwargs = {name: float(value) for name, value in zip(WEIGHT_FIELDS, vec)}
    return HeuristicWeights(**kwargs)


# ---------------------------------------------------------------------------
# 적합도(fitness) 평가: 그리디 자가 플레이 시뮬레이션
# ---------------------------------------------------------------------------
def _greedy_self_play(
    weights: HeuristicWeights,
    max_turns: int,
    rng: random.Random,
) -> Tuple[int, float]:
    """
    매 턴 무작위 블록 3개를 받아, 각 단계마다 evaluate() 가 가장 높은
    (블록, 배치)를 그리디하게 선택한다. (survived_turns, total_score) 반환.
    """
    board_bb = 0
    combo = 0
    total_score = 0.0
    survived = 0

    for _ in range(max_turns):
        tray = random_pieces(3, rng)
        if all(not can_place_anywhere(board_bb, p) for p in tray if p):
            break

        remaining = list(tray)
        while any(p for p in remaining):
            best = None
            for idx, cells in enumerate(remaining):
                if not cells:
                    continue
                for placement in valid_placements(board_bb, idx, cells):
                    step = apply_placement(board_bb, placement, combo)
                    score = evaluate(step.board_bb, step.score_gain, step.combo, remaining, weights)
                    if best is None or score > best[0]:
                        best = (score, idx, step)
            if best is None:
                break
            _, idx, step = best
            board_bb = step.board_bb
            combo = step.combo
            total_score += step.score_gain
            remaining[idx] = tuple()

        survived += 1

    return survived, total_score


def evaluate_weights(
    weights: HeuristicWeights,
    episodes: int = 3,
    max_turns: int = 25,
    rng: Optional[random.Random] = None,
) -> float:
    """survived_turns 와 score 를 결합한 적합도 점수(높을수록 좋음)."""
    rng = rng or random.Random()
    total = 0.0
    for _ in range(episodes):
        survived, score = _greedy_self_play(weights, max_turns, rng)
        total += survived * 10.0 + score
    return total / episodes


# ---------------------------------------------------------------------------
# 유전 알고리즘
# ---------------------------------------------------------------------------
@dataclass
class TuningResult:
    best_weights: HeuristicWeights
    best_fitness: float
    history: List[float]


def genetic_algorithm(
    base_weights: Optional[HeuristicWeights] = None,
    population_size: int = 12,
    generations: int = 8,
    episodes_per_eval: int = 2,
    max_turns: int = 20,
    mutation_rate: float = 0.2,
    mutation_scale: float = 0.3,
    elite_count: int = 2,
    rng: Optional[random.Random] = None,
) -> TuningResult:
    """
    base_weights 를 시드로 한 유전 알고리즘으로 HeuristicWeights 를 탐색한다.

    각 세대마다:
    1. 모집단의 각 개체를 evaluate_weights 로 평가
    2. 상위 elite_count 개체는 그대로 보존
    3. 나머지는 상위 개체들 중에서 부모를 뽑아 교차(crossover) + 변이(mutation)
    """
    rng = rng or random.Random()
    base = base_weights or HeuristicWeights()
    base_vec = weights_to_vector(base)
    dim = len(base_vec)

    # 초기 모집단: base + 약간의 변이
    population: List[np.ndarray] = [base_vec.copy()]
    for _ in range(population_size - 1):
        noise = (np.random.RandomState(rng.randrange(2**31)).randn(dim)) * mutation_scale * np.abs(base_vec)
        population.append(np.clip(base_vec + noise, 0.0, None))

    history: List[float] = []
    best_vec = base_vec
    best_fitness = -float("inf")

    for _gen in range(generations):
        scored: List[Tuple[float, np.ndarray]] = []
        for vec in population:
            w = vector_to_weights(vec)
            fitness = evaluate_weights(w, episodes=episodes_per_eval, max_turns=max_turns, rng=rng)
            scored.append((fitness, vec))

        scored.sort(key=lambda x: x[0], reverse=True)
        if scored[0][0] > best_fitness:
            best_fitness = scored[0][0]
            best_vec = scored[0][1]
        history.append(scored[0][0])

        # 다음 세대 구성
        next_population: List[np.ndarray] = [vec.copy() for _, vec in scored[:elite_count]]
        parents = [vec for _, vec in scored[: max(2, population_size // 2)]]

        while len(next_population) < population_size:
            p1, p2 = rng.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
            mask = np.random.RandomState(rng.randrange(2**31)).rand(dim) < 0.5
            child = np.where(mask, p1, p2)

            mutate_mask = np.random.RandomState(rng.randrange(2**31)).rand(dim) < mutation_rate
            noise = np.random.RandomState(rng.randrange(2**31)).randn(dim) * mutation_scale * np.abs(child)
            child = np.where(mutate_mask, child + noise, child)
            child = np.clip(child, 0.0, None)
            next_population.append(child)

        population = next_population

    return TuningResult(
        best_weights=vector_to_weights(best_vec),
        best_fitness=best_fitness,
        history=history,
    )


# ---------------------------------------------------------------------------
# 블록 확률 모델 (#10)
# ---------------------------------------------------------------------------
def _shape_signature(cells: PieceCells) -> Tuple[Tuple[int, int], ...]:
    return normalize_piece(cells)


PIECE_SIGNATURES: List[Tuple[Tuple[int, int], ...]] = [_shape_signature(p) for p in PIECE_LIBRARY]


def build_piece_probabilities(logger: DataLogger) -> Dict[int, float]:
    """
    로그에 기록된 pieces_json 에 등장한 블록 모양의 빈도를 세어,
    PIECE_LIBRARY 인덱스 -> 등장 확률(0~1) 딕셔너리를 반환한다.
    데이터가 없으면 균등 분포를 반환한다.
    """
    counts = {i: 0 for i in range(len(PIECE_LIBRARY))}
    total = 0

    for row in logger.fetch_all():
        try:
            pieces = json.loads(row["pieces_json"])
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        for p in pieces:
            if not p:
                continue
            arr = np.asarray(p)
            if arr.ndim == 2 and arr.shape[1] == 2:
                cells = tuple(tuple(int(v) for v in c) for c in p)
            else:
                cells = tuple(
                    (r, c)
                    for r in range(arr.shape[0])
                    for c in range(arr.shape[1])
                    if arr[r, c]
                )
            sig = _shape_signature(cells)
            for i, ref_sig in enumerate(PIECE_SIGNATURES):
                if sig == ref_sig:
                    counts[i] += 1
                    total += 1
                    break

    if total == 0:
        n = len(PIECE_LIBRARY)
        return {i: 1.0 / n for i in range(n)}

    return {i: c / total for i, c in counts.items()}


def weighted_random_pieces(
    probabilities: Dict[int, float],
    n: int = 3,
    rng: Optional[random.Random] = None,
) -> List[PieceCells]:
    """학습된 블록 확률 분포에 따라 n개의 블록을 무작위로 뽑는다."""
    rng = rng or random.Random()
    indices = list(probabilities.keys())
    weights = [probabilities[i] for i in indices]
    chosen = rng.choices(indices, weights=weights, k=n)
    return [PIECE_LIBRARY[i] for i in chosen]

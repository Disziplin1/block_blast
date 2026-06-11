"""
shape_templates.py
===================
Block Blast 에 등장하는 모든 블록 모양(Shape)을 미리 정의한 템플릿 목록과,
인식된 셀 점유 격자(occupancy grid)를 가장 유사한 템플릿으로 매칭하는
Template Matching 유틸리티.

Bounding Box 추정 대신, Grid Detection 결과(occupancy grid)를 표준 템플릿과
비교(Jaccard 유사도)하여 가장 비슷한 모양을 Solver 에 전달할 Shape Matrix 로
사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from utils import PieceCells, piece_cells_to_grid, piece_grid_to_cells


@dataclass(frozen=True)
class ShapeTemplate:
    name: str
    grid: np.ndarray
    cells: PieceCells


def _template_from_string(name: str, shape: str) -> ShapeTemplate:
    rows = shape.strip("\n").split("\n")
    grid = [[1 if ch == "#" else 0 for ch in row] for row in rows]
    cells = piece_grid_to_cells(grid)
    return ShapeTemplate(name=name, grid=piece_cells_to_grid(cells), cells=cells)


# Block Blast 에서 등장하는 모든 블록 모양 (회전 변형 포함).
_TEMPLATE_DEFS: List[Tuple[str, str]] = [
    # 1칸
    ("1x1", "#"),
    # 2칸
    ("1x2", "##"),
    ("2x1", "#\n#"),
    # 3칸
    ("1x3", "###"),
    ("3x1", "#\n#\n#"),
    # 4칸
    ("1x4", "####"),
    ("4x1", "#\n#\n#\n#"),
    # 5칸
    ("1x5", "#####"),
    ("5x1", "#\n#\n#\n#\n#"),
    # 정사각형
    ("2x2", "##\n##"),
    ("3x3", "###\n###\n###"),
    # L 모양 (4가지 회전)
    ("L1", "#.\n#.\n##"),
    ("L2", "##\n#.\n#."),
    ("L3", "##\n.#\n.#"),
    ("L4", ".#\n.#\n##"),
    # 역L(J) 모양 (4가지 회전)
    ("J1", ".#\n.#\n##"),
    ("J2", "##\n.#\n.#"),
    ("J3", "##\n#.\n#."),
    ("J4", "#.\n#.\n##"),
    # T 모양 (4가지 회전)
    ("T1", "###\n.#."),
    ("T2", ".#\n##\n.#"),
    ("T3", ".#.\n###"),
    ("T4", "#.\n##\n#."),
    # S / Z 모양
    ("S1", ".##\n##."),
    ("S2", "##.\n.##"),
    ("Z1", "#.\n##\n.#"),
    ("Z2", ".#\n##\n#."),
    # 큰 L (3칸짜리 변, 4가지 회전)
    ("BigL1", "#..\n#..\n###"),
    ("BigL2", "###\n#..\n#.."),
    ("BigL3", "###\n..#\n..#"),
    ("BigL4", "..#\n..#\n###"),
    # 십자 모양
    ("Plus", ".#.\n###\n.#."),
]

SHAPE_TEMPLATES: List[ShapeTemplate] = [
    _template_from_string(name, shape) for name, shape in _TEMPLATE_DEFS
]


def _grid_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 0/1 격자의 Jaccard 유사도(교집합/합집합)를 계산한다.

    두 격자 모두 좌상단(0,0) 기준으로 정규화(trim)되어 있다고 가정하고,
    더 큰 쪽의 크기에 맞춰 0으로 패딩한 뒤 같은 좌표끼리 비교한다.
    """
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])
    pa = np.zeros((h, w), dtype=np.int8)
    pb = np.zeros((h, w), dtype=np.int8)
    pa[: a.shape[0], : a.shape[1]] = a
    pb[: b.shape[0], : b.shape[1]] = b

    union = int(np.logical_or(pa, pb).sum())
    if union == 0:
        return 1.0
    intersection = int(np.logical_and(pa, pb).sum())
    return intersection / union


def match_template(grid: np.ndarray) -> Tuple[ShapeTemplate, float]:
    """occupancy grid(좌상단 정규화됨)와 가장 유사한 SHAPE_TEMPLATES 항목과
    유사도(0~1, 1이 완전 일치)를 반환한다."""
    best = SHAPE_TEMPLATES[0]
    best_score = -1.0
    for tmpl in SHAPE_TEMPLATES:
        score = _grid_similarity(grid, tmpl.grid)
        if score > best_score:
            best_score = score
            best = tmpl
    return best, best_score

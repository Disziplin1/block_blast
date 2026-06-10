"""
main.py
=======
Ultimate Block Blast AI Assistant 진입점.

사용법
------
    python main.py          # GUI 실행
    python main.py --selftest   # GUI 없이 솔버 스모크 테스트만 실행
"""

from __future__ import annotations

import argparse
import sys


def run_gui() -> None:
    from gui import main as gui_main

    gui_main()


def run_selftest() -> None:
    """PyQt5/화면 캡처 없이 핵심 알고리즘만 빠르게 점검한다."""
    import numpy as np
    from solver import solve

    board = np.zeros((8, 8), dtype=np.int8)
    pieces = [
        [[1, 1], [1, 1]],
        [[1, 1, 1]],
        [[1], [1], [1]],
    ]
    result = solve(board, pieces, run_monte_carlo=True)
    print(f"Evaluation time: {result.evaluation_time_sec * 1000:.1f} ms")
    for rec in result.recommendations:
        print(
            f"#{rec.rank} piece={rec.piece_index} pos=({rec.anchor_row},{rec.anchor_col}) "
            f"score={rec.expected_score} survival={rec.survival_stars} risk={rec.risk_level}"
        )
        print(f"   reasons: {rec.reasons}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ultimate Block Blast AI Assistant")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="GUI 없이 솔버 핵심 로직만 빠르게 테스트",
    )
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
        return

    run_gui()


if __name__ == "__main__":
    sys.exit(main() or 0)

#!/usr/bin/env python3
"""Sprawdza, czy threshold_sweep osiąga cel:
FPR_adv_fp_trap < 0.15 oraz TPR_aibaseline > 0.75.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

DEFAULT = Path(__file__).parent.parent / "results" / "latest" / "threshold_sweep.csv"


def to_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"[ERROR] Missing file: {path}")
        sys.exit(1)

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    winners = [
        r for r in rows
        if to_float(r.get("FPR_adv_fp_trap", "1"), 1.0) < 0.15
        and to_float(r.get("TPR_aibaseline", "0"), 0.0) > 0.75
    ]

    print(f"Total sweep rows: {len(rows)}")
    print(f"Rows meeting target: {len(winners)}")

    if winners:
        print("\nTop candidates:")
        top = sorted(
            winners,
            key=lambda r: (to_float(r.get("FPR_adv_fp_trap", "1"), 1.0), -to_float(r.get("TPR_aibaseline", "0"), 0.0)),
        )[:10]
        for r in top:
            print(
                f"  pts={r.get('points_thr','?')} "
                f"FPR={r.get('FPR_adv_fp_trap','?')} "
                f"TPR={r.get('TPR_aibaseline','?')} "
                f"F1={r.get('f1','?')}"
            )


if __name__ == "__main__":
    main()

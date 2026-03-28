#!/usr/bin/env python3
"""Audyt FP w adv_fp_trap: ile przypadków ma ai_specific=0.

Użycie:
  python kod/tools/fp_aispecific_audit.py
  python kod/tools/fp_aispecific_audit.py kod/results/latest/evaluation_results.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

DEFAULT = Path(__file__).parent.parent / "results" / "latest" / "evaluation_results.csv"


def to_int(x: str | int | float | None, default: int = 0) -> int:
    if x in ("", None):
        return default
    return int(float(x))


def parse_ai_specific(row: dict) -> int:
    if "ai_specific" in row and row["ai_specific"] not in ("", None):
        return to_int(row["ai_specific"], 0)

    mode = row.get("fusion_mode", "")
    if "ai_specific=1" in mode:
        return 1
    return 0


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"[ERROR] Missing file: {path}")
        sys.exit(1)

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    fp_rows = [
        r for r in rows
        if r.get("category") == "adv_fp_trap"
        and to_int(r.get("ground_truth"), 0) == 0
        and to_int(r.get("detected"), 0) == 1
    ]

    fp_ai0 = [r for r in fp_rows if parse_ai_specific(r) == 0]
    fp_ai1 = [r for r in fp_rows if parse_ai_specific(r) == 1]
    fp_broadcast = [r for r in fp_rows if to_int(r.get("broadcast_trap"), 0) == 1]
    fp_pattern_trap = [r for r in fp_rows if to_int(r.get("broadcast_pattern_trap"), 0) == 1]
    fp_guard_violation = [
        r for r in fp_rows
        if parse_ai_specific(r) == 0 and to_int(r.get("detected"), 0) == 1
    ]
    fp_lowerthird_ai0 = [
        r for r in fp_rows
        if parse_ai_specific(r) == 0 and float(r.get("of_lower_third_roi_ratio", 0.0) or 0.0) > 0.60
    ]

    print(f"adv_fp_trap FP total: {len(fp_rows)}")
    print(f"  with ai_specific=0: {len(fp_ai0)}")
    print(f"  with ai_specific=1: {len(fp_ai1)}")
    print(f"  with broadcast_trap=1: {len(fp_broadcast)}")
    print(f"  with broadcast_pattern_trap=1: {len(fp_pattern_trap)}")
    print(f"  guard violations (detected=1 & ai_specific=0): {len(fp_guard_violation)}")
    print(f"  lower-third + ai_specific=0: {len(fp_lowerthird_ai0)}")

    if fp_ai0:
        print("\nTop FP (ai_specific=0):")
        for r in fp_ai0[:20]:
            print(f"  - {r.get('filename','')} | mode={r.get('fusion_mode','')}")

    if fp_guard_violation:
        print("\n[WARN] Guard violations found (should be 0):")
        for r in fp_guard_violation[:20]:
            print(f"  - {r.get('filename','')} | detected={r.get('detected')} | mode={r.get('fusion_mode','')}")


if __name__ == "__main__":
    main()

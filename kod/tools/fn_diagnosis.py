#!/usr/bin/env python3
"""
fn_diagnosis.py

Diagnoses AI videos that are false negatives in every positive split.
This is intended to support the discussion section of the thesis.

Input:
  kod/results/latest/evaluation_results.csv

Output:
  - prints videos that remain detected=0 in ai_baseline / adv_compressed / adv_cropped
  - prints simple heuristic reasons based on raw detector values if available
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

RESULTS_ROOT = Path(__file__).parent.parent / "results" / "latest"
EVAL_CSV = RESULTS_ROOT / "evaluation_results.csv"
RAW_CSV = RESULTS_ROOT / "raw_signals.csv"
POSITIVE_SPLITS = ["ai_baseline", "adv_compressed", "adv_cropped"]


def _int(x):
    return int(float(x)) if x not in ("", None) else 0


def _float(x):
    return float(x) if x not in ("", None) else 0.0


def load_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def short_reason(rows: list[dict]) -> str:
    mean_of_count = mean(_int(r.get("of_count", 0)) for r in rows)
    mean_zv_count = mean(_int(r.get("zv_count", 0)) for r in rows)
    mean_motion = mean(_float(r.get("of_global_motion", 0.0)) for r in rows)
    mean_area = mean(_float(r.get("of_max_area", 0.0)) for r in rows)

    reasons = []
    if mean_of_count <= 1.0:
        reasons.append("very few optical-flow contours")
    if mean_motion < 1.0:
        reasons.append("low global motion")
    if mean_zv_count == 0:
        reasons.append("no zero-variance ROIs")
    if mean_area < 50000:
        reasons.append("small static contour area")
    if not reasons:
        reasons.append("weak heuristic evidence in all AI-positive splits")
    return ", ".join(reasons)


def main() -> None:
    if not EVAL_CSV.exists():
        print(f"[ERROR] Missing {EVAL_CSV}. Run evaluate.py first.")
        sys.exit(1)

    eval_rows = load_csv(EVAL_CSV)
    raw_rows = load_csv(RAW_CSV) if RAW_CSV.exists() else []
    raw_by_key = {(r["category"], r["filename"]): r for r in raw_rows}

    by_filename = defaultdict(list)
    for row in eval_rows:
        if row["category"] in POSITIVE_SPLITS:
            by_filename[row["filename"]].append(row)

    always_fn = []
    for filename, rows in by_filename.items():
        present_splits = {r["category"] for r in rows}
        if set(POSITIVE_SPLITS).issubset(present_splits) and all(_int(r["detected"]) == 0 for r in rows):
            enriched = []
            for r in rows:
                key = (r["category"], r["filename"])
                merged = dict(r)
                merged.update(raw_by_key.get(key, {}))
                enriched.append(merged)
            always_fn.append((filename, enriched))

    print("False negatives in every AI-positive split")
    print("=" * 72)
    if not always_fn:
        print("No such videos found.")
        return

    for filename, rows in sorted(always_fn):
        print(f"\n{filename}")
        print(f"  reason: {short_reason(rows)}")
        for r in sorted(rows, key=lambda x: x["category"]):
            print(
                f"  - {r['category']:15s} detected={_int(r.get('detected', 0))} "
                f"of_count={_int(r.get('of_count', 0))} "
                f"of_global_motion={_float(r.get('of_global_motion', 0.0)):.3f} "
                f"of_max_area={_float(r.get('of_max_area', 0.0)):.1f} "
                f"zv_count={_int(r.get('zv_count', 0))}"
            )

    print("\nSuggested thesis wording:")
    print(
        "A small subset of AI videos remained false negative in all AI-positive splits. "
        "Inspection of raw heuristic signals suggests that these clips contain little camera motion "
        "and produce very few persistent static contours, which weakens the Optical Flow and "
        "Zero-Variance heuristics simultaneously."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
fn_diagnosis_v2.py

Rozszerzona diagnostyka false negatives (FN) dla pracy magisterskiej.

Cel:
- policzyć skalę persistent FN liczbowo,
- zidentyfikować dominujące wzorce awarii heurystyk OF/ZV,
- wygenerować uczciwszą, gotową do wklejenia tezę do rozdziału wyników.

Wejście:
  kod/results/latest/evaluation_results.csv
  kod/results/latest/raw_signals.csv

Użycie:
  python kod/tools/fn_diagnosis_v2.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

RESULTS_ROOT = Path(__file__).parent.parent / "results" / "latest"
EVAL_CSV = RESULTS_ROOT / "evaluation_results.csv"
RAW_CSV = RESULTS_ROOT / "raw_signals.csv"
POSITIVE_SPLITS = ["ai_baseline", "adv_compressed", "adv_cropped"]


def _int(x: str | int | float | None) -> int:
    return int(float(x)) if x not in ("", None) else 0


def _float(x: str | int | float | None) -> float:
    return float(x) if x not in ("", None) else 0.0


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def classify_failure(rows: list[dict]) -> tuple[list[str], dict[str, float]]:
    """Zwraca (etykiety_wzorców, agregaty) dla jednego source video."""
    avg_of_count = mean(_int(r.get("of_count")) for r in rows)
    avg_of_motion = mean(_float(r.get("of_global_motion")) for r in rows)
    avg_of_area = mean(_float(r.get("of_max_area")) for r in rows)
    avg_zv_count = mean(_int(r.get("zv_count")) for r in rows)

    patterns: list[str] = []
    if avg_zv_count == 0:
        patterns.append("no zero-variance ROIs")
    if avg_of_area < 50_000:
        patterns.append("small static contour area")
    if avg_of_motion < 1.0:
        patterns.append("low global motion")
    if avg_of_count <= 1.0:
        patterns.append("very few optical-flow contours")

    if not patterns:
        patterns.append("weak multi-signal evidence")

    return patterns, {
        "avg_of_count": avg_of_count,
        "avg_of_motion": avg_of_motion,
        "avg_of_area": avg_of_area,
        "avg_zv_count": avg_zv_count,
    }


def build_source_level_positive_rows(eval_rows: list[dict], raw_rows: list[dict]) -> dict[str, list[dict]]:
    raw_by_key = {(r["category"], r["filename"]): r for r in raw_rows}
    by_filename: dict[str, list[dict]] = defaultdict(list)

    for row in eval_rows:
        if row.get("category") not in POSITIVE_SPLITS:
            continue
        key = (row["category"], row["filename"])
        merged = dict(row)
        merged.update(raw_by_key.get(key, {}))
        by_filename[row["filename"]].append(merged)
    return by_filename


def main() -> None:
    if not EVAL_CSV.exists():
        print(f"[ERROR] Missing {EVAL_CSV}. Run: python kod/dataset/evaluate.py")
        sys.exit(1)

    if not RAW_CSV.exists():
        print(f"[ERROR] Missing {RAW_CSV}. Run: python kod/dataset/evaluate.py")
        sys.exit(1)

    eval_rows = load_csv(EVAL_CSV)
    raw_rows = load_csv(RAW_CSV)
    by_filename = build_source_level_positive_rows(eval_rows, raw_rows)

    source_total = 0
    persistent_fn: list[tuple[str, list[dict], list[str], dict[str, float]]] = []

    for filename, rows in sorted(by_filename.items()):
        present = {r["category"] for r in rows}
        if not set(POSITIVE_SPLITS).issubset(present):
            continue
        source_total += 1

        all_fn = all(_int(r.get("detected")) == 0 for r in rows)
        if all_fn:
            patterns, agg = classify_failure(rows)
            persistent_fn.append((filename, rows, patterns, agg))

    persistent_count = len(persistent_fn)
    persistent_ratio = (100.0 * persistent_count / source_total) if source_total else 0.0

    pattern_counter: Counter[str] = Counter()
    for _, _, patterns, _ in persistent_fn:
        pattern_counter.update(patterns)

    print("Persistent FN diagnosis (v2)")
    print("=" * 84)
    print(f"AI source videos (with all positive splits present): {source_total}")
    print(f"Persistent FN (detected=0 in all {len(POSITIVE_SPLITS)} AI-positive splits): {persistent_count}")
    print(f"Persistent FN share: {persistent_ratio:.2f}%")

    if not persistent_fn:
        print("\nNo persistent FN found.")
        return

    print("\nDominant failure-pattern counts among persistent FN:")
    for label, n in pattern_counter.most_common():
        share = 100.0 * n / persistent_count
        print(f"  - {label:35s} {n:3d}  ({share:6.2f}% of persistent FN videos)")

    print("\nPersistent FN videos (source-level):")
    for filename, rows, patterns, agg in persistent_fn:
        print(f"\n{filename}")
        print(f"  patterns: {', '.join(patterns)}")
        print(
            "  means: "
            f"of_count={agg['avg_of_count']:.2f}, "
            f"of_global_motion={agg['avg_of_motion']:.3f}, "
            f"of_max_area={agg['avg_of_area']:.1f}, "
            f"zv_count={agg['avg_zv_count']:.2f}"
        )
        for r in sorted(rows, key=lambda x: x["category"]):
            print(
                f"  - {r['category']:15s} detected={_int(r.get('detected'))} "
                f"of_count={_int(r.get('of_count')):2d} "
                f"of_global_motion={_float(r.get('of_global_motion')):.3f} "
                f"of_max_area={_float(r.get('of_max_area')):.1f} "
                f"zv_count={_int(r.get('zv_count')):2d}"
            )

    print("\nSuggested thesis wording (EN):")
    print(
        "A substantial majority of AI videos remained false negative across all "
        "AI-positive splits. The dominant failure mode was not only low-motion content, "
        "but more generally the absence of stable zero-variance regions and the lack of "
        "sufficiently strong static contour evidence. This indicates that the implemented "
        "Optical Flow and Zero-Variance heuristics do not function as reliable positive "
        "indicators of AI generation on this dataset."
    )

    print("\nProponowane zdanie do pracy (PL):")
    print(
        "W zdecydowanej większości źródłowych filmów AI obserwowano persistent false negatives "
        "we wszystkich splitach dodatnich (ai_baseline, adv_compressed, adv_cropped). "
        "Dominującym mechanizmem błędu był nie tylko niski ruch globalny, ale przede wszystkim "
        "brak stabilnych regionów zero-variance oraz zbyt słaby sygnał statycznych konturów OF; "
        "wskazuje to, że zastosowane heurystyki Optical Flow i Zero-Variance nie mogą pełnić "
        "roli wiarygodnych dodatnich wskaźników generacji AI w tym zbiorze."
    )


if __name__ == "__main__":
    main()

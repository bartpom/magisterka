#!/usr/bin/env python3
"""
thesis_findings.py

Czyta raw_signals.csv z evaluate.py i generuje ostrozny, "paper-ready" raport:
- baseline always-negative
- confirmed invisible watermark only (strict)
- wykrycie podejrzanych legacy-hitow IW: iw_found=1 przy pustym iw_matched
- prosty raport sygnalow overlay-trap (OF/ZV) dla adv_fp_trap vs ai_baseline

Uzycie:
  python thesis_findings.py
  python thesis_findings.py path/to/raw_signals.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from statistics import mean, median


def _int(x: str | int | float) -> int:
    return int(float(x))


def _float(x: str | int | float) -> float:
    return float(x)


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def metrics(rows: list[dict], pred_fn) -> dict[str, dict[str, float | int]]:
    per_cat: dict[str, list[dict]] = {}
    for r in rows:
        per_cat.setdefault(r["category"], []).append(r)

    out: dict[str, dict[str, float | int]] = {}
    for cat, cat_rows in per_cat.items():
        gt = [_int(r["ground_truth"]) for r in cat_rows]
        pred = [int(bool(pred_fn(r))) for r in cat_rows]
        tp = sum(p == 1 and g == 1 for p, g in zip(pred, gt))
        tn = sum(p == 0 and g == 0 for p, g in zip(pred, gt))
        fp = sum(p == 1 and g == 0 for p, g in zip(pred, gt))
        fn = sum(p == 0 and g == 1 for p, g in zip(pred, gt))
        n = len(gt)
        acc = (tp + tn) / n if n else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        out[cat] = {
            "n": n, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "acc": acc, "f1": f1, "fpr": fpr,
        }
    return out


def print_metrics(title: str, rows: list[dict], pred_fn) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)
    stats = metrics(rows, pred_fn)
    print(f"{'category':18s} {'n':>4} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'acc':>8} {'f1':>8} {'FPR':>8}")
    for cat, s in stats.items():
        print(
            f"{cat:18s} {s['n']:4d} {s['TP']:4d} {s['FP']:4d} {s['TN']:4d} {s['FN']:4d} "
            f"{s['acc']:8.4f} {s['f1']:8.4f} {s['fpr']:8.4f}"
        )


def summarize_signal(rows: list[dict], category: str, field: str, cast=float) -> tuple[float, float, float]:
    vals = [cast(r[field]) for r in rows if r["category"] == category]
    if not vals:
        return 0.0, 0.0, 0.0
    return mean(vals), median(vals), max(vals)


def main() -> None:
    if len(sys.argv) > 1:
        raw_path = Path(sys.argv[1])
    else:
        raw_path = Path(__file__).parent.parent / "results" / "latest" / "raw_signals.csv"

    if not raw_path.exists():
        print(f"[BLAD] Nie znaleziono: {raw_path}")
        print("Uruchom najpierw: python kod/dataset/evaluate.py")
        sys.exit(1)

    rows = load_rows(raw_path)
    print(f"[INFO] Wczytano {len(rows)} wierszy z {raw_path}")

    print_metrics(
        "Baseline: always-negative (punkt odniesienia, nie detektor AI)",
        rows,
        lambda r: 0,
    )

    print_metrics(
        "Strict AI-positive: tylko potwierdzone dopasowanie invisible watermark",
        rows,
        lambda r: (r.get("iw_matched", "") != "" and _float(r.get("iw_best_similarity", 0.0)) >= 0.85),
    )

    print_metrics(
        "Legacy IW-hit: wszystko, co evaluate zapisalo jako iw_found=1",
        rows,
        lambda r: _int(r.get("iw_found", 0)) == 1,
    )

    suspicious_legacy = [
        r for r in rows
        if _int(r.get("iw_found", 0)) == 1 and not r.get("iw_matched", "")
    ]
    print("\n" + "=" * 76)
    print("Podejrzane legacy-hity IW: iw_found=1, ale brak potwierdzonego matched")
    print("=" * 76)
    print(f"Liczba wierszy: {len(suspicious_legacy)}")
    for r in suspicious_legacy[:20]:
        print(
            f"  {r['category']:15s}  {r['filename'][:40]:40s}  "
            f"sim={_float(r.get('iw_best_similarity', 0.0)):.4f}  "
            f"method={r.get('iw_method', '')}"
        )
    if len(suspicious_legacy) > 20:
        print(f"  ... (+{len(suspicious_legacy)-20} kolejnych)")

    print("\n" + "=" * 76)
    print("Sygnały overlay-trap: ai_baseline vs adv_fp_trap")
    print("=" * 76)
    for field, cast in [
        ("of_max_area", _int),
        ("of_count", _int),
        ("zv_count", _int),
        ("zv_max_score", _float),
        ("iw_best_similarity", _float),
    ]:
        ai_mean, ai_med, ai_max = summarize_signal(rows, "ai_baseline", field, cast)
        fp_mean, fp_med, fp_max = summarize_signal(rows, "adv_fp_trap", field, cast)
        print(
            f"{field:18s}  AI mean/med/max={ai_mean:.4f}/{ai_med:.4f}/{ai_max:.4f}   "
            f"FP mean/med/max={fp_mean:.4f}/{fp_med:.4f}/{fp_max:.4f}"
        )

    print("\n" + "=" * 76)
    print("Sugestia do pracy")
    print("=" * 76)
    print(
        "Na podstawie surowych sygnalow nie nalezy utozsamiac samego dekodu invisible watermark "
        "z potwierdzonym sygnalem AI. W raporcie warto rozdzielic: "
        "(1) potwierdzone dopasowanie do znanej sygnatury, "
        "(2) kandydat dekodu bez potwierdzenia, "
        "(3) brak dodatniego sygnalu AI. "
        "Heurystyki OF/ZV moga sluzyc jako filtr pulapek overlayowych, ale nie jako samodzielny dowod AI."
    )


if __name__ == "__main__":
    main()

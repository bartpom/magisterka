#!/usr/bin/env python3
"""
compare_external_apps.py

Porównuje skuteczność Twojej aplikacji z innymi narzędziami detekcji watermarków.

Wejście:
1) CSV z wynikami własnego pipeline (domyślnie: kod/results/latest/evaluation_results.csv)
2) CSV z wynikami narzędzia zewnętrznego (własny format, mapowany przez argumenty)

Wynik:
- metryki per narzędzie: TP/TN/FP/FN, accuracy, precision, recall, F1, FPR
- CSV z porównaniem per plik

Przykład:
python kod/tools/compare_external_apps.py \
  --external-csv tmp/tool_x.csv \
  --external-name ToolX \
  --ext-filename-col filename \
  --ext-pred-col detected
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_int(value: str | int | float | None, default: int = 0) -> int:
    if value in ("", None):
        return default
    return int(float(value))


def compute_metrics(rows: list[dict], pred_key: str, gt_key: str = "ground_truth") -> dict[str, float | int]:
    tp = tn = fp = fn = 0
    for r in rows:
        gt = as_int(r.get(gt_key), 0)
        pred = as_int(r.get(pred_key), 0)
        tp += int(pred == 1 and gt == 1)
        tn += int(pred == 0 and gt == 0)
        fp += int(pred == 1 and gt == 0)
        fn += int(pred == 0 and gt == 1)

    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "n": n,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Porównanie metryk własnej aplikacji z narzędziem zewnętrznym.")
    parser.add_argument(
        "--ours-csv",
        type=Path,
        default=Path("kod/results/latest/evaluation_results.csv"),
        help="CSV z evaluate.py (z kolumnami filename, ground_truth, detected)",
    )
    parser.add_argument("--external-csv", type=Path, required=True)
    parser.add_argument("--external-name", type=str, default="ExternalTool")
    parser.add_argument("--ext-filename-col", type=str, default="filename")
    parser.add_argument("--ext-pred-col", type=str, default="detected")
    parser.add_argument("--ext-positive-value", type=str, default="1")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("kod/results/latest/external_comparison.csv"),
    )
    args = parser.parse_args()

    if not args.ours_csv.exists():
        raise FileNotFoundError(f"Brak pliku ours CSV: {args.ours_csv}")
    if not args.external_csv.exists():
        raise FileNotFoundError(f"Brak pliku external CSV: {args.external_csv}")

    ours = load_csv(args.ours_csv)
    ext = load_csv(args.external_csv)

    ours_by_filename = {r.get("filename", ""): r for r in ours}

    merged: list[dict] = []
    for r in ext:
        filename = r.get(args.ext_filename_col, "")
        if not filename or filename not in ours_by_filename:
            continue

        ours_row = ours_by_filename[filename]
        ext_raw = r.get(args.ext_pred_col, "")
        ext_pred = int(str(ext_raw).strip() == args.ext_positive_value)

        merged.append({
            "filename": filename,
            "category": ours_row.get("category", ""),
            "ground_truth": as_int(ours_row.get("ground_truth")),
            "our_detected": as_int(ours_row.get("detected")),
            "external_detected": ext_pred,
        })

    if not merged:
        raise RuntimeError("Brak wspólnych plików między CSV aplikacji własnej i zewnętrznej.")

    ours_metrics = compute_metrics(merged, pred_key="our_detected")
    ext_metrics = compute_metrics(merged, pred_key="external_detected")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "category", "ground_truth", "our_detected", "external_detected"],
        )
        writer.writeheader()
        writer.writerows(merged)

    print(f"[INFO] Wspólne próbki: {len(merged)}")
    print("\n[OUR APP]")
    for k, v in ours_metrics.items():
        print(f"  {k}: {v}")

    print(f"\n[{args.external_name}]")
    for k, v in ext_metrics.items():
        print(f"  {k}: {v}")

    print(f"\n[CSV] zapisano: {args.out_csv}")


if __name__ == "__main__":
    main()

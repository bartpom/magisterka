#!/usr/bin/env python3
"""
summarize_latest_results.py

Tworzy zwięzły raport markdown z katalogu `kod/results/latest/`.

Wejście (domyślnie):
- evaluation_results.csv
- metrics_summary.csv
- opcjonalnie:
  - run_info.txt
  - best_config_selection.csv
  - pareto_frontier.csv
  - feature_activation_summary.csv

Wyjście:
- thesis_summary.md
- global_confusion_matrix.csv

Cel:
- ograniczyć ręczne przepisywanie liczb do pracy / README,
- mieć jedno spójne źródło globalnych TP/TN/FP/FN i metryk,
- ułatwić budowę tabel do rozdziału ewaluacyjnego.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_int(value, default: int = 0) -> int:
    if value in ("", None):
        return default
    return int(float(value))


def as_float(value, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    return float(value)


def md_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    rows = list(rows)
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def compute_global_metrics(eval_rows: list[dict]) -> dict[str, float | int]:
    tp = tn = fp = fn = 0
    for row in eval_rows:
        gt = as_int(row.get("ground_truth"))
        pred = as_int(row.get("detected"))
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
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "N": n,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "specificity": round(specificity, 4),
    }


def parse_run_info(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    info: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        info[key.strip()] = value.strip()
    return info


def write_confusion_csv(path: Path, metrics: dict[str, float | int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["actual", "predicted", "count"])
        writer.writeheader()
        writer.writerows([
            {"actual": "AI", "predicted": "AI", "count": metrics["TP"]},
            {"actual": "AI", "predicted": "not-AI", "count": metrics["FN"]},
            {"actual": "not-AI", "predicted": "AI", "count": metrics["FP"]},
            {"actual": "not-AI", "predicted": "not-AI", "count": metrics["TN"]},
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown summary from kod/results/latest.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "latest",
        help="Path to latest benchmark results (default: kod/results/latest)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output markdown path. Default: <results-root>/thesis_summary.md",
    )
    args = parser.parse_args()

    results_root = args.results_root
    out_path = args.out or (results_root / "thesis_summary.md")

    metrics_csv = results_root / "metrics_summary.csv"
    eval_csv = results_root / "evaluation_results.csv"
    best_csv = results_root / "best_config_selection.csv"
    pareto_csv = results_root / "pareto_frontier.csv"
    activation_csv = results_root / "feature_activation_summary.csv"
    run_info_txt = results_root / "run_info.txt"

    if not metrics_csv.exists() or not eval_csv.exists():
        raise FileNotFoundError(
            "Missing metrics_summary.csv or evaluation_results.csv. Run: python kod/dataset/evaluate.py"
        )

    metrics_rows = read_csv(metrics_csv)
    eval_rows = read_csv(eval_csv)
    best_rows = read_csv(best_csv) if best_csv.exists() else []
    pareto_rows = read_csv(pareto_csv) if pareto_csv.exists() else []
    activation_rows = read_csv(activation_csv) if activation_csv.exists() else []
    run_info = parse_run_info(run_info_txt)
    global_metrics = compute_global_metrics(eval_rows)

    confusion_csv = results_root / "global_confusion_matrix.csv"
    write_confusion_csv(confusion_csv, global_metrics)

    lines: list[str] = []
    lines.append("# Benchmark summary (auto-generated)")
    lines.append("")
    lines.append("This file was generated from `kod/results/latest/`. Do not edit it manually.")
    lines.append("")

    if run_info:
        lines.append("## Run metadata")
        lines.append("")
        lines.append(md_table(["field", "value"], [[k, v] for k, v in run_info.items()]))
        lines.append("")

    lines.append("## Per-split metrics")
    lines.append("")
    lines.append(md_table(
        ["category", "n", "TP", "TN", "FP", "FN", "accuracy", "precision", "recall", "f1", "FPR", "specificity"],
        [
            [
                row.get("category", ""),
                row.get("n", ""),
                row.get("TP", ""),
                row.get("TN", ""),
                row.get("FP", ""),
                row.get("FN", ""),
                row.get("accuracy", ""),
                row.get("precision", ""),
                row.get("recall", ""),
                row.get("f1", ""),
                row.get("FPR", ""),
                row.get("specificity", ""),
            ]
            for row in metrics_rows
        ],
    ))
    lines.append("")

    lines.append("## Global confusion matrix")
    lines.append("")
    lines.append(md_table(
        ["actual \\ predicted", "AI", "not-AI"],
        [
            ["AI", global_metrics["TP"], global_metrics["FN"]],
            ["not-AI", global_metrics["FP"], global_metrics["TN"]],
        ],
    ))
    lines.append("")
    lines.append(md_table(["metric", "value"], [[k, v] for k, v in global_metrics.items()]))
    lines.append("")

    if best_rows:
        lines.append("## Best configuration from sweep")
        lines.append("")
        best = best_rows[0]
        lines.append(md_table(list(best.keys()), [list(best.values())]))
        lines.append("")

    if pareto_rows:
        lines.append("## Pareto frontier (first rows)")
        lines.append("")
        head = pareto_rows[: min(10, len(pareto_rows))]
        lines.append(md_table(list(head[0].keys()), [list(row.values()) for row in head]))
        lines.append("")

    if activation_rows and best_rows:
        best = best_rows[0]
        hf = str(best.get("hf_ratio_threshold", ""))
        tex = str(best.get("low_texture_threshold", ""))
        selected = [
            row for row in activation_rows
            if str(row.get("hf_ratio_threshold", "")) == hf
            and str(row.get("low_texture_threshold", "")) == tex
        ]
        if selected:
            lines.append("## Feature activation rates for selected HF/texture thresholds")
            lines.append("")
            lines.append(md_table(
                [
                    "category",
                    "low_hf_rate",
                    "low_texture_rate",
                    "corner_compact_rate",
                    "scoreboard_trap_rate",
                    "billboard_trap_rate",
                    "pattern_trap_rate",
                ],
                [
                    [
                        row.get("category", ""),
                        row.get("low_hf_rate", ""),
                        row.get("low_texture_rate", ""),
                        row.get("corner_compact_rate", ""),
                        row.get("scoreboard_trap_rate", ""),
                        row.get("billboard_trap_rate", ""),
                        row.get("pattern_trap_rate", ""),
                    ]
                    for row in selected
                ],
            ))
            lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Global precision should always be interpreted together with recall and FPR.")
    lines.append("- `global_confusion_matrix.csv` was written next to this summary for plotting or thesis tables.")
    lines.append("- The summary intentionally mirrors benchmark artifacts to reduce manual transcription errors.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {out_path}")
    print(f"[OK] wrote {confusion_csv}")


if __name__ == "__main__":
    main()

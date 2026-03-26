#!/usr/bin/env python3
"""
plot_results_v2.py

Final thesis-ready figures based on raw_signals.csv.

Improvements over plot_results.py:
- all figure titles are in English,
- signal-separation view includes all AI splits (baseline / compressed / cropped),
- adds an explicit OF robustness figure to show whether compression/cropping
  materially changes the Optical Flow signal,
- keeps the strict IW definition conservative:
    confirmed iw_matched AND similarity >= 0.85
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

DATASET_ROOT = Path(__file__).parent
RESULTS_DIR = DATASET_ROOT.parent / "results"
RAW_CSV = RESULTS_DIR / "latest" / "raw_signals.csv"
OUT_DIR = RESULTS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

_IW_EMPTY = {"", "nieznany", "none", "(brak)", "null"}
CATEGORY_ORDER = ["ai_baseline", "adv_compressed", "adv_cropped", "adv_fp_trap"]
CATEGORY_LABELS = {
    "ai_baseline": "AI baseline",
    "adv_compressed": "AI compressed",
    "adv_cropped": "AI cropped",
    "adv_fp_trap": "Real TV / FP trap",
}
CATEGORY_COLORS = {
    "ai_baseline": "#2c7bb6",
    "adv_compressed": "#1a9641",
    "adv_cropped": "#fdae61",
    "adv_fp_trap": "#d7191c",
}

plt.rcParams.update({"font.size": 12, "font.family": "serif"})


def _float(x):
    return float(x) if x not in ("", None) else 0.0


def _int(x):
    return int(float(x)) if x not in ("", None) else 0


def is_strict_iw(row):
    matched = str(row.get("iw_matched", "")).strip().lower()
    return matched not in _IW_EMPTY and _float(row.get("iw_best_similarity", 0)) >= 0.85


def load_rows():
    if not RAW_CSV.exists():
        print(f"[ERROR] Missing {RAW_CSV}. Run evaluate.py first.")
        raise SystemExit(1)
    with RAW_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compute_metrics(rows, pred_fn):
    stats = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)

    for category, rr in grouped.items():
        gt = [_int(r["ground_truth"]) for r in rr]
        pred = [int(bool(pred_fn(r))) for r in rr]
        tp = sum(p == 1 and g == 1 for p, g in zip(pred, gt))
        tn = sum(p == 0 and g == 0 for p, g in zip(pred, gt))
        fp = sum(p == 1 and g == 0 for p, g in zip(pred, gt))
        fn = sum(p == 0 and g == 1 for p, g in zip(pred, gt))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        stats[category] = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "n": len(gt),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
        }
    return stats


def boxplot_for_categories(ax, rows, field, categories, ylabel, title):
    data = [[_float(r[field]) for r in rows if r["category"] == cat] for cat in categories]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        widths=0.55,
    )
    for patch, cat in zip(bp["boxes"], categories):
        patch.set_facecolor(CATEGORY_COLORS[cat])
        patch.set_alpha(0.72)

    ax.set_xticks(range(1, len(categories) + 1))
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in categories], rotation=12)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.35)


def plot_confusion_matrix(rows):
    gt = [_int(r["ground_truth"]) for r in rows]
    pred = [int(bool(is_strict_iw(r))) for r in rows]

    tp = sum(p == 1 and g == 1 for p, g in zip(pred, gt))
    tn = sum(p == 0 and g == 0 for p, g in zip(pred, gt))
    fp = sum(p == 1 and g == 0 for p, g in zip(pred, gt))
    fn = sum(p == 0 and g == 1 for p, g in zip(pred, gt))
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Real", "Predicted AI"])
    ax.set_yticklabels(["Actual Real", "Actual AI"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=20,
            )
    ax.set_title(
        "Confusion Matrix — Strict Invisible Watermark Definition\n"
        "(confirmed iw_matched and similarity >= 0.85)",
        fontsize=14,
        pad=12,
    )
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    fig.tight_layout()
    out = OUT_DIR / "confusion_matrix_v2.png"
    fig.savefig(out, dpi=300)
    print(f"Saved {out}")
    plt.close(fig)


def plot_metrics_bar(rows):
    stats = compute_metrics(rows, is_strict_iw)
    cats = [c for c in CATEGORY_ORDER if c in stats]
    labels = [CATEGORY_LABELS[c] for c in cats]
    metrics = ["precision", "recall", "f1"]
    metric_colors = ["#2c7bb6", "#1a9641", "#d7191c"]
    x = np.arange(len(cats))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (metric_name, color) in enumerate(zip(metrics, metric_colors)):
        vals = [stats[c][metric_name] for c in cats]
        bars = ax.bar(x + i * width - width, vals, width, label=metric_name.capitalize(), color=color, alpha=0.86)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_xlabel("Category")
    ax.set_title(
        "Performance Metrics — Strict Invisible Watermark Definition\n"
        "(TP = 0 everywhere: no confirmed watermark detected)",
        fontsize=14,
    )
    ax.legend(title="Metric")
    ax.axhline(0, color="black", linewidth=0.8)
    if "adv_fp_trap" in stats:
        fpr = stats["adv_fp_trap"]["fpr"]
        ax.annotate(
            f"FPR = {fpr:.0%}",
            xy=(cats.index("adv_fp_trap"), 0.05),
            ha="center",
            color="gray",
            fontsize=10,
        )

    fig.tight_layout()
    out = OUT_DIR / "metrics_bar_chart_v2.png"
    fig.savefig(out, dpi=300)
    print(f"Saved {out}")
    plt.close(fig)


def plot_signal_separation_all(rows):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    ai_categories = ["ai_baseline", "adv_compressed", "adv_cropped"]
    all_categories = ["ai_baseline", "adv_compressed", "adv_cropped", "adv_fp_trap"]

    boxplot_for_categories(
        axes[0],
        rows,
        "of_count",
        all_categories,
        ylabel="Number of OF contours",
        title="Signal separation: of_count",
    )
    boxplot_for_categories(
        axes[1],
        rows,
        "of_max_area",
        all_categories,
        ylabel="OF max area [px²]",
        title="Signal separation: of_max_area",
    )
    boxplot_for_categories(
        axes[2],
        rows,
        "zv_max_score",
        all_categories,
        ylabel="ZV max score",
        title="Signal separation: zv_max_score",
    )

    legend_handles = [
        mpatches.Patch(color=CATEGORY_COLORS[c], alpha=0.72, label=CATEGORY_LABELS[c])
        for c in all_categories
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=10)
    fig.suptitle(
        "Heuristic signal comparison across all dataset splits\n"
        "TV / sports overlays remain stronger than AI exports even after compression and cropping",
        fontsize=14,
    )
    fig.tight_layout()
    out = OUT_DIR / "signal_separation_all_splits.png"
    fig.savefig(out, dpi=300)
    print(f"Saved {out}")
    plt.close(fig)

    # Additional robustness-only view for the three AI-positive splits.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    boxplot_for_categories(
        axes[0],
        rows,
        "of_count",
        ai_categories,
        ylabel="Number of OF contours",
        title="OF robustness across AI-only splits: of_count",
    )
    boxplot_for_categories(
        axes[1],
        rows,
        "of_max_area",
        ai_categories,
        ylabel="OF max area [px²]",
        title="OF robustness across AI-only splits: of_max_area",
    )
    ai_handles = [
        mpatches.Patch(color=CATEGORY_COLORS[c], alpha=0.72, label=CATEGORY_LABELS[c])
        for c in ai_categories
    ]
    fig.legend(handles=ai_handles, loc="upper right", fontsize=10)
    fig.suptitle(
        "Optical Flow robustness for AI videos\n"
        "Compression and cropping do not materially collapse the OF signal distribution",
        fontsize=14,
    )
    fig.tight_layout()
    out = OUT_DIR / "of_robustness_ai_splits.png"
    fig.savefig(out, dpi=300)
    print(f"Saved {out}")
    plt.close(fig)


def main():
    print("Loading raw_signals.csv...")
    rows = load_rows()
    print(f"  loaded {len(rows)} rows")

    print("Generating strict-IW confusion matrix...")
    plot_confusion_matrix(rows)

    print("Generating strict-IW metrics bar chart...")
    plot_metrics_bar(rows)

    print("Generating all-split signal separation figures...")
    plot_signal_separation_all(rows)

    print("Done.")


if __name__ == "__main__":
    main()

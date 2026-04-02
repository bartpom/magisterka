#!/usr/bin/env python3
"""
plot_thesis.py

Kompletny skrypt generowania wykresow do magisterki.
Wczytuje kod/results/latest/metrics_summary.csv oraz raw_signals.csv.

Generowane pliki (w kod/results/figures/):
  01_metrics_per_category.png  — Recall / Precision / F1 / FPR per split
  02_confusion_heatmap.png     — macierz konfuzji (TP/TN/FP/FN)
  03_signal_boxplots.png       — rozklad sygnalu OF/ZV/HF per kategoria
  04_detector_contribution.png — ile filmow wykryl kazdy detektor (stacked bar)
  05_roc_curve.png             — krzywa ROC na podstawie fusion_score
  06_score_distribution.png    — rozklad fusion_score AI vs Real

Uzycie:
  python kod/dataset/plot_thesis.py

Wymagane pliki:
  kod/results/latest/metrics_summary.csv
  kod/results/latest/raw_signals.csv
  kod/results/latest/evaluation_results.csv
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# --- sciezki ---------------------------------------------------------------
DATASET_ROOT = Path(__file__).parent
RESULTS_DIR  = DATASET_ROOT.parent / "results" / "latest"
OUT_DIR      = DATASET_ROOT.parent / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = RESULTS_DIR / "metrics_summary.csv"
RAW_CSV     = RESULTS_DIR / "raw_signals.csv"
EVAL_CSV    = RESULTS_DIR / "evaluation_results.csv"

# --- styl ------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

CATEGORY_ORDER = ["ai_baseline", "adv_compressed", "adv_cropped", "adv_fp_trap"]
CATEGORY_LABELS = {
    "ai_baseline":    "AI baseline",
    "adv_compressed": "AI compressed",
    "adv_cropped":    "AI cropped",
    "adv_fp_trap":    "Real TV / FP trap",
}
CATEGORY_COLORS = {
    "ai_baseline":    "#2c7bb6",
    "adv_compressed": "#1a9641",
    "adv_cropped":    "#e08214",
    "adv_fp_trap":    "#d7191c",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(x) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0


def _i(x) -> int:
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return 0


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[ERROR] Brak pliku: {path}")
        sys.exit(1)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save(fig: plt.Figure, name: str) -> None:
    out = OUT_DIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Zapisano: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 01 — Metryki per kategoria
# ---------------------------------------------------------------------------

def plot_metrics_per_category(rows_metrics: list[dict]) -> None:
    cats   = [r["category"] for r in rows_metrics if r["category"] in CATEGORY_ORDER]
    cats   = sorted(cats, key=lambda c: CATEGORY_ORDER.index(c))
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]

    metric_keys    = ["recall", "precision", "f1", "FPR"]
    metric_labels  = ["Recall", "Precision", "F1", "FPR"]
    metric_colors  = ["#2c7bb6", "#1a9641", "#6a3d9a", "#d7191c"]

    data_map = {r["category"]: r for r in rows_metrics}
    x     = np.arange(len(cats))
    width = 0.19

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, metric_colors)):
        vals = [_f(data_map[c].get(key, 0)) for c in cats]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.015,
                    f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Warto\u015b\u0107")
    ax.set_title("Metryki klasyfikatora per kategoria zbioru testowego", fontsize=14, pad=10)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.axhline(0.80, color="gray", linestyle=":", linewidth=1.4, label="pr\xf3g recall = 0.80")
    ax.axhline(1/7, color="#d7191c", linestyle=":", linewidth=1.2, label="pr\xf3g FPR = 1/7")

    # adnotacja dla FP trap — metryka kluczowa to FPR, nie recall
    fp_idx = cats.index("adv_fp_trap") if "adv_fp_trap" in cats else None
    if fp_idx is not None:
        ax.annotate(
            "Metryka\nkluczowa: FPR",
            xy=(fp_idx, 0.12),
            xytext=(fp_idx - 0.55, 0.32),
            fontsize=9,
            color="#d7191c",
            arrowprops=dict(arrowstyle="->", color="#d7191c", lw=1.2),
        )

    fig.tight_layout()
    save(fig, "01_metrics_per_category.png")


# ---------------------------------------------------------------------------
# 02 — Macierz konfuzji
# ---------------------------------------------------------------------------

def plot_confusion_heatmap(rows_metrics: list[dict]) -> None:
    total_tp = sum(_i(r["TP"]) for r in rows_metrics if r["category"] != "adv_fp_trap")
    total_fn = sum(_i(r["FN"]) for r in rows_metrics if r["category"] != "adv_fp_trap")
    fp_row   = next((r for r in rows_metrics if r["category"] == "adv_fp_trap"), {})
    total_fp = _i(fp_row.get("FP", 0))
    total_tn = _i(fp_row.get("TN", 0))

    cm = np.array([[total_tn, total_fp],
                   [total_fn, total_tp]])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Przewidywane: Real", "Przewidywane: AI"], fontsize=11)
    ax.set_yticklabels(["Rzeczywiste: Real", "Rzeczywiste: AI"], fontsize=11)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.55 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=22, fontweight="bold", color=color)
    ax.set_title("Macierz konfuzji \u2014 system fuzji multi-sygna\u0142\xf3w", fontsize=13, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, "02_confusion_heatmap.png")


# ---------------------------------------------------------------------------
# 03 — Boxploty sygnałów
# ---------------------------------------------------------------------------

def plot_signal_boxplots(rows_raw: list[dict]) -> None:
    grouped = defaultdict(list)
    for r in rows_raw:
        grouped[r["category"]].append(r)

    cats = [c for c in CATEGORY_ORDER if c in grouped]
    signals = [
        ("of_count",          "Liczba kontur\xf3w OF",     "OF count"),
        ("freq_hf_ratio_mean", "\u015aredni udzia\u0142 HF", "HF ratio"),
        ("zv_max_score",       "Maks. wynik ZV",            "ZV max score"),
        ("ai_style_prob",      "Prawdopod. AI (CLIP)",      "CLIP AI prob"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
    for ax, (field, ylabel, title) in zip(axes, signals):
        data = [[_f(r[field]) for r in grouped[c] if field in r] for c in cats]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            widths=0.55,
            showfliers=True,
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
        )
        for patch, cat in zip(bp["boxes"], cats):
            patch.set_facecolor(CATEGORY_COLORS[cat])
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(cats) + 1))
        ax.set_xticklabels([CATEGORY_LABELS[c] for c in cats], rotation=14, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)

    legend_handles = [
        mpatches.Patch(color=CATEGORY_COLORS[c], alpha=0.75, label=CATEGORY_LABELS[c])
        for c in cats
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Rozk\u0142ady sygna\u0142\xf3w diagnostycznych per kategoria",
        fontsize=13, y=1.07,
    )
    fig.tight_layout()
    save(fig, "03_signal_boxplots.png")


# ---------------------------------------------------------------------------
# 04 — Wklad detektorow
# ---------------------------------------------------------------------------

def plot_detector_contribution(rows_raw: list[dict]) -> None:
    """
    Dla filmow AI (gt=1) liczy ile z nich zostalo wykrytych przez kazdy
    z detektorow niezaleznie (bez fuzji).
    """
    ai_rows = [r for r in rows_raw if _i(r.get("ground_truth", 0)) == 1]
    n = len(ai_rows)
    if n == 0:
        print("  [SKIP] Brak wierszy AI w raw_signals.csv")
        return

    detectors = {
        "Optical Flow":      lambda r: _i(r.get("of_count", 0)) >= 5,
        "Zero Variance":     lambda r: _i(r.get("zv_count", 0)) >= 1,
        "Invis. Watermark":  lambda r: str(r.get("iw_matched", "")).strip().lower() not in ("", "none", "null", "nieznany", "(brak)"),
        "Flux Watermark":    lambda r: _i(r.get("flux_combined", 0)) == 1,
        "CLIP AI Style":     lambda r: _f(r.get("ai_style_prob", 0)) >= 0.44,
        "HF Spectral":       lambda r: _f(r.get("freq_hf_ratio_mean", 1.0)) < 0.50,
        "Temp. Consistency": lambda r: _i(r.get("tc_detected", 0)) == 1,
        "C2PA":              lambda r: _i(r.get("c2pa_ai", 0)) == 1,
    }

    cats = [c for c in CATEGORY_ORDER if c != "adv_fp_trap"]
    cat_groups = defaultdict(list)
    for r in ai_rows:
        cat_groups[r["category"]].append(r)

    det_names = list(detectors.keys())
    x = np.arange(len(det_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6.5))
    for i, cat in enumerate(cats):
        grp = cat_groups.get(cat, [])
        if not grp:
            continue
        counts = [sum(1 for r in grp if fn(r)) / len(grp) for fn in detectors.values()]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, counts, width,
                      label=CATEGORY_LABELS[cat],
                      color=CATEGORY_COLORS[cat], alpha=0.82)
        for bar in bars:
            h = bar.get_height()
            if h > 0.04:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                        f"{h:.0%}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(det_names, rotation=18, ha="right", fontsize=10)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Odsetek wykrytych film\xf3w AI")
    ax.set_title("Skuteczno\u015b\u0107 poszczeg\xf3lnych detektor\xf3w (niezale\u017cnie od fuzji)",
                 fontsize=13, pad=10)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    save(fig, "04_detector_contribution.png")


# ---------------------------------------------------------------------------
# 05 — Krzywa ROC
# ---------------------------------------------------------------------------

def plot_roc_curve(rows_eval: list[dict]) -> None:
    scores = [_f(r.get("fusion_score", 0)) for r in rows_eval]
    labels = [_i(r.get("ground_truth", 0)) for r in rows_eval]

    if not any(labels) or all(labels):
        print("  [SKIP] ROC wymaga obu klas w evaluation_results.csv")
        return

    thresholds = sorted(set(scores), reverse=True)
    tprs, fprs = [], []
    pos = sum(labels)
    neg = len(labels) - pos

    for thr in thresholds:
        pred = [1 if s >= thr else 0 for s in scores]
        tp = sum(p == 1 and g == 1 for p, g in zip(pred, labels))
        fp = sum(p == 1 and g == 0 for p, g in zip(pred, labels))
        tprs.append(tp / pos if pos else 0)
        fprs.append(fp / neg if neg else 0)

    tprs = [0.0] + tprs + [1.0]
    fprs = [0.0] + fprs + [1.0]
    auc = float(np.trapz(tprs, fprs))

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot(fprs, tprs, color="#2c7bb6", linewidth=2.5, label=f"System (AUC\u00a0=\u00a0{auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Losowy klasyfikator")
    # zaznaczamy punkt operacyjny (FPR=0.119, TPR=recall)
    op_fpr = 0.119
    op_tpr_candidates = [(abs(f - op_fpr), t) for f, t in zip(fprs, tprs)]
    op_tpr = min(op_tpr_candidates)[1]
    ax.scatter([op_fpr], [op_tpr], color="#d7191c", zorder=5, s=60,
               label=f"Punkt operacyjny (FPR={op_fpr:.2f}, TPR={op_tpr:.2f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("Krzywa ROC \u2014 system fuzji multi-sygna\u0142\xf3w", fontsize=13, pad=10)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    save(fig, "05_roc_curve.png")


# ---------------------------------------------------------------------------
# 06 — Rozklad fusion_score
# ---------------------------------------------------------------------------

def plot_score_distribution(rows_eval: list[dict]) -> None:
    ai_scores   = [_f(r["fusion_score"]) for r in rows_eval if _i(r.get("ground_truth", 0)) == 1]
    real_scores = [_f(r["fusion_score"]) for r in rows_eval if _i(r.get("ground_truth", 0)) == 0]

    if not ai_scores or not real_scores:
        print("  [SKIP] Brak obu klas w evaluation_results.csv")
        return

    bins = np.arange(
        min(min(ai_scores), min(real_scores)) - 0.5,
        max(max(ai_scores), max(real_scores)) + 1.5,
        1.0,
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(ai_scores,   bins=bins, alpha=0.68, color="#2c7bb6", label="AI (gt=1)",   edgecolor="white")
    ax.hist(real_scores, bins=bins, alpha=0.68, color="#d7191c", label="Real (gt=0)", edgecolor="white")
    ax.axvline(5, color="black", linestyle="--", linewidth=1.8, label="Pr\xf3g decyzji = 5")
    ax.set_xlabel("Wynik fuzji (fusion_score)")
    ax.set_ylabel("Liczba film\xf3w")
    ax.set_title("Rozk\u0142ad wynik\xf3w fuzji \u2014 AI vs. filmy rzeczywiste", fontsize=13, pad=10)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    save(fig, "06_score_distribution.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== plot_thesis.py ===")
    print(f"Wczytywanie danych z: {RESULTS_DIR}")

    rows_metrics = load_csv(METRICS_CSV)
    rows_raw     = load_csv(RAW_CSV)
    rows_eval    = load_csv(EVAL_CSV)

    print(f"  metrics_summary: {len(rows_metrics)} wierszy")
    print(f"  raw_signals:     {len(rows_raw)} wierszy")
    print(f"  eval_results:    {len(rows_eval)} wierszy")
    print()

    print("[1/6] Metryki per kategoria...")
    plot_metrics_per_category(rows_metrics)

    print("[2/6] Macierz konfuzji...")
    plot_confusion_heatmap(rows_metrics)

    print("[3/6] Boxploty sygnalow...")
    plot_signal_boxplots(rows_raw)

    print("[4/6] Wklad detektorow...")
    plot_detector_contribution(rows_raw)

    print("[5/6] Krzywa ROC...")
    plot_roc_curve(rows_eval)

    print("[6/6] Rozklad fusion_score...")
    plot_score_distribution(rows_eval)

    print(f"\nGotowe. Wykresy zapisano w: {OUT_DIR}")


if __name__ == "__main__":
    main()

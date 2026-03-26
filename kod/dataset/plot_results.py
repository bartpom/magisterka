#!/usr/bin/env python3
"""
plot_results.py  —  wykresy do pracy magisterskiej

Uzywa raw_signals.csv (nie evaluation_results.csv).
Decyzja AI oparta na strict IW: iw_matched potwierdzony I similarity >= 0.85.
Generuje:
  1. confusion_matrix.png       — macierz pomylek (strict IW)
  2. metrics_bar_chart.png      — precision/recall/f1 per kategoria (strict IW vs always-neg baseline)
  3. signal_separation.png      — boxplot OF/ZV: ai_baseline vs adv_fp_trap
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

# ── sciezki ────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(__file__).parent
RESULTS_DIR  = DATASET_ROOT.parent / "results"
RAW_CSV      = RESULTS_DIR / "latest" / "raw_signals.csv"
OUT_DIR      = RESULTS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── stałe ──────────────────────────────────────────────────────────────────
_IW_EMPTY = {"", "nieznany", "none", "(brak)", "null"}
CATEGORY_ORDER = ["ai_baseline", "adv_compressed", "adv_cropped", "adv_fp_trap"]
CATEGORY_LABELS = {
    "ai_baseline":    "AI baseline",
    "adv_compressed": "Adv compressed",
    "adv_cropped":    "Adv cropped",
    "adv_fp_trap":    "FP trap (TV)",
}
plt.rcParams.update({"font.size": 12, "font.family": "serif"})

# ── pomocnicze ─────────────────────────────────────────────────────────────
def _float(x): return float(x) if x not in ("", None) else 0.0
def _int(x):   return int(float(x)) if x not in ("", None) else 0

def is_strict_iw(row):
    matched = str(row.get("iw_matched", "")).strip().lower()
    return matched not in _IW_EMPTY and _float(row.get("iw_best_similarity", 0)) >= 0.85

def load_rows():
    if not RAW_CSV.exists():
        print(f"[BLAD] Brak {RAW_CSV}. Uruchom najpierw evaluate.py")
        raise SystemExit(1)
    with RAW_CSV.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def compute_metrics(rows, pred_fn):
    stats = {}
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for cat, rr in by_cat.items():
        gt   = [_int(r["ground_truth"]) for r in rr]
        pred = [int(bool(pred_fn(r)))   for r in rr]
        tp = sum(p==1 and g==1 for p,g in zip(pred,gt))
        tn = sum(p==0 and g==0 for p,g in zip(pred,gt))
        fp = sum(p==1 and g==0 for p,g in zip(pred,gt))
        fn = sum(p==0 and g==1 for p,g in zip(pred,gt))
        prec = tp/(tp+fp) if (tp+fp) else 0.0
        rec  = tp/(tp+fn) if (tp+fn) else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        fpr  = fp/(fp+tn) if (fp+tn) else 0.0
        stats[cat] = dict(TP=tp,TN=tn,FP=fp,FN=fn,
                          precision=prec,recall=rec,f1=f1,fpr=fpr,n=len(gt))
    return stats

# ── wykres 1: confusion matrix ─────────────────────────────────────────────
def plot_confusion_matrix(rows):
    gt   = [_int(r["ground_truth"]) for r in rows if r["category"] != "adv_fp_trap" or True]
    pred = [int(bool(is_strict_iw(r))) for r in rows]

    tp = sum(p==1 and g==1 for p,g in zip(pred,gt))
    tn = sum(p==0 and g==0 for p,g in zip(pred,gt))
    fp = sum(p==1 and g==0 for p,g in zip(pred,gt))
    fn = sum(p==0 and g==1 for p,g in zip(pred,gt))
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Predicted Real", "Predicted AI"])
    ax.set_yticklabels(["Actual Real", "Actual AI"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=16)
    ax.set_title("Confusion Matrix: Strict IW Detection\n(iw_matched confirmed, similarity ≥ 0.85)",
                 fontsize=13, pad=12)
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    fig.tight_layout()
    out = OUT_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=300)
    print(f"Zapisano {out}")
    plt.close(fig)

# ── wykres 2: metryki per kategoria ───────────────────────────────────────
def plot_metrics_bar(rows):
    stats = compute_metrics(rows, is_strict_iw)
    cats  = [c for c in CATEGORY_ORDER if c in stats]
    labels = [CATEGORY_LABELS[c] for c in cats]

    metrics = ["precision", "recall", "f1"]
    colors  = ["#2c7bb6", "#1a9641", "#d7191c"]
    x = np.arange(len(cats))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (m, col) in enumerate(zip(metrics, colors)):
        vals = [stats[c][m] for c in cats]
        bars = ax.bar(x + i*width - width, vals, width, label=m.capitalize(), color=col, alpha=0.85)
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(b.get_x()+b.get_width()/2, h),
                        xytext=(0,4), textcoords="offset points",
                        ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_xlabel("Category")
    ax.set_title("Performance Metrics — Strict IW Definition\n(TP=0 everywhere: no confirmed watermark detected)",
                 fontsize=13)
    ax.legend(title="Metric")
    ax.axhline(0, color="black", linewidth=0.8)

    # adnotacja FPR dla fp_trap
    if "adv_fp_trap" in stats:
        fpr = stats["adv_fp_trap"]["fpr"]
        ax.annotate(f"FPR={fpr:.0%}",
                    xy=(cats.index("adv_fp_trap"), 0.05),
                    ha="center", color="gray", fontsize=10)

    fig.tight_layout()
    out = OUT_DIR / "metrics_bar_chart.png"
    fig.savefig(out, dpi=300)
    print(f"Zapisano {out}")
    plt.close(fig)

# ── wykres 3: separacja OF/ZV ──────────────────────────────────────────────
def plot_signal_separation(rows):
    cats_of_interest = ["ai_baseline", "adv_fp_trap"]
    signals = {
        "of_max_area":  "OF max area [px²]",
        "zv_max_score": "ZV max score",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = {"ai_baseline": "#2c7bb6", "adv_fp_trap": "#d7191c"}
    labels_map = {"ai_baseline": "AI baseline", "adv_fp_trap": "FP trap (TV)"}

    for ax, (field, ylabel) in zip(axes, signals.items()):
        data_by_cat = {}
        for cat in cats_of_interest:
            vals = [_float(r[field]) for r in rows if r["category"] == cat]
            data_by_cat[cat] = vals

        bp = ax.boxplot(
            [data_by_cat[c] for c in cats_of_interest],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            widths=0.5,
        )
        for patch, cat in zip(bp["boxes"], cats_of_interest):
            patch.set_facecolor(colors[cat])
            patch.set_alpha(0.7)

        ax.set_xticks([1, 2])
        ax.set_xticklabels([labels_map[c] for c in cats_of_interest])
        ax.set_ylabel(ylabel)
        ax.set_title(f"Separacja sygnału: {field}")
        ax.grid(axis="y", alpha=0.4)

    patches = [mpatches.Patch(color=colors[c], alpha=0.7, label=labels_map[c])
               for c in cats_of_interest]
    fig.legend(handles=patches, loc="upper right", fontsize=10)
    fig.suptitle("Heurystyki OF/ZV: AI baseline vs FP trap (TV/Sport)\n"
                 "Wyższe wartości u TV potwierdzają nakładki graficzne",
                 fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / "signal_separation.png"
    fig.savefig(out, dpi=300)
    print(f"Zapisano {out}")
    plt.close(fig)

# ── main ───────────────────────────────────────────────────────────────────
def main():
    print("Ladowanie danych z raw_signals.csv...")
    rows = load_rows()
    print(f"  {len(rows)} wierszy")

    print("Generowanie Confusion Matrix (strict IW)...")
    plot_confusion_matrix(rows)

    print("Generowanie wykresu metryk (strict IW)...")
    plot_metrics_bar(rows)

    print("Generowanie wykresu separacji sygnalow OF/ZV...")
    plot_signal_separation(rows)

    print("Generowanie zakonczone pomyslnie.")

if __name__ == "__main__":
    main()

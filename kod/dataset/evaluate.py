#!/usr/bin/env python3
"""
evaluate.py
Benchmark detektora AI-wideo.

Zapisuje trzy pliki wynikowe:
  raw_signals.csv        — surowe wartości każdego detektora (do strojenia progów)
  evaluation_results.csv — decyzja finalna per wideo
  metrics_summary.csv    — TP/TN/FP/FN + Accuracy/F1/FPR per kategoria
  threshold_sweep.csv    — sweep progów invisible_wm i optical_flow

Kategorie:
  ai_baseline    (filmy AI z watermarkiem)        → gt=1
  adv_compressed (filmy AI skompresowane)          → gt=1
  adv_cropped    (filmy AI przycięte)              → gt=1
  adv_fp_trap    (filmy ludzkie / pułapki FP)      → gt=0
"""

from __future__ import annotations
import csv
import sys
import time
from pathlib import Path
from typing import Any
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from advanced_detectors import run_advanced_scan

DATASET_ROOT = Path(__file__).parent
RESULTS_DIR  = DATASET_ROOT.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV      = RESULTS_DIR / "raw_signals.csv"
EVAL_CSV     = RESULTS_DIR / "evaluation_results.csv"
METRICS_CSV  = RESULTS_DIR / "metrics_summary.csv"
SWEEP_CSV    = RESULTS_DIR / "threshold_sweep.csv"

CATEGORIES = {
    "ai_baseline":    (DATASET_ROOT / "ai_baseline",    1),
    "adv_compressed": (DATASET_ROOT / "adv_compressed", 1),
    "adv_cropped":    (DATASET_ROOT / "adv_cropped",    1),
    "adv_fp_trap":    (DATASET_ROOT / "adv_fp_trap",    0),
}

# ————————————————————————————————————————————————————————————————————————
# Definicja kolumn CSV
# ————————————————————————————————————————————————————————————————————————

RAW_FIELDS = [
    "category", "filename", "ground_truth",
    # Zero-variance
    "zv_count",           # liczba ROI z zerową wariancją
    "zv_max_score",       # max score (frakcja pikseli < thr) wśród ROI
    # Optical Flow
    "of_count",           # liczba konturów statycznych
    "of_max_area",        # pole największego konturu
    "of_global_motion",   # średni globalny ruch kamery (px)
    # Invisible Watermark
    "iw_found",           # 1/0 — czy cokolwiek dopasowano
    "iw_best_similarity", # best_similarity do znanych sygnatur [0.0–1.0]
    "iw_matched",         # nazwa dopasowanej sygnatury lub pusty string
    "iw_method",          # metoda: dwtDct / dwtDctSvd / rivaGan
    # FFT
    "fft_found",          # 1/0
    "fft_score",          # score z FFT [0.0–1.0]
    # Czas
    "frames_sampled",
    "duration_s",
    "detector_version",
]

EVAL_FIELDS = [
    "category", "filename", "ground_truth",
    "detected", "fusion_score", "fusion_mode",
    "zv_count", "of_count", "iw_best_similarity", "iw_matched",
    "fft_score", "duration_s",
]

DETECTOR_VERSION = "adv_v2_sweep"


# ————————————————————————————————————————————————————————————————————————
# Fuzja sygnałów — domyślne progi (kalibrowane przez threshold_sweep)
# ————————————————————————————————————————————————————————————————————————

def fuse(
    zv_count: int,
    of_count: int,
    iw_similarity: float,
    iw_matched: str,
    fft_score: float,
    c2pa_found: bool = False,
    # Progi sygnałów mocnych
    iw_strong_threshold: float = 0.85,
    # Progi sygnałów słabych (heurystycznych)
    of_threshold: int = 6,
    iw_weak_threshold: float = 0.65,
    zv_threshold: int = 2,
    fft_threshold: float = 0.30,
    # Próg score ważonego + minimalna liczba głosów
    score_threshold: float = 0.62,
    min_weak_votes: int = 2,
) -> tuple[int, float, str]:
    """
    Fuzja sygnałów — zwraca (detected: 0/1, fusion_score: float, mode: str).

    Logika dwustopniowa:
    1. Sygnały wysokiej precyzji: C2PA lub silny invisible WM -> natychmiastowa detekcja.
    2. Sygnały heurystyczne: wymagana zgodność ≥ min_weak_votes i score ≥ score_threshold.
    """
    # --- STOPIEŃ 1: sygnały wysokiej precyzji ---
    if c2pa_found:
        return 1, 1.0, "c2pa"
    if iw_matched and iw_similarity >= iw_strong_threshold:
        return 1, iw_similarity, f"iw_strong:{iw_matched}"

    # --- STOPIEŃ 2: score ważony + głosowanie ---
    iw_norm = min(1.0, iw_similarity)  # już w [0,1]
    of_norm = min(1.0, of_count / max(of_threshold, 1))
    zv_norm = min(1.0, zv_count / max(zv_threshold, 1))

    score = (
        0.45 * iw_norm +
        0.25 * of_norm +
        0.20 * zv_norm +
        0.10 * fft_score
    )

    votes = sum([
        iw_similarity >= iw_weak_threshold,
        of_count >= of_threshold,
        zv_count >= zv_threshold,
        fft_score >= fft_threshold,
    ])

    if score >= score_threshold and votes >= min_weak_votes:
        return 1, round(score, 4), f"weighted:votes={votes}"
    return 0, round(score, 4), f"below_thr:votes={votes}"


# ————————————————————————————————————————————————————————————————————————
# Skanowanie wideo
# ————————————————————————————————————————————————————————————————————————

def scan_video(video_path: Path) -> tuple[dict[str, Any], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć: {video_path}")
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t0 = time.time()
    result = run_advanced_scan(
        cap, fps, total_frames,
        n_frames_median=30,
        check_invisible=True,
        check_fft=True,
        check_optical_flow=True,
        of_scale=0.5,
    )
    elapsed = time.time() - t0
    cap.release()
    return result, elapsed


def extract_signals(result: dict[str, Any]) -> dict[str, Any]:
    """Wyciąga i normalizuje surowe sygnały z wyniku run_advanced_scan."""
    zv_rois = result.get("zero_variance_rois", [])
    of_rois = result.get("optical_flow_rois", [])
    iw_data = result.get("invisible_wm", {})
    fft_data = result.get("fft_artifacts", {})

    zv_count    = len(zv_rois)
    zv_max_score = max((r.get("score", 0.0) for r in zv_rois), default=0.0)

    of_count    = len(of_rois)
    of_max_area = max((r.get("area", 0) for r in of_rois), default=0)
    of_global   = of_rois[0].get("global_motion", 0.0) if of_rois else 0.0

    # Kluczowa zmiana: best_similarity zamiast stałego 0.5
    iw_similarity = float(iw_data.get("score", 0.0))
    iw_matched    = iw_data.get("matched") or ""
    iw_found      = 1 if iw_data.get("found", False) else 0
    iw_method     = iw_data.get("method", "")

    fft_found = 1 if fft_data.get("found", False) else 0
    fft_score = float(fft_data.get("score", 0.0))

    return {
        "zv_count":          zv_count,
        "zv_max_score":      round(zv_max_score, 4),
        "of_count":          of_count,
        "of_max_area":       of_max_area,
        "of_global_motion":  round(of_global, 4),
        "iw_found":          iw_found,
        "iw_best_similarity": round(iw_similarity, 4),
        "iw_matched":        iw_matched,
        "iw_method":         iw_method,
        "fft_found":         fft_found,
        "fft_score":         round(fft_score, 4),
    }


# ————————————————————————————————————————————————————————————————————————
# Metryki
# ————————————————————————————————————————————————————————————————————————

def compute_metrics(rows: list[dict], pred_field: str = "detected") -> list[dict]:
    cats: dict[str, list] = {}
    for row in rows:
        cats.setdefault(row["category"], []).append(row)
    metric_rows = []
    for cat, cat_rows in cats.items():
        gt   = [int(r["ground_truth"]) for r in cat_rows]
        pred = [int(r[pred_field])     for r in cat_rows]
        tp = sum(p == 1 and g == 1 for p, g in zip(pred, gt))
        tn = sum(p == 0 and g == 0 for p, g in zip(pred, gt))
        fp = sum(p == 1 and g == 0 for p, g in zip(pred, gt))
        fn = sum(p == 0 and g == 1 for p, g in zip(pred, gt))
        n  = len(gt)
        acc  = (tp + tn) / n if n else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec  = tp / (tp + fn) if (tp + fn) else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        fpr  = fp / (fp + tn) if (fp + tn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        metric_rows.append({
            "category": cat, "n": n,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "accuracy":    f"{acc:.4f}",
            "precision":   f"{prec:.4f}",
            "recall":      f"{rec:.4f}",
            "f1":          f"{f1:.4f}",
            "FPR":         f"{fpr:.4f}",
            "specificity": f"{spec:.4f}",
        })
    return metric_rows


# ————————————————————————————————————————————————————————————————————————
# Threshold sweep
# ————————————————————————————————————————————————————————————————————————

def run_threshold_sweep(raw_rows: list[dict]) -> list[dict]:
    """
    Sweep progów iw_similarity i of_count bez ponownego przetwarzania wideo.
    Dla każdej kombinacji (iw_thr, of_thr, score_thr) oblicza globalne metryki.
    """
    import itertools

    iw_thresholds  = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    of_thresholds  = [3, 5, 8, 10, 15]
    score_thresholds = [0.55, 0.60, 0.62, 0.65, 0.70]

    sweep_rows = []
    for iw_thr, of_thr, sc_thr in itertools.product(
        iw_thresholds, of_thresholds, score_thresholds
    ):
        preds, gts = [], []
        for r in raw_rows:
            gt_val = int(r["ground_truth"])
            det, _, _ = fuse(
                zv_count=int(r["zv_count"]),
                of_count=int(r["of_count"]),
                iw_similarity=float(r["iw_best_similarity"]),
                iw_matched=r["iw_matched"],
                fft_score=float(r["fft_score"]),
                iw_strong_threshold=0.85,
                of_threshold=of_thr,
                iw_weak_threshold=iw_thr,
                score_threshold=sc_thr,
                min_weak_votes=2,
            )
            preds.append(det)
            gts.append(gt_val)

        tp = sum(p == 1 and g == 1 for p, g in zip(preds, gts))
        tn = sum(p == 0 and g == 0 for p, g in zip(preds, gts))
        fp = sum(p == 1 and g == 0 for p, g in zip(preds, gts))
        fn = sum(p == 0 and g == 1 for p, g in zip(preds, gts))
        n  = len(gts)
        acc  = (tp + tn) / n if n else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec  = tp / (tp + fn) if (tp + fn) else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        fpr  = fp / (fp + tn) if (fp + tn) else 0

        sweep_rows.append({
            "iw_weak_thr":    iw_thr,
            "of_thr":         of_thr,
            "score_thr":      sc_thr,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "accuracy":  f"{acc:.4f}",
            "precision": f"{prec:.4f}",
            "recall":    f"{rec:.4f}",
            "f1":        f"{f1:.4f}",
            "FPR":       f"{fpr:.4f}",
        })

    # Sortuj po FPR ASC, potem F1 DESC
    sweep_rows.sort(key=lambda x: (float(x["FPR"]), -float(x["f1"])))
    return sweep_rows


# ————————————————————————————————————————————————————————————————————————
# Main
# ————————————————————————————————————————————————————————————————————————

def main() -> None:
    # Czyść stare wyniki
    for f in [RAW_CSV, EVAL_CSV, METRICS_CSV, SWEEP_CSV]:
        if f.exists():
            print(f"[INFO] Usuwam: {f}")
            f.unlink()

    raw_rows:  list[dict] = []
    eval_rows: list[dict] = []

    for category, (folder, gt) in CATEGORIES.items():
        videos = sorted(folder.glob("*.mp4"))
        if not videos:
            print(f"[WARN] Brak .mp4 w {folder}", file=sys.stderr)
            continue
        print(f"\n=== {category} ({len(videos)} filmów) ===")

        for vp in videos:
            print(f"  [SCAN] {vp.name} ... ", end="", flush=True)
            try:
                result, elapsed = scan_video(vp)
                sig = extract_signals(result)
                frames_sampled = len(result.get("zero_variance_rois", []).__class__.__mro__) or 30

                # Surowy wiersz
                raw_row = {
                    "category":      category,
                    "filename":      vp.name,
                    "ground_truth":  gt,
                    **sig,
                    "frames_sampled":  30,
                    "duration_s":      f"{elapsed:.2f}",
                    "detector_version": DETECTOR_VERSION,
                }
                raw_rows.append(raw_row)

                # Decyzja fuzji z domyślnymi progami
                det, score, mode = fuse(
                    zv_count=sig["zv_count"],
                    of_count=sig["of_count"],
                    iw_similarity=sig["iw_best_similarity"],
                    iw_matched=sig["iw_matched"],
                    fft_score=sig["fft_score"],
                )

                eval_row = {
                    "category":          category,
                    "filename":          vp.name,
                    "ground_truth":      gt,
                    "detected":          det,
                    "fusion_score":      score,
                    "fusion_mode":       mode,
                    "zv_count":          sig["zv_count"],
                    "of_count":          sig["of_count"],
                    "iw_best_similarity": sig["iw_best_similarity"],
                    "iw_matched":        sig["iw_matched"],
                    "fft_score":         sig["fft_score"],
                    "duration_s":        f"{elapsed:.2f}",
                }
                eval_rows.append(eval_row)

                det_str = "WYKRYTO" if det else "brak"
                print(f"{det_str}  score={score:.3f}  ({elapsed:.1f}s)")

            except Exception as e:
                print(f"BŁĄD: {e}", file=sys.stderr)

    # --- Zapis raw_signals.csv ---
    if raw_rows:
        with RAW_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
            writer.writeheader()
            writer.writerows(raw_rows)
        print(f"\n[RAW]     {RAW_CSV}")

    # --- Zapis evaluation_results.csv ---
    if eval_rows:
        with EVAL_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
            writer.writeheader()
            writer.writerows(eval_rows)
        print(f"[EVAL]    {EVAL_CSV}")

    # --- Metryki ---
    if eval_rows:
        metrics = compute_metrics(eval_rows)
        with METRICS_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)
        print(f"[METRICS] {METRICS_CSV}")

        print("\n--- Podsumowanie metryk (domyślne progi) ---")
        for m in metrics:
            print(
                f"  {m['category']:18s}: n={m['n']:3d} "
                f"acc={m['accuracy']}  f1={m['f1']}  "
                f"FPR={m['FPR']}  spec={m['specificity']}  "
                f"TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}"
            )

    # --- Threshold sweep (na surowych sygnałach, bez ponownego skanowania) ---
    if raw_rows:
        print("\n[SWEEP] Obliczam threshold sweep...")
        sweep = run_threshold_sweep(raw_rows)
        with SWEEP_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
            writer.writeheader()
            writer.writerows(sweep)
        print(f"[SWEEP]   {SWEEP_CSV}")

        # Pokaż top-5 konfiguracji z najniższym FPR
        print("\n--- Top-5 konfiguracji (najniższy FPR) ---")
        for row in sweep[:5]:
            print(
                f"  iw_thr={row['iw_weak_thr']}  of_thr={row['of_thr']}  "
                f"sc_thr={row['score_thr']}  "
                f"FPR={row['FPR']}  F1={row['f1']}  acc={row['accuracy']}"
            )


if __name__ == "__main__":
    main()

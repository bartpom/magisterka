#!/usr/bin/env python3
"""
evaluate.py
Benchmark detektora AI-wideo.

Zapisuje wyniki do kod/results/<YYYY-MM-DD_<hash>/ (snapshot) oraz do
kod/results/latest/ (symlink / kopia dla szybkiego dostępu).

Pliki wynikowe:
  raw_signals.csv        — surowe wartości każdego detektora
  evaluation_results.csv — decyzja finalna per wideo
  metrics_summary.csv    — TP/TN/FP/FN + Accuracy/F1/FPR/specificity per kategoria
  threshold_sweep.csv    — sweep 175+ kombinacji progów
  run_info.txt           — hash commita, data, liczba filmów

Kategorie:
  ai_baseline    (filmy AI z watermarkiem)        → gt=1
  adv_compressed (filmy AI skompresowane)          → gt=1
  adv_cropped    (filmy AI przycięte)              → gt=1
  adv_fp_trap    (filmy ludzkie / pułapki FP)      → gt=0
"""

from __future__ import annotations
import csv
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from statistics import mean
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from advanced_detectors import run_advanced_scan
from fusion_params import (
    HF_RATIO_THRESHOLD,
    LOWER_THIRD_HARD_THRESHOLD,
    LOW_TEXTURE_THRESHOLD,
    MAX_AREA_RATIO_THRESHOLD,
    POINTS_THRESHOLD_DEFAULT,
    POINTS_THRESHOLD_SWEEP,
)

DATASET_ROOT = Path(__file__).parent
RESULTS_BASE = DATASET_ROOT.parent / "results"
RESULTS_BASE.mkdir(parents=True, exist_ok=True)

CATEGORIES = {
    "ai_baseline":    (DATASET_ROOT / "ai_baseline",    1),
    "adv_compressed": (DATASET_ROOT / "adv_compressed", 1),
    "adv_cropped":    (DATASET_ROOT / "adv_cropped",    1),
    "adv_fp_trap":    (DATASET_ROOT / "adv_fp_trap",    0),
}

RAW_FIELDS = [
    "category", "filename", "ground_truth",
    "zv_count", "zv_max_score", "zv_lower_third_roi_count",
    "of_count", "of_max_area", "of_max_area_ratio", "of_global_motion",
    "of_texture_variance_mean", "of_low_texture_roi_count",
    "of_wide_lower_roi_count", "of_corner_compact_roi_count", "of_lower_third_roi_ratio",
    "iw_found", "iw_best_similarity", "iw_matched", "iw_method",
    "fft_found", "fft_score", "freq_hf_ratio_mean",
    "frames_sampled", "duration_s", "detector_version",
]

EVAL_FIELDS = [
    "category", "filename", "ground_truth",
    "detected", "fusion_score", "fusion_mode", "ai_specific", "broadcast_trap",
    "zv_count", "zv_lower_third_roi_count", "of_count", "of_max_area_ratio", "iw_best_similarity", "iw_matched",
    "fft_score", "of_texture_variance_mean", "of_low_texture_roi_count",
    "of_wide_lower_roi_count", "of_corner_compact_roi_count", "of_lower_third_roi_ratio",
    "freq_hf_ratio_mean", "duration_s",
]

DETECTOR_VERSION = "adv_v5_geom_penalty"


# ───────────────────────────────────────────────────────────────────────
# Snapshot katalogu wynikow
# ───────────────────────────────────────────────────────────────────────

def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(DATASET_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def make_snapshot_dir() -> Path:
    """Tworzy unikalny katalog wynikow results/YYYY-MM-DD_<hash>/."""
    date_str  = datetime.now().strftime("%Y-%m-%d")
    git_hash  = get_git_hash()
    snap_name = f"{date_str}_{git_hash}"
    snap_dir  = RESULTS_BASE / snap_name
    snap_dir.mkdir(parents=True, exist_ok=True)
    return snap_dir


def copy_to_latest(snap_dir: Path) -> None:
    """Kopiuje snapshot do results/latest/ (nadpisuje)."""
    latest = RESULTS_BASE / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(snap_dir, latest)


# ───────────────────────────────────────────────────────────────────────
# Fuzja sygnalow
# ───────────────────────────────────────────────────────────────────────

def compute_ai_score(
    row: dict[str, Any],
    low_texture_threshold: int = LOW_TEXTURE_THRESHOLD,
    hf_ratio_threshold: float = HF_RATIO_THRESHOLD,
    max_area_ratio_threshold: float = MAX_AREA_RATIO_THRESHOLD,
    lower_third_hard_threshold: float = LOWER_THIRD_HARD_THRESHOLD,
) -> int:
    """
    Multi-signal AI score (0..6) wg założeń zadania.
    """
    score = 0

    of_count = int(row.get("of_count", 0))
    area_ratio = float(row.get("of_max_area_ratio", 1.0))
    low_texture = int(row.get("of_low_texture_roi_count", 0)) >= low_texture_threshold
    low_hf = float(row.get("freq_hf_ratio_mean", 1.0)) < hf_ratio_threshold
    iw_strong = bool(row.get("iw_matched")) and float(row.get("iw_similarity", 0.0)) >= 0.85
    wide_lower = int(row.get("of_wide_lower_roi_count", 0))
    corner_compact = int(row.get("of_corner_compact_roi_count", 0))
    lower_third_ratio = float(row.get("of_lower_third_roi_ratio", 0.0))
    zv_lower_third = int(row.get("zv_lower_third_roi_count", 0))

    # Signal 1: OF obecny i niezdominowany przez gigantyczny overlay.
    if of_count >= 5:
        score += 1
    if area_ratio < max_area_ratio_threshold:
        score += 1
    else:
        score -= 1
    # Signal 2: niski texture variance w ROI OF (AI smoothness)
    if low_texture:
        score += 2
    # Signal 3: niski udział HF (AI smoothness w domenie częstotliwości)
    if low_hf:
        score += 2
    # Signal 4: regiony zero-variance (statyczne overlaye/logotypy)
    if int(row.get("zv_count", 0)) >= 1:
        score += 1
    if iw_strong:
        score += 2
    # Geometry-aware refinements:
    if wide_lower >= 1:
        score -= 2
    if corner_compact >= 1:
        score += 1
    # Hard broadcast-trap heuristic: dolny-dominujacy OF + statyczne ROI na dole.
    if lower_third_ratio > lower_third_hard_threshold and zv_lower_third > 0:
        score -= 3
    return score


def compute_ai_flags(row: dict[str, Any]) -> tuple[int, int]:
    ai_specific = int(
        row.get("of_low_texture_roi_count", 0) >= LOW_TEXTURE_THRESHOLD
        or row.get("freq_hf_ratio_mean", 1.0) < HF_RATIO_THRESHOLD
        or (bool(row.get("iw_matched")) and row.get("iw_similarity", 0.0) >= 0.85)
    )
    broadcast_trap = int(
        row.get("of_lower_third_roi_ratio", 0.0) > LOWER_THIRD_HARD_THRESHOLD
        and row.get("zv_lower_third_roi_count", 0) > 0
    )
    return ai_specific, broadcast_trap


def fuse(
    zv_count: int,
    zv_lower_third_roi_count: int,
    of_count: int,
    of_max_area: float,
    of_max_area_ratio: float,
    iw_similarity: float,
    iw_matched: str,
    fft_score: float,
    of_texture_variance_mean: float,
    of_low_texture_roi_count: int,
    of_wide_lower_roi_count: int,
    of_corner_compact_roi_count: int,
    of_lower_third_roi_ratio: float,
    freq_hf_ratio_mean: float,
    points_threshold: int = POINTS_THRESHOLD_DEFAULT,
) -> tuple[int, float, str, int, int]:
    row = {
        "zv_count": zv_count,
        "zv_lower_third_roi_count": zv_lower_third_roi_count,
        "of_count": of_count,
        "of_max_area": of_max_area,
        "of_max_area_ratio": of_max_area_ratio,
        "iw_similarity": iw_similarity,
        "iw_matched": iw_matched,
        "fft_score": fft_score,
        "of_texture_variance_mean": of_texture_variance_mean,
        "of_low_texture_roi_count": of_low_texture_roi_count,
        "of_wide_lower_roi_count": of_wide_lower_roi_count,
        "of_corner_compact_roi_count": of_corner_compact_roi_count,
        "of_lower_third_roi_ratio": of_lower_third_roi_ratio,
        "freq_hf_ratio_mean": freq_hf_ratio_mean,
    }
    score = compute_ai_score(row)
    ai_specific, broadcast_trap = compute_ai_flags(row)
    lower_third_ok = not broadcast_trap
    detected = int(score >= points_threshold and ai_specific and lower_third_ok)
    return detected, float(score), (
        f"ai_score={score};ai_specific={int(ai_specific)};lower_third_ok={int(lower_third_ok)}"
    ), int(ai_specific), int(broadcast_trap)


# ───────────────────────────────────────────────────────────────────────
# Skanowanie wideo
# ───────────────────────────────────────────────────────────────────────

def scan_video(video_path: Path) -> tuple[dict[str, Any], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nie mozna otworzyc: {video_path}")
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
    zv_rois  = result.get("zero_variance_rois", [])
    of_rois  = result.get("optical_flow_rois",  [])
    iw_data  = result.get("invisible_wm",       {})
    fft_data = result.get("fft_artifacts",      {})

    zv_count     = len(zv_rois)
    zv_max_score = max((r.get("score", 0.0) for r in zv_rois), default=0.0)
    zv_lower_third_roi_count = sum(
        1 for r in zv_rois
        if r.get("name") in {"CORNER-BL", "CORNER-BR"}
    )
    of_count     = len(of_rois)
    of_max_area  = max((r.get("area",  0)   for r in of_rois), default=0)
    of_max_area_ratio = max((float(r.get("area_ratio", 0.0)) for r in of_rois), default=0.0)
    of_global    = of_rois[0].get("global_motion", 0.0) if of_rois else 0.0
    of_texture_variance_mean = (
        float(mean(r.get("texture_variance", 0.0) for r in of_rois)) if of_rois else 0.0
    )
    of_low_texture_roi_count = sum(1 for r in of_rois if float(r.get("texture_variance", 0.0)) < 50.0)
    of_wide_lower_roi_count = sum(
        1 for r in of_rois
        if float(r.get("width_ratio", 0.0)) >= 0.45
        and float(r.get("height_ratio", 0.0)) <= 0.25
        and float(r.get("cy_rel", 0.0)) >= 0.65
    )
    lower_third_roi_count = sum(1 for r in of_rois if float(r.get("cy_rel", 0.0)) >= 0.67)
    of_lower_third_roi_ratio = (
        float(lower_third_roi_count) / float(max(of_count, 1)) if of_count > 0 else 0.0
    )
    of_corner_compact_roi_count = sum(
        1 for r in of_rois
        if float(r.get("area_ratio", 0.0)) >= 0.0002
        and float(r.get("area_ratio", 0.0)) <= 0.03
        and (
            (float(r.get("cx_rel", 0.5)) <= 0.28 and float(r.get("cy_rel", 0.5)) <= 0.28)
            or (float(r.get("cx_rel", 0.5)) >= 0.72 and float(r.get("cy_rel", 0.5)) <= 0.28)
            or (float(r.get("cx_rel", 0.5)) <= 0.28 and float(r.get("cy_rel", 0.5)) >= 0.72)
            or (float(r.get("cx_rel", 0.5)) >= 0.72 and float(r.get("cy_rel", 0.5)) >= 0.72)
        )
    )

    iw_similarity = float(iw_data.get("score",   0.0))
    iw_matched    = iw_data.get("matched") or ""
    iw_found      = 1 if iw_data.get("found", False) else 0
    iw_method     = iw_data.get("method", "")

    fft_found = 1 if fft_data.get("found", False) else 0
    fft_score = float(fft_data.get("score", 0.0))
    freq_hf_ratio_mean = float(
        fft_data.get("freq_hf_ratio_mean", fft_data.get("freq_hf_ratio", 0.0))
    )

    return {
        "zv_count":           zv_count,
        "zv_max_score":       round(zv_max_score,  4),
        "zv_lower_third_roi_count": zv_lower_third_roi_count,
        "of_count":           of_count,
        "of_max_area":        of_max_area,
        "of_max_area_ratio":  round(of_max_area_ratio, 6),
        "of_global_motion":   round(of_global,     4),
        "of_texture_variance_mean": round(of_texture_variance_mean, 4),
        "of_low_texture_roi_count": of_low_texture_roi_count,
        "of_wide_lower_roi_count": of_wide_lower_roi_count,
        "of_corner_compact_roi_count": of_corner_compact_roi_count,
        "of_lower_third_roi_ratio": round(of_lower_third_roi_ratio, 4),
        "iw_found":           iw_found,
        "iw_best_similarity": round(iw_similarity, 4),
        "iw_matched":         iw_matched,
        "iw_method":          iw_method,
        "fft_found":          fft_found,
        "fft_score":          round(fft_score,     4),
        "freq_hf_ratio_mean": round(freq_hf_ratio_mean, 4),
    }


# ───────────────────────────────────────────────────────────────────────
# Metryki
# ───────────────────────────────────────────────────────────────────────

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
        prec = tp / (tp + fp)   if (tp + fp) else 0
        rec  = tp / (tp + fn)   if (tp + fn) else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        fpr  = fp / (fp + tn)   if (fp + tn) else 0
        spec = tn / (tn + fp)   if (tn + fp) else 0
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


# ───────────────────────────────────────────────────────────────────────
# Threshold sweep (rozszerzony o iw_strong)
# ───────────────────────────────────────────────────────────────────────

def run_threshold_sweep(raw_rows: list[dict]) -> list[dict]:
    """
    Sweep bez ponownego przetwarzania wideo dla reguly punktowej 3/6.
    """
    points_thresholds = POINTS_THRESHOLD_SWEEP

    sweep_rows = []
    for pts_thr in points_thresholds:
        preds, gts = [], []
        # Zliczaj FP per split osobno (potrzebne do wyboru konfiguracji wg TPR/FPR per-split)
        split_tp:  dict[str, int] = {}
        split_fp:  dict[str, int] = {}
        split_tn:  dict[str, int] = {}
        split_fn:  dict[str, int] = {}
        for r in raw_rows:
            gt_val = int(r["ground_truth"])
            cat    = r["category"]
            det, _, _, _, _ = fuse(
                zv_count      = int(r["zv_count"]),
                zv_lower_third_roi_count = int(r.get("zv_lower_third_roi_count", 0)),
                of_count      = int(r["of_count"]),
                of_max_area   = float(r.get("of_max_area", 0.0)),
                of_max_area_ratio = float(r.get("of_max_area_ratio", 0.0)),
                iw_similarity = float(r["iw_best_similarity"]),
                iw_matched    = r["iw_matched"],
                fft_score     = float(r["fft_score"]),
                of_texture_variance_mean = float(r.get("of_texture_variance_mean", 0.0)),
                of_low_texture_roi_count = int(r.get("of_low_texture_roi_count", 0)),
                of_wide_lower_roi_count = int(r.get("of_wide_lower_roi_count", 0)),
                of_corner_compact_roi_count = int(r.get("of_corner_compact_roi_count", 0)),
                of_lower_third_roi_ratio = float(r.get("of_lower_third_roi_ratio", 0.0)),
                freq_hf_ratio_mean = float(r.get("freq_hf_ratio_mean", 0.0)),
                points_threshold = pts_thr,
            )
            preds.append(det)
            gts.append(gt_val)
            if det == 1 and gt_val == 1: split_tp[cat] = split_tp.get(cat, 0) + 1
            if det == 1 and gt_val == 0: split_fp[cat] = split_fp.get(cat, 0) + 1
            if det == 0 and gt_val == 0: split_tn[cat] = split_tn.get(cat, 0) + 1
            if det == 0 and gt_val == 1: split_fn[cat] = split_fn.get(cat, 0) + 1

        tp = sum(p == 1 and g == 1 for p, g in zip(preds, gts))
        tn = sum(p == 0 and g == 0 for p, g in zip(preds, gts))
        fp = sum(p == 1 and g == 0 for p, g in zip(preds, gts))
        fn = sum(p == 0 and g == 1 for p, g in zip(preds, gts))
        n  = len(gts)
        acc  = (tp + tn) / n if n else 0
        prec = tp / (tp + fp)   if (tp + fp) else 0
        rec  = tp / (tp + fn)   if (tp + fn) else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        fpr  = fp / (fp + tn)   if (fp + tn) else 0

        # TPR per split (klucz: TPR_aibaseline, FPR_adv_fp_trap)
        def _tpr(cat: str) -> str:
            tp_c = split_tp.get(cat, 0)
            fn_c = split_fn.get(cat, 0)
            return f"{tp_c / (tp_c + fn_c):.4f}" if (tp_c + fn_c) > 0 else "N/A"

        def _fpr_split(cat: str) -> str:
            fp_c = split_fp.get(cat, 0)
            tn_c = split_tn.get(cat, 0)
            return f"{fp_c / (fp_c + tn_c):.4f}" if (fp_c + tn_c) > 0 else "N/A"

        sweep_rows.append({
            "points_thr":            pts_thr,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "accuracy":              f"{acc:.4f}",
            "precision":             f"{prec:.4f}",
            "recall":                f"{rec:.4f}",
            "f1":                    f"{f1:.4f}",
            "FPR_global":            f"{fpr:.4f}",
            "TPR_aibaseline":        _tpr("ai_baseline"),
            "FPR_adv_fp_trap":       _fpr_split("adv_fp_trap"),
            "TPR_adv_compressed":    _tpr("adv_compressed"),
            "TPR_adv_cropped":       _tpr("adv_cropped"),
        })

    # Sortuj: najpierw FPR_adv_fp_trap ASC, potem TPR_aibaseline DESC
    def _sort_key(x: dict):
        fpr_fp = float(x["FPR_adv_fp_trap"]) if x["FPR_adv_fp_trap"] != "N/A" else 1.0
        tpr_ai = float(x["TPR_aibaseline"])  if x["TPR_aibaseline"]  != "N/A" else 0.0
        return (fpr_fp, -tpr_ai)

    sweep_rows.sort(key=_sort_key)
    return sweep_rows


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main() -> None:
    snap_dir  = make_snapshot_dir()
    git_hash  = get_git_hash()
    run_time  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[SNAP]  Snapshot: {snap_dir}")
    print(f"[SNAP]  Commit:   {git_hash}")

    raw_rows:  list[dict] = []
    eval_rows: list[dict] = []
    total_videos = 0

    for category, (folder, gt) in CATEGORIES.items():
        videos = sorted(folder.glob("*.mp4"))
        if not videos:
            print(f"[WARN] Brak .mp4 w {folder}", file=sys.stderr)
            continue
        print(f"\n=== {category} ({len(videos)} filmow) ===")
        total_videos += len(videos)

        for vp in videos:
            print(f"  [SCAN] {vp.name} ... ", end="", flush=True)
            try:
                result, elapsed = scan_video(vp)
                sig = extract_signals(result)

                raw_row = {
                    "category":         category,
                    "filename":         vp.name,
                    "ground_truth":     gt,
                    **sig,
                    "frames_sampled":   30,
                    "duration_s":       f"{elapsed:.2f}",
                    "detector_version": DETECTOR_VERSION,
                }
                raw_rows.append(raw_row)

                det, score, mode, ai_specific, broadcast_trap = fuse(
                    zv_count      = sig["zv_count"],
                    zv_lower_third_roi_count = sig["zv_lower_third_roi_count"],
                    of_count      = sig["of_count"],
                    of_max_area   = sig["of_max_area"],
                    of_max_area_ratio = sig["of_max_area_ratio"],
                    iw_similarity = sig["iw_best_similarity"],
                    iw_matched    = sig["iw_matched"],
                    fft_score     = sig["fft_score"],
                    of_texture_variance_mean = sig["of_texture_variance_mean"],
                    of_low_texture_roi_count = sig["of_low_texture_roi_count"],
                    of_wide_lower_roi_count = sig["of_wide_lower_roi_count"],
                    of_corner_compact_roi_count = sig["of_corner_compact_roi_count"],
                    of_lower_third_roi_ratio = sig["of_lower_third_roi_ratio"],
                    freq_hf_ratio_mean = sig["freq_hf_ratio_mean"],
                )

                eval_row = {
                    "category":           category,
                    "filename":           vp.name,
                    "ground_truth":       gt,
                    "detected":           det,
                    "fusion_score":       score,
                    "fusion_mode":        mode,
                    "ai_specific":        ai_specific,
                    "broadcast_trap":     broadcast_trap,
                    "zv_count":           sig["zv_count"],
                    "zv_lower_third_roi_count": sig["zv_lower_third_roi_count"],
                    "of_count":           sig["of_count"],
                    "of_max_area_ratio":  sig["of_max_area_ratio"],
                    "iw_best_similarity": sig["iw_best_similarity"],
                    "iw_matched":         sig["iw_matched"],
                    "fft_score":          sig["fft_score"],
                    "of_texture_variance_mean": sig["of_texture_variance_mean"],
                    "of_low_texture_roi_count": sig["of_low_texture_roi_count"],
                    "of_wide_lower_roi_count": sig["of_wide_lower_roi_count"],
                    "of_corner_compact_roi_count": sig["of_corner_compact_roi_count"],
                    "of_lower_third_roi_ratio": sig["of_lower_third_roi_ratio"],
                    "freq_hf_ratio_mean": sig["freq_hf_ratio_mean"],
                    "duration_s":         f"{elapsed:.2f}",
                }
                eval_rows.append(eval_row)

                det_str = "WYKRYTO" if det else "brak"
                print(f"{det_str}  score={score:.3f}  mode={mode}  ({elapsed:.1f}s)")

            except Exception as e:
                print(f"BLAD: {e}", file=sys.stderr)

    # —— Zapis plikow wynikowych ——
    def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    if raw_rows:
        _write_csv(snap_dir / "raw_signals.csv",        RAW_FIELDS,  raw_rows)
    if eval_rows:
        _write_csv(snap_dir / "evaluation_results.csv", EVAL_FIELDS, eval_rows)

    metrics: list[dict] = []
    if eval_rows:
        metrics = compute_metrics(eval_rows)
        _write_csv(snap_dir / "metrics_summary.csv", list(metrics[0].keys()), metrics)

    # —— run_info.txt ——
    with (snap_dir / "run_info.txt").open("w", encoding="utf-8") as f:
        f.write(f"commit:       {git_hash}\n")
        f.write(f"run_time:     {run_time}\n")
        f.write(f"total_videos: {total_videos}\n")
        f.write(f"categories:   {list(CATEGORIES.keys())}\n")
        f.write(f"detector_ver: {DETECTOR_VERSION}\n")

    print(f"\n[RAW]     {snap_dir / 'raw_signals.csv'}")
    print(f"[EVAL]    {snap_dir / 'evaluation_results.csv'}")
    print(f"[METRICS] {snap_dir / 'metrics_summary.csv'}")
    print(f"[INFO]    {snap_dir / 'run_info.txt'}")

    if metrics:
        print("\n--- Podsumowanie metryk (domyslne progi) ---")
        for m in metrics:
            print(
                f"  {m['category']:18s}: n={m['n']:3d} "
                f"acc={m['accuracy']}  f1={m['f1']}  "
                f"FPR={m['FPR']}  spec={m['specificity']}  "
                f"TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}"
            )

    # —— Threshold sweep (na surowych sygnałach) ——
    if raw_rows:
        print("\n[SWEEP] Obliczam threshold sweep (progi punktowe)...")
        sweep = run_threshold_sweep(raw_rows)
        sweep_path = snap_dir / "threshold_sweep.csv"
        with sweep_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
            writer.writeheader()
            writer.writerows(sweep)
        print(f"[SWEEP]   {sweep_path}  ({len(sweep)} kombinacji)")

        print("\n--- Top-5 konfiguracji (FPR_adv_fp_trap ASC, TPR_aibaseline DESC) ---")
        for row in sweep[:5]:
            print(
                f"  pts={row['points_thr']}  "
                f"FPR_fptrap={row['FPR_adv_fp_trap']}  "
                f"TPR_ai={row['TPR_aibaseline']}  F1={row['f1']}"
            )

    # —— Kopiuj do latest/ ——
    copy_to_latest(snap_dir)
    print(f"\n[LATEST] {RESULTS_BASE / 'latest'}")


if __name__ == "__main__":
    main()

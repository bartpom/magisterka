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
import json
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
from advanced_detectors import initialize_invisible_watermark, run_advanced_scan
try:
    from c2pa_detector import detect_c2pa  # type: ignore
except Exception:
    detect_c2pa = None
from fusion_params import (
    BILLBOARD_CENTER_RATIO_MIN,
    BILLBOARD_GLOBAL_MOTION_MIN,
    BILLBOARD_TEXTURE_MIN,
    BROADCAST_SOFT_BILLBOARD_OF_COUNT_MIN,
    BROADCAST_SOFT_BILLBOARD_TEXTURE_VAR_MIN,
    BROADCAST_SOFT_BILLBOARD_UPPER_RATIO_MAX,
    BROADCAST_SOFT_LOWERTHIRD_RATIO_MIN,
    BROADCAST_SOFT_SCOREBOARD_CENTER_RATIO_MAX,
    BROADCAST_SOFT_SCOREBOARD_LOWER_RATIO_MIN,
    BROADCAST_SOFT_SCOREBOARD_OF_COUNT_MIN,
    BROADCAST_SOFT_SCOREBOARD_UPPER_RATIO_MIN,
    CROPPED_OF_RECOVERY_BONUS_POINTS,
    CROPPED_OF_RECOVERY_MAX_AREA_RATIO,
    CROPPED_OF_RECOVERY_MAX_CENTER_RATIO,
    CROPPED_OF_RECOVERY_MAX_LOWER_RATIO,
    CROPPED_OF_RECOVERY_MIN_LOW_TEXTURE_SHARE,
    CROPPED_OF_RECOVERY_MIN_OF_COUNT,
    CLEAN_AI_HF_THRESHOLD,
    CLEAN_AI_MAX_AREA_RATIO,
    ENABLE_CROPPED_OF_RECOVERY,
    ENABLE_BROADCAST_SOFT_GUARD,
    ENABLE_CLEAN_AI_RESCUE,
    HF_RATIO_THRESHOLD_SWEEP,
    HF_RATIO_THRESHOLD,
    HIGH_SCORE_OVERRIDE_THRESHOLD,
    LOW_TEXTURE_THRESHOLD_SWEEP,
    LOWER_THIRD_HARD_THRESHOLD,
    LOWER_THIRD_HARD_UPPER_MAX,
    LOW_TEXTURE_THRESHOLD,
    MAX_AREA_RATIO_THRESHOLD,
    POINTS_THRESHOLD_DEFAULT,
    POINTS_THRESHOLD_SWEEP,
    SCOREBOARD_HF_MIN,
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
    "of_upper_third_roi_ratio", "of_center_roi_ratio", "of_wide_top_bottom_count",
    "broadcast_scoreboard_trap", "broadcast_billboard_trap",
    "broadcast_pattern_trap", "broadcast_lower_third_pattern", "broadcast_scoreboard_pattern", "broadcast_billboard_pattern",
    "iw_found", "iw_best_similarity", "iw_matched", "iw_method",
    "fft_found", "fft_score", "freq_hf_ratio_mean",
    "c2pa_found", "c2pa_ai", "c2pa_generator", "c2pa_error",
    "frames_sampled", "duration_s", "detector_version",
]

EVAL_FIELDS = [
    "category", "filename", "ground_truth",
    "detected", "fusion_score", "fusion_mode", "ai_specific", "broadcast_trap", "high_score_override",
    "zv_count", "zv_lower_third_roi_count", "of_count", "of_max_area_ratio", "iw_best_similarity", "iw_matched",
    "fft_score", "of_texture_variance_mean", "of_low_texture_roi_count",
    "of_wide_lower_roi_count", "of_corner_compact_roi_count", "of_lower_third_roi_ratio",
    "of_upper_third_roi_ratio", "of_center_roi_ratio", "of_wide_top_bottom_count",
    "broadcast_scoreboard_trap", "broadcast_billboard_trap",
    "broadcast_pattern_trap", "broadcast_lower_third_pattern", "broadcast_scoreboard_pattern", "broadcast_billboard_pattern",
    "freq_hf_ratio_mean", "c2pa_found", "c2pa_ai", "c2pa_generator", "duration_s",
]

DETECTOR_VERSION = "adv_v7_cinematic_rescue"

# Próg HF dla ścieżki kinematograficznej (luźniejszy niż CLEAN_AI_HF_THRESHOLD)
CINEMATIC_HF_THRESHOLD = 0.52
# Maksymalne area_ratio dla ścieżki kinematograficznej (filmy z dużym ruchem)
CINEMATIC_MAX_AREA_RATIO = 0.60
# Minimalny of_count dla ścieżki kinematograficznej
CINEMATIC_MIN_OF_COUNT = 3


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


C2PA_AI_ORIGIN_KEYWORDS = (
    "openai",
    "sora",
    "runway",
    "adobe firefly",
    "firefly",
    "luma",
    "pika",
    "kling",
    "hailuo",
    "stable video",
    "cogvideo",
    "generative ai",
    "ai generated",
    "synthetic media",
)


def detect_c2pa_signal(video_path: Path) -> dict[str, Any]:
    """
    C2PA check nigdy nie blokuje skanu:
    - found+ai_origin => c2pa_ai=1
    - brak/blad => c2pa_ai=0
    """
    default = {
        "c2pa_found": 0,
        "c2pa_ai": 0,
        "c2pa_generator": "",
        "c2pa_error": "",
    }
    if detect_c2pa is None:
        print("ADV C2PA: not found")
        return default

    try:
        res = detect_c2pa(str(video_path))
    except Exception as e:
        print("ADV C2PA: not found")
        default["c2pa_error"] = str(e)
        return default

    if not getattr(res, "found", False):
        print("ADV C2PA: not found")
        err = getattr(res, "error", None)
        if err:
            default["c2pa_error"] = str(err)
        return default

    generator = str(getattr(res, "generator", "") or "")
    producer = str(getattr(res, "producer", "") or "")
    manifest_blob = ""
    raw_manifest = getattr(res, "raw_manifest", None)
    if raw_manifest:
        try:
            manifest_blob = json.dumps(raw_manifest, ensure_ascii=False).lower()
        except Exception:
            manifest_blob = str(raw_manifest).lower()

    claim_blob = " ".join((generator.lower(), producer.lower(), manifest_blob))
    ai_origin = int(any(k in claim_blob for k in C2PA_AI_ORIGIN_KEYWORDS))
    print(f"ADV C2PA: found, ai_origin={ai_origin}")
    return {
        "c2pa_found": 1,
        "c2pa_ai": ai_origin,
        "c2pa_generator": generator,
        "c2pa_error": str(getattr(res, "error", "") or ""),
    }


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
    c2pa_ai = int(row.get("c2pa_ai", 0)) == 1
    wide_lower = int(row.get("of_wide_lower_roi_count", 0))
    corner_compact = int(row.get("of_corner_compact_roi_count", 0))
    lower_third_ratio = float(row.get("of_lower_third_roi_ratio", 0.0))
    upper_third_ratio = float(row.get("of_upper_third_roi_ratio", 0.0))
    zv_lower_third = int(row.get("zv_lower_third_roi_count", 0))
    hf_ratio = float(row.get("freq_hf_ratio_mean", 1.0))

    # Signal 1: OF obecny i niezdominowany przez gigantyczny overlay.
    if of_count >= 5:
        score += 1
    if area_ratio < max_area_ratio_threshold:
        score += 1
    else:
        score -= 1

    # candidate_ai_shape: trzy niezależne ścieżki aktywacji sygnałów texture/HF:
    # (a) klasyczna: corner watermark LUB kompaktowy overlay
    # (b) kinematyczna: dużo OF + niskie HF — film AI bez overlayów (Sora, Runway Gen3)
    candidate_ai_shape = (
        corner_compact >= 1
        or (of_count >= 3 and area_ratio < max_area_ratio_threshold)
        or (of_count >= 5 and hf_ratio < CINEMATIC_HF_THRESHOLD and wide_lower == 0)  # ścieżka kinematyczna
    )

    # Signal 2: niski texture variance w ROI OF (AI smoothness)
    if low_texture and candidate_ai_shape:
        score += 1
    # Signal 3: niski udział HF (AI smoothness w domenie częstotliwości)
    if low_hf and candidate_ai_shape:
        score += 1
    # Signal 4: regiony zero-variance (statyczne overlaye/logotypy)
    if int(row.get("zv_count", 0)) >= 1:
        score += 1
    if iw_strong:
        score += 2
    if c2pa_ai:
        score += 2
    # Geometry-aware refinements:
    if wide_lower >= 1:
        score -= 2
    if corner_compact >= 1:
        score += 1
    # Hard broadcast-trap heuristic: dolny-dominujacy OF + statyczne ROI na dole.
    if (
        lower_third_ratio > lower_third_hard_threshold
        and upper_third_ratio < LOWER_THIRD_HARD_UPPER_MAX
        and zv_lower_third > 0
    ):
        score -= 3
    if int(row.get("broadcast_scoreboard_trap", 0)) == 1:
        score -= 3
    if int(row.get("broadcast_billboard_trap", 0)) == 1:
        score -= 2
    if int(row.get("broadcast_pattern_trap", 0)) == 1:
        score -= 2
    if (
        lower_third_ratio >= 0.50
        and wide_lower >= 1
        and corner_compact == 0
    ):
        score -= 1
    # Cropping can remove corner anchors and shift OF ROI zones; recover one point
    # using position-agnostic OF/texture cues when the overlay footprint is compact.
    low_texture_share = (
        float(int(row.get("of_low_texture_roi_count", 0))) / float(max(of_count, 1))
    )
    if (
        ENABLE_CROPPED_OF_RECOVERY
        and of_count >= CROPPED_OF_RECOVERY_MIN_OF_COUNT
        and area_ratio <= CROPPED_OF_RECOVERY_MAX_AREA_RATIO
        and lower_third_ratio <= CROPPED_OF_RECOVERY_MAX_LOWER_RATIO
        and float(row.get("of_center_roi_ratio", 0.0)) <= CROPPED_OF_RECOVERY_MAX_CENTER_RATIO
        and wide_lower == 0
        and low_texture_share >= CROPPED_OF_RECOVERY_MIN_LOW_TEXTURE_SHARE
    ):
        score += CROPPED_OF_RECOVERY_BONUS_POINTS
    return score


def compute_ai_flags(
    row: dict[str, Any],
    low_texture_threshold: int = LOW_TEXTURE_THRESHOLD,
    hf_ratio_threshold: float = HF_RATIO_THRESHOLD,
    max_area_ratio_threshold: float = MAX_AREA_RATIO_THRESHOLD,
) -> tuple[int, int]:
    iw_strong = bool(row.get("iw_matched")) and float(row.get("iw_similarity", 0.0)) >= 0.85
    hf_ratio = float(row.get("freq_hf_ratio_mean", 1.0))
    of_count = int(row.get("of_count", 0))
    area_ratio = float(row.get("of_max_area_ratio", 1.0))
    wide_lower = int(row.get("of_wide_lower_roi_count", 0))

    shape_signal = (
        int(row.get("of_corner_compact_roi_count", 0)) >= 1
        or area_ratio < max_area_ratio_threshold
        or (of_count >= 5 and hf_ratio < CINEMATIC_HF_THRESHOLD and wide_lower == 0)  # ścieżka kinematyczna
    )
    ai_signals = [
        shape_signal and int(row.get("of_low_texture_roi_count", 0)) >= low_texture_threshold,
        shape_signal and hf_ratio < hf_ratio_threshold,
        int(row.get("of_corner_compact_roi_count", 0)) >= 1,
        of_count >= 3 and area_ratio < max_area_ratio_threshold,
        # ścieżka kinematyczna jako samodzielny sygnał AI
        of_count >= CINEMATIC_MIN_OF_COUNT and hf_ratio < CINEMATIC_HF_THRESHOLD and wide_lower == 0,
    ]
    c2pa_ai = int(row.get("c2pa_ai", 0)) == 1
    ai_specific = int(iw_strong or c2pa_ai or sum(ai_signals) >= 2)
    lower_third_ratio = float(row.get("of_lower_third_roi_ratio", 0.0))
    upper_third_ratio = float(row.get("of_upper_third_roi_ratio", 1.0))
    center_ratio = float(row.get("of_center_roi_ratio", 0.0))
    texture_var = float(row.get("of_texture_variance_mean", 0.0))
    lower_third_trap = int(
        lower_third_ratio > LOWER_THIRD_HARD_THRESHOLD
        and upper_third_ratio < LOWER_THIRD_HARD_UPPER_MAX
        and row.get("zv_lower_third_roi_count", 0) > 0
    )
    scoreboard_trap = int(row.get("broadcast_scoreboard_trap", 0))
    billboard_trap = int(row.get("broadcast_billboard_trap", 0))
    pattern_trap = int(row.get("broadcast_pattern_trap", 0))
    scoreboard_soft = int(
        scoreboard_trap
        or (
            lower_third_ratio >= BROADCAST_SOFT_SCOREBOARD_LOWER_RATIO_MIN
            and upper_third_ratio >= BROADCAST_SOFT_SCOREBOARD_UPPER_RATIO_MIN
            and center_ratio <= BROADCAST_SOFT_SCOREBOARD_CENTER_RATIO_MAX
            and of_count >= BROADCAST_SOFT_SCOREBOARD_OF_COUNT_MIN
        )
    )
    billboard_soft = int(
        billboard_trap
        or (
            texture_var >= BROADCAST_SOFT_BILLBOARD_TEXTURE_VAR_MIN
            and upper_third_ratio <= BROADCAST_SOFT_BILLBOARD_UPPER_RATIO_MAX
            and of_count >= BROADCAST_SOFT_BILLBOARD_OF_COUNT_MIN
        )
    )
    lower_third_soft = int(lower_third_ratio >= BROADCAST_SOFT_LOWERTHIRD_RATIO_MIN)
    # Pattern trap bywa nadwrażliwy (np. krótkie klipy AI z overlayami YouTube),
    # więc traktujemy go jako sygnał pomocniczy, nie samodzielny hard gate.
    pattern_confirmed = int(pattern_trap and (scoreboard_trap or billboard_trap or lower_third_trap))
    soft_broadcast_guard = int(
        ENABLE_BROADCAST_SOFT_GUARD
        and (scoreboard_soft or (billboard_soft and lower_third_soft))
    )
    broadcast_trap = int(
        lower_third_trap
        or scoreboard_trap
        or billboard_trap
        or pattern_confirmed
        or soft_broadcast_guard
    )
    if broadcast_trap:
        ai_specific = 0
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
    of_upper_third_roi_ratio: float,
    of_center_roi_ratio: float,
    of_wide_top_bottom_count: int,
    broadcast_scoreboard_trap: int,
    broadcast_billboard_trap: int,
    broadcast_pattern_trap: int,
    broadcast_lower_third_pattern: int,
    broadcast_scoreboard_pattern: int,
    broadcast_billboard_pattern: int,
    freq_hf_ratio_mean: float,
    c2pa_ai: int = 0,
    points_threshold: int = POINTS_THRESHOLD_DEFAULT,
    low_texture_threshold: int = LOW_TEXTURE_THRESHOLD,
    hf_ratio_threshold: float = HF_RATIO_THRESHOLD,
    max_area_ratio_threshold: float = MAX_AREA_RATIO_THRESHOLD,
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
        "of_upper_third_roi_ratio": of_upper_third_roi_ratio,
        "of_center_roi_ratio": of_center_roi_ratio,
        "of_wide_top_bottom_count": of_wide_top_bottom_count,
        "broadcast_scoreboard_trap": broadcast_scoreboard_trap,
        "broadcast_billboard_trap": broadcast_billboard_trap,
        "broadcast_pattern_trap": broadcast_pattern_trap,
        "broadcast_lower_third_pattern": broadcast_lower_third_pattern,
        "broadcast_scoreboard_pattern": broadcast_scoreboard_pattern,
        "broadcast_billboard_pattern": broadcast_billboard_pattern,
        "freq_hf_ratio_mean": freq_hf_ratio_mean,
        "c2pa_ai": c2pa_ai,
    }
    score = compute_ai_score(
        row,
        low_texture_threshold=low_texture_threshold,
        hf_ratio_threshold=hf_ratio_threshold,
        max_area_ratio_threshold=max_area_ratio_threshold,
    )
    ai_specific, broadcast_trap = compute_ai_flags(
        row,
        low_texture_threshold=low_texture_threshold,
        hf_ratio_threshold=hf_ratio_threshold,
        max_area_ratio_threshold=max_area_ratio_threshold,
    )
    lower_third_ok = not broadcast_trap
    points_score = score

    # ── clean_ai_rescue: filmy AI bez overlayów (krótkie Sora-clips, proste gen) ──
    # Ścieżka 1 (oryginalna): brak OF lub jeden OF, czysty spektralnie
    clean_ai_candidate = (
        int(row.get("of_count", 0)) <= 8  # poluzowano z <=1 — pokrywa więcej Sora-clips
        and int(row.get("zv_count", 0)) == 0
        and int(row.get("broadcast_scoreboard_trap", 0)) == 0
        and int(row.get("broadcast_billboard_trap", 0)) == 0
        and int(row.get("broadcast_pattern_trap", 0)) == 0
        and int(row.get("of_wide_lower_roi_count", 0)) == 0
        and float(row.get("freq_hf_ratio_mean", 1.0)) < CLEAN_AI_HF_THRESHOLD
    )
    clean_ai_rescue_strict = (
        clean_ai_candidate
        and float(row.get("of_max_area_ratio", 1.0)) < CLEAN_AI_MAX_AREA_RATIO
    )

    # ── clean_ai_rescue_motion: filmy kinematograficzne AI z dużą ilością OF ──
    # Runway Gen3 "The Shadow" (of=54, hf=0.396), cyberpunk city, Luma Modify itp.
    # Warunki: dużo ruchu (of>=3), niskie HF, brak jakiegokolwiek broadcast trapa,
    #          brak wide_lower (szeroki dolny pasek = typowy overlay/lower-third),
    #          brak zv (statyczny overlay byłby FP jak BBC news).
    clean_ai_rescue_motion = (
        int(row.get("of_count", 0)) >= CINEMATIC_MIN_OF_COUNT
        and float(row.get("freq_hf_ratio_mean", 1.0)) < CINEMATIC_HF_THRESHOLD
        and int(row.get("zv_count", 0)) == 0
        and int(row.get("of_wide_lower_roi_count", 0)) == 0
        and int(row.get("broadcast_scoreboard_trap", 0)) == 0
        and int(row.get("broadcast_billboard_trap", 0)) == 0
        and int(row.get("broadcast_pattern_trap", 0)) == 0
        and float(row.get("of_max_area_ratio", 1.0)) < CINEMATIC_MAX_AREA_RATIO
    )

    detected = int(
        points_score >= POINTS_THRESHOLD_DEFAULT
        or (ENABLE_CLEAN_AI_RESCUE and clean_ai_rescue_strict)
        or (ENABLE_CLEAN_AI_RESCUE and clean_ai_rescue_motion)
    )
    rescue_path = ""
    if ENABLE_CLEAN_AI_RESCUE and clean_ai_rescue_strict and points_score < POINTS_THRESHOLD_DEFAULT:
        rescue_path = ";rescue=clean_strict"
    elif ENABLE_CLEAN_AI_RESCUE and clean_ai_rescue_motion and points_score < POINTS_THRESHOLD_DEFAULT:
        rescue_path = ";rescue=cinematic_motion"

    return detected, float(score), (
        f"ai_score={score};ai_specific={int(ai_specific)};lower_third_ok={int(lower_third_ok)}"
        f";lower_third_effective={int(lower_third_ok)};c2pa_ai={int(c2pa_ai)}"
        f";flux=0;ai_style_prob=0.00;flux_fft_score=0;fft_bonus=0;tc_score=0;tc_bonus=0"
        f"{rescue_path}"
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
    trap_data = result.get("broadcast_traps",   {})

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
    upper_third_roi_count = sum(1 for r in of_rois if float(r.get("cy_rel", 0.0)) <= 0.33)
    center_roi_count = sum(
        1 for r in of_rois
        if 0.33 < float(r.get("cy_rel", 0.0)) < 0.67
    )
    of_lower_third_roi_ratio = (
        float(lower_third_roi_count) / float(max(of_count, 1)) if of_count > 0 else 0.0
    )
    of_upper_third_roi_ratio = (
        float(upper_third_roi_count) / float(max(of_count, 1)) if of_count > 0 else 0.0
    )
    of_center_roi_ratio = (
        float(center_roi_count) / float(max(of_count, 1)) if of_count > 0 else 0.0
    )
    of_wide_top_bottom_count = sum(
        1 for r in of_rois
        if float(r.get("width_ratio", 0.0)) >= 0.80
        and (float(r.get("cy_rel", 0.0)) <= 0.30 or float(r.get("cy_rel", 0.0)) >= 0.70)
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
    broadcast_scoreboard_trap = int(
        of_wide_top_bottom_count >= 1
        and freq_hf_ratio_mean >= SCOREBOARD_HF_MIN
        and of_low_texture_roi_count == 0
    )
    broadcast_billboard_trap = int(
        of_center_roi_ratio >= BILLBOARD_CENTER_RATIO_MIN
        and of_global >= BILLBOARD_GLOBAL_MOTION_MIN
        and of_texture_variance_mean >= BILLBOARD_TEXTURE_MIN
        and of_low_texture_roi_count == 0
    )
    broadcast_pattern_trap = int(trap_data.get("broadcast_trap", False))
    broadcast_lower_third_pattern = int(trap_data.get("lower_third_anim", False))
    broadcast_scoreboard_pattern = int(trap_data.get("scoreboard_top_pair", False))
    broadcast_billboard_pattern = int(trap_data.get("billboard_center_large", False))

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
        "of_upper_third_roi_ratio": round(of_upper_third_roi_ratio, 4),
        "of_center_roi_ratio": round(of_center_roi_ratio, 4),
        "of_wide_top_bottom_count": of_wide_top_bottom_count,
        "broadcast_scoreboard_trap": broadcast_scoreboard_trap,
        "broadcast_billboard_trap": broadcast_billboard_trap,
        "broadcast_pattern_trap": broadcast_pattern_trap,
        "broadcast_lower_third_pattern": broadcast_lower_third_pattern,
        "broadcast_scoreboard_pattern": broadcast_scoreboard_pattern,
        "broadcast_billboard_pattern": broadcast_billboard_pattern,
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

def run_threshold_sweep(raw_rows: list[dict]) -> dict[str, list[dict]]:
    """
    Rozszerzony sweep po progach:
    - points threshold
    - HF ratio threshold
    - low-texture threshold
    """
    points_thresholds = POINTS_THRESHOLD_SWEEP
    hf_thresholds = HF_RATIO_THRESHOLD_SWEEP
    texture_thresholds = LOW_TEXTURE_THRESHOLD_SWEEP

    heuristic_rows: list[dict] = []
    activation_rows: list[dict] = []
    score_dist_rows: list[dict] = []

    categories = sorted({r["category"] for r in raw_rows})

    for hf_thr in hf_thresholds:
        for tex_thr in texture_thresholds:
            activation_totals = {
                cat: {
                    "n": 0,
                    "low_hf": 0,
                    "low_texture": 0,
                    "corner_compact": 0,
                    "scoreboard_trap": 0,
                    "billboard_trap": 0,
                    "pattern_trap": 0,
                }
                for cat in categories
            }
            for r in raw_rows:
                cat = r["category"]
                activation_totals[cat]["n"] += 1
                if float(r.get("freq_hf_ratio_mean", 1.0)) < hf_thr:
                    activation_totals[cat]["low_hf"] += 1
                if int(r.get("of_low_texture_roi_count", 0)) >= tex_thr:
                    activation_totals[cat]["low_texture"] += 1
                if int(r.get("of_corner_compact_roi_count", 0)) >= 1:
                    activation_totals[cat]["corner_compact"] += 1
                if int(r.get("broadcast_scoreboard_trap", 0)) == 1:
                    activation_totals[cat]["scoreboard_trap"] += 1
                if int(r.get("broadcast_billboard_trap", 0)) == 1:
                    activation_totals[cat]["billboard_trap"] += 1
                if int(r.get("broadcast_pattern_trap", 0)) == 1:
                    activation_totals[cat]["pattern_trap"] += 1

            for cat, vals in activation_totals.items():
                n_cat = max(vals["n"], 1)
                activation_rows.append({
                    "hf_ratio_threshold": f"{hf_thr:.2f}",
                    "low_texture_threshold": tex_thr,
                    "category": cat,
                    "low_hf_rate": f"{vals['low_hf'] / n_cat:.4f}",
                    "low_texture_rate": f"{vals['low_texture'] / n_cat:.4f}",
                    "corner_compact_rate": f"{vals['corner_compact'] / n_cat:.4f}",
                    "scoreboard_trap_rate": f"{vals['scoreboard_trap'] / n_cat:.4f}",
                    "billboard_trap_rate": f"{vals['billboard_trap'] / n_cat:.4f}",
                    "pattern_trap_rate": f"{vals['pattern_trap'] / n_cat:.4f}",
                })

            for pts_thr in points_thresholds:
                preds, gts = [], []
                split_tp: dict[str, int] = {}
                split_fp: dict[str, int] = {}
                split_tn: dict[str, int] = {}
                split_fn: dict[str, int] = {}
                score_buckets: dict[str, dict[int, int]] = {cat: {} for cat in categories}

                for r in raw_rows:
                    gt_val = int(r["ground_truth"])
                    cat = r["category"]
                    det, score, _, ai_specific, _ = fuse(
                        zv_count=int(r["zv_count"]),
                        zv_lower_third_roi_count=int(r.get("zv_lower_third_roi_count", 0)),
                        of_count=int(r["of_count"]),
                        of_max_area=float(r.get("of_max_area", 0.0)),
                        of_max_area_ratio=float(r.get("of_max_area_ratio", 0.0)),
                        iw_similarity=float(r["iw_best_similarity"]),
                        iw_matched=r["iw_matched"],
                        fft_score=float(r["fft_score"]),
                        of_texture_variance_mean=float(r.get("of_texture_variance_mean", 0.0)),
                        of_low_texture_roi_count=int(r.get("of_low_texture_roi_count", 0)),
                        of_wide_lower_roi_count=int(r.get("of_wide_lower_roi_count", 0)),
                        of_corner_compact_roi_count=int(r.get("of_corner_compact_roi_count", 0)),
                        of_lower_third_roi_ratio=float(r.get("of_lower_third_roi_ratio", 0.0)),
                        of_upper_third_roi_ratio=float(r.get("of_upper_third_roi_ratio", 0.0)),
                        of_center_roi_ratio=float(r.get("of_center_roi_ratio", 0.0)),
                        of_wide_top_bottom_count=int(r.get("of_wide_top_bottom_count", 0)),
                        broadcast_scoreboard_trap=int(r.get("broadcast_scoreboard_trap", 0)),
                        broadcast_billboard_trap=int(r.get("broadcast_billboard_trap", 0)),
                        broadcast_pattern_trap=int(r.get("broadcast_pattern_trap", 0)),
                        broadcast_lower_third_pattern=int(r.get("broadcast_lower_third_pattern", 0)),
                        broadcast_scoreboard_pattern=int(r.get("broadcast_scoreboard_pattern", 0)),
                        broadcast_billboard_pattern=int(r.get("broadcast_billboard_pattern", 0)),
                        freq_hf_ratio_mean=float(r.get("freq_hf_ratio_mean", 0.0)),
                        c2pa_ai=int(r.get("c2pa_ai", 0)),
                        points_threshold=pts_thr,
                        low_texture_threshold=tex_thr,
                        hf_ratio_threshold=hf_thr,
                    )
                    if ai_specific == 0:
                        det = 0
                    preds.append(det)
                    gts.append(gt_val)
                    bucket = int(round(score))
                    score_buckets[cat][bucket] = score_buckets[cat].get(bucket, 0) + 1
                    if det == 1 and gt_val == 1:
                        split_tp[cat] = split_tp.get(cat, 0) + 1
                    if det == 1 and gt_val == 0:
                        split_fp[cat] = split_fp.get(cat, 0) + 1
                    if det == 0 and gt_val == 0:
                        split_tn[cat] = split_tn.get(cat, 0) + 1
                    if det == 0 and gt_val == 1:
                        split_fn[cat] = split_fn.get(cat, 0) + 1

                for cat, buckets in score_buckets.items():
                    for score_val, count in sorted(buckets.items()):
                        score_dist_rows.append({
                            "hf_ratio_threshold": f"{hf_thr:.2f}",
                            "low_texture_threshold": tex_thr,
                            "points_threshold": pts_thr,
                            "category": cat,
                            "score_bucket": score_val,
                            "count": count,
                        })

                tp = sum(p == 1 and g == 1 for p, g in zip(preds, gts))
                tn = sum(p == 0 and g == 0 for p, g in zip(preds, gts))
                fp = sum(p == 1 and g == 0 for p, g in zip(preds, gts))
                fn = sum(p == 0 and g == 1 for p, g in zip(preds, gts))
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

                def _tpr(cat: str) -> float:
                    tp_c = split_tp.get(cat, 0)
                    fn_c = split_fn.get(cat, 0)
                    return (tp_c / (tp_c + fn_c)) if (tp_c + fn_c) > 0 else 0.0

                def _fpr(cat: str) -> float:
                    fp_c = split_fp.get(cat, 0)
                    tn_c = split_tn.get(cat, 0)
                    return (fp_c / (fp_c + tn_c)) if (fp_c + tn_c) > 0 else 0.0

                heuristic_rows.append({
                    "hf_ratio_threshold": f"{hf_thr:.2f}",
                    "low_texture_threshold": tex_thr,
                    "points_threshold": pts_thr,
                    "points_thr": pts_thr,
                    "TPR_aibaseline": f"{_tpr('ai_baseline'):.4f}",
                    "TPR_adv_compressed": f"{_tpr('adv_compressed'):.4f}",
                    "TPR_adv_cropped": f"{_tpr('adv_cropped'):.4f}",
                    "FPR_adv_fp_trap": f"{_fpr('adv_fp_trap'):.4f}",
                    "F1_global": f"{f1:.4f}",
                    "f1": f"{f1:.4f}",
                    "precision_global": f"{prec:.4f}",
                    "precision": f"{prec:.4f}",
                    "recall_global": f"{rec:.4f}",
                    "recall": f"{rec:.4f}",
                    "objective": f"{(_tpr('ai_baseline') - 2.0 * _fpr('adv_fp_trap')):.4f}",
                })

    feasible = [r for r in heuristic_rows if float(r["FPR_adv_fp_trap"]) <= 0.15]
    if feasible:
        best = sorted(
            feasible,
            key=lambda x: (float(x["TPR_aibaseline"]), -float(x["FPR_adv_fp_trap"])),
            reverse=True,
        )[0]
    else:
        best = sorted(heuristic_rows, key=lambda x: float(x["objective"]), reverse=True)[0]

    # Pareto frontier (maximize TPR, minimize FPR)
    sorted_rows = sorted(
        heuristic_rows,
        key=lambda x: (float(x["FPR_adv_fp_trap"]), -float(x["TPR_aibaseline"])),
    )
    pareto_rows: list[dict] = []
    best_tpr_seen = -1.0
    for row in sorted_rows:
        tpr = float(row["TPR_aibaseline"])
        if tpr > best_tpr_seen:
            pareto_rows.append(row)
            best_tpr_seen = tpr

    heuristic_rows.sort(
        key=lambda x: (float(x["FPR_adv_fp_trap"]), -float(x["TPR_aibaseline"]))
    )
    return {
        "heuristic_rows": heuristic_rows,
        "activation_rows": activation_rows,
        "score_dist_rows": score_dist_rows,
        "best_row": [best],
        "pareto_rows": pareto_rows,
    }


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main() -> None:
    snap_dir  = make_snapshot_dir()
    git_hash  = get_git_hash()
    run_time  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[SNAP]  Snapshot: {snap_dir}")
    print(f"[SNAP]  Commit:   {git_hash}")
    print(
        "[CFG] Using: "
        f"pts>={POINTS_THRESHOLD_DEFAULT}, hf_thr={HF_RATIO_THRESHOLD}  "
        f"cinematic_rescue: hf<{CINEMATIC_HF_THRESHOLD}, of>={CINEMATIC_MIN_OF_COUNT}, area<{CINEMATIC_MAX_AREA_RATIO}"
    )
    rivagan_init = initialize_invisible_watermark()
    if rivagan_init.get("rivaGan_ready", False):
        print("[INIT] rivaGAN initialized once before scan loop.")
    else:
        reason = rivagan_init.get("reason", "unknown")
        print(f"[INIT] rivaGAN unavailable: {reason}")

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
                c2pa_sig = detect_c2pa_signal(vp)

                raw_row = {
                    "category":         category,
                    "filename":         vp.name,
                    "ground_truth":     gt,
                    **sig,
                    **c2pa_sig,
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
                    of_upper_third_roi_ratio = sig["of_upper_third_roi_ratio"],
                    of_center_roi_ratio = sig["of_center_roi_ratio"],
                    of_wide_top_bottom_count = sig["of_wide_top_bottom_count"],
                    broadcast_scoreboard_trap = sig["broadcast_scoreboard_trap"],
                    broadcast_billboard_trap = sig["broadcast_billboard_trap"],
                    broadcast_pattern_trap = sig["broadcast_pattern_trap"],
                    broadcast_lower_third_pattern = sig["broadcast_lower_third_pattern"],
                    broadcast_scoreboard_pattern = sig["broadcast_scoreboard_pattern"],
                    broadcast_billboard_pattern = sig["broadcast_billboard_pattern"],
                    freq_hf_ratio_mean = sig["freq_hf_ratio_mean"],
                    c2pa_ai = c2pa_sig["c2pa_ai"],
                )
                # Twarda reguła #1 i #2 z tasku naprawczego:
                # 1) ai_specific=0 -> zawsze brak (chyba że rescue_guard_override)
                # 2) lower-third wykryty + ai_specific=0 -> zawsze brak
                rescue_guard_override = (
                    ai_specific == 0
                    and det == 1
                    and float(score) < float(POINTS_THRESHOLD_DEFAULT)
                    and int(sig.get("of_count", 0)) >= 1
                    and int(sig.get("of_wide_lower_roi_count", 0)) == 0
                    and float(sig.get("iw_best_similarity", 0.0)) >= 0.60
                    and float(sig.get("freq_hf_ratio_mean", 1.0)) < CLEAN_AI_HF_THRESHOLD
                    and int(sig.get("broadcast_scoreboard_trap", 0)) == 0
                    and int(sig.get("broadcast_billboard_trap", 0)) == 0
                    and int(sig.get("broadcast_pattern_trap", 0)) == 0
                )
                if ai_specific == 0 and not rescue_guard_override:
                    det = 0
                    mode = mode + ";guard_no_ai_specific=1"
                if rescue_guard_override:
                    mode = mode + ";guard_rescue_override=1"
                if sig.get("of_lower_third_roi_ratio", 0.0) > LOWER_THIRD_HARD_THRESHOLD and ai_specific == 0:
                    det = 0
                    mode = mode + ";guard_lowerthird_without_ai=1"
                high_score_override = 0
                if (
                    det == 0
                    and float(score) >= float(HIGH_SCORE_OVERRIDE_THRESHOLD)
                    and int(broadcast_trap) == 1
                ):
                    det = 1
                    high_score_override = 1
                    mode = mode + ";high_score_override=1"
                    print(
                        f"  ADV HIGH_SCORE_OVERRIDE: fusion_score={float(score):.3f}, "
                        "broadcast_trap=1 -> det=1"
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
                    "high_score_override": high_score_override,
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
                    "of_upper_third_roi_ratio": sig["of_upper_third_roi_ratio"],
                    "of_center_roi_ratio": sig["of_center_roi_ratio"],
                    "of_wide_top_bottom_count": sig["of_wide_top_bottom_count"],
                    "broadcast_scoreboard_trap": sig["broadcast_scoreboard_trap"],
                    "broadcast_billboard_trap": sig["broadcast_billboard_trap"],
                    "broadcast_pattern_trap": sig["broadcast_pattern_trap"],
                    "broadcast_lower_third_pattern": sig["broadcast_lower_third_pattern"],
                    "broadcast_scoreboard_pattern": sig["broadcast_scoreboard_pattern"],
                    "broadcast_billboard_pattern": sig["broadcast_billboard_pattern"],
                    "freq_hf_ratio_mean": sig["freq_hf_ratio_mean"],
                    "c2pa_found":        c2pa_sig["c2pa_found"],
                    "c2pa_ai":           c2pa_sig["c2pa_ai"],
                    "c2pa_generator":    c2pa_sig["c2pa_generator"],
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
        print("\n[SWEEP] Obliczam rozszerzony sweep heurystyk...")
        sweep = run_threshold_sweep(raw_rows)
        heuristic_rows = sweep["heuristic_rows"]
        activation_rows = sweep["activation_rows"]
        score_dist_rows = sweep["score_dist_rows"]
        best_row = sweep["best_row"]
        pareto_rows = sweep["pareto_rows"]

        threshold_sweep_path = snap_dir / "threshold_sweep.csv"
        heuristic_sweep_path = snap_dir / "heuristic_param_sweep.csv"
        activation_path = snap_dir / "feature_activation_summary.csv"
        score_dist_path = snap_dir / "score_distribution_by_split.csv"
        best_path = snap_dir / "best_config_selection.csv"
        pareto_path = snap_dir / "pareto_frontier.csv"

        with threshold_sweep_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(heuristic_rows[0].keys()))
            writer.writeheader()
            writer.writerows(heuristic_rows)
        shutil.copyfile(threshold_sweep_path, heuristic_sweep_path)
        with activation_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(activation_rows[0].keys()))
            writer.writeheader()
            writer.writerows(activation_rows)
        with score_dist_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(score_dist_rows[0].keys()))
            writer.writeheader()
            writer.writerows(score_dist_rows)
        with best_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(best_row[0].keys()))
            writer.writeheader()
            writer.writerows(best_row)
        with pareto_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pareto_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pareto_rows)

        print(f"[SWEEP]   {threshold_sweep_path}  ({len(heuristic_rows)} kombinacji)")
        print(f"[SWEEP]   {activation_path}")
        print(f"[SWEEP]   {score_dist_path}")
        print(f"[SWEEP]   {best_path}")
        print(f"[SWEEP]   {pareto_path}")

        print("\n--- Top-5 konfiguracji (FPR_adv_fp_trap ASC, TPR_aibaseline DESC) ---")
        for row in heuristic_rows[:5]:
            print(
                f"  hf={row['hf_ratio_threshold']} tex={row['low_texture_threshold']} pts={row['points_threshold']}  "
                f"FPR_fptrap={row['FPR_adv_fp_trap']}  "
                f"TPR_ai={row['TPR_aibaseline']}  F1={row['F1_global']}"
            )

    # —— Kopiuj do latest/ ——
    copy_to_latest(snap_dir)
    print(f"\n[LATEST] {RESULTS_BASE / 'latest'}")


if __name__ == "__main__":
    main()

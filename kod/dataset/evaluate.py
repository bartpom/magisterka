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
import os
import shutil
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from statistics import mean
import cv2

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from advanced_detectors import initialize_invisible_watermark, run_advanced_scan
try:
    from ai_style_clip_detector import AIStyleCLIPDetector  # type: ignore
except Exception:
    AIStyleCLIPDetector = None  # type: ignore
try:
    from flux_fft_detector import FluxFFTDetector  # type: ignore
except Exception:
    FluxFFTDetector = None  # type: ignore
try:
    from temporal_consistency_detector import TemporalConsistencyDetector  # type: ignore
except Exception:
    TemporalConsistencyDetector = None  # type: ignore
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
    AI_STYLE_CLIP_HIGH_CONF_THRESHOLD,
    AI_STYLE_SOFT_OVERRIDE_SCORE,
    LOWER_THIRD_QUALITY_PROB_THRESHOLD,
    TC_MIN_AI_STYLE_PROB,
    TC_ELIGIBLE_SCORES,
    TC_MAX_LOWER_THIRD_PROB,
    CINEMATIC_CLIP_MIN_PROB,
    CINEMATIC_RESCUE_MIN_CLIP_PROB,
    ZERO_CLIP_PENALTY_THRESHOLD,
    OF_LARGE_AREA_RATIO_GUARD,
    OF_UPPER_DOMINANCE_RATIO,
    OF_LOWER_MAX_FOR_UPPER_GUARD,
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
    "flux_found", "flux_similarity", "flux_similarity_std", "flux_method",
    "ai_style_prob", "ai_style_detected", "flux_fft_score", "fft_bonus", "flux_combined",
    "tc_score", "tc_detected", "tc_frame_diff_variance", "tc_of_smoothness", "tc_luminance_temporal_std", "tc_bonus",
    "fft_found", "fft_score", "freq_hf_ratio_mean",
    "c2pa_found", "c2pa_ai", "c2pa_generator", "c2pa_error",
    "frames_sampled", "duration_s", "analysis_time_s", "eval_mode", "detector_version",
]

EVAL_FIELDS = [
    "category", "filename", "ground_truth",
    "detected", "fusion_score", "fusion_mode", "ai_specific", "broadcast_trap", "high_score_override",
    "zv_count", "zv_lower_third_roi_count", "of_count", "of_max_area_ratio", "iw_best_similarity", "iw_matched",
    "flux_found", "flux_similarity", "flux_similarity_std", "flux_method",
    "ai_style_prob", "ai_style_detected", "flux_fft_score", "fft_bonus", "flux_combined",
    "tc_score", "tc_detected", "tc_frame_diff_variance", "tc_of_smoothness", "tc_luminance_temporal_std", "tc_bonus",
    "fft_score", "of_texture_variance_mean", "of_low_texture_roi_count",
    "of_wide_lower_roi_count", "of_corner_compact_roi_count", "of_lower_third_roi_ratio",
    "of_upper_third_roi_ratio", "of_center_roi_ratio", "of_wide_top_bottom_count",
    "broadcast_scoreboard_trap", "broadcast_billboard_trap",
    "broadcast_pattern_trap", "broadcast_lower_third_pattern", "broadcast_scoreboard_pattern", "broadcast_billboard_pattern",
    "freq_hf_ratio_mean", "c2pa_found", "c2pa_ai", "c2pa_generator", "duration_s",
    "analysis_time_s", "eval_mode",
]

DETECTOR_VERSION = "adv_v7_ai_style_clip_fft"

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
        shutil.rmtree(latest, ignore_errors=True)
        if latest.exists():
            # Fallback: usuń zawartość plik po pliku (Windows read-only / symlink)
            for child in latest.rglob("*"):
                try:
                    if child.is_file() or child.is_symlink():
                        child.unlink(missing_ok=True)
                except Exception:
                    pass
            try:
                latest.rmdir()
            except Exception:
                pass
    try:
        shutil.copytree(snap_dir, latest, dirs_exist_ok=True)
    except Exception as e:
        print(f"[WARN] copy_to_latest: {e}", file=sys.stderr)


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


def initialize_ai_style_clip_detector() -> tuple[Any, str]:
    if AIStyleCLIPDetector is None:
        return None, "AIStyleCLIPDetector import failed"
    try:
        detector = AIStyleCLIPDetector(model_path=DATASET_ROOT.parent.parent / "clip_classifier.pkl")
        if not getattr(detector, "enabled", False):
            reason = getattr(detector, "load_error", "detector_disabled")
            return None, reason
        return detector, "OK"
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def initialize_flux_fft_detector() -> tuple[Any, str]:
    if FluxFFTDetector is None:
        return None, "FluxFFTDetector import failed"
    try:
        detector = FluxFFTDetector(thresholds_path=DATASET_ROOT.parent.parent / "flux_fft_thresholds.json")
        return detector, "OK"
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def initialize_temporal_detector() -> tuple[Any, str]:
    if TemporalConsistencyDetector is None:
        return None, "TemporalConsistencyDetector import failed"
    try:
        detector = TemporalConsistencyDetector()
        return detector, "OK"
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


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
    flux_detected = int(row.get("flux_combined", row.get("flux_detected", 0))) == 1
    zv_count = int(row.get("zv_count", 0))
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
    if zv_count >= 1:
        score += 1
    if iw_strong:
        score += 2
    if c2pa_ai:
        score += 2
    if flux_detected:
        score += 3
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
) -> tuple[int, int, int]:
    iw_strong = bool(row.get("iw_matched")) and float(row.get("iw_similarity", 0.0)) >= 0.85
    flux_detected = int(row.get("flux_combined", row.get("flux_detected", 0))) == 1
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
    ai_specific = int(iw_strong or c2pa_ai or flux_detected or sum(ai_signals) >= 2)
    
    # Guard: ścieżka kinematyczna nie może być jedynym uzasadnieniem ai_specific,
    # jeśli CLIP nie daje żadnego sygnału AI (natural high-motion footage).
    clip_prob_local = float(row.get("ai_style_prob", 0.0))
    only_cinematic = (
        not iw_strong
        and not c2pa_ai
        and not flux_detected
        and int(row.get("of_corner_compact_roi_count", 0)) == 0
        and not (of_count >= 3 and area_ratio < max_area_ratio_threshold)
        and clip_prob_local < CINEMATIC_CLIP_MIN_PROB
    )
    if only_cinematic and ai_specific == 1:
        ai_specific = 0
    
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
    return ai_specific, broadcast_trap, int(only_cinematic)


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
    flux_detected: int = 0,
    flux_similarity: float = 0.0,
    flux_similarity_std: float = 0.0,
    ai_style_prob: float = 0.0,
    ai_style_detected: int = 0,
    flux_fft_score: int = 0,
    fft_bonus: int = 0,
    flux_combined: int = 0,
    tc_score: int = 0,
    tc_detected: int = 0,
    tc_bonus: int = 0,
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
        "flux_detected": flux_detected,
        "flux_similarity": flux_similarity,
        "flux_similarity_std": flux_similarity_std,
        "ai_style_prob": ai_style_prob,
        "ai_style_detected": ai_style_detected,
        "flux_fft_score": flux_fft_score,
        "fft_bonus": fft_bonus,
        "flux_combined": flux_combined,
        "tc_score": tc_score,
        "tc_detected": tc_detected,
        "tc_bonus": tc_bonus,
    }
    score = compute_ai_score(
        row,
        low_texture_threshold=low_texture_threshold,
        hf_ratio_threshold=hf_ratio_threshold,
        max_area_ratio_threshold=max_area_ratio_threshold,
    )
    ai_specific, broadcast_trap, only_cinematic_guard = compute_ai_flags(
        row,
        low_texture_threshold=low_texture_threshold,
        hf_ratio_threshold=hf_ratio_threshold,
        max_area_ratio_threshold=max_area_ratio_threshold,
    )
    score += int(max(0, fft_bonus))
    score += int(max(0, tc_bonus))
    clip_prob = float(ai_style_prob)

    tc_conditional_boost_applied = 0
    if (
        int(tc_detected) == 1
        and clip_prob >= float(TC_MIN_AI_STYLE_PROB)
        and int(score) in set(TC_ELIGIBLE_SCORES)
        and float(clip_prob) < float(TC_MAX_LOWER_THIRD_PROB)
    ):
        score += 1
        tc_conditional_boost_applied = 1

    lower_third_ok = not broadcast_trap
    # Fix D: lower-third quality gate (nie wzmacniaj sygnalu lower-third,
    # jesli brak AI-specific i CLIP ma niskie prawdopodobienstwo AI).
    lower_third_effective = int(
        lower_third_ok and (ai_specific == 1 or clip_prob > LOWER_THIRD_QUALITY_PROB_THRESHOLD)
    )

    # Fix A: broadcast_score_cap
    broadcast_score_cap_applied = 0
    if (
        int(row.get("broadcast_pattern_trap", 0)) == 1
        and int(c2pa_ai) == 0
        and int(flux_combined) == 0
        and clip_prob < 0.55
        and float(score) > 3.0
    ):
        score = 3
        broadcast_score_cap_applied = 1

    points_score = score

    # Fix B: optical_flow_penalty
    of_penalty_applied = 0
    if (
        int(row.get("of_count", 0)) > 20
        and int(ai_specific) == 0
        and int(c2pa_ai) == 0
        and clip_prob < 0.55
    ):
        points_score -= 2
        score = points_score
        of_penalty_applied = 1

    # Poprawka 3: ai_style_prob == 0 penalty
    # Gdy CLIP nie potwierdza AI (prob<ZERO_CLIP_PENALTY_THRESHOLD), flux nie aktywny i brak c2pa,
    # obniż score o 1 punkt aby zmniejszyć progi FP z wiadomością kinematyczną.
    # Nie stosuj gdy jest tc_bonus (temporal sygnał zazwyczaj potwierdzony).
    zero_clip_penalty_applied = 0
    if (
        clip_prob < float(ZERO_CLIP_PENALTY_THRESHOLD)
        and int(flux_combined) == 0
        and int(c2pa_ai) == 0
        and int(tc_bonus) == 0
        and ai_specific == 1
    ):
        points_score -= 1
        zero_clip_penalty_applied = 1

    # Poprawka B: dużą część OF w górnej części bez clip/c2pa/flux
    # Typowe dla sportowych overlayów vMix/scoreboard, ale bez lower-third pattern.
    guard_large_of_no_clip = 0
    if (
        float(row.get("of_max_area_ratio", 0.0)) >= float(OF_LARGE_AREA_RATIO_GUARD)
        and int(row.get("broadcast_lower_third_pattern", 0)) == 0
        and float(ai_style_prob) < 0.01
        and int(flux_combined) == 0
        and int(c2pa_ai) == 0
    ):
        points_score -= 1
        guard_large_of_no_clip = 1

    # Poprawka C: dominacja górnej tercji OF w naturalnym nagraniu korytarza/przejścia
    # Kings Cross, Broadcast Intro: duży statyczny sufit/ściany u góry, mało OF na dole.
    guard_upper_of_dominance = 0
    if (
        float(row.get("of_upper_third_roi_ratio", 0.0)) >= float(OF_UPPER_DOMINANCE_RATIO)
        and float(row.get("of_lower_third_roi_ratio", 0.0)) < float(OF_LOWER_MAX_FOR_UPPER_GUARD)
        and float(ai_style_prob) < 0.01
        and int(flux_combined) == 0
        and int(c2pa_ai) == 0
    ):
        points_score -= 1
        guard_upper_of_dominance = 1

    # Fix C: soft_threshold_tighten
    soft_threshold_min_score = 3
    if ai_specific == 0 and int(lower_third_effective) == 0 and clip_prob < 0.55:
        soft_threshold_min_score = 4
    elif clip_prob < 0.55 and int(c2pa_ai) == 0 and int(flux_combined) == 0:
        soft_threshold_min_score = 4

    soft_threshold_hit = (
        ai_specific == 1
        and int(broadcast_trap) == 0
        and int(row.get("of_count", 0)) >= 10
        and points_score >= soft_threshold_min_score
        and points_score < points_threshold
        and int(c2pa_ai) == 0
    )

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
        and (
            float(row.get("ai_style_prob", 0.0)) >= CINEMATIC_RESCUE_MIN_CLIP_PROB
            or int(row.get("of_corner_compact_roi_count", 0)) >= 1
        )
    )

    flux_soft_override = (
        int(flux_combined) == 1
        and float(points_score) >= float(AI_STYLE_SOFT_OVERRIDE_SCORE)
        and int(row.get("broadcast_pattern_trap", 0)) == 0
    )

    detected = int(
        soft_threshold_hit
        or points_score >= POINTS_THRESHOLD_DEFAULT
        or (ENABLE_CLEAN_AI_RESCUE and clean_ai_rescue_strict)
        or (ENABLE_CLEAN_AI_RESCUE and clean_ai_rescue_motion)
        or flux_soft_override
    )

    mode = (
        f"ai_score={score};ai_specific={int(ai_specific)};"
        f"lower_third_ok={int(lower_third_ok)};lower_third_effective={int(lower_third_effective)};"
        f"c2pa_ai={int(c2pa_ai)};"
        f"flux={int(flux_combined)};ai_style_prob={float(ai_style_prob):.2f};"
        f"flux_fft_score={int(flux_fft_score)};fft_bonus={int(fft_bonus)};"
        f"tc_score={int(tc_score)};tc_bonus={int(tc_bonus)}"
    )
    if soft_threshold_hit:
        mode += ";soft_threshold=1"
    if soft_threshold_min_score > 3:
        mode += f";soft_threshold_min={int(soft_threshold_min_score)}"
    if flux_soft_override:
        mode += ";flux_soft_override=1"
    if broadcast_score_cap_applied:
        mode += ";guard_broadcast_score_cap=1"
    if of_penalty_applied:
        mode += ";guard_of_penalty=1"
    if zero_clip_penalty_applied:
        mode += ";guard_zero_clip_penalty=1"
    if guard_large_of_no_clip:
        mode += ";guard_large_of_no_clip=1"
    if guard_upper_of_dominance:
        mode += ";guard_upper_of_dominance=1"
    if tc_conditional_boost_applied:
        mode += ";tc_conditional_boost=1"
    if only_cinematic_guard:
        mode += ";guard_cinematic_no_clip=1"
    if ENABLE_CLEAN_AI_RESCUE and clean_ai_rescue_motion and points_score < POINTS_THRESHOLD_DEFAULT:
        mode += ";rescue=cinematic_motion"
    if ENABLE_CLEAN_AI_RESCUE and clean_ai_rescue_strict and points_score < POINTS_THRESHOLD_DEFAULT:
        mode += ";rescue=clean_strict"

    return detected, float(score), mode, int(ai_specific), int(broadcast_trap)


# ───────────────────────────────────────────────────────────────────────
# Skanowanie wideo
# ───────────────────────────────────────────────────────────────────────

def scan_video(video_path: Path, mode: str = "standard") -> tuple[dict[str, Any], float, int | None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nie mozna otworzyc: {video_path}")
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fast = (mode == "fast")
    t0 = time.perf_counter()
    result = run_advanced_scan(
        cap, fps, total_frames,
        n_frames_median=3 if fast else 30,
        check_invisible=not fast,
        check_fft=True,
        check_optical_flow=not fast,
        of_scale=0.5,
        fast_mode=fast,
    )
    elapsed = time.perf_counter() - t0
    cap.release()
    # W trybie fast zwracamy również podwyższony próg decyzji (brak OF -> wyższe FPR)
    fast_threshold_override = 5 if fast else None
    return result, elapsed, fast_threshold_override


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
                        flux_detected=int(r.get("flux_found", 0)),
                        flux_similarity=float(r.get("flux_similarity", 0.0)),
                        flux_similarity_std=float(r.get("flux_similarity_std", 0.0)),
                        ai_style_prob=float(r.get("ai_style_prob", 0.0)),
                        ai_style_detected=int(r.get("ai_style_detected", 0)),
                        flux_fft_score=int(r.get("flux_fft_score", 0)),
                        fft_bonus=int(r.get("fft_bonus", 0)),
                        flux_combined=int(r.get("flux_combined", 0)),
                        tc_score=int(r.get("tc_score", 0)),
                        tc_detected=int(r.get("tc_detected", 0)),
                        tc_bonus=int(r.get("tc_bonus", 0)),
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
    parser = argparse.ArgumentParser(description="Benchmark detektora AI-wideo")
    parser.add_argument(
        "--mode", choices=["standard", "fast"], default="standard",
        help="standard: pełna analiza OF+ZV+FFT (30 klatek); fast: bez OF, 3 klatki, FFT na 10%%"
    )
    parser.add_argument(
        "--input", default=None, metavar="DIR",
        help="Katalog z plikami wideo do analizy (tryb standalone). "
             "Jeśli podać labels.csv w tym katalogu, obliczone zostaną recall/FPR."
    )
    parser.add_argument(
        "--output", default=None, metavar="CSV",
        help="Ścieżka do pliku CSV z wynikami trybu --input."
    )
    args = parser.parse_args()
    eval_mode = args.mode

    if args.input is not None:
        run_input_mode(args.input, args.output, eval_mode)
        return

    snap_dir  = make_snapshot_dir()
    git_hash  = get_git_hash()
    run_time  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[SNAP]  Snapshot: {snap_dir}")
    print(f"[SNAP]  Commit:   {git_hash}")
    print(f"[MODE]  Tryb analizy: {eval_mode}")
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
    ai_style_clip, ai_style_clip_reason = initialize_ai_style_clip_detector()
    if ai_style_clip is None:
        print(f"[INIT] AIStyleCLIPDetector unavailable: {ai_style_clip_reason}")
    else:
        clip_thr = float(getattr(ai_style_clip, "threshold", 0.5))
        print(f"[INIT] AIStyleCLIPDetector loaded (threshold={clip_thr:.2f}).")

    flux_fft_detector, flux_fft_reason = initialize_flux_fft_detector()
    if flux_fft_detector is None:
        print(f"[INIT] FluxFFTDetector unavailable: {flux_fft_reason}")
    else:
        print("[INIT] FluxFFTDetector loaded (bonus mode).")
    temporal_detector, temporal_reason = initialize_temporal_detector()
    if temporal_detector is None:
        print(f"[INIT] TemporalConsistencyDetector unavailable: {temporal_reason}")
    else:
        print("[INIT] TemporalConsistencyDetector loaded.")

    raw_rows:  list[dict] = []
    eval_rows: list[dict] = []
    total_videos = 0

    for category, (folder, gt) in CATEGORIES.items():
        videos = sorted(folder.glob("*.mp4"))
        if not videos:
            print(f"[WARN] Brak .mp4 w {folder}", file=sys.stderr)
            continue
        total = len(videos)
        print(f"\n=== {category} ({total} filmow) ===")
        total_videos += total

        for idx, vp in enumerate(videos, 1):
            print(f"  [SCAN] ({idx}/{total}) {vp.name} ... ", end="", flush=True)
            try:
                t_perf0 = time.perf_counter()
                result, elapsed, _fast_pts = scan_video(vp, mode=eval_mode)
                analysis_time_s = time.perf_counter() - t_perf0
                sig = extract_signals(result)
                c2pa_sig = detect_c2pa_signal(vp)
                if ai_style_clip is not None:
                    try:
                        clip_result = ai_style_clip.detect_video(vp)
                    except Exception as clip_err:  # noqa: BLE001
                        clip_result = {
                            "ai_style_prob": 0.0,
                            "ai_style_detected": False,
                            "ai_style_threshold": 0.5,
                            "ai_style_top_dims": [],
                            "error": str(clip_err),
                        }
                else:
                    clip_result = {
                        "ai_style_prob": 0.0,
                        "ai_style_detected": False,
                        "ai_style_threshold": 0.5,
                        "ai_style_top_dims": [],
                        "error": ai_style_clip_reason,
                    }

                if flux_fft_detector is not None:
                    try:
                        fft_flux = flux_fft_detector.detect_video(vp)
                    except Exception as fft_err:  # noqa: BLE001
                        fft_flux = {
                            "fft_score": 0,
                            "fft_bonus": 0,
                            "metrics": {},
                            "active_metrics": [],
                            "error": str(fft_err),
                        }
                else:
                    fft_flux = {
                        "fft_score": 0,
                        "fft_bonus": 0,
                        "metrics": {},
                        "active_metrics": [],
                        "error": flux_fft_reason,
                    }

                if temporal_detector is not None:
                    try:
                        tc_result = temporal_detector.detect_video(vp)
                    except Exception as tc_err:  # noqa: BLE001
                        tc_result = {
                            "tc_score": 0,
                            "tc_detected": False,
                            "frame_diff_variance": 0.0,
                            "of_smoothness": 0.0,
                            "luminance_temporal_std": 0.0,
                            "error": str(tc_err),
                        }
                else:
                    tc_result = {
                        "tc_score": 0,
                        "tc_detected": False,
                        "frame_diff_variance": 0.0,
                        "of_smoothness": 0.0,
                        "luminance_temporal_std": 0.0,
                        "error": temporal_reason,
                    }

                clip_prob = float(
                    clip_result.get("ai_style_prob", clip_result.get("clip_ai_prob", 0.0))
                )
                clip_detected = bool(
                    clip_result.get("ai_style_detected", clip_result.get("clip_detected", False))
                )
                fft_score_flux = int(fft_flux.get("fft_score", 0))
                fft_bonus = int(fft_flux.get("fft_bonus", 0))
                tc_score = int(tc_result.get("tc_score", 0))
                tc_detected = bool(tc_result.get("tc_detected", False))
                temporal_bonus = 0
                # S1: conservative temporal bonus (requires moderate CLIP confidence)
                # to avoid FP growth on broadcast-like real footage.
                if clip_prob >= 0.50:
                    if tc_detected:
                        temporal_bonus = 2
                    elif tc_score == 1:
                        temporal_bonus = 1
                flux_combined = bool(clip_prob > AI_STYLE_CLIP_HIGH_CONF_THRESHOLD)
                print(
                    "ADV AI-Style CLIP: "
                    f"prob={clip_prob:.2f} "
                    f"-> {'WYKRYTO' if flux_combined else 'brak'}"
                )
                if fft_bonus > 0:
                    print(
                        f"  ADV FFT bonus: +{fft_bonus} "
                        f"(metrics: {fft_flux.get('metrics', {})})"
                    )
                if temporal_bonus > 0:
                    print(
                        f"  TEMPORAL +{temporal_bonus}: tc_score={tc_score}, "
                        f"fdv={float(tc_result.get('frame_diff_variance', 0.0)):.4f}, "
                        f"of_smooth={float(tc_result.get('of_smoothness', 0.0)):.3f}, "
                        f"lum_std={float(tc_result.get('luminance_temporal_std', 0.0)):.3f}"
                    )

                raw_row = {
                    "category":         category,
                    "filename":         vp.name,
                    "ground_truth":     gt,
                    **sig,
                    **c2pa_sig,
                    "flux_found":       int(flux_combined),
                    "flux_similarity":  float(clip_prob),
                    "flux_similarity_std": 0.0,
                    "flux_method":      "clip+fft",
                    "ai_style_prob":    float(clip_prob),
                    "ai_style_detected": int(clip_detected),
                    "flux_fft_score":   int(fft_score_flux),
                    "fft_bonus":        int(fft_bonus),
                    "flux_combined":    int(flux_combined),
                    "tc_score":         int(tc_score),
                    "tc_detected":      int(tc_detected),
                    "tc_frame_diff_variance": float(tc_result.get("frame_diff_variance", 0.0)),
                    "tc_of_smoothness": float(tc_result.get("of_smoothness", 0.0)),
                    "tc_luminance_temporal_std": float(tc_result.get("luminance_temporal_std", 0.0)),
                    "tc_bonus":         int(temporal_bonus),
                    "frames_sampled":   3 if eval_mode == "fast" else 30,
                    "duration_s":       f"{elapsed:.2f}",
                    "analysis_time_s":  f"{analysis_time_s:.3f}",
                    "eval_mode":        eval_mode,
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
                    flux_detected = int(flux_combined),
                    flux_similarity = float(clip_prob),
                    flux_similarity_std = 0.0,
                    ai_style_prob = float(clip_prob),
                    ai_style_detected = int(clip_detected),
                    flux_fft_score = int(fft_score_flux),
                    fft_bonus = int(fft_bonus),
                    flux_combined = int(flux_combined),
                    tc_score = int(tc_score),
                    tc_detected = int(tc_detected),
                    tc_bonus = int(temporal_bonus),
                    points_threshold=_fast_pts if _fast_pts is not None else POINTS_THRESHOLD_DEFAULT,
                )
                c2pa_override = int(c2pa_sig.get("c2pa_ai", 0)) == 1
                if c2pa_override:
                    det = 1
                    mode = mode + ";c2pa_override=1"

                # Poprawka 2: kling_static_ai blokada billboard
                # Blokuje aktywację gdy detektor statycznych ZV się aktywuje,
                # ale film zawiera grafiki broadcastowe (lower third, billboard).
                # Chroni przed FP na nagraniach z overlayami, które mają ZV obszary.
                kling_static_ai = (
                    int(sig.get("zv_count", 0)) >= 2
                    and float(sig.get("iw_best_similarity", 0.0)) >= 0.40
                    and int(broadcast_trap) == 0
                    and int(sig.get("broadcast_lower_third_pattern", 0)) == 0
                    and int(sig.get("broadcast_billboard_pattern", 0)) == 0
                )
                if kling_static_ai:
                    det = 1
                    mode = mode + ";kling_static_ai=1"

                sora_static_override = (
                    int(sig.get("of_count", 0)) == 0
                    and int(sig.get("zv_count", 0)) == 0
                    and float(sig.get("freq_hf_ratio_mean", 1.0)) < 0.38
                )
                if sora_static_override:
                    det = 1
                    mode = mode + ";sora_static_override=1"

                guard_bypass = c2pa_override or kling_static_ai or sora_static_override
                rescue_guard_override = False
                if not guard_bypass:
                    # Twarda reguła #1 i #2 z tasku naprawczego:
                    # 1) ai_specific=0 -> zawsze brak
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
                    if (
                        sig.get("of_lower_third_roi_ratio", 0.0) > LOWER_THIRD_HARD_THRESHOLD
                        and ai_specific == 0
                    ):
                        det = 0
                        mode = mode + ";guard_lowerthird_without_ai=1"

                # Poprawka A: high_score_override blokada broadcast
                # Override działa tylko gdy nie ma żadnego trapu broadcastowego oraz
                # gdy istnieje sygnał ai_specific. Bez ai_specific wysokie score nie może
                # wygenerować detekcji.
                high_score_override = 0
                if (not c2pa_override) and (
                    det == 0
                    and float(score) >= float(HIGH_SCORE_OVERRIDE_THRESHOLD)
                    and ai_specific == 1
                    and int(sig.get("broadcast_scoreboard_trap", 0)) == 0
                    and int(sig.get("broadcast_billboard_trap", 0)) == 0
                    and int(sig.get("broadcast_pattern_trap", 0)) == 0
                ):
                    det = 1
                    high_score_override = 1
                    mode = mode + ";high_score_override=1"
                    print(
                        f"  ADV HIGH_SCORE_OVERRIDE: fusion_score={float(score):.3f}, "
                        "broadcast traps absent -> det=1"
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
                    "flux_found":         int(flux_combined),
                    "flux_similarity":    float(clip_prob),
                    "flux_similarity_std": 0.0,
                    "flux_method":        "clip+fft",
                    "ai_style_prob":      float(clip_prob),
                    "ai_style_detected":  int(clip_detected),
                    "flux_fft_score":     int(fft_score_flux),
                    "fft_bonus":          int(fft_bonus),
                    "flux_combined":      int(flux_combined),
                    "tc_score":           int(tc_score),
                    "tc_detected":        int(tc_detected),
                    "tc_frame_diff_variance": float(tc_result.get("frame_diff_variance", 0.0)),
                    "tc_of_smoothness":   float(tc_result.get("of_smoothness", 0.0)),
                    "tc_luminance_temporal_std": float(tc_result.get("luminance_temporal_std", 0.0)),
                    "tc_bonus":           int(temporal_bonus),
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
                    "analysis_time_s":    f"{analysis_time_s:.3f}",
                    "eval_mode":          eval_mode,
                }
                eval_rows.append(eval_row)

                if ";guard_broadcast_score_cap=1" in mode:
                    print("  GUARD broadcast_score_cap applied: score capped at 3")
                if ";guard_of_penalty=1" in mode:
                    print(
                        f"  GUARD of_penalty: -2 (contours={int(sig.get('of_count', 0))}, "
                        "no ai_specific)"
                    )
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
        f.write(f"eval_mode:    {eval_mode}\n")

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
    if raw_rows and os.getenv("EVAL_SKIP_SWEEP", "0") != "1":
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
    elif raw_rows:
        print("\n[SWEEP] Pominięty (EVAL_SKIP_SWEEP=1).")

    # —— Kopiuj do latest/ ——
    copy_to_latest(snap_dir)
    print(f"\n[LATEST] {RESULTS_BASE / 'latest'}")


def run_input_mode(input_dir: str, output_csv: str | None, mode: str = "standard") -> None:
    """
    Tryb standalone: iteruje po plikach wideo w input_dir,
    uruchamia detektor, zapisuje wyniki do CSV.
    Jeśli w input_dir istnieje labels.csv z kolumnami filename,ground_truth,
    oblicza recall/FPR i wypisuje summary.
    """
    VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"[ERR] --input '{input_dir}' nie jest katalogiem.", file=sys.stderr)
        sys.exit(1)

    videos = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"[WARN] Brak plików wideo w: {input_path}", file=sys.stderr)
        return

    # Wczytaj opcjonalne etykiety ground truth
    labels: dict[str, int] = {}
    labels_file = input_path / "labels.csv"
    if labels_file.exists():
        with labels_file.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["filename"]] = int(row["ground_truth"])
        print(f"[LABELS] Wczytano {len(labels)} etykiet z {labels_file}")

    out_path = Path(output_csv) if output_csv else input_path / f"results_{mode}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rivagan_init = initialize_invisible_watermark()
    if not rivagan_init.get("rivaGan_ready"):
        print(f"[INIT] rivaGAN niedostępny: {rivagan_init.get('reason', '?')}")

    rows: list[dict] = []
    total = len(videos)
    times: list[float] = []

    print(f"[INPUT] {total} plików w '{input_path}' | tryb={mode} | output={out_path}")
    print("-" * 70)

    for idx, vp in enumerate(videos, 1):
        print(f"  [{idx:3d}/{total}] {vp.name} ... ", end="", flush=True)
        try:
            result, elapsed, _fast_pts = scan_video(vp, mode=mode)
            sig = extract_signals(result)
            c2pa_sig = detect_c2pa_signal(vp)

            det, score, fuse_mode, ai_specific, broadcast_trap = fuse(
                zv_count=sig["zv_count"],
                zv_lower_third_roi_count=sig["zv_lower_third_roi_count"],
                of_count=sig["of_count"],
                of_max_area=sig["of_max_area"],
                of_max_area_ratio=sig["of_max_area_ratio"],
                iw_similarity=sig["iw_best_similarity"],
                iw_matched=sig["iw_matched"],
                fft_score=sig["fft_score"],
                of_texture_variance_mean=sig["of_texture_variance_mean"],
                of_low_texture_roi_count=sig["of_low_texture_roi_count"],
                of_wide_lower_roi_count=sig["of_wide_lower_roi_count"],
                of_corner_compact_roi_count=sig["of_corner_compact_roi_count"],
                of_lower_third_roi_ratio=sig["of_lower_third_roi_ratio"],
                of_upper_third_roi_ratio=sig["of_upper_third_roi_ratio"],
                of_center_roi_ratio=sig["of_center_roi_ratio"],
                of_wide_top_bottom_count=sig["of_wide_top_bottom_count"],
                broadcast_scoreboard_trap=sig["broadcast_scoreboard_trap"],
                broadcast_billboard_trap=sig["broadcast_billboard_trap"],
                broadcast_pattern_trap=sig["broadcast_pattern_trap"],
                broadcast_lower_third_pattern=sig["broadcast_lower_third_pattern"],
                broadcast_scoreboard_pattern=sig["broadcast_scoreboard_pattern"],
                broadcast_billboard_pattern=sig["broadcast_billboard_pattern"],
                freq_hf_ratio_mean=sig["freq_hf_ratio_mean"],
                c2pa_ai=c2pa_sig["c2pa_ai"],
                flux_detected=0,
                flux_similarity=0.0,
                flux_similarity_std=0.0,
                ai_style_prob=0.0,
                ai_style_detected=0,
                flux_fft_score=0,
                fft_bonus=0,
                flux_combined=0,
                tc_score=0,
                tc_detected=0,
                tc_bonus=0,
                points_threshold=_fast_pts if _fast_pts is not None else POINTS_THRESHOLD_DEFAULT,
            )

            label = "AI" if det == 1 else "NOT_AI"
            times.append(elapsed)
            rows.append({
                "filename": vp.name,
                "predicted_label": label,
                "fusion_score": f"{score:.3f}",
                "analysis_time_s": f"{elapsed:.3f}",
                "mode": mode,
            })
            gt_info = ""
            if vp.name in labels:
                gt = labels[vp.name]
                correct = (det == gt)
                gt_info = f"  gt={gt} {'OK' if correct else 'ERR'}"
            print(f"{label}  score={score:.3f}  ({elapsed:.1f}s){gt_info}")

        except Exception as exc:  # noqa: BLE001
            print(f"BLAD: {exc}", file=sys.stderr)
            rows.append({"filename": vp.name, "predicted_label": "ERROR",
                         "fusion_score": "", "analysis_time_s": "", "mode": mode})

    # Zapisz CSV
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "predicted_label",
                                               "fusion_score", "analysis_time_s", "mode"])
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 70)
    print(f"[OUT]  Zapisano: {out_path}")
    print(f"[SUMMARY] Pliki: {total}  Przeanalizowane: {len(times)}")
    if times:
        print(f"[SUMMARY] Czas analizy: średn.={sum(times)/len(times):.2f}s  "
              f"min={min(times):.2f}s  max={max(times):.2f}s")

    # Oblicz recall/FPR jeśli dostępne etykiety
    if labels:
        matched = [(r, labels[r["filename"]]) for r in rows
                   if r["filename"] in labels and r["predicted_label"] in ("AI", "NOT_AI")]
        if matched:
            tp = sum(1 for r, gt in matched if r["predicted_label"] == "AI" and gt == 1)
            fn = sum(1 for r, gt in matched if r["predicted_label"] == "NOT_AI" and gt == 1)
            fp = sum(1 for r, gt in matched if r["predicted_label"] == "AI" and gt == 0)
            tn = sum(1 for r, gt in matched if r["predicted_label"] == "NOT_AI" and gt == 0)
            recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            fpr    = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
            print(f"[SUMMARY] TP={tp} FN={fn} FP={fp} TN={tn}")
            print(f"[SUMMARY] Recall={recall:.3f}  FPR={fpr:.3f}")



if __name__ == "__main__":
    main()

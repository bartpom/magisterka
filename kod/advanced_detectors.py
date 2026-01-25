# ===============================
# FINAL – High-Precision Deepfake Detector Core
# ===============================

from __future__ import annotations
import os, math, cv2, threading, contextlib
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import config

# -------------------------------
# Utilities
# -------------------------------

def clamp(x: float, lo=0.0, hi=100.0) -> float:
    return float(max(lo, min(hi, x)))

def safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

# -------------------------------
# Robust aggregation
# -------------------------------

def robust_agg(vals: List[float], mode="p90") -> Optional[float]:
    arr = np.asarray([v for v in vals if safe_float(v) is not None], dtype=np.float32)
    if arr.size == 0:
        return None

    if mode == "mean":
        return float(arr.mean())
    if mode == "median":
        return float(np.median(arr))
    if mode.startswith("p"):
        return float(np.percentile(arr, int(mode[1:])))
    return float(np.percentile(arr, 90))

# -------------------------------
# Video score fusion (VideoMAE + D3)
# -------------------------------

def fuse_video_scores(
    vm: Optional[float],
    d3: Optional[float],
    policy: str
) -> Optional[float]:

    if vm is None and d3 is None:
        return None

    if policy == "high_recall":
        return max(v for v in [vm, d3] if v is not None)

    # high_precision / balanced
    if vm is not None and d3 is not None:
        return 0.9 * vm + 0.1 * d3

    if vm is not None:
        return vm

    # D3 only: trust only if extremely high
    return d3 if d3 is not None and d3 >= 85.0 else None

# -------------------------------
# Final fusion
# -------------------------------

def fuse_scores(features: Dict[str, Any]) -> float:
    weights = config.FUSE_WEIGHTS
    acc = 0.0
    wsum = 0.0

    for k, w in weights.items():
        v = safe_float(features.get(k))
        if v is None or w <= 0:
            continue
        acc += v * w
        wsum += w

    return clamp(acc / wsum) if wsum > 0 else 0.0

# -------------------------------
# Verdict logic (CORE)
# -------------------------------

def decision_policy(
    score: float,
    details: Dict[str, Any],
    policy: str
) -> str:

    REAL_MAX = config.REAL_MAX
    FAKE_MIN = config.FAKE_MIN
    eps = 0.01

    face = safe_float(details.get("ai_face_score"))
    video = safe_float(details.get("ai_video_score"))
    wm = safe_float(details.get("watermark_score"))
    forensic = safe_float(details.get("forensic_score"))
    ratio = safe_float(details.get("fake_ratio"))

    # -----------------
    # FAKE corroboration
    # -----------------
    primary = any([
        video is not None and video >= 90,
        wm is not None and wm >= 85,
        forensic is not None and forensic >= 85,
    ])

    secondary = any([
        ratio is not None and ratio >= 40,
        face is not None and face >= 85,
    ])

    corroborated = primary and secondary

    # -----------------
    # REAL gate
    # -----------------
    suspect = any([
        face is not None and face >= 50,
        video is not None and video >= 50,
        wm is not None and wm >= 60,
    ])

    if policy == "high_precision":
        if score >= FAKE_MIN and not corroborated:
            score = FAKE_MIN - eps
        if score <= REAL_MAX and suspect:
            score = REAL_MAX + eps

    elif policy == "high_recall":
        if score <= REAL_MAX:
            score = REAL_MAX + eps

    # -----------------
    # Verdict
    # -----------------
    if score <= REAL_MAX:
        return "REAL (PRAWDOPODOBNE)"
    if score >= FAKE_MIN:
        return "FAKE (PRAWDOPODOBNE)"
    return "GREY ZONE (NIEPEWNE)"

# -------------------------------
# MAIN ENTRYPOINT
# -------------------------------

def scan_for_deepfake(
    input_path: str,
    *,
    decision_policy_name: Optional[str] = None,
    do_face_ai=True,
    do_forensic=False,
    do_watermark=False,
) -> Tuple[str, float, float, Dict[str, Any]]:

    policy = (decision_policy_name or config.DECISION_POLICY).lower()

    # --- tutaj zakładamy, że wcześniej policzyłeś ---
    # ai_face_score, ai_scene_score, ai_video_score, fake_ratio itd.

    details: Dict[str, Any] = {}

    # (tu wklejasz swoje istniejące obliczenia modeli)

    final_score = fuse_scores({
        "face": details.get("ai_face_score"),
        "scene": details.get("ai_scene_score"),
        "video": details.get("ai_video_score"),
    })

    verdict = decision_policy(final_score, details, policy)

    details["final_score"] = final_score
    details["verdict"] = verdict

    return verdict, final_score, details.get("fake_ratio", 0.0), details

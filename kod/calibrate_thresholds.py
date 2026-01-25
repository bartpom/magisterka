# calibrate_thresholds.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def label_from_filename(path: str) -> Optional[int]:
    """
    Zwraca etykietę na podstawie nazwy pliku:
      *_fake* -> 1
      *_real* -> 0
    """
    name = os.path.basename(path).lower()
    if "_fake" in name:
        return 1
    if "_real" in name:
        return 0
    return None


def _verdict_from_score(score: float, real_max: float, fake_min: float) -> str:
    if score >= fake_min:
        return "FAKE (PRAWDOPODOBNE)"
    if score <= real_max:
        return "REAL (PRAWDOPODOBNE)"
    return "NIEPEWNE / GREY ZONE"


@dataclass
class Thresholds:
    real_max: float
    fake_min: float

    def to_dict(self) -> Dict[str, float]:
        return {"REAL_MAX": float(self.real_max), "FAKE_MIN": float(self.fake_min)}


def _best_thresholds_from_scores(real_scores: List[float], fake_scores: List[float]) -> Thresholds:
    """
    Dobiera (REAL_MAX, FAKE_MIN) tak, żeby:
    - REAL_MAX = wysoki percentyl wyników real (np. 95%)
    - FAKE_MIN = niski percentyl wyników fake (np. 5%)
    To daje szeroką "grey zone" tam, gdzie rozkłady się nakładają, ale minimalizuje FP/FN.
    """
    if not real_scores or not fake_scores:
        return Thresholds(real_max=30.0, fake_min=70.0)

    r = np.array(real_scores, dtype=float)
    f = np.array(fake_scores, dtype=float)

    real_max = float(np.percentile(r, 95))
    fake_min = float(np.percentile(f, 5))

    # jeżeli progi się "przecinają" (nakładanie), ustaw wide grey-zone zamiast psuć decyzje
    if fake_min <= real_max:
        mid = float((fake_min + real_max) / 2.0)
        real_max = max(5.0, min(45.0, mid - 5.0))
        fake_min = min(95.0, max(55.0, mid + 5.0))

    # sanity clamp
    real_max = max(0.0, min(49.0, real_max))
    fake_min = max(51.0, min(100.0, fake_min))

    return Thresholds(real_max=real_max, fake_min=fake_min)


def compute_thresholds_from_details(details_list: List[Dict[str, Any]]) -> Dict[str, Thresholds]:
    """
    details_list: lista dictów z gui (po normalize_details), musi mieć:
      - full_path
      - ai_final_score
      - deepfake_final_score
    """
    real_ai: List[float] = []
    fake_ai: List[float] = []
    real_df: List[float] = []
    fake_df: List[float] = []

    for d in details_list:
        p = d.get("full_path", "")
        y = label_from_filename(p)
        if y is None:
            continue

        ai_s = _safe_float(d.get("ai_final_score"))
        df_s = _safe_float(d.get("deepfake_final_score"))

        if ai_s is not None:
            (fake_ai if y == 1 else real_ai).append(ai_s)

        if df_s is not None:
            (fake_df if y == 1 else real_df).append(df_s)

    thr_ai = _best_thresholds_from_scores(real_ai, fake_ai)
    thr_df = _best_thresholds_from_scores(real_df, fake_df)

    return {"ai_detector": thr_ai, "deepfake_detector": thr_df}


def save_thresholds(path: str, thresholds: Dict[str, Thresholds]) -> None:
    payload = {
        "ai_detector": thresholds["ai_detector"].to_dict(),
        "deepfake_detector": thresholds["deepfake_detector"].to_dict(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_thresholds(path: str) -> Optional[Dict[str, Thresholds]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return {
            "ai_detector": Thresholds(
                real_max=float(payload["ai_detector"]["REAL_MAX"]),
                fake_min=float(payload["ai_detector"]["FAKE_MIN"]),
            ),
            "deepfake_detector": Thresholds(
                real_max=float(payload["deepfake_detector"]["REAL_MAX"]),
                fake_min=float(payload["deepfake_detector"]["FAKE_MIN"]),
            ),
        }
    except Exception:
        return None


def verdict_for(detector_name: str, score: float, thresholds: Dict[str, Thresholds]) -> str:
    thr = thresholds.get(detector_name)
    if not thr:
        return "NIEPEWNE / BRAK PROGÓW"
    return _verdict_from_score(score, thr.real_max, thr.fake_min)

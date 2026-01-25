#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _get_first_non_none(d: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
            return _to_float(d[k])
    return None


@dataclass
class MetaFusion:
    pipeline: Any
    features: Sequence[str]
    real_cut: float
    fake_cut: float

    @staticmethod
    def load(path: str) -> Optional["MetaFusion"]:
        try:
            import joblib
            payload = joblib.load(path)
            thr = payload.get("thresholds", {})
            return MetaFusion(
                pipeline=payload["pipeline"],
                features=payload["features"],
                real_cut=float(thr.get("real_cut", 0.33)),
                fake_cut=float(thr.get("fake_cut", 0.66)),
            )
        except Exception:
            return None

    def _vector_from_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        # Mapuj różne możliwe klucze z Twojego pipeline’u na spójne kolumny.
        # Wszystko w procentach/liczbach jak w raporcie (0..100 dla ai_*).
        mapped: Dict[str, Any] = {}

        mapped["ai_face"] = _get_first_non_none(details, ["ai_face_score", "ai_face_raw", "ai_face"])
        mapped["ai_scene"] = _get_first_non_none(details, ["ai_scene_score", "ai_scene_raw", "ai_scene"])
        mapped["ai_video"] = _get_first_non_none(details, ["ai_video_score", "ai_video_raw", "ai_video"])
        mapped["jitter_px"] = _get_first_non_none(details, ["jitter_px", "jitter_score"])
        mapped["blink_per_min"] = _get_first_non_none(details, ["blink_per_min", "blinks_per_min"])
        mapped["ela"] = _get_first_non_none(details, ["ela_score", "ela"])
        mapped["fft"] = _get_first_non_none(details, ["fft_score", "fft"])
        mapped["border"] = _get_first_non_none(details, ["border_artifacts", "border_score", "border"])
        mapped["sharp"] = _get_first_non_none(details, ["face_sharpness", "sharp_face", "sharp"])

        # Upewnij się, że masz wszystkie cechy (braki zostaną zaimputowane przez pipeline).
        for f in self.features:
            mapped.setdefault(f, None)

        return mapped

    def predict_proba_fake(self, details: Dict[str, Any]) -> Optional[float]:
        vec = self._vector_from_details(details)

        # pipeline oczekuje DataFrame-like; najprościej podać dict w liście.
        try:
            import pandas as pd
            X = pd.DataFrame([vec], columns=list(self.features))
            p = float(self.pipeline.predict_proba(X)[0, 1])
            if p < 0.0:
                p = 0.0
            if p > 1.0:
                p = 1.0
            return p
        except Exception:
            return None

    def verdict_from_proba(self, p_fake: float) -> str:
        if p_fake >= self.fake_cut:
            return "FAKE (PRAWDOPODOBNE)"
        if p_fake <= self.real_cut:
            return "REAL (PRAWDOPODOBNE)"
        return "NIEPEWNE / GREY ZONE"

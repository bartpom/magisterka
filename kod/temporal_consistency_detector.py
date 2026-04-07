#!/usr/bin/env python3
# Created: 2026-03-31
# Part of: AI Video Detector v2 (CLIP+FFT integration)
# Status: ACTIVE

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class TemporalConsistencyDetector:
    def __init__(
        self,
        frame_diff_variance_thr: float = 0.15,
        of_smoothness_thr: float = 3.0,
        luminance_temporal_std_thr: float = 8.0,
        max_frames: int = 90,
        stride: int = 3,
        resize_w: int = 320,
        resize_h: int = 180,
    ) -> None:
        self.frame_diff_variance_thr = float(frame_diff_variance_thr)
        self.of_smoothness_thr = float(of_smoothness_thr)
        self.luminance_temporal_std_thr = float(luminance_temporal_std_thr)
        self.max_frames = int(max_frames)
        self.stride = int(stride)
        self.resize_w = int(resize_w)
        self.resize_h = int(resize_h)

    def _sample_gray_frames(self, path: str | Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return []
        frames: list[np.ndarray] = []
        idx = 0
        while idx < self.max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx % self.stride == 0:
                fr = cv2.resize(frame, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
            idx += 1
        cap.release()
        return frames

    def detect_video(self, path: str | Path) -> dict[str, Any]:
        frames = self._sample_gray_frames(path)
        if len(frames) < 2:
            return {
                "tc_score": 0,
                "tc_detected": False,
                "frame_diff_variance": 0.0,
                "of_smoothness": 0.0,
                "luminance_temporal_std": 0.0,
                "thresholds_used": {
                    "frame_diff_variance": self.frame_diff_variance_thr,
                    "of_smoothness": self.of_smoothness_thr,
                    "luminance_temporal_std": self.luminance_temporal_std_thr,
                },
            }

        # 1) frame_diff_variance
        diffs = []
        for i in range(1, len(frames)):
            d = cv2.absdiff(frames[i], frames[i - 1])
            diffs.append(float(np.mean(d) / 255.0))
        frame_diff_variance = float(np.var(diffs)) if diffs else 0.0

        # 2) optical_flow_smoothness
        pair_idx = np.linspace(0, max(0, len(frames) - 2), num=min(10, len(frames) - 1), dtype=int)
        ratios = []
        for i in pair_idx:
            prev = frames[i]
            cur = frames[i + 1]
            flow = cv2.calcOpticalFlowFarneback(
                prev,
                cur,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mean_mag = float(np.mean(mag))
            std_mag = float(np.std(mag))
            ratio = std_mag / max(mean_mag, 1e-6)
            ratios.append(float(ratio))
        of_smoothness = float(np.mean(ratios)) if ratios else 0.0

        # 3) luminance_temporal_std
        lum = [float(np.mean(f)) for f in frames]
        luminance_temporal_std = float(np.std(lum)) if lum else 0.0

        tc_score = 0
        if frame_diff_variance < self.frame_diff_variance_thr:
            tc_score += 1
        if of_smoothness > self.of_smoothness_thr:
            tc_score += 1
        if luminance_temporal_std < self.luminance_temporal_std_thr:
            tc_score += 1

        return {
            "tc_score": int(tc_score),
            "tc_detected": bool(tc_score >= 2),
            "frame_diff_variance": frame_diff_variance,
            "of_smoothness": of_smoothness,
            "luminance_temporal_std": luminance_temporal_std,
            "thresholds_used": {
                "frame_diff_variance": self.frame_diff_variance_thr,
                "of_smoothness": self.of_smoothness_thr,
                "luminance_temporal_std": self.luminance_temporal_std_thr,
            },
        }


# Created: 2026-03-31
# Part of: AI Video Detector v2 (CLIP+FFT integration)
# Status: ACTIVE

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def _sample_frame_indices(total_frames: int, n_samples: int) -> np.ndarray:
    if total_frames <= 1:
        return np.zeros(n_samples, dtype=int)
    return np.linspace(0, total_frames - 1, num=n_samples, dtype=int)


def _oversmoothing_ratio(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    h, w = gray.shape
    x0, x1 = int(0.30 * w), int(0.70 * w)
    y0, y1 = int(0.30 * h), int(0.70 * h)
    center = grad[y0:y1, x0:x1]

    border_mask = np.ones_like(gray, dtype=bool)
    border_mask[y0:y1, x0:x1] = False
    border = grad[border_mask]

    center_mean = float(np.mean(center)) if center.size else 1e-6
    border_mean = float(np.mean(border)) if border.size else 0.0
    return float(border_mean / max(center_mean, 1e-6))


def _bimodality_coeff(frame_bgr: np.ndarray) -> float:
    from scipy.stats import kurtosis, skew  # local import to keep module lightweight

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32).reshape(-1)
    if gray.size < 8:
        return 0.0
    s = float(skew(gray, bias=False))
    k = float(kurtosis(gray, fisher=False, bias=False))
    if (not np.isfinite(s)) or (not np.isfinite(k)) or abs(k) < 1e-6:
        return 0.0
    bc = float((s * s + 1.0) / k)
    if not np.isfinite(bc):
        return 0.0
    return bc


def _hf_noise_ratio_dark(frame_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    dark_mask = v < 50.0
    if np.count_nonzero(dark_mask) < 64:
        return 0.0

    signal = np.zeros_like(v, dtype=np.float32)
    signal[dark_mask] = v[dark_mask]

    fft = np.fft.fft2(signal)
    fft_shift = np.fft.fftshift(fft)
    mag = np.abs(fft_shift)

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = max(1e-6, 0.5 * min(h, w))
    r = dist / max_radius

    band_mask = (r >= 0.30) & (r <= 0.50)
    low_mask = r < 0.30
    band_energy = float(np.mean(mag[band_mask])) if np.any(band_mask) else 0.0
    low_energy = float(np.mean(mag[low_mask])) if np.any(low_mask) else 1e-6
    return float(band_energy / max(low_energy, 1e-6))


def _ssim_variance(frames_bgr: list[np.ndarray]) -> float:
    if len(frames_bgr) < 2:
        return 0.0
    vals: list[float] = []
    prev = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    for frm in frames_bgr[1:]:
        cur = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        try:
            ssim = structural_similarity(prev, cur, data_range=255)
        except Exception:
            ssim = 0.0
        vals.append(float(ssim))
        prev = cur
    return float(np.var(vals)) if vals else 0.0


def compute_video_flux_fft_metrics(video_path: str | Path, n_frames: int = 8) -> dict[str, float]:
    vp = Path(video_path)
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        return {
            "oversmoothing_ratio": 0.0,
            "bimodality_coeff": 0.0,
            "hf_noise_ratio": 0.0,
            "ssim_variance": 0.0,
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = _sample_frame_indices(total_frames, n_frames)
    frames: list[np.ndarray] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()

    if not frames:
        return {
            "oversmoothing_ratio": 0.0,
            "bimodality_coeff": 0.0,
            "hf_noise_ratio": 0.0,
            "ssim_variance": 0.0,
        }

    overs = [float(np.nan_to_num(_oversmoothing_ratio(f), nan=0.0, posinf=0.0, neginf=0.0)) for f in frames]
    bimod = [float(np.nan_to_num(_bimodality_coeff(f), nan=0.0, posinf=0.0, neginf=0.0)) for f in frames]
    hf = [float(np.nan_to_num(_hf_noise_ratio_dark(f), nan=0.0, posinf=0.0, neginf=0.0)) for f in frames]
    ssim_var = _ssim_variance(frames)
    return {
        "oversmoothing_ratio": float(np.nan_to_num(np.mean(overs), nan=0.0, posinf=0.0, neginf=0.0)),
        "bimodality_coeff": float(np.nan_to_num(np.mean(bimod), nan=0.0, posinf=0.0, neginf=0.0)),
        "hf_noise_ratio": float(np.nan_to_num(np.mean(hf), nan=0.0, posinf=0.0, neginf=0.0)),
        "ssim_variance": float(np.nan_to_num(ssim_var, nan=0.0, posinf=0.0, neginf=0.0)),
    }


class FluxFFTDetector:
    """
    FFT component acts as soft bonus (+1 per active metric, max +3).
    Not used as hard gate — see debug_fft_scores.csv for AUC analysis.
    """

    def __init__(
        self,
        thresholds_path: str | Path = "flux_fft_thresholds.json",
        thresholds: dict[str, Any] | None = None,
        n_frames: int = 8,
    ) -> None:
        self.n_frames = int(n_frames)
        self.thresholds_path = self._resolve_thresholds_path(thresholds_path)
        self.metrics_cfg: dict[str, dict[str, Any]] = {
            "oversmoothing_ratio": {"threshold": 0.60, "direction": "le", "enabled": True},
            "bimodality_coeff": {"threshold": 0.55, "direction": "ge", "enabled": True},
            "ssim_variance": {"threshold": 0.02, "direction": "le", "enabled": True},
        }
        if thresholds is not None:
            self._load_from_dict(thresholds)
        elif self.thresholds_path.exists():
            try:
                payload = json.loads(self.thresholds_path.read_text(encoding="utf-8"))
                self._load_from_dict(payload)
            except Exception:
                pass

    @staticmethod
    def _resolve_thresholds_path(thresholds_path: str | Path) -> Path:
        p = Path(thresholds_path)
        if p.exists():
            return p
        here = Path(__file__).resolve().parent
        candidates = [here.parent / str(thresholds_path), here / str(thresholds_path)]
        for c in candidates:
            if c.exists():
                return c
        return p

    def _load_from_dict(self, payload: dict[str, Any]) -> None:
        metrics = payload.get("metrics", payload)
        for name in self.metrics_cfg:
            if name in metrics:
                cfg = metrics[name]
                self.metrics_cfg[name]["threshold"] = float(cfg.get("threshold", self.metrics_cfg[name]["threshold"]))
                self.metrics_cfg[name]["direction"] = str(cfg.get("direction", self.metrics_cfg[name]["direction"]))
                self.metrics_cfg[name]["enabled"] = bool(cfg.get("enabled", self.metrics_cfg[name]["enabled"]))

    def _trigger(self, metric_name: str, metric_value: float) -> bool:
        cfg = self.metrics_cfg.get(metric_name, {})
        if not cfg or not bool(cfg.get("enabled", True)):
            return False
        thr = float(cfg.get("threshold", 0.0))
        direction = str(cfg.get("direction", "ge")).lower()
        if direction == "le":
            return metric_value <= thr
        return metric_value >= thr

    def detect_video(self, video_path: str | Path) -> dict[str, Any]:
        metrics = compute_video_flux_fft_metrics(video_path, n_frames=self.n_frames)
        active: list[str] = []
        for name, value in metrics.items():
            if self._trigger(name, float(value)):
                active.append(name)
        fft_score = len(active)
        return {
            "fft_score": int(fft_score),
            "fft_bonus": int(fft_score),
            "metrics": metrics,
            "active_metrics": active,
        }

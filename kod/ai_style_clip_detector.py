# Created: 2026-03-31
# Part of: AI Video Detector v2 (CLIP+FFT integration)
# Status: ACTIVE

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def _sample_frame_indices(total_frames: int, n_samples: int) -> np.ndarray:
    if total_frames <= 1:
        return np.zeros(n_samples, dtype=int)
    return np.linspace(0, total_frames - 1, num=n_samples, dtype=int)


class AIStyleCLIPDetector:
    """
    Detects AI-generated video style via CLIP embeddings.
    Originally trained to detect Flux.1 artifacts, generalizes to
    broader AI video style detection.
    """

    def __init__(
        self,
        model_path: str | Path = "clip_classifier.pkl",
        clip_model_name: str = "openai/clip-vit-base-patch32",
        n_frames: int = 8,
    ) -> None:
        self.n_frames = int(n_frames)
        self.clip_model_name = clip_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Any = None
        self.threshold: float = 0.5
        self.feature_dim: int = 1024
        self.enabled = False
        self.load_error = ""
        self._coef: np.ndarray | None = None

        self._clip_model: CLIPModel | None = None
        self._clip_processor: CLIPProcessor | None = None

        self.model_path = self._resolve_model_path(model_path)
        self._load_classifier()
        if self.enabled:
            self._load_clip()

    @staticmethod
    def _resolve_model_path(model_path: str | Path) -> Path:
        p = Path(model_path)
        if p.exists():
            return p
        here = Path(__file__).resolve().parent
        candidates = [
            here.parent / str(model_path),
            here / str(model_path),
        ]
        for c in candidates:
            if c.exists():
                return c
        return p

    def _load_classifier(self) -> None:
        try:
            payload = joblib.load(self.model_path)
            self.model = payload["model"]
            self.threshold = float(payload.get("threshold", 0.5))
            self.feature_dim = int(payload.get("feature_dim", 1024))
            coef = payload.get("coef")
            if coef is not None:
                self._coef = np.array(coef, dtype=np.float32).reshape(-1)
            self.enabled = True
        except Exception as exc:  # noqa: BLE001
            self.enabled = False
            self.load_error = str(exc)

    def _load_clip(self) -> None:
        try:
            self._clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self._clip_model.to(self.device)
            self._clip_model.eval()
        except Exception as exc:  # noqa: BLE001
            self.enabled = False
            self.load_error = f"CLIP load failed: {exc}"

    def _extract_embedding(self, video_path: Path) -> np.ndarray:
        if self._clip_model is None or self._clip_processor is None:
            raise RuntimeError("CLIP not initialized")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("video_open_failed")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = _sample_frame_indices(total_frames, self.n_frames)
        images: list[Image.Image] = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))
        cap.release()
        if not images:
            raise RuntimeError("no_frames")

        inputs = self._clip_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self._clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        arr = feats.detach().cpu().numpy().astype(np.float32)
        emb = np.concatenate([arr.mean(axis=0), arr.std(axis=0)], axis=0)
        return emb.reshape(1, -1)

    def _top_dims(self, emb_1d: np.ndarray, k: int = 5) -> list[int]:
        if self._coef is not None and self._coef.size == emb_1d.size:
            contrib = np.abs(emb_1d * self._coef)
        else:
            contrib = np.abs(emb_1d)
        idx = np.argsort(contrib)[-k:][::-1]
        return [int(i) for i in idx.tolist()]

    def detect_video(self, video_path: str | Path) -> dict[str, Any]:
        vp = Path(video_path)
        if not self.enabled or self.model is None:
            return {
                "ai_style_prob": 0.0,
                "ai_style_detected": False,
                "ai_style_threshold": self.threshold,
                "ai_style_top_dims": [],
                "error": self.load_error or "detector_disabled",
            }

        try:
            emb = self._extract_embedding(vp)
            if hasattr(self.model, "predict_proba"):
                prob = float(self.model.predict_proba(emb)[0, 1])
            else:
                dec = float(self.model.decision_function(emb)[0])
                prob = float(1.0 / (1.0 + np.exp(-dec)))
            emb_1d = emb.reshape(-1)
            return {
                "ai_style_prob": prob,
                "ai_style_detected": bool(prob >= self.threshold),
                "ai_style_threshold": float(self.threshold),
                "ai_style_top_dims": self._top_dims(emb_1d, k=5),
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ai_style_prob": 0.0,
                "ai_style_detected": False,
                "ai_style_threshold": float(self.threshold),
                "ai_style_top_dims": [],
                "error": str(exc),
            }


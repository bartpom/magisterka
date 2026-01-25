from __future__ import annotations

"""
videomae_detector.py

Detektor deepfake / AI-video oparty o VideoMAE fine-tuned (Hugging Face).
To daje brakujący u Ciebie sygnał "AI Video Model Score" (u Ciebie często 0.00%),
czyli klasyfikację na poziomie sekwencji klatek, a nie pojedynczej klatki.

Uwagi praktyczne:
- Modele VideoMAE są cięższe niż klasyfikatory obrazowe. Na CPU to będzie wolniejsze.
- Warto uruchamiać to na próbkowanych klatkach (np. 16) i cache'ować model w pamięci.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
except Exception as e:  # pragma: no cover
    VideoMAEForVideoClassification = None  # type: ignore
    VideoMAEImageProcessor = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


@dataclass
class VideoMAEConfig:
    # Domyślnie biorę model "base" (mniejszy) żeby CPU dało radę.
    # Alternatywa (większa/cięższa): "shylhy/videomae-large-finetuned-deepfake-subset"
    model_id: str = "Ammar2k/videomae-base-finetuned-deepfake-subset"
    device: str = "cpu"
    # Jeśli None: bierzemy model.config.num_frames
    num_frames: Optional[int] = None
    # Gdy wideo jest długie, uniform sampling daje lepszą reprezentację.
    sampling: str = "uniform"
    # Maksymalna szer./wys. wejściowa; preprocessor i tak zrobi resize/crop.
    max_side: int = 256


class VideoMAEDeepfakeDetector:
    def __init__(self, cfg: Optional[VideoMAEConfig] = None) -> None:
        self.cfg = cfg or VideoMAEConfig()
        self._device = torch.device(self.cfg.device)
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                f"Brak transformers VideoMAE. Zainstaluj/aktualizuj transformers. "
                f"Szczegóły importu: {_IMPORT_ERROR}"
            )
        if self._model is not None and self._processor is not None:
            return

        # Lazy-load (ważne dla GUI: nie blokuj startu aplikacji)
        self._processor = VideoMAEImageProcessor.from_pretrained(self.cfg.model_id)
        self._model = VideoMAEForVideoClassification.from_pretrained(self.cfg.model_id)
        self._model.to(self._device)
        self._model.eval()

    def _sample_frames_uniform(self, video_path: str, num_frames: int) -> List[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total = int(total) if total and not np.isnan(total) else 0

        frames: List[Image.Image] = []
        if total > 0:
            idxs = np.linspace(0, max(total - 1, 0), num_frames).astype(int)
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        else:
            # Fallback: czytaj sekwencyjnie i potem przytnij/rozszerz
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                if len(frames) >= num_frames:
                    break

        cap.release()
        if not frames:
            return []

        # Dopasuj do num_frames (duplikacja ostatniej klatki, jeśli braki)
        if len(frames) < num_frames:
            last = frames[-1]
            frames.extend([last] * (num_frames - len(frames)))
        elif len(frames) > num_frames:
            frames = frames[:num_frames]

        return frames

    def analyze(self, video_path: str) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Zwraca:
          - p_fake w [0,1] lub None
          - details dict (label, raw probs)
        """
        try:
            self._ensure_loaded()
        except Exception as e:
            return None, {"videomae_status": "load_failed", "videomae_error": str(e)}

        assert self._model is not None and self._processor is not None

        # Ile klatek chce model?
        num_frames = int(self.cfg.num_frames or getattr(self._model.config, "num_frames", 16) or 16)
        num_frames = max(num_frames, 8)

        frames = self._sample_frames_uniform(video_path, num_frames=num_frames)
        if len(frames) < num_frames:
            return None, {"videomae_status": "no_frames"}

        try:
            inputs = self._processor(frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self._device)

            with torch.inference_mode():
                out = self._model(pixel_values=pixel_values)
                logits = out.logits  # [1, C]
                probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

            # Ustal indeks klasy FAKE (różne modele mogą mieć różne id2label)
            id2label = getattr(self._model.config, "id2label", {}) or {}
            fake_idx = None
            for k, v in id2label.items():
                try:
                    ki = int(k)
                except Exception:
                    continue
                label = str(v).lower()
                if "fake" in label or "deepfake" in label:
                    fake_idx = ki
                    break
            if fake_idx is None:
                # w wielu fine-tuned modelach jest: 0=fake/deepfake, 1=real
                fake_idx = 0

            p_fake = float(probs[fake_idx])

            # Najlepsza etykieta wg modelu
            best_idx = int(np.argmax(probs))
            best_label = str(id2label.get(str(best_idx), id2label.get(best_idx, f"label_{best_idx}")))

            details = {
                "videomae_status": "ok",
                "videomae_model": self.cfg.model_id,
                "videomae_num_frames": num_frames,
                "videomae_best_label": best_label,
                "videomae_p_fake": p_fake,
                "videomae_probs": probs.tolist(),
                "videomae_id2label": {str(k): str(v) for k, v in id2label.items()},
            }
            return p_fake, details

        except Exception as e:
            return None, {"videomae_status": "inference_failed", "videomae_error": str(e)}
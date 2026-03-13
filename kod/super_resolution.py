"""
super_resolution.py

Opcjonalny modul Super-Resolution dla polepszenia jakosci ROI przed OCR.
Uzywa cv2.dnn_superres (wbudowany w OpenCV contrib) – nie wymaga PyTorch.

Obslugiwane modele (EDSR, FSRCNN, ESPCN, LapSRN):
  - EDSR_x4.pb        – najlepsza jakosc, wolniejszy  (~50ms/ROI na CPU)
  - FSRCNN_x4.pb      – szybszy, nieco slabsza jakosc (~5ms/ROI na CPU)
  - ESPCN_x4.pb       – najszybszy, minimalnie gorsza jakosc

Jak pobrac modele:
  pip install opencv-contrib-python
  # Pobierz .pb z https://github.com/Saafke/EDSR_Opencv/tree/master/models
  # lub https://github.com/opencv/opencv_contrib (modules/dnn_superres/models)
  # Wrzuc do folderu models/ obok skryptow.

UZYCIE:
  from super_resolution import upscale_roi
  roi_hq = upscale_roi(roi_bgr, scale=4)  # zwraca powiekszone ROI lub oryginalne jesli blad
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np

# Kolejnosc probowania modeli: jakosc vs szybkosc
_MODEL_CANDIDATES = [
    ("EDSR",   4, "EDSR_x4.pb"),
    ("FSRCNN", 4, "FSRCNN_x4.pb"),
    ("ESPCN",  4, "ESPCN_x4.pb"),
    ("EDSR",   2, "EDSR_x2.pb"),
    ("FSRCNN", 2, "FSRCNN_x2.pb"),
]

# Folder z modelami – sprawdzamy kolejno te lokalizacje
_SEARCH_DIRS = [
    os.path.join(os.path.dirname(__file__), "models"),
    os.path.join(os.path.dirname(__file__), "..", "models"),
    "models",
]

_sr: Optional[object] = None          # zaladowany cv2.dnn_superres.DnnSuperResImpl
_sr_scale: int = 1                     # faktyczny skalar zaladowanego modelu
_sr_available: Optional[bool] = None  # None = nie sprawdzono


def _find_model_path(filename: str) -> Optional[str]:
    for d in _SEARCH_DIRS:
        p = os.path.abspath(os.path.join(d, filename))
        if os.path.isfile(p):
            return p
    return None


def _load_sr_model() -> Tuple[Optional[object], int]:
    """
    Probuje zaladowac pierwszy dostepny model SR.
    Zwraca (sr_object, scale) lub (None, 1) jesli brak.
    """
    global _sr_available

    # Sprawdz czy cv2.dnn_superres jest dostepny
    if not hasattr(cv2, 'dnn_superres'):
        _sr_available = False
        print(
            "[SR] cv2.dnn_superres niedostepny. "
            "Zainstaluj: pip install opencv-contrib-python",
            file=sys.stderr
        )
        return None, 1

    for algo, scale, filename in _MODEL_CANDIDATES:
        model_path = _find_model_path(filename)
        if model_path is None:
            continue
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(model_path)
            sr.setModel(algo.lower(), scale)
            # Test z miniaturka – jesli zaladowal sie OK, nie rzuci wyjatku
            test = np.zeros((8, 8, 3), dtype=np.uint8)
            _ = sr.upsample(test)
            _sr_available = True
            print(f"[SR] Zaladowano model: {algo}_x{scale} ({model_path})", file=sys.stderr)
            return sr, scale
        except Exception as e:
            print(f"[SR] Blad ladowania {filename}: {e}", file=sys.stderr)
            continue

    _sr_available = False
    print(
        "[SR] Brak modeli SR w folderze models/. "
        "Pobierz EDSR_x4.pb lub FSRCNN_x4.pb i wrzuc do kod/models/.",
        file=sys.stderr
    )
    return None, 1


def is_available() -> bool:
    """Czy Super-Resolution jest dostepne (zaladowany model)."""
    global _sr, _sr_scale, _sr_available
    if _sr_available is None:
        _sr, _sr_scale = _load_sr_model()
    return _sr_available is True


def upscale_roi(
    roi_bgr: np.ndarray,
    scale: Optional[int] = None,
    fallback_interpolation: int = cv2.INTER_CUBIC
) -> np.ndarray:
    """
    Powieksza ROI przez Super-Resolution (EDSR/FSRCNN).
    Jesli model niedostepny – fallback do cv2.resize z INTER_CUBIC.

    Args:
        roi_bgr               : fragment klatki BGR uint8
        scale                 : skalar (2 lub 4) – jesli None, uzywa zaladowanego modelu
        fallback_interpolation: metoda interpolacji gdy SR niedostepny

    Returns:
        Powiekszone ROI (uint8 BGR)
    """
    global _sr, _sr_scale, _sr_available

    if roi_bgr is None or roi_bgr.size == 0:
        return roi_bgr

    if _sr_available is None:
        _sr, _sr_scale = _load_sr_model()

    effective_scale = scale or _sr_scale or 4

    # Uzyj SR jesli dostepny i skalar sie zgadza (lub nie wymuszono konkretnego)
    if _sr is not None and (scale is None or scale == _sr_scale):
        try:
            # cv2.dnn_superres nie lubi bardzo malych ROI (<4px) – guard
            if roi_bgr.shape[0] < 4 or roi_bgr.shape[1] < 4:
                raise ValueError("ROI za male dla SR")
            result = _sr.upsample(roi_bgr)  # type: ignore
            return result
        except Exception as e:
            print(f"[SR] upscale_roi blad SR, fallback INTER_CUBIC: {e}", file=sys.stderr)

    # Fallback: klasyczny resize
    h, w = roi_bgr.shape[:2]
    return cv2.resize(
        roi_bgr,
        (w * effective_scale, h * effective_scale),
        interpolation=fallback_interpolation
    )


def upscale_for_ocr(
    roi_bgr: np.ndarray,
    target_min_dim: int = 64
) -> np.ndarray:
    """
    Inteligentne powiekszanie ROI dla OCR:
    - Jesli ROI jest juz duze (min wymiar > target_min_dim) – zwraca bez zmian
    - W przeciwnym razie aplikuje SR (lub fallback INTER_CUBIC)
    - Zapewnia ze wynik ma min. target_min_dim pikseli w krotszym wymiarze

    Args:
        roi_bgr        : fragment klatki BGR uint8
        target_min_dim : minimalna pozadana dlugosc krotszego wymiaru po SR

    Returns:
        ROI gotowe do podania do OCR
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return roi_bgr

    h, w = roi_bgr.shape[:2]
    min_dim = min(h, w)

    if min_dim >= target_min_dim:
        return roi_bgr  # wystarczajaco duze, nie powieksza

    # Oblicz minimalny potrzebny skalar
    needed_scale = max(2, int(np.ceil(target_min_dim / max(min_dim, 1))))
    # Zaokraglij do 2 lub 4 (obslugiwane przez SR)
    sr_scale = 4 if needed_scale > 2 else 2

    return upscale_roi(roi_bgr, scale=sr_scale)

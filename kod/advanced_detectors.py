"""
advanced_detectors.py

Zaawansowane metody detekcji znakow wodnych:
1. Temporal Median Filtering  – wydobywa statyczny znak wodny z sekwencji klatek
2. Invisible Watermark        – dekoduje ukryty token DWT/DWT-DCT/RivaGAN (imwatermark)
3. Noise Residual / FFT       – wykrywa periodyczne artefakty upsamplingu AI
4. Zero-Variance ROI          – wykrywa regiony bez zmian w czasie (statyczny overlay)

Kazda metoda zwraca slownik z kluczami:
  found   : bool
  method  : str
  details : str       (opis znaleziska)
  score   : float     (0-1, pewnosc)
  image   : np.ndarray | None   (obraz diagnostyczny, opcjonalny)
"""

from __future__ import annotations

import sys
from typing import List, Optional, Dict, Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. TEMPORAL MEDIAN FILTERING
# ---------------------------------------------------------------------------

def build_temporal_median(
    frames: List[np.ndarray],
    max_frames: int = 50
) -> np.ndarray:
    """
    Oblicza mediane po osi czasu dla listy klatek BGR.
    Ruchome obiekty 'znikaja', statyczny znak wodny pozostaje ostry.

    Args:
        frames: lista klatek np.ndarray (BGR uint8, takie same rozmiary)
        max_frames: uzyj co najwyzej tylu klatek (rowno rozniesione)

    Returns:
        Obraz mediany (uint8 BGR)
    """
    if not frames:
        raise ValueError("Pusta lista klatek")

    # Probkuj rowno jesli za duzo klatek
    if len(frames) > max_frames:
        step = len(frames) // max_frames
        frames = frames[::step][:max_frames]

    # Konwertuj do float32 i ukladaj w tensor (T, H, W, C)
    stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
    # np.median po osi 0 jest wolny dla duzych tensorow – uzywamy partial sort
    median_frame = np.median(stack, axis=0).astype(np.uint8)
    return median_frame


def extract_static_overlay(
    median_frame: np.ndarray,
    reference_frame: np.ndarray,
    amp: float = 4.0
) -> np.ndarray:
    """
    Odejmuje "gładkie tło" (mediane) od pojedynczej klatki referencyjnej,
    wzmacnia roznice – to co zostaje to statyczny overlay (znak wodny).

    Args:
        median_frame  : wynik build_temporal_median()
        reference_frame: dowolna klatka z tego samego wideo
        amp           : wspolczynnik wzmocnienia roznicy (domyslnie 4x)

    Returns:
        Obraz diff (uint8 BGR) z wyeksponowanym statycznym overlayem
    """
    diff = cv2.absdiff(reference_frame.astype(np.float32),
                       median_frame.astype(np.float32))
    diff = np.clip(diff * amp, 0, 255).astype(np.uint8)
    return diff


def detect_zero_variance_rois(
    frames: List[np.ndarray],
    corner_ratio: float = 0.20,
    variance_threshold: float = 8.0,
    min_fraction: float = 0.30
) -> List[Dict[str, Any]]:
    """
    Szuka regionow w kadrze (glownie narozniki), gdzie wariancja pikselowa
    w czasie jest bliska 0 – silna heurystyka statycznego nałozonego elementu.

    Args:
        frames            : lista klatek BGR
        corner_ratio      : jaka czesc krawedzi kadru analizowac (domyslnie 20%)
        variance_threshold: prog wariancji – ponizej = "statyczny"
        min_fraction      : jaki procent pikseli w ROI musi byc ponizej progu

    Returns:
        Lista slownikow {name, bbox, score, variance_map}
    """
    if len(frames) < 5:
        return []

    h, w = frames[0].shape[:2]
    ch = int(h * corner_ratio)
    cw = int(w * corner_ratio)

    corners = [
        ("CORNER-TL", (0,      0,      cw,     ch)),
        ("CORNER-TR", (w - cw, 0,      w,      ch)),
        ("CORNER-BL", (0,      h - ch, cw,     h)),
        ("CORNER-BR", (w - cw, h - ch, w,      h)),
    ]

    # Tensor szarych klatek: (T, H, W)
    gray_stack = np.stack(
        [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames],
        axis=0
    )
    variance_map = np.var(gray_stack, axis=0)  # (H, W)

    results = []
    for name, (x1, y1, x2, y2) in corners:
        roi_var = variance_map[y1:y2, x1:x2]
        if roi_var.size == 0:
            continue
        low_var_fraction = float(np.mean(roi_var < variance_threshold))
        if low_var_fraction >= min_fraction:
            results.append({
                "name": name,
                "bbox": (x1, y1, x2, y2),
                "score": low_var_fraction,
                "variance_map": roi_var
            })
    return results


# ---------------------------------------------------------------------------
# 2. INVISIBLE WATERMARK (imwatermark / DWT / RivaGAN)
# ---------------------------------------------------------------------------

# Znane sygnatury binarne roznych systemow
_KNOWN_SIGNATURES: Dict[str, str] = {
    # Stable Diffusion / Stability AI (invisible-watermark default 48-bit)
    "STABILITY_AI": "110100110100101000001111111011010101010001000111",
    # Przykladowe – rozszerz wg potrzeb
    "RUNWAY_WATERMARK": "101010101010101010101010",
}

_INVISIBLE_WM_AVAILABLE = None  # None = nie sprawdzono, True/False po pierwszym wywolaniu


def _check_imwatermark() -> bool:
    global _INVISIBLE_WM_AVAILABLE
    if _INVISIBLE_WM_AVAILABLE is not None:
        return _INVISIBLE_WM_AVAILABLE
    try:
        from imwatermark import WatermarkDecoder  # type: ignore  # noqa
        _INVISIBLE_WM_AVAILABLE = True
    except ImportError:
        _INVISIBLE_WM_AVAILABLE = False
        print("[INVIS-WM] imwatermark niedostepny. Zainstaluj: pip install invisible-watermark",
              file=sys.stderr)
    return _INVISIBLE_WM_AVAILABLE


def detect_invisible_watermark(
    frame_bgr: np.ndarray,
    methods: Optional[List[str]] = None,
    watermark_length: int = 48
) -> Dict[str, Any]:
    """
    Proba zdekodowania ukrytego znaku wodnego z pojedynczej klatki.

    Args:
        frame_bgr       : klatka BGR uint8
        methods         : lista metod do sprawdzenia: ["dwtDct", "dwtDctSvd", "rivaGan"]
                          domyslnie sprawdzamy wszystkie
        watermark_length: dlugosc bitu klucza do dekodowania (domyslnie 48)

    Returns:
        slownik:
          found   : bool
          method  : str
          bits    : str  (zdekodowany ciag bitow)
          matched : str | None  (nazwa pasujacego systemu lub None)
          score   : float
    """
    result = {"found": False, "method": "invisible_watermark",
              "bits": "", "matched": None, "score": 0.0, "details": ""}

    if not _check_imwatermark():
        result["details"] = "imwatermark niedostepny"
        return result

    if methods is None:
        methods = ["dwtDct", "dwtDctSvd"]
        try:
            import torch  # type: ignore  # noqa
            methods.append("rivaGan")
        except ImportError:
            pass

    try:
        from imwatermark import WatermarkDecoder  # type: ignore
    except ImportError:
        result["details"] = "import error"
        return result

    bgr = frame_bgr
    if bgr.shape[2] == 3:
        pass  # OK
    else:
        bgr = bgr[:, :, :3]

    for method in methods:
        try:
            decoder = WatermarkDecoder('bits', watermark_length)
            watermark_bits = decoder.decode(bgr, method)
            bits_str = ''.join(str(int(b)) for b in watermark_bits)

            # Sprawdz czy pasuje do znanych sygnatur
            matched = None
            best_similarity = 0.0
            for sig_name, sig_bits in _KNOWN_SIGNATURES.items():
                if len(sig_bits) <= len(bits_str):
                    # Hamming similarity na pierwszych len(sig_bits) bitach
                    sub = bits_str[:len(sig_bits)]
                    matches = sum(a == b for a, b in zip(sub, sig_bits))
                    sim = matches / len(sig_bits)
                    if sim > best_similarity:
                        best_similarity = sim
                        if sim >= 0.85:
                            matched = sig_name

            # Heurystyka: nierandomowy ciag bitow = nie jest czystym szumem
            ones_ratio = bits_str.count('1') / max(len(bits_str), 1)
            is_nontrivial = 0.15 < ones_ratio < 0.85

            # Wykryj dlugie serie (artifact szumu vs struktura)
            max_run = max(
                (sum(1 for _ in g) for _, g in
                 __import__('itertools').groupby(bits_str)),
                default=0
            )
            has_structure = max_run < len(bits_str) * 0.6

            if matched or (is_nontrivial and has_structure):
                result["found"] = True
                result["method"] = f"invisible_watermark:{method}"
                result["bits"] = bits_str
                result["matched"] = matched
                result["score"] = best_similarity if matched else 0.5
                result["details"] = (
                    f"Metoda={method}, bits={bits_str[:32]}..., "
                    f"pasuje_do={matched or 'nieznany'}, podobienstwo={best_similarity:.2f}"
                )
                return result

        except Exception as e:
            result["details"] = f"blad {method}: {e}"
            continue

    return result


# ---------------------------------------------------------------------------
# 3. NOISE RESIDUAL + FFT – artefakty upsamplingu AI
# ---------------------------------------------------------------------------

def detect_ai_noise_artifacts(
    frame_bgr: np.ndarray,
    fft_peak_threshold: float = 0.35,
    wiener_ksize: int = 5
) -> Dict[str, Any]:
    """
    Wykrywa periodyczne artefakty upsamplingu charakterystyczne dla modeli AI
    poprzez:
    a) ekstrakcje szumu resztkowego (residual po filtrze Wienera),
    b) FFT na szumie – szuka ostrych pik poza centrum.

    Args:
        frame_bgr          : klatka BGR uint8
        fft_peak_threshold : stosunek max_peak / mean_background w FFT
        wiener_ksize       : rozmiar kernela filtru Wienera

    Returns:
        slownik {found, score, details, fft_image}
    """
    result = {"found": False, "method": "noise_residual_fft",
              "score": 0.0, "details": "", "fft_image": None}

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Filtr Wienera (aproksymacja przez Gaussian)
    blurred = cv2.GaussianBlur(gray, (wiener_ksize, wiener_ksize), 0)
    noise = gray - blurred  # residual szumu

    # FFT na szumie
    fft = np.fft.fft2(noise)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shifted))

    # Zamaskuj centrum (niskoczestotliwosci, zawsze silne)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    mask = np.ones_like(magnitude)
    r = min(h, w) // 8  # promien maski centrum
    cv2.circle(mask, (cx, cy), r, 0, -1)
    magnitude_no_center = magnitude * mask

    # Czy sa ostre piki poza centrum?
    mean_bg = float(np.mean(magnitude_no_center[magnitude_no_center > 0]))
    max_peak = float(np.max(magnitude_no_center))
    ratio = max_peak / (mean_bg + 1e-6)

    # Normalizuj do uint8 dla podgladu
    vis = cv2.normalize(magnitude_no_center, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    result["fft_image"] = vis_color

    if ratio > fft_peak_threshold * 10:  # skalujemy prog bo log1p zmienia skale
        result["found"] = True
        result["score"] = min(1.0, (ratio - fft_peak_threshold * 10) / 50.0)
        result["details"] = (
            f"FFT peak ratio={ratio:.2f}, mean_bg={mean_bg:.3f}, "
            f"max_peak={max_peak:.3f} – mozliwe artefakty AI upsamplingu"
        )
    else:
        result["details"] = f"FFT ratio={ratio:.2f} – brak anomalii"

    return result


# ---------------------------------------------------------------------------
# 4. FASADA: skan zaawansowany dla jednego pliku wideo
# ---------------------------------------------------------------------------

def run_advanced_scan(
    cap: cv2.VideoCapture,
    fps: float,
    total_frames: int,
    n_frames_median: int = 40,
    check_invisible: bool = True,
    check_fft: bool = True,
    log_fn=None
) -> Dict[str, Any]:
    """
    Zbiera klatki z otwartego VideoCapture i uruchamia wszystkie zaawansowane metody.

    Args:
        cap              : otwarty cv2.VideoCapture
        fps              : FPS wideo
        total_frames     : liczba klatek
        n_frames_median  : ile klatek pobrac do mediany
        check_invisible  : czy sprawdzac invisible watermark
        check_fft        : czy sprawdzac FFT noise artifacts
        log_fn           : opcjonalna funkcja logowania (str) -> None

    Returns:
        slownik z kluczami:
          temporal_median_frame : np.ndarray (uint8 BGR) | None
          overlay_diff          : np.ndarray (uint8 BGR) | None
          zero_variance_rois    : list
          invisible_wm          : dict
          fft_artifacts         : dict
          summary               : str
    """
    def _log(msg):
        print(msg, file=sys.stderr)
        if log_fn:
            try:
                log_fn(msg)
            except Exception:
                pass

    result: Dict[str, Any] = {
        "temporal_median_frame": None,
        "overlay_diff": None,
        "zero_variance_rois": [],
        "invisible_wm": {"found": False},
        "fft_artifacts": {"found": False},
        "summary": ""
    }

    # Zbierz klatki rowno rozniesione w czasie
    _log("[ADV] Pobieram klatki do analizy temporalnej...")
    step = max(1, total_frames // n_frames_median)
    frames: List[np.ndarray] = []
    pos = 0
    while len(frames) < n_frames_median and pos < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
        pos += step

    if len(frames) < 3:
        result["summary"] = "Za malo klatek do analizy zaawansowanej."
        return result

    _log(f"[ADV] Zebrano {len(frames)} klatek. Licze mediane temporalna...")

    # 1. Temporal median
    try:
        median_frame = build_temporal_median(frames)
        result["temporal_median_frame"] = median_frame
        # Diff na srodkowej klatce
        mid = frames[len(frames) // 2]
        result["overlay_diff"] = extract_static_overlay(median_frame, mid, amp=5.0)
        _log("[ADV] Mediana temporalna obliczona.")
    except Exception as e:
        _log(f"[ADV] Blad mediany temporalnej: {e}")

    # 2. Zero variance ROIs
    try:
        zv_rois = detect_zero_variance_rois(frames)
        result["zero_variance_rois"] = zv_rois
        if zv_rois:
            names = [r['name'] for r in zv_rois]
            _log(f"[ADV] Zerowa wariancja ROI: {names} – mozliwy statyczny overlay")
    except Exception as e:
        _log(f"[ADV] Blad zero-variance: {e}")

    # 3. Invisible watermark – sprawdz na medianie (najczystszy sygnal)
    if check_invisible and result["temporal_median_frame"] is not None:
        _log("[ADV] Sprawdzam invisible watermark (imwatermark)...")
        try:
            iw = detect_invisible_watermark(result["temporal_median_frame"])
            result["invisible_wm"] = iw
            if iw["found"]:
                _log(f"[ADV] INVISIBLE WM ZNALEZIONY: {iw['details']}")
            else:
                _log(f"[ADV] Invisible WM: brak / {iw['details']}")
        except Exception as e:
            _log(f"[ADV] Blad invisible WM: {e}")

    # 4. FFT noise – sprawdz na jednej srodkowej klatce
    if check_fft:
        _log("[ADV] Sprawdzam artefakty FFT noise...")
        try:
            fft_res = detect_ai_noise_artifacts(frames[len(frames) // 2])
            result["fft_artifacts"] = fft_res
            if fft_res["found"]:
                _log(f"[ADV] FFT artefakty: {fft_res['details']}")
            else:
                _log(f"[ADV] FFT: {fft_res['details']}")
        except Exception as e:
            _log(f"[ADV] Blad FFT: {e}")

    # Podsumowanie
    findings = []
    if result["zero_variance_rois"]:
        findings.append(f"statyczny_overlay({len(result['zero_variance_rois'])} ROI)")
    if result["invisible_wm"].get("found"):
        findings.append(f"invisible_wm({result['invisible_wm'].get('matched', 'nieznany')})")
    if result["fft_artifacts"].get("found"):
        findings.append(f"fft_artefakty(score={result['fft_artifacts'].get('score', 0):.2f})")
    result["summary"] = "ZNALEZIONO: " + ", ".join(findings) if findings else "Brak wynikow zaawansowanych."
    _log(f"[ADV] Podsumowanie: {result['summary']}")

    return result

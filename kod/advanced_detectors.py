"""
advanced_detectors.py

Zaawansowane metody detekcji znakow wodnych:
1. Temporal Median Filtering  – wydobywa statyczny znak wodny z sekwencji klatek
2. Invisible Watermark        – dekoduje ukryty token DWT/DWT-DCT/RivaGAN (imwatermark)
3. Noise Residual / FFT       – wykrywa periodyczne artefakty upsamplingu AI
4. Zero-Variance ROI          – wykrywa regiony bez zmian w czasie (statyczny overlay)
5. Optical Flow Overlay       – wykrywa statyczne piksele mimo globalnego ruchu kamery
                                 przez analize konturow na mapie static_mask (Farneback)
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
    if not frames:
        raise ValueError("Pusta lista klatek")
    if len(frames) > max_frames:
        step = len(frames) // max_frames
        frames = frames[::step][:max_frames]
    stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def extract_static_overlay(
    median_frame: np.ndarray,
    reference_frame: np.ndarray,
    amp: float = 4.0
) -> np.ndarray:
    diff = cv2.absdiff(reference_frame.astype(np.float32),
                       median_frame.astype(np.float32))
    return np.clip(diff * amp, 0, 255).astype(np.uint8)


def detect_zero_variance_rois(
    frames: List[np.ndarray],
    corner_ratio: float = 0.20,
    variance_threshold: float = 8.0,
    min_fraction: float = 0.30
) -> List[Dict[str, Any]]:
    if len(frames) < 5:
        return []
    h, w = frames[0].shape[:2]
    ch, cw = int(h * corner_ratio), int(w * corner_ratio)
    corners = [
        ("CORNER-TL", (0,      0,      cw,     ch)),
        ("CORNER-TR", (w - cw, 0,      w,      ch)),
        ("CORNER-BL", (0,      h - ch, cw,     h)),
        ("CORNER-BR", (w - cw, h - ch, w,      h)),
    ]
    gray_stack = np.stack(
        [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames], axis=0
    )
    variance_map = np.var(gray_stack, axis=0)
    results = []
    for name, (x1, y1, x2, y2) in corners:
        roi_var = variance_map[y1:y2, x1:x2]
        if roi_var.size == 0:
            continue
        low_var_fraction = float(np.mean(roi_var < variance_threshold))
        if low_var_fraction >= min_fraction:
            results.append({"name": name, "bbox": (x1, y1, x2, y2),
                             "score": low_var_fraction, "variance_map": roi_var})
    return results


# ---------------------------------------------------------------------------
# 5. OPTICAL FLOW OVERLAY DETECTION (Farneback + contour search)
# ---------------------------------------------------------------------------

def detect_optical_flow_overlay(
    frames: List[np.ndarray],
    flow_zero_threshold: float = 0.5,
    min_global_motion: float = 0.8,
    min_contour_area: int = 40,
    morph_kernel_size: int = 5
) -> List[Dict[str, Any]]:
    """
    Wykrywa statyczne piksele (nalozone overlaye) pomimo globalnego ruchu kamery.

    Algorytm:
    1. Dense Optical Flow Farneback miedzy kolejnymi parami klatek.
    2. Akumulacja sredniej mapy ruchu.
    3. Jesli global_mean_motion < min_global_motion -> kamera stoi -> zwroc [] 
       (klasyczne zero-variance wystarczy, OF nic nie wnosi).
    4. Binaryzacja: piksele z ruchem < flow_zero_threshold = 'statyczne'.
    5. Morfologiczne zamkniecie (CLOSE) laczace bliskie litery w jedna bryle.
    6. cv2.findContours -> filtracja po polu powierzchni -> cv2.boundingRect.
       BRAK sztywnego ograniczenia do naroznikow: wykrywa watermarki w centrum,
       na dole kadru, oraz jest odporny na crop-attack (przycinanie krawedzi).

    Args:
        frames             : lista klatek BGR (min. 3)
        flow_zero_threshold: prog wektora ruchu (px) – ponizej = statyczny
        min_global_motion  : minimalny globalny ruch (px/klatke) by OF mial sens
        min_contour_area   : minimalne pole konturu (eliminuje szum kompresji)
        morph_kernel_size  : rozmiar kernela morfologicznego zamkniecia

    Returns:
        Lista {name, bbox, score, area, global_motion} lub []
    """
    if len(frames) < 3:
        return []

    h, w = frames[0].shape[:2]

    _FB_PARAMS = dict(
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    # Akumuluj mape ruchu (probkuj max 10 par klatek)
    step = max(1, len(frames) // 10)
    sampled = frames[::step][:11]  # max 11 klatek -> 10 par

    magnitude_acc = np.zeros((h, w), dtype=np.float32)
    n_pairs = 0
    prev_gray = cv2.cvtColor(sampled[0], cv2.COLOR_BGR2GRAY)
    for curr_frame in sampled[1:]:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        try:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **_FB_PARAMS)
            magnitude_acc += np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            n_pairs += 1
        except Exception:
            pass
        prev_gray = curr_gray

    if n_pairs == 0:
        return []

    avg_magnitude = magnitude_acc / n_pairs
    global_mean_motion = float(np.mean(avg_magnitude))

    if global_mean_motion < min_global_motion:
        return []  # kamera stoi – zero-variance wystarczy

    # Binaryzacja: piksele statyczne = 255
    static_mask = np.where(avg_magnitude < flow_zero_threshold,
                           np.uint8(255), np.uint8(0))

    # Morfologiczne zamkniecie – laczy bliskie litery napisu w jedna bryle
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
    )
    closed_mask = cv2.morphologyEx(static_mask, cv2.MORPH_CLOSE, kernel)

    # Znajdz kontury na calym kadrze (nie tylko w naroznikach!)
    contours, _ = cv2.findContours(
        closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    results = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue  # szum kompresji wideo

        bx, by, bw_cnt, bh_cnt = cv2.boundingRect(cnt)

        # Fraction statycznych pikseli wewnatrz bounding box
        roi_mask = static_mask[by:by + bh_cnt, bx:bx + bw_cnt]
        score = float(np.mean(roi_mask > 0)) if roi_mask.size > 0 else 0.0

        # Nazwa na podstawie pozycji w kadrze
        cx_rel = (bx + bw_cnt / 2) / w
        cy_rel = (by + bh_cnt / 2) / h
        pos_name = (
            "OF-TOP" if cy_rel < 0.25 else
            "OF-BOTTOM" if cy_rel > 0.75 else
            "OF-CENTER"
        )
        if cx_rel < 0.25:
            pos_name += "-L"
        elif cx_rel > 0.75:
            pos_name += "-R"

        results.append({
            "name": f"{pos_name}-{i}",
            "bbox": (bx, by, bx + bw_cnt, by + bh_cnt),
            "score": score,
            "area": area,
            "global_motion": global_mean_motion
        })

    # Sortuj malejaco wg pola – najwazniejsze kontury najpierw
    results.sort(key=lambda x: x["area"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# 2. INVISIBLE WATERMARK (imwatermark / DWT / RivaGAN)
# ---------------------------------------------------------------------------

_KNOWN_SIGNATURES: Dict[str, str] = {
    "STABILITY_AI": "110100110100101000001111111011010101010001000111",
    "RUNWAY_WATERMARK": "101010101010101010101010",
}

_INVISIBLE_WM_AVAILABLE = None


def _check_imwatermark() -> bool:
    global _INVISIBLE_WM_AVAILABLE
    if _INVISIBLE_WM_AVAILABLE is not None:
        return _INVISIBLE_WM_AVAILABLE
    try:
        from imwatermark import WatermarkDecoder  # type: ignore  # noqa
        _INVISIBLE_WM_AVAILABLE = True
    except ImportError:
        _INVISIBLE_WM_AVAILABLE = False
        print("[INVIS-WM] imwatermark niedostepny. pip install invisible-watermark",
              file=sys.stderr)
    return _INVISIBLE_WM_AVAILABLE


def _torch_available() -> bool:
    if 'torch' in sys.modules:
        return True
    try:
        import importlib
        importlib.import_module('torch')
        return True
    except Exception:
        return False


def detect_invisible_watermark(
    frame_bgr: np.ndarray,
    methods: Optional[List[str]] = None,
    watermark_length: int = 48
) -> Dict[str, Any]:
    result = {"found": False, "method": "invisible_watermark",
              "bits": "", "matched": None, "score": 0.0, "details": ""}

    if not _check_imwatermark():
        result["details"] = "imwatermark niedostepny"
        return result

    if methods is None:
        methods = ["dwtDct", "dwtDctSvd"]
        if _torch_available():
            methods.append("rivaGan")

    try:
        from imwatermark import WatermarkDecoder  # type: ignore
    except ImportError:
        result["details"] = "import error"
        return result

    bgr = frame_bgr[:, :, :3] if frame_bgr.ndim == 3 and frame_bgr.shape[2] != 3 else frame_bgr

    for method in methods:
        try:
            decoder = WatermarkDecoder('bits', watermark_length)
            watermark_bits = decoder.decode(bgr, method)
            bits_str = ''.join(str(int(b)) for b in watermark_bits)

            matched = None
            best_similarity = 0.0
            for sig_name, sig_bits in _KNOWN_SIGNATURES.items():
                if len(sig_bits) <= len(bits_str):
                    sub = bits_str[:len(sig_bits)]
                    matches = sum(a == b for a, b in zip(sub, sig_bits))
                    sim = matches / len(sig_bits)
                    if sim > best_similarity:
                        best_similarity = sim
                        if sim >= 0.85:
                            matched = sig_name

            ones_ratio = bits_str.count('1') / max(len(bits_str), 1)
            is_nontrivial = 0.15 < ones_ratio < 0.85
            max_run = max(
                (sum(1 for _ in g) for _, g in
                 __import__('itertools').groupby(bits_str)), default=0
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
# 3. NOISE RESIDUAL + FFT
# ---------------------------------------------------------------------------

def detect_ai_noise_artifacts(
    frame_bgr: np.ndarray,
    fft_peak_threshold: float = 0.35,
    wiener_ksize: int = 5
) -> Dict[str, Any]:
    result = {"found": False, "method": "noise_residual_fft",
              "score": 0.0, "details": "", "fft_image": None}

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (wiener_ksize, wiener_ksize), 0)
    noise = gray - blurred

    fft = np.fft.fft2(noise)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shifted))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8

    mask_u8 = np.ones((h, w), dtype=np.uint8)
    cv2.circle(mask_u8, (cx, cy), r, 0, -1)
    mask = mask_u8.astype(np.float64)
    magnitude_no_center = magnitude * mask

    nonzero = magnitude_no_center[magnitude_no_center > 0]
    mean_bg = float(np.mean(nonzero)) if nonzero.size > 0 else 1e-6
    max_peak = float(np.max(magnitude_no_center))
    ratio = max_peak / (mean_bg + 1e-6)

    vis = cv2.normalize(
        magnitude_no_center.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    result["fft_image"] = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)

    if ratio > fft_peak_threshold * 10:
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
# 4. FASADA
# ---------------------------------------------------------------------------

def run_advanced_scan(
    cap: cv2.VideoCapture,
    fps: float,
    total_frames: int,
    n_frames_median: int = 40,
    check_invisible: bool = True,
    check_fft: bool = True,
    check_optical_flow: bool = True,
    log_fn=None
) -> Dict[str, Any]:
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
        "optical_flow_rois": [],
        "invisible_wm": {"found": False},
        "fft_artifacts": {"found": False},
        "summary": ""
    }

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

    try:
        median_frame = build_temporal_median(frames)
        result["temporal_median_frame"] = median_frame
        mid = frames[len(frames) // 2]
        result["overlay_diff"] = extract_static_overlay(median_frame, mid, amp=5.0)
        _log("[ADV] Mediana temporalna obliczona.")
    except Exception as e:
        _log(f"[ADV] Blad mediany temporalnej: {e}")

    try:
        zv_rois = detect_zero_variance_rois(frames)
        result["zero_variance_rois"] = zv_rois
        if zv_rois:
            _log(f"[ADV] Zerowa wariancja ROI: {[r['name'] for r in zv_rois]}")
    except Exception as e:
        _log(f"[ADV] Blad zero-variance: {e}")

    if check_optical_flow and len(frames) >= 3:
        _log("[ADV] Sprawdzam Optical Flow (Farneback + contour search)...")
        try:
            of_rois = detect_optical_flow_overlay(frames)
            result["optical_flow_rois"] = of_rois
            if of_rois:
                _log(f"[ADV] OF – znaleziono {len(of_rois)} konturow statycznych: "
                     f"{[r['name'] for r in of_rois[:5]]}")
            else:
                _log("[ADV] OF: brak statycznych konturow / kamera statyczna")
        except Exception as e:
            _log(f"[ADV] Blad Optical Flow: {e}")

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

    findings = []
    if result["zero_variance_rois"]:
        findings.append(f"statyczny_overlay({len(result['zero_variance_rois'])} ROI)")
    if result["optical_flow_rois"]:
        findings.append(f"optical_flow_overlay({len(result['optical_flow_rois'])} konturow)")
    if result["invisible_wm"].get("found"):
        findings.append(f"invisible_wm({result['invisible_wm'].get('matched', 'nieznany')})")
    if result["fft_artifacts"].get("found"):
        findings.append(f"fft_artefakty(score={result['fft_artifacts'].get('score', 0):.2f})")
    result["summary"] = "ZNALEZIONO: " + ", ".join(findings) if findings else "Brak wynikow zaawansowanych."
    _log(f"[ADV] Podsumowanie: {result['summary']}")

    return result

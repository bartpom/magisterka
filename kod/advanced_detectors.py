"""
advanced_detectors.py

Zaawansowane metody detekcji znakow wodnych:
1. Temporal Median Filtering  – wydobywa statyczny znak wodny z sekwencji klatek
2. Invisible Watermark        – dekoduje ukryty token DWT/DWT-DCT/RivaGAN (imwatermark)
3. Noise Residual / FFT       – wykrywa periodyczne artefakty upsamplingu AI
4. Zero-Variance ROI          – wykrywa regiony bez zmian w czasie (statyczny overlay)
5. Optical Flow Overlay       – wykrywa statyczne piksele mimo globalnego ruchu kamery
                                 (Farneback Dense OF + contour search, crop-attack resistant)

Optymalizacje wydajnosci:
- OF liczony na klatkach zmniejszonych do of_scale (domyslnie 0.5) -> 4x mniej obliczen
- static_mask przeskalowana z powrotem do oryginalnych wymiarow przed nalozeiem konturow
- Opcjonalne CUDA: cv2.cuda_FarnebackOpticalFlow jesli dostepne (karta NVIDIA)
"""

from __future__ import annotations

import sys
from typing import List, Optional, Dict, Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Wykrywanie CUDA
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    """
    Sprawdza czy OpenCV zostal skompilowany z CUDA i czy jest dostepna karta NVIDIA.
    Uzywa cv2.cuda.getCudaEnabledDeviceCount() bez rzucania wyjatkow.
    """
    try:
        return hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


_CUDA_AVAILABLE: Optional[bool] = None  # None = nie sprawdzono


def _get_cuda() -> bool:
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is None:
        _CUDA_AVAILABLE = _cuda_available()
        if _CUDA_AVAILABLE:
            print("[ADV] CUDA dostepna – OF bezie liczony na GPU.", file=sys.stderr)
        else:
            print("[ADV] CUDA niedostepna – OF na CPU (Farneback).", file=sys.stderr)
    return _CUDA_AVAILABLE


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
# 5. OPTICAL FLOW OVERLAY DETECTION
# ---------------------------------------------------------------------------

def detect_optical_flow_overlay(
    frames: List[np.ndarray],
    flow_zero_threshold: float = 0.5,
    min_global_motion: float = 0.8,
    min_contour_area: int = 40,
    morph_kernel_size: int = 5,
    of_scale: float = 0.5,
    low_texture_threshold: float = 50.0,
    use_cuda: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """
    Wykrywa statyczne piksele (nalozone overlaye) pomimo globalnego ruchu kamery.

    Optymalizacje wydajnosci:
    - of_scale (domyslnie 0.5): klatki zmniejszone do 50% przed OF -> 4x mniej obliczen
      na CPU. Ruch skaluje sie liniowo (flow_zero_threshold korygowany automatycznie).
      static_mask przeskalowywana z powrotem do oryginalnych wymiarow przed konturami.
    - use_cuda=True: uzywa cv2.cuda_FarnebackOpticalFlow (GPU NVIDIA).
      Jesli CUDA niedostepna lub use_cuda=False -> fallback do CPU Farneback.

    Algorytm:
    1. Downscale klatek do of_scale.
    2. Dense Optical Flow (Farneback CPU lub CUDA GPU) miedzy parami klatek.
    3. Akumulacja sredniej mapy ruchu.
    4. Jesli global_mean_motion < min_global_motion -> kamera stoi -> zwroc [].
    5. Binaryzacja: piksele z ruchem < flow_zero_threshold = statyczne.
    6. Upscale static_mask do oryginalnych wymiarow.
    7. Morfologiczne zamkniecie (CLOSE) + cv2.findContours na calym kadrze.
    8. Filtracja po polu konturu, cv2.boundingRect, dynamiczne nazwy pozycji.

    Args:
        frames             : lista klatek BGR (min. 3)
        flow_zero_threshold: prog wektora ruchu (px w skali oryginalnej)
        min_global_motion  : minimalny globalny ruch (px) by OF mial sens
        min_contour_area   : minimalne pole konturu (eliminuje szum kompresji)
        morph_kernel_size  : rozmiar kernela morfologicznego zamkniecia
        of_scale           : skalar zmniejszenia klatki przed OF (0.25-1.0)
        use_cuda           : None=autodetect, True=wymusz GPU, False=wymusz CPU

    Returns:
        Lista {name, bbox, score, area, global_motion} lub []
    """
    if len(frames) < 3:
        return []

    h_orig, w_orig = frames[0].shape[:2]

    # Parametry Farneback dla CPU
    _FB_PARAMS = dict(
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    # Ustal czy uzyc CUDA
    if use_cuda is None:
        use_cuda = _get_cuda()

    # Przygotuj obiekt CUDA OF jesli dostepny
    cuda_of = None
    if use_cuda:
        try:
            cuda_of = cv2.cuda_FarnebackOpticalFlow.create(
                numLevels=3, pyrScale=0.5, fastPyramids=False,
                winSize=15, numIters=3, polyN=5, polySigma=1.2, flags=0
            )
        except Exception as e:
            print(f"[ADV] Blad tworzenia CUDA OF: {e} – fallback CPU", file=sys.stderr)
            cuda_of = None
            use_cuda = False

    # Wymiary po downscale
    of_scale = max(0.1, min(1.0, of_scale))  # guard: 0.1 - 1.0
    h_of = max(1, int(h_orig * of_scale))
    w_of = max(1, int(w_orig * of_scale))

    # Koryguj prog ruchu proporcjonalnie do skali
    threshold_scaled = flow_zero_threshold * of_scale
    min_motion_scaled = min_global_motion * of_scale

    # Probkuj max 10 par klatek
    step = max(1, len(frames) // 10)
    sampled = frames[::step][:11]

    magnitude_acc = np.zeros((h_of, w_of), dtype=np.float32)
    n_pairs = 0

    def _resize_gray(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if of_scale < 1.0:
            gray = cv2.resize(gray, (w_of, h_of), interpolation=cv2.INTER_AREA)
        return gray

    prev_gray = _resize_gray(sampled[0])
    texture_frame_gray = cv2.cvtColor(sampled[len(sampled) // 2], cv2.COLOR_BGR2GRAY)

    for curr_frame in sampled[1:]:
        curr_gray = _resize_gray(curr_frame)
        try:
            if cuda_of is not None:
                # CUDA path
                prev_gpu = cv2.cuda_GpuMat()
                curr_gpu = cv2.cuda_GpuMat()
                prev_gpu.upload(prev_gray)
                curr_gpu.upload(curr_gray)
                flow_gpu = cuda_of.calc(prev_gpu, curr_gpu, None)
                flow = flow_gpu.download()
            else:
                # CPU path
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, **_FB_PARAMS
                )
            magnitude_acc += np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            n_pairs += 1
        except Exception as e:
            print(f"[ADV] OF para klatek blad: {e}", file=sys.stderr)
        prev_gray = curr_gray

    if n_pairs == 0:
        return []

    avg_magnitude = magnitude_acc / n_pairs
    global_mean_motion_scaled = float(np.mean(avg_magnitude))

    if global_mean_motion_scaled < min_motion_scaled:
        return []  # kamera stoi – zero-variance wystarczy

    # Binaryzacja w skali OF
    static_mask_small = np.where(
        avg_magnitude < threshold_scaled, np.uint8(255), np.uint8(0)
    )

    # Upscale maski do oryginalnych wymiarow
    if of_scale < 1.0:
        static_mask = cv2.resize(
            static_mask_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST
        )
    else:
        static_mask = static_mask_small

    # global_motion w oryginalnej skali pikselowej
    global_mean_motion = global_mean_motion_scaled / of_scale

    # Morfologiczne zamkniecie – laczy bliskie piksele liter w jedna bryle
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
    )
    closed_mask = cv2.morphologyEx(static_mask, cv2.MORPH_CLOSE, kernel)

    # Znajdz kontury na CALYM kadrze (nie tylko w naroznikach)
    contours, _ = cv2.findContours(
        closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    results = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue

        bx, by, bw_cnt, bh_cnt = cv2.boundingRect(cnt)

        roi_mask = static_mask[by:by + bh_cnt, bx:bx + bw_cnt]
        score = float(np.mean(roi_mask > 0)) if roi_mask.size > 0 else 0.0
        roi_texture = texture_frame_gray[by:by + bh_cnt, bx:bx + bw_cnt]
        texture_variance = float(np.var(roi_texture)) if roi_texture.size > 0 else 0.0

        cx_rel = (bx + bw_cnt / 2) / w_orig
        cy_rel = (by + bh_cnt / 2) / h_orig
        pos_v = "TOP" if cy_rel < 0.25 else ("BOTTOM" if cy_rel > 0.75 else "CENTER")
        pos_h = "-L" if cx_rel < 0.25 else ("-R" if cx_rel > 0.75 else "")

        results.append({
            "name": f"OF-{pos_v}{pos_h}-{i}",
            "bbox": (bx, by, bx + bw_cnt, by + bh_cnt),
            "width_ratio": float(bw_cnt) / float(max(w_orig, 1)),
            "height_ratio": float(bh_cnt) / float(max(h_orig, 1)),
            "cx_rel": cx_rel,
            "cy_rel": cy_rel,
            "score": score,
            "area": area,
            "area_ratio": float(area) / float(max(w_orig * h_orig, 1)),
            "global_motion": global_mean_motion,
            "texture_variance": texture_variance,
            "is_low_texture": bool(texture_variance < low_texture_threshold),
        })

    results.sort(key=lambda x: x["area"], reverse=True)
    return results


def detect_broadcast_trap_patterns(
    of_rois: List[Dict[str, Any]],
    zv_rois: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Heurystyki pułapek broadcast:
    - lower_third_anim: szeroki pas w dolnej części kadru (często lower-third)
    - scoreboard_top_pair: jednoczesne sygnały top-left + top-right
    - billboard_center_large: duży, centralny prostokątny overlay
    """
    if not of_rois:
        return {
            "broadcast_trap": False,
            "lower_third_anim": False,
            "scoreboard_top_pair": False,
            "billboard_center_large": False,
        }

    lower_third_anim = sum(
        1 for r in of_rois
        if float(r.get("cy_rel", 0.0)) >= 0.75
        and float(r.get("height_ratio", 1.0)) <= 0.25
        and float(r.get("width_ratio", 0.0)) >= 0.60
    ) >= 1

    has_tl = any(
        float(r.get("cx_rel", 1.0)) <= 0.30 and float(r.get("cy_rel", 1.0)) <= 0.30
        for r in of_rois
    ) or any(z.get("name") == "CORNER-TL" for z in zv_rois)
    has_tr = any(
        float(r.get("cx_rel", 0.0)) >= 0.70 and float(r.get("cy_rel", 1.0)) <= 0.30
        for r in of_rois
    ) or any(z.get("name") == "CORNER-TR" for z in zv_rois)
    scoreboard_top_pair = has_tl and has_tr

    billboard_center_large = any(
        0.30 <= float(r.get("cx_rel", 0.0)) <= 0.70
        and 0.30 <= float(r.get("cy_rel", 0.0)) <= 0.70
        and float(r.get("area_ratio", 0.0)) >= 0.12
        for r in of_rois
    )

    broadcast_trap = lower_third_anim or scoreboard_top_pair or billboard_center_large
    return {
        "broadcast_trap": broadcast_trap,
        "lower_third_anim": lower_third_anim,
        "scoreboard_top_pair": scoreboard_top_pair,
        "billboard_center_large": billboard_center_large,
    }


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
            method_bits = 32 if method == "rivaGan" else watermark_length
            decoder = WatermarkDecoder('bits', method_bits)
            watermark_bits = decoder.decode(bgr, method)
            bits_str = ''.join(str(int(b)) for b in watermark_bits)

            matched = None
            best_similarity = 0.0
            for sig_name, sig_bits in _KNOWN_SIGNATURES.items():
                cmp_len = min(len(sig_bits), len(bits_str))
                if cmp_len >= 16:
                    sub = bits_str[:cmp_len]
                    ref = sig_bits[:cmp_len]
                    matches = sum(a == b for a, b in zip(sub, ref))
                    sim = matches / cmp_len
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
            # Fallback dla starszych wersji rivaGan, które wspierają tylko 32-bit.
            if method == "rivaGan" and "32 bits" in str(e):
                try:
                    decoder = WatermarkDecoder('bits', 32)
                    watermark_bits = decoder.decode(bgr, method)
                    bits_str = ''.join(str(int(b)) for b in watermark_bits)
                    result["found"] = bool(bits_str)
                    result["method"] = "invisible_watermark:rivaGan"
                    result["bits"] = bits_str
                    result["matched"] = None
                    result["score"] = 0.5 if bits_str else 0.0
                    result["details"] = "rivaGan fallback: decode 32-bit watermark"
                    if bits_str:
                        return result
                except Exception as fallback_e:
                    result["details"] = f"blad {method} fallback32: {fallback_e}"
                    continue
            result["details"] = f"blad {method}: {e}"
            continue

    return result


# ---------------------------------------------------------------------------
# 3. NOISE RESIDUAL + FFT
# ---------------------------------------------------------------------------

def _compute_hf_ratio(frame_bgr: np.ndarray, cutoff_ratio: float = 0.30) -> float:
    """
    Oblicza high-frequency energy ratio:
        hf_ratio = energy(f > cutoff_ratio * nyquist) / total_energy
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = 0.5 * min(h, w)
    hf_mask = dist >= (cutoff_ratio * max_radius)

    total_energy = float(np.sum(magnitude))
    hf_energy = float(np.sum(magnitude[hf_mask])) if np.any(hf_mask) else 0.0
    return hf_energy / (total_energy + 1e-6)


def compute_freq_hf_ratio_mean(frames: List[np.ndarray], n_samples: int = 5) -> float:
    """Liczy mean hf_ratio na N równomiernie próbkowanych klatkach."""
    if not frames:
        return 0.0
    n = min(n_samples, len(frames))
    idxs = np.linspace(0, len(frames) - 1, num=n, dtype=int)
    ratios = []
    for i in idxs:
        try:
            ratios.append(_compute_hf_ratio(frames[i]))
        except Exception:
            continue
    return float(np.mean(ratios)) if ratios else 0.0


def detect_ai_noise_artifacts(
    frame_bgr: np.ndarray,
    fft_peak_threshold: float = 0.35,
    wiener_ksize: int = 5
) -> Dict[str, Any]:
    result = {"found": False, "method": "noise_residual_fft",
              "score": 0.0, "details": "", "fft_image": None, "freq_hf_ratio": 0.0}

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

    # High-frequency energy ratio dla biezacej klatki.
    hf_ratio = _compute_hf_ratio(frame_bgr, cutoff_ratio=0.30)
    result["freq_hf_ratio"] = hf_ratio

    vis = cv2.normalize(
        magnitude_no_center.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    result["fft_image"] = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)

    if ratio > fft_peak_threshold * 10:
        result["found"] = True
        result["score"] = min(1.0, (ratio - fft_peak_threshold * 10) / 50.0)
        result["details"] = (
            f"FFT peak ratio={ratio:.2f}, hf_ratio={hf_ratio:.3f}, mean_bg={mean_bg:.3f}, "
            f"max_peak={max_peak:.3f} – mozliwe artefakty AI upsamplingu"
        )
    else:
        result["details"] = f"FFT ratio={ratio:.2f}, hf_ratio={hf_ratio:.3f} – brak anomalii"

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
    of_scale: float = 0.5,
    log_fn=None
) -> Dict[str, Any]:
    """
    Zbiera klatki i uruchamia wszystkie zaawansowane metody detekcji.

    Args:
        of_scale: skalar zmniejszenia klatek przed OF (0.5 = 4x szybciej na CPU).
                  Automatyczna korekcja progów wewnątrz detect_optical_flow_overlay.
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
        "optical_flow_rois": [],
        "broadcast_traps": {
            "broadcast_trap": False,
            "lower_third_anim": False,
            "scoreboard_top_pair": False,
            "billboard_center_large": False,
        },
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
        cuda_str = "GPU" if _get_cuda() else "CPU"
        _log(f"[ADV] Sprawdzam Optical Flow (Farneback {cuda_str}, scale={of_scale}) + contour search...")
        try:
            of_rois = detect_optical_flow_overlay(frames, of_scale=of_scale)
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
            fft_res["freq_hf_ratio_mean"] = compute_freq_hf_ratio_mean(frames, n_samples=5)
            result["fft_artifacts"] = fft_res
            if fft_res["found"]:
                _log(f"[ADV] FFT artefakty: {fft_res['details']}")
            else:
                _log(f"[ADV] FFT: {fft_res['details']}")
            _log(f"[ADV] FFT hf_ratio_mean={fft_res.get('freq_hf_ratio_mean', 0.0):.4f}")
        except Exception as e:
            _log(f"[ADV] Blad FFT: {e}")

    try:
        trap_flags = detect_broadcast_trap_patterns(
            result.get("optical_flow_rois", []),
            result.get("zero_variance_rois", []),
        )
        result["broadcast_traps"] = trap_flags
        if trap_flags.get("broadcast_trap"):
            # Hard gate: przy trapie broadcast nie licz konturow center jako sygnalu AI.
            of_rois = result.get("optical_flow_rois", [])
            filtered_rois = [
                r for r in of_rois
                if not (
                    0.30 <= float(r.get("cx_rel", 0.0)) <= 0.70
                    and 0.30 <= float(r.get("cy_rel", 0.0)) <= 0.70
                )
            ]
            suppressed = len(of_rois) - len(filtered_rois)
            result["optical_flow_rois"] = filtered_rois
            _log(
                "[ADV] Broadcast trap detected: "
                f"lower_third={int(trap_flags.get('lower_third_anim', False))}, "
                f"scoreboard={int(trap_flags.get('scoreboard_top_pair', False))}, "
                f"billboard={int(trap_flags.get('billboard_center_large', False))}, "
                f"suppressed_center_rois={suppressed}"
            )
    except Exception as e:
        _log(f"[ADV] Blad broadcast trap patterns: {e}")

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

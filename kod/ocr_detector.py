"""
ocr_detector.py

Detekcja znakow wodnych / napisow AI w obrazach i wideo.
- OCR (RapidOCR / PaddleOCR / EasyOCR)
- Template Matching
- Corner ROI scanning  [Super-Resolution via super_resolution.upscale_for_ocr]
- Temporal Median Filtering + Zero-Variance ROI  [advanced_detectors]
- Invisible Watermark (imwatermark DWT/RivaGAN)  [advanced_detectors]
- FFT Noise Residual analysis                    [advanced_detectors]
- Optical Flow overlay detection                 [advanced_detectors]
"""

from __future__ import annotations

import os
import csv
import math
import sys
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable

import cv2
import numpy as np

import config
from advanced_detectors import run_advanced_scan
import super_resolution  # Super-Resolution przed OCR (EDSR/FSRCNN, fallback INTER_CUBIC)

_OCR_READER = None
_OCR_ENGINE_TYPE = None
_OCR_INIT_ERROR = None
_YOLO_MODEL = None
_OCR_LOCK = threading.Lock()

CORNER_RATIO = 0.15
# CORNER_SCALE uzywany tylko do przeliczania wspolrzednych bbox z powrotem na oryginalne
# Faktyczne powiekszenie ROI delegowane do super_resolution.upscale_for_ocr()
CORNER_SCALE = 4.0   # max skalar SR (x4) – uzyj do back-projekcji bbox
_SR_TARGET_MIN_DIM = 120  # minimalna dlugosc krotszego boku ROI podawanego do OCR

_LABEL_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE     = 0.55
_LABEL_THICKNESS = 1
_LABEL_PAD       = 3


class TextTracker:
    def __init__(self):
        self.history = {}

    def update(self, frame_idx, type_id, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if type_id not in self.history:
            self.history[type_id] = []
            status = "NOWY"
        else:
            self.history[type_id].sort(key=lambda x: x['frame'])
            closest = min(self.history[type_id], key=lambda x: abs(x['frame'] - frame_idx))
            dist = math.hypot(cx - closest['centroid'][0], cy - closest['centroid'][1])
            frames_diff = abs(frame_idx - closest['frame'])
            if frames_diff > 30:
                status = "POJAWIENIE"
            elif dist < 25:
                status = "STATYCZNY"
            else:
                status = "RUCHOMY"

        self.history[type_id].append({"frame": frame_idx, "centroid": (cx, cy), "bbox": bbox})
        return status


def _draw_label(img: np.ndarray, text: str, anchor_x: int, anchor_y: int,
                color=(0, 255, 0), used_rects: list = None) -> tuple:
    """
    Rysuje etykiete PONIZEJ punktu anchor (anchor_y = dolna krawedz bbox).
    """
    h_img, w_img = img.shape[:2]
    (tw, th), baseline = cv2.getTextSize(text, _LABEL_FONT, _LABEL_SCALE, _LABEL_THICKNESS)
    box_w = min(tw + 2 * _LABEL_PAD, w_img)
    box_h = th + baseline + 2 * _LABEL_PAD

    tx = max(0, min(anchor_x, w_img - box_w))

    ty_below = anchor_y + _LABEL_PAD + th
    if anchor_y + box_h > h_img:
        ty_below = anchor_y - _LABEL_PAD
    ty = max(th + _LABEL_PAD, min(ty_below, h_img - baseline - _LABEL_PAD))

    if used_rects is not None:
        for _ in range(40):
            rect = (tx, ty - th - _LABEL_PAD, tx + box_w, ty + baseline + _LABEL_PAD)
            overlap = any(
                not (rect[2] <= r[0] or rect[0] >= r[2] or rect[3] <= r[1] or rect[1] >= r[3])
                for r in used_rects
            )
            if not overlap:
                break
            ty += box_h + 2
            if ty > h_img:
                ty = h_img - _LABEL_PAD
                break

    bx1 = tx
    by1 = max(0, ty - th - _LABEL_PAD)
    bx2 = min(w_img, tx + box_w)
    by2 = min(h_img, ty + baseline + _LABEL_PAD)

    cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, text, (bx1 + _LABEL_PAD, ty), _LABEL_FONT,
                _LABEL_SCALE, color, _LABEL_THICKNESS, cv2.LINE_AA)

    if used_rects is not None:
        used_rects.append((bx1, by1, bx2, by2))
    return tx, ty


def reset_reader():
    global _OCR_READER, _OCR_ENGINE_TYPE, _OCR_INIT_ERROR
    with _OCR_LOCK:
        _OCR_READER = None
        _OCR_ENGINE_TYPE = None
        _OCR_INIT_ERROR = None


def warmup_reader(log_fn=None):
    global _OCR_READER, _OCR_ENGINE_TYPE, _OCR_INIT_ERROR

    def _log(msg):
        print(msg, file=sys.stderr)
        if log_fn:
            try:
                log_fn(msg)
            except Exception:
                pass

    with _OCR_LOCK:
        if _OCR_READER is not None:
            _log(f"[OCR] Engine juz zaladowany: {_OCR_ENGINE_TYPE}")
            return _OCR_ENGINE_TYPE, None

        _OCR_INIT_ERROR = None

        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
            _log("[OCR] Probuje RapidOCR (onnxruntime)...")
            _OCR_READER = RapidOCR()
            _OCR_ENGINE_TYPE = "rapid"
            _log("[OCR] RapidOCR zaladowany pomyslnie.")
            return "rapid", None
        except ImportError:
            _log("[OCR] RapidOCR niedostepny, probuje PaddleOCR...")
        except Exception as e:
            _log(f"[OCR] RapidOCR blad: {e}, probuje PaddleOCR...")

        try:
            from paddleocr import PaddleOCR  # type: ignore
            _log("[OCR] Probuje PaddleOCR...")
            _OCR_READER = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
            _OCR_ENGINE_TYPE = "paddle"
            _log("[OCR] PaddleOCR zaladowany pomyslnie.")
            return "paddle", None
        except ImportError:
            _log("[OCR] PaddleOCR niedostepny, probuje EasyOCR...")
        except Exception as e:
            _log(f"[OCR] PaddleOCR blad: {e}, probuje EasyOCR...")

        try:
            import easyocr  # type: ignore
            _log("[OCR] Inicjalizuje EasyOCR (moze chwile potrwac)...")
            _OCR_READER = easyocr.Reader(["en", "pl"], gpu=False, verbose=False)
            _OCR_ENGINE_TYPE = "easyocr"
            _log("[OCR] EasyOCR zaladowany pomyslnie.")
            return "easyocr", None
        except Exception as e:
            err = f"Wszystkie silniki OCR niedostepne. Ostatni blad (EasyOCR): {e}"
            _log(f"[OCR] BLAD KRYTYCZNY: {err}")
            _OCR_READER = None
            _OCR_ENGINE_TYPE = None
            _OCR_INIT_ERROR = err
            return None, err


def _get_reader():
    global _OCR_READER
    if _OCR_READER is not None:
        return _OCR_READER
    warmup_reader()
    return _OCR_READER


def get_init_error() -> Optional[str]:
    return _OCR_INIT_ERROR


def _get_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    path = getattr(config, "SORA_YOLO_MODEL_PATH", "") or ""
    model_id = getattr(config, "WATERMARK_YOLO_MODEL_ID", "") or ""
    if not path and not model_id:
        return None

    try:
        from ultralytics import YOLO  # type: ignore
        if path and os.path.exists(path):
            _YOLO_MODEL = YOLO(path)
        elif model_id:
            _YOLO_MODEL = YOLO(model_id)
        return _YOLO_MODEL
    except Exception:
        _YOLO_MODEL = None
        return None


def _detect_yolo_watermark(frame_bgr, min_conf: float) -> List[Tuple]:
    model = _get_yolo()
    if model is None:
        return []
    try:
        results = model(frame_bgr, verbose=False)
    except Exception:
        return []
    detections = []
    class_map = {0: "WATERMARK", 1: "SORA", 2: "OPENAI"}
    for r in results:
        if not hasattr(r, "boxes"):
            continue
        for b in r.boxes:
            try:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                if conf < min_conf:
                    continue
                label = class_map.get(cls_id, "WATERMARK")
                if hasattr(r, "names") and cls_id in r.names:
                    label = r.names[cls_id].upper()
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                detections.append((x1, y1, x2, y2, conf, label))
            except Exception:
                continue
    return detections


def _detect_template_watermarks(image_to_scan: np.ndarray, confidence: float) -> List[dict]:
    templates_dir = getattr(config, "TEMPLATES_DIR", "watermark_templates")
    if not os.path.exists(templates_dir):
        try:
            os.makedirs(templates_dir, exist_ok=True)
        except Exception:
            return []
    detections = []
    gray_frame = cv2.cvtColor(image_to_scan, cv2.COLOR_BGR2GRAY) if len(image_to_scan.shape) == 3 else image_to_scan
    for fname in os.listdir(templates_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        tpl = cv2.imread(os.path.join(templates_dir, fname), cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            continue
        try:
            res = cv2.matchTemplate(gray_frame, tpl, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= confidence)
            h, w = tpl.shape
            label = os.path.splitext(fname)[0].upper()
            for pt in zip(*loc[::-1]):
                x1, y1 = int(pt[0]), int(pt[1])
                detections.append({
                    "type": f"LOGO-{label}",
                    "confidence": float(res[y1, x1]),
                    "text": f"[IMG: {label}]",
                    "bbox": (x1, y1, x1 + w, y1 + h)
                })
        except Exception:
            pass
    return detections


def _preprocess_for_ocr(roi_bgr: np.ndarray) -> np.ndarray:
    try:
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    except Exception:
        return roi_bgr


def _get_advanced_filters(frame_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    filters = []
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
        filters.append(("AGGR-ADAPT", cv2.cvtColor(adapt, cv2.COLOR_GRAY2BGR)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        tophat = cv2.normalize(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel), None, 0, 255, cv2.NORM_MINMAX)
        filters.append(("AGGR-TOPHAT", cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR)))
        blackhat = cv2.normalize(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel), None, 0, 255, cv2.NORM_MINMAX)
        filters.append(("AGGR-BLACKHAT", cv2.cvtColor(blackhat, cv2.COLOR_GRAY2BGR)))
        gaussian = cv2.GaussianBlur(frame_bgr, (9, 9), 10.0)
        filters.append(("AGGR-SHARPEN", cv2.addWeighted(frame_bgr, 1.5, gaussian, -0.5, 0)))
        table = np.array([((i / 255.0) ** (1.0 / 0.4)) * 255 for i in range(256)]).astype("uint8")
        dark_inv_clahe = _preprocess_for_ocr(cv2.bitwise_not(cv2.LUT(frame_bgr, table)))
        filters.append(("AGGR-EXTREME-WHITE", dark_inv_clahe))
        bg = cv2.medianBlur(gray, 51)
        diff = cv2.normalize(cv2.absdiff(gray, bg), None, 0, 255, cv2.NORM_MINMAX)
        filters.append(("AGGR-BGSUB", cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)))
    except Exception:
        pass
    return filters


def _make_session_dir(input_path: str) -> str:
    filename_clean = os.path.basename(input_path).replace(".", "_")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = getattr(config, "REPORTS_BASE_DIR", "reports")
    out_dir = os.path.join(base, "watermarks", f"{filename_clean}_{timestamp_str}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _extract_corner_rois(frame_bgr: np.ndarray) -> List[Tuple[str, np.ndarray, int, int]]:
    """
    Wycina narozniki kadru i powiekasza je Super-Resolution (EDSR/FSRCNN)
    zamiast kubicznego resize. Jesli model SR niedostepny – automatyczny
    fallback do INTER_CUBIC wewnatrz super_resolution.upscale_for_ocr().

    Zwracany CORNER_SCALE odzwierciedla faktyczny uzyty skalar SR (2 lub 4)
    potrzebny do back-projekcji bbox na wspolrzedne oryginalnej klatki.
    """
    h, w = frame_bgr.shape[:2]
    ch = int(h * CORNER_RATIO)
    cw = int(w * CORNER_RATIO)
    corners = [
        ("CORNER-TL", frame_bgr[0:ch, 0:cw],        0,      0),
        ("CORNER-TR", frame_bgr[0:ch, w - cw:w],     w - cw, 0),
        ("CORNER-BL", frame_bgr[h - ch:h, 0:cw],     0,      h - ch),
        ("CORNER-BR", frame_bgr[h - ch:h, w - cw:w], w - cw, h - ch),
    ]
    result = []
    for name, roi, ox, oy in corners:
        if roi.size == 0:
            continue
        # Super-Resolution: automatycznie dobiera x2 lub x4 tak, zeby
        # krotszy bok mial co najmniej _SR_TARGET_MIN_DIM pikseli.
        # Jesli brak modelu SR -> fallback INTER_CUBIC (zero crashy).
        upscaled = super_resolution.upscale_for_ocr(roi, target_min_dim=_SR_TARGET_MIN_DIM)
        result.append((name, upscaled, ox, oy))
    return result


def _get_actual_corner_scale(roi_original: np.ndarray, roi_upscaled: np.ndarray) -> float:
    """
    Oblicza faktyczny skalar powiekszenia (szerokos upscaled / szerokos oryginalu).
    Uzywane do precyzyjnej back-projekcji wspolrzednych bbox na oryginalna klatke.
    """
    if roi_original.shape[1] == 0:
        return CORNER_SCALE
    return roi_upscaled.shape[1] / roi_original.shape[1]


def _corner_versions(roi_upscaled: np.ndarray) -> List[np.ndarray]:
    versions = [roi_upscaled]
    try:
        gray = cv2.cvtColor(roi_upscaled, cv2.COLOR_BGR2GRAY)
        versions.append(_preprocess_for_ocr(roi_upscaled))
        versions.append(cv2.bitwise_not(roi_upscaled))
        versions.append(cv2.bitwise_not(_preprocess_for_ocr(roi_upscaled)))
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        versions.append(cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR))
        versions.append(cv2.cvtColor(cv2.bitwise_not(norm), cv2.COLOR_GRAY2BGR))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        blackhat = cv2.normalize(
            cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel), None, 0, 255, cv2.NORM_MINMAX
        )
        versions.append(cv2.cvtColor(blackhat, cv2.COLOR_GRAY2BGR))
        tophat = cv2.normalize(
            cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel), None, 0, 255, cv2.NORM_MINMAX
        )
        versions.append(cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR))
        adapt = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        versions.append(cv2.cvtColor(adapt, cv2.COLOR_GRAY2BGR))
        versions.append(cv2.cvtColor(cv2.bitwise_not(adapt), cv2.COLOR_GRAY2BGR))
        blur = cv2.GaussianBlur(roi_upscaled, (3, 3), 0)
        versions.append(cv2.addWeighted(roi_upscaled, 2.0, blur, -1.0, 0))
    except Exception:
        pass
    return versions


def _ocr_on_image(image: np.ndarray, confidence: float, keywords: List[str]) -> List[Tuple]:
    reader = _get_reader()
    if reader is None:
        return []

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    raw_results = []
    try:
        if _OCR_ENGINE_TYPE == "rapid":
            result, _ = reader(image)
            if result:
                for item in result:
                    raw_results.append((item[0], item[1], float(item[2]) if item[2] is not None else 0.0))
        elif _OCR_ENGINE_TYPE == "paddle":
            raw = reader.ocr(image, cls=False)
            if raw and raw[0]:
                for line in raw[0]:
                    raw_results.append((line[0], line[1][0], line[1][1]))
        else:
            raw_results = reader.readtext(image)
    except Exception:
        return []

    matches = []
    for (bbox, text, prob) in raw_results:
        if float(prob) < confidence:
            continue
        t_clean = str(text).upper().replace(" ", "")
        for k in keywords:
            if k.replace(" ", "") in t_clean:
                bbox_norm = _normalize_bbox(bbox)
                matches.append((bbox_norm, text, float(prob), k))
                break
    return matches


def _normalize_bbox(bbox) -> list:
    try:
        pts = list(bbox)
        if len(pts) == 4:
            result = []
            for p in pts:
                if hasattr(p, '__len__') and len(p) >= 2:
                    result.append([float(p[0]), float(p[1])])
                else:
                    result.append([float(p), 0.0])
            return result
        return [[0, 0], [10, 0], [10, 10], [0, 10]]
    except Exception:
        return [[0, 0], [10, 0], [10, 10], [0, 10]]


def _perform_scan(
    frame_original: np.ndarray,
    confidence: float,
    keywords: List[str],
    versions_to_scan: List[Tuple[str, np.ndarray]],
    scale_factor: float = 1.0
) -> List[dict]:
    frame_detections = []
    found_keys: set = set()

    for (x1, y1, x2, y2, conf, label) in _detect_yolo_watermark(frame_original, confidence):
        frame_detections.append({"type": label, "confidence": conf, "text": f"[{label}]",
                                  "bbox": (x1, y1, x2, y2), "source": "YOLO"})
        found_keys.add(label)

    for source_name, base_image in versions_to_scan:
        for det in _detect_template_watermarks(base_image, confidence):
            if det["type"] not in found_keys:
                det["source"] = f"TEMPLATE-{source_name}"
                frame_detections.append(det)
                found_keys.add(det["type"])

        img = cv2.resize(base_image, None, fx=scale_factor, fy=scale_factor,
                         interpolation=cv2.INTER_CUBIC) if scale_factor != 1.0 else base_image

        for (bbox, text, prob, kw) in _ocr_on_image(img, confidence, keywords):
            if kw in found_keys:
                continue
            x1 = int(bbox[0][0] / scale_factor)
            y1 = int(bbox[0][1] / scale_factor)
            x2 = int(bbox[2][0] / scale_factor)
            y2 = int(bbox[2][1] / scale_factor)
            frame_detections.append({"type": kw, "confidence": prob, "text": str(text).upper(),
                                      "bbox": (x1, y1, x2, y2), "source": source_name})
            found_keys.add(kw)

    h_orig, w_orig = frame_original.shape[:2]
    corner_confidence = max(0.25, confidence - 0.25)

    for (corner_name, roi_upscaled, ox, oy) in _extract_corner_rois(frame_original):
        # Oblicz faktyczny skalar SR dla tej konkretnej ROI
        orig_h = int(h_orig * CORNER_RATIO)
        orig_w = int(w_orig * CORNER_RATIO)
        actual_scale_x = roi_upscaled.shape[1] / max(orig_w, 1)
        actual_scale_y = roi_upscaled.shape[0] / max(orig_h, 1)

        for rv in _corner_versions(roi_upscaled):
            for (bbox, text, prob, kw) in _ocr_on_image(rv, corner_confidence, keywords):
                if kw in found_keys:
                    continue
                x1 = ox + int(bbox[0][0] / actual_scale_x)
                y1 = oy + int(bbox[0][1] / actual_scale_y)
                x2 = ox + int(bbox[2][0] / actual_scale_x)
                y2 = oy + int(bbox[2][1] / actual_scale_y)
                x1, x2 = max(0, x1), min(w_orig, x2)
                y1, y2 = max(0, y1), min(h_orig, y2)
                frame_detections.append({"type": kw, "confidence": prob, "text": str(text).upper(),
                                          "bbox": (x1, y1, x2, y2), "source": corner_name})
                found_keys.add(kw)

    return frame_detections


def _annotate_frame(frame: np.ndarray, detections: List[dict], tracker: 'TextTracker',
                    frame_idx: int, aggr: bool = False) -> Dict[str, str]:
    used_rects: List[Tuple] = []
    motion_map: Dict[str, str] = {}

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        motion = tracker.update(frame_idx, det['type'], det['bbox'])
        motion_map[det['type']] = motion
        conf_pct = int(det['confidence'] * 100)
        label = f"{det['type']} {conf_pct}%" + ("*" if aggr else "")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _draw_label(frame, label, anchor_x=x1, anchor_y=y2,
                    color=(0, 255, 0), used_rects=used_rects)

    return motion_map


def scan_for_watermarks(
    media_path: str,
    check_stop=None,
    progress_callback=None,
    confidence: float = 0.6,
    sample_rate: int = 30,
    detailed_scan: bool = False,
    preview_callback: Optional[Callable[[np.ndarray, list], None]] = None
) -> Dict[str, Any]:

    if not isinstance(sample_rate, int) or sample_rate <= 0:
        sample_rate = 1

    if _OCR_READER is None:
        warmup_reader()
    if _OCR_READER is None:
        return {
            "status": "ERROR",
            "error": f"OCR reader niedostepny: {_OCR_INIT_ERROR or 'nieznany blad'}",
            "watermark_found": False,
            "watermark_count": 0,
            "watermark_types": [],
        }

    is_video = os.path.splitext(media_path)[1].lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    cap = cv2.VideoCapture(os.path.abspath(media_path))
    if not cap.isOpened():
        return {"status": "ERROR", "error": "Nie mozna otworzyc pliku."}

    default_keywords = [
        "SORA", "OPENAI", "GENERATED", "AI VIDEO", "MADE WITH", "AI GENERATED",
        "RUNWAY", "PIKA", "LUMA", "GEN-2", "GEN-3", "GEN-4", "GEN-5",
        "GEN3", "GEN4", "GEN5", "GEN 3", "GEN 4", "GEN 5",
        "TIKTOK", "KWAI", "CAPCUT", "STABLE VIDEO",
        "KLING", "VEED", "INVIDEO", "KAPWING", "SYNTHID", "MINIMAX", "HAIPER", "DREAMLUX"
    ]
    keywords = [str(k).upper() for k in getattr(config, "WATERMARK_KEYWORDS", default_keywords)]

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else 1

    out_dir = _make_session_dir(media_path)
    csv_path = os.path.join(out_dir, "report.csv")

    saved_paths: List[str] = []
    missed_frames: List[int] = []

    frame_idx = 0
    detections_count = 0
    found_types: set = set()
    tracker = TextTracker()

    # ----------------------------------------------------------------
    # FAZA 0: Zaawansowana analiza
    # ----------------------------------------------------------------
    advanced_results: Dict[str, Any] = {}
    if is_video:
        try:
            def _adv_log(msg):
                print(msg, file=sys.stderr)

            advanced_results = run_advanced_scan(
                cap=cap,
                fps=fps,
                total_frames=total_frames,
                n_frames_median=40,
                check_invisible=True,
                check_fft=True,
                check_optical_flow=True,
                log_fn=_adv_log
            )
            if advanced_results.get("temporal_median_frame") is not None:
                cv2.imwrite(
                    os.path.join(out_dir, "temporal_median.jpg"),
                    advanced_results["temporal_median_frame"]
                )
            if advanced_results.get("overlay_diff") is not None:
                cv2.imwrite(
                    os.path.join(out_dir, "overlay_diff.jpg"),
                    advanced_results["overlay_diff"]
                )
            if advanced_results.get("fft_artifacts", {}).get("fft_image") is not None:
                cv2.imwrite(
                    os.path.join(out_dir, "fft_noise.jpg"),
                    advanced_results["fft_artifacts"]["fft_image"]
                )
        except Exception as e:
            print(f"[ADV] Blad analizy zaawansowanej: {e}", file=sys.stderr)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Plik", "Typ", "Klatka", "Timestamp", "Typ watermarku",
                         "Confidence", "Tekst", "Ruch", "Zrodlo", "Zapisany plik"])

        if advanced_results:
            if advanced_results.get("invisible_wm", {}).get("found"):
                iw = advanced_results["invisible_wm"]
                writer.writerow([
                    os.path.basename(media_path), "INVISIBLE_WM", "-", "-",
                    iw.get("matched", "UNKNOWN"), f"{iw.get('score', 0):.2f}",
                    iw.get("bits", "")[:32], "-", iw.get("method", ""), "-"
                ])
                found_types.add("INVISIBLE_WM")
                detections_count += 1

            for roi in advanced_results.get("zero_variance_rois", []):
                writer.writerow([
                    os.path.basename(media_path), "STATIC_OVERLAY", "-", "-",
                    roi["name"], f"{roi['score']:.2f}",
                    "zero-variance ROI", "-", "temporal_analysis", "-"
                ])
                found_types.add("STATIC_OVERLAY")
                detections_count += 1

            for roi in advanced_results.get("optical_flow_rois", []):
                writer.writerow([
                    os.path.basename(media_path), "OF_STATIC_OVERLAY", "-", "-",
                    roi["name"], f"{roi['score']:.2f}",
                    f"optical_flow global_motion={roi.get('global_motion', 0):.2f}px",
                    "-", "optical_flow_farneback", "-"
                ])
                found_types.add("OF_STATIC_OVERLAY")
                detections_count += 1

            if advanced_results.get("fft_artifacts", {}).get("found"):
                fa = advanced_results["fft_artifacts"]
                writer.writerow([
                    os.path.basename(media_path), "FFT_ARTIFACT", "-", "-",
                    "AI_UPSAMPLE", f"{fa.get('score', 0):.2f}",
                    fa.get("details", ""), "-", "fft_noise_residual", "-"
                ])
                found_types.add("FFT_ARTIFACT")
                detections_count += 1

        median_frame = advanced_results.get("temporal_median_frame")
        overlay_diff  = advanced_results.get("overlay_diff")

        # ----------------------------------------------------------------
        # FAZA 1: Glowna petla OCR
        # ----------------------------------------------------------------
        while True:
            if check_stop and check_stop():
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if progress_callback and is_video and frame_idx % 10 == 0:
                progress_callback(frame_idx, total_frames)

            if is_video and frame_idx % sample_rate != 0 and frame_idx != 1:
                continue

            now_sec = frame_idx / float(fps) if is_video else 0.0

            table = np.array([((i / 255.0) ** (1.0 / 0.5)) * 255 for i in range(256)]).astype("uint8")
            darkened = cv2.LUT(frame, table)

            versions_to_scan = [
                ("OCR-RAW",   frame),
                ("OCR-CLAHE", _preprocess_for_ocr(frame)),
                ("OCR-INV",   cv2.bitwise_not(frame)),
                ("OCR-DARK",  darkened),
            ]
            if median_frame is not None:
                versions_to_scan.append(("OCR-TEMPORAL-MEDIAN", median_frame))
            if overlay_diff is not None:
                versions_to_scan.append(("OCR-OVERLAY-DIFF", overlay_diff))

            frame_detections = _perform_scan(frame, confidence, keywords, versions_to_scan, scale_factor=1.0)

            if not frame_detections:
                missed_frames.append(frame_idx)

            frame_to_draw = frame.copy()
            motion_map = _annotate_frame(frame_to_draw, frame_detections, tracker, frame_idx, aggr=False)

            for det in frame_detections:
                found_types.add(det['type'])
                detections_count += 1
                fname = f"frame_{frame_idx}_t_{now_sec:.2f}s.jpg"
                save_path = os.path.join(out_dir, fname)
                writer.writerow([
                    os.path.basename(media_path), "Video" if is_video else "Image",
                    frame_idx, f"{now_sec:.2f}", det['type'],
                    f"{det['confidence']:.2f}", det['text'],
                    motion_map.get(det['type'], ""),
                    det.get('source', ''), save_path
                ])
                try:
                    cv2.imwrite(save_path, frame_to_draw)
                    saved_paths.append(save_path)
                except Exception:
                    pass

            if preview_callback:
                if not frame_detections:
                    cv2.putText(frame_to_draw, f"Brak detekcji (klatka {frame_idx})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                preview_callback(frame_to_draw, frame_detections)

        # ----------------------------------------------------------------
        # FAZA 2: Szczegolowa analiza missed frames
        # ----------------------------------------------------------------
        if detailed_scan and missed_frames and not (check_stop and check_stop()):
            for i, m_idx in enumerate(missed_frames):
                if check_stop and check_stop():
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, m_idx - 1)
                ok, frame = cap.read()
                if not ok:
                    continue

                if progress_callback:
                    progress_callback(i + 1, len(missed_frames))

                now_sec = m_idx / float(fps)
                aggr_versions = _get_advanced_filters(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                aggr_versions.append(("AGGR-BW",   cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)))
                aggr_versions.append(("AGGR-ORIG", frame))
                if median_frame is not None:
                    aggr_versions.append(("AGGR-TEMPORAL", median_frame))
                if overlay_diff is not None:
                    aggr_versions.append(("AGGR-DIFF", overlay_diff))

                frame_detections = _perform_scan(frame, confidence, keywords, aggr_versions, scale_factor=2.0)

                if frame_detections:
                    frame_to_draw = frame.copy()
                    motion_map = _annotate_frame(frame_to_draw, frame_detections, tracker, m_idx, aggr=True)

                    for det in frame_detections:
                        found_types.add(det['type'])
                        detections_count += 1
                        writer.writerow([
                            os.path.basename(media_path), "Video (Aggr)" if is_video else "Image (Aggr)",
                            m_idx, f"{now_sec:.2f}", det['type'],
                            f"{det['confidence']:.2f}", det['text'],
                            motion_map.get(det['type'], ""),
                            det.get('source', ''), ""
                        ])

                    save_path = os.path.join(out_dir, f"frame_{m_idx}_aggr_t_{now_sec:.2f}s.jpg")
                    try:
                        cv2.imwrite(save_path, frame_to_draw)
                        saved_paths.append(save_path)
                    except Exception:
                        pass

                    if preview_callback:
                        preview_callback(frame_to_draw, frame_detections)
                else:
                    if preview_callback:
                        tmp = frame.copy()
                        cv2.putText(tmp, f"Brak detekcji (AGGR klatka {m_idx})",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                        preview_callback(tmp, [])

    cap.release()

    return {
        "status": "OK",
        "watermark_found": detections_count > 0,
        "watermark_types": list(found_types),
        "watermark_count": detections_count,
        "watermark_folder": out_dir,
        "csv_path": csv_path,
        "watermark_frames": saved_paths,
        "advanced": advanced_results,
    }

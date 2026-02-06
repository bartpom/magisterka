"""
ocr_detector.py

Detekcja znaków wodnych / napisów „generatora” w wideo.
Wersja odporna na zawieszanie GUI:
- EasyOCR i YOLO są ładowane leniwie (dopiero podczas analizy).
- Zwracany jest słownik wyników spójny z GUI.

Wymagania (opcjonalne):
- easyocr
- ultralytics (jeśli używasz YOLO)
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import cv2

import config

# Lazy singletons
_OCR_READER = None
_YOLO_MODEL = None


def _get_reader():
    global _OCR_READER
    if _OCR_READER is not None:
        return _OCR_READER
    try:
        import easyocr  # type: ignore

        print(" [OCR] Ładowanie modelu EasyOCR…")
        _OCR_READER = easyocr.Reader(["en", "pl"], gpu=False)
        print(" [OCR] Model EasyOCR gotowy.")
    except Exception as e:
        print(f" [OCR] EasyOCR niedostępny: {e}")
        _OCR_READER = None
    return _OCR_READER


def _get_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    path = getattr(config, "SORA_YOLO_MODEL_PATH", "") or ""
    model_id = getattr(config, "WATERMARK_YOLO_MODEL_ID", "") or ""

    if not path and not model_id:
        _YOLO_MODEL = None
        return None

    try:
        from ultralytics import YOLO  # type: ignore

        if path and os.path.exists(path):
            print(f" [OCR] Ładowanie modelu YOLO watermark (lokalnie): {path}")
            _YOLO_MODEL = YOLO(path)
            print(" [OCR] Model YOLO watermark gotowy.")
            return _YOLO_MODEL

        if model_id:
            print(f" [OCR] Ładowanie modelu YOLO watermark (ID): {model_id}")
            _YOLO_MODEL = YOLO(model_id)
            print(" [OCR] Model YOLO watermark gotowy.")
            return _YOLO_MODEL

    except Exception as e:
        print(f" [OCR] Nie udało się załadować YOLO: {e}")
        _YOLO_MODEL = None
        return None

    _YOLO_MODEL = None
    return None


def _detect_yolo_watermark(frame_bgr) -> Tuple[Optional[str], List[Tuple[int, int, int, int, float]]]:
    model = _get_yolo()
    if model is None:
        return None, []

    try:
        results = model(frame_bgr, verbose=False)
    except Exception:
        return None, []

    detections: List[Tuple[int, int, int, int, float]] = []
    label_name = None

    class_map = {0: "WATERMARK", 1: "SORA", 2: "OPENAI"}

    for r in results:
        if not hasattr(r, "boxes"):
            continue
        for b in r.boxes:
            try:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                if conf < 0.5:
                    continue
                label_name = class_map.get(cls_id, "WATERMARK")
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                detections.append((x1, y1, x2, y2, conf))
            except Exception:
                continue

    return label_name, detections


def _make_session_dir(input_path: str) -> str:
    filename_clean = os.path.basename(input_path).replace(".", "_")
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_folder_name = f"{filename_clean}_{timestamp_str}"
    base = getattr(config, "WATERMARK_BASE_DIR", "suspicious_frames")
    out_dir = os.path.join(base, session_folder_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _normalize_text_for_match(s: str) -> str:
    s = s.upper()
    s = re.sub(r"\s+", " ", s).strip()
    # usuń część znaków, które OCR lubi wstawiać przypadkowo
    s = re.sub(r"[^A-Z0-9 ]+", "", s)
    return s


def _match_keyword(text_u: str, keywords_u: List[str], *, fuzzy_ratio: float) -> Optional[str]:
    """Zwraca dopasowane keyword (UPPER) albo None."""
    if not text_u:
        return None

    # fast path: substring
    for k in keywords_u:
        if k and k in text_u:
            return k

    # fuzzy path: tolerancja literówek / złamań
    if fuzzy_ratio <= 0.0:
        return None

    for k in keywords_u:
        if not k:
            continue
        # porównuj też wersję bez spacji (np. "AIGENERATED")
        t0 = text_u.replace(" ", "")
        k0 = k.replace(" ", "")
        r = SequenceMatcher(None, t0, k0).ratio()
        if r >= fuzzy_ratio:
            return k

    return None


def _preprocess_roi_for_ocr(roi_bgr):
    """Lekki preprocessing, który zwykle poprawia OCR na półprzezroczystych watermarkach."""
    try:
        if roi_bgr is None or roi_bgr.size == 0:
            return roi_bgr

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # powiększ, żeby OCR miało łatwiej (szczególnie na małych napisach)
        h, w = gray.shape[:2]
        scale = 2 if max(h, w) < 900 else 1
        if scale > 1:
            gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        # kontrast (CLAHE)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            pass

        # odszumienie + threshold
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        thr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5
        )

        return thr
    except Exception:
        return roi_bgr


def _iter_rois(frame_bgr):
    """Zwraca ROI + offsety (x_off, y_off) dla mapowania bbox na pełny obraz."""
    h, w = frame_bgr.shape[:2]

    # dotychczasowe paski (góra/dół)
    yield frame_bgr[int(h * 0.75):h, 0:w], 0, int(h * 0.75)
    yield frame_bgr[0:int(h * 0.20), 0:w], 0, 0

    # dodatkowe rogi (często watermark jest w rogu)
    cw = int(w * 0.35)
    ch = int(h * 0.30)
    yield frame_bgr[0:ch, 0:cw], 0, 0
    yield frame_bgr[0:ch, w - cw:w], w - cw, 0
    yield frame_bgr[h - ch:h, 0:cw], 0, h - ch
    yield frame_bgr[h - ch:h, w - cw:w], w - cw, h - ch


def scan_for_watermarks(video_path: str, check_stop=None, progress_callback=None) -> Dict[str, Any]:
    cap = cv2.VideoCapture(os.path.abspath(video_path))
    if not cap.isOpened():
        return {
            "status": "ERROR",
            "watermark_found": False,
            "watermark_label": None,
            "watermark_score": 0.0,
            "watermark_folder": None,
            "watermark_frames": [],
            "watermark_hits": {},
        }

    keywords = [str(k).upper() for k in getattr(config, "WATERMARK_KEYWORDS", [])]
    max_frames = int(getattr(config, "WATERMARK_MAX_FRAMES", 600))
    stride = int(getattr(config, "WATERMARK_STRIDE", 10))
    min_gap = float(getattr(config, "WATERMARK_MIN_SAVE_GAP_SEC", 1.0))

    min_prob = float(getattr(config, "WATERMARK_OCR_MIN_PROB", 0.40))
    fuzzy_ratio = float(getattr(config, "WATERMARK_FUZZY_RATIO", 0.86))
    min_hits = int(getattr(config, "WATERMARK_MIN_HITS", 2))
    early_exit = bool(getattr(config, "WATERMARK_EARLY_EXIT", True))

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_dir = None
    saved_paths: List[str] = []
    found_label: Optional[str] = None
    last_save_time = -999.0

    hit_counts: Dict[str, int] = {}

    frame_idx = 0
    while True:
        if check_stop and check_stop():
            cap.release()
            return {
                "status": "STOPPED",
                "watermark_found": bool(found_label),
                "watermark_label": found_label,
                "watermark_score": 100.0 if found_label else 0.0,
                "watermark_folder": out_dir,
                "watermark_frames": saved_paths,
                "watermark_hits": hit_counts,
            }

        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        if frame_idx > max_frames:
            break

        if progress_callback and frame_idx % 10 == 0:
            progress_callback(frame_idx, max_frames)

        if frame_idx % stride != 0:
            continue

        now_sec = frame_idx / float(fps or 30.0)
        frame_to_draw = frame.copy()
        detected_this_frame: Optional[str] = None

        # 1) YOLO watermark (jeśli skonfigurowany)
        yolo_label, yolo_boxes = _detect_yolo_watermark(frame)
        if yolo_label and yolo_boxes:
            detected_this_frame = yolo_label
            found_label = found_label or yolo_label
            hit_counts[yolo_label] = hit_counts.get(yolo_label, 0) + 1
            for (x1, y1, x2, y2, conf) in yolo_boxes:
                cv2.rectangle(frame_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    frame_to_draw,
                    f"{yolo_label} ({int(conf * 100)}%)",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        # 2) OCR – ROI (jeśli dostępny EasyOCR)
        if detected_this_frame is None:
            reader = _get_reader()
            if reader is not None:
                for roi, x_off, y_off in _iter_rois(frame):
                    roi_pp = _preprocess_roi_for_ocr(roi)
                    try:
                        results = reader.readtext(roi_pp)
                    except Exception:
                        results = []

                    for (bbox, text, prob) in results:
                        if float(prob) < min_prob:
                            continue

                        t = _normalize_text_for_match(str(text))
                        matched = _match_keyword(t, keywords, fuzzy_ratio=fuzzy_ratio)
                        if matched:
                            detected_this_frame = matched
                            found_label = found_label or matched
                            hit_counts[matched] = hit_counts.get(matched, 0) + 1

                            # bbox -> prostokąt (offsety ROI)
                            try:
                                x1 = int(bbox[0][0]) + x_off
                                y1 = int(bbox[0][1]) + y_off
                                x2 = int(bbox[2][0]) + x_off
                                y2 = int(bbox[2][1]) + y_off
                                cv2.rectangle(frame_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                cv2.putText(
                                    frame_to_draw,
                                    f"{matched} ({int(float(prob) * 100)}%)",
                                    (x1, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 0),
                                    2,
                                )
                            except Exception:
                                pass

                            break
                    if detected_this_frame:
                        break

        # zapis klatki dowodowej
        if detected_this_frame:
            if (now_sec - last_save_time) >= min_gap:
                if out_dir is None:
                    out_dir = _make_session_dir(video_path)

                fname = f"frame_{frame_idx}_t_{int(now_sec)}s.jpg"
                save_path = os.path.join(out_dir, fname)
                try:
                    cv2.imwrite(save_path, frame_to_draw)
                    saved_paths.append(save_path)
                    last_save_time = now_sec
                except Exception:
                    pass

        # early exit: jeśli mamy stabilne trafienia, nie marnuj czasu na kolejne OCR
        if early_exit and found_label and hit_counts.get(found_label, 0) >= max(1, min_hits) and saved_paths:
            break

    cap.release()

    watermark_found = bool(found_label)
    # stabilizacja: wymagaj min_hits dla labela (jeśli ustawione)
    if watermark_found and min_hits > 1 and found_label:
        if hit_counts.get(found_label, 0) < min_hits:
            watermark_found = False
            found_label = None

    return {
        "status": "OK",
        "watermark_found": bool(watermark_found),
        "watermark_label": found_label,
        "watermark_score": 100.0 if watermark_found else 0.0,
        "watermark_folder": out_dir,
        "watermark_frames": saved_paths,
        "watermark_hits": hit_counts,
    }


# kompatybilność ze starszymi nazwami
def scan_for_watermark(video_path: str, check_stop=None, progress_callback=None):
    res = scan_for_watermarks(video_path, check_stop=check_stop, progress_callback=progress_callback)
    return res.get("watermark_label"), res.get("watermark_frames", [])

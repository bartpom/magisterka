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
from datetime import datetime
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
        }

    keywords = [str(k).upper() for k in getattr(config, "WATERMARK_KEYWORDS", [])]
    max_frames = int(getattr(config, "WATERMARK_MAX_FRAMES", 600))
    stride = int(getattr(config, "WATERMARK_STRIDE", 10))
    min_gap = float(getattr(config, "WATERMARK_MIN_SAVE_GAP_SEC", 1.0))

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_dir = None
    saved_paths: List[str] = []
    found_label: Optional[str] = None
    last_save_time = -999.0

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
        h, w = frame.shape[:2]
        frame_to_draw = frame.copy()
        detected_this_frame = None

        # 1) YOLO watermark (jeśli skonfigurowany)
        yolo_label, yolo_boxes = _detect_yolo_watermark(frame)
        if yolo_label and yolo_boxes:
            detected_this_frame = yolo_label
            found_label = found_label or yolo_label
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

        # 2) OCR – paski tekstowe (jeśli dostępny EasyOCR)
        if detected_this_frame is None:
            reader = _get_reader()
            if reader is not None:
                strips = [
                    (frame[int(h * 0.75):h, 0:w], int(h * 0.75)),  # dół
                    (frame[0:int(h * 0.20), 0:w], 0),              # góra
                ]
                for roi, y_off in strips:
                    try:
                        results = reader.readtext(roi)
                    except Exception:
                        results = []
                    for (bbox, text, prob) in results:
                        if float(prob) < 0.40:
                            continue
                        t = str(text).upper().strip()
                        for k in keywords:
                            if k and k in t:
                                detected_this_frame = k
                                found_label = found_label or k

                                # bbox -> prostokąt
                                try:
                                    x1 = int(bbox[0][0])
                                    y1 = int(bbox[0][1]) + y_off
                                    x2 = int(bbox[2][0])
                                    y2 = int(bbox[2][1]) + y_off
                                    cv2.rectangle(frame_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                    cv2.putText(
                                        frame_to_draw,
                                        f"{k} ({int(float(prob) * 100)}%)",
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

    cap.release()

    return {
        "status": "OK",
        "watermark_found": bool(found_label),
        "watermark_label": found_label,
        "watermark_score": 100.0 if found_label else 0.0,
        "watermark_folder": out_dir,
        "watermark_frames": saved_paths,
    }


# kompatybilność ze starszymi nazwami
def scan_for_watermark(video_path: str, check_stop=None, progress_callback=None):
    res = scan_for_watermarks(video_path, check_stop=check_stop, progress_callback=progress_callback)
    return res.get("watermark_label"), res.get("watermark_frames", [])
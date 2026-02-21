"""
ocr_detector.py

Detekcja znaków wodnych / napisów „generatora” w obrazach i wideo.
Dostosowane do wytycznych:
- Zapis CSV z detekcjami (Plik, Typ, Numer klatki, Timestamp, Typ watermarku, Confidence, Tekst, Ścieżka).
- Konfiguracja progu pewności (confidence) oraz próbkowania (sample_rate).
- PaddleOCR (szybszy) + EasyOCR (fallback).
"""

from __future__ import annotations

import os
import csv
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import config

# Lazy singletons
_OCR_READER = None
_OCR_ENGINE_TYPE = None  # "paddle" lub "easyocr"
_YOLO_MODEL = None


def _get_reader():
    global _OCR_READER, _OCR_ENGINE_TYPE
    if _OCR_READER is not None:
        return _OCR_READER

    try:
        from paddleocr import PaddleOCR  # type: ignore
        print(" [OCR] Ładowanie modelu PaddleOCR...")
        _OCR_READER = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
        _OCR_ENGINE_TYPE = "paddle"
        return _OCR_READER
    except ImportError:
        pass

    try:
        import easyocr  # type: ignore
        print(" [OCR] Ładowanie modelu EasyOCR...")
        _OCR_READER = easyocr.Reader(["en", "pl"], gpu=False)
        _OCR_ENGINE_TYPE = "easyocr"
        return _OCR_READER
    except Exception as e:
        print(f" [OCR] Błąd ładowania OCR: {e}")
        _OCR_READER = None
        _OCR_ENGINE_TYPE = None

    return _OCR_READER


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

        if _YOLO_MODEL is not None and hasattr(_YOLO_MODEL, "set_classes"):
            _YOLO_MODEL.set_classes(["watermark", "tiktok logo", "ai generated logo", "brand mark"])
        return _YOLO_MODEL
    except Exception:
        _YOLO_MODEL = None
        return None


def _detect_yolo_watermark(frame_bgr, min_conf: float) -> List[Tuple[int, int, int, int, float, str]]:
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
                
                label_name = class_map.get(cls_id, "WATERMARK")
                if hasattr(r, "names") and isinstance(r.names, dict) and cls_id in r.names:
                    label_name = r.names[cls_id].upper()

                x1, y1, x2, y2 = map(int, b.xyxy[0])
                detections.append((x1, y1, x2, y2, conf, label_name))
            except Exception:
                continue

    return detections


def _preprocess_for_ocr(roi_bgr: np.ndarray) -> np.ndarray:
    try:
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except Exception:
        return roi_bgr


def _make_session_dir(input_path: str) -> str:
    filename_clean = os.path.basename(input_path).replace(".", "_")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder_name = f"{filename_clean}_{timestamp_str}"
    base = getattr(config, "REPORTS_BASE_DIR", "reports")
    out_dir = os.path.join(base, "watermarks", session_folder_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def scan_for_watermarks(
    media_path: str, 
    check_stop=None, 
    progress_callback=None, 
    confidence: float = 0.6, 
    sample_rate: int = 30
) -> Dict[str, Any]:
    
    is_video = os.path.splitext(media_path)[1].lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    cap = cv2.VideoCapture(os.path.abspath(media_path))
    
    if not cap.isOpened():
        return {"status": "ERROR", "error": "Nie można otworzyć pliku."}

    default_keywords = [
        "SORA", "OPENAI", "GENERATED", "AI VIDEO", "MADE WITH", "AI GENERATED", 
        "RUNWAY", "PIKA", "LUMA", "GEN-2", "TIKTOK", "KWAI", "CAPCUT", "STABLE VIDEO"
    ]
    keywords = [str(k).upper() for k in getattr(config, "WATERMARK_KEYWORDS", default_keywords)]
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else 1

    out_dir = _make_session_dir(media_path)
    csv_path = os.path.join(out_dir, "report.csv")
    
    csv_rows = []
    saved_paths: List[str] = []
    
    frame_idx = 0
    detections_count = 0
    found_types = set()

    # Otwórz plik CSV do zapisu strumieniowego
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Nagłówki z pliku PDF
        csv_writer.writerow(["Plik", "Typ", "Numer klatki", "Timestamp", "Typ watermarku", "Confidence", "Tekst", "Ścieżka zapisu"])

        while True:
            if check_stop and check_stop():
                break

            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1

            if progress_callback and is_video and frame_idx % 10 == 0:
                progress_callback(frame_idx, total_frames)

            # Pomijanie klatek w wideo
            if is_video and frame_idx % sample_rate != 0 and frame_idx != 1:
                continue

            now_sec = frame_idx / float(fps) if is_video else 0.0
            h, w = frame.shape[:2]
            frame_to_draw = frame.copy()
            
            frame_detections = []

            # 1) YOLO watermark
            yolo_boxes = _detect_yolo_watermark(frame, min_conf=confidence)
            for (x1, y1, x2, y2, conf, label) in yolo_boxes:
                frame_detections.append({
                    "type": label,
                    "confidence": conf,
                    "text": f"[{label}]",
                    "bbox": (x1, y1, x2, y2),
                    "source": "YOLO"
                })

            # 2) OCR (Smart Corners + CLAHE)
            reader = _get_reader()
            if reader is not None:
                w_margin = int(w * 0.35)
                h_margin = int(h * 0.25)
                
                smart_corners = [
                    (frame[0:h_margin, 0:w_margin], 0, 0),                                # TL
                    (frame[0:h_margin, w - w_margin:w], w - w_margin, 0),                 # TR
                    (frame[h - h_margin:h, 0:w_margin], 0, h - h_margin),                 # BL
                    (frame[h - h_margin:h, w - w_margin:w], w - w_margin, h - h_margin),  # BR
                ]

                for roi, x_off, y_off in smart_corners:
                    if roi.size == 0: continue
                    enhanced_roi = _preprocess_for_ocr(roi)

                    try:
                        if _OCR_ENGINE_TYPE == "paddle":
                            results = reader.ocr(enhanced_roi, cls=False)
                            parsed_results = []
                            if results and results[0] is not None:
                                for line in results[0]:
                                    parsed_results.append((line[0], line[1][0], line[1][1]))
                        else:
                            parsed_results = reader.readtext(enhanced_roi)
                    except Exception:
                        parsed_results = []

                    for (bbox, text, prob) in parsed_results:
                        if float(prob) < confidence:
                            continue
                        
                        t = str(text).upper()
                        t_clean = t.replace(" ", "")
                        
                        matched_keyword = "UNKNOWN"
                        for k in keywords:
                            if k.replace(" ", "") in t_clean:
                                matched_keyword = k
                                break
                                
                        if matched_keyword != "UNKNOWN": # Zapisujemy tylko jeśli pasuje do słów kluczowych
                            x1 = int(bbox[0][0]) + x_off
                            y1 = int(bbox[0][1]) + y_off
                            x2 = int(bbox[2][0]) + x_off
                            y2 = int(bbox[2][1]) + y_off
                            
                            frame_detections.append({
                                "type": matched_keyword,
                                "confidence": float(prob),
                                "text": t,
                                "bbox": (x1, y1, x2, y2),
                                "source": "OCR"
                            })

            # Zapis klatki dowodowej i wiersza CSV jeśli coś wykryto
            if frame_detections:
                fname = f"frame_{frame_idx}_t_{now_sec:.2f}s.jpg"
                save_path = os.path.join(out_dir, fname)
                
                for det in frame_detections:
                    x1, y1, x2, y2 = det["bbox"]
                    color = (0, 255, 0) if det["source"] == "YOLO" else (0, 165, 255)
                    cv2.rectangle(frame_to_draw, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(
                        frame_to_draw,
                        f"{det['type']} ({int(det['confidence'] * 100)}%)",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                    )
                    
                    found_types.add(det['type'])
                    detections_count += 1
                    
                    # Zapis do CSV
                    csv_writer.writerow([
                        os.path.basename(media_path),
                        "Video" if is_video else "Image",
                        frame_idx,
                        f"{now_sec:.2f}",
                        det['type'],
                        f"{det['confidence']:.2f}",
                        det['text'],
                        save_path
                    ])
                
                try:
                    cv2.imwrite(save_path, frame_to_draw)
                    saved_paths.append(save_path)
                except Exception:
                    pass

    cap.release()

    return {
        "status": "OK",
        "watermark_found": detections_count > 0,
        "watermark_types": list(found_types),
        "watermark_count": detections_count,
        "watermark_folder": out_dir,
        "csv_path": csv_path,
        "watermark_frames": saved_paths,
    }

"""
ocr_detector.py

Detekcja znaków wodnych / napisów „generatora” w wideo.
Zoptymalizowana wersja SOTA (State of the Art):
- Wspiera PaddleOCR (szybszy i skuteczniejszy) z fallbackiem do EasyOCR.
- Inteligentne narożnikowe ROI (Region of Interest) zamiast całych pasków.
- Pre-processing CLAHE (podbicie kontrastu półprzezroczystych znaków).
- Early Stopping (przerywa sprawdzanie gdy znajdzie X dowodów).
- Kompatybilność z YOLO-World (Zero-Shot object detection).
"""

from __future__ import annotations

import os
import sys
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

    # 1. Próba załadowania PaddleOCR (Zalecane - dużo lepsze dla małych tekstów)
    try:
        from paddleocr import PaddleOCR  # type: ignore
        print(" [OCR] Ładowanie modelu PaddleOCR (zoptymalizowany pod mały tekst)…")
        # use_angle_cls=False przyspiesza; show_log=False ukrywa spam w konsoli
        _OCR_READER = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
        _OCR_ENGINE_TYPE = "paddle"
        print(" [OCR] Model PaddleOCR gotowy.")
        return _OCR_READER
    except ImportError:
        print(" [OCR] PaddleOCR nie jest zainstalowany (pip install paddlepaddle paddleocr).")
    except Exception as e:
        print(f" [OCR] Błąd inicjalizacji PaddleOCR: {e}")

    # 2. Fallback do EasyOCR
    try:
        import easyocr  # type: ignore
        print(" [OCR] Ładowanie modelu EasyOCR (Fallback)…")
        _OCR_READER = easyocr.Reader(["en", "pl"], gpu=False)
        _OCR_ENGINE_TYPE = "easyocr"
        print(" [OCR] Model EasyOCR gotowy.")
        return _OCR_READER
    except Exception as e:
        print(f" [OCR] EasyOCR niedostępny: {e}")
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
            print(f" [OCR] Ładowanie modelu YOLO (lokalnie): {path}")
            _YOLO_MODEL = YOLO(path)
        elif model_id:
            print(f" [OCR] Ładowanie modelu YOLO (ID): {model_id}")
            _YOLO_MODEL = YOLO(model_id)

        # Wsparcie dla YOLO-World (Zero-shot) - jeśli model ma metodę set_classes
        if _YOLO_MODEL is not None and hasattr(_YOLO_MODEL, "set_classes"):
            print(" [OCR] Wykryto model YOLO-World. Ustawiam niestandardowe klasy tekstowe.")
            _YOLO_MODEL.set_classes(["watermark", "tiktok logo", "ai generated logo", "brand mark"])

        print(" [OCR] Model YOLO gotowy.")
        return _YOLO_MODEL

    except Exception as e:
        print(f" [OCR] Nie udało się załadować YOLO: {e}")
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
                if conf < 0.45:  # lekko niższy próg żeby łapać zblendowane logotypy
                    continue
                
                # Zero-shot (YOLO-World) ma nazwy klas w r.names
                if hasattr(r, "names") and isinstance(r.names, dict) and cls_id in r.names:
                    label_name = r.names[cls_id].upper()
                else:
                    label_name = class_map.get(cls_id, "WATERMARK")

                x1, y1, x2, y2 = map(int, b.xyxy[0])
                detections.append((x1, y1, x2, y2, conf))
            except Exception:
                continue

    return label_name, detections


def _preprocess_for_ocr(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Poprawia czytelność półprzezroczystych (zblendowanych z tłem) znaków wodnych.
    Używa CLAHE (Contrast Limited Adaptive Histogram Equalization) w przestrzeni LAB.
    """
    try:
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Opcjonalne wyostrzenie (Laplacian) można dodać tutaj, ale CLAHE zwykle wystarcza.
        return enhanced
    except Exception:
        return roi_bgr


def _make_session_dir(input_path: str) -> str:
    filename_clean = os.path.basename(input_path).replace(".", "_")
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_folder_name = f"{filename_clean}_{timestamp_str}"
    base = getattr(config, "REPORTS_BASE_DIR", "ai_reports")
    out_dir = os.path.join(base, "watermarks", session_folder_name)
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

    # Rozszerzona lista słów kluczowych domyślnie używanych w deepfake'ach i stockach
    default_keywords = [
        "SORA", "OPENAI", "GENERATED", "AI VIDEO", "MADE WITH", "AI GENERATED", 
        "RUNWAY", "PIKA", "LUMA", "GEN-2", "TIKTOK", "KWAI", "CAPCUT", "STABLE VIDEO"
    ]
    keywords = [str(k).upper() for k in getattr(config, "WATERMARK_KEYWORDS", default_keywords)]
    
    max_frames = int(getattr(config, "WATERMARK_MAX_FRAMES", 600))
    stride = int(getattr(config, "WATERMARK_STRIDE", 5))
    min_gap = float(getattr(config, "WATERMARK_MIN_SAVE_GAP_SEC", 1.0))
    
    # Early Stopping: Jeśli znajdziemy ewidentny watermark na 3 klatkach, przerywamy skanowanie by oszczędzić CPU!
    max_evidence_frames = 3 

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_dir = None
    saved_paths: List[str] = []
    found_label: Optional[str] = None
    last_save_time = -999.0

    frame_idx = 0
    evidence_count = 0

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

        # 1) YOLO watermark
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )

        # 2) OCR (Smart Corners + CLAHE)
        if detected_this_frame is None:
            reader = _get_reader()
            if reader is not None:
                # Zamiast gigantycznych poziomych pasów, badamy 4 narożniki, co drastycznie przyspiesza
                # i redukuje false-positives z tekstów na koszulkach postaci na środku ekranu.
                w_margin = int(w * 0.35)
                h_margin = int(h * 0.25)
                
                # (roi, x_offset, y_offset)
                smart_corners = [
                    (frame[0:h_margin, 0:w_margin], 0, 0),                                # TL
                    (frame[0:h_margin, w - w_margin:w], w - w_margin, 0),                 # TR
                    (frame[h - h_margin:h, 0:w_margin], 0, h - h_margin),                 # BL
                    (frame[h - h_margin:h, w - w_margin:w], w - w_margin, h - h_margin),  # BR
                ]

                for roi, x_off, y_off in smart_corners:
                    if roi.size == 0: continue
                    
                    # Pre-processing dla ukrytych, zblendowanych logotypów
                    enhanced_roi = _preprocess_for_ocr(roi)

                    try:
                        if _OCR_ENGINE_TYPE == "paddle":
                            results = reader.ocr(enhanced_roi, cls=False)
                            # Paddle zwraca: [[[[x,y],[x,y],[x,y],[x,y]], ('text', confidence)], ...]
                            # Dla pustego obrazu zwraca [None]
                            parsed_results = []
                            if results and results[0] is not None:
                                for line in results[0]:
                                    box = line[0]
                                    text = line[1][0]
                                    prob = line[1][1]
                                    parsed_results.append((box, text, prob))
                        else:
                            # EasyOCR zwraca: [([[x,y],[x,y],[x,y],[x,y]], 'text', confidence), ...]
                            parsed_results = reader.readtext(enhanced_roi)
                    except Exception as e:
                        parsed_results = []

                    for (bbox, text, prob) in parsed_results:
                        if float(prob) < 0.50:  # Zwiększony próg pewności OCR by unikać śmieci
                            continue
                        
                        t = str(text).upper().replace(" ", "")  # 'Made With' -> 'MADEWITH' ułatwia string matching
                        
                        for k in keywords:
                            k_clean = k.replace(" ", "")
                            if k_clean and k_clean in t:
                                detected_this_frame = k
                                found_label = found_label or k

                                try:
                                    x1 = int(bbox[0][0]) + x_off
                                    y1 = int(bbox[0][1]) + y_off
                                    x2 = int(bbox[2][0]) + x_off
                                    y2 = int(bbox[2][1]) + y_off
                                    cv2.rectangle(frame_to_draw, (x1, y1), (x2, y2), (0, 165, 255), 3) # Pomarańczowy dla OCR
                                    cv2.putText(
                                        frame_to_draw,
                                        f"OCR: {k} ({int(float(prob) * 100)}%)",
                                        (x1, max(0, y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
                                    )
                                except Exception:
                                    pass
                                break
                        if detected_this_frame:
                            break
                    if detected_this_frame:
                        break

        # Zapis klatki dowodowej i Early Stopping
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
                    evidence_count += 1
                except Exception:
                    pass
            
            # EARLY STOPPING: Nie badamy całego wideo, jeśli mamy już niezbite dowody!
            if evidence_count >= max_evidence_frames:
                print(f" [OCR] Zatrzymano wcześnie: Znaleziono wystarczającą ilość dowodów ({evidence_count}) w {int(now_sec)} sek.")
                break

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

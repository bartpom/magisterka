"""
ocr_detector.py

Detekcja znaków wodnych / napisów „generatora” w obrazach i wideo.
Dostosowane do wytycznych:
- Zapis CSV z detekcjami.
- Konfiguracja progu pewności (confidence) oraz próbkowania (sample_rate).
- Opcjonalne drugie przejście (szczegółowa analiza dwufazowa z zaawansowanymi filtrami morfologicznymi).
- Zapisywanie na dysk wersji oryginalnej i przefiltrowanej po pomyślnej agresywnej detekcji.
- Template Matching dla graficznych znaków wodnych (logo).
- Śledzenie ruchu napisów (pojawianie się, znikanie, ruch, statyczność).
- Ekstremalne radzenie sobie z jasnym, białym tłem.
"""

from __future__ import annotations

import os
import csv
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable

import cv2
import numpy as np

import config

# Lazy singletons
_OCR_READER = None
_OCR_ENGINE_TYPE = None
_YOLO_MODEL = None


class TextTracker:
    """Śledzi pojawianie się, znikanie i ruch znaków wodnych w czasie."""
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
            closest_record = min(self.history[type_id], key=lambda x: abs(x['frame'] - frame_idx))
            
            last_cx, last_cy = closest_record['centroid']
            dist = math.hypot(cx - last_cx, cy - last_cy)
            frames_diff = abs(frame_idx - closest_record['frame'])
            
            if frames_diff > 30: 
                status = "POJAWIENIE"
            elif dist < 25:       
                status = "STATYCZNY"
            else:
                status = "RUCHOMY"
                
        self.history[type_id].append({
            "frame": frame_idx,
            "centroid": (cx, cy),
            "bbox": bbox
        })
        return status


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


def _detect_template_watermarks(image_to_scan: np.ndarray, confidence: float) -> List[dict]:
    templates_dir = getattr(config, "TEMPLATES_DIR", "watermark_templates")
    if not os.path.exists(templates_dir):
        try:
            os.makedirs(templates_dir, exist_ok=True)
        except Exception:
            return []
            
    detections = []
    if len(image_to_scan.shape) == 3:
        gray_frame = cv2.cvtColor(image_to_scan, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = image_to_scan
        
    for fname in os.listdir(templates_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        tpl_path = os.path.join(templates_dir, fname)
        tpl = cv2.imread(tpl_path, cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            continue
            
        try:
            res = cv2.matchTemplate(gray_frame, tpl, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= confidence)
            h, w = tpl.shape
            label = os.path.splitext(fname)[0].upper()
            
            for pt in zip(*loc[::-1]):
                x1, y1 = int(pt[0]), int(pt[1])
                x2, y2 = x1 + w, y1 + h
                conf = float(res[y1, x1])
                detections.append({
                    "type": f"LOGO-{label}",
                    "confidence": conf,
                    "text": f"[IMG: {label}]",
                    "bbox": (x1, y1, x2, y2)
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
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except Exception:
        return roi_bgr


def _get_advanced_filters(frame_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Zaawansowane techniki Image Processing pod wyciąganie znaków z kompresji, cieni i jaskrawych/białych środowisk."""
    filters = []
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Adaptive Thresholding 
        adapt_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
        filters.append(("AGGR-ADAPT-THRESH", cv2.cvtColor(adapt_thresh, cv2.COLOR_GRAY2BGR)))
        
        # 2. Morphological Top-Hat 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        tophat_norm = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
        filters.append(("AGGR-TOPHAT", cv2.cvtColor(tophat_norm, cv2.COLOR_GRAY2BGR)))
        
        # 3. Morphological Black-Hat 
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        blackhat_norm = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
        filters.append(("AGGR-BLACKHAT", cv2.cvtColor(blackhat_norm, cv2.COLOR_GRAY2BGR)))
        
        # 4. Unsharp Masking 
        gaussian = cv2.GaussianBlur(frame_bgr, (9,9), 10.0)
        sharpened = cv2.addWeighted(frame_bgr, 1.5, gaussian, -0.5, 0)
        filters.append(("AGGR-SHARPEN", sharpened))
        
        # 5. Edge Enhancement 
        laplacian = cv2.Laplacian(gray, cv2.CV_8U)
        lap_bgr = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        edge_enhanced = cv2.addWeighted(frame_bgr, 1.0, lap_bgr, 0.8, 0)
        filters.append(("AGGR-EDGE", edge_enhanced))

        # 6. EXTREME WHITE BACKGROUND FIX (Gamma + Invert CLAHE)
        # Przy bardzo jasnym tle (jak słońce, chmury, śnieg) obniżamy jasność przez krzywą gammy (przyciemniamy),
        # żeby biały znak wodny zaczął odstawać, a następnie odwracamy kolory by napis stał się czarny i przepuszczamy przez CLAHE.
        gamma = 0.4 # Silne przyciemnienie
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        darkened = cv2.LUT(frame_bgr, table)
        dark_inv = cv2.bitwise_not(darkened)
        dark_inv_clahe = _preprocess_for_ocr(dark_inv)
        filters.append(("AGGR-EXTREME-WHITE", dark_inv_clahe))

        # 7. Localized Background Subtraction (Tło lokalne odjęte)
        # Wykorzystywane aby usunąć gradient szarości nieba i zostawić twarde znaki
        bg = cv2.medianBlur(gray, 51) 
        diff = cv2.absdiff(gray, bg)
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        diff_bgr = cv2.cvtColor(diff_norm, cv2.COLOR_GRAY2BGR)
        filters.append(("AGGR-BGSUB", diff_bgr))
        
    except Exception:
        pass
        
    return filters


def _make_session_dir(input_path: str) -> str:
    filename_clean = os.path.basename(input_path).replace(".", "_")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder_name = f"{filename_clean}_{timestamp_str}"
    base = getattr(config, "REPORTS_BASE_DIR", "reports")
    out_dir = os.path.join(base, "watermarks", session_folder_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _perform_scan(frame_original, confidence, keywords, versions_to_scan, scale_factor=1.0):
    frame_detections = []
    
    # 1) YOLO watermark (zawsze na oryginalnej skali)
    yolo_boxes = _detect_yolo_watermark(frame_original, min_conf=confidence)
    for (x1, y1, x2, y2, conf, label) in yolo_boxes:
        frame_detections.append({
            "type": label,
            "confidence": conf,
            "text": f"[{label}]",
            "bbox": (x1, y1, x2, y2),
            "source": "YOLO"
        })

    # 2) OCR & Template
    reader = _get_reader()
    found_words_this_frame = set()
    
    for source_name, base_image in versions_to_scan:
        # a) Szukanie graficznych logo (Template Matching)
        tpl_boxes = _detect_template_watermarks(base_image, confidence)
        for det in tpl_boxes:
            if det["type"] not in found_words_this_frame:
                det["source"] = f"TEMPLATE-{source_name}"
                frame_detections.append(det)
                found_words_this_frame.add(det["type"])

        # b) Skalowanie (Super Resolution dla lepszego OCR małych znaków)
        if scale_factor != 1.0:
            image_to_scan = cv2.resize(base_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        else:
            image_to_scan = base_image

        # c) OCR
        if reader is not None:
            try:
                if _OCR_ENGINE_TYPE == "paddle":
                    results = reader.ocr(image_to_scan, cls=False)
                    parsed_results = []
                    if results and results[0] is not None:
                        for line in results[0]:
                            parsed_results.append((line[0], line[1][0], line[1][1]))
                else:
                    parsed_results = reader.readtext(image_to_scan)
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
                        
                if matched_keyword != "UNKNOWN" and matched_keyword not in found_words_this_frame:
                    x1 = int(bbox[0][0] / scale_factor)
                    y1 = int(bbox[0][1] / scale_factor)
                    x2 = int(bbox[2][0] / scale_factor)
                    y2 = int(bbox[2][1] / scale_factor)
                    
                    frame_detections.append({
                        "type": matched_keyword,
                        "confidence": float(prob),
                        "text": t,
                        "bbox": (x1, y1, x2, y2),
                        "source": source_name
                    })
                    found_words_this_frame.add(matched_keyword)
                    
    return frame_detections


def scan_for_watermarks(
    media_path: str, 
    check_stop=None, 
    progress_callback=None, 
    confidence: float = 0.6, 
    sample_rate: int = 30,
    detailed_scan: bool = False,
    preview_callback: Optional[Callable[[np.ndarray, list], None]] = None
) -> Dict[str, Any]:
    
    is_video = os.path.splitext(media_path)[1].lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    cap = cv2.VideoCapture(os.path.abspath(media_path))
    
    if not cap.isOpened():
        return {"status": "ERROR", "error": "Nie można otworzyć pliku."}

    default_keywords = [
        "SORA", "OPENAI", "GENERATED", "AI VIDEO", "MADE WITH", "AI GENERATED", 
        "RUNWAY", "PIKA", "LUMA", "GEN-2", "TIKTOK", "KWAI", "CAPCUT", "STABLE VIDEO",
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
    found_types = set()
    
    tracker = TextTracker()

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Plik", "Typ", "Numer klatki", "Timestamp", "Typ watermarku", "Confidence", "Tekst", "Ruch", "Ścieżka zapisu"])

        # ==========================================
        # FAZA 1: Skanowanie podstawowe
        # ==========================================
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
            
            # W fazie 1 dodajemy szybki filtr gamma do białych teł by zwiększyć szansę już na wejściu bez czekania na AGGR
            gamma = 0.5
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            darkened = cv2.LUT(frame, table)
            
            versions_to_scan = [
                ("OCR-RAW", frame),
                ("OCR-CLAHE", _preprocess_for_ocr(frame)),
                ("OCR-INV", cv2.bitwise_not(frame)),
                ("OCR-DARK", darkened)
            ]
            
            frame_detections = _perform_scan(frame, confidence, keywords, versions_to_scan, scale_factor=1.0)

            if not frame_detections:
                missed_frames.append(frame_idx)

            frame_to_draw = frame.copy()

            for det in frame_detections:
                x1, y1, x2, y2 = det["bbox"]
                
                motion_status = tracker.update(frame_idx, det['type'], det['bbox'])
                
                color = (0, 255, 0)
                cv2.rectangle(frame_to_draw, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    frame_to_draw,
                    f"{det['type']} [{motion_status}] ({int(det['confidence'] * 100)}%)",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                )
                
                found_types.add(det['type'])
                detections_count += 1
                
                fname = f"frame_{frame_idx}_t_{now_sec:.2f}s.jpg"
                save_path = os.path.join(out_dir, fname)
                csv_writer.writerow([
                    os.path.basename(media_path), "Video" if is_video else "Image",
                    frame_idx, f"{now_sec:.2f}", det['type'], f"{det['confidence']:.2f}", det['text'], motion_status, save_path
                ])
                try:
                    cv2.imwrite(save_path, frame_to_draw)
                    saved_paths.append(save_path)
                except Exception:
                    pass

            if preview_callback:
                if not frame_detections:
                     cv2.putText(frame_to_draw, f"Brak detekcji (klatka {frame_idx})", (10, 30), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                preview_callback(frame_to_draw, frame_detections)

        # ==========================================
        # FAZA 2: Szczegółowa Analiza (Zaawansowana)
        # Dla obrazków robimy też Faze 2 automatycznie, jeśli 'detailed_scan' i brak detekcji,
        # dla filmów tylko wtedy gdy detections_count > 0.
        # ==========================================
        run_aggr = False
        if detailed_scan and missed_frames and not (check_stop and check_stop()):
            if not is_video:
                run_aggr = True  # Zawsze dociskamy zdjęcia jak nic nie znaleziono
            elif detections_count > 0:
                run_aggr = True  # Wideo dociskamy tylko jak jest dowód na innej klatce

        if run_aggr:
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
                aggr_versions.append(("AGGR-BW", cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)))
                aggr_versions.append(("AGGR-ORIG", frame))
                
                # Dodatkowa super rozdzielczość x2.0 pod kątem zblendowanych napisów Sora na niebie/chmurach
                frame_detections = _perform_scan(frame, confidence, keywords, aggr_versions, scale_factor=2.0)
                
                if frame_detections:
                    frame_to_draw_orig = frame.copy()
                    for det in frame_detections:
                        x1, y1, x2, y2 = det["bbox"]
                        motion_status = tracker.update(m_idx, det['type'], det['bbox'])
                        
                        color = (0, 255, 0)
                        cv2.rectangle(frame_to_draw_orig, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(
                            frame_to_draw_orig,
                            f"{det['type']} [AGGR:{motion_status}]",
                            (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                        )
                        found_types.add(det['type'])
                        detections_count += 1
                        
                    fname_orig = f"frame_{m_idx}_aggr_orig_t_{now_sec:.2f}s.jpg"
                    save_path_orig = os.path.join(out_dir, fname_orig)
                    try:
                        cv2.imwrite(save_path_orig, frame_to_draw_orig)
                        saved_paths.append(save_path_orig)
                    except Exception:
                        pass
                        
                    version_images = dict(aggr_versions)
                    from collections import defaultdict
                    dets_by_source = defaultdict(list)
                    
                    for det in frame_detections:
                        src = det.get("source", "")
                        base_source = src.replace("TEMPLATE-", "") if src.startswith("TEMPLATE-") else src
                        dets_by_source[base_source].append(det)
                        
                        csv_writer.writerow([
                            os.path.basename(media_path), "Video (Aggr)" if is_video else "Image (Aggr)",
                            m_idx, f"{now_sec:.2f}", det['type'], f"{det['confidence']:.2f}", det['text'], motion_status, save_path_orig
                        ])
                        
                    for src_name, dets in dets_by_source.items():
                        if src_name in version_images:
                            img_to_draw = version_images[src_name].copy()
                            for det in dets:
                                x1, y1, x2, y2 = det["bbox"]
                                color = (0, 255, 0)
                                cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), color, 3)
                                cv2.putText(
                                    img_to_draw,
                                    f"[{src_name}] {det['type']}",
                                    (x1, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                                )
                            
                            fname_mod = f"frame_{m_idx}_{src_name}_t_{now_sec:.2f}s.jpg"
                            save_path_mod = os.path.join(out_dir, fname_mod)
                            try:
                                cv2.imwrite(save_path_mod, img_to_draw)
                                saved_paths.append(save_path_mod)
                            except Exception:
                                pass
                                
                    if preview_callback:
                        preview_callback(frame_to_draw_orig, frame_detections)
                else:
                    if preview_callback:
                        frame_to_draw_empty = frame.copy()
                        cv2.putText(frame_to_draw_empty, f"Brak detekcji (AGGR klatka {m_idx})", (10, 30), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
                        preview_callback(frame_to_draw_empty, [])

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

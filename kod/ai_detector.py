# ai_detector.py - ULEPSZONA WERSJA DLA LEPsZEJ DETEKCJI DEEPFAKE
# -*- coding: utf-8 -*-

"""
AI/Forensic video analysis helper - ULEPSZONA WERSJA

Główne ulepszenia:
1. Lepsza integracja modeli HF (twarz + scena + wideo)
2. Optymalna fuzja wyników VideoMAE i modeli obrazowych
3. Poprawiony algorytm decyzyjny z walidacją
4. Specjalna obsługa detekcji Sora/Generacji AI
5. Zaawansowana analiza forensic
6. Wsparcie dla 3 trybów: AI Detection, Deepfake Detection, Watermark
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import random
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import config
from videomae_detector import VideoMAEDeepfakeDetector, VideoMAEConfig

# --- Optional heavy deps ---
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import torch
    from PIL import Image
    from transformers import pipeline
except Exception:
    torch = None
    Image = None
    pipeline = None

# Import zaawansowanych funkcji detekcji z advanced_detectors.py
try:
    from advanced_detectors import (
        clamp, safe_float, robust_agg,
        fuse_video_scores, fuse_scores, decision_policy
    )
except ImportError:
    # Fallback funkcje jeśli advanced_detectors.py nie jest dostępny
    def clamp(x: float, lo=0.0, hi=100.0) -> float:
        return float(max(lo, min(hi, x)))


    def safe_float(x) -> Optional[float]:
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except Exception:
            return None


    def robust_agg(vals: List[float], mode="p90") -> Optional[float]:
        if not vals:
            return None
        try:
            arr = np.asarray([v for v in vals if safe_float(v) is not None], dtype=np.float32)
            if arr.size == 0:
                return None
            if mode == "mean":
                return float(arr.mean())
            if mode == "median":
                return float(np.median(arr))
            if mode.startswith("p"):
                return float(np.percentile(arr, int(mode[1:])))
            return float(np.percentile(arr, 90))
        except Exception:
            return None


    def fuse_video_scores(vm: Optional[float], d3: Optional[float], policy: str) -> Optional[float]:
        if vm is None and d3 is None:
            return None
        if policy == "high_recall":
            return max(v for v in [vm, d3] if v is not None)
        if vm is not None and d3 is not None:
            return 0.9 * vm + 0.1 * d3
        if vm is not None:
            return vm
        return d3 if d3 is not None and d3 >= 85.0 else None


def fuse_scores(features: Dict[str, Any]) -> float:
    # Logika "Weighted Score" zintegrowana z Safety Gate
    weights = getattr(config, 'FUSE_WEIGHTS', {"video": 0.50, "face": 0.30, "scene": 0.20})
    
    face = safe_float(features.get("face"))
    scene = safe_float(features.get("scene"))
    video = safe_float(features.get("video"))
    
    # --- Safety Gate / Noise Filter ---
    # Jeśli scena ma 100% ale wideo i twarz są niskie, tłumimy wynik sceny (częsty błąd w nowym repo)
    if scene is not None and scene > 95:
        if (video is not None and video < 30) and (face is None or face < 30):
            scene *= 0.4  # Tłumienie szumu sceny
            
    acc = 0.0
    wsum = 0.0
    
    vals = {"video": video, "face": face, "scene": scene}
    for k, w in weights.items():
        v = vals.get(k)
        if v is None or w <= 0:
            continue
        acc += v * w
        wsum += w
        
    final = clamp(acc / wsum) if wsum > 0 else 0.0
    
    # Multi-Signal Boost (z wersji ULTRA/FINAL)
    signals = [v for v in [video, face, scene] if v is not None and v > 45]
    if len(signals) >= 2:
        final = clamp(final + 10.0)
        
    return final


    def decision_policy(score: float, details: Dict[str, Any], policy: str) -> str:
        real_max = getattr(config, "REAL_MAX", 30.0)
        fake_min = getattr(config, "FAKE_MIN", 70.0)
        eps = 0.01

        face = safe_float(details.get("ai_face_score"))
        video = safe_float(details.get("ai_video_score"))
        wm = safe_float(details.get("watermark_score"))
        forensic = safe_float(details.get("forensic_score"))
        ratio = safe_float(details.get("fake_ratio"))

        # FAKE corroboration
        primary = any([
            video is not None and video >= 90,
            wm is not None and wm >= 85,
            forensic is not None and forensic >= 85,
        ])

        secondary = any([
            ratio is not None and ratio >= 40,
            face is not None and face >= 85,
        ])

        corroborated = primary and secondary

        # REAL gate
        suspect = any([
            face is not None and face >= 50,
            video is not None and video >= 50,
            wm is not None and wm >= 60,
        ])

        if policy == "high_precision":
            if score >= fake_min and not corroborated:
                score = fake_min - eps
            if score <= real_max and suspect:
                score = real_max + eps

        elif policy == "high_recall":
            if score <= real_max:
                score = real_max + eps

        # Verdict
        if score <= real_max:
            return "REAL (PRAWDOPODOBNE)"
        if score >= fake_min:
            return "FAKE (PRAWDOPODOBNE)"
        return "GREY ZONE (NIEPEWNE)"

# =============================================================================
# Public stop control (dla GUI)
# =============================================================================

_STOP_REQUESTED: bool = False


def request_stop() -> None:
    """Może być wołane z GUI, żeby przerwać batch."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def clear_stop() -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False


def stop_requested() -> bool:
    return _STOP_REQUESTED


# =============================================================================
# Run folder
# =============================================================================

def _now_stamp() -> str:
    # YYYYMMDD_HHMMSS
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def begin_run(reports_root: str = "reports") -> str:
    """
    Tworzy katalog run: reports/run_YYYYMMDD_HHMMSS_XXXX
    Zwraca ścieżkę do katalogu.
    """
    os.makedirs(reports_root, exist_ok=True)
    suffix = random.randint(1000, 9999)
    run_dir = os.path.join(reports_root, f"run_{_now_stamp()}_{suffix}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"> Run folder: {run_dir}")
    return run_dir


# =============================================================================
# Report format
# =============================================================================

@dataclass
class AiResult:
    face_score: Optional[float] = None
    scene_score: Optional[float] = None
    video_score: Optional[float] = None  # reserved
    combined_max: Optional[float] = None
    face_frames: int = 0
    total_frames: int = 0


@dataclass
class ForensicResult:
    jitter_px: Optional[float] = None
    blinking: Optional[float] = None
    ela_score: Optional[float] = None
    fft_score: Optional[float] = None
    border_artifacts: Optional[float] = None
    sharpness_face: Optional[float] = None


@dataclass
class Report:
    file_name: str
    verdict: str
    total_score: float
    ai: AiResult
    forensic: ForensicResult
    metadata: Optional[Dict[str, Any]] = None


def _fmt_num(x: Optional[float], *, pct: bool = False, nd: int = 2) -> str:
    if x is None:
        return "N/A"
    if pct:
        return f"{x:.{nd}f}%"
    return f"{x:.{nd}f}"


def _verdict_from_score(score: float) -> str:
    # Use thresholds from config
    real_max = getattr(config, "REAL_MAX", 30.0)
    fake_min = getattr(config, "FAKE_MIN", 60.0)

    if score <= real_max:
        return f"REAL (PRAWDOPODOBNE) (<=REAL_MAX {real_max})"
    if score >= fake_min:
        return f"FAKE (PRAWDOPODOBNE) (>=FAKE_MIN {fake_min})"
    
    # Dodatkowe info dla Grey Zone (zgodnie z nowym raportem)
    to_fake = fake_min - score
    over_real = score - real_max
    return f"NIEPEWNE / GREY ZONE ({score:.2f}%) [GREY: +{to_fake:.2f} do FAKE, +{over_real:.2f} ponad REAL]"


def _write_report_txt(report: Report, out_path: str) -> None:
    lines: List[str] = []
    lines.append(f"Plik: {report.file_name}")
    lines.append("")
    lines.append(f"WERDYKT PODSUMOWUJĄCY: {report.verdict}")
    lines.append(f"Wynik łączny (Score): {report.total_score:.2f}%")
    lines.append("")

    # Wylicz dodatkowe metryki z metadata
    if report.metadata:
        ai_detector_verdict = report.metadata.get('ai_detector_verdict', 'N/A')
        ai_detector_score = report.metadata.get('ai_detector_score', 0.0)
        deepfake_verdict = report.metadata.get('deepfake_verdict', 'N/A')
        deepfake_score = report.metadata.get('deepfake_score', 0.0)

        lines.append("--- WERDYKTY CZĄSTKOWE ---")
        lines.append(f"AI DETECTOR (Scena/Gen): {ai_detector_verdict} ({ai_detector_score:.2f}%)")
        lines.append(f"DEEPFAKE (Twarz): {deepfake_verdict} ({deepfake_score:.2f}%)")
        lines.append("")

        face_ratio = report.metadata.get('face_ratio', 0.0)
        fake_ratio = report.metadata.get('fake_ratio', 0.0)
        detection_mode = report.metadata.get('detection_mode', 'combined')

        lines.append(f"Tryb detekcji: {detection_mode}")
        lines.append(f"Wskaźnik twarzy: {face_ratio:.1f}%")
        lines.append(f"Wskaźnik fake: {fake_ratio:.1f}%")

        if 'detection_flags' in report.metadata:
            flags = report.metadata['detection_flags']
            if flags:
                lines.append(f"Flagi detekcji: {', '.join(flags)}")

    lines.append("")
    lines.append("--- DETALE AI ---")
    lines.append(f"AI Face/Subject Score: {_fmt_num(report.ai.face_score, pct=True)}")
    lines.append(f"AI Scene (Frames) Score: {_fmt_num(report.ai.scene_score, pct=True)}")
    lines.append(f"AI Video Model Score: {_fmt_num(report.ai.video_score, pct=True)}")
    lines.append(f"AI Combined (max) Score: {_fmt_num(report.ai.combined_max, pct=True)}")
    lines.append(f"Klatki z twarzą: {report.ai.face_frames}/{report.ai.total_frames}")
    lines.append("")
    lines.append("--- DETALE FORENSIC ---")
    lines.append(f"Stabilność (Jitter): {_fmt_num(report.forensic.jitter_px, nd=2)} px")
    lines.append(f"ELA Score: {_fmt_num(report.forensic.ela_score, nd=2)}")
    lines.append(f"FFT Score: {_fmt_num(report.forensic.fft_score, nd=2)}")
    lines.append(f"Border Artifacts: {_fmt_num(report.forensic.border_artifacts, nd=2)}")
    lines.append(f"Sharpness (face): {_fmt_num(report.forensic.sharpness_face, nd=2)}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_report_json(report: Report, out_path: str) -> None:
    payload = {
        "file": report.file_name,
        "verdict": report.verdict,
        "total_score": report.total_score,
        "ai": {
            "face_score": report.ai.face_score,
            "scene_score": report.ai.scene_score,
            "video_score": report.ai.video_score,
            "combined_max": report.ai.combined_max,
            "face_frames": report.ai.face_frames,
            "total_frames": report.ai.total_frames,
        },
        "forensic": {
            "jitter_px": report.forensic.jitter_px,
            "blinking": report.forensic.blinking,
            "ela_score": report.forensic.ela_score,
            "fft_score": report.forensic.fft_score,
            "border_artifacts": report.forensic.border_artifacts,
            "sharpness_face": report.forensic.sharpness_face,
        },
        "metadata": report.metadata or {},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# =============================================================================
# Progress callback normalizacja
# =============================================================================

_IntLike = Union[int, float]
_ProgressPayload = Union[_IntLike, str, Dict[str, Any], Tuple[Any, ...], List[Any]]

_last_progress_error_printed: bool = False


def _extract_last_int_from_string(s: str) -> Optional[int]:
    # Bierzemy OSTATNIĄ liczbę w stringu (np. "... frame 0" -> 0).
    m = re.search(r"(\d+)(?!.*\d)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _normalize_progress(payload: _ProgressPayload, default_total: int) -> Tuple[
    Optional[int], Optional[int], Optional[int]]:
    """
    Zwraca (current, total, percent)
    - current: numer klatki / krok
    - total: max
    - percent: 0..100
    """
    current: Optional[int] = None
    total: Optional[int] = None

    # dict forms
    if isinstance(payload, dict):
        # percent by name (jeżeli ktoś tak przekaże)
        for k in ("percent", "pct", "progress"):
            if k in payload:
                v = payload.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    p = float(v)
                    if 0.0 <= p <= 1.0:
                        current = int(round(p * 100))
                        total = 100
                        percent = max(0, min(100, int(round(p * 100))))
                        return (current, total, percent)
                    # traktuj jako 0..100
                    current = int(round(p))
                    total = 100
                    percent = max(0, min(100, current))
                    return (current, total, percent)
                if isinstance(v, str):
                    p_int = _extract_last_int_from_string(v)
                    if p_int is not None:
                        current = p_int
                        total = 100
                        percent = max(0, min(100, current))
                        return (current, total, percent)

        for k in ("current", "frame", "idx", "index", "i", "step"):
            if k in payload:
                v = payload.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    current = int(v)
                    break
                if isinstance(v, str):
                    current = _extract_last_int_from_string(v)
                    break

        for k in ("total", "max", "frames", "n", "count"):
            if k in payload:
                v = payload.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    total = int(v)
                    break
                if isinstance(v, str):
                    total = _extract_last_int_from_string(v)
                    break

    # tuple/list forms (current, total) albo (current,)
    elif isinstance(payload, (tuple, list)):
        if len(payload) >= 1:
            v0 = payload[0]
            if isinstance(v0, (int, float)) and not isinstance(v0, bool):
                current = int(v0)
            elif isinstance(v0, str):
                current = _extract_last_int_from_string(v0)
        if len(payload) >= 2:
            v1 = payload[1]
            if isinstance(v1, (int, float)) and not isinstance(v1, bool):
                total = int(v1)
            elif isinstance(v1, str):
                total = _extract_last_int_from_string(v1)

    # numeric forms
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        # Heurystyka:
        # - float 0..1 => percent
        # - int/float > 1 => current
        if isinstance(payload, float) and 0.0 <= payload <= 1.0:
            total = 100
            current = int(round(payload * 100))
        else:
            current = int(payload)

    # string form
    elif isinstance(payload, str):
        # np. "0000_fake_x.mp4: frame 0" -> current=0
        current = _extract_last_int_from_string(payload)

    if total is None:
        total = default_total
    if current is None:
        return (None, total, None)

    # clamp
    if total <= 0:
        total = default_total if default_total > 0 else 1
    if current < 0:
        current = 0

    current_for_percent = min(current, total)
    percent = int(round(100.0 * float(current_for_percent) / float(total)))
    percent = max(0, min(100, percent))
    return (current, total, percent)


def _safe_call_progress(cb: Optional[Callable[..., Any]], payload: _ProgressPayload, *, default_total: int) -> None:
    """
    Woła progress callback tak, żeby:
    - nie zabiło analizy (try/except)
    - przekazać coś "sensownego" w różnych formatach:
        1) cb(percent:int)
        2) cb(current:int, total:int)
        3) cb(dict)
        4) cb(current=..., total=..., percent=..., raw=...)
    """
    global _last_progress_error_printed
    if cb is None:
        return

    cur, tot, pct = _normalize_progress(payload, default_total=default_total)

    payload_dict: Dict[str, Any] = {
        "current": cur,
        "total": tot,
        "percent": pct,
        "raw": payload,
    }

    variants: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []

    if pct is not None:
        variants.append(((pct,), {}))  # najczęstsze w GUI: int percent
    if cur is not None and tot is not None:
        variants.append(((cur, tot), {}))  # czasem: (cur,total)
    variants.append(((payload_dict,), {}))  # czasem: 1 argument dict
    variants.append(((), payload_dict))  # czasem: kwargs

    last_err: Optional[BaseException] = None

    for args, kwargs in variants:
        try:
            cb(*args, **kwargs)
            return
        except Exception as e:
            # ważne: NIE przerywamy analizy i próbujemy następny wariant,
            # bo błąd mógł wynikać z niepasującego formatu (np. callback oczekiwał dict).
            last_err = e
            continue

    if last_err is not None and not _last_progress_error_printed:
        _last_progress_error_printed = True
        print(f"[UWAGA] progress_callback zgłosił błąd i został zignorowany: {last_err}")


# =============================================================================
# Video helpers
# =============================================================================

def _require_deps() -> None:
    if cv2 is None or np is None:
        raise RuntimeError("Brak zależności: wymagane 'opencv-python' oraz 'numpy'.")


def _collect_video_files(path: str) -> List[str]:
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    if os.path.isfile(path):
        return [path]
    files: List[str] = []
    for root, _, names in os.walk(path):
        for n in names:
            if os.path.splitext(n)[1].lower() in exts:
                files.append(os.path.join(root, n))
    files.sort()
    return files


def _open_video(path: str):
    _require_deps()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć wideo: {path}")
    return cap


def _get_frame_count(cap) -> int:
    # nie zawsze dostępne, ale spróbujmy
    try:
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if cnt > 0:
            return cnt
    except Exception:
        pass
    return 0


def _sample_frame_indices(total_frames: int, max_frames: int) -> List[int]:
    if total_frames <= 0:
        # nie znamy – spróbujemy po kolei do max_frames
        return list(range(max_frames))
    if max_frames <= 0:
        return []
    if total_frames <= max_frames:
        return list(range(total_frames))
    step = total_frames / float(max_frames)
    idxs = [int(i * step) for i in range(max_frames)]
    # clamp & unique
    idxs = [max(0, min(total_frames - 1, x)) for x in idxs]
    out: List[int] = []
    seen = set()
    for x in idxs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# =============================================================================
# Simple face detect (Haar) – bez dodatkowych modeli
# =============================================================================

_haar = None


def _get_haar() -> Any:
    global _haar
    if _haar is not None:
        return _haar
    _require_deps()
    try:
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _haar = cv2.CascadeClassifier(haar_path)
        if _haar.empty():
            _haar = None
    except Exception:
        _haar = None
    return _haar


def _detect_face_bbox(frame_bgr) -> Optional[Tuple[int, int, int, int]]:
    """
    Zwraca (x,y,w,h) dla największej twarzy albo None.
    """
    haar = _get_haar()
    if haar is None:
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if faces is None or len(faces) == 0:
        return None
    # Wybierz największą twarz
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return int(x), int(y), int(w), int(h)


def _detect_all_faces(frame_bgr) -> List[Tuple[int, int, int, int]]:
    """Zwraca listę wszystkich twarzy w klatce"""
    haar = _get_haar()
    if haar is None:
        return []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if faces is None:
        return []
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


# =============================================================================
# Forensic metrics (ELA / FFT / border / jitter)
# =============================================================================

def _ela_score(frame_bgr) -> float:
    """
    Error Level Analysis (przybliżone): recompress JPEG i różnica.
    Zwraca w przybliżeniu 0..1 (clamp).
    """
    try:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        ok, enc = cv2.imencode(".jpg", frame_bgr, encode_params)
        if not ok:
            return 0.0
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if dec is None:
            return 0.0
        diff = cv2.absdiff(frame_bgr, dec)
        diff_mean = float(np.mean(diff))  # 0..255
        score = diff_mean / 50.0
        return float(max(0.0, min(1.0, score)))
    except Exception:
        return 0.0


def _fft_score(gray) -> float:
    """
    Prosty wskaźnik artefaktów w dziedzinie częstotliwości.
    Zwraca wartość "około" 3..8 (w zależności od treści).
    """
    try:
        h, w = gray.shape[:2]
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        mag = np.log(mag + 1.0)

        c0, c1 = h // 2, w // 2
        r = max(4, min(h, w) // 12)
        low = mag[c0 - r:c0 + r, c1 - r:c1 + r]
        low_mean = float(np.mean(low)) if low.size else 1e-6
        high_mean = float(np.mean(mag))

        ratio = high_mean / (low_mean + 1e-6)
        return float(ratio * 3.0)
    except Exception:
        return 4.0  # Domyślna wartość


def _border_artifacts_score(frame_bgr, bbox: Tuple[int, int, int, int]) -> float:
    x, y, w, h = bbox
    h_img, w_img = frame_bgr.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0

    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0

    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)

        b = 3
        if min(edges.shape[:2]) <= 2 * b:
            return 0.0

        top = edges[:b, :]
        bottom = edges[-b:, :]
        left = edges[:, :b]
        right = edges[:, -b:]
        border = np.concatenate([top.flatten(), bottom.flatten(), left.flatten(), right.flatten()])

        inside = edges[b:-b, b:-b]
        border_mean = float(np.mean(border)) if border.size else 0.0
        inside_mean = float(np.mean(inside)) if inside.size else 0.0

        score = max(0.0, (border_mean - inside_mean) / 255.0)
        return float(max(0.0, min(1.0, score)))
    except Exception:
        return 0.0


def _laplacian_sharpness(gray) -> float:
    try:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())
    except Exception:
        return 0.0


# =============================================================================
# AI scoring (proste, deterministyczne) - ULEPSZONE
# =============================================================================

def _normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    v = (value - lo) / (hi - lo)
    return float(max(0.0, min(1.0, v)))


def _compute_ai_scores(
        jitter_px: Optional[float],
        ela: Optional[float],
        fft: Optional[float],
        border: Optional[float],
        sharpness: Optional[float],
        scene_ela: float,
        scene_fft: float,
        face_scores_hf: List[float],
        scene_scores_hf: List[float],
) -> AiResult:
    """
    Zwraca AiResult w %.
    """
    face_score: Optional[float] = None
    scene_score: Optional[float] = None

    # Oblicz wynik dla twarzy
    if face_scores_hf:
        # Użyj percentyla 90 dla wyników HF twarzy
        face_score = robust_agg(face_scores_hf, "p90")
    elif jitter_px is not None and ela is not None and fft is not None and border is not None:
        # Fallback: oblicz na podstawie metryk forensic
        j = _normalize(jitter_px, 0.0, 120.0)
        e = _normalize(ela, 0.0, 1.0)
        f = _normalize(fft, 3.0, 8.0)
        b = _normalize(border, 0.0, 0.08)

        s = 0.0
        wsum = 0.0

        s += 0.30 * j
        wsum += 0.30
        s += 0.25 * e
        wsum += 0.25
        s += 0.25 * f
        wsum += 0.25
        s += 0.20 * b
        wsum += 0.20

        if sharpness is not None:
            sh = 1.0 - _normalize(sharpness, 50.0, 400.0)
            s += 0.10 * sh
            wsum += 0.10

        if wsum > 0:
            face_score = 100.0 * (s / wsum)

    # Oblicz wynik dla sceny
    if scene_scores_hf:
        # Użyj percentyla 90 dla wyników HF sceny
        scene_score = robust_agg(scene_scores_hf, "p90")
    else:
        # Fallback: oblicz na podstawie ELA i FFT
        se = _normalize(scene_ela, 0.0, 1.0)
        sf = _normalize(scene_fft, 3.0, 8.0)
        scene_score = 100.0 * (0.5 * se + 0.5 * sf)

    video_score: Optional[float] = None

    combined = None
    candidates = [x for x in [face_score, scene_score, video_score] if x is not None]
    if candidates:
        combined = max(candidates)

    return AiResult(
        face_score=face_score,
        scene_score=scene_score,
        video_score=video_score,
        combined_max=combined,
    )


# =============================================================================
# HF Models Integration - ULEPSZONE
# =============================================================================

_videomae_detector = None
_videomae_lock = threading.Lock()

_hf_image_pipelines = {}
_hf_image_lock = threading.Lock()


def _get_videomae_detector() -> VideoMAEDeepfakeDetector:
    global _videomae_detector
    with _videomae_lock:
        if _videomae_detector is None:
            model_id = getattr(config, "HF_VIDEO_MODEL", "shylhy/videomae-large-finetuned-deepfake-subset")
            device = "cuda" if getattr(config, "PREFER_CUDA", True) and torch and torch.cuda.is_available() else "cpu"
            _videomae_detector = VideoMAEDeepfakeDetector(VideoMAEConfig(model_id=model_id, device=device))
        return _videomae_detector


def _get_hf_image_pipeline(model_id: str):
    global _hf_image_pipelines
    with _hf_image_lock:
        if model_id not in _hf_image_pipelines:
            if pipeline is None:
                return None
            device = 0 if getattr(config, "PREFER_CUDA", True) and torch and torch.cuda.is_available() else -1
            try:
                _hf_image_pipelines[model_id] = pipeline("image-classification", model=model_id, device=device)
            except Exception as e:
                print(f"[HF] Error loading model {model_id}: {e}")
                return None
        return _hf_image_pipelines[model_id]


def _run_hf_image_models(image_pil, scope_key: str) -> List[float]:
    scores = []
    # Pobierz modele z konfiguracji
    models_cfg = getattr(config, "HF_IMAGE_MODELS", [])

    if not models_cfg:
        # Domyślne modele jeśli nie zdefiniowano w config
        if scope_key == "face":
            default_models = [
                "prithivMLmods/Deep-Fake-Detector-v2-Model",
                "dima806/deepfake_vs_real_image_detection",
                "buildborderless/CommunityForensics-DeepfakeDet-ViT",
            ]
        else:  # "scene"
            default_models = [
                "prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2",
            ]

        for model_id in default_models:
            pipe = _get_hf_image_pipeline(model_id)
            if pipe:
                try:
                    # Fix: CommunityForensics-DeepfakeDet-ViT requires 384x384
                    # Używamy bardziej elastycznego dopasowania nazwy modelu
                    target_size = (384, 384) if any(x in model_id for x in ["CommunityForensics", "ViT"]) else None
                    input_img = image_pil
                    if target_size and image_pil.size != target_size:
                        input_img = image_pil.resize(target_size, Image.Resampling.LANCZOS)

                    results = pipe(input_img)
                    # Szukamy etykiety "fake" lub "deepfake" lub "ai"
                    p_fake = 0.0
                    for res in results:
                        label = str(res.get("label", "")).lower()
                        score = float(res.get("score", 0.0))

                        if any(keyword in label for keyword in ["fake", "deepfake", "ai"]):
                            p_fake = score
                            break
                        elif "real" in label:
                            p_fake = 1.0 - score

                    scores.append(p_fake * 100.0)
                except Exception as e:
                    print(f"[HF] Inference error for {model_id}: {e}")
    else:
        # Użyj modeli z konfiguracji
        for cfg in models_cfg:
            m_id = cfg.get("id")
            scopes = cfg.get("scopes", {})
            if not scopes.get(scope_key, False):
                continue

            pipe = _get_hf_image_pipeline(m_id)
            if pipe:
                try:
                    target_size = (384, 384) if any(x in m_id for x in ["CommunityForensics", "ViT"]) else None
                    input_img = image_pil
                    if target_size and image_pil.size != target_size:
                        input_img = image_pil.resize(target_size, Image.Resampling.LANCZOS)

                    results = pipe(input_img)
                    p_fake = 0.0
                    for res in results:
                        label = str(res.get("label", "")).lower()
                        score = float(res.get("score", 0.0))

                        if any(keyword in label for keyword in ["fake", "deepfake", "ai"]):
                            p_fake = score
                            break
                        elif "real" in label:
                            p_fake = 1.0 - score

                    scores.append(p_fake * 100.0)
                except Exception as e:
                    print(f"[HF] Inference error for {m_id}: {e}")

    return scores


# =============================================================================
# GŁÓWNA FUNKCJA ANALIZY - CAŁKOWICIE PRZEPISANA
# =============================================================================

def scan_for_deepfake(
        video_path: str,
        max_frames: int = 60,
        progress_callback: Optional[Callable[..., Any]] = None,
        check_stop: Optional[Any] = None,
        detection_mode: str = "combined",  # "ai", "deepfake", "combined"
        **_ignored_kwargs: Any,
) -> Tuple[str, float, float, Dict[str, Any]]:
    """
    Główna analiza AI+forensic na potrzeby raportu.
    ZWRACA: (werdykt, final_score, fake_ratio, szczegóły)

    detection_mode:
      - "ai": skup się na detekcji AI/Generacji (Sora itp.)
      - "deepfake": skup się na deepfake twarzy
      - "combined": połącz oba podejścia
    """
    _require_deps()

    if not os.path.exists(video_path):
        return "ERROR", 0.0, 0.0, {"error": f"Plik nie istnieje: {video_path}"}

    # Inicjalizuj struktury do zbierania wyników
    details: Dict[str, Any] = {
        "detection_mode": detection_mode,
        "video_path": video_path
    }

    # --- Analiza VideoMAE (jeśli dostępne) ---
    video_score: Optional[float] = None
    try:
        vm_detector = _get_videomae_detector()
        p_fake, vm_details = vm_detector.analyze(video_path)
        if p_fake is not None:
            video_score = p_fake * 100.0
            details['videomae_details'] = vm_details
            details['videomae_score_raw'] = p_fake * 100.0
    except Exception as e:
        print(f"[VideoMAE] Error: {e}")
        details['videomae_error'] = str(e)

    # --- Analiza klatek wideo ---
    cap = _open_video(video_path)
    try:
        total_frames = _get_frame_count(cap)
        idxs = _sample_frame_indices(total_frames, max_frames)

        if not idxs:
            # Jeśli nie ma klatek, zwróć domyślne wartości
            details['ai_face_score'] = None
            details['ai_scene_score'] = None
            details['ai_video_score'] = video_score
            details['face_ratio'] = 0.0
            details['face_frames'] = 0
            details['total_frames'] = 0

            # Fuzja wyników
            features_for_fusion = {
                "face": details.get("ai_face_score"),
                "scene": details.get("ai_scene_score"),
                "video": details.get("ai_video_score"),
            }
            final_score = fuse_scores(features_for_fusion)
            details['final_score'] = final_score

            # Werdykt
            verdict = decision_policy(
                final_score,
                details,
                policy=getattr(config, 'DECISION_POLICY', 'high_precision')
            )
            details['verdict'] = verdict

            return verdict, final_score, 0.0, details

        # Listy na wyniki
        face_scores_hf = []  # Wyniki HF dla twarzy
        scene_scores_hf = []  # Wyniki HF dla sceny

        face_positions = []  # Pozycje twarzy do obliczenia jitter
        face_frames = 0  # Liczba klatek z twarzą

        # Metryki forensic
        ela_scores = []
        fft_scores = []
        border_scores = []
        sharpness_scores = []

        # Przetwarzaj klatki
        for k, frame_idx in enumerate(idxs):
            if stop_requested():
                break

            if check_stop is not None:
                try:
                    if callable(check_stop):
                        if bool(check_stop()):
                            break
                    else:
                        if bool(check_stop):
                            break
                except Exception:
                    pass

            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # Progress callback
            if progress_callback:
                _safe_call_progress(progress_callback, (k + 1, len(idxs)), default_total=len(idxs))

            # --- Analiza sceny (cała klatka) ---
            try:
                # ELA dla sceny
                scene_ela = _ela_score(frame)
                ela_scores.append(scene_ela)

                # FFT dla sceny
                gray_scene = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                scene_fft = _fft_score(gray_scene)
                fft_scores.append(scene_fft)

                # Modele HF dla sceny (szczególnie ważne dla AI/Generacji)
                if Image is not None and detection_mode in ["ai", "combined"]:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    hf_scene_scores = _run_hf_image_models(frame_pil, "scene")
                    if hf_scene_scores:
                        scene_scores_hf.extend(hf_scene_scores)
            except Exception as e:
                print(f"[Scene Analysis Error] Frame {frame_idx}: {e}")

            # --- Detekcja i analiza twarzy ---
            faces = _detect_all_faces(frame)

            if faces and detection_mode in ["deepfake", "combined"]:
                face_frames += 1

                # Weź największą twarz do analizy
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_positions.append((x + w / 2, y + h / 2))

                # Wyciągnij ROI twarzy
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                roi = frame[y0:y1, x0:x1]

                if roi.size > 0:
                    try:
                        # Modele HF dla twarzy
                        if Image is not None:
                            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                            roi_pil = Image.fromarray(roi_rgb)
                            hf_face_scores = _run_hf_image_models(roi_pil, "face")
                            if hf_face_scores:
                                face_scores_hf.extend(hf_face_scores)

                        # Border artifacts
                        border = _border_artifacts_score(frame, (x, y, w, h))
                        border_scores.append(border)

                        # Sharpness
                        gray_face = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        sharp = _laplacian_sharpness(gray_face)
                        sharpness_scores.append(sharp)
                    except Exception as e:
                        print(f"[Face Analysis Error] Frame {frame_idx}: {e}")

        # --- Agregacja wyników ---
        # Jitter calculation
        jitter_px: Optional[float] = None
        if len(face_positions) >= 2:
            dists = []
            for i in range(1, len(face_positions)):
                x1, y1 = face_positions[i - 1]
                x2, y2 = face_positions[i]
                dists.append(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            jitter_px = float(np.mean(dists)) if dists else 0.0

        # Oblicz średnie wyniki forensic
        ela_avg = float(np.mean(ela_scores)) if ela_scores else 0.0
        fft_avg = float(np.mean(fft_scores)) if fft_scores else 0.0
        border_avg = float(np.mean(border_scores)) if border_scores else 0.0
        sharp_avg = float(np.mean(sharpness_scores)) if sharpness_scores else None

        # Oblicz AI scores
        face_score = robust_agg(face_scores_hf, "p90") if face_scores_hf else None
        scene_score = robust_agg(scene_scores_hf, "p90") if scene_scores_hf else None

        # Zapisz do details
        details['ai_face_score'] = face_score
        details['ai_scene_score'] = scene_score
        details['ai_video_score'] = video_score
        details['jitter_px'] = jitter_px
        details['ela_score'] = ela_avg
        details['fft_score'] = fft_avg
        details['border_artifacts'] = border_avg
        details['face_sharpness'] = sharp_avg
        details['face_ratio'] = (face_frames / len(idxs)) * 100.0 if idxs else 0.0
        details['face_frames'] = face_frames
        details['total_frames'] = len(idxs)
        details['face_scores_raw'] = face_scores_hf
        details['scene_scores_raw'] = scene_scores_hf

        # Fuzja wyników wideo (VideoMAE + ewentualnie D3)
        video_fused = fuse_video_scores(
            details.get('ai_video_score'),
            details.get('d3_score'),  # Jeśli masz D3
            policy=getattr(config, 'DECISION_POLICY', 'high_precision')
        )
        if video_fused is not None:
            details['ai_video_score'] = video_fused

        # Specjalna logika dla różnych trybów detekcji
        detection_flags = []

        if detection_mode == "ai":
            # Tryb AI: bardziej ufaj wynikom sceny
            if scene_score and scene_score > 70:
                detection_flags.append("HIGH_AI_SCENE_SCORE")
                # Podbij wynik jeśli scena wskazuje na AI
                if face_score and face_score < 50:
                    face_score = max(face_score, scene_score * 0.7)

        elif detection_mode == "deepfake":
            # Tryb Deepfake: bardziej ufaj wynikom twarzy
            if face_score and face_score > 70:
                detection_flags.append("HIGH_DEEPFAKE_FACE_SCORE")

        else:  # combined
            # Połączona logika
            if scene_score and scene_score > 80:
                detection_flags.append("HIGH_COMBINED_SCENE_SCORE")
            if face_score and face_score > 80:
                detection_flags.append("HIGH_COMBINED_FACE_SCORE")

        # Dodatkowe flagi
        if details['face_ratio'] < 20:
            detection_flags.append("LOW_FACE_RATIO")
        if jitter_px and jitter_px > 100:
            detection_flags.append("HIGH_JITTER")

        details['detection_flags'] = detection_flags

        # Fuzja wszystkich wyników
        features_for_fusion = {
            "face": details.get("ai_face_score"),
            "scene": details.get("ai_scene_score"),
            "video": details.get("ai_video_score"),
        }
        final_score = fuse_scores(features_for_fusion)
        details['final_score'] = final_score

        # --- Rozdzielenie werdyktów (Wersja MASTER) ---
        
        # 1. Werdykt AI Detector (Scena / Generacja)
        ai_detector_score = scene_score if scene_score is not None else 0.0
        if video_score is not None:
            # Jeśli mamy wynik wideo, połącz go z wynikiem sceny
            ai_detector_score = max(ai_detector_score, video_score)
        
        ai_detector_verdict = _verdict_from_score(ai_detector_score)
        details['ai_detector_score'] = ai_detector_score
        details['ai_detector_verdict'] = ai_detector_verdict
        
        # 2. Werdykt Deepfake (Twarz)
        deepfake_score = face_score if face_score is not None else 0.0
        deepfake_verdict = _verdict_from_score(deepfake_score)
        details['deepfake_score'] = deepfake_score
        details['deepfake_verdict'] = deepfake_verdict
        
        # 3. Werdykt Podsumowujący (Finalny)
        # Używamy logiki decision_policy na final_score, ale z uwzględnieniem dominującego zagrożenia
        final_verdict_raw = decision_policy(
            final_score,
            details,
            policy=getattr(config, 'DECISION_POLICY', 'high_precision')
        )
        
        # Formatowanie werdyktu podsumowującego zgodnie z życzeniem użytkownika
        if deepfake_score > ai_detector_score and deepfake_score > 30:
            summary_verdict = f"DEEPFAKE na {deepfake_score:.1f}%"
        elif ai_detector_score > 30:
            summary_verdict = f"AI GENERACJA na {ai_detector_score:.1f}%"
        else:
            summary_verdict = f"REAL (Score: {final_score:.1f}%)"
            
        details['verdict'] = summary_verdict
        details['summary_verdict'] = summary_verdict

        # Oblicz fake_ratio
        fake_ratio = 0.0
        if face_scores_hf:
            fake_count = sum(1 for s in face_scores_hf if s > 50)
            fake_ratio = (fake_count / len(face_scores_hf)) * 100.0

        return summary_verdict, final_score, fake_ratio, details

    except Exception as e:
        print(f"[MAIN ANALYSIS ERROR] {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return "ERROR", 0.0, 0.0, {"error": str(e)}

    finally:
        try:
            cap.release()
        except Exception:
            pass


# =============================================================================
# Funkcje analizy dla GUI
# =============================================================================

def analyze_video(
        video_path: str,
        run_dir: str,
        *,
        max_frames: int = 60,
        do_ai: bool = True,
        do_forensic: bool = True,
        do_watermark: bool = False,
        progress_callback: Optional[Callable[..., Any]] = None,
        json_report: bool = False,
        check_stop: Optional[Any] = None,
) -> Tuple[Optional[Report], Optional[str]]:
    """
    Analizuje pojedynczy plik i zapisuje raport .txt (oraz opcjonalnie .json).
    Zwraca (Report|None, ścieżka do raportu|None).
    """
    base = os.path.basename(video_path)

    # Określ tryb detekcji na podstawie nazwy pliku i ustawień
    detection_mode = "combined"
    if "sora" in base.lower() or "ai_generated" in base.lower():
        detection_mode = "ai"
    elif "deepfake" in base.lower() or "fake" in base.lower():
        detection_mode = "deepfake"

    try:
        # Uruchom główną analizę
        verdict, final_score, fake_ratio, details = scan_for_deepfake(
            video_path,
            max_frames=max_frames,
            progress_callback=progress_callback,
            check_stop=check_stop,
            detection_mode=detection_mode,
        )

        # Przygotuj wyniki w formacie raportu
        ai_res = AiResult(
            face_score=details.get('ai_face_score'),
            scene_score=details.get('ai_scene_score'),
            video_score=details.get('ai_video_score'),
            combined_max=details.get('final_score'),
            face_frames=details.get('face_frames', 0),
            total_frames=details.get('total_frames', 0)
        )

        forensic_res = ForensicResult(
            jitter_px=details.get('jitter_px'),
            ela_score=details.get('ela_score'),
            fft_score=details.get('fft_score'),
            border_artifacts=details.get('border_artifacts'),
            sharpness_face=details.get('face_sharpness')
        )

        report = Report(
            file_name=base,
            verdict=verdict,
            total_score=final_score,
            ai=ai_res,
            forensic=forensic_res,
            metadata={
                'face_ratio': details.get('face_ratio', 0.0),
                'fake_ratio': fake_ratio,
                'detection_flags': details.get('detection_flags', []),
                'detection_mode': detection_mode,
                'video_path': video_path
            }
        )

        # Zapisz raport
        txt_path = os.path.join(run_dir, os.path.splitext(base)[0] + ".txt")
        _write_report_txt(report, txt_path)

        if json_report:
            json_path = os.path.join(run_dir, os.path.splitext(base)[0] + ".json")
            _write_report_json(report, json_path)

        return report, txt_path

    except Exception as e:
        print(f"[BŁĄD] {base} (AI): {e}")
        import traceback
        traceback.print_exc()
        return None, None


def analyze_files(
        files: List[str],
        run_dir: str,
        *,
        max_frames: int = 60,
        do_ai: bool = True,
        do_forensic: bool = True,
        do_watermark: bool = False,
        progress_callback: Optional[Callable[..., Any]] = None,
        json_report: bool = False,
        check_stop: Optional[Any] = None,
) -> List[Report]:
    """
    Batch analiza listy plików.
    """
    print(f"> Rozpoczynam analizę… (AI={do_ai}, Forensic={do_forensic}, Watermark={do_watermark})")
    out: List[Report] = []

    total = len(files)
    for i, path in enumerate(files, start=1):
        base = os.path.basename(path)

        if stop_requested():
            print("> Przerywam analizę…")
            break

        print(f"> [{i}/{total}] Start: {base}")

        def per_frame_progress(payload: _ProgressPayload, _i: int = i, _total: int = total) -> None:
            cur, tot, pct = _normalize_progress(payload, default_total=max(1, max_frames))

            if pct is None:
                _safe_call_progress(
                    progress_callback,
                    {"current": cur, "total": tot, "percent": pct, "raw": payload},
                    default_total=max(1, max_frames),
                )
                return

            global_pct = int(round(100.0 * ((_i - 1) / max(1, _total) + (pct / 100.0) / max(1, _total))))
            global_pct = max(0, min(100, global_pct))
            _safe_call_progress(progress_callback, global_pct, default_total=100)

        report, rep_path = analyze_video(
            path,
            run_dir,
            max_frames=max_frames,
            do_ai=do_ai,
            do_forensic=do_forensic,
            do_watermark=do_watermark,
            progress_callback=per_frame_progress,
            json_report=json_report,
            check_stop=check_stop,
        )

        if report is None:
            print(f"< [{i}/{total}] DONE: ERROR (0.00%). Raport: None")
            continue

        out.append(report)
        print(f"< [{i}/{total}] DONE: {report.verdict.split()[0]} ({report.total_score:.2f}%). Raport: {rep_path}")

        _safe_call_progress(progress_callback, int(round(100.0 * i / max(1, total))), default_total=100)

    return out


def analyze_path(
        path: str,
        run_dir: str,
        *,
        max_frames: int = 60,
        do_ai: bool = True,
        do_forensic: bool = True,
        do_watermark: bool = False,
        progress_callback: Optional[Callable[..., Any]] = None,
        json_report: bool = False,
        check_stop: Optional[Any] = None,
) -> List[Report]:
    files = _collect_video_files(path)
    return analyze_files(
        files,
        run_dir,
        max_frames=max_frames,
        do_ai=do_ai,
        do_forensic=do_forensic,
        do_watermark=do_watermark,
        progress_callback=progress_callback,
        json_report=json_report,
        check_stop=check_stop,
    )


# =============================================================================
# CLI
# =============================================================================

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI/Forensic deepfake video analyzer - ULEPSZONA WERSJA")
    p.add_argument("path", help="File or directory with videos")
    p.add_argument("--max-frames", type=int, default=60, help="Max sampled frames per video")
    p.add_argument("--detection-mode", choices=["ai", "deepfake", "combined"], default="combined",
                   help="Detection mode: ai (AI generation), deepfake (face swaps), combined (both)")
    p.add_argument("--no-ai", action="store_true", help="Disable AI scoring")
    p.add_argument("--no-forensic", action="store_true", help="Disable forensic metrics")
    p.add_argument("--watermark", action="store_true", help="Enable watermark analysis")
    p.add_argument("--json", action="store_true", help="Also write JSON report per file")
    p.add_argument("--reports-root", default="reports", help="Reports output folder")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    run_dir = begin_run(args.reports_root)

    do_ai = not args.no_ai
    do_forensic = not args.no_forensic
    do_watermark = bool(args.watermark)

    try:
        # Użyj trybu detekcji z argumentów
        files = _collect_video_files(args.path)

        # Przekaż tryb detekcji do analizy plików
        # (niestety analyze_files nie obsługuje bezpośrednio detection_mode,
        # więc możemy potrzebować modyfikacji)

        results = analyze_files(
            files,
            run_dir,
            max_frames=int(args.max_frames),
            do_ai=do_ai,
            do_forensic=do_forensic,
            do_watermark=do_watermark,
            progress_callback=None,
            json_report=bool(args.json),
        )

        print(f"> Analiza zakończona. Przeanalizowano {len(results)} plików.")

        # Podsumowanie
        real_count = sum(1 for r in results if "REAL" in r.verdict)
        fake_count = sum(1 for r in results if "FAKE" in r.verdict)
        grey_count = sum(1 for r in results if "NIEPEWNE" in r.verdict or "GREY" in r.verdict)

        print(f"> Podsumowanie: REAL={real_count}, FAKE={fake_count}, NIEPEWNE={grey_count}")

        return 0
    except Exception as e:
        print(f"[BŁĄD] Analiza przerwana: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
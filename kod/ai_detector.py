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

Uwaga: W trybie watermark-only (AI=False i Forensic=False) analiza nie uruchamia modeli HF,
żeby nie generować błędów i opóźnień (np. wymaganych rozmiarów wejścia).
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
        weights = getattr(config, 'FUSE_WEIGHTS', {"video": 0.70, "face": 0.30, "scene": 0.0})
        acc = 0.0
        wsum = 0.0
        for k, w in weights.items():
            v = safe_float(features.get(k))
            if v is None or w <= 0:
                continue
            acc += v * w
            wsum += w
        return clamp(acc / wsum) if wsum > 0 else 0.0


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
    real_max = getattr(config, "REAL_MAX", 25.0)
    fake_min = getattr(config, "FAKE_MIN", 45.0)

    if score <= real_max:
        return "REAL (PRAWDOPODOBNE)"
    if score >= fake_min:
        return "FAKE (PRAWDOPODOBNE)"
    return "NIEPEWNE / GREY ZONE"


def _write_report_txt(report: Report, out_path: str) -> None:
    lines: List[str] = []
    lines.append(f"Plik: {report.file_name}")
    lines.append("")
    lines.append(f"WERDYKT: {report.verdict}")
    lines.append(f"Wynik łączny (Score): {report.total_score:.2f}%")
    lines.append("")

    # Wylicz dodatkowe metryki z metadata
    if report.metadata:
        face_ratio = report.metadata.get('face_ratio', 0.0)
        fake_ratio = report.metadata.get('fake_ratio', 0.0)
        detection_mode = report.metadata.get('detection_mode', 'combined')

        lines.append(f"Tryb detekcji: {detection_mode}")
        lines.append(f"Wskaźnik twarzy: {face_ratio:.1f}%")
        lines.append(f"Wskaźnik fake: {fake_ratio:.1f}%")

        if report.metadata.get('watermark_found') is not None:
            lines.append(f"Watermark: {'TAK' if report.metadata.get('watermark_found') else 'NIE'}")
            if report.metadata.get('watermark_label'):
                lines.append(f"Watermark label: {report.metadata.get('watermark_label')}")

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
    try:
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if cnt > 0:
            return cnt
    except Exception:
        pass
    return 0


def _sample_frame_indices(total_frames: int, max_frames: int) -> List[int]:
    if total_frames <= 0:
        return list(range(max_frames))
    if max_frames <= 0:
        return []
    if total_frames <= max_frames:
        return list(range(total_frames))
    step = total_frames / float(max_frames)
    idxs = [int(i * step) for i in range(max_frames)]
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
    haar = _get_haar()
    if haar is None:
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if faces is None or len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return int(x), int(y), int(w), int(h)


def _detect_all_faces(frame_bgr) -> List[Tuple[int, int, int, int]]:
    haar = _get_haar()
    if haar is None:
        return []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if faces is None:
        return []
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


# =============================================================================
# Forensic metrics ... (bez zmian)
# =============================================================================

# ... (reszta pliku bez zmian aż do scan_for_deepfake) ...


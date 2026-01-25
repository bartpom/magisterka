# config.py - ZOPTYMALIZOWANA KONFIGURACJA DLA DETEKCJI DEEPFAKE
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
from typing import List, Dict, Any

# ============ GŁÓWNE USTAWIENIA ============
REPORTS_BASE_DIR = os.getenv("REPORTS_BASE_DIR", "ai_reports")
TIMESTAMP_FMT = "%Y-%m-%d_%H-%M-%S"

# ============ URZĄDZENIE ============
PREFER_CUDA = True  # jeśli masz GPU + CUDA w torch, użyj GPU

# ============ TRYB DETEKCJI ============
# Dostępne tryby: "high_precision", "high_recall", "balanced"
DECISION_POLICY = "high_precision"  # Dla minimalizacji false positives

# ============# Progi detekcji ============
# Ustawione dla minimalizacji false positives (Wersja FINAL v3 - Master)
REAL_MAX = 30.0  # Poniżej = REAL
FAKE_MIN = 60.0  # Powyżej = FAKE
# ============ MODELE HF DLA OBRAZÓW ============
# Modele dla twarzy (face detection)
HF_FACE_MODELS = [
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "dima806/deepfake_vs_real_image_detection",
    "buildborderless/CommunityForensics-DeepfakeDet-ViT",
    "hamzenium/ViT-Deepfake-Classifier",
]

# Modele dla sceny/Sora (scene detection)
HF_SCENE_MODELS = [
    "prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2",
]

# Modele w formacie kompatybilnym z GUI
HF_IMAGE_MODELS = [
    {"id": "prithivMLmods/Deep-Fake-Detector-v2-Model", "scopes": {"face": True, "scene": False}},
    {"id": "dima806/deepfake_vs_real_image_detection", "scopes": {"face": True, "scene": False}},
    {"id": "buildborderless/CommunityForensics-DeepfakeDet-ViT", "scopes": {"face": True, "scene": False}},
    {"id": "hamzenium/ViT-Deepfake-Classifier", "scopes": {"face": True, "scene": False}},
    {"id": "prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2", "scopes": {"face": False, "scene": True}},
]

# ============ MODELE WIDEO ============
# Główny model VideoMAE
HF_VIDEO_MODEL = "shylhy/videomae-large-finetuned-deepfake-subset"

# Alternatywne modele wideo (do ewentualnej fuzji)
HF_VIDEO_MODELS = [
    "shylhy/videomae-large-finetuned-deepfake-subset",
    "Ammar2k/videomae-base-finetuned-deepfake-subset",
    "Hemgg/deepfake_model_Video-MAE"
]

# ============ SAMPLING / ANALIZA ============
ANALYZE_NUM_FRAMES = 48  # Więcej klatek dla lepszej dokładności
VIDEO_NUM_FRAMES = 32  # Ile klatek podać do modelu wideo

# Percentyl dla agregacji z wielu klatek
FRAME_PERCENTILE = 90.0  # 90-ty percentyl - mniej wrażliwy na outlierów

# ============ D3 (CLIP-temporal) – opcjonalne ============
D3_ENABLED = True
D3_MODEL_ID = "openai/clip-vit-base-patch32"
D3_NUM_FRAMES = 24
D3_MOTION_MIN = 0.0
D3_RATIO_REF = 0.55
D3_RATIO_SCALE = 18.0
D3_LOWER_RATIO_MEANS_FAKE = True

# ============ FUSJA WYNIKÓW ============
# Wagi zoptymalizowane pod precyzję (mniej false positives)
FUSE_WEIGHTS = {
    "video": 0.50,  # VideoMAE - najważniejszy, bo analizuje całe wideo
    "face": 0.35,  # Twarz - ważna ale nie decydująca
    "scene": 0.15,  # Scena - pomocnicza, często daje false positives
}

# Tryb fuzji
FUSE_MODE = "weighted"  # "weighted", "noisy_or", "max"

# Tłumienie wyniku sceny gdy mamy twarz lub wideo (dla uniknięcia false positives)
SUPPRESS_SCENE_WHEN_FACE_OR_VIDEO = True
SUPPRESS_SCENE_ONLY_IF_VIDEO = True

# Forensic (jitter/blink/ela/fft) ma wpływać tylko gdy AI już jest podejrzane
FORENSIC_GATE_MIN = 60.0

# ============ HIGH PRECISION TUNING ============
# (Używane gdy DECISION_POLICY = "high_precision")

# Progi dla potwierdzania FAKE (wymagaj wysokiej pewności)
HP_FAKE_CONFIRM_FACE = 85.0  # Wymagaj bardzo wysokiego wyniku twarzy
HP_FAKE_CONFIRM_VIDEO = 80.0  # Wymagaj wysokiego video
HP_FAKE_CONFIRM_SCENE = 90.0  # Wymagaj bardzo wysokiego sceny
HP_FAKE_CONFIRM_RATIO = 30.0  # % podejrzanych próbek (0..100)

# Minimalna ilość sygnałów do potwierdzenia FAKE
HP_FAKE_MIN_SIGNALS = 2

# Blokowanie REAL jeśli jakikolwiek sygnał jest podejrzany
HP_REAL_BLOCK_VIDEO = 40.0  # Blokuj REAL jeśli video > 40%
HP_REAL_BLOCK_FACE = 60.0  # Blokuj REAL jeśli face > 60%
HP_REAL_BLOCK_RATIO_PCT = 20.0  # Blokuj REAL jeśli ratio > 20%
HP_REAL_BLOCK_BORDER = 0.06  # Border artifacts threshold
HP_REAL_BLOCK_D3 = 30.0  # d3 score (0..100)

# Minimalna ilość klatek z twarzą dla wiarygodnego wyniku REAL
HP_MIN_FACE_FRAMES = 5
HP_MIN_FACE_AREA_RATIO = 0.020  # 2% obszaru klatki
HP_MIN_FACE_SIZE_PX = 80  # Minimalny rozmiar twarzy w pikselach

# ============ HIGH RECALL TUNING ============
# (Używane gdy DECISION_POLICY = "high_recall")

# Progi dla detekcji FAKE (bardziej agresywne)
HR_FACE_STRONG = 70.0
HR_VIDEO_STRONG = 60.0
HR_VIDEO_MED = 30.0
HR_FACE_MED = 40.0
HR_VIDEO_PAIR = 20.0
HR_FAKE_RATIO_MED = 15.0  # 15% a NIE 0.15 (bo fake_ratio w %)
HR_BORDER_MED = 0.055
HR_ELA_MED = 47.0

# "Near-threshold push" (dopychanie bliskich wyników do FAKE)
HR_NEAR_FAKE_MARGIN = 15.0  # ile % poniżej FAKE_MIN uznajemy za "blisko"
HR_NEAR_FAKE_VIDEO = 12.0  # jeśli video >= 12% i raw blisko progu -> dopchnij
HR_NEAR_FAKE_FACE = 60.0  # jeśli face >= 60% i raw blisko progu -> dopchnij
HR_NEAR_FAKE_RATIO = 15.0  # dodatkowy warunek: fake_ratio >= 15% (w %)
HR_NEAR_FAKE_BORDER = 0.055
HR_NEAR_FAKE_ELA = 47.0

# Video sampling dla high recall (więcej klipów)
VIDEO_NUM_CLIPS_HIGH_RECALL = 11  # więcej klipów dla lepszego pokrycia

# Forensic forcing dla trudnych FN
HR_VIDEO_LOW_MAX = 10.0  # "video jest niskie" (<=)
HR_FORENSIC_FORCE_HITS = 2  # ile warunków forensic musi "kliknąć"
HR_FORENSIC_MIN_FACE_SAMPLES = 10  # zabezpieczenie przed face_samples=1
HR_BORDER_FORCE = 0.060  # border_artifacts (0..1)
HR_ELA_FORCE = 50.0  # ELA
HR_JITTER_FORCE = 200.0  # jitter_px
HR_SHARP_FORCE = 25.0  # sharpness (Laplacian var)
HR_FFT_FORCE = 140.0  # FFT score

# Anti-FP (video overconfidence -> GREY)
HR_VIDEO_CONFLICT_DEMOTE = True
HR_CONFLICT_VIDEO_STRONG = 90.0
HR_CONFLICT_FACE_MAX = 60.0
HR_CONFLICT_FAKE_RATIO_MAX = 10.0  # w %, bo fake_ratio masz w %
HR_CONFLICT_BORDER_MAX = 0.050
HR_CONFLICT_ELA_MAX = 50.0

# ============ DETEKCJA SORA/GENERACJI AI ============
SORA_DETECTION_ENABLED = True
SORA_SCENE_THRESHOLD = 85.0  # Próg dla detekcji Sora/Generacji AI
SORA_MIN_FRAMES = 10  # Minimalna ilość klatek do analizy

# ============ WATERMARK DETEKCJA ============
WATERMARK_KEYWORDS = ["sora", "openai", "generated", "ai video", "made with", "ai generated"]
WATERMARK_MAX_FRAMES = 300
WATERMARK_STRIDE = 5
WATERMARK_MIN_SAVE_GAP_SEC = 1.0
WATERMARK_BASE_DIR = "suspicious_frames"
WATERMARK_YOLO_MODEL_ID = ""  # np. "username/yolo-watermark-detector"

# ============ BLINK / MEDIAPIPE ============
EAR_BLINK_THRESH = 0.19  # próg EAR (mniejszy => oko zamknięte)

# ============ AGGREGACJA WYNIKÓW ============
IMAGE_ENSEMBLE_AGG = "median"  # "mean" | "median" | "max" | "p90"
VIDEO_SCORE_AGG = "median"  # "mean" | "median" | "max" | "p90"
VIDEO_NUM_CLIPS = 5  # start/mid/end
VIDEO_CLIP_AGG = "max"  # "max" albo "mean"

# Dla modeli typu: AI vs Deepfake vs Real
COUNT_AI_AS_FAKE = True  # Traktuj AI-generated jako FAKE

# ============ WALIDACJA ============
MIN_FACE_RATIO = 20.0  # Minimalny % klatek z twarzą dla wiarygodności
MAX_JITTER = 200.0  # Maksymalny jitter (powyżej = podejrzane)
MIN_FACE_SAMPLES_FOR_REAL = 3  # Minimalna ilość próbek twarzy dla wyniku REAL

# ============ META-KALIBRATOR ============
# Jeśli chcesz: ustaw ścieżkę do pliku joblib z pipeline/estimatorem.
META_CALIBRATOR_PATH = os.getenv("META_CALIBRATOR_PATH", "")

# ============ ZAAWANSOWANE ============
# Jeśli masz problemy z wątkami na GPU, możesz wymusić serializację inferencji:
SERIALIZE_INFERENCE = None  # None = auto-detect, True = wymuś serializację

DEBUG_VIDEO_MODEL = False  # Debugowanie modelu wideo

# ============ GUI COMPATIBILITY ============
# GUI oczekuje config.THRESHOLDS
THRESHOLDS = {
    "FAKE_MIN": FAKE_MIN,
    "REAL_MAX": REAL_MAX,
}

# ============ RÓŻNE PROGI DLA RÓŻNYCH TRYBÓW ============
# Dostosuj w zależności od typu zawartości

# Dla filmów z twarzami (face deepfakes)
FACE_DEEPFAKE_THRESHOLDS = {
    "FAKE_MIN": 70.0,
    "REAL_MAX": 30.0,
    "MIN_FACE_RATIO": 30.0,
}

# Dla generacji AI/Sora (bez twarzy)
AI_GENERATION_THRESHOLDS = {
    "FAKE_MIN": 65.0,
    "REAL_MAX": 25.0,
    "MIN_FACE_RATIO": 5.0,  # Może nie mieć twarzy
}

# Dla ogólnych deepfake (mixed)
GENERAL_THRESHOLDS = {
    "FAKE_MIN": 75.0,
    "REAL_MAX": 25.0,
    "MIN_FACE_RATIO": 15.0,
}

# ============ MODE TRYBÓW DETEKCJI ============
# Definicje trybów detekcji dla aplikacji
DETECTION_MODES = {
    "ai": {
        "name": "AI Detection",
        "description": "Wykrywanie generacji AI (Sora, DALL-E, itp.)",
        "focus": "scene",  # Skup się na scenie, nie na twarzach
        "weights": {"video": 0.40, "face": 0.20, "scene": 0.40},
        "thresholds": AI_GENERATION_THRESHOLDS,
    },
    "deepfake": {
        "name": "Deepfake Detection",
        "description": "Wykrywanie deepfake twarzy",
        "focus": "face",  # Skup się na twarzach
        "weights": {"video": 0.40, "face": 0.50, "scene": 0.10},
        "thresholds": FACE_DEEPFAKE_THRESHOLDS,
    },
    "combined": {
        "name": "Combined Detection",
        "description": "Wykrywanie wszystkich typów manipulacji",
        "focus": "balanced",  # Zrównoważone podejście
        "weights": FUSE_WEIGHTS,  # Użyj głównych wag
        "thresholds": GENERAL_THRESHOLDS,
    },
    "watermark": {
        "name": "Watermark Detection",
        "description": "Wykrywanie znaków wodnych/napisów",
        "focus": "watermark",  # Tylko watermark
        "weights": {"watermark": 1.0},
        "thresholds": {"FAKE_MIN": 50.0, "REAL_MAX": 10.0},
    }
}


# ============ ZAAWANSOWANE FUNKCJE WALIDACJI ============

def get_mode_config(mode: str = "combined") -> Dict[str, Any]:
    """
    Zwraca konfigurację dla danego trybu detekcji.

    Args:
        mode: "ai", "deepfake", "combined", "watermark"

    Returns:
        Słownik z konfiguracją trybu
    """
    if mode not in DETECTION_MODES:
        mode = "combined"

    config = DETECTION_MODES[mode].copy()

    # Ustaw globalne progi na podstawie trybu
    global REAL_MAX, FAKE_MIN, FUSE_WEIGHTS
    REAL_MAX = config["thresholds"]["REAL_MAX"]
    FAKE_MIN = config["thresholds"]["FAKE_MIN"]

    if mode != "watermark":
        FUSE_WEIGHTS = config["weights"]

    # Aktualizuj THRESHOLDS dla GUI
    THRESHOLDS["FAKE_MIN"] = FAKE_MIN
    THRESHOLDS["REAL_MAX"] = REAL_MAX

    return config


def get_model_ids_for_scope(scope: str) -> List[str]:
    """
    Zwraca listę ID modeli HF dla danego zakresu.

    Args:
        scope: "face", "scene", "video"

    Returns:
        Lista ID modeli
    """
    if scope == "face":
        return HF_FACE_MODELS
    elif scope == "scene":
        return HF_SCENE_MODELS
    elif scope == "video":
        return [HF_VIDEO_MODEL]
    else:
        return []


def should_suppress_scene(face_score: float, video_score: float) -> bool:
    """
    Decyduje czy stłumić wynik sceny na podstawie wyników twarzy i wideo.
    Pomaga zmniejszyć false positives.
    """
    if not SUPPRESS_SCENE_WHEN_FACE_OR_VIDEO:
        return False

    if SUPPRESS_SCENE_ONLY_IF_VIDEO:
        # Tłum tylko jeśli mamy wynik wideo
        return video_score is not None and video_score > 20.0
    else:
        # Tłum jeśli mamy albo twarz albo wideo
        has_face_signal = face_score is not None and face_score > 30.0
        has_video_signal = video_score is not None and video_score > 20.0
        return has_face_signal or has_video_signal


def validate_face_detection(face_frames: int, total_frames: int, face_ratio: float) -> Dict[str, Any]:
    """
    Waliduje wyniki detekcji twarzy.

    Returns:
        Słownik z wynikami walidacji
    """
    result = {
        "is_valid": True,
        "warnings": [],
        "face_ratio": face_ratio,
    }

    # Sprawdź minimalny ratio twarzy
    min_ratio = getattr(config, "MIN_FACE_RATIO", 20.0)
    if face_ratio < min_ratio:
        result["warnings"].append(f"Low face ratio: {face_ratio:.1f}% < {min_ratio}%")
        result["is_valid"] = False

    # Sprawdź minimalną ilość klatek z twarzą
    min_face_frames = getattr(config, "MIN_FACE_SAMPLES_FOR_REAL", 3)
    if face_frames < min_face_frames:
        result["warnings"].append(f"Insufficient face frames: {face_frames} < {min_face_frames}")
        result["is_valid"] = False

    return result


def get_optimal_frame_count(mode: str = "combined") -> int:
    """
    Zwraca optymalną ilość klatek do analizy dla danego trybu.
    """
    base_frames = getattr(config, "ANALYZE_NUM_FRAMES", 48)

    if mode == "ai":
        # Dla AI potrzebujemy więcej klatek do analizy sceny
        return min(72, base_frames * 1.5)
    elif mode == "deepfake":
        # Dla deepfake twarzy mniej klatek ale lepsze próbkowanie
        return base_frames
    elif mode == "combined":
        return base_frames
    else:
        return base_frames


# ============ FUNKCJE POMOCNICZE ============

def now_str() -> str:
    """Zwraca aktualny timestamp w formacie dla raportów."""
    import datetime
    return datetime.datetime.now().strftime(TIMESTAMP_FMT)


def get_detection_mode_from_filename(filename: str) -> str:
    """
    Automatycznie określa tryb detekcji na podstawie nazwy pliku.
    """
    filename_lower = filename.lower()

    if any(keyword in filename_lower for keyword in ["sora", "ai_generated", "dalle", "midjourney", "generated"]):
        return "ai"
    elif any(keyword in filename_lower for keyword in ["deepfake", "faceswap", "face_swap"]):
        return "deepfake"
    elif any(keyword in filename_lower for keyword in ["watermark", "logo", "brand"]):
        return "watermark"
    else:
        return "combined"


# ============ INICJALIZACJA ============
# Ustaw domyślny tryb na starcie
_CURRENT_MODE = "combined"
get_mode_config(_CURRENT_MODE)

print(f"[CONFIG] Loaded with mode: {_CURRENT_MODE}, REAL_MAX: {REAL_MAX}, FAKE_MIN: {FAKE_MIN}")
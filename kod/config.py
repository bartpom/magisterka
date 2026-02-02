# config.py - ZOPTYMALIZOWANA KONFIGURACJA DLA DETEKCJI DEEPFAKE
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
from typing import List, Dict, Any

REPORTS_BASE_DIR = os.getenv("REPORTS_BASE_DIR", "ai_reports")
TIMESTAMP_FMT = "%Y-%m-%d_%H-%M-%S"

PREFER_CUDA = True

DECISION_POLICY = "high_precision"

REAL_MAX = 30.0
FAKE_MIN = 60.0

HF_FACE_MODELS = [
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "dima806/deepfake_vs_real_image_detection",
    "buildborderless/CommunityForensics-DeepfakeDet-ViT",
    "hamzenium/ViT-Deepfake-Classifier",
]

HF_SCENE_MODELS = [
    "prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2",
]

HF_IMAGE_MODELS = [
    {"id": "prithivMLmods/Deep-Fake-Detector-v2-Model", "scopes": {"face": True, "scene": False}},
    {"id": "dima806/deepfake_vs_real_image_detection", "scopes": {"face": True, "scene": False}},
    {"id": "buildborderless/CommunityForensics-DeepfakeDet-ViT", "scopes": {"face": True, "scene": False}},
    {"id": "hamzenium/ViT-Deepfake-Classifier", "scopes": {"face": True, "scene": False}},
    {"id": "prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2", "scopes": {"face": False, "scene": True}},
]

HF_VIDEO_MODEL = "shylhy/videomae-large-finetuned-deepfake-subset"

HF_VIDEO_MODELS = [
    "shylhy/videomae-large-finetuned-deepfake-subset",
    "Ammar2k/videomae-base-finetuned-deepfake-subset",
    "Hemgg/deepfake_model_Video-MAE"
]

ANALYZE_NUM_FRAMES = 48
VIDEO_NUM_FRAMES = 32
FRAME_PERCENTILE = 90.0

D3_ENABLED = True
D3_MODEL_ID = "openai/clip-vit-base-patch32"
D3_NUM_FRAMES = 24
D3_MOTION_MIN = 0.0
D3_RATIO_REF = 0.55
D3_RATIO_SCALE = 18.0
D3_LOWER_RATIO_MEANS_FAKE = True

# ZMIANA: mniej ufamy scenie, bo u Ciebie saturuje do ~100%
FUSE_WEIGHTS = {
    "video": 0.60,
    "face": 0.30,
    "scene": 0.10,
}

FUSE_MODE = "weighted"

SUPPRESS_SCENE_WHEN_FACE_OR_VIDEO = True
SUPPRESS_SCENE_ONLY_IF_VIDEO = True

FORENSIC_GATE_MIN = 60.0

HP_FAKE_CONFIRM_FACE = 85.0
HP_FAKE_CONFIRM_VIDEO = 80.0
HP_FAKE_CONFIRM_SCENE = 90.0
HP_FAKE_CONFIRM_RATIO = 30.0
HP_FAKE_MIN_SIGNALS = 2

HP_REAL_BLOCK_VIDEO = 40.0
HP_REAL_BLOCK_FACE = 60.0
HP_REAL_BLOCK_RATIO_PCT = 20.0
HP_REAL_BLOCK_BORDER = 0.06
HP_REAL_BLOCK_D3 = 30.0

HP_MIN_FACE_FRAMES = 5
HP_MIN_FACE_AREA_RATIO = 0.020
HP_MIN_FACE_SIZE_PX = 80

HR_FACE_STRONG = 70.0
HR_VIDEO_STRONG = 60.0
HR_VIDEO_MED = 30.0
HR_FACE_MED = 40.0
HR_VIDEO_PAIR = 20.0
HR_FAKE_RATIO_MED = 15.0
HR_BORDER_MED = 0.055
HR_ELA_MED = 47.0

HR_NEAR_FAKE_MARGIN = 15.0
HR_NEAR_FAKE_VIDEO = 12.0
HR_NEAR_FAKE_FACE = 60.0
HR_NEAR_FAKE_RATIO = 15.0
HR_NEAR_FAKE_BORDER = 0.055
HR_NEAR_FAKE_ELA = 47.0

VIDEO_NUM_CLIPS_HIGH_RECALL = 11

HR_VIDEO_LOW_MAX = 10.0
HR_FORENSIC_FORCE_HITS = 2
HR_FORENSIC_MIN_FACE_SAMPLES = 10
HR_BORDER_FORCE = 0.060
HR_ELA_FORCE = 50.0
HR_JITTER_FORCE = 200.0
HR_SHARP_FORCE = 25.0
HR_FFT_FORCE = 140.0

HR_VIDEO_CONFLICT_DEMOTE = True
HR_CONFLICT_VIDEO_STRONG = 90.0
HR_CONFLICT_FACE_MAX = 60.0
HR_CONFLICT_FAKE_RATIO_MAX = 10.0
HR_CONFLICT_BORDER_MAX = 0.050
HR_CONFLICT_ELA_MAX = 50.0

SORA_DETECTION_ENABLED = True
SORA_SCENE_THRESHOLD = 85.0
SORA_MIN_FRAMES = 10

WATERMARK_KEYWORDS = ["sora", "openai", "generated", "ai video", "made with", "ai generated"]
WATERMARK_MAX_FRAMES = 300
WATERMARK_STRIDE = 5
WATERMARK_MIN_SAVE_GAP_SEC = 1

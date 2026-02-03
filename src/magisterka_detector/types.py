"""Core types and dataclasses for the detector API."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

# Callback types
ProgressCb = Callable[[int, int], Any]  # (current, total)
CancelCb = Callable[[], bool]

# Enums
MediaKind = Literal["video", "image"]
DetectMode = Literal["ai", "deepfake", "combined"]
PolicyName = Literal["high_precision", "high_recall", "balanced"]


@dataclass
class AnalyzeOptions:
    """Configuration for analyze_media() / analyze_batch()."""
    
    mode: DetectMode = "combined"
    max_frames: int = 60
    
    # Feature extractors
    enable_hf_face: bool = True
    enable_hf_scene: bool = True
    enable_videomae: bool = True
    enable_forensic: bool = True
    enable_enhanced: bool = False
    enable_watermark: bool = False
    
    # Scoring
    decision_policy: PolicyName = "high_precision"
    calibration_thresholds_path: Optional[str] = None
    
    # Reporting
    reports_root: str = "reports"
    write_txt: bool = True
    write_json: bool = False


@dataclass
class AnalysisFeatures:
    """Raw features extracted from media file."""
    
    media_kind: MediaKind
    path: str
    
    # AI signals (0..100 scale)
    ai_face_score: Optional[float] = None
    ai_scene_score: Optional[float] = None
    ai_video_score: Optional[float] = None
    
    # Forensic signals
    jitter_px: Optional[float] = None
    ela_score: Optional[float] = None
    fft_score: Optional[float] = None
    border_artifacts: Optional[float] = None
    face_sharpness: Optional[float] = None
    face_ratio: Optional[float] = None
    face_frames: int = 0
    total_frames: int = 0
    
    # Debug/raw data
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Final analysis result with verdict and scoring."""
    
    verdict: str
    final_score: float
    features: AnalysisFeatures
    flags: List[str] = field(default_factory=list)
    
    # Report paths (if written)
    report_txt_path: Optional[str] = None
    report_json_path: Optional[str] = None
    report_folder: Optional[str] = None

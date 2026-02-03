"""Video frame sampling and basic OpenCV utilities."""

import cv2
import numpy as np
from typing import List, Optional, Tuple


def detect_media_kind(path: str) -> str:
    """Detect if file is video or image.
    
    Returns:
        "video" or "image"
    """
    ext = path.lower().split('.')[-1]
    if ext in {'mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'm4v'}:
        return "video"
    elif ext in {'jpg', 'jpeg', 'png', 'bmp', 'webp', 'tif', 'tiff'}:
        return "image"
    else:
        # Try to open as video first
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            cap.release()
            return "video"
        return "image"


def sample_frame_indices(total_frames: int, max_frames: int) -> List[int]:
    """Generate evenly spaced frame indices for sampling.
    
    Args:
        total_frames: Total number of frames in video
        max_frames: Maximum frames to sample
        
    Returns:
        List of frame indices
    """
    if total_frames <= max_frames:
        return list(range(total_frames))
    
    step = total_frames / max_frames
    return [int(i * step) for i in range(max_frames)]


def extract_frames(
    video_path: str,
    max_frames: int = 60,
    progress_callback=None,
    check_stop=None,
) -> Tuple[List[np.ndarray], int]:
    """Extract frames from video.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        progress_callback: Optional callback(current, total)
        check_stop: Optional callback() -> bool to check for cancellation
        
    Returns:
        (list of frames as numpy arrays, total_frame_count)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")
    
    indices = sample_frame_indices(total_frames, max_frames)
    frames = []
    
    for i, frame_idx in enumerate(indices):
        if check_stop and check_stop():
            break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret and frame is not None:
            frames.append(frame)
        
        if progress_callback:
            progress_callback(i + 1, len(indices))
    
    cap.release()
    return frames, total_frames


def load_image(image_path: str) -> np.ndarray:
    """Load a single image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (BGR format)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    return img

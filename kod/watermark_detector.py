from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

@dataclass(frozen=True)
class OCRHit:
    text: str
    conf: float      # 0..1
    frame_idx: int
    bbox: Optional[Tuple[int, int, int, int]] = None


def watermark_score_from_ocr_hits(
    hits: Sequence[OCRHit],
    min_conf: float = 0.70,
    min_distinct_frames: int = 3,
    require_repeat_text: bool = True
) -> float:
    """
    High-precision: watermark zwykle jest stały, więc oczekujemy powtórzeń.
    Zwraca score 0..100.
    """
    good = [h for h in hits if h.conf >= min_conf and h.text and h.text.strip()]
    if not good:
        return 0.0

    frames = {h.frame_idx for h in good}
    if len(frames) < min_distinct_frames:
        return 0.0

    def norm(s: str) -> str:
        s = s.lower().strip()
        return " ".join(s.split())

    if require_repeat_text:
        buckets = {}
        for h in good:
            buckets.setdefault(norm(h.text), []).append(h)

        best_text, best_hits = max(buckets.items(), key=lambda kv: len({x.frame_idx for x in kv[1]}))
        best_frames = {h.frame_idx for h in best_hits}
        if len(best_frames) < min_distinct_frames:
            return 0.0

        avg_conf = sum(h.conf for h in best_hits) / max(1, len(best_hits))
        frame_factor = min(1.0, len(best_frames) / 8.0)
        return float(max(0.0, min(100.0, 100.0 * frame_factor * avg_conf)))

    avg_conf = sum(h.conf for h in good) / max(1, len(good))
    frame_factor = min(1.0, len(frames) / 8.0)
    return float(max(0.0, min(100.0, 100.0 * frame_factor * avg_conf)))




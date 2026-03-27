"""Parametry fuzji heurystyk AI-video.

Cel: ograniczyć konflikty przy kolejnych PR-ach. Strojenie progów odbywa się tutaj,
bez ciągłego edytowania `evaluate.py`.
"""

LOW_TEXTURE_THRESHOLD = 2
HF_RATIO_THRESHOLD = 0.15
MAX_AREA_RATIO_THRESHOLD = 0.18
LOWER_THIRD_HARD_THRESHOLD = 0.60
POINTS_THRESHOLD_DEFAULT = 4
POINTS_THRESHOLD_SWEEP = [3, 4, 5]

"""Parametry fuzji heurystyk AI-video.

Cel: ograniczyć konflikty przy kolejnych PR-ach. Strojenie progów odbywa się tutaj,
bez ciągłego edytowania `evaluate.py`.
"""

LOW_TEXTURE_THRESHOLD = 1
HF_RATIO_THRESHOLD = 0.15
MAX_AREA_RATIO_THRESHOLD = 0.18
LOWER_THIRD_HARD_THRESHOLD = 0.60
POINTS_THRESHOLD_DEFAULT = 4
POINTS_THRESHOLD_SWEEP = [3, 4, 5]

# Broadcast trap heuristics
LOWER_THIRD_HARD_UPPER_MAX = 0.20
SCOREBOARD_HF_MIN = 0.65
BILLBOARD_CENTER_RATIO_MIN = 0.55
BILLBOARD_GLOBAL_MOTION_MIN = 10.0
BILLBOARD_TEXTURE_MIN = 1200.0

# Final Detector Evaluation Report

## Overall Metrics
| Dataset | Recall / FPR | TP | FP | TN | FN |
|---|---|---|---|---|---|
| ai_baseline (40 videos) | 0.5250 | 21 | - | - | 19 |
| adv_compressed (27 videos) | 0.6296 | 17 | - | - | 10 |
| adv_cropped (27 videos) | 0.6667 | 18 | - | - | 9 |
| adv_fp_trap (37 videos) | FPR 0.1081 | - | 4 | 33 | - |

## Per-Generator Recall (new videos only)
| Generator | Detected | Total | Recall |
|---|---|---|---|
| Runway Gen-4 Turbo | 3 | 5 | 0.60 |
| Kling | 0 | 3 | 0.00 |
| Pika | 2 | 5 | 0.40 |

## Signal Contribution
| Signal | Videos where signal was decisive |
|---|---|
| Optical Flow | 60 |
| FFT | 7 |
| rivaGAN | 0 |
| C2PA | 3 |
| high_score_override | 1 (ai_runway_05) |

## Known Limitations
- Kling nature videos (organic motion, natural textures): 0/3 detected
- C2PA coverage is still limited in this dataset (5/40 ai_baseline videos with Content Credentials)
- Pika abstract/futuristic scenes: score <= 2, below detection threshold

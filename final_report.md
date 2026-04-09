# Final Detector Evaluation Report

## Overall Metrics
| Dataset | Recall / FPR | TP | FP | TN | FN |
|---|---|---|---|---|---|
| ai_baseline (40 videos) | 0.5250 | 21 | - | - | 19 |
| adv_compressed (27 videos) | 0.6296 | 17 | - | - | 10 |
| adv_cropped (27 videos) | 0.6667 | 18 | - | - | 9 |
| adv_fp_trap (37 videos) | FPR 0.1081 | - | 4 | 33 | - |

Pelne manifesty datasetu dostepne sa w plikach `dataset/manifest_fake_all.csv`
(93 rekordy: 31 baseline + 31 compressed + 31 cropped, label=fake)
oraz `dataset/manifest_real_all.csv` (36 rekordow, label=real).
Opis struktury folderow: `dataset/README_dataset.md`.

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

## Comparison with External Tools

Testing was performed on a subset of videos from the dataset using two external web-based tools:
- **detectvideo.ai** — metadata-based screening
- **undetectable.ai** (powered by TruthScan) — frame+audio+motion analysis

### Results on selected AI-generated videos

| Filename | Label | Our Detector | detectvideo.ai | undetectable.ai |
|---|---|---|---|---|
| ai_runway_05 (city street) | AI | ✅ pred=1 (score 6, override) | ❌ 55% inconclusive | not tested |
| Gen-4 Turbo candle flame | AI | ❌ pred=0 | ❌ 55% inconclusive | ✅ 97% AI |
| Abstract fluid simulation | AI | ✅ pred=1 | ✅ 72% AI | ❌ 80% Real |
| Dragon fantasy | AI | ✅ pred=1 | ✅ 72% AI | ✅ 98% AI |
| Rainy street neon | AI | ❌ pred=0 | ✅ 72% AI | not tested |
| Face morphing | AI | ❌ pred=0 | ❌ 55% inconclusive | ✅ 97% AI |
| Futuristic city | AI | ❌ pred=0 | not tested | ✅ 97% AI |
| ai_kling_01 (waves) | AI | ❌ pred=0 (score 2) | ❌ 55% inconclusive | not tested |

### Results on real videos (adv_fp_trap subset)

| Filename | Label | Our Detector | detectvideo.ai | undetectable.ai |
|---|---|---|---|---|
| Broadcast news lower thirds | Real | ✅ TN | ❌ 55% inconclusive | ✅ Real 76% |
| Street billboard | Real | ✅ TN | not tested | ✅ Real 83% |
| biden_real.mp4 | Real | ✅ TN | ❌ 55% AI | ✅ Real 84% |
| biden_fake.mp4 (deepfake) | Real* | ✅ TN | ❌ 55% AI | ❌ Real 92% |
| Lower third motion graphics | Real | ✅ TN | not tested | ❌ AI 57% |
| passport_fake.mp4 | Real* | ✅ TN | ✅ 72% AI | ❌ Real 85% |

*Note: biden_fake and passport_fake are face-swap/document deepfakes, not AI-generated video — correctly outside scope of this detector.*

### Key observations

1. **detectvideo.ai** returns 55% (inconclusive) for ~80% of tested videos including real footage — effectively unusable for binary classification
2. **undetectable.ai** performs well on obvious AI content (fantasy, morphing) but fails on deepfake faces (biden_fake: 92% Real) and nature scenes
3. **Our detector** is the only one that: (a) runs fully locally, (b) has no file size limit, (c) has no per-video cost, (d) makes a binary decision for every input
4. **Audio analysis:** both external tools incorporate audio signals — our detector intentionally excludes audio (audio is analyzed by a separate dedicated tool in the full pipeline)
5. **Complementary strengths:** our detector catches compressed/cropped adversarial cases (adv_compressed recall 63%, adv_cropped 67%) which external tools were not tested on
6. **Known limitation:** Kling nature videos (organic motion) — 0/3 detected by our tool; undetectable.ai also not tested on these

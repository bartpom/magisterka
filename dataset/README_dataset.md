# Dataset Structure

## AI-Generated Videos (Fake)

All fake videos are catalogued in `manifest_fake_all.csv`.

| Folder | Count | Description |
|---|---|---|
| ai_baseline/ | 31 | Original AI-generated videos |
| adv_compressed/ | 31 | Same videos, re-encoded at lower bitrate |
| adv_cropped/ | 31 | Same videos, spatially cropped |

Generators represented: OpenAI Sora, Runway Gen-3/4/4.5, Luma Dream Machine, Pika, Kling, Google Veo, InVideo, and others.

## Real Videos (Non-AI)

All real videos are catalogued in `manifest_real_all.csv`.

| Folder | Count | Description |
|---|---|---|
| adv_fp_trap/ | 36 | Real videos with AI-like visual overlays (billboards, lower thirds, scoreboards) |

Categories: billboard, broadcast_lower_third, scoreboard_overlay, subway_signage.

## Notes
- `adv_compressed` and `adv_cropped` are adversarial variants of the same 31 videos from `ai_baseline`
- `adv_fp_trap` tests false positive rate - no AI generation involved
- Incomplete downloads (`.part`, `.m4a`) in `adv_fp_trap/` are excluded from manifests

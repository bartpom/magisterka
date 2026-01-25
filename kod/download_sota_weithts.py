"""
download_sota_weights.py

Downloads DeepFake / AI-detection models from Hugging Face into ./weights/*,
so ai_detector.py can load them offline (USE_LOCAL_SOTA_MODELS=True).

Why snapshot_download?
- hf_hub_download("model.safetensors") alone is NOT enough for Transformers,
  because you also need config.json + preprocessor config.

Usage:
  python download_sota_weights.py

Requirements:
  pip install huggingface_hub
"""

from __future__ import annotations

from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except Exception as e:
    raise SystemExit(
        "Missing dependency: huggingface_hub\n"
        "Install with: pip install huggingface_hub\n"
        f"Original error: {e}"
    )

WEIGHTS_DIR = Path(__file__).with_name("weights")

MODELS = {
    "vit_v2": "prithivMLmods/Deep-Fake-Detector-v2-Model",
}

def download(repo_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {out_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    print("Done.\n")

def main() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for subdir, repo_id in MODELS.items():
        download(repo_id, WEIGHTS_DIR / subdir)

    print("All downloads finished.")
    print("Next step:")
    print("  - in config.py set USE_LOCAL_SOTA_MODELS = True")
    print("  - run your GUI: python main.py")

if __name__ == "__main__":
    main()

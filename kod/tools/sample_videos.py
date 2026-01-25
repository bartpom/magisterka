from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _load_metadata(meta_path: Path) -> Dict[str, str]:
    """
    DFDC metadata.json format:
      {
        "a.mp4": {"label":"FAKE","original":"b.mp4"},
        "b.mp4": {"label":"REAL"},
        ...
      }
    Returns: {filename -> "REAL"/"FAKE"}
    """
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    labels: Dict[str, str] = {}
    if isinstance(meta, dict):
        for fname, info in meta.items():
            if not isinstance(info, dict):
                continue
            lab = info.get("label")
            if lab in ("REAL", "FAKE"):
                labels[str(fname)] = lab
    return labels


def collect_dfdc_videos(root: Path) -> Tuple[List[Path], List[Path]]:
    meta_files = list(root.rglob("metadata.json"))
    if not meta_files:
        raise RuntimeError(f"Nie znaleziono metadata.json w: {root}")

    real: List[Path] = []
    fake: List[Path] = []
    seen: set[Path] = set()

    for meta_path in meta_files:
        labels = _load_metadata(meta_path)
        base_dir = meta_path.parent

        for fname, lab in labels.items():
            video_path = (base_dir / fname)

            if not video_path.exists():
                # rzadkie, ale zostawiamy fallback
                hits = list(base_dir.rglob(fname))
                if not hits:
                    continue
                video_path = hits[0]

            if video_path.suffix.lower() not in VIDEO_EXTS:
                continue

            rp = video_path.resolve()
            if rp in seen:
                continue
            seen.add(rp)

            if lab == "REAL":
                real.append(rp)
            else:
                fake.append(rp)

    return real, fake


def _materialize(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return

    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except OSError:
            # na Windows symlink często wymaga uprawnień/dev mode
            shutil.copy2(src, dst)
            return

    raise ValueError(f"Nieznany mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="root DFDC (gdzieś pod spodem jest metadata.json)")
    ap.add_argument("--out", required=True, help="folder wyjściowy (stworzy real/ i fake/)")
    ap.add_argument("--n_real", type=int, default=100)
    ap.add_argument("--n_fake", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--clean", action="store_true", help="czyści out/real i out/fake przed zapisem")
    ap.add_argument("--append", action="store_true", help="dokłada kolejne pliki (nie zaczyna od 0000)")
    ap.add_argument("--mode", choices=("copy", "hardlink", "symlink"), default="copy")
    args = ap.parse_args()

    random.seed(args.seed)

    root = Path(args.input)
    out = Path(args.out)
    out_real = out / "real"
    out_fake = out / "fake"

    if args.clean:
        shutil.rmtree(out_real, ignore_errors=True)
        shutil.rmtree(out_fake, ignore_errors=True)

    out_real.mkdir(parents=True, exist_ok=True)
    out_fake.mkdir(parents=True, exist_ok=True)

    real_files, fake_files = collect_dfdc_videos(root)
    print(f"Znaleziono: REAL={len(real_files)}  FAKE={len(fake_files)}")

    if len(real_files) < args.n_real:
        print(f"UWAGA: za mało REAL ({len(real_files)} < {args.n_real}) – wezmę ile jest.")
    if len(fake_files) < args.n_fake:
        print(f"UWAGA: za mało FAKE ({len(fake_files)} < {args.n_fake}) – wezmę ile jest.")

    real_sel = random.sample(real_files, min(args.n_real, len(real_files))) if args.n_real > 0 else []
    fake_sel = random.sample(fake_files, min(args.n_fake, len(fake_files))) if args.n_fake > 0 else []

    start_real = len(list(out_real.glob("*"))) if args.append else 0
    start_fake = len(list(out_fake.glob("*"))) if args.append else 0

    for i, src in enumerate(real_sel, start=start_real):
        # nazwa zachowuje src.name dla łatwiejszego trace/debug
        dst = out_real / f"{i:04d}_real_{src.name}"
        _materialize(src, dst, args.mode)

    for i, src in enumerate(fake_sel, start=start_fake):
        dst = out_fake / f"{i:04d}_fake_{src.name}"
        _materialize(src, dst, args.mode)

    print(f"OK: zapisano REAL={len(real_sel)} → {out_real}")
    print(f"OK: zapisano FAKE={len(fake_sel)} → {out_fake}")
    if args.mode != "copy":
        print(f"(mode={args.mode})")


if __name__ == "__main__":
    main()

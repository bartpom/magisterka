import argparse
import os
import sys
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd: list, cwd: Optional[Path] = None) -> int:
    print("[CMD]", " ".join(cmd))
    try:
        r = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
        return int(r.returncode)
    except FileNotFoundError:
        return 127

def unzip_all(dir_path: Path) -> None:
    zips = sorted(dir_path.glob("*.zip"))
    if not zips:
        print("[INFO] Brak .zip do rozpakowania w:", dir_path)
        return
    for z in zips:
        out = dir_path / z.stem
        ensure_dir(out)
        print(f"[UNZIP] {z.name} -> {out}")
        try:
            with zipfile.ZipFile(z, "r") as f:
                f.extractall(out)
        except Exception as e:
            print("[WARN] Nie udało się rozpakować:", z, "err:", e)

def download_kaggle_competition(competition: str, out_dir: Path, unzip: bool) -> None:
    ensure_dir(out_dir)
    # wymaga: pip install kaggle + token kaggle.json
    code = run_cmd(["kaggle", "competitions", "download", "-c", competition, "-p", str(out_dir), "--force"])
    if code == 127:
        print("[ERROR] Nie znaleziono komendy 'kaggle'. Zainstaluj: pip install kaggle")
        return
    if code != 0:
        print("[ERROR] Kaggle download nie powiódł się. Sprawdź token kaggle.json i nazwę competition.")
        return
    if unzip:
        unzip_all(out_dir)

def download_kaggle_dataset(dataset: str, out_dir: Path, unzip: bool) -> None:
    ensure_dir(out_dir)
    code = run_cmd(["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_dir), "--force"])
    if code == 127:
        print("[ERROR] Nie znaleziono komendy 'kaggle'. Zainstaluj: pip install kaggle")
        return
    if code != 0:
        print("[ERROR] Kaggle dataset download nie powiódł się. Sprawdź token kaggle.json i nazwę dataset.")
        return
    if unzip:
        unzip_all(out_dir)

def download_http(url: str, out_dir: Path, filename: Optional[str], unzip: bool) -> None:
    ensure_dir(out_dir)
    try:
        import requests
    except Exception:
        print("[ERROR] Brak requests. Zainstaluj: pip install requests")
        return

    if not filename:
        filename = url.split("/")[-1] or "download.bin"
    out_path = out_dir / filename

    print("[HTTP] Pobieram:", url)
    print("[HTTP] Zapis do:", out_path)

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", "0") or "0")
            got = 0
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    got += len(chunk)
                    if total > 0:
                        pct = got * 100.0 / total
                        print(f"\r[HTTP] {pct:6.2f}% ({got}/{total} bytes)", end="")
        print()
    except Exception as e:
        print("[ERROR] HTTP download failed:", e)
        return

    if unzip and out_path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(out_path, "r") as zf:
                out = out_dir / out_path.stem
                ensure_dir(out)
                print(f"[UNZIP] {out_path.name} -> {out}")
                zf.extractall(out)
        except Exception as e:
            print("[WARN] unzip nieudany:", e)

def hf_snapshot(repo_id: str, out_dir: Path) -> None:
    ensure_dir(out_dir)
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        print("[ERROR] Brak huggingface_hub. Zainstaluj: pip install huggingface_hub")
        return

    print("[HF] snapshot_download:", repo_id)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as e:
        print("[ERROR] HF download failed:", e)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Katalog docelowy, np. data/raw/DFDC")
    ap.add_argument("--unzip", action="store_true", help="Rozpakuj .zip po pobraniu (jeśli dotyczy)")

    # Kaggle
    ap.add_argument("--kaggle-competition", default="", help="Nazwa competition, np. deepfake-detection-challenge")
    ap.add_argument("--kaggle-dataset", default="", help="Nazwa dataset w formacie owner/dataset")

    # HTTP
    ap.add_argument("--http-url", default="", help="Bezpośredni URL do pliku/archiwum")
    ap.add_argument("--http-filename", default="", help="Wymuś nazwę pliku dla HTTP")

    # HuggingFace
    ap.add_argument("--hf-repo", default="", help="HuggingFace repo_id (np. datasets/...)")

    args = ap.parse_args()
    out_dir = Path(args.out)

    if args.kaggle_competition:
        download_kaggle_competition(args.kaggle_competition, out_dir, args.unzip)
        return 0

    if args.kaggle_dataset:
        download_kaggle_dataset(args.kaggle_dataset, out_dir, args.unzip)
        return 0

    if args.http_url:
        download_http(args.http_url, out_dir, args.http_filename or None, args.unzip)
        return 0

    if args.hf_repo:
        hf_snapshot(args.hf_repo, out_dir)
        return 0

    print("[ERROR] Podaj jedną z opcji: --kaggle-competition / --kaggle-dataset / --http-url / --hf-repo")
    return 2

if __name__ == "__main__":
    raise SystemExit(main())

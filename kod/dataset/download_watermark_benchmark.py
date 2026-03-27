#!/usr/bin/env python3
"""
download_watermark_benchmark.py

Buduje bardziej zróżnicowany benchmark filmów z watermarkami (widocznymi i ukrytymi).
Skrypt pobiera materiały przez yt-dlp i zapisuje manifest z typem watermarku,
źródłem i przeznaczeniem do ewaluacji.

Cele:
- zwiększyć różnorodność watermarków (logo generatora, stacja TV, stock, social media),
- ułatwić porównanie aplikacji własnej z aplikacjami zewnętrznymi,
- zachować reprodukowalność przez manifest CSV.

Użycie (przykłady):
  python kod/dataset/download_watermark_benchmark.py
  python kod/dataset/download_watermark_benchmark.py --per-query 8 --max-duration 240
  python kod/dataset/download_watermark_benchmark.py --profile ai_visible --dry-run
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from yt_dlp import YoutubeDL


@dataclass(frozen=True)
class QueryDef:
    profile: str
    wm_type: str
    source_kind: str
    query: str


DEFAULT_OUT = Path("dataset/watermark_benchmark")
MANIFEST_NAME = "manifest.csv"

# profile: logiczne grupy do późniejszego raportowania / ablacji
QUERIES: list[QueryDef] = [
    QueryDef("ai_visible", "visible_generator_logo", "youtube", "Runway Gen-4 watermark demo"),
    QueryDef("ai_visible", "visible_generator_logo", "youtube", "Pika AI watermark sample"),
    QueryDef("ai_visible", "visible_generator_logo", "youtube", "Luma Dream Machine watermark"),
    QueryDef("ai_visible", "visible_generator_logo", "youtube", "Kling AI generated video watermark"),
    QueryDef("ai_visible", "visible_generator_logo", "youtube", "InVideo AI generated clip watermark"),
    QueryDef("broadcast_overlay", "visible_tv_overlay", "youtube", "sports broadcast score overlay full match"),
    QueryDef("broadcast_overlay", "visible_tv_overlay", "youtube", "news lower third live broadcast"),
    QueryDef("broadcast_overlay", "visible_tv_overlay", "youtube", "TV channel logo corner watermark"),
    QueryDef("stock_watermark", "visible_stock_watermark", "youtube", "shutterstock watermark sample video"),
    QueryDef("stock_watermark", "visible_stock_watermark", "youtube", "pond5 watermark video preview"),
    QueryDef("stock_watermark", "visible_stock_watermark", "youtube", "getty images watermark footage"),
    QueryDef("social_overlay", "visible_platform_overlay", "youtube", "TikTok repost with watermark"),
    QueryDef("social_overlay", "visible_platform_overlay", "youtube", "instagram reels watermark compilation"),
    QueryDef("social_overlay", "visible_platform_overlay", "youtube", "CapCut template watermark export"),
    QueryDef("c2pa_candidate", "invisible_c2pa_metadata_candidate", "youtube", "Content Credentials C2PA demo video"),
    QueryDef("c2pa_candidate", "invisible_c2pa_metadata_candidate", "youtube", "Adobe Firefly content credentials video"),
]


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    return re.sub(r"\s+", " ", name).strip()


def load_seen_video_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row.get("video_id", "") for row in csv.DictReader(f) if row.get("video_id")}


def append_manifest(path: Path, rows: list[dict]) -> None:
    fields = [
        "video_id",
        "title",
        "uploader",
        "duration_s",
        "webpage_url",
        "profile",
        "wm_type",
        "source_kind",
        "query",
        "local_path",
        "downloaded_at_utc",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def search_candidates(query: str, limit: int) -> list[dict]:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
    return info.get("entries", []) if info else []


def download_video(url: str, out_dir: Path) -> tuple[dict, str | None]:
    outtmpl = str(out_dir / "%(id)s_%(title).100B.%(ext)s")
    opts = {
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/b",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "windowsfilenames": True,
        "overwrites": False,
        "ffmpeg_location": shutil.which("ffmpeg") or "ffmpeg",
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)

    path_obj = Path(path)
    if path_obj.suffix.lower() != ".mp4":
        mp4_path = path_obj.with_suffix(".mp4")
        if mp4_path.exists():
            path_obj = mp4_path

    return info, str(path_obj) if path_obj.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Pobiera zróżnicowany benchmark watermarków.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-query", type=int, default=6)
    parser.add_argument("--max-duration", type=int, default=300)
    parser.add_argument("--min-duration", type=int, default=4)
    parser.add_argument(
        "--profile",
        type=str,
        default="all",
        help="Filtr profilu: ai_visible | broadcast_overlay | stock_watermark | social_overlay | c2pa_candidate | all",
    )
    parser.add_argument("--dry-run", action="store_true", help="Tylko wybór kandydatów, bez pobierania")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / MANIFEST_NAME
    seen_ids = load_seen_video_ids(manifest)

    filtered_queries = [q for q in QUERIES if args.profile == "all" or q.profile == args.profile]
    if not filtered_queries:
        print("[ERROR] Brak zapytań dla podanego --profile")
        return

    selected: list[tuple[QueryDef, dict]] = []
    for qd in filtered_queries:
        print(f"[SEARCH] {qd.profile} :: {qd.query}")
        try:
            entries = search_candidates(qd.query, args.per_query)
        except Exception as exc:
            print(f"  [ERROR] Search failed: {exc}")
            continue

        for e in entries:
            if not e:
                continue
            vid = e.get("id")
            duration = e.get("duration")
            url = e.get("url") or (f"https://www.youtube.com/watch?v={vid}" if vid else None)
            if not vid or not url or vid in seen_ids:
                continue
            if duration is None or not (args.min_duration <= duration <= args.max_duration):
                continue

            selected.append((qd, {
                "video_id": vid,
                "title": sanitize_filename(e.get("title") or "untitled"),
                "duration_s": duration,
                "webpage_url": url,
            }))
            seen_ids.add(vid)

    print(f"[INFO] Wybrano unikalnych kandydatów: {len(selected)}")
    if args.dry_run:
        return

    rows: list[dict] = []
    for qd, item in selected:
        print(f"[DOWNLOAD] ({qd.profile}) {item['title']}")
        try:
            info, local_path = download_video(item["webpage_url"], args.output)
        except Exception as exc:
            print(f"  [ERROR] Download failed: {exc}")
            continue

        rows.append({
            "video_id": item["video_id"],
            "title": sanitize_filename(info.get("title") or item["title"]),
            "uploader": info.get("uploader") or "",
            "duration_s": info.get("duration") or item["duration_s"],
            "webpage_url": info.get("webpage_url") or item["webpage_url"],
            "profile": qd.profile,
            "wm_type": qd.wm_type,
            "source_kind": qd.source_kind,
            "query": qd.query,
            "local_path": local_path or "",
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    if rows:
        append_manifest(manifest, rows)

    print(f"\n[MANIFEST] {manifest}")
    print(f"[RESULT] Newly downloaded: {len(rows)}")


if __name__ == "__main__":
    main()

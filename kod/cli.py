#!/usr/bin/env python
"""
cli.py  —  Interfejs wiersza poleceń do detektora watermarków AI.

Przykłady użycia:
    python cli.py film.mp4
    python cli.py folder/ --confidence 0.65 --sample-rate 15 --detailed
    python cli.py *.mp4 --output /mnt/wyniki --format json --quiet
    python cli.py film.mp4 --no-c2pa --log-level DEBUG
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Pomijamy zbędne logi OpenCV / PIL przed importem modułów projektu
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config  # noqa: E402

try:
    import ocr_detector
except ImportError as _e:
    print(f"[BŁĄD] Nie można zaimportować ocr_detector: {_e}", file=sys.stderr)
    ocr_detector = None  # type: ignore

try:
    import c2pa_detector
except ImportError:
    c2pa_detector = None  # type: ignore

# ---------------------------------------------------------------------------
# Stałe
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
}

_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Kolory ANSI (wyłączane przez --no-color lub brak TTY)
# ---------------------------------------------------------------------------

class _Clr:
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

def _c(code: str, text: str, use_color: bool) -> str:
    return f"{code}{text}{_Clr.RESET}" if use_color else text


# ---------------------------------------------------------------------------
# Budowanie listy plików wejściowych
# ---------------------------------------------------------------------------

def _collect_inputs(paths: list[str]) -> list[Path]:
    """Rozwiń globs, foldery i pojedyncze pliki → unikalna posortowana lista."""
    result: list[Path] = []
    seen: set[Path] = set()

    for raw in paths:
        # Glob (np. *.mp4 przekazane jako jeden string)
        expanded = glob.glob(raw, recursive=True)
        targets = [Path(p) for p in expanded] if expanded else [Path(raw)]

        for t in targets:
            if t.is_dir():
                for f in sorted(t.rglob("*")):
                    if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
                        if f not in seen:
                            seen.add(f)
                            result.append(f)
            elif t.is_file():
                if t.suffix.lower() not in SUPPORTED_EXTS:
                    print(f"[POMIŃ] Nieobsługiwany format: {t}", file=sys.stderr)
                    continue
                if t not in seen:
                    seen.add(t)
                    result.append(t)
            else:
                print(f"[OSTRZEŻENIE] Nie znaleziono: {raw}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Pasek postępu w CLI
# ---------------------------------------------------------------------------

def _progress_bar(curr: int, total: int, width: int = 30, quiet: bool = False) -> None:
    if quiet or total <= 0:
        return
    filled = int(width * curr / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(curr * 100 / total)
    print(f"\r  [{bar}] {pct:3d}% ({curr}/{total})", end="", flush=True)
    if curr >= total:
        print()


# ---------------------------------------------------------------------------
# Analiza pojedynczego pliku
# ---------------------------------------------------------------------------

def _analyze_one(
    path: Path,
    confidence: float,
    sample_rate: int,
    detailed: bool,
    output_dir: Optional[str],
    skip_c2pa: bool,
    quiet: bool,
    use_color: bool,
    log_level: str,
) -> dict:
    """Analizuje jeden plik; zwraca słownik wyników."""

    if not quiet:
        print(f"\n  Plik : {path.name}")
        print(f"  Rozm : {path.stat().st_size / 1024:.1f} KB")

    # --- C2PA ---
    c2pa_result: dict = {}
    if not skip_c2pa and c2pa_detector is not None:
        try:
            c2pa_result = c2pa_detector.detect_c2pa(str(path)) or {}
        except Exception as ce:
            c2pa_result = {"error": str(ce)}

    if ocr_detector is None:
        return {
            "file": str(path),
            "status": "ERROR",
            "error": "Moduł ocr_detector niedostępny",
            "c2pa": c2pa_result,
        }

    original_base = getattr(config, "REPORTS_BASE_DIR", "reports")
    if output_dir:
        setattr(config, "REPORTS_BASE_DIR", output_dir)

    def _cb(curr: int, tot: int) -> None:
        _progress_bar(curr, tot, quiet=quiet)

    t0 = time.monotonic()
    try:
        res = ocr_detector.scan_for_watermarks(
            str(path),
            check_stop=lambda: False,
            progress_callback=_cb,
            confidence=confidence,
            sample_rate=sample_rate,
            detailed_scan=detailed,
        )
        details: dict = res if isinstance(res, dict) else {}
        details["file"]     = str(path)
        details["c2pa"]     = c2pa_result
        details["elapsed_s"] = round(time.monotonic() - t0, 2)
    except Exception as exc:
        details = {
            "file": str(path),
            "status": "ERROR",
            "error": str(exc),
            "c2pa": c2pa_result,
            "elapsed_s": round(time.monotonic() - t0, 2),
        }
    finally:
        setattr(config, "REPORTS_BASE_DIR", original_base)

    return details


# ---------------------------------------------------------------------------
# Formatowanie wyniku
# ---------------------------------------------------------------------------

def _print_result(details: dict, use_color: bool, quiet: bool) -> None:
    if quiet:
        return

    status = details.get("status", "UNKNOWN")
    count  = details.get("watermark_count", 0) or 0
    elapsed = details.get("elapsed_s", 0)

    if status == "ERROR":
        print(_c(_Clr.RED, f"  Wynik : ERROR — {details.get('error', '')}", use_color))
        return

    if count > 0:
        types = ", ".join(details.get("watermark_types", []))
        label = _c(_Clr.RED, f"🔴 AI DETECTED ({count} ramek, typy: {types})", use_color)
    else:
        label = _c(_Clr.GREEN, "✅ AI CLEAR", use_color)

    c2pa = details.get("c2pa", {})
    if isinstance(c2pa, dict) and c2pa.get("found"):
        c2pa_label = _c(_Clr.GREEN, "✅ C2PA znalezione", use_color)
    elif isinstance(c2pa, dict) and "error" in c2pa:
        c2pa_label = _c(_Clr.YELLOW, "⚠️  C2PA błąd", use_color)
    else:
        c2pa_label = "❌ Brak C2PA"

    print(f"  Wynik : {label}")
    print(f"  C2PA  : {c2pa_label}")
    print(f"  Czas  : {elapsed:.2f}s")
    if details.get("csv_path"):
        print(f"  Raport: {details['csv_path']}")


# ---------------------------------------------------------------------------
# Zapis wyników
# ---------------------------------------------------------------------------

def _save_output(results: list[dict], fmt: str, out_path: str) -> None:
    """Zapisz zbiorczy wynik do pliku (json lub csv)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
    elif fmt == "csv":
        import csv
        keys = ["file", "status", "watermark_count", "watermark_types", "elapsed_s",
                "csv_path", "c2pa"]
        with open(p, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                row = dict(r)
                row["watermark_types"] = ", ".join(row.get("watermark_types") or [])
                row["c2pa"] = str(row.get("c2pa", ""))
                writer.writerow(row)
    print(f"\n[✓] Zbiorczy raport zapisano: {p}")


# ---------------------------------------------------------------------------
# Parser argumentów
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description=(
            "Detektor watermarków AI w filmach i obrazach.\n"
            "Analizuje pliki wideo/obrazy pod kątem obecności tekstowych \n"
            "watermarków generatorów AI (Runway, Sora, Kling itd.) \n"
            "oraz sprawdza metadane C2PA / Content Credentials."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Przykłady:\n"
            "  python cli.py film.mp4\n"
            "  python cli.py folder/ --confidence 0.65 --sample-rate 15 --detailed\n"
            "  python cli.py *.mp4 --output /mnt/wyniki --save-report wyniki.json\n"
            "  python cli.py film.mp4 --no-c2pa --log-level DEBUG\n"
            "  python cli.py folder/ --quiet --save-report raport.csv --format csv\n"
        ),
    )

    # ---- Wejście ----
    parser.add_argument(
        "inputs",
        metavar="PLIK_LUB_FOLDER",
        nargs="+",
        help=(
            "Pliki wideo/obrazy lub foldery do analizy. "
            "Obsługiwane formaty: mp4, mov, avi, mkv, webm, jpg, jpeg, png, bmp, webp, tif, tiff. "
            "Foldery są przeszukiwane rekurencyjnie. Obsługiwane globs (np. *.mp4)."
        ),
    )

    # ---- Parametry detekcji ----
    detect = parser.add_argument_group("Parametry detekcji")
    detect.add_argument(
        "-c", "--confidence",
        type=float,
        default=0.60,
        metavar="FLOAT",
        help=(
            "Minimalny próg pewności OCR (0.1–1.0). "
            "Niższy = więcej detekcji (w tym fałszywe alarmy). "
            "Wyższy = tylko pewne wyniki. "
            "[domyślnie: 0.60]"
        ),
    )
    detect.add_argument(
        "-s", "--sample-rate",
        type=int,
        default=30,
        metavar="N",
        help=(
            "Co ile klatek analizowana jest jedna klatka wideo. "
            "Wartość 30 = 1 klatka na sekundę przy 30fps. "
            "Mniejsza = dokładniej, ale wolniej. "
            "[domyślnie: 30]"
        ),
    )
    detect.add_argument(
        "-d", "--detailed",
        action="store_true",
        default=False,
        help=(
            "Tryb szczegółowy (dwufazowy): po szybkim skanie uruchamia "
            "zaawansowane filtry obrazu (CLAHE, Top-Hat, gamma, odwrócone kolory) "
            "na klatkach bez detekcji. Wolniejszy, ale wykrywa trudne watermarki "
            "(np. białe logo na jasnym tle). [domyślnie: wyłączony]"
        ),
    )
    detect.add_argument(
        "--no-c2pa",
        action="store_true",
        default=False,
        help="Pomiń analizę metadanych C2PA / Content Credentials. [domyślnie: włączona]",
    )

    # ---- Wyjście ----
    output = parser.add_argument_group("Wyjście i raporty")
    output.add_argument(
        "-o", "--output",
        metavar="FOLDER",
        default=None,
        help=(
            "Folder docelowy dla raportów CSV i ramek z detekcją. "
            "Nadpisuje wartość z config.py. [domyślnie: config.REPORTS_BASE_DIR]"
        ),
    )
    output.add_argument(
        "--save-report",
        metavar="PLIK",
        default=None,
        help=(
            "Ścieżka do zbiorczego pliku wynikowego "
            "(np. wyniki.json lub raport.csv). "
            "Format wykrywany automatycznie na podstawie rozszerzenia "
            "lub można ustawić --format."
        ),
    )
    output.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Format zbiorczego raportu przy użyciu --save-report. [domyślnie: json]",
    )
    output.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Wycisz wszystkie logi oprócz błędów krytycznych i wyniku końcowego.",
    )
    output.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="Wyłącz kolory ANSI (przydatne przy przekierowaniu do pliku).",
    )
    output.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Poziom szczegółowości logów [domyślnie: INFO].",
    )

    # ---- GUI ----
    gui_grp = parser.add_argument_group("Interfejs graficzny")
    gui_grp.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Uruchom GUI (PyQt6) zamiast trybu CLI.",
    )

    # ---- Informacyjne ----
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_VERSION}",
    )

    return parser


# ---------------------------------------------------------------------------
# Punkt wejścia
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()

    # Brak argumentów → pokaż help
    if not (argv or sys.argv[1:]):
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    # --- Tryb GUI ---
    if args.gui:
        try:
            from gui import run as run_gui
            run_gui()
        except ImportError as e:
            print(f"[BŁĄD] Nie można uruchomić GUI: {e}", file=sys.stderr)
            return 1
        return 0

    use_color = not args.no_color and sys.stdout.isatty()

    # --- Inicjalizacja OCR ---
    if ocr_detector is not None and not args.quiet:
        print("[OCR] Inicjalizacja silnika...", flush=True)
        engine, err = ocr_detector.warmup_reader()
        if err:
            print(f"[BŁĄD] OCR: {err}", file=sys.stderr)
            return 1
        if not args.quiet:
            print(f"[OCR] Silnik gotowy: {engine}")

    # --- Zbierz pliki ---
    files = _collect_inputs(args.inputs)
    if not files:
        print("[BŁĄD] Nie znaleziono obsługiwanych plików.", file=sys.stderr)
        return 1

    if not args.quiet:
        print(_c(_Clr.BOLD, f"\nAnaliza {len(files)} plik(ów):", use_color))
        print(f"  Confidence : {args.confidence}")
        print(f"  Sample rate: {args.sample_rate}")
        print(f"  Detailed   : {args.detailed}")
        print(f"  C2PA       : {'pominięta' if args.no_c2pa else 'włączona'}")
        print(f"  Output dir : {args.output or getattr(config, 'REPORTS_BASE_DIR', 'reports')}")

    # --- Analiza ---
    results: list[dict] = []
    errors = 0

    for i, path in enumerate(files, 1):
        if not args.quiet:
            print(_c(_Clr.BOLD, f"\n[{i}/{len(files)}] {path.name}", use_color))
            print(f"  Format : {path.suffix.upper().lstrip('.')}")

        det = _analyze_one(
            path=path,
            confidence=args.confidence,
            sample_rate=args.sample_rate,
            detailed=args.detailed,
            output_dir=args.output,
            skip_c2pa=args.no_c2pa,
            quiet=args.quiet,
            use_color=use_color,
            log_level=args.log_level,
        )
        _print_result(det, use_color=use_color, quiet=args.quiet)
        results.append(det)
        if det.get("status") == "ERROR":
            errors += 1

    # --- Podsumowanie ---
    detected = sum(1 for r in results if (r.get("watermark_count") or 0) > 0)
    clear    = len(results) - detected - errors

    if not args.quiet:
        print(_c(_Clr.BOLD, "\n═" * 50, use_color))
        print(_c(_Clr.BOLD, "PODSUMOWANIE", use_color))
        print(f"  Łącznie plików : {len(results)}")
        print(_c(_Clr.RED,   f"  AI DETECTED    : {detected}", use_color))
        print(_c(_Clr.GREEN, f"  AI CLEAR       : {clear}",    use_color))
        if errors:
            print(_c(_Clr.YELLOW, f"  Błędy          : {errors}", use_color))
        print(_c(_Clr.BOLD, "═" * 50, use_color))

    # --- Zbiorczy raport ---
    if args.save_report:
        _save_output(results, fmt=args.format, out_path=args.save_report)

    return 1 if errors == len(results) else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
analyze_results.py
Diagnostyka wynikow benchmarku — odpowiada na 3 kluczowe pytania:

  A. FP stage 1 vs stage 2
     Ile false positive pochodzi z IW stage 1 (iw_matched + similarity >= prog),
     a ile wpada przez stage 2 (score wazony + glosy).

  B. Separacja sygnalow: aibaseline vs adv_fp_trap
     Rozklady iw_best_similarity, of_count, of_max_area, of_global_motion,
     zv_count, zv_max_score, fft_score z podsumowaniem mean/median/max.

  C. FP per metoda IW (iw_method)
     Ktora metoda (dwtDct / dwtDctSvd / rivaGan) generuje FP na adv_fp_trap,
     a ktora skutecznie trafia w ai_baseline.

Uzycie:
  python analyze_results.py                        # szuka w results/latest/
  python analyze_results.py <sciezka/do/raw.csv>   # konkretny plik
"""

from __future__ import annotations
import csv
import sys
from pathlib import Path
from collections import defaultdict
from statistics import mean, median

# ───────────────────────────────────────────────────────────────────────
HEADER = "\n" + "=" * 72 + "\n"


def _h(title: str) -> None:
    print(f"{HEADER}  {title}{HEADER}")


def load_raw(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _int(x: str) -> int:
    """Bezpieczna konwersja string -> int, obsługuje '955.0' z CSV."""
    return int(float(x))


# ───────────────────────────────────────────────────────────────────────
# A: FP stage 1 vs stage 2
# ───────────────────────────────────────────────────────────────────────

def analyze_fp_stage(
    rows: list[dict],
    iw_strong_threshold: float = 0.85,
) -> None:
    """
    Klasyfikuje FP z adv_fp_trap na:
    - stage1_iw: iw_matched nie-pusty AND iw_best_similarity >= iw_strong_threshold
    - stage2:    wszystkie pozostale (score wazony / glosy)
    Rowniez pokazuje rozklad similarity wsrod FP.
    """
    _h("A. FALSE POSITIVE: STAGE 1 (IW silny) vs STAGE 2 (score wazony)")

    fp_rows = [
        r for r in rows
        if r["category"] == "adv_fp_trap" and _int(r["ground_truth"]) == 0
    ]
    if not fp_rows:
        print("  Brak wierszy adv_fp_trap w danych.")
        return

    print(f"  Liczba filmow w adv_fp_trap:    {len(fp_rows)}")

    # Symulujemy decyzje fuzji z evaluate.py
    sys.path.insert(0, str(Path(__file__).parent.parent / "dataset"))
    try:
        from evaluate import fuse as _fuse
    except ImportError:
        print("  [WARN] Nie mozna zaimportowac fuse() z evaluate.py")
        _fuse = None

    stage1_iw_fp, stage2_fp, no_det = 0, 0, 0
    stage1_details: list[str] = []

    for r in fp_rows:
        iw_sim     = float(r["iw_best_similarity"])
        iw_matched = r["iw_matched"]

        # Stage 1: IW silny
        if iw_matched and iw_sim >= iw_strong_threshold:
            stage1_iw_fp += 1
            stage1_details.append(
                f"    FP-stage1: {r['filename'][:45]:45s}  "
                f"iw_matched={iw_matched}  sim={iw_sim:.4f}"
            )
            continue

        # Sprawdz czy fuse() w ogole daje detekcje (stage 2)
        if _fuse:
            det, score, mode = _fuse(
                zv_count      = _int(r["zv_count"]),
                of_count      = _int(r["of_count"]),
                iw_similarity = iw_sim,
                iw_matched    = iw_matched,
                fft_score     = float(r["fft_score"]),
            )
            if det:
                stage2_fp += 1
            else:
                no_det += 1
        else:
            stage2_fp += 1  # fallback: zakladamy detekcje

    total_fp = stage1_iw_fp + stage2_fp
    print(f"  \u2550 Zidentyfikowane FP (z detekcja):  {total_fp} / {len(fp_rows)}")
    print(f"    - Stage 1 (iw_strong >= {iw_strong_threshold}):  {stage1_iw_fp}  "
          f"({100*stage1_iw_fp/max(total_fp,1):.0f}% FP)")
    print(f"    - Stage 2 (score + glosy):         {stage2_fp}  "
          f"({100*stage2_fp/max(total_fp,1):.0f}% FP)")
    print(f"    - Bez detekcji (TN):               {no_det}")

    if stage1_details:
        print("\n  Filmy FP przez stage 1:")
        for d in stage1_details:
            print(d)

    # Rozklad iw_best_similarity dla FP
    sims = [float(r["iw_best_similarity"]) for r in fp_rows]
    print(f"\n  Rozklad iw_best_similarity wsrod adv_fp_trap:")
    print(f"    mean   = {mean(sims):.4f}")
    print(f"    median = {median(sims):.4f}")
    print(f"    max    = {max(sims):.4f}")
    print(f"    min    = {min(sims):.4f}")
    buckets = [0]*10
    for s in sims:
        idx = min(9, int(s * 10))
        buckets[idx] += 1
    print("  Histogram similarity [0.0-1.0] w przedzialkach 0.1:")
    for i, cnt in enumerate(buckets):
        bar = "#" * cnt
        print(f"    [{i*0.1:.1f}-{(i+1)*0.1:.1f})  {bar} ({cnt})")

    print(f"\n  >>> WNIOSEK: ", end="")
    if stage1_iw_fp > stage2_fp:
        print("Stage 1 (IW silny) jest glownym zrodlem FP.")
        print("  Zalecenie: podniesc iw_strong_threshold albo zdegradowac IW do stage 2.")
    elif stage2_fp > 0:
        print("Stage 2 (score + glosy) jest glownym zrodlem FP.")
        print("  Zalecenie: podniesc of_threshold / score_threshold / min_weak_votes.")
    else:
        print("Brak FP w danych lub brak detekcji.")


# ───────────────────────────────────────────────────────────────────────
# B: Separacja sygnalow
# ───────────────────────────────────────────────────────────────────────

def analyze_signal_separation(rows: list[dict]) -> None:
    _h("B. SEPARACJA SYGNALOW: ai_baseline vs adv_fp_trap")

    ai_rows = [r for r in rows if r["category"] == "ai_baseline"]
    fp_rows = [r for r in rows if r["category"] == "adv_fp_trap"]

    if not ai_rows or not fp_rows:
        print("  Brak wystarczajacych danych dla obu kategorii.")
        return

    signals = [
        ("iw_best_similarity", float),
        ("of_count",           _int),
        ("of_max_area",        _int),
        ("of_global_motion",   float),
        ("zv_count",           _int),
        ("zv_max_score",       float),
        ("fft_score",          float),
    ]

    print(f"  {'Sygnal':22s}  {'AI mean':>9}  {'AI med':>8}  "
          f"{'FP mean':>9}  {'FP med':>8}  {'Delta':>8}  Ocena")
    print("  " + "-" * 82)

    for field, cast in signals:
        ai_vals = [cast(r[field]) for r in ai_rows  if r.get(field) not in ("", None)]
        fp_vals = [cast(r[field]) for r in fp_rows  if r.get(field) not in ("", None)]
        if not ai_vals or not fp_vals:
            print(f"  {field:22s}  brak danych")
            continue

        ai_mean = mean(ai_vals)
        ai_med  = median(ai_vals)
        fp_mean = mean(fp_vals)
        fp_med  = median(fp_vals)
        delta   = ai_mean - fp_mean
        # Ocena separacji: abs(delta) / (srednia obu mean)
        base = (abs(ai_mean) + abs(fp_mean)) / 2 or 1
        sep_ratio = abs(delta) / base
        if sep_ratio >= 0.5:
            rating = "DOBRA SEPARACJA"
        elif sep_ratio >= 0.2:
            rating = "slaba separacja"
        else:
            rating = "BRAK separacji"

        print(f"  {field:22s}  {ai_mean:9.4f}  {ai_med:8.4f}  "
              f"{fp_mean:9.4f}  {fp_med:8.4f}  {delta:+8.4f}  {rating}")

    print("\n  Legenda:")
    print("    DOBRA SEPARACJA  → sygnal roznia AI od FP, moze byc w fuzji z wyzszym progiem")
    print("    slaba separacja  → sygnal wnosi niewiele, ostroznosc")
    print("    BRAK separacji   → sygnal praktycznie nie roznia AI od FP, rozważ usuniecie z fuzji")


# ───────────────────────────────────────────────────────────────────────
# C: FP per metoda IW
# ───────────────────────────────────────────────────────────────────────

def analyze_iw_methods(rows: list[dict]) -> None:
    _h("C. INVISIBLE WATERMARK: skutecznosc per metoda (iw_method)")

    # Zbierz po metodach
    method_stats: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for r in rows:
        method = r.get("iw_method", "") or "(brak)"
        cat    = r["category"]
        sim    = float(r["iw_best_similarity"])
        found  = _int(r["iw_found"])
        method_stats[method][cat].append((found, sim))

    all_methods = sorted(method_stats.keys())
    cats_order  = ["ai_baseline", "adv_compressed", "adv_cropped", "adv_fp_trap"]

    for method in all_methods:
        print(f"\n  Metoda: {method}")
        print(f"  {'Kategoria':20s}  {'n':>4}  {'iw_found':>8}  "
              f"{'mean_sim':>9}  {'max_sim':>8}")
        print("  " + "-" * 58)
        for cat in cats_order:
            entries = method_stats[method].get(cat, [])
            if not entries:
                continue
            n         = len(entries)
            n_found   = sum(e[0] for e in entries)
            sims      = [e[1] for e in entries]
            mean_sim  = mean(sims)
            max_sim   = max(sims)
            marker    = "  <-- FP" if cat == "adv_fp_trap" and n_found > 0 else ""
            print(f"  {cat:20s}  {n:4d}  {n_found:8d}  "
                  f"{mean_sim:9.4f}  {max_sim:8.4f}{marker}")

    print("\n  Interpretacja:")
    print("    Jesli metoda ma iw_found > 0 na adv_fp_trap → generuje FP")
    print("    Porownaj: ktora metoda ma dobra detekcje AI a mala na FP")
    print("    Rozważ wylaczenie metod ktore szumia na FP bez zysku na AI")


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) > 1:
        raw_path = Path(sys.argv[1])
    else:
        # Domyslnie szukaj w results/latest/
        default = Path(__file__).parent.parent / "results" / "latest" / "raw_signals.csv"
        raw_path = default

    if not raw_path.exists():
        print(f"[BLAD] Nie znaleziono: {raw_path}")
        print("Uruchom najpierw: python kod/dataset/evaluate.py")
        sys.exit(1)

    print(f"[ANALIZA] Wczytuję: {raw_path}")
    rows = load_raw(raw_path)
    print(f"[ANALIZA] Wierszy: {len(rows)}")

    analyze_fp_stage(rows)
    analyze_signal_separation(rows)
    analyze_iw_methods(rows)

    print(f"{HEADER}  GOTOWE  {HEADER}")
    print("Kolejne kroki:")
    print("  1. Jesli FP robi stage 1 (IW) -> podniesc iw_strong_threshold w threshold_sweep.csv")
    print("  2. Jesli FP robi stage 2 (OF/ZV) -> podniesc of_threshold lub min_weak_votes")
    print("  3. Jesli metoda IW szumi -> odkomentuj wylaczenie jej w advanced_detectors.py")
    print("  4. Wybierz konfiguracje z threshold_sweep.csv: FPR_adv_fp_trap min + TPR_aibaseline >= 0.80")


if __name__ == "__main__":
    main()

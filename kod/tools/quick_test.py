from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass
class ItemResult:
    gt: str
    path: Path
    status: str
    final_score: float
    verdict: str
    ai_face: Optional[float]
    ai_scene: Optional[float]
    ai_video: Optional[float]
    fake_ratio: Optional[float]
    face_samples: Optional[int]
    force_fake: bool
    force_reason: str
    err: Optional[str]


# =========================
# HELPERS
# =========================

def _list_videos(dir_path: Path) -> List[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    return sorted(
        p for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def _sample(items: List[Path], n: int, rng: random.Random) -> List[Path]:
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    return rng.sample(items, n)


def _ts_file() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _pred_from_score(score: float, real_max: float, fake_min: float) -> str:
    try:
        s = float(score)
    except Exception:
        return "error"
    if s <= real_max:
        return "real"
    if s >= fake_min:
        return "fake"
    return "grey"


def _safe_mean(xs: List[float]) -> Optional[float]:
    return statistics.mean(xs) if xs else None


def _safe_median(xs: List[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


def _fmt_num(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{x:.4f}"


def _fmt_pct(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{x:.2f}%"


# =========================
# METRICS
# =========================

def compute_tri_metrics(
    rows: List[Dict[str, Any]],
    real_max: float,
    fake_min: float,
) -> Dict[str, Any]:
    cm = {
        "real": {"real": 0, "grey": 0, "fake": 0},
        "fake": {"real": 0, "grey": 0, "fake": 0},
    }

    for r in rows:
        gt = str(r["gt"]).strip().lower()
        score = float(r["final_score"])
        pred = _pred_from_score(score, real_max, fake_min)
        if gt in cm and pred in cm[gt]:
            cm[gt][pred] += 1

    total = sum(sum(v.values()) for v in cm.values())
    if total == 0:
        return {"cm": cm}

    strict_acc = (cm["real"]["real"] + cm["fake"]["fake"]) / total

    non_grey = (
        cm["real"]["real"] + cm["real"]["fake"] +
        cm["fake"]["real"] + cm["fake"]["fake"]
    )
    coverage = non_grey / total if total else 0.0
    acc_non_grey = ((cm["real"]["real"] + cm["fake"]["fake"]) / non_grey) if non_grey > 0 else None

    tp = cm["fake"]["fake"]
    fp = cm["real"]["fake"]
    fn = cm["fake"]["real"] + cm["fake"]["grey"]  # grey = miss dla recall fake

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    return {
        "cm": cm,
        "strict_accuracy": strict_acc,
        "coverage": coverage,
        "accuracy_non_grey": acc_non_grey,
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
    }


def tune_fake_min_for_recall(
    rows: List[Dict[str, Any]],
    real_max: float,
    fake_min_min: int = 40,
    fake_min_max: int = 80,
    min_precision: float = 0.80,
) -> Optional[Dict[str, Any]]:
    """
    Prosty tuning: szukamy FAKE_MIN, który maksymalizuje recall_fake przy constraint precision_fake >= min_precision.
    """
    best = None
    for fm in range(fake_min_min, fake_min_max + 1):
        m = compute_tri_metrics(rows, real_max=real_max, fake_min=float(fm))
        pf = m.get("precision_fake", None)
        rf = m.get("recall_fake", None)
        if pf is None or rf is None:
            continue
        if pf < min_precision:
            continue

        if best is None:
            best = dict(m)
            best["fake_min"] = fm
            continue

        # maximize recall, tie-break: precision, then coverage
        if rf > best["recall_fake"]:
            best = dict(m)
            best["fake_min"] = fm
        elif rf == best["recall_fake"] and pf > best["precision_fake"]:
            best = dict(m)
            best["fake_min"] = fm
        elif rf == best["recall_fake"] and pf == best["precision_fake"]:
            if (m.get("coverage") or 0.0) > (best.get("coverage") or 0.0):
                best = dict(m)
                best["fake_min"] = fm

    return best


# =========================
# MAIN
# =========================

def main() -> int:
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import ai_detector  # noqa
    import config  # noqa

    parser = argparse.ArgumentParser("Quick deepfake test (N real + N fake)")
    parser.add_argument("--real_dir", default=str(project_root / "data" / "sample" / "real"))
    parser.add_argument("--fake_dir", default=str(project_root / "data" / "sample" / "fake"))
    parser.add_argument("--n_real", type=int, default=10)
    parser.add_argument("--n_fake", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_dir", default=str(project_root))  # raport zbiorczy w root projektu
    parser.add_argument("--no_face_ai", action="store_true")
    parser.add_argument("--no_forensic", action="store_true")
    parser.add_argument("--watermark", action="store_true")
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    do_face_ai = not bool(args.no_face_ai)
    do_forensic = not bool(args.no_forensic)
    do_watermark = bool(args.watermark)

    rng = random.Random(int(args.seed))

    real_all = _list_videos(real_dir)
    fake_all = _list_videos(fake_dir)

    real_pick = _sample(real_all, int(args.n_real), rng)
    fake_pick = _sample(fake_all, int(args.n_fake), rng)

    run_dir = ai_detector.begin_run()   # raporty per plik
    ai_detector.init_models()

    results: List[ItemResult] = []
    t0 = time.time()

    def run_one(gt: str, p: Path, idx: int, total: int) -> None:
        print(f"[{idx}/{total}] {gt.upper()} -> {p.name}")
        try:
            status, score, fake_ratio, details = ai_detector.scan_for_deepfake(
                str(p),
                do_face_ai=do_face_ai,
                do_forensic=do_forensic,
                do_watermark=do_watermark,
                run_dir=run_dir,
            )

            # HIGH-RECALL postprocess (pod recall_fake)
            status, score, fake_ratio, details = ai_detector.postprocess_score_for_high_recall(
                status, score, fake_ratio, details
            )

            # Utrzymaj spójność raportu per-plik (nadpisz report.txt po dopchnięciu)
            try:
                if isinstance(details, dict) and score is not None:
                    details["final_score"] = float(score)
                    ai_detector.save_report(str(p), details, run_dir=run_dir)
            except Exception:
                pass

            score_f = float(score) if score is not None else 0.0
            details = details if isinstance(details, dict) else {}

            results.append(
                ItemResult(
                    gt=gt,
                    path=p,
                    status=str(status),
                    final_score=score_f,
                    verdict=str(details.get("verdict", "")),
                    ai_face=details.get("ai_face_score"),
                    ai_scene=details.get("ai_scene_score"),
                    ai_video=details.get("ai_video_score"),
                    fake_ratio=float(fake_ratio) if fake_ratio is not None else None,
                    face_samples=details.get("face_samples"),
                    force_fake=bool(details.get("force_fake", False)),
                    force_reason=str(details.get("force_fake_reason", "")),
                    err=None,
                )
            )
        except Exception as e:
            results.append(
                ItemResult(
                    gt=gt,
                    path=p,
                    status="ERROR",
                    final_score=0.0,
                    verdict="",
                    ai_face=None,
                    ai_scene=None,
                    ai_video=None,
                    fake_ratio=None,
                    face_samples=None,
                    force_fake=False,
                    force_reason="",
                    err=str(e),
                )
            )

    total = len(real_pick) + len(fake_pick)
    idx = 1
    for p in real_pick:
        run_one("real", p, idx, total)
        idx += 1
    for p in fake_pick:
        run_one("fake", p, idx, total)
        idx += 1

    dt = time.time() - t0

    ok = [r for r in results if r.status == "OK"]
    err = [r for r in results if r.status != "OK"]

    real_max = float(getattr(config, "REAL_MAX", 30.0))
    fake_min = float(getattr(config, "FAKE_MIN", 70.0))

    rows = [{"gt": r.gt, "final_score": r.final_score} for r in ok]
    metrics = compute_tri_metrics(rows, real_max, fake_min)
    cm = metrics.get("cm", {"real": {"real": 0, "grey": 0, "fake": 0}, "fake": {"real": 0, "grey": 0, "fake": 0}})

    # staty score
    real_scores = [r.final_score for r in ok if r.gt == "real"]
    fake_scores = [r.final_score for r in ok if r.gt == "fake"]

    real_sorted_hi = sorted([r for r in ok if r.gt == "real"], key=lambda x: x.final_score, reverse=True)
    fake_sorted_lo = sorted([r for r in ok if r.gt == "fake"], key=lambda x: x.final_score)

    # tuning progu FAKE_MIN pod recall (opcjonalna sugestia)
    tuned = tune_fake_min_for_recall(rows, real_max=real_max, fake_min_min=40, fake_min_max=80, min_precision=0.80)

    report_path = out_dir / f"raport_zbiorczy - {_ts_file()}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("RAPORT ZBIORCZY – Deepfake Detector (quick test)\n")
        f.write(f"Data/godzina: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Czas trwania: {dt:.2f} s\n\n")

        f.write("WEJŚCIE\n")
        f.write(f"  real_dir: {real_dir.resolve()}\n")
        f.write(f"  fake_dir: {fake_dir.resolve()}\n")
        f.write(f"  n_real (requested/used): {args.n_real} / {len(real_pick)} (available={len(real_all)})\n")
        f.write(f"  n_fake (requested/used): {args.n_fake} / {len(fake_pick)} (available={len(fake_all)})\n")
        f.write(f"  seed: {args.seed}\n")
        f.write(f"  run_dir (raporty per plik): {Path(run_dir).resolve()}\n\n")

        f.write("KONFIG\n")
        f.write(f"  DECISION_POLICY={getattr(config, 'DECISION_POLICY', 'balanced')}\n")
        f.write(f"  do_face_ai={do_face_ai}\n")
        f.write(f"  do_forensic={do_forensic}\n")
        f.write(f"  do_watermark={do_watermark}\n")
        f.write(f"  PROGI: REAL_MAX={real_max:.2f}  FAKE_MIN={fake_min:.2f}\n")
        f.write(f"  FUSE_MODE={getattr(config, 'FUSE_MODE', 'weighted_mean')}\n")
        f.write(f"  FUSE_WEIGHTS={getattr(config, 'FUSE_WEIGHTS', {})}\n\n")

        f.write("PODSUMOWANIE\n")
        f.write(f"  Total picked: {total}\n")
        f.write(f"  OK: {len(ok)}\n")
        f.write(f"  ERROR: {len(err)}\n\n")

        f.write("MACIERZ (GT x PRED) – tylko OK\n")
        f.write(f"  GT=REAL: pred_real={cm['real']['real']}  pred_grey={cm['real']['grey']}  pred_fake={cm['real']['fake']}\n")
        f.write(f"  GT=FAKE: pred_real={cm['fake']['real']}  pred_grey={cm['fake']['grey']}  pred_fake={cm['fake']['fake']}\n\n")

        f.write("METRYKI (tylko OK)\n")
        sa = metrics.get("strict_accuracy", None)
        cov = metrics.get("coverage", None)
        ang = metrics.get("accuracy_non_grey", None)
        pf = metrics.get("precision_fake", None)
        rf = metrics.get("recall_fake", None)
        f1 = metrics.get("f1_fake", None)

        f.write(f"  strict_accuracy (grey=error): {('N/A' if sa is None else f'{sa*100:.2f}%')}\n")
        f.write(f"  coverage (non-grey rate): {('N/A' if cov is None else f'{cov*100:.2f}%')}\n")
        f.write(f"  accuracy_non_grey: {('N/A' if ang is None else f'{ang*100:.2f}%')}\n")
        f.write(f"  precision_fake: {('N/A' if pf is None else f'{pf*100:.2f}%')}\n")
        f.write(f"  recall_fake: {('N/A' if rf is None else f'{rf*100:.2f}%')}\n")
        f.write(f"  f1_fake: {('N/A' if f1 is None else f'{f1*100:.2f}%')}\n\n")

        f.write("STATYSTYKI SCORE (final_score)\n")
        f.write(
            "  REAL: "
            f"mean={_fmt_num(_safe_mean(real_scores))}  "
            f"median={_fmt_num(_safe_median(real_scores))}  "
            f"min={_fmt_num(min(real_scores) if real_scores else None)}  "
            f"max={_fmt_num(max(real_scores) if real_scores else None)}\n"
        )
        f.write(
            "  FAKE: "
            f"mean={_fmt_num(_safe_mean(fake_scores))}  "
            f"median={_fmt_num(_safe_median(fake_scores))}  "
            f"min={_fmt_num(min(fake_scores) if fake_scores else None)}  "
            f"max={_fmt_num(max(fake_scores) if fake_scores else None)}\n\n"
        )

        f.write("NAJGORSZE PRZYPADKI\n")
        f.write("  TOP REAL (najwyższy score):\n")
        for r in real_sorted_hi[:5]:
            pred = _pred_from_score(r.final_score, real_max, fake_min)
            f.write(f"    {r.final_score:6.2f}%  pred={pred:4s}  {r.path.name}\n")
        f.write("\n")

        f.write("  BOTTOM FAKE (najniższy score):\n")
        for r in fake_sorted_lo[:5]:
            pred = _pred_from_score(r.final_score, real_max, fake_min)
            f.write(f"    {r.final_score:6.2f}%  pred={pred:4s}  {r.path.name}\n")
        f.write("\n")

        if tuned:
            f.write("SUGESTIA: FAKE_MIN pod recall (min precision >= 80%)\n")
            f.write(
                f"  best_fake_min={tuned['fake_min']}  "
                f"precision_fake={tuned['precision_fake']*100:.2f}%  "
                f"recall_fake={tuned['recall_fake']*100:.2f}%  "
                f"coverage={tuned['coverage']*100:.2f}%\n\n"
            )

        f.write("WYNIKI PER PLIK (tylko OK)\n")
        for r in ok:
            pred = _pred_from_score(r.final_score, real_max, fake_min)
            forced = "FORCED" if r.force_fake else ""
            f.write(
                f"- GT={r.gt:4s}  PRED={pred:4s}  SCORE={r.final_score:6.2f}%  "
                f"FACE={_fmt_pct(r.ai_face)}  SCENE={_fmt_pct(r.ai_scene)}  VIDEO={_fmt_pct(r.ai_video)}  "
                f"fake_ratio={_fmt_pct(r.fake_ratio)}  face_samples={r.face_samples if r.face_samples is not None else 'N/A'}  "
                f"{forced}  verdict='{r.verdict}'  file={r.path}\n"
            )
            if r.force_fake and r.force_reason:
                f.write(f"    force_reason: {r.force_reason}\n")

        if err:
            f.write("\nBŁĘDY\n")
            for r in err:
                f.write(f"- GT={r.gt} file={r.path} status={r.status} err={r.err}\n")

    print(f"[OK] Raport zapisany: {report_path.resolve()}")
    print(f"[OK] Raporty per plik: {Path(run_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

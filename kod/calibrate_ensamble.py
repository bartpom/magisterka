from __future__ import annotations

"""
calibrate_ensemble.py

Skrypt do kalibracji i "domknięcia" problemu, który widać w Twoich wynikach:
większość filmów ląduje w GREY ZONE, bo surowe score'y z różnych detektorów
nie są skalibrowane do Twojej dystrybucji danych.

Ten skrypt:
1) Przechodzi po folderze z *.mp4 / *.mov / *.mkv
2) Nadaje etykiety na podstawie nazwy pliku: zawiera "_real" -> 0, "_fake" -> 1
3) Wylicza cechy: VideoMAE p_fake, D3 std_abs + D3 p_fake
4) Trenuje prostą regresję logistyczną i zapisuje kalibrator do pliku joblib.

Użycie (PowerShell):
  python calibrate_ensemble.py --data "C:\sciezka\do\wideo" --out calibration.joblib
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from temporal_d3 import D3Config, D3TemporalDetector
from videomae_detector import VideoMAEConfig, VideoMAEDeepfakeDetector


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def infer_label_from_name(name: str) -> int:
    n = name.lower()
    if "_fake" in n or "fake" in n:
        return 1
    if "_real" in n or "real" in n:
        return 0
    raise ValueError("Nie można ustalić etykiety z nazwy (oczekuję _real lub _fake).")


def build_feature_vector(videomae_p: float | None, d3_std: float | None, d3_p: float | None) -> List[float]:
    return [
        float(videomae_p) if videomae_p is not None else float("nan"),
        float(d3_std) if d3_std is not None else float("nan"),
        float(d3_p) if d3_p is not None else float("nan"),
    ]


def nan_impute(X: np.ndarray) -> np.ndarray:
    X2 = X.copy()
    for j in range(X2.shape[1]):
        col = X2[:, j]
        m = np.nanmedian(col)
        if np.isnan(m):
            m = 0.0
        col[np.isnan(col)] = m
        X2[:, j] = col
    return X2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder z wideo (nazwy *_real* / *_fake*)")
    ap.add_argument("--out", default="calibration.joblib", help="Plik wyjściowy joblib")
    ap.add_argument("--videomae", default="Ammar2k/videomae-base-finetuned-deepfake-subset", help="HF model_id")
    ap.add_argument("--device", default="cpu", help="cpu/cuda")
    args = ap.parse_args()

    data_dir = Path(args.data)
    files = [p for p in data_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    if not files:
        raise SystemExit(f"Brak plików wideo w: {data_dir}")

    d3 = D3TemporalDetector(D3Config(device=args.device))
    vm = VideoMAEDeepfakeDetector(VideoMAEConfig(model_id=args.videomae, device=args.device))

    X_list: List[List[float]] = []
    y_list: List[int] = []
    names: List[str] = []

    print(f"[INFO] Liczę cechy dla {len(files)} plików...")

    for p in files:
        try:
            y = infer_label_from_name(p.name)
        except Exception:
            continue

        videomae_p, _ = vm.analyze(str(p))
        d3_p, d3_det = d3.analyze(str(p))
        d3_std = d3_det.get("d3_std_abs") if isinstance(d3_det, dict) else None

        fv = build_feature_vector(videomae_p, d3_std, d3_p)
        X_list.append(fv)
        y_list.append(y)
        names.append(p.name)

        print(f"  - {p.name}: y={y} vm_p={videomae_p} d3_std={d3_std} d3_p={d3_p}")

    if len(set(y_list)) < 2:
        raise SystemExit("Potrzebujesz co najmniej po 1 pliku real i fake do kalibracji.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    X = nan_impute(X)

    feature_names = ["videomae_p_fake", "d3_std_abs", "d3_p_fake"]

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    skf = StratifiedKFold(n_splits=min(5, len(y)), shuffle=True, random_state=42)
    aucs: List[float] = []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], p)
        aucs.append(float(auc))
        print(f"[CV] Fold {fold}: AUC={auc:.3f}")

    print(f"[CV] Mean AUC={np.mean(aucs):.3f} (+/- {np.std(aucs):.3f})")

    clf.fit(X, y)
    joblib.dump(
        {
            "model": clf,
            "feature_names": feature_names,
            "label_convention": {"0": "real", "1": "fake"},
            "videomae_model": args.videomae,
            "d3_config": asdict(d3.cfg),
        },
        args.out,
    )

    meta_path = str(Path(args.out).with_suffix(".json"))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_names": feature_names,
                "n_samples": int(len(y)),
                "cv_auc_mean": float(np.mean(aucs)),
                "cv_auc_std": float(np.std(aucs)),
                "label_convention": {"0": "real", "1": "fake"},
                "videomae_model": args.videomae,
                "d3_config": asdict(d3.cfg),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] Zapisano: {args.out}")
    print(f"[OK] Zapisano: {meta_path}")

    pred = clf.predict(X)
    print(classification_report(y, pred, target_names=["real", "fake"]))


if __name__ == "__main__":
    main()
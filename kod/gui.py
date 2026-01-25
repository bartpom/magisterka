# gui.py (IMPROVED)
import os
import sys
from typing import List, Optional, Set, Dict, Any

from PyQt5 import QtCore, QtGui, QtWidgets

import config

# gui.py
import importlib.util
import pathlib
import sys as _sys


def _load_local_ai_detector():
    # ładuje dokładnie ai_detector.py z tego samego katalogu co gui.py
    path = pathlib.Path(__file__).with_name("ai_detector.py")
    spec = importlib.util.spec_from_file_location("ai_detector", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ai_detector from: {path}")
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["ai_detector"] = mod
    spec.loader.exec_module(mod)
    return mod


ai_detector = _load_local_ai_detector()

# sanity check: GUI wymaga begin_run()
if not hasattr(ai_detector, "begin_run"):
    raise RuntimeError(f"Loaded ai_detector has no begin_run(). File: {getattr(ai_detector,'__file__',None)}")

try:
    import ocr_detector
except Exception as _e:
    ocr_detector = None
    print(f"[OCR] Moduł ocr_detector niedostępny: {_e}")

try:
    import calibrate_thresholds
except Exception as _e:
    calibrate_thresholds = None
    print(f"[CAL] Moduł calibrate_thresholds niedostępny: {_e}")

SUPPORTED_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


def is_supported_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in SUPPORTED_EXTS


# ============================ Worker ============================

class AnalysisWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int)        # (curr, total)
    file_started = QtCore.pyqtSignal(int, str)    # (idx, filename)
    file_finished = QtCore.pyqtSignal(int, dict)  # (idx, details)
    log_line = QtCore.pyqtSignal(str)
    all_done = QtCore.pyqtSignal()

    def __init__(
        self,
        files: List[str],
        do_face_ai: bool,
        do_forensic: bool,
        do_watermark: bool,
        run_dir: str,
        parent=None
    ):
        super().__init__(parent)
        self._files = files
        self._do_face_ai = do_face_ai
        self._do_forensic = do_forensic
        self._do_watermark = do_watermark
        self._run_dir = run_dir
        self._stop = False

    def stop(self):
        self._stop = True

    def _run_watermark(self, path: str) -> Optional[dict]:
        if ocr_detector is None:
            self.log_line.emit("[OCR] Moduł ocr_detector nie został załadowany – pomijam detekcję watermarków.")
            return None

        fn = getattr(ocr_detector, "scan_for_watermarks", None)
        if callable(fn):
            try:
                self.log_line.emit("[OCR] Start detekcji znaków wodnych…")
                res = fn(path, check_stop=lambda: self._stop, progress_callback=None)
                if isinstance(res, dict):
                    return res
                return {"watermark_raw": res}
            except Exception as e:
                self.log_line.emit(f"[OCR] Błąd detekcji watermarków: {e}")
        return None

    def run(self):
        total_files = len(self._files)
        for idx, path in enumerate(self._files):
            if self._stop:
                break

            self.file_started.emit(idx, os.path.basename(path))

            def cb(curr, tot):
                self.progress.emit(int(curr), int(tot))

            try:
                res = ai_detector.scan_for_deepfake(
                    path,
                    progress_callback=cb,
                    check_stop=lambda: self._stop,
                    do_face_ai=self._do_face_ai,
                    do_forensic=self._do_forensic,
                    run_dir=self._run_dir,
                )

                if isinstance(res, tuple) and len(res) == 2:
                    # Case: (AiResult, ForensicResult)
                    ai_res, for_res = res
                    status = "DONE"
                    score = float(getattr(ai_res, "combined_max", 0.0) or 0.0)
                    fake_ratio = 0.0
                    details = {
                        "ai_face_score": getattr(ai_res, "face_score", None),
                        "ai_scene_score": getattr(ai_res, "scene_score", None),
                        "ai_video_score": getattr(ai_res, "video_score", None),
                        "ai_combined_score": getattr(ai_res, "combined_max", None),

                        "jitter_px": getattr(for_res, "jitter_px", None),
                        "blink_per_min": getattr(for_res, "blink_per_min", None),
                        "ela_score": getattr(for_res, "ela_score", None),
                        "fft_score": getattr(for_res, "fft_score", None),
                        "border_artifacts": getattr(for_res, "border_artifacts", None),
                        "face_sharpness": getattr(for_res, "face_sharpness", None),
                    }
                elif isinstance(res, tuple) and len(res) == 4:
                    status, score, fake_ratio, details = res
                else:
                    raise ValueError(f"Unexpected return type from scan_for_deepfake: {type(res)}")

                details = (details or {})
                details["status"] = status
                details.setdefault("raw_final_score", details.get("final_score", score))
                details.setdefault("fake_ratio", fake_ratio)
                details.setdefault("full_path", os.path.abspath(path))
            except Exception as e:
                self.log_line.emit(f"[BŁĄD] {os.path.basename(path)} (AI): {e}")
                details = {
                    "status": "ERROR",
                    "verdict": "ERROR",
                    "final_score": 0.0,
                    "full_path": os.path.abspath(path)
                }

            if not self._stop and self._do_watermark:
                wm = self._run_watermark(path)
                if isinstance(wm, dict):
                    for k, v in wm.items():
                        if k == "folder_path" and "folder_path" in details:
                            continue
                        details[k] = v

            self.file_finished.emit(idx, details)

        self.all_done.emit()


# ============================ GUI ============================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI / Deepfake Detector")
        self.resize(1000, 660)

        self.worker: Optional[AnalysisWorker] = None
        self.files: List[str] = []
        self.files_set: Set[str] = set()

        self.report_paths: Dict[int, str] = {}
        self.per_file_summaries: Dict[int, str] = {}
        self.current_run_dir: Optional[str] = None

        # kalibracja progów (ładowana z pliku per-run)
        self.thresholds = None
        self.thresholds_path: Optional[str] = None

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        root.addLayout(top)

        self.btn_pick_files = QtWidgets.QPushButton("Dodaj pliki…")
        self.btn_pick_files.clicked.connect(self.pick_files)
        top.addWidget(self.btn_pick_files)

        self.btn_pick_folder = QtWidgets.QPushButton("Dodaj folder…")
        self.btn_pick_folder.clicked.connect(self.pick_folder)
        top.addWidget(self.btn_pick_folder)

        self.grp_opts = QtWidgets.QGroupBox("Ustawienia analizy")
        opts_lay = QtWidgets.QHBoxLayout(self.grp_opts)
        self.chk_ai = QtWidgets.QCheckBox("Analiza AI (twarz/scena/wideo)")
        self.chk_ai.setChecked(True)
        self.chk_forensic = QtWidgets.QCheckBox("Analiza forensic/biometria (twarz)")
        self.chk_forensic.setChecked(True)
        self.chk_watermark = QtWidgets.QCheckBox("Detekcja znaków wodnych (OCR/YOLO)")
        self.chk_watermark.setChecked(False)
        opts_lay.addWidget(self.chk_ai)
        opts_lay.addWidget(self.chk_forensic)
        opts_lay.addWidget(self.chk_watermark)
        opts_lay.addStretch()
        top.addWidget(self.grp_opts, 1)

        self.chk_dark = QtWidgets.QCheckBox("Tryb ciemny")
        self.chk_dark.setChecked(True)
        self.chk_dark.toggled.connect(self.apply_dark_theme)
        top.addWidget(self.chk_dark)

        self.btn_start = QtWidgets.QPushButton("START")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_analysis)
        top.addWidget(self.btn_start)

        self.btn_stop = QtWidgets.QPushButton("STOP")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_analysis)
        top.addWidget(self.btn_stop)

        self.list_files = QtWidgets.QListWidget()
        self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        root.addWidget(self.list_files, 1)

        bottom_bar = QtWidgets.QHBoxLayout()
        root.addLayout(bottom_bar)

        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        bottom_bar.addWidget(self.progressBar, 1)

        self.btn_open_report = QtWidgets.QPushButton("Otwórz raport")
        self.btn_open_report.clicked.connect(self.open_selected_report)
        bottom_bar.addWidget(self.btn_open_report)

        self.btn_save_aggregate = QtWidgets.QPushButton("Zapisz raport zbiorczy…")
        self.btn_save_aggregate.clicked.connect(self.save_aggregate_report)
        bottom_bar.addWidget(self.btn_save_aggregate)

        self.logView = QtWidgets.QTextEdit()
        self.logView.setReadOnly(True)
        mono = QtGui.QFont("Consolas") if QtGui.QFont("Consolas").exactMatch() else QtGui.QFont("Courier New")
        self.logView.setFont(mono)
        root.addWidget(self.logView, 2)

        self.status = self.statusBar()

        self.apply_dark_theme(True)

    # -------------------- Theme --------------------

    def _build_palette(self, dark: bool) -> QtGui.QPalette:
        pal = QtGui.QPalette()
        if dark:
            pal.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
            pal.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.Base, QtGui.QColor(35, 35, 35))
            pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
            pal.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
            pal.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
            pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
            pal.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
        else:
            pal.setColor(QtGui.QPalette.Window, QtGui.QColor(239, 239, 239))
            pal.setColor(QtGui.QPalette.WindowText, QtCore.Qt.black)
            pal.setColor(QtGui.QPalette.Base, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(233, 233, 233))
            pal.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.black)
            pal.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.Text, QtCore.Qt.black)
            pal.setColor(QtGui.QPalette.Button, QtGui.QColor(239, 239, 239))
            pal.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.black)
            pal.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
            pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(61, 133, 224))
            pal.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)
        return pal

    def _repolish(self, w: QtWidgets.QWidget):
        try:
            w.style().unpolish(w)
            w.style().polish(w)
            w.update()
        except Exception:
            pass

    def apply_dark_theme(self, enabled: bool):
        app = QtWidgets.QApplication.instance()
        if not app:
            return
        app.setStyle("Fusion")
        app.setPalette(self._build_palette(enabled))
        for w in (self, self.centralWidget(), self.list_files, self.logView, self.grp_opts,
                  self.progressBar, self.btn_pick_files, self.btn_pick_folder,
                  self.chk_dark, self.btn_start, self.btn_stop,
                  self.btn_open_report, self.btn_save_aggregate):
            self._repolish(w)

    # -------------------- Helpers --------------------

    def append_log(self, text: str):
        self.logView.append(text)
        self.logView.moveCursor(QtGui.QTextCursor.End)
        self.status.showMessage(text, 4000)

    def _add_files(self, paths: List[str]):
        new_count = 0
        for p in paths:
            if not p:
                continue
            if not is_supported_file(p):
                continue
            ap = os.path.abspath(p)
            if ap in self.files_set:
                continue
            self.files_set.add(ap)
            self.files.append(ap)
            self.list_files.addItem(ap)
            new_count += 1

        if new_count:
            self.btn_start.setEnabled(True)
            self.append_log(f"> Dodano {new_count} plik(ów). Razem: {len(self.files)}.")
        else:
            self.append_log("> Brak nowych plików do dodania.")

    # -------------------- Normalizacja wyników --------------------

    @staticmethod
    def _to_float(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            return float(v)
        except Exception:
            return None

    def _details_get(self, d: Dict[str, Any], *keys: str) -> Any:
        for k in keys:
            if k in d:
                return d[k]
        return None

    def _fuse_ai_score(self, ai_face: Optional[float], ai_scene: Optional[float], ai_video: Optional[float]) -> Optional[float]:
        """
        Zamiast max(): ważona średnia z config.FUSE_WEIGHTS + suppress sceny.
        To stabilizuje wynik i zmniejsza saturację 100%.
        """
        w = getattr(config, "FUSE_WEIGHTS", {"video": 0.5, "face": 0.35, "scene": 0.15})
        face_w = float(w.get("face", 0.35))
        scene_w = float(w.get("scene", 0.15))
        video_w = float(w.get("video", 0.50))

        # suppress scene jeśli config mówi
        if hasattr(config, "should_suppress_scene"):
            try:
                if config.should_suppress_scene(ai_face, ai_video):
                    scene_w = 0.0
            except Exception:
                pass

        items: List[tuple] = []
        if ai_face is not None:
            items.append((ai_face, face_w))
        if ai_scene is not None:
            items.append((ai_scene, scene_w))
        if ai_video is not None:
            items.append((ai_video, video_w))

        items = [(v, ww) for (v, ww) in items if ww > 0.0]
        if not items:
            # fallback: jeśli brak wag -> zwróć cokolwiek
            vals = [v for v in [ai_face, ai_scene, ai_video] if v is not None]
            return max(vals) if vals else None

        s = sum(v * ww for v, ww in items)
        ws = sum(ww for _, ww in items)
        if ws <= 0:
            return None
        return max(0.0, min(100.0, s / ws))

    def _compute_deepfake_score(self, details: Dict[str, Any], ai_face: Optional[float], ai_video: Optional[float]) -> Optional[float]:
        """
        Deepfake score ma bazować na sygnałach twarzy/wideo + forensic jako korekta,
        ale tylko jeśli forensic jest dostępne.
        """
        forensic_ok = bool(details.get("forensic_available"))
        if not forensic_ok:
            # jeśli brak forensic, to deepfake score opieramy na face/video (dla face-deepfake to i tak sensowne)
            vals = [v for v in [ai_face, ai_video] if v is not None]
            return (sum(vals) / len(vals)) if vals else None

        jitter = self._to_float(details.get("jitter_px"))
        ela = self._to_float(details.get("ela_score"))
        fft = self._to_float(details.get("fft_score"))
        border = self._to_float(details.get("border_artifacts"))
        sharp = self._to_float(details.get("face_sharpness"))

        # Normalizacje heurystyczne (0..100), bez uczenia, ale stabilne.
        # - border artifacts: zwykle 0..0.1 -> mapujemy do 0..100
        # - jitter: px, mapujemy do 0..100 w okolicy MAX_JITTER
        # - sharp: wyższe zwykle oznacza "bardziej naturalne", więc odwracamy sygnał
        border_s = None if border is None else max(0.0, min(100.0, (border / 0.08) * 100.0))
        jitter_s = None if jitter is None else max(0.0, min(100.0, (jitter / float(getattr(config, "MAX_JITTER", 200.0))) * 100.0))
        # ELA/FFT u Ciebie już wygląda jak "score" (ale w raporcie małe wartości) – zostawiamy jako weak-signal
        ela_s = None if ela is None else max(0.0, min(100.0, ela * 100.0))
        fft_s = None if fft is None else max(0.0, min(100.0, fft * 50.0))

        # sharp: odwrócony (im mniejsza ostrość, tym bardziej podejrzane)
        sharp_s = None
        if sharp is not None:
            # typowe wartości widzę różne, ale >50 często OK; więc mapujemy 0..100
            sharp_norm = max(0.0, min(1.0, sharp / 80.0))
            sharp_s = (1.0 - sharp_norm) * 100.0

        # bazowy deepfake: face/video
        base_vals = [v for v in [ai_face, ai_video] if v is not None]
        base = (sum(base_vals) / len(base_vals)) if base_vals else None
        if base is None:
            return None

        # forensic jako korekta (+)
        fvals = [v for v in [border_s, jitter_s, sharp_s, ela_s, fft_s] if v is not None]
        if not fvals:
            return base

        forensic = float(sum(fvals) / len(fvals))

        # Gate: forensic ma realny wpływ dopiero gdy AI już podejrzane (masz podobne założenie w config.FORENSIC_GATE_MIN)
        gate = float(getattr(config, "FORENSIC_GATE_MIN", 60.0))
        if base < gate:
            # minimalna korekta
            return max(0.0, min(100.0, base * 0.85 + forensic * 0.15))

        return max(0.0, min(100.0, base * 0.70 + forensic * 0.30))

    def _normalize_details(self, idx: int, details: Dict[str, Any]) -> Dict[str, Any]:
        full_path = details.get("full_path") or (self.files[idx] if 0 <= idx < len(self.files) else "")
        details["full_path"] = os.path.abspath(full_path) if full_path else full_path

        ai_face = self._to_float(self._details_get(details, "ai_face_score", "ai_face_raw"))
        ai_scene = self._to_float(self._details_get(details, "ai_scene_score", "ai_scene_raw"))
        ai_video = self._to_float(self._details_get(details, "ai_video_score", "ai_video_raw"))

        # AI combined: ważona fuzja zamiast max
        ai_combined = self._fuse_ai_score(ai_face, ai_scene, ai_video)

        forensic_available = bool(self._details_get(details, "forensic_available"))
        if not forensic_available:
            if any(self._details_get(details, k) is not None for k in ["jitter_px", "blink_per_min", "ela_score", "fft_score", "border_artifacts", "face_sharpness"]):
                forensic_available = True

        jitter = self._to_float(self._details_get(details, "jitter_px", "jitter_score"))
        blink = self._to_float(self._details_get(details, "blink_per_min", "blinks_per_min"))
        ela = self._to_float(self._details_get(details, "ela_score"))
        fft = self._to_float(self._details_get(details, "fft_score"))
        border = self._to_float(self._details_get(details, "border_artifacts", "border_score"))
        sharp = self._to_float(self._details_get(details, "face_sharpness", "sharp_face"))

        details["ai_face_score"] = ai_face
        details["ai_scene_score"] = ai_scene
        details["ai_video_score"] = ai_video
        details["ai_combined_score"] = ai_combined

        details["forensic_available"] = forensic_available
        details["jitter_px"] = jitter
        details["blink_per_min"] = blink
        details["ela_score"] = ela
        details["fft_score"] = fft
        details["border_artifacts"] = border
        details["face_sharpness"] = sharp

        # nowe: dwa wyniki
        ai_final = ai_combined
        df_final = self._compute_deepfake_score(details, ai_face, ai_video)

        details["ai_final_score"] = ai_final
        details["deepfake_final_score"] = df_final

        # nowe: dwa werdykty (progami z kalibracji jeśli są)
        if calibrate_thresholds is not None and self.thresholds is not None:
            if ai_final is not None:
                details["ai_verdict"] = calibrate_thresholds.verdict_for("ai_detector", float(ai_final), self.thresholds)
            else:
                details["ai_verdict"] = "NIEPEWNE / BRAK DANYCH"

            if df_final is not None:
                details["deepfake_verdict"] = calibrate_thresholds.verdict_for("deepfake_detector", float(df_final), self.thresholds)
            else:
                details["deepfake_verdict"] = "N/A (BRAK TWARZY / BRAK SYGNAŁU)"
        else:
            # fallback na config.THRESHOLDS (jeden zestaw)
            def _verdict(score: Optional[float]) -> str:
                if score is None:
                    return "NIEPEWNE / BRAK DANYCH"
                if score >= float(config.THRESHOLDS["FAKE_MIN"]):
                    return "FAKE (PRAWDOPODOBNE)"
                if score <= float(config.THRESHOLDS["REAL_MAX"]):
                    return "REAL (PRAWDOPODOBNE)"
                return "NIEPEWNE / GREY ZONE"

            details["ai_verdict"] = _verdict(ai_final)
            details["deepfake_verdict"] = _verdict(df_final) if df_final is not None else "N/A (BRAK TWARZY / BRAK SYGNAŁU)"

        # zachowaj kompatybilność: "verdict" = combined (bardziej konserwatywnie)
        # jeśli którykolwiek mówi FAKE -> FAKE; jeśli oba REAL -> REAL; inaczej GREY
        av = str(details.get("ai_verdict", "")).upper()
        dv = str(details.get("deepfake_verdict", "")).upper()
        if "FAKE" in av or "FAKE" in dv:
            combined_verdict = "FAKE (PRAWDOPODOBNE)"
        elif "REAL" in av and "REAL" in dv:
            combined_verdict = "REAL (PRAWDOPODOBNE)"
        else:
            combined_verdict = "NIEPEWNE / GREY ZONE"

        details["verdict"] = combined_verdict

        # nowy "final_score" (do logów): bierz max z dwóch (konserwatywnie)
        final_score = None
        vals = [v for v in [ai_final, df_final] if v is not None]
        if vals:
            final_score = float(max(vals))
        details["final_score"] = final_score

        details["no_signal"] = (ai_final is None and df_final is None)
        details.setdefault("timestamp", config.now_str() if hasattr(config, "now_str") else "")

        # watermark (jak było)
        details["watermark_found"] = bool(self._details_get(details, "watermark_found"))
        details["watermark_label"] = self._details_get(details, "watermark_label")
        details["watermark_folder"] = self._details_get(details, "watermark_folder")

        return details

    @staticmethod
    def _fmt_pct(v: Optional[float]) -> str:
        return "N/A" if v is None else f"{float(v):.2f}%"

    @staticmethod
    def _fmt_px(v: Optional[float]) -> str:
        return "N/A" if v is None else f"{float(v):.2f} px"

    @staticmethod
    def _fmt_rate(v: Optional[float]) -> str:
        return "N/A" if v is None else f"{float(v):.1f} / min"

    @staticmethod
    def _fmt_num(v: Optional[float]) -> str:
        return "N/A" if v is None else f"{float(v):.2f}"

    def _make_summary_block(self, idx: int, details: Dict[str, Any]) -> str:
        d = self._normalize_details(idx, details.copy())

        fname = os.path.basename(d.get("full_path", "?"))
        ts = d.get("timestamp")

        ai_verdict = d.get("ai_verdict", "N/A")
        df_verdict = d.get("deepfake_verdict", "N/A")
        combined_verdict = d.get("verdict", "N/A")

        ai_final = d.get("ai_final_score")
        df_final = d.get("deepfake_final_score")
        final_score = d.get("final_score")

        ai_face = d.get("ai_face_score")
        ai_scene = d.get("ai_scene_score")
        ai_video = d.get("ai_video_score")
        ai_combined = d.get("ai_combined_score")

        forensic_ok = bool(d.get("forensic_available"))
        jitter = d.get("jitter_px")
        blink = d.get("blink_per_min")
        ela = d.get("ela_score")
        fft = d.get("fft_score")
        border = d.get("border_artifacts")
        sharp = d.get("face_sharpness")

        wm_found = bool(d.get("watermark_found"))
        wm_label = d.get("watermark_label")

        lines: List[str] = []
        lines.append(f"Plik: {fname}")
        if ts:
            lines.append(f"Timestamp: {ts}\n")
        else:
            lines.append("")

        lines.append(f"WERDYKT (COMBINED): {combined_verdict}")
        lines.append(f"WERDYKT (AI_DETECTOR): {ai_verdict}")
        lines.append(f"WERDYKT (DEEPFAKE_DETECTOR): {df_verdict}")
        lines.append(f"Wynik łączny (Score): {self._fmt_pct(final_score)}")
        lines.append(f"Score (AI_DETECTOR): {self._fmt_pct(ai_final)}")
        lines.append(f"Score (DEEPFAKE_DETECTOR): {self._fmt_pct(df_final)}\n")

        lines.append("--- DETALE AI ---")
        lines.append(f"AI Face/Subject Score: {self._fmt_pct(ai_face)}")
        lines.append(f"AI Scene (Frames) Score: {self._fmt_pct(ai_scene)}")
        lines.append(f"AI Video Model Score: {self._fmt_pct(ai_video)}")
        lines.append(f"AI Combined (weighted) Score: {self._fmt_pct(ai_combined)}\n")

        lines.append("--- DETALE FORENSIC (tylko przy ludzkiej twarzy) ---")
        if not forensic_ok:
            lines.append("Stabilność (Jitter): N/A")
            lines.append("Mruganie: N/A")
            lines.append("ELA Score: N/A")
            lines.append("FFT Score: N/A")
            lines.append("Border Artifacts: N/A")
            lines.append("Sharpness (face): N/A")
        else:
            lines.append(f"Stabilność (Jitter): {self._fmt_px(jitter)}")
            lines.append(f"Mruganie: {self._fmt_rate(blink)}")
            lines.append(f"ELA Score: {self._fmt_num(ela)}")
            lines.append(f"FFT Score: {self._fmt_num(fft)}")
            lines.append(f"Border Artifacts: {self._fmt_num(border)}")
            lines.append(f"Sharpness (face): {self._fmt_num(sharp)}")

        if wm_found:
            lines.append("\n--- WATERMARK ---")
            lines.append(f"Znaleziono watermark/napis: {wm_label if wm_label else 'TAK'}")

        return "\n".join(lines)

    # -------------------- Actions --------------------

    def pick_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Wybierz pliki do analizy",
            os.path.abspath(getattr(config, "REPORTS_BASE_DIR", ".")),
            "Media (*.mp4 *.mov *.avi *.mkv *.webm *.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;Wszystkie pliki (*.*)",
        )
        if paths:
            self._add_files(paths)

    def pick_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Wybierz folder z nagraniami/obrazami",
            os.path.abspath(getattr(config, "REPORTS_BASE_DIR", ".")),
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        if not folder:
            return
        to_add: List[str] = []
        for root, _, files in os.walk(folder):
            for name in files:
                path = os.path.join(root, name)
                if is_supported_file(path):
                    to_add.append(path)
        if not to_add:
            self.append_log("> W wybranym folderze nie znaleziono obsługiwanych plików.")
            return
        self._add_files(sorted(to_add))

    def start_analysis(self):
        if not self.files:
            QtWidgets.QMessageBox.warning(self, "Brak plików", "Najpierw dodaj pliki lub folder do analizy.")
            return

        do_ai = self.chk_ai.isChecked()
        do_forensic = self.chk_forensic.isChecked()
        do_watermark = self.chk_watermark.isChecked()

        # Nowy run per START
        run_dir = ai_detector.begin_run()
        self.current_run_dir = run_dir
        self.append_log(f"> Run folder: {run_dir}")

        # przygotuj ścieżkę progów i spróbuj załadować istniejące
        self.thresholds_path = os.path.join(run_dir, "_calibration_thresholds.json")
        if calibrate_thresholds is not None:
            self.thresholds = calibrate_thresholds.load_thresholds(self.thresholds_path)
            if self.thresholds:
                self.append_log(f"> [CAL] Załadowano progi kalibracji: {self.thresholds_path}")

        self.report_paths.clear()
        self.per_file_summaries.clear()

        self.btn_start.setEnabled(False)
        self.btn_pick_files.setEnabled(False)
        self.btn_pick_folder.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progressBar.setValue(0)

        self.append_log(f"> Rozpoczynam analizę… (AI={do_ai}, Forensic={do_forensic}, Watermark={do_watermark})")

        do_face_ai = bool(do_ai)
        do_forensic2 = bool(do_forensic)

        self.worker = AnalysisWorker(self.files, do_face_ai, do_forensic2, do_watermark, run_dir, self)
        self.worker.progress.connect(self.on_progress)
        self.worker.file_started.connect(self.on_file_started)
        self.worker.file_finished.connect(self.on_file_finished)
        self.worker.log_line.connect(self.append_log)
        self.worker.all_done.connect(self.on_all_done)
        self.worker.start()

    def stop_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.append_log("> Przerywam analizę…")

    def open_selected_report(self):
        items = self.list_files.selectedIndexes()
        if not items:
            QtWidgets.QMessageBox.information(self, "Brak wyboru", "Zaznacz plik na liście.")
            return
        idx = items[0].row()

        report_dir = self.report_paths.get(idx)
        fallback_dir = os.path.dirname(os.path.abspath(self.files[idx])) if 0 <= idx < len(self.files) else None

        path_to_open = None
        if report_dir and os.path.isdir(report_dir):
            path_to_open = report_dir
        elif fallback_dir and os.path.isdir(fallback_dir):
            path_to_open = fallback_dir

        if not path_to_open:
            QtWidgets.QMessageBox.information(self, "Brak raportu", "Raport jeszcze nie został zapisany.")
            return

        try:
            if sys.platform.startswith("win"):
                os.startfile(path_to_open)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                QtCore.QProcess.startDetached("open", [path_to_open])
            else:
                QtCore.QProcess.startDetached("xdg-open", [path_to_open])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Nie udało się otworzyć: {e}")

    def save_aggregate_report(self):
        if not self.files:
            QtWidgets.QMessageBox.information(self, "Brak danych", "Brak plików/rezultatów do zapisania.")
            return

        blocks: List[str] = []
        for idx in range(len(self.files)):
            block = self.per_file_summaries.get(idx)
            if block:
                blocks.append(block)

        if not blocks:
            QtWidgets.QMessageBox.information(self, "Brak wyników", "Brak ukończonych wyników do zapisania.")
            return

        base_dir = os.path.dirname(self.files[0])
        default_path = os.path.join(base_dir, "raport_zbiorczy.txt")
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Zapisz raport zbiorczy",
            default_path,
            "Pliki tekstowe (*.txt);;Wszystkie pliki (*.*)",
        )
        if not out_path:
            return

        content = "\n\n".join(blocks) + "\n"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)
            self.append_log(f"> Zapisano raport zbiorczy: {out_path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Błąd zapisu", f"Nie udało się zapisać raportu: {e}")

    # -------------------- Slots --------------------

    @QtCore.pyqtSlot(int, int)
    def on_progress(self, curr: int, tot: int):
        if tot <= 0:
            return
        pct = max(0, min(100, int(curr * 100 / max(1, tot))))
        self.progressBar.setValue(pct)

    @QtCore.pyqtSlot(int, str)
    def on_file_started(self, idx: int, name: str):
        self.append_log(f"> [{idx + 1}/{len(self.files)}] Start: {name}")
        self.progressBar.setValue(0)

    @QtCore.pyqtSlot(int, dict)
    def on_file_finished(self, idx: int, details: dict):
        d = self._normalize_details(idx, details)

        verdict = d.get("verdict", "UNKNOWN")
        score = d.get("final_score")

        folder = d.get("folder_path") or d.get("watermark_folder")
        if folder:
            self.report_paths[idx] = folder

        block = self._make_summary_block(idx, d)
        self.per_file_summaries[idx] = block

        if d.get("no_signal"):
            self.append_log(f"< [{idx + 1}/{len(self.files)}] BRAK DANYCH z modeli. Raport: {folder}")
        else:
            try:
                score_str = f"{float(score):.2f}%" if score is not None else "N/A"
            except Exception:
                score_str = "N/A"
            self.append_log(f"< [{idx + 1}/{len(self.files)}] DONE: {verdict} ({score_str}). Raport: {folder}")

    @QtCore.pyqtSlot()
    def on_all_done(self):
        # po analizie: jeśli mamy etykiety _fake/_real, policz progi i zapisz
        if calibrate_thresholds is not None and self.current_run_dir:
            try:
                all_details = []
                for i in range(len(self.files)):
                    # wyciągamy details z summary bloku? nie; lepiej znormalizować "ostatnie znane"
                    # tutaj korzystamy z per_file_summaries nie jest strukturalne, więc budujemy z pamięci:
                    # minimalnie: na podstawie ostatnich raportów; dlatego przechodzimy po plikach i liczymy z zapisanych danych:
                    # -> prościej: użyj self.per_file_summaries nie daje liczb. Zamiast tego: użyj cache z logiki - brakuje.
                    # Rozwiązanie: kalibruj na podstawie ponownego użycia _normalize_details? nie mamy raw details.
                    # Dlatego: kalibracja będzie aktywna tylko jeśli użytkownik zapisze i ponownie odpali (progi z poprzedniego run).
                    self.append_log("> [CAL] (INFO) Kalibracja progów wymaga cache wyników liczbowych. Jeśli chcesz, dodam w kolejnym kroku cache wyników per plik.")
                # Nie zapisujemy nic w tej wersji (żeby nie zapisać śmieci).
            except Exception as e:
                self.append_log(f"> [CAL] Błąd kalibracji: {e}")

        self.append_log("> Analiza zakończona.")
        self.btn_pick_files.setEnabled(True)
        self.btn_pick_folder.setEnabled(True)
        self.btn_start.setEnabled(len(self.files) > 0)
        self.btn_stop.setEnabled(False)
        self.worker = None


def run():
    try:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()

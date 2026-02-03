import os
import sys
from typing import List, Optional, Set, Dict, Any

from PyQt5 import QtCore, QtGui, QtWidgets

import ai_detector
import config

# gui.py
import importlib.util
import pathlib
import sys

def _load_local_ai_detector():
    # ładuje dokładnie ai_detector.py z tego samego katalogu co gui.py
    path = pathlib.Path(__file__).with_name("ai_detector.py")
    spec = importlib.util.spec_from_file_location("ai_detector", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ai_detector from: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ai_detector"] = mod
    spec.loader.exec_module(mod)
    return mod

ai_detector = _load_local_ai_detector()

# sanity check: GUI wymaga begin_run()
if not hasattr(ai_detector, "begin_run"):
    raise RuntimeError(f"Loaded ai_detector has no begin_run(). File: {getattr(ai_detector,'__file__',None)}")


def detect_one(video_path: str) -> str:
    res = scan_video(video_path, opts=ScanOptions(max_frames=24, do_ai=True, do_forensic=True))
    return format_report_pl(res)

try:
    import ocr_detector
except Exception as _e:
    ocr_detector = None
    print(f"[OCR] Moduł ocr_detector niedostępny: {_e}")

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
                # Fix: ai_detector.scan_for_deepfake might return (AiResult, ForensicResult) [2 values]
                # but the GUI expects (status, score, fake_ratio, details) [4 values].
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
                    score = ai_res.combined_max if hasattr(ai_res, "combined_max") else 0.0
                    fake_ratio = 0.0 # Default if not provided
                    details = {
                        "ai_face_score": ai_res.face_score if hasattr(ai_res, "face_score") else None,
                        "ai_scene_score": ai_res.scene_score if hasattr(ai_res, "scene_score") else None,
                        "ai_video_score": ai_res.video_score if hasattr(ai_res, "video_score") else None,
                        "jitter_px": for_res.jitter_px if hasattr(for_res, "jitter_px") else None,
                        "ela_score": for_res.ela_score if hasattr(for_res, "ela_score") else None,
                        "fft_score": for_res.fft_score if hasattr(for_res, "fft_score") else None,
                        "border_artifacts": for_res.border_artifacts if hasattr(for_res, "border_artifacts") else None,
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
    def _to_pct(v: Any) -> Optional[float]:
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

    def _normalize_details(self, idx: int, details: Dict[str, Any]) -> Dict[str, Any]:
        full_path = details.get("full_path") or (self.files[idx] if 0 <= idx < len(self.files) else "")
        details["full_path"] = os.path.abspath(full_path) if full_path else full_path

        ai_face = self._to_pct(self._details_get(details, "ai_face_score", "ai_face_raw"))
        ai_scene = self._to_pct(self._details_get(details, "ai_scene_score", "ai_scene_raw"))
        ai_video = self._to_pct(self._details_get(details, "ai_video_score", "ai_video_raw"))
        ai_combined = self._to_pct(self._details_get(details, "ai_combined_score", "ai_raw"))

        if ai_combined is None:
            vals = [v for v in [ai_face, ai_scene, ai_video] if v is not None]
            ai_combined = max(vals) if vals else None

        final_score = self._to_pct(self._details_get(details, "final_score", "raw_final_score"))
        if final_score is None:
            final_score = ai_combined

        forensic_available = bool(self._details_get(details, "forensic_available"))
        faces_any = bool(self._details_get(details, "faces_detected_any"))
        if not forensic_available:
            if any(self._details_get(details, k) is not None for k in ["jitter_px", "blink_per_min", "ela_score", "fft_score"]):
                forensic_available = True

        jitter = self._to_pct(self._details_get(details, "jitter_px", "jitter_score"))
        blink = self._to_pct(self._details_get(details, "blink_per_min", "blinks_per_min"))
        ela = self._to_pct(self._details_get(details, "ela_score"))
        fft = self._to_pct(self._details_get(details, "fft_score"))
        border = self._to_pct(self._details_get(details, "border_artifacts", "border_score"))
        sharp = self._to_pct(self._details_get(details, "face_sharpness", "sharp_face"))

        watermark_found = bool(self._details_get(details, "watermark_found"))
        watermark_label = self._details_get(details, "watermark_label")
        watermark_folder = self._details_get(details, "watermark_folder")

        details["ai_face_score"] = ai_face
        details["ai_scene_score"] = ai_scene
        details["ai_video_score"] = ai_video
        details["ai_combined_score"] = ai_combined
        details["final_score"] = final_score

        details["forensic_available"] = forensic_available
        details["faces_detected_any"] = faces_any

        details["jitter_px"] = jitter
        details["blink_per_min"] = blink
        details["ela_score"] = ela
        details["fft_score"] = fft
        details["border_artifacts"] = border
        details["face_sharpness"] = sharp

        details["watermark_found"] = watermark_found
        details["watermark_label"] = watermark_label
        details["watermark_folder"] = watermark_folder

        verdict = details.get("verdict")
        if not verdict:
            if final_score is None:
                verdict = "NIEPEWNE / BRAK DANYCH"
            else:
                verdict = self._verdict_from_score(float(final_score))
        details["verdict"] = verdict

        details["no_signal"] = (final_score is None and ai_combined is None and ai_face is None and ai_scene is None and ai_video is None)
        details.setdefault("timestamp", config.now_str() if hasattr(config, "now_str") else "")

        return details

    @staticmethod
    def _verdict_from_score(score: float) -> str:
        if score >= float(config.THRESHOLDS["FAKE_MIN"]):
            return "FAKE (PRAWDOPODOBNE)"
        if score <= float(config.THRESHOLDS["REAL_MAX"]):
            return "REAL (PRAWDOPODOBNE)"
        return "NIEPEWNE / GREY ZONE"

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
        verdict = d.get("verdict", "NIEZNANY")
        ts = d.get("timestamp")

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

        lines.append(f"WERDYKT PODSUMOWUJĄCY: {verdict}")
        lines.append(f"Wynik łączny (Score): {self._fmt_pct(final_score)}\n")

        ai_detector_verdict = d.get('ai_detector_verdict', 'N/A')
        ai_detector_score = d.get('ai_detector_score', 0.0)
        deepfake_verdict = d.get('deepfake_verdict', 'N/A')
        deepfake_score = d.get('deepfake_score', 0.0)

        lines.append("--- WERDYKTY CZĄSTKOWE ---")
        lines.append(f"AI DETECTOR (Scena/Gen): {ai_detector_verdict} ({ai_detector_score:.2f}%)")
        lines.append(f"DEEPFAKE (Twarz): {deepfake_verdict} ({deepfake_score:.2f}%)")
        lines.append("")

        lines.append("--- DETALE AI ---")
        lines.append(f"AI Face/Subject Score: {self._fmt_pct(ai_face)}")
        lines.append(f"AI Scene (Frames) Score: {self._fmt_pct(ai_scene)}")
        lines.append(f"AI Video Model Score: {self._fmt_pct(ai_video)}")
        lines.append(f"AI Combined (max) Score: {self._fmt_pct(ai_combined)}\n")

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

        self.report_paths.clear()
        self.per_file_summaries.clear()

        self.btn_start.setEnabled(False)
        self.btn_pick_files.setEnabled(False)
        self.btn_pick_folder.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progressBar.setValue(0)

        self.append_log(f"> Rozpoczynam analizę… (AI={do_ai}, Forensic={do_forensic}, Watermark={do_watermark})")

        # do_ai kontroluje tylko to, czy liczymy AI; w praktyce do_face_ai = do_ai itd.
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

# ------------------------------------------------------------
# Backward-compat: GUI oczekuje config.THRESHOLDS
# ------------------------------------------------------------
FAKE_MIN = float(globals().get("FAKE_MIN", 70.0))
REAL_MAX = float(globals().get("REAL_MAX", 30.0))

if "THRESHOLDS" not in globals():
    THRESHOLDS = {
        "FAKE_MIN": FAKE_MIN,
        "REAL_MAX": REAL_MAX,
    }


if __name__ == "__main__":
    run()

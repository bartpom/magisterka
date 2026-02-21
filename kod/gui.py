# gui.py  (PyQt6 + Modern QSS  –  drop-in replacement for PyQt5 version)
import os
import sys
import pathlib
import importlib.util
import sys as _sys
from typing import Optional, Any

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QCheckBox, QGroupBox, QListWidget,
    QProgressBar, QTextEdit, QFileDialog, QMessageBox, QAbstractItemView,
    QTableWidget, QTableWidgetItem, QHeaderView, QDoubleSpinBox, QSpinBox, QLabel,
    QFormLayout
)

import config

# ---------- lazy-load ai_detector from same directory ----------

def _load_local_ai_detector():
    path = pathlib.Path(__file__).with_name("ai_detector.py")
    spec = importlib.util.spec_from_file_location("ai_detector", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ai_detector from: {path}")
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["ai_detector"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


ai_detector = _load_local_ai_detector()

if not hasattr(ai_detector, "begin_run"):
    raise RuntimeError(
        f"Loaded ai_detector has no begin_run(). File: {getattr(ai_detector, '__file__', None)}"
    )

try:
    import ocr_detector  # type: ignore[import]
except Exception as _e:
    ocr_detector = None  # type: ignore[assignment]
    print(f"[OCR] Moduł ocr_detector niedostępny: {_e}")

try:
    import calibrate_thresholds  # type: ignore[import]
except Exception as _e:
    calibrate_thresholds = None  # type: ignore[assignment]
    print(f"[CAL] Moduł calibrate_thresholds niedostępny: {_e}")

SUPPORTED_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
}

# ============================ QSS Themes ============================

_DARK_QSS = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 5px 14px;
    min-height: 28px;
}
QPushButton:hover  { background-color: #45475a; border-color: #89b4fa; }
QPushButton:pressed { background-color: #585b70; }
QPushButton:disabled { background-color: #2a2a3d; color: #6c7086; border-color: #313244; }
QPushButton#btn_start {
    background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; border: none;
}
QPushButton#btn_start:hover    { background-color: #89d18a; }
QPushButton#btn_start:disabled { background-color: #2a3b2a; color: #4a5a4a; }
QPushButton#btn_stop {
    background-color: #f38ba8; color: #1e1e2e; font-weight: bold; border: none;
}
QPushButton#btn_stop:hover    { background-color: #e07090; }
QPushButton#btn_stop:disabled { background-color: #3b2a2a; color: #5a4a4a; }
QGroupBox {
    border: 1px solid #45475a; border-radius: 6px;
    margin-top: 8px; padding: 4px;
    color: #89b4fa; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; }
QCheckBox { color: #cdd6f4; spacing: 6px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid #45475a; border-radius: 3px; background-color: #313244;
}
QCheckBox::indicator:checked { background-color: #89b4fa; border-color: #89b4fa; }
QListWidget {
    background-color: #181825; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 6px; outline: none;
}
QListWidget::item:selected { background-color: #313244; color: #89b4fa; }
QListWidget::item:hover    { background-color: #292938; }
QTableWidget {
    background-color: #181825; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 6px;
    gridline-color: #313244;
}
QTableWidget::item:selected { background-color: #313244; color: #89b4fa; }
QHeaderView::section {
    background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; padding: 4px; font-weight: bold;
}
QDoubleSpinBox, QSpinBox {
    background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 4px; padding: 2px;
}
QTextEdit {
    background-color: #11111b; color: #a6e3a1;
    border: 1px solid #45475a; border-radius: 6px;
    font-family: "Consolas", "JetBrains Mono", "Courier New", monospace;
    font-size: 12px;
}
QProgressBar {
    background-color: #313244; border: 1px solid #45475a;
    border-radius: 5px; text-align: center; color: #cdd6f4; min-height: 20px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #89b4fa,stop:1 #a6e3a1);
    border-radius: 4px;
}
QScrollBar:vertical   { background: #1e1e2e; width: 10px; border-radius: 5px; }
QScrollBar::handle:vertical { background: #45475a; border-radius: 5px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QSplitter::handle { background-color: #45475a; }
QStatusBar { background-color: #181825; color: #6c7086; border-top: 1px solid #45475a; }
"""

_LIGHT_QSS = """
QMainWindow, QWidget {
    background-color: #eff1f5; color: #4c4f69;
    font-family: "Segoe UI", "Inter", sans-serif; font-size: 13px;
}
QPushButton {
    background-color: #e6e9ef; color: #4c4f69;
    border: 1px solid #bcc0cc; border-radius: 6px;
    padding: 5px 14px; min-height: 28px;
}
QPushButton:hover   { background-color: #dce0e8; border-color: #1e66f5; }
QPushButton:pressed { background-color: #ccd0da; }
QPushButton:disabled { background-color: #e6e9ef; color: #9ca0b0; border-color: #ccd0da; }
QPushButton#btn_start {
    background-color: #40a02b; color: #eff1f5; font-weight: bold; border: none;
}
QPushButton#btn_start:hover    { background-color: #379128; }
QPushButton#btn_start:disabled { background-color: #c8e6c0; color: #9abf93; }
QPushButton#btn_stop {
    background-color: #d20f39; color: #eff1f5; font-weight: bold; border: none;
}
QPushButton#btn_stop:hover    { background-color: #b50e33; }
QPushButton#btn_stop:disabled { background-color: #f5b8c5; color: #c08090; }
QGroupBox {
    border: 1px solid #bcc0cc; border-radius: 6px;
    margin-top: 8px; padding: 4px; color: #1e66f5; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; }
QCheckBox { color: #4c4f69; spacing: 6px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid #bcc0cc; border-radius: 3px; background-color: #e6e9ef;
}
QCheckBox::indicator:checked { background-color: #1e66f5; border-color: #1e66f5; }
QListWidget {
    background-color: #dce0e8; color: #4c4f69;
    border: 1px solid #bcc0cc; border-radius: 6px; outline: none;
}
QListWidget::item:selected { background-color: #c8d4f5; color: #1e66f5; }
QListWidget::item:hover    { background-color: #e6e9ef; }
QTableWidget {
    background-color: #dce0e8; color: #4c4f69;
    border: 1px solid #bcc0cc; border-radius: 6px;
    gridline-color: #bcc0cc;
}
QTableWidget::item:selected { background-color: #c8d4f5; color: #1e66f5; }
QHeaderView::section {
    background-color: #e6e9ef; color: #4c4f69;
    border: 1px solid #bcc0cc; padding: 4px; font-weight: bold;
}
QDoubleSpinBox, QSpinBox {
    background-color: #e6e9ef; color: #4c4f69;
    border: 1px solid #bcc0cc; border-radius: 4px; padding: 2px;
}
QTextEdit {
    background-color: #e6e9ef; color: #4c4f69;
    border: 1px solid #bcc0cc; border-radius: 6px;
    font-family: "Consolas", "JetBrains Mono", "Courier New", monospace;
    font-size: 12px;
}
QProgressBar {
    background-color: #dce0e8; border: 1px solid #bcc0cc;
    border-radius: 5px; text-align: center; color: #4c4f69; min-height: 20px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1e66f5,stop:1 #40a02b);
    border-radius: 4px;
}
QScrollBar:vertical   { background: #eff1f5; width: 10px; border-radius: 5px; }
QScrollBar::handle:vertical { background: #bcc0cc; border-radius: 5px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QSplitter::handle { background-color: #bcc0cc; }
QStatusBar { background-color: #dce0e8; color: #9ca0b0; border-top: 1px solid #bcc0cc; }
"""


# ============================ helpers ============================

def is_supported_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTS


# ============================ Worker ============================

class AnalysisWorker(QtCore.QThread):
    progress = pyqtSignal(int, int)
    file_started = pyqtSignal(int, str)
    file_finished = pyqtSignal(int, dict)
    log_line = pyqtSignal(str)
    all_done = pyqtSignal()

    def __init__(
        self,
        files: list[str],
        do_face_ai: bool,
        do_forensic: bool,
        do_watermark: bool,
        run_dir: str,
        confidence: float,
        sample_rate: int,
        parent=None,
    ):
        super().__init__(parent)
        self._files = files
        self._do_face_ai = do_face_ai
        self._do_forensic = do_forensic
        self._do_watermark = do_watermark
        self._run_dir = run_dir
        self._confidence = confidence
        self._sample_rate = sample_rate
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _run_watermark(self, path: str) -> Optional[dict]:
        if ocr_detector is None:
            self.log_line.emit("[OCR] Moduł ocr_detector nie został załadowany – pomijam detekcję watermarków.")
            return None
        fn = getattr(ocr_detector, "scan_for_watermarks", None)
        if callable(fn):
            try:
                self.log_line.emit(f"[OCR] Start detekcji znaków wodnych (conf={self._confidence}, sample_rate={self._sample_rate})…")
                # Przekazanie przywróconych parametrów
                res = fn(path, check_stop=lambda: self._stop, progress_callback=None, confidence=self._confidence, sample_rate=self._sample_rate)
                return res if isinstance(res, dict) else {"watermark_raw": res}
            except Exception as e:
                self.log_line.emit(f"[OCR] Błąd detekcji watermarków: {e}")
        return None

    def run(self) -> None:
        for idx, path in enumerate(self._files):
            if self._stop:
                break
            self.file_started.emit(idx, os.path.basename(path))

            def cb(curr, tot):
                self.progress.emit(int(curr), int(tot))

            try:
                # Wywołanie zgodne z nowym API ai_detector.scan_for_deepfake
                res = ai_detector.scan_for_deepfake(
                    path,
                    progress_callback=cb,
                    check_stop=lambda: self._stop,
                    detection_mode="combined",
                )

                if isinstance(res, tuple) and len(res) == 4:
                    verdict, score, fake_ratio, det = res
                    details: dict[str, Any] = det or {}
                    details["status"] = verdict
                    details.setdefault("raw_final_score", details.get("final_score", score))
                    details.setdefault("fake_ratio", fake_ratio)
                elif isinstance(res, tuple) and len(res) == 2:
                    ai_res, for_res = res
                    details = {
                        "status": "DONE",
                        "ai_face_score":     getattr(ai_res,  "face_score",       None),
                        "ai_scene_score":    getattr(ai_res,  "scene_score",      None),
                        "ai_video_score":    getattr(ai_res,  "video_score",      None),
                        "ai_combined_score": getattr(ai_res,  "combined_max",     None),
                        "jitter_px":         getattr(for_res, "jitter_px",        None),
                        "blink_per_min":     getattr(for_res, "blink_per_min",    None),
                        "ela_score":         getattr(for_res, "ela_score",        None),
                        "fft_score":         getattr(for_res, "fft_score",        None),
                        "border_artifacts":  getattr(for_res, "border_artifacts", None),
                        "face_sharpness":    getattr(for_res, "face_sharpness",   None),
                    }
                    details.setdefault("raw_final_score", float(getattr(ai_res, "combined_max", 0.0) or 0.0))
                    details.setdefault("fake_ratio", 0.0)
                else:
                    raise ValueError(f"Nieoczekiwany format zwrotny z scan_for_deepfake: {type(res)}")

                details.setdefault("full_path", os.path.abspath(path))

            except Exception as e:
                self.log_line.emit(f"[BŁĄD] {os.path.basename(path)} (AI): {e}")
                details = {
                    "status": "ERROR", "verdict": "ERROR",
                    "final_score": 0.0, "full_path": os.path.abspath(path),
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI / Deepfake Detector & Watermark (PyQt6)")
        self.resize(1100, 760)

        self.worker: Optional[AnalysisWorker] = None
        self.files: list[str] = []
        self.files_set: set[str] = set()
        self.report_paths: dict[int, str] = {}
        self.per_file_summaries: dict[int, str] = {}
        self.current_run_dir: Optional[str] = None
        self.thresholds = None
        self.thresholds_path: Optional[str] = None

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        # ---- toolbar row ----
        top = QHBoxLayout()
        top.setSpacing(6)
        root.addLayout(top)

        self.btn_pick_files = QPushButton("📂 Dodaj pliki…")
        self.btn_pick_files.clicked.connect(self.pick_files)
        top.addWidget(self.btn_pick_files)

        self.btn_pick_folder = QPushButton("📁 Dodaj folder…")
        self.btn_pick_folder.clicked.connect(self.pick_folder)
        top.addWidget(self.btn_pick_folder)

        self.grp_opts = QGroupBox("Ustawienia analizy")
        opts_lay = QVBoxLayout(self.grp_opts)
        opts_lay.setContentsMargins(8, 4, 8, 4)
        
        # Checkboxy
        chk_lay = QHBoxLayout()
        self.chk_ai        = QCheckBox("Analiza AI (twarz/scena/wideo)")
        self.chk_ai.setChecked(True)
        self.chk_forensic  = QCheckBox("Forensic / biometria")
        self.chk_forensic.setChecked(True)
        self.chk_watermark = QCheckBox("Znaki wodne (OCR/YOLO)")
        self.chk_watermark.setChecked(True) # Domyślnie włączone, bo o to prosił user
        chk_lay.addWidget(self.chk_ai)
        chk_lay.addWidget(self.chk_forensic)
        chk_lay.addWidget(self.chk_watermark)
        opts_lay.addLayout(chk_lay)

        # Przywrócone opcje Watermark
        param_lay = QFormLayout()
        
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.1, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.60) # Domyślnie z pliku PDF
        self.spin_conf.setToolTip("Minimalna pewność OCR")
        
        self.spin_sample = QSpinBox()
        self.spin_sample.setRange(1, 300)
        self.spin_sample.setValue(30) # Domyślnie z pliku PDF
        self.spin_sample.setToolTip("Odstęp próbkowania klatek (co X klatkę)")
        
        param_lay.addRow("OCR Confidence:", self.spin_conf)
        param_lay.addRow("Sample rate:", self.spin_sample)
        opts_lay.addLayout(param_lay)

        top.addWidget(self.grp_opts, 1)

        self.chk_dark = QCheckBox("🌙 Ciemny")
        self.chk_dark.setChecked(True)
        self.chk_dark.toggled.connect(self._apply_theme)
        top.addWidget(self.chk_dark)

        self.btn_start = QPushButton("▶ START")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_analysis)
        top.addWidget(self.btn_start)

        self.btn_stop = QPushButton("■ STOP")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_analysis)
        top.addWidget(self.btn_stop)

        # ---- splitter: table + log ----
        splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(splitter, 1)

        # PRZYWRÓCONA TABELA zamiast QListWidget
        self.table_results = QTableWidget()
        self.table_results.setColumnCount(4)
        self.table_results.setHorizontalHeaderLabels(["Plik", "Typ", "Liczba Detekcji (WM)", "Ścieżka CSV/Raport"])
        self.table_results.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table_results.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_results.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_results.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table_results.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_results.setAlternatingRowColors(True)
        splitter.addWidget(self.table_results)

        self.logView = QTextEdit()
        self.logView.setReadOnly(True)
        splitter.addWidget(self.logView)
        splitter.setSizes([300, 300])

        # ---- bottom bar ----
        bottom = QHBoxLayout()
        bottom.setSpacing(6)
        root.addLayout(bottom)

        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.setFormat("%p%")
        bottom.addWidget(self.progressBar, 1)

        # Przywrócone Otwórz Output
        self.btn_open_folder = QPushButton("📂 Open Output Folder")
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        self.btn_open_folder.setEnabled(False)
        bottom.addWidget(self.btn_open_folder)

        self.btn_open_report = QPushButton("📄 Otwórz raport pliku")
        self.btn_open_report.clicked.connect(self.open_selected_report)
        bottom.addWidget(self.btn_open_report)

        self.btn_save_aggregate = QPushButton("💾 Zapisz raport zbiorczy…")
        self.btn_save_aggregate.clicked.connect(self.save_aggregate_report)
        bottom.addWidget(self.btn_save_aggregate)

        self.status = self.statusBar()
        self._apply_theme(True)

    # -------------------- Theme --------------------

    def _apply_theme(self, dark: bool) -> None:
        app = QApplication.instance()
        if not app:
            return
        app.setStyle("Fusion")  # type: ignore[union-attr]
        app.setStyleSheet(_DARK_QSS if dark else _LIGHT_QSS)  # type: ignore[union-attr]

    # -------------------- Helpers --------------------

    def append_log(self, text: str) -> None:
        self.logView.append(text)
        self.logView.moveCursor(QTextCursor.MoveOperation.End)
        self.status.showMessage(text, 4000)

    def _add_files(self, paths: list[str]) -> None:
        added = 0
        for p in paths:
            if not p or not is_supported_file(p):
                continue
            ap = os.path.abspath(p)
            if ap in self.files_set:
                continue
            self.files_set.add(ap)
            self.files.append(ap)
            
            # Dodaj wiersz do tabeli
            row = self.table_results.rowCount()
            self.table_results.insertRow(row)
            
            fname = os.path.basename(ap)
            ext = os.path.splitext(fname)[1].lower()
            file_type = "Video" if ext in {".mp4", ".mkv", ".avi", ".webm"} else "Image"
            
            self.table_results.setItem(row, 0, QTableWidgetItem(fname))
            self.table_results.setItem(row, 1, QTableWidgetItem(file_type))
            self.table_results.setItem(row, 2, QTableWidgetItem("-"))
            self.table_results.setItem(row, 3, QTableWidgetItem("-"))
            
            added += 1
            
        if added:
            self.btn_start.setEnabled(True)
            self.append_log(f"> Dodano {added} plik(ów). Razem: {len(self.files)}.")
        else:
            self.append_log("> Brak nowych plików do dodania.")

    # -------------------- Score normalization --------------------

    @staticmethod
    def _to_float(v: Any) -> Optional[float]:
        try:
            return None if v is None else float(v)
        except Exception:
            return None

    def _details_get(self, d: dict[str, Any], *keys: str) -> Any:
        for k in keys:
            if k in d:
                return d[k]
        return None

    def _fuse_ai_score(
        self,
        ai_face: Optional[float],
        ai_scene: Optional[float],
        ai_video: Optional[float],
    ) -> Optional[float]:
        w = getattr(config, "FUSE_WEIGHTS", {"video": 0.5, "face": 0.35, "scene": 0.15})
        face_w  = float(w.get("face",  0.35))
        scene_w = float(w.get("scene", 0.15))
        video_w = float(w.get("video", 0.50))
        if hasattr(config, "should_suppress_scene"):
            try:
                if config.should_suppress_scene(ai_face, ai_video):
                    scene_w = 0.0
            except Exception:
                pass
        items = [
            (v, ww)
            for v, ww in [(ai_face, face_w), (ai_scene, scene_w), (ai_video, video_w)]
            if v is not None and ww > 0
        ]
        if not items:
            vals = [v for v in [ai_face, ai_scene, ai_video] if v is not None]
            return max(vals) if vals else None
        s  = sum(v * ww for v, ww in items)
        ws = sum(ww       for _, ww in items)
        return max(0.0, min(100.0, s / ws)) if ws > 0 else None

    def _compute_deepfake_score(
        self,
        details: dict[str, Any],
        ai_face: Optional[float],
        ai_video: Optional[float],
    ) -> Optional[float]:
        if not bool(details.get("forensic_available")):
            vals = [v for v in [ai_face, ai_video] if v is not None]
            return (sum(vals) / len(vals)) if vals else None
        jitter = self._to_float(details.get("jitter_px"))
        ela    = self._to_float(details.get("ela_score"))
        fft    = self._to_float(details.get("fft_score"))
        border = self._to_float(details.get("border_artifacts"))
        sharp  = self._to_float(details.get("face_sharpness"))
        border_s = None if border is None else max(0.0, min(100.0, (border / 0.08) * 100.0))
        jitter_s = None if jitter is None else max(0.0, min(100.0, (jitter / float(getattr(config, "MAX_JITTER", 200.0))) * 100.0))
        ela_s    = None if ela    is None else max(0.0, min(100.0, ela * 100.0))
        fft_s    = None if fft    is None else max(0.0, min(100.0, fft * 50.0))
        sharp_s  = None if sharp  is None else (1.0 - max(0.0, min(1.0, sharp / 80.0))) * 100.0
        base_vals = [v for v in [ai_face, ai_video] if v is not None]
        base = (sum(base_vals) / len(base_vals)) if base_vals else None
        if base is None:
            return None
        fvals = [v for v in [border_s, jitter_s, sharp_s, ela_s, fft_s] if v is not None]
        if not fvals:
            return base
        forensic = sum(fvals) / len(fvals)
        gate = float(getattr(config, "FORENSIC_GATE_MIN", 60.0))
        mix  = (0.85, 0.15) if base < gate else (0.70, 0.30)
        return max(0.0, min(100.0, base * mix[0] + forensic * mix[1]))

    def _verdict_from_fallback(self, score: Optional[float]) -> str:
        if score is None:
            return "NIEPEWNE / BRAK DANYCH"
        fake_min = float(getattr(config, "FAKE_MIN", 60.0))
        real_max = float(getattr(config, "REAL_MAX", 30.0))
        if score >= fake_min:
            return "FAKE (PRAWDOPODOBNE)"
        if score <= real_max:
            return "REAL (PRAWDOPODOBNE)"
        return "NIEPEWNE / GREY ZONE"

    def _normalize_details(self, idx: int, details: dict[str, Any]) -> dict[str, Any]:
        fp = details.get("full_path") or (self.files[idx] if 0 <= idx < len(self.files) else "")
        details["full_path"] = os.path.abspath(fp) if fp else fp

        ai_face  = self._to_float(self._details_get(details, "ai_face_score",  "ai_face_raw"))
        ai_scene = self._to_float(self._details_get(details, "ai_scene_score", "ai_scene_raw"))
        ai_video = self._to_float(self._details_get(details, "ai_video_score", "ai_video_raw"))
        ai_combined = self._fuse_ai_score(ai_face, ai_scene, ai_video)

        forensic_available = bool(self._details_get(details, "forensic_available"))
        if not forensic_available:
            forensic_available = any(
                self._details_get(details, k) is not None
                for k in ["jitter_px", "blink_per_min", "ela_score", "fft_score", "border_artifacts", "face_sharpness"]
            )

        details.update(
            ai_face_score=ai_face, ai_scene_score=ai_scene, ai_video_score=ai_video,
            ai_combined_score=ai_combined, forensic_available=forensic_available,
            jitter_px=self._to_float(self._details_get(details, "jitter_px",       "jitter_score")),
            blink_per_min=self._to_float(self._details_get(details, "blink_per_min", "blinks_per_min")),
            ela_score=self._to_float(details.get("ela_score")),
            fft_score=self._to_float(details.get("fft_score")),
            border_artifacts=self._to_float(self._details_get(details, "border_artifacts", "border_score")),
            face_sharpness=self._to_float(self._details_get(details, "face_sharpness",  "sharp_face")),
        )

        ai_final = ai_combined
        df_final = self._compute_deepfake_score(details, ai_face, ai_video)
        details["ai_final_score"] = ai_final
        details["deepfake_final_score"] = df_final

        if calibrate_thresholds is not None and self.thresholds is not None:
            details["ai_verdict"] = (
                calibrate_thresholds.verdict_for("ai_detector", float(ai_final), self.thresholds)
                if ai_final is not None else "NIEPEWNE / BRAK DANYCH"
            )
            details["deepfake_verdict"] = (
                calibrate_thresholds.verdict_for("deepfake_detector", float(df_final), self.thresholds)
                if df_final is not None else "N/A (BRAK TWARZY / BRAK SYGNAŁU)"
            )
        else:
            details["ai_verdict"]       = self._verdict_from_fallback(ai_final)
            details["deepfake_verdict"] = (
                self._verdict_from_fallback(df_final) if df_final is not None
                else "N/A (BRAK TWARZY / BRAK SYGNAŁU)"
            )

        av = str(details.get("ai_verdict",       "")).upper()
        dv = str(details.get("deepfake_verdict", "")).upper()
        if   "FAKE" in av or "FAKE" in dv:    details["verdict"] = "FAKE (PRAWDOPODOBNE)"
        elif "REAL" in av and "REAL" in dv:   details["verdict"] = "REAL (PRAWDOPODOBNE)"
        else:                                  details["verdict"] = "NIEPEWNE / GREY ZONE"

        vals = [v for v in [ai_final, df_final] if v is not None]
        details["final_score"] = float(max(vals)) if vals else None
        details["no_signal"]   = (ai_final is None and df_final is None)
        details.setdefault("timestamp", config.now_str() if hasattr(config, "now_str") else "")
        details["watermark_found"]  = bool(self._details_get(details, "watermark_found"))
        details["watermark_label"]  = self._details_get(details, "watermark_label")
        details["watermark_folder"] = self._details_get(details, "watermark_folder")
        details["watermark_count"]  = self._details_get(details, "watermark_count", "watermarks_detected")
        return details

    @staticmethod
    def _fmt_pct(v: Optional[float])  -> str: return "N/A" if v is None else f"{float(v):.2f}%"
    @staticmethod
    def _fmt_px(v:  Optional[float])  -> str: return "N/A" if v is None else f"{float(v):.2f} px"
    @staticmethod
    def _fmt_rate(v: Optional[float]) -> str: return "N/A" if v is None else f"{float(v):.1f} / min"
    @staticmethod
    def _fmt_num(v:  Optional[float]) -> str: return "N/A" if v is None else f"{float(v):.2f}"

    def _make_summary_block(self, idx: int, details: dict[str, Any]) -> str:
        d = self._normalize_details(idx, details.copy())
        fname = os.path.basename(d.get("full_path", "?"))
        ts = d.get("timestamp")
        lines: list[str] = [
            f"Plik: {fname}",
            f"Timestamp: {ts}" if ts else "",
            "",
            f"WERDYKT (COMBINED):        {d.get('verdict',          'N/A')}",
            f"WERDYKT (AI_DETECTOR):     {d.get('ai_verdict',       'N/A')}",
            f"WERDYKT (DEEPFAKE_DETECT): {d.get('deepfake_verdict', 'N/A')}",
            f"Wynik łączny  (Score):     {self._fmt_pct(d.get('final_score'))}",
            f"Score AI_DETECTOR:         {self._fmt_pct(d.get('ai_final_score'))}",
            f"Score DEEPFAKE_DETECTOR:   {self._fmt_pct(d.get('deepfake_final_score'))}",
            "",
            "--- DETALE AI ---",
            f"AI Face/Subject Score:     {self._fmt_pct(d.get('ai_face_score'))}",
            f"AI Scene (Frames) Score:   {self._fmt_pct(d.get('ai_scene_score'))}",
            f"AI Video Model Score:      {self._fmt_pct(d.get('ai_video_score'))}",
            f"AI Combined Score:         {self._fmt_pct(d.get('ai_combined_score'))}",
            "",
            "--- DETALE FORENSIC ---",
        ]
        if not bool(d.get("forensic_available")):
            lines += ["Stabilność (Jitter): N/A", "Mruganie: N/A",
                      "ELA Score: N/A", "FFT Score: N/A",
                      "Border Artifacts: N/A", "Sharpness (face): N/A"]
        else:
            lines += [
                f"Stabilność (Jitter):  {self._fmt_px(d.get('jitter_px'))}",
                f"Mruganie:             {self._fmt_rate(d.get('blink_per_min'))}",
                f"ELA Score:            {self._fmt_num(d.get('ela_score'))}",
                f"FFT Score:            {self._fmt_num(d.get('fft_score'))}",
                f"Border Artifacts:     {self._fmt_num(d.get('border_artifacts'))}",
                f"Sharpness (face):     {self._fmt_num(d.get('face_sharpness'))}",
            ]
        if bool(d.get("watermark_found")):
            lbl = d.get("watermark_label") or "TAK"
            lines += ["", "--- WATERMARK ---", f"Znaleziono watermark/napis: {lbl}"]
        return "\n".join(lines)

    # -------------------- Actions --------------------

    def pick_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Wybierz pliki do analizy",
            os.path.abspath(getattr(config, "REPORTS_BASE_DIR", ".")),
            "Media (*.mp4 *.mov *.avi *.mkv *.webm *.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)"
            ";;Wszystkie pliki (*.*)",
        )
        if paths:
            self._add_files(paths)

    def pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Wybierz folder z nagraniami/obrazami",
            os.path.abspath(getattr(config, "REPORTS_BASE_DIR", ".")),
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        if not folder:
            return
        to_add = [
            os.path.join(root, name)
            for root, _, files in os.walk(folder)
            for name in files
            if is_supported_file(name)
        ]
        if not to_add:
            self.append_log("> W wybranym folderze nie znaleziono obsługiwanych plików.")
            return
        self._add_files(sorted(to_add))

    def start_analysis(self) -> None:
        if not self.files:
            QMessageBox.warning(self, "Brak plików", "Najpierw dodaj pliki lub folder do analizy.")
            return
        run_dir = ai_detector.begin_run()
        self.current_run_dir = run_dir
        self.btn_open_folder.setEnabled(True) # Odblokowujemy przycisk
        
        self.append_log(f"> Run folder: {run_dir}")
        self.thresholds_path = os.path.join(run_dir, "_calibration_thresholds.json")
        if calibrate_thresholds is not None:
            self.thresholds = calibrate_thresholds.load_thresholds(self.thresholds_path)
            if self.thresholds:
                self.append_log(f"> [CAL] Załadowano progi kalibracji: {self.thresholds_path}")
        self.report_paths.clear()
        self.per_file_summaries.clear()
        for btn in (self.btn_start, self.btn_pick_files, self.btn_pick_folder):
            btn.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progressBar.setValue(0)
        
        # Pobieramy ustawienia
        do_ai        = self.chk_ai.isChecked()
        do_forensic  = self.chk_forensic.isChecked()
        do_watermark = self.chk_watermark.isChecked()
        conf_val     = self.spin_conf.value()
        sample_val   = self.spin_sample.value()
        
        self.append_log(f"> Rozpoczynam analizę… (AI={do_ai}, Forensic={do_forensic}, Watermark={do_watermark})")
        self.worker = AnalysisWorker(
            self.files, bool(do_ai), bool(do_forensic), bool(do_watermark), run_dir, 
            confidence=conf_val, sample_rate=sample_val, parent=self
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.file_started.connect(self.on_file_started)
        self.worker.file_finished.connect(self.on_file_finished)
        self.worker.log_line.connect(self.append_log)
        self.worker.all_done.connect(self.on_all_done)
        self.worker.start()

    def stop_analysis(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.append_log("> Przerywam analizę…")

    def open_output_folder(self) -> None:
        if not self.current_run_dir or not os.path.isdir(self.current_run_dir):
            QMessageBox.information(self, "Brak Outputu", "Brak aktywnego folderu z raportami.")
            return
        path_to_open = os.path.abspath(self.current_run_dir)
        try:
            if sys.platform.startswith("win"):
                os.startfile(path_to_open)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                QtCore.QProcess.startDetached("open",     [path_to_open])
            else:
                QtCore.QProcess.startDetached("xdg-open", [path_to_open])
        except Exception as e:
            QMessageBox.warning(self, "Błąd", f"Nie udało się otworzyć: {e}")

    def open_selected_report(self) -> None:
        items = self.table_results.selectedItems()
        if not items:
            QMessageBox.information(self, "Brak wyboru", "Zaznacz plik w tabeli.")
            return
        idx = items[0].row()
        path_to_open = (
            self.report_paths.get(idx)
            or (os.path.dirname(os.path.abspath(self.files[idx])) if 0 <= idx < len(self.files) else None)
        )
        if not path_to_open or not os.path.isdir(path_to_open):
            QMessageBox.information(self, "Brak raportu", "Raport jeszcze nie został zapisany.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(path_to_open)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                QtCore.QProcess.startDetached("open",     [path_to_open])
            else:
                QtCore.QProcess.startDetached("xdg-open", [path_to_open])
        except Exception as e:
            QMessageBox.warning(self, "Błąd", f"Nie udało się otworzyć: {e}")

    def save_aggregate_report(self) -> None:
        if not self.files:
            QMessageBox.information(self, "Brak danych", "Brak plików/rezultatów do zapisania.")
            return
        blocks = [b for i in range(len(self.files)) if (b := self.per_file_summaries.get(i))]
        if not blocks:
            QMessageBox.information(self, "Brak wyników", "Brak ukończonych wyników do zapisania.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Zapisz raport zbiorczy",
            os.path.join(os.path.dirname(self.files[0]), "raport_zbiorczy.txt"),
            "Pliki tekstowe (*.txt);;Wszystkie pliki (*.*)",
        )
        if not out_path:
            return
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(blocks) + "\n")
            self.append_log(f"> Zapisano raport zbiorczy: {out_path}")
        except Exception as e:
            QMessageBox.warning(self, "Błąd zapisu", f"Nie udało się zapisać raportu: {e}")

    # -------------------- Slots --------------------

    @pyqtSlot(int, int)
    def on_progress(self, curr: int, tot: int) -> None:
        if tot > 0:
            self.progressBar.setValue(max(0, min(100, int(curr * 100 / max(1, tot)))))

    @pyqtSlot(int, str)
    def on_file_started(self, idx: int, name: str) -> None:
        self.append_log(f"> [{idx + 1}/{len(self.files)}] Start: {name}")
        self.progressBar.setValue(0)

    @pyqtSlot(int, dict)
    def on_file_finished(self, idx: int, details: dict) -> None:
        d = self._normalize_details(idx, details)
        folder = d.get("folder_path") or d.get("watermark_folder")
        if folder:
            self.report_paths[idx] = folder
            
        # Aktualizacja tabeli z wynikami
        count = d.get("watermark_count")
        if count is not None:
            self.table_results.setItem(idx, 2, QTableWidgetItem(str(count)))
        elif d.get("watermark_found"):
            self.table_results.setItem(idx, 2, QTableWidgetItem("Wykryto (nieznana ilość)"))
        else:
            self.table_results.setItem(idx, 2, QTableWidgetItem("Brak"))
            
        report_file = d.get("csv_path", folder if folder else "Brak")
        self.table_results.setItem(idx, 3, QTableWidgetItem(str(report_file)))
            
        self.per_file_summaries[idx] = self._make_summary_block(idx, d)
        verdict = d.get("verdict", "UNKNOWN")
        score   = d.get("final_score")
        if d.get("no_signal"):
            self.append_log(f"< [{idx + 1}/{len(self.files)}] BRAK DANYCH z modeli. Raport: {folder}")
        else:
            try:
                score_str = f"{float(score):.2f}%" if score is not None else "N/A"
            except Exception:
                score_str = "N/A"
            self.append_log(f"< [{idx + 1}/{len(self.files)}] DONE: {verdict} ({score_str}). Raport: {folder}")

    @pyqtSlot()
    def on_all_done(self) -> None:
        self.append_log("> Analiza zakończona.")
        for btn in (self.btn_pick_files, self.btn_pick_folder):
            btn.setEnabled(True)
        self.btn_start.setEnabled(len(self.files) > 0)
        self.btn_stop.setEnabled(False)
        self.worker = None


# ============================ entry point ============================

def run() -> None:
    # PyQt6: HiDPI scaling is always enabled — no QApplication attributes needed
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()

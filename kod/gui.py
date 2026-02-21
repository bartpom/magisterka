# gui.py  (PyQt6 + Modern QSS) - STRICLY WATERMARK DETECTOR
import os
import sys
from typing import Optional, Any

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QCheckBox, QGroupBox,
    QProgressBar, QTextEdit, QFileDialog, QMessageBox, QAbstractItemView,
    QTableWidget, QTableWidgetItem, QHeaderView, QDoubleSpinBox, QSpinBox,
    QFormLayout
)

try:
    import ocr_detector  # type: ignore[import]
except Exception as _e:
    ocr_detector = None  # type: ignore[assignment]
    print(f"[OCR] Moduł ocr_detector niedostępny: {_e}")

SUPPORTED_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
}

# ============================ QSS Themes ============================

_DARK_QSS = """
QMainWindow, QWidget {
    background-color: #1e1e2e; color: #cdd6f4;
    font-family: "Segoe UI", "Inter", sans-serif; font-size: 13px;
}
QPushButton {
    background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 6px;
    padding: 5px 14px; min-height: 28px;
}
QPushButton:hover  { background-color: #45475a; border-color: #89b4fa; }
QPushButton:pressed { background-color: #585b70; }
QPushButton:disabled { background-color: #2a2a3d; color: #6c7086; border-color: #313244; }
QPushButton#btn_start { background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; border: none; }
QPushButton#btn_start:hover { background-color: #89d18a; }
QPushButton#btn_start:disabled { background-color: #2a3b2a; color: #4a5a4a; }
QPushButton#btn_stop { background-color: #f38ba8; color: #1e1e2e; font-weight: bold; border: none; }
QPushButton#btn_stop:hover { background-color: #e07090; }
QPushButton#btn_stop:disabled { background-color: #3b2a2a; color: #5a4a4a; }
QGroupBox {
    border: 1px solid #45475a; border-radius: 6px;
    margin-top: 8px; padding: 4px; color: #89b4fa; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; }
QCheckBox { color: #cdd6f4; spacing: 6px; }
QTableWidget {
    background-color: #181825; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 6px; gridline-color: #313244;
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
    font-family: "Consolas", "JetBrains Mono", "Courier New", monospace; font-size: 12px;
}
QProgressBar {
    background-color: #313244; border: 1px solid #45475a;
    border-radius: 5px; text-align: center; color: #cdd6f4; min-height: 20px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #89b4fa,stop:1 #a6e3a1); border-radius: 4px;
}
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
QPushButton:hover { background-color: #dce0e8; border-color: #1e66f5; }
QPushButton:pressed { background-color: #ccd0da; }
QPushButton:disabled { background-color: #e6e9ef; color: #9ca0b0; border-color: #ccd0da; }
QPushButton#btn_start { background-color: #40a02b; color: #eff1f5; font-weight: bold; border: none; }
QPushButton#btn_start:hover { background-color: #379128; }
QPushButton#btn_start:disabled { background-color: #c8e6c0; color: #9abf93; }
QPushButton#btn_stop { background-color: #d20f39; color: #eff1f5; font-weight: bold; border: none; }
QPushButton#btn_stop:hover { background-color: #b50e33; }
QPushButton#btn_stop:disabled { background-color: #f5b8c5; color: #c08090; }
QGroupBox {
    border: 1px solid #bcc0cc; border-radius: 6px;
    margin-top: 8px; padding: 4px; color: #1e66f5; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; }
QCheckBox { color: #4c4f69; spacing: 6px; }
QTableWidget {
    background-color: #dce0e8; color: #4c4f69;
    border: 1px solid #bcc0cc; border-radius: 6px; gridline-color: #bcc0cc;
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
    font-family: "Consolas", "JetBrains Mono", "Courier New", monospace; font-size: 12px;
}
QProgressBar {
    background-color: #dce0e8; border: 1px solid #bcc0cc;
    border-radius: 5px; text-align: center; color: #4c4f69; min-height: 20px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1e66f5,stop:1 #40a02b); border-radius: 4px;
}
QSplitter::handle { background-color: #bcc0cc; }
QStatusBar { background-color: #dce0e8; color: #9ca0b0; border-top: 1px solid #bcc0cc; }
"""

# ============================ helpers ============================

def is_supported_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTS


# ============================ Worker ============================

class WatermarkWorker(QtCore.QThread):
    progress = pyqtSignal(int, int)
    file_started = pyqtSignal(int, str)
    file_finished = pyqtSignal(int, dict)
    log_line = pyqtSignal(str)
    all_done = pyqtSignal()

    def __init__(
        self,
        files: list[str],
        confidence: float,
        sample_rate: int,
        parent=None,
    ):
        super().__init__(parent)
        self._files = files
        self._confidence = confidence
        self._sample_rate = sample_rate
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        if ocr_detector is None:
            self.log_line.emit("[BŁĄD] Moduł ocr_detector nie został poprawnie załadowany.")
            self.all_done.emit()
            return

        for idx, path in enumerate(self._files):
            if self._stop:
                break
            
            fname = os.path.basename(path)
            self.file_started.emit(idx, fname)
            self.log_line.emit(f"[{idx+1}/{len(self._files)}] Rozpoczynam analizę: {fname} (Conf: {self._confidence}, Sample: {self._sample_rate})")

            def cb(curr, tot):
                self.progress.emit(int(curr), int(tot))

            try:
                # API scan_for_watermarks: (media_path, check_stop, progress_callback, confidence, sample_rate) -> dict
                res = ocr_detector.scan_for_watermarks(
                    path,
                    check_stop=lambda: self._stop,
                    progress_callback=cb,
                    confidence=self._confidence,
                    sample_rate=self._sample_rate
                )
                
                details = res if isinstance(res, dict) else {}
                details["full_path"] = os.path.abspath(path)
                
            except Exception as e:
                self.log_line.emit(f"[BŁĄD] {fname}: {e}")
                details = {"status": "ERROR", "full_path": os.path.abspath(path), "error": str(e)}

            self.file_finished.emit(idx, details)
            
        self.all_done.emit()


# ============================ GUI ============================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Watermark Detector (PyQt6)")
        self.resize(1000, 680)

        self.worker: Optional[WatermarkWorker] = None
        self.files: list[str] = []
        self.files_set: set[str] = set()
        self.report_paths: dict[int, str] = {}
        self.current_run_dir: Optional[str] = None

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

        self.grp_opts = QGroupBox("Parametry OCR (Watermark)")
        opts_lay = QHBoxLayout(self.grp_opts)
        opts_lay.setContentsMargins(8, 4, 8, 4)
        
        param_lay = QFormLayout()
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.1, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.60)
        self.spin_conf.setToolTip("Minimalna pewność OCR")
        
        self.spin_sample = QSpinBox()
        self.spin_sample.setRange(1, 300)
        self.spin_sample.setValue(30)
        self.spin_sample.setToolTip("Odstęp próbkowania klatek w wideo (co X klatkę)")
        
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

        self.btn_open_folder = QPushButton("📂 Open Output Folder")
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        self.btn_open_folder.setEnabled(False)
        bottom.addWidget(self.btn_open_folder)

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

    # -------------------- Actions --------------------

    def pick_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Wybierz pliki do analizy",
            "",
            "Media (*.mp4 *.mov *.avi *.mkv *.webm *.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;Wszystkie pliki (*.*)",
        )
        if paths:
            self._add_files(paths)

    def pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Wybierz folder z nagraniami/obrazami",
            "",
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
            
        for btn in (self.btn_start, self.btn_pick_files, self.btn_pick_folder):
            btn.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progressBar.setValue(0)
        
        conf_val     = self.spin_conf.value()
        sample_val   = self.spin_sample.value()
        
        self.worker = WatermarkWorker(self.files, conf_val, sample_val, parent=self)
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
            QMessageBox.information(self, "Brak Outputu", "Najpierw przeprowadź analizę.")
            return
        
        # Otwiera folder nadrzędny raportów watermarków
        base_reports = os.path.dirname(self.current_run_dir)
        path_to_open = os.path.abspath(base_reports)
        try:
            if sys.platform.startswith("win"):
                os.startfile(path_to_open)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                QtCore.QProcess.startDetached("open",     [path_to_open])
            else:
                QtCore.QProcess.startDetached("xdg-open", [path_to_open])
        except Exception as e:
            QMessageBox.warning(self, "Błąd", f"Nie udało się otworzyć: {e}")

    # -------------------- Slots --------------------

    @pyqtSlot(int, int)
    def on_progress(self, curr: int, tot: int) -> None:
        if tot > 0:
            self.progressBar.setValue(max(0, min(100, int(curr * 100 / max(1, tot)))))

    @pyqtSlot(int, str)
    def on_file_started(self, idx: int, name: str) -> None:
        self.progressBar.setValue(0)

    @pyqtSlot(int, dict)
    def on_file_finished(self, idx: int, details: dict) -> None:
        folder = details.get("watermark_folder")
        if folder:
            self.report_paths[idx] = folder
            self.current_run_dir = folder # Zapisujemy ostatni folder sesji
            self.btn_open_folder.setEnabled(True)
            
        count = details.get("watermark_count")
        if count is not None and count > 0:
            types = ", ".join(details.get("watermark_types", []))
            self.table_results.setItem(idx, 2, QTableWidgetItem(f"{count} ({types})"))
        else:
            self.table_results.setItem(idx, 2, QTableWidgetItem("Brak detekcji"))
            
        report_file = details.get("csv_path", folder if folder else "Brak")
        self.table_results.setItem(idx, 3, QTableWidgetItem(str(report_file)))
        
        self.append_log(f"   ➔ Plik zakończony. Znaleziono {count or 0} watermarków.")

    @pyqtSlot()
    def on_all_done(self) -> None:
        self.append_log("> Analiza wszystkich plików zakończona.")
        for btn in (self.btn_pick_files, self.btn_pick_folder):
            btn.setEnabled(True)
        self.btn_start.setEnabled(len(self.files) > 0)
        self.btn_stop.setEnabled(False)
        self.worker = None


# ============================ entry point ============================

def run() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()

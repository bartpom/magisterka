# gui.py  (PyQt6 + Modern QSS) - WATERMARK DETECTOR + PREVIEW
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QSettings
from PyQt6.QtGui import QTextCursor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QGroupBox,
    QProgressBar, QTextEdit, QFileDialog, QMessageBox, QAbstractItemView,
    QTableWidget, QTableWidgetItem, QHeaderView, QDoubleSpinBox, QSpinBox,
    QFormLayout, QLabel, QLineEdit
)

import config

try:
    import ocr_detector  # type: ignore[import]
except Exception as _e:
    ocr_detector = None  # type: ignore[assignment]
    print(f"[OCR] Moduł ocr_detector niedostępny: {_e}")

try:
    import c2pa_detector  # type: ignore[import]
except Exception as _e2:
    c2pa_detector = None  # type: ignore[assignment]
    print(f"[C2PA] Moduł c2pa_detector niedostępny: {_e2}")

SUPPORTED_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
}

COL_FILE   = 0
COL_TYPE   = 1
COL_STATUS = 2
COL_PCT    = 3
COL_C2PA   = 4
COL_CSV    = 5

# Minimalne szerokosci kolumn — kolumna CSV nigdy nie moze zniknac
_COL_MIN_WIDTHS = [80, 45, 90, 100, 50, 80]

_SETTINGS_ORG  = "WatermarkDetector"
_SETTINGS_APP  = "GUI"
_KEY_MAIN_SPLIT  = "splitter/main4"
_KEY_LEFT_SPLIT  = "splitter/left4"
_KEY_TABLE_COLS  = "table/cols"

# ======================== QSS Themes ========================

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
QLineEdit {
    background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 4px; padding: 4px;
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
QSplitter::handle:horizontal {
    width: 14px;
    background-color: #313244;
    border-left: 1px solid #45475a;
    border-right: 1px solid #45475a;
    image: url(none);
}
QSplitter::handle:horizontal:hover {
    background-color: #45475a;
    border-left: 1px solid #89b4fa;
    border-right: 1px solid #89b4fa;
}
QSplitter::handle:vertical {
    height: 8px;
    background-color: #313244;
    border-top: 1px solid #45475a;
    border-bottom: 1px solid #45475a;
}
QSplitter::handle:vertical:hover {
    background-color: #45475a;
}
QStatusBar { background-color: #181825; color: #6c7086; border-top: 1px solid #45475a; }
QLabel#preview_label, QLabel#zoom_label {
    background-color: #11111b; border: 1px solid #45475a; border-radius: 6px;
}
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
QLineEdit {
    background-color: #e6e9ef; color: #4c4f69;
    border: 1px solid #bcc0cc; border-radius: 4px; padding: 4px;
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
QSplitter::handle:horizontal {
    width: 14px;
    background-color: #dce0e8;
    border-left: 1px solid #bcc0cc;
    border-right: 1px solid #bcc0cc;
}
QSplitter::handle:horizontal:hover {
    background-color: #c0c4d0;
    border-left: 1px solid #1e66f5;
    border-right: 1px solid #1e66f5;
}
QSplitter::handle:vertical {
    height: 8px;
    background-color: #dce0e8;
    border-top: 1px solid #bcc0cc;
    border-bottom: 1px solid #bcc0cc;
}
QSplitter::handle:vertical:hover {
    background-color: #c0c4d0;
}
QStatusBar { background-color: #dce0e8; color: #9ca0b0; border-top: 1px solid #bcc0cc; }
QLabel#preview_label, QLabel#zoom_label {
    background-color: #e6e9ef; border: 1px solid #bcc0cc; border-radius: 6px;
}
"""

# ======================== Toggle Switch ========================

class ToggleSwitch(QtWidgets.QAbstractButton):
    def __init__(self, label: str = "", parent=None):
        super().__init__(parent)
        self._label = label
        self.setCheckable(True)
        self.setChecked(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)

    def sizeHint(self):
        fm = QtGui.QFontMetrics(self.font())
        text_w = fm.horizontalAdvance(self._label) + 8 if self._label else 0
        return QtCore.QSize(52 + text_w, 26)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        track_color = QtGui.QColor("#a6e3a1") if self.isChecked() else QtGui.QColor("#45475a")
        p.setBrush(track_color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 3, 46, 20, 10, 10)
        thumb_x = 26 if self.isChecked() else 2
        p.setBrush(QtGui.QColor("#1e1e2e") if self.isChecked() else QtGui.QColor("#cdd6f4"))
        p.drawEllipse(thumb_x, 5, 18, 16)
        if self._label:
            p.setPen(QtGui.QColor("#cdd6f4"))
            p.setFont(self.font())
            p.drawText(52, 0, self.width() - 52, 26,
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._label)
        p.end()


# ======================== SplitterHandle z ikona ========================

class GripSplitterHandle(QtWidgets.QSplitterHandle):
    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self.setCursor(Qt.CursorShape.SizeHorCursor)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        dot_color = QtGui.QColor("#89b4fa")
        p.setBrush(dot_color)
        p.setPen(Qt.PenStyle.NoPen)
        cx = self.width() // 2
        cy = self.height() // 2
        r = 3
        gap = 9
        for dy in (-gap, 0, gap):
            p.drawEllipse(cx - r, cy + dy - r, r * 2, r * 2)
        p.end()


class GripSplitter(QSplitter):
    def createHandle(self):
        if self.orientation() == Qt.Orientation.Horizontal:
            return GripSplitterHandle(self.orientation(), self)
        return super().createHandle()


# ======================== Drop Overlay ========================

class DropOverlay(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.hide()

    def resizeEvent(self, event):
        self.setGeometry(self.parent().rect())  # type: ignore

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(30, 30, 46, 200))
        pen = QtGui.QPen(QtGui.QColor("#89b4fa"), 4, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(self.rect().adjusted(20, 20, -20, -20), 18, 18)
        p.setPen(QtGui.QColor("#cdd6f4"))
        p.setFont(QtGui.QFont("Segoe UI", 28, QtGui.QFont.Weight.Bold))
        p.drawText(self.rect().adjusted(0, -60, 0, 0), Qt.AlignmentFlag.AlignCenter, "📂")
        p.setFont(QtGui.QFont("Segoe UI", 16, QtGui.QFont.Weight.Bold))
        p.setPen(QtGui.QColor("#89b4fa"))
        p.drawText(self.rect().adjusted(0, 40, 0, 0), Qt.AlignmentFlag.AlignCenter, "Upuść pliki lub folder tutaj")
        p.setFont(QtGui.QFont("Segoe UI", 10))
        p.setPen(QtGui.QColor("#6c7086"))
        p.drawText(self.rect().adjusted(0, 80, 0, 0), Qt.AlignmentFlag.AlignCenter,
                   "obsługiwane: mp4 mov avi mkv webm jpg png bmp")
        p.end()


# ======================== helpers ========================

def is_supported_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTS


def collect_paths_from_urls(urls) -> list:
    paths = []
    for url in urls:
        local = url.toLocalFile()
        if os.path.isdir(local):
            for root, _, files in os.walk(local):
                for f in files:
                    paths.append(os.path.join(root, f))
        else:
            paths.append(local)
    return paths


def _fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _get_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(os.path.abspath(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    return count


def _safe_crop_for_zoom(
    frame_bgr: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    padding: int = 40
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    bx1, by1 = max(0, x1), max(0, y1)
    bx2, by2 = min(w - 1, x2), min(h - 1, y2)
    cx1, cy1 = max(0, bx1 - padding), max(0, by1 - padding)
    cx2, cy2 = min(w, bx2 + padding), min(h, by2 + padding)
    if bx2 + padding > w and cx1 > 0:
        cx1 -= min(cx1, (bx2 + padding) - w)
        cx2 = min(w, cx2)
    if by2 + padding > h and cy1 > 0:
        cy1 -= min(cy1, (by2 + padding) - h)
        cy2 = min(h, cy2)
    if bx1 - padding < 0 and cx2 < w:
        cx2 += min(w - cx2, padding - bx1)
    if by1 - padding < 0 and cy2 < h:
        cy2 += min(h - cy2, padding - by1)
    cx1, cy1, cx2, cy2 = max(0, cx1), max(0, cy1), min(w, cx2), min(h, cy2)
    if cx2 <= cx1 or cy2 <= cy1:
        return frame_bgr
    return frame_bgr[cy1:cy2, cx1:cx2]


def _fill_zoom_label(crop_bgr: np.ndarray, label_w: int, label_h: int) -> QPixmap:
    ch, cw = crop_bgr.shape[:2]
    if cw == 0 or ch == 0 or label_w == 0 or label_h == 0:
        return QPixmap()
    scale = max(label_w / cw, label_h / ch)
    new_w, new_h = int(cw * scale), int(ch * scale)
    resized = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    off_x, off_y = (new_w - label_w) // 2, (new_h - label_h) // 2
    cropped = resized[off_y:off_y + label_h, off_x:off_x + label_w]
    if cropped.shape[0] != label_h or cropped.shape[1] != label_w:
        cropped = cv2.resize(cropped, (label_w, label_h), interpolation=cv2.INTER_LINEAR)
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    qt_img = QImage(cropped_rgb.data, label_w, label_h, 3 * label_w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qt_img)


# ======================== AspectRatioWidget (16:9 container) ========================

class AspectRatioWidget(QWidget):
    def __init__(self, inner: QLabel, aspect_w: int = 16, aspect_h: int = 9, parent=None):
        super().__init__(parent)
        self._inner = inner
        self._aw = aspect_w
        self._ah = aspect_h
        inner.setParent(self)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Ignored,
        )

    def sizeHint(self):
        return QtCore.QSize(1, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        avail_w, avail_h = self.width(), self.height()
        if avail_w <= 0 or avail_h <= 0:
            return
        target_w = avail_w
        target_h = avail_w * self._ah // self._aw
        if target_h > avail_h:
            target_h = avail_h
            target_w = avail_h * self._aw // self._ah
        off_x = (avail_w - target_w) // 2
        off_y = (avail_h - target_h) // 2
        self._inner.setGeometry(off_x, off_y, target_w, target_h)


# ======================== Worker ========================

class WatermarkWorker(QtCore.QThread):
    progress       = pyqtSignal(int, int)
    eta_update     = pyqtSignal(str)
    file_started   = pyqtSignal(int, str, int)   # global_idx, name, total_frames
    file_finished  = pyqtSignal(int, dict)        # global_idx, details
    log_line       = pyqtSignal(str)
    frame_detected = pyqtSignal(np.ndarray, list)
    all_done       = pyqtSignal()

    def __init__(self, files, row_offset, confidence, sample_rate, output_dir, detailed_scan, parent=None):
        super().__init__(parent)
        self._files         = list(files)
        self._row_offset    = row_offset   # indeks pierwszego nowego wiersza w tabeli
        self._confidence    = confidence
        self._sample_rate   = max(1, int(sample_rate)) if sample_rate else 1
        self._output_dir    = output_dir
        self._detailed_scan = detailed_scan
        self._stop          = False

    def stop(self):
        self._stop = True

    def run(self):
        if ocr_detector is None:
            self.log_line.emit("[BŁĄD] Moduł ocr_detector nie został poprawnie załadowany.")
            self.all_done.emit()
            return

        engine, err = ocr_detector.warmup_reader(log_fn=self.log_line.emit)
        if err or engine is None:
            self.log_line.emit(f"[BŁĄD KRYTYCZNY] OCR nie mógł się zainicjalizować: {err}")
            self.log_line.emit("Sprawdź czy easyocr lub paddlepaddle jest zainstalowane w venv.")
            self.all_done.emit()
            return

        self.log_line.emit(f"[OCR] Engine: {engine} — gotowy.")

        original_base = getattr(config, "REPORTS_BASE_DIR", "reports")
        if self._output_dir:
            setattr(config, "REPORTS_BASE_DIR", self._output_dir)

        all_frame_counts = [_get_frame_count(p) for p in self._files]
        frame_times: list[float] = []

        try:
            for local_idx, path in enumerate(self._files):
                if self._stop:
                    break

                global_idx = self._row_offset + local_idx
                fname = os.path.basename(path)
                total_frames = all_frame_counts[local_idx]

                self.file_started.emit(global_idx, fname, total_frames)
                self.log_line.emit(
                    f"[{local_idx+1}/{len(self._files)}] Rozpoczynam analizę: {fname} "
                    f"(Conf: {self._confidence}, Sample: {self._sample_rate}, "
                    f"Detailed: {self._detailed_scan})"
                )

                sampled = max(1, total_frames // self._sample_rate) if total_frames > 0 else 1
                frames_done = [0]
                file_start_t = time.monotonic()

                remaining_frames_after = sum(
                    max(1, all_frame_counts[j] // self._sample_rate)
                    for j in range(local_idx + 1, len(self._files))
                    if all_frame_counts[j] > 0
                )

                def cb(curr, tot,
                       _sampled=sampled,
                       _remaining_after=remaining_frames_after,
                       _ft=frame_times,
                       _fst=file_start_t,
                       _fd=frames_done):
                    _fd[0] += 1
                    elapsed = time.monotonic() - _fst
                    if _fd[0] > 0:
                        avg = elapsed / _fd[0]
                        _ft.append(avg)
                        if len(_ft) > 20:
                            _ft.pop(0)
                        avg_global = sum(_ft) / len(_ft)
                        remaining_this = max(0, _sampled - _fd[0]) * avg_global
                        eta_sec = remaining_this + _remaining_after * avg_global
                        self.eta_update.emit(f"ETA: {_fmt_eta(eta_sec)}")
                    self.progress.emit(int(curr), int(tot))

                def preview_cb(*args):
                    if len(args) == 1:
                        self.frame_detected.emit(args[0], [])
                    elif len(args) >= 2:
                        self.frame_detected.emit(args[0], args[1])

                c2pa_result: dict = {}
                if c2pa_detector is not None:
                    try:
                        c2pa_result = c2pa_detector.detect_c2pa(path) or {}
                    except Exception as _ce:
                        c2pa_result = {"error": str(_ce)}

                try:
                    res = ocr_detector.scan_for_watermarks(
                        path,
                        check_stop=lambda: self._stop,
                        progress_callback=cb,
                        confidence=self._confidence,
                        sample_rate=self._sample_rate,
                        detailed_scan=self._detailed_scan,
                        preview_callback=preview_cb
                    )
                    details = res if isinstance(res, dict) else {}
                    details["full_path"]    = os.path.abspath(path)
                    details["total_frames"] = total_frames
                    details["c2pa"]         = c2pa_result
                except Exception as e:
                    self.log_line.emit(f"[BŁĄD] {fname}: {e}")
                    details = {
                        "status": "ERROR",
                        "full_path": os.path.abspath(path),
                        "error": str(e),
                        "total_frames": total_frames,
                        "c2pa": c2pa_result,
                    }

                self.file_finished.emit(global_idx, details)
        finally:
            setattr(config, "REPORTS_BASE_DIR", original_base)

        self.eta_update.emit("")
        self.all_done.emit()


# ======================== GUI ========================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Watermark Detector (PyQt6) z Podglądem")
        self.resize(1400, 800)
        self.setAcceptDrops(True)

        self._settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._splitters_initialized = False

        self.worker: Optional[WatermarkWorker] = None
        self.files: list[str] = []           # wszystkie pliki (stare + nowe)
        self.files_set: set[str] = set()     # deduplikacja
        self._analyzed_set: set[str] = set() # juz przeanalizowane
        self.report_paths: dict[int, str] = {}
        self.current_run_dir: Optional[str] = None
        self._file_frame_data: dict[int, dict] = {}

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(0)

        self.main_splitter = GripSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(16)
        root_layout.addWidget(self.main_splitter)

        # ---- LEFT PANEL ----
        left_panel = QWidget()
        left_panel.setMinimumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(6)
        left_layout.addLayout(top)

        self.btn_pick_files = QPushButton("📂 Dodaj pliki…")
        self.btn_pick_files.setToolTip("Dodaj pojedyncze pliki wideo lub obrazów do kolejki analizy")
        self.btn_pick_files.clicked.connect(self.pick_files)
        top.addWidget(self.btn_pick_files)

        self.btn_pick_folder = QPushButton("📁 Dodaj folder…")
        self.btn_pick_folder.setToolTip("Dodaj cały folder — wszystkie obsługiwane pliki zostaną dodane rekurencyjnie")
        self.btn_pick_folder.clicked.connect(self.pick_folder)
        top.addWidget(self.btn_pick_folder)

        self.grp_opts = QGroupBox("Parametry OCR (Watermark)")
        opts_lay = QVBoxLayout(self.grp_opts)
        opts_lay.setContentsMargins(8, 4, 8, 4)
        param_lay = QFormLayout()

        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.1, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.60)
        self.spin_conf.setToolTip(
            "OCR Confidence — minimalna pewność rozpoznania tekstu (0.1–1.0).\n"
            "Niższa wartość = więcej detekcji (ale też fałszywe alarmy).\n"
            "Wyższa wartość = tylko bardzo pewne wyniki. Zalecane: 0.55–0.70."
        )

        self.spin_sample = QSpinBox()
        self.spin_sample.setRange(1, 300)
        self.spin_sample.setValue(30)
        self.spin_sample.setToolTip(
            "Sample Rate — co ile klatek wideo jest analizowana jedna klatka.\n"
            "Wartość 30 = analizuj co 30. klatkę (1 raz na sekundę przy 30fps).\n"
            "Mniejsza wartość = dokładniej, ale wolniej. Zalecane: 15–30."
        )

        lbl_conf = QLabel("OCR Confidence:")
        lbl_conf.setToolTip(self.spin_conf.toolTip())
        lbl_sample = QLabel("Sample rate:")
        lbl_sample.setToolTip(self.spin_sample.toolTip())
        param_lay.addRow(lbl_conf, self.spin_conf)
        param_lay.addRow(lbl_sample, self.spin_sample)

        self.toggle_detailed = ToggleSwitch("Szczegółowa analiza (Dwufazowa)")
        self.toggle_detailed.setToolTip(
            "Tryb dwufazowy — po szybkim skanie uruchamia zaawansowane filtry obrazu\n"
            "(CLAHE, Top-Hat, gamma, odwrócone kolory) na klatkach bez detekcji.\n"
            "Wolniejszy, ale wykrywa trudne watermarki (np. białe na jasnym tle)."
        )
        toggle_row = QHBoxLayout()
        toggle_row.addWidget(self.toggle_detailed)
        toggle_row.addStretch()
        param_lay.addRow("", toggle_row)

        dir_lay = QHBoxLayout()
        self.txt_output_dir = QLineEdit()
        self.txt_output_dir.setPlaceholderText("Domyślny folder z config.py")
        self.txt_output_dir.setReadOnly(True)
        self.btn_pick_out_dir = QPushButton("Zmień folder zapisu...")
        self.btn_pick_out_dir.clicked.connect(self.pick_output_dir)
        dir_lay.addWidget(self.txt_output_dir)
        dir_lay.addWidget(self.btn_pick_out_dir)

        opts_lay.addLayout(param_lay)
        opts_lay.addLayout(dir_lay)
        top.addWidget(self.grp_opts, 1)

        self.toggle_dark = ToggleSwitch("Ciemny")
        self.toggle_dark.setChecked(True)
        self.toggle_dark.toggled.connect(self._apply_theme)
        top.addWidget(self.toggle_dark)

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

        self.left_splitter = QSplitter(Qt.Orientation.Vertical)
        self.left_splitter.setChildrenCollapsible(False)
        left_layout.addWidget(self.left_splitter, 1)

        self.table_results = QTableWidget()
        self.table_results.setMinimumSize(0, 0)
        self.table_results.setColumnCount(6)
        self.table_results.setHorizontalHeaderLabels([
            "Plik", "Typ", "Status AI", "% AI w wideo", "C2PA", "Raport CSV"
        ])
        hh = self.table_results.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        # Ostatnia kolumna NIE rozciaga sie automatycznie — ma swoj minWidth
        hh.setStretchLastSection(False)
        _default_col_widths = [200, 55, 130, 160, 60, 200]
        for col, w in enumerate(_default_col_widths):
            self.table_results.setColumnWidth(col, w)
            self.table_results.horizontalHeader().setMinimumSectionSize(_COL_MIN_WIDTHS[col])
        # Wymuszamy minimalne szerokosci per kolumna przez delegate
        hh.sectionResized.connect(self._enforce_col_min_widths)
        self.table_results.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_results.setAlternatingRowColors(True)
        self.table_results.setToolTip(
            "Dwuklik na wierszu — otworzy folder z raportem dla tego pliku.\n"
            "Dwuklik na wierszu z wynikiem — wyświetla szczegóły detekcji."
        )
        self.table_results.cellDoubleClicked.connect(self._on_row_double_clicked)
        self.left_splitter.addWidget(self.table_results)

        self.logView = QTextEdit()
        self.logView.setReadOnly(True)
        self.logView.setMinimumSize(0, 0)
        self.left_splitter.addWidget(self.logView)

        self.left_splitter.setStretchFactor(0, 3)
        self.left_splitter.setStretchFactor(1, 1)

        bottom = QHBoxLayout()
        bottom.setSpacing(6)
        left_layout.addLayout(bottom)

        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.setFormat("%p%")
        bottom.addWidget(self.progressBar, 1)

        self.lbl_eta = QLabel("")
        self.lbl_eta.setMinimumWidth(90)
        self.lbl_eta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_eta.setStyleSheet("color: #89b4fa; font-weight: bold;")
        bottom.addWidget(self.lbl_eta)

        self.btn_open_folder = QPushButton("📂 Otwórz folder wyników")
        self.btn_open_folder.setToolTip(
            "Otwiera folder z ostatnim raportem.\n"
            "Możesz też dwukliknąć wiersz w tabeli, żeby otworzyć folder konkretnego pliku."
        )
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        self.btn_open_folder.setEnabled(False)
        bottom.addWidget(self.btn_open_folder)

        self.main_splitter.addWidget(left_panel)

        # ---- RIGHT PANEL (podglady) ----
        right_panel = QWidget()
        right_panel.setMinimumWidth(200)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(4)

        lbl_title = QLabel("<b>Klatka z kamery:</b>")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setMinimumSize(0, 0)
        right_layout.addWidget(lbl_title)

        self.lbl_preview = QLabel("Brak podglądu")
        self.lbl_preview.setObjectName("preview_label")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setMinimumSize(0, 0)
        self._preview_ar = AspectRatioWidget(self.lbl_preview, 16, 9)
        right_layout.addWidget(self._preview_ar, 3)

        lbl_zoom_title = QLabel("<b>Przybliżenie detekcji (Zoom):</b>")
        lbl_zoom_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_zoom_title.setMinimumSize(0, 0)
        right_layout.addWidget(lbl_zoom_title)

        self.lbl_zoom = QLabel("Brak detekcji")
        self.lbl_zoom.setObjectName("zoom_label")
        self.lbl_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_zoom.setMinimumSize(0, 0)
        self.lbl_zoom.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Ignored,
        )
        right_layout.addWidget(self.lbl_zoom, 2)

        self.main_splitter.addWidget(right_panel)

        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 1)

        self._drop_overlay = DropOverlay(central)
        self._drop_overlay.setGeometry(central.rect())
        self._drop_overlay.raise_()

        self.status = self.statusBar()
        self._apply_theme(True)

        if ocr_detector is not None:
            QtCore.QTimer.singleShot(500, self._warmup_ocr)

    # ----------------------------------------------------------------
    # Wymuszanie minimalnych szerokosci kolumn przy resize
    # ----------------------------------------------------------------

    @pyqtSlot(int, int, int)
    def _enforce_col_min_widths(self, col: int, old_size: int, new_size: int) -> None:
        """Pilnuje, żeby żadna kolumna nie była węższa niż _COL_MIN_WIDTHS[col]."""
        min_w = _COL_MIN_WIDTHS[col] if col < len(_COL_MIN_WIDTHS) else 40
        if new_size < min_w:
            # Blokujemy sygnał żeby nie wywolac petli
            hh = self.table_results.horizontalHeader()
            hh.blockSignals(True)
            self.table_results.setColumnWidth(col, min_w)
            hh.blockSignals(False)

    # ----------------------------------------------------------------
    # showEvent: dopiero tu okno ma prawdziwa geometrie
    # ----------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        if not self._splitters_initialized:
            self._splitters_initialized = True
            self._restore_splitters()

    # ----------------------------------------------------------------
    # Zapis / odczyt pozycji splitterow i szerokosci kolumn
    # ----------------------------------------------------------------

    def _restore_splitters(self):
        total_w = self.main_splitter.width()
        data = self._settings.value(_KEY_MAIN_SPLIT)
        if data is not None:
            try:
                self.main_splitter.restoreState(data)
                sizes = self.main_splitter.sizes()
                if sum(sizes) < 100 or any(s < 10 for s in sizes):
                    raise ValueError("bad sizes")
            except Exception:
                self._settings.remove(_KEY_MAIN_SPLIT)
                half = total_w // 2
                self.main_splitter.setSizes([half, total_w - half])
        else:
            half = total_w // 2
            self.main_splitter.setSizes([half, total_w - half])

        total_h = self.left_splitter.height()
        data2 = self._settings.value(_KEY_LEFT_SPLIT)
        if data2 is not None:
            try:
                self.left_splitter.restoreState(data2)
                sizes2 = self.left_splitter.sizes()
                if sum(sizes2) < 100 or any(s < 10 for s in sizes2):
                    raise ValueError("bad sizes")
            except Exception:
                self._settings.remove(_KEY_LEFT_SPLIT)
                self.left_splitter.setSizes([int(total_h * 0.65), int(total_h * 0.35)])
        else:
            self.left_splitter.setSizes([int(total_h * 0.65), int(total_h * 0.35)])

        col_data = self._settings.value(_KEY_TABLE_COLS)
        if col_data is not None:
            try:
                widths = [int(x) for x in col_data]
                for col, w in enumerate(widths):
                    if col < self.table_results.columnCount():
                        safe_w = max(w, _COL_MIN_WIDTHS[col] if col < len(_COL_MIN_WIDTHS) else 40)
                        self.table_results.setColumnWidth(col, safe_w)
            except Exception:
                self._settings.remove(_KEY_TABLE_COLS)

    def _save_splitters(self):
        self._settings.setValue(_KEY_MAIN_SPLIT, self.main_splitter.saveState())
        self._settings.setValue(_KEY_LEFT_SPLIT, self.left_splitter.saveState())
        widths = [self.table_results.columnWidth(c)
                  for c in range(self.table_results.columnCount())]
        self._settings.setValue(_KEY_TABLE_COLS, widths)
        self._settings.sync()

    def closeEvent(self, event):
        self._save_splitters()
        super().closeEvent(event)

    # ----------------------------------------------------------------

    def _warmup_ocr(self):
        import threading
        def _do():
            engine, err = ocr_detector.warmup_reader()
            if err:
                QtCore.QMetaObject.invokeMethod(
                    self.logView, "append",
                    Qt.ConnectionType.QueuedConnection,
                    QtCore.Q_ARG(str, f"[OCR] BŁĄD inicjalizacji: {err}")
                )
            else:
                QtCore.QMetaObject.invokeMethod(
                    self.logView, "append",
                    Qt.ConnectionType.QueuedConnection,
                    QtCore.Q_ARG(str, f"[OCR] Engine gotowy: {engine}")
                )
        threading.Thread(target=_do, daemon=True).start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cw = self.centralWidget()
        if cw and self._drop_overlay:
            self._drop_overlay.setGeometry(cw.rect())

    def _apply_theme(self, dark: bool) -> None:
        app = QApplication.instance()
        if not app:
            return
        app.setStyle("Fusion")  # type: ignore
        app.setStyleSheet(_DARK_QSS if dark else _LIGHT_QSS)  # type: ignore

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            self._drop_overlay.setGeometry(self.centralWidget().rect())  # type: ignore
            self._drop_overlay.show()
            self._drop_overlay.raise_()
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragLeaveEvent(self, event: QtGui.QDragLeaveEvent):
        self._drop_overlay.hide()
        super().dragLeaveEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent):
        self._drop_overlay.hide()
        if event.mimeData().hasUrls():
            self._add_files(collect_paths_from_urls(event.mimeData().urls()))
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    @pyqtSlot(int, int)
    def _on_row_double_clicked(self, row: int, _col: int) -> None:
        folder = self.report_paths.get(row)
        if not folder or not os.path.isdir(folder):
            self.append_log("> Folder raportu nie jest jeszcze dostępny dla tego pliku.")
            return
        self._open_path(folder)

    def _open_path(self, path: str) -> None:
        try:
            if sys.platform.startswith("win"):
                os.startfile(os.path.abspath(path))  # type: ignore
            elif sys.platform == "darwin":
                QtCore.QProcess.startDetached("open", [path])
            else:
                QtCore.QProcess.startDetached("xdg-open", [path])
        except Exception as e:
            QMessageBox.warning(self, "Błąd", f"Nie udało się otworzyć: {e}")

    def append_log(self, text: str) -> None:
        self.logView.append(text)
        self.logView.moveCursor(QTextCursor.MoveOperation.End)
        self.status.showMessage(text, 4000)

    def _add_files(self, paths: list) -> None:
        added = 0
        for p in paths:
            if not p or not is_supported_file(p):
                continue
            ap = os.path.abspath(p)
            if ap in self.files_set:
                continue
            self.files_set.add(ap)
            self.files.append(ap)

            row = self.table_results.rowCount()
            self.table_results.insertRow(row)

            fname = os.path.basename(ap)
            ext = os.path.splitext(fname)[1].lower()
            file_type = "Video" if ext in {".mp4", ".mkv", ".avi", ".webm", ".mov"} else "Image"

            self.table_results.setItem(row, COL_FILE, QTableWidgetItem(fname))
            self.table_results.setItem(row, COL_TYPE, QTableWidgetItem(file_type))

            si = QTableWidgetItem("⏳ Oczekuje")
            si.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_results.setItem(row, COL_STATUS, si)

            pi = QTableWidgetItem("-")
            pi.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_results.setItem(row, COL_PCT, pi)

            ci = QTableWidgetItem("⏳")
            ci.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            ci.setToolTip("C2PA / Content Credentials — oczekuje na analizę")
            self.table_results.setItem(row, COL_C2PA, ci)

            self.table_results.setItem(row, COL_CSV, QTableWidgetItem("-"))
            added += 1

        if added:
            self.btn_start.setEnabled(True)
            self.append_log(f"> Dodano {added} plik(ów). Razem: {len(self.files)}.")

    @pyqtSlot(np.ndarray, list)
    def set_preview_image(self, frame_bgr: np.ndarray, detections: list) -> None:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qt_img = QImage(frame_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.lbl_preview.setPixmap(
            pixmap.scaled(
                self.lbl_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

        if detections:
            det = detections[0]
            x1, y1, x2, y2 = det.get("bbox", (0, 0, 10, 10))
            crop = _safe_crop_for_zoom(frame_bgr, x1, y1, x2, y2, padding=40)
            if crop.size > 0:
                lw = max(1, self.lbl_zoom.width())
                lh = max(1, self.lbl_zoom.height())
                zoom_pixmap = _fill_zoom_label(crop, lw, lh)
                if not zoom_pixmap.isNull():
                    self.lbl_zoom.setPixmap(zoom_pixmap)
        else:
            self.lbl_zoom.setText("Brak detekcji")

    def pick_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Wybierz folder docelowy dla raportów")
        if folder:
            self.txt_output_dir.setText(os.path.abspath(folder))

    def pick_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Wybierz pliki do analizy", "",
            "Media (*.mp4 *.mov *.avi *.mkv *.webm *.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)"
            ";;Wszystkie pliki (*.*)",
        )
        if paths:
            self._add_files(paths)

    def pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Wybierz folder z nagraniami/obrazami", "",
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
        # BUG FIX: analizuj TYLKO pliki ktore jeszcze nie zostaly przeanalizowane
        pending = [f for f in self.files if f not in self._analyzed_set]
        if not pending:
            QMessageBox.information(
                self, "Brak nowych plików",
                "Wszystkie pliki zostały już przeanalizowane.\n"
                "Dodaj nowe pliki lub folder, żeby uruchomić kolejną analizę."
            )
            return

        # Indeks pierwszego oczekujacego wiersza w tabeli
        row_offset = len(self.files) - len(pending)

        self._file_frame_data = {}
        for btn in (self.btn_start, self.btn_pick_files, self.btn_pick_folder):
            btn.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progressBar.setValue(0)
        self.lbl_eta.setText("")
        self.lbl_preview.setText("Analiza w toku...")
        self.lbl_zoom.setText("Analiza w toku...")

        self.worker = WatermarkWorker(
            pending,
            row_offset,
            self.spin_conf.value(),
            self.spin_sample.value(),
            self.txt_output_dir.text().strip(),
            self.toggle_detailed.isChecked(),
            parent=self
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.eta_update.connect(self.lbl_eta.setText)
        self.worker.file_started.connect(self.on_file_started)
        self.worker.file_finished.connect(self.on_file_finished)
        self.worker.log_line.connect(self.append_log)
        self.worker.frame_detected.connect(self.set_preview_image)
        self.worker.all_done.connect(self.on_all_done)
        self.worker.start()

        # Oznacz jako w trakcie analizy (dodamy do _analyzed_set po zakonczeniu)
        self._current_pending = pending

    def stop_analysis(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.append_log("> Przerywam analizę…")

    def open_output_folder(self) -> None:
        if not self.current_run_dir or not os.path.isdir(self.current_run_dir):
            QMessageBox.information(self, "Brak Outputu", "Najpierw przeprowadź analizę.")
            return
        self._open_path(self.current_run_dir)

    @pyqtSlot(int, int)
    def on_progress(self, curr: int, tot: int) -> None:
        if tot > 0:
            self.progressBar.setValue(max(0, min(100, int(curr * 100 / max(1, tot)))))

    @pyqtSlot(int, str, int)
    def on_file_started(self, idx: int, name: str, total_frames: int) -> None:
        self.progressBar.setValue(0)
        self._file_frame_data[idx] = {"total_frames": total_frames}
        item = self.table_results.item(idx, COL_STATUS)
        if item:
            item.setText("🔍 Analiza...")
        c2pa_item = self.table_results.item(idx, COL_C2PA)
        if c2pa_item:
            c2pa_item.setText("🔍")

    @pyqtSlot(int, dict)
    def on_file_finished(self, idx: int, details: dict) -> None:
        folder = details.get("watermark_folder")
        if folder:
            self.report_paths[idx] = folder
            self.current_run_dir = os.path.dirname(folder)
            self.btn_open_folder.setEnabled(True)

        count = details.get("watermark_count", 0) or 0
        total_frames = details.get("total_frames", 0) or 0
        sampled = max(1, total_frames // max(1, self.spin_sample.value())) if total_frames > 0 else 1

        if count > 0:
            types = ", ".join(details.get("watermark_types", []))
            si = QTableWidgetItem(f"🔴 AI DETECTED\n{types}")
            si.setForeground(QtGui.QBrush(QtGui.QColor("#f38ba8")))
        else:
            si = QTableWidgetItem("✅ AI CLEAR")
            si.setForeground(QtGui.QBrush(QtGui.QColor("#a6e3a1")))
        si.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table_results.setItem(idx, COL_STATUS, si)

        if total_frames > 0 and count > 0:
            ai_pct = min(100.0, count / sampled * 100)
            pi = QTableWidgetItem(f"🔴 {ai_pct:.0f}% AI  |  ✅ {100-ai_pct:.0f}% CLEAR")
            pi.setForeground(QtGui.QBrush(
                QtGui.QColor("#f38ba8") if ai_pct >= 50 else QtGui.QColor("#fab387")
            ))
        else:
            pi = QTableWidgetItem("✅ 100% CLEAR")
            pi.setForeground(QtGui.QBrush(QtGui.QColor("#a6e3a1")))
        pi.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table_results.setItem(idx, COL_PCT, pi)

        c2pa = details.get("c2pa", {})
        if isinstance(c2pa, dict) and c2pa:
            if "error" in c2pa:
                c2pa_text = "⚠️ Błąd"
                c2pa_color = "#fab387"
                c2pa_tip = str(c2pa["error"])
            elif c2pa.get("found") or c2pa.get("has_c2pa"):
                c2pa_text = "✅ C2PA"
                c2pa_color = "#a6e3a1"
                issuer = c2pa.get("issuer") or c2pa.get("producer") or ""
                c2pa_tip = f"Content Credentials znalezione.\nWydawca: {issuer}" if issuer else "Content Credentials znalezione."
            else:
                c2pa_text = "❌ Brak"
                c2pa_color = "#6c7086"
                c2pa_tip = "Brak metadanych C2PA / Content Credentials."
        else:
            if c2pa_detector is None:
                c2pa_text = "— N/A"
                c2pa_color = "#6c7086"
                c2pa_tip = "Moduł c2pa_detector niedostępny."
            else:
                c2pa_text = "❌ Brak"
                c2pa_color = "#6c7086"
                c2pa_tip = "Brak metadanych C2PA."
        ci = QTableWidgetItem(c2pa_text)
        ci.setForeground(QtGui.QBrush(QtGui.QColor(c2pa_color)))
        ci.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        ci.setToolTip(c2pa_tip)
        self.table_results.setItem(idx, COL_C2PA, ci)

        report_file = details.get("csv_path", folder if folder else "Brak")
        self.table_results.setItem(idx, COL_CSV, QTableWidgetItem(str(report_file)))

        self.append_log(f"   ➔ Plik zakończony. Znaleziono {count} watermarków. C2PA: {c2pa_text}")
        self.table_results.resizeRowsToContents()

    @pyqtSlot()
    def on_all_done(self) -> None:
        # Oznacz przetworzone pliki jako przeanalizowane
        if hasattr(self, "_current_pending"):
            self._analyzed_set.update(self._current_pending)
            self._current_pending = []

        self.append_log("> Analiza wszystkich plików zakończona.")
        self.progressBar.setValue(0)
        self.lbl_eta.setText("")
        for btn in (self.btn_pick_files, self.btn_pick_folder):
            btn.setEnabled(True)

        # START aktywny tylko jesli sa nowe (nieprzetworzone) pliki
        pending_left = [f for f in self.files if f not in self._analyzed_set]
        self.btn_start.setEnabled(len(pending_left) > 0)
        self.btn_stop.setEnabled(False)
        self.worker = None


# ======================== entry point ========================

def run() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()

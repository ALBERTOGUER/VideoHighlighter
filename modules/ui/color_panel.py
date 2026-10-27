"""Colour settings: the camera LUT, the grade sliders, "Recommend" and a
before/after preview. In Video Output, and as a section of the before-run
dialog. See ``modules.media.color_grade``.

Recommend and Preview read frames of the first video in the list; that takes
seconds on 4K, so it runs on a worker thread and the window stays live.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QWidget,
)

from modules.media import color_grade as cg
from modules.ui.stabilization_panel import SliderSpin

SLIDERS = (
    # key, label, step, decimals, suffix, tooltip
    ("exposure", "Exposure", 0.05, 2, " EV", "Brightness in stops."),
    ("contrast", "Contrast", 0.01, 2, "", "1 = unchanged; above 1 deepens shadows and brightens highlights."),
    ("saturation", "Saturation", 0.01, 2, "", "0 = black and white, 1 = unchanged."),
    ("temperature", "Temperature", 1, 0, "", "White balance: negative cooler (blue), positive warmer (orange)."),
    ("tint", "Tint", 1, 0, "", "White balance: negative green, positive magenta."),
    ("highlights", "Highlight protection", 0.01, 2, "",
     "Rolls highlights off smoothly towards white instead of clipping them."),
)


class _Job(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn())
        except Exception as e:  # shown to the user, never raised in Qt's thread
            self.failed.emit(str(e))


class ColorPanel(QWidget):
    def __init__(self, settings=None, in_dialog=False, video_provider=None, parent=None):
        super().__init__(parent)
        self._video = video_provider or (lambda: None)
        # What the before-run dialog passes when it builds its own copy.
        self.dialog_kwargs = {"video_provider": self._video}
        self._job = None
        self._then = None   # what to run once the current job's thread is done
        # Own flag, not isRunning(): QThread emits finished while still running.
        self._busy = False
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)

        lut_row = QHBoxLayout()
        self.lut_input = QLineEdit()
        self.lut_input.setPlaceholderText("None — the source colours are kept")
        self.lut_input.setToolTip(
            "Your camera's 3D LUT (.cube), e.g. DJI's D-Log M to Rec.709.\n"
            "Applied to every clip and the best segment; the source is never changed.")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.lut_input.clear)
        lut_row.addWidget(self.lut_input, 1)
        lut_row.addWidget(browse)
        lut_row.addWidget(clear)
        form.addRow("Camera LUT:", lut_row)

        self.sliders = {}
        for key, label, step, decimals, suffix, tip in SLIDERS:
            _field, lo, hi, _neutral = cg.FIELDS[key]
            slider = SliderSpin(lo, hi, step, decimals, suffix)
            slider.setToolTip(tip)
            self.sliders[key] = slider
            form.addRow(f"{label}:", slider)

        buttons = QHBoxLayout()
        self.detect_btn = QPushButton("Detect camera")
        self.detect_btn.setToolTip(
            "Read the first video's metadata: which camera and colour profile it was\n"
            "recorded in, and the LUT that profile needs (DroneReel's camera profiles).\n"
            "A LUT is only set when the profile is confirmed.")
        self.detect_btn.clicked.connect(self.detect)
        buttons.addWidget(self.detect_btn)
        self.recommend_btn = QPushButton("Recommend")
        self.recommend_btn.setToolTip(
            "Measure frames of the first video (after the LUT) and suggest values.\n"
            "A starting point: every suggestion says what it measured.")
        self.recommend_btn.clicked.connect(self.recommend)
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.setToolTip("A frame with the LUT only | with these settings.")
        self.preview_btn.clicked.connect(self.preview)
        reset = QPushButton("Reset")
        reset.setToolTip("Back to the LUT alone, no grade.")
        reset.clicked.connect(self.reset_grade)
        buttons.addWidget(self.recommend_btn)
        buttons.addWidget(self.preview_btn)
        buttons.addWidget(reset)
        buttons.addStretch()
        form.addRow(buttons)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: gray; font-size: 9pt;")
        form.addRow(self.status)
        self.preview_lbl = QLabel()
        self.preview_lbl.setAlignment(Qt.AlignCenter)
        self.preview_lbl.setVisible(False)
        form.addRow(self.preview_lbl)

        self.ask_chk = QCheckBox("Ask before each run")
        self.ask_chk.setToolTip("Show this section when Analyze is pressed.")
        if not in_dialog:
            form.addRow(self.ask_chk)

        self.set_settings(settings or cg.DEFAULTS)

    # ── settings ────────────────────────────────────────────────────────────
    def settings(self) -> dict:
        return cg.resolve_settings({
            "lut_path": self.lut_input.text(),
            "ask": self.ask_chk.isChecked(),
            **{k: s.value() for k, s in self.sliders.items()},
        })

    def set_settings(self, raw):
        s = cg.resolve_settings(raw)
        self.lut_input.setText(s["lut_path"])
        self.ask_chk.setChecked(s["ask"])
        for key, slider in self.sliders.items():
            slider.setValue(s[key])

    def reset_grade(self):
        s = self.settings()
        self.set_settings({"lut_path": s["lut_path"], "ask": s["ask"]})
        self.status.setText("Grade reset: the LUT alone.")

    def _browse(self):
        from modules.media.lut import LUT_SUFFIXES
        start = os.path.dirname(self.lut_input.text().strip())
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a colour LUT", start,
            f"3D LUTs ({' '.join('*' + s for s in LUT_SUFFIXES)});;All files (*)")
        if path:
            self.lut_input.setText(path)

    # ── recommend / preview ─────────────────────────────────────────────────
    def _start(self, what, fn, on_done):
        video = self._video()
        if not video or not os.path.isfile(video):
            self.status.setText("Add a video first: Recommend and Preview read its frames.")
            return
        if self._busy:
            return
        self._busy = True
        for b in (self.detect_btn, self.recommend_btn, self.preview_btn):
            b.setEnabled(False)
        self.status.setText(f"{what} {os.path.basename(video)}…")
        self._job = _Job(lambda: fn(video), self)
        self._job.done.connect(on_done)
        self._job.failed.connect(lambda msg: self.status.setText(f"⚠️ {msg}"))
        self._job.finished.connect(self._job_finished)
        self._job.start()

    def _job_finished(self):
        self._busy = False
        for b in (self.detect_btn, self.recommend_btn, self.preview_btn):
            b.setEnabled(True)
        then, self._then = self._then, None
        if then:
            then()

    def detect(self):
        self._start("Reading", cg.detect, self._detected)

    def _detected(self, result):
        if result["lut"]:
            self.lut_input.setText(result["lut"])
        elif result["profile"] != "unknown" and not result["log"] and self.lut_input.text():
            # A log-to-Rec.709 LUT on footage already in Rec.709 crushes it.
            self.lut_input.clear()
            result["message"] += " The LUT was cleared."
        self.status.setText(result["message"])

    def recommend(self):
        current = self.settings()
        self._start("Measuring", lambda video: cg.recommend(video, current), self._recommended)

    def _recommended(self, result):
        settings, reasons = result
        self.set_settings({**settings, "ask": self.ask_chk.isChecked()})
        self.status.setText("Suggested:\n• " + "\n• ".join(reasons))
        self._then = self.preview   # after this thread ends; one job at a time

    def preview(self):
        current = self.settings()
        self._start("Rendering a preview of", lambda video: cg.preview(video, current),
                    self._show_preview)

    def _show_preview(self, rgb):
        reasons = self.status.text()
        if not reasons.startswith("Suggested"):
            self.status.setText("Left: the LUT only. Right: with these settings.")
        h, w, _ = rgb.shape
        image = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()
        self.preview_lbl.setPixmap(QPixmap.fromImage(image).scaledToWidth(
            min(w, 520), Qt.SmoothTransformation))
        self.preview_lbl.setToolTip("Left: the LUT only. Right: with these settings.")
        self.preview_lbl.setVisible(True)

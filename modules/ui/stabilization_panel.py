"""Gyroflow stabilization settings: the panel in Video Output, also shown as a
section of the before-run dialog (``modules.ui.before_run_dialog``), so a
value means the same thing in either place."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QLabel, QRadioButton, QButtonGroup, QSlider, QSpinBox,
    QVBoxLayout, QWidget,
)

from modules.media.stabilize import (
    CODECS, DEFAULTS, OUTPUT_SIZES, find_gyroflow, is_sandboxed, resolve_settings,
)

MODE_LABELS = (
    ("manual", "Manual", "Your smoothness and low-pass filter for every clip."),
    ("auto_clip", "Automatic per clip",
     "Each clip's gyro decides its smoothness, and a low-pass filter where it vibrates."),
    ("auto_frame", "Automatic per frame",
     "Per clip, plus smoothness keyframed through the clip: more where it shakes, "
     "less where it is still or turning on purpose."),
)


class SliderSpin(QWidget):
    """A slider and a spin box bound to one float value."""

    valueChanged = Signal(float)

    def __init__(self, minimum, maximum, step, decimals, suffix="", parent=None):
        super().__init__(parent)
        self._scale = 10 ** decimals
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(minimum * self._scale), int(maximum * self._scale))
        self.slider.setSingleStep(max(1, int(step * self._scale)))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        if suffix:
            self.spin.setSuffix(suffix)
        self.slider.valueChanged.connect(lambda v: self.spin.setValue(v / self._scale))
        self.spin.valueChanged.connect(self._from_spin)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)

    def _from_spin(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(v * self._scale)))
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)

    def value(self):
        return float(self.spin.value())

    def setValue(self, v):
        self.spin.setValue(float(v))


class StabilizationPanel(QWidget):
    def __init__(self, settings=None, in_dialog=False, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)

        self.enabled_chk = QCheckBox("Stabilize the clips with Gyroflow")
        self.enabled_chk.setToolTip(
            "Uses the gyroscope the camera recorded (DJI, GoPro, Insta360, Sony…).\n"
            "Record with the camera's own stabilization (EIS/RockSteady) off.\n"
            "Runs before the color LUT, so the LUT still sees the log footage.")
        form.addRow(self.enabled_chk)

        exe = find_gyroflow()
        if not exe:
            status = "⚠️ Gyroflow not found — install it from gyroflow.xyz or the App Store."
        elif is_sandboxed(exe):
            status = ("Gyroflow (App Store build) found. Footage must be on the startup "
                      "disk: it is linked into Gyroflow's folder, not copied.")
        else:
            status = "Gyroflow found."
        self.status_lbl = QLabel(status)
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color: gray; font-size: 9pt;")
        form.addRow(self.status_lbl)

        mode_row = QVBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = {}
        for key, label, tip in MODE_LABELS:
            rb = QRadioButton(label)
            rb.setToolTip(tip)
            self.mode_group.addButton(rb)
            self.mode_buttons[key] = rb
            mode_row.addWidget(rb)
        form.addRow("Mode:", mode_row)

        self.size_combo = QComboBox()
        for value, label in OUTPUT_SIZES:
            self.size_combo.addItem(label, value)
        form.addRow("Output:", self.size_combo)

        codec_row = QHBoxLayout()
        self.codec_combo = QComboBox()
        self.codec_combo.addItems(CODECS)
        self.bitrate_spin = QSpinBox()
        self.bitrate_spin.setRange(5, 800)
        self.bitrate_spin.setSuffix(" Mb/s")
        self.bitrate_spin.setToolTip(
            "Gyroflow's intermediate file; the final clip is encoded afterwards.")
        codec_row.addWidget(self.codec_combo, 1)
        codec_row.addWidget(self.bitrate_spin)
        form.addRow("Codec:", codec_row)

        self.smooth = SliderSpin(0.0, 1.0, 0.01, 2)
        self.smooth.setToolTip("Gyroflow's smoothness: higher is steadier but crops more.")
        self.smooth_lbl = QLabel("Smoothness:")
        form.addRow(self.smooth_lbl, self.smooth)

        self.smooth_min = SliderSpin(0.0, 1.0, 0.01, 2)
        self.smooth_max = SliderSpin(0.0, 1.0, 0.01, 2)
        self.smooth_min.setToolTip("Smoothness for the calmest moments.")
        self.smooth_max.setToolTip("Smoothness for the roughest moments.")
        self.smooth_min_lbl = QLabel("Smoothness min:")
        self.smooth_max_lbl = QLabel("Smoothness max:")
        form.addRow(self.smooth_min_lbl, self.smooth_min)
        form.addRow(self.smooth_max_lbl, self.smooth_max)

        self.pan = SliderSpin(0.0, 1.0, 0.05, 2)
        self.pan.setToolTip(
            "How much a deliberate turn lowers the smoothing so the frame follows it.\n"
            "0 = ignore turns, 1 = follow them closely.")
        self.pan_lbl = QLabel("Follow turns:")
        form.addRow(self.pan_lbl, self.pan)

        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.1, 5.0)
        self.step_spin.setSingleStep(0.1)
        self.step_spin.setSuffix(" s")
        self.step_spin.setToolTip("Time between smoothness keyframes.")
        self.step_lbl = QLabel("Keyframe every:")
        form.addRow(self.step_lbl, self.step_spin)

        lpf_row = QHBoxLayout()
        self.lpf_chk = QCheckBox()
        self.lpf = SliderSpin(1.0, 100.0, 0.5, 1, " Hz")
        self.lpf.setToolTip(
            "Gyro low-pass filter: removes vibration above this frequency.\n"
            "In the automatic modes this allows a per-clip filter set from the\n"
            "vibration measured; the value here is used in manual mode.")
        self.lpf_chk.toggled.connect(self.lpf.setEnabled)
        lpf_row.addWidget(self.lpf_chk)
        lpf_row.addWidget(self.lpf, 1)
        form.addRow("Low-pass filter:", lpf_row)

        self.ask_chk = QCheckBox("Ask before each run")
        self.ask_chk.setToolTip("Show these settings when Analyze is pressed.")
        if not in_dialog:
            form.addRow(self.ask_chk)

        self.mode_group.buttonToggled.connect(lambda *_: self._sync())
        self.enabled_chk.toggled.connect(lambda *_: self._sync())
        self.set_settings(settings or DEFAULTS)

    def _mode(self):
        return next((k for k, b in self.mode_buttons.items() if b.isChecked()), "manual")

    def _sync(self):
        mode = self._mode()
        manual, frame = mode == "manual", mode == "auto_frame"
        for w in (self.smooth, self.smooth_lbl):
            w.setVisible(manual)
        for w in (self.smooth_min, self.smooth_min_lbl, self.smooth_max, self.smooth_max_lbl):
            w.setVisible(not manual)
        for w in (self.pan, self.pan_lbl, self.step_spin, self.step_lbl):
            w.setVisible(frame)
        on = self.enabled_chk.isChecked()
        for w in self.findChildren(QWidget):
            if w not in (self.enabled_chk, self.status_lbl, self.ask_chk) and w.parent() is self:
                w.setEnabled(on)
        self.lpf.setEnabled(on and self.lpf_chk.isChecked())

    def settings(self) -> dict:
        return resolve_settings({
            "enabled": self.enabled_chk.isChecked(),
            "ask": self.ask_chk.isChecked(),
            "mode": self._mode(),
            "output_size": self.size_combo.currentData(),
            "codec": self.codec_combo.currentText(),
            "bitrate": self.bitrate_spin.value(),
            "use_gpu": self._use_gpu,
            "smoothness": self.smooth.value(),
            "lpf_enabled": self.lpf_chk.isChecked(),
            "lpf_hz": self.lpf.value(),
            "smoothness_min": self.smooth_min.value(),
            "smoothness_max": self.smooth_max.value(),
            "pan_response": self.pan.value(),
            "keyframe_step": self.step_spin.value(),
        })

    def set_settings(self, raw):
        s = resolve_settings(raw)
        self._use_gpu = s["use_gpu"]
        self.enabled_chk.setChecked(s["enabled"])
        self.ask_chk.setChecked(s["ask"])
        self.mode_buttons[s["mode"]].setChecked(True)
        self.size_combo.setCurrentIndex(max(0, self.size_combo.findData(s["output_size"])))
        self.codec_combo.setCurrentText(s["codec"])
        self.bitrate_spin.setValue(s["bitrate"])
        self.smooth.setValue(s["smoothness"])
        self.lpf_chk.setChecked(s["lpf_enabled"])
        self.lpf.setValue(s["lpf_hz"])
        self.smooth_min.setValue(s["smoothness_min"])
        self.smooth_max.setValue(s["smoothness_max"])
        self.pan.setValue(s["pan_response"])
        self.step_spin.setValue(s["keyframe_step"])
        self._sync()


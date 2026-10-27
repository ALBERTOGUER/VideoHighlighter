"""Where the stabilized window looks — the panel in Video Output, and a
section of the before-run dialog. See ``modules.media.framing``."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QFormLayout, QLabel, QRadioButton, QVBoxLayout, QWidget,
)

from modules.media.framing import DEFAULTS, resolve_settings
from modules.ui.stabilization_panel import SliderSpin

MODE_LABELS = (
    ("center", "Centre", "The window stays in the middle of the picture."),
    ("follow", "Follow what stands out",
     "The window drifts, smoothly and slowly, towards what stands out in the picture\n"
     "(contrast, colour, detail — no person or object detection)."),
    ("fixed", "Best fixed position",
     "One position for the whole clip, where the most interesting part sits."),
)


class FramingPanel(QWidget):
    def __init__(self, settings=None, in_dialog=False, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)

        note = QLabel("Steers Gyroflow's crop, so it needs stabilization on. Most useful "
                      "when the output is narrower than the footage (9:16 from 4:3 or 16:9).")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 9pt;")
        form.addRow(note)

        modes = QVBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = {}
        for key, label, tip in MODE_LABELS:
            rb = QRadioButton(label)
            rb.setToolTip(tip)
            self.mode_group.addButton(rb)
            self.mode_buttons[key] = rb
            modes.addWidget(rb)
        form.addRow("Window:", modes)

        self.max_shift = SliderSpin(0.0, 25.0, 1.0, 0, " %")
        self.max_shift.setToolTip(
            "How far the window may move off centre, in % of its width.\n"
            "Gyroflow zooms in to keep a moved window free of borders: about ×1.05 at 7 %,\n"
            "×1.25 at 17 %. More following costs more crop.")
        self.speed = SliderSpin(0.01, 1.0, 0.01, 2, " widths/s")
        self.speed.setToolTip("Fastest the window may move: a fast virtual pan on top of "
                              "the camera's own movement is tiring to watch.")
        self.smooth = SliderSpin(0.0, 5.0, 0.1, 1, " s")
        self.smooth.setToolTip("How calmly the path is smoothed (Gaussian, in seconds).")
        self.sticky = SliderSpin(0.0, 10.0, 0.1, 1)
        self.sticky.setToolTip("Reluctance to move between two places that are equally interesting.")
        self.bias = SliderSpin(0.0, 1.0, 0.01, 2)
        self.bias.setToolTip("Pull towards the centre.")
        self._follow_rows = []
        for label, widget, follow_only in (("Max shift:", self.max_shift, False),
                                           ("Max speed:", self.speed, True),
                                           ("Smoothness:", self.smooth, True),
                                           ("Stickiness:", self.sticky, True),
                                           ("Centre pull:", self.bias, False)):
            lbl = QLabel(label)
            form.addRow(lbl, widget)
            self._follow_rows.append((lbl, widget, follow_only))

        self.ask_chk = QCheckBox("Ask before each run")
        self.ask_chk.setToolTip("Show this section when Analyze is pressed.")
        if not in_dialog:
            form.addRow(self.ask_chk)

        self.mode_group.buttonToggled.connect(lambda *_: self._sync())
        self.set_settings(settings or DEFAULTS)

    def _mode(self):
        return next((k for k, b in self.mode_buttons.items() if b.isChecked()), "center")

    def _sync(self):
        mode = self._mode()
        for lbl, widget, follow_only in self._follow_rows:
            on = mode == "follow" if follow_only else mode != "center"
            lbl.setEnabled(on)
            widget.setEnabled(on)

    def settings(self) -> dict:
        s = resolve_settings({
            "mode": self._mode(),
            "ask": self.ask_chk.isChecked(),
            "max_shift": self.max_shift.value() / 100,
            "max_speed": self.speed.value(),
            "smoothness": self.smooth.value(),
            "stickiness": self.sticky.value(),
            "center_bias": self.bias.value(),
            "keyframe_step": self._step,
        })
        return s

    def set_settings(self, raw):
        s = resolve_settings(raw)
        self._step = s["keyframe_step"]
        self.mode_buttons[s["mode"]].setChecked(True)
        self.ask_chk.setChecked(s["ask"])
        self.max_shift.setValue(s["max_shift"] * 100)
        self.speed.setValue(s["max_speed"])
        self.smooth.setValue(s["smoothness"])
        self.sticky.setValue(s["stickiness"])
        self.bias.setValue(s["center_bias"])
        self._sync()

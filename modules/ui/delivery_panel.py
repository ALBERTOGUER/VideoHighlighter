"""What a run delivers — the panel in Video Output, and a section of the
before-run dialog. See ``modules.media.delivery``."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QRadioButton, QVBoxLayout, QWidget,
)

from modules.media.delivery import DEFAULTS, resolve_settings

MODE_LABELS = (
    ("reel", "Highlight reel",
     "Many short moments joined into one video, plus each clip on its own."),
    ("segment", "Best continuous segment",
     "DroneReel picks the single best unbroken stretch: no cuts inside it."),
    ("both", "Both", "The reel and the best segment, from the same run."),
)


def _seconds(minimum, maximum, step=1.0, special=None):
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setSingleStep(step)
    spin.setDecimals(1)
    spin.setSuffix(" s")
    if special:
        spin.setSpecialValueText(special)
    return spin


class DeliveryPanel(QWidget):
    def __init__(self, settings=None, in_dialog=False, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)

        modes = QVBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = {}
        for key, label, tip in MODE_LABELS:
            rb = QRadioButton(label)
            rb.setToolTip(tip)
            self.mode_group.addButton(rb)
            self.mode_buttons[key] = rb
            modes.addWidget(rb)
        form.addRow("Deliver:", modes)

        self.segment_lbl = QLabel("Best segment:")
        self.segment_lbl.setStyleSheet("font-weight: 600;")
        form.addRow(self.segment_lbl)

        length = QHBoxLayout()
        self.min_spin = _seconds(1, 600)
        self.max_spin = _seconds(0, 3600, special="no limit")
        self.min_spin.setToolTip("Shortest segment DroneReel may choose.")
        self.max_spin.setToolTip("Longest segment; 'no limit' = up to the whole shot.")
        length.addWidget(QLabel("from"))
        length.addWidget(self.min_spin)
        length.addWidget(QLabel("to"))
        length.addWidget(self.max_spin)
        length.addStretch()
        self.length_lbl = QLabel("Length:")
        form.addRow(self.length_lbl, length)

        context = QHBoxLayout()
        self.pre_spin = _seconds(0, 30, 0.5)
        self.post_spin = _seconds(0, 30, 0.5)
        self.pre_spin.setToolTip("Lead-in kept before the best moment.")
        self.post_spin.setToolTip("Time kept after the best moment.")
        context.addWidget(QLabel("before"))
        context.addWidget(self.pre_spin)
        context.addWidget(QLabel("after"))
        context.addWidget(self.post_spin)
        context.addStretch()
        self.context_lbl = QLabel("Context:")
        form.addRow(self.context_lbl, context)

        self.model_chk = QCheckBox("Use my learned taste when it beats the selector")
        self.model_chk.setToolTip(
            "DroneReel's model trained from your ratings ('python -m dronereel train').\n"
            "Used only once its evaluation shows it picks better than the hand-made selector.")
        form.addRow(self.model_chk)

        self.ask_chk = QCheckBox("Ask before each run")
        self.ask_chk.setToolTip("Show this section when Analyze is pressed.")
        if not in_dialog:
            form.addRow(self.ask_chk)

        self._segment_widgets = (self.segment_lbl, self.length_lbl, self.context_lbl,
                                 self.min_spin, self.max_spin, self.pre_spin,
                                 self.post_spin, self.model_chk)
        self.mode_group.buttonToggled.connect(lambda *_: self._sync())
        self.set_settings(settings or DEFAULTS)

    def _mode(self):
        return next((k for k, b in self.mode_buttons.items() if b.isChecked()), "reel")

    def _sync(self):
        on = self._mode() != "reel"
        for w in self._segment_widgets:
            w.setEnabled(on)

    def settings(self) -> dict:
        return resolve_settings({
            "mode": self._mode(),
            "ask": self.ask_chk.isChecked(),
            "min_duration": self.min_spin.value(),
            "max_duration": self.max_spin.value(),
            "pre_roll": self.pre_spin.value(),
            "post_roll": self.post_spin.value(),
            "use_model": self.model_chk.isChecked(),
        })

    def set_settings(self, raw):
        s = resolve_settings(raw)
        self.mode_buttons[s["mode"]].setChecked(True)
        self.ask_chk.setChecked(s["ask"])
        self.min_spin.setValue(s["min_duration"])
        self.max_spin.setValue(s["max_duration"])
        self.pre_spin.setValue(s["pre_roll"])
        self.post_spin.setValue(s["post_roll"])
        self.model_chk.setChecked(s["use_model"])
        self._sync()

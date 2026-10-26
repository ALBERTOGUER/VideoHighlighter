"""One dialog on Analyze, with a section per settings panel set to "Ask
before each run" — rather than one dialog per panel in a row.

A section is ``(key, title, panel_class, settings)``; the panel classes share
one shape: ``Panel(settings, in_dialog=True)`` with ``settings()`` and an
``ask`` key in what they return.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QGroupBox, QScrollArea, QVBoxLayout, QWidget,
)


class BeforeRunDialog(QDialog):
    def __init__(self, sections, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Before analyzing")
        outer = QVBoxLayout(self)

        body = QWidget()
        column = QVBoxLayout(body)
        self.panels = {}
        for key, title, panel_class, settings in sections:
            box = QGroupBox(title)
            box_layout = QVBoxLayout(box)
            panel = panel_class(settings, in_dialog=True)
            box_layout.addWidget(panel)
            column.addWidget(box)
            self.panels[key] = panel
        column.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        self.dont_ask = QCheckBox(
            "Don't ask again (change it in Detailed settings → Video Output)")
        outer.addWidget(self.dont_ask)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Analyze with these settings")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self.resize(560, min(820, 160 + 300 * len(self.panels)))

    def results(self) -> dict:
        """``{key: settings}`` for every section, ``ask`` cleared when the user
        said not to ask again."""
        out = {}
        for key, panel in self.panels.items():
            s = panel.settings()
            s["ask"] = not self.dont_ask.isChecked()
            out[key] = s
        return out

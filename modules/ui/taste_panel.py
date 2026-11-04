"""The Taste tab: rate the best segments runs delivered, then train
DroneReel's learned taste on those ratings. See ``modules.media.taste``."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

from modules.media import taste

VERDICTS = (("accept", "Keep it"), ("reject", "Not good"), (None, "No verdict"))


def _clock(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{t % 60:04.1f}"


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


class TastePanel(QWidget):
    def __init__(self, folder_provider=None, dataset=None, model_path=None, parent=None):
        super().__init__(parent)
        self._folder = folder_provider or (lambda: "")
        self._dataset, self._model_path = dataset, model_path
        self._items = []            # [(record_path, record, rated)]
        self._job = None
        self._busy = False

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Rate the best continuous segments your runs delivered. DroneReel learns "
            "a small correction to its selector from your ratings, and a run uses it "
            "only once it has proved better than the selector on videos it did not see.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # ── where the segments are ──
        row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Folder with best segments (the run's output folder)")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        row.addWidget(self.folder_edit, 1)
        row.addWidget(browse)
        row.addWidget(self.refresh_btn)
        layout.addLayout(row)

        self.list = QListWidget()
        self.list.setMinimumHeight(110)
        self.list.currentRowChanged.connect(self._show)
        layout.addWidget(self.list)

        # ── one segment ──
        box = QGroupBox("Rate this segment")
        form = QFormLayout(box)
        self.info = QLabel("Pick a segment above.")
        self.info.setWordWrap(True)
        self.play_btn = QPushButton("Play segment")
        self.play_btn.clicked.connect(self._play)
        info_row = QHBoxLayout()
        info_row.addWidget(self.info, 1)
        info_row.addWidget(self.play_btn)
        form.addRow(info_row)

        self.score = QComboBox()
        self.score.addItems(["—", "1 ★", "2 ★★", "3 ★★★", "4 ★★★★", "5 ★★★★★"])
        form.addRow("Score:", self.score)

        verdicts = QHBoxLayout()
        self.verdict_group = QButtonGroup(self)
        self.verdict_buttons = {}
        for key, label in VERDICTS:
            b = QRadioButton(label)
            self.verdict_group.addButton(b)
            self.verdict_buttons[key] = b
            verdicts.addWidget(b)
        verdicts.addStretch(1)
        form.addRow("Verdict:", verdicts)

        alt_row = QHBoxLayout()
        self.prefer = QComboBox()
        self.prefer.setToolTip("Another stretch the selector considered. Watch it with "
                               "Preview before saying it was better.")
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self._preview)
        self.prefer.currentIndexChanged.connect(self._sync)
        alt_row.addWidget(self.prefer, 1)
        alt_row.addWidget(self.preview_btn)
        form.addRow("Better choice:", alt_row)

        manual_row = QHBoxLayout()
        self.manual_chk = QCheckBox("I'd have picked")
        self.manual_from, self.manual_to = QDoubleSpinBox(), QDoubleSpinBox()
        for spin in (self.manual_from, self.manual_to):
            spin.setDecimals(1)
            spin.setSuffix(" s")
            spin.setRange(0, 24 * 3600)
        self.manual_chk.toggled.connect(self._sync)
        manual_row.addWidget(self.manual_chk)
        manual_row.addWidget(self.manual_from)
        manual_row.addWidget(QLabel("to"))
        manual_row.addWidget(self.manual_to)
        manual_row.addStretch(1)
        form.addRow("Own pick:", manual_row)

        self.note = QLineEdit()
        self.note.setPlaceholderText("Optional, for you")
        form.addRow("Note:", self.note)

        self.save_btn = QPushButton("Save rating")
        self.save_btn.clicked.connect(self._save)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        save_row = QHBoxLayout()
        save_row.addWidget(self.save_btn)
        save_row.addWidget(self.message, 1)
        form.addRow(save_row)
        layout.addWidget(box)

        # ── the model ──
        model_box = QGroupBox("Learned taste")
        model_layout = QVBoxLayout(model_box)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.train_btn = QPushButton("Train on my ratings")
        self.train_btn.clicked.connect(self._train)
        self.train_result = QLabel("")
        self.train_result.setWordWrap(True)
        model_layout.addWidget(self.status_label)
        train_row = QHBoxLayout()
        train_row.addWidget(self.train_btn)
        train_row.addStretch(1)
        model_layout.addLayout(train_row)
        model_layout.addWidget(self.train_result)
        layout.addWidget(model_box)
        layout.addStretch(1)

        self._clear_form()
        self._show_status()

    # ── the list ────────────────────────────────────────────────────────────

    def showEvent(self, event):
        super().showEvent(event)
        if not self.folder_edit.text():
            self.folder_edit.setText(self._folder() or "")
            self.refresh()

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Folder with best segments",
                                                  self.folder_edit.text() or "")
        if folder:
            self.folder_edit.setText(folder)
            self.refresh()

    def refresh(self):
        folder = self.folder_edit.text().strip()
        self._items = taste.segments(folder, self._dataset) if folder else []
        self.list.blockSignals(True)
        self.list.clear()
        for _, rec, rated in self._items:
            e = rec["export"]
            name = os.path.basename(e.get("path") or "")
            item = QListWidgetItem(f"{'✓' if rated else '•'}  {name}   "
                                   f"({_clock(e['start'])}–{_clock(e['end'])})")
            item.setToolTip("Rated already — another rating adds to it" if rated
                            else "Not rated yet")
            self.list.addItem(item)
        self.list.blockSignals(False)
        if not self._items:
            self.list.addItem("No best segments here yet — run with Delivery set to "
                              "“Best continuous segment” or “Both”.")
            self.list.item(0).setFlags(Qt.NoItemFlags)
            self._clear_form()
        else:
            self.list.setCurrentRow(0)
        self._show_status()

    def _current(self):
        row = self.list.currentRow()
        return self._items[row] if 0 <= row < len(self._items) else None

    def _clear_form(self):
        self.score.setCurrentIndex(0)
        self.verdict_buttons[None].setChecked(True)
        self.prefer.clear()
        self.prefer.addItem("The chosen one was best", None)
        self.manual_chk.setChecked(False)
        self.note.clear()
        has = self._current() is not None
        for w in (self.play_btn, self.save_btn, self.score, self.prefer, self.manual_chk):
            w.setEnabled(has)
        self._sync()

    def _show(self, _row=None):
        current = self._current()
        self._clear_form()
        self.message.setText("")
        if current is None:
            self.info.setText("Pick a segment above.")
            return
        path, rec, rated = current
        e, src = rec["export"], rec["source"]
        self.info.setText(
            f"<b>{os.path.basename(e.get('path') or '')}</b><br>"
            f"{_clock(e['start'])}–{_clock(e['end'])} of {os.path.basename(src['path'])}"
            + ("<br><i>Rated already.</i>" if rated else ""))
        for number, s, t in taste.alternatives(rec):
            self.prefer.addItem(f"Alternative {number}: {_clock(s)}–{_clock(t)}", number)
        self.manual_from.setValue(e["start"])
        self.manual_to.setValue(e["end"])
        self.play_btn.setEnabled(os.path.exists(e.get("path") or ""))
        self._sync()

    def _sync(self, *_):
        self.preview_btn.setEnabled(self.prefer.currentData() is not None and not self._busy)
        manual = self.manual_chk.isChecked() and self.manual_chk.isEnabled()
        self.manual_from.setEnabled(manual)
        self.manual_to.setEnabled(manual)

    # ── actions ─────────────────────────────────────────────────────────────

    def _start(self, what, fn, on_done):
        if self._busy:
            return
        self._busy = True
        for b in (self.save_btn, self.train_btn, self.preview_btn, self.refresh_btn):
            b.setEnabled(False)
        self.message.setText(what)
        self._job = _Job(fn, self)
        self._job.done.connect(on_done)
        self._job.failed.connect(lambda err: self.message.setText(f"⚠️ {err}"))
        self._job.finished.connect(self._job_finished)
        self._job.start()

    def _job_finished(self):
        self._busy = False
        for b in (self.train_btn, self.refresh_btn):
            b.setEnabled(True)
        self.save_btn.setEnabled(self._current() is not None)
        self._sync()

    def _play(self):
        current = self._current()
        if current:
            taste.open_file(current[1]["export"]["path"])

    def _preview(self):
        current, number = self._current(), self.prefer.currentData()
        if not current or number is None:
            return
        rec = current[1]
        _, s, t = next(a for a in taste.alternatives(rec) if a[0] == number)
        self._start(f"Preparing a preview of alternative {number} (4K takes about "
                    f"3 s per second of video; cached after)…",
                    lambda: taste.make_preview(rec, s, t), self._opened)

    def _opened(self, path):
        self.message.setText("Preview opened.")
        taste.open_file(path)

    def _choice(self):
        verdict = next(k for k, b in self.verdict_buttons.items() if b.isChecked())
        score = self.score.currentIndex() or None
        return score, verdict, self.prefer.currentData()

    def _save(self):
        current = self._current()
        if not current:
            return
        path, rec, _ = current
        score, verdict, prefer = self._choice()
        note = self.note.text().strip()
        manual = self.manual_chk.isChecked()
        start, end = self.manual_from.value(), self.manual_to.value()
        if not manual and score is None and verdict is None and prefer is None:
            self.message.setText("Give a score, a verdict, a better choice or your own pick.")
            return
        if manual and end <= start:
            self.message.setText("Your own pick has to end after it starts.")
            return

        def work():
            saved = []
            if score is not None or verdict is not None or prefer is not None:
                taste.rate(path, rec, score=score, verdict=verdict, prefer=prefer,
                           note=note, dataset=self._dataset)
                saved.append("rating")
            if manual:
                taste.rate_manual(rec, start, end, score=None, note=note,
                                  dataset=self._dataset)
                saved.append("your own pick")
            return saved

        self._start("Saving…" if not manual else "Saving (scoring your pick)…",
                    work, self._saved)

    def _saved(self, saved):
        row = self.list.currentRow()
        self.refresh()
        self.message.setText("Saved: " + " and ".join(saved) + ".")
        if self._items:
            # The next unrated one, if any, else stay where we were.
            nxt = next((i for i, (_, _, r) in enumerate(self._items) if not r), None)
            self.list.setCurrentRow(nxt if nxt is not None else min(row, len(self._items) - 1))
            self.message.setText("Saved: " + " and ".join(saved) + ".")

    def _train(self):
        self._start("Training…", lambda: taste.train(self._dataset, self._model_path),
                    self._trained)

    def _trained(self, r):
        self.message.setText("")
        lines = [f"{r['pairs'] or 0} preferences from {r['videos'] or 0} video(s)."]
        if r.get("after") is not None:
            lines.append(f"On videos it had not seen, the selector alone agreed with you "
                         f"{r['before'] * 100:.0f}% of the time; with your taste, "
                         f"{r['after'] * 100:.0f}% ({r['test_pairs']} test pairs).")
        if r.get("learned"):
            lines.append("Leans on: " + ", ".join(r["learned"]) + ".")
        if r.get("saved"):
            lines.append("✅ Better than the selector — runs will use it." if r["better"]
                         else "Saved, but runs keep the selector until it proves better.")
        else:
            lines.append("No model trained.")
        if r.get("reason"):
            lines.append(f"Why: {r['reason']}.")
        self.train_result.setText("\n".join(lines))
        self._show_status()

    def _show_status(self):
        try:
            st = taste.status(self._dataset, self._model_path)
        except Exception as e:
            self.status_label.setText(f"⚠️ {e}")
            return
        sm, need = st["summary"], st["needed"]
        text = (f"{sm['ratings']} rating(s) and {sm['manual']} own pick(s) from "
                f"{sm['sources']} video(s). A model is used once it has ratings from at "
                f"least {need['videos']} videos and beats the selector on them.")
        model = st.get("model")
        if st.get("model_error"):
            text += f"\nSaved model unusable: {st['model_error']} — train again."
        elif model is None:
            text += "\nNo model yet: runs use the selector alone."
        elif model.get("better"):
            text += f"\nModel from {model.get('date', '?')[:10]}: in use."
        else:
            why = (model.get("evaluation") or {}).get("reason") or "it does not beat the selector"
            text += f"\nModel from {model.get('date', '?')[:10]}: not in use yet ({why})."
        self.status_label.setText(text)

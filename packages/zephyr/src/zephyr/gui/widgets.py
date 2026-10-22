"""Widgets the pages share: a job's log panel, training curves, and a trace with the
crop frames it came from, played back in step with a cursor."""

import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import features
from ..features import ClipEntry
from .jobs import Job
from .theme import SERIES, THEME

pg.setConfigOptions(antialias=True, background=THEME["card"], foreground=THEME["ink"])

POLL_MS = 150


def entry_from(fields: dict) -> ClipEntry:
    """A :class:`ClipEntry` back from the strings a snippet carries."""
    ints = {"n_frames", "n_output"}
    paths = {"stem", "frames", "frame_times", "times", "target", "events"}
    paths |= {"thermistor", "video"}
    return ClipEntry(
        **{
            k: int(v) if k in ints else (Path(v) if v else None) if k in paths else v
            for k, v in fields.items()
        }
    )


def find_data(combo: QtWidgets.QComboBox, value) -> int:
    """The index of *value* among *combo*'s item data, by ``==`` (``findData``
    compares Python objects such as paths by identity); -1 if absent."""
    for index in range(combo.count()):
        if value is not None and combo.itemData(index) == value:
            return index
    return -1


def heading(text: str) -> QtWidgets.QLabel:
    return QtWidgets.QLabel(text, objectName="heading")


def note(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text, objectName="sub", wordWrap=True)
    label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def accent(text: str) -> QtWidgets.QPushButton:
    return QtWidgets.QPushButton(text, objectName="accent")


class JobPanel(QtWidgets.QWidget):
    """Runs one :class:`Job` at a time: its live log, Cancel, and the command line."""

    line = QtCore.Signal(str)
    snippets = QtCore.Signal(dict)
    finished = QtCore.Signal(object)
    """The job's return code (None if it was cancelled before it ended)."""

    def __init__(self) -> None:
        super().__init__()
        self.job: Job | None = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("idle")
        bar.addWidget(self.status, 1)
        self.copy = QtWidgets.QPushButton("Copy command")
        self.copy.setToolTip("The same command, to run in a terminal from the project")
        self.copy.clicked.connect(self._copy)
        self.cancel = QtWidgets.QPushButton("Cancel")
        self.cancel.clicked.connect(self.stop)
        for button in (self.copy, self.cancel):
            button.setEnabled(False)
            bar.addWidget(button)
        layout.addLayout(bar)
        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(20000)
        layout.addWidget(self.log, 1)
        self._timer = QtCore.QTimer(self, interval=POLL_MS)
        self._timer.timeout.connect(self._poll)

    @property
    def busy(self) -> bool:
        return self.job is not None and self.job.running

    def run(self, job: Job) -> bool:
        """Start *job*; False (after saying so) while another one runs."""
        if self.busy:
            QtWidgets.QMessageBox.information(
                self, "Busy", "Another command is running; cancel it or wait."
            )
            return False
        self.job = job
        self.log.appendPlainText(f"\n$ {job.command_line()}")
        try:
            job.start()
        except OSError as exc:
            self.log.appendPlainText(f"could not start: {exc}")
            return False
        self.status.setText(f"running: zephyr {job.command}")
        self.copy.setEnabled(True)
        self.cancel.setEnabled(True)
        self._timer.start()
        return True

    def stop(self) -> None:
        if self.busy:
            self.job.cancel()
            self.status.setText("cancelling...")

    def _copy(self) -> None:
        if self.job is not None:
            QtGui.QGuiApplication.clipboard().setText(self.job.command_line())

    def _poll(self) -> None:
        if self.job is None:
            return
        for kind, payload in self.job.poll():
            if kind == "log":
                self.log.appendPlainText(payload)
                self.line.emit(payload)
            elif kind == "snippets":
                self.snippets.emit(payload)
            elif kind == "exit":
                self._timer.stop()
                self.cancel.setEnabled(False)
                cancelled = self.job.cancelled
                self.status.setText(
                    "cancelled"
                    if cancelled
                    else "done"
                    if payload == 0
                    else f"failed (exit {payload}): see the log"
                )
                self.finished.emit(None if cancelled else payload)


def _plot(title: str, left: str) -> pg.PlotWidget:
    plot = pg.PlotWidget(title=title)
    plot.showGrid(x=True, y=True, alpha=0.15)
    plot.setLabel("bottom", "epoch")
    plot.setLabel("left", left)
    plot.getAxis("left").enableAutoSIPrefix(False)
    plot.addLegend(offset=(-10, 10))
    return plot


class HistoryPlot(QtWidgets.QWidget):
    """A run's ``history.json``: losses per epoch and the whole-clip validation scores,
    with each ``best.pt`` save marked."""

    LOSSES = (("train_loss", "train"), ("val_loss", "validation"))
    SCORES = (
        ("xcorr", "correlation"),
        ("inhale_f1", "inhale F1"),
        ("exhale_f1", "exhale F1"),
        ("ema_xcorr", "correlation (EMA)"),
    )

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.loss = _plot("Loss", "loss")
        self.scores = _plot("Validation, whole clips", "score")
        self.scores.setXLink(self.loss)
        layout.addWidget(self.loss, 1)
        layout.addWidget(self.scores, 1)
        self.curves = {}
        for i, (key, name) in enumerate(self.LOSSES):
            self.curves[key] = self.loss.plot(
                name=name, pen=pg.mkPen(SERIES[i], width=2)
            )
        for i, (key, name) in enumerate(self.SCORES):
            style = QtCore.Qt.PenStyle.DashLine if key.startswith("ema") else None
            pen = pg.mkPen(SERIES[i % 3], width=2)
            if style is not None:
                pen.setStyle(style)
            self.curves[key] = self.scores.plot(
                name=name, pen=pen, symbol="o", symbolSize=5, symbolBrush=SERIES[i % 3]
            )
        self.best = pg.ScatterPlotItem(
            symbol="star", size=14, brush=THEME["span"], pen=None, name="best.pt"
        )
        self.scores.addItem(self.best)

    def set_history(self, rows: list[dict]) -> None:
        for key, curve in self.curves.items():
            pairs = [
                (r["epoch"] + 1, r[key])
                for r in rows
                if isinstance(r.get(key), int | float) and np.isfinite(r[key])
            ]
            curve.setData(*(zip(*pairs) if pairs else ([], [])))
        best = [
            (r["epoch"] + 1, r["xcorr"])
            for r in rows
            if r.get("since_best") == 0 and "xcorr" in r
        ]
        self.best.setData(*(zip(*best) if best else ([], [])))


def read_history(run_dir: Path) -> list[dict]:
    """The rows of *run_dir*'s ``history.json`` (empty while it does not exist, or is
    being rewritten)."""
    try:
        return json.loads((run_dir / "history.json").read_text())
    except (OSError, ValueError):
        return []


class TraceView(QtWidgets.QWidget):
    """A predicted trace (and the true one, if any) with inhale ticks, beside the crop
    frames of the same span read from the feature cache; Play steps a cursor along
    the trace and shows the frame under it."""

    def __init__(self) -> None:
        super().__init__()
        self._frames: np.ndarray | None = None
        self._frame_times: np.ndarray | None = None
        self._t = np.empty(0)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        side = QtWidgets.QVBoxLayout()
        self.title = QtWidgets.QLabel(objectName="sub", wordWrap=True)
        self.title.setMaximumWidth(220)
        side.addWidget(self.title)
        self.view = pg.GraphicsLayoutWidget()
        self.view.setFixedSize(180, 180)
        box = self.view.addViewBox(lockAspect=True, invertY=True, enableMouse=False)
        self.image = pg.ImageItem(axisOrder="row-major")
        box.addItem(self.image)
        side.addWidget(self.view)
        self.play = QtWidgets.QPushButton("Play")
        self.play.setCheckable(True)
        self.play.toggled.connect(self._toggle)
        side.addWidget(self.play)
        side.addStretch(1)
        layout.addLayout(side)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, alpha=0.15)
        self.plot.setLabel("bottom", "time (s)")
        self.legend = self.plot.addLegend(offset=(-10, 5))
        layout.addWidget(self.plot, 1)
        self.cursor = pg.InfiniteLine(
            angle=90, movable=True, pen=pg.mkPen(THEME["muted"], width=1)
        )
        self.cursor.sigPositionChanged.connect(
            lambda: self.show_frame(self.cursor.value())
        )
        self._timer = QtCore.QTimer(self, interval=33)
        self._timer.timeout.connect(self._step)
        self._clock = QtCore.QElapsedTimer()
        self._start = 0.0

    def set_trace(
        self,
        title: str,
        time: np.ndarray,
        signal: np.ndarray,
        *,
        truth: np.ndarray | None = None,
        inhale: np.ndarray | None = None,
        truth_inhale: np.ndarray | None = None,
        entry: ClipEntry | None = None,
        bad_spans: list[tuple[float, float]] = (),
    ) -> None:
        """Show one trace; *entry* (its cache entry) gives the crop frames."""
        self.play.setChecked(False)
        self.title.setText(title)
        self._t = np.asarray(time, float)
        plot = self.plot
        plot.clear()
        self.legend.clear()
        for start, end in bad_spans:
            region = pg.LinearRegionItem(
                (start, end), movable=False, brush=pg.mkBrush(217, 83, 79, 40)
            )
            region.setZValue(-10)
            plot.addItem(region)
        if truth is not None:
            plot.plot(self._t, _z(truth), name="sensor", pen=pg.mkPen(THEME["raw"]))
        plot.plot(
            self._t,
            _z(signal),
            name="prediction",
            pen=pg.mkPen(THEME["accent"], width=2),
        )
        top = 3.2
        if truth_inhale is not None and len(truth_inhale):
            plot.plot(
                np.asarray(truth_inhale),
                np.full(len(truth_inhale), top),
                pen=None,
                symbol="t",
                symbolBrush=THEME["raw"],
                symbolSize=8,
                name="true inhale",
            )
        if inhale is not None and len(inhale):
            plot.plot(
                np.asarray(inhale),
                np.full(len(inhale), -top),
                pen=None,
                symbol="t1",
                symbolBrush=THEME["inhale"],
                symbolSize=8,
                name="predicted inhale",
            )
        plot.addItem(self.cursor)
        if len(self._t):
            self.cursor.setValue(self._t[0])
            self.plot.setXRange(self._t[0], min(self._t[-1], self._t[0] + 20))
        self._load_frames(entry)
        self.show_frame(self.cursor.value())

    def _load_frames(self, entry: ClipEntry | None) -> None:
        self._frames = self._frame_times = None
        self.view.setVisible(entry is not None)
        self.play.setEnabled(len(self._t) > 1)
        if entry is None or entry.frames is None or not entry.frames.is_file():
            self.view.setVisible(False)
            return
        if not len(self._t):
            return
        times = np.load(entry.frame_times)
        rows = np.flatnonzero((times >= self._t[0] - 0.05) & (times <= self._t[-1]))
        if len(rows) == 0:
            return
        window = slice(int(rows[0]), int(rows[-1]) + 1)
        self._frames = features.reader(entry).read(window)[:, 0]
        self._frame_times = times[window]

    def show_frame(self, t: float) -> None:
        if self._frames is None or not len(self._frames):
            return
        k = int(
            np.clip(np.searchsorted(self._frame_times, t), 0, len(self._frames) - 1)
        )
        self.image.setImage(self._frames[k], levels=(0, 255))

    def _toggle(self, on: bool) -> None:
        self.play.setText("Pause" if on else "Play")
        if on and len(self._t):
            if self.cursor.value() >= self._t[-1]:
                self.cursor.setValue(self._t[0])
            self._start = self.cursor.value()
            self._clock.start()
            self._timer.start()
        else:
            self._timer.stop()

    def _step(self) -> None:
        t = self._start + self._clock.elapsed() / 1000.0
        if t >= self._t[-1]:
            self.play.setChecked(False)
            t = self._t[-1]
        self.cursor.setValue(t)
        lo, hi = self.plot.getViewBox().viewRange()[0]
        if not lo <= t <= hi:
            self.plot.setXRange(t, t + (hi - lo), padding=0)


def _z(x) -> np.ndarray:
    x = np.asarray(x, float)
    scale = x.std()
    return (x - x.mean()) / (scale if scale > 0 else 1.0)


class SnippetGallery(QtWidgets.QWidget):
    """The validation snippets a live run sends after each scoring, with a slider
    over the scorings received so far (kept in memory only)."""

    def __init__(self) -> None:
        super().__init__()
        self.received: list[dict] = []
        layout = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLabel("No snippets yet: they arrive at each scoring.")
        bar.addWidget(self.label, 1)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._show)
        self.follow = QtWidgets.QCheckBox("follow latest", checked=True)
        bar.addWidget(self.slider, 2)
        bar.addWidget(self.follow)
        layout.addLayout(bar)
        scroll = QtWidgets.QScrollArea(widgetResizable=True)
        inner = QtWidgets.QWidget()
        self.column = QtWidgets.QVBoxLayout(inner)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)
        self.views: list[TraceView] = []

    def clear(self) -> None:
        self.received = []
        self.slider.setEnabled(False)
        self.label.setText("No snippets yet: they arrive at each scoring.")
        for view in self.views:
            view.setVisible(False)

    def add(self, payload: dict) -> None:
        self.received.append(payload)
        self.slider.setEnabled(True)
        self.slider.setRange(0, len(self.received) - 1)
        if self.follow.isChecked():
            self.slider.setValue(len(self.received) - 1)
        if len(self.received) == 1:
            self._show(0)

    def _show(self, index: int) -> None:
        if not self.received:
            return
        payload = self.received[index]
        self.label.setText(
            f"epoch {payload['epoch'] + 1} (scoring {index + 1} of {len(self.received)})"
        )
        while len(self.views) < len(payload["clips"]):
            view = TraceView()
            view.setMinimumHeight(220)
            self.views.append(view)
            self.column.addWidget(view)
        for view, clip in zip(self.views, payload["clips"], strict=False):
            view.setVisible(True)
            scores = clip["scores"]
            view.set_trace(
                f"<b>{clip['clip_id']}</b><br>correlation "
                f"{scores.get('correlation', float('nan')):+.3f}",
                np.asarray(clip["time"]),
                np.asarray(clip["signal"]),
                truth=np.asarray(clip["truth"]),
                inhale=np.asarray(clip["inhale"]),
                truth_inhale=np.asarray(clip["truth_inhale"]),
                entry=entry_from(clip["entry"]),
            )
        for view in self.views[len(payload["clips"]) :]:
            view.setVisible(False)

"""Widgets the pages share: a job's log panel, training curves, and a trace with the
crop frames it came from, played back in step with a cursor."""

import json
import shutil
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import features
from ..features import ClipEntry
from .jobs import Job
from .project import cache_need, human_bytes
from .theme import SERIES, THEME

pg.setConfigOptions(antialias=True, background=THEME["card"], foreground=THEME["ink"])

POLL_MS = 150
IDLE = "Nothing running. Commands started on this page show their progress here."
LOG_HEIGHT = (100, 220)
"""Smallest and largest height of a job's log when it is open, in pixels."""
ONSET_HEIGHT = 110
"""Height of the onset head's strip under a trace, in pixels."""
TRACE_S = 30.0
"""The span of a trace shown at first, in seconds (zoom and pan show the rest)."""
PLOT_HEIGHT = (200, 320)
"""Smallest and largest height of a training-curve plot, in pixels."""


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


def ask_cache_space(parent, lists: list[Path], cache: Path) -> bool:
    """Whether to write *lists*' frames into *cache*, after saying how much space it
    takes and how much the drive has (nothing to ask when all is cached)."""
    try:
        need = cache_need(lists, cache)
    except ValueError:
        return True  # the command says what is wrong with the list
    if not need.todo:
        return True
    drive = cache
    while not drive.exists() and drive.parent != drive:  # the nearest folder there is
        drive = drive.parent
    free = shutil.disk_usage(drive).free
    unknown = (
        f" ({need.unknown} more of unknown size: their timestamps cannot be read)"
        if need.unknown
        else ""
    )
    text = (
        f"This writes about {human_bytes(need.bytes)} of frames into {cache.name}/ "
        f"for {need.todo} clip(s){unknown}; {need.cached} are cached already. The "
        f"drive has {human_bytes(free)} free."
    )
    if need.bytes > free:
        text += "\n\nThat is not enough: free some space or cancel."
    answer = QtWidgets.QMessageBox.question(parent, "Space for the cache", text)
    return answer == QtWidgets.QMessageBox.StandardButton.Yes


class ElidedLabel(QtWidgets.QLabel):
    """One line that never widens its window: cut with an ellipsis to the room it
    has, the whole of it in the tooltip (:meth:`text` is the whole of it too)."""

    def __init__(self, text: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred
        )
        self._full = ""
        self.setText(text)

    def setText(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        self._elide()

    def text(self) -> str:
        return self._full

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        metrics = self.fontMetrics()
        width = max(self.width(), 1)
        shown = metrics.elidedText(
            self._full, QtCore.Qt.TextElideMode.ElideRight, width
        )
        super().setText(shown)


class JobPanel(QtWidgets.QWidget):
    """Runs one :class:`Job` at a time: a line saying what runs (or how it ended),
    Cancel, the command line, and its live log, folded away until asked for or the
    command fails. The main window shows the current page's panel below it."""

    line = QtCore.Signal(str)
    snippets = QtCore.Signal(dict)
    started = QtCore.Signal(str)
    """The command line of a job that just started."""
    finished = QtCore.Signal(object)
    """The job's return code (None if it was cancelled before it ended)."""

    def __init__(self) -> None:
        super().__init__()
        self.job: Job | None = None
        self._since = QtCore.QElapsedTimer()
        self._last = ""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        self.status = ElidedLabel(IDLE, objectName="jobstatus")
        bar.addWidget(self.status, 1)
        self.copy = QtWidgets.QPushButton("Copy command")
        self.copy.setToolTip("The same command, to run in a terminal from the project")
        self.copy.clicked.connect(self._copy)
        self.cancel = QtWidgets.QPushButton("Cancel")
        self.cancel.clicked.connect(self.stop)
        for button in (self.copy, self.cancel):
            button.setEnabled(False)
            bar.addWidget(button)
        self.toggle = QtWidgets.QToolButton(
            text="Log",
            checkable=True,
            toolButtonStyle=QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
        )
        self.toggle.setToolTip("Show or hide everything the commands printed")
        self.toggle.toggled.connect(self.show_log)
        bar.addWidget(self.toggle)
        layout.addLayout(bar)
        self.progress = QtWidgets.QProgressBar(textVisible=False, maximumHeight=6)
        self.progress.setRange(0, 0)  # busy: no end is known
        self.progress.hide()
        layout.addWidget(self.progress)
        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(20000)
        self.log.setMinimumHeight(LOG_HEIGHT[0])
        self.log.setMaximumHeight(LOG_HEIGHT[1])
        layout.addWidget(self.log, 1)
        self.show_log(False)
        self._timer = QtCore.QTimer(self, interval=POLL_MS)
        self._timer.timeout.connect(self._poll)

    @property
    def busy(self) -> bool:
        return self.job is not None and self.job.running

    def show_log(self, shown: bool) -> None:
        self.toggle.blockSignals(True)
        self.toggle.setChecked(shown)
        self.toggle.blockSignals(False)
        self.toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if shown else QtCore.Qt.ArrowType.UpArrow
        )
        self.log.setVisible(shown)

    def say(self, text: str) -> None:
        """Put *text* in the log and open it (a page's error that is not a command's)."""
        self.log.appendPlainText(text)
        self.show_log(True)

    def run(self, job: Job, *, show_log: bool = False) -> bool:
        """Start *job* (opening the log with *show_log*, for a command whose output
        is its progress); False (after saying so) while another one runs."""
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
            self._show_state("failed", f"Could not start zephyr {job.command}: {exc}")
            self.say(f"could not start: {exc}")
            return False
        if show_log:
            self.show_log(True)
        self._since.start()
        self._last = ""
        self._show_state("running")
        self.progress.show()
        self.copy.setEnabled(True)
        self.cancel.setEnabled(True)
        self._timer.start()
        self.started.emit(job.command_line())
        return True

    def describe(self) -> str:
        """What runs, for how long, and its latest line of output."""
        if self.job is None:
            return ""
        seconds = self._since.elapsed() // 1000
        text = f"zephyr {self.job.command} {self.job.argv[0] if self.job.argv else ''}"
        text = f"{text.strip()} · {seconds // 60}:{seconds % 60:02d}"
        return f"{text} · {self._last}" if self._last else text

    def _show_state(self, state: str, text: str = "") -> None:
        self.status.setProperty("state", state)
        self.status.style().polish(self.status)
        self.status.setText(text or f"Running: {self.describe()}")

    def stop(self) -> None:
        if self.busy:
            self.job.cancel()
            self._show_state("running", "Cancelling...")

    def _copy(self) -> None:
        if self.job is not None:
            QtGui.QGuiApplication.clipboard().setText(self.job.command_line())

    def _poll(self) -> None:
        if self.job is None:
            return
        for kind, payload in self.job.poll():
            if kind == "log":
                self.log.appendPlainText(payload)
                if payload.strip():
                    self._last = payload.strip()[:90]
                self.line.emit(payload)
            elif kind == "snippets":
                self.snippets.emit(payload)
            elif kind == "exit":
                self._timer.stop()
                self.progress.hide()
                self.cancel.setEnabled(False)
                cancelled = self.job.cancelled
                name = f"zephyr {self.job.command}"
                if cancelled:
                    self._show_state("idle", f"Cancelled: {name}")
                elif payload == 0:
                    self._show_state("done", f"Done: {name}")
                else:
                    last = f": {self._last}" if self._last else ""
                    self._show_state("failed", f"Failed (exit {payload}): {name}{last}")
                    self.show_log(True)
                self.finished.emit(None if cancelled else payload)
                return
        if self.busy:
            self._show_state("running")


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
        plots = QtWidgets.QHBoxLayout()
        self.loss = _plot("Loss", "loss")
        self.scores = _plot("Validation, whole clips", "score")
        self.scores.setXLink(self.loss)
        for plot in (self.loss, self.scores):
            plot.setMinimumHeight(PLOT_HEIGHT[0])
            plot.setMaximumHeight(PLOT_HEIGHT[1])
            plots.addWidget(plot, 1)
        layout.addLayout(plots)
        layout.addStretch(1)
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
        self._offset = 0.0
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
        plots = QtWidgets.QVBoxLayout()
        plots.setSpacing(0)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, alpha=0.15)
        self.plot.setLabel("bottom", "time from the start (s)")
        self.legend = self.plot.addLegend(offset=(-10, 5))
        plots.addWidget(self.plot, 1)
        # The onset head's own output, under the trace: how sure it is of an inhale
        # at each moment, not only where it crossed the threshold.
        self.onset_plot = pg.PlotWidget()
        self.onset_plot.setMaximumHeight(ONSET_HEIGHT)
        self.onset_plot.setMinimumHeight(ONSET_HEIGHT // 2)
        self.onset_plot.setXLink(self.plot)
        self.onset_plot.setYRange(0, 1, padding=0.05)
        self.onset_plot.setMouseEnabled(y=False)
        self.onset_plot.setLabel("left", "onset head")
        self.onset_plot.showGrid(x=True, alpha=0.15)
        self.onset_plot.hide()
        plots.addWidget(self.onset_plot)
        layout.addLayout(plots, 1)
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
        onset: np.ndarray | None = None,
        threshold: float | None = None,
    ) -> None:
        """Show one trace; *entry* (its cache entry) gives the crop frames, *onset*
        the onset head's probability (drawn below, with the *threshold* its inhales
        were found at; default the scorer's). Times (any clock) are shown in seconds
        from the trace's start."""
        self.play.setChecked(False)
        self.title.setText(title)
        time = np.asarray(time, float)
        self._offset = float(time[0]) if len(time) else 0.0
        self._t = time - self._offset
        plot = self.plot
        plot.clear()
        self.legend.clear()
        for start, end in bad_spans:
            region = pg.LinearRegionItem(
                (start - self._offset, end - self._offset),
                movable=False,
                brush=pg.mkBrush(217, 83, 79, 40),
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
                np.asarray(truth_inhale) - self._offset,
                np.full(len(truth_inhale), top),
                pen=None,
                symbol="t",
                symbolBrush=THEME["raw"],
                symbolSize=8,
                name="true inhale",
            )
        if inhale is not None and len(inhale):
            plot.plot(
                np.asarray(inhale) - self._offset,
                np.full(len(inhale), -top),
                pen=None,
                symbol="t1",
                symbolBrush=THEME["inhale"],
                symbolSize=8,
                name="predicted inhale",
            )
        plot.addItem(self.cursor)
        self._show_onset(onset, threshold)
        if len(self._t):
            self.cursor.setValue(self._t[0])
            self.plot.setXRange(self._t[0], min(self._t[-1], self._t[0] + TRACE_S))
        self._load_frames(entry)
        self.show_frame(self.cursor.value())

    def _show_onset(
        self, onset: np.ndarray | None, threshold: float | None = None
    ) -> None:
        """The onset head's probability under the trace, or nothing for a network
        without the head (no output, or one that never moves)."""
        self.onset_plot.clear()
        onset = None if onset is None else np.asarray(onset, float)[: len(self._t)]
        if onset is None or not len(onset) or np.nanstd(onset) == 0:
            self.onset_plot.hide()
            return
        if threshold is None:
            from ..evaluate import HEAD_THRESHOLD as threshold  # the scorer's default

        self.onset_plot.plot(
            self._t[: len(onset)],
            onset,
            pen=pg.mkPen(THEME["inhale"], width=1),
            fillLevel=0,
            brush=pg.mkBrush(47, 158, 99, 50),
        )
        self.onset_plot.addItem(
            pg.InfiniteLine(
                threshold,
                angle=0,
                pen=pg.mkPen(THEME["muted"], style=QtCore.Qt.PenStyle.DashLine),
            )
        )
        self.onset_plot.show()

    def clear(self, title: str) -> None:
        """Show no trace, only *title* (why there is none)."""
        self.play.setChecked(False)
        self.title.setText(title)
        self._t = np.empty(0)
        self.plot.clear()
        self.legend.clear()
        self._show_onset(None)
        self._load_frames(None)

    def _load_frames(self, entry: ClipEntry | None) -> None:
        self._frames = self._frame_times = None
        self.view.setVisible(entry is not None)
        self.play.setEnabled(len(self._t) > 1)
        if entry is None or entry.frames is None or not entry.frames.is_file():
            self.view.setVisible(False)
            return
        if not len(self._t):
            return
        times = np.load(entry.frame_times) - self._offset
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
                onset=np.asarray(clip["onset"]) if "onset" in clip else None,
            )
        for view in self.views[len(payload["clips"]) :]:
            view.setVisible(False)

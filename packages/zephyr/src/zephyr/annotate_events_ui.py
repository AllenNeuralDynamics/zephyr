"""The detector tuner's window: pyqtgraph traces, live stats and a per-clip table.

Needs the ``ui`` extra (``pyqtgraph`` and ``PySide6``) and a display; the logic it
drives is :class:`zephyr.annotate_events.TuneState`.  Events are never edited: the
Editor tab shows a clip's traces, events and metrics, the Methods tab one row per
clip with its detector; only the detector, its parameters and the clip's excluded
spans change.  Everything is in the menus; wheel zooms, shift+wheel or horizontal
scroll pans, n/p change clip, x marks an excluded span (start, then end) and a
right-click deletes one.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import events
from .annotate_events import (
    METRICS,
    POINT_KINDS,
    RATE_BAND_HZ,
    TuneState,
    mean_metrics,
)
from .rates import breath_rates

THEME = {
    "bg": "#f3f4f7",
    "card": "#ffffff",
    "ink": "#1f2933",
    "muted": "#7b8794",
    "line": "#e1e5ea",
    "accent": "#3b6fd4",
    "raw": "#6b7785",
    "inhale": "#2f9e63",
    "exhale": "#d9534f",
    "span": "#f0a53a",
    "bad": "#d9534f",
}
SYMBOLS = {"inhale": "t1", "exhale": "t"}
HIST_MAX_HZ = 20.0
SEEKER_BUCKETS = 2000
AMPLITUDE_WINDOW_S = 1.0
"""Width of the rolling window whose raw 5th-95th percentile range is the amplitude."""
AMPLITUDE_COLOUR = "#8e5bd0"
AMPLITUDE_HZ = 50.0
SERIES_BINS = 120
"""Bins across the window for the rate and amplitude lines (coarser when zoomed out)."""
MARKER_MAX_WINDOW_S = 60.0
"""Beyond this window the event markers are hidden: there would be thousands."""
MIN_WINDOW_S = 0.5

STYLE = f"""
QWidget {{ background: {THEME["bg"]}; color: {THEME["ink"]}; font-family: "Segoe UI"; font-size: 13px; }}
QFrame#card {{ background: {THEME["card"]}; border-radius: 6px; }}
QFrame#card QLabel {{ background: transparent; }}
QLabel#title {{ color: {THEME["muted"]}; font-size: 9px; }}
QLabel#value {{ font-size: 19px; font-weight: 600; }}
QLabel#sub {{ color: {THEME["muted"]}; font-size: 10px; }}
QListWidget, QTableView, QLineEdit, QComboBox, QDoubleSpinBox {{
    background: {THEME["card"]}; border: 1px solid {THEME["line"]}; border-radius: 4px; }}
QListWidget, QTableView {{ font-size: 12px; }}
QListWidget::item {{ padding: 1px 4px; }}
QTableView::item {{ padding: 0px 4px; }}
QListWidget::item:selected, QTableView::item:selected {{ background: {THEME["accent"]}; color: white; }}
QHeaderView::section {{ background: {THEME["bg"]}; border: none; padding: 3px; color: {THEME["muted"]}; }}
QPushButton {{ background: {THEME["line"]}; border: none; border-radius: 4px; padding: 6px 14px; }}
QPushButton:hover {{ background: #d3d9e0; }}
QPushButton#accent {{ background: {THEME["accent"]}; color: white; font-weight: 600; }}
QPushButton#accent:hover {{ background: #2f5bb5; }}
QPushButton:disabled {{ color: {THEME["muted"]}; }}
QSplitter::handle {{ background: {THEME["line"]}; }}
"""


def _binned(t, y, t0: float, width: float, median: bool = False):
    """*y* reduced to its mean (or median) in bins of *width* s from *t0*;
    returns bin centres and values, skipping empty bins."""
    index = np.floor((np.asarray(t) - t0) / width).astype(np.int64)
    grouped = pd.Series(np.asarray(y)).groupby(index)
    values = grouped.median() if median else grouped.mean()
    return t0 + (values.index.to_numpy() + 0.5) * width, values.to_numpy()


def _missing(value) -> bool:
    """Whether a table value is empty (NaN) and so sorts last."""
    return isinstance(value, float) and np.isnan(value)


def _alpha(colour: str, alpha: int) -> QtGui.QColor:
    """*colour* (``#rrggbb``) with *alpha* out of 255."""
    out = QtGui.QColor(colour)
    out.setAlpha(alpha)
    return out


def _envelope(t: np.ndarray, x: np.ndarray, buckets: int):
    """Bucket times with the minimum and maximum of *x* in each (a faithful outline)."""
    size = max(1, len(x) // buckets)
    n = size * (len(x) // size)
    blocks = x[:n].reshape(-1, size)
    return t[:n].reshape(-1, size)[:, size // 2], blocks.min(axis=1), blocks.max(axis=1)


class PanBox(pg.ViewBox):
    """A view box whose horizontal scroll (or shift + wheel) pans time; the plain
    wheel zooms as usual."""

    def wheelEvent(self, ev, axis=None) -> None:
        sideways = ev.orientation() == QtCore.Qt.Orientation.Horizontal or bool(
            ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
        )
        if not sideways:
            return super().wheelEvent(ev, axis)
        lo, hi = self.viewRange()[0]
        shift = -ev.delta() / 120 * 0.2 * (hi - lo)
        self.setXRange(lo + shift, hi + shift, padding=0)
        ev.accept()


class SpanBox(PanBox):
    """A pan/zoom box where a right-click deletes the excluded span under it."""

    def __init__(self, window: "EditorWindow") -> None:
        super().__init__(enableMenu=False)
        self.win = window

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.win.delete_span(self.mapSceneToView(ev.scenePos()).x())
            ev.accept()
        else:
            super().mouseClickEvent(ev)


class SeekBox(pg.ViewBox):
    """The seeker's view box: click or drag outside the window to move it."""

    def __init__(self, window: "EditorWindow") -> None:
        super().__init__(enableMenu=False)
        self.win = window
        self.setMouseEnabled(x=False, y=False)

    def mouseClickEvent(self, ev) -> None:
        self.win.centre_on(self.mapSceneToView(ev.scenePos()).x())
        ev.accept()

    def mouseDragEvent(self, ev, axis=None) -> None:
        self.win.centre_on(self.mapSceneToView(ev.scenePos()).x())
        ev.accept()


class EventModel(QtCore.QAbstractTableModel):
    """The event list: one row per inhale or exhale; fast to reset."""

    HEAD = ("time", "kind", "rate")

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple] = []
        self.tags: list[str] = []
        self.times: list[float] = []

    def rowCount(self, parent=None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.rows)

    def columnCount(self, parent=None) -> int:
        return len(self.HEAD)

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if (
            role == QtCore.Qt.ItemDataRole.DisplayRole
            and orientation == QtCore.Qt.Orientation.Horizontal
        ):
            return self.HEAD[section]

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        row = index.row()
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return self.rows[row][index.column()]
        if role == QtCore.Qt.ItemDataRole.ForegroundRole:
            tag = self.tags[row]
            if tag:
                return QtGui.QBrush(QtGui.QColor(THEME[tag]))
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            return (
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )

    def reset(self, rows, tags, times) -> None:
        self.beginResetModel()
        self.rows, self.tags, self.times = rows, tags, times
        self.endResetModel()


class _Signals(QtCore.QObject):
    done = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)


class _Job(QtCore.QRunnable):
    """Run *fn* off the UI thread; ``signals.done`` carries an error text or ''."""

    def __init__(self, fn, reports: bool = False) -> None:
        super().__init__()
        self.fn = fn
        self.reports = reports  # fn(progress) rather than fn()
        self.signals = _Signals()

    def run(self) -> None:
        try:
            self.fn(self.signals.progress.emit) if self.reports else self.fn()
            error = ""
        except Exception as exc:  # noqa: BLE001 - shown to the user
            error = str(exc) or type(exc).__name__
        self.signals.done.emit(error)


class DetectorCard(QtWidgets.QFrame):
    """Pick a detector and its parameters; each button applies them somewhere."""

    def __init__(self, current, buttons, horizontal: bool = False):
        """*current* returns the (method, params) to show; *buttons* is a list of
        (text, callback, accent)."""
        super().__init__(objectName="card")
        self.current, self.horizontal = current, horizontal
        outer = (
            QtWidgets.QHBoxLayout(self) if horizontal else QtWidgets.QVBoxLayout(self)
        )
        outer.setContentsMargins(10, 8, 10, 8)
        outer.addWidget(QtWidgets.QLabel("DETECTOR", objectName="title"))
        self.picker = QtWidgets.QComboBox()
        self.picker.addItems(sorted(events.DETECTORS))
        self.picker.setCurrentText(current()[0])
        self.picker.currentTextChanged.connect(self._build_params)
        outer.addWidget(self.picker)
        host = QtWidgets.QWidget()
        host.setStyleSheet("background: transparent;")
        self.grid = QtWidgets.QGridLayout(host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(8)
        outer.addWidget(host, 1 if horizontal else 0)
        for text, callback, accent in buttons:
            button = QtWidgets.QPushButton(text, objectName="accent" if accent else "")
            button.clicked.connect(callback)
            outer.addWidget(button)
        self.edits: dict[str, QtWidgets.QLineEdit] = {}
        self._build_params(self.picker.currentText())

    def _build_params(self, name: str) -> None:
        while self.grid.count():
            self.grid.takeAt(0).widget().deleteLater()
        self.edits.clear()
        method, params = self.current()
        values = params if name == method else {}
        fields = events.DETECTORS[name][0].model_fields
        for n, (field, info) in enumerate(fields.items()):
            edit = QtWidgets.QLineEdit(str(values.get(field, info.default)))
            label = QtWidgets.QLabel(field)
            if self.horizontal:
                edit.setMaximumWidth(80)
                self.grid.addWidget(label, 0, 2 * n)
                self.grid.addWidget(edit, 0, 2 * n + 1)
            else:
                self.grid.addWidget(label, n, 0)
                self.grid.addWidget(edit, n, 1)
            self.edits[field] = edit

    def sync(self) -> None:
        """Show the method and parameters :attr:`current` reports now."""
        self.picker.blockSignals(True)
        self.picker.setCurrentText(self.current()[0])
        self.picker.blockSignals(False)
        self._build_params(self.current()[0])

    def values(self) -> tuple[str, dict]:
        """The chosen detector and its parameters; ValueError if one is not a number."""
        return self.picker.currentText(), {
            k: float(e.text()) for k, e in self.edits.items()
        }


def _card(title: str) -> tuple[QtWidgets.QFrame, QtWidgets.QLabel, QtWidgets.QLabel]:
    frame = QtWidgets.QFrame(objectName="card")
    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(10, 6, 10, 6)
    layout.setSpacing(0)
    layout.addWidget(QtWidgets.QLabel(title, objectName="title"))
    value = QtWidgets.QLabel("-", objectName="value")
    sub = QtWidgets.QLabel("", objectName="sub")
    layout.addWidget(value)
    layout.addWidget(sub)
    for label in frame.findChildren(QtWidgets.QLabel):
        label.setSizePolicy(  # let the strip shrink on a narrow window
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred
        )
    frame.setMinimumWidth(70)
    return frame, value, sub


def _step_hist(plot: pg.PlotItem, colour: str) -> pg.PlotCurveItem:
    """A stepped, filled, antialiased histogram: set with bin edges and counts."""
    item = pg.PlotCurveItem(
        pen=pg.mkPen(None),  # an outline would draw the empty bins' baseline
        brush=pg.mkBrush(_alpha(colour, 200)),
        fillLevel=0,
        antialias=True,
    )
    item.setData([0.0, 1.0], [0.0], stepMode=True)
    plot.addItem(item)
    return item


def _hist_plot(title: str, xlabel: str) -> pg.PlotItem:
    plot = pg.PlotItem()
    plot.setMenuEnabled(False)
    plot.setMouseEnabled(x=False, y=False)
    plot.hideButtons()
    plot.setTitle(title, color=THEME["muted"], size="9pt")
    plot.showGrid(y=True, alpha=0.25)
    plot.getAxis("bottom").enableAutoSIPrefix(False)
    plot.getAxis("left").setWidth(46)
    return plot


class EditorWindow(QtWidgets.QMainWindow):
    def __init__(self, state: TuneState, window_s: float = 5.0) -> None:
        super().__init__()
        self.state = state
        self.width_s = self.initial_s = window_s
        self.band = list(RATE_BAND_HZ)
        self.cursor_x: float | None = None
        self._busy_sync = False
        empty = np.empty(0)
        self._rate_full = (empty, empty)
        self._amp_full = (empty, empty)
        self._bin_key: float | None = None
        self._series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._skip_prompt = False
        self.span_start: float | None = None
        self.span_items: list[pg.LinearRegionItem] = []
        self._jobs: list[_Job] = []
        screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.resize(min(1700, screen.width()), min(900, screen.height()))
        self._build()
        self._timer = QtCore.QTimer(singleShot=True, interval=25)
        self._timer.timeout.connect(self.refresh_analysis)
        self.load_clip()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not getattr(self, "_shown", False):
            self._shown = True  # the view box reports a default range once laid out
            lo = float(self.state.trace().t[0])
            QtCore.QTimer.singleShot(
                0, lambda: self.set_window(lo, lo + self.initial_s)
            )

    # ---- construction ----------------------------------------------------------

    def _build(self) -> None:
        split = QtWidgets.QSplitter()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(split, "Editor")
        self.tabs.addTab(self._build_methods(), "Methods")
        self.tabs.currentChanged.connect(self._tab_changed)
        self.setCentralWidget(self.tabs)
        split.addWidget(self._build_left())
        split.addWidget(self._build_centre())
        split.addWidget(self._build_right())
        split.setStretchFactor(1, 1)
        split.setSizes([340, 900, 380])
        split.setChildrenCollapsible(False)
        self.statusBar().setStyleSheet(f"color: {THEME['muted']};")
        hint = QtWidgets.QLabel(
            "wheel: zoom | shift+wheel: pan | drag: pan | x: span start/end | right-click: delete span | n/p: clip",
            objectName="sub",
        )
        self.statusBar().addPermanentWidget(hint)
        self._build_menus()

    def _build_left(self) -> QtWidgets.QWidget:
        panes = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.clips = QtWidgets.QListWidget()
        self.clips.addItems([c.name for c in self.state.clips])
        self.clips.setTextElideMode(QtCore.Qt.TextElideMode.ElideLeft)
        self.clips.currentRowChanged.connect(self._clip_picked)
        panes.addWidget(self.clips)

        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(0, 4, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        self.show_kind = {}
        for name, label in (
            ("inhale", "inhale"),
            ("exhale", "exhale"),
            ("violations only", "violations"),
        ):
            check = QtWidgets.QCheckBox(label)
            check.setChecked(name != "violations only")
            check.toggled.connect(self.refresh_list)
            self.show_kind[name] = check
            bar.addWidget(check)
        layout.addLayout(bar)
        self.model = EventModel()
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.horizontalHeader().setStretchLastSection(True)
        for column, width in enumerate((80, 70)):
            self.table.setColumnWidth(column, width)
        self.table.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setShowGrid(False)
        self.table.selectionModel().currentRowChanged.connect(self._event_picked)
        layout.addWidget(self.table)
        self.list_tabs = QtWidgets.QTabWidget()
        self.list_tabs.addTab(box, "Events")
        self.list_tabs.addTab(self._build_span_list(), "Spans")
        panes.addWidget(self.list_tabs)
        panes.setSizes([200, 700])
        return panes

    def _build_span_list(self) -> QtWidgets.QWidget:
        """The clip's excluded spans: pick one to look at it, edit its start or end
        by typing, remove it."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        self.span_table = QtWidgets.QTableWidget(0, 3)
        self.span_table.setHorizontalHeaderLabels(
            ["start (s)", "end (s)", "length (s)"]
        )
        self.span_table.verticalHeader().hide()
        self.span_table.verticalHeader().setDefaultSectionSize(24)
        self.span_table.horizontalHeader().setStretchLastSection(True)
        self.span_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Interactive
        )
        self.span_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.span_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.span_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.span_table.setShowGrid(False)
        self.span_table.itemChanged.connect(self._span_edited)
        self.span_table.currentCellChanged.connect(self._span_picked)
        layout.addWidget(self.span_table, 1)
        buttons = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add span")
        add.setToolTip("A span in the middle of the view; then type its start and end")
        add.clicked.connect(self.add_span_here)
        remove = QtWidgets.QPushButton("Delete selected")
        remove.clicked.connect(self.delete_selected_spans)
        buttons.addWidget(add)
        buttons.addWidget(remove)
        layout.addLayout(buttons)
        layout.addWidget(
            QtWidgets.QLabel("double-click a start or end to type it", objectName="sub")
        )
        return page

    def _build_centre(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        self.cards = {}
        for key, title in (
            ("inhales", "INHALES"),
            ("rate", "MEAN RATE"),
            ("outside", "OUT OF BAND"),
            ("ii", "INHALE > INHALE"),
            ("ee", "EXHALE > EXHALE"),
        ):
            frame, value, sub = _card(title)
            bar.addWidget(frame)
            self.cards[key] = (value, sub)
        bar.addStretch(1)
        bar.addWidget(QtWidgets.QLabel("Window (s)"))
        self.width_box = QtWidgets.QDoubleSpinBox()
        self.width_box.setRange(MIN_WINDOW_S, 600)
        self.width_box.setDecimals(1)
        self.width_box.setSingleStep(1)
        self.width_box.setValue(self.width_s)
        self.width_box.valueChanged.connect(self._width_typed)
        bar.addWidget(self.width_box)
        layout.addLayout(bar)

        limits = QtWidgets.QHBoxLayout()
        limits.addWidget(QtWidgets.QLabel("y (empty = fit)", objectName="sub"))
        self.limit_edits: dict[
            str, tuple[QtWidgets.QLineEdit, QtWidgets.QLineEdit]
        ] = {}
        for name, title in (
            ("raw", "raw"),
            ("proc", "proc"),
            ("rate", "rate"),
            ("amp", "amp"),
        ):
            limits.addWidget(QtWidgets.QLabel(title))
            pair = []
            for hint in ("min", "max"):
                edit = QtWidgets.QLineEdit()
                edit.setPlaceholderText(hint)
                edit.setFixedWidth(52)
                edit.setValidator(QtGui.QDoubleValidator())
                edit.editingFinished.connect(lambda n=name: self.apply_limits(n))
                limits.addWidget(edit)
                pair.append(edit)
            self.limit_edits[name] = (pair[0], pair[1])
            limits.addSpacing(6)
        limits.addStretch(1)
        layout.addLayout(limits)

        pg.setConfigOptions(
            antialias=False, background=THEME["card"], foreground=THEME["muted"]
        )
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        self.raw = self.graphics.addPlot(row=0, col=0, viewBox=SpanBox(self))
        self.proc = self.graphics.addPlot(row=1, col=0, viewBox=SpanBox(self))
        self.rate = self.graphics.addPlot(row=2, col=0, viewBox=SpanBox(self))
        self.seek = self.graphics.addPlot(row=3, col=0, viewBox=SeekBox(self))
        for row, stretch in enumerate((3, 3, 2, 1)):
            self.graphics.ci.layout.setRowStretchFactor(row, stretch)
        self.proc.setXLink(self.raw)
        self.rate.setXLink(self.raw)
        for plot in (self.raw, self.proc, self.rate, self.seek):
            plot.showGrid(x=True, y=True, alpha=0.2)
            plot.getAxis("left").setWidth(62)
            plot.hideButtons()
        for plot in (self.raw, self.proc):
            vb = plot.getViewBox()
            vb.setMouseEnabled(x=True, y=False)
            vb.enableAutoRange(x=False, y=True)
            vb.setAutoVisible(y=True)
            vb.setDefaultPadding(0.12)
        rate_vb = self.rate.getViewBox()
        rate_vb.setMouseEnabled(x=True, y=False)
        rate_vb.enableAutoRange(x=False, y=True)
        rate_vb.setAutoVisible(y=True)
        rate_vb.setDefaultPadding(0.15)
        rate_vb.setLimits(yMin=0, yMax=HIST_MAX_HZ)
        self.raw.getAxis("bottom").setStyle(showValues=False)
        self.proc.getAxis("bottom").setStyle(showValues=False)
        self.seek.getAxis("left").setStyle(showValues=False)

        self.raw_curve = self.raw.plot(
            pen=pg.mkPen(THEME["raw"], width=2), skipFiniteCheck=True
        )
        self.proc_curve = self.proc.plot(
            pen=pg.mkPen(THEME["ink"], width=2), skipFiniteCheck=True
        )
        self.seek_lo = self.seek.plot(pen=pg.mkPen(None), skipFiniteCheck=True)
        self.seek_hi = self.seek.plot(pen=pg.mkPen(None), skipFiniteCheck=True)
        self.seek.addItem(
            pg.FillBetweenItem(
                self.seek_lo, self.seek_hi, brush=pg.mkBrush(_alpha(THEME["raw"], 150))
            )
        )
        self.rate_curve = self.rate.plot(
            pen=pg.mkPen(THEME["accent"], width=3),
            stepMode="left",
            skipFiniteCheck=True,
        )
        self.amp_vb = pg.ViewBox(enableMenu=False)
        self.amp_vb.setMouseEnabled(x=False, y=False)
        self.amp_vb.setXLink(self.rate)
        self.amp_vb.enableAutoRange(x=False, y=True)
        self.amp_vb.setAutoVisible(y=True)
        self.amp_vb.setDefaultPadding(0.15)
        self.rate.showAxis("right")
        self.rate.scene().addItem(self.amp_vb)
        self.rate.getAxis("right").linkToView(self.amp_vb)
        self.rate.getAxis("right").setPen(AMPLITUDE_COLOUR)
        for plot in (self.raw, self.proc, self.seek, self.rate):
            right = plot.getAxis("right")  # equal gutters keep the time axes aligned
            right.setWidth(70)
            if plot is not self.rate:
                plot.showAxis("right")
                right.setStyle(showValues=False, tickLength=0)
                right.setPen(THEME["line"])
        self.rate.getViewBox().sigResized.connect(self._sync_amplitude)
        self.amp_curve = pg.PlotDataItem(
            pen=pg.mkPen(AMPLITUDE_COLOUR, width=2), skipFiniteCheck=True
        )
        self.amp_curve.setClipToView(True)
        self.amp_vb.addItem(self.amp_curve)
        self.rate_band = pg.LinearRegionItem(
            self.band,
            orientation="horizontal",
            movable=False,
            brush=pg.mkBrush(_alpha(THEME["accent"], 28)),
            pen=pg.mkPen(None),
        )
        self.rate_band.setZValue(-20)
        for line in self.rate_band.lines:
            line.setPen(pg.mkPen(THEME["muted"], style=QtCore.Qt.PenStyle.DashLine))
        self.rate.addItem(self.rate_band, ignoreBounds=True)
        self.rate_flags = pg.ScatterPlotItem(
            symbol="o", size=7, pen=None, brush=pg.mkBrush(THEME["bad"])
        )
        self.rate.addItem(self.rate_flags)
        self.cross = [
            pg.InfiniteLine(
                angle=90,
                pen=pg.mkPen(THEME["muted"], width=2, style=QtCore.Qt.PenStyle.DotLine),
            )
            for _ in range(3)
        ]
        for plot, line in zip((self.raw, self.proc, self.rate), self.cross):
            line.hide()
            plot.addItem(line, ignoreBounds=True)
        for curve in (self.raw_curve, self.proc_curve):
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")

        self.markers: dict[tuple[str, str], pg.ScatterPlotItem] = {}
        for name, plot, size in (("raw", self.raw, 8), ("proc", self.proc, 11)):
            for kind in POINT_KINDS:
                colour = THEME[kind]
                item = pg.ScatterPlotItem(
                    symbol=SYMBOLS[kind],
                    size=size,
                    pen=pg.mkPen(colour, width=1.6),
                    brush=pg.mkBrush(colour),
                )
                item.setZValue(5)
                plot.addItem(item)
                self.markers[(name, kind)] = item
        self.flags = pg.ScatterPlotItem(
            symbol="o", size=20, pen=pg.mkPen(THEME["bad"], width=2), brush=None
        )
        self.flags.setZValue(4)
        self.proc.addItem(self.flags)
        self.seek_flags = pg.ScatterPlotItem(
            symbol="o", size=6, pen=None, brush=pg.mkBrush(THEME["bad"])
        )
        self.seek.addItem(self.seek_flags)

        for plot, entries in (
            (self.raw, [(self.raw_curve, "raw")]),
            (
                self.proc,
                [(self.proc_curve, "processed (z)"), (self.flags, "out of band")],
            ),
            (
                self.rate,
                [
                    (self.rate_curve, "rate (Hz)"),
                    (self.amp_curve, "raw amplitude"),
                    (self.rate_flags, "out of band"),
                ],
            ),
        ):
            legend = plot.addLegend(offset=(8, 4), labelTextColor=THEME["muted"])
            legend.setBrush(pg.mkBrush(_alpha(THEME["card"], 120)))
            legend.setPen(pg.mkPen(None))
            for item, name in entries:
                legend.addItem(item, name)

        self.pending = [
            pg.InfiniteLine(
                angle=90,
                pen=pg.mkPen(THEME["span"], width=2, style=QtCore.Qt.PenStyle.DashLine),
            )
            for _ in range(2)
        ]
        for plot, line in zip((self.raw, self.proc), self.pending):
            line.hide()
            plot.addItem(line, ignoreBounds=True)

        self.region = pg.LinearRegionItem(
            brush=pg.mkBrush(_alpha(THEME["accent"], 56)),
            pen=pg.mkPen(THEME["accent"], width=1.5),
        )
        self.region.setZValue(10)
        self.seek.addItem(self.region)
        self.region.sigRegionChanged.connect(self._region_moved)
        self.raw.getViewBox().sigXRangeChanged.connect(self._range_changed)
        self.graphics.scene().sigMouseMoved.connect(self._mouse_moved)
        layout.addWidget(self.graphics)
        return page

    def _build_right(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        hists = pg.GraphicsLayoutWidget()
        self.rate_plot = _hist_plot("instantaneous breathing rate (Hz)", "")
        self.ie_plot = _hist_plot("inhale to exhale (s)", "")
        self.ei_plot = _hist_plot("exhale to inhale (s)", "")
        for row, plot in enumerate((self.rate_plot, self.ie_plot, self.ei_plot)):
            hists.addItem(plot, row=row, col=0)
        self.rate_in = _step_hist(self.rate_plot, THEME["accent"])
        self.rate_out = _step_hist(self.rate_plot, THEME["bad"])
        self.rate_mean = pg.InfiniteLine(angle=90, pen=pg.mkPen(THEME["ink"], width=2))
        self.rate_plot.addItem(self.rate_mean)
        self.band_lines = []
        for value in self.band:
            line = pg.InfiniteLine(
                value,
                angle=90,
                movable=True,
                bounds=(0, HIST_MAX_HZ),
                pen=pg.mkPen(
                    THEME["muted"], width=1.5, style=QtCore.Qt.PenStyle.DashLine
                ),
                hoverPen=pg.mkPen(THEME["accent"], width=2.5),
                label="{value:.1f}",
                labelOpts={"position": 0.92, "color": THEME["muted"]},
            )
            line.sigPositionChanged.connect(self._band_dragged)
            self.rate_plot.addItem(line)
            self.band_lines.append(line)
        self.rate_plot.setXRange(0, HIST_MAX_HZ, padding=0)
        self.hist_items = {}
        for key, plot in (("ie", self.ie_plot), ("ei", self.ei_plot)):
            bars = _step_hist(plot, THEME["accent"])
            median = pg.InfiniteLine(angle=90, pen=pg.mkPen(THEME["ink"], width=2))
            plot.addItem(median)
            self.hist_items[key] = (bars, median)
        hists.setMinimumHeight(420)
        layout.addWidget(hists, 1)
        reset = QtWidgets.QPushButton("Reset band to 2-15 Hz")
        reset.clicked.connect(self._reset_band)
        layout.addWidget(reset)
        self.detector_card = DetectorCard(
            lambda: (self.state.entry.method, self.state.entry.params),
            [("Apply to this clip", self.apply_current, True)],
        )
        layout.addWidget(self.detector_card)
        return page

    # ---- clip, view -----------------------------------------------------------------

    def load_clip(self) -> None:
        trace = self.state.trace()
        raw = trace.x if trace.raw is None else trace.raw
        self.raw_curve.setData(trace.t, raw)
        self.proc_curve.setData(trace.t, trace.x)
        tt, floor, ceil = _envelope(trace.t, trace.x, SEEKER_BUCKETS)
        window = max(3, round(AMPLITUDE_WINDOW_S * trace.fs))
        rolling = pd.Series(raw).rolling(window, center=True, min_periods=1)
        amplitude = (rolling.quantile(0.95) - rolling.quantile(0.05)).to_numpy()
        step = max(1, round(trace.fs / AMPLITUDE_HZ))
        self._amp_full = (trace.t[::step], amplitude[::step])
        self._series.update(raw=(trace.t, raw), proc=(trace.t, trace.x))
        self._rebin(force=True)
        self._sync_amplitude()
        self.seek_lo.setData(tt, floor)
        self.seek_hi.setData(tt, ceil)
        self.seek.setYRange(float(floor.min()), float(ceil.max()), padding=0.1)
        lo, hi = float(trace.t[0]), float(trace.t[-1])
        for plot in (self.raw, self.proc):
            plot.getViewBox().setLimits(
                xMin=lo, xMax=hi, minXRange=MIN_WINDOW_S, maxXRange=hi - lo
            )
        self.seek.getViewBox().setLimits(xMin=lo, xMax=hi)
        self.seek.setXRange(lo, hi, padding=0)
        self.region.setBounds((lo, hi))
        self.set_window(lo, min(hi, lo + self.width_s))
        self.detector_card.sync()
        self.span_start = None
        for line in self.pending:
            line.hide()
        self.clips.blockSignals(True)
        self.clips.setCurrentRow(self.state.index)
        self.clips.blockSignals(False)
        self.refresh_markers()
        self.refresh_analysis()
        self._refresh_clip_names()

    def set_window(self, lo: float, hi: float) -> None:
        self.raw.setXRange(lo, hi, padding=0)

    def centre_on(self, t: float) -> None:
        lo, hi = self.raw.getViewBox().viewRange()[0]
        half = (hi - lo) / 2
        self.set_window(t - half, t + half)

    def _range_changed(self, _vb, rng) -> None:
        if self._busy_sync:
            return
        self._busy_sync = True
        lo, hi = rng
        self.width_s = hi - lo
        self.region.setRegion((lo, hi))
        self.width_box.blockSignals(True)
        self.width_box.setValue(self.width_s)
        self.width_box.blockSignals(False)
        self._busy_sync = False
        self._rebin()
        show = self.width_s <= MARKER_MAX_WINDOW_S
        for item in self.markers.values():
            item.setVisible(show)
        self._follow_partial_limits()

    def _region_moved(self) -> None:
        if self._busy_sync:
            return
        self._busy_sync = True
        lo, hi = self.region.getRegion()
        self.raw.setXRange(lo, hi, padding=0)
        self.width_s = hi - lo
        self._busy_sync = False

    def _width_typed(self, width: float) -> None:
        lo, hi = self.raw.getViewBox().viewRange()[0]
        mid = (lo + hi) / 2
        self.set_window(mid - width / 2, mid + width / 2)

    def _rebin(self, force: bool = False) -> None:
        """Draw the rate and amplitude lines binned to the window: about
        :data:`SERIES_BINS` bins across it (a power of two of seconds, so panning
        never re-bins), unbinned once a bin would hold under a breath."""
        width = 2 ** round(np.log2(max(self.width_s / SERIES_BINS, 1e-3)))
        if width == self._bin_key and not force:
            return
        self._bin_key = width
        t0 = float(self.state.trace().t[0])
        onsets, rates = self._rate_full
        if len(rates) and width > float(np.median(np.diff(onsets))):
            x, y = _binned(onsets[:-1], rates, t0, width)
            self.rate_curve.setData(x, y, stepMode=False)
            self._series["rate"] = (x, y)
        elif len(rates):
            self.rate_curve.setData(
                onsets, np.append(rates, rates[-1]), stepMode="left"
            )
            self._series["rate"] = (onsets[:-1], rates)
        else:
            self.rate_curve.setData([], [])
            self._series["rate"] = (np.empty(0), np.empty(0))
        t, amp = self._amp_full
        if len(amp) and width > 2 / AMPLITUDE_HZ:
            t, amp = _binned(t, amp, t0, width, median=True)
        self.amp_curve.setData(t, amp)
        self._series["amp"] = (t, amp)
        self._follow_partial_limits()

    def _view_box(self, name: str) -> pg.ViewBox:
        return {
            "raw": self.raw.getViewBox(),
            "proc": self.proc.getViewBox(),
            "rate": self.rate.getViewBox(),
            "amp": self.amp_vb,
        }[name]

    def apply_limits(self, name: str) -> None:
        """Fix a plot's y range to the typed min and max; an empty end follows the
        data in view, and with both empty the plot fits freely."""
        vb = self._view_box(name)
        texts = [e.text().strip() for e in self.limit_edits[name]]
        low, high = (float(t) if t else None for t in texts)
        if low is None and high is None:
            vb.enableAutoRange(y=True)
            vb.setAutoVisible(y=True)
            return
        vb.enableAutoRange(y=False)
        if low is None or high is None:
            t, y = self._series.get(name, (np.empty(0), np.empty(0)))
            x0, x1 = self.raw.getViewBox().viewRange()[0]
            seen = y[np.searchsorted(t, x0) : np.searchsorted(t, x1)]
            if not len(seen):
                return
            pad = 0.12 * (float(seen.max()) - float(seen.min()) or 1.0)
            low = float(seen.min()) - pad if low is None else low
            high = float(seen.max()) + pad if high is None else high
        if high > low:
            vb.setYRange(low, high, padding=0)

    def _follow_partial_limits(self) -> None:
        for name, pair in self.limit_edits.items():
            if bool(pair[0].text().strip()) != bool(pair[1].text().strip()):
                self.apply_limits(name)

    def _sync_amplitude(self) -> None:
        self.amp_vb.setGeometry(self.rate.getViewBox().sceneBoundingRect())
        self.amp_vb.linkedViewChanged(self.rate.getViewBox(), self.amp_vb.XAxis)

    def _mouse_moved(self, pos) -> None:
        for plot in (self.raw, self.proc, self.rate):
            if plot.sceneBoundingRect().contains(pos):
                self.cursor_x = plot.getViewBox().mapSceneToView(pos).x()
                for line in self.cross:
                    line.setValue(self.cursor_x)
                    line.show()
                return
        for line in self.cross:
            line.hide()

    def _clip_picked(self, row: int) -> None:
        if row >= 0 and row != self.state.index:
            self.state.select(row)
            self.load_clip()

    def _refresh_clip_names(self) -> None:
        for i, clip in enumerate(self.state.clips):
            method, _ = self.state.describe(i)
            mark = "\u25cf " if i in self.state.dirty else ""
            rejected = self.state.entries[i].rejected
            item = self.clips.item(i)
            item.setText(
                f"{mark}{clip.name}   [excluded]"
                if rejected
                else f"{mark}{clip.name}   [{method}]"
            )
            item.setForeground(
                QtGui.QBrush(QtGui.QColor(THEME["muted" if rejected else "ink"]))
            )
            font = item.font()
            font.setStrikeOut(rejected)
            item.setFont(font)
        mark = "*" if self.state.modified else ""
        self.setWindowTitle(f"{mark}Breathing event detectors - {self.state.path}")

    def _reload_clips(self) -> None:
        """The list of clips changed: rebuild the list widget and show a clip."""
        self.clips.blockSignals(True)
        self.clips.clear()
        self.clips.addItems([c.name for c in self.state.clips])
        self.clips.blockSignals(False)
        self.load_clip()
        if self.tabs.currentIndex() == 1:
            self.refresh_table()

    # ---- drawing -------------------------------------------------------------------------

    def refresh_spans(self) -> None:
        """Draw the clip's excluded spans on the traces, rate plot and seeker."""
        self.refresh_span_list()
        for item in self.span_items:
            if item.scene() is not None:
                item.scene().removeItem(item)
        self.span_items.clear()
        brush = pg.mkBrush(_alpha(THEME["span"], 90))
        for start, end in self.state.spans():
            for plot in (self.raw, self.proc, self.rate, self.seek):
                item = pg.LinearRegionItem(
                    (start, end), movable=False, brush=brush, pen=pg.mkPen(None)
                )
                item.setZValue(-10)
                for line in item.lines:
                    line.setPen(pg.mkPen(None))
                plot.addItem(item, ignoreBounds=True)
                self.span_items.append(item)

    def refresh_span_list(self) -> None:
        """Fill the Spans tab from the clip's spans (sorted by start)."""
        spans = self.state.spans()
        table = self.span_table
        table.blockSignals(True)
        table.setRowCount(len(spans))
        for k, (start, end) in enumerate(spans):
            for column, value in enumerate((start, end, end - start)):
                item = QtWidgets.QTableWidgetItem(f"{value:.3f}")
                item.setTextAlignment(
                    QtCore.Qt.AlignmentFlag.AlignRight
                    | QtCore.Qt.AlignmentFlag.AlignVCenter
                )
                if column == 2:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                table.setItem(k, column, item)
        table.blockSignals(False)
        self.list_tabs.setTabText(1, f"Spans ({len(spans)})" if len(spans) else "Spans")

    def _span_picked(
        self, row: int, _column: int, _old_row: int, _old_col: int
    ) -> None:
        spans = self.state.spans()
        if 0 <= row < len(spans):
            self.centre_on(float(spans[row].mean()))

    def _span_edited(self, item) -> None:
        """A start or end was typed: apply it, or say why not and put it back."""
        try:
            k = item.row()
            start = float(self.span_table.item(k, 0).text())
            end = float(self.span_table.item(k, 1).text())
            self.state.update_span(k, start, end)
        except ValueError as exc:
            self.statusBar().showMessage(f"span not changed: {exc}")
            self.refresh_span_list()
            return
        self._spans_changed()

    def add_span_here(self) -> None:
        lo, hi = self.raw.getViewBox().viewRange()[0]
        width = hi - lo
        self.state.exclude(lo + width / 4, hi - width / 4)
        self._spans_changed()

    def delete_selected_spans(self) -> None:
        rows = sorted(
            {r.row() for r in self.span_table.selectionModel().selectedRows()}
        )
        if rows and self.state.remove_spans(rows):
            self._spans_changed()

    def _spans_changed(self) -> None:
        self.refresh_spans()
        self.refresh_analysis()
        self._refresh_clip_names()
        if self.tabs.currentIndex() == 1:
            self.refresh_table()

    def mark_span(self) -> None:
        """Start an excluded span at the cursor, or end the one that was started."""
        if self.cursor_x is None:
            self.statusBar().showMessage("point at a plot first")
            return
        if self.span_start is None:
            self.span_start = self.cursor_x
            for line in self.pending:
                line.setValue(self.span_start)
                line.show()
            self.statusBar().showMessage("span started: x again to end it")
            return
        start, self.span_start = self.span_start, None
        for line in self.pending:
            line.hide()
        if abs(self.cursor_x - start) > 1e-3:
            self.state.exclude(start, self.cursor_x)
            self._spans_changed()

    def delete_span(self, t: float | None = None) -> None:
        t = self.cursor_x if t is None else t
        if t is not None and self.state.delete_span(t):
            self._spans_changed()

    def undo_span(self) -> None:
        if self.state.undo_span():
            self._spans_changed()

    def refresh_markers(self) -> None:
        self.refresh_spans()
        trace, rows = self.state.trace(), self.state.rows()
        raw = trace.x if trace.raw is None else trace.raw
        for kind in POINT_KINDS:
            part = rows.loc[rows["kind"] == kind, "time"].to_numpy()
            for name, y in (("raw", raw), ("proc", trace.x)):
                self.markers[(name, kind)].setData(part, np.interp(part, trace.t, y))

    def refresh_analysis(self) -> None:
        """Stats, histograms, flags and the event list, for the current band."""
        stats = self.state.breath_stats(band=tuple(self.band))
        lo, hi = self.band
        rate = stats["mean"]
        self.cards["inhales"][0].setText(str(stats["inhales"]))
        self.cards["inhales"][1].setText(f"{stats['exhales']} exhales")
        self.cards["rate"][0].setText("-" if np.isnan(rate) else f"{rate:.2f} Hz")
        self.cards["rate"][1].setText(f"{len(stats['rates'])} breaths")
        self.cards["outside"][0].setText(str(stats["outside"]))
        self.cards["outside"][1].setText(f"outside {lo:.1f}-{hi:.1f} Hz")
        for key, count in (
            ("outside", stats["outside"]),
            ("ii", stats["ii_breaks"]),
            ("ee", stats["ee_breaks"]),
        ):
            self.cards[key][0].setStyleSheet(
                f"color: {THEME['bad'] if count else THEME['ink']};"
            )
        self.cards["ii"][0].setText(str(stats["ii_breaks"]))
        self.cards["ii"][1].setText("without an exhale")
        self.cards["ee"][0].setText(str(stats["ee_breaks"]))
        self.cards["ee"][1].setText("without an inhale")

        edges = np.arange(0, HIST_MAX_HZ + 0.5, 0.5)
        counts, _ = np.histogram(np.clip(stats["rates"], 0, HIST_MAX_HZ), edges)
        centres = edges[:-1] + 0.25
        inside = (centres >= lo) & (centres <= hi)
        self.rate_in.setData(edges, np.where(inside, counts, 0), stepMode=True)
        self.rate_out.setData(edges, np.where(inside, 0, counts), stepMode=True)
        self.rate_mean.setVisible(not np.isnan(rate))
        if not np.isnan(rate):
            self.rate_mean.setValue(rate)
        for key in ("ie", "ei"):
            bars, median = self.hist_items[key]
            data = stats[key]
            if len(data):
                top = max(0.2, float(np.percentile(data, 99)) * 1.1)
                bins = np.linspace(0, top, 41)
                counts, _ = np.histogram(np.clip(data, 0, top), bins)
                bars.setData(bins, counts, stepMode=True)
                median.setValue(float(np.median(data)))
                median.show()
            else:
                bars.setData([0.0, 1.0], [0.0], stepMode=True)
                median.hide()
        trace = self.state.trace()
        flagged = stats["flagged"]
        self.flags.setData(flagged, np.interp(flagged, trace.t, trace.x))
        self.seek_flags.setData(flagged, np.interp(flagged, trace.t, trace.x))
        onsets = self.state.times("inhale")
        if len(onsets) > 1:
            rates = np.minimum(breath_rates(onsets), HIST_MAX_HZ)
            self._rate_full = (onsets, rates)
            bad = (rates < lo) | (rates > hi)
            self.rate_flags.setData((onsets[:-1] + onsets[1:])[bad] / 2, rates[bad])
        else:
            self._rate_full = (np.empty(0), np.empty(0))
            self.rate_flags.setData([], [])
        self.rate_band.setRegion((lo, hi))
        self._rebin(force=True)
        self._follow_partial_limits()
        self.refresh_list(stats)
        method, changed = self.state.describe(self.state.index)
        self.statusBar().showMessage(
            f"{self.state.clip.name}   {method} {changed}   "
            f"{len(self.state.spans())} excluded span(s)   "
            f"{'VIDEO EXCLUDED   ' if self.state.entry.rejected else ''}"
            f"{'UNSAVED' if self.state.modified else 'saved'}"
        )

    def refresh_list(self, stats: dict | None = None) -> None:
        if not isinstance(stats, dict):
            stats = self.state.breath_stats(band=tuple(self.band))
        rows = self.state.rows()
        onsets = self.state.times("inhale")
        flagged = stats["flagged"]
        only_bad = self.show_kind["violations only"].isChecked()
        out, tags, times = [], [], []
        later = np.searchsorted(onsets, rows["time"].to_numpy(), side="right")
        for pos, (kind, time) in enumerate(zip(rows["kind"], rows["time"])):
            if not self.show_kind[kind].isChecked():
                continue
            bad = kind == "inhale" and bool(np.isin(time, flagged))
            if only_bad and not bad:
                continue
            rate = ""
            if kind == "inhale" and later[pos] < len(onsets):
                rate = f"{1 / (onsets[later[pos]] - time):.2f}"
            out.append((f"{time:.2f}", kind, rate))
            tags.append("bad" if bad else "")
            times.append(float(time))
        self.model.reset(out, tags, times)

    def _event_picked(self, current, _previous) -> None:
        if current.isValid():
            self.centre_on(self.model.times[current.row()])

    # ---- band ----------------------------------------------------------------------------

    def _band_dragged(self) -> None:
        self.band = sorted(line.value() for line in self.band_lines)
        self._timer.start()

    def _reset_band(self) -> None:
        for line, value in zip(self.band_lines, RATE_BAND_HZ):
            line.setValue(value)

    # ---- actions -------------------------------------------------------------------------

    # ---- methods tab -----------------------------------------------------------

    TABLE_HEAD = (
        "use",
        "video",
        "method",
        "parameters",
        "inhales",
        "exhales",
        "mean rate (Hz)",
        "out of band",
        "I>E median (s)",
        "E>I median (s)",
        "I>I breaks",
        "E>E breaks",
        "spans",
    )

    def _build_methods(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.methods_card = DetectorCard(
            lambda: (self.state.entry.method, self.state.entry.params),
            [
                ("Apply to selected", self.apply_selected, False),
                ("Apply to all", self.apply_all, True),
            ],
            horizontal=True,
        )
        layout.addWidget(self.methods_card)
        self.summary = QtWidgets.QTableWidget(0, len(self.TABLE_HEAD))
        self.summary.setHorizontalHeaderLabels(self.TABLE_HEAD)
        self.summary.verticalHeader().hide()
        self.summary.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.summary.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.summary.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.summary.setShowGrid(False)
        header = self.summary.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        header.setMinimumSectionSize(40)
        header.setSectionsClickable(True)
        header.sectionClicked.connect(self._sort_clicked)
        self._sort: tuple[int, QtCore.Qt.SortOrder] | None = None
        self._table_rows: list[dict] = []
        self._row_clip: list[int] = []
        self._widths_set = False
        self.summary.cellDoubleClicked.connect(self._summary_opened)
        self.summary.itemChanged.connect(self._use_toggled)
        self.summary.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.summary.customContextMenuRequested.connect(self._table_menu)
        layout.addWidget(self.summary, 1)
        layout.addWidget(
            QtWidgets.QLabel(
                "untick a video to exclude it | top row: mean over videos | blue method: own setting, grey: list default | click a header to sort (descending, ascending, off) | drag header edges to resize | double-click a video to view it",
                objectName="sub",
            )
        )
        return page

    def _work(self, text: str, fn, then, total: int | None = None) -> None:
        """Run fn(progress) off the UI thread behind a progress pop-up; then(error)."""
        total = len(self.state.clips) if total is None else total
        dialog = QtWidgets.QProgressDialog(text, "", 0, total, self)
        dialog.setCancelButton(None)
        dialog.setWindowTitle("Working")
        dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(300)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        job = _Job(fn, reports=True)
        job.signals.progress.connect(lambda done, _total: dialog.setValue(done))

        def finished(error: str) -> None:
            dialog.close()
            self._jobs.remove(job)
            then(error)

        job.signals.done.connect(finished)
        self._jobs.append(job)
        QtCore.QThreadPool.globalInstance().start(job)

    def _tab_changed(self, index: int) -> None:
        if index == 1:
            self.methods_card.sync()
            self.refresh_table()
        self.statusBar().clearMessage()

    def refresh_table(self) -> None:
        rows: list[dict] = []
        band = tuple(self.band)

        def compute(progress) -> None:
            rows.clear()
            for i in range(len(self.state.clips)):
                rejected = self.state.entries[i].rejected
                rows.append(
                    dict.fromkeys(METRICS, np.nan)
                    if rejected
                    else self.state.clip_metrics(i, band)
                )
                progress(i + 1, len(self.state.clips))

        def done(error: str) -> None:
            if error:
                self.statusBar().showMessage(f"metrics failed: {error}")
            else:
                self._fill_table(rows)

        self._work("Computing metrics for every video...", compute, done)

    def _fill_table(self, rows: list[dict]) -> None:
        def cell(text: str, bold: bool = False, bad: bool = False, left: bool = False):
            item = QtWidgets.QTableWidgetItem(text)
            side = (
                QtCore.Qt.AlignmentFlag.AlignLeft
                if left
                else QtCore.Qt.AlignmentFlag.AlignRight
            )
            item.setTextAlignment(side | QtCore.Qt.AlignmentFlag.AlignVCenter)
            if bold:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                item.setBackground(QtGui.QBrush(QtGui.QColor(THEME["line"])))
            if bad:
                item.setForeground(QtGui.QBrush(QtGui.QColor(THEME["bad"])))
            return item

        def fmt(value: float, digits: int) -> str:
            return "-" if np.isnan(value) else f"{value:.{digits}f}"

        def numbers(row: dict, count: int) -> list[str]:
            digits = (count, count, 2, count, 3, 3, count, count, count)
            return [fmt(row[k], d) for k, d in zip(METRICS, digits)]

        flagged = {"outside", "ii_breaks", "ee_breaks"}
        selected = set(self._selected_clips())
        self._table_rows = rows
        order = self._sorted_order(rows)
        self._row_clip = order
        self.summary.setRowCount(len(rows) + 1)
        used = [rows[i] for i in self.state.accepted() if i < len(rows)]
        average = mean_metrics(used) if used else dict.fromkeys(METRICS, np.nan)
        dropped = len(rows) - len(used)
        label = f"mean of {len(used)} videos" + (
            f" ({dropped} excluded)" if dropped else ""
        )
        self.summary.blockSignals(True)
        self.summary.setItem(0, 0, cell("", True))
        self.summary.setItem(0, 1, cell(label, True, left=True))
        for column in (2, 3):
            self.summary.setItem(0, column, cell("", True))
        for column, text in enumerate(numbers(average, 1), start=4):
            self.summary.setItem(0, column, cell(text, True))
        for n, i in enumerate(order, start=1):
            clip, row = self.state.clips[i], rows[i]
            method, changed = self.state.describe(i)
            mark = "\u25cf " if i in self.state.dirty else ""
            tick = cell("")
            tick.setFlags(
                QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
            )
            tick.setCheckState(
                QtCore.Qt.CheckState.Unchecked
                if self.state.entries[i].rejected
                else QtCore.Qt.CheckState.Checked
            )
            self.summary.setItem(n, 0, tick)
            if self.state.entries[i].rejected:
                for column in range(1, len(self.TABLE_HEAD)):
                    gone = cell(
                        f"{mark}{clip.name}"
                        if column == 1
                        else "excluded"
                        if column == 2
                        else "",
                        left=column < 4,
                    )
                    gone.setForeground(QtGui.QBrush(QtGui.QColor(THEME["muted"])))
                    font = gone.font()
                    font.setStrikeOut(column == 1)
                    font.setItalic(column == 2)
                    gone.setFont(font)
                    self.summary.setItem(n, column, gone)
                continue
            self.summary.setItem(n, 1, cell(mark + clip.name, left=True))
            chosen = cell(method, left=True)
            colour = THEME["accent"] if self.state.entries[i].custom else THEME["muted"]
            chosen.setForeground(QtGui.QBrush(QtGui.QColor(colour)))
            self.summary.setItem(n, 2, chosen)
            self.summary.setItem(n, 3, cell(changed, left=True))
            for k, text in enumerate(numbers(row, 0)):
                bad = METRICS[k] in flagged and row[METRICS[k]] > 0
                self.summary.setItem(n, 4 + k, cell(text, bad=bad))
        if not self._widths_set and rows:
            self.summary.resizeColumnsToContents()
            self.summary.setColumnWidth(0, 50)
            for column, least in ((1, 260), (2, 150), (3, 170)):
                self.summary.setColumnWidth(
                    column, max(self.summary.columnWidth(column), least)
                )
            self._widths_set = True
        self.summary.blockSignals(False)
        for n, i in enumerate(order, start=1):
            if i in selected:
                self.summary.selectionModel().select(
                    self.summary.model().index(n, 0),
                    QtCore.QItemSelectionModel.SelectionFlag.Select
                    | QtCore.QItemSelectionModel.SelectionFlag.Rows,
                )

    def _sorted_order(self, rows: list[dict]) -> list[int]:
        """Clip indices in the table's order: as listed, or by the sorted column
        (empty values last); the mean row is not part of it."""
        order = list(range(len(rows)))
        if self._sort is None:
            return order
        column, direction = self._sort

        def value(i: int):
            if column == 0:
                return int(not self.state.entries[i].rejected)
            if column == 1:
                return self.state.clips[i].name.lower()
            if column in (2, 3):
                return self.state.describe(i)[column - 2]
            return rows[i][METRICS[column - 4]]

        present = [i for i in order if not _missing(value(i))]
        absent = [i for i in order if _missing(value(i))]
        present.sort(
            key=value, reverse=direction == QtCore.Qt.SortOrder.DescendingOrder
        )
        return present + absent

    def _sort_clicked(self, column: int) -> None:
        """Click a header: sort by it descending, then ascending, then not at all."""
        if self._sort is None or self._sort[0] != column:
            self._sort = (column, QtCore.Qt.SortOrder.DescendingOrder)
        elif self._sort[1] == QtCore.Qt.SortOrder.DescendingOrder:
            self._sort = (column, QtCore.Qt.SortOrder.AscendingOrder)
        else:
            self._sort = None
        header = self.summary.horizontalHeader()
        header.setSortIndicatorShown(self._sort is not None)
        if self._sort is not None:
            header.setSortIndicator(*self._sort)
        if self._table_rows:
            self._fill_table(self._table_rows)

    def _use_toggled(self, item) -> None:
        """The tick in the first column: unticked videos are excluded from use."""
        n = item.row()
        if item.column() != 0 or not 1 <= n <= len(self._row_clip):
            return
        i = self._row_clip[n - 1]
        use = item.checkState() == QtCore.Qt.CheckState.Checked
        message = ""
        try:
            self.state.set_rejected([i], not use)
        except ValueError as exc:
            message = f"not changed: {exc}"
        else:
            verb = "included again" if use else "excluded"
            message = f"{self.state.clips[i].name} {verb}; save to keep it"
        self._refresh_clip_names()
        self.refresh_analysis()
        self.refresh_table()
        self.statusBar().showMessage(message)

    def _summary_opened(self, row: int, _column: int) -> None:
        if 1 <= row <= len(self._row_clip):
            self.state.select(self._row_clip[row - 1])
            self.load_clip()
            self.tabs.setCurrentIndex(0)

    # ---- applying a method ---------------------------------------------------------

    def _selected_clips(self) -> list[int]:
        rows = {r.row() for r in self.summary.selectionModel().selectedRows()}
        return sorted(
            self._row_clip[r - 1] for r in rows if 1 <= r <= len(self._row_clip)
        )

    def _apply(self, indices: list[int], card: DetectorCard) -> None:
        if not indices:
            self.statusBar().showMessage("no clips selected")
            return
        try:
            name, params = card.values()
            events.detector_params(name, params)
        except ValueError as exc:
            self.statusBar().showMessage(f"bad parameters: {exc}")
            return
        self._work(
            f"Applying {name} to {len(indices)} video(s)...",
            lambda progress: self.state.apply(indices, name, params, progress),
            self._applied,
            total=len(indices),
        )

    def _applied(self, error: str) -> None:
        if error:
            self.statusBar().showMessage(f"detector failed: {error}")
        self.refresh_markers()
        self.refresh_analysis()
        self._refresh_clip_names()
        self.detector_card.sync()
        if self.tabs.currentIndex() == 1:
            self.refresh_table()

    def apply_current(self) -> None:
        self._apply([self.state.index], self.detector_card)

    def apply_selected(self) -> None:
        self._apply(self._selected_clips(), self.methods_card)

    def apply_all(self) -> None:
        self._apply(self.state.accepted(), self.methods_card)

    def _targets(self) -> list[int]:
        """The videos a menu command acts on: the table's selection, else this one."""
        if self.tabs.currentIndex() == 1:
            return self._selected_clips()
        return [self.state.index]

    def set_rejected(self, rejected: bool) -> None:
        """Exclude (or include again) the target videos: an excluded video is never
        used once the list is saved."""
        indices = self._targets()
        if not indices:
            self.statusBar().showMessage("no videos selected")
            return
        try:
            self.state.set_rejected(indices, rejected)
        except ValueError as exc:
            self.statusBar().showMessage(f"not changed: {exc}")
            return
        verb = "excluded" if rejected else "included again"
        self.statusBar().showMessage(f"{len(indices)} video(s) {verb}; save to keep it")
        self._refresh_clip_names()
        self.refresh_analysis()
        if self.tabs.currentIndex() == 1:
            self.refresh_table()

    def _table_menu(self, position) -> None:
        menu = QtWidgets.QMenu(self)
        menu.addAction("Exclude selected videos").triggered.connect(
            lambda: self.set_rejected(True)
        )
        menu.addAction("Include selected videos").triggered.connect(
            lambda: self.set_rejected(False)
        )
        menu.exec(self.summary.viewport().mapToGlobal(position))

    def _apply_from_menu(self, scope: str) -> None:
        on_methods = self.tabs.currentIndex() == 1
        card = self.methods_card if on_methods else self.detector_card
        if scope == "all":
            indices = self.state.accepted()
        elif scope == "selected" and on_methods:
            indices = self._selected_clips()
        else:
            indices = [self.state.index]
        self._apply(indices, card)

    # ---- the list: open, add, remove, save ---------------------------------------------

    def _build_menus(self) -> None:
        bar = self.menuBar()

        def action(menu, text, slot, shortcut=None):
            act = QtGui.QAction(text, self)
            if shortcut:
                act.setShortcut(QtGui.QKeySequence(shortcut))
            act.triggered.connect(lambda _checked=False: slot())
            menu.addAction(act)
            return act

        file_menu = bar.addMenu("&File")
        action(file_menu, "&Open clip list...", self.open_list, "Ctrl+O")
        file_menu.addSeparator()
        action(file_menu, "&Add videos...", self.add_videos, "Ctrl+Shift+A")
        action(file_menu, "&Remove selected videos", self.remove_videos, "Del")
        file_menu.addSeparator()
        action(file_menu, "&Save", self.save, "Ctrl+S")
        file_menu.addSeparator()
        action(file_menu, "&Quit", self.close, "Ctrl+Q")
        edit_menu = bar.addMenu("&Edit")
        action(
            edit_menu,
            "Apply method to this &clip",
            lambda: self._apply_from_menu("clip"),
            "Ctrl+Return",
        )
        action(
            edit_menu,
            "Apply method to &selected clips",
            lambda: self._apply_from_menu("selected"),
            "Ctrl+Alt+Return",
        )
        action(
            edit_menu,
            "Apply method to &all clips",
            lambda: self._apply_from_menu("all"),
            "Ctrl+Shift+Return",
        )
        edit_menu.addSeparator()
        action(
            edit_menu,
            "Exclude video(s) from use",
            lambda: self.set_rejected(True),
            "Ctrl+E",
        )
        action(
            edit_menu,
            "Include video(s) again",
            lambda: self.set_rejected(False),
            "Ctrl+Shift+E",
        )
        edit_menu.addSeparator()
        action(edit_menu, "Mark excluded span start/end", self.mark_span, "X")
        action(edit_menu, "Delete span at cursor", self.delete_span, "Shift+X")
        action(edit_menu, "Delete selected spans", self.delete_selected_spans)
        action(edit_menu, "Undo span change", self.undo_span, "Ctrl+Z")
        view_menu = bar.addMenu("&View")
        action(view_menu, "&Editor tab", lambda: self.tabs.setCurrentIndex(0), "Ctrl+1")
        action(
            view_menu, "&Methods tab", lambda: self.tabs.setCurrentIndex(1), "Ctrl+2"
        )
        view_menu.addSeparator()
        action(view_menu, "&Next clip", lambda: self._step_clip(1), "N")
        action(view_menu, "&Previous clip", lambda: self._step_clip(-1), "P")

    def _step_clip(self, step: int) -> None:
        self.state.select(self.state.index + step)
        self.load_clip()

    def _confirm_discard(self) -> bool:
        if not self.state.modified:
            return True
        answer = QtWidgets.QMessageBox.question(
            self,
            "Unsaved changes",
            "Save the changes to the clip list first?",
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QtWidgets.QMessageBox.StandardButton.Discard

    def open_list(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open clip list",
            str(self.state.path.parent),
            "Clip lists (*.toml)",
        )
        if not path or not self._confirm_discard():
            return
        if open_window(Path(path), self.initial_s, self):
            self._skip_prompt = True
            self.close()

    def add_videos(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Add videos",
            str(self.state.path.parent),
            "Videos (*.mp4 *.avi *.mov *.mkv)",
        )
        if not paths:
            return
        try:
            added = self.state.add_videos([Path(p) for p in paths])
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot add videos", str(exc))
            return
        self.statusBar().showMessage(f"added {len(added)} video(s); save to keep them")
        self._reload_clips()

    def remove_videos(self) -> None:
        indices = (
            self._selected_clips()
            if self.tabs.currentIndex() == 1
            else [self.state.index]
        )
        if not indices:
            return
        names = "\n".join(self.state.clips[i].name for i in indices[:8])
        answer = QtWidgets.QMessageBox.question(
            self,
            "Remove videos",
            f"Remove {len(indices)} video(s) from the clip list?\n\n{names}",
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            self.state.remove(indices)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot remove", str(exc))
            return
        self.statusBar().showMessage(
            f"removed {len(indices)} video(s); save to keep it"
        )
        self._reload_clips()

    def save(self) -> bool:
        try:
            n = self.state.save()
        except Exception as exc:  # noqa: BLE001 - shown to the user
            QtWidgets.QMessageBox.warning(self, "Cannot save", str(exc))
            return False
        self._refresh_clip_names()
        self.refresh_analysis()
        if self.tabs.currentIndex() == 1:
            self.refresh_table()
        self.statusBar().showMessage(
            f"saved {n} clip detector setting(s) to {self.state.path}"
        )
        return True

    def keyPressEvent(self, event) -> None:
        if self.tabs.currentIndex() != 0:
            return super().keyPressEvent(event)
        key = QtCore.Qt.Key
        shift = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        lo, hi = self.raw.getViewBox().viewRange()[0]
        width = hi - lo
        step = width * (1.0 if shift else 0.5)
        k = event.key()
        if k == key.Key_Right:
            self.set_window(lo + step, hi + step)
        elif k == key.Key_Left:
            self.set_window(lo - step, hi - step)
        elif k in (key.Key_Up, key.Key_Down):
            new = width / 1.5 if k == key.Key_Up else width * 1.5
            self.set_window((lo + hi) / 2 - new / 2, (lo + hi) / 2 + new / 2)
        else:
            return super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if not self._skip_prompt and not self._confirm_discard():
            event.ignore()
            return
        event.accept()


_windows: list[EditorWindow] = []
"""Open windows, kept alive for the application's lifetime."""


def open_window(path: Path, window_s: float, parent=None) -> bool:
    """Open *path* in a new window; False (after saying why) if it cannot load."""
    try:
        window = EditorWindow(TuneState(path), window_s)
    except Exception as exc:  # noqa: BLE001 - shown to the user
        QtWidgets.QMessageBox.warning(
            parent, "Cannot open clip list", f"{path}\n\n{exc}"
        )
        return False
    _windows.append(window)
    window.showMaximized()
    return True


def run_ui(path: Path | None, window_s: float = 5.0) -> None:  # pragma: no cover
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(STYLE)
    if path is None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Open clip list", "", "Clip lists (*.toml)"
        )
        if not chosen:
            return
        path = Path(chosen)
    if open_window(path, window_s):
        app.exec()

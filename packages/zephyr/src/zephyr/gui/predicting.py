"""The Predict page: a checkpoint's traces for a clip list, or for videos from any folder
(``clips scan --checkpoint``, their crops placed at the network's size, then ``zephyr
predict``), on request with quality control of each (``--quality-control``),
shown per video with the crop it came from; the Quality tab sets each video's scores
beside the checkpoint's validation windows (:mod:`.quality`)."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .. import features
from ..annotate import open_state
from ..clips import update_clip_list
from ..config import Derive
from ..predict import SETTINGS, SUFFIX
from ..quality_control.metrics import MEASURES
from .boxes import BoxWindow
from .jobs import Job
from .pages import Page
from .project import Project, is_smoke, run_label, run_status
from .quality import GOOD_F1, SHOWN, QualityState
from .widgets import (
    JobPanel,
    TraceView,
    accent,
    ask_cache_space,
    find_data,
    heading,
    note,
)

QC_SUFFIX = ".qc.json"
QC_COLOURS = {"good": (110, 110, 110), "poor": (213, 94, 0), "video": (0, 114, 178)}
"""Overview rows: the reference's good and poor windows, and the folder's videos."""
VIDEO_LIST = "videos.toml"
"""The Videos tab's list, in the predictions folder it fills."""


def _threshold(settings: Path) -> float | None:
    """The onset threshold a folder's inhales were found at (its ``predictions.json``),
    or None when it does not say."""
    try:
        recorded = json.loads(settings.read_text()).get("head_threshold")
    except (OSError, ValueError):
        return None
    return recorded["value"] if recorded else None


def _measure(value: float | None) -> str:
    """A quality-control measure for the verdict line; ``n/a`` when it has none."""
    return "n/a" if value is None else f"{value:.3g}"


class PredictPage(Page):
    title = "Predict"

    def __init__(self, app) -> None:
        super().__init__(app)
        layout = QtWidgets.QHBoxLayout(self)
        holder = QtWidgets.QWidget(maximumWidth=420)
        left = QtWidgets.QVBoxLayout(holder)
        layout.addWidget(holder)
        left.addWidget(heading("Predict"))
        top = QtWidgets.QFormLayout()
        row = QtWidgets.QHBoxLayout()
        self.checkpoint = QtWidgets.QComboBox()
        row.addWidget(self.checkpoint, 1)
        top.addRow("Checkpoint", row)
        browse = QtWidgets.QPushButton("Open a checkpoint...")
        browse.setToolTip("Any best.pt, from this project or another")
        browse.clicked.connect(self._browse)
        top.addRow("", browse)
        left.addLayout(top)
        self.source = QtWidgets.QTabWidget()
        left.addWidget(self.source)
        listed = QtWidgets.QWidget()
        self.source.addTab(listed, "Clip list")
        self.source.addTab(self._build_videos(), "Videos")
        tab = QtWidgets.QVBoxLayout(listed)
        form = QtWidgets.QFormLayout()
        self.clip_list = QtWidgets.QComboBox()
        form.addRow("Clip list", self.clip_list)
        self.groups = QtWidgets.QLineEdit()
        self.groups.setPlaceholderText("all groups")
        form.addRow("Groups", self.groups)
        self.name = QtWidgets.QLineEdit("predictions")
        form.addRow("Into predictions/", self.name)
        self.keep = QtWidgets.QCheckBox(
            "keep the frames in the project's cache (shows the crop; takes space)"
        )
        form.addRow("", self.keep)
        self.qc = QtWidgets.QCheckBox("judge them (quality control)")
        form.addRow("", self.qc)
        tab.addLayout(form)
        tab.addWidget(
            note(
                "The clips need boxes, not a breathing sensor. Without the cache the "
                "frames are decoded into a temporary folder and removed; the crop is "
                "then not shown. Quality control refuses clips the checkpoint trained "
                "on."
            )
        )
        go = accent("Predict")
        go.clicked.connect(self.predict)
        tab.addWidget(go)
        tab.addStretch(1)
        left.addStretch(1)
        self.jobs = JobPanel()
        self.jobs.finished.connect(self._finished)

        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 1)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(heading("Predictions"))
        bar.addSpacing(12)
        bar.addWidget(QtWidgets.QLabel("in predictions/"))
        self.folder = QtWidgets.QComboBox(minimumWidth=200)
        self.folder.currentIndexChanged.connect(lambda _: self.fill_videos())
        bar.addWidget(self.folder)
        bar.addStretch(1)
        right.addLayout(bar)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.browse = QtWidgets.QTabWidget()
        self.videos = QtWidgets.QListWidget()
        self.videos.currentRowChanged.connect(lambda _: self.show_video())
        self.browse.addTab(self.videos, "Videos")
        self.browse.addTab(self._build_quality(), "Quality")
        split.addWidget(self.browse)
        view = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(view)
        column.setContentsMargins(0, 0, 0, 0)
        self.verdict = QtWidgets.QLabel(wordWrap=True)
        self.verdict.setTextFormat(QtCore.Qt.TextFormat.RichText)
        column.addWidget(self.verdict)
        self.trace = TraceView()
        column.addWidget(self.trace, 1)
        # The chosen measure per window, under the onset probability on the trace's own
        # time axis; a click zooms the trace to its window.
        self.strip = pg.PlotWidget()
        self.strip.setMaximumHeight(90)
        self.strip.setMinimumHeight(60)
        self.strip.showGrid(x=True, alpha=0.15)
        self.strip.setXLink(self.trace.plot)
        self.strip.scene().sigMouseClicked.connect(self._strip_clicked)
        self.trace.plots.addWidget(self.strip)
        split.addWidget(view)
        # Its plots ask for more height than a window has: the page fits the window
        # and the splitter shares it, rather than the page scrolling.
        split.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Ignored
        )
        split.setSizes([280, 670])
        self._quality: QualityState | None = None
        self._origin = 0.0
        self._stem: str | None = None
        right.addWidget(split, 1)
        self._queue: list[Job] = []
        self._then = None
        self.box_window: BoxWindow | None = None
        self._reopen = False

    # ---- quality control --------------------------------------------------------

    def _build_quality(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Measure"))
        self.qc_measure = QtWidgets.QComboBox()
        self.qc_measure.currentIndexChanged.connect(lambda _: self.show_quality())
        bar.addWidget(self.qc_measure)
        bar.addSpacing(12)
        bar.addWidget(QtWidgets.QLabel("Reference split at true F1 ≥"))
        self.good_f1 = QtWidgets.QDoubleSpinBox(
            minimum=0.0, maximum=1.0, singleStep=0.05, decimals=2, value=GOOD_F1
        )
        self.good_f1.setToolTip(
            "The checkpoint's validation windows, where the truth was known, are shown "
            "in two rows: those predicted at least this well (true inhale F1), and "
            "the rest. A description of the reference, never a verdict on a video."
        )
        self.good_f1.valueChanged.connect(self._resplit)
        bar.addWidget(self.good_f1)
        bar.addStretch(1)
        column.addLayout(bar)
        self.qc_note = note("")
        column.addWidget(self.qc_note)
        parts = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.overview = pg.PlotWidget()
        self.overview.showGrid(x=True, alpha=0.15)
        self.overview.invertY(True)
        self.overview.setMouseEnabled(x=False, y=False)
        self.overview.scene().sigMouseClicked.connect(self._overview_clicked)
        parts.addWidget(self.overview)
        self.qc_table = QtWidgets.QTableWidget()
        self.qc_table.setSortingEnabled(True)
        self.qc_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.qc_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.qc_table.cellClicked.connect(self._table_clicked)
        parts.addWidget(self.qc_table)
        parts.setSizes([500, 400])
        column.addWidget(parts, 1)
        self._rows = []
        return page

    def fill_quality(self, folder: Path | None) -> None:
        """Read the folder's quality-control reports, and its checkpoint's reference."""
        state = QualityState(folder, self.good_f1.value()) if folder else None
        self._quality = None if state is None or state.empty else state
        current = self.qc_measure.currentData()
        self.qc_measure.blockSignals(True)
        self.qc_measure.clear()
        if self._quality is not None:
            for name in self._quality.measures():
                self.qc_measure.addItem(MEASURES[name][1], name)
            found = self.qc_measure.findData(current)
            self.qc_measure.setCurrentIndex(max(found, 0))
        self.qc_measure.blockSignals(False)
        self.fill_table()
        self.show_quality()

    def _resplit(self, value: float) -> None:
        if self._quality is not None:
            self._quality.good_f1 = value
        self.show_quality()

    def show_quality(self) -> None:
        """The overview of the chosen measure, and its strip under the trace."""
        plot = self.overview
        plot.clear()
        self._rows = []
        state, measure = self._quality, self.qc_measure.currentData()
        if state is None or measure is None:
            self.qc_note.setText(
                "No quality-control reports in this folder: tick 'judge them' when "
                "predicting."
            )
            self.show_strip()
            return
        self.qc_note.setText(
            "Each row is the spread of the measure over 30-s windows (box: middle "
            "half; whiskers: 5-95%; line: median). "
            + (
                "The top rows are the checkpoint's validation windows, where the "
                "truth was known."
                if state.split
                else "The checkpoint has no reference: the videos are shown against "
                "each other."
            )
        )
        self._rows = state.rows(measure)
        for y, row in enumerate(self._rows):
            spread = row.spread()
            if spread is None:
                continue
            lo, q1, median, q3, hi = spread
            colour = QC_COLOURS[row.kind]
            strong = row.stem is not None and row.stem == self._stem
            plot.addItem(pg.PlotDataItem([lo, hi], [y, y], pen=pg.mkPen(colour)))
            plot.addItem(
                pg.BarGraphItem(
                    x0=[q1],
                    width=[max(q3 - q1, 1e-9)],
                    y=[y],
                    height=0.6,
                    brush=pg.mkBrush(*colour, 230 if strong else 120),
                    pen=pg.mkPen("k" if strong else colour),
                )
            )
            plot.addItem(
                pg.PlotDataItem(
                    [median, median], [y - 0.3, y + 0.3], pen=pg.mkPen("k", width=2)
                )
            )
        plot.getAxis("left").setTicks(
            [[(y, row.label) for y, row in enumerate(self._rows)]]
        )
        plot.setLabel("bottom", MEASURES[measure][1])
        plot.setYRange(-0.6, len(self._rows) - 0.4)
        self.show_strip()

    def fill_table(self) -> None:
        """One row per video: minutes, and each measure's median over its windows
        (with its median percentile among the validation windows, when known)."""
        table = self.qc_table
        table.setSortingEnabled(False)
        table.clear()
        frame = self._quality.table() if self._quality is not None else pd.DataFrame()
        columns = [c for c in frame.columns if c != "stem"]
        heads = ["Video"] + [
            "Minutes"
            if c == "minutes"
            else f"{MEASURES[c[4:]][1]} (pct)"
            if c.startswith("pct_")
            else MEASURES[c][1]
            for c in columns
        ]
        table.setColumnCount(len(heads))
        table.setHorizontalHeaderLabels(heads)
        table.setRowCount(len(frame))
        for r, row in enumerate(frame.itertuples(index=False)):
            values = row._asdict()
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(values["stem"]))
            for c, name in enumerate(columns, start=1):
                item = QtWidgets.QTableWidgetItem()
                value = values[name]
                if value is not None and np.isfinite(value):
                    item.setData(QtCore.Qt.ItemDataRole.DisplayRole, round(value, 3))
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        table.setSortingEnabled(True)

    def select_video(self, stem: str) -> None:
        """Show video *stem*: its trace, onset probability and crop, and its strip."""
        for k in range(self.videos.count()):
            if self.videos.item(k).text() == stem:
                self.videos.setCurrentRow(k)
                return

    def _overview_clicked(self, event) -> None:
        point = self.overview.plotItem.vb.mapSceneToView(event.scenePos())
        k = round(point.y())
        if 0 <= k < len(self._rows) and self._rows[k].stem is not None:
            self.select_video(self._rows[k].stem)

    def _table_clicked(self, row: int, _column: int) -> None:
        item = self.qc_table.item(row, 0)
        if item is not None:
            self.select_video(item.text())

    def show_strip(self) -> None:
        """The chosen measure over the shown video's windows, on the trace's time axis,
        with the reference's good windows' 5-95% range shaded."""
        strip = self.strip
        strip.clear()
        state, measure = self._quality, self.qc_measure.currentData()
        if state is None or measure is None or self._stem is None:
            strip.setVisible(False)
            return
        windows = state.over_time(self._stem, measure)
        strip.setVisible(not windows.empty)
        if windows.empty:
            return
        edges = (
            np.r_[windows["start"].to_numpy(), windows["end"].iloc[-1]] - self._origin
        )
        strip.plot(
            edges,
            windows["value"].to_numpy(),
            stepMode="center",
            pen=pg.mkPen(QC_COLOURS["video"], width=2),
        )
        good = state.good_range(measure)
        if good is not None:
            band = pg.LinearRegionItem(
                values=good,
                orientation="horizontal",
                movable=False,
                brush=pg.mkBrush(*QC_COLOURS["good"], 50),
            )
            strip.addItem(band)
        strip.setLabel("left", MEASURES[measure][1])

    def _strip_clicked(self, event) -> None:
        if self._quality is None or self._stem is None:
            return
        x = self.strip.plotItem.vb.mapSceneToView(event.scenePos()).x()
        window = self._quality.window_at(self._stem, x + self._origin)
        if window is not None:
            self.trace.plot.setXRange(
                window[0] - self._origin, window[1] - self._origin, padding=0
            )

    # ---- videos from anywhere ----------------------------------------------------

    def _build_videos(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        tab = QtWidgets.QVBoxLayout(page)
        form = QtWidgets.QFormLayout()
        folder = QtWidgets.QHBoxLayout()
        self.video_folder = QtWidgets.QLineEdit()
        folder.addWidget(self.video_folder, 1)
        pick = QtWidgets.QPushButton("...")
        pick.clicked.connect(self._pick_folder)
        folder.addWidget(pick)
        form.addRow("Folder", folder)
        self.video_glob = QtWidgets.QLineEdit("*.mp4")
        form.addRow("Videos", self.video_glob)
        self.video_times = QtWidgets.QLineEdit(Derive().timestamps)
        form.addRow("Timestamps", self.video_times)
        self.video_rate = QtWidgets.QLineEdit(placeholderText="(from the timestamps)")
        self.video_rate.setToolTip(
            "Videos with no timestamps: their frames per second (frame i at i / rate "
            "s, assuming no dropped frames); empty: each needs its timestamps"
        )
        form.addRow("Frame rate (Hz)", self.video_rate)
        self.video_name = QtWidgets.QLineEdit("videos")
        self.video_name.textChanged.connect(lambda _: self.refresh_listed())
        form.addRow("Into predictions/", self.video_name)
        tab.addLayout(form)
        tab.addWidget(
            note(
                "Videos and their timestamps (or a frame rate) only, from any folder. "
                "The list takes the "
                "checkpoint's frame size, and every crop is the size the network "
                "trained on: you choose only where it goes, on one frame per video."
            )
        )
        make = QtWidgets.QPushButton("1. Make the list")
        make.clicked.connect(self.make_video_list)
        tab.addWidget(make)
        self.listed_label = note("No list yet.")
        tab.addWidget(self.listed_label)
        self.listed = QtWidgets.QListWidget()
        self.listed.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.listed.setMinimumHeight(120)
        tab.addWidget(self.listed, 1)
        edits = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add videos...")
        add.setToolTip("Add videos from anywhere to the list (each needs timestamps)")
        add.clicked.connect(self.add_videos)
        edits.addWidget(add)
        remove = QtWidgets.QPushButton("Remove selected")
        remove.clicked.connect(self.remove_videos)
        edits.addWidget(remove)
        edits.addStretch(1)
        tab.addLayout(edits)
        place = QtWidgets.QPushButton("2. Place the crops")
        place.clicked.connect(self.place_crops)
        tab.addWidget(place)
        self.video_cache = QtWidgets.QCheckBox(
            "keep the frames in the project's cache (shows the crop; takes space)"
        )
        self.video_cache.setToolTip(
            "Off: the frames are read from the videos into a temporary folder and "
            "removed afterwards"
        )
        tab.addWidget(self.video_cache)
        self.video_qc = QtWidgets.QCheckBox("judge them (quality control)")
        tab.addWidget(self.video_qc)
        go = accent("3. Predict")
        go.clicked.connect(self.predict_videos)
        tab.addWidget(go)
        return page

    def refresh_listed(self) -> None:
        """The videos in the Videos tab's list, each with its crop (placed or not)
        and whether its timestamps were found."""
        self.listed.clear()
        path = self.video_list()
        if path is None or not path.is_file():
            self.listed_label.setText("No list yet: 1. Make the list.")
            return
        try:
            clips = self.project.load_list(path).resolve()
        except ValueError as exc:
            self.listed_label.setText(f"The list does not load: {exc}")
            return
        placed = sum(c.box is not None for c in clips)
        self.listed_label.setText(
            f"{len(clips)} video(s) in predictions/{path.parent.name}, {placed} with "
            "a crop placed:"
        )
        for clip in clips:
            state = "crop placed" if clip.box is not None else "no crop yet"
            if clip.timestamps is not None and not clip.timestamps.is_file():
                state = "no timestamps"
            item = QtWidgets.QListWidgetItem(f"{clip.video.name}   ({state})")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, clip.video)
            item.setToolTip(str(clip.video))
            self.listed.addItem(item)

    def _edit_listed(self, **change) -> bool:
        """Change the Videos tab's list (``zephyr.clips``), its crop window closed
        first (it saves) so nothing it holds overwrites the change."""
        path = self.video_list()
        if path is None or not path.is_file():
            QtWidgets.QMessageBox.information(
                self, "No list yet", "Make the list of videos first."
            )
            return False
        reopen = self.box_window is not None and self.box_window.isVisible()
        if self.box_window is not None:
            self.box_window.close()
            self.box_window = None
        try:
            update_clip_list(path, **change)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "The list is unchanged", str(exc))
            return False
        finally:
            self.refresh_listed()
        self._reopen = reopen
        return True

    def add_videos(self) -> None:
        """Add videos to the list, then cache a frame of each for its crop."""
        start = self.video_folder.text().strip()
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Videos to add", start, "Videos (*.mp4 *.avi *.mov *.mkv);;All (*)"
        )
        rate = self._frame_rate()
        if rate is False or not files:
            return
        if not self._edit_listed(
            add=list(map(Path, files)), unlabelled=True, frame_rate=rate
        ):
            return
        path = self.video_list()
        prepare = ["--prepare", "--frame-dir", str(self.project.frames_dir)]
        self._then = self.place_crops if self._reopen else self.refresh_listed
        self.jobs.run(Job("annotate", [str(path), *prepare], self.project.root))

    def remove_videos(self) -> None:
        chosen = {
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in self.listed.selectedItems()
        }
        if chosen and self._edit_listed(remove=chosen) and self._reopen:
            self.place_crops()

    def _frame_rate(self) -> float | None | bool:
        """The Frame rate field: ``None`` when empty, ``False`` (after saying so)
        when it is not a number above 0."""
        text = self.video_rate.text().strip()
        if not text:
            return None
        try:
            rate = float(text)
        except ValueError:
            rate = 0.0
        if rate > 0:
            return rate
        QtWidgets.QMessageBox.warning(
            self, "Frame rate", f"{text!r} is not a frame rate (frames per second)"
        )
        return False

    def _pick_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Video folder")
        if folder:
            self.video_folder.setText(folder)

    def video_list(self) -> Path | None:
        """Where the Videos tab's list is written: ``predictions/<name>/videos.toml``."""
        if self.project is None:
            return None
        name = self.video_name.text().strip() or "videos"
        return self.project.predictions_dir / name / VIDEO_LIST

    def _video_inputs(self) -> tuple[Path, Path] | None:
        checkpoint, path = self.checkpoint.currentData(), self.video_list()
        if checkpoint is None or path is None:
            QtWidgets.QMessageBox.information(
                self, "Predict", "Open a project and choose a checkpoint first."
            )
            return None
        return checkpoint, path

    def make_video_list(self) -> None:
        """Scan the folder into a list for the checkpoint, then cache one frame per
        video and open the crops."""
        found = self._video_inputs()
        folder = self.video_folder.text().strip()
        if found is None or not folder:
            return
        checkpoint, path = found
        if path.exists():
            answer = QtWidgets.QMessageBox.question(
                self,
                "Replace the list?",
                f"predictions/{path.parent.name} already has a list of videos; scan "
                "the folder again and replace it (its crops are lost)?",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        if self.box_window is not None:  # it would write the old crops back
            self.box_window.close()
            self.box_window = None
        rate = self._frame_rate()
        if rate is False:
            return
        argv = ["scan", folder, "--glob", self.video_glob.text(), "-o", str(path)]
        argv += ["--checkpoint", str(checkpoint)]
        if rate is not None:
            argv += ["--frame-rate", f"{rate:g}"]
        elif self.video_times.text() != Derive().timestamps:
            argv += ["--timestamps", self.video_times.text()]
        prepare = ["--prepare", "--frame-dir", str(self.project.frames_dir)]
        self._queue = [Job("annotate", [str(path), *prepare], self.project.root)]
        self._then = self.place_crops
        self.jobs.run(Job("clips", argv, self.project.root))

    def place_crops(self) -> None:
        """The box window on the list's frames, the crop size the network's."""
        found = self._video_inputs()
        if found is None:
            return
        checkpoint, path = found
        if not path.is_file():
            QtWidgets.QMessageBox.information(
                self, "No list yet", "Make the list of videos first."
            )
            return
        try:
            state = open_state(
                path, self.project.frames_dir, remote=True, checkpoint=checkpoint
            )
        except (SystemExit, ValueError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot place the crops", str(exc))
            return
        self.box_window = BoxWindow(state, self.project.frames_dir, path)
        self.box_window.saved.connect(self.refresh_listed)
        self.refresh_listed()
        self.box_window.setWindowTitle(f"Crops of predictions/{path.parent.name}")
        self.box_window.resize(1300, 850)
        self.box_window.show()
        self.box_window.canvas.setFocus()

    def predict_videos(self) -> None:
        found = self._video_inputs()
        if found is None:
            return
        checkpoint, path = found
        if not path.is_file():
            QtWidgets.QMessageBox.information(
                self, "No list yet", "Make the list of videos first."
            )
            return
        if self.box_window is not None:
            self.box_window.close()  # saves every crop
            self.box_window = None
        argv = ["--checkpoint", str(checkpoint), "--clips", str(path)]
        argv += ["--out", str(path.parent)]
        again = self._again(path.parent)
        if again is None:
            return
        argv += again
        if self.video_cache.isChecked():
            cache = self.project.cache_dir
            if not ask_cache_space(self, [path], cache):
                return
            argv += ["--cache", str(cache)]
        if self.video_qc.isChecked():
            argv.append("--quality-control")
        self._out = path.parent
        self.jobs.run(Job("predict", argv, self.project.root))

    def _again(self, out: Path) -> list[str] | None:
        """For a folder that has predictions already: ``--restart`` to predict every
        video again (new crops, list or checkpoint), nothing to predict only the
        videos without one, None to cancel."""
        done = list(out.glob(f"*{SUFFIX}"))
        if not done:
            return []
        box = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Icon.Question,
            "Predictions exist",
            f"predictions/{out.name} has {len(done)} prediction(s) already. Predict "
            "every video again (they are replaced: do this after moving a crop or "
            "with another checkpoint), or only the videos without one?",
            parent=self,
        )
        everything = box.addButton("Predict all again", box.ButtonRole.AcceptRole)
        new = box.addButton("Only the new ones", box.ButtonRole.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is everything:
            return ["--restart"]
        return [] if box.clickedButton() is new else None

    def set_project(self, project: Project | None) -> None:
        self.shown()

    def shown(self) -> None:
        project = self.project
        keep = self.checkpoint.currentData()
        self.checkpoint.clear()
        self.clip_list.clear()
        if project is None:
            return
        checkpoints = project.checkpoints()
        checkpoints.sort(key=lambda path: is_smoke(path.parent, project.runs_dir))
        for path in checkpoints:
            status = run_status(path.parent)
            best = re.search(r"best\.pt from epoch \d+", status)
            label = run_label(path.parent, project.runs_dir)
            self.checkpoint.addItem(
                f"{label} · {best.group(0)}" if best else label, path
            )
            self.checkpoint.setItemData(
                self.checkpoint.count() - 1,
                str(path),
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        if not checkpoints and keep is None:
            self.checkpoint.addItem("none yet: train one on 4 Train")
        if keep is not None and find_data(self.checkpoint, keep) < 0:
            self.checkpoint.addItem(str(keep), keep)
        self.checkpoint.setCurrentIndex(max(find_data(self.checkpoint, keep), 0))
        for path in project.clip_lists():
            self.clip_list.addItem(path.name, path)
        index = find_data(self.clip_list, project.clip_list)
        self.clip_list.setCurrentIndex(max(index, 0))
        self.refresh_listed()
        self.fill_folders()

    def _browse(self) -> None:
        start = str(self.project.runs_dir) if self.project else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Checkpoint", start, "Checkpoints (*.pt)"
        )
        if path:
            self.checkpoint.addItem(
                f"{Path(path).parent.name} (elsewhere): {path}", Path(path)
            )
            self.checkpoint.setCurrentIndex(self.checkpoint.count() - 1)

    def fill_folders(self, select: Path | None = None) -> None:
        project = self.project
        current = select or self.folder.currentData()
        self.folder.blockSignals(True)
        self.folder.clear()
        if project is not None and project.predictions_dir.is_dir():
            for folder in sorted(
                p for p in project.predictions_dir.iterdir() if p.is_dir()
            ):
                self.folder.addItem(folder.name, folder)
        self.folder.setCurrentIndex(max(find_data(self.folder, current), 0))
        self.folder.blockSignals(False)
        self.fill_videos()

    def predict(self) -> None:
        project = self.project
        checkpoint, clips = self.checkpoint.currentData(), self.clip_list.currentData()
        if project is None or checkpoint is None or clips is None:
            QtWidgets.QMessageBox.information(
                self, "Predict", "Choose a checkpoint and a clip list first."
            )
            return
        out = project.predictions_dir / (self.name.text().strip() or "predictions")
        common = ["--checkpoint", str(checkpoint), "--clips", str(clips)]
        groups = self.groups.text().split()
        selection = ["--groups", *groups] if groups else []
        argv = [*common, "--out", str(out), *selection]
        again = self._again(out)
        if again is None:
            return
        argv += again
        if self.keep.isChecked():
            if not ask_cache_space(self, [clips], project.cache_dir):
                return
            argv += ["--cache", str(project.cache_dir)]
        self._queue = []
        if self.qc.isChecked():
            argv.append("--quality-control")
        self._out = out
        self.jobs.run(Job("predict", argv, project.root))

    def _finished(self, code) -> None:
        if code == 0 and self._queue:
            self.jobs.run(self._queue.pop(0))
            return
        then, self._then, self._queue = self._then, None, []
        if code == 0 and then is not None:
            then()
        self.fill_folders(getattr(self, "_out", None))

    def fill_videos(self) -> None:
        folder = self.folder.currentData()
        self.videos.clear()
        self._stem = None
        if folder is None:
            self.fill_quality(None)
            return
        for path in sorted(folder.glob(f"*{SUFFIX}")):
            item = QtWidgets.QListWidgetItem(path.name[: -len(SUFFIX)])
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            self.videos.addItem(item)
        self.fill_quality(folder)
        if self.videos.count():
            self.videos.setCurrentRow(0)

    def _entry(self, stem: str):
        """The cache entry of the video *stem* in the folder's own list of videos, else
        the chosen clip list, if cached."""
        project, path = self.project, self.clip_list.currentData()
        folder = self.folder.currentData()
        if folder is not None and (folder / VIDEO_LIST).is_file():
            path = folder / VIDEO_LIST
        if project is None or path is None:
            return None
        try:
            clip_list = project.load_list(path)
        except ValueError:
            return None
        for clip in clip_list.resolve():
            if clip.video is not None and clip.video.stem == stem and clip.box:
                return features.lookup(clip, clip_list.preprocess, project.cache_dir)
        return None

    def show_video(self) -> None:
        item = self.videos.currentItem()
        if item is None:
            return
        path: Path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        stem = path.name[: -len(SUFFIX)]
        table = pd.read_parquet(path)
        time = table["time"].to_numpy()
        self._stem, self._origin = stem, float(time[0]) if len(time) else 0.0
        report_path = path.with_name(stem + QC_SUFFIX)
        verdict = ""
        if report_path.is_file():
            summary = json.loads(report_path.read_text())["summary"]
            # A measure without a value (expected precision with no reference) is null;
            # percentiles place it among the checkpoint's validation windows.
            position = summary.get("percentiles") or {}
            cells = [
                f"<b>{name}</b> {_measure(summary.get(name))}"
                + (
                    f" ({position[name]:.0f}th pct)"
                    if position.get(name) is not None
                    else ""
                )
                for name in SHOWN
                if name in summary
            ]
            verdict = (
                f"Whole video: {'; '.join(cells)} (percentiles among the checkpoint's "
                "validation windows; every measure is on the Quality tab)."
            )
        self.verdict.setText(verdict)
        self.verdict.setVisible(bool(verdict))
        self.trace.set_trace(
            f"<b>{stem}</b>",
            time,
            table["signal"].to_numpy(),
            inhale=time[table["inhale"].to_numpy(bool)],
            entry=self._entry(stem),
            onset=table["onset"].to_numpy() if "onset" in table else None,
            threshold=_threshold(path.parent / SETTINGS),
        )
        self.show_quality()

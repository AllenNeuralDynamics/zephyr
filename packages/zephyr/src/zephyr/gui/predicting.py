"""The Predict page: a checkpoint's traces for a clip list, or for videos from any folder
(``clips scan --checkpoint``, their crops placed at the network's size, then ``zephyr
predict``), and, on request, quality control of them (``zephyr quality-control run``),
shown per video with the crop it came from."""

import json
from pathlib import Path

import pandas as pd
from PySide6 import QtCore, QtWidgets

from .. import features
from ..annotate import open_state
from ..config import Derive
from ..predict import SUFFIX
from .boxes import BoxWindow
from .jobs import Job
from .pages import Page
from .project import Project
from .widgets import JobPanel, TraceView, accent, find_data, heading, note

QC_SUFFIX = ".qc.json"
VIDEO_LIST = "videos.toml"
"""The Videos tab's list, in the predictions folder it fills."""


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
        browse = QtWidgets.QPushButton("...")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        top.addRow("Checkpoint", row)
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
        self.keep = QtWidgets.QCheckBox("keep the frames in the project's cache")
        self.keep.setChecked(True)
        form.addRow("", self.keep)
        self.qc = QtWidgets.QCheckBox("judge them (quality control; keeps the frames)")
        self.qc.toggled.connect(lambda on: self.keep.setChecked(True) if on else None)
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
        self.jobs = JobPanel()
        self.jobs.finished.connect(self._finished)
        left.addWidget(self.jobs, 1)

        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 1)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Folder"))
        self.folder = QtWidgets.QComboBox()
        self.folder.currentIndexChanged.connect(lambda _: self.fill_videos())
        bar.addWidget(self.folder, 1)
        right.addLayout(bar)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.videos = QtWidgets.QListWidget()
        self.videos.currentRowChanged.connect(lambda _: self.show_video())
        split.addWidget(self.videos)
        self.trace = TraceView()
        split.addWidget(self.trace)
        self.verdict = QtWidgets.QLabel(wordWrap=True)
        self.verdict.setTextFormat(QtCore.Qt.TextFormat.RichText)
        split.addWidget(self.verdict)
        split.setSizes([160, 420, 120])
        right.addWidget(split, 1)
        self._queue: list[Job] = []
        self._then = None
        self.box_window: BoxWindow | None = None

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
        self.video_name = QtWidgets.QLineEdit("videos")
        form.addRow("Into predictions/", self.video_name)
        tab.addLayout(form)
        tab.addWidget(
            note(
                "Videos and their timestamps only, from any folder. The list takes the "
                "checkpoint's frame size, and every crop is the size the network "
                "trained on: you choose only where it goes, on one frame per video."
            )
        )
        steps = (
            ("1. Make the list", self.make_video_list),
            ("2. Place the crops", self.place_crops),
            ("3. Predict", self.predict_videos),
        )
        for text, slot in steps:
            button = (
                accent(text) if text.startswith("3") else QtWidgets.QPushButton(text)
            )
            button.clicked.connect(slot)
            tab.addWidget(button)
        tab.addStretch(1)
        return page

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
            path.unlink()
        argv = ["scan", folder, "--glob", self.video_glob.text(), "-o", str(path)]
        argv += ["--checkpoint", str(checkpoint)]
        if self.video_times.text() != Derive().timestamps:
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
        argv += ["--out", str(path.parent), "--cache", str(self.project.cache_dir)]
        self._out = path.parent
        self.jobs.run(Job("predict", argv, self.project.root))

    def set_project(self, project: Project | None) -> None:
        self.shown()

    def shown(self) -> None:
        project = self.project
        keep = self.checkpoint.currentData()
        self.checkpoint.clear()
        self.clip_list.clear()
        if project is None:
            return
        for path in project.checkpoints():
            self.checkpoint.addItem(str(path.relative_to(project.runs_dir)), path)
        if keep is not None and find_data(self.checkpoint, keep) < 0:
            self.checkpoint.addItem(str(keep), keep)
        self.checkpoint.setCurrentIndex(max(find_data(self.checkpoint, keep), 0))
        for path in project.clip_lists():
            self.clip_list.addItem(path.name, path)
        index = find_data(self.clip_list, project.clip_list)
        self.clip_list.setCurrentIndex(max(index, 0))
        self.fill_folders()

    def _browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Checkpoint", "", "Checkpoints (*.pt)"
        )
        if path:
            self.checkpoint.addItem(path, Path(path))
            self.checkpoint.setCurrentIndex(self.checkpoint.count() - 1)

    def fill_folders(self, select: Path | None = None) -> None:
        project = self.project
        current = select or self.folder.currentData()
        self.folder.blockSignals(True)
        self.folder.clear()
        if project is not None:
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
        if self.keep.isChecked():
            argv += ["--cache", str(project.cache_dir)]
        self._queue = []
        if self.qc.isChecked():
            self._queue.append(
                Job(
                    "quality-control",
                    [
                        "run",
                        *common,
                        "--cache",
                        str(project.cache_dir),
                        "--out",
                        str(out),
                        *selection,
                    ],
                    project.root,
                )
            )
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
        if folder is None:
            return
        for path in sorted(folder.glob(f"*{SUFFIX}")):
            item = QtWidgets.QListWidgetItem(path.name[: -len(SUFFIX)])
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            self.videos.addItem(item)
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
        report_path = path.with_name(stem + QC_SUFFIX)
        bad, verdict = [], "No quality control for this video."
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            bad = [
                (w["start"], w["end"])
                for w in report["windows"]
                if any(v is False for v in w["checks"].values())
            ]
            summary = report["summary"]
            cells = [
                f"<b>{name}</b> {summary.get(name, float('nan')):.3g} "
                + ("pass" if ok else "fail" if ok is False else "n/a")
                for name, ok in summary["checks"].items()
            ]
            verdict = (
                f"Whole video: {'; '.join(cells)}.<br>{len(bad)} of "
                f"{len(report['windows'])} windows fail a measure (shaded). A "
                "measure is judged per window; a video is never failed as a whole."
            )
        self.verdict.setText(verdict)
        self.trace.set_trace(
            f"<b>{stem}</b>",
            time,
            table["signal"].to_numpy(),
            inhale=time[table["inhale"].to_numpy(bool)],
            entry=self._entry(stem),
            bad_spans=bad,
        )

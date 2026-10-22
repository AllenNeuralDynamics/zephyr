"""The Predict page: a checkpoint's traces for a clip list (``zephyr predict``), and,
on request, quality control of them (``zephyr quality-control run``), shown per video
with the crop it came from."""

import json
from pathlib import Path

import pandas as pd
from PySide6 import QtCore, QtWidgets

from .. import features
from ..predict import SUFFIX
from .jobs import Job
from .pages import Page
from .project import Project
from .widgets import JobPanel, TraceView, accent, find_data, heading, note

QC_SUFFIX = ".qc.json"


class PredictPage(Page):
    title = "Predict"

    def __init__(self, app) -> None:
        super().__init__(app)
        layout = QtWidgets.QHBoxLayout(self)
        holder = QtWidgets.QWidget(maximumWidth=420)
        left = QtWidgets.QVBoxLayout(holder)
        layout.addWidget(holder)
        left.addWidget(heading("Predict"))
        form = QtWidgets.QFormLayout()
        row = QtWidgets.QHBoxLayout()
        self.checkpoint = QtWidgets.QComboBox()
        row.addWidget(self.checkpoint, 1)
        browse = QtWidgets.QPushButton("...")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        form.addRow("Checkpoint", row)
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
        left.addLayout(form)
        left.addWidget(
            note(
                "The clips need boxes, not a breathing sensor. Without the cache the "
                "frames are decoded into a temporary folder and removed; the crop is "
                "then not shown. Quality control refuses clips the checkpoint trained "
                "on."
            )
        )
        go = accent("Predict")
        go.clicked.connect(self.predict)
        left.addWidget(go)
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
        self._queue = []
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
        """The cache entry of the video *stem* in the chosen clip list, if cached."""
        project, path = self.project, self.clip_list.currentData()
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

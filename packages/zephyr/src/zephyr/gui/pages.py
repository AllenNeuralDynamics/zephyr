"""The first pages: Project (open or create one), Clips (make a clip list, check its
files), Boxes (:mod:`.boxes`) and Breathing (:mod:`.breathing`)."""

import html
import json
import tempfile
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .. import features
from ..annotate import DEFAULT_BOX_SIZE, frame_path, missing_frames, open_state
from ..annotate_events import TuneState
from ..config import Derive
from .boxes import BoxWindow
from .breathing import EditorWindow
from .jobs import Job
from .project import Project
from .widgets import JobPanel, accent, heading, note


class Page(QtWidgets.QWidget):
    """One step of the app. *set_project* is called when the project or its clip
    list changes; *leave* may refuse to switch away (unsaved changes)."""

    title = ""

    def __init__(self, app) -> None:
        super().__init__()
        self.app = app

    @property
    def project(self) -> Project | None:
        return self.app.project

    def set_project(self, project: Project | None) -> None:
        pass

    def shown(self) -> None:
        pass

    def leave(self) -> bool:
        return True


def _table(headers: list[str]) -> QtWidgets.QTableWidget:
    table = QtWidgets.QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def _fill(table: QtWidgets.QTableWidget, rows: list[list]) -> None:
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            table.setItem(
                r, c, QtWidgets.QTableWidgetItem("" if value is None else str(value))
            )
    table.resizeColumnsToContents()


class ProjectPage(Page):
    title = "Project"

    def __init__(self, app) -> None:
        super().__init__(app)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(heading("Project"))
        layout.addWidget(
            note(
                "A project is a folder: its clip lists (*.toml) and, beside them, "
                "experiments/, frames/, cache/, runs/ and predictions/. Everything the "
                "app makes stays in it, and every step is also a zephyr command."
            )
        )
        buttons = QtWidgets.QHBoxLayout()
        for text, slot in (
            ("Open or create a project folder...", self.app.choose_project),
            ("Open a clip list...", self.app.choose_clip_list),
        ):
            button = (
                accent(text) if not buttons.count() else QtWidgets.QPushButton(text)
            )
            button.clicked.connect(slot)
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.summary = QtWidgets.QLabel(wordWrap=True)
        self.summary.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.summary.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.summary, 1)

    def set_project(self, project: Project | None) -> None:
        self.shown()

    def shown(self) -> None:
        project = self.project
        if project is None:
            self.summary.setText("No project open.")
            return
        lines = [f"<p><b>{html.escape(str(project.root))}</b></p><ul>"]
        lists = project.clip_lists()
        if not lists:
            lines.append(
                "<li>No clip list yet: make one on the <b>Clips</b> page.</li>"
            )
        for path in lists:
            mark = " (in use)" if path == project.clip_list else ""
            if path in project.errors:
                lines.append(
                    f"<li>{path.name}{mark}: <b>does not load</b>: "
                    f"{html.escape(project.errors[path][:400])}</li>"
                )
                continue
            clip_list = project.load_list(path)
            clips = clip_list.resolve()
            boxed = sum(c.box is not None for c in clips)
            labelled = sum(c.labelled for c in clips)
            cached = sum(
                c.video is not None
                and c.box is not None
                and features.lookup(c, clip_list.preprocess, project.cache_dir)
                is not None
                for c in clips
            )
            groups = len({c.group for c in clips})
            lines.append(
                f"<li>{path.name}{mark}: {len(clips)} clips in {groups} groups, "
                f"{labelled} with a breathing sensor, {boxed} boxed, {cached} "
                "preprocessed</li>"
            )
        lines.append(
            f"<li>{len(project.experiments())} experiment(s), {len(project.runs())} "
            "run(s)</li></ul>"
        )
        lines.append(
            "<p>Steps: <b>Clips</b> (list and check the files) &rarr; <b>Boxes</b> "
            "(one crop per recording) &rarr; <b>Breathing</b> (check the events) "
            "&rarr; <b>Train</b> &rarr; <b>Results</b> &rarr; <b>Predict</b>.</p>"
        )
        self.summary.setText("".join(lines))


class ClipsPage(Page):
    """Make a clip list from a folder of videos (``zephyr clips scan``), see its
    clips, and check every file (``zephyr clips check``)."""

    title = "Clips"

    def __init__(self, app) -> None:
        super().__init__(app)
        layout = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        layout.addLayout(left, 1)
        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 2)

        left.addWidget(heading("New clip list"))
        form = QtWidgets.QFormLayout()
        folder_row = QtWidgets.QHBoxLayout()
        self.folder = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("...")
        browse.clicked.connect(self._browse)
        folder_row.addWidget(self.folder)
        folder_row.addWidget(browse)
        form.addRow("Video folder", folder_row)
        self.glob = QtWidgets.QLineEdit("video_*.mp4")
        form.addRow("Videos", self.glob)
        self.name = QtWidgets.QLineEdit("clips")
        form.addRow("List name", self.name)
        self.unlabelled = QtWidgets.QCheckBox("no breathing sensor (predict only)")
        form.addRow("", self.unlabelled)
        defaults = Derive()
        self.pattern = QtWidgets.QLineEdit(defaults.pattern)
        self.timestamps = QtWidgets.QLineEdit(defaults.timestamps)
        self.thermistor = QtWidgets.QLineEdit(defaults.thermistor)
        form.addRow("Name pattern", self.pattern)
        form.addRow("Timestamps", self.timestamps)
        form.addRow("Breathing file", self.thermistor)
        left.addLayout(form)
        left.addWidget(
            note(
                "The pattern is a regex searched in each video's file name (without "
                "extension); it must capture (?P<group>...), the session. The two "
                "files are templates over {stem} and the pattern's groups, next to "
                "the video. The defaults match video_<camera>_<session>_part_<n>.mp4 "
                "with <stem>.parquet and thermistor_<session>_part_<n>.parquet."
            )
        )
        scan = accent("Make the clip list")
        scan.clicked.connect(self.scan)
        left.addWidget(scan)
        left.addStretch(1)

        bar = QtWidgets.QHBoxLayout()
        self.list_label = heading("Clips")
        bar.addWidget(self.list_label, 1)
        check = accent("Check files")
        check.clicked.connect(self.check)
        bar.addWidget(check)
        right.addLayout(bar)
        self.table = _table(
            ["clip", "group", "sensor", "box", "events", "mark", "check"]
        )
        right.addWidget(self.table, 3)
        self.jobs = JobPanel()
        self.jobs.finished.connect(self._finished)
        right.addWidget(self.jobs, 1)
        self._pending: tuple[str, Path] | None = None
        self._checks: dict[str, dict] = {}

    def _browse(self) -> None:
        start = str(self.project.root) if self.project else ""
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Video folder", start)
        if folder:
            self.folder.setText(folder)

    def set_project(self, project: Project | None) -> None:
        self._checks = {}
        self.shown()

    def shown(self) -> None:
        project = self.project
        if project is None or project.clip_list is None:
            self.list_label.setText("Clips: no clip list")
            self.table.setRowCount(0)
            return
        self.list_label.setText(f"Clips of {project.clip_list.name}")
        try:
            clip_list = project.load_list()
        except ValueError as exc:
            self.table.setRowCount(0)
            self.jobs.log.appendPlainText(str(exc))
            return
        rows = []
        resolved_clips = clip_list.resolve(include_excluded=True)
        for clip, resolved in zip(clip_list.clip, resolved_clips, strict=True):
            checked = self._checks.get(resolved.name)
            rows.append(
                [
                    resolved.name,
                    resolved.group,
                    "yes" if resolved.labelled else "no",
                    "" if clip.box is None else " ".join(map(str, clip.box)),
                    resolved.events.method,
                    clip.selection or "",
                    ""
                    if checked is None
                    else "ok"
                    if not checked["problems"]
                    else "; ".join(checked["problems"]),
                ]
            )
        _fill(self.table, rows)

    def scan(self) -> None:
        if self.project is None:
            self.app.choose_project()
            if self.project is None:
                return
        name = self.name.text().strip() or "clips"
        out = self.project.root / f"{name}.toml"
        if out.exists():
            QtWidgets.QMessageBox.warning(
                self, "Exists", f"{out.name} exists; choose another list name."
            )
            return
        argv = [self.folder.text(), "--glob", self.glob.text(), "-o", str(out)]
        if self.unlabelled.isChecked():
            argv.append("--unlabelled")
        defaults = Derive()
        for flag, edit, default in (
            ("--pattern", self.pattern, defaults.pattern),
            ("--timestamps", self.timestamps, defaults.timestamps),
            ("--thermistor", self.thermistor, defaults.thermistor),
        ):
            if edit.text() != default:
                argv += [flag, edit.text()]
        self._pending = ("scan", out)
        self.jobs.run(Job("clips", ["scan", *argv], self.project.root))

    def check(self) -> None:
        if self.project is None or self.project.clip_list is None:
            return
        report = Path(tempfile.mkstemp(suffix=".json")[1])
        self._pending = ("check", report)
        self.jobs.run(
            Job(
                "clips",
                ["check", str(self.project.clip_list), "--json", str(report)],
                self.project.root,
            )
        )

    def _finished(self, code) -> None:
        kind, path = self._pending or (None, None)
        self._pending = None
        if kind == "scan" and code == 0:
            self.app.open_project(path)
        elif kind == "check" and path is not None:
            try:
                self._checks = {c["name"]: c for c in json.loads(path.read_text())}
            except (OSError, ValueError):
                self._checks = {}
            path.unlink(missing_ok=True)
            self.shown()


class BoxesPage(Page):
    """Place one crop box per recording, on the project's clip list (the
    :class:`~.boxes.BoxWindow`); frames come from ``frames/`` (``--prepare``)."""

    title = "Boxes"

    def __init__(self, app) -> None:
        super().__init__(app)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(9, 9, 9, 0)
        self.label = QtWidgets.QLabel()
        bar.addWidget(self.label, 1)
        bar.addWidget(QtWidgets.QLabel("box size for unplaced lists"))
        self.box_size = QtWidgets.QSpinBox(minimum=16, maximum=1024, singleStep=8)
        self.box_size.setValue(DEFAULT_BOX_SIZE)
        bar.addWidget(self.box_size)
        self.prepare = accent("Cache one frame per clip")
        self.prepare.clicked.connect(self.cache_frames)
        bar.addWidget(self.prepare)
        reload = QtWidgets.QPushButton("Reload")
        reload.clicked.connect(self.load)
        bar.addWidget(reload)
        layout.addLayout(bar)
        self.stack = QtWidgets.QStackedWidget()
        self.jobs = JobPanel()
        self.jobs.finished.connect(lambda code: self.load() if code == 0 else None)
        self.stack.addWidget(self.jobs)
        layout.addWidget(self.stack, 1)
        self.window: BoxWindow | None = None
        self._loaded: Path | None = None

    def set_project(self, project: Project | None) -> None:
        self._loaded = None
        if self.isVisible():
            self.load()

    def shown(self) -> None:
        if self.project is not None and self._loaded != self.project.clip_list:
            self.load()

    def cache_frames(self) -> None:
        if self.project is None or self.project.clip_list is None:
            return
        self.stack.setCurrentWidget(self.jobs)
        self.jobs.run(
            Job(
                "annotate",
                [
                    str(self.project.clip_list),
                    "--prepare",
                    "--frame-dir",
                    str(self.project.frames_dir),
                ],
                self.project.root,
            )
        )

    def load(self) -> None:
        project = self.project
        if self.window is not None:
            self.stack.removeWidget(self.window)
            self.window.deleteLater()
            self.window = None
        if project is None or project.clip_list is None:
            self.label.setText("No clip list: make one on the Clips page.")
            return
        self._loaded = project.clip_list
        try:
            clips = project.load_list().resolve()
            # With a frame cached for every clip (the videos may be elsewhere), place
            # on those frames without opening a video (``--remote``).
            remote = all(
                frame_path(project.frames_dir, c.video).is_file()
                for c in clips
                if c.video is not None
            )
            state = open_state(
                project.clip_list,
                project.frames_dir,
                box_size=self.box_size.value(),
                remote=remote,
            )
        except (ValueError, SystemExit, OSError) as exc:
            self.label.setText(f"Cannot place boxes: {exc}")
            self.stack.setCurrentWidget(self.jobs)
            return
        missing = missing_frames(state, project.frames_dir)
        done, total = state.progress()
        self.label.setText(
            f"{project.clip_list.name}: {done} of {total} recordings placed"
            + (f"; {len(missing)} clip(s) have no cached frame" if missing else "")
        )
        self.prepare.setVisible(bool(missing))
        self.window = BoxWindow(state, project.frames_dir, project.clip_list)
        self.window.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.stack.addWidget(self.window)
        self.stack.setCurrentWidget(self.window)
        self.window.canvas.setFocus()


class BreathingPage(Page):
    """Check each clip's breathing events and choose its detector, mark bad
    stretches and clips (the :class:`~.breathing.EditorWindow`)."""

    title = "Breathing"

    def __init__(self, app) -> None:
        super().__init__(app)
        self.layout_ = QtWidgets.QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.message = QtWidgets.QLabel(wordWrap=True)
        self.layout_.addWidget(self.message)
        self.editor: EditorWindow | None = None
        self._loaded: Path | None = None

    def set_project(self, project: Project | None) -> None:
        if self.isVisible():
            self.load()

    def shown(self) -> None:
        if self.project is not None and self._loaded != self.project.clip_list:
            self.load()

    def open_list(self, path: Path) -> None:
        """File > Open in the editor: the clip list becomes the project's."""
        self.app.open_project(path)

    def load(self) -> None:
        project = self.project
        if self.editor is not None:
            self.editor._skip_prompt = True
            self.layout_.removeWidget(self.editor)
            self.editor.deleteLater()
            self.editor = None
        self._loaded = project.clip_list if project else None
        if project is None or project.clip_list is None:
            self.message.setText("No clip list: make one on the Clips page.")
            self.message.setVisible(True)
            return
        try:
            state = TuneState(project.clip_list)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.message.setText(f"Cannot open {project.clip_list.name}: {exc}")
            self.message.setVisible(True)
            return
        self.message.setVisible(False)
        self.editor = EditorWindow(state, opener=self.open_list)
        self.editor.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.editor.menuBar().setNativeMenuBar(False)
        self.layout_.addWidget(self.editor, 1)
        self.editor.show()

    def leave(self) -> bool:
        return self.editor is None or self.editor._confirm_discard()

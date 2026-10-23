"""The first pages: Project (open or create one), Clips (make a clip list, check its
files), Boxes (:mod:`.boxes`) and Breathing (:mod:`.breathing`)."""

import html
import json
import os
import tempfile
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .. import features
from ..annotate import frame_path, missing_frames, open_state
from ..annotate_events import TuneState
from ..config import Derive
from ..video import probe_size
from .boxes import BoxWindow
from .breathing import EditorWindow
from .jobs import Job
from .project import Project, preview_scan
from .widgets import JobPanel, accent, heading, note

FRAME_DIVISORS = (("native", 1), ("1/2", 2), ("1/3", 3), ("1/4", 4))
"""The frame sizes a new list can take, as fractions of the first video's."""
PREVIEW_MS = 300
"""How long typing pauses before the Clips page looks for the files again."""
PREVIEW_ROWS = 500
STEP_HELP = {
    "Clips": "list the videos and check their files",
    "Boxes": "one crop box per recording, around the face",
    "Breathing": "check the breathing events the network learns from",
    "Train": "split the sessions, then train",
    "Results": "score a run on its held-out sessions",
    "Predict": "traces for new videos",
}
STATE_MARK = {"done": "✓", "partly": "◐", "todo": "○"}


class Page(QtWidgets.QWidget):
    """One step of the app. *set_project* is called when the project or its clip
    list changes; *leave* may refuse to switch away (unsaved changes), and
    *attention* says what the user must still do here (shown above the page)."""

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

    def attention(self) -> str:
        return ""


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
        steps = project.steps()
        lines.append("<p><b>Where the project is</b></p><table cellpadding='3'>")
        for number, (title, what) in enumerate(STEP_HELP.items(), start=1):
            step = steps.get(title)
            state = step.state if step else "todo"
            lines.append(
                f"<tr><td>{STATE_MARK[state]}</td><td><b>{number} {title}</b></td>"
                f"<td>{what}</td><td style='color:#7b8794'>"
                f"{html.escape(step.detail) if step else ''}</td></tr>"
            )
        lines.append("</table>")
        lines.append(
            "<p>Work down the list on the left; <b>Next</b> (top right) goes on once "
            "a step is done.</p>"
        )
        self.summary.setText("".join(lines))


class ClipsPage(Page):
    """Make a clip list from a folder of videos (``zephyr clips scan``), seeing first
    what it would list, then its clips, and check every file (``zephyr clips
    check``)."""

    title = "Clips"

    def __init__(self, app) -> None:
        super().__init__(app)
        layout = QtWidgets.QHBoxLayout(self)
        self.new_form = QtWidgets.QWidget(maximumWidth=520)
        left = QtWidgets.QVBoxLayout(self.new_form)
        left.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.new_form, 1)
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
        self.glob = QtWidgets.QLineEdit("*.mp4")
        form.addRow("Videos", self.glob)
        self.name = QtWidgets.QLineEdit("clips")
        form.addRow("List name", self.name)
        self.frame = QtWidgets.QComboBox()
        for label, divisor in FRAME_DIVISORS:
            self.frame.addItem(label, divisor)
        self.frame.setToolTip(
            "The size every frame is shrunk to before the crop; boxes are placed on "
            "it, and it is fixed for the list (a new size is a new list)"
        )
        form.addRow("Frame size", self.frame)
        left.addLayout(form)
        self.naming = QtWidgets.QToolButton(
            text="Other file names",
            checkable=True,
            toolButtonStyle=QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
        )
        self.naming.toggled.connect(self._show_naming)
        left.addWidget(self.naming)
        self.naming_form = QtWidgets.QWidget()
        names = QtWidgets.QFormLayout(self.naming_form)
        names.setContentsMargins(0, 0, 0, 0)
        defaults = Derive()
        self.pattern = QtWidgets.QLineEdit(defaults.pattern)
        self.timestamps = QtWidgets.QLineEdit(defaults.timestamps)
        self.thermistor = QtWidgets.QLineEdit(defaults.thermistor)
        names.addRow("Name pattern", self.pattern)
        names.addRow("Timestamps", self.timestamps)
        names.addRow("Breathing file", self.thermistor)
        names.addRow(
            note(
                "The pattern is a regex searched in each video's file name (without "
                "extension); it must capture (?P<group>...), the session. The two "
                "files are templates over {stem} and the pattern's groups, next to "
                "the video. The defaults match video_<camera>_<session>_part_<n>.mp4 "
                "with <stem>.parquet and thermistor_<session>_part_<n>.parquet."
            )
        )
        left.addWidget(self.naming_form)
        self._show_naming(False)
        self.found_label = note("Choose the folder of videos.")
        left.addWidget(self.found_label)
        self.found = _table(["video", "session", "timestamps", "sensor"])
        left.addWidget(self.found, 1)
        self.scan_button = accent("Make the clip list")
        self.scan_button.clicked.connect(self.scan)
        left.addWidget(self.scan_button)
        self._preview = QtCore.QTimer(self, singleShot=True, interval=PREVIEW_MS)
        self._preview.timeout.connect(self.preview)
        for edit in (self.folder, self.glob, self.pattern):
            edit.textChanged.connect(lambda _=None: self._preview.start())
        for edit in (self.timestamps, self.thermistor):
            edit.textChanged.connect(lambda _=None: self._preview.start())
        self.frame.currentIndexChanged.connect(lambda _: self._show_size())
        self._native: tuple[Path, tuple[int, int] | None] | None = None

        bar = QtWidgets.QHBoxLayout()
        self.list_label = heading("Clips")
        bar.addWidget(self.list_label, 1)
        self.another = QtWidgets.QToolButton(
            text="New clip list",
            checkable=True,
            toolButtonStyle=QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
        )
        self.another.toggled.connect(self._show_form)
        bar.addWidget(self.another)
        self.edit = QtWidgets.QPushButton("Edit TOML")
        self.edit.setToolTip(
            "Open the clip list in your text editor (add or change clips, file "
            "names, [derive]); save it there, then Reload"
        )
        self.edit.clicked.connect(self.edit_list)
        bar.addWidget(self.edit)
        self.reload = QtWidgets.QPushButton("Reload")
        self.reload.setToolTip("Read the clip list again, as saved on disk")
        self.reload.clicked.connect(self.reload_list)
        bar.addWidget(self.reload)
        check = accent("Check files")
        check.setToolTip("Read every file the list names and say what would be refused")
        check.clicked.connect(self.check)
        bar.addWidget(check)
        right.addLayout(bar)
        self.hint = note("")
        self.hint.setVisible(False)
        right.addWidget(self.hint)
        self.problem = QtWidgets.QLabel(objectName="bad", wordWrap=True)
        self.problem.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.problem.setVisible(False)
        right.addWidget(self.problem)
        self.table = _table(
            ["clip", "group", "sensor", "box", "events", "mark", "check"]
        )
        right.addWidget(self.table, 1)
        self.jobs = JobPanel()
        self.jobs.finished.connect(self._finished)
        self._pending: tuple[str, Path] | None = None
        self._checks: dict[str, dict] = {}

    def _show_naming(self, shown: bool) -> None:
        self.naming_form.setVisible(shown)
        self.naming.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if shown else QtCore.Qt.ArrowType.RightArrow
        )

    def _show_form(self, shown: bool) -> None:
        self.new_form.setVisible(shown)
        self.another.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if shown else QtCore.Qt.ArrowType.LeftArrow
        )

    def _browse(self) -> None:
        start = str(self.project.root) if self.project else ""
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Video folder", start)
        if folder:
            self.folder.setText(folder)

    def _derive(self) -> Derive:
        return Derive(
            pattern=self.pattern.text(),
            timestamps=self.timestamps.text(),
            thermistor=self.thermistor.text(),
        )

    def preview(self) -> None:
        """What the scan would list, as the fields are typed: each video's session
        and whether its timestamps and breathing file are there."""
        self.found.setRowCount(0)
        folder = self.folder.text().strip()
        if not folder:
            self.found_label.setText("Choose the folder of videos.")
            return
        try:
            found = preview_scan(Path(folder), self.glob.text().strip(), self._derive())
        except ValueError as exc:
            self.found_label.setText(f"File names: {exc}")
            return
        if not found:
            self.found_label.setText(
                f"No video in {folder} matches {self.glob.text()!r}."
            )
            self._native = None
            self._show_size()
            return
        unmatched = sum(f.group is None for f in found)
        stamped = sum(f.timestamps for f in found)
        sensed = sum(f.sensor for f in found)
        problems = []
        if unmatched:
            problems.append(
                f"{unmatched} name(s) do not match the pattern (Other file names)"
            )
        if stamped < len(found):
            problems.append(f"{len(found) - stamped} without timestamps")
        if sensed < len(found):
            problems.append(f"{len(found) - sensed} without a breathing file")
        groups = len({f.group for f in found if f.group is not None})
        self.found_label.setText(
            f"{len(found)} video(s) in {groups} session(s). "
            + ("; ".join(problems) + "." if problems else "Every file is there.")
        )
        _fill(
            self.found,
            [
                [
                    f.video.name,
                    f.group or "no match",
                    "✓" if f.timestamps else "✗",
                    "✓" if f.sensor else "✗",
                ]
                for f in found[:PREVIEW_ROWS]
            ],
        )
        first = found[0].video
        if self._native is None or self._native[0] != first:
            try:
                self._native = (first, probe_size(first))
            except Exception:  # noqa: BLE001 - only the size label goes without
                self._native = (first, None)
        self._show_size()

    def frame_size(self) -> tuple[int, int] | None:
        """The frame size chosen, from the first video's (None while unknown)."""
        if self._native is None or self._native[1] is None:
            return None
        width, height = self._native[1]
        divisor = self.frame.currentData()
        return width // divisor, height // divisor

    def _show_size(self) -> None:
        size = self.frame_size()
        for i, (label, divisor) in enumerate(FRAME_DIVISORS):
            native = self._native[1] if self._native else None
            text = label
            if native is not None:
                text = f"{label}: {native[0] // divisor} x {native[1] // divisor}"
            self.frame.setItemText(i, text)
        self.frame.setEnabled(size is not None)

    def set_project(self, project: Project | None) -> None:
        self._checks = {}
        listed = project is not None and project.clip_list is not None
        self.another.setChecked(not listed)  # no list yet: make one
        self._show_form(self.another.isChecked())
        self.shown()

    def edit_list(self) -> None:
        """Open the clip list in the system's editor for ``.toml`` files (it is the
        user's file to edit; the app only reads it again on Reload)."""
        path = self.project.clip_list if self.project else None
        if path is None:
            return
        url = QtCore.QUrl.fromLocalFile(str(path))
        if not QtGui.QDesktopServices.openUrl(url):
            QtGui.QGuiApplication.clipboard().setText(str(path))
            QtWidgets.QMessageBox.information(
                self,
                "No editor for .toml files",
                f"No program opens .toml files here. Open {path} in a text editor "
                "(its path is on the clipboard), save it, then Reload.",
            )
        self.hint.setText(
            f"{path.name} is open in your editor: save it there, then Reload."
        )
        self.hint.setVisible(True)

    def reload_list(self) -> None:
        """Read the clip list again, after it was edited outside the app."""
        if self.project is not None and self.project.clip_list is not None:
            self.hint.setVisible(False)
            self.app.open_project(self.project.clip_list)

    def shown(self) -> None:
        project = self.project
        self.problem.setVisible(False)
        listed = project is not None and project.clip_list is not None
        self.edit.setEnabled(listed)
        self.reload.setEnabled(listed)
        if not listed:
            self.list_label.setText("Clips: no clip list yet")
            self.table.setRowCount(0)
            return
        self.list_label.setText(f"Clips of {project.clip_list.name}")
        try:
            clip_list = project.load_list()
        except ValueError as exc:
            self.table.setRowCount(0)
            self.problem.setText(
                f"{project.clip_list.name} does not load: {exc}\nFix it with Edit "
                "TOML, then Reload."
            )
            self.problem.setVisible(True)
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
        size = self.frame_size()
        if size is not None and self.frame.currentData() != 1:
            argv += ["--target-size", str(size[0]), str(size[1])]
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
        handle, name = tempfile.mkstemp(suffix=".json")
        os.close(handle)  # Windows cannot remove a file still open
        report = Path(name)
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
    :class:`~.boxes.BoxWindow`, its frame size the list's, set on Clips); frames
    come from ``frames/`` (``--prepare``)."""

    title = "Boxes"

    def __init__(self, app) -> None:
        super().__init__(app)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(9, 9, 9, 0)
        self.label = QtWidgets.QLabel(wordWrap=True)
        bar.addWidget(self.label, 1)
        self.prepare = accent("Cache one frame per clip")
        self.prepare.setToolTip(
            "Decode one frame of every clip into frames/, to place the boxes on"
        )
        self.prepare.clicked.connect(self.cache_frames)
        bar.addWidget(self.prepare)
        reload = QtWidgets.QPushButton("Reload")
        reload.clicked.connect(self.load)
        bar.addWidget(reload)
        layout.addLayout(bar)
        self.stack = QtWidgets.QStackedWidget()
        self.empty = QtWidgets.QWidget()
        self.stack.addWidget(self.empty)
        layout.addWidget(self.stack, 1)
        self.jobs = JobPanel()
        self.jobs.finished.connect(lambda code: self.load() if code == 0 else None)
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
        self.prepare.setVisible(False)
        if project is None or project.clip_list is None:
            self.label.setText("No clip list yet: make one on 1 Clips.")
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
            state = open_state(project.clip_list, project.frames_dir, remote=remote)
        except (ValueError, SystemExit, OSError) as exc:
            self.label.setText(f"Cannot place boxes: {exc}")
            return
        missing = missing_frames(state, project.frames_dir)
        done, total = state.progress()
        if missing:
            self.label.setText(
                f"{len(missing)} clip(s) have no cached frame yet, so their boxes "
                "would be placed on black: cache them first."
            )
        else:
            self.label.setText(
                f"{done} of {total} recordings placed. Click the face (mouth and "
                "nose) in each; Auto (A) places the rest from those placed by hand. "
                "Every change is saved."
            )
        self.prepare.setVisible(bool(missing))
        self.window = BoxWindow(
            state, project.frames_dir, project.clip_list, frame_fixed=True
        )
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
        self._loaded = None  # the list may have changed on disk (Reload): read again
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
            self.message.setText("No clip list yet: make one on 1 Clips.")
            self.message.setVisible(True)
            return
        try:
            state = TuneState(project.clip_list)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.message.setText(f"Cannot open {project.clip_list.name}: {exc}")
            self.message.setVisible(True)
            return
        self.message.setVisible(False)
        self.editor = EditorWindow(state, opener=self.open_list, suggest=True)
        self.editor.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.editor.menuBar().setNativeMenuBar(False)
        self.layout_.addWidget(self.editor, 1)
        self.editor.show()

    def leave(self) -> bool:
        return self.editor is None or self.editor._confirm_discard()

    def attention(self) -> str:
        if self.editor is not None and self.editor.state.modified:
            return "Breathing has unsaved changes: Save (Ctrl+S) keeps them"
        return ""

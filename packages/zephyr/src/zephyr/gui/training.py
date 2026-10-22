"""The Train page (write an experiment, preprocess, smoke-run and run it, watch its
curves and live validation snippets) and the Results page (a run's scores)."""

import json
import math
import re
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ..config import Experiment, load
from .jobs import Job
from .pages import Page, _fill, _table
from .project import Project, run_status, split_groups
from .widgets import (
    HistoryPlot,
    JobPanel,
    SnippetGallery,
    accent,
    find_data,
    heading,
    note,
    read_history,
)

RUN_LINE = re.compile(r"^=== (\S+)\s+seed (\S+)\s+-> (.+)$")
"""What ``zephyr run`` prints as each (fold, seed) starts."""
HISTORY_MS = 2000


def _checklist(items: list[str]) -> QtWidgets.QListWidget:
    widget = QtWidgets.QListWidget()
    for item in items:
        entry = QtWidgets.QListWidgetItem(item)
        entry.setFlags(entry.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        entry.setCheckState(QtCore.Qt.CheckState.Unchecked)
        widget.addItem(entry)
    widget.setMaximumHeight(110)
    return widget


def _checked(widget: QtWidgets.QListWidget) -> list[str]:
    return [
        widget.item(i).text()
        for i in range(widget.count())
        if widget.item(i).checkState() == QtCore.Qt.CheckState.Checked
    ]


def experiment_lists(experiment: Experiment) -> list[Path]:
    """Every clip list an experiment's folds read, once each."""
    paths = []
    for fold in experiment.fold:
        paths += [s.clips for s in fold.train] + [s.clips for s in fold.test.values()]
        if fold.val is not None:
            paths.append(fold.val.clips)
    return list(dict.fromkeys(paths))


class NewExperiment(QtWidgets.QGroupBox):
    """A form writing a one-fold experiment over the project's clip list."""

    written = QtCore.Signal(Path)

    def __init__(self, page: "TrainPage") -> None:
        super().__init__()
        self.page = page
        form = QtWidgets.QFormLayout(self)
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("e.g. mouse-5-held-out")
        form.addRow("Name", self.name)
        self.train = _checklist([])
        self.val = _checklist([])
        self.test = _checklist([])
        split = QtWidgets.QHBoxLayout()
        self.shares = {}
        for side, value in (("train", 60), ("val", 20), ("test", 20)):
            box = QtWidgets.QSpinBox(minimum=0, maximum=100, value=value, suffix=" %")
            split.addWidget(QtWidgets.QLabel(side))
            split.addWidget(box)
            self.shares[side] = box
        split.addStretch(1)
        form.addRow("Split", split)
        draw = QtWidgets.QHBoxLayout()
        self.draw = QtWidgets.QSpinBox(minimum=0, maximum=99999, value=1)
        draw.addWidget(self.draw)
        auto = QtWidgets.QPushButton("Auto split")
        auto.clicked.connect(self.auto_split)
        draw.addWidget(auto)
        draw.addStretch(1)
        form.addRow("Draw", draw)
        form.addRow("Train on", self.train)
        form.addRow("Stop early on", self.val)
        form.addRow("Test on", self.test)
        form.addRow(
            note(
                "Groups are sessions. Auto split draws them in those shares (a new "
                "draw number, another split); the ticks can then be changed. Hold "
                "out whole animals for a fair test; a video may not be on two sides."
            )
        )
        self.seeds = QtWidgets.QLineEdit("1")
        form.addRow("Seeds", self.seeds)
        self.epochs = QtWidgets.QSpinBox(minimum=1, maximum=10000, value=200)
        self.steps = QtWidgets.QSpinBox(minimum=1, maximum=100000, value=200)
        form.addRow("Epochs", self.epochs)
        form.addRow("Steps per epoch", self.steps)
        self.device = QtWidgets.QComboBox()
        self.device.addItems(["(default)", "cuda", "cpu"])
        self.amp = QtWidgets.QComboBox()
        self.amp.addItems(["(default)", "bf16", "fp16", "off"])
        self.workers = QtWidgets.QSpinBox(minimum=-1, maximum=64, value=-1)
        self.workers.setSpecialValueText("(default)")
        form.addRow("Device", self.device)
        form.addRow("Precision", self.amp)
        form.addRow("Data workers", self.workers)
        write = accent("Write experiment")
        write.clicked.connect(self.write)
        form.addRow(write)

    def set_groups(self, groups: list[str]) -> None:
        for widget in (self.train, self.val, self.test):
            widget.clear()
            for group in groups:
                entry = QtWidgets.QListWidgetItem(group)
                entry.setFlags(entry.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                entry.setCheckState(QtCore.Qt.CheckState.Unchecked)
                widget.addItem(entry)

    def auto_split(self) -> None:
        """Tick the groups of a draw in the chosen shares."""
        groups = [self.train.item(i).text() for i in range(self.train.count())]
        try:
            sides = split_groups(
                groups,
                tuple(self.shares[s].value() for s in ("train", "val", "test")),
                self.draw.value(),
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot split", str(exc))
            return
        for widget, chosen in zip((self.train, self.val, self.test), sides):
            for i in range(widget.count()):
                item = widget.item(i)
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked
                    if item.text() in chosen
                    else QtCore.Qt.CheckState.Unchecked
                )
        if not self.name.text().strip():
            self.name.setText(f"split-{self.draw.value()}")

    def write(self) -> None:
        project = self.page.project
        if project is None:
            return
        try:
            seeds = tuple(int(s) for s in self.seeds.text().replace(",", " ").split())
            params = {}
            if self.epochs.value() != 200:
                params["epochs"] = self.epochs.value()
            if self.steps.value() != 200:
                params["steps_per_epoch"] = self.steps.value()
            machine = {
                "device": None
                if self.device.currentIndex() == 0
                else self.device.currentText(),
                "amp": None if self.amp.currentIndex() == 0 else self.amp.currentText(),
                "num_workers": None
                if self.workers.value() < 0
                else self.workers.value(),
            }
            path = project.new_experiment(
                self.name.text().strip(),
                train=_checked(self.train),
                val=_checked(self.val) or None,
                test={"test": _checked(self.test)},
                seeds=seeds or (1,),
                train_params=params,
                machine=machine,
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot write the experiment", str(exc))
            return
        self.written.emit(path)


class TrainPage(Page):
    title = "Train"

    def __init__(self, app) -> None:
        super().__init__(app)
        layout = QtWidgets.QHBoxLayout(self)
        columns = QtWidgets.QSplitter()
        layout.addWidget(columns)
        scroll = self.scroll = QtWidgets.QScrollArea(widgetResizable=True)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        left_widget = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(left_widget)
        scroll.setWidget(left_widget)
        columns.addWidget(scroll)

        self.buttons = {}

        def button(key: str, text: str) -> QtWidgets.QPushButton:
            made = accent(text) if key == "run" else QtWidgets.QPushButton(text)
            made.clicked.connect(lambda _=False: self.start(key))
            self.buttons[key] = made
            return made

        left.addWidget(heading("1. Preprocess"))
        left.addWidget(button("preprocess", "Preprocess the clips"))
        left.addWidget(
            note(
                "Crops every clip into cache/ once; any experiment on these clips "
                "reuses it, so this can run while you set one up."
            )
        )

        left.addWidget(heading("2. Set up the experiment"))
        self.experiment = QtWidgets.QComboBox()
        self.experiment.currentIndexChanged.connect(self._experiment_changed)
        left.addWidget(self.experiment)
        self.summary = note("")
        left.addWidget(self.summary)
        self.define = QtWidgets.QToolButton(
            text="Define a new experiment",
            checkable=True,
            toolButtonStyle=QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
        )
        self.define.toggled.connect(self._show_form)
        left.addWidget(self.define)
        self.form = NewExperiment(self)
        self.form.written.connect(self._written)
        left.addWidget(self.form)

        left.addWidget(heading("3. Train"))
        left.addWidget(button("smoke", "Smoke run (2 steps)"))
        left.addWidget(button("run", "Run"))
        left.addWidget(button("resume", "Resume a stopped run"))
        snippets = QtWidgets.QHBoxLayout()
        self.want_snippets = QtWidgets.QCheckBox("validation snippets", checked=True)
        self.snippet_clips = QtWidgets.QSpinBox(minimum=1, maximum=12, value=4)
        self.snippet_seconds = QtWidgets.QSpinBox(minimum=2, maximum=120, value=20)
        snippets.addWidget(self.want_snippets)
        snippets.addWidget(self.snippet_clips)
        snippets.addWidget(QtWidgets.QLabel("clips x"))
        snippets.addWidget(self.snippet_seconds)
        snippets.addWidget(QtWidgets.QLabel("s"))
        left.addLayout(snippets)
        left.addWidget(
            note(
                "Snippets are shown while a run started here trains, and never saved. "
                "The curves come from the run's history.json, so they show any run."
            )
        )
        left.addStretch(1)
        # The column is as wide as its widest row (the form, measured while shown),
        # so nothing is cut off on the right.
        bar_width = scroll.verticalScrollBar().sizeHint().width()
        scroll.setMinimumWidth(left_widget.minimumSizeHint().width() + bar_width + 4)
        self._show_form(False)

        right_widget = QtWidgets.QWidget()
        right = QtWidgets.QVBoxLayout(right_widget)
        right.setContentsMargins(0, 0, 0, 0)
        columns.addWidget(right_widget)
        columns.setStretchFactor(1, 1)
        columns.setCollapsible(0, False)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Run"))
        self.run_dir = QtWidgets.QComboBox()
        self.run_dir.currentIndexChanged.connect(lambda _: self.refresh_history())
        bar.addWidget(self.run_dir, 1)
        self.status = QtWidgets.QLabel()
        bar.addWidget(self.status)
        right.addLayout(bar)
        self.tabs = QtWidgets.QTabWidget()
        self.history = HistoryPlot()
        self.gallery = SnippetGallery()
        self.jobs = JobPanel()
        self.jobs.line.connect(self._line)
        self.jobs.snippets.connect(self.gallery.add)
        self.jobs.finished.connect(lambda _: self._fill_runs())
        self.tabs.addTab(self.history, "Progress")
        self.tabs.addTab(self.gallery, "Snippets")
        self.tabs.addTab(self.jobs, "Log")
        right.addWidget(self.tabs, 1)
        self._timer = QtCore.QTimer(self, interval=HISTORY_MS)
        self._timer.timeout.connect(self.refresh_history)
        self._timer.start()
        self._smoke = False

    # ---- project ---------------------------------------------------------------

    def set_project(self, project: Project | None) -> None:
        self.shown()

    def shown(self) -> None:
        project = self.project
        current = self.experiment.currentData()
        self.experiment.blockSignals(True)
        self.experiment.clear()
        if project is not None:
            for path in project.experiments():
                self.experiment.addItem(path.name, path)
            try:
                self.form.set_groups(project.groups())
            except ValueError:
                self.form.set_groups([])
        index = find_data(self.experiment, current)
        self.experiment.setCurrentIndex(max(index, 0))
        self.experiment.blockSignals(False)
        if self.experiment.count() == 0:
            self.define.setChecked(True)  # nothing to train yet: define one
        self._experiment_changed()

    def _show_form(self, shown: bool) -> None:
        self.form.setVisible(shown)
        self.define.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if shown else QtCore.Qt.ArrowType.RightArrow
        )

    def _written(self, path: Path) -> None:
        self.shown()
        self.experiment.setCurrentIndex(find_data(self.experiment, path))
        self.define.setChecked(False)
        self.summary.setText(
            f"Wrote experiments/{path.name}.\n{self.summary.text()}\n"
            "Next: 3. Smoke run, then Run."
        )
        self.app.refresh_steps()

    def _loaded(self) -> Experiment | None:
        path = self.experiment.currentData()
        if path is None:
            return None
        return load(Experiment, path)

    def _experiment_changed(self) -> None:
        path = self.experiment.currentData()
        for button in self.buttons.values():
            button.setEnabled(path is not None)
        listed = self.project is not None and self.project.clip_list is not None
        self.buttons["preprocess"].setEnabled(path is not None or listed)
        if path is None:
            self.summary.setText("No experiment yet: define one below.")
            self.run_dir.clear()
            return
        try:
            experiment = self._loaded()
        except ValueError as exc:
            self.summary.setText(f"Does not load: {exc}")
            for button in self.buttons.values():
                button.setEnabled(False)
            return
        lines = []
        for fold in experiment.fold:
            groups = [g for s in fold.train for g in (s.groups or ["all"])]
            val = fold.val.groups if fold.val is not None else None
            lines.append(
                f"fold {fold.name}: train {', '.join(groups)}; "
                + (
                    f"stop on {', '.join(val or ['all'])}; "
                    if fold.val is not None
                    else f"stop on the last {fold.train_params.val_fraction:.0%} of "
                    "each clip; "
                )
                + "test "
                + "; ".join(
                    f"{name} = {', '.join(s.groups or ['all'])}"
                    for name, s in fold.test.items()
                )
                + f"; {fold.train_params.epochs} epochs"
            )
        lines.append(f"seeds {experiment.seeds} -> {experiment.output_dir}")
        self.summary.setText("\n".join(lines))
        self._fill_runs()

    def _fill_runs(self) -> None:
        path = self.experiment.currentData()
        current = self.run_dir.currentData()
        self.run_dir.blockSignals(True)
        self.run_dir.clear()
        if path is not None and self.project is not None:
            try:
                dirs = self.project.run_dirs(path) + self.project.run_dirs(
                    path, smoke=True
                )
            except ValueError:
                dirs = []
            for run in dirs:
                label = f"{run.parent.name}/{run.name}"
                if run.parent.parent.name == "smoke":
                    label = "smoke: " + label
                self.run_dir.addItem(label, run)
        index = find_data(self.run_dir, current)
        self.run_dir.setCurrentIndex(max(index, 0))
        self.run_dir.blockSignals(False)
        self.refresh_history()

    # ---- running ---------------------------------------------------------------

    def start(self, step: str) -> None:
        project, path = self.project, self.experiment.currentData()
        if project is None:
            return
        if step == "preprocess":
            if path is not None:  # the experiment's lists, into its cache
                experiment = self._loaded()
                argv = [*map(str, experiment_lists(experiment))]
                argv += ["--cache", str(experiment.features_dir)]
            elif project.clip_list is not None:  # where new experiments look
                argv = [str(project.clip_list), "--cache", str(project.cache_dir)]
            else:
                return
            job = Job("preprocess", argv, project.root)
        elif path is None:
            return
        else:
            argv = [str(path)]
            if step == "smoke":  # a check of the plumbing: always from scratch
                argv += ["--smoke", "--restart"]
            if step == "resume":
                argv.append("--resume")
            if step == "run":
                if not self._may_start_over(project, path):
                    return
                argv.append("--restart")
            snippets = (
                {
                    "clips": self.snippet_clips.value(),
                    "seconds": self.snippet_seconds.value(),
                }
                if self.want_snippets.isChecked()
                else None
            )
            job = Job("run", argv, project.root, snippets=snippets)
            self.gallery.clear()
        self._smoke = step == "smoke"
        if self.jobs.run(job):
            self.tabs.setCurrentWidget(
                self.jobs if step == "preprocess" else self.history
            )

    def _may_start_over(self, project: Project, path: Path) -> bool:
        """Whether Run may train *path* from scratch: yes when none of its runs
        exists, else only if the user agrees to delete them."""
        existing = [
            (run, run_status(run)) for run in project.run_dirs(path) if run_status(run)
        ]
        if not existing:
            return True
        listed = "\n".join(f"  {r.parent.name}/{r.name}: {s}" for r, s in existing)
        answer = QtWidgets.QMessageBox.question(
            self,
            "Start over?",
            f"{path.name} already has a run:\n{listed}\n\nRun deletes it (checkpoints, "
            "history and scores) and trains from scratch. To continue it instead use "
            "Resume; to score it as it is use Score on 5 Results.",
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def _line(self, line: str) -> None:
        match = RUN_LINE.match(line.strip())
        if match:
            self._fill_runs()
            index = find_data(self.run_dir, Path(match.group(3)))
            if index >= 0:
                self.run_dir.setCurrentIndex(index)

    def refresh_history(self) -> None:
        run = self.run_dir.currentData()
        if run is None or not self.isVisible():
            return
        rows = read_history(run)
        self.history.set_history(rows)
        if not rows:
            self.status.setText("not started")
            return
        config = run / "config.json"
        epochs = None
        try:
            epochs = json.loads(config.read_text())["fold"]["train_params"]["epochs"]
        except (OSError, ValueError, KeyError):
            pass
        last = rows[-1]
        seconds = sum(r.get("seconds", 0) for r in rows) / len(rows)
        text = f"epoch {last['epoch'] + 1}"
        if epochs:
            left = epochs - last["epoch"] - 1
            text += f" of {epochs}, {seconds:.0f} s each"
            if left > 0 and self.jobs.busy:
                text += f", at most {left * seconds / 60:.0f} min left"
        if (run / "evaluation.json").is_file():
            text += "; finished and scored"
        self.status.setText(text)


class ResultsPage(Page):
    """A run's ``evaluation.json`` (and any later scoring) per test group and clip,
    its training curves, and scoring its checkpoint on another clip list."""

    title = "Results"

    def __init__(self, app) -> None:
        super().__init__(app)
        layout = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(heading("Runs"))
        self.runs = QtWidgets.QListWidget()
        self.runs.currentRowChanged.connect(lambda _: self.show_run())
        left.addWidget(self.runs, 1)
        other = QtWidgets.QPushButton("Open a run folder...")
        other.clicked.connect(self._browse)
        left.addWidget(other)
        holder = QtWidgets.QWidget(maximumWidth=380)
        holder.setLayout(left)
        layout.addWidget(holder)

        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 1)
        bar = QtWidgets.QHBoxLayout()
        self.title_label = heading("")
        bar.addWidget(self.title_label)
        self.run_state = note("")
        bar.addWidget(self.run_state, 1)
        own = accent("Score its test groups")
        own.setToolTip(
            "Score this run's best.pt, finished or stopped, on its experiment's test "
            "groups (zephyr run --evaluate-only), writing evaluation.json"
        )
        own.clicked.connect(self.score_own)
        bar.addWidget(own)
        bar.addWidget(QtWidgets.QLabel("scores"))
        self.result = QtWidgets.QComboBox()
        self.result.currentIndexChanged.connect(lambda _: self.show_result())
        bar.addWidget(self.result)
        right.addLayout(bar)
        self.tabs = QtWidgets.QTabWidget()
        self.groups = _table(
            [
                "group",
                "recordings",
                "correlation",
                "inhale F1",
                "exhale F1",
                "KL-IBI",
                "head inhale F1",
            ]
        )
        self.clips = _table(
            [
                "group",
                "clip",
                "correlation",
                "inhale F1",
                "exhale F1",
                "KL-IBI",
                "head inhale F1",
            ]
        )
        self.clips.setSortingEnabled(True)
        self.history = HistoryPlot()
        score = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(score)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Score best.pt on"))
        self.score_list = QtWidgets.QComboBox()
        row.addWidget(self.score_list, 1)
        go = accent("Score")
        go.clicked.connect(self.score)
        row.addWidget(go)
        form.addLayout(row)
        form.addWidget(
            note(
                "Writes scores/<list>.json in the run folder (zephyr evaluate, on the "
                "project's cache: preprocess the list first). Clips the checkpoint "
                "trained on are refused."
            )
        )
        self.jobs = JobPanel()
        self.jobs.finished.connect(lambda _: self.show_run())
        form.addWidget(self.jobs, 1)
        self.tabs.addTab(self.groups, "Test groups")
        self.tabs.addTab(self.clips, "Clips")
        self.tabs.addTab(self.history, "Training")
        self.tabs.addTab(score, "Score more / log")
        right.addWidget(self.tabs, 1)
        self._extra: list[Path] = []

    def set_project(self, project: Project | None) -> None:
        self.shown()

    def shown(self) -> None:
        project = self.project
        current = self._current()
        self.runs.blockSignals(True)
        self.runs.clear()
        dirs = (project.runs() if project else []) + self._extra
        for run in dirs:
            label = (
                str(run.relative_to(project.runs_dir))
                if (project and run.is_relative_to(project.runs_dir))
                else str(run)
            )
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, run)
            self.runs.addItem(item)
        self.score_list.clear()
        if project is not None:
            for path in project.clip_lists():
                self.score_list.addItem(path.name, path)
        rows = [
            self.runs.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
            for i in range(self.runs.count())
        ]
        self.runs.setCurrentRow(rows.index(current) if current in rows else 0)
        self.runs.blockSignals(False)
        self.show_run()

    def _browse(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Run folder")
        if folder:
            self._extra.append(Path(folder))
            self.shown()

    def _current(self) -> Path | None:
        item = self.runs.currentItem()
        return item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None

    def show_run(self) -> None:
        run = self._current()
        self.result.blockSignals(True)
        self.result.clear()
        self.run_state.setText(run_status(run) if run is not None else "")
        if run is None:
            self.title_label.setText("No runs yet")
            self.result.blockSignals(False)
            self.groups.setRowCount(0)
            self.clips.setRowCount(0)
            self.history.set_history([])
            return
        self.title_label.setText(f"{run.parent.name}/{run.name}")
        for path in [run / "evaluation.json", *sorted((run / "scores").glob("*.json"))]:
            if path.is_file():
                self.result.addItem(path.relative_to(run).as_posix(), path)
        self.result.blockSignals(False)
        self.history.set_history(read_history(run))
        self.show_result()

    def show_result(self) -> None:
        path = self.result.currentData()
        if path is None:
            self.groups.setRowCount(0)
            self.clips.setRowCount(0)
            return
        result = json.loads(path.read_text())

        def cells(summary: dict) -> list:
            return [
                _fmt(summary.get(k))
                for k in (
                    "correlation",
                    "inhale_f1",
                    "exhale_f1",
                    "kl_ibi",
                    "head_inhale_f1",
                )
            ]

        _fill(
            self.groups,
            [
                [name, len(group["recordings"]), *cells(group["summary"])]
                for name, group in result["groups"].items()
            ],
        )
        self.clips.setSortingEnabled(False)
        _fill(
            self.clips,
            [
                [row.get("group"), row["clip_id"], *cells(row)]
                for row in result["clips"]
            ],
        )
        self.clips.setSortingEnabled(True)

    def score_own(self) -> None:
        """Score the run's best.pt on its fold's test groups, asking before an
        evaluation.json is replaced."""
        run, project = self._current(), self.project
        if run is None or project is None:
            return
        found = project.run_of(run) if run.is_relative_to(project.runs_dir) else None
        if found is None:
            QtWidgets.QMessageBox.information(
                self,
                "Not an experiment's run",
                "Only runs of this project's experiments are scored here: a run "
                "folder opened from elsewhere is read, never written to.",
            )
            return
        if not (run / "best.pt").is_file():
            QtWidgets.QMessageBox.information(
                self, "Nothing to score", f"{run.name} has no best.pt yet."
            )
            return
        if (run / "evaluation.json").is_file():
            answer = QtWidgets.QMessageBox.question(
                self,
                "Replace the scores?",
                f"{run.parent.name}/{run.name} already has an evaluation.json. Score "
                "best.pt again and replace it?",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        experiment, fold, seed, smoke = found
        argv = [str(experiment), "--evaluate-only", "--folds", fold]
        argv += ["--seeds", str(seed)] + (["--smoke"] if smoke else [])
        self.tabs.setCurrentIndex(self.tabs.count() - 1)  # where the log is
        self.jobs.run(Job("run", argv, project.root))

    def score(self) -> None:
        run, clips = self._current(), self.score_list.currentData()
        if run is None or clips is None or self.project is None:
            return
        if not run.is_relative_to(self.project.runs_dir):
            QtWidgets.QMessageBox.information(
                self,
                "Not this project's run",
                "Only runs of this project are scored here: a run folder opened "
                "from elsewhere is read, never written to.",
            )
            return
        out = run / "scores" / f"{clips.stem}.json"
        self.jobs.run(
            Job(
                "evaluate",
                [
                    "--checkpoint",
                    str(run / "best.pt"),
                    "--clips",
                    str(clips),
                    "--cache",
                    str(self.project.cache_dir),
                    "--out",
                    str(out),
                ],
                self.project.root,
            )
        )


def _fmt(value) -> str:
    if isinstance(value, int | float):
        return f"{value:.3f}" if math.isfinite(value) else "n/a"
    return "" if value is None else str(value)

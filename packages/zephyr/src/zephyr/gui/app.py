"""The app's main window: the steps on the left, each marked done, partly done or to
do (:meth:`.project.Project.steps`), and the current one on the right."""

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .pages import BoxesPage, BreathingPage, ClipsPage, ProjectPage
from .predicting import PredictPage
from .project import DONE, PARTLY, TODO, Project
from .theme import STYLE, THEME
from .training import ResultsPage, TrainPage
from .widgets import find_data

STEP_MARKS = {DONE: ("✓", "placed"), PARTLY: ("◐", "span"), TODO: ("○", "muted")}
"""The mark and theme colour of a step in the list, by its state."""
STEPS_MS = 3000
"""How often the steps' marks are read again from the project folder."""

PAGES = (
    ProjectPage,
    ClipsPage,
    BoxesPage,
    BreathingPage,
    TrainPage,
    ResultsPage,
    PredictPage,
)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, project: Project | None = None) -> None:
        super().__init__()
        self.project: Project | None = None
        body = QtWidgets.QWidget()
        split = QtWidgets.QHBoxLayout(body)
        split.setContentsMargins(0, 0, 0, 0)
        self.nav = QtWidgets.QListWidget(objectName="nav")
        self.nav.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.nav.setFixedWidth(180)
        self.stack = QtWidgets.QStackedWidget()
        self.pages = [page(self) for page in PAGES]
        for page in self.pages:
            self.nav.addItem("")
            self.stack.addWidget(page)
            if getattr(page, "jobs", None) is not None:
                page.jobs.started.connect(lambda _line, p=page: self._busy(p))
                page.jobs.finished.connect(lambda _code, p=page: self._busy(p))
        self._held: dict[int, list] = {}
        self.running = QtWidgets.QLabel(objectName="jobstatus")
        self.running.setProperty("state", "running")
        self.statusBar().addWidget(self.running, 1)
        self._busy_timer = QtCore.QTimer(self, interval=1000)
        self._busy_timer.timeout.connect(self._show_running)
        self._steps_timer = QtCore.QTimer(self, interval=STEPS_MS)
        self._steps_timer.timeout.connect(self.refresh_steps)
        self._steps_timer.start()
        self.nav.currentRowChanged.connect(self._go)
        split.addWidget(self.nav)
        split.addWidget(self.stack, 1)
        self.setCentralWidget(body)
        self._build_menu()
        self.lists = QtWidgets.QComboBox()
        self.lists.activated.connect(self._list_chosen)
        self.statusBar().addPermanentWidget(QtWidgets.QLabel("clip list"))
        self.statusBar().addPermanentWidget(self.lists)
        self.set_project(project)
        self.nav.setCurrentRow(0 if project is None else 1)

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("&Project")
        for text, slot, keys in (
            (
                "&Open or create a project folder...",
                self.choose_project,
                "Ctrl+Shift+O",
            ),
            ("Open a &clip list...", self.choose_clip_list, None),
            ("&Quit", self.close, "Ctrl+Q"),
        ):
            action = QtGui.QAction(text, self)
            if keys:
                action.setShortcut(QtGui.QKeySequence(keys))
            action.triggered.connect(lambda _=False, s=slot: s())
            menu.addAction(action)

    # ---- the project -----------------------------------------------------------

    def choose_project(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Project folder (an empty one starts a new project)"
        )
        if folder:
            self.open_project(Path(folder))

    def choose_clip_list(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Clip list", "", "Clip lists (*.toml)"
        )
        if path:
            self.open_project(Path(path))

    def open_project(self, path: Path) -> None:
        """Open the project of *path* (a folder or a clip list in one)."""
        if not all(page.leave() for page in self.pages):
            return
        try:
            project = Project.open(path)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot open the project", str(exc))
            return
        self.set_project(project)

    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.setWindowTitle(f"zephyr: {project.root}" if project else "zephyr")
        self.lists.blockSignals(True)
        self.lists.clear()
        if project is not None:
            for path in project.clip_lists():
                self.lists.addItem(path.name, path)
            self.lists.setCurrentIndex(max(find_data(self.lists, project.clip_list), 0))
        self.lists.blockSignals(False)
        for page in self.pages:
            page.set_project(project)
        self.refresh_steps()

    def refresh_steps(self) -> None:
        """Mark each step in the list as done, partly done or to do, from the
        project folder; the tooltip says what the mark rests on."""
        steps = self.project.steps() if self.project is not None else {}
        for number, page in enumerate(self.pages):
            item = self.nav.item(number)
            step = steps.get(page.title)
            state = step.state if step else TODO
            mark, colour = STEP_MARKS[state]
            label = f"{number}  {page.title}" if number else page.title
            item.setText(f"{mark}  {label}")
            item.setForeground(QtGui.QColor(THEME[colour]))
            item.setToolTip(
                f"{page.title}: {state}" + (f"\n{step.detail}" if step else "")
            )

    def _busy(self, page) -> None:
        """While *page*'s command runs, grey out its buttons (not the log's Cancel and
        Copy) so nothing starts twice; give them back once it ends. Read from the job,
        not the signal, so a chain of commands holds them throughout."""
        key = id(page)
        if page.jobs.busy and key not in self._held:
            held = [
                button
                for button in page.findChildren(QtWidgets.QAbstractButton)
                if button.isEnabled() and not page.jobs.isAncestorOf(button)
            ]
            for button in held:
                button.setEnabled(False)
            self._held[key] = held
        elif not page.jobs.busy and key in self._held:
            for button in self._held.pop(key):
                button.setEnabled(True)
            self.refresh_steps()
        self._show_running()

    def _show_running(self) -> None:
        """What runs, on which page, in the status bar (any page can be shown)."""
        busy = [p for p in self.pages if getattr(p, "jobs", None) and p.jobs.busy]
        self.running.setText(
            "   ".join(f"{p.title}: running {p.jobs.describe()}" for p in busy)
        )
        if busy:
            self._busy_timer.start()
        else:
            self._busy_timer.stop()

    def _list_chosen(self, index: int) -> None:
        path = self.lists.itemData(index)
        if self.project is not None and path != self.project.clip_list:
            self.open_project(path)

    def _go(self, row: int) -> None:
        self.stack.setCurrentIndex(row)
        self.pages[row].shown()
        self.refresh_steps()

    def closeEvent(self, event) -> None:
        busy = [p.title for p in self.pages if getattr(p, "jobs", None) and p.jobs.busy]
        if busy:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Commands running",
                f"Stop what runs on {', '.join(busy)} and quit? A stopped run "
                "continues with Resume.",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        if not all(page.leave() for page in self.pages):
            event.ignore()
            return
        for page in self.pages:
            if getattr(page, "jobs", None) is not None:
                page.jobs.stop()
        event.accept()


def run_app(path: Path | None = None) -> None:  # pragma: no cover
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("zephyr")
    app.setStyleSheet(STYLE)
    window = MainWindow(Project.open(path) if path is not None else None)
    window.resize(1600, 950)
    window.show()
    app.exec()

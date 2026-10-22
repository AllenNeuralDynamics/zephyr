"""The app's main window: the steps on the left, each marked done, partly done or to
do (:meth:`.project.Project.steps`); above the current one the project, its clip list
and the next step, below it what the page's command is doing."""

import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .pages import BoxesPage, BreathingPage, ClipsPage, ProjectPage
from .predicting import PredictPage
from .project import DONE, PARTLY, TODO, Project
from .theme import STYLE, THEME
from .training import ResultsPage, TrainPage
from .widgets import ElidedLabel, find_data

STEP_MARKS = {DONE: ("✓", "placed"), PARTLY: ("◐", "span"), TODO: ("○", "muted")}
"""The mark and theme colour of a step in the list, by its state."""
STEPS_MS = 3000
"""How often the steps' marks are read again from the project folder."""
WINDOW = (1600, 950)
"""The window's size at most, in pixels."""
SCREEN_SHARE = 0.92
"""The share of the screen's free area the window takes at most."""
RECENT = 8
"""How many projects Project > Recent projects remembers."""
ICON = Path(__file__).with_name("icon.svg")
"""The app's icon: a mouse's snout breathing out."""

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
        # The user's settings (recent projects), in an .ini file, never the project.
        self.settings = QtCore.QSettings(
            QtCore.QSettings.Format.IniFormat,
            QtCore.QSettings.Scope.UserScope,
            "zephyr",
            "gui",
        )
        body = QtWidgets.QWidget()
        split = QtWidgets.QHBoxLayout(body)
        split.setContentsMargins(0, 0, 0, 0)
        split.setSpacing(0)
        self.nav = QtWidgets.QListWidget(objectName="nav")
        self.nav.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.nav.setFixedWidth(190)
        self.stack = QtWidgets.QStackedWidget()
        self.pages = [page(self) for page in PAGES]
        # Each page's command panel lives below the pages; only the shown page's is.
        self.activity = QtWidgets.QFrame(objectName="activity")
        activity = QtWidgets.QVBoxLayout(self.activity)
        activity.setContentsMargins(12, 6, 12, 6)
        for page in self.pages:
            self.nav.addItem("")
            # A page taller than the window scrolls, so the window can always be
            # made smaller (the stack would otherwise be as tall as its tallest page).
            scroll = QtWidgets.QScrollArea(widgetResizable=True)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
            if getattr(page, "jobs", None) is not None:
                activity.addWidget(page.jobs)
                page.jobs.started.connect(lambda _line, p=page: self._busy(p))
                page.jobs.finished.connect(lambda _code, p=page: self._busy(p))
        self._held: dict[int, list] = {}
        self.running = ElidedLabel(objectName="jobstatus")
        self.running.setProperty("state", "running")
        self.statusBar().addWidget(self.running, 1)
        self._busy_timer = QtCore.QTimer(self, interval=1000)
        self._busy_timer.timeout.connect(self._show_running)
        self._steps_timer = QtCore.QTimer(self, interval=STEPS_MS)
        self._steps_timer.timeout.connect(self.refresh_steps)
        self._steps_timer.start()
        self.nav.currentRowChanged.connect(self._go)
        right = QtWidgets.QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        right.addWidget(self._build_header())
        right.addWidget(self.stack, 1)
        right.addWidget(self.activity)
        split.addWidget(self.nav)
        split.addLayout(right, 1)
        self.setCentralWidget(body)
        self._build_menu()
        self.set_project(project)
        self.nav.setCurrentRow(0 if project is None else 1)

    def _build_header(self) -> QtWidgets.QWidget:
        """The project, the clip list every page works on, what needs attention on
        this page, and the way on."""
        header = QtWidgets.QFrame(objectName="header")
        bar = QtWidgets.QHBoxLayout(header)
        bar.setContentsMargins(12, 8, 12, 8)
        self.project_label = QtWidgets.QLabel(objectName="heading")
        bar.addWidget(self.project_label)
        self.project_where = ElidedLabel(objectName="sub")
        self.project_where.setMinimumWidth(60)
        self.project_where.setMaximumWidth(320)
        bar.addWidget(self.project_where, 1)
        bar.addSpacing(18)
        bar.addWidget(QtWidgets.QLabel("Clip list"))
        self.lists = QtWidgets.QComboBox(minimumContentsLength=14)
        self.lists.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.lists.setToolTip("The clip list every step works on")
        self.lists.activated.connect(self._list_chosen)
        bar.addWidget(self.lists)
        bar.addSpacing(12)
        self.attention = ElidedLabel(objectName="bad")
        self.attention.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        bar.addWidget(self.attention, 1)
        self.next = QtWidgets.QPushButton()
        self.next.clicked.connect(
            lambda: self.nav.setCurrentRow(self.stack.currentIndex() + 1)
        )
        bar.addWidget(self.next)
        return header

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
            if slot == self.choose_clip_list:
                self.recent_menu = menu.addMenu("&Recent projects")
                self.recent_menu.aboutToShow.connect(self._fill_recent)

    # ---- the project -----------------------------------------------------------

    def recent(self) -> list[Path]:
        """The projects opened last, newest first, that still exist."""
        stored = self.settings.value("recent", []) or []
        if isinstance(stored, str):  # QSettings gives a lone value as a string
            stored = [stored]
        return [Path(p) for p in stored if Path(p).is_dir()]

    def _remember(self, root: Path) -> None:
        kept = [str(root), *(str(p) for p in self.recent() if p != root)]
        self.settings.setValue("recent", kept[:RECENT])

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        for root in self.recent():
            action = self.recent_menu.addAction(f"{root.name}   ({root.parent})")
            action.triggered.connect(lambda _=False, r=root: self.open_project(r))
        if self.recent_menu.isEmpty():
            self.recent_menu.addAction("none yet").setEnabled(False)

    def choose_project(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open a project folder (or an empty folder for a new project)"
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
        create = not Project.exists(path)
        if create:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Start a new project here?",
                f"{path} is not a zephyr project yet. Make it one? It gets the "
                "folders experiments/, frames/, cache/, runs/ and predictions/, and "
                "everything made from now on goes in it.\n\nTo list videos kept in "
                "this folder instead, stay in your project and choose it as the "
                "Video folder on 1 Clips.",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        if not all(page.leave() for page in self.pages):
            return
        try:
            project = Project.open(path, create=create)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot open the project", str(exc))
            return
        self.set_project(project)

    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.setWindowTitle(f"zephyr: {project.root}" if project else "zephyr")
        self.project_label.setText(project.root.name if project else "No project")
        self.project_label.setToolTip(str(project.root) if project else "")
        self.project_where.setText(f"in {project.root.parent}" if project else "")
        if project is not None:
            self._remember(project.root)
        self.lists.blockSignals(True)
        self.lists.clear()
        if project is not None:
            for path in project.clip_lists():
                self.lists.addItem(path.name, path)
            self.lists.setCurrentIndex(max(find_data(self.lists, project.clip_list), 0))
        if not self.lists.count():
            self.lists.addItem("none yet: make one on 1 Clips")
        self.lists.setEnabled(project is not None and project.clip_list is not None)
        self.lists.blockSignals(False)
        for page in self.pages:
            page.set_project(project)
        self.refresh_steps()

    def refresh_steps(self) -> None:
        """Mark each step in the list as done, partly done or to do, from the
        project folder (the tooltip says what the mark rests on, a dot that the
        page has unsaved changes); offer the next step, highlighted once this one
        is done."""
        steps = self.project.steps() if self.project is not None else {}
        for number, page in enumerate(self.pages):
            item = self.nav.item(number)
            step = steps.get(page.title)
            state = step.state if step else TODO
            mark, colour = STEP_MARKS[state]
            label = f"{number}  {page.title}" if number else page.title
            unsaved = "  ●" if page.attention() else ""
            item.setText(f"{mark}  {label}{unsaved}")
            item.setForeground(QtGui.QColor(THEME[colour]))
            item.setToolTip(
                f"{page.title}: {state}" + (f"\n{step.detail}" if step else "")
            )
        row = self.stack.currentIndex()
        page = self.pages[row]
        self.attention.setText(page.attention())
        following = row + 1 < len(self.pages)
        self.next.setVisible(following and self.project is not None)
        if following:
            title = self.pages[row + 1].title
            self.next.setText(f"Next: {row + 1} {title}  →")
            step = steps.get(page.title)
            done = step is not None and step.state == DONE
            if self.next.objectName() != ("accent" if done else ""):
                self.next.setObjectName("accent" if done else "")
                self.next.style().polish(self.next)

    def _busy(self, page) -> None:
        """While *page*'s command runs, grey out its buttons (not the command
        panel's) so nothing starts twice; give them back once it ends. Read from the
        job, not the signal, so a chain of commands holds them throughout."""
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
            page.shown()  # the page's own rules again (what changed while it ran)
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
        if self.project is not None and path and path != self.project.clip_list:
            self.open_project(path)

    def go_to(self, name: str) -> None:
        """Show the page titled *name* (any case)."""
        titles = [page.title.lower() for page in self.pages]
        self.nav.setCurrentRow(titles.index(name.lower()))

    def _go(self, row: int) -> None:
        self.stack.setCurrentIndex(row)
        jobs = getattr(self.pages[row], "jobs", None)
        for page in self.pages:
            if getattr(page, "jobs", None) is not None:
                page.jobs.setVisible(page.jobs is jobs)
        self.activity.setVisible(jobs is not None)
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


def place(window: QtWidgets.QWidget, screen: QtGui.QScreen) -> None:
    """Size *window* to at most :data:`WINDOW` and :data:`SCREEN_SHARE` of *screen*'s
    free area, centred on it (Windows may otherwise put it where a monitor used to
    be, off every screen)."""
    free = screen.availableGeometry()
    width = min(WINDOW[0], int(free.width() * SCREEN_SHARE))
    height = min(WINDOW[1], int(free.height() * SCREEN_SHARE))
    window.resize(width, height)
    window.move(
        free.x() + (free.width() - width) // 2, free.y() + (free.height() - height) // 2
    )


def run_app(
    path: Path | None = None, page: str | None = None
) -> None:  # pragma: no cover
    if sys.platform == "win32":  # the taskbar shows the app's icon, not Python's
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("zephyr.gui")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("zephyr")
    app.setWindowIcon(QtGui.QIcon(str(ICON)))
    app.setStyleSheet(STYLE)
    project = None
    if path is not None:  # from the command line: a new folder is made a project
        project = Project.open(path, create=not Project.exists(path))
    window = MainWindow(project)
    screen = app.screenAt(QtGui.QCursor.pos()) or app.primaryScreen()
    place(window, screen)
    if page is not None:
        window.go_to(page)
    window.show()
    window.raise_()
    window.activateWindow()
    app.exec()

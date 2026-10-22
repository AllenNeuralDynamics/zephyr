"""The app's main window: the steps on the left, the current one on the right."""

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .pages import BoxesPage, BreathingPage, ClipsPage, ProjectPage
from .predicting import PredictPage
from .project import Project
from .theme import STYLE
from .training import ResultsPage, TrainPage
from .widgets import find_data

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
        self.nav.setFixedWidth(150)
        self.stack = QtWidgets.QStackedWidget()
        self.pages = [page(self) for page in PAGES]
        for number, page in enumerate(self.pages):
            self.nav.addItem(f"{number}  {page.title}" if number else page.title)
            self.stack.addWidget(page)
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

    def _list_chosen(self, index: int) -> None:
        path = self.lists.itemData(index)
        if self.project is not None and path != self.project.clip_list:
            self.open_project(path)

    def _go(self, row: int) -> None:
        self.stack.setCurrentIndex(row)
        self.pages[row].shown()

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

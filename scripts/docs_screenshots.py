"""Take the documentation's screenshots (docs/assets/screens/) from the real app.

Drives `zephyr gui` offscreen through the tutorial's project, on the tutorial's data
(docs/tutorial/index.md), and saves each page as the tutorial shows it. Needs the
`ui` extra and a GPU for the training stage (about 40 minutes):

    uv run python scripts/docs_screenshots.py <workdir>            # every stage
    uv run python scripts/docs_screenshots.py <workdir> --stage predict

<workdir> holds vr-foraging/ (the labelled sessions) and new-animal/ (a video to
predict), downloaded as the tutorial says; the project is made in
<workdir>/mouse-breathing. --show opens the window on screen.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

STAGES = ("setup", "train", "predict")
SIZE = (1600, 950)
"""The window's size, in logical pixels; shots are taken at 1.5x and saved at WIDTH."""
WIDTH = 1500
QUALITY = 85
"""WebP quality: small UI text stays sharp, a page is about 50 kB."""
GLOB = "*_FaceCamera.mp4"
NAMES = (
    r"(?P<group>.+)_span(?P<span>\d+)_FaceCamera$",
    "{group}_span{span}_thermistor.parquet",
)
"""The corpus's file names: the [derive] pattern and breathing-file template."""
SESSIONS = {
    "789903": "behavior_789903_2025-10-29_20-06-45",
    "789907": "behavior_789907_2025-10-23_17-07-36",
    "794591": "behavior_794591_2025-10-14_17-57-50",
    "798279": "behavior_798279_2025-10-20_16-00-19",
    "789917": "behavior_789917_2025-10-22_19-48-21",
    "806527": "behavior_806527_2025-10-24_18-04-57",
}
"""The tutorial's labelled sessions (group = session), one per animal."""
SPLIT = {
    "train": ["789903", "789907", "794591", "798279"],
    "val": ["789917"],
    "test": ["806527"],
}
"""The animals on each side: the corpus's train, validation and test A animals."""
BOX_CENTRES = {"789903": (278, 146), "789907": (298, 162), "794591": (222, 137)}
"""Where the boxes of three animals are placed by hand (360 x 270 frames), from the
corpus's published boxes; Auto places the rest."""
NEW_CENTRE = (264, 102)
"""Where the new animal's crop is placed by hand."""

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("workdir", type=Path)
parser.add_argument("--stage", choices=STAGES, help="Only this stage (default: all).")
parser.add_argument("--out", type=Path, default=Path("docs/assets/screens"))
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--show", action="store_true", help="Show the window on screen.")
args = parser.parse_args()

if not args.show:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if sys.platform == "win32":  # offscreen Qt finds no fonts on Windows by itself
        os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
os.environ.setdefault("QT_SCALE_FACTOR", "1.5")

from PySide6 import QtCore, QtGui, QtWidgets

from zephyr.gui.app import ICON, MainWindow
from zephyr.gui.project import Project
from zephyr.gui.theme import STYLE

ROOT = args.workdir / "mouse-breathing"
LABELLED = args.workdir / "vr-foraging"
NEW = args.workdir / "new-animal"
Checked, Unchecked = QtCore.Qt.CheckState.Checked, QtCore.Qt.CheckState.Unchecked

# Every question is answered yes; what the app says is printed.
Yes = QtWidgets.QMessageBox.StandardButton.Yes
QtWidgets.QMessageBox.question = staticmethod(
    lambda _parent, _title, text, *_a, **_k: print("asked:", text[:100]) or Yes
)
QtWidgets.QMessageBox.information = staticmethod(lambda _p, _t, text, *_a: print(text))
QtWidgets.QMessageBox.warning = staticmethod(lambda _p, _t, text, *_a: print(text))

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
app.setApplicationName("zephyr")
app.setWindowIcon(QtGui.QIcon(str(ICON)))
app.setStyleSheet(STYLE)
QtCore.QSettings.setPath(  # recent projects: never the user's
    QtCore.QSettings.Format.IniFormat,
    QtCore.QSettings.Scope.UserScope,
    str(args.workdir / "qt-settings"),
)
args.out.mkdir(parents=True, exist_ok=True)


def spin(seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def wait(page, every: float = 0, then=None) -> None:
    """Spin until *page*'s commands end (idle for 2 s: a page may chain several),
    calling *then* every *every* seconds."""
    last = idle = time.time()
    while time.time() - idle < 2.0:
        spin(0.5)
        if page.jobs.busy:
            idle = time.time()
        if then and time.time() - last > every:
            then()
            last = time.time()


def grab(name: str, widget=None) -> None:
    if widget is None:
        window.resize(*SIZE)
    spin(0.6)
    image = (widget or window).grab().toImage()
    image = image.scaledToWidth(
        WIDTH, QtCore.Qt.TransformationMode.SmoothTransformation
    )
    image.save(str(args.out / f"{name}.webp"), "WEBP", QUALITY)
    print("shot", name)


def go(title: str):
    window.go_to(title)
    spin(1.0)
    return window.pages[[p.title.lower() for p in window.pages].index(title)]


def key_of(box, session: str) -> str:
    """The Boxes page's key of *session*'s recording."""
    return next(r.key for r in box.state.recordings if r.clips[0].group == session)


def setup() -> None:
    """Project, Clips, Boxes and Breathing."""
    ROOT.mkdir(parents=True, exist_ok=True)
    window.open_project(ROOT)
    go("project")
    grab("project-new")

    clips = go("clips")
    clips.folder.setText(str(LABELLED))
    clips.glob.setText(GLOB)
    spin(2.5)  # the preview: the default names do not match
    clips.frame.setCurrentIndex(1)  # 1/2: 360 x 270
    grab("clips-preview")
    clips.naming.setChecked(True)
    clips.pattern.setText(NAMES[0])
    clips.thermistor.setText(NAMES[1])
    spin(2.5)
    grab("clips-other-names")
    clips.scan()
    wait(clips)
    grab("clips-list")

    boxes = go("boxes")
    boxes.cache_frames()
    wait(boxes)
    boxes.load()
    spin(1.0)
    box = boxes.window
    while box.state.box_size != 96:
        box.resize_box(8 if box.state.box_size < 96 else -8)
    for animal, centre in BOX_CENTRES.items():
        box.select_key(key_of(box, SESSIONS[animal]))
        box.place(*centre)
    grab("boxes-placed")
    box.auto()
    spin(1.0)
    box.select_key(key_of(box, SESSIONS["798279"]))
    grab("boxes-matched")

    clips = go("clips")
    clips.check()
    wait(clips)
    grab("clips-checked")

    editor = go("breathing").editor
    spin(3.0)  # the suggestions
    grab("breathing-selection")
    editor.tabs.setCurrentIndex(0)
    grab("breathing-editor")
    editor.auto()
    QtCore.QThreadPool.globalInstance().waitForDone()
    editor.tabs.setCurrentIndex(1)
    every = list(range(len(editor.state.entries)))
    editor._mark(every, "include")
    editor._review(every, True)
    grab("breathing-reviewed")
    editor.save()
    go("project")
    grab("project-progress")


def train() -> None:
    """Train: preprocess, an experiment, a smoke run and a run."""
    page = go("train")
    page.start("preprocess")
    wait(page)
    page.jobs.show_log(False)
    grab("train-preprocessed")
    form = page.form
    form.auto_split()
    for side, widget in (("train", form.train), ("val", form.val), ("test", form.test)):
        for i in range(widget.count()):
            item = widget.item(i)
            chosen = {SESSIONS[a] for a in SPLIT[side]}
            item.setCheckState(Checked if item.text() in chosen else Unchecked)
    form.name.setText("first-network")
    form.epochs.setValue(args.epochs)
    grab("train-new-experiment")
    form.write()
    page.start("smoke")
    wait(page)
    grab("train-smoke")
    page.start("run")

    def snippets() -> None:  # the latest scoring's, kept until the run ends
        page.tabs.setCurrentWidget(page.gallery)
        grab("train-snippets")
        page.tabs.setCurrentWidget(page.history)

    wait(page, every=200, then=snippets)
    grab("train-progress")

    results = go("results")
    spin(2.0)
    grab("results-groups")
    results.tabs.setCurrentIndex(1)
    results.clips.selectRow(0)
    spin(2.0)
    grab("results-clips")
    results.tabs.setCurrentIndex(2)
    grab("results-training")


def predict() -> None:
    """Predict: videos from a folder, their crops, and quality control."""
    # Predicting into a folder that has predictions asks which to redo: start afresh.
    shutil.rmtree(ROOT / "predictions", ignore_errors=True)
    (ROOT / "predictions").mkdir()
    page = go("predict")
    page.source.setCurrentIndex(1)
    page.video_folder.setText(str(NEW))
    page.video_glob.setText(GLOB)
    page.video_name.setText("new-animals")
    page.video_cache.setChecked(True)
    grab("predict-videos")
    page.make_video_list()
    wait(page)
    box = page.box_window
    box.place(*NEW_CENTRE)
    grab("predict-crops", box)
    page.predict_videos()
    wait(page)
    grab("predict-trace")

    page.source.setCurrentIndex(0)
    page.groups.setText(SESSIONS[SPLIT["test"][0]])
    page.name.setText("held-out")
    page.qc.setChecked(True)
    page.predict()
    wait(page)
    grab("predict-qc")


window = MainWindow(Project.open(ROOT) if Project.exists(ROOT) else None)
window.resize(*SIZE)
window.show()
spin(1.5)
for stage in [args.stage] if args.stage else STAGES:
    print(f"== {stage}")
    globals()[stage]()
for page in window.pages:  # leave without asking about unsaved changes
    if getattr(page, "editor", None) is not None:
        page.editor._skip_prompt = True
window.close()

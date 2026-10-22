"""The Boxes page (``zephyr gui <list> --page boxes``): one crop box per recording, on the downsampled
frame.

Recordings on the left, the frame with its box on the right; every change is saved to
the clip list at once (only ``box`` and ``target_size``). The logic is
:class:`zephyr.annotate.BoxState`. Keys: click place | arrows nudge (shift x10) |
[ ] resize | n/p recording | 1/2/3 clip | o own box | c copy previous | r re-centre |
a auto (the hand-placed boxes matched on the unplaced recordings).
"""

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..annotate import (
    BoxState,
    box_centre,
    downsample,
    load_recording_frames,
    save_boxes,
)
from .theme import THEME

STEP = 8
"""Pixels a box grows or shrinks by ([ ] and the -/+ buttons)."""
DIVISORS = (("native", 1), ("1/2", 2), ("1/3", 3), ("1/4", 4))
HELP = (
    "click=place   arrows=nudge (shift x10)   [ ]=box size   n/p=recording   "
    "1/2/3=clip   o=own box   c=copy prev   r=re-centre   a=auto"
)


class _FrameBox(pg.ViewBox):
    """A view whose left click places the box (dragging still pans)."""

    def __init__(self, window: "BoxWindow") -> None:
        super().__init__(lockAspect=True, invertY=True, enableMenu=False)
        self.window = window

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            point = self.mapSceneToView(ev.scenePos())
            self.window.place(point.x(), point.y())
            ev.accept()
        else:
            super().mouseClickEvent(ev)


class _Mark(pg.ScatterPlotItem):
    """The box's centre cross; a click on it places the box, like anywhere else."""

    def mouseClickEvent(self, ev) -> None:
        ev.ignore()


class _Canvas(pg.GraphicsView):
    """The frame's view; its keys are the box's (a graphics view would scroll)."""

    def __init__(self, window: "BoxWindow") -> None:
        super().__init__()
        self.window = window

    def keyPressEvent(self, event) -> None:
        if not self.window.handle_key(event):
            super().keyPressEvent(event)


class BoxWindow(QtWidgets.QMainWindow):
    """Place the boxes of *state*, saving each change to the clip list *out_path*.
    With *frame_fixed* (the app, where Clips chose it) the frame size is shown, not
    offered."""

    saved = QtCore.Signal()
    """The clip list was written (after every change)."""

    def __init__(
        self,
        state: BoxState,
        frame_dir: Path,
        out_path: Path,
        *,
        frame_fixed: bool = False,
    ) -> None:
        super().__init__()
        self.state = state
        self.frame_dir = frame_dir
        self.out_path = out_path
        self.frame_fixed = frame_fixed
        self.setWindowTitle("Crop ROI per recording")
        self._frames: dict[str, list[np.ndarray]] = {}
        self._items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        self._build()
        for recording in state.recordings:
            self._refresh_row(recording.key)
        self.select_key(state.key)

    # ---- construction ----------------------------------------------------------

    def _build(self) -> None:
        split = QtWidgets.QSplitter()
        self.setCentralWidget(split)

        left = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(left)
        column.addWidget(QtWidgets.QLabel("Recordings", objectName="heading"))
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Recording", "Clips", "Box"])
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for section in (1, 2):
            header.setSectionResizeMode(
                section, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
        self.tree.setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle)
        # Keys belong to the box, as in the frame: the tree only takes clicks.
        self.tree.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        folders: dict[Path, QtWidgets.QTreeWidgetItem] = {}
        one_folder = len({r.clips[0].video.parent for r in self.state.recordings}) == 1
        self.tree.setRootIsDecorated(not one_folder)
        for recording in self.state.recordings:
            folder = recording.clips[0].video.parent
            if one_folder:  # no folder level to tell them apart
                parent = self.tree
            else:
                if folder not in folders:
                    folders[folder] = QtWidgets.QTreeWidgetItem(
                        self.tree, [folder.name]
                    )
                    folders[folder].setExpanded(True)
                parent = folders[folder]
            item = QtWidgets.QTreeWidgetItem(
                parent, [recording.label, str(len(recording.clips)), ""]
            )
            item.setToolTip(0, recording.label)
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, recording.key)
            self._items[recording.key] = item
        self.tree.currentItemChanged.connect(self._tree_picked)
        column.addWidget(self.tree)
        split.addWidget(left)

        right = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(right)
        controls = QtWidgets.QHBoxLayout()
        downsample = QtWidgets.QLabel("Downsample to:")
        controls.addWidget(downsample)
        self.width_edit = QtWidgets.QLineEdit(maximumWidth=60)
        self.height_edit = QtWidgets.QLineEdit(maximumWidth=60)
        for edit in (self.width_edit, self.height_edit):
            edit.setValidator(QtGui.QIntValidator(1, 100000))
            edit.returnPressed.connect(self.apply_target)
        controls.addWidget(self.width_edit)
        by = QtWidgets.QLabel("x")
        controls.addWidget(by)
        controls.addWidget(self.height_edit)
        apply = QtWidgets.QPushButton("Apply")
        apply.clicked.connect(self.apply_target)
        controls.addWidget(apply)
        sizing = [self.width_edit, self.height_edit, apply]
        for label, divisor in DIVISORS:
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda _=False, d=divisor: self.set_divisor(d))
            controls.addWidget(button)
            sizing.append(button)
        if self.frame_fixed:
            for widget in [downsample, by, *sizing]:
                widget.hide()
        else:
            controls.addSpacing(16)
        controls.addWidget(QtWidgets.QLabel("Box:"))
        smaller = QtWidgets.QPushButton("-")
        smaller.clicked.connect(lambda: self.resize_box(-STEP))
        controls.addWidget(smaller)
        self.size_label = QtWidgets.QLabel(
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.size_label.setMinimumWidth(40)
        controls.addWidget(self.size_label)
        larger = QtWidgets.QPushButton("+")
        larger.clicked.connect(lambda: self.resize_box(STEP))
        controls.addWidget(larger)
        sizing += [smaller, larger]
        if self.state.locked:  # the network's crop and frame: only the place moves
            for widget in sizing:
                widget.setEnabled(False)
            self.size_label.setToolTip("The network's crop size: it cannot change")
        controls.addSpacing(16)
        auto = QtWidgets.QPushButton("Auto")
        auto.setToolTip(
            "Box every unplaced recording where the boxes placed by hand match its "
            "frame (A); the rest stay unplaced"
        )
        auto.clicked.connect(self.auto)
        controls.addWidget(auto)
        controls.addSpacing(16)
        self.part_label = QtWidgets.QLabel()
        controls.addWidget(self.part_label)
        controls.addStretch(1)
        self.progress_label = QtWidgets.QLabel()
        controls.addWidget(self.progress_label)
        for button in right.findChildren(QtWidgets.QPushButton):
            button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        column.addLayout(controls)

        self.title = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self.title)
        self.view = _FrameBox(self)
        self.canvas = _Canvas(self)
        self.canvas.setCentralItem(self.view)
        self.canvas.setBackground(THEME["bg"])
        self.canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.view.addItem(self.image)
        self.rect = QtWidgets.QGraphicsRectItem()
        self.view.addItem(self.rect)
        self.crosshair = _Mark(symbol="+", size=16, pen=pg.mkPen("c", width=2))
        self.view.addItem(self.crosshair)
        column.addWidget(self.canvas, 1)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 1050])

        self.info = QtWidgets.QLabel()
        self.statusBar().addWidget(self.info, 1)
        self.statusBar().addPermanentWidget(QtWidgets.QLabel(HELP, objectName="sub"))

    # ---- drawing ----------------------------------------------------------------

    def _frames_for(self, key: str) -> list[np.ndarray]:
        if key not in self._frames:
            recording = next(r for r in self.state.recordings if r.key == key)
            self._frames[key] = load_recording_frames(recording, self.frame_dir)
        return self._frames[key]

    def _refresh_row(self, key: str) -> None:
        placed = self.state.is_placed(key)
        item = self._items[key]
        score = self.state.matched.get(key)
        item.setText(
            2,
            f"match {score:.2f}" if score is not None else "set" if placed else "",
        )
        colour = QtGui.QColor(THEME["placed" if placed else "auto"])
        for column in range(3):
            item.setForeground(column, colour)

    def redraw(self) -> None:
        state = self.state
        frames = self._frames_for(state.key)
        if frames:
            state.set_part(state.part_index)
            data = downsample(frames[state.part_index], state.target_size)
            suffix = "  (own box)" if state.has_override else ""
            self.part_label.setText(f"{state.clip.video.stem}{suffix}")
        else:
            data = np.zeros(state.target_size[::-1], dtype=np.uint8)
            self.part_label.setText("no cached frame")
        if data.ndim == 3:
            data = data.mean(axis=2)
        self.image.setImage(data, levels=(0, 255))
        width, height = state.target_size
        self.view.setLimits(xMin=-width, xMax=2 * width, yMin=-height, yMax=2 * height)
        if not self._fitted:
            self.view.setRange(xRange=(0, width), yRange=(0, height), padding=0.02)
            self._fitted = True

        box = state.box
        self.rect.setRect(*box)
        pen = pg.mkPen(
            "c" if state.is_placed(state.key) else "y",
            width=2,
            style=QtCore.Qt.PenStyle.DashLine
            if state.has_override
            else QtCore.Qt.PenStyle.SolidLine,
        )
        pen.setCosmetic(True)
        self.rect.setPen(pen)
        self.crosshair.setData([box_centre(box)[0]], [box_centre(box)[1]])

        scale = state.scale_from_native
        fixed = " (the list's, set when it was made)" if self.frame_fixed else ""
        self.title.setText(
            f"<b>{state.recording.label}</b><br>frame {width}x{height}{fixed} "
            f"({scale:.2f}x downsample)   box {state.box_size}px = "
            f"{state.box_size * scale:.0f} native px   at {box[:2]}"
        )
        done, total = state.progress()
        self.progress_label.setText(f"{done} / {total} placed")
        self.info.setText(
            f"clips: {', '.join(c.video.stem for c in state.recording.clips)}"
        )
        self.width_edit.setText(str(width))
        self.height_edit.setText(str(height))
        self.size_label.setText(str(state.box_size))

    _fitted = False

    # ---- actions ----------------------------------------------------------------

    def _save(self) -> None:
        save_boxes(self.out_path, self.state)
        self.saved.emit()

    def commit(self) -> None:
        """Save after a change to the boxes, as every change is."""
        self._refresh_row(self.state.key)
        self._save()
        self.redraw()

    def select_key(self, key: str) -> None:
        self.state.select(key)
        item = self._items[key]
        self.tree.blockSignals(True)
        self.tree.setCurrentItem(item)
        self.tree.blockSignals(False)
        self.tree.scrollToItem(item)
        self.redraw()

    def _tree_picked(self, item, _previous) -> None:
        key = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None
        if key is not None:
            self.state.select(key)
            self.redraw()
        self.canvas.setFocus()

    def place(self, x: float, y: float) -> None:
        self.state.place(x, y)
        self.commit()

    def resize_box(self, delta: int) -> None:
        self.state.resize(delta)
        self.commit()

    def auto(self) -> None:
        """Match the hand-placed boxes on every unplaced recording, and save."""
        try:
            scores = self.state.auto(self._frames_for_recording)
        except ValueError as exc:
            self.statusBar().showMessage(str(exc), 8000)
            return
        for key in scores:
            self._refresh_row(key)
        self._save()
        self.redraw()
        boxed = sum(key in self.state.matched for key in scores)
        self.statusBar().showMessage(
            f"Auto boxed {boxed} of {len(scores)} unplaced recording(s); the rest "
            "need a box by hand (then Auto again)",
            8000,
        )

    def _frames_for_recording(self, recording) -> list[np.ndarray]:
        return self._frames_for(recording.key)

    def apply_target(self) -> None:
        try:
            width, height = int(self.width_edit.text()), int(self.height_edit.text())
        except ValueError:
            self.redraw()
            return
        self.state.set_target(width, height)
        self._fitted = False
        self._save()
        self.redraw()
        self.canvas.setFocus()

    def set_divisor(self, divisor: int) -> None:
        native = self.state.native_size
        self.state.set_target(native[0] // divisor, native[1] // divisor)
        self._fitted = False
        self._save()
        self.redraw()

    def keyPressEvent(self, event) -> None:
        # Keys a size field leaves (arrows) bubble up here: they are not box keys.
        typing = isinstance(QtWidgets.QApplication.focusWidget(), QtWidgets.QLineEdit)
        if typing or not self.handle_key(event):
            super().keyPressEvent(event)

    def handle_key(self, event) -> bool:
        """Apply a box key; False for any other key."""
        key = QtCore.Qt.Key
        k = event.key()
        step = 10 if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier else 1
        moves = {
            key.Key_Left: (-step, 0),
            key.Key_Right: (step, 0),
            key.Key_Up: (0, -step),
            key.Key_Down: (0, step),
        }
        state = self.state
        if k in moves:
            state.nudge(*moves[k])
            self.commit()
        elif k in (key.Key_N, key.Key_P):
            state.advance(1 if k == key.Key_N else -1)
            self.select_key(state.key)
        elif k == key.Key_C:
            state.copy_previous()
            self.commit()
        elif k == key.Key_R:
            state.reset()
            self.commit()
        elif k == key.Key_A:
            self.auto()
        elif k == key.Key_O:
            state.toggle_override()
            self.commit()
        elif k in (key.Key_1, key.Key_2, key.Key_3):
            state.set_part(k - key.Key_1)
            self.redraw()
        elif k == key.Key_BracketRight:
            self.resize_box(STEP)
        elif k == key.Key_BracketLeft:
            self.resize_box(-STEP)
        else:
            return False
        event.accept()
        return True

    def closeEvent(self, event) -> None:
        self._save()
        print(f"saved {len(self.state.placed)} recording box(es) -> {self.out_path}")
        event.accept()

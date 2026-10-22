"""The app's windows, offscreen (needs the ``ui`` extra): every key and control of the
box window reaches the saved clip list as the Tk tool's did, and the pages open."""

import os
import tempfile
import tomllib
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from _clipfiles import touch_clip

HAS_UI = bool(find_spec("PySide6") and find_spec("pyqtgraph"))

if HAS_UI:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtWidgets
    from PySide6.QtTest import QTest

    from zephyr.annotate import frame_path, open_state
    from zephyr.gui.app import MainWindow
    from zephyr.gui.boxes import BoxWindow
    from zephyr.gui.project import Project
    from zephyr.gui.widgets import HistoryPlot, SnippetGallery

    APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

NATIVE = (100, 80)
Key = QtCore.Qt.Key if HAS_UI else None


def _boxes(path: Path) -> dict[str, list[int] | None]:
    doc = tomllib.loads(path.read_text())
    return {Path(c["video"]).stem: c.get("box") for c in doc["clip"]}


def _layout(root: Path, *, sensors: bool = False) -> Path:
    """Recording 1 (two parts), 2 and 3, a cached frame for every clip, no boxes."""
    videos = [
        touch_clip(root / "data", 1, 1),
        touch_clip(root / "data", 1, 2),
        touch_clip(root / "data", 2, 1),
        touch_clip(root / "data", 3, 1),
    ]
    lines = [f"[preprocess]\ntarget_size = {list(NATIVE)}\n"]
    lines += [f'[[clip]]\nvideo = "data/{v.name}"\n' for v in videos]
    path = root / "clips.toml"
    path.write_text("\n".join(lines))
    frames = root / "frames"
    frames.mkdir()
    for k, video in enumerate(videos):
        frame = np.full(NATIVE[::-1], 40 * k, np.uint8)
        np.save(frame_path(frames, video.resolve()), frame)
    if sensors:
        t = np.arange(0, 30, 1 / 500)
        for video in videos:
            name = video.stem.replace("video_face_", "thermistor_")
            pd.DataFrame({"Time": t, "Signal": np.sin(2 * np.pi * 3 * t)}).to_parquet(
                root / "data" / f"{name}.parquet"
            )
    return path


@unittest.skipUnless(HAS_UI, "needs the ui extra")
class BoxWindowTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.list = _layout(self.root)
        self.frames = self.root / "frames"
        self.open()

    def open(self, **kwargs):
        state = open_state(self.list, self.frames, remote=True, box_size=32, **kwargs)
        self.window = BoxWindow(state, self.frames, self.list)
        self.window.resize(1000, 700)
        self.window.show()
        self.window.canvas.setFocus()
        APP.processEvents()
        self.state = state

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        APP.processEvents()
        self._tmp.cleanup()

    def key(self, key, modifier=None):
        QTest.keyClick(
            self.window.canvas, key, modifier or QtCore.Qt.KeyboardModifier.NoModifier
        )

    def click(self, x: float, y: float) -> None:
        scene = self.window.view.mapViewToScene(QtCore.QPointF(x, y))
        point = self.window.canvas.mapFromScene(scene)
        viewport = self.window.canvas.viewport()
        QTest.mouseClick(
            viewport, QtCore.Qt.MouseButton.LeftButton, pos=point, delay=300
        )

    def test_a_click_places_the_recordings_box_and_saves_it(self):
        self.click(50, 40)
        boxes = _boxes(self.list)
        self.assertEqual(boxes["video_face_1_part_1"], [34, 24, 32, 32])
        self.assertEqual(boxes["video_face_1_part_2"], [34, 24, 32, 32])
        self.assertIsNone(boxes["video_face_2_part_1"])
        self.assertEqual(self.window.progress_label.text(), "1 / 3 placed")
        item = self.window._items[self.state.key]
        self.assertEqual(item.text(2), "set")

    def test_arrows_nudge_and_shift_moves_ten(self):
        self.click(50, 40)
        self.key(Key.Key_Right)
        self.key(Key.Key_Down, QtCore.Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(_boxes(self.list)["video_face_1_part_1"], [35, 34, 32, 32])
        self.key(Key.Key_Left, QtCore.Qt.KeyboardModifier.ShiftModifier)
        self.key(Key.Key_Up)
        self.assertEqual(_boxes(self.list)["video_face_1_part_1"], [25, 33, 32, 32])

    def test_brackets_and_buttons_resize_every_box(self):
        self.click(50, 40)
        self.key(Key.Key_N)
        self.click(30, 30)
        self.key(Key.Key_BracketRight)
        self.assertEqual({b[2] for b in _boxes(self.list).values() if b}, {40})
        self.key(Key.Key_BracketLeft)
        self.key(Key.Key_BracketLeft)
        boxes = _boxes(self.list)
        self.assertEqual(boxes["video_face_2_part_1"], [18, 18, 24, 24])
        self.assertEqual(self.window.size_label.text(), "24")

    def test_n_and_p_move_between_recordings(self):
        self.key(Key.Key_N)
        self.key(Key.Key_N)
        self.assertEqual(self.state.recording.label, "data/3")
        self.key(Key.Key_N)
        self.assertEqual(self.state.recording.label, "data/3")
        self.key(Key.Key_P)
        self.assertEqual(self.state.recording.label, "data/2")
        self.assertIs(
            self.window.tree.currentItem(), self.window._items[self.state.key]
        )

    def test_c_copies_the_previous_recordings_box(self):
        self.click(20, 20)
        self.key(Key.Key_N)
        self.key(Key.Key_C)
        boxes = _boxes(self.list)
        self.assertEqual(boxes["video_face_2_part_1"], boxes["video_face_1_part_1"])

    def test_o_gives_one_clip_its_own_box(self):
        self.click(50, 40)
        self.key(Key.Key_2)
        self.assertEqual(self.state.clip.video.stem, "video_face_1_part_2")
        self.key(Key.Key_O)
        self.assertTrue(self.window.part_label.text().endswith("(own box)"))
        self.key(Key.Key_Right)
        boxes = _boxes(self.list)
        self.assertEqual(boxes["video_face_1_part_1"], [34, 24, 32, 32])
        self.assertEqual(boxes["video_face_1_part_2"], [35, 24, 32, 32])
        self.key(Key.Key_1)
        self.assertEqual(self.state.clip.video.stem, "video_face_1_part_1")

    def test_r_recentres_and_unplaces(self):
        self.click(20, 20)
        self.key(Key.Key_R)
        self.assertIsNone(_boxes(self.list)["video_face_1_part_1"])
        self.assertEqual(self.state.box, (34, 24, 32, 32))
        self.assertEqual(self.window.progress_label.text(), "0 / 3 placed")

    def test_the_target_size_carries_the_boxes_across(self):
        self.click(50, 40)
        self.window.width_edit.setText("50")
        self.window.height_edit.setText("40")
        self.window.apply_target()
        doc = tomllib.loads(self.list.read_text())
        self.assertEqual(doc["preprocess"]["target_size"], [50, 40])
        self.assertEqual(_boxes(self.list)["video_face_1_part_1"], [9, 4, 32, 32])
        self.window.set_divisor(1)
        doc = tomllib.loads(self.list.read_text())
        self.assertEqual(doc["preprocess"]["target_size"], list(NATIVE))

    def test_typing_a_size_is_not_a_box_key(self):
        self.window.width_edit.setFocus()
        QTest.keyClicks(self.window.width_edit, "2")
        QTest.keyClick(self.window.width_edit, Key.Key_N)
        self.assertEqual(self.state.index, 0)
        self.assertEqual(self.state.part_index, 0)

    def test_the_tree_picks_a_recording(self):
        self.window.tree.setCurrentItem(
            self.window._items[self.state.recordings[2].key]
        )
        self.assertEqual(self.state.recording.label, "data/3")

    def test_placed_boxes_resume(self):
        self.click(50, 40)
        self.window.close()
        self.open()
        self.assertEqual(self.state.progress(), (1, 3))
        self.assertEqual(self.state.box, (34, 24, 32, 32))


@unittest.skipUnless(HAS_UI, "needs the ui extra")
class AppTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.list = _layout(self.root, sensors=True)
        self.said: list[str] = []
        for name in ("warning", "information", "question"):
            patcher = mock.patch.object(
                QtWidgets.QMessageBox,
                name,
                side_effect=lambda _p, _title, text, *a, **k: self.said.append(text),
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        self.window = MainWindow(Project.open(self.root))
        self.window.show()

    def tearDown(self):
        for page in self.window.pages:
            editor = getattr(page, "editor", None)
            if editor is not None:
                editor._skip_prompt = True
        self.window.close()
        self.window.deleteLater()
        APP.processEvents()
        self._tmp.cleanup()

    def go(self, title: str):
        titles = [p.title for p in self.window.pages]
        self.window.nav.setCurrentRow(titles.index(title))
        APP.processEvents()
        return self.window.pages[titles.index(title)]

    def test_every_page_opens_on_a_project(self):
        for page in self.window.pages:
            self.go(page.title)
        self.assertEqual(self.go("Clips").table.rowCount(), 4)
        self.assertIsNotNone(self.go("Boxes").window)
        self.assertIsNotNone(self.go("Breathing").editor)
        train = self.go("Train")
        self.assertEqual(train.form.train.count(), 3)

    def test_boxes_are_placed_on_cached_frames_of_videos_kept_elsewhere(self):
        for video in (self.root / "data").glob("*.mp4"):
            video.unlink()
        boxes = self.go("Boxes")
        boxes.box_size.setValue(32)
        boxes.load()
        self.assertEqual(boxes.window.state.native_size, NATIVE)
        boxes.window.place(50, 40)
        self.assertEqual(_boxes(self.list)["video_face_1_part_1"], [34, 24, 32, 32])

    def test_a_hidden_pages_shortcuts_stay_quiet(self):
        editor = self.go("Breathing").editor
        boxes = self.go("Boxes").window
        QTest.keyClick(boxes.canvas, Key.Key_N)
        self.assertEqual((boxes.state.index, editor.state.index), (1, 0))
        self.go("Breathing")
        editor.setFocus()
        QTest.keyClick(editor, Key.Key_N)
        self.assertEqual((boxes.state.index, editor.state.index), (1, 1))

    def test_the_train_form_writes_an_experiment(self):
        train = self.go("Train")
        train.form.name.setText("first")
        for widget, row in ((train.form.train, 0), (train.form.test, 2)):
            widget.item(row).setCheckState(QtCore.Qt.CheckState.Checked)
        train.form.write()
        self.assertRegex(" ".join(self.said), "no box")  # a fold needs every box
        self.assertFalse((self.root / "experiments" / "first.toml").exists())
        self.list.write_text(
            self.list.read_text().replace('.mp4"\n', '.mp4"\nbox = [10, 10, 32, 32]\n')
        )
        train.form.write()
        self.assertEqual(train.experiment.currentText(), "first.toml", self.said)
        self.assertTrue(train.buttons["run"].isEnabled())


@unittest.skipUnless(HAS_UI, "needs the ui extra")
class WidgetTests(unittest.TestCase):
    def test_history_marks_each_best(self):
        plot = HistoryPlot()
        rows = [
            {
                "epoch": 0,
                "train_loss": 1.0,
                "val_loss": 1.1,
                "xcorr": 0.2,
                "since_best": 0,
            },
            {"epoch": 1, "train_loss": 0.9, "val_loss": 1.0},
            {
                "epoch": 2,
                "train_loss": 0.8,
                "val_loss": 0.9,
                "xcorr": 0.4,
                "since_best": 0,
            },
            {
                "epoch": 3,
                "train_loss": 0.7,
                "val_loss": 0.9,
                "xcorr": 0.3,
                "since_best": 1,
            },
        ]
        plot.set_history(rows)
        x, _ = plot.curves["xcorr"].getData()
        self.assertEqual(list(x), [1, 3, 4])
        self.assertEqual(list(plot.best.getData()[0]), [1, 3])

    def test_the_gallery_keeps_every_scoring_it_received(self):
        gallery = SnippetGallery()
        t = np.arange(0, 5, 1 / 60)
        clip = {
            "clip_id": "a/b",
            "entry": {"frames": "", "n_frames": "0", "n_output": "0"},
            "time": t.tolist(),
            "truth": np.sin(t).tolist(),
            "signal": np.cos(t).tolist(),
            "onset": np.zeros_like(t).tolist(),
            "truth_inhale": [1.0],
            "inhale": [1.1],
            "scores": {"correlation": 0.5},
        }
        clip["entry"] |= {k: None for k in ("clip_id", "recording", "stem")}
        clip["entry"] |= {
            k: None
            for k in ("frame_times", "times", "target", "events", "thermistor", "video")
        }
        clip["entry"]["clip_id"] = "a/b"
        for epoch in range(3):
            gallery.add({"epoch": epoch, "clips": [clip]})
        self.assertEqual(gallery.slider.maximum(), 2)
        self.assertEqual(gallery.slider.value(), 2)
        self.assertIn("epoch 3", gallery.label.text())
        gallery.slider.setValue(0)
        self.assertIn("epoch 1", gallery.label.text())
        gallery.clear()
        self.assertEqual(gallery.received, [])


if __name__ == "__main__":
    unittest.main()

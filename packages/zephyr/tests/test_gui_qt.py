"""The app's windows, offscreen (needs the ``ui`` extra): every key and control of the
box window reaches the saved clip list as the Tk tool's did, and the pages open."""

import json
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
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtTest import QTest

    from zephyr.annotate import frame_path, open_state
    from zephyr.gui.app import ICON, MainWindow, place
    from zephyr.gui.boxes import BoxWindow
    from zephyr.gui.jobs import Job
    from zephyr.gui.project import Project
    from zephyr.gui.training import _checked
    from zephyr.gui.widgets import HistoryPlot, JobPanel, SnippetGallery, TraceView

    APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _SETTINGS = tempfile.TemporaryDirectory()  # recent projects: never the user's
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope,
        _SETTINGS.name,
    )

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

    def test_a_matches_the_placed_box_on_the_other_recordings(self):
        y, x = np.mgrid[0 : NATIVE[1], 0 : NATIVE[0]]
        for stem, (cx, cy) in (
            ("video_face_1_part_1", (50, 40)),
            ("video_face_1_part_2", (50, 40)),
            ("video_face_2_part_1", (30, 30)),
            ("video_face_3_part_1", (70, 50)),
        ):
            blob = 200 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 6.0**2))
            video = (self.root / "data" / f"{stem}.mp4").resolve()
            np.save(frame_path(self.frames, video), (20 + blob).astype(np.uint8))
        self.window.close()
        self.open()
        self.key(Key.Key_A)  # nothing placed by hand: nothing to look for
        self.assertEqual(
            set(map(tuple, filter(None, _boxes(self.list).values()))), set()
        )
        self.click(50, 40)
        self.key(Key.Key_A)
        boxes = _boxes(self.list)
        self.assertEqual(boxes["video_face_2_part_1"], [14, 14, 32, 32])
        self.assertEqual(boxes["video_face_3_part_1"], [54, 34, 32, 32])
        self.assertEqual(self.window.progress_label.text(), "3 / 3 placed")
        second = self.state.recordings[1].key
        self.assertRegex(self.window._items[second].text(2), r"^match 0\.9\d|^match 1")

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
        QtCore.QThreadPool.globalInstance().waitForDone()  # Breathing's suggestions
        APP.processEvents()
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
        boxes.load()
        boxes.window.resize_box(32 - boxes.window.state.box_size)
        self.assertEqual(boxes.window.state.native_size, NATIVE)
        boxes.window.place(50, 40)
        self.assertEqual(_boxes(self.list)["video_face_1_part_1"], [34, 24, 32, 32])

    def test_a_hidden_pages_shortcuts_stay_quiet(self):
        editor = self.go("Breathing").editor
        QtCore.QThreadPool.globalInstance().waitForDone()  # the suggestions, on open
        APP.processEvents()
        self.assertIsNotNone(editor._suggestions)
        boxes = self.go("Boxes").window
        QTest.keyClick(boxes.canvas, Key.Key_N)
        self.assertEqual((boxes.state.index, editor.state.index), (1, 0))
        self.go("Breathing")
        self.assertEqual(editor.tabs.tabText(editor.tabs.currentIndex()), "Selection")
        editor.tabs.setCurrentIndex(0)  # the keys are the Editor's
        editor.setFocus()
        QTest.keyClick(editor, Key.Key_N)
        self.assertEqual((boxes.state.index, editor.state.index), (1, 1))

    def test_a_running_command_greys_its_page_and_says_what_runs(self):
        clips = self.go("Clips")
        job = mock.MagicMock(command="clips", argv=["scan"], running=True)
        clips.jobs.job = job
        clips.jobs.started.emit("zephyr clips scan")
        make = next(
            b
            for b in clips.findChildren(QtWidgets.QPushButton)
            if b.text() == "Make the clip list"
        )
        self.assertFalse(make.isEnabled())
        self.assertIn("Clips: running zephyr clips scan", self.window.running.text())
        job.running = False
        clips.jobs.finished.emit(0)
        self.assertTrue(make.isEnabled())
        self.assertEqual(self.window.running.text(), "")

    def test_the_videos_tab_lists_crops_and_predicts_videos_from_anywhere(self):
        predict = self.go("Predict")
        self.assertEqual(predict.source.tabText(1), "Videos")
        checkpoint = self.root / "best.pt"
        predict.checkpoint.addItem("best.pt", checkpoint)
        predict.checkpoint.setCurrentIndex(predict.checkpoint.count() - 1)
        predict.video_folder.setText("D:/elsewhere")
        with mock.patch.object(predict.jobs, "run") as started:
            predict.make_video_list()
            argv = started.call_args.args[0].argv
            listed = self.window.project.predictions_dir / "videos" / "videos.toml"
            self.assertEqual(argv[:2], ["scan", "D:/elsewhere"])
            self.assertEqual(argv[argv.index("-o") + 1], str(listed))
            self.assertEqual(argv[argv.index("--checkpoint") + 1], str(checkpoint))
            (then,) = predict._queue  # next: one frame per video, for the crops
            self.assertEqual(then.command, "annotate")
            self.assertIn("--prepare", then.argv)
            predict.predict_videos()  # no list yet: nothing runs
            self.assertRegex(self.said[-1], "Make the list")
            listed.parent.mkdir(parents=True)
            listed.write_text("")
            predict.predict_videos()
            argv = started.call_args.args[0].argv
            self.assertEqual(argv[argv.index("--out") + 1], str(listed.parent))

    def test_the_train_form_writes_an_experiment(self):
        train = self.go("Train")
        self.assertTrue(train.form.isVisibleTo(train))  # no experiment: define one
        self.assertTrue(train.buttons["preprocess"].isEnabled())  # needs only the list
        self.assertFalse(train.buttons["run"].isEnabled())
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
        self.assertIn("Wrote experiments/first.toml", train.summary.text())
        self.assertFalse(train.form.isVisibleTo(train))  # folded away once written
        self.assertTrue(self.window.nav.item(4).text().startswith("◐"))  # not run yet

    def test_the_steps_are_marked_in_the_list(self):
        nav = [self.window.nav.item(i).text() for i in range(self.window.nav.count())]
        self.assertEqual(nav[0], "✓  Project")
        self.assertEqual(nav[1], "✓  1  Clips")
        self.assertEqual(nav[2], "○  2  Boxes")  # no box yet
        self.assertIn("0 of 4 clip(s) boxed", self.window.nav.item(2).toolTip())

    def _trained(self) -> Path:
        """An experiment over the list (boxed) with a stopped run: its run folder."""
        self.list.write_text(
            self.list.read_text().replace('.mp4"\n', '.mp4"\nbox = [10, 10, 32, 32]\n')
        )
        project = self.window.project
        experiment = project.new_experiment("e", train=["1"], test={"test": ["2"]})
        (run,) = project.run_dirs(experiment)
        run.mkdir(parents=True)
        (run / "best.pt").touch()
        (run / "last.pt").touch()
        (run / "history.json").write_text('[{"epoch": 0, "since_best": 0}]')
        return run

    def _yes(self):
        return mock.patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.StandardButton.Yes,
        )

    def test_run_asks_before_starting_an_existing_run_over(self):
        run = self._trained()
        train = self.go("Train")
        with mock.patch.object(train.jobs, "run") as started:
            train.start("run")  # declined (the patched question answers nothing)
            self.assertRegex(self.said[-1], "already has a run")
            self.assertFalse(started.called)
            with self._yes():
                train.start("run")
            self.assertIn("--restart", started.call_args.args[0].argv)
            train.start("smoke")  # a smoke run always starts over, unasked
            self.assertIn("--restart", started.call_args.args[0].argv)
        self.assertTrue(run.is_dir())  # deleting is the command's, not the app's

    def test_resume_asks_before_scoring_finished_runs_again(self):
        run = self._trained()
        (run / "evaluation.json").write_text('{"groups": {}, "clips": []}')
        train = self.go("Train")
        with mock.patch.object(train.jobs, "run") as started:
            train.start("resume")  # declined (the patched question answers nothing)
            self.assertRegex(self.said[-1], "scores these finished ones again")
            self.assertFalse(started.called)
            with self._yes():
                train.start("resume")
            self.assertIn("--resume", started.call_args.args[0].argv)

    def test_runs_outside_the_project_are_never_started_over(self):
        self._trained()
        experiment = self.root / "experiments" / "e.toml"
        elsewhere = (self.root.parent / f"{self.root.name}-elsewhere").as_posix()
        experiment.write_text(
            experiment.read_text().replace('"../runs/e"', f'"{elsewhere}"')
        )
        train = self.go("Train")
        with mock.patch.object(train.jobs, "run") as started, self._yes():
            train.start("run")
            train.start("smoke")
        self.assertFalse(started.called)
        self.assertRegex(self.said[-1], "outside this project's runs/")

    def test_results_scores_a_stopped_run_and_asks_before_replacing(self):
        run = self._trained()
        results = self.go("Results")
        self.assertIn(
            "1 epochs, best.pt from epoch 1, not scored", results.run_state.text()
        )
        with mock.patch.object(results.jobs, "run") as started:
            results.score_own()
            argv = started.call_args.args[0].argv
            self.assertEqual(
                argv[1:],
                ["--evaluate-only", "--folds", "e", "--seeds", "1", "--save-traces"],
            )
            (run / "evaluation.json").write_text('{"groups": {}, "clips": []}')
            started.reset_mock()
            results.score_own()
            self.assertRegex(self.said[-1], "already has an evaluation.json")
            self.assertFalse(started.called)

    def test_auto_split_ticks_each_session_on_one_side(self):
        form = self.go("Train").form
        form.auto_split()
        sides = [_checked(w) for w in (form.train, form.val, form.test)]
        self.assertEqual([len(s) for s in sides], [1, 1, 1])
        self.assertEqual(len({group for side in sides for group in side}), 3)
        self.assertEqual(form.name.text(), "split-1")
        form.shares["val"].setValue(0)
        form.draw.setValue(2)
        form.auto_split()
        self.assertEqual(_checked(form.val), [])
        self.assertEqual(len(_checked(form.train) + _checked(form.test)), 3)

    def test_clips_shows_what_a_scan_would_list_before_writing(self):
        clips = self.go("Clips")
        self.assertFalse(clips.new_form.isVisibleTo(clips))  # a list exists: folded
        clips.another.setChecked(True)
        with mock.patch("zephyr.gui.pages.probe_size", return_value=NATIVE):
            clips.folder.setText(str(self.root / "data"))
            clips.preview()
        self.assertEqual(clips.found.rowCount(), 4)
        self.assertIn(
            "4 video(s) in 3 session(s). Every file", clips.found_label.text()
        )
        self.assertEqual(clips.found.item(0, 3).text(), "✓")
        clips.frame.setCurrentIndex(1)
        self.assertEqual(clips.frame.currentText(), "1/2: 50 x 40")
        clips.name.setText("half")
        with mock.patch.object(clips.jobs, "run") as started:
            clips.scan()
        argv = started.call_args.args[0].argv
        self.assertEqual(argv[argv.index("--target-size") + 1 :][:2], ["50", "40"])
        clips.thermistor.setText("nothing_{stem}.parquet")
        clips.preview()
        self.assertIn("4 without a breathing file", clips.found_label.text())

    def test_the_list_is_edited_outside_the_app_and_read_again(self):
        clips = self.go("Clips")
        with mock.patch.object(
            QtGui.QDesktopServices, "openUrl", return_value=True
        ) as opened:
            clips.edit_list()
        self.assertEqual(
            Path(opened.call_args.args[0].toLocalFile()), self.window.project.clip_list
        )
        self.assertIn("save it there, then Reload", clips.hint.text())
        self.list.write_text(self.list.read_text() + "\nnot toml = = \n")
        clips.reload_list()
        clips = self.go("Clips")
        self.assertFalse(clips.hint.isVisibleTo(clips))
        self.assertTrue(clips.problem.isVisibleTo(clips))
        self.assertIn("does not load", clips.problem.text())
        self.list.write_text(self.list.read_text().replace("\nnot toml = = \n", ""))
        clips.reload_list()
        self.assertEqual(self.go("Clips").table.rowCount(), 4)
        self.assertFalse(clips.problem.isVisibleTo(clips))

    def test_the_videos_to_predict_are_listed_added_and_removed(self):
        predict = self.go("Predict")
        listed = self.window.project.predictions_dir / "videos" / "videos.toml"
        videos = sorted((self.root / "data").glob("*.mp4"))
        listed.parent.mkdir(parents=True)
        listed.write_text(
            f"[preprocess]\ntarget_size = {list(NATIVE)}\n[derive]\n"
            'pattern = "(?P<group>.+)"\nthermistor = "{stem}.thermistor.parquet"\n'
            + "".join(
                f'[[clip]]\nvideo = "{v.resolve().as_posix()}"\nthermistor = false\n'
                "box = [10, 10, 32, 32]\n"
                for v in videos[:2]
            )
        )
        predict.refresh_listed()
        self.assertEqual(predict.listed.count(), 2)
        self.assertIn("(crop placed)", predict.listed.item(0).text())
        with (
            mock.patch.object(
                QtWidgets.QFileDialog,
                "getOpenFileNames",
                return_value=([str(videos[2])], ""),
            ),
            mock.patch.object(predict.jobs, "run") as started,
        ):
            predict.add_videos()
        self.assertEqual(predict.listed.count(), 3)
        self.assertIn("--prepare", started.call_args.args[0].argv)  # its crop frame
        self.assertFalse(tomllib.loads(listed.read_text())["clip"][2]["thermistor"])
        predict.listed.item(0).setSelected(True)
        predict.remove_videos()
        self.assertEqual(predict.listed.count(), 2)
        checkpoint = self.root / "best.pt"
        predict.checkpoint.addItem("best.pt", checkpoint)
        predict.checkpoint.setCurrentIndex(predict.checkpoint.count() - 1)
        with mock.patch.object(predict.jobs, "run") as started:
            predict.predict_videos()  # from the videos: nothing cached
            self.assertNotIn("--cache", started.call_args.args[0].argv)
            predict.video_cache.setChecked(True)
            started.reset_mock()
            predict.predict_videos()  # asks first (the patched question declines)
            self.assertRegex(self.said[-1], "writes about")
            self.assertFalse(started.called)

    def test_the_onset_heads_output_is_drawn_under_the_trace(self):
        view = TraceView()
        t = np.arange(0, 10, 1 / 60)
        view.set_trace("x", t, np.sin(t), onset=(np.sin(t) + 1) / 2)
        self.assertFalse(view.onset_plot.isHidden())
        view.set_trace("x", t, np.sin(t), onset=np.zeros(len(t)))  # no head
        self.assertTrue(view.onset_plot.isHidden())

    def test_preprocess_asks_with_the_space_it_needs(self):
        self.list.write_text(
            self.list.read_text().replace('.mp4"\n', '.mp4"\nbox = [10, 10, 32, 32]\n')
        )
        train = self.go("Train")
        with mock.patch.object(train.jobs, "run") as started:
            train.start("preprocess")  # declined (the patched question answers nothing)
            self.assertRegex(self.said[-1], r"writes about .* for 4 clip\(s\)")
            self.assertRegex(self.said[-1], "free")
            self.assertFalse(started.called)
            with self._yes():
                train.start("preprocess")
            self.assertEqual(started.call_args.args[0].command, "preprocess")

    def test_a_folder_becomes_a_project_only_when_asked(self):
        other = self.root / "videos-elsewhere"
        other.mkdir()
        self.window.open_project(
            other
        )  # declined (the patched question answers nothing)
        self.assertRegex(self.said[-1], "not a zephyr project yet")
        self.assertFalse((other / "runs").exists())
        self.assertEqual(self.window.project.root, self.root.resolve())
        with self._yes():
            self.window.open_project(other)
        self.assertTrue((other / "runs").is_dir())
        self.assertEqual(self.window.project_label.text(), "videos-elsewhere")
        self.assertEqual(
            self.window.project_where.text(), f"in {other.resolve().parent}"
        )
        self.assertEqual(
            self.window.recent()[:2], [other.resolve(), self.root.resolve()]
        )
        self.said.clear()
        self.window.open_project(self.root)  # a project already: no question
        self.assertEqual(self.said, [])

    def test_the_window_opens_on_a_screen_and_fits_it(self):
        self.assertFalse(QtGui.QIcon(str(ICON)).pixmap(32, 32).isNull())
        free = QtCore.QRect(3840, 6, 1920, 1032)  # a second monitor, to the right
        screen = mock.Mock(availableGeometry=lambda: free)
        window = QtWidgets.QWidget()
        place(window, screen)
        self.assertEqual(window.size(), QtCore.QSize(1600, 949))  # 92% of 1032 high
        self.assertTrue(free.contains(window.geometry()))
        self.assertEqual(window.geometry().center().x(), free.center().x())

    def test_a_short_window_scrolls_the_page_instead_of_growing(self):
        self.go("Breathing")
        self.window.pages[4].jobs.show_log(True)
        self.assertLess(self.window.minimumSizeHint().height(), 600)

    def test_the_header_offers_the_next_step_once_this_one_is_done(self):
        self.go("Clips")
        self.assertEqual(self.window.next.text(), "Next: 2 Boxes  →")
        self.assertEqual(self.window.next.objectName(), "accent")  # Clips is done
        self.window.next.click()
        APP.processEvents()
        self.assertEqual(
            self.window.pages[self.window.stack.currentIndex()].title, "Boxes"
        )
        self.assertEqual(self.window.next.objectName(), "")  # no box yet
        self.assertEqual(self.window.lists.currentText(), "clips.toml")

    def test_only_the_shown_pages_command_panel_is_shown(self):
        clips, train = self.go("Clips"), self.window.pages[4]
        self.assertTrue(clips.jobs.isVisibleTo(self.window))
        self.assertFalse(train.jobs.isVisibleTo(self.window))
        self.go("Project")
        self.assertFalse(self.window.activity.isVisibleTo(self.window))

    def test_unsaved_breathing_changes_are_marked(self):
        editor = self.go("Breathing").editor
        QtCore.QThreadPool.globalInstance().waitForDone()
        APP.processEvents()
        editor.state.set_selection([0], "exclude")
        self.window.refresh_steps()
        self.assertTrue(self.window.nav.item(3).text().endswith("●"))
        self.assertIn("unsaved", self.window.attention.text())
        self.assertTrue(editor.save())
        self.window.refresh_steps()
        self.assertFalse(self.window.nav.item(3).text().endswith("●"))

    def test_boxes_in_the_app_show_the_frame_size_and_do_not_offer_it(self):
        window = self.go("Boxes").window
        self.assertTrue(window.frame_fixed)
        self.assertFalse(window.width_edit.isVisibleTo(window))
        self.assertIn("set when it was made", window.title.text())

    def test_results_says_when_a_clip_has_no_trace(self):
        run = self._trained()
        (run / "evaluation.json").write_text(
            '{"groups": {"test": {"recordings": ["2"], "summary": {}}}, '
            '"clips": [{"group": "test", "clip_id": "c2", "correlation": 0.5}]}'
        )
        results = self.go("Results")
        self.assertEqual(results.result.currentText(), "its test groups")
        self.assertEqual(results.runs.item(0).text(), "e · seed 1")
        results.clips.selectRow(0)
        self.assertIn("No trace saved", results.trace.title.text())

    def test_predict_shows_a_report_with_a_measure_without_a_value(self):
        folder = self.root / "predictions" / "judged"
        folder.mkdir(parents=True)
        t = np.arange(0, 10, 1 / 60)
        pd.DataFrame(
            {"time": t, "signal": np.sin(t), "onset": 0.0, "inhale": False}
        ).to_parquet(folder / "clip.prediction.parquet")
        checks = {"expected_precision": None, "too_fast_frac": True}
        summary = {"expected_precision": None, "too_fast_frac": 0.0, "checks": checks}
        window = {"start": 0.0, "end": 10.0, "checks": {"too_fast_frac": False}}
        report = {"summary": summary, "windows": [window]}
        (folder / "clip.qc.json").write_text(json.dumps(report))
        predict = self.go("Predict")
        predict.fill_folders(folder)
        self.assertIn("expected_precision</b> n/a n/a", predict.verdict.text())
        self.assertIn("1 of 1 windows fail", predict.verdict.text())

    def test_breathing_auto_makes_the_default_the_lists(self):
        editor = self.go("Breathing").editor
        editor.auto()
        QtCore.QThreadPool.globalInstance().waitForDone()
        APP.processEvents()
        self.assertTrue(editor.save())
        self.assertEqual(
            tomllib.loads(self.list.read_text())["events"]["method"],
            "adaptive_prominence",
        )


@unittest.skipUnless(HAS_UI, "needs the ui extra")
class WidgetTests(unittest.TestCase):
    def test_the_job_panel_shows_a_command_running_then_failing(self):
        panel = JobPanel()
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(panel.run(Job("clips", ["check", "missing.toml"], tmp)))
            self.assertTrue(panel.status.text().startswith("Running: zephyr clips"))
            self.assertFalse(panel.progress.isHidden())
            timer = QtCore.QElapsedTimer()
            timer.start()
            while panel.busy or panel._timer.isActive():
                APP.processEvents()
                self.assertLess(timer.elapsed(), 60000)
        self.assertRegex(panel.status.text(), r"^Failed \(exit \d+\): zephyr clips")
        self.assertTrue(panel.progress.isHidden())
        self.assertFalse(panel.log.isHidden())  # a failure opens the log
        self.assertTrue(panel.toggle.isChecked())

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

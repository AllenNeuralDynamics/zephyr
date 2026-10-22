"""A copy of a dataset finds the caches and training records made from the original."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from _clipfiles import touch_clip, write_list

from zephyr import evaluate, features
from zephyr.config import PATH_MAP_ENV, ClipList, canonical, load


def _mapped(copy: Path, original: str):
    return mock.patch.dict(os.environ, {PATH_MAP_ENV: f"{copy.as_posix()}={original}"})


class CanonicalTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.copy = self.root / "copy"

    def test_without_a_map_a_path_is_its_slash_form(self):
        with mock.patch.dict(os.environ, {PATH_MAP_ENV: ""}):
            self.assertEqual(
                canonical(self.copy / "a.mp4"), (self.copy / "a.mp4").as_posix()
            )

    def test_inside_a_copy_a_path_is_named_as_the_original(self):
        with _mapped(self.copy, "/data/set/"):
            self.assertEqual(
                canonical(self.copy / "clips" / "a.mp4"), "/data/set/clips/a.mp4"
            )
            self.assertEqual(canonical(self.copy), "/data/set")

    def test_a_folder_sharing_the_prefix_is_not_the_copy(self):
        other = self.root / "copy2" / "a.mp4"
        with _mapped(self.copy, "/data/set"):
            self.assertEqual(canonical(other), other.as_posix())

    def test_a_malformed_map_is_an_error(self):
        bad = mock.patch.dict(os.environ, {PATH_MAP_ENV: "no-equals-sign"})
        with bad, self.assertRaisesRegex(ValueError, "<copy>=<original>"):
            canonical(self.copy)


class CopyTests(unittest.TestCase):
    """The original dataset is cached; a copy of it (another folder) is looked up."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.original, self.copy, self.cache = (
            self.root / n for n in ("orig", "copy", "cache")
        )
        video = touch_clip(self.original / "clips", 1, 1)
        write_list(self.original / "l.toml", [video])
        (clip,) = load(ClipList, self.original / "l.toml").resolve()
        params = load(ClipList, self.original / "l.toml").preprocess
        sidecar = {
            "key": features.cache_key(clip, params),
            "thermistor": canonical(clip.thermistor),
            "events": features.events_identity(clip),
            "n_frames": 5,
            "n_output": 5,
            "stats": {},
        }
        self.cache.mkdir()
        name = features.prefix(clip, params)
        files = features.paths(self.cache, name)
        files["sidecar"].write_text(json.dumps(sidecar))
        np.save(files["frames"], np.zeros((5, 1, 2, 2), np.uint8))
        self.trained = {
            "train_videos": [canonical(clip.video)],
        }
        shutil.copytree(self.original, self.copy)

    def _copy_clip(self):
        clip_list = load(ClipList, self.copy / "l.toml", check_files=False)
        (clip,) = clip_list.resolve()
        return clip, clip_list.preprocess

    def test_a_copy_finds_the_original_cache_only_through_the_map(self):
        clip, params = self._copy_clip()
        self.assertIsNone(features.lookup(clip, params, self.cache))
        with _mapped(self.copy, self.original.as_posix()):
            entry = features.lookup(clip, params, self.cache)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.thermistor, clip.thermistor)  # read from the copy

    def test_a_copy_of_a_trained_clip_is_refused(self):
        clip, _ = self._copy_clip()
        evaluate.refuse_leaked([self.trained], [clip])  # unmapped: another dataset
        mapped = _mapped(self.copy, self.original.as_posix())
        with mapped, self.assertRaisesRegex(SystemExit, "trained on this video"):
            evaluate.refuse_leaked([self.trained], [clip])

    def test_scoring_needs_the_thermistor_but_not_the_video(self):
        for f in (self.copy / "clips").glob("video_*"):
            f.unlink()  # a copy of the cache and thermistors only
        argv = [
            "--checkpoint",
            "x.pt",
            "--clips",
            str(self.copy / "l.toml"),
            "--cache",
            str(self.cache),
        ]

        class Loaded(Exception):
            pass

        def loader(path, device):
            raise Loaded  # the lists loaded: scoring would start here

        with self.assertRaises(Loaded):
            evaluate.main(argv, loader=loader)
        for f in (self.copy / "clips").glob("thermistor_*"):
            f.unlink()
        with self.assertRaisesRegex(SystemExit, "thermistor file"):
            evaluate.main(argv, loader=loader)


if __name__ == "__main__":
    unittest.main()

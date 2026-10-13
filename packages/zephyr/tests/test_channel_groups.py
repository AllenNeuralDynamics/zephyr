import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from _clipfiles import touch_clip, write_list

from zephyr import features
from zephyr.channels import CHANNEL_NAMES, ChannelSet, encode_stack, make_flow_estimator
from zephyr.config import ClipList, load
from zephyr.dataset import channel_stats
from zephyr.evaluate import models_channels
from zephyr.features import ClipEntry, FeatureReader

T, H, W = 50, 6, 8
VALUES = {"gray": [10], "diff": [20], "flow": [30, 31]}


def _write_entry(folder: Path, name: str, groups) -> ClipEntry:
    """An entry of constant channels (gray 10, diff 20, flow 30/31) in *groups*."""
    for g in groups:
        plane = np.stack([np.full((T, H, W), v, np.uint8) for v in VALUES[g]], axis=1)
        np.save(folder / f"{g}-{name}.npy", plane)
    return ClipEntry(
        clip_id=name,
        recording=name,
        n_frames=T,
        n_output=T,
        stem=folder / name,
        frame_times=folder / f"ftime-{name}.npy",
        baselines=folder / f"dt-{name}.npy",
        times=folder / f"time-{name}.npy",
        target=None,
        events=None,
        groups=tuple(groups),
    )


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)


class ReaderTests(_Tmp):
    def test_a_read_opens_only_the_groups_it_needs(self):
        entry = _write_entry(self.root, "a", features.GROUPS)
        reader = FeatureReader(entry)
        block = reader.read(slice(0, 3), ChannelSet.parse("gray"))
        self.assertEqual(block.shape, (3, 1, H, W))
        self.assertTrue((block == 10).all())
        self.assertEqual(set(reader._maps), {"gray"})
        flow_x = reader.read(slice(5, 9), ChannelSet.parse("flow_x"))
        self.assertTrue((flow_x == 30).all())
        self.assertEqual(set(reader._maps), {"gray", "flow"})

    def test_any_channel_selection_reads_in_stored_order(self):
        reader = FeatureReader(_write_entry(self.root, "s", features.GROUPS))
        value = dict(zip(CHANNEL_NAMES, [10, 20, 30, 31], strict=True))
        for spec in ("gray", "diff", "flow", "gray+flow_y", "gray+diff+flow"):
            channels = ChannelSet.parse(spec)
            for rows in (slice(0, 7), np.array([1, 4, 9])):
                block = reader.read(rows, channels)
                self.assertEqual(
                    [int(block[0, i, 0, 0]) for i in range(block.shape[1])],
                    [value[n] for n in channels.names],
                )

    def test_a_channel_not_stored_is_refused(self):
        reader = FeatureReader(_write_entry(self.root, "g", ("gray",)))
        self.assertEqual(reader.entry.stored.names, ("gray",))
        with self.assertRaisesRegex(ValueError, "does not store"):
            reader.read(slice(0, 2), ChannelSet.parse("gray+diff"))


class StatsTests(_Tmp):
    def test_missing_channels_are_nan_and_recomputed_once_stored(self):
        entries = [_write_entry(self.root, n, ("gray",)) for n in ("a", "b")]
        cache = self.root / "stats.json"
        mean, std = channel_stats(entries, cache, frames_per_clip=10)
        self.assertEqual(mean[0], 10.0)
        self.assertTrue(np.isnan(mean[1:]).all() and np.isnan(std[1:]).all())
        # The diff group is added to both entries: the cached file lacks it, so it is
        # measured again rather than reused.
        entries = [_write_entry(self.root, n, ("gray", "diff")) for n in ("a", "b")]
        mean, _ = channel_stats(entries, cache, frames_per_clip=10)
        np.testing.assert_array_equal(mean[:2], [10.0, 20.0])
        self.assertTrue(np.isnan(mean[2:]).all())

    def test_a_run_measures_only_its_channels_and_needs_only_their_files(self):
        entries = [_write_entry(self.root, n, ("gray",)) for n in ("a", "b")]
        # The sidecars say all groups (the files of the others were never staged).
        entries = [
            e.__class__(**{**e.__dict__, "groups": features.GROUPS}) for e in entries
        ]
        cache = self.root / "only.json"
        gray = ChannelSet.parse("gray")
        mean, _ = channel_stats(entries, cache, frames_per_clip=10, channels=gray)
        self.assertEqual(mean[0], 10.0)
        self.assertTrue(np.isnan(mean[1:]).all())

    def test_a_channel_counts_only_where_every_entry_stores_it(self):
        entries = [
            _write_entry(self.root, "a", features.GROUPS),
            _write_entry(self.root, "b", ("gray", "diff")),
        ]
        mean, _ = channel_stats(entries, self.root / "s.json", frames_per_clip=10)
        np.testing.assert_array_equal(mean[:2], [10.0, 20.0])
        self.assertTrue(np.isnan(mean[2:]).all())


class RequireTests(_Tmp):
    def _cached(self, groups) -> tuple[list, ClipList]:
        video = touch_clip(self.root / "d", 1, 1)
        clip_list = load(ClipList, write_list(self.root / "l.toml", [video]))
        (clip,) = clip_list.resolve()
        params = clip_list.preprocess
        sidecar = {
            "key": features.cache_key(clip, params),
            "thermistor": clip.thermistor.as_posix(),
            "events": features.events_identity(clip),
            "n_frames": T,
            "n_output": T,
            "stats": {},
            "groups": list(groups),
        }
        name = features.prefix(clip, params)
        (self.root / "cache").mkdir(exist_ok=True)
        features.paths(self.root / "cache", name)["sidecar"].write_text(
            json.dumps(sidecar)
        )
        return [clip], clip_list

    def test_entries_lacking_a_group_are_refused_with_the_command(self):
        clips, clip_list = self._cached(["gray"])
        cache = self.root / "cache"
        (entry,) = features.require(clips, clip_list.preprocess, cache)
        self.assertEqual(entry.groups, ("gray",))
        features.require(clips, clip_list.preprocess, cache, ChannelSet.parse("gray"))
        with self.assertRaisesRegex(SystemExit, "--channels gray\\+flow"):
            features.require(
                clips, clip_list.preprocess, cache, ChannelSet.parse("gray+flow_x")
            )


class ModelsChannelsTests(unittest.TestCase):
    def test_the_union_of_the_models_channels_is_required(self):
        class Net:
            def __init__(self, spec):
                self.channels = ChannelSet.parse(spec)

        models = [(Net("gray"), None, None), (Net("gray+flow_x"), None, None)]
        self.assertEqual(models_channels(models).names, ("gray", "flow_x"))
        self.assertIsNone(models_channels([(object(), None, None)]))


class EncodeStackTests(unittest.TestCase):
    def test_only_the_groups_asked_for_are_computed(self):
        rng = np.random.default_rng(0)
        gray = rng.integers(0, 255, (16, 16), dtype=np.uint8)
        reference = rng.integers(0, 255, (16, 16), dtype=np.uint8)
        planes, _, n_flow = encode_stack(gray, reference, None, groups=("gray", "diff"))
        self.assertEqual(set(planes), {"gray", "diff"})
        self.assertEqual(n_flow, 0)
        np.testing.assert_array_equal(planes["gray"][0], gray)
        everything, _, _ = encode_stack(gray, reference, make_flow_estimator())
        self.assertEqual(
            {k: v.shape[0] for k, v in everything.items()},
            {"gray": 1, "diff": 1, "flow": 2},
        )
        np.testing.assert_array_equal(everything["diff"], planes["diff"])

    def test_the_first_frame_has_neutral_motion(self):
        gray = np.full((4, 4), 7, np.uint8)
        planes, _, _ = encode_stack(gray, None, None, groups=("diff", "flow"))
        self.assertTrue((planes["diff"] == 128).all() and (planes["flow"] == 128).all())
        self.assertEqual(planes["flow"].shape, (2, 4, 4))


if __name__ == "__main__":
    unittest.main()

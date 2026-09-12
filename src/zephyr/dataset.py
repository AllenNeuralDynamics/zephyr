"""Windowed torch dataset over the preprocessed uint8 feature arrays.

Windows, not whole clips
------------------------
A clip is much longer than fits usefully in one gradient step, so training
samples fixed-length windows instead, long enough that the TCN's receptive
field is filled well inside the window and short enough to batch.

Window starts are drawn at random rather than being fixed on a grid, so the
effective sample count is every valid offset in a clip rather than one
window per receptive-field-length. Validation windows *are* gridded and
non-overlapping, so the reported number is stable between epochs.

A window is two sequences, not one
----------------------------------
``window`` counts *output* samples at 60 Hz; the frames fed to the CNN live on
the *selection* grid.  The two are joined inside the model
(:func:`~.model.resample_embeddings`), so every item carries ``t_in`` and
``t_out`` alongside the pixels, both relative to the window's own start.

Input frames are read at **fractional** selection positions spaced ``stretch``
apart, which is what keeps time-stretch free: a 3x stretch spans three times as
much clip but hands the CNN the same number of frames.  Unstretched on a 60 Hz
selection grid the positions are integers and the read is a contiguous slice.

The start position is a *float*, so the output grid's phase relative to the
native video frames is continuous rather than locked to the selection grid.

The draw is keyed on the dataset index alone, never on a mutable epoch counter.
DataLoader workers are spawned on Windows and hold a frozen copy of the dataset,
so an epoch attribute set in the parent would never reach them and every epoch
would silently replay the same windows.  Instead the index space runs the whole
length of training and :class:`EpochRangeSampler` -- which lives in the parent --
hands out a fresh slice of it each epoch.

What never happens here
-----------------------
Windows are never mixed across clips: a window lives inside exactly one clip.

Normalisation
-------------
Stored values are the *companded* uint8 codes, and that is what the model is
fed, standardised per channel.  Decoding flow back to pixels first would undo
the variance stabilisation that ``asinh`` provides: the companded code is the
better-conditioned input, and the network does not care that the mapping to
pixels is non-linear as long as it is monotonic and fixed.

Statistics are measured once over a sample of the training clips and cached, so
they cannot drift between training and inference, and are not recomputed per run.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from .augment import (
    AugmentConfig,
    add_motion_noise,
    apply_spatial,
    gather_positions,
    jitter_positions,
    rate_targeted_stretch,
    scale_motion_channels,
)
from .channels import ALL_CHANNELS, CHANNEL_NAMES, N_CHANNELS, ChannelSet
from .targets import onset_heatmap

STATS_FILENAME = "channel_stats.json"

INTERP_MARGIN = 2
"""Extra input frames kept either side of a window's output span.

The selection grid is only *nearly* uniform (each anchor sits within half a
native frame period of its tick) while the output grid is exactly uniform.  Two
frames is far more slack than that jitter can consume, which keeps every output
sample inside the input span so the interpolation never clamps.
"""


@dataclass(frozen=True)
class ClipEntry:
    """One preprocessed clip: where its arrays live and which session it is.

    Two lengths because a clip lives on two grids: *n_frames* on the selection
    grid (``features`` / ``frame_times``) and *n_output* on the 60 Hz output
    grid (``times`` / ``target``).
    """

    clip_id: str
    session_idx: int
    part: int
    n_frames: int
    n_output: int
    features: Path
    frame_times: Path
    baselines: Path
    times: Path
    target: Path | None
    events: Path | None

    @property
    def has_target(self) -> bool:
        return self.target is not None


def load_manifest(
    features_dir: Path, split: str = "train", camera: str = "face"
) -> tuple[dict, list[ClipEntry]]:
    """Read ``manifest_{split}_{camera}.json`` and resolve its paths."""
    path = features_dir / f"manifest_{split}_{camera}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m breathing_cnn_tcn.preprocess` first."
        )
    manifest = json.loads(path.read_text())
    entries = [
        ClipEntry(
            clip_id=c["clip_id"],
            session_idx=c["session_idx"],
            part=c["part"],
            n_frames=c["n_frames"],
            n_output=c["n_output"],
            features=features_dir / c["features"],
            frame_times=features_dir / c["frame_times"],
            baselines=features_dir / c["baselines"],
            times=features_dir / c["times"],
            target=features_dir / c["target"] if c.get("target") else None,
            events=features_dir / c["events"] if c.get("events") else None,
        )
        for c in manifest["clips"]
    ]
    return manifest["config"], entries


def channel_stats(
    entries: list[ClipEntry],
    cache_path: Path,
    *,
    frames_per_clip: int = 400,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and standard deviation of the stored uint8 codes.

    Sampled rather than exhaustive: *frames_per_clip* frames spread evenly over
    each clip is enough to pin the mean well under a count.  Cached to
    *cache_path* so training and inference cannot disagree.

    Always measured over every stored channel, whatever subset a run trains on
    -- so one cache serves every :class:`~.channels.ChannelSet` with no
    recompute, and two runs can never disagree about a channel's mean.  Slice
    the result with :meth:`~.channels.ChannelSet.take_stats`.
    """
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        return np.array(cached["mean"], np.float32), np.array(cached["std"], np.float32)

    rng = np.random.default_rng(seed)
    total = np.zeros(N_CHANNELS, np.float64)
    total_sq = np.zeros(N_CHANNELS, np.float64)
    count = 0
    for entry in entries:
        array = np.load(entry.features, mmap_mode="r")
        n = min(frames_per_clip, len(array))
        # Skip frame 0: it has no predecessor, so its diff and flow channels are
        # zero by construction and would drag the motion statistics down.
        indices = rng.choice(np.arange(1, len(array)), size=n, replace=False)
        block = np.asarray(array[np.sort(indices)], dtype=np.float64)
        total += block.sum(axis=(0, 2, 3))
        total_sq += (block**2).sum(axis=(0, 2, 3))
        count += block.shape[0] * block.shape[2] * block.shape[3]

    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean**2, 1e-12))
    cache_path.write_text(
        json.dumps(
            {
                "channels": list(CHANNEL_NAMES),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "n_clips": len(entries),
                "frames_per_clip": frames_per_clip,
                "n_pixels": int(count),
            },
            indent=2,
        )
    )
    return mean.astype(np.float32), std.astype(np.float32)


def _load_target_frame(
    entry: ClipEntry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Output-grid times, the target signal on them, and the onset times.

    The stored ``onset_heatmap`` column is deliberately unused: a window lands
    on its own output grid, and rebuilding from onset *times* places each bump
    exactly, where resampling an already-discretised heatmap would smear a
    sigma=20 ms bump that is barely one sample wide.
    """
    frame = pd.read_parquet(entry.target)
    onsets = np.load(entry.events)["onset_times"]
    return (
        frame["time"].to_numpy(np.float64),
        frame["signal"].to_numpy(np.float32),
        onsets.astype(np.float64),
    )


class WindowDataset(Dataset):
    """Fixed-length windows over a set of clips.

    Parameters
    ----------
    entries:
        Clips to draw from.  All must carry a target.
    window:
        Window length in *output* samples.  How many frames the CNN reads for
        it follows from *select_fs* and the drawn stretch.
    select_fs, output_fs, motion_tau_s, onset_sigma_s:
        From the preprocessing manifest's ``config``, never defaulted: a cache
        built at one selection rate read as if it were another would mis-scale
        every motion channel with no visible symptom.
    mean, std:
        Per-channel standardisation constants from :func:`channel_stats`, over
        *all* stored channels.  Sliced here to match *channels*, so callers
        pass the full-width arrays and cannot mismatch them.
    channels:
        Which stored channels this dataset yields, and in what order.  The
        stored array always holds every channel; this selects the subset a
        model is trained on.
    stride:
        When given, windows are laid out on a fixed grid with this hop and the
        dataset is deterministic -- the validation mode.  When ``None``, each
        ``__getitem__`` draws a random start inside the clip.
    length:
        Size of the random-window index space.  Should cover all of training
        (steps x batch x epochs), since :class:`EpochRangeSampler` walks through
        it once rather than restarting each epoch.
    seed:
        Base seed for random starts.  Combined with the dataset index, so the
        window drawn for a given (seed, index) pair is always the same.
    span:
        Fraction of each clip this dataset may draw from, as ``(start, stop)``.
        ``(0, 0.75)`` and ``(0.75, 1)`` split every clip in time, which is how a
        run training on every session still gets a stopping signal without
        holding any session out entirely.  A within-session signal is not the
        same thing as cross-session generalisation.
    """

    def __init__(
        self,
        entries: list[ClipEntry],
        *,
        window: int,
        mean: np.ndarray,
        std: np.ndarray,
        select_fs: float,
        output_fs: float,
        motion_tau_s: float,
        onset_sigma_s: float = 0.020,
        stride: int | None = None,
        length: int | None = None,
        seed: int = 0,
        augment: AugmentConfig | None = None,
        span: tuple[float, float] = (0.0, 1.0),
        channels: ChannelSet = ALL_CHANNELS,
    ) -> None:
        missing = [e.clip_id for e in entries if not e.has_target]
        if missing:
            raise ValueError(f"clips without a target cannot be trained on: {missing}")
        if not entries:
            raise ValueError("WindowDataset needs at least one clip")

        self.entries = entries
        self.window = window
        self.seed = seed
        self.augment = augment or AugmentConfig()
        self.select_fs = select_fs
        self.output_fs = output_fs
        self.motion_tau_s = motion_tau_s
        self.onset_sigma_s = onset_sigma_s

        # Input positions are spaced `stretch` apart, so `n_core` of them span
        # exactly as much clip as the window's output samples do.
        self.step_base = select_fs / output_fs
        self.margin = INTERP_MARGIN
        self.n_core = round((window - 1) * self.step_base) + 1
        self.n_in = self.n_core + 2 * self.margin
        if stride is not None and self.augment.enabled:
            # Gridded mode is the validation path; augmenting it would make the
            # reported number drift with the augmentation strength rather than
            # with the model.
            raise ValueError("augmentation must not be enabled on gridded windows")
        self.channels = channels
        selected_mean, selected_std = channels.take_stats(mean, std)
        # (1, C, 1, 1) so broadcasting hits the channel axis of a (T, C, H, W) window.
        self.mean = torch.from_numpy(np.asarray(selected_mean, np.float32)).view(
            1, -1, 1, 1
        )
        self.std = torch.from_numpy(np.asarray(selected_std, np.float32)).view(
            1, -1, 1, 1
        )

        self._arrays: dict[str, np.ndarray] = {}
        self._targets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._frame_time_cache: dict[str, np.ndarray] = {}

        # A stretch factor above 1 spans more of the clip than it returns, so a
        # clip must be long enough for the widest draw, not just the window.
        self.max_source = int(np.ceil(self.extent(self.augment.max_source_frames))) + 1
        if not 0.0 <= span[0] < span[1] <= 1.0:
            raise ValueError(f"span must satisfy 0 <= start < stop <= 1, got {span}")
        self.span = span
        self.bounds = {
            e.clip_id: (int(e.n_frames * span[0]), int(e.n_frames * span[1]))
            for e in entries
        }
        usable = [
            e
            for e in entries
            if self.bounds[e.clip_id][1] - self.bounds[e.clip_id][0] >= self.max_source
        ]
        if not usable:
            raise ValueError(
                f"no clip has {self.max_source} selection frames inside span "
                f"{span} (longest clip is {max(e.n_frames for e in entries)})"
            )
        self.usable = usable

        if stride is not None:
            # Gridded mode never stretches, so the span of one window is fixed.
            extent = self.extent(1.0)
            hop = max(1.0, stride * self.step_base)
            self.index: list[tuple[int, float]] = [
                (i, float(start))
                for i, entry in enumerate(usable)
                for start in np.arange(
                    self.bounds[entry.clip_id][0],
                    self.bounds[entry.clip_id][1] - extent,
                    hop,
                )
            ]
            self.random = False
        else:
            self.index = []
            self.random = True
            # Sample clips in proportion to their usable offsets, so a short clip
            # is not over-represented relative to the data it actually holds.
            offsets = np.array(
                [
                    self.bounds[e.clip_id][1]
                    - self.bounds[e.clip_id][0]
                    - self.max_source
                    + 1
                    for e in usable
                ],
                np.float64,
            )
            self.weights = offsets / offsets.sum()
            self.length = length if length is not None else 8 * len(usable)

    def extent(self, stretch: float) -> float:
        """Selection-grid span one window covers at *stretch*, including margin."""
        return (self.n_in - 1) * stretch

    def motion_scale(self, stretch: float) -> float:
        """Factor the motion channels need at *stretch*.

        Stored values are displacement per ``motion_tau_s``, but a stretched
        window makes one output sample cover ``stretch / output_fs`` seconds.
        Equals *stretch* exactly at the default 60 Hz / 16.67 ms pairing.
        """
        return (stretch / self.output_fs) / self.motion_tau_s

    def __len__(self) -> int:
        return len(self.index) if not self.random else self.length

    def _array(self, entry: ClipEntry) -> np.ndarray:
        # Opened lazily and cached per process: DataLoader workers are spawned on
        # Windows, so a memmap opened in the parent would not survive the fork.
        array = self._arrays.get(entry.clip_id)
        if array is None:
            array = np.load(entry.features, mmap_mode="r")
            self._arrays[entry.clip_id] = array
        return array

    def _target(self, entry: ClipEntry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target = self._targets.get(entry.clip_id)
        if target is None:
            target = _load_target_frame(entry)
            self._targets[entry.clip_id] = target
        return target

    def _frame_times(self, entry: ClipEntry) -> np.ndarray:
        times = self._frame_time_cache.get(entry.clip_id)
        if times is None:
            times = np.load(entry.frame_times)
            self._frame_time_cache[entry.clip_id] = times
        return times

    def _draw_start(
        self, rng: np.random.Generator, lo: int, hi: int, stretch: float
    ) -> float:
        """Uniform float start position leaving room for the whole window."""
        top = hi - 1 - self.extent(stretch)
        return float(rng.uniform(lo, top)) if top > lo else float(lo)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        window = self.window
        stretch = 1.0
        if self.random:
            rng = np.random.default_rng((self.seed, i))
            clip_i = int(rng.choice(len(self.usable), p=self.weights))
            entry = self.usable[clip_i]
            lo, hi = self.bounds[entry.clip_id]
            target_times, signal_all, onset_times = self._target(entry)
            frame_times = self._frame_times(entry)

            start = self._draw_start(rng, lo, hi, 1.0)
            if self.augment.time_stretch > 1.0:
                # Measure the rate on a provisional window first, then choose
                # the stretch: the target is already in memory, so this costs a
                # peak count and nothing else.
                probe_t = float(frame_times[int(start)])
                probe = int(
                    np.clip(
                        round((probe_t - target_times[0]) * self.output_fs),
                        0,
                        max(0, len(signal_all) - window),
                    )
                )
                stretch = rate_targeted_stretch(
                    signal_all[probe : probe + window],
                    rng,
                    self.augment,
                    self.output_fs,
                )
                start = self._draw_start(rng, lo, hi, stretch)
        else:
            rng = None
            clip_i, start = self.index[i]
            entry = self.usable[clip_i]
            target_times, signal_all, onset_times = self._target(entry)
            frame_times = self._frame_times(entry)

        # `stretch` selection-steps apart, so a stretched window reaches
        # further into the clip while the CNN still sees exactly `n_in` frames.
        positions = start + np.arange(self.n_in) * stretch
        if rng is not None and self.augment.select_jitter > 0:
            positions = jitter_positions(positions, rng, self.augment.select_jitter)
        positions = np.clip(positions, 0.0, entry.n_frames - 1)

        # One contiguous slab covers every position; channels are selected
        # first, so nothing pays to cast planes this model never sees.
        first = int(np.floor(positions[0]))
        last = int(np.ceil(positions[-1]))
        slab = self.channels.take(np.asarray(self._array(entry)[first : last + 1]))
        block = gather_positions(slab, positions - first)

        in_times = gather_positions(frame_times, positions, dtype=np.float64)
        # Anchoring at the first non-margin input frame keeps the window's
        # output span inside its input span.
        origin = float(in_times[self.margin])
        out_times = origin + np.arange(window) * (stretch / self.output_fs)

        signal = np.interp(out_times, target_times, signal_all).astype(np.float32)
        heatmap = onset_heatmap(out_times, onset_times, sigma_s=self.onset_sigma_s)

        if rng is not None and self.augment.enabled:
            if self.augment.scale_motion:
                block = scale_motion_channels(
                    block, self.motion_scale(stretch), self.channels
                )
            block = add_motion_noise(block, rng, self.augment, self.channels)
            block = apply_spatial(block, rng, self.augment, self.channels)

        features = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32))
        features = (features - self.mean) / self.std

        return {
            "features": features,  # (T_in, C, H, W) float32
            # Window-relative, never absolute clip time -- float32 could not
            # resolve the interpolation weight otherwise.
            "t_in": torch.from_numpy((in_times - origin).astype(np.float32)),
            "t_out": torch.from_numpy((out_times - origin).astype(np.float32)),
            "signal": torch.from_numpy(signal),  # (T_out,)
            "onset": torch.from_numpy(heatmap.astype(np.float32)),  # (T_out,)
            "clip_index": torch.tensor(clip_i),
            "start": torch.tensor(start, dtype=torch.float32),
        }


class EpochRangeSampler(Sampler[int]):
    """Hand out a fresh contiguous slice of the index space each epoch.

    Lives in the parent process -- DataLoader regenerates indices from the
    sampler on every ``iter()`` -- so bumping :attr:`epoch` changes which
    windows are drawn even when ``persistent_workers`` keeps the same frozen
    worker copies of the dataset alive.
    """

    def __init__(self, per_epoch: int, epoch: int = 0) -> None:
        self.per_epoch = per_epoch
        self.epoch = epoch

    def __iter__(self):
        base = self.epoch * self.per_epoch
        return iter(range(base, base + self.per_epoch))

    def __len__(self) -> int:
        return self.per_epoch


def reserve_test_sessions(
    entries: list[ClipEntry],
    path: Path,
    *,
    n_test: int = 3,
    seed: int = 0,
) -> list[int]:
    """Draw and persist a set of sessions that training never sees.

    Validation selects the checkpoint, so a run's own score is optimistically
    biased.  These sessions are removed before training and touched by nothing
    -- no training, no validation, no selection -- which makes them the only
    clean local estimate of generalisation.

    Persisted on first use and reloaded verbatim after.  Re-drawing per run
    would leak: a session held out in one run becomes training data in the next.
    The file is the source of truth, not the seed.
    """
    sessions = sorted({e.session_idx for e in entries})
    if path.exists():
        record = json.loads(path.read_text())
        reserved = list(record["test_sessions"])
        missing = [s for s in reserved if s not in sessions]
        if missing:
            raise ValueError(
                f"{path} reserves sessions {missing} that are absent from the "
                "manifest.  Refusing to continue: silently re-drawing would put "
                "previously held-out sessions into training."
            )
        return reserved

    if n_test >= len(sessions):
        raise ValueError(
            f"cannot reserve {n_test} of {len(sessions)} sessions -- nothing "
            "would be left to train on"
        )
    rng = np.random.default_rng(seed)
    reserved = sorted(rng.choice(sessions, size=n_test, replace=False).tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "test_sessions": reserved,
                # Seed and pool together re-derive test_sessions exactly -- the
                # draw is over list positions -- so the file can be audited
                # rather than merely trusted.  It is still the source of truth:
                # the pool changes if the manifest does.
                "seed": seed,
                "drawn_from": sessions,
            },
            indent=2,
        )
    )
    return reserved

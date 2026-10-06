"""Windowed torch dataset over the preprocessed uint8 feature arrays.

Training samples fixed-length windows from random starts, so every valid offset is
reachable; validation windows are gridded and non-overlapping so the number is stable.
``window`` counts 60 Hz output samples; input frames sit on the selection grid at
fractional positions ``stretch`` apart.
The draw is keyed on the dataset index alone, never an epoch counter (spawned DataLoader
workers hold a frozen copy); :class:`EpochRangeSampler` hands out a fresh slice of the
index space each epoch. The companded uint8 codes are standardised per channel with
statistics measured once over the training clips and cached.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from zephyr.augment import (
    AugmentConfig,
    apply_spatial,
    gather_positions,
    jitter_positions,
    rate_targeted_stretch,
    scale_motion_channels,
)
from zephyr.channels import ALL_CHANNELS, CHANNEL_NAMES, N_CHANNELS, ChannelSet

from .features import ClipEntry
from .rates import DEFAULT_RATE_BINS_HZ, balance_weights, rate_track, window_rates
from .targets import onset_heatmap

INTERP_MARGIN = 2
"""Extra input frames kept either side of a window's output span, so the
selection grid's jitter never pushes an output sample outside the input span."""

BALANCE_HOP_S = 0.25
"""Spacing of the candidate window starts that rate balancing weighs; a drawn start
is jittered uniformly within its cell, so every offset stays reachable."""


def stats_path(cache_dir: Path, entries: Sequence[ClipEntry]) -> Path:
    """Where the statistics over exactly *entries* are cached: next to the
    features, named for the ordered clip set so another set never reuses them."""
    blob = json.dumps([e.features.name for e in entries]).encode()
    return cache_dir / f"channel_stats-{hashlib.sha256(blob).hexdigest()[:10]}.json"


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
    *cache_path* (see :func:`stats_path`) so repeated runs over the same clips
    cannot disagree.

    Always measured over every stored channel, whatever subset a run trains on
    -- so one cache serves every :class:`zephyr.channels.ChannelSet` with no
    recompute, and two runs can never disagree about a channel's mean.  Slice
    the result with :meth:`~.channels.ChannelSet.take_stats`.
    """
    names = [entry.features.name for entry in entries]
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("clips") == names:
            return np.array(cached["mean"], np.float32), np.array(
                cached["std"], np.float32
            )

    rng = np.random.default_rng(seed)
    total = np.zeros(N_CHANNELS, np.float64)
    total_sq = np.zeros(N_CHANNELS, np.float64)
    count = 0
    for entry in entries:
        array = np.load(entry.features, mmap_mode="r")
        n = min(frames_per_clip, len(array) - 1)
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
                "clips": names,
                "frames_per_clip": frames_per_clip,
                "n_pixels": int(count),
            },
            indent=2,
        )
    )
    return mean.astype(np.float32), std.astype(np.float32)


def mixture_weights(
    offsets: np.ndarray, source_of: Sequence[int], weights: Sequence[float]
) -> np.ndarray:
    """Clip draw probabilities for a weighted mix of clip lists.

    Each list gets its relative share of the mass (*weights* are normalised, so
    0.5 / 0.5 and 1 / 1 agree); within a list, clips are drawn in proportion to
    their usable *offsets*, as a single list always was.  *source_of* gives each
    clip's list.  With one list this reduces exactly to ``offsets / offsets.sum()``.
    """
    source_of = np.asarray(source_of)
    share = np.asarray(weights, np.float64) / float(np.sum(weights))
    empty = [i for i in range(len(share)) if not (source_of == i).any()]
    if empty:
        raise ValueError(
            f"training list(s) {empty} have no clip long enough to draw windows from"
        )
    probabilities = np.zeros(len(offsets))
    for i, fraction in enumerate(share):
        members = source_of == i
        probabilities[members] = fraction * offsets[members] / offsets[members].sum()
    return probabilities


def _load_target_frame(
    entry: ClipEntry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Output-grid times, the target signal on them, and the onset times.

    The stored ``onset_heatmap`` column is unused; a window rebuilds it from
    onset *times* on its own output grid instead of resampling the column.
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
    sources:
        Clip lists to draw from, each a list of entries; all must carry a
        target.  Windows are drawn across lists in proportion to *weights*.
    window:
        Window length in *output* samples. Input frame count follows from
        *select_fs* and the drawn stretch.
    select_fs, output_fs, motion_tau_s, onset_sigma_s:
        From the clips' :class:`~zephyr.config.PreprocessParams`.
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
    weights:
        Relative share of the windows each list in *sources* receives (default
        equal), each list still drawing its clips in proportion to their usable
        offsets.  Lets a handful of clips from a new camera view be oversampled
        against a much larger set instead of drowning in it; see
        :func:`mixture_weights`.
    span:
        Fraction of each clip this dataset may draw from, as ``(start, stop)``.
        ``(0, 0.75)`` and ``(0.75, 1)`` split every clip in time, which is how a
        run training on every session still gets a stopping signal without
        holding any session out entirely.  A within-session signal is not the
        same thing as cross-session generalisation.
    """

    def __init__(
        self,
        sources: Sequence[Sequence[ClipEntry]],
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
        weights: Sequence[float] | None = None,
        rate_balance: float = 0.0,
        rate_bins_hz: Sequence[float] = DEFAULT_RATE_BINS_HZ,
    ) -> None:
        entries = [e for source in sources for e in source]
        source_of = [i for i, source in enumerate(sources) for _ in source]
        weights = [1.0] * len(sources) if weights is None else list(weights)
        if len(weights) != len(sources):
            raise ValueError(f"{len(sources)} clip lists but {len(weights)} weights")
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

        self.step_base = select_fs / output_fs
        self.margin = INTERP_MARGIN
        self.n_core = round((window - 1) * self.step_base) + 1
        self.n_in = self.n_core + 2 * self.margin
        if stride is not None and self.augment.enabled:
            # Gridded mode is the validation path; augmenting it would make the
            # reported number drift with the augmentation strength rather than
            # with the model.
            raise ValueError("augmentation must not be enabled on gridded windows")
        if rate_balance and stride is not None:
            raise ValueError("rate balancing applies to random windows only")
        self.rate_balance = rate_balance
        self.channels = channels
        selected_mean, selected_std = channels.take_stats(mean, std)
        # (1, C, 1, 1) so broadcasting hits the channel axis of a (T, C, H, W) window.
        self.mean = torch.from_numpy(np.asarray(selected_mean, np.float32)).view(
            1, -1, 1, 1
        )
        self.std = torch.from_numpy(np.asarray(selected_std, np.float32)).view(
            1, -1, 1, 1
        )

        # Keyed by feature path: the clip label is for people and need not be unique.
        self._arrays: dict[Path, np.ndarray] = {}
        self._targets: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._frame_time_cache: dict[Path, np.ndarray] = {}

        # A clip must be long enough for the widest stretch draw, not just window.
        self.max_source = int(np.ceil(self.extent(self.augment.max_source_frames))) + 1
        if not 0.0 <= span[0] < span[1] <= 1.0:
            raise ValueError(f"span must satisfy 0 <= start < stop <= 1, got {span}")
        self.span = span
        self.bounds = {
            e.features: (int(e.n_frames * span[0]), int(e.n_frames * span[1]))
            for e in entries
        }
        keep = [
            i
            for i, e in enumerate(entries)
            if self.bounds[e.features][1] - self.bounds[e.features][0]
            >= self.max_source
        ]
        usable = [entries[i] for i in keep]
        if not usable:
            raise ValueError(
                f"no clip has {self.max_source} selection frames inside span "
                f"{span} (longest clip is {max(e.n_frames for e in entries)})"
            )
        self.usable = usable

        if stride is not None:
            extent = self.extent(1.0)  # gridded mode never stretches
            hop = max(1.0, stride * self.step_base)
            self.index: list[tuple[int, float]] = [
                (i, float(start))
                for i, entry in enumerate(usable)
                for start in np.arange(
                    self.bounds[entry.features][0],
                    self.bounds[entry.features][1] - extent,
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
                    self.bounds[e.features][1]
                    - self.bounds[e.features][0]
                    - self.max_source
                    + 1
                    for e in usable
                ],
                np.float64,
            )
            self.weights = mixture_weights(
                offsets, [source_of[i] for i in keep], weights
            )
            self.length = length if length is not None else 8 * len(usable)
            if rate_balance > 0:
                self._balance(
                    [source_of[i] for i in keep], weights, rate_balance, rate_bins_hz
                )

    def _balance(
        self,
        source_of: Sequence[int],
        weights: Sequence[float],
        power: float,
        bins_hz: Sequence[float],
    ) -> None:
        """Candidate starts every :data:`BALANCE_HOP_S` with draw probabilities that
        even out their true breathing rate within each source; each source keeps
        its share of *weights*."""
        self.hop = max(1.0, BALANCE_HOP_S * self.select_fs)
        share = np.asarray(weights, np.float64) / float(np.sum(weights))
        clip_of, starts, rates = [], [], []
        for clip_i, entry in enumerate(self.usable):
            lo, hi = self.bounds[entry.features]
            top = hi - 1 - self.extent(1.0)
            grid = np.arange(lo, top, self.hop) if top > lo else np.array([float(lo)])
            target_times, _, onsets = _load_target_frame(entry)
            first = np.load(entry.frame_times)[grid.astype(int)]
            index = np.searchsorted(target_times, first)
            track = rate_track(target_times, onsets)
            clip_of.append(np.full(len(grid), clip_i))
            starts.append(grid)
            rates.append(window_rates(track, index, self.window))
        clip_of, starts, rates = map(np.concatenate, (clip_of, starts, rates))
        source = np.asarray(source_of)[clip_of]
        probability = np.zeros(len(starts))
        for s, fraction in enumerate(share):
            members = source == s
            probability[members] = fraction * balance_weights(
                rates[members], bins_hz, power
            )
        self.candidate_clip = clip_of
        self.candidate_start = starts
        self.candidate_rate = rates
        self._candidate_cdf = np.cumsum(probability) / probability.sum()

    def extent(self, stretch: float) -> float:
        """Selection-grid span one window covers at *stretch*, incl. margin."""
        return (self.n_in - 1) * stretch

    def motion_scale(self, stretch: float) -> float:
        """Motion-channel rescale factor at *stretch* (equals *stretch* at the
        default 60 Hz / 16.67 ms pairing)."""
        return (stretch / self.output_fs) / self.motion_tau_s

    def __len__(self) -> int:
        return len(self.index) if not self.random else self.length

    def _array(self, entry: ClipEntry) -> np.ndarray:
        # Opened lazily and cached per process: DataLoader workers are spawned on
        # Windows, so a memmap opened in the parent would not survive the fork.
        array = self._arrays.get(entry.features)
        if array is None:
            array = np.load(entry.features, mmap_mode="r")
            self._arrays[entry.features] = array
        return array

    def _target(self, entry: ClipEntry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target = self._targets.get(entry.features)
        if target is None:
            target = _load_target_frame(entry)
            self._targets[entry.features] = target
        return target

    def _frame_times(self, entry: ClipEntry) -> np.ndarray:
        times = self._frame_time_cache.get(entry.features)
        if times is None:
            times = np.load(entry.frame_times)
            self._frame_time_cache[entry.features] = times
        return times

    def _draw_start(
        self, rng: np.random.Generator, lo: int, hi: int, stretch: float
    ) -> float:
        """Float start position leaving room for the whole window."""
        top = hi - 1 - self.extent(stretch)
        return float(rng.uniform(lo, top)) if top > lo else float(lo)

    def _draw(self, i: int) -> tuple[np.random.Generator | None, int, float, float]:
        """The ``(rng, clip index, start, stretch)`` of window *i*.

        *rng* is the generator positioned after the draw, for the augmentation
        that follows; ``None`` in gridded mode, which draws nothing.
        """
        stretch = 1.0
        if not self.random:
            clip_i, start = self.index[i]
            return None, clip_i, start, stretch

        rng = np.random.default_rng((self.seed, i))
        balanced = self.rate_balance > 0
        if balanced:
            c = int(np.searchsorted(self._candidate_cdf, rng.random(), side="right"))
            c = min(c, len(self._candidate_cdf) - 1)
            clip_i = int(self.candidate_clip[c])
            entry = self.usable[clip_i]
            lo, hi = self.bounds[entry.features]
            top = max(float(lo), hi - 1 - self.extent(1.0))
            start = min(float(self.candidate_start[c] + rng.uniform(0, self.hop)), top)
        else:
            clip_i = int(rng.choice(len(self.usable), p=self.weights))
            entry = self.usable[clip_i]
            lo, hi = self.bounds[entry.features]
            start = self._draw_start(rng, lo, hi, 1.0)
        if self.augment.time_stretch > 1.0:
            # Measure the rate on a provisional window first, then choose
            # the stretch: the target is already in memory, so this costs a
            # peak count and nothing else.
            target_times, signal_all, _ = self._target(entry)
            probe_t = float(self._frame_times(entry)[int(start)])
            probe = int(
                np.clip(
                    round((probe_t - target_times[0]) * self.output_fs),
                    0,
                    max(0, len(signal_all) - self.window),
                )
            )
            stretch = rate_targeted_stretch(
                signal_all[probe : probe + self.window],
                rng,
                self.augment,
                self.output_fs,
            )
            if balanced:  # keep the balanced start; only make the window fit
                start = min(start, max(float(lo), hi - 1 - self.extent(stretch)))
            else:
                start = self._draw_start(rng, lo, hi, stretch)
        return rng, clip_i, start, stretch

    def draw(self, i: int) -> tuple[int, float, float]:
        """Which clip, where, and at what stretch window *i* is cut.

        Exposed because the draw is the dataset's whole contract with training:
        a test pins the first draws of a known run, so a refactor that shifted
        them would silently change what every seed trains on.
        """
        _, clip_i, start, stretch = self._draw(i)
        return clip_i, start, stretch

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        window = self.window
        rng, clip_i, start, stretch = self._draw(i)
        entry = self.usable[clip_i]
        target_times, signal_all, onset_times = self._target(entry)
        frame_times = self._frame_times(entry)

        positions = start + np.arange(self.n_in) * stretch
        if rng is not None and self.augment.select_jitter > 0:
            positions = jitter_positions(positions, rng, self.augment.select_jitter)
        positions = np.clip(positions, 0.0, entry.n_frames - 1)

        first = int(np.floor(positions[0]))
        last = int(np.ceil(positions[-1]))
        slab = self.channels.take(np.asarray(self._array(entry)[first : last + 1]))
        block = gather_positions(slab, positions - first)

        in_times = gather_positions(frame_times, positions, dtype=np.float64)
        origin = float(in_times[self.margin])
        out_times = origin + np.arange(window) * (stretch / self.output_fs)

        signal = np.interp(out_times, target_times, signal_all).astype(np.float32)
        heatmap = onset_heatmap(out_times, onset_times, sigma_s=self.onset_sigma_s)

        if rng is not None and self.augment.enabled:
            if self.augment.scale_motion:
                block = scale_motion_channels(
                    block, self.motion_scale(stretch), self.channels
                )
            block = apply_spatial(block, rng, self.augment, self.channels)

        features = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32))
        features = (features - self.mean) / self.std

        return {
            "features": features,  # (T_in, C, H, W) float32
            "t_in": torch.from_numpy(
                (in_times - origin).astype(np.float32)
            ),  # window-relative
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

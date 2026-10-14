"""The on-disk feature cache: one entry per (video, preprocessing, box).

Files share a prefix ``{video stem}-{hash}`` (video, timestamps, recipe, box; see
:func:`cache_key`). Per entry: the uint8 channels, one file per group (``gray-``,
``diff-``, ``flow-``; T x n x h x w), only the groups it was asked for, then
``ftime-``, ``dt-``, ``time-``, ``target-`` and ``events-``, and a ``{prefix}.json`` sidecar (key, stats, thermistor, events, groups),
written last so its presence means the arrays are complete. Read channels through
:func:`reader`, which opens only the groups a model uses.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .channels import CHANNEL_GROUPS, ChannelSet
from .config import PreprocessParams, ResolvedClip, canonical, dump

GROUPS = ("gray", "diff", "flow")
"""How channels are stored: one file per group, in stored channel order."""


def groups_of(channels: ChannelSet) -> tuple[str, ...]:
    """The stored groups holding *channels*."""
    return tuple(g for g in GROUPS if set(CHANNEL_GROUPS[g]) & set(channels.names))


@dataclass(frozen=True)
class ClipEntry:
    """One preprocessed clip: where its arrays live and what it is.

    *n_frames* is on the selection grid (channels/``frame_times``);
    *n_output* on the output grid (``times``/``target``).  *clip_id* is a
    human label (folder and file stem); *recording* is the leakage identity,
    see :attr:`zephyr.config.ResolvedClip.recording_id`.  *thermistor* is the raw
    ground-truth trace scoring uses, ``None`` for an unlabelled clip.
    """

    clip_id: str
    recording: str
    n_frames: int
    n_output: int
    stem: Path
    """Cache folder / prefix: the entry's identity (keys caches and statistics), and
    what its channel files are named from (:meth:`group_file`)."""
    frame_times: Path
    baselines: Path
    times: Path
    target: Path | None
    events: Path | None
    thermistor: Path | None = None
    video: Path | None = None
    groups: tuple[str, ...] = GROUPS
    """Channel groups stored."""

    @property
    def has_target(self) -> bool:
        return self.target is not None

    @property
    def stored(self) -> ChannelSet:
        """The channels this entry holds."""
        return ChannelSet.parse([n for g in self.groups for n in CHANNEL_GROUPS[g]])

    def group_file(self, group: str) -> Path:
        return self.stem.with_name(f"{group}-{self.stem.name}.npy")


def cache_key(clip: ResolvedClip, params: PreprocessParams) -> dict:
    """Everything that determines a clip's arrays."""
    if clip.video is None or clip.timestamps is None:
        raise ValueError(f"{clip.name} is thermistor only; features need its video")
    if clip.box is None:
        raise ValueError(f"{clip.name} has no box; place it with `zephyr annotate`")
    recipe = {
        k: v
        for k, v in dump(params).items()
        if params.KEY_DEFAULTS.get(k, object()) != v
    }
    return {
        "video": canonical(clip.video),
        "timestamps": canonical(clip.timestamps),
        "preprocess": recipe,
        "box": list(clip.box),
    }


def events_identity(clip: ResolvedClip) -> dict:
    """What a labelled clip's cached events depend on; recorded in the sidecar.

    A cache made before event sets existed records none, so it is stale: its
    target is rebuilt (the video arrays are kept), with events at the
    thermistor's own resolution instead of the output grid's.
    """
    return clip.events.identity(clip)


def prefix(clip: ResolvedClip, params: PreprocessParams) -> str:
    blob = json.dumps(cache_key(clip, params), sort_keys=True).encode()
    return f"{clip.source.stem}-{hashlib.sha256(blob).hexdigest()[:10]}"


def paths(cache_dir: Path, name: str) -> dict[str, Path]:
    """The files of the entry with prefix *name*."""
    return {
        **{group: cache_dir / f"{group}-{name}.npy" for group in GROUPS},
        "frame_times": cache_dir / f"ftime-{name}.npy",
        "baselines": cache_dir / f"dt-{name}.npy",
        "times": cache_dir / f"time-{name}.npy",
        "target": cache_dir / f"target-{name}.parquet",
        "events": cache_dir / f"events-{name}.npz",
        "sidecar": cache_dir / f"{name}.json",
    }


def lookup(
    clip: ResolvedClip, params: PreprocessParams, cache_dir: Path
) -> ClipEntry | None:
    """The cached entry for *clip*, or ``None`` if it is missing or stale.

    "Stale" means the arrays are there but the clip is labelled with a
    different thermistor or event set than the target on disk was built from,
    or is labelled and has no target yet; the arrays are reusable, only the
    target is not.
    """
    name = prefix(clip, params)
    files = paths(cache_dir, name)
    if not files["sidecar"].is_file():
        return None
    sidecar = json.loads(files["sidecar"].read_text())
    if sidecar["key"] != cache_key(clip, params):
        return None
    wanted = canonical(clip.thermistor) if clip.thermistor else None
    if sidecar["thermistor"] != wanted:
        return None
    if wanted and sidecar.get("events") != events_identity(clip):
        return None
    return ClipEntry(
        clip_id=clip.name,
        recording=clip.recording_id,
        n_frames=sidecar["n_frames"],
        n_output=sidecar["n_output"],
        stem=cache_dir / name,
        frame_times=files["frame_times"],
        baselines=files["baselines"],
        times=files["times"],
        target=files["target"] if wanted else None,
        events=files["events"] if wanted else None,
        thermistor=clip.thermistor,
        video=clip.video,
        groups=tuple(sidecar["groups"]),
    )


def require(
    clips: list[ResolvedClip],
    params: PreprocessParams,
    cache_dir: Path,
    channels: ChannelSet | None = None,
) -> list[ClipEntry]:
    """Entries for *clips*, in order; fail naming what is not preprocessed yet, or
    (given *channels*) what lacks a channel group they need."""
    entries, missing, lacking = [], [], []
    need = set(groups_of(channels)) if channels is not None else set()
    for clip in clips:
        entry = lookup(clip, params, cache_dir)
        if entry is None:
            missing.append(clip.name)
        elif need - set(entry.groups):
            lacking.append(clip.name)
        else:
            entries.append(entry)
    if missing:
        raise SystemExit(
            f"{len(missing)} clip(s) are not in the feature cache {cache_dir}: "
            f"{missing}\nRun `zephyr preprocess <clip list> --cache {cache_dir}`."
        )
    if lacking:
        groups = "+".join(g for g in GROUPS if g in need)
        raise SystemExit(
            f"{len(lacking)} clip(s) in {cache_dir} lack channels {channels}: "
            f"{lacking}\nRun `zephyr preprocess <clip list> --cache {cache_dir} "
            f"--channels {groups}`."
        )
    return entries


class FeatureReader:
    """An entry's channels, opened lazily: only the group files a read needs.

    ``read(rows, channels)`` returns ``(n, len(channels), h, w)`` uint8 for *rows* (a
    slice or index array on the selection grid), channels in stored order.  Memmaps are
    cached on the reader, so make one per process (DataLoader workers are spawned).
    """

    def __init__(self, entry: ClipEntry) -> None:
        self.entry = entry
        self._maps: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return self.entry.n_frames

    def _map(self, group: str) -> np.ndarray:
        array = self._maps.get(group)
        if array is None:
            array = np.load(self.entry.group_file(group), mmap_mode="r")
            self._maps[group] = array
        return array

    def read(self, rows, channels: ChannelSet | None = None) -> np.ndarray:
        channels = channels or self.entry.stored
        lacking = sorted(set(channels.names) - set(self.entry.stored.names))
        if lacking:
            raise ValueError(f"{self.entry.clip_id} does not store {lacking}")
        parts = []
        for group in groups_of(channels):
            names = CHANNEL_GROUPS[group]
            block = np.asarray(self._map(group)[rows])
            keep = [i for i, n in enumerate(names) if n in channels.names]
            parts.append(block if len(keep) == len(names) else block[:, keep])
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)


def reader(entry: ClipEntry) -> FeatureReader:
    return FeatureReader(entry)

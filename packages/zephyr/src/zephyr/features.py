"""The on-disk feature cache: one entry per (video, preprocessing, box).

Files share a prefix ``{video stem}-{hash}`` (video, timestamps, recipe, box; see
:func:`cache_key`). Per entry: the uint8 gray frames (``gray-``, T x 1 x h x w),
then ``ftime-``, ``time-``, ``target-`` and ``events-``, and a ``{prefix}.json`` sidecar
(key, stats, thermistor, events), written last so its presence means the arrays are
complete. Read frames through :func:`reader`.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import EventsSource, PreprocessParams, ResolvedClip, canonical, dump


@dataclass(frozen=True)
class ClipEntry:
    """One preprocessed clip: where its arrays live and what it is.

    *n_frames* is on the selection grid (frames/``frame_times``);
    *n_output* on the output grid (``times``/``target``).  *clip_id* is a
    human label (folder and file stem); *recording* is the leakage identity,
    see :attr:`zephyr.config.ResolvedClip.recording_id`.  *thermistor* is the raw
    ground-truth trace scoring uses, ``None`` for an unlabelled clip.  *detector* is
    the clip's event detector, which scoring also runs on a predicted trace.
    """

    clip_id: str
    recording: str
    n_frames: int
    n_output: int
    stem: Path
    """Cache folder / prefix: the entry's identity (keys caches and statistics)."""
    frames: Path
    frame_times: Path
    times: Path
    target: Path | None
    events: Path | None
    thermistor: Path | None = None
    video: Path | None = None
    detector: EventsSource | None = None

    @property
    def has_target(self) -> bool:
        return self.target is not None


def cache_key(clip: ResolvedClip, params: PreprocessParams) -> dict:
    """Everything that determines a clip's arrays."""
    if clip.video is None or clip.timestamps is None:
        raise ValueError(f"{clip.name} is thermistor only; features need its video")
    if clip.box is None:
        raise ValueError(f"{clip.name} has no box; place it in `zephyr gui` (Boxes)")
    recipe = params.KEY_RETIRED | {
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
        "frames": cache_dir / f"gray-{name}.npy",
        "frame_times": cache_dir / f"ftime-{name}.npy",
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
    if not files["sidecar"].is_file() or not files["frames"].is_file():
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
        frames=files["frames"],
        frame_times=files["frame_times"],
        times=files["times"],
        target=files["target"] if wanted else None,
        events=files["events"] if wanted else None,
        thermistor=clip.thermistor,
        video=clip.video,
        detector=clip.events if wanted else None,
    )


def require(
    clips: list[ResolvedClip], params: PreprocessParams, cache_dir: Path
) -> list[ClipEntry]:
    """Entries for *clips*, in order; fail naming what is not preprocessed yet."""
    entries, missing = [], []
    for clip in clips:
        entry = lookup(clip, params, cache_dir)
        if entry is None:
            missing.append(clip.name)
        else:
            entries.append(entry)
    if missing:
        raise SystemExit(
            f"{len(missing)} clip(s) are not in the feature cache {cache_dir}: "
            f"{missing}\nRun `zephyr preprocess <clip list> --cache {cache_dir}`."
        )
    return entries


class FeatureReader:
    """An entry's frames, opened lazily.

    ``read(rows)`` returns ``(n, 1, h, w)`` uint8 for *rows* (a slice or index array on
    the selection grid).  The memmap is cached on the reader, so make one per process
    (DataLoader workers are spawned).
    """

    def __init__(self, entry: ClipEntry) -> None:
        self.entry = entry
        self._map: np.ndarray | None = None

    def __len__(self) -> int:
        return self.entry.n_frames

    def read(self, rows) -> np.ndarray:
        if self._map is None:
            self._map = np.load(self.entry.frames, mmap_mode="r")
        return np.asarray(self._map[rows])


def reader(entry: ClipEntry) -> FeatureReader:
    return FeatureReader(entry)

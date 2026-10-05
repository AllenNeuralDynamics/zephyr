"""The on-disk feature cache: one entry per (video, preprocessing, box).

Files share a prefix ``{video stem}-{hash}``; the hash covers the video and timestamp
paths, every PreprocessParams field and the box, so changing any gives a new entry and
nothing is overwritten with different content. Per entry: ``feat-`` (uint8 T x 4 x h x
w), ``ftime-``, ``dt-``, ``time-`` (60 Hz grid), ``target-`` and ``events-`` for
labelled clips, and a ``{prefix}.json`` sidecar (full key, stats, thermistor used),
written last so its presence means the arrays are complete.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .config import PreprocessParams, ResolvedClip, dump


@dataclass(frozen=True)
class ClipEntry:
    """One preprocessed clip: where its arrays live and what it is.

    *n_frames* is on the selection grid (``features``/``frame_times``);
    *n_output* on the 60 Hz output grid (``times``/``target``).  *clip_id* is a
    human label (folder and file stem); *recording* is the leakage identity,
    see :attr:`zephyr.config.ResolvedClip.recording_id`.  *thermistor* is the raw
    ground-truth trace scoring uses, ``None`` for an unlabelled clip.
    """

    clip_id: str
    recording: str
    n_frames: int
    n_output: int
    features: Path
    frame_times: Path
    baselines: Path
    times: Path
    target: Path | None
    events: Path | None
    thermistor: Path | None = None
    video: Path | None = None

    @property
    def has_target(self) -> bool:
        return self.target is not None


def cache_key(clip: ResolvedClip, params: PreprocessParams) -> dict:
    """Everything that determines a clip's arrays."""
    if clip.box is None:
        raise ValueError(f"{clip.name} has no box; place it with `zephyr annotate`")
    return {
        "video": clip.video.as_posix(),
        "timestamps": clip.timestamps.as_posix(),
        "preprocess": dump(params),
        "box": list(clip.box),
    }


def prefix(clip: ResolvedClip, params: PreprocessParams) -> str:
    blob = json.dumps(cache_key(clip, params), sort_keys=True).encode()
    return f"{clip.video.stem}-{hashlib.sha256(blob).hexdigest()[:10]}"


def paths(cache_dir: Path, name: str) -> dict[str, Path]:
    """The files of the entry with prefix *name*."""
    return {
        "features": cache_dir / f"feat-{name}.npy",
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
    different thermistor than the target on disk was built from, or is labelled
    and has no target yet; the arrays are reusable, only the target is not.
    """
    name = prefix(clip, params)
    files = paths(cache_dir, name)
    if not files["sidecar"].is_file():
        return None
    sidecar = json.loads(files["sidecar"].read_text())
    if sidecar["key"] != cache_key(clip, params):
        return None
    wanted = clip.thermistor.as_posix() if clip.thermistor else None
    if sidecar["thermistor"] != wanted:
        return None
    return ClipEntry(
        clip_id=clip.name,
        recording=clip.recording_id,
        n_frames=sidecar["n_frames"],
        n_output=sidecar["n_output"],
        features=files["features"],
        frame_times=files["frame_times"],
        baselines=files["baselines"],
        times=files["times"],
        target=files["target"] if wanted else None,
        events=files["events"] if wanted else None,
        thermistor=clip.thermistor,
        video=clip.video,
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

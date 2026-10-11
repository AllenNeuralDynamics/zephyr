"""``select``: the final rule, written with everything else to ``results.json``.

The clip list is the source of truth: a session passes when its mark is ``include``,
and its clips come from its ``included`` spans (the QC's, or a reviewer's). Each
passing session draws ``clips_per_session`` non-overlapping clips uniformly over its
included time (seeded by the spec's seed and its session_id), labelled ``span1``,
``span2``, ... in time order (thermistor seconds); with ``require_camera`` only
sessions whose camera folder holds ``video.mp4`` and ``metadata.csv`` are selected.
"""

import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd

from zephyr.config import ClipList, load

from .config import DatasetSpec, Layout, session_of
from .qc import METRICS
from .qc import metadata as qc_metadata
from .remote import filesystem, find, thread_map

SCHEMA = 1
TRIES = 200


def starts(spans, clip_s: float) -> list[tuple[float, float]]:
    """Intervals of valid clip starts inside the spans."""
    return [(a, b - clip_s) for a, b in spans if b - a >= clip_s]


def cut(intervals, lo: float, hi: float) -> list[tuple[float, float]]:
    """*intervals* with ``(lo, hi)`` taken out."""
    out = []
    for a, b in intervals:
        if b <= lo or a >= hi:
            out.append((a, b))
            continue
        if a < lo:
            out.append((a, lo))
        if hi < b:
            out.append((hi, b))
    return out


def draw(intervals, rng) -> float:
    """A point uniform over the intervals' total length."""
    lengths = np.array([b - a for a, b in intervals])
    if lengths.sum() <= 0:  # only point intervals left: pick one exactly
        return float(intervals[rng.integers(len(intervals))][0])
    k = rng.choice(len(intervals), p=lengths / lengths.sum())
    a, b = intervals[k]
    return float(rng.uniform(a, b))


def session_seed(seed: int, session_id: str) -> int:
    return int(hashlib.sha256(f"{seed}:{session_id}".encode()).hexdigest()[:8], 16)


def draw_clips(
    session_id: str, spans, clip_s: float, n: int, seed: int
) -> dict[str, dict[str, float]] | None:
    """*n* non-overlapping clips inside *spans*, or None when they do not fit."""
    rng = np.random.default_rng(session_seed(seed, session_id))
    first = starts(spans, clip_s)
    if not first:
        return None
    for _ in range(TRIES):
        chosen, free = [], first
        for _ in range(n):
            if not free:
                break
            start = draw(free, rng)
            chosen.append(start)
            free = cut(free, start - clip_s, start + clip_s)
        if len(chosen) == n:
            return {
                f"span{k}": {"start": s, "end": s + clip_s}
                for k, s in enumerate(sorted(chosen), start=1)
            }
    return None


def camera_folder(fs, spec: DatasetSpec, source: str) -> str | None:
    """``s3://`` path of the session's camera folder when it holds a video."""
    path = find(fs, spec.source.bucket, source, f"behavior-videos/{spec.source.camera}")
    if path is None:
        return None
    names = {n.rsplit("/", 1)[-1] for n in fs.ls(path, detail=False)}
    if {"video.mp4", "metadata.csv"} <= names:
        return f"s3://{path}/"
    return None


def _clean(value):
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _package_version() -> str | None:
    try:
        return version("zephyr-dataset")
    except PackageNotFoundError:
        return None


def run(spec: DatasetSpec, workers: int = 32, fs=None) -> dict:
    layout = Layout(spec.root)
    rule = spec.select
    clip_list = load(ClipList, layout.clips)
    qc = pd.read_parquet(layout.qc).set_index("session_id")
    inventory = pd.read_parquet(layout.inventory).set_index("session_id")
    sessions = []
    for clip in clip_list.clip:
        sid = session_of(clip.thermistor)
        meta = inventory.loc[sid] if sid in inventory.index else None
        scored = qc.loc[sid] if sid in qc.index else None
        record = {
            "session_id": sid,
            "subject_id": None if meta is None else meta["subject_id"],
            "rig": None if meta is None else meta["rig"],
            "date": None if meta is None else meta["date"],
            "dataset_version": None if meta is None else meta["dataset_version"],
            "asset": None if meta is None else meta["location"],
            "source_session": None
            if meta is None
            else str(meta["source_sessions"]).split(";")[0],
            "selection": clip.selection,
            "reviewed": clip.reviewed,
            "suggested": None if scored is None else scored["suggested"],
            "reason": None if scored is None else (scored["reason"] or None),
            "metrics": {}
            if scored is None
            else {k: _clean(scored[k]) for k in METRICS},
            "included": [[float(a), float(b)] for a, b in clip.included],
            "excluded": [[float(a), float(b)] for a, b in clip.excluded],
            "clips": None,
            "camera": None,
        }
        if clip.selection == "include":
            record["clips"] = draw_clips(
                sid, clip.included, rule.clip_s, rule.clips_per_session, rule.seed
            )
        sessions.append(record)
    passed = [s for s in sessions if s["selection"] == "include"]
    fits = [s for s in passed if s["clips"]]
    if rule.require_camera:
        fs = fs or filesystem()
        found = thread_map(
            lambda s: camera_folder(fs, spec, s["source_session"] or s["session_id"]),
            fits,
            workers,
        )
        for s, path in zip(fits, found):
            s["camera"] = path
        selected = [s for s in fits if s["camera"]]
    else:
        selected = fits
    chosen = {s["session_id"] for s in selected}
    for s in sessions:
        if s["session_id"] not in chosen:
            s["clips"] = None
    results = {
        "schema": SCHEMA,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "zephyr_dataset": _package_version(),
        "signal": f"{spec.source.device}.harp {spec.source.signal} (raw)",
        "inventory_queried": str(inventory["queried"].iloc[0])
        if "queried" in inventory
        else None,
        "spec": spec.model_dump(mode="json", exclude={"root"}),
        "qc": qc_metadata(layout.qc),
        "counts": {
            "sessions": len(sessions),
            "passed": len(passed),
            "clips_fit": len(fits),
            "selected": len(selected),
        },
        "passed": [s["session_id"] for s in passed],
        "selected": [s["session_id"] for s in selected],
        "sessions": sessions,
    }
    layout.results.write_text(json.dumps(results, indent=1, default=_clean))
    print(
        f"select: {len(passed)} passed, {len(fits)} fit {rule.clips_per_session} "
        f"clips of {rule.clip_s:g} s, {len(selected)} selected"
    )
    return results

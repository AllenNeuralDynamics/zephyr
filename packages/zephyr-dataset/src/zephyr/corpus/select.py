"""``select``: the final rule, written with everything else to ``results.json``.

The clip list is the source of truth: a session passes when its mark is ``include``,
and its clips come from its ``included`` spans (the QC's, or a reviewer's). Each
passing session draws ``clips_per_session`` non-overlapping clips uniformly over its
included time (seeded by the spec's seed and its session_id), labelled ``span1``,
``span2``, ... in time order (thermistor seconds); with ``require_cameras`` only
sessions with every camera of the spec are kept, and (once ``videos`` has run) only
those whose cameras are all clean: the others get ``dropped`` with the reason.
"""

import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd
import tomlkit

from . import camera, storage
from .config import DatasetSpec, Layout, session_of
from .lists import validate
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


def camera_folders(fs, spec: DatasetSpec, source: str) -> dict[str, str]:
    """``s3://`` paths of the session's camera folders that hold a video and its
    ``metadata.csv``, by camera name (cameras without them are left out)."""
    root = find(fs, spec.source.bucket, source, "behavior-videos")
    if root is None:
        return {}
    found = {}
    for name in spec.source.cameras:
        path = f"{root}/{name}"
        if not fs.exists(path):
            continue
        names = {n.rsplit("/", 1)[-1] for n in fs.ls(path, detail=False)}
        if {"video.mp4", "metadata.csv"} <= names:
            found[name] = f"s3://{path.lstrip('/')}/"
    return found


def camera_problems(layout: Layout) -> dict[str, list[str]]:
    """Sessions with a camera whose timestamps are not clean after the repair, and
    why ("<camera>: <problem>"); none before ``videos`` has run."""
    if not layout.cameras.exists():
        return {}
    table = storage.read_parquet(layout.cameras)
    out: dict[str, list[str]] = {}
    for _, row in table.iterrows():
        if row["error"] is not None and not pd.isna(row["error"]):
            continue  # not checked yet: retried by the next `videos`
        if not bool(row["clean"]):
            found = camera.problems(row) or ["not clean"]
            out.setdefault(row["session_id"], []).extend(
                f"{row['camera']}: {p}" for p in found
            )
    return out


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
    text = layout.session_list.read_text(encoding="utf-8")
    entries = validate(tomlkit.parse(text), layout)
    qc = storage.read_parquet(layout.qc).set_index("session_id")
    inventory = storage.read_parquet(layout.inventory).set_index("session_id")
    sessions = []
    for clip in entries:
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
            "cameras": {},
        }
        if clip.selection == "include":
            record["clips"] = draw_clips(
                sid, clip.included, rule.clip_s, rule.clips_per_session, rule.seed
            )
        sessions.append(record)
    passed = [s for s in sessions if s["selection"] == "include"]
    fits = [s for s in passed if s["clips"]]
    if rule.require_cameras:
        fs = fs or filesystem()
        found = thread_map(
            lambda s: camera_folders(fs, spec, s["source_session"] or s["session_id"]),
            fits,
            workers,
        )
        for s, paths in zip(fits, found):
            s["cameras"] = paths
        wanted = set(spec.source.cameras)
        selected = [s for s in fits if wanted <= set(s["cameras"])]
    else:
        selected = fits
    dropped = camera_problems(layout)  # from `videos`, once it has run
    for s in selected:
        if s["session_id"] in dropped:
            s["dropped"] = dropped[s["session_id"]]
    selected = [s for s in selected if s["session_id"] not in dropped]
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
            "dropped_for_cameras": sum("dropped" in s for s in sessions),
            "selected": len(selected),
        },
        "passed": [s["session_id"] for s in passed],
        "selected": [s["session_id"] for s in selected],
        "sessions": sessions,
    }
    storage.write_text(layout.results, json.dumps(results, indent=1, default=_clean))
    print(
        f"select: {len(passed)} passed, {len(fits)} fit {rule.clips_per_session} "
        f"clips of {rule.clip_s:g} s, {results['counts']['dropped_for_cameras']} "
        f"dropped for their cameras, {len(selected)} selected"
    )
    return results

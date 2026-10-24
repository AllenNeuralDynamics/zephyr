"""``videos``: each selected session's camera timestamps, checked and repaired.

For every camera of the spec (an optional one where a session ``folds`` assigns has it,
beside its other cameras), caches ``metadata.csv`` unmodified in
``camera_metadata/<session_id>_<camera>.parquet`` (with the video's frame count from
ffprobe), measures it (:func:`camera.measure`), repairs ``ReferenceTime`` rollbacks
(:func:`camera.fix_rollbacks`), measures again and writes the repaired timestamps to
``videos/<session_id>_<camera>.parquet``; ``cameras.parquet`` keeps both measurements.
Resumable, and a cached camera is re-repaired without touching S3.
"""

import io
import json

import pandas as pd
import pyarrow as pa

from . import camera, storage
from .config import DatasetSpec, Layout
from .remote import camera_folder, filesystem, thread_map

COLUMNS = ["ReferenceTime", "CameraFrameNumber", "CameraFrameTime"]
BATCH = 200


def https(s3_folder: str) -> str:
    """The public HTTPS URL of an ``s3://bucket/key/`` folder."""
    bucket, _, key = s3_folder.removeprefix("s3://").partition("/")
    return f"https://{bucket}.s3.amazonaws.com/{key.rstrip('/')}"


def one(fs, layout: Layout, session_id: str, name: str, folder: str, frames) -> dict:
    """Check, repair and write one camera; returns its row of ``cameras.parquet``."""
    row: dict = {"session_id": session_id, "camera": name, "folder": folder}
    row["error"] = None
    try:
        raw, row["video_frames"] = cached(fs, layout, session_id, name, folder, frames)
        before = camera.measure(raw)
        fixed, row["repaired_rows"] = camera.fix_rollbacks(raw)
        after = camera.measure(fixed)
    except Exception as exc:  # noqa: BLE001 -- recorded; retried on the next run
        return row | {"error": f"{type(exc).__name__}: {exc}"[:300]}
    row |= {f"before_{k}": v for k, v in before.items()}
    row |= {f"after_{k}": v for k, v in after.items()}
    row["clean"] = camera.clean(after, row["video_frames"])
    write(fixed[COLUMNS], layout.timestamps(session_id, name))
    return row


def cached(fs, layout: Layout, session_id: str, name: str, folder: str, frames):
    """The camera's ``metadata.csv`` rows and its video's frame count, from the cache,
    else from S3 (then cached)."""
    path = layout.camera_metadata(session_id, name)
    if path.exists():
        meta = storage.schema_metadata(path)
        if b"video_frames" in meta:
            return storage.read_parquet(path), int(meta[b"video_frames"])
    source = f"{folder.removeprefix('s3://').rstrip('/')}/metadata.csv"
    raw = pd.read_csv(io.BytesIO(fs.cat(source)), usecols=COLUMNS)  # one request
    count = frames(f"{https(folder)}/video.mp4")
    write(raw, path, {"video_frames": str(count), "source": f"s3://{source}"})
    return raw, count


def write(frame: pd.DataFrame, path, metadata: dict | None = None) -> None:
    """Lossless and about 7x smaller than the defaults: the clocks step regularly, so
    delta-encoded counters and byte-split floats compress well."""
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if metadata:
        table = table.replace_schema_metadata(
            {
                **(table.schema.metadata or {}),
                **{k.encode(): v.encode() for k, v in metadata.items()},
            }
        )
    storage.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        column_encoding={
            "ReferenceTime": "BYTE_STREAM_SPLIT",
            "CameraFrameNumber": "DELTA_BINARY_PACKED",
            "CameraFrameTime": "DELTA_BINARY_PACKED",
        },
    )


def run(
    spec: DatasetSpec,
    workers: int = 16,
    fs=None,
    frames=None,
    limit: int | None = None,
) -> pd.DataFrame:
    fs = fs or filesystem()
    frames = frames or camera.video_frames
    layout = Layout(spec.root)
    results = json.loads(layout.selection.read_text(encoding="utf-8"))
    by_id = {s["session_id"]: s for s in results["sessions"]}
    done = storage.read_parquet(layout.cameras) if layout.cameras.exists() else None
    finished = set()
    if done is not None:
        ok = done[done["error"].isna()]
        written = storage.names(layout.video_dir, "*.parquet")
        finished = {
            (s, c)
            for s, c in zip(ok["session_id"], ok["camera"])
            if layout.timestamps(s, c).name in written
        }
    todo = [
        (sid, name, by_id[sid]["cameras"][name])
        for sid in results["selected"][:limit]
        for name in spec.source.cameras
        if (sid, name) not in finished and name in by_id[sid]["cameras"]
    ]
    todo += optional_cameras(
        spec, layout, fs, by_id, results["selected"], finished, limit, workers
    )
    table = done if done is not None else pd.DataFrame()
    for start in range(0, len(todo), BATCH):  # saved per batch: a stop loses little
        batch = todo[start : start + BATCH]
        rows = pd.DataFrame(
            thread_map(lambda t: one(fs, layout, *t, frames), batch, workers)
        )
        table = merge(table, rows)
        storage.write_frame(table, layout.cameras)
        print(f"videos: {start + len(batch)}/{len(todo)}", flush=True)
    report(table)
    return table


def optional_cameras(spec, layout, fs, by_id, selected, finished, limit, workers):
    """``(session, camera, folder)`` of the optional cameras still to check: in the
    sessions ``folds`` assigns (all selected before it has run), where the folder is
    beside the session's other cameras; the sessions without one are counted."""
    if not spec.source.optional_cameras:
        return []
    sessions = selected
    if layout.folds.exists():
        folds = storage.read_parquet(layout.folds, columns=["session_id", "set"])
        assigned = set(folds.loc[folds["set"].notna(), "session_id"])
        sessions = [sid for sid in selected if sid in assigned]
    wanted = [
        (sid, name)
        for sid in sessions[:limit]
        for name in spec.source.optional_cameras
        if (sid, name) not in finished and by_id[sid]["cameras"]
    ]

    def folder(item):
        sid, name = item
        known = next(iter(by_id[sid]["cameras"].values()))
        return camera_folder(fs, known.rstrip("/").rsplit("/", 1)[0], name)

    found = thread_map(folder, wanted, workers)
    for name in spec.source.optional_cameras:
        absent = sum(f is None for (_, n), f in zip(wanted, found) if n == name)
        if absent:
            print(f"videos: {absent} session(s) have no {name}", flush=True)
    return [(sid, name, f) for (sid, name), f in zip(wanted, found) if f is not None]


def merge(table: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """*table* with *rows* replacing any earlier row of the same camera."""
    if table.empty:
        merged = rows
    else:
        key = ["session_id", "camera"]
        again = table.set_index(key).index.isin(rows.set_index(key).index)
        merged = pd.concat([table[~again], rows], ignore_index=True)
    return merged.sort_values(["session_id", "camera"]).reset_index(drop=True)


def report(table: pd.DataFrame) -> None:
    if table.empty:
        print("videos: nothing to do")
        return
    failed = table["error"].notna()
    clean = table["clean"].fillna(False).astype(bool)
    repaired = table["repaired_rows"].fillna(0) > 0
    print(
        f"videos: {len(table)} cameras, {int(clean.sum())} clean "
        f"({int((clean & repaired).sum())} after repair), "
        f"{int((~clean & ~failed).sum())} not clean, {int(failed.sum())} failed"
    )

"""``sync``: whether each selected session's thermistor device keeps the reference clock.

Harp devices on one clock answer the read dump that starts a session (the WRITE to
``OperationControl``) within a few ms of each other; a device off the clock timestamps
everything it logs off by as much. Reads only that register's ``.bin`` of the two
devices from the raw session and writes their times to ``sync.parquet``; ``folds``
takes the sessions over ``[sync] max_offset_s`` out after the draw. Resumable.
"""

import json

import pandas as pd

from . import harp, storage
from .config import DatasetSpec, Layout
from .remote import filesystem, find, thread_map


def dump_time(fs, folder: str, device: str) -> float:
    """The time of *device*'s read dump: its last WRITE to ``OperationControl``."""
    name = f"{device}_{harp.OPERATION_CONTROL}.bin"
    times = harp.write_times(fs.cat(f"{folder}/{device}.harp/{name}"))
    if not len(times):
        raise ValueError(f"no read dump in {device}.harp/{name}")
    return float(times[-1])


def one(fs, spec: DatasetSpec, session_id: str, source: str) -> dict:
    row = {"session_id": session_id, "error": None}
    try:
        folder = find(fs, spec.source.bucket, source, "behavior")
        if folder is None:
            raise FileNotFoundError("no behavior folder")
        device = dump_time(fs, folder, spec.source.device)
        reference = dump_time(fs, folder, spec.sync.reference)
    except Exception as exc:  # noqa: BLE001 -- recorded; retried on the next run
        return row | {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return row | {
        "device_s": device,
        "reference_s": reference,
        "offset_s": device - reference,
    }


def out_of_sync(spec: DatasetSpec, layout: Layout, sessions: list[str]) -> dict:
    """Each of *sessions* whose clock is off (or could not be checked): the reason."""
    if not layout.sync.exists():
        raise SystemExit("sync: no sync.parquet, run sync first")
    table = storage.read_parquet(layout.sync).set_index("session_id")
    missing = sorted(set(sessions) - set(table.index))
    if missing:
        raise SystemExit(f"sync: {len(missing)} sessions not checked, run sync first")
    table = table.loc[sessions]
    off = {
        s: f"read dump {1000 * v:+.0f} ms from {spec.sync.reference}"
        for s, v in table["offset_s"].items()
        if abs(v) > spec.sync.max_offset_s
    }
    return off | {s: e for s, e in table["error"].items() if isinstance(e, str)}


def run(spec: DatasetSpec, workers: int = 32, fs=None, limit: int | None = None):
    if spec.sync is None:
        raise SystemExit("sync: the spec has no [sync]")
    fs = fs or filesystem()
    layout = Layout(spec.root)
    results = json.loads(layout.results.read_text(encoding="utf-8"))
    by_id = {s["session_id"]: s for s in results["sessions"]}
    done = storage.read_parquet(layout.sync) if layout.sync.exists() else pd.DataFrame()
    have = set(done.loc[done["error"].isna(), "session_id"]) if len(done) else set()
    todo = [s for s in results["selected"][:limit] if s not in have]
    rows = thread_map(
        lambda s: one(fs, spec, s, by_id[s]["source_session"] or s), todo, workers
    )
    table = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    table = table.drop_duplicates("session_id", keep="last").sort_values("session_id")
    storage.write_frame(table.reset_index(drop=True), layout.sync)
    off = out_of_sync(spec, layout, results["selected"][:limit])
    print(f"sync: {len(table)} sessions checked, {len(off)} off the clock or unchecked")
    for session, reason in off.items():
        print(f"  {session}: {reason}")
    return table

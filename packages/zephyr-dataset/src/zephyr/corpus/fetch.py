"""``fetch``: each wanted session's raw thermistor, straight from its Harp files on S3.

Reads ``device.yml`` and the register's ``.bin`` files of
``<raw session>/behavior/<device>.harp`` (nothing else is downloaded) and writes the
whole session to ``thermistor/<session_id>.parquet`` (``Time``, ``Signal``).
Resumable: sessions already present are skipped.
"""

import pandas as pd

from . import harp
from .config import DatasetSpec, Layout
from .inventory import wanted
from .remote import filesystem, find, thread_map


def first_source(row) -> str:
    return str(row.source_sessions).split(";")[0]


def read_thermistor(fs, spec: DatasetSpec, source: str) -> pd.DataFrame | None:
    """The register's messages as ``Time``, ``Signal``; None without raw data."""
    src = spec.source
    folder = find(fs, src.bucket, source, f"behavior/{src.device}.harp")
    if folder is None:
        return None
    addr = harp.address(fs.cat(f"{folder}/device.yml"), src.signal)
    names = harp.chunks(fs.ls(folder, detail=False), src.device, addr)
    if not names:
        return None
    frame = harp.read_messages(b"".join(fs.cat(n) for n in names))
    frame = frame.rename(columns={"value": "Signal"}).sort_values("Time", kind="stable")
    return frame.reset_index(drop=True)


def fetch_one(fs, spec: DatasetSpec, layout: Layout, row) -> str:
    target = layout.thermistor(row.session_id)
    if target.exists():
        return "have"
    try:
        frame = read_thermistor(fs, spec, first_source(row))
    except Exception as exc:  # noqa: BLE001 -- reported; retried on the next run
        return type(exc).__name__
    if frame is None or frame.empty:
        return "no raw data"
    tmp = target.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(target)
    return "ok"


def run(spec: DatasetSpec, workers: int = 16, fs=None, limit: int | None = None):
    fs = fs or filesystem()
    layout = Layout(spec.root)
    layout.thermistor_dir.mkdir(parents=True, exist_ok=True)
    table = wanted(spec, pd.read_parquet(layout.inventory))
    if limit:
        table = table.head(limit)
    status = thread_map(
        lambda row: fetch_one(fs, spec, layout, row), list(table.itertuples()), workers
    )
    counts = pd.Series(status, dtype=object).value_counts().to_dict()
    print(f"fetch: {len(status)} sessions: {counts}")
    return dict(zip(table["session_id"], status))

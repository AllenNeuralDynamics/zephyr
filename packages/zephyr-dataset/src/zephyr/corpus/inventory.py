"""``inventory``: every packaged session of the source process, from DocDB.

Derived assets with the spec's packaging process, created by ``query.until``; per raw
source session the highest packaging version (then the newest) wins, and an asset must
claim every source it lists. Each asset's ``session.parquet`` gives session_id, subject,
date, rig and dataset version. The DocDB return is saved (``docdb.json.gz``) and reused
unless refreshed, so a rebuild sees the same metadata; writes ``inventory.parquet``.
"""

import gzip
import json
import urllib.request
from datetime import UTC, datetime

import pandas as pd
import pyarrow.parquet as pq
from packaging.version import InvalidVersion, Version

from . import storage
from .config import DatasetSpec, Layout
from .remote import filesystem, thread_map

PAGE = 500
COLUMNS = [
    "session_id",
    "subject_id",
    "date",
    "rig",
    "curriculum_stage_name",
    "dataset_version",
]


def aggregate(url: str, pipeline: list[dict]) -> list[dict]:
    """Run a read-only aggregation on DocDB's REST API (no credentials needed)."""
    request = urllib.request.Request(
        f"{url}/aggregate",
        data=json.dumps(pipeline).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def query_docdb(url: str, process: str) -> list[dict]:
    """The raw DocDB records of every derived asset of *process* (see :func:`asset_of`)."""
    match = {
        "data_description.data_level": "derived",
        "processing.data_processes": {
            "$elemMatch": {
                "name": process,
                "output_parameters.packaging_version": {"$exists": True},
            }
        },
    }
    project = {
        "name": 1,
        "location": 1,
        "data_description.creation_time": 1,
        "data_description.source_data": 1,
        "processing.data_processes.name": 1,
        "processing.data_processes.output_parameters.packaging_version": 1,
    }
    found: list[dict] = []
    while True:  # pages: the API refuses large responses
        page = aggregate(
            url,
            [
                {"$match": match},
                {"$sort": {"_id": 1}},
                {"$skip": len(found)},
                {"$limit": PAGE},
                {"$project": project},
            ],
        )
        found += page
        if len(page) < PAGE:
            return found


def records(spec: DatasetSpec, refresh: bool = False) -> list[dict]:
    """The saved DocDB return (``docdb.json.gz``), or a new query saved there when
    there is none or *refresh*: re-runs read the same metadata."""
    layout = Layout(spec.root)
    if layout.docdb.exists() and not refresh:
        return snapshot(layout)["records"]
    found = query_docdb(spec.source.docdb, spec.source.process)
    saved = {
        "queried": datetime.now(UTC).isoformat(timespec="seconds"),
        "url": spec.source.docdb,
        "process": spec.source.process,
        "records": found,
    }
    storage.write_bytes(layout.docdb, gzip.compress(json.dumps(saved).encode()))
    return found


def snapshot(layout: Layout) -> dict:
    return json.loads(gzip.decompress(layout.docdb.read_bytes()))


def queried(spec: DatasetSpec) -> str | None:
    """When the saved DocDB return was queried."""
    layout = Layout(spec.root)
    if not layout.docdb.exists():
        return None
    return snapshot(layout).get("queried")


def _utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def created_by(assets: list[dict], until: datetime | None) -> list[dict]:
    """The assets created at or before *until* (all when None; an asset without a
    readable creation time is dropped)."""
    if until is None:
        return assets
    kept = []
    for asset in assets:
        try:
            created = _utc(datetime.fromisoformat(asset["created"]))
        except ValueError:
            continue
        if created <= _utc(until):
            kept.append(asset)
    return kept


def asset_of(record: dict, process: str) -> dict | None:
    processes = (record.get("processing") or {}).get("data_processes") or []
    version = next(
        (
            (p.get("output_parameters") or {}).get("packaging_version")
            for p in processes
            if p.get("name") == process
        ),
        None,
    )
    if version is None:
        return None
    description = record.get("data_description") or {}
    source = description.get("source_data") or []
    return {
        "asset": record["name"],
        "location": record["location"],
        "version": str(version),
        "created": str(description.get("creation_time")),
        "source": [source] if isinstance(source, str) else list(source),
    }


def _version(text: str) -> Version:
    try:
        return Version(text)
    except InvalidVersion:
        return Version("0")


def latest_per_source(assets: list[dict]) -> list[dict]:
    """The highest packaging version (then newest) asset of each source session."""
    ordered = sorted(
        assets, key=lambda a: (_version(a["version"]), a["created"]), reverse=True
    )
    chosen, claimed = [], set()
    for asset in ordered:
        if set(asset["source"]) & claimed:
            continue
        chosen.append(asset)
        claimed.update(asset["source"])
    return chosen


def read_session(fs, asset: dict) -> dict:
    """The asset joined with its ``session.parquet`` row (or an ``error``)."""
    row = {k: v for k, v in asset.items() if k != "source"}
    row["source_sessions"] = ";".join(asset["source"])
    try:
        path = asset["location"].removeprefix("s3://") + "/session.parquet"
        session = pq.read_table(path, columns=COLUMNS, filesystem=fs).to_pylist()[0]
    except Exception as exc:  # noqa: BLE001 -- recorded, the asset is skipped later
        return row | {"error": type(exc).__name__}
    rig = json.loads(session["rig"]) if session["rig"] else {}
    return row | {
        "session_id": session["session_id"],
        "subject_id": str(session["subject_id"]),
        "date": str(session["date"]),
        "rig": rig.get("rig_name"),
        "stage": session["curriculum_stage_name"],
        "dataset_version": session["dataset_version"],
        "error": None,
    }


def build(
    spec: DatasetSpec, workers: int = 32, fs=None, refresh: bool = False
) -> pd.DataFrame:
    fs = fs or filesystem()
    found = records(spec, refresh)
    process = spec.source.process
    assets = [a for r in found if (a := asset_of(r, process)) is not None]
    assets = created_by(assets, spec.query.until)
    chosen = latest_per_source(assets)
    rows = thread_map(lambda a: read_session(fs, a), chosen, workers)
    table = pd.DataFrame(rows)
    table["queried"] = queried(spec)
    storage.write_frame(table, Layout(spec.root).inventory)
    print(
        f"inventory: {len(found)} records, {len(assets)} assets created by "
        f"{spec.query.until or 'now'}, {len(chosen)} latest per source, "
        f"{table['error'].notna().sum()} unreadable"
    )
    return table


def wanted(spec: DatasetSpec, table: pd.DataFrame) -> pd.DataFrame:
    """The inventoried sessions the spec's query keeps, sorted by session_id."""
    query = spec.query
    table = table[table["error"].isna() & table["session_id"].notna()]
    keep = pd.Series(True, table.index)
    if query.min_dataset_version is not None:
        floor = Version(query.min_dataset_version)
        keep &= table["dataset_version"].map(lambda v: _version(str(v)) >= floor)
    keep |= table["session_id"].isin(query.include_sessions)
    keep &= ~table["session_id"].isin(query.exclude_sessions)
    keep &= ~table["rig"].isin(query.exclude_rigs)
    return table[keep].sort_values("session_id").reset_index(drop=True)

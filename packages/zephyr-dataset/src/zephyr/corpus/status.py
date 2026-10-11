"""``status``: how far a dataset got: counts per stage, and marks per rig."""

import json

import pandas as pd

from . import storage
from .config import DatasetSpec, Layout
from .inventory import wanted


def run(spec: DatasetSpec) -> dict:
    layout = Layout(spec.root)
    counts: dict[str, int | None] = dict.fromkeys(
        ["inventoried", "wanted", "fetched", "scored", "passed", "selected", "clips"]
    )
    if layout.inventory.exists():
        inventory = storage.read_parquet(layout.inventory)
        counts["inventoried"] = len(inventory)
        counts["wanted"] = len(wanted(spec, inventory))
    counts["fetched"] = len(layout.sessions())
    if layout.qc.exists():
        counts["scored"] = len(storage.read_parquet(layout.qc, columns=["session_id"]))
    if layout.clip_table.exists():
        table = storage.read_parquet(layout.clip_table, columns=["error"])
        counts["clips"] = int(table["error"].isna().sum())
    rigs = None
    if layout.results.exists():
        results = json.loads(layout.results.read_text(encoding="utf-8"))
        counts["passed"] = len(results["passed"])
        counts["selected"] = len(results["selected"])
        frame = pd.DataFrame(results["sessions"])
        frame["mark"] = frame["selection"].fillna("unset")
        rigs = pd.crosstab(frame["rig"].fillna("?"), frame["mark"])
        rigs["selected"] = frame.groupby(frame["rig"].fillna("?"))["clips"].apply(
            lambda c: int(c.notna().sum())
        )
    for name, value in counts.items():
        print(f"{name:>12}: {'-' if value is None else value}")
    if rigs is not None:
        print(rigs.to_string())
    return counts

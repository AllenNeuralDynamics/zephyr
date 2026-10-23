"""``qc``: score every session of ``sessions.toml`` with zephyr's session selection.

Runs :func:`zephyr.session_selection.assess` with the spec's thresholds, writes the
suggested marks and spans into the clips no one has reviewed (reviewed ones never
change), then the spec's ``[[review]]`` decisions over them (marked reviewed), and
keeps each session's metrics and suggestion in ``qc.parquet`` (SNR gate and
thresholds in its metadata) for ``select``.
"""

import json
import time

import pandas as pd
import pyarrow as pa

from zephyr.annotate_events import TuneState
from zephyr.session_selection import apply, assess

from . import storage
from .config import DatasetSpec, Layout, session_of

METRICS = [
    "snr_db",
    "amplitude",
    "mean_rate",
    "break_frac",
    "outside_frac",
    "rate_cv",
    "amp_cv",
    "rail_frac",
    "consistency_z",
    "good_frac",
    "artifacts",
    "accepted_s",
]


def apply_reviews(spec: DatasetSpec, state: TuneState) -> None:
    """Write the spec's ``[[review]]`` decisions over the suggestions, reviewed: the
    fields a review leaves out keep what QC wrote (on the first run) or had."""
    entries = {session_of(e.clip.source): e for e in state.entries}
    reviews = [  # one that only drops a camera is for ``clips``
        r
        for r in spec.review
        if (r.selection, r.excluded, r.included) != (None, None, None)
    ]
    missing = [r.session_id for r in reviews if r.session_id not in entries]
    if missing:
        raise SystemExit(f"qc: reviewed sessions not in the list: {missing}")
    for review in reviews:
        entry = entries[review.session_id]
        if review.selection is not None:
            entry.selection = review.selection
        if review.excluded is not None:
            entry.excluded = sorted(review.excluded)
        if review.included is not None:
            entry.included = sorted(review.included)
        entry.reviewed = True


def run(spec: DatasetSpec) -> pd.DataFrame:
    layout = Layout(spec.root)
    if not layout.local:
        raise SystemExit(
            "qc: zephyr reads the clip list and thermistors from local files; run qc "
            f"on a local copy of {layout.root} (sessions.toml, thermistor/, "
            "inventory.parquet) and upload sessions.toml and qc.parquet back"
        )
    state = TuneState(layout.session_list)
    start = time.time()

    def progress(done: int, total: int) -> None:
        if done % 250 == 0 or done == total:
            print(f"qc: {done}/{total} {time.time() - start:.0f}s", flush=True)

    table = assess(state, spec.qc.settings(), progress=progress, release=True)
    table.insert(0, "session_id", [session_of(e.clip.source) for e in state.entries])
    changed = apply(state, table)
    apply_reviews(spec, state)
    state.save()
    out = table[["session_id", *METRICS, "dead_sensor", "suggested", "reason"]].copy()
    out["included"] = [[[float(a), float(b)] for a, b in s] for s in table["included"]]
    out["excluded"] = [[[float(a), float(b)] for a, b in s] for s in table["excluded"]]
    meta = {
        "snr_gate_db": table.attrs.get("snr_threshold"),
        "qc": spec.qc.model_dump(),
        "events": spec.events.model_dump(),
    }
    arrow = pa.Table.from_pandas(out, preserve_index=False)
    arrow = arrow.replace_schema_metadata(
        {**(arrow.schema.metadata or {}), b"zephyr_dataset": json.dumps(meta).encode()}
    )
    storage.write_table(arrow, layout.qc)
    passed = int((out["suggested"] == "include").sum())
    print(
        f"qc: {len(out)} sessions scored in {time.time() - start:.0f}s, {passed} "
        f"suggested include, {changed} unreviewed clips updated"
    )
    return out


def metadata(path) -> dict:
    """What ``qc.parquet`` was made with (SNR gate, thresholds, detector)."""
    return json.loads(storage.schema_metadata(path).get(b"zephyr_dataset", b"{}"))

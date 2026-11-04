"""Quality-control reports for the quality notebook: copied once, read as tables.

A run is a folder of ``zephyr predict --quality-control`` (``*.qc.json`` beside each
``*.prediction.parquet``), local or ``s3://``;
an ``s3://`` folder and a reference file there are copied once into
``cache/quality-control/`` (read unsigned: the bucket is public). A folder of probe
reports (``*.probe.json``, ``zephyr-benchmarks probe``: noisy frames) is read the
same way. ``ZEPHYR_QC_REPORTS``,
``ZEPHYR_QC_REFERENCE`` and ``ZEPHYR_QC_PROBES`` give the notebook's defaults.
"""

import functools
import json
import os
from pathlib import Path

import pandas as pd

from zephyr.benchmarks.probe import load_probes
from zephyr.quality_control import load_reports
from zephyr.quality_control.metrics import MEASURES

from . import runs
from .test_sets import CACHE

LOCAL = CACHE / "quality-control"
REPORTS = os.environ.get("ZEPHYR_QC_REPORTS", "")
"""The reports folder the notebook opens first (empty: none)."""
REFERENCE = os.environ.get("ZEPHYR_QC_REFERENCE", "")
"""The reference file it opens first (empty: none)."""
PROBES = os.environ.get("ZEPHYR_QC_PROBES", "")
"""The probe reports folder it opens first (empty: none)."""
JUDGED = {name: better for name, (_, _, better) in MEASURES.items() if better}
"""The measures that judge a prediction, and which way is better."""


def local(location: str) -> Path | None:
    """*location* on this machine: itself when local, else its copy in :data:`LOCAL`
    (made once); ``None`` when it does not exist."""
    if not location.startswith("s3://"):
        path = Path(location)
        return path if path.exists() else None
    import fsspec

    s3 = fsspec.filesystem("s3", anon=True)  # read only, and the bucket is public
    key = location.removeprefix("s3://").rstrip("/")
    copy = LOCAL / key
    if copy.exists():
        return copy
    if not s3.exists(key):
        return None
    copy.parent.mkdir(parents=True, exist_ok=True)
    part = copy.with_name(copy.name + ".part")
    s3.get(key, str(part), recursive=s3.isdir(key))
    part.replace(copy)
    return copy


@functools.cache
def tables(location: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per video and one per window of the reports at *location* (empty when
    there are none), each with its real scores (``truth_<metric>``) where the clip's
    thermistor is in the dataset copy: joined here, never read by quality control."""
    path = local(location) if location else None
    if path is None:
        return pd.DataFrame(), pd.DataFrame()
    videos, windows = load_reports(path)
    truth = runs.report_truth(path)
    if len(truth):
        whole = truth[truth["whole"]].drop(columns=["whole", "start", "end"])
        videos = videos.merge(whole, on="stem", how="left")
        windows = runs.with_truth(
            windows, truth[~truth["whole"]].drop(columns=["whole", "end"])
        )
    return videos, windows


def reference(location: str) -> dict | None:
    path = local(location) if location else None
    return json.loads(path.read_text()) if path else None


def traces(location: str, stem: str) -> pd.DataFrame:
    """A video's saved trace, onset probability and inhales."""
    return pd.read_parquet(local(location) / f"{stem}.qc.parquet")


@functools.cache
def probes(location: str) -> pd.DataFrame:
    """The noisy frames at *location*, one row per video and noise level (empty when
    there are none)."""
    path = local(location) if location else None
    return load_probes(path) if path is not None else pd.DataFrame()

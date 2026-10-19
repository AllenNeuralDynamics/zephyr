"""Quality-control runs for the quality notebook: downloaded once, read as tables.

Which runs, under which labels, is a quality manifest of ``zephyr-benchmarks``
(``configs/results/``, :data:`MANIFEST`; ``ZEPHYR_QUALITY`` names another). A run's
reports (``*.qc.json``, ``*.qc.npz``) and its reference are copied once into
``cache/quality-control/`` (credentials from ``AWS_PROFILE`` for ``s3://`` roots).
"""

import functools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from zephyr.benchmarks.config import QualityManifest, QualityReport
from zephyr.config import load
from zephyr.quality_control import load_reports
from zephyr.quality_control.metrics import MEASURES

from .test_sets import CACHE

REPO = Path(__file__).resolve().parents[2]
MANIFEST = Path(
    os.environ.get(
        "ZEPHYR_QUALITY",
        REPO / "packages/zephyr-benchmarks/configs/results/quality-control-v3.toml",
    )
)
RESULTS: QualityManifest = load(QualityManifest, MANIFEST)
LOCAL = CACHE / "quality-control"
REPORTS: dict[str, QualityReport] = {r.label: r for r in RESULTS.report}
JUDGED = {name: better for name, (_, _, better) in MEASURES.items() if better}
"""The measures that judge a prediction, and which way is better."""


@functools.cache
def _fs():
    import fsspec

    return fsspec.filesystem("s3" if RESULTS.root.startswith("s3://") else "file")


def _remote(key: str) -> str:
    return f"{RESULTS.root.rstrip('/').removeprefix('s3://')}/{key}"


def _fetch(key: str) -> Path | None:
    """``<root>/<key>`` (file or folder) into :data:`LOCAL`, once; ``None`` if absent."""
    local = LOCAL / key
    if local.exists():
        return local
    if not _fs().exists(_remote(key)):
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    part = local.with_name(local.name + ".part")
    _fs().get(_remote(key), str(part), recursive=_fs().isdir(_remote(key)))
    part.replace(local)
    return local


def folder(label: str) -> Path | None:
    return _fetch(REPORTS[label].folder)


@functools.cache
def tables(label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per video and one per window of a run (empty if not uploaded)."""
    path = folder(label)
    if path is None:
        return pd.DataFrame(), pd.DataFrame()
    return load_reports(path)


def reference(label: str) -> dict | None:
    key = REPORTS[label].reference
    path = _fetch(key) if key else None
    return json.loads(path.read_text()) if path else None


def traces(label: str, stem: str) -> dict[str, np.ndarray]:
    """A video's saved trace, onset probability, draws' spread and inhales."""
    with np.load(folder(label) / f"{stem}.qc.npz") as data:
        return dict(data)


def report(label: str, stem: str) -> dict:
    return json.loads((folder(label) / f"{stem}.qc.json").read_text())

"""A predictions folder's quality control as the Predict page shows it (no Qt here).

:class:`QualityState` reads what ``zephyr predict --quality-control`` wrote: each video's
report (``<stem>.qc.json``) and, when the folder's checkpoint has one, its reference
(the windows of its validation animals, each beside its true scores). It never writes,
and never judges a video: the reference's windows are split at a true inhale F1 the
user sets, to show what good and poor predictions scored there.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..predict import SETTINGS
from ..quality_control.metrics import MEASURES
from ..quality_control.reference import Reference
from ..quality_control.report import REFERENCE, load_reports

SHOWN = ("head_trace_f1", "alternation_break_frac", "peak_prob", "onset_background")
"""The measures shown first: those that told good predictions from bad ones on held-out
animals and on another task alike."""
GOOD_F1 = 0.9
"""The default true inhale F1 that splits the reference's windows."""


@dataclass(frozen=True)
class Row:
    """One row of the overview: a label, what it is (``good``, ``poor`` or ``video``),
    the video's stem (videos only) and its windows' values of one measure."""

    label: str
    kind: str
    stem: str | None
    values: np.ndarray

    def spread(self) -> tuple[float, float, float, float, float] | None:
        """5th, 25th, 50th, 75th and 95th percentiles, or None without values."""
        v = self.values[np.isfinite(self.values)]
        return tuple(np.percentile(v, [5, 25, 50, 75, 95])) if len(v) else None


class QualityState:
    """The quality control of the predictions in *folder*, against the reference of the
    checkpoint they came from when there is one (and it is that checkpoint's)."""

    def __init__(self, folder: Path, good_f1: float = GOOD_F1) -> None:
        self.folder = Path(folder)
        self.good_f1 = good_f1
        self.videos, self.windows = load_reports(self.folder)
        self.reference = self._reference()

    def _reference(self) -> Reference | None:
        try:
            settings = json.loads((self.folder / SETTINGS).read_text())
        except (OSError, ValueError):
            return None
        quality = settings.get("quality_control") or {}
        candidates = [quality.get("reference"), settings.get("checkpoint")]
        paths = [Path(candidates[0])] if candidates[0] else []
        if candidates[1]:
            paths.append(Path(candidates[1]).parent / REFERENCE)
        for path in paths:
            if not path.is_file():
                continue
            reference = Reference.load(path)
            if reference.checkpoint_sha256 == settings.get("checkpoint_sha256"):
                return reference
        return None

    @property
    def empty(self) -> bool:
        return self.windows.empty

    @property
    def split(self) -> bool:
        """Whether the reference's windows carry the truth to split them by."""
        return bool(self.reference and self.reference.validation) and any(
            w.get("truth_inhale_f1") is not None for w in self.reference.validation
        )

    def measures(self) -> list[str]:
        """The measures the reports hold, the :data:`SHOWN` ones first."""
        held = [m for m in MEASURES if m in self.windows]
        return [m for m in SHOWN if m in held] + [m for m in held if m not in SHOWN]

    def rows(self, measure: str) -> list[Row]:
        """The reference's good and poor windows (true inhale F1 at or above
        :attr:`good_f1`, and below), when it has them, then each video's windows."""
        out = []
        if self.split:
            ref = pd.DataFrame(self.reference.validation)
            if measure in ref:
                f1 = ref["truth_inhale_f1"].astype(float)
                good = f1 >= self.good_f1
                for kind, mask, label in (
                    ("good", good, f"Validation, F1 ≥ {self.good_f1:g}"),
                    ("poor", ~good & f1.notna(), f"Validation, F1 < {self.good_f1:g}"),
                ):
                    values = ref.loc[mask, measure].to_numpy(float)
                    out.append(Row(f"{label} (n={mask.sum()})", kind, None, values))
        for stem, group in self.windows.groupby("stem", sort=True):
            values = group[measure].to_numpy(float) if measure in group else np.empty(0)
            out.append(Row(stem, "video", stem, values))
        return out

    def table(self) -> pd.DataFrame:
        """One row per video: ``stem``, ``minutes``, and per measure its median over the
        video's windows and, with a reference, the median percentile."""
        rows = []
        for stem, group in self.windows.groupby("stem", sort=True):
            row = {
                "stem": stem,
                "minutes": float(group["end"].max() - group["start"].min()) / 60,
            }
            for m in self.measures():
                row[m] = float(group[m].median())
                if f"pct_{m}" in group:
                    row[f"pct_{m}"] = float(group[f"pct_{m}"].median())
            rows.append(row)
        return pd.DataFrame(rows)

    def over_time(self, stem: str, measure: str) -> pd.DataFrame:
        """A video's windows in time order: ``start``, ``end`` and the measure's
        ``value`` (and ``percentile``, with a reference)."""
        group = self.windows[self.windows["stem"] == stem].sort_values("start")
        out = group[["start", "end"]].assign(value=group[measure].astype(float))
        if f"pct_{measure}" in group:
            out["percentile"] = group[f"pct_{measure}"].astype(float)
        return out.reset_index(drop=True)

    def window_at(self, stem: str, t: float) -> tuple[float, float] | None:
        """The window of video *stem* holding time *t* (its clock), if any."""
        rows = self.windows[self.windows["stem"] == stem]
        hit = rows[(rows["start"] <= t) & (t < rows["end"])]
        return (
            None
            if hit.empty
            else (float(hit["start"].iloc[0]), float(hit["end"].iloc[0]))
        )

    def good_range(self, measure: str) -> tuple[float, float] | None:
        """The 5th to 95th percentile of *measure* over the reference's good windows."""
        good = next((r for r in self.rows(measure) if r.kind == "good"), None)
        spread = good.spread() if good else None
        return (spread[0], spread[-1]) if spread else None

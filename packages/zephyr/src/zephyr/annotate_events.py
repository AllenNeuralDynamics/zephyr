"""Compare and choose the breathing-event detector of each clip in a clip list.

Each labelled clip's thermistor is shown with the inhale peaks and exhale troughs of
its detector from :mod:`zephyr.events`, with live quality metrics, so a method (and its
parameters) can be picked per clip.  Events are never placed by hand, only stretches
to exclude are marked: saving writes each clip's ``events = {method, params}`` and
``excluded = [[start, end], ...]`` next to its ``box`` in the clip list.

CLI
---
    zephyr annotate events [<list.toml>]

Menu: File > Open / Add videos / Remove / Save; Edit > apply a method to the current
clip, the selected clips or all of them.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import events
from .clips import update_clip_list
from .config import Clip, ClipList, ResolvedClip, load, resolve_clip
from .rates import breath_rates

POINT_KINDS = ("inhale", "exhale")
RATE_BAND_HZ = (2.0, 15.0)
"""Expected breathing rates; a breath outside them is flagged in the stats."""
METRICS = (
    "inhales",
    "exhales",
    "mean_rate",
    "outside",
    "ie_median",
    "ei_median",
    "ii_breaks",
    "ee_breaks",
    "spans",
)
"""The per-clip numbers :meth:`TuneState.clip_metrics` reports."""


@dataclass(frozen=True)
class Trace:
    """A clip's thermistor as the viewer shows it: at its own rate, filtered and
    z-scored, as every detector sees it."""

    t: np.ndarray
    x: np.ndarray
    fs: float
    raw: np.ndarray | None = None
    """The unfiltered thermistor on the same grid, if known."""


def load_trace(clip: ResolvedClip) -> Trace:
    frame = pd.read_parquet(clip.thermistor)
    t_raw, v_raw = frame["Time"].to_numpy(float), frame["Signal"].to_numpy(float)
    t, x, fs = events.native_trace(t_raw, v_raw)
    return Trace(t, x, fs, np.interp(t, t_raw, v_raw))


def mean_metrics(rows: list[dict]) -> dict:
    """Mean of each metric over clips (each clip counts once); NaNs are skipped."""
    out = {}
    for key in METRICS:
        values = np.array([r[key] for r in rows], float)
        out[key] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
    return out


@dataclass
class Entry:
    """One clip being tuned: its detector setting and what it produced."""

    clip: ResolvedClip
    method: str
    params: dict
    """Every parameter of *method*, defaults included."""
    custom: bool
    """Whether the clip list gives it its own ``events`` (else it uses the list's)."""
    rejected: bool = False
    """Whether the video is never to be used (``rejected = true`` in the list)."""
    excluded: list[tuple[float, float]] = field(default_factory=list)
    """The clip's excluded spans (start, end), sorted."""
    undo: list[list[tuple[float, float]]] = field(default_factory=list)
    dirty: bool = False
    trace: Trace | None = None
    rows: pd.DataFrame | None = None
    stats: dict = field(default_factory=dict)


class TuneState:
    """The clips of a clip list and the detector chosen for each; no UI."""

    def __init__(self, path: Path, trace_loader=load_trace) -> None:
        self.path = Path(path).resolve()
        self.clip_list = load(ClipList, self.path)
        self._load_trace = trace_loader
        self.entries: list[Entry] = []
        everything = self.clip_list.resolve(include_rejected=True)
        for raw, clip in zip(self.clip_list.clip, everything):
            if clip.labelled:
                entry = self._entry(clip, raw.events is not None)
                entry.rejected = raw.rejected
                self.entries.append(entry)
        if not self.entries:
            raise ValueError("the clip list has no labelled clips")
        self.index = 0
        self.added: list[Path] = []
        self.removed: set[Path] = set()

    @staticmethod
    def _entry(clip: ResolvedClip, custom: bool) -> Entry:
        source = clip.events
        params = events.detector_params(source.method, source.params).model_dump()
        return Entry(clip, source.method, dict(params), custom, list(clip.excluded))

    # ---- the clips -------------------------------------------------------------

    @property
    def clips(self) -> list[ResolvedClip]:
        return [e.clip for e in self.entries]

    @property
    def clip(self) -> ResolvedClip:
        return self.entries[self.index].clip

    @property
    def entry(self) -> Entry:
        return self.entries[self.index]

    @property
    def dirty(self) -> set[int]:
        """Indices of clips whose detector changed since the last save."""
        return {i for i, e in enumerate(self.entries) if e.dirty}

    @property
    def modified(self) -> bool:
        return bool(self.dirty or self.added or self.removed)

    def select(self, i: int) -> None:
        self.index = int(i) % len(self.entries)

    def trace(self, i: int | None = None) -> Trace:
        entry = self.entries[self.index if i is None else i]
        if entry.trace is None:
            entry.trace = self._load_trace(entry.clip)
        return entry.trace

    # ---- events of the chosen detector --------------------------------------------

    def rows(self, i: int | None = None) -> pd.DataFrame:
        """The clip's events (``kind``, ``time``, ``end``, ``source`` columns)."""
        entry = self.entries[self.index if i is None else i]
        if entry.rows is None:
            found = events.detect(
                entry.method, pd.read_parquet(entry.clip.thermistor), None, entry.params
            )
            parts = [
                pd.DataFrame({"kind": kind, "time": times})
                for kind, times in (("inhale", found.inhale), ("exhale", found.exhale))
            ]
            rows = pd.concat(parts, ignore_index=True).sort_values(
                "time", kind="stable"
            )
            rows = rows.reset_index(drop=True).assign(end=np.nan, source="auto")
            entry.rows = rows[["kind", "time", "end", "source"]]
        return entry.rows

    def times(self, kind: str, i: int | None = None) -> np.ndarray:
        rows = self.rows(i)
        return rows.loc[rows["kind"] == kind, "time"].to_numpy()

    def describe(self, i: int) -> tuple[str, str]:
        """The clip's method, and the parameters that differ from its defaults."""
        entry = self.entries[i]
        defaults = events.detector_params(entry.method).model_dump()
        changed = {k: v for k, v in entry.params.items() if v != defaults[k]}
        return entry.method, ", ".join(f"{k}={v:g}" for k, v in changed.items())

    def apply(
        self,
        indices: list[int],
        method: str,
        params: dict | None = None,
        progress=None,
    ) -> None:
        """Give each of *indices* the detector *method* with *params* and run it;
        *progress* is called with (clips done, clips total).  Bad names or
        parameters raise ValueError before anything changes."""
        full = events.detector_params(method, params).model_dump()
        for n, i in enumerate(indices, start=1):
            entry = self.entries[i]
            entry.method, entry.params = method, dict(full)
            entry.custom = entry.dirty = True
            entry.rows = None
            self.rows(i)
            if progress is not None:
                progress(n, len(indices))

    # ---- rejected videos ------------------------------------------------------------

    def set_rejected(self, indices: list[int], rejected: bool = True) -> None:
        """Reject (or take back) the videos at *indices*: a rejected video is never
        used downstream.  At least one video must stay in use."""
        change = set(indices)
        left = [
            i
            for i, e in enumerate(self.entries)
            if not (e.rejected if i not in change else rejected)
        ]
        if not left:
            raise ValueError("at least one video must stay in use")
        for i in change:
            if self.entries[i].rejected != rejected:
                self.entries[i].rejected = rejected
                self.entries[i].dirty = True

    def accepted(self) -> list[int]:
        """Indices of the videos still in use."""
        return [i for i, e in enumerate(self.entries) if not e.rejected]

    # ---- excluded spans -------------------------------------------------------------

    def spans(self, i: int | None = None) -> np.ndarray:
        """The clip's excluded spans, shape ``(n, 2)``."""
        entry = self.entries[self.index if i is None else i]
        return np.asarray(entry.excluded, np.float64).reshape(-1, 2)

    def _set_spans(self, entry: Entry, spans: list[tuple[float, float]]) -> None:
        entry.undo.append(list(entry.excluded))
        entry.excluded = sorted(spans)
        entry.dirty = True

    def exclude(self, t0: float, t1: float, i: int | None = None) -> None:
        """Exclude the span between *t0* and *t1* (either order) from the clip."""
        start, end = sorted((float(t0), float(t1)))
        if not start < end:
            raise ValueError("an excluded span needs a start before its end")
        entry = self.entries[self.index if i is None else i]
        self._set_spans(entry, [*entry.excluded, (start, end)])

    def delete_span(self, t: float, i: int | None = None) -> bool:
        """Remove the span containing time *t*; False if there is none."""
        entry = self.entries[self.index if i is None else i]
        keep = [s for s in entry.excluded if not s[0] <= t <= s[1]]
        if len(keep) == len(entry.excluded):
            return False
        self._set_spans(entry, keep)
        return True

    def update_span(
        self, k: int, start: float, end: float, i: int | None = None
    ) -> None:
        """Give the clip's *k*-th span (in start order) new bounds."""
        start, end = float(start), float(end)
        if not start < end:
            raise ValueError("an excluded span needs a start before its end")
        entry = self.entries[self.index if i is None else i]
        spans = list(entry.excluded)
        spans[k] = (start, end)
        self._set_spans(entry, spans)

    def remove_spans(self, indices: list[int], i: int | None = None) -> bool:
        """Remove the clip's spans at *indices* (in start order) as one change."""
        entry = self.entries[self.index if i is None else i]
        drop = set(indices)
        keep = [s for n, s in enumerate(entry.excluded) if n not in drop]
        if len(keep) == len(entry.excluded):
            return False
        self._set_spans(entry, keep)
        return True

    def undo_span(self, i: int | None = None) -> bool:
        """Take back the last span change of the clip."""
        entry = self.entries[self.index if i is None else i]
        if not entry.undo:
            return False
        entry.excluded = entry.undo.pop()
        entry.dirty = True
        return True

    # ---- the list ------------------------------------------------------------------

    def add_videos(self, videos: list[Path]) -> list[Path]:
        """Add videos to the list (their thermistors must exist); returns those
        added, skipping any already present."""
        present = {e.clip.video.resolve() for e in self.entries}
        added = []
        for video in (Path(v).resolve() for v in videos):
            if video in present:
                continue
            clip = resolve_clip(
                Clip(video=video), self.clip_list.derive, self.clip_list.events
            )
            if not clip.labelled or not clip.thermistor.is_file():
                raise ValueError(f"{video.name}: no thermistor file {clip.thermistor}")
            self.entries.append(self._entry(clip, False))
            self.added.append(video)
            self.removed.discard(video)
            present.add(video)
            added.append(video)
        return added

    def remove(self, indices: list[int]) -> None:
        """Drop clips from the list; the last clip cannot be removed."""
        gone = set(indices)
        if all(e.rejected or i in gone for i, e in enumerate(self.entries)):
            raise ValueError("a clip list needs at least one video in use")
        for i in sorted(gone, reverse=True):
            video = self.entries[i].clip.video.resolve()
            if video in self.added:
                self.added.remove(video)
            else:
                self.removed.add(video)
            del self.entries[i]
        self.index = min(self.index, len(self.entries) - 1)

    # ---- checks --------------------------------------------------------------------

    def breath_stats(
        self, i: int | None = None, band: tuple[float, float] = RATE_BAND_HZ
    ) -> dict:
        """Live breathing features of the clip: counts, the instantaneous rate of
        each breath (mean, and those outside *band*), inhale->exhale and
        exhale->inhale durations, and same-kind neighbours (``ii_breaks``,
        ``ee_breaks``)."""
        spans = self.spans(i)

        def touching(first, last):
            if not len(spans) or not len(first):
                return np.zeros(len(first), bool)
            return (
                (first[:, None] <= spans[None, :, 1])
                & (last[:, None] >= spans[None, :, 0])
            ).any(axis=1)

        onsets = self.times("inhale", i)
        keep = ~touching(onsets[:-1], onsets[1:])
        rates = breath_rates(onsets)[keep] if len(onsets) > 1 else np.empty(0)
        out = (rates < band[0]) | (rates > band[1])
        rows = self.rows(i)
        points = rows[rows["kind"].isin(POINT_KINDS)].sort_values("time", kind="stable")
        kinds, times = points["kind"].to_numpy(), points["time"].to_numpy()
        gap = times[1:] - times[:-1]
        ok = ~touching(times[:-1], times[1:])

        def pair(x, y):
            return ok & (kinds[:-1] == x) & (kinds[1:] == y)

        return {
            "inhales": len(onsets),
            "exhales": int(np.sum(kinds == "exhale")),
            "rates": rates,
            "mean": float(rates.mean()) if len(rates) else float("nan"),
            "outside": int(out.sum()),
            "flagged": onsets[:-1][keep][out],
            "ie": gap[pair("inhale", "exhale")],
            "ei": gap[pair("exhale", "inhale")],
            "ii_breaks": int(pair("inhale", "inhale").sum()),
            "ee_breaks": int(pair("exhale", "exhale").sum()),
        }

    def clip_metrics(
        self, i: int | None = None, band: tuple[float, float] = RATE_BAND_HZ
    ) -> dict:
        """The summary of one clip's events, as a row of the all-clips table."""
        stats = self.breath_stats(i, band)

        def median(a):
            return float(np.median(a)) if len(a) else np.nan

        return {
            "inhales": stats["inhales"],
            "exhales": stats["exhales"],
            "mean_rate": stats["mean"],
            "outside": stats["outside"],
            "ie_median": median(stats["ie"]),
            "ei_median": median(stats["ei"]),
            "ii_breaks": stats["ii_breaks"],
            "ee_breaks": stats["ee_breaks"],
            "spans": len(self.spans(i)),
        }

    # ---- saving ----------------------------------------------------------------------

    def save(self) -> int:
        """Write each customised clip's ``events``, the added videos and the removals
        into the clip list; returns the number of clips given their own detector."""
        chosen = {
            e.clip.video.resolve(): (e.method, e.params)
            for e in self.entries
            if e.custom
        }
        spans = {e.clip.video.resolve(): e.excluded for e in self.entries}
        rejected = {e.clip.video.resolve(): e.rejected for e in self.entries}
        update_clip_list(
            self.path,
            events=chosen,
            excluded=spans,
            rejected=rejected,
            add=self.added,
            remove=self.removed,
        )
        for entry in self.entries:
            entry.dirty = False
            entry.undo.clear()
        self.added, self.removed = [], set()
        return len(chosen)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "clips", nargs="?", type=Path, help="Clip list TOML file (else File > Open)."
    )
    parser.add_argument("--window-s", type=float, default=5.0)
    args = parser.parse_args(argv)
    from .annotate_events_ui import run_ui

    run_ui(args.clips, args.window_s)

"""Compare and choose the breathing-event detector of each clip in a clip list.

Each labelled clip's thermistor is shown with the inhale peaks and exhale troughs of
its detector from :mod:`zephyr.events`, with live quality metrics, so a method (and its
parameters) can be picked per clip.  Events are never placed by hand, only stretches
to exclude are marked: saving writes each clip's ``events = {method, params}`` and
``excluded = [[start, end], ...]`` next to its ``box`` in the clip list.

:class:`TuneState` is the logic of the app's Breathing page (``zephyr gui <list>
--page breathing``).

CLI
---
    zephyr annotate events <list.toml> --auto   # the default detector for all, saved
"""

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import events
from .clips import SCAN_EVENTS, update_clip_list
from .config import Clip, ClipList, EventsSource, ResolvedClip, load, resolve_clip
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
    "amplitude",
    "snr_db",
    "spans",
)
"""The per-clip numbers :meth:`TuneState.clip_metrics` reports."""


def parallel_map(fn, items, progress=None) -> list:
    """``[fn(x) for x in items]`` on threads (the detectors' numpy / pandas work
    releases the GIL), in order; *progress* is called with (done, total)."""
    items = list(items)
    done, lock = [0], threading.Lock()

    def run(item):
        result = fn(item)
        if progress is not None:
            with lock:
                done[0] += 1
                progress(done[0], len(items))
        return result

    with ThreadPoolExecutor(max(1, min(8, os.cpu_count() or 1))) as pool:
        return list(pool.map(run, items))


@dataclass(frozen=True)
class Trace:
    """A clip's thermistor as the viewer shows it: at its own rate, filtered and
    z-scored, as every detector sees it."""

    t: np.ndarray
    x: np.ndarray
    fs: float
    raw: np.ndarray | None = None
    """The unfiltered thermistor on the same grid, if known."""


def load_trace(clip: ResolvedClip, highpass_hz: float = 0.2) -> Trace:
    frame = pd.read_parquet(clip.thermistor)
    t_raw, v_raw = frame["Time"].to_numpy(float), frame["Signal"].to_numpy(float)
    t, x, fs = events.native_trace(t_raw, v_raw, highpass_hz)
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
    selection: str | None = None
    """``None``, ``"include"`` or ``"exclude"`` (``selection`` in the list)."""
    reviewed: bool = False
    """Whether a person confirmed the selection (``reviewed = true`` in the list)."""
    excluded: list[tuple[float, float]] = field(default_factory=list)
    """The clip's excluded spans (start, end), sorted."""
    included: list[tuple[float, float]] = field(default_factory=list)
    """The clip's included (good) spans (start, end), sorted; a note only."""
    undo: list[tuple[list, list]] = field(default_factory=list)
    dirty: bool = False
    trace: Trace | None = None
    rows: pd.DataFrame | None = None
    stats: dict = field(default_factory=dict)

    @property
    def rejected(self) -> bool:
        return self.selection == "exclude"


class TuneState:
    """The clips of a clip list and the detector chosen for each; no UI."""

    def __init__(self, path: Path, trace_loader=load_trace) -> None:
        self.path = Path(path).resolve()
        self.clip_list = load(ClipList, self.path)
        self._load_trace = trace_loader
        self.entries: list[Entry] = []
        everything = self.clip_list.resolve(include_excluded=True)
        for raw, clip in zip(self.clip_list.clip, everything):
            if clip.labelled:
                entry = self._entry(clip, raw.events is not None)
                entry.selection = raw.selection
                entry.reviewed = raw.reviewed
                entry.included = sorted(raw.included)
                self.entries.append(entry)
        if not self.entries:
            raise ValueError("the clip list has no labelled clips")
        self.index = 0
        self.added: list[Path] = []
        self.removed: set[Path] = set()
        self.list_events: EventsSource | None = None
        """The list's ``[events]`` to write on save (set by :meth:`auto`)."""

    @staticmethod
    def _entry(clip: ResolvedClip, custom: bool) -> Entry:
        source = clip.events
        params = events.detector_params(source.method, source.params).model_dump()
        return Entry(
            clip, source.method, dict(params), custom, excluded=list(clip.excluded)
        )

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
            entry.trace = self._load_trace(
                entry.clip, entry.params.get("highpass_hz", 0.2)
            )
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
        for i in indices:
            entry = self.entries[i]
            entry.method, entry.params = method, dict(full)
            entry.custom = entry.dirty = True
            entry.rows = entry.trace = None  # the filter may have changed
        parallel_map(self.rows, indices, progress)

    def auto(self, progress=None) -> None:
        """Make the default detector (:data:`zephyr.clips.SCAN_EVENTS`) the list's
        ``[events]`` and every clip's, dropping their own; marks, spans and reviews
        are kept, and nothing is marked reviewed."""
        source = SCAN_EVENTS
        full = events.detector_params(source.method, source.params).model_dump()
        self.list_events = source
        for entry in self.entries:
            entry.method, entry.params = source.method, dict(full)
            entry.custom, entry.dirty = False, True
            entry.rows = entry.trace = None
        parallel_map(self.rows, list(range(len(self.entries))), progress)

    # ---- include / exclude marks ----------------------------------------------------

    SELECTIONS = (None, "include", "exclude")

    def set_selection(self, indices: list[int], value: str | None) -> None:
        """Mark the videos at *indices* ``None``, ``"include"`` or ``"exclude"``:
        an excluded video is never used downstream, the marks change nothing else.
        At least one video must stay un-excluded."""
        if value not in self.SELECTIONS:
            raise ValueError(f"selection must be one of {self.SELECTIONS}")
        change = set(indices)
        if not any(
            (value if i in change else e.selection) != "exclude"
            for i, e in enumerate(self.entries)
        ):
            raise ValueError("at least one video must stay in use")
        for i in change:
            if self.entries[i].selection != value:
                self.entries[i].selection = value
                self.entries[i].dirty = True

    def set_reviewed(self, indices: list[int], reviewed: bool = True) -> None:
        """Mark the videos at *indices* reviewed (or not) by a person."""
        for i in set(indices):
            if self.entries[i].reviewed != reviewed:
                self.entries[i].reviewed = reviewed
                self.entries[i].dirty = True

    def next_selection(self, i: int) -> str | None:
        """The mark after video *i*'s in the cycle none, include, exclude."""
        order = self.SELECTIONS
        return order[(order.index(self.entries[i].selection) + 1) % len(order)]

    def accepted(self) -> list[int]:
        """Indices of the videos not excluded."""
        return [i for i, e in enumerate(self.entries) if e.selection != "exclude"]

    # ---- excluded and included spans ----------------------------------------------

    KINDS = ("exclude", "include")
    """Span kinds: stretches whose events are untrusted, and good stretches (a note)."""

    @staticmethod
    def _kind(entry: Entry, kind: str) -> list[tuple[float, float]]:
        if kind not in TuneState.KINDS:
            raise ValueError(f"span kind must be one of {TuneState.KINDS}")
        return entry.excluded if kind == "exclude" else entry.included

    def spans(self, i: int | None = None, kind: str = "exclude") -> np.ndarray:
        """The clip's spans of *kind*, shape ``(n, 2)``."""
        entry = self.entries[self.index if i is None else i]
        return np.asarray(self._kind(entry, kind), np.float64).reshape(-1, 2)

    def _set_spans(
        self, entry: Entry, spans: list[tuple[float, float]], kind: str = "exclude"
    ) -> None:
        self._kind(entry, kind)
        entry.undo.append((list(entry.excluded), list(entry.included)))
        if kind == "exclude":
            entry.excluded = sorted(spans)
        else:
            entry.included = sorted(spans)
        entry.dirty = True

    def exclude(
        self, t0: float, t1: float, i: int | None = None, kind: str = "exclude"
    ) -> None:
        """Add the span between *t0* and *t1* (either order) to the clip: excluded
        (default) or, with ``kind="include"``, a good stretch."""
        start, end = sorted((float(t0), float(t1)))
        if not start < end:
            raise ValueError("a span needs a start before its end")
        entry = self.entries[self.index if i is None else i]
        self._set_spans(entry, [*self._kind(entry, kind), (start, end)], kind)

    def delete_span(
        self, t: float, i: int | None = None, kind: str | None = None
    ) -> bool:
        """Remove the span of *kind* (either, if None) containing time *t*; False if
        there is none."""
        entry = self.entries[self.index if i is None else i]
        changed = False
        for k in (kind,) if kind else self.KINDS:
            spans = self._kind(entry, k)
            keep = [s for s in spans if not s[0] <= t <= s[1]]
            if len(keep) != len(spans):
                self._set_spans(entry, keep, k)
                changed = True
        return changed

    def update_span(
        self,
        k: int,
        start: float,
        end: float,
        i: int | None = None,
        kind: str = "exclude",
    ) -> None:
        """Give the clip's *k*-th span of *kind* (in start order) new bounds."""
        start, end = float(start), float(end)
        if not start < end:
            raise ValueError("a span needs a start before its end")
        entry = self.entries[self.index if i is None else i]
        spans = list(self._kind(entry, kind))
        spans[k] = (start, end)
        self._set_spans(entry, spans, kind)

    def remove_spans(
        self, indices: list[int], i: int | None = None, kind: str = "exclude"
    ) -> bool:
        """Remove the clip's spans of *kind* at *indices* (in start order) as one
        change."""
        entry = self.entries[self.index if i is None else i]
        drop = set(indices)
        spans = self._kind(entry, kind)
        keep = [s for n, s in enumerate(spans) if n not in drop]
        if len(keep) == len(spans):
            return False
        self._set_spans(entry, keep, kind)
        return True

    def clear_spans(self, indices: list[int]) -> int:
        """Remove every span (both kinds) of the videos at *indices*, each as one
        undoable change; returns how many videos had any."""
        cleared = 0
        for i in indices:
            entry = self.entries[i]
            if entry.excluded or entry.included:
                entry.undo.append((list(entry.excluded), list(entry.included)))
                entry.excluded, entry.included = [], []
                entry.dirty = True
                cleared += 1
        return cleared

    def undo_span(self, i: int | None = None) -> bool:
        """Take back the last span change of the clip."""
        entry = self.entries[self.index if i is None else i]
        if not entry.undo:
            return False
        excluded, included = entry.undo.pop()
        entry.excluded, entry.included = excluded, included
        entry.dirty = True
        return True

    # ---- the list ------------------------------------------------------------------

    def add_videos(self, videos: list[Path]) -> list[Path]:
        """Add videos to the list (their thermistors must exist); returns those
        added, skipping any already present."""
        present = {e.clip.source.resolve() for e in self.entries}
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
        if all(
            e.selection == "exclude" or i in gone for i, e in enumerate(self.entries)
        ):
            raise ValueError("a clip list needs at least one video in use")
        for i in sorted(gone, reverse=True):
            video = self.entries[i].clip.source.resolve()
            if video in self.added:
                self.added.remove(video)
            else:
                self.removed.add(video)
            del self.entries[i]
        self.index = min(self.index, len(self.entries) - 1)

    # ---- checks --------------------------------------------------------------------

    def breath_stats(
        self,
        i: int | None = None,
        band: tuple[float, float] = RATE_BAND_HZ,
        whole: bool = False,
    ) -> dict:
        """Live breathing features of the clip: counts, the instantaneous rate of
        each breath (mean, and those outside *band*), inhale->exhale and
        exhale->inhale durations, and same-kind neighbours (``ii_breaks``,
        ``ee_breaks``); events touching an excluded span are left out unless
        *whole*."""
        spans = np.empty((0, 2)) if whole else self.spans(i)

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

    def signal_quality(self, i: int | None = None) -> tuple[float, float]:
        """Mean amplitude and SNR (dB) of the raw thermistor: the amplitude is the
        mean 5th-to-95th percentile range over 1 s blocks, the noise the robust
        standard deviation of its sample-to-sample differences."""
        trace = self.trace(i)
        raw = np.asarray(trace.x if trace.raw is None else trace.raw, float)
        block = max(3, round(trace.fs))
        blocks = raw[: len(raw) // block * block].reshape(-1, block)
        if not len(blocks):
            return np.nan, np.nan
        low, high = np.percentile(blocks, [5, 95], axis=1)
        amplitude = float(np.mean(high - low))
        step = np.diff(raw)
        noise = 1.4826 * float(np.median(np.abs(step - np.median(step)))) / np.sqrt(2)
        if noise <= 0:
            return amplitude, np.nan
        return amplitude, float(20 * np.log10(amplitude / noise))

    def clip_metrics(
        self, i: int | None = None, band: tuple[float, float] = RATE_BAND_HZ
    ) -> dict:
        """The summary of one clip's events, as a row of the all-clips table."""
        stats = self.breath_stats(i, band)

        def median(a):
            return float(np.median(a)) if len(a) else np.nan

        amplitude, snr_db = self.signal_quality(i)
        return {
            "inhales": stats["inhales"],
            "exhales": stats["exhales"],
            "mean_rate": stats["mean"],
            "outside": stats["outside"],
            "ie_median": median(stats["ie"]),
            "ei_median": median(stats["ei"]),
            "ii_breaks": stats["ii_breaks"],
            "ee_breaks": stats["ee_breaks"],
            "amplitude": amplitude,
            "snr_db": snr_db,
            "spans": len(self.spans(i)),
        }

    # ---- saving ----------------------------------------------------------------------

    def save(self) -> int:
        """Write each customised clip's ``events``, the added videos and the removals
        into the clip list; returns the number of clips given their own detector."""
        chosen = {
            e.clip.source.resolve(): (e.method, e.params)
            for e in self.entries
            if e.custom
        }
        spans = {e.clip.source.resolve(): e.excluded for e in self.entries}
        selection = {e.clip.source.resolve(): e.selection for e in self.entries}
        update_clip_list(
            self.path,
            events={e.clip.source.resolve(): None for e in self.entries} | chosen,
            list_events=self.list_events,
            excluded=spans,
            included={e.clip.source.resolve(): e.included for e in self.entries},
            selection=selection,
            reviewed={e.clip.source.resolve(): e.reviewed for e in self.entries},
            add=self.added,
            remove=self.removed,
        )
        for entry in self.entries:
            entry.dirty = False
            entry.undo.clear()
        self.added, self.removed = [], set()
        self.list_events = None
        return len(chosen)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Give every clip the default detector (choosing one by hand is "
        "the app's: zephyr gui <list> --page breathing)."
    )
    parser.add_argument("clips", type=Path, help="Clip list TOML file.")
    parser.add_argument(
        "--auto",
        action="store_true",
        required=True,
        help="Give every clip the default detector and save (Auto in the app).",
    )
    args = parser.parse_args(argv)
    state = TuneState(args.clips)
    state.auto()
    state.save()
    print(f"{args.clips}: {len(state.entries)} clip(s) use {SCAN_EVENTS.method}")

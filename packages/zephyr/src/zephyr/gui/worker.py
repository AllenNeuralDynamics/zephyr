"""The child process the app runs a ``zephyr`` command in (``python -m``; no Qt).

It calls the command's own ``main(argv)``, exactly as the command line would, so its
output is the command's. For ``run`` it also passes training an ``on_scored`` hook that
writes a few validation snippets to stdout, one JSON line after :data:`MARK`: they
exist only on that pipe, and nothing of them is saved.
"""

import importlib
import json
import sys
from dataclasses import asdict

import numpy as np

COMMANDS = {
    "clips": "zephyr.clips",
    "annotate": "zephyr.annotate",
    "preprocess": "zephyr.preprocess",
    "run": "zephyr.run",
    "evaluate": "zephyr.evaluate",
    "predict": "zephyr.predict",
    "quality-control": "zephyr.quality_control",
}
"""The ``zephyr`` subcommands the app runs, and the module whose ``main`` each is."""
MARK = "\x1esnippets "
"""Starts a line of snippets on the worker's stdout."""


def _window(n: int, span: tuple[float, float], width: int) -> slice:
    """*width* samples centred in the scored *span* (fractions) of *n* samples."""
    lo, hi = int(n * span[0]), int(n * span[1])
    width = min(width, hi - lo)
    start = lo + (hi - lo - width) // 2
    return slice(start, start + width)


def snippets(epoch: int, clips, n_clips: int = 4, seconds: float = 20.0) -> dict:
    """A JSON-ready window of up to *n_clips* validation clips (one per recording, in
    clip order) from :class:`zephyr.train.ScoredClip` predictions."""
    from ..evaluate import head_event_indices

    chosen, recordings = [], set()
    for clip in sorted(clips, key=lambda c: c.entry.clip_id):
        if clip.entry.recording not in recordings and len(chosen) < n_clips:
            recordings.add(clip.entry.recording)
            chosen.append(clip)
    out = []
    for clip in chosen:
        times = clip.times
        fs = 1.0 / float(np.median(np.diff(times)))
        rows = _window(len(times), clip.span, round(seconds * fs))
        t = times[rows]
        truth_inhale = np.empty(0)
        if clip.entry.events is not None:
            stored = np.load(clip.entry.events)
            onsets = stored["onset_times"]
            truth_inhale = onsets[(onsets >= t[0]) & (onsets <= t[-1])]
        head = head_event_indices(clip.onset[rows], t)
        out.append(
            {
                "clip_id": clip.entry.clip_id,
                "entry": {
                    k: None if v is None else str(v)
                    for k, v in asdict(clip.entry).items()
                },
                "time": t.tolist(),
                "truth": np.asarray(clip.truth[rows], float).round(4).tolist(),
                "signal": np.asarray(clip.signal[rows], float).round(4).tolist(),
                "onset": np.asarray(clip.onset[rows], float).round(4).tolist(),
                "truth_inhale": truth_inhale.tolist(),
                "inhale": t[head].tolist(),
                "scores": {
                    k: v for k, v in clip.scores.items() if isinstance(v, int | float)
                },
            }
        )
    return {"epoch": epoch, "clips": out}


def main(argv: list[str] | None = None) -> None:
    spec = json.loads((sys.argv[1:] if argv is None else argv)[0])
    module = importlib.import_module(COMMANDS[spec["command"]])
    kwargs = {}
    wanted = spec.get("snippets")
    if spec["command"] == "run" and wanted:
        stdout = sys.stdout

        def on_scored(epoch: int, clips) -> None:
            payload = snippets(epoch, clips, wanted["clips"], wanted["seconds"])
            stdout.write(MARK + json.dumps(payload) + "\n")
            stdout.flush()

        kwargs["on_scored"] = on_scored
    module.main(spec["argv"], **kwargs)


if __name__ == "__main__":
    main()

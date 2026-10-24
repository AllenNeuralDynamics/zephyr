"""Where the corpus' sessions go: every packaged session, step by step, to a set.

Reads a dataset folder (local) and its spec: ``inventory.parquet``, ``qc.parquet``,
``selection.json`` and ``folds.parquet`` of its ``raw/``. A session left out is counted
under the first rule it fails, in the order the steps apply them (``qc`` names one
reason per session).
"""

import json
from pathlib import Path

import matplotlib.patches as mpatches
import pandas as pd
from matplotlib.path import Path as MPath

from zephyr.corpus.config import DatasetSpec
from zephyr.corpus.inventory import wanted

from . import style

ROOT_ENV = "ZEPHYR_DATASET_ROOT"
"""A local copy of the dataset folder; the notebook offers it as the default."""
SETS = {
    "train": "train",
    "val": "validation",
    "test_a": "test A (new animals)",
    "test_b": "test B (new sessions)",
    "test_c": "test C (359 Hz)",
}
LOSS = "#c8c8c8"
KEPT = "#9bb7d4"


def _qc_reason(row, spec: DatasetSpec) -> tuple[str, str]:
    """(label, spec rule) of the first rule a session fails in ``qc``."""
    q = spec.qc
    reason = row["reason"] or ""
    if row["reviewed"]:
        return "excluded by review", "[[review]]"
    if reason.startswith("SNR"):
        return "dead sensor (low SNR mode)", f"min_snr_gap_db = {q.min_snr_gap_db:g}"
    if reason.startswith("amplitude"):
        return (
            f"weak signal (amplitude < {q.min_amplitude:g})",
            f"min_amplitude = {q.min_amplitude:g}",
        )
    if reason.startswith("alternation"):
        return (
            f"events fail to alternate (> {100 * q.max_session_breaks:g}%)",
            f"max_session_breaks = {q.max_session_breaks:g}",
        )
    if reason.startswith("inconsistent"):
        return (
            f"inconsistent events (z > {q.outlier_z:g})",
            f"outlier_z = {q.outlier_z:g}",
        )
    if row["rail_frac"] > 0:
        return (
            f"no strong, unclipped {q.min_span_s:g} s run",
            f"rails = {list(q.rails)}, max_rail_frac = {q.max_rail_frac:g}",
        )
    return (
        f"no strong, clean {q.min_span_s:g} s run",
        (
            f"min_amplitude = {q.min_amplitude:g}, max_outside = {q.max_outside:g}, "
            f"max_breaks = {q.max_breaks:g} per window"
        ),
    )


def flow(root: Path, spec: DatasetSpec) -> list[dict]:
    """The steps from the packaged sessions to the sets: per step its name, the
    sessions and animals it keeps, and what it drops as ``(label, rule, sessions)``."""
    inventory = pd.read_parquet(root / "raw" / "inventory.parquet")
    kept = wanted(spec, inventory)
    qc = pd.read_parquet(root / "raw" / "qc.parquet").set_index("session_id")
    results = json.loads((root / "raw" / "selection.json").read_text(encoding="utf-8"))
    sessions = pd.DataFrame(results["sessions"]).set_index("session_id")
    sessions = sessions.join(qc[["amplitude", "rail_frac"]])
    folds = pd.read_parquet(root / "raw" / "folds.parquet").set_index("session_id")
    subject = inventory.set_index("session_id")["subject_id"].astype(str)
    counts = results["counts"]

    def animals(ids) -> int:
        return int(subject.reindex(list(ids)).nunique())

    q, sel = spec.query, spec.select
    old = len(inventory) - len(kept)
    steps = [
        {
            "name": "packaged",
            "sessions": len(inventory),
            "animals": animals(inventory.session_id),
            "lost": [],
        },
        {
            "name": "query",
            "sessions": len(kept),
            "animals": animals(kept.session_id),
            "lost": [
                (
                    (
                        f"packaged before dataset {q.min_dataset_version} "
                        f"({len(q.include_sessions)} ephys sessions kept by hand)"
                    ),
                    f"min_dataset_version = {q.min_dataset_version}",
                    old,
                )
            ],
        },
    ]
    out = sessions[sessions["selection"] != "include"]
    reasons = pd.DataFrame(
        [_qc_reason(r, spec) for _, r in out.iterrows()], columns=["label", "rule"]
    )
    lost = reasons.value_counts(sort=True).reset_index(name="n")
    steps.append(
        {
            "name": "thermistor QC",
            "sessions": counts["passed"],
            "animals": animals(results["passed"]),
            "lost": [(r.label, r.rule, int(r.n)) for r in lost.itertuples()],
        }
    )
    no_fit = counts["passed"] - counts["clips_fit"]
    no_cams = counts["clips_fit"] - counts["dropped_for_cameras"] - counts["selected"]
    camera_lost = [
        (
            f"{sel.clips_per_session} clips of {sel.clip_s:g} s do not fit",
            f"clip_s = {sel.clip_s:g}, clips_per_session = {sel.clips_per_session}",
            no_fit,
        ),
        ("a camera missing", "require_cameras = true", no_cams),
        (
            "camera timestamps not clean",
            "cameras.parquet",
            counts["dropped_for_cameras"],
        ),
    ]
    steps.append(
        {
            "name": "clips & cameras",
            "sessions": counts["selected"],
            "animals": animals(results["selected"]),
            "lost": [x for x in camera_lost if x[2]],
        }
    )
    f = spec.folds
    used = folds[folds["set"].notna()]
    dark = folds.index.isin(f.exclude) & folds["set"].isna()
    spare = folds[folds["set"].isna() & ~dark]
    caps = {
        "train": (
            (
                f"training animals: over {f.cap_train} per animal and rig "
                f"(test B first: up to {f.cap_test_b}, 1 in {f.test_b_one_in})"
            ),
            (
                f"cap_train = {f.cap_train}, cap_test_b = {f.cap_test_b}, "
                f"test_b_one_in = {f.test_b_one_in}, train_keep = {f.train_keep}"
            ),
        ),
        "val": (
            f"validation animals: over {f.cap_val} per animal and rig",
            f"cap_val = {f.cap_val}",
        ),
        "test_a": (
            f"test A animals: over {f.cap_test} per animal and rig",
            f"cap_test = {f.cap_test}",
        ),
        "group_c": (
            "group C animals: their 240 Hz sessions beyond the caps",
            f"cap_train = {f.cap_train}, group_c",
        ),
    }
    by_role = spare["role"].value_counts()
    steps.append(
        {
            "name": "split",
            "sessions": len(used),
            "animals": int(used["subject_id"].nunique()),
            "lost": [(*caps[r], int(by_role.get(r, 0))) for r in caps]
            + [
                (
                    "taken out after the draw (dark video, rejected trace)",
                    "[folds] exclude",
                    int(dark.sum()),
                )
            ],
            "sets": [
                (
                    SETS[k],
                    int((used["set"] == k).sum()),
                    used.loc[used["set"] == k, "subject_id"].nunique(),
                )
                for k in SETS
            ],
        }
    )
    for s in steps:
        s["lost"] = [x for x in s["lost"] if x[2]]
    return steps


def table(steps: list[dict]) -> pd.DataFrame:
    """Every loss: the step, why, the spec's rule, and the sessions it drops."""
    rows = [
        {"step": s["name"], "reason": label, "rule": rule, "sessions": n}
        for s in steps
        for label, rule, n in s["lost"]
    ]
    return pd.DataFrame(rows)


def _band(ax, x0, top0, bot0, x1, top1, bot1, colour, alpha=1.0) -> None:
    """A band from [bot0, top0] at x0 to [bot1, top1] at x1, eased in between."""
    mid = (x0 + x1) / 2
    verts = [
        (x0, top0),
        (mid, top0),
        (mid, top1),
        (x1, top1),
        (x1, bot1),
        (mid, bot1),
        (mid, bot0),
        (x0, bot0),
        (x0, top0),
    ]
    codes = [MPath.MOVETO, *[MPath.CURVE4] * 3, MPath.LINETO, *[MPath.CURVE4] * 3]
    codes.append(MPath.CLOSEPOLY)
    ax.add_patch(
        mpatches.PathPatch(MPath(verts, codes), fc=colour, ec="none", alpha=alpha)
    )


def flow_figure(steps: list[dict]):
    """The sessions as a band that narrows step by step; each loss peels off below,
    labelled with its reason and count, and the last step splits into the sets."""
    fig, ax = style.figure("double", 0.62)
    total = steps[0]["sessions"]
    gap = 0.012 * total
    row = 0.032 * total  # the height of one label
    width = 1.0
    y_low = 0.0
    for i, s in enumerate(steps):
        x = i * width
        n = s["sessions"]
        ax.add_patch(mpatches.Rectangle((x - 0.04, -n), 0.08, n, fc="0.35", ec="none"))
        ax.text(
            x,
            gap,
            f"{s['name']}\n{n:,} sessions\n{s['animals']} animals",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
        if i + 1 == len(steps):
            break
        nxt = steps[i + 1]
        _band(
            ax,
            x + 0.04,
            0,
            -nxt["sessions"],
            x + width - 0.04,
            0,
            -nxt["sessions"],
            KEPT,
        )
        y = -nxt["sessions"]
        sink = -n - 3 * gap
        for label, _rule, k in nxt["lost"]:
            _band(ax, x + 0.04, y, y - k, x + 0.62 * width, sink, sink - k, LOSS)
            ax.add_patch(
                mpatches.Rectangle(
                    (x + 0.62 * width, sink - k), 0.02, k, fc="0.55", ec="none"
                )
            )
            ax.text(
                x + 0.66 * width,
                sink - k / 2,
                f"{label}: {k:,}",
                va="center",
                fontsize=5.5,
            )
            y -= k
            sink -= max(k, row) + gap
        y_low = min(y_low, sink)
    last = steps[-1]
    x = (len(steps) - 1) * width
    y, sink = 0.0, 0.0
    colours = style.series(len(last["sets"]))
    for (name, k, a), colour in zip(last["sets"], colours, strict=True):
        _band(ax, x + 0.04, y, y - k, x + 0.55 * width, sink, sink - k, colour, 0.8)
        ax.add_patch(
            mpatches.Rectangle(
                (x + 0.55 * width, sink - k), 0.02, k, fc=colour, ec="none"
            )
        )
        ax.text(
            x + 0.6 * width,
            sink - k / 2,
            f"{name}: {k} sessions, {a} animals",
            va="center",
            fontsize=5.5,
        )
        y -= k
        sink -= max(k, row) + 0.5 * gap
    ax.set_xlim(-0.3, len(steps) - 1 + 1.25)
    ax.set_ylim(y_low - gap, 0.16 * total)
    ax.set_axis_off()
    return fig

"""``readme``: the dataset folder's ``README.md`` and ``raw/README.md``.

Filled from ``templates/`` with the counts of ``raw/inventory.parquet``,
``raw/selection.json``, ``sessions.parquet`` and ``clips.toml``, so they describe the
folder as it is. Run it after ``cross_validation`` (and again after ``clips``).
"""

import json
from importlib import resources

import pandas as pd
import tomlkit

from . import storage
from .config import DatasetSpec, Layout

FLOW = [  # (count, what, where)
    ("inventoried", "packaged sessions found", "inventory.parquet"),
    ("scored", "kept by the query", "thermistor/, sessions.toml, qc.parquet"),
    ("passed", "pass thermistor QC", "qc.parquet"),
    ("clips_fit", "fit their clips in good stretches", "selection.json"),
    ("with_cameras", "have the required cameras' videos", "selection.json (cameras)"),
    ("selected", "have clean camera timestamps", "cameras.parquet, selection.json"),
    ("assigned", "are given a set", "folds.parquet, ../sessions.parquet"),
]


def template(name: str) -> str:
    return (resources.files(__package__) / "templates" / name).read_text("utf-8")


def flow(counts: dict[str, int]) -> str:
    rows = [(f"{counts[k]:,}", what, where) for k, what, where in FLOW]
    w0 = max(len(r[0]) for r in rows)
    w1 = max(len(r[1]) for r in rows)
    lines = [f"{n:>{w0}}  {what:<{w1}}  {where}" for n, what, where in rows]
    return "```\n" + "\n".join(lines) + "\n```"


def cameras(layout: Layout) -> str:
    """Sessions and clips of each camera in ``clips.toml``, as a Markdown table."""
    doc = tomlkit.parse(layout.clip_list.read_text(encoding="utf-8"))
    clips = pd.DataFrame(
        {
            "session": [str(e["group"]) for e in doc["clip"]],
            "camera": [
                str(e["video"]).rsplit("_", 1)[-1].split(".")[0] for e in doc["clip"]
            ],
        }
    )
    by = clips.groupby("camera").agg(
        sessions=("session", "nunique"), clips=("session", "size")
    )
    rows = [f"| {c} | {r.sessions:,} | {r.clips:,} |" for c, r in by.iterrows()]
    return "\n".join(["| camera | sessions | clips |", "| --- | --- | --- |", *rows])


def counts(spec: DatasetSpec, layout: Layout) -> dict[str, int]:
    selection = json.loads(layout.selection.read_text(encoding="utf-8"))
    c = selection["counts"]
    sessions = storage.read_parquet(layout.session_table)
    return {
        "inventoried": len(
            storage.read_parquet(layout.inventory, columns=["session_id"])
        ),
        "scored": c["sessions"],
        "passed": c["passed"],
        "clips_fit": c["clips_fit"],
        "with_cameras": c["selected"] + c["dropped_for_cameras"],
        "no_video": c["clips_fit"] - c["selected"] - c["dropped_for_cameras"],
        "dropped": c["dropped_for_cameras"],
        "selected": c["selected"],
        "assigned": len(sessions),
        "animals": sessions["subject_id"].nunique(),
        "n_folds": spec.cross_validation.n_folds,
    }


def run(spec: DatasetSpec) -> None:
    if spec.cross_validation is None:
        raise SystemExit("readme: the spec has no [cross_validation]")
    layout = Layout(spec.root)
    for path, step in [
        (layout.session_table, "cross_validation"),
        (layout.clip_list, "clips"),
    ]:
        if not path.exists():
            raise SystemExit(f"readme: no {path.name} (run {step} first)")
    n = counts(spec, layout)
    fields = {k: f"{v:,}" for k, v in n.items()}
    storage.write_text(
        layout.readme,
        template("dataset.md").format(
            **fields, last_fold=n["n_folds"] - 1, cameras=cameras(layout)
        ),
    )
    storage.write_text(
        layout.raw_readme, template("raw.md").format(**fields, flow=flow(n))
    )
    print(f"readme: wrote {layout.readme} and {layout.raw_readme}")

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import json
    import os
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    from zephyr.config import load
    from zephyr.corpus import (
        fetch,
        inventory,
        lists,
        qc,
        select,
        status,
        storage,
        videos,
    )
    from zephyr.corpus.config import DatasetSpec, Layout

    REPO = Path(__file__).resolve().parents[1]
    SPEC = REPO / "packages/zephyr-dataset/configs/vr-foraging-sniff.toml"


@app.cell
def _():
    mo.md("""
    # Rebuild the corpus

    Runs the default `zephyr-dataset` pipeline (`inventory`, `fetch`, `lists`, `qc`,
    `select`, `videos`; not `clips`, which copies terabytes) from the spec into a new
    local folder (`qc` reads local files), then compares it with the corpus the spec
    names. The DocDB return is copied from that corpus (`docdb.json.gz`) and the
    spec's `until` drops later assets, so the rebuild reads the same metadata; the
    spec's `[[review]]` entries restore the decisions made by hand.
    """)
    return


@app.cell
def _():
    spec_path = mo.ui.text(value=str(SPEC), label="Spec", full_width=True)
    out_dir = mo.ui.text(
        value=str(Path.home() / "zephyr-dataset-rebuild"),
        label="Rebuild into (local)",
        full_width=True,
    )
    profile = mo.ui.text(value="IAmAnywhere", label="AWS profile of an s3:// corpus")
    workers = mo.ui.number(start=1, stop=64, value=16, label="S3 threads")
    mo.vstack([spec_path, out_dir, profile, workers])
    return out_dir, profile, spec_path, workers


@app.cell
def _(out_dir, profile, spec_path):
    if profile.value:
        os.environ["AWS_PROFILE"] = profile.value
    reference = load(DatasetSpec, Path(spec_path.value))
    rebuilt = reference.model_copy(update={"root": str(Path(out_dir.value).resolve())})
    mo.stop(
        rebuilt.root == reference.root,
        mo.callout(
            mo.md("Choose a folder other than the corpus itself."), kind="danger"
        ),
    )
    mo.md(f"Reference corpus: `{reference.root}`  \nRebuild: `{rebuilt.root}`")
    return rebuilt, reference


@app.cell
def _():
    run = mo.ui.run_button(label="Rebuild (about an hour)")
    run
    return (run,)


@app.cell
def _(rebuilt, reference, run, workers):
    mo.stop(not run.value)
    layout = Layout(rebuilt.root)
    layout.root.mkdir(parents=True, exist_ok=True)
    saved = Layout(reference.root).docdb
    if saved.exists() and not layout.docdb.exists():
        storage.write_bytes(layout.docdb, saved.read_bytes())
    with mo.status.spinner(title="inventory") as spinner:
        inventory.build(rebuilt, workers=workers.value)
        spinner.update(title="fetch (raw Harp files from S3)")
        fetch.run(rebuilt, workers=workers.value)
        spinner.update(title="lists")
        lists.run(rebuilt)
        spinner.update(title="qc")
        qc.run(rebuilt)
        spinner.update(title="select (camera folders on S3)")
        select.run(rebuilt, workers=workers.value)
        spinner.update(title="videos (camera timestamps from S3)")
        videos.run(rebuilt, workers=workers.value)
        spinner.update(title="select again (drops sessions with unclean cameras)")
        results = select.run(rebuilt, workers=workers.value)
    return (results,)


@app.cell
def _(rebuilt, reference, results):
    ref = json.loads(Layout(reference.root).results.read_text(encoding="utf-8"))

    def by_session(found):
        return {s["session_id"]: s for s in found["sessions"]}

    old, new = by_session(ref), by_session(results)
    keys = ("selection", "reviewed", "included", "excluded", "clips", "cameras")
    differ = sorted(
        sid
        for sid in old.keys() | new.keys()
        if sid not in old
        or sid not in new
        or any(old[sid][k] != new[sid][k] for k in keys)
        or old[sid].get("dropped") != new[sid].get("dropped")
    )
    inv_old = storage.read_parquet(Layout(reference.root).inventory)
    inv_new = storage.read_parquet(Layout(rebuilt.root).inventory)
    summary = pd.DataFrame(
        {
            "reference": [
                len(inv_old),
                ref["counts"]["passed"],
                ref["counts"].get("dropped_for_cameras"),
                ref["counts"]["selected"],
                "",
            ],
            "rebuild": [
                len(inv_new),
                results["counts"]["passed"],
                results["counts"]["dropped_for_cameras"],
                results["counts"]["selected"],
                len(differ),
            ],
        },
        index=[
            "inventoried",
            "passed",
            "dropped for cameras",
            "selected",
            "sessions that differ",
        ],
    )
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    "The rebuild matches the corpus."
                    if not differ and ref["selected"] == results["selected"]
                    else f"{len(differ)} session(s) differ: {differ[:20]}"
                ),
                kind="success" if not differ else "warn",
            ),
            summary,
        ]
    )
    return


@app.cell
def _(rebuilt, results):
    status.run(rebuilt)
    frame = pd.DataFrame(results["sessions"])
    frame["mark"] = frame["selection"].fillna("unset")
    pd.crosstab(frame["rig"].fillna("?"), frame["mark"], margins=True)
    return


@app.cell
def _(rebuilt, results):
    cams = pd.read_parquet(Layout(rebuilt.root).cameras)
    failed = cams["error"].notna()
    clean = cams["clean"].fillna(False).astype(bool)
    repaired = cams["repaired_rows"].fillna(0) > 0
    per_camera = (
        pd.DataFrame(
            {
                "cameras": cams.groupby("camera").size(),
                "clean": cams[clean].groupby("camera").size(),
                "clean after repair": cams[clean & repaired].groupby("camera").size(),
                "not clean": cams[~clean & ~failed].groupby("camera").size(),
                "failed": cams[failed].groupby("camera").size(),
            }
        )
        .fillna(0)
        .astype(int)
    )
    dropped = pd.DataFrame(
        [
            {
                "session_id": s["session_id"],
                "subject": s["subject_id"],
                "rig": s["rig"],
                "date": s["date"],
                "why": "; ".join(s["dropped"]),
            }
            for s in results["sessions"]
            if s.get("dropped")
        ]
    )
    mo.vstack(
        [
            mo.md("## Camera timestamps"),
            per_camera,
            mo.md(f"{len(dropped)} session(s) dropped for their cameras"),
            dropped,
            mo.md(f"{int(failed.sum())} camera(s) failed (retried by the next run)"),
            cams.loc[failed, ["session_id", "camera", "error"]],
        ]
    )
    return


if __name__ == "__main__":
    app.run()

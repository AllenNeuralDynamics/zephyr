import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import json
    import shutil
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    from zephyr.config import load
    from zephyr.corpus import fetch, inventory, lists, qc, select, status
    from zephyr.corpus.config import DatasetSpec, Layout

    REPO = Path(__file__).resolve().parents[1]
    SPEC = REPO / "packages/zephyr-dataset/configs/vr-foraging-sniff.toml"


@app.cell
def _():
    mo.md("""
    # Rebuild the corpus

    Runs the default `zephyr-dataset` pipeline (`inventory`, `fetch`, `lists`, `qc`,
    `select`) from the spec into a new folder, then compares it with the corpus the
    spec names. The DocDB return is copied from that corpus (`docdb.json.gz`) and the
    spec's `until` drops later assets, so the rebuild reads the same metadata; the
    spec's `[[review]]` entries restore the decisions made by hand.
    """)
    return


@app.cell
def _():
    spec_path = mo.ui.text(value=str(SPEC), label="Spec", full_width=True)
    out_dir = mo.ui.text(
        value=str(load(DatasetSpec, SPEC).root.with_name("vr-foraging-sniff-rebuild")),
        label="Rebuild into",
        full_width=True,
    )
    workers = mo.ui.number(start=1, stop=64, value=16, label="S3 threads")
    mo.vstack([spec_path, out_dir, workers])
    return out_dir, spec_path, workers


@app.cell
def _(out_dir, spec_path):
    reference = load(DatasetSpec, Path(spec_path.value))
    rebuilt = reference.model_copy(update={"root": Path(out_dir.value)})
    mo.stop(
        rebuilt.root.resolve() == reference.root.resolve(),
        mo.callout(
            mo.md("Choose a folder other than the corpus itself."), kind="danger"
        ),
    )
    mo.md(f"Reference corpus: `{reference.root}`  \nRebuild: `{rebuilt.root}`")
    return rebuilt, reference


@app.cell
def _():
    run = mo.ui.run_button(label="Rebuild (about 15 minutes)")
    run
    return (run,)


@app.cell
def _(rebuilt, reference, run, workers):
    mo.stop(not run.value)
    layout = Layout(rebuilt.root)
    layout.root.mkdir(parents=True, exist_ok=True)
    saved = Layout(reference.root).docdb
    if saved.exists() and not layout.docdb.exists():
        shutil.copy(saved, layout.docdb)
    with mo.status.spinner(title="inventory") as spinner:
        inventory.build(rebuilt, workers=workers.value)
        spinner.update(title="fetch (raw Harp files from S3)")
        fetch.run(rebuilt, workers=workers.value)
        spinner.update(title="lists")
        lists.run(rebuilt)
        spinner.update(title="qc")
        qc.run(rebuilt)
        spinner.update(title="select (camera folders on S3)")
        results = select.run(rebuilt, workers=workers.value)
    return (results,)


@app.cell
def _(rebuilt, reference, results):
    ref = json.loads(Layout(reference.root).results.read_text())

    def by_session(found):
        return {s["session_id"]: s for s in found["sessions"]}

    old, new = by_session(ref), by_session(results)
    keys = ("selection", "reviewed", "included", "excluded", "clips", "camera")
    differ = sorted(
        sid
        for sid in old.keys() | new.keys()
        if sid not in old
        or sid not in new
        or any(old[sid][k] != new[sid][k] for k in keys)
    )
    inv_old = pd.read_parquet(Layout(reference.root).inventory)
    inv_new = pd.read_parquet(Layout(rebuilt.root).inventory)
    summary = pd.DataFrame(
        {
            "reference": [
                len(inv_old),
                ref["counts"]["passed"],
                ref["counts"]["selected"],
                "",
            ],
            "rebuild": [
                len(inv_new),
                results["counts"]["passed"],
                results["counts"]["selected"],
                len(differ),
            ],
        },
        index=["inventoried", "passed", "selected", "sessions that differ"],
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


if __name__ == "__main__":
    app.run()

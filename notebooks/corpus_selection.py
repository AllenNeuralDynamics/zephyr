import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")

with app.setup:
    import os
    import tomllib
    from pathlib import Path

    import marimo as mo

    from utils import corpus_flow, style
    from zephyr.corpus.config import DatasetSpec

    style.use_style()
    SPEC = (
        Path(__file__).parent.parent
        / "packages/zephyr-dataset/configs/vr-foraging-sniff.toml"
    )


@app.cell
def _():
    mo.md("""
    # Corpus selection
    """)
    return


@app.cell
def _():
    root_text = mo.ui.text(
        value=os.environ.get(corpus_flow.ROOT_ENV, "C:/Data/zephyr-local-v2/dataset"),
        label="Dataset folder (local copy)",
        full_width=True,
    )
    root_text
    return (root_text,)


@app.cell
def _(root_text):
    root = Path(root_text.value)
    needed = [
        "raw/inventory.parquet",
        "raw/qc.parquet",
        "raw/selection.json",
        "raw/folds.parquet",
    ]
    absent = [n for n in needed if not (root / n).is_file()]
    mo.stop(bool(absent), mo.md("Missing from the folder: " + ", ".join(absent)))
    with SPEC.open("rb") as handle:
        spec = DatasetSpec.model_validate(tomllib.load(handle))
    steps = corpus_flow.flow(root, spec)
    return (steps,)


@app.cell
def _():
    mo.md("""
    ## Where the sessions go
    """)
    return


@app.cell
def _(steps):
    corpus_flow.flow_figure(steps)
    return


@app.cell
def _():
    mo.md("""
    ## The rules
    """)
    return


@app.cell
def _(steps):
    mo.ui.table(corpus_flow.table(steps), selection=None, page_size=20)
    return


if __name__ == "__main__":
    app.run()

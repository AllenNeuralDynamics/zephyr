# Figures

Each figure is assembled from panels drawn at their printed size.

| File | What |
| --- | --- |
| `layout.toml` | every panel's size (mm), kind and content; the one place sizes are set |
| `lib.typ` | places a figure's panels row by row, adds the letters, draws missing panels as placeholders |
| `figN.typ` | one figure: which panels exist yet; compiles to `figN.pdf` |
| `figN/<panel>.svg` | the panels: data panels written by the notebook, schematics drawn by hand; the `.typ` embeds them as vectors |

## Workflow

1. Set or change a panel's size in `layout.toml`.
2. Data panels: run `notebooks/manuscript_figures.py` (from `notebooks/`:
   `uv run python manuscript_figures.py`, or `uv run marimo edit manuscript_figures.py`).
   It reads the layout, draws each panel with `notebooks/utils/style.py` and writes
   `figN/<panel>.svg`. Runs come from the results manifest and clips from the local
   dataset copy (`ZEPHYR_TEST_ROOT`), as for the walkthrough.
3. Schematics: draw in Inkscape on a page of the panel's size, text in Arial 6–7 pt, and
   save as plain SVG to `figN/<panel>.svg`. Never let the notebook write a schematic's name.
4. Add the panel to `files` in `figN.typ`, then from `manuscript/`:
   `typst compile figures/figN.typ` (the PDF to submit) or `typst compile main.typ`
   (the manuscript, which imports the same figure).

Nature Methods: 89 mm (one column) or 183 mm (two columns) wide, sans-serif text 5–7 pt,
vector where possible.

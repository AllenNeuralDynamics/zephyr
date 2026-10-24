# Notebooks

[marimo](https://marimo.io) notebooks for plotting and exploring runs and
benchmark results. They are plain `.py` files, so they diff and review like code.

```bash
uv run marimo edit notebooks/walkthrough.py     # edit interactively
uv run marimo run notebooks/walkthrough.py      # serve as a read-only app
uv run marimo export html notebooks/walkthrough.py -o walkthrough.html --no-include-code
```

- `walkthrough.py`: the benchmark on the corpus's test sets A, B and C. One test clip
  from crop to breathing trace, every method on it (with how each works), then every
  method on every test clip: scores with bootstrap intervals over animals, by animal,
  by breathing rate and how alike their per-clip scores are; breathing-rate agreement (MAE, Bland-Altman);
  inhale timing; inference wall time on the GPU; the augmentation variant against
  Zephyr; examples; and Zephyr's worst sessions with every missed and false inhale and
  what was happening there. Which runs, under which labels, is the results manifest
  `packages/zephyr-benchmarks/configs/results/corpus-v3.toml` (`ZEPHYR_RESULTS` names
  another); the measures are `zephyr.benchmarks.analysis`. The runs come from
  `s3://aind-scratch-data/vr-foraging/zephyr-runs/` (read unsigned, no credentials: the bucket is public),
  downloaded once into `cache/runs/`; the clips
  from a local copy of the dataset and its feature cache (`ZEPHYR_TEST_ROOT`, train and
  validation clips included for the rate distribution).
  Scores are the runs' own `evaluation.json`; traces are the runs' own `traces/` (and
  the pixel methods'), downloaded, where a run saved them, else predicted here (GPU,
  inference only); either way cached in `cache/walkthrough/`. A network's inhales come
  from its onset head at a threshold chosen on its own validation clips, as training now
  chooses one (`ZEPHYR_HEAD_THRESHOLD=default` keeps 0.5); what is chosen, and why, is
  in `cache/walkthrough/head-validation/head-thresholds.json`. The clip-by-clip sections and the wall
  times read only that cache: the notebook never predicts a whole test set or times a
  method, so a method missing from the cache is left out of them (the traces on the
  shown clip are still predicted, one clip each).
- `corpus.py`: reruns the default `zephyr-dataset` pipeline
  (`packages/zephyr-dataset/configs/vr-foraging-sniff.toml`) up to `select` into a new
  folder, from the corpus's saved DocDB return, and compares the selection with the
  corpus (not `folds`, which would draw a new split: `raw/folds.parquet` is the record).
  It writes only that new folder, and only when its Rebuild button is pressed
  (about an hour; the raw thermistors and camera timestamps come from S3).
- `manuscript_figures.py`: every data panel of the manuscript's figures
  (`manuscript/figures/<figure>/<panel>.svg`, at the sizes in
  `manuscript/figures/layout.toml`) and the tables it reads (`manuscript/tables/`),
  from the same runs, clips and cache as `walkthrough.py`, and the quality-control
  reports for Fig. 5 (`ZEPHYR_MANUSCRIPT_QC`).
- `quality_control.py`: `zephyr quality-control` checked where the truth is known.
  Every measure of each video's report against its real inhale F1 (per video), how well
  each ranks videos and windows, how much keeping the best-judged videos raises the
  mean F1, one video's trace and onset head and its measures over time, and
  the reference's onset calibration. It reads one folder of reports and, optionally,
  its reference (local or `s3://`, given in the notebook or by `ZEPHYR_QC_REPORTS` and
  `ZEPHYR_QC_REFERENCE`); `s3://` ones are copied once into `cache/quality-control/`
  (`utils/quality.py`).
- `utils/style.py`: figure sizes, fonts and one fixed colour per method, variant
  and set of clips. Every figure calls `style.use_style()` and takes its colours
  from here.
- `utils/plots.py`: reusable figure components. `utils/runs.py`: downloads the
  manifest's runs, predicts and caches their traces, and caches the
  `zephyr.benchmarks.analysis` tables built from them.

Notebooks only read results (`runs/`, `benchmarks/`, `data/`); they never write
into them. Keep notebooks declarative: logic that a second notebook would need
goes into `utils/`.

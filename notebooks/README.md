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
  by frequency band and by breathing rate; breathing-rate agreement (MAE, Bland-Altman);
  inhale timing; inference wall time on the GPU; the augmentation variant against
  Zephyr; examples; and Zephyr's worst sessions with every missed and false inhale and
  what was happening there. Which runs, under which labels, is the results manifest
  `packages/zephyr-benchmarks/configs/results/corpus-v3.toml` (`ZEPHYR_RESULTS` names
  another); the measures are `zephyr.benchmarks.analysis`. The runs come from
  `s3://aind-scratch-data/vr-foraging/zephyr-runs/` (read unsigned, no credentials: the bucket is public),
  downloaded once into `cache/runs/`; the clips
  from a local copy of the dataset and its feature cache (`ZEPHYR_TEST_ROOT`, as for
  `test_sets.py`, train and validation clips included for the rate distribution).
  Scores are the runs' own `evaluation.json`; traces are predicted here (GPU, inference
  only) and cached in `cache/walkthrough/`. Predicting every method on every test clip
  and timing them are behind buttons: hours the first time, seconds after.
- `corpus.py`: reruns the default `zephyr-dataset` pipeline
  (`packages/zephyr-dataset/configs/vr-foraging-sniff.toml`) up to `select` into a new
  folder, from the corpus's saved DocDB return, and compares the selection with the
  corpus (not `folds`, which would draw a new split: `folds.parquet` is the record).
  It writes only that new folder, and only when its Rebuild button is pressed
  (about an hour; the raw thermistors and camera timestamps come from S3).
- `corpus_selection.py`: where the corpus' sessions go, from every packaged session to
  the sets, as a flow: each step's losses peel off with their reason and count, each
  session under the first rule it fails, with a table of the spec's rule behind each
  (`utils/corpus_flow.py`). It reads a local copy of the dataset folder
  (`ZEPHYR_DATASET_ROOT`: `inventory.parquet`, `qc.parquet`, `results.json`,
  `folds.parquet`) and the spec.
- `test_sets.py`: networks scored on the split's test sets A, B and C, clip by clip:
  scores per video and per animal, against what is known of each clip (thermistor
  quality, image statistics, rig), and the time offset between each network's
  prediction and the thermistor. It reads a local folder (`utils/test_sets.py`, `LAYOUT`;
  `ZEPHYR_TEST_ROOT`) holding the test lists, their cache entries, the checkpoints and
  one `zephyr evaluate` result per network, made with `ZEPHYR_PATH_MAP` pointing the
  folder's dataset copy at `/data/zephyr-dataset`. Image statistics and lags (inference
  on every clip, behind a button) are cached in `cache/`.
- `utils/style.py`: figure sizes, fonts and one fixed colour per method, channel
  and objective. Every figure calls `style.use_style()` and takes its colours
  from here.
- `utils/plots.py`: reusable figure components. `utils/runs.py`: downloads the
  manifest's runs, predicts and caches their traces, and caches the
  `zephyr.benchmarks.analysis` tables built from them.

Notebooks only read results (`runs/`, `benchmarks/`, `data/`); they never write
into them. Keep notebooks declarative: logic that a second notebook would need
goes into `utils/`.

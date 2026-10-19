# Working in zephyr

Rules for coding agents. Read before changing anything. Details live in each package's
README and in the pydantic models; this file only says what must and must not be done.

## Packages

A uv workspace (`uv sync --all-packages --extra cpu`, or `--extra gpu`; `--extra ui` for
the annotation windows):

| Package | Import | Command | Details |
| --- | --- | --- | --- |
| `zephyr` (released) | `zephyr` | `zephyr` | `packages/zephyr/README.md` |
| `zephyr-benchmarks` | `zephyr.benchmarks` | `zephyr-benchmarks` | `packages/zephyr-benchmarks/README.md` |
| `zephyr-dataset` (Allen-internal) | `zephyr.corpus` | `zephyr-dataset` | `packages/zephyr-dataset/README.md` |

- `zephyr` runs only the zephyr network: it never imports `zephyr.benchmarks` or
  `zephyr.corpus` and never names a comparison method (`tests/test_boundary.py`).
- A new comparison method goes in zephyr-benchmarks. If it needs something from zephyr,
  add a **generic** seam (like `Architecture`, `Runner`, `evaluate`'s `loader`), never a
  special case.
- Core's `zephyr/__init__.py` extends `__path__`; keep it, and never add an
  `__init__.py` at `packages/*/src/zephyr/`.
- `notebooks/` (marimo) only read results and never train (the one exception,
  `corpus.py`, rebuilds the corpus into a folder of its own); every figure uses
  `notebooks/utils/style.py`. Which runs a report compares is a results manifest
  (`zephyr-benchmarks/configs/results/`), never a list in a notebook, and what it
  measures beyond `evaluation.json` lives in `zephyr.benchmarks.analysis`; a notebook
  only caches, downloads and draws.
- Inhale events of a network with an onset head come from the head
  (`head_events`), never from DSP on its trace. DSP is for the thermistor, headless
  methods (labelled "DSP") and exhale F1, or an explicitly labelled comparison.

## Experiments are TOML, not code

Every input (clips, what trains, what is scored, hyperparameters, seeds) is a TOML file
of paths relative to itself, validated by `packages/zephyr/src/zephyr/config.py`
(extended by `zephyr-benchmarks`' `config.py`). A misspelt key is an error. To run a new
experiment, **write or copy a TOML file**. Never:

- add CLI flags for experiment settings (`zephyr preprocess --channels` is not one: it
  chooses which channel files to store, never what any stored value is);
- hardcode paths, split names, camera names, session numbers or dataset layout in `src/`;
- write scripts that bypass `zephyr run` / `evaluate` (or `zephyr-benchmarks run` / `evaluate`);
- edit an existing config to repurpose it: copy it under a new name.

Clip lists and experiments:

- Put all sessions of a dataset in **one clip list**; folds select them with `groups`.
- Create lists with `zephyr clips scan`, place boxes with `zephyr annotate`; never
  hand-edit boxes in bulk.
- Seeds and machine settings belong to the experiment, never to a fold; fold names are
  distinct plain file names.
- To early-stop on held-out clips (e.g. validation animals), give the fold a `[fold.val]`
  selection; it may not overlap training or a test group, so never list it as a test
  group too (its scores at `best.pt` are in `best_per_clip.json` and `history.json`).
- Loading refuses leakage, unlabelled training clips, missing boxes and mixed
  `[preprocess]` recipes, and `evaluate` refuses clips a checkpoint trained on. If one
  refuses, **fix the config, never the check**.
- Comparison networks (`arch` other than `zephyr`) run only through `zephyr-benchmarks`.
- **Always smoke-run** (`--smoke`) before a full run; use a new `output_dir` for a new
  experiment rather than deleting results.
- Never compare scores made before and after a scoring rule changed (time alignment,
  event tolerance; see the zephyr README, "Scoring").

## Quality control (zephyr.quality_control)

Judges predictions on videos without a thermistor (not the corpus's `qc` step, which
scores thermistor sessions). Both steps are TOML (`QualityReference`, `QualityControl`).

- No measure reads a thermistor; a labelled clip's real scores go only into the report's
  separate `truth` block, which exists to check the measures.
- A measure reads only the network's ordinary output (trace, onset probability);
  never add one that needs more network passes or the video (dropout draws, shifted
  crops, other checkpoints, embeddings and image statistics were tried and dropped).
- A reference belongs to one checkpoint (SHA-256; a run refuses another's). Its
  `calibration` clips must be labelled and held out by the checkpoint (refused
  otherwise); never calibrate on a test set the measures are then checked on. Training
  does not write one.
- Judge rates only from above (`breathing_ceiling_hz`): never threshold a rate
  distribution, which changes with running, odours and task.
- A measure is only trusted once checked against held-out animals
  (`notebooks/quality_control.py`); thresholds, when added, go in `QualityParams`.

## Reference configs

`configs/experiments/benchmark-gray-diff-flow-multitask.toml` and `baselines-nets.toml`
(zephyr-benchmarks) hold the settings of the runs in `benchmarks/`, pinned by
`tests/test_reproduction.py`. Copy them; never edit them. Time stretching has been removed
from zephyr, and the benchmark fold's runs were trained with it (`time_stretch = 3`), so
that config no longer reproduces them exactly; `baselines-nets.toml` does.

## The corpus (zephyr-dataset)

Built from one spec (`packages/zephyr-dataset/configs/vr-foraging-sniff.toml`) by
resumable steps; what each does and the folder layout are in its README.

- **Every setting is in the spec**; `[qc]` writes out every threshold, none defaulted.
  Hand decisions go in `[[review]]`, so a rebuild gives the same QC and selection rules.
  The published split is `folds.parquet` (with the clips in `results.json`): it is the
  record, and re-running `select` or `folds` draws a new one, so never re-run them on the
  published corpus.
- `qc` scores whole sessions, whatever marks the list holds, so re-running it never
  changes a metric.
- **The subject always comes from session metadata**, never from a session name.
- **Never hard-code a camera rate**: thresholds are relative to each camera's own
  interval, and clips read rate, time base and keyframes from the stream.
- Clips are **stream-copied, never re-encoded**, and checked before upload.
- A camera that is not clean is reported and its session dropped, never kept silently;
  keep the README's "Known data issues" current. Likewise a session whose thermistor
  device is off the reference Harp clock (`sync`, `[sync]`): `folds` takes it out after
  the draw.
- The split (`folds`): test A and validation hold out **whole animals**; never weaken
  `folds.check`, which refuses a held-out animal in training. Keep the README's "Why these
  sessions" current with any change to the split.
- Face crop boxes are made by `python -m zephyr.corpus.boxes` (template match, your
  corrections in `zephyr annotate --remote`, then `publish`) and live in `clips.toml`;
  `clips` keeps them, and lists the side camera as excluded (`[clips] exclude_cameras`). `python -m zephyr.corpus.rois` is a helper that writes only to a
  local folder, never the dataset or the spec; OWLv2 needs `--extra rois`.
- Cross-validation (`[cross_validation]`, `cross_validation.parquet`) is drawn beside the
  published split and only reads `folds.parquet`; never write folds into
  `folds.parquet`, and never weaken `cross_validation.check_all` (each fold passes
  `folds.check`). `build` does not run it.
- The small dataset (`[small]`, `clips-small.toml`) is a subset of the assigned sessions that
  keeps the split's structure; never run a small experiment's test sets against animals
  its training uses (`folds.check` guards the full split).
- In `clips.toml` the `group` is the session; the animal split lives in `folds.parquet`.
- The root may be remote (`s3://...`, through `storage.py`). Zephyr core reads only
  local files and must not learn about remote paths: `qc`, `zephyr annotate events` and
  `preprocess` run on a local copy (`zephyr annotate --remote` only skips checking that
  a list's clips exist and reads cached frames). Never rename remote objects; delete only with
  `clips --prune`.
- Paths that identify a clip (cache keys, sidecars, `train_videos`, recordings) go
  through `config.canonical`, never `str()` or a raw comparison; a local copy of a
  dataset is named as the original with `ZEPHYR_PATH_MAP` (zephyr README), never by
  moving or renaming cache entries.
- Feature caches store one file per channel group; core reads channels only through
  `features.reader`.

## When code must change

Only when a setting does not exist yet:

1. Add a field to the right pydantic model with a default that keeps current behaviour,
   so existing configs and `test_reproduction.py` pass unchanged (network settings in
   zephyr's `config.py`, comparison-only settings in `BenchmarkTrainParams`).
2. Read it from the model where it is used; no new CLI flag.
3. Add a test; keep module docstrings under 10 lines (`test_module_docstrings.py`); run
   `uv run ruff format . && uv run ruff check . && uv run pytest -q` from the root.

A change of on-disk format (cache, run layout) keeps no backwards compatibility and no
migration code in the packages: existing data is converted once by a throwaway script,
and the code reads only the new format.

Whenever a change touches config models, the CLI, run layout, cache, package boundary
or a rule here, update this file (and the relevant README) in the same change; fix or
remove a rule that became wrong rather than working around it.

## Hard rules

- `data/` and `benchmarks/` are read-only. Write outputs to new directories. The one
  exception: when the scorer itself is corrected, an existing run's `evaluation.json` may
  be re-scored in place, after keeping the original as `evaluation.old-scorer.json`, and
  only if re-scoring its checkpoint with the old scorer reproduces the stored values.
  Checkpoints, `config.json` / `args.json`, histories and traces are never touched.
- Do not commit unless asked.

# Working in zephyr

Rules for coding agents. Read before changing anything. Details live in each package's
README and in the pydantic models; this file only says what must and must not be done.

## Packages

A uv workspace (`uv sync --all-packages --extra cpu`, or `--extra gpu`; `--extra ui` for
the app and its windows):

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
  (`zephyr-benchmarks/configs/results/`), never a list in a notebook (the quality
  notebook reads one reports folder, given in the notebook), and what it
  measures beyond `evaluation.json` lives in `zephyr.benchmarks.analysis`; a notebook
  only caches, downloads and draws.
- Gray frames are the only input, and only zephyr has an onset head: never add a channel
  selection, or an onset head to a comparison network.
- Inhale events of a network with an onset head come from the head
  (`head_event_indices`), never from DSP on its trace. DSP is for the thermistor, headless
  methods (labelled "DSP") and exhale F1, or an explicitly labelled comparison; DSP on a
  predicted trace is always the clip's own event detector (`evaluate.predicted_events`),
  never a detector of its own.
- The head's onset threshold belongs to the checkpoint (`head_threshold` in its state,
  chosen on `[fold.val]` clips by `train_params.head_thresholds`; 0.5 without). Read it
  with `evaluate.head_threshold` / `checkpoint_head_threshold`, never the constant;
  only `--head-threshold` on `evaluate`/`predict` overrides it, and every output
  records the value and its source. Never choose it on a test set. The 50-ms peak
  spacing stays fixed.

## Experiments

Never:

- hardcode paths, split names, camera names, session numbers or dataset layout in `src/`;
- write scripts that bypass `zephyr run` / `evaluate` (or `zephyr-benchmarks run` / `evaluate`);
- edit an existing config to repurpose it: copy it under a new name.

Clip lists and experiments:

- Put all sessions of a dataset in **one clip list**; folds select them with `groups`.
- Create lists with `zephyr clips scan` (`[derive]` flags for other file names), check
  them with `zephyr clips check`, place boxes in the app (`zephyr gui`, Boxes); never hand-edit
  boxes in bulk.
- Seeds and machine settings belong to the experiment, never to a fold; fold names are
  distinct plain file names.
- To early-stop on held-out clips (e.g. validation animals), give the fold a `[fold.val]`
  selection; it may not overlap training or a test group, so never list it as a test
  group too (its scores at `best.pt` are in `best_per_clip.json` and `history.json`).
- Loading refuses a video on both sides of a fold, unlabelled training clips, missing boxes and mixed
  `[preprocess]` recipes, and `evaluate` refuses clips a checkpoint trained on. If one
  refuses, **fix the config, never the check**.
- Comparison networks (`arch` other than `zephyr`) run only through `zephyr-benchmarks`.
- Published results (reports, figures, papers) come from the cross-validation folds
  (`cv_<k>`, from the dataset's `sessions.parquet`) for every method; the first draw
  (`raw/folds.parquet`) is never a fold, only the record of which sessions are used.
- **Always smoke-run** (`--smoke`) before a full run; use a new `output_dir` for a new
  experiment rather than deleting results.
- Never compare scores made before and after a scoring rule changed (time alignment,
  event tolerance, the onset threshold; see the zephyr README, "Scoring").

## The app (zephyr.gui)

`zephyr gui` walks a project folder (zephyr README, "The app") through the same steps
as the commands.

- It **runs only `zephyr` commands**, the ones in `gui/worker.py`'s `COMMANDS`, through
  their own `main(argv)` in a local child process (`gui/jobs.py`); never reimplement a
  step in the app, never call one in the app's own process, never launch remote jobs.
  A step the app needs that no command does is a new command first.
- It **writes only clip lists** (through `zephyr.clips`, as the windows do; a list of
  videos to predict, from `clips scan --checkpoint`, lives in its `predictions/<name>/`
  and gains or loses videos through `update_clip_list`) **and new experiments**
  (`Project.new_experiment`, validated before it is kept, never overwriting); it never
  edits an experiment, and never writes into a run folder outside its project's `runs/`
  (a folder opened from elsewhere, e.g. `benchmarks/`, is read only). Its own settings
  (recent projects) go in the user's Qt settings file, never a project. A folder
  becomes a project (its layout made) only after the user agrees, and the feature
  cache is filled only after saying how much space it takes (`gui/project.py`'s
  `cache_need`).
- **Run** never silently re-scores: an experiment that already has a run is started
  over (`zephyr run --restart`, which deletes that run folder) only after the user
  agrees; scoring a stopped or finished run is Results' button (`--evaluate-only`),
  which asks before replacing an `evaluation.json`.
- Validation snippets reach the app only through `train_fold`'s `on_scored` hook, on the
  worker's stdout, and are **never saved**; the hook never changes training
  (`test_predict.py` runs training with and without it).
- Qt (`PySide6`, `pyqtgraph`) is imported only by `gui/`'s window modules; everything
  else (`gui/project.py`, `gui/jobs.py`, `gui/worker.py`, the logic of each window, the
  commands) imports without the `ui` extra and is tested headless. A window's logic
  lives in a headless class (`annotate.BoxState`, `annotate_events.TuneState`).
- Placing boxes and choosing detectors by hand happen only in the app (`zephyr gui
  <list> --page boxes|breathing`); `zephyr annotate` keeps only what needs no window
  (`--prepare`, `--auto`, `annotate events --auto`). A key or control of the Boxes or
  Breathing page is never dropped (`tests/test_gui_qt.py` checks each against the saved
  clip list, offscreen). A clip may have its own `frame` (`clips frame --input-scale`,
  or Input scale on the Boxes page): its box is in those pixels, it keys its cache entry
  as a list at that size would, and the list still needs one box size (the network sees
  only the crop). The Boxes page shows the list's frame size (chosen on Clips)
  instead of offering it (`frame_fixed`). Opening a project writes nothing; a folder
  is made only by the step that fills it (a dataset's own folder stays as it was).

## Quality control (zephyr.quality_control)

Judges predictions on videos without a thermistor (not the corpus's `qc` step, which
scores thermistor sessions). Both steps take flags, like `zephyr evaluate`.

- No measure reads a thermistor; a labelled clip's real scores go only into the report's
  separate `truth` block, which exists to check the measures.
- A measure reads only the network's ordinary output (trace, onset probability);
  never add one that needs more network passes or the video (dropout draws, shifted
  crops, other checkpoints, embeddings and image statistics were tried and dropped).
- A reference belongs to one checkpoint (SHA-256; a run refuses another's) and is
  calibrated at its onset threshold, which measures use too (a run refuses a reference
  made at another). Its
  `calibration` clips must be labelled and held out by the checkpoint (refused
  otherwise); never calibrate on a test set the measures are then checked on. Training
  does not write one.
- Judge rates only from above (`breathing_ceiling_hz`): never threshold a rate
  distribution, which changes with running, odours and task.
- A measure is only trusted once checked against held-out animals
  (`notebooks/quality_control.py`). Its pass bound lives in `QualityThresholds`, chosen
  on validation animals only (never a test set); every report keeps the value beside
  the pass/fail, and never fails a video as a whole.

## Reference configs

`configs/experiments/baselines-nets.toml` (zephyr-benchmarks) holds the settings of the
runs in `benchmarks/baselines-v1/`, pinned by `tests/test_reproduction.py`. Copy it; never
edit it.

## The corpus (zephyr-dataset)

Built from one spec (`packages/zephyr-dataset/configs/vr-foraging-sniff.toml`) by
resumable steps; what each does and the folder layout are in its README.

- **Every setting is in the spec**; `[qc]` writes out every threshold, none defaulted.
  Hand decisions go in `[[review]]`, so a rebuild gives the same QC and selection rules.
  Which sessions are used is `raw/folds.parquet` (with the clips in `raw/selection.json`):
  it is the record, and re-running `select` or `folds` draws a new one, so never re-run
  them on the published corpus.
- `qc` scores whole sessions, whatever marks the list holds, so re-running it never
  changes a metric.
- **The subject always comes from session metadata**, never from a session name.
- **Never hard-code a camera rate**: thresholds are relative to each camera's own
  interval, and clips read rate, time base and keyframes from the stream.
- Clips are **stream-copied, never re-encoded**, and checked before upload.
- A required camera (`[source] cameras`) that is not clean is reported and its session
  dropped, never kept silently; keep the README's "Known data issues" current. An optional
  camera (`[source] optional_cameras`) never selects or drops a session and never changes
  a clip's thermistor: it is cut only where it is clean and no `[[review]]` drops it
  (`drop_cameras`, whose files `clips --prune` deletes). Likewise a session whose thermistor
  device is off the reference Harp clock (`sync`, `[sync]`): `folds` takes it out after
  the draw.
- The split (`folds`): test A and validation hold out **whole animals**; never weaken
  `folds.check`, which refuses a held-out animal in training. Keep the README's "Why these
  sessions" current with any change to the split.
- Face crop boxes are made by `python -m zephyr.corpus.boxes` (template match, your
  corrections on the app's Boxes page, `zephyr gui boxes/face_camera_snout.toml --page boxes`, then
  `publish`) and live in `clips.toml`;
  `clips` keeps them, and lists the side and front cameras as excluded (`[clips] exclude_cameras`). `python -m zephyr.corpus.rois` is a helper that writes only to a
  local folder, never the dataset or the spec; OWLv2 needs `--extra rois`.
- Cross-validation (`[cross_validation]`) is the published split: it only reads
  `raw/folds.parquet` and writes `sessions.parquet` (the used sessions' columns of it,
  without the first draw's `set` and `role`, and the folds); never write into
  `raw/folds.parquet`, and never weaken `cross_validation.check_all` (each fold passes
  `folds.check`). `build` does not run it, nor `readme`, which writes `README.md` and
  `raw/README.md` from `templates/` (keep them current with the folder).
- The dataset folder publishes `clips/`, `clips.toml`, `thumbnails/`, `sessions.parquet`
  and the READMEs at its top; everything the build reads or records is in `raw/`, which
  training and evaluation never read.
- In `clips.toml` the `group` is the session; the animal split lives in `sessions.parquet`.
- The root may be remote (`s3://...`, through `storage.py`). Zephyr core reads only
  local files and must not learn about remote paths: `qc`, the app's Breathing page and
  `preprocess` run on a local copy (the Boxes page needs only the cached frames, not the
  clips). Never rename remote objects; delete only with
  `clips --prune`, or when a change of layout is converted once by a throwaway script
  (copy, check the copy, then delete the old objects, after the user agrees).
- Paths that identify a clip (cache keys, sidecars, `train_videos`, recordings) go
  through `config.canonical`, never `str()` or a raw comparison; a local copy of a
  dataset is named as the original with `ZEPHYR_PATH_MAP` (zephyr README), never by
  moving or renaming cache entries.
- Feature caches store the gray frames; core reads them only through `features.reader`.

## When code must change

Only when a setting does not exist yet:

1. Add a field to the right pydantic model with a default that keeps current behaviour,
   so existing configs and `test_reproduction.py` pass unchanged (network settings in
   zephyr's `config.py`, comparison-only settings in `BenchmarkTrainParams`).
2. Read it from the model where it is used.
3. Add a test; keep module docstrings under 10 lines (`test_module_docstrings.py`); run
   `uv run ruff format . && uv run ruff check . && uv run pytest -q` from the root.

A change of on-disk format (cache, run layout) keeps no backwards compatibility and no
migration code in the packages: existing data is converted once by a throwaway script,
and the code reads only the new format.

Whenever a change touches config models, the CLI, run layout, cache, package boundary
or a rule here, update this file (and the relevant README) in the same change; fix or
remove a rule that became wrong rather than working around it.

## The documentation site

`docs/` is the user documentation of the `zephyr` package only (never benchmarks or the
corpus build; the tutorial only uses a few of the corpus's clips), built with Zensical
from `zensical.toml` (`uvx zensical serve`; the theme override is `docs/overrides/`,
not published). CI builds it `--strict` and publishes it to GitHub Pages when a release
is published (from its tag) or when run by hand. A change to the
app, a command's flags, a config field or an output file updates its page in the same
change. Screenshots in `docs/assets/screens/` come only from
`scripts/docs_screenshots.py`, which drives the real app through the tutorial's project
(never mock or hand-edit one); re-run it when a page they show changes.

## Hard rules

- `data/` and `benchmarks/` are read-only. Write outputs to new directories. The one
  exception: when the scorer itself is corrected, an existing run's `evaluation.json` may
  be re-scored in place, after keeping the original as `evaluation.old-scorer.json`, and
  only if re-scoring its checkpoint with the old scorer reproduces the stored values.
  Checkpoints, `config.json` / `args.json`, histories and traces are never touched.
- Do not commit unless asked.

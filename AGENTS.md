# Working in zephyr

Instructions for coding agents. Read this before changing anything.

## Two packages, one way

The repository is a uv workspace of two packages (`uv sync --all-packages
--extra cpu`, or `--extra gpu`):

| Package | Path | Import | Command |
| --- | --- | --- | --- |
| `zephyr` | `packages/zephyr/` | `zephyr` | `zephyr` |
| `zephyr-benchmarks` | `packages/zephyr-benchmarks/` | `zephyr.benchmarks` | `zephyr-benchmarks` |

- **`zephyr` runs only the zephyr network.** It never imports
  `zephyr.benchmarks` and never names a comparison method (TS-CAN, PhysNet,
  pixel, Facemap, ...); `packages/zephyr/tests/test_boundary.py` enforces both.
  It is the only released package.
- **`zephyr-benchmarks`** holds everything about our dataset and the methods
  zephyr is compared against: the benchmark configs, the comparison networks
  and baselines, timing, the collected tables and the report. It builds on
  zephyr and is never released.
- A new comparison method goes in zephyr-benchmarks. If it needs zephyr to do
  something new, add a **generic** seam to zephyr (as with `Architecture` in
  `train.py`, `Runner` in `run.py`, the `loader` of `evaluate.py`) that does
  not know the method exists, never a special case.
- `zephyr` lives under one import name across two distributions: core's
  `zephyr/__init__.py` extends `__path__`. Keep that line; never add an
  `__init__.py` at `packages/zephyr-benchmarks/src/zephyr/`.
- **`notebooks/`** holds [marimo](https://marimo.io) notebooks (plain `.py`
  files) for plotting runs and results; marimo is in the `dev` dependency group.
  Notebooks are declarative, only read results, never train, and share
  `notebooks/utils/` (`style.py` fixes figure sizes, fonts and one colour per
  method/channel/objective; reuse it for every figure). They are not an
  experiment path (see below).
- **Inhale events of a network with an onset head come from the head, never
  from DSP on its predicted trace**, in every plot and table: event markers,
  inhale F1, KL-IBI, per-rate analyses (`results.head_events`). DSP
  (`results.events`) is only for the thermistor, for methods without a head
  (labelled "DSP"), and for exhale F1 (no head exists). A DSP view of a headed
  network is allowed only as an explicitly labelled comparison next to the head
  version (e.g. the §5 ablation, the "Local F1 (DSP)" column of §9).

## The rule: experiments are TOML, not code

Every input zephyr uses (which clips exist, what trains, what is scored, with
which hyperparameters and seeds) is a **TOML file of local paths**, validated
by the pydantic models in `packages/zephyr/src/zephyr/config.py` (extended for
the benchmarks by `packages/zephyr-benchmarks/src/zephyr/benchmarks/config.py`).
The benchmark's files live in `packages/zephyr-benchmarks/configs/`;
`examples/` holds one generic file of each type.

To run a new experiment, **write or copy a TOML file. Do not edit Python.**
Specifically, do not:

- add CLI flags for experiment settings (learning rate, epochs, sessions,
  weights, ...);
- hardcode paths, split names (`train`, `test`, `ood`), camera names, session
  numbers or dataset layout anywhere in either package's `src/`;
- write one-off training or evaluation scripts that bypass `zephyr run` /
  `zephyr evaluate` (or `zephyr-benchmarks run` / `evaluate`);
- edit an existing config to repurpose it. Copy it under a new name instead,
  so earlier runs stay reproducible.

Change code only when a setting genuinely does not exist yet (see
[When code does need to change](#when-code-does-need-to-change)).

## The three file types

Paths in every file are **relative to the file they are written in**.
`zephyr schema <dir>` writes JSON schemas of all three for editor validation.
A misspelt key is an error, never silently ignored. Examples below are from
`packages/zephyr-benchmarks/configs/`, four levels below the repository root
where `data/`, `benchmarks/` and `runs/` live.

### Clip list (`configs/clips/*.toml`, model `ClipList`)

Names videos and how to preprocess them.

```toml
[preprocess]
target_size = [360, 270]          # frame size the boxes are measured in

[[clip]]
video = "../../../../data/ood/video_side_right_13_part_1.mp4"
box = [228, 85, 96, 96]           # x, y, w, h in target_size pixels
```

- `timestamps`, `thermistor` and `group` are **derived from the video name** by
  `[derive]` (defaults match the example dataset); give them explicitly only
  when the naming differs. `thermistor = false` marks an unlabelled clip.
- `group` is the recording (session). A *recording* is (video folder, group):
  clips of one recording never go to both train and test.
- Create lists with `zephyr clips scan <dir> --glob 'video_*.mp4' -o <file>`,
  then place boxes with `zephyr annotate <file>` (it writes `box =` back).
  Never hand-edit boxes in bulk with a script.
- `[preprocess]` also sets `select_fs` (rate of the frames the CNN sees) and
  `output_fs` (rate of the grid the network predicts and trains on; both
  default 60 Hz). The TCN's dilations count output samples, so its context in
  seconds scales with `1 / output_fs`. Scoring always resamples to 60 Hz.
  `output_fs` enters the cache key only off its default, so caches made before
  it existed still match.
- Put **all sessions of a dataset in one list**; folds select sessions with
  `groups`. Do not make one file per session or per subset.

### Fold (`configs/folds/*.toml`, model `Fold`; `BenchmarkFold` in zephyr-benchmarks)

What trains, what is scored, and how.

```toml
init_from = "../../../../benchmarks/.../best.pt"   # optional: start from these weights

[[train]]
clips = "../clips/face_train.toml"
weight = 0.5                                 # relative share of training windows

[[train]]
clips = "../clips/ood_side_right.toml"
groups = ["13", "14"]                        # optional: only these recordings
weight = 0.5

[test]                                       # name -> clips; names become report groups
ood_held_out = { clips = "../clips/ood_side_right.toml", groups = ["15"] }
new_animals = "../clips/face_test_new_animals.toml"

[train_params]                               # anything omitted keeps its default
epochs = 20
lr = 1e-4

[train_params.augmentation]
time_stretch = 1.0
```

- Every training knob lives in `TrainParams` / `Augmentation` in zephyr's
  `config.py`; read that file for the full list and defaults.
- `rate_balance` (0 to 1, default 0 = off) draws training windows by their true
  breathing rate, from the clips' stored inhale onsets, over `rate_bins_hz`
  (default 2-15 Hz): each window is weighted by `share(bin) ** -rate_balance`,
  within each `[[train]]` source. Bins under 1% of windows count as 1%. Helpers
  are in `zephyr/rates.py`; validation windows are never balanced.
- `signal_pool` (default 1 = off) makes the signal head read the TCN's output
  average-pooled by that factor in time and upsampled back to 60 Hz (a smoother
  trace). `onset_input` (`full`, the default, `pooled` or `both`) says what the
  onset head reads: the full-rate features, the branch's, or both concatenated;
  the last two need `signal_pool > 1`. Both are recorded in the checkpoint's
  `arch_kwargs`, and both packages' loaders rebuild from them.
- zephyr-benchmarks' `BenchmarkTrainParams` adds `arch` (`zephyr`, the
  default, `tscan` or `physnet`; the last two need `channels = "gray"`) and
  `tscan_img_size`. zephyr's own `Fold` refuses both keys: a fold naming a
  comparison network runs only through `zephyr-benchmarks run`.
- Loading refuses leakage (a test clip sharing a video or recording with
  training), unlabelled training clips, clips without boxes, and lists with
  different `[preprocess]` recipes. If a fold fails to load, fix the fold,
  not the validator.
- `all` is reserved as a test group name.
- Leave-one-out is one fold file per held-out recording (see
  `configs/folds/ood-hold{13,14,15}.toml`); they differ only in `groups`.

### Experiment (`configs/experiments/*.toml`, model `Experiment`; `BenchmarkExperiment`)

Which folds run, with which seeds, and where.

```toml
folds = ["../folds/ood-hold13.toml", "../folds/ood-hold14.toml"]
seeds = [42]                      # each fold runs once per seed
output_dir = "../../../../runs/my-experiment"
features_dir = "../../../../data/features-v2"
device = "cuda"                   # machine settings: optional, CLI overrides
amp = "bf16"
```

Seeds belong to the experiment, never to the fold.

## Workflow

From the repository root, with `C=packages/zephyr-benchmarks/configs`:

```bash
uv run zephyr preprocess $C/clips/<list>.toml --cache data/features-v2
uv run zephyr run $C/experiments/<exp>.toml --smoke --output-dir runs/smoke   # always first
uv run zephyr run $C/experiments/<exp>.toml [--seeds 42]
uv run zephyr evaluate --checkpoint <best.pt> --clips <a>.toml <b>.toml --cache data/features-v2 --out <file>
```

- `zephyr-benchmarks run` / `evaluate` take the same arguments and are needed
  for folds with `arch = "tscan"` or `"physnet"` and their checkpoints;
  `zephyr evaluate` refuses non-zephyr checkpoints. Baselines and tables:
  `zephyr-benchmarks pixel|facemap|timing <fold> --cache <dir>`,
  `zephyr-benchmarks collect --runs <dir>`, `zephyr-benchmarks report`.
- `zephyr-benchmarks occlusion --checkpoint <best.pt> --clips <a>.toml <b>.toml --cache data/features-v2 --out <file.npz>`
  is inference-only analysis (not an experiment): per-channel patch-occlusion
  maps of event F1 on the first `--window-s` seconds of each clip; `--patch`,
  `--stride`, `--window-s` are analysis args.
- **Always smoke-run** (`--smoke`, 1 epoch of 2 steps) before a full run. Smoke
  output goes to `<output_dir>/smoke/` unless `--output-dir` is given.
- A run writes `<output_dir>/<fold stem>/seed-<n>/` with `config.json`,
  `train_videos.json`, checkpoints and `evaluation.json`. With
  `train_params.save_every = N` it also keeps `epoch-<n>.pt` every N epochs
  (same weights a final `best.pt` would hold); score one with `evaluate
  --checkpoint`. These were trained mid-schedule (learning rate not yet
  decayed), so they show how the score evolves, not what a shorter run gives. A seed whose `best.pt`
  exists is re-scored, not retrained; use a new `output_dir` for a new
  experiment rather than deleting results.
- Without a GPU add `--device cpu --amp off --num-workers 0`; expect ~1.7 s
  per training step.
- Standalone `evaluate` reports each `--clips` list as a group named after its
  file stem (`--name` renames a single list). To score selected `groups`
  under chosen names, put them in a fold's `[test]` instead.
- `evaluate` refuses clips a checkpoint (or the checkpoint it was initialised
  from) trained on. Do not work around the refusal.
- Scoring and training targets go by time, never by sample index: a clip's video
  and thermistor need not start together (side-camera clips start up to ~0.9 s
  apart, face clips a few ms). `score_clip` compares the two at the prediction's
  timestamps over the span both cover, and `load_target` holds the thermistor's
  edge value outside its span instead of extrapolating. Correlation against the
  raw thermistor moves by several hundredths for a few ms of misalignment, so
  never compare correlations scored before and after this rule.
- To check that a new config means what you intend before training, load it:
  `uv run python -c "from zephyr.config import load; from zephyr.benchmarks.config import BenchmarkFold; f = load(BenchmarkFold, 'packages/zephyr-benchmarks/configs/folds/x.toml'); print([len(c) for c in f.train_clips()], {k: len(v) for k, v in f.test_clips().items()})"`.

## Reference configs: do not change their meaning

`configs/folds/benchmark-gray-diff-flow-multitask.toml`,
`configs/folds/baseline-{tscan,physnet}.toml` and their experiments (all in
`packages/zephyr-benchmarks/`) reproduce published results;
`packages/zephyr-benchmarks/tests/test_reproduction.py` pins them to the
reference runs' settings. Copy them to experiment; never edit them.

## When code does need to change

Only when an experiment needs a setting that does not exist yet:

1. Add it as a field on the right pydantic model, with a default that keeps
   current behaviour, so every existing config and `test_reproduction.py`
   still pass unchanged. A setting of the zephyr network goes in zephyr's
   `config.py` (`TrainParams`, `Augmentation`, `PreprocessParams`,
   `Selection`, ...); a setting only a comparison method has goes in
   `BenchmarkTrainParams`.
2. Read it from the model where it is used; no new CLI flag.
3. Add a test in that package's `tests/`, keep module docstrings under 10
   lines (excluding CLI examples; each package's `test_module_docstrings.py`
   enforces it), and run
   `uv run ruff format . && uv run ruff check . && uv run pytest -q` from the
   repository root (it tests both packages).

## Keeping this file current

This file describes the infrastructure as it is now, and it will go stale if
nobody maintains it. Whenever a change touches the config models, the CLI, the
run layout, the cache, the package boundary or a workflow described here,
**consider what the change means for this file and update it in the same
change**. That covers, for example: a new or renamed field, file type or
subcommand; a new validation rule; a moved directory; a new reference config.
If a change makes a rule here wrong or obsolete, fix or remove the rule rather
than working around it. When in doubt, mention the implication to the user.

## Hard rules

- **Never read, store or use animal or subject identity** (`subject_id` or
  similar) anywhere: code, configs, logs. Test group names like
  `new_animals` are labels a config author chose, nothing more.
- `data/` and `benchmarks/` are read-only inputs. Write new outputs to new
  directories (`runs/...`, a new feature cache). One exception: when the scorer
  itself is corrected, the score files (`evaluation.json`) of existing runs, in
  `benchmarks/` too, may be re-scored in place, only after the original is kept
  beside it as `evaluation.old-scorer.json` and only for a run whose checkpoint,
  re-scored with the old scorer, reproduces the stored values. Checkpoints,
  `args.json`, histories and traces are never touched.
- zephyr never imports zephyr-benchmarks (see above).
- Do not commit unless asked.

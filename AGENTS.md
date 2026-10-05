# Working in zephyr

Instructions for coding agents. Read this before changing anything.

## The rule: experiments are TOML, not code

Every input zephyr uses (which clips exist, what trains, what is scored, with
which hyperparameters and seeds) is a **TOML file of local paths** under
`configs/`, validated by the pydantic models in `src/zephyr/config.py`.

To run a new experiment, **write or copy a TOML file. Do not edit Python.**
Specifically, do not:

- add CLI flags for experiment settings (learning rate, epochs, sessions,
  weights, ...);
- hardcode paths, split names (`train`, `test`, `ood`), camera names, session
  numbers or dataset layout anywhere in `src/`;
- write one-off training or evaluation scripts that bypass `zephyr run` /
  `zephyr evaluate`;
- edit an existing config to repurpose it. Copy it under a new name instead,
  so earlier runs stay reproducible.

Change code only when a setting genuinely does not exist yet (see
[When code does need to change](#when-code-does-need-to-change)).

## The three file types

Paths in every file are **relative to the file they are written in**.
`zephyr schema <dir>` writes JSON schemas of all three for editor validation.
A misspelt key is an error, never silently ignored.

### Clip list (`configs/clips/*.toml`, model `ClipList`)

Names videos and how to preprocess them.

```toml
[preprocess]
target_size = [360, 270]          # frame size the boxes are measured in

[[clip]]
video = "../../data/ood/video_side_right_13_part_1.mp4"
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
- Put **all sessions of a dataset in one list**; folds select sessions with
  `groups`. Do not make one file per session or per subset.

### Fold (`configs/folds/*.toml`, model `Fold`)

What trains, what is scored, and how.

```toml
init_from = "../../benchmarks/.../best.pt"   # optional: start from these weights

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

- Every training knob lives in `TrainParams` / `Augmentation` in
  `src/zephyr/config.py`; read that file for the full list and defaults.
- Loading refuses leakage (a test clip sharing a video or recording with
  training), unlabelled training clips, clips without boxes, and lists with
  different `[preprocess]` recipes. If a fold fails to load, fix the fold,
  not the validator.
- `all` is reserved as a test group name.
- Leave-one-out is one fold file per held-out recording (see
  `configs/folds/ood-hold{13,14,15}.toml`); they differ only in `groups`.

### Experiment (`configs/experiments/*.toml`, model `Experiment`)

Which folds run, with which seeds, and where.

```toml
folds = ["../folds/ood-hold13.toml", "../folds/ood-hold14.toml"]
seeds = [42]                      # each fold runs once per seed
output_dir = "../../runs/my-experiment"
features_dir = "../../data/features-v2"
device = "cuda"                   # machine settings: optional, CLI overrides
amp = "bf16"
```

Seeds belong to the experiment, never to the fold.

## Workflow

```bash
uv run zephyr preprocess configs/clips/<list>.toml --cache data/features-v2
uv run zephyr run configs/experiments/<exp>.toml --smoke --output-dir runs/smoke   # always first
uv run zephyr run configs/experiments/<exp>.toml [--seeds 42]
uv run zephyr evaluate --checkpoint <best.pt> --clips <a>.toml <b>.toml --cache data/features-v2 --out <file>
```

- **Always smoke-run** (`--smoke`, 1 epoch of 2 steps) before a full run. Smoke
  output goes to `<output_dir>/smoke/` unless `--output-dir` is given.
- A run writes `<output_dir>/<fold stem>/seed-<n>/` with `config.json`,
  `train_videos.json`, checkpoints and `evaluation.json`. A seed whose `best.pt`
  exists is re-scored, not retrained; use a new `output_dir` for a new
  experiment rather than deleting results.
- Without a GPU add `--device cpu --amp off --num-workers 0`; expect ~1.7 s
  per training step.
- Standalone `evaluate` reports each `--clips` list as a group named after its
  file stem (`--name` renames a single list). To score selected `groups`
  under chosen names, put them in a fold's `[test]` instead.
- `evaluate` refuses clips a checkpoint (or the checkpoint it was initialised
  from) trained on. Do not work around the refusal.
- To check that a new config means what you intend before training, load it:
  `uv run python -c "from zephyr.config import load, Fold; f = load(Fold, 'configs/folds/x.toml'); print([len(c) for c in f.train_clips()], {k: len(v) for k, v in f.test_clips().items()})"`.

## Reference configs: do not change their meaning

`configs/folds/benchmark-gray-diff-flow-multitask.toml`,
`configs/folds/baseline-{tscan,physnet}.toml` and their experiments reproduce
published results; `tests/test_reproduction.py` pins them to the reference
runs' settings. Copy them to experiment; never edit them.

## When code does need to change

Only when an experiment needs a setting that does not exist yet:

1. Add it as a field on the right pydantic model in `src/zephyr/config.py`
   (`TrainParams`, `Augmentation`, `PreprocessParams`, `Selection`, ...), with a
   default that keeps current behaviour, so every existing config and
   `tests/test_reproduction.py` still pass unchanged.
2. Read it from the model where it is used; no new CLI flag.
3. Add a test, keep module docstrings under 10 lines (excluding CLI examples;
   `tests/test_module_docstrings.py` enforces it), and run
   `uv run ruff format . && uv run ruff check . && uv run pytest -q`.

## Keeping this file current

This file describes the infrastructure as it is now, and it will go stale if
nobody maintains it. Whenever a change touches the config models, the CLI, the
run layout, the cache, or a workflow described here, **consider what the change
means for this file and update it in the same change**. That covers, for
example: a new or renamed field, file type or subcommand; a new validation
rule; a moved directory; a new reference config. If a change makes a rule here
wrong or obsolete, fix or remove the rule rather than working around it. When
in doubt, mention the implication to the user.

## Hard rules

- **Never read, store or use animal or subject identity** (`subject_id` or
  similar) anywhere: code, configs, logs. Test group names like
  `new_animals` are labels a config author chose, nothing more.
- `data/` and `benchmarks/` are read-only inputs. Write new outputs to new
  directories (`runs/...`, a new feature cache).
- Do not commit unless asked.

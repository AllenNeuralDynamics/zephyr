# Working in zephyr

Instructions for coding agents. Read this before changing anything.

## Three packages, one way

The repository is a uv workspace of three packages (`uv sync --all-packages
--extra cpu`, or `--extra gpu`):

| Package | Path | Import | Command |
| --- | --- | --- | --- |
| `zephyr` | `packages/zephyr/` | `zephyr` | `zephyr` |
| `zephyr-benchmarks` | `packages/zephyr-benchmarks/` | `zephyr.benchmarks` | `zephyr-benchmarks` |
| `zephyr-dataset` | `packages/zephyr-dataset/` | `zephyr.corpus` | `zephyr-dataset` |

- **`zephyr` runs only the zephyr network.** It never imports
  `zephyr.benchmarks` or `zephyr.corpus` and never names a comparison method
  (TS-CAN, PhysNet, pixel, Facemap, ...); `packages/zephyr/tests/test_boundary.py`
  enforces all three. It is the only released package.
- **`zephyr-dataset`** (Allen-internal, never released) builds the corpus of
  VR-foraging sessions; see [Building the corpus](#building-the-corpus). Its import
  name is `zephyr.corpus` because `zephyr.dataset` is core's training dataset module.
- **`zephyr-benchmarks`** holds everything about our dataset and the methods
  zephyr is compared against: the benchmark configs, the comparison networks
  and baselines, timing, the collected tables and the report. It builds on
  zephyr and is never released.
- A new comparison method goes in zephyr-benchmarks. If it needs zephyr to do
  something new, add a **generic** seam to zephyr (as with `Architecture` in
  `train.py`, `Runner` in `run.py`, the `loader` of `evaluate.py`) that does
  not know the method exists, never a special case.
- `zephyr` lives under one import name across three distributions: core's
  `zephyr/__init__.py` extends `__path__`. Keep that line; never add an
  `__init__.py` at `packages/zephyr-benchmarks/src/zephyr/` or
  `packages/zephyr-dataset/src/zephyr/`.
- **`notebooks/`** holds [marimo](https://marimo.io) notebooks (plain `.py`
  files) for plotting runs and results; marimo is in the `dev` dependency group.
  Notebooks are declarative, only read results (the one exception, `corpus.py`,
  rebuilds the corpus into a new folder of its own), never train, and share
  `notebooks/utils/` (`style.py` fixes figure sizes, fonts and one colour per
  method/channel/objective; reuse it for every figure). They are not an
  experiment path (see below).
- **Inhale events of a network with an onset head come from the head, never
  from DSP on its predicted trace**, in every plot and table: event markers,
  inhale F1, KL-IBI, per-rate analyses (`head_events` in
  `notebooks/utils/results.py`). DSP (`results.events`) is only for the
  thermistor, for methods without a head (labelled "DSP"), and for exhale F1
  (no head exists). A DSP view of a headed network is allowed only as an
  explicitly labelled comparison next to the head version (e.g. in
  `notebooks/walkthrough.py`, the §5 ablation and the "Local F1 (DSP)" column
  of §9).

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

## The two file types

Paths in every file are **relative to the file they are written in**.
`zephyr schema <dir>` writes JSON schemas of both for editor validation.
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

- An entry **without a `video`** is thermistor only (`thermistor = "..."`, optional
  `group`, defaulting to the file stem; no timestamps or box): a whole recording,
  e.g. a session before clips are cut from it. `zephyr annotate events` and
  `zephyr clips select` read it; everything that needs frames (preprocess, folds,
  `evaluate`, `zephyr annotate`, the feature cache) refuses it (`require_videos`).
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
- A clip's breathing events (inhale peaks, exhale troughs), which both training
  (onset head, rate balancing) and scoring use, come from a detector in
  `zephyr/events.py` run on its thermistor. Without any setting every clip uses
  `peak_detection` with its defaults. `[events]` (model `EventsSource`) sets
  `method` (`"adaptive_prominence"`, `"hysteresis_cycles"`, ...) and `params = {...}`
  for the whole list; a clip's own `events = { method = "...", params = {...} }`,
  written next to its `box`, overrides it (see
  [Choosing the event detector](#choosing-the-event-detector)). Detectors work on
  the thermistor at its own rate, so event times are never snapped to a grid.
  Every detector's `params` also take `highpass_hz` (default 0.2), the corner of the
  high-pass applied to the thermistor first (a detector that filters it takes it); it enters the cache identity only off its default.
  A clip's `excluded = [[start, end], ...]` (seconds on the thermistor's clock) marks
  stretches whose events are untrusted: nothing is scored inside them and no
  training window overlaps them. Changing any of these rebuilds only the targets at
  the next `preprocess`. A clip with `selection = "exclude"` is **never used**: the list's
  `resolve()` leaves it out, so preprocessing, folds, training and scoring never see
  it (only the detector tuner lists it, to include it again), whether or not it is
  reviewed. `reviewed = true` records that a person confirmed the mark and spans. At
  least one clip of a list must stay in use.

### Experiment (`configs/experiments/*.toml`, model `Experiment`; `BenchmarkExperiment`)

Which seeds run and where, plus its folds inline: each `[[fold]]` block is one
train/test arrangement (model `Fold`; `BenchmarkFold` in zephyr-benchmarks) --
what trains, what is scored, and how.

```toml
seeds = [42]                      # each fold runs once per seed
output_dir = "../../../../runs/my-experiment"
features_dir = "../../../../data/features-v2"
device = "cuda"                   # machine settings: optional, CLI overrides
amp = "bf16"

[[fold]]
name = "ood-hold15"               # its run directory: <output_dir>/ood-hold15/seed-<n>/
init_from = "../../../../benchmarks/.../best.pt"   # optional: start from these weights

[[fold.train]]
clips = "../clips/face_train.toml"
weight = 0.5                      # relative share of training windows

[[fold.train]]
clips = "../clips/ood_side_right.toml"
groups = ["13", "14"]             # optional: only these recordings
weight = 0.5

[fold.test]                       # name -> clips; names become report groups
ood_held_out = { clips = "../clips/ood_side_right.toml", groups = ["15"] }
new_animals = "../clips/face_test_new_animals.toml"

[fold.train_params]               # anything omitted keeps its default
epochs = 20
lr = 1e-4

[fold.train_params.augmentation]
time_stretch = 1.0
```

- Seeds and machine settings belong to the experiment, never to a fold. Fold
  names must be distinct, plain file names; `zephyr run --folds <name>...`
  runs only some of them.
- Every training knob lives in `TrainParams` / `Augmentation` in zephyr's
  `config.py`; read that file for the full list and defaults.
- `rate_balance` (0 to 1, default 0 = off) draws training windows by their true
  breathing rate, from the clips' stored inhale onsets, over `rate_bins_hz`
  (default 2-15 Hz): each window is weighted by `share(bin) ** -rate_balance`,
  within each `[[fold.train]]` source. Bins under 1% of windows count as 1%.
  Helpers are in `zephyr/rates.py`; validation windows are never balanced.
- `signal_pool` (default 1 = off) makes the signal head read the TCN's output
  average-pooled by that factor in time and upsampled back to 60 Hz (a smoother
  trace). `onset_input` (`full`, the default, `pooled` or `both`) says what the
  onset head reads: the full-rate features, the branch's, or both concatenated;
  the last two need `signal_pool > 1`. Both are recorded in the checkpoint's
  `arch_kwargs`, and both packages' loaders rebuild from them.
- zephyr-benchmarks' `BenchmarkTrainParams` adds `arch` (`zephyr`, the
  default, `tscan`, `physnet` or `physnet_event`; every arch but `zephyr` needs
  `channels = "gray"` and leaves `signal_pool` / `onset_input` at their
  defaults) and `tscan_img_size`. zephyr's own `Fold` refuses both keys: an
  experiment naming a comparison network runs only through `zephyr-benchmarks run`.
- Loading refuses leakage (a test clip sharing a video or recording with
  training), unlabelled training clips, clips without boxes, and lists with
  different `[preprocess]` recipes. If a fold fails to load, fix the fold,
  not the validator.
- `all` is reserved as a test group name.
- Leave-one-out is one `[[fold]]` per held-out recording in one experiment
  (see `configs/experiments/ood-finetune.toml`); the blocks differ only in
  `name` and `groups`. Duplicating them is intended: each fold reads on its own.

## Workflow

From the repository root, with `C=packages/zephyr-benchmarks/configs`:

```bash
uv run zephyr preprocess $C/clips/<list>.toml --cache data/features-v2
uv run zephyr run $C/experiments/<exp>.toml --smoke --output-dir runs/smoke   # always first
uv run zephyr run $C/experiments/<exp>.toml [--seeds 42]
uv run zephyr evaluate --checkpoint <best.pt> --clips <a>.toml <b>.toml --cache data/features-v2 --out <file>
```

- `zephyr-benchmarks run` / `evaluate` take the same arguments and are needed
  for folds with any `arch` other than `zephyr`, and for their checkpoints;
  `zephyr evaluate` refuses non-zephyr checkpoints. Baselines and tables:
  `zephyr-benchmarks pixel|facemap|timing <experiment> [--fold <name>]`,
  `zephyr-benchmarks collect --runs <dir>`,
  `zephyr-benchmarks report --runs <dir> --out <dir>`.
- `zephyr-benchmarks occlusion --checkpoint <best.pt> --clips <a>.toml <b>.toml --cache data/features-v2 --out <file.npz>`
  is inference-only analysis (not an experiment): per-channel patch-occlusion
  maps of event F1 on the first `--window-s` seconds of each clip; `--patch`,
  `--stride`, `--window-s` are analysis args.
- **Always smoke-run** (`--smoke`, 1 epoch of 2 steps) before a full run. Smoke
  output goes to `<output_dir>/smoke/` unless `--output-dir` is given.
- A run writes `<output_dir>/<fold name>/seed-<n>/` with `config.json`,
  `train_videos.json`, checkpoints and `evaluation.json`. With
  `train_params.save_every = N` it also keeps `epoch-<nnn>.pt` every N epochs,
  in the same form as `best.pt`, so `evaluate --checkpoint` scores one as is.
  They were trained mid-schedule (learning rate not yet decayed), so they show
  how the score evolves, not what a shorter run gives. A seed whose `best.pt`
  exists is re-scored, not retrained; use a new `output_dir` for a new
  experiment rather than deleting results.
- Without a GPU add `--device cpu --amp off --num-workers 0`; expect ~1.7 s
  per training step.
- Standalone `evaluate` reports each `--clips` list as a group named after its
  file stem (`--name` renames a single list). To score selected `groups`
  under chosen names, put them in a fold's `[fold.test]` instead.
- `evaluate` refuses clips a checkpoint (or the checkpoint it was initialised
  from) trained on. Do not work around the refusal.
- Scoring and training targets go by time, never by sample index: a clip's video
  and thermistor need not start together (side-camera clips start up to ~0.9 s
  apart, face clips a few ms). `score_clip` compares the two at the prediction's
  timestamps over the span both cover, and `load_target` holds the thermistor's
  edge value outside its span instead of extrapolating. Correlation against the
  raw thermistor moves by several hundredths for a few ms of misalignment, so
  never compare correlations scored before and after this rule.
- Truth events come from the clip's cached event set (`evaluate.score_entry`),
  the same events its target was built from; nothing re-detects them from the
  raw thermistor. Inside an excluded span nothing is scored and no training
  window is drawn. Events match within `EVENT_TOLERANCE_S`, one 60 Hz frame
  (16.7 ms) either side: against exact thermistor times, a predicted event must
  be on one of the two frames bracketing the true one. Scores from before this
  rule (truth re-detected on the raw thermistor on its own 60 Hz grid, 17 ms)
  are not comparable.
- A cache entry whose event set is stale (made before event sets, or under
  another `[events]`) is refused as "not in the feature cache". Rebuild its
  target, without decoding video, with
  `uv run python scripts/rebuild_targets.py --cache <dir> --clips <lists>`
  (`--select-fs 30` for `data/features-v2-30hz`; `--dry-run` reports only).
- To check that a new config means what you intend before training, load it:
  `uv run python -c "from zephyr.config import load; from zephyr.benchmarks.config import BenchmarkExperiment; e = load(BenchmarkExperiment, 'packages/zephyr-benchmarks/configs/experiments/x.toml'); print({f.name: ([len(c) for c in f.train_clips()], {k: len(v) for k, v in f.test_clips().items()}) for f in e.fold})"`.

## Choosing the event detector

`uv run zephyr annotate events [<list.toml>]` (`zephyr/annotate_events.py`; the
window is `annotate_events_ui.py`, pyqtgraph + PySide6, installed with
`uv sync --all-packages --extra ui`) compares the detectors on a clip list. It is
read-only apart from excluded spans: events are never placed by hand. The Editor tab shows a clip's raw and
processed traces, its detected events, rate, amplitude and histograms; the Methods
tab has one row per clip with its method, parameters and quality metrics (and a
mean row). Pick a method and parameters and apply them to the current clip, the
selected clips or all of them. Mark a stretch to exclude with `x` (start, then end,
at the cursor), delete one with a right-click (or Shift+X), undo with Ctrl+Z. The
**Spans** tab beside the event list lists the current clip's spans (start, end,
length): pick one to jump to it, double-click a start or end to type it, add one
in the middle of the view, or delete the selected ones.
Each video has a mark, unset (default, no key), include or exclude, and a reviewed flag
(default false). Set them in the first column of the Methods table, the "This video" strip
and box in the Editor tab, the buttons above the Methods and Selection tables (they act
on all selected rows), the right-click menu, Edit > Cycle video(s) (Ctrl+E) and Edit >
Toggle reviewed (Ctrl+R). An excluded video keeps its row, metrics, spans and detector
(marks never remove settings from the list); it is struck out, left out of the mean row
and of Apply to all, and saved as `selection = "exclude"`. `selection = "include"` is a
note only. Spans have two kinds: exclude (amber, key `x`) and include (teal, key `i`; a
note of good stretches, saved as `included = [[start, end], ...]`, read by nothing
downstream and not used by the metrics). Drag a span's edges on the traces to move them. Sorting the Methods or Selection table
reorders the Editor's clip list (and N / P, < / >) to match.
The **Selection** tab (`zephyr/session_selection.py`, also
`uv run zephyr clips select <list.toml> [--out <list.toml>] [--min-span-s 300] [--window-s 30] [--report <csv>]`)
suggests marks and spans: each clip is scored in rolling windows (SNR, amplitude,
out-of-band breaths, alternation breaks), and `artifact_pad_s` either side of each stretch where
the processed trace (as plotted, z-scored over the clip) goes beyond `artifact_z` in
either direction is cut out; runs of good time at least the minimum span long become `included` spans, bad
windows and artifact stretches `excluded` spans. A clip is suggested exclude when its SNR
is in the list's low (dead sensor) mode, when its amplitude is below `min_amplitude`,
when it is a robust outlier in event consistency
among the rest, or when it has no accepted span. Suggestions are only written to clips
that are not reviewed (Apply to unreviewed videos, or `--out`), replacing their mark and
both span kinds; the thresholds are editable in the tab (analysis settings, not
experiment settings).
**File > Save** writes each chosen `events = { method, params }`, any
`excluded` / `included` spans, any `selection` mark and `reviewed = true` next to the clip's `box` in the list (comments and layout are kept). File > Add videos / Remove selected videos edit the list itself.
The method and parameters may differ from clip to clip.

## Building the corpus

`zephyr-dataset` (`packages/zephyr-dataset/`, import `zephyr.corpus`) builds the corpus
of VR-foraging sessions with a sniff detector from one spec, `DatasetSpec` in
`zephyr/corpus/config.py` (`packages/zephyr-dataset/configs/vr-foraging-sniff.toml`):
`root` (the dataset folder: a URL such as `s3://...`, or a path relative to the spec),
`[source]` (bucket, DocDB, Harp device and register, cameras), `[query]` (`until`,
minimum dataset version, sessions and rigs in or out), `[preprocess]` and `[events]` (the
clip list's header), `[qc]` (**every** `session_selection.Settings` value, none
defaulted), `[select]` (clip length, clips per session, seed, camera required) and
`[clips]` (thermistor pad). Steps, each resumable:

```bash
uv run zephyr-dataset inventory|fetch|lists|qc|select|videos|clips|status|build <spec.toml> [--workers N] [--profile P]
```

The corpus root is `s3://aind-scratch-data/vr-foraging/zephyr-dataset`. Every step reads
and writes the root through `zephyr/corpus/storage.py` (`upath.UPath`, local or remote;
`--profile` sets `AWS_PROFILE`), never renaming or deleting remote objects. Zephyr
itself reads only local files and must not learn about remote paths, so `qc` (and
`zephyr annotate events`, `preprocess`) run on a local copy; a remote clip list is
validated entry by entry because zephyr cannot check its files exist (see the package
README).

- `inventory` queries DocDB (read-only REST, no credentials, paged) for the packaging
  process's derived assets, saves the raw return as `docdb.json.gz` and reuses it on
  later runs (`--refresh` queries again), drops assets created after `[query] until`,
  keeps the latest per raw source session and writes `inventory.parquet` (session_id,
  subject, rig, date, dataset version, asset, source). Both make a rebuild reproducible.
- `fetch` reads only `device.yml` and the register's `.bin` files of each raw session's
  Harp folder on S3 (`zephyr/corpus/harp.py`, no other Harp library) into the whole
  session's `thermistor/<session_id>.parquet` (`Time`, `Signal`).
- `lists` writes `sessions.toml`: one thermistor-only entry per session, header from
  the spec, and each entry's `group` set to its **subject from the session metadata** (folds hold out
  animals). Never infer the animal from a session name: corrected sessions keep their
  old name. It only adds clips, so marks, spans and reviews survive.
- `qc` runs `session_selection.assess` / `apply` (reviewed clips never change) and keeps
  the metrics and suggestions in `qc.parquet`.
- `select` writes `results.json`: a session passes when its mark in `sessions.toml` is
  `include` (the list is the source of truth, so reviews win); passing sessions draw
  `clips_per_session` non-overlapping clips (`span1`, `span2`, ...) uniformly over their
  `included` spans, seeded by the spec's seed and the session_id, and need the camera
  folders (`video.mp4` + `metadata.csv`) of every camera in `[source] cameras` when
  `require_cameras`.
- `videos` (selected sessions only) caches each camera's `metadata.csv` unmodified in
  `camera_metadata/<session_id>_<camera>.parquet` (with the video's frame count from
  ffprobe), measures it (`zephyr/corpus/camera.py`: dropped frames, backward or
  oversized steps in either clock, clock jitter, rows vs video frames; every threshold
  relative to the camera's own median interval, never a fixed rate), repairs one-row
  `ReferenceTime` rollbacks by interpolation and writes the repaired timestamps to
  `videos/<session_id>_<camera>.parquet`; `cameras.parquet` keeps both measurements and
  `clean`. A camera that is not clean after the repair is reported, never silently kept:
  the next `select` drops its session and records why (`dropped` in `results.json`).
  A re-run reads the cache, so changing the repair needs no S3. The known problems, why
  their sessions are dropped and how they could be repaired are in
  `packages/zephyr-dataset/README.md` ("Known data issues"); keep that section current.
- `clips` (selected sessions, after the second `select`) cuts each span of `results.json`
  from each camera's video **without re-encoding**: from the keyframe at or before the
  span's first frame to the keyframe at or after its last (or the video's end), as a
  fragmented MP4 that ffmpeg writes to a pipe, checked (one packet per frame, evenly
  spaced) before upload to `clips/<session_id>_<span>_<camera>.mp4`. Frame rate, time base
  and keyframes come from each stream (cameras run at 240.38 or 359.2 Hz; never assume
  one). Beside it go the frames' `Time` (repaired `ReferenceTime`, row *i* = frame *i*)
  and, per span, the thermistor over both cameras ± `[clips] thermistor_pad_s`; each
  session's camera gets one full-resolution PNG, `thumbnails/<session_id>_<camera>.png`
  (first frame of its first span; missing ones are made from the stored clip). Rows go to
  `clips.parquet` (errors kept, retried next run) and the clips to `clips.toml`, a clip list
  grouped by subject, without boxes.

Decisions made by hand go in the spec as `[[review]]` (`session_id`, optional
`selection`, `excluded`, `included`); `qc` writes them over its suggestions, marked
reviewed (fields left out keep QC's), so a rebuild from nothing gives the same corpus. `notebooks/corpus.py` reruns the metadata stages
into a new local folder and compares it with the corpus.

The dataset folder holds only these entries (plus `camera_metadata/`, `videos/` and
`cameras.parquet` from `videos`, and `clips/`, `thumbnails/`, `clips.parquet`,
`clips.toml` from `clips`); no logs or reports. Review with
`uv run zephyr annotate events <copy>/sessions.toml` on a local copy, record decisions in
`[[review]]`, then run `qc` (on the copy) and `select` again.

## Reference configs: do not change their meaning

`configs/experiments/benchmark-gray-diff-flow-multitask.toml` and
`configs/experiments/baselines-nets.toml` (in `packages/zephyr-benchmarks/`)
reproduce the reference runs in `benchmarks/`;
`packages/zephyr-benchmarks/tests/test_reproduction.py` pins their folds to
those runs' settings. Copy them to experiment; never edit them.

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

- `data/` and `benchmarks/` are read-only inputs. Write new outputs to new
  directories (`runs/...`, a new feature cache). One exception: when the scorer
  itself is corrected, the score files (`evaluation.json`) of existing runs, in
  `benchmarks/` too, may be re-scored in place, only after the original is kept
  beside it as `evaluation.old-scorer.json` and only for a run whose checkpoint,
  re-scored with the old scorer, reproduces the stored values. Checkpoints,
  `config.json` (or older runs' `args.json`), histories and traces are never
  touched.
- zephyr never imports zephyr-benchmarks (see above).
- Do not commit unless asked.

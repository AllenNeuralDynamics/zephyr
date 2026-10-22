# Zephyr

CNN + TCN network that predicts a mouse's breathing trace from video, covering
everything from raw video to a trained, evaluated checkpoint.

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/) and `ffmpeg`/`ffprobe` on
`PATH`. A display (and the `ui` extra) is needed only for the app and the box and
event windows. Everything runs through the single `zephyr <subcommand>` command, or the
app that runs those commands for you; `zephyr <subcommand> --help` prints a one-line
summary, so see each module's docstring (e.g. `src/zephyr/run.py`) for its flags.

Benchmarks on our dataset, and the methods zephyr is compared against, live in
the separate `zephyr-benchmarks` package of the same repository.

## How inputs are described

Zephyr knows nothing about where data came from. Every input is a **TOML file
of local paths**, validated by a pydantic model (`src/zephyr/config.py`); paths
are relative to the file they are written in.

- A **clip list** names videos (everything else -- timestamps, thermistor,
  recording group -- is derived from the file name, or given explicitly), the
  `[preprocess]` recipe, and a hand-placed crop `box` per clip.
- An **experiment** says which seeds run, where results go, and holds one or
  more **folds** inline as `[[fold]]` blocks. A fold has a `name` (its run
  directory) and names the clip lists to train on (with relative weights), the
  clip lists to score (each a named group in the report), the training
  hyperparameters, and optionally a checkpoint to start from. Leakage is
  refused at load: no test clip may share a video, or a *recording* (video
  folder + group), with a training clip.

`zephyr schema <dir>` writes JSON schemas of both for editor validation.
The repository's `examples/` holds one of each, used below; they read the face
clips of the AIND breathing challenge data (see `zephyr-benchmarks`' README for
the download).

## The app

`zephyr gui` is a desktop app (Qt, on Windows, macOS and Linux) that walks through a
project from videos to predictions, as SLEAP's does:

```bash
uv sync --locked --all-packages --extra gpu --extra ui   # --extra cpu without a GPU
uv run zephyr gui my-project/          # a folder; an empty one starts a project
uv run zephyr gui my-project/clips.toml
```

A project is a folder: its clip lists (`*.toml`) and, beside them, `experiments/`,
`frames/` (one frame per clip, for boxes), `cache/` (the feature cache), `runs/` and
`predictions/`. Opening a folder that is not a project yet asks before making it
one, the header says which project is open and where, and **Project → Recent
projects** goes back to another. The steps are listed on the left, each marked done (✓), partly done (◐)
or to do (○) from what the folder holds; above the page are the clip list every step
works on and **Next**, highlighted once the step is done; below it, what the page's
command is doing (its log opens on **Log**, and on its own when a command fails). The
pages, in order:

| Page | Does | Command it runs |
| --- | --- | --- |
| Project | opens or creates a project; where each step stands | |
| Clips | makes a clip list from a video folder, showing first what it would list (each video's session, timestamps and breathing file; other file names under **Other file names**) and the frame size, fixed for the list; checks every file; **Edit TOML** opens the list in your text editor (add clips or paths by hand), **Reload** reads it again and says why if it does not load | `clips scan`, `clips check` |
| Boxes | one crop box per recording, on the list's frame size; **Auto** matches your boxes on the rest | `annotate --prepare` |
| Breathing | each clip's events and detector, bad stretches and clips, opening on the suggestions for every clip; **Auto** gives every clip the default detector; a dot in the list marks unsaved changes | |
| Train | writes a new experiment (one fold over the clip list's groups; **Auto split** draws sessions in train/val/test shares), preprocesses, smoke-runs, runs and resumes it | `preprocess`, `run --save-traces` |
| Results | a run's scores per test group and clip, a clip's trace against its sensor (the run's `traces/`), its curves; scoring it on its test groups or another list | `run --evaluate-only`, `evaluate` |
| Predict | a checkpoint's trace for each video, with the onset head's probability under it, the crop it came from (when its frames are cached) and its quality control; **Videos** takes a folder of videos and timestamps from anywhere into a list you can see, add videos to and remove them from, its crops placed at the network's size; frames are read straight from the videos unless you ask to keep them in the cache | `clips scan --checkpoint`, `annotate --prepare`, `predict`, `quality-control run` |

The app writes only clip lists (as the two windows do) and new experiments (never
editing one). Every command runs on this machine, in a child process of the app, with
the same arguments the command line takes (**Copy command** gives that line), so a crash
or **Cancel** leaves the app running and a stopped run continues with **Resume**.
**Run** on an experiment that already has a run asks before starting it over
(`--restart`), and **Score its test groups** on Results scores a run's `best.pt`,
finished or stopped (`--evaluate-only`), asking before it replaces scores. Runs and
scorings the app starts save the traces they score (`--save-traces`, about 1 MB a
test clip), which Results draws against the sensor. Before anything fills the
feature cache (**Preprocess**, or a prediction keeping its frames), the app says how
much space it takes (frames kept × crop pixels, one byte each) and how much the drive
has. While a
run trains, its curves are read from its `history.json`, and after each validation
scoring it sends the app a window of a few validation clips (the true and predicted
traces, their inhales and the crop frames to play along): those snippets are shown,
never saved.

## Clone to an evaluated network

1. **Install** (from the repository root; `--extra gpu` for CUDA):

   ```bash
   git clone https://github.com/AllenNeuralDynamics/zephyr
   cd zephyr
   uv sync --locked --all-packages --extra cpu
   ```

2. **Scan** a folder into a clip list (videos only; the scan validates that
   every derived file exists, so unlabelled clips need `--unlabelled`):

   ```bash
   uv run zephyr clips scan data/train --glob 'video_face_*.mp4' -o my_clips.toml
   ```

   Files named otherwise are scanned with `--pattern`, `--timestamps` and
   `--thermistor`, written as the list's `[derive]` (see "Clip lists" below). The
   list's `[events]` starts as `adaptive_prominence` with the corpus's settings
   (`zephyr.clips.SCAN_EVENTS`); a list without `[events]` uses `peak_detection`.
   `zephyr clips check my_clips.toml` then reads every file the list names and reports,
   per clip, what preprocessing or training would refuse: missing files or columns,
   frames that differ from the timestamp rows, a camera slower than `select_fs`, a
   breathing sensor at 80 Hz or less, a sensor and video sharing no time, a missing box.

   To predict on videos that have only timestamps, from any folder, scan them for the
   network: `--checkpoint` writes its `[preprocess]`, no sensor, each video its own group
   (unless `--pattern`) and absolute paths. The crop must be the size the network trained
   on (read from the run's training lists in the checkpoint): the app's Predict →
   Videos places it at that size, so only the place is chosen, and `zephyr predict`
   refuses any other size (`--restart` predicts a folder's videos again).

   ```bash
   uv run zephyr clips scan D:/new-videos --glob '*.mp4' -o new.toml --checkpoint best.pt
   uv run zephyr annotate new.toml --prepare --frame-dir frames
   uv run zephyr annotate new.toml --auto --frame-dir frames --checkpoint best.pt  # or by hand
   uv run zephyr predict --checkpoint best.pt --clips new.toml --out predictions/new
   ```

3. **Annotate**: place one crop box per recording (one box covers a
   recording's parts; `o` gives a single clip its own) on the app's Boxes page (needs
   `--extra ui`). Boxes are written into the clip list as `box = [x, y, w, h]`, and
   re-centring a recording (`r`) removes its box. `zephyr annotate --prepare` caches
   each clip's frame at 150 s, or halfway through a shorter clip, into `--frame-dir`;
   the app places boxes on the project's `frames/`, so a list whose clips are not on
   this machine needs only those frames. `examples/clips.toml` already carries boxes:

   ```bash
   uv run zephyr annotate my_clips.toml --prepare --frame-dir frames
   uv run zephyr gui my_clips.toml --page boxes   # frames from frames/ beside the list
   ```

   **Auto** (`a` on the page, or `zephyr annotate --auto`) places a few boxes for you: the crop
   of every recording you placed is a template, matched (normalised cross-correlation,
   on the cached frames at the list's size) against each unplaced recording; one whose
   best match reaches `--min-score` (0.5) gets a box there, shown as "match 0.73" until
   you move it, and the rest stay unplaced. Place one of those and run Auto again.

   ```bash
   uv run zephyr annotate my_clips.toml --auto --frame-dir artifacts/frames
   ```

4. **Preprocess** into a feature cache (~1-2 min per clip on CPU). Entries are
   keyed by video, recipe and box, so a cache directory is safe to share and
   re-running only does what changed:

   ```bash
   uv run zephyr preprocess examples/clips.toml --cache data/features-example
   ```

   Each entry stores the clip's gray frames (`gray-<prefix>.npy`, uint8), the only input
   the network reads.

5. **Run.** `zephyr run` trains each (fold, seed) into
   `<output_dir>/<fold name>/seed-<n>/` (`--folds <name>...` runs only some) -- with the resolved config, the list of
   training videos, checkpoints and `evaluation.json` -- then scores the
   fold's test groups. Check the plumbing first with `--smoke` (1 epoch of 2
   steps; add `--device cpu --amp off --num-workers 0` without a GPU):

   ```bash
   uv run zephyr run examples/experiment.toml --smoke
   uv run zephyr run examples/experiment.toml
   ```

   A run that has a `best.pt` is not trained again: `zephyr run` only scores it (a
   stopped one too). `--resume` continues a stopped run, `--restart` deletes each run
   folder and trains it from scratch, and `--evaluate-only` never trains, scoring
   `best.pt` as it is.

   `evaluation.json` reports each clip, each recording, and each named test
   group (plus the onset head's inhale F1 for multitask networks). With
   `save_traces = true` in the experiment (or `--save-traces`; off by default) the run
   also writes `traces/feat-<entry prefix>.parquet` per test clip, one row per output
   sample: the `time`, z-scored `signal` and (with an onset head) `onset` probability
   that were scored, so reports need not run the network again.

6. **Evaluate** any checkpoint on any clip lists; each list is reported as a
   group. A checkpoint records the videos it trained on and `evaluate` refuses
   clips that share a video or recording with them (older checkpoints without
   the record are scored with a warning):

   ```bash
   uv run zephyr evaluate --checkpoint runs/example/fold/seed-42/best.pt --clips <list.toml> --cache data/features-example --out evaluation.json
   ```

   Scoring reads only the cache and each clip's thermistor, so the videos and
   timestamps need not be on the machine once their entries are cached.

**A copy of a dataset elsewhere.** Cache keys, sidecars and the training record name
clips by their absolute paths (in `/` form, `config.canonical`), so a cache made at
`/data/zephyr-dataset` is not found from a copy at `C:/Data/zephyr-dataset`, and a
checkpoint's record would not catch its own training clips there. Say which folder is
a copy of which, and the copy is named as the original everywhere a path is an identity
(files are still read from the copy):

```bash
ZEPHYR_PATH_MAP="C:/Data/zephyr-dataset=/data/zephyr-dataset" uv run zephyr evaluate ...
```

Several pairs are separated by `;`. With it set, a cache made on Code Ocean, copied to
this machine, is used as it is, and one preprocessed here is the one Code Ocean would
make. Without it nothing changes.

**Predict** breathing from videos with no sensor (a clip list with boxes, `thermistor =
false` or not) with `zephyr predict`. It preprocesses the clips, into `--cache` or
else a temporary folder it removes, and writes `<video stem>.prediction.parquet` per
video: `time`, the z-scored `signal`, the onset head's `onset` probability and `inhale`
(an inhale the head found there). The lists' `[preprocess]` must be the checkpoint's
recipe, and a folder holds one checkpoint's predictions:

```bash
uv run zephyr predict --checkpoint runs/example/fold/seed-42/best.pt --clips <list.toml> --out predictions/
```

`zephyr.infer.predict_clip` runs inference on an arbitrary preprocessed clip
and checkpoint; see its docstring.

## Quality control without a thermistor

On an animal without a thermistor nothing can be scored, so `zephyr quality-control`
judges each prediction from the network's output alone, its trace and onset
probability: one ordinary inference per video and nothing more. Both steps take flags,
like `zephyr evaluate` (`--clips` takes clip lists, `--groups` keeps some of their
groups):

```bash
uv run zephyr quality-control run --checkpoint best.pt --clips <list.toml>     --cache data/features-example --out qc/ [--reference qc-reference.json]
uv run zephyr quality-control reference --checkpoint best.pt --clips <list.toml>     --groups <validation sessions> --cache data/features-example --out qc-reference.json
```

1. **Reference** (optional), once per checkpoint: predicts labelled clips the checkpoint
   never trained on (e.g. its validation animals; refused if it trained on them;
   `--max-clips` to predict fewer) and fits a logistic calibration of the onset head:
   the chance that an inhale it asserts with a given peak is real. The JSON file it
   writes is tied to the checkpoint by its SHA-256; ship it beside it. Without it every
   measure but expected precision is still reported.
2. **Run**, on any preprocessed clips (`thermistor = false` is fine): writes
   `<video stem>.qc.json` per video into `--out`: every measure over the whole clip
   (`summary`) and per window (`windows`, about `--window-s` wide), plus
   `<video stem>.qc.parquet` (one row per output sample: time, trace, onset
   probability, inhale; `--no-traces` skips it). A finished report is kept on a re-run;
   a folder holding reports made with other settings is refused. Every summary and
   window keeps each measure's value and adds `checks`: each measure against its pass
   bound (`true` passes, `false` fails, `null` when the measure has no value, e.g.
   expected precision without a reference). No video is failed as a whole; which checks
   matter is the reader's call. The default bounds (`QualityThresholds` in `config.py`)
   each pass 99% of the good windows (true inhale F1 >= 0.9) of the corpus's
   validation animals; `--threshold <measure>=<bound>` overrides one, and a bound of
   `nan` leaves that measure unchecked. `zephyr.quality_control.load_reports(folder)`
   reads a folder back as two tables (checks as `ok_<measure>` columns).

The measures (`quality_control.metrics.MEASURES`, with which way is better):

| Family | Measures |
| --- | --- |
| rhythm | rate (a description), breaths faster than `breathing_ceiling_hz` (faster than a mouse breathes), inhale/exhale alternation breaks, flat breaths (the trace swings less than `flat_ratio` of the clip's median breath between two inhales). No slower rate is judged: usual rates change with running, odours and task |
| head | onset head vs DSP peaks on its own trace (a labelled comparison), median peak height, ambiguous peaks (`ambiguous_floor` to 0.5), background probability, expected precision (reference calibration), inhale phase spread (how much the point of the trace's trough-to-trough cycle where the head puts each inhale wanders) |

Inhales come from the onset head (or DSP on the trace for a headless checkpoint), as
everywhere. A labelled clip's report also holds a `truth` block, its real scores scored
as `zephyr evaluate` scores it (over the whole clip exactly so); no measure reads it.
It is there to check the measures: they are proxies, and only held-out animals with a
thermistor say which of them track the real F1 (`notebooks/quality_control.py`).

## Reference

The models in `src/zephyr/config.py` are the full list of fields and defaults; this
section explains the ones whose behaviour is not obvious.

### Clip lists

- An entry **without a `video`** is thermistor only (`thermistor`, optional `group`
  defaulting to the file stem, no timestamps or box): a whole recording, e.g. a session
  before clips are cut. The app's Breathing page (and its suggestions) read it;
  everything that needs frames refuses it (`require_videos`).
- `timestamps`, `thermistor` and `group` are derived from the video name by `[derive]`;
  `thermistor = false` marks an unlabelled clip. `group` is the recording (session); a
  *recording* is (video folder, group).
- `[preprocess]` sets `select_fs` (rate of the frames the CNN sees) and `output_fs` (rate
  of the predicted grid; both default 60 Hz). The TCN's dilations count output samples,
  so its context in seconds scales with `1 / output_fs`. Scoring always resamples to 60 Hz.
- Breathing events (inhale peaks, exhale troughs) come from a detector in
  `zephyr/events.py` run on the thermistor at its own rate: `peak_detection` by default,
  `[events]` (`method`, `params`) for the list, a clip's own `events = {...}` to override.
  Every detector takes `highpass_hz` (default 0.2). A clip's `excluded` spans (thermistor
  seconds) are never scored or trained on; `selection = "exclude"` removes the clip from
  everything downstream; `included` spans and `selection = "include"` are notes only;
  `reviewed = true` records that a person confirmed the mark. Changing any of these
  rebuilds only targets at the next `preprocess`.

### Experiments

- Validation picks `best.pt` and stops the run early (`score_every`, `patience`,
  `min_epochs`). By default it is the last `val_fraction` of every training clip; a
  `[fold.val]` selection (written like a test group) validates on whole clips instead,
  e.g. held-out animals. They never train, must share no recording with the training or
  test clips, and replace `val_fraction` (leave it out). `val_fraction = 0` without
  `[fold.val]` trains a fixed number of epochs. With `ema_decay > 0` it selects on the
  averaged weights once they have relaxed (about 3 / (1 - `ema_decay`) steps), and the
  count starts afresh then: a live-weights best, noisier, is no bar for them.
- `rate_balance` (0 to 1, default 0) weights training windows by
  `share(rate bin) ** -rate_balance` over `rate_bins_hz` (default 2-15 Hz), per
  `[[fold.train]]` source, bins under 1% counting as 1% (`zephyr/rates.py`); validation
  windows are never balanced.
- `[fold.train_params.augmentation]` (defaults on): `shift_px` moves the crop, `scale`
  zooms it and `rotation` (degrees) turns it about its centre (both default 0), `blur`
  (maximum Gaussian sigma in pixels, default 0), `brightness` and `contrast` act on the
  gray frames.
- `temporal_kernel` (odd, default 1) makes the encoder's first layer a 3D convolution over
  that many neighbouring frames, so the network sees change between frames in its first
  layer; later layers stay per frame. Stored in the checkpoint's `arch_kwargs`.
- `signal_pool` (default 1) makes the signal head read the TCN output average-pooled by
  that factor; `onset_input` (`full`, `pooled` or `both`) picks what the onset head reads
  (the last two need `signal_pool > 1`). Both are stored in the checkpoint's `arch_kwargs`.
- With `train_params.save_every = N` a run also keeps `epoch-<nnn>.pt`, scorable like
  `best.pt`; they are mid-schedule (learning rate not decayed), so they show how the
  score evolves, not what a shorter run gives. A seed whose `best.pt` exists is
  re-scored, not retrained.
- Leave-one-out is one `[[fold]]` per held-out recording, each training on the others
  with `groups`. `all` is a reserved test group name.
- Without a GPU: `--device cpu --amp off --num-workers 0` (about 1.7 s per step).
- To check a config before training, load it and print its folds' clip counts with
  `zephyr.config.load`.

### Scoring

- By time, never by sample index: video and thermistor need not start together
  (side-camera clips up to about 0.9 s apart). `score_clip` compares them at the
  prediction's timestamps over the span both cover; `load_target` holds the thermistor's
  edge value outside its span. Correlations scored before this rule are not comparable.
- Truth events come from the clip's cached event set (`evaluate.score_entry`), never
  re-detected. A predicted event matches a true one on the frame nearest it or the frame
  before or after: `EVENT_TOLERANCE_FRAMES` (1.5) frames of the prediction's own output
  grid, so 25 ms at 60 Hz and 50 ms at 30 Hz. Event F1 scored before this rule (one
  60 Hz frame, 16.7 ms) is not comparable.
- The onset head's target is a Gaussian (`onset_sigma_s`) centred on each event's own
  time, between frames where it falls there (`targets.onset_heatmap`), built per window
  from the cached event times; it is not stored.
  Heads trained before this rule peaked on the frame at or after each event (half a
  frame late on average) and learned a false onset at both ends of every window.
- A cache entry whose event set is stale is refused as "not in the feature cache";
  re-running `zephyr preprocess` on its list rebuilds only the target, without decoding
  video.

### Choosing the event detector

The app's Breathing page (`uv run zephyr gui <list.toml> --page breathing`, needs `--extra
ui`) compares detectors on a clip list; it never places events by hand. The **Editor** tab shows a clip's traces,
events, rate and amplitude; **Methods** has one row per clip with its method, parameters
and quality metrics; **Spans** lists the clip's spans. Apply a method to the current,
selected or all clips, or **Auto** (`zephyr annotate events <list> --auto` saves it without the app):
the default detector (`zephyr.clips.SCAN_EVENTS`) becomes the list's `[events]` and
every clip's own `events` is removed, marks, spans and `reviewed` untouched; mark an excluded stretch with `x` (start, then end), an included
one with `i`, delete with right-click or Shift+X, undo with Ctrl+Z, drag edges to move.
Each video has a mark (unset, include, exclude) and a reviewed flag (Ctrl+E cycles the
mark, Ctrl+R toggles reviewed); an excluded video keeps its settings and is left out of
the mean row and of Apply to all. The **Selection** tab (`zephyr.session_selection`)
suggests marks and spans from rolling-window SNR, amplitude, out-of-band breaths,
alternation breaks and artifacts (`artifact_z`, `artifact_pad_s`), writing only to
unreviewed clips. **File > Save** writes methods, spans, marks and `reviewed` next to each
clip's `box`, keeping comments and layout.

# Zephyr

CNN + TCN network that predicts a mouse's breathing trace from video, covering
everything from raw video to a trained, evaluated checkpoint.

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/) and `ffmpeg`/`ffprobe` on
`PATH`. A display is only needed for the optional annotation step. Everything
runs through the single `zephyr <subcommand>` command; `zephyr <subcommand>
--help` prints a one-line summary, so see each module's docstring (e.g.
`src/zephyr/run.py`) for its flags.

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

3. **Annotate**: place one crop box per recording (one box covers a
   recording's parts; `o` gives a single clip its own). Boxes are written into
   the clip list as `box = [x, y, w, h]`. `examples/clips.toml` already
   carries boxes:

   ```bash
   uv run zephyr annotate my_clips.toml --prepare
   uv run zephyr annotate my_clips.toml --width 360 --height 270 --box-size 96
   ```

   For a list whose clips are not on this machine, `--remote` skips checking that they
   exist and reads the frames you cached in `--frame-dir` (one `frame_path` file per clip).

4. **Preprocess** into a feature cache (~1-2 min per clip on CPU). Entries are
   keyed by video, recipe and box, so a cache directory is safe to share and
   re-running only does what changed:

   ```bash
   uv run zephyr preprocess examples/clips.toml --cache data/features-example
   ```

5. **Run.** `zephyr run` trains each (fold, seed) into
   `<output_dir>/<fold name>/seed-<n>/` (`--folds <name>...` runs only some) -- with the resolved config, the list of
   training videos, checkpoints and `evaluation.json` -- then scores the
   fold's test groups. Check the plumbing first with `--smoke` (1 epoch of 2
   steps; add `--device cpu --amp off --num-workers 0` without a GPU):

   ```bash
   uv run zephyr run examples/experiment.toml --smoke
   uv run zephyr run examples/experiment.toml
   ```

   `evaluation.json` reports each clip, each recording, and each named test
   group (plus the onset head's inhale F1 for multitask networks).

6. **Evaluate** any checkpoint on any clip lists; each list is reported as a
   group. A checkpoint records the videos it trained on and `evaluate` refuses
   clips that share a video or recording with them (older checkpoints without
   the record are scored with a warning):

   ```bash
   uv run zephyr evaluate --checkpoint runs/example/fold/seed-42/best.pt --clips <list.toml> --cache data/features-example --out evaluation.json
   ```

`zephyr.infer.predict_clip` runs inference on an arbitrary preprocessed clip
and checkpoint; see its docstring.

## Reference

The models in `src/zephyr/config.py` are the full list of fields and defaults; this
section explains the ones whose behaviour is not obvious.

### Clip lists

- An entry **without a `video`** is thermistor only (`thermistor`, optional `group`
  defaulting to the file stem, no timestamps or box): a whole recording, e.g. a session
  before clips are cut. `zephyr annotate events` and `zephyr clips select` read it;
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
  `[fold.val]` trains a fixed number of epochs.
- `rate_balance` (0 to 1, default 0) weights training windows by
  `share(rate bin) ** -rate_balance` over `rate_bins_hz` (default 2-15 Hz), per
  `[[fold.train]]` source, bins under 1% counting as 1% (`zephyr/rates.py`); validation
  windows are never balanced.
- `[fold.train_params.augmentation]` (defaults on): `shift_px` moves the crop and `scale`
  zooms it about its centre (default 0), `brightness`/`contrast` act on gray,
  `motion_noise` on diff and flow. `scale` needs channels without flow (gray, gray+diff):
  a model that reads the flow channels refuses it.
- `temporal_kernel` (odd, default 1) makes the encoder's first layer a 3D convolution over
  that many neighbouring frames, so a gray-only network sees change between frames
  itself rather than through the precomputed `diff`/flow channels; later layers stay per
  frame. Stored in the checkpoint's `arch_kwargs`.
- `signal_pool` (default 1) makes the signal head read the TCN output average-pooled by
  that factor; `onset_input` (`full`, `pooled` or `both`) picks what the onset head reads
  (the last two need `signal_pool > 1`). Both are stored in the checkpoint's `arch_kwargs`.
- With `train_params.save_every = N` a run also keeps `epoch-<nnn>.pt`, scorable like
  `best.pt`; they are mid-schedule (learning rate not decayed), so they show how the
  score evolves, not what a shorter run gives. A seed whose `best.pt` exists is
  re-scored, not retrained.
- Leave-one-out is one `[[fold]]` per held-out recording (see
  `configs/experiments/ood-finetune.toml` in zephyr-benchmarks). `all` is a reserved
  test group name.
- Without a GPU: `--device cpu --amp off --num-workers 0` (about 1.7 s per step).
- To check a config before training, load it and print its folds' clip counts with
  `zephyr.config.load`.

### Scoring

- By time, never by sample index: video and thermistor need not start together
  (side-camera clips up to about 0.9 s apart). `score_clip` compares them at the
  prediction's timestamps over the span both cover; `load_target` holds the thermistor's
  edge value outside its span. Correlations scored before this rule are not comparable.
- Truth events come from the clip's cached event set (`evaluate.score_entry`), never
  re-detected. Events match within `EVENT_TOLERANCE_S`, one 60 Hz frame (16.7 ms) either
  side. Scores from before this rule are not comparable.
- A cache entry whose event set is stale is refused as "not in the feature cache";
  re-running `zephyr preprocess` on its list rebuilds only the target, without decoding
  video.

### Choosing the event detector

`uv run zephyr annotate events [<list.toml>]` (needs `--extra ui`) compares detectors on
a clip list; it never places events by hand. The **Editor** tab shows a clip's traces,
events, rate and amplitude; **Methods** has one row per clip with its method, parameters
and quality metrics; **Spans** lists the clip's spans. Apply a method to the current,
selected or all clips; mark an excluded stretch with `x` (start, then end), an included
one with `i`, delete with right-click or Shift+X, undo with Ctrl+Z, drag edges to move.
Each video has a mark (unset, include, exclude) and a reviewed flag (Ctrl+E cycles the
mark, Ctrl+R toggles reviewed); an excluded video keeps its settings and is left out of
the mean row and of Apply to all. The **Selection** tab (also
`uv run zephyr clips select <list.toml> [--out ...] [--min-span-s 300] [--window-s 30]`)
suggests marks and spans from rolling-window SNR, amplitude, out-of-band breaths,
alternation breaks and artifacts (`artifact_z`, `artifact_pad_s`), writing only to
unreviewed clips. **File > Save** writes methods, spans, marks and `reviewed` next to each
clip's `box`, keeping comments and layout.

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
- A **fold** names the clip lists to train on (with relative weights), the
  clip lists to score (each a named group in the report), the training
  hyperparameters, and optionally a checkpoint to start from. Leakage is
  refused at load: no test clip may share a video, or a *recording* (video
  folder + group), with a training clip.
- An **experiment** names folds, the seeds each runs with, and where results go.

`zephyr schema <dir>` writes JSON schemas of all three for editor validation.
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

4. **Preprocess** into a feature cache (~1-2 min per clip on CPU). Entries are
   keyed by video, recipe and box, so a cache directory is safe to share and
   re-running only does what changed:

   ```bash
   uv run zephyr preprocess examples/clips.toml --cache data/features-example
   ```

5. **Run.** `zephyr run` trains each (fold, seed) into
   `<output_dir>/<fold>/seed-<n>/` -- with the resolved config, the list of
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

# The command line

Everything the app does is a `zephyr` command, and the app runs exactly these commands
(**Copy command** on any page gives you the line). Use the command line on a cluster,
in scripts, or when you prefer a terminal. The only steps that need a window are placing
boxes and choosing detectors by hand.

This walk-through uses the tutorial's six labelled sessions, downloaded into
`data/vr-foraging/` (see [Get the example data](../tutorial/index.md#get-the-example-data)).
The repository's `examples/` folder holds a clip list of them, with boxes placed, and an
experiment that splits them by animal.

## 1. Make a clip list

```bash
uv run zephyr clips scan data/vr-foraging --glob '*_FaceCamera.mp4' \
    --target-size 360 270 -o my_clips.toml \
    --pattern '(?P<group>.+)_span(?P<span>\d+)_FaceCamera$' \
    --thermistor '{group}_span{span}_thermistor.parquet'
uv run zephyr clips check my_clips.toml
```

The scan checks that every derived file exists, so a folder of videos without sensors
needs `--unlabelled`. `--pattern`, `--timestamps` and `--thermistor` describe file names
other than the default (see [Preparing your own data](your-data.md#other-names)); they
become the list's `[derive]`.

## 2. Place the boxes

```bash
uv run zephyr annotate my_clips.toml --prepare --frame-dir frames   # one frame per clip
uv run zephyr gui my_clips.toml --page boxes                        # place a few by hand
uv run zephyr annotate my_clips.toml --auto --frame-dir frames      # match the rest
```

The app reads frames from a `frames/` folder beside the list. `examples/clips.toml`
already has boxes.

## 3. Choose the breathing events

```bash
uv run zephyr annotate events my_clips.toml --auto   # the default detector for every clip
uv run zephyr gui my_clips.toml --page breathing     # optional: check and mark by hand
```

## 4. Preprocess

```bash
uv run zephyr preprocess examples/clips.toml --cache data/features-example
```

This takes about one to two minutes per clip on CPU. Clips are processed in parallel
(`--workers`). Re-running does only what changed.

## 5. Train and score

```bash
uv run zephyr run examples/experiment.toml --smoke    # 1 epoch of 2 steps: check the plumbing
uv run zephyr run examples/experiment.toml
```

Each fold and seed trains into `<output_dir>/<fold>/seed-<n>/`, then scores the fold's
test groups into `evaluation.json`. See [Output files](../reference/outputs.md).

## 6. Score a checkpoint on more clips

```bash
uv run zephyr evaluate --checkpoint runs/example/fold/seed-42/best.pt \
    --clips new-animals.toml --cache data/features-example --out new-animals.json
```

Each `--clips` list is reported as its own group. A checkpoint records the videos it
trained on, and `evaluate` refuses them. Scoring reads only the cache and each clip's
sensor, so the videos need not be on the machine once cached.

## 7. Predict new videos

```bash
uv run zephyr clips scan D:/new-videos --glob '*.mp4' -o new.toml --checkpoint best.pt
uv run zephyr annotate new.toml --prepare --frame-dir frames
uv run zephyr annotate new.toml --auto --frame-dir frames --checkpoint best.pt
uv run zephyr predict --checkpoint best.pt --clips new.toml --out predictions/new
```

`clips scan --checkpoint` writes a list for that network: its `[preprocess]`, no
sensor, and each video its own group. The crop must be the size the network trained on,
and `predict` refuses any other. See [Quality control](quality-control.md) to judge the
predictions.

## Editor support

```bash
uv run zephyr schema schemas/
```

writes JSON schemas of clip lists and experiments. Point your editor's TOML extension at
them (for example *Even Better TOML* in VS Code) for completion and validation as you
type.

## From Python

Each command is a module with a `main(argv)` function, so a script can call the same
code:

```python
from zephyr import run

run.main(["examples/experiment.toml", "--smoke"])
```

`zephyr.infer.predict_clip` predicts one preprocessed clip with a checkpoint, and
`zephyr.config.load` loads and validates a clip list or experiment:

```python
from zephyr.config import Experiment, load

experiment = load(Experiment, "examples/experiment.toml")
for fold in experiment.folds():
    print(fold.name, [source.groups for source in fold.train])
```

See [Commands](../reference/commands.md) for every flag.

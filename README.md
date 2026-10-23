# Zephyr

CNN + TCN network that predicts a mouse's breathing trace from video.

**Documentation:** <https://allenneuraldynamics.github.io/zephyr/> (installation, a
tutorial of the app, guides and reference; sources in `docs/`).

This repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
of three packages:

| Package | Import | Command | What it is |
| --- | --- | --- | --- |
| [`zephyr`](packages/zephyr/README.md) | `zephyr` | `zephyr` | The network: clip lists, preprocessing, training, evaluation, inference. The only released package. |
| [`zephyr-benchmarks`](packages/zephyr-benchmarks/README.md) | `zephyr.benchmarks` | `zephyr-benchmarks` | Benchmarks on our dataset and the methods zephyr is compared against (TS-CAN, PhysNet, pixel, Facemap-style). Workspace only. |
| [`zephyr-dataset`](packages/zephyr-dataset/README.md) | `zephyr.corpus` | `zephyr-dataset` | Builds the corpus of VR-foraging sessions (Allen-internal). Workspace only. |

`zephyr` never imports `zephyr.benchmarks`; a test enforces it.

```bash
git clone https://github.com/AllenNeuralDynamics/zephyr
cd zephyr
uv sync --locked --all-packages --extra cpu     # or --extra gpu for CUDA
```

For the desktop app, which walks a project from videos to a trained network and its
predictions, add `--extra ui` and run `uv run zephyr gui` (see
[The app](packages/zephyr/README.md#the-app)).

## Data

The dataset (face-camera videos, frame timestamps and thermistor traces) is on
a public S3 bucket; no AWS credentials are needed. The download commands are in
the [benchmarks README, under Dataset](packages/zephyr-benchmarks/README.md#dataset).
Everything goes into `data/` at the repository root.

- `examples/` has one clip list and one experiment to start from; the
  [zephyr README](packages/zephyr/README.md) walks through them.
- To train or predict on your own recordings, see
  [Preparing your own data](https://allenneuraldynamics.github.io/zephyr/guides/your-data/)
  (sources in `docs/guides/your-data.md`).
- `packages/zephyr-benchmarks/configs/` holds the benchmark's configs; its
  [README](packages/zephyr-benchmarks/README.md) covers downloading the data
  and reproducing the published results.
- `data/` (inputs), `benchmarks/` (reference runs) and `runs/` (outputs) live
  at the repository root and are ignored by Git.

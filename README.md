# Zephyr

CNN + TCN network that predicts a mouse's breathing trace from video.

This repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
of two packages:

| Package | Import | Command | What it is |
| --- | --- | --- | --- |
| [`zephyr`](packages/zephyr/README.md) | `zephyr` | `zephyr` | The network: clip lists, preprocessing, training, evaluation, inference. The only released package. |
| [`zephyr-benchmarks`](packages/zephyr-benchmarks/README.md) | `zephyr.benchmarks` | `zephyr-benchmarks` | Benchmarks on our dataset and the methods zephyr is compared against (TS-CAN, PhysNet, pixel, Facemap-style). Workspace only. |

`zephyr` never imports `zephyr.benchmarks`; a test enforces it.

```bash
git clone https://github.com/AllenNeuralDynamics/zephyr
cd zephyr
uv sync --locked --all-packages --extra cpu     # or --extra gpu for CUDA
```

## Data

The dataset (face-camera videos, frame timestamps and thermistor traces) is on
a public S3 bucket; no AWS credentials are needed. The download commands are in
the [benchmarks README, under Dataset](packages/zephyr-benchmarks/README.md#dataset).
Everything goes into `data/` at the repository root.

- `examples/` has one clip list and one experiment to start from; the
  [zephyr README](packages/zephyr/README.md) walks through them.
- `packages/zephyr-benchmarks/configs/` holds the benchmark's configs; its
  [README](packages/zephyr-benchmarks/README.md) covers downloading the data
  and reproducing the published results.
- `data/` (inputs), `benchmarks/` (reference runs) and `runs/` (outputs) live
  at the repository root and are ignored by Git.

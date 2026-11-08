# Zephyr

CNN + TCN network that predicts a mouse's breathing trace from video.

**Documentation:** <https://allenneuraldynamics.github.io/zephyr/> (installation, a
tutorial of the app, guides and reference; sources in `docs/`).

**Try it online:** <https://allenneuraldynamics.github.io/zephyr/app/>, the network in
your browser on your own GPU (sources in `web/`, see `web/README.md`).

This repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
of one package, [`zephyr`](packages/zephyr/README.md) (import `zephyr`, command
`zephyr`): the network, from clip lists, preprocessing, training and evaluation to
inference.

The paper lives in
[zephyr-manuscript](https://github.com/AllenNeuralDynamics/zephyr-manuscript): its
manuscript, notebooks, the benchmarks against other methods (`zephyr-benchmarks`) and the
build of the corpus (`zephyr-dataset`), which install under the `zephyr` import name
(`zephyr.benchmarks`, `zephyr.corpus`) and pin a commit of this repository. `zephyr`
never imports them; a test enforces it.

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
a public S3 bucket; no AWS credentials are needed. The
[tutorial](https://allenneuraldynamics.github.io/zephyr/tutorial/) downloads the
sessions it uses into `data/` at the repository root.

- `examples/` has one clip list and one experiment to start from; the
  [zephyr README](packages/zephyr/README.md) walks through them.
- To train or predict on your own recordings, see
  [Preparing your own data](https://allenneuraldynamics.github.io/zephyr/guides/your-data/)
  (sources in `docs/guides/your-data.md`).
- Reproducing the published results is
  [zephyr-manuscript](https://github.com/AllenNeuralDynamics/zephyr-manuscript)'s.
- `data/` (inputs) and `runs/` (outputs) live at the repository root and are ignored
  by Git.

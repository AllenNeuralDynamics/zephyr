# Notebooks

[marimo](https://marimo.io) notebooks for plotting and exploring runs and
benchmark results. They are plain `.py` files, so they diff and review like code.

```bash
uv run marimo edit notebooks/walkthrough.py     # edit interactively
uv run marimo run notebooks/walkthrough.py      # serve as a read-only app
uv run marimo export html notebooks/walkthrough.py -o walkthrough.html --no-include-code
```

- `walkthrough.py`: one test clip from video to breathing trace, every method on
  it, the benchmark, two ablations, and the time-stretch augmentation ablation
  (scores and a trace comparison against the no-stretch network), the training
  set's breathing-rate distribution, and test performance by breathing rate
  (`utils/breathing.py`; the latter needs Zephyr's outputs on all 24 test clips,
  cached after the first run). Inference only (CPU is enough, about a
  minute for a new clip, then cached in `cache/`); it never trains a network. A
  first cell lists any missing input with the command that produces it.
- `utils/animation.py`: `session_movie(...)` renders a whole test recording (both
  videos) as a real-time MP4 of the scrubber figure (video, the four channels, a
  scrolling trace window), e.g. `uv run --project .. python -m utils.animation 6
  figures/rec-6.mp4` from `notebooks/` (about 30 min for a 10 min session;
  `--seconds 10` for a preview). `pipeline_gif(...)` renders a short excerpt as a
  GIF. Neither is used by the notebook.
- `utils/style.py`: figure sizes, fonts and one fixed colour per method, channel
  and objective. Every figure calls `style.use_style()` and takes its colours
  from here.
- `utils/plots.py`: reusable figure components. `utils/results.py`: where results
  live and how they are loaded.

Notebooks only read results (`runs/`, `benchmarks/`, `data/`); they never write
into them. Keep notebooks declarative: logic that a second notebook would need
goes into `utils/`.

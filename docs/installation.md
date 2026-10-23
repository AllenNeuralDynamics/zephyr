# Installation

Zephyr runs on Windows, macOS and Linux. Training wants a CUDA GPU. Everything else,
including the app, runs on any desktop.

## Requirements

- **Python 3.11 or newer.**
- **[uv](https://docs.astral.sh/uv/)**, the package manager the repository is set up
  for.
- **`ffmpeg` and `ffprobe`** on your `PATH`. Zephyr decodes video with them.
- **A display**, for the app only. Training, prediction and scoring run headless.

??? tip "Installing ffmpeg"

    === "Windows"

        ```powershell
        winget install Gyan.FFmpeg
        ```

    === "macOS"

        ```bash
        brew install ffmpeg
        ```

    === "Linux"

        ```bash
        sudo apt-get install ffmpeg
        ```

    Check that it is found with `ffprobe -version`.

## Install from a clone

```bash
git clone https://github.com/AllenNeuralDynamics/zephyr
cd zephyr
```

Then pick the extras you need:

=== "GPU + app (recommended)"

    ```bash
    uv sync --locked --all-packages --extra gpu --extra ui
    ```

=== "CPU + app"

    ```bash
    uv sync --locked --all-packages --extra cpu --extra ui
    ```

=== "Headless GPU (cluster)"

    ```bash
    uv sync --locked --all-packages --extra gpu
    ```

| Extra | Adds | Needed for |
| --- | --- | --- |
| `gpu` | PyTorch built for CUDA 12.6 | training and prediction on an NVIDIA GPU |
| `cpu` | PyTorch for CPU | machines without a GPU (`gpu` and `cpu` exclude each other) |
| `ui` | PySide6 and pyqtgraph | the app (`zephyr gui`) |

Check the install:

```bash
uv run zephyr --help
uv run zephyr gui        # opens the app
```

!!! note "Run commands through `uv run`"
    `uv run zephyr ...` runs the command in the repository's environment. If you
    activate the environment instead (`.venv\Scripts\activate` on Windows,
    `source .venv/bin/activate` elsewhere), plain `zephyr ...` works too.

## Install a released wheel

Each [GitHub release](https://github.com/AllenNeuralDynamics/zephyr/releases) carries a
wheel of the `zephyr` package. Install it into your own environment with PyTorch:

```bash
pip install torch                       # the build for your platform and CUDA
pip install "zephyr[ui] @ <url of the .whl from the release>"
```

## Hardware

**GPU.** An epoch of 200 steps takes about a minute on a desktop GPU (an RTX 4060 Ti),
so a full run of 200 epochs takes a few hours. Early stopping often ends it sooner. On
CPU it takes about 1.7 s per step, roughly 19 hours for a full run. Without a GPU, give
`zephyr run` the flags `--device cpu --amp off --num-workers 0`.

**Disk.** The feature cache keeps each clip's cropped gray frames, one byte per pixel,
at `select_fs`. A 96 × 96 box at 60 Hz takes about **2 GB per hour of video**. The app
tells you how much space a step needs before it fills the cache. Once a clip is cached,
training and scoring no longer read its video.

**Memory.** Inference runs in overlapping windows, so clip length is not limited by GPU
memory.

## Next steps

- New to zephyr? Follow the [tutorial](tutorial/index.md).
- Bringing your own recordings? Read [Preparing your own data](guides/your-data.md)
  first.

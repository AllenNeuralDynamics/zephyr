# Working across machines

A common setup: place boxes on a desktop, preprocess and train on a GPU machine or
cluster, then read the results back on the desktop. Zephyr is built for this. Every
input is a TOML file of local paths, every step is a command, and the steps after
preprocessing need no video.

## What each step needs

| Step | Needs | Not needed |
| --- | --- | --- |
| Place boxes | the clip list and `frames/` | the videos |
| Check breathing | the clip list and the sensor files | the videos |
| Preprocess | the videos, timestamps and sensor files | |
| Train, evaluate | the clip list, the feature cache and the sensor files | the videos |
| Predict | the videos (or their cache entries) and the checkpoint | the sensor files |

So you can:

1. Run `zephyr annotate --prepare` once where the videos are, and copy `frames/` to your
   desktop to place boxes.
2. Copy the clip list back, preprocess on the machine that has the videos, and train
   there or anywhere the cache is copied to.

## Headless machines

Training, prediction and scoring run without a display. Install without the `ui` extra:

```bash
uv sync --locked --all-packages --extra gpu
```

The app's **Copy command** gives you the exact command line of any step, to paste into a
job script.

## A copy of a dataset elsewhere

Cache keys, the training record in each checkpoint and the run outputs name clips by
their **absolute paths**. A cache made at `/data/zephyr-dataset` is therefore not found
from a copy at `C:/Data/zephyr-dataset`, and a checkpoint would not recognise its own
training clips there.

Say which folder is a copy of which with `ZEPHYR_PATH_MAP`. The copy is then named as
the original everywhere a path is an identity, while files are still read from the
copy:

=== "Linux / macOS"

    ```bash
    ZEPHYR_PATH_MAP="/mnt/copy/zephyr-dataset=/data/zephyr-dataset" uv run zephyr evaluate ...
    ```

=== "Windows (PowerShell)"

    ```powershell
    $env:ZEPHYR_PATH_MAP = "C:/Data/zephyr-dataset=/data/zephyr-dataset"
    uv run zephyr evaluate ...
    ```

Separate several pairs with `;`. With it set, a cache made on a cluster and copied to
your machine is used as it is, and one preprocessed locally is the one the cluster would
make. Without it, nothing changes.

!!! warning "Never rename cache entries"
    To use a dataset from a new place, set `ZEPHYR_PATH_MAP`. Do not move or rename
    files in the cache.

## Sharing a cache

Cache entries are keyed by video, preprocessing recipe and box, so one cache folder can
safely serve many clip lists and experiments. Re-running `zephyr preprocess` fills in
only what is missing or changed.

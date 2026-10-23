# Troubleshooting

Every check names the clip it failed on. Run `zephyr clips check <list>` (**Check
files** in the app) to see every problem of a list at once.

!!! tip "Fix the config, not the check"
    When zephyr refuses something, the refusal protects a result: a video on both sides
    of a fold, a missing box, a clip a checkpoint trained on. Change the clip list or
    experiment, never work around the check.

## Clip lists

| Message | Cause and fix |
| --- | --- |
| `cannot derive timestamps: pattern ... does not match the video name` | Your file names differ from the default. Add `[derive]` (**Other file names**), or give the paths explicitly. |
| `N file(s) do not exist (mark an unlabelled clip thermistor = false)` | A derived file is missing, or the clip has no breathing sensor. Scan videos without sensors with `--unlabelled`. |
| `... does not load` on the Clips page | The TOML has an error or an unknown key. **Edit TOML**, fix it, then **Reload**. |

## Preprocessing

| Message | Cause and fix |
| --- | --- |
| `decoded N frames but the timestamp parquet has M rows` | The timestamps are not one row per frame: dropped frames, or timestamps from another file. |
| `select_fs 60.0 Hz exceeds the clip's native rate` | The camera is slower than `select_fs`. Lower `select_fs` in `[preprocess]`. |
| `Digital filter critical frequencies must be 0 < Wn < fs/2` | The breathing sensor is sampled at 80 Hz or less. |
| `boxes of mixed sizes` | Every clip must crop to one box size. Re-place them at one size. |
| `N clip(s) have no box` | Place the boxes (Boxes page, or `zephyr annotate`). |
| `ffmpeg` / `ffprobe` not found | Install ffmpeg and put it on `PATH` (see [Installation](../installation.md)). |

## Experiments and training

| Message | Cause and fix |
| --- | --- |
| `test group ...: ... is also a training clip` | A video is on both sides of a fold. Fix the fold's groups. |
| `groups [...] are not in the list` | A fold names a group the list does not have. Group names are strings: `"3"`, not `3`. |
| `fold name ... must be a plain file name` | Use only letters, digits, `.`, `_` and `-`. |
| `... not in the feature cache` | The clip was not preprocessed with this recipe and box, or its events changed. Re-run `zephyr preprocess` on the list. |
| `CUDA out of memory` | Lower `batch_size`, or pass `--amp bf16` if you turned it off. |
| very slow training | You are on CPU. Check `torch.cuda.is_available()`, and that you installed the `gpu` extra. |

## Scoring and prediction

| Message | Cause and fix |
| --- | --- |
| `evaluate` refuses clips | The checkpoint trained on them. Score held-out clips only. |
| `predict` refuses the crop size | Boxes must be the size the network trained on. Make the list with `clips scan --checkpoint`, which fixes the size. |
| a cache made elsewhere is not found | Paths are identities. Set `ZEPHYR_PATH_MAP` (see [Working across machines](../guides/machines.md)). |

## The app

| Problem | Fix |
| --- | --- |
| `the zephyr app needs the ui extra` | `uv sync --extra ui` (with your `cpu` or `gpu` extra). |
| the window opens off-screen | It centres itself on the screen under the mouse. Move the mouse to the screen you want and start it again. |
| a command failed | Its log opens on its own. **Copy command** gives the line to run in a terminal for the full traceback. |

Still stuck? Open an issue on
[GitHub](https://github.com/AllenNeuralDynamics/zephyr/issues) with the command, the
full message, and your clip list.

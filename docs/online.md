# Try it online!

Zephyr runs in your browser: choose a video, place the crop box on the snout, and get
the predicted breathing as a plot and as a file. Nothing to install, and the video never
leaves your computer: it is decoded and predicted on your own GPU.

[Open Zephyr in the browser](app/){ .md-button .md-button--primary }

## What it needs

- **A recent Chrome or Edge.** The page decodes the video with WebCodecs and runs the
  network with WebGPU. Without a usable GPU it falls back to the CPU (**CPU only**),
  much more slowly.
- **An MP4** (H.264), ordinary or fragmented, as cameras and ffmpeg write them.
- **The camera's frame rate.** The page fills in the container's; check it, since frame
  *i* is taken to be at *i* / rate seconds (as `zephyr predict --video --frame-rate`
  does).

## What you get

The same columns as [`zephyr predict`](reference/outputs.md): per 60 Hz sample its
`time`, the video `frame` there, the z-scored breathing `signal`, the onset head's
`onset` probability and each `inhale` (at the network's own onset threshold), as a
parquet file to download.

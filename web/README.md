# Zephyr in the browser

zephyr's breathing network running in the browser, on the computer's own GPU. Choose an
mp4, place the crop, and get the predicted breathing as a plot and as a parquet with
`zephyr predict`'s columns (`time`, `frame`, `signal`, `onset`, `inhale`). The video
never leaves the machine. Published with the documentation as **Try it online!**
(`docs/online.md`; the app is the site's `app/`).

Needs a browser with WebCodecs and WebGPU (a recent Chrome or Edge); without a GPU it
runs on the CPU through WebAssembly (**CPU only**), much more slowly. It reads ordinary
and fragmented MP4s (the datasets' clips are fragmented): only the index boxes are read
whole (`moov`, and each fragment's `moof`), the frames straight from the file.

## Running it

```bash
cd web
npm install
npm run models       # the networks of models.json, from their releases
npm run dev          # http://localhost:5173
```

`npm run build` writes a static site to `web/dist/`. `.github/workflows/docs.yml` builds
it, with its networks, into the documentation site's `app/` when the documentation is
published; zephyr's CI only checks that it type-checks and builds. GitHub Pages cannot
send the cross-origin isolation headers in `vite.config.ts`, so there the CPU fallback
runs on one thread; WebGPU does not need them.

## The networks

No network is in the repository. `models.json` lists the networks the app offers, each
the three files of one GitHub release of zephyr; `npm run models` downloads them into
`public/models/<name>/` (gitignored) and lists them in `public/models/index.json`.

To publish a network:

1. Export its checkpoint (checked against PyTorch):

   ```bash
   uv run --with onnx --with onnxruntime python web/tools/export_onnx.py \
       --checkpoint <run>/<fold>/seed-<n>/best.pt --out <folder>
   ```

   `<folder>` gets `encoder.onnx`, `temporal.onnx` and `meta.json` (crop, recipe,
   normalisation, onset threshold, and the checkpoint with its SHA-256).
2. Name it in `models.json` (`{ "name": ..., "release": ... }`) with the zephyr release
   that will publish it (`vX.Y.Z`), before that release is made.
3. Make the release as a draft, attach the three files, then publish it. Publishing
   deploys the documentation, whose build fetches them: files attached after
   publishing would miss it. A release carries one network (the file names are fixed);
   a network stays where it was published until `models.json` names another release.

The app offers `finetune-df-cv-v4-cv_0-df`: `full-gray-all-cv-v4`'s fold `cv_0` network
(face, side and front cameras of VR foraging) refined on the side and bottom cameras of
the dynamic-foraging dataset (`zephyr-df-dataset`), a cross-validation network with its
test animals held out.

## How it works

| Step | zephyr (`zephyr predict --video`) | Here |
| --- | --- | --- |
| Read the video | ffmpeg | mp4box.js reads the index; each frame's bytes are read from the file and decoded with WebCodecs (hardware) |
| Frame times | `--frame-rate` (frame *i* at *i* / rate s) | the same; the container's rate is filled in |
| Pick frames | nearest each *k* / 60 s | the same |
| Crop | scale the frame (area) so the box becomes the network's crop, crop, full-range gray | area-average only the box's luma onto the crop, video range to full range |
| Network | PyTorch | two ONNX graphs on ONNX Runtime Web (WebGPU): the frame encoder on chunks of crops as they decode, then the temporal network in overlapping 1024-sample windows |
| Between the two | `resample_embeddings` | the same linear interpolation, in JavaScript |
| Inhales | the onset head's peaks above the checkpoint's threshold, 50 ms apart | the same (`findPeaks` mirrors `scipy.signal.find_peaks`) |

Only one 128-number embedding per kept frame is held, so an hour of video takes about
130 MB, not the 2.4 GB of crops.

### Speed

Decoding sets the pace, not the network. On a 240 Hz 720x540 face video (Edge, one
GPU), measured on a 60-second clip:

| | frames/s |
| --- | --- |
| decoding every frame | ~2,400 |
| decoding only what is needed (frames no other frame refers to are skipped unless kept: half of them here) | ~4,300 |
| that, and copying each kept frame's box off the GPU | ~2,000 |

so an hour at 240 Hz should take about 9 minutes. A 5-minute 500 Hz dynamic-foraging
clip took about a minute. `npm run bench` measures it.

### Matching zephyr

Against `zephyr predict --video` with the same checkpoint, box and frame rate:

| Video | Signal correlation | Inhales at the same sample | Within one |
| --- | --- | --- | --- |
| 60 s of a 240 Hz VR face video (`full-gray-all-cv-v4`, `cv_0`) | 0.99986 | 195 of 200 | 199 |
| 300 s of a 500 Hz dynamic-foraging side video, a held-out animal (`finetune-df-cv-v4-cv_0-df`; the dataset's fragmented clip and an ordinary remux of it alike) | 0.9993 | 702 of 729 | 725 |

The rest is the crop's luma: within one gray level of ffmpeg's everywhere (69% exact on
the face video), since ffmpeg's area scaling and range conversion round differently.

## Checking it

Test videos go in `web/testdata/` (gitignored).

- `tools/pipeline.py`: the app's algorithm in numpy and ONNX Runtime, compared with a
  zephyr prediction.
- `tools/check_luma.py`, `tools/check_gpu_luma.py`: the crop's gray against ffmpeg's.
- `tests/predict.spec.ts` (`npm test`, in Edge): uploads a clip, places the box,
  predicts and downloads the parquet. `VIDEO`, `RATE`, `BOX=x,y,size` (the video's own
  pixels) and `OUT` choose another video.
- `tools/compare.py`: that parquet against zephyr's.

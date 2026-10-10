"""Rebuild the thermistor targets of clips already in a feature cache.

A cache entry whose event set is stale (made before event sets existed, or
under another ``[events]``) is refused until ``zephyr preprocess`` rebuilds
its target.  This does that for every clip list and cache given, but only for
clips the cache already holds: the video arrays are kept, nothing is decoded,
and a clip missing from a cache is skipped (and reported), never added.

    uv run python scripts/rebuild_targets.py --cache data/features-v2 \\
        --clips packages/zephyr-benchmarks/configs/clips/*.toml ephys/clips-96.toml
"""

import argparse
import json
from pathlib import Path

from zephyr import features
from zephyr.config import ClipList, load
from zephyr.preprocess import preprocess_clip

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--cache", type=Path, nargs="+", required=True)
ap.add_argument("--clips", type=Path, nargs="+", required=True)
ap.add_argument("--dry-run", action="store_true", help="only report what is stale")
ap.add_argument(
    "--select-fs",
    type=float,
    help="override the lists' select_fs, for a cache made at another frame rate "
    "(data/features-v2-30hz holds the face lists at 30)",
)
args = ap.parse_args()

for cache in args.cache:
    for path in args.clips:
        try:
            clip_list = load(ClipList, path)
        except ValueError as exc:
            print(f"{path}: skipped, does not load ({str(exc).splitlines()[0]})")
            continue
        params = clip_list.preprocess
        if args.select_fs is not None:
            params = params.model_copy(update={"select_fs": args.select_fs})
        counts = {"current": 0, "rebuilt": 0, "not cached": 0, "unlabelled": 0}
        for clip in clip_list.resolve():
            if not clip.labelled:
                counts["unlabelled"] += 1
                continue
            sidecar = features.paths(cache, features.prefix(clip, params))["sidecar"]
            if not sidecar.is_file() or json.loads(sidecar.read_text())["key"] != (
                features.cache_key(clip, params)
            ):
                counts["not cached"] += 1
                continue
            if features.lookup(clip, params, cache) is not None:
                counts["current"] += 1
                continue
            if not args.dry_run:
                preprocess_clip(clip, params, cache)
            counts["rebuilt"] += 1
        if counts["current"] + counts["rebuilt"]:
            summary = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
            verb = "would rebuild" if args.dry_run else "done"
            print(f"{cache} <- {path}: {summary} ({verb})")

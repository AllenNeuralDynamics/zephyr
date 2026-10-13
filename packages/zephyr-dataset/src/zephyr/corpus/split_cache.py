"""Rewrite a feature cache made before zephyr stored each channel group in its own file.

Each entry's ``feat-`` array (all four channels) becomes ``gray-``, ``diff-`` and
``flow-`` files, bit for bit, so a network reads only the channels it uses. In place by
default, local or remote: the group files first, then the sidecar records them, then
``feat-`` goes, so an interrupted run leaves every entry readable and a rerun goes on.
With ``--out`` (a local folder) the source is kept, e.g. for the read-only ``data/``.

    python -m zephyr.corpus.split_cache s3://bucket/zephyr-cache/<recipe> --workers 16
    python -m zephyr.corpus.split_cache data/features-v2 --out runs/features-v2-split
"""

import argparse
import json
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from fsspec.core import url_to_fs

from zephyr import features
from zephyr.channels import CHANNEL_GROUPS, CHANNEL_NAMES

SMALL = ("frame_times", "baselines", "times", "target", "events")
"""Entry files besides the channels; copied as they are when writing to ``--out``."""


def entries(fs, folder: str) -> list[str]:
    """Prefixes of the entries in *folder* (one ``{prefix}.json`` sidecar each)."""
    names = [Path(p).name for p in fs.ls(folder, detail=False)]
    return sorted(
        n[: -len(".json")]
        for n in names
        if n.endswith(".json") and not n.startswith("channel_stats-")
    )


def split_entry(fs, folder: str, name: str, scratch: Path, out: Path | None) -> str:
    """Split one entry; return what was done (``split``, ``copied`` or ``already``)."""
    remote = {
        key: f"{folder}/{path.name}"
        for key, path in features.paths(Path(), name).items()
    }
    sidecar = json.loads(fs.cat_file(remote["sidecar"]))
    dst = features.paths(out, name) if out is not None else None
    if "groups" in sidecar:
        if out is None:
            return "already"
        for key in (*SMALL, *features.GROUPS):
            if fs.exists(remote[key]):
                fs.get_file(remote[key], str(dst[key]))
        (dst["sidecar"]).write_text(json.dumps(sidecar, indent=2))
        return "copied"

    work = Path(tempfile.mkdtemp(dir=scratch))
    try:
        local = work / f"feat-{name}.npy"
        fs.get_file(remote["features"], str(local))
        combined = np.load(local, mmap_mode="r")
        for group in features.GROUPS:
            index = [CHANNEL_NAMES.index(n) for n in CHANNEL_GROUPS[group]]
            part = work / f"{group}-{name}.npy"
            np.save(part, np.ascontiguousarray(combined[:, index]))
            if out is None:
                fs.put_file(str(part), remote[group])
            else:
                shutil.move(part, dst[group])
        del combined
        if out is not None:
            for key in SMALL:
                if fs.exists(remote[key]):
                    fs.get_file(remote[key], str(dst[key]))
        text = json.dumps(sidecar | {"groups": list(features.GROUPS)}, indent=2)
        if out is None:
            fs.pipe_file(remote["sidecar"], text.encode())  # now the groups count
            fs.rm_file(remote["features"])
        else:
            dst["sidecar"].write_text(text)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return "split"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "cache", help="cache folder: a local path or any fsspec URL (s3://...)"
    )
    p.add_argument(
        "--out", type=Path, help="write the split cache here (local) instead"
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--scratch", type=Path, default=Path(tempfile.gettempdir()))
    a = p.parse_args(argv)

    fs, _ = url_to_fs(a.cache)
    folder = a.cache.rstrip("/").replace("\\", "/")
    if a.out is not None:
        a.out.mkdir(parents=True, exist_ok=True)
        for stats in fs.glob(f"{folder}/channel_stats-*.json"):
            fs.get_file(stats, str(a.out / Path(stats).name))
    a.scratch.mkdir(parents=True, exist_ok=True)
    names = entries(fs, folder)
    print(f"{folder}: {len(names)} entries -> {a.out or 'in place'}", flush=True)

    counts: dict[str, int] = {}
    lock, began = threading.Lock(), time.time()
    with ThreadPoolExecutor(a.workers) as pool:
        futures = {
            pool.submit(split_entry, fs, folder, n, a.scratch, a.out): n for n in names
        }
        for i, future in enumerate(as_completed(futures), 1):
            done = future.result()
            with lock:
                counts[done] = counts.get(done, 0) + 1
                if i % 50 == 0 or i == len(names):
                    rate = i / max(time.time() - began, 1e-9) * 3600
                    print(
                        f"  [{i}/{len(names)}] {counts} | {rate:.0f} entries/h",
                        flush=True,
                    )


if __name__ == "__main__":
    main()

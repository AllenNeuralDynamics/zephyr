"""The local copy of the dataset the notebooks read clips from (``ROOT_ENV``), and the
notebooks' own cache (``CACHE``)."""

import functools
import os
from pathlib import Path

from zephyr.config import PATH_MAP_ENV, ClipList, load

CACHE = Path(__file__).resolve().parents[1] / "cache"
"""``notebooks/cache/``: everything the notebooks compute once (ignored by git)."""
ROOT_ENV = "ZEPHYR_TEST_ROOT"
"""The local folder; the notebooks use it (or ``D:/zephyr-local-v2``) by default."""
DATASET_ROOT = "/data/zephyr-dataset"
"""Where the dataset the copy was taken from lives (the cache keys name it)."""
RECIPE = "features-360x270-box96x96-60hz"


def use_copy(root: Path) -> None:
    """Name *root*'s dataset copy as the original, so its cache entries are found."""
    os.environ[PATH_MAP_ENV] = f"{(root / 'dataset').as_posix()}={DATASET_ROOT}"


@functools.cache
def _clips(root: Path) -> dict[str, tuple]:
    """Clip stem -> (resolved clip, recipe) over the test lists."""
    out = {}
    for path in sorted((root / "dataset").glob("full_test_*.toml")):
        clip_list = load(ClipList, path, check_files=False)
        out |= {c.source.stem: (c, clip_list.preprocess) for c in clip_list.resolve()}
    return out

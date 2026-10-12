"""The S3 bucket: an anonymous filesystem and the raw asset folders of a session.

A derived asset lists its raw source session by name; the raw folder is found under
that name as given, then with and without a ``behavior_`` prefix. Tests pass any
fsspec filesystem (e.g. ``memory``) in place of S3.
"""

from concurrent.futures import ThreadPoolExecutor


def filesystem():
    import s3fs

    return s3fs.S3FileSystem(anon=True)


def candidates(source: str) -> list[str]:
    bare = source.removeprefix("behavior_")
    return list(dict.fromkeys([source, bare, "behavior_" + bare]))


def find(fs, bucket: str, source: str, subfolder: str) -> str | None:
    """``<bucket>/<raw name>/<subfolder>`` for the first raw name that has it."""
    for name in candidates(source):
        path = f"{bucket}/{name}/{subfolder}"
        if fs.exists(path):
            return path
    return None


def thread_map(fn, items, workers: int) -> list:
    with ThreadPoolExecutor(max(1, workers)) as pool:
        return list(pool.map(fn, items))

"""Reading and writing the dataset folder, local or remote (any fsspec URL, e.g. S3).

Paths are :class:`upath.UPath`. A write never leaves a partial file: a local one goes
through a temporary file, a remote one is a single upload (S3 makes an object visible
only once it is complete), so nothing is ever renamed or deleted remotely. Credentials
for a remote root come from the environment (``AWS_PROFILE``; ``--profile`` sets it).
"""

import io
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from upath import UPath


def root(value) -> UPath:
    return value if isinstance(value, UPath) else UPath(str(value))


def is_local(path) -> bool:
    return isinstance(path, Path) and getattr(path, "protocol", "") in ("", "file")


def write_bytes(path: UPath, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_local(path):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    else:
        path.write_bytes(data)


def write_text(path: UPath, text: str) -> None:
    write_bytes(path, text.encode("utf-8"))


def read_parquet(path: UPath, **kwargs) -> pd.DataFrame:
    with path.open("rb") as handle:
        return pd.read_parquet(handle, **kwargs)


def schema_metadata(path: UPath) -> dict[bytes, bytes]:
    with path.open("rb") as handle:
        return pq.read_schema(handle).metadata or {}


def write_table(table: pa.Table, path: UPath, **kwargs) -> None:
    buffer = io.BytesIO()
    pq.write_table(table, buffer, **kwargs)
    write_bytes(path, buffer.getvalue())


def write_frame(frame: pd.DataFrame, path: UPath) -> None:
    write_table(pa.Table.from_pandas(frame, preserve_index=False), path)


def names(folder: UPath, pattern: str = "*") -> set[str]:
    """File names in *folder* matching *pattern* (one listing, not a request each)."""
    if not folder.exists():
        return set()
    return {p.name for p in folder.glob(pattern)}

"""Fake Harp files and camera timestamps for the zephyr-dataset tests."""

import numpy as np
import pandas as pd

from zephyr.corpus import harp


def camera_frame(
    n=2400, start=5000.0, rollback=None, drop=None, hz=240.0
) -> pd.DataFrame:
    """A *hz* camera's ``metadata.csv`` rows: *rollback* puts a spike + rollback of the
    Harp clock at that row; *drop* removes that frame (a dropped frame)."""
    number = np.arange(n) + 1000
    cam = (np.arange(n) * 1e9 / hz).astype(np.int64) + 7_000_000_000
    ref = start + np.arange(n) / hz
    if rollback is not None:
        ref[rollback] += 50.0  # the spike ...
        ref[rollback + 1] -= 1.0  # ... and the rollback after it
    frame = pd.DataFrame(
        {"ReferenceTime": ref, "CameraFrameNumber": number, "CameraFrameTime": cam}
    )
    if drop is not None:
        frame = frame.drop(index=drop).reset_index(drop=True)
    return frame


def harp_bytes(times, values, address=32) -> bytes:
    """Timestamped U16 event messages, as a Harp device logs them."""
    dtype = np.dtype(
        [
            ("type", "u1"),
            ("length", "u1"),
            ("address", "u1"),
            ("port", "u1"),
            ("payload_type", "u1"),
            ("seconds", "<u4"),
            ("ticks", "<u2"),
            ("value", "<u2"),
            ("checksum", "u1"),
        ]
    )
    msgs = np.zeros(len(times), dtype)
    msgs["type"], msgs["length"], msgs["address"] = 3, dtype.itemsize - 2, address
    msgs["port"], msgs["payload_type"] = 255, harp.TIMESTAMPED | 0x02
    times = np.asarray(times, float)
    msgs["seconds"] = np.floor(times).astype(np.uint32)
    msgs["ticks"] = np.round((times - np.floor(times)) / 32e-6).astype(np.uint16)
    msgs["value"] = values
    return msgs.tobytes()

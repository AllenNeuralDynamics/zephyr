"""Fake Harp files for the zephyr-dataset tests."""

import numpy as np

from zephyr.corpus import harp


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

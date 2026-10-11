"""Read one register of a Harp device folder (``device.yml`` plus ``<name>_<address>.bin``).

Each ``.bin`` file is a run of fixed-size Harp messages: type, length, address, port,
payload type, timestamp (seconds and 32 us ticks), payload, checksum. A register may be
split over several files, read in natural order of their names.
"""

import re

import numpy as np
import pandas as pd
import yaml

DTYPES = {
    0x01: "u1",
    0x02: "<u2",
    0x04: "<u4",
    0x08: "<u8",
    0x81: "i1",
    0x82: "<i2",
    0x84: "<i4",
    0x88: "<i8",
    0x44: "<f4",
}
TIMESTAMPED = 0x10


def natural_key(text: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", text))


def address(device_yml: bytes | str, register: str) -> int:
    """The register's address from the device's ``device.yml``."""
    registers = yaml.safe_load(device_yml)["registers"]
    if register not in registers:
        raise KeyError(f"register {register!r} not in device.yml")
    return int(registers[register]["address"])


def chunks(names: list[str], device: str, addr: int) -> list[str]:
    """The ``.bin`` files of one register among *names*, in reading order."""
    pattern = re.compile(rf"^{re.escape(device)}_{addr}(?:_.*)?\.bin$")
    found = [n for n in names if pattern.match(n.rsplit("/", 1)[-1])]
    return sorted(found, key=natural_key)


def read_messages(data: bytes) -> pd.DataFrame:
    """``Time`` (seconds) and ``value`` (first payload element) of every message."""
    if not data:
        return pd.DataFrame({"Time": [], "value": []})
    buf = np.frombuffer(data, np.uint8)
    stride = int(buf[1]) + 2
    if len(buf) % stride:
        raise ValueError("Harp data is not a whole number of messages")
    msgs = buf.reshape(-1, stride)
    kind = int(msgs[0, 4])
    if not kind & TIMESTAMPED:
        raise ValueError("Harp messages without timestamps")
    seconds = msgs[:, 5:9].copy().view("<u4").ravel()
    ticks = msgs[:, 9:11].copy().view("<u2").ravel()
    payload = msgs[:, 11:-1].copy().view(DTYPES[kind & ~TIMESTAMPED])
    return pd.DataFrame(
        {"Time": seconds + ticks * 32e-6, "value": payload[:, 0].astype(float)}
    )

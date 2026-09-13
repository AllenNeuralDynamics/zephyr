"""Multi-channel per-frame input stack: gray, signed diff, flow_x, flow_y.

``diff`` is ``I(t) - I(t - dt)``; ``flow_x``/``flow_y`` are signed DIS optical
flow. Both are measured over the achieved interval ``dt`` (nearest source frame
to ``t - TAU``) and scaled by ``TAU / dt``, so the stored value is displacement
per :data:`MOTION_TAU_S` regardless of camera frame rate.

Preprocessing always writes all four channels; :class:`ChannelSet` selects the
training-time subset.

Quantisation: ``gray`` stored as-is; ``diff`` as ``round(diff) + 128`` (not
exactly lossless, since the ``TAU/dt`` scale isn't integer-valued); flow
companded through ``asinh`` rather than clipped linearly, for fine steps near
zero while still covering large excursions.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

CHANNEL_NAMES = ("gray", "diff", "flow_x", "flow_y")
N_CHANNELS = len(CHANNEL_NAMES)

DIFF_CLIP = 127
"""Frame-difference clip in intensity counts, after normalisation to ``TAU``."""

MOTION_TAU_S = 1.0 / 60.0
"""Canonical motion baseline, in seconds. Must be >= the coarsest native frame
period to support (16.7 ms covers 70/120/240/504 fps) and >= the selection
gap."""

FLOW_SCALE_PX = 0.1
"""Knee of the ``asinh`` flow companding, in pixels."""

FLOW_CLIP_PX = 16.0
"""Flow range bound, in pixels."""

CHANNEL_GROUPS: dict[str, tuple[str, ...]] = {
    "gray": ("gray",),
    "diff": ("diff",),
    "flow": ("flow_x", "flow_y"),
    "all": CHANNEL_NAMES,
}
"""Group names accepted when selecting channels, beyond the raw channel names."""


@dataclass(frozen=True)
class ChannelSet:
    """The channels one model is trained on, always ordered as stored.

    :attr:`indices` positions within the stored ``(T, 4, H, W)`` array;
    :meth:`position` positions within the selection itself.
    """

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("a ChannelSet needs at least one channel")
        unknown = [n for n in self.names if n not in CHANNEL_NAMES]
        if unknown:
            raise ValueError(
                f"unknown channel(s) {unknown}; stored channels are "
                f"{list(CHANNEL_NAMES)}"
            )
        if list(self.names) != [n for n in CHANNEL_NAMES if n in set(self.names)]:
            raise ValueError(
                f"channels must be in stored order {list(CHANNEL_NAMES)}, "
                f"got {list(self.names)} -- use ChannelSet.parse, which sorts"
            )

    @classmethod
    def parse(cls, spec: str | Iterable[str]) -> "ChannelSet":
        """Build from ``"gray+diff+flow"``, ``"gray,diff"``, or a list of names.

        Group names expand, duplicates collapse, result is ordered as stored.
        """
        tokens = (
            [t for t in re.split(r"[+,\s]+", spec.strip()) if t]
            if isinstance(spec, str)
            else [str(t) for t in spec]
        )
        if not tokens:
            raise ValueError(f"no channels named in {spec!r}")
        wanted: set[str] = set()
        for token in tokens:
            expanded = CHANNEL_GROUPS.get(token)
            if expanded is None:
                if token not in CHANNEL_NAMES:
                    raise ValueError(
                        f"unknown channel {token!r}; choose from "
                        f"{sorted(set(CHANNEL_NAMES) | set(CHANNEL_GROUPS))}"
                    )
                expanded = (token,)
            wanted.update(expanded)
        return cls(tuple(n for n in CHANNEL_NAMES if n in wanted))

    def __len__(self) -> int:
        return len(self.names)

    def __str__(self) -> str:
        return "+".join(self._folded())

    @property
    def is_complete(self) -> bool:
        return self.names == CHANNEL_NAMES

    @property
    def indices(self) -> tuple[int, ...]:
        """Positions of these channels in the stored array."""
        return tuple(CHANNEL_NAMES.index(n) for n in self.names)

    def _folded(self) -> list[str]:
        """Channel names with the complete flow pair folded back to ``flow``."""
        flow = ("flow_x", "flow_y")
        parts = [n for n in self.names if n not in flow]
        if all(f in self.names for f in flow):
            parts.append("flow")
        else:
            parts.extend(f for f in flow if f in self.names)
        return parts

    @property
    def slug(self) -> str:
        """Filesystem-safe name for this selection, for run directories."""
        return "_".join(self._folded())

    def position(self, name: str) -> int | None:
        """Index of *name* within the selection, or ``None`` if not selected."""
        return self.names.index(name) if name in self.names else None

    def take(self, block: np.ndarray) -> np.ndarray:
        """Slice the channel axis of a ``(T, C, H, W)`` stored block."""
        if self.is_complete:
            return block
        return block[:, self.indices]

    def take_stats(
        self, mean: np.ndarray, std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slice per-channel statistics measured over *all* stored channels."""
        index = list(self.indices)
        return np.asarray(mean)[index], np.asarray(std)[index]


ALL_CHANNELS = ChannelSet(CHANNEL_NAMES)
"""Every stored channel -- what preprocessing writes, and the training default."""


def _flow_gain(scale_px: float, clip_px: float) -> float:
    """uint8 codes per unit of ``asinh(flow / scale_px)``."""
    return 127.0 / float(np.arcsinh(clip_px / scale_px))


def encode_flow(
    flow: np.ndarray, *, scale_px: float = FLOW_SCALE_PX, clip_px: float = FLOW_CLIP_PX
) -> np.ndarray:
    """Compand a signed flow field in pixels to uint8 codes centred on 128."""
    gain = _flow_gain(scale_px, clip_px)
    codes = 128.0 + gain * np.arcsinh(flow / scale_px)
    return np.clip(codes, 0.0, 255.0).astype(np.uint8)


def decode_flow(
    codes: np.ndarray,
    *,
    scale_px: float = FLOW_SCALE_PX,
    clip_px: float = FLOW_CLIP_PX,
) -> np.ndarray:
    """Inverse of :func:`encode_flow`, returning pixels."""
    gain = _flow_gain(scale_px, clip_px)
    return scale_px * np.sinh((codes.astype(np.float32) - 128.0) / gain)


def make_flow_estimator(
    *, finest_scale: int = 0, variational_iters: int = 5
) -> cv2.DISOpticalFlow:
    """DIS flow configured for sub-pixel displacement."""
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow.setFinestScale(finest_scale)
    flow.setUseSpatialPropagation(True)
    flow.setVariationalRefinementIterations(variational_iters)
    return flow


def channel_encoding(
    *,
    flow_scale_px: float = FLOW_SCALE_PX,
    flow_clip_px: float = FLOW_CLIP_PX,
    motion_tau_s: float = MOTION_TAU_S,
) -> dict[str, dict]:
    """How to map each stored uint8 channel back to physical units.

    Recorded in the preprocessing manifest. ``gray`` in intensity counts;
    ``diff`` in counts per *motion_tau_s*; flow in pixels per *motion_tau_s*
    (invert companding with :func:`decode_flow` first).
    """
    return {
        "gray": {"kind": "identity", "units": "counts"},
        "diff": {
            "kind": "affine",
            "gain": 1.0,
            "offset": -128.0,
            "units": "counts/tau",
        },
        "flow_x": {
            "kind": "asinh",
            "scale_px": flow_scale_px,
            "clip_px": flow_clip_px,
            "units": "px/tau",
        },
        "flow_y": {
            "kind": "asinh",
            "scale_px": flow_scale_px,
            "clip_px": flow_clip_px,
            "units": "px/tau",
        },
        "tau_s": {"kind": "constant", "value": motion_tau_s, "units": "s"},
    }


def encode_stack(
    gray: np.ndarray,
    reference: np.ndarray | None,
    flow_estimator: cv2.DISOpticalFlow,
    *,
    dt: float = MOTION_TAU_S,
    tau: float = MOTION_TAU_S,
    flow_scale_px: float = FLOW_SCALE_PX,
    flow_clip_px: float = FLOW_CLIP_PX,
) -> tuple[np.ndarray, int, int]:
    """Encode one anchor frame into a ``(4, H, W)`` uint8 stack.

    Parameters
    ----------
    gray:
        The anchor frame, uint8.
    reference:
        The source frame nearest ``t - tau``, or ``None`` at clip start (motion
        channels then set to their zero level).
    flow_estimator:
        From :func:`make_flow_estimator`; reused across calls for DIS's
        internal buffers.
    dt:
        Interval *reference* to *gray* actually spans, in seconds. Motion is
        scaled by ``tau / dt``.

    Returns
    -------
    (stack, n_diff_saturated, n_flow_saturated)
        Saturation counted after normalisation.
    """
    height, width = gray.shape
    stack = np.empty((N_CHANNELS, height, width), dtype=np.uint8)
    stack[0] = gray

    if reference is None or dt <= 0:
        stack[1:] = 128
        return stack, 0, 0

    gain = tau / dt

    diff = (gray.astype(np.float32) - reference.astype(np.float32)) * gain
    n_diff_saturated = int((np.abs(diff) > DIFF_CLIP).sum())
    stack[1] = np.clip(np.rint(diff) + 128.0, 0.0, 255.0).astype(np.uint8)

    # DIS asserts on non-contiguous input, and decoded frames arrive as views
    # into a shared read-only buffer -- so normalise before handing them over.
    flow = (
        flow_estimator.calc(
            np.ascontiguousarray(reference), np.ascontiguousarray(gray), None
        )
        * gain
    )
    n_flow_saturated = int((np.abs(flow) > flow_clip_px).sum())
    codes = encode_flow(flow, scale_px=flow_scale_px, clip_px=flow_clip_px)
    stack[2] = codes[..., 0]
    stack[3] = codes[..., 1]

    return stack, n_diff_saturated, n_flow_saturated

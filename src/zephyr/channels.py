"""Build the multi-channel per-frame input stack: gray, signed diff, flow_x, flow_y.

Channel rationale
-----------------
``gray``
    Raw appearance.  The CNN is per-frame and therefore temporally blind, so it
    cannot derive anything time-varying from this channel alone.
``diff``
    ``I(t) - I(t - stride)``.  A hand-crafted temporal high-pass that lets the
    spatial trunk respond to motion at all.
``flow_x`` / ``flow_y``
    Signed dense optical flow.  Kept signed rather than as a magnitude because
    the sign is what distinguishes opposite directions of motion; a magnitude
    channel folds the two together.

Choice of flow algorithm
------------------------
DIS flow, not Farneback or TV-L1: DIS is built for speed on small motion and is
the only one of the three fast enough to run over a full dataset. Variational
refinement and the finest pyramid scale are both enabled so it resolves
sub-pixel displacement rather than giving up at a coarser level.

``stride`` should match the decimation stride used elsewhere, so the motion
channels describe exactly the interval between consecutive *stored* frames.

Storing every channel, training on a subset
-------------------------------------------
Preprocessing always writes all four channels.  Which of them a model is
actually *trained* on is a separate, training-time choice, expressed as a
:class:`ChannelSet` -- so comparing a 1-channel model against a 4-channel one
needs no second preprocessing pass, and every variant reads byte-identical
crops and shares one set of normalisation statistics.

Quantisation
------------
Every channel is stored as uint8 to keep one compact array per clip:

* ``gray`` is already 8-bit, stored as-is.
* ``diff`` is an exact integer difference of two 8-bit frames, so
  ``raw = diff + 128`` is *lossless* for the |diff| <= 127 that covers nearly
  every pixel.
* Flow is companded through ``asinh`` (see below) rather than clipped linearly.

Why flow is companded and not clipped
-------------------------------------
Flow can span several orders of magnitude between the signal of interest and
occasional large motion, so a linear quantiser forces a choice between
resolving the small signal and covering the large excursions.  ``asinh``
resolves both at once -- linear near zero and logarithmic beyond it -- giving
fine steps close to zero while still representing large excursions.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

CHANNEL_NAMES = ("gray", "diff", "flow_x", "flow_y")
N_CHANNELS = len(CHANNEL_NAMES)

DIFF_CLIP = 127
"""Frame-difference clip in intensity counts.  Lossless below this magnitude."""

FLOW_SCALE_PX = 0.1
"""Knee of the ``asinh`` flow companding, in pixels.

Below this the encoding is effectively linear.
"""

FLOW_CLIP_PX = 16.0
"""Flow bound in pixels.  With companding this is large enough that saturation
is negligible, so it functions as a range limit rather than a lossy clip."""

CHANNEL_GROUPS: dict[str, tuple[str, ...]] = {
    "gray": ("gray",),
    "diff": ("diff",),
    "flow": ("flow_x", "flow_y"),
    "all": CHANNEL_NAMES,
}
"""Names accepted when selecting channels, beyond the raw channel names.

``flow`` names both flow planes at once: they are one estimator's two outputs,
and keeping them together is what lets a horizontal flip stay coherent -- a
mirrored frame has to negate ``flow_x``, which is meaningless if only ``flow_y``
is present.
"""


@dataclass(frozen=True)
class ChannelSet:
    """The channels one model is trained on, and where they sit in the array.

    Preprocessing writes all of :data:`CHANNEL_NAMES`; this holds the subset a
    model consumes, always ordered as stored.  Two index spaces, which differ:

    :attr:`indices`
        Positions in the stored ``(T, 4, H, W)`` array.
    :meth:`position`
        Position within the selection -- the block a model actually sees, and
        what :mod:`.augment` indexes by.
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

        Group names from :data:`CHANNEL_GROUPS` are expanded, duplicates
        collapse, and the result is always ordered as stored -- so the same
        selection written two ways gives the same object, and the same slug.
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
        """Channel names with the complete flow pair folded back to ``flow``.

        So a selection is echoed the way it was asked for -- ``gray+diff+flow``
        rather than ``gray+diff+flow_x+flow_y``.  A lone flow plane, which no
        group can name, is left spelled out.
        """
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
        """Slice the channel axis of a ``(T, C, H, W)`` stored block.

        Returned unchanged for the complete set, so the default path costs
        nothing and stays byte-identical to reading the array directly.
        """
        if self.is_complete:
            return block
        return block[:, self.indices]

    def take_stats(
        self, mean: np.ndarray, std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slice per-channel statistics measured over *all* stored channels.

        Statistics are always measured and cached for the full stored set, so
        one cache serves every selection and no two runs can disagree about
        what a channel's mean is.
        """
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
    *, flow_scale_px: float = FLOW_SCALE_PX, flow_clip_px: float = FLOW_CLIP_PX
) -> dict[str, dict]:
    """How to map each stored uint8 channel back to physical units.

    Recorded in the preprocessing manifest so a consumer never has to guess.
    ``gray`` and ``diff`` are in intensity counts, the flow channels in pixels
    (invert with :func:`decode_flow`).
    """
    return {
        "gray": {"kind": "identity", "units": "counts"},
        "diff": {"kind": "affine", "gain": 1.0, "offset": -128.0, "units": "counts"},
        "flow_x": {
            "kind": "asinh",
            "scale_px": flow_scale_px,
            "clip_px": flow_clip_px,
            "units": "px",
        },
        "flow_y": {
            "kind": "asinh",
            "scale_px": flow_scale_px,
            "clip_px": flow_clip_px,
            "units": "px",
        },
    }


def encode_stack(
    gray: np.ndarray,
    reference: np.ndarray | None,
    flow_estimator: cv2.DISOpticalFlow,
    *,
    flow_scale_px: float = FLOW_SCALE_PX,
    flow_clip_px: float = FLOW_CLIP_PX,
) -> tuple[np.ndarray, int, int]:
    """Encode one anchor frame into a ``(4, H, W)`` uint8 stack.

    Parameters
    ----------
    gray:
        The anchor frame, uint8.
    reference:
        The frame ``stride`` earlier, or ``None`` for the first anchor -- in
        which case the motion channels are set to their zero level rather than
        left undefined.
    flow_estimator:
        From :func:`make_flow_estimator`; reused across calls so DIS keeps its
        internal buffers.
    flow_scale_px, flow_clip_px:
        Companding knee and range bound, in pixels.

    Returns
    -------
    (stack, n_diff_saturated, n_flow_saturated)
        The saturation counts let :mod:`.preprocess` report whether the bounds
        are actually appropriate for this data.
    """
    height, width = gray.shape
    stack = np.empty((N_CHANNELS, height, width), dtype=np.uint8)
    stack[0] = gray

    if reference is None:
        stack[1:] = 128
        return stack, 0, 0

    diff = gray.astype(np.int16) - reference.astype(np.int16)
    n_diff_saturated = int((np.abs(diff) > DIFF_CLIP).sum())
    stack[1] = (np.clip(diff, -DIFF_CLIP, DIFF_CLIP) + 128).astype(np.uint8)

    # DIS asserts on non-contiguous input, and decoded frames arrive as views
    # into a shared read-only buffer -- so normalise before handing them over.
    flow = flow_estimator.calc(
        np.ascontiguousarray(reference), np.ascontiguousarray(gray), None
    )
    n_flow_saturated = int((np.abs(flow) > flow_clip_px).sum())
    codes = encode_flow(flow, scale_px=flow_scale_px, clip_px=flow_clip_px)
    stack[2] = codes[..., 0]
    stack[3] = codes[..., 1]

    return stack, n_diff_saturated, n_flow_saturated

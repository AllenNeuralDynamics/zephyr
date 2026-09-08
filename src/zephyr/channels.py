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

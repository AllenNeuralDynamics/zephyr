from dataclasses import dataclass

import cv2
import numpy as np

from .channels import ChannelSet

MOTION_CHANNELS = ("diff", "flow_x", "flow_y")
"""Channels that measure motion between frames rather than appearance."""

# Every function here is handed the block a model actually sees, which holds
# only the selected channels -- so a channel is addressed by
# ``ChannelSet.position``, its index *within that selection*, and never by its
# position in the stored array.  An augmentation whose channel is not selected
# is simply skipped.


@dataclass(frozen=True)
class AugmentConfig:
    """Augmentation strengths.  Every field at its default is a no-op.

    Attributes
    ----------
    select_jitter:
        Maximum perturbation, in selection-grid units, applied independently to
        each input frame's position (its timestamp moves with it).
    motion_noise:
        Standard deviation of extra Gaussian noise on the motion channels only,
        in uint8 code units, drawn per window from a log-uniform band around
        this value. Covers the estimator noise that per-``TAU`` normalisation
        amplifies when the achieved baseline is shorter than ``TAU``.
    shift_px:
        Maximum spatial translation in pixels, drawn uniformly per axis.  Guards
        against the model keying on the crop's absolute position.
    scale:
        Maximum fractional zoom (``0.1`` draws 0.9 to 1.1), drawn uniformly, one per
        window about the crop centre with the edge pixels repeated.  Covers camera
        distance.  The flow channels are companded vectors, which a zoom would have
        to decode, rescale and encode, so a model that reads them refuses it
        (``TrainParams``).
    rotation:
        Maximum rotation in degrees, drawn uniformly per window and applied with the
        zoom about the crop centre.  Covers head tilt and camera mounting.  Flow
        vectors would have to rotate too, so a model that reads them refuses it.
    blur:
        Maximum Gaussian blur sigma in pixels, drawn uniformly per window, on the
        ``gray`` channel only.  Covers focus and optics that differ between rigs.
    brightness / contrast:
        Applied to the ``gray`` channel only, as fractional deviations from 1.0.
        The motion channels are differences and are already invariant to a
        constant offset.  A no-op when ``gray`` is not selected.
    noise:
        Standard deviation of additive Gaussian noise, in uint8 code units,
        applied to every channel before standardisation.
    flip:
        Horizontal mirror probability.  Mirroring must negate ``flow_x``,
        otherwise the flow field contradicts the image -- skipped when
        ``flow_x`` is not selected, since then there is nothing to contradict.
        Off by default when a camera only ever views one side of the animal.

    Note that these do not all apply to every channel, so two models trained on
    different channel sets are not regularised quite equally: a ``gray``-only
    model sees no motion scaling, a motion-only model no photometric jitter.
    """

    select_jitter: float = 0.0
    motion_noise: float = 0.0
    shift_px: int = 0
    scale: float = 0.0
    rotation: float = 0.0
    blur: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    noise: float = 0.0
    flip: float = 0.0

    @property
    def enabled(self) -> bool:
        return (
            self.select_jitter > 0
            or self.motion_noise > 0
            or self.shift_px > 0
            or self.scale > 0
            or self.rotation > 0
            or self.blur > 0
            or self.brightness > 0
            or self.contrast > 0
            or self.noise > 0
            or self.flip > 0
        )


def gather_positions(
    x: np.ndarray, positions: np.ndarray, dtype: type = np.float32
) -> np.ndarray:
    """Linearly interpolate *x* along axis 0 at fractional *positions*.

    Exact integer positions take a plain gather (no interpolation cost).
    """
    n_in = x.shape[0]
    positions = np.clip(positions, 0.0, n_in - 1)
    lo = np.floor(positions).astype(np.intp)
    weight = positions - lo
    if not weight.any():
        return x[lo].astype(dtype)

    hi = np.minimum(lo + 1, n_in - 1)
    # Broadcast the weight over whatever trailing axes the array has.
    weight = weight.astype(dtype).reshape(-1, *([1] * (x.ndim - 1)))
    left = x[lo].astype(dtype)
    right = x[hi].astype(dtype)
    return left * (1.0 - weight) + right * weight


def jitter_positions(
    positions: np.ndarray, rng: np.random.Generator, amount: float
) -> np.ndarray:
    """Perturb each selection position independently; re-sorted to stay monotonic."""
    if amount <= 0:
        return positions
    return np.sort(positions + rng.uniform(-amount, amount, size=positions.shape))


MOTION_NOISE_SPREAD = 2.0
"""Factor the noise level is drawn log-uniformly within, either side of
``AugmentConfig.motion_noise``."""


def noise_scales(
    rng: np.random.Generator, config: AugmentConfig, channels: ChannelSet
) -> np.ndarray | None:
    """Per-channel noise standard deviation for one window, or ``None``.

    Global and motion-only terms are independent Gaussians, combined in
    quadrature into a single sd per channel -- one draw instead of two.
    """
    sd = np.full(len(channels), float(config.noise), np.float32)
    if config.motion_noise > 0:
        log_spread = np.log(MOTION_NOISE_SPREAD)
        extra = config.motion_noise * float(
            np.exp(rng.uniform(-log_spread, log_spread))
        )
        for name in MOTION_CHANNELS:
            position = channels.position(name)
            if position is not None:
                sd[position] = np.hypot(sd[position], extra)
    return sd if sd.any() else None


def zoom(block: np.ndarray, factor: float, angle: float = 0.0) -> np.ndarray:
    """Zoom (and rotate by *angle* degrees) every frame of a ``(T, C, H, W)`` block
    about its centre."""
    height, width = block.shape[2:]
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, factor)
    out = np.empty_like(block)
    for t in range(block.shape[0]):
        frame = np.ascontiguousarray(block[t].transpose(1, 2, 0))  # H, W, C
        warped = cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        out[t] = warped.reshape(height, width, -1).transpose(2, 0, 1)
    return out


def blur(frames: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur every frame of a ``(T, H, W)`` stack, edges repeated."""
    if sigma <= 0:
        return frames
    return np.stack(
        [
            cv2.GaussianBlur(
                np.ascontiguousarray(f), (0, 0), sigma, borderType=cv2.BORDER_REPLICATE
            )
            for f in frames
        ]
    )


def apply_spatial(
    block: np.ndarray,
    rng: np.random.Generator,
    config: AugmentConfig,
    channels: ChannelSet,
) -> np.ndarray:
    """Zoom and rotate, shift, flip, blur, and photometrically jitter a ``(T, C, H, W)``
    float block.

    *block* holds only the channels in *channels*, in that order.
    """
    # Each draw only when its knob is on, so a config without it draws the same
    # random sequence as before the knob existed.
    if config.scale > 0 or config.rotation > 0:
        factor = (
            float(rng.uniform(1.0 - config.scale, 1.0 + config.scale))
            if config.scale > 0
            else 1.0
        )
        angle = (
            float(rng.uniform(-config.rotation, config.rotation))
            if config.rotation > 0
            else 0.0
        )
        block = zoom(block, factor, angle)

    if config.shift_px > 0:
        dy = int(rng.integers(-config.shift_px, config.shift_px + 1))
        dx = int(rng.integers(-config.shift_px, config.shift_px + 1))
        if dy or dx:
            # Edge padding rather than zeros: a zero border is a hard artificial
            # edge that the motion channels would read as a huge displacement.
            block = np.pad(
                block,
                (
                    (0, 0),
                    (0, 0),
                    (abs(dy), abs(dy)),
                    (abs(dx), abs(dx)),
                ),
                mode="edge",
            )
            h, w = block.shape[2], block.shape[3]
            y0 = abs(dy) + dy
            x0 = abs(dx) + dx
            block = block[:, :, y0 : y0 + h - 2 * abs(dy), x0 : x0 + w - 2 * abs(dx)]

    if config.flip > 0 and rng.random() < config.flip:
        block = block[:, :, :, ::-1].copy()
        # A mirrored frame reverses horizontal motion, so the encoded flow_x must
        # be reflected about its 128 centre or it would disagree with the image.
        flow_x = channels.position("flow_x")
        if flow_x is not None:
            block[:, flow_x] = 255.0 - block[:, flow_x]

    gray_channel = channels.position("gray")
    if config.blur > 0 and gray_channel is not None:
        sigma = float(rng.uniform(0.0, config.blur))
        block[:, gray_channel] = blur(block[:, gray_channel], sigma)

    if (config.brightness > 0 or config.contrast > 0) and gray_channel is not None:
        gain = 1.0 + rng.uniform(-config.contrast, config.contrast)
        offset = 255.0 * rng.uniform(-config.brightness, config.brightness)
        gray = block[:, gray_channel]
        block[:, gray_channel] = np.clip(
            (gray - 128.0) * gain + 128.0 + offset, 0.0, 255.0
        )

    sd = noise_scales(rng, config, channels)
    if sd is not None:
        block += rng.standard_normal(block.shape, dtype=np.float32) * sd.reshape(
            1, -1, 1, 1
        )

    return block

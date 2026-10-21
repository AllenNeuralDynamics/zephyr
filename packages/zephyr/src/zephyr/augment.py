from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AugmentConfig:
    """Augmentation strengths.  Every field at its default is a no-op.

    Attributes
    ----------
    select_jitter:
        Maximum perturbation, in selection-grid units, applied independently to
        each input frame's position (its timestamp moves with it).
    shift_px:
        Maximum spatial translation in pixels, drawn uniformly per axis.  Guards
        against the model keying on the crop's absolute position.
    scale:
        Maximum fractional zoom (``0.1`` draws 0.9 to 1.1), drawn uniformly, one per
        window about the crop centre with the edge pixels repeated.  Covers camera
        distance.
    rotation:
        Maximum rotation in degrees, drawn uniformly per window and applied with the
        zoom about the crop centre.  Covers head tilt and camera mounting.
    blur:
        Maximum Gaussian blur sigma in pixels, drawn uniformly per window.  Covers
        focus and optics that differ between rigs.
    brightness / contrast:
        Fractional deviations from 1.0.
    noise:
        Standard deviation of additive Gaussian noise, in uint8 code units,
        applied before standardisation.
    flip:
        Horizontal mirror probability.  Off by default when a camera only ever
        views one side of the animal.
    """

    select_jitter: float = 0.0
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
    block: np.ndarray, rng: np.random.Generator, config: AugmentConfig
) -> np.ndarray:
    """Zoom and rotate, shift, flip, blur, and photometrically jitter a ``(T, 1, H, W)``
    float block of gray frames."""
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
            # Edge padding rather than zeros: a zero border is a hard artificial edge.
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

    if config.blur > 0:
        sigma = float(rng.uniform(0.0, config.blur))
        block[:, 0] = blur(block[:, 0], sigma)

    if config.brightness > 0 or config.contrast > 0:
        gain = 1.0 + rng.uniform(-config.contrast, config.contrast)
        offset = 255.0 * rng.uniform(-config.brightness, config.brightness)
        block[:] = np.clip((block - 128.0) * gain + 128.0 + offset, 0.0, 255.0)

    if config.noise > 0:
        block += rng.standard_normal(block.shape, dtype=np.float32) * np.float32(
            config.noise
        )

    return block

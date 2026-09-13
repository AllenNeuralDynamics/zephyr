from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

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
    time_stretch:
        Bound on the stretch factor.  With *rate_range* unset, factors are drawn
        log-uniformly on ``[1/time_stretch, time_stretch]``.  With *rate_range*
        set, this is the clamp applied to the rate-targeted factor.  1.0
        disables stretching entirely.
    rate_range:
        Target signal-rate band in Hz.  When set, each window's own rate is
        measured and the stretch chosen to land the output at a rate drawn
        uniformly from this band, which flattens the rate distribution the
        model sees rather than merely widening it.
    scale_motion:
        Rescale the motion channels by the stretch factor.  ``diff`` is
        ``I(t) - I(t-dt)``, proportional to ``dt`` for small intervals, so
        changing the effective frame interval must scale it or a stretched
        sample pairs a slow rhythm with fast-rhythm motion magnitudes.  A no-op
        when no motion channel is selected.
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
        Off by default when a camera only ever views one side of the subject.

    Note that these do not all apply to every channel, so two models trained on
    different channel sets are not regularised quite equally: a ``gray``-only
    model sees no motion scaling, a motion-only model no photometric jitter.
    """

    time_stretch: float = 1.0
    rate_range: tuple[float, float] | None = None
    scale_motion: bool = True
    select_jitter: float = 0.0
    motion_noise: float = 0.0
    shift_px: int = 0
    brightness: float = 0.0
    contrast: float = 0.0
    noise: float = 0.0
    flip: float = 0.0

    @property
    def enabled(self) -> bool:
        return (
            self.time_stretch > 1.0
            or self.select_jitter > 0
            or self.motion_noise > 0
            or self.shift_px > 0
            or self.brightness > 0
            or self.contrast > 0
            or self.noise > 0
            or self.flip > 0
        )

    @property
    def max_source_frames(self) -> float:
        """Multiplier on the window's selection-grid span at the widest draw."""
        return max(1.0, self.time_stretch)


def draw_stretch(rng: np.random.Generator, max_factor: float) -> float:
    """Draw a stretch factor log-uniformly on ``[1/max, max]``.

    Log-uniform rather than uniform so that halving and doubling the rate are
    equally likely; a uniform draw on ``[0.5, 2]`` would spend three quarters of
    its mass on speeding the recording up, which is the direction that does not
    need help.
    """
    if max_factor <= 1.0:
        return 1.0
    log_max = np.log(max_factor)
    return float(np.exp(rng.uniform(-log_max, log_max)))


def local_rate_hz(signal: np.ndarray, fs: float = 60.0) -> float:
    """Oscillation rate of one window, by counting prominent peaks.

    Peak counting rather than a spectral peak: a short window gives a
    frequency-domain estimate too little resolution, and its maximum is
    readily captured by envelope drift.
    """
    if len(signal) < 4:
        return 0.0
    x = signal - signal.mean()
    sd = x.std()
    if sd <= 0:
        return 0.0
    peaks, _ = find_peaks(x / sd, distance=max(1, int(fs / 20)), prominence=0.5)
    return len(peaks) / (len(signal) / fs)


def rate_targeted_stretch(
    signal: np.ndarray,
    rng: np.random.Generator,
    config: AugmentConfig,
    fs: float = 60.0,
) -> float:
    """Stretch factor that lands this window's rate inside ``config.rate_range``.

    Falls back to 1.0 when the window's rate cannot be measured.
    """
    if config.rate_range is None:
        return draw_stretch(rng, config.time_stretch)
    rate = local_rate_hz(signal, fs)
    if rate <= 0:
        return 1.0
    desired = rng.uniform(*config.rate_range)
    bound = max(1.0, config.time_stretch)
    return float(np.clip(desired / rate, 1.0 / bound, bound))


def scale_motion_channels(
    block: np.ndarray, stretch: float, channels: ChannelSet
) -> np.ndarray:
    """Scale whichever motion channels are selected about their 128 centre."""
    if stretch == 1.0:
        return block
    for name in MOTION_CHANNELS:
        position = channels.position(name)
        if position is None:
            continue
        block[:, position] = np.clip(
            (block[:, position] - 128.0) * stretch + 128.0, 0.0, 255.0
        )
    return block


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


def apply_spatial(
    block: np.ndarray,
    rng: np.random.Generator,
    config: AugmentConfig,
    channels: ChannelSet,
) -> np.ndarray:
    """Shift, flip, and photometrically jitter a ``(T, C, H, W)`` float block.

    *block* holds only the channels in *channels*, in that order.
    """
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

    if config.brightness > 0 or config.contrast > 0:
        gray_channel = channels.position("gray")
        if gray_channel is not None:
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

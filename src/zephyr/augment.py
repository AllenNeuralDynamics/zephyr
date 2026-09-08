from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .channels import CHANNEL_NAMES

GRAY = CHANNEL_NAMES.index("gray")
DIFF = CHANNEL_NAMES.index("diff")
FLOW_X = CHANNEL_NAMES.index("flow_x")
FLOW_Y = CHANNEL_NAMES.index("flow_y")
MOTION = (DIFF, FLOW_X, FLOW_Y)


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
        sample pairs a slow rhythm with fast-rhythm motion magnitudes.
    shift_px:
        Maximum spatial translation in pixels, drawn uniformly per axis.  Guards
        against the model keying on the crop's absolute position.
    brightness / contrast:
        Applied to the ``gray`` channel only, as fractional deviations from 1.0.
        The motion channels are differences and are already invariant to a
        constant offset.
    noise:
        Standard deviation of additive Gaussian noise, in uint8 code units,
        applied to every channel before standardisation.
    flip:
        Horizontal mirror probability.  Mirroring must negate ``flow_x``,
        otherwise the flow field contradicts the image.  Off by default when a
        camera only ever views one side of the subject.
    """

    time_stretch: float = 1.0
    rate_range: tuple[float, float] | None = None
    scale_motion: bool = True
    shift_px: int = 0
    brightness: float = 0.0
    contrast: float = 0.0
    noise: float = 0.0
    flip: float = 0.0

    @property
    def enabled(self) -> bool:
        return (
            self.time_stretch > 1.0
            or self.shift_px > 0
            or self.brightness > 0
            or self.contrast > 0
            or self.noise > 0
            or self.flip > 0
        )

    @property
    def max_source_frames(self) -> float:
        """Multiplier on the window length that a stretched draw may need."""
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


def scale_motion_channels(block: np.ndarray, stretch: float) -> np.ndarray:
    """Scale the motion channels about their 128 centre by *stretch*."""
    if stretch == 1.0:
        return block
    for channel in MOTION:
        block[:, channel] = np.clip(
            (block[:, channel] - 128.0) * stretch + 128.0, 0.0, 255.0
        )
    return block


def resample_time(x: np.ndarray, n_out: int) -> np.ndarray:
    """Linearly resample *x* along axis 0 to *n_out* samples."""
    n_in = x.shape[0]
    if n_in == n_out:
        return x.astype(np.float32, copy=False)
    positions = np.linspace(0.0, n_in - 1, n_out)
    lo = np.floor(positions).astype(np.intp)
    hi = np.minimum(lo + 1, n_in - 1)
    weight = (positions - lo).astype(np.float32)
    # Broadcast the weight over whatever trailing axes the array has.
    weight = weight.reshape(-1, *([1] * (x.ndim - 1)))
    left = x[lo].astype(np.float32)
    right = x[hi].astype(np.float32)
    return left * (1.0 - weight) + right * weight


def apply_spatial(
    block: np.ndarray, rng: np.random.Generator, config: AugmentConfig
) -> np.ndarray:
    """Shift, flip, and photometrically jitter a ``(T, C, H, W)`` float block."""
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
        block[:, FLOW_X] = 255.0 - block[:, FLOW_X]

    if config.brightness > 0 or config.contrast > 0:
        gain = 1.0 + rng.uniform(-config.contrast, config.contrast)
        offset = 255.0 * rng.uniform(-config.brightness, config.brightness)
        gray = block[:, GRAY]
        block[:, GRAY] = np.clip((gray - 128.0) * gain + 128.0 + offset, 0.0, 255.0)

    if config.noise > 0:
        block = block + rng.normal(0.0, config.noise, block.shape).astype(np.float32)

    return block

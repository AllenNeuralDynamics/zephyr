"""CNN frame encoder + dilated TCN sequence decoder.

Shape of the problem
--------------------
The input is a per-frame image crop; the output is one target sample per frame.
The two halves do different jobs:

``FrameEncoder``
    Runs on every frame independently and collapses the crop to a short vector.
    It is temporally blind by construction -- which is why motion channels are
    precomputed rather than left for the network to discover.
``TemporalNet``
    Sees only that sequence of vectors and reconstructs the trace.  Dilated
    convolutions give it a wide receptive field for a fixed number of conv
    layers, where a stack of plain k=3 convolutions would need many more layers
    to reach the same span.

Two grids, joined in embedding space
------------------------------------
The encoder runs on the *selection* grid (whichever source frames
:mod:`.preprocess` picked, at whatever rate); the TCN runs on the fixed 60 Hz
*output* grid. :func:`resample_embeddings` interpolates between them using the
real frame timestamps.

Non-causal on purpose
---------------------
Inference runs offline on complete clips, so there is no reason to hide the
future: padding is symmetric and each output sample sees context on both sides.
We may want to revisit this choice if we ever need causal inference or tight
online closed-loop operation.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .channels import ALL_CHANNELS, ChannelSet


class FrameEncoder(nn.Module):
    """Encode one multi-channel crop to an *embed*-dimensional vector.

    Downsamples by 2 at every stage.  The final pooling keeps a small spatial
    grid rather than collapsing to a single vector: a global average would let
    opposite-signed motion in different parts of the crop cancel out, which
    would destroy directional information the loss needs.
    """

    def __init__(
        self,
        in_channels: int,
        widths: tuple[int, ...] = (32, 64, 96, 128),
        embed: int = 128,
        pool: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for i, width in enumerate(widths):
            kernel = 5 if i == 0 else 3
            layers += [
                nn.Conv2d(
                    prev, width, kernel, stride=2, padding=kernel // 2, bias=False
                ),
                nn.BatchNorm2d(width),
                nn.GELU(),
            ]
            prev = width
        self.trunk = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(pool)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(prev * pool * pool, embed),
            nn.GELU(),
        )
        self.embed = embed

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) -> (N, embed)."""
        return self.head(self.pool(self.trunk(frames)))


def resample_embeddings(
    embeddings: torch.Tensor, t_in: torch.Tensor, t_out: torch.Tensor
) -> torch.Tensor:
    """``(B, T_in, E)`` sampled at *t_in* -> ``(B, T_out, E)`` at *t_out*.

    Linear interpolation; targets outside the input span clamp rather than
    extrapolate. Both time arrays must be relative to the window (not absolute
    clip time), for float32 precision.
    """
    n_in = embeddings.shape[1]
    if n_in < 2:
        raise ValueError("resampling needs at least two input samples")

    right = torch.searchsorted(t_in.contiguous(), t_out.contiguous()).clamp(1, n_in - 1)
    left = right - 1
    t0 = t_in.gather(1, left)
    t1 = t_in.gather(1, right)
    # clamp_min guards a repeated timestamp; clamp(0, 1) turns what would be
    # extrapolation at the edges into a hold.
    weight = ((t_out - t0) / (t1 - t0).clamp_min(1e-9)).clamp(0.0, 1.0).unsqueeze(-1)

    embed = embeddings.shape[-1]
    low = embeddings.gather(1, left.unsqueeze(-1).expand(-1, -1, embed))
    high = embeddings.gather(1, right.unsqueeze(-1).expand(-1, -1, embed))
    return low + (high - low) * weight


class ResidualBlock(nn.Module):
    """Two dilated non-causal convolutions with a residual connection."""

    def __init__(
        self, channels: int, dilation: int, kernel: int = 3, dropout: float = 0.1
    ) -> None:
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.body = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel,
                dilation=dilation,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel,
                dilation=dilation,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class TemporalNet(nn.Module):
    """Dilated residual stack over the frame-embedding sequence."""

    def __init__(
        self,
        embed: int,
        channels: int = 128,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        kernel: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.project = nn.Conv1d(embed, channels, 1)
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels, d, kernel, dropout) for d in dilations]
        )
        self.signal_head = nn.Conv1d(channels, 1, 1)
        self.onset_head = nn.Conv1d(channels, 1, 1)
        self.kernel = kernel
        self.dilations = dilations

    @property
    def receptive_field(self) -> int:
        """Output-frame span each prediction depends on."""
        return 1 + 2 * (self.kernel - 1) * sum(self.dilations)

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, embed) -> signal (B, T) and onset logits (B, T)."""
        x = self.blocks(self.project(embeddings.transpose(1, 2)))
        return self.signal_head(x).squeeze(1), self.onset_head(x).squeeze(1)


class BreathingNet(nn.Module):
    """The full model: per-frame CNN, then TCN over the frame sequence.

    ``channels`` names the stored channels this model consumes, in order, and
    sets the encoder's input width.  :func:`~.infer.predict_clip` reads it to
    slice the stored array and its per-channel statistics; :mod:`.train`
    records it in the checkpoint.
    """

    def __init__(
        self,
        channels: ChannelSet = ALL_CHANNELS,
        widths: tuple[int, ...] = (32, 64, 96, 128),
        embed: int = 128,
        tcn_channels: int = 128,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.1,
        encoder_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.encoder = FrameEncoder(
            len(channels), widths, embed=embed, dropout=encoder_dropout
        )
        self.temporal = TemporalNet(
            embed, channels=tcn_channels, dilations=dilations, dropout=dropout
        )

    @property
    def receptive_field(self) -> int:
        """Output-grid samples per prediction."""
        return self.temporal.receptive_field

    def encode(self, features: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
        """(B, T, C, H, W) -> (B, T, embed).

        *chunk* bounds how many frames go through the CNN at once.  Under
        ``no_grad`` this caps activation memory without changing the result,
        which is what makes whole-clip inference fit in 8 GB.  It is not useful
        during training, where activations must be retained anyway.
        """
        b, t = features.shape[:2]
        flat = features.reshape(b * t, *features.shape[2:])
        if chunk is None or chunk >= flat.shape[0]:
            embeddings = self.encoder(flat)
        else:
            embeddings = torch.cat(
                [
                    self.encoder(flat[i : i + chunk])
                    for i in range(0, flat.shape[0], chunk)
                ]
            )
        return embeddings.view(b, t, -1)

    def forward(
        self,
        features: torch.Tensor,
        t_in: torch.Tensor,
        t_out: torch.Tensor,
        chunk: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, T_in, C, H, W)`` -> signal and onset logits, both ``(B, T_out)``.

        *t_in*/*t_out* per :func:`resample_embeddings`.
        """
        embeddings = self.encode(features, chunk=chunk)
        return self.temporal(resample_embeddings(embeddings, t_in, t_out))


def pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """``1 - r`` per sequence, averaged over the batch.

    Correlation rather than MSE because the evaluation metric is amplitude-
    invariant and the target's own amplitude is arbitrary, so MSE would spend
    capacity fitting a scale that is not scored.
    """
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    numerator = (pred * target).sum(dim=-1)
    denominator = pred.norm(dim=-1) * target.norm(dim=-1) + eps
    return 1.0 - (numerator / denominator).mean()


def multiscale_pearson_loss(
    pred: torch.Tensor, target: torch.Tensor, scales: tuple[int, ...] = (1, 4, 16)
) -> torch.Tensor:
    """``pearson_loss`` averaged over average-pooled copies of the sequence.

    A single whole-window correlation is dominated by whichever component holds
    the most variance in that window, which leaves slower structure free to be
    wrong without moving the loss.  Pooling by a larger factor averages the
    fast component away and leaves the slow envelope as the dominant signal at
    that scale, so an error there is no longer invisible.  Scale 1 is the
    original per-frame signal; each additional scale also folds in an implicit
    amplitude/shape constraint.
    """
    total = pred.new_zeros(())
    for scale in scales:
        if scale > 1:
            p = F.avg_pool1d(pred.unsqueeze(1), scale).squeeze(1)
            t = F.avg_pool1d(target.unsqueeze(1), scale).squeeze(1)
        else:
            p, t = pred, target
        total = total + pearson_loss(p, t)
    return total / len(scales)


class BreathingLoss(nn.Module):
    """Multi-scale correlation loss on the trace, plus one auxiliary term.

    ``w_corr`` / ``scales``
        See :func:`multiscale_pearson_loss`.
    ``w_onset``
        Binary cross-entropy against a soft event-onset heatmap.  A head that
        has to localise onsets sharpens the trunk's temporal resolution beyond
        what a smooth correlation target demands.  ``pos_weight`` is set per
        batch from the target's own positive ratio rather than a fixed
        constant, since that ratio varies with the signal's own rate.
    """

    def __init__(
        self,
        w_corr: float = 1.0,
        w_onset: float = 0.5,
        scales: tuple[int, ...] = (1, 4, 16),
    ) -> None:
        super().__init__()
        self.w_corr = w_corr
        self.w_onset = w_onset
        self.scales = scales

    def forward(
        self,
        signal_pred: torch.Tensor,
        onset_logits: torch.Tensor,
        signal_true: torch.Tensor,
        onset_true: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        corr = multiscale_pearson_loss(signal_pred, signal_true, self.scales)
        pos_frac = onset_true.mean().clamp_min(1e-3)
        pos_weight = (1.0 - pos_frac) / pos_frac
        onset = F.binary_cross_entropy_with_logits(
            onset_logits, onset_true, pos_weight=pos_weight.detach()
        )
        total = self.w_corr * corr + self.w_onset * onset
        return total, {
            "loss": float(total.detach()),
            "corr": float(1.0 - corr.detach()),
            "onset": float(onset.detach()),
        }

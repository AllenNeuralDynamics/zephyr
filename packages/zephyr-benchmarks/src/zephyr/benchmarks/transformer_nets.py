"""PhysFormer and RhythmFormer, reimplemented in PyTorch from the authors' code (both MIT).

PhysFormer: Yu et al., CVPR 2022 (github.com/ZitongYu/PhysFormer). RhythmFormer: Zou, Guo
et al., 2024 (github.com/zizheng-guo/RhythmFormer). Layers and settings are the published
ones (PhysFormer: dim 96, 12 layers, theta 0.7, sharpness 2; RhythmFormer: 3 stages of 2
blocks, t_patch 2/4/8, top-k 40). Deviations forced by the data: one gray channel, the 96 px
crop upsampled to the 128 px both were built for (so their token grids are the published
ones). Each trains on its paper's loss (:mod:`.transformer_losses`).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from zephyr.model import resample_embeddings

SIZE = 128
"""Input size both networks were built for: the 96 px crop is upsampled to it."""

CONTEXT = 160
"""Frames trimmed in total from a window's two edges when windows are stitched
(:func:`zephyr.infer.predict_clip`), declared as ``receptive_field``. Attention spans a
whole window, so this is not a receptive field: set ``infer_window`` to the training
``window`` (e.g. 320, half of it kept per window)."""


class CDC_T(nn.Module):
    """Temporal centre-difference 3-D convolution (PhysFormer): a 3x3x3 convolution minus
    *theta* times the response to its temporal-edge weights summed, a learned difference."""

    def __init__(self, cin: int, cout: int, theta: float) -> None:
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, 3, stride=1, padding=1, bias=False)
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if math.fabs(self.theta) < 1e-8:
            return out
        w = self.conv.weight
        diff = (w[:, :, 0].sum((2, 3)) + w[:, :, 2].sum((2, 3)))[:, :, None, None, None]
        return out - self.theta * F.conv3d(x, diff, stride=self.conv.stride, padding=0)


# --- PhysFormer -------------------------------------------------------------------------


class _PFAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, theta: float) -> None:
        super().__init__()
        self.proj_q = nn.Sequential(CDC_T(dim, dim, theta), nn.BatchNorm3d(dim))
        self.proj_k = nn.Sequential(CDC_T(dim, dim, theta), nn.BatchNorm3d(dim))
        self.proj_v = nn.Conv3d(dim, dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.heads = heads

    def forward(self, x: torch.Tensor, grid: tuple, sharpness: float) -> torch.Tensor:
        b, p, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, *grid)
        q, k, v = (
            f(x)
            .flatten(2)
            .transpose(1, 2)
            .reshape(b, p, self.heads, -1)
            .transpose(1, 2)
            for f in (self.proj_q, self.proj_k, self.proj_v)
        )
        scores = self.drop(F.softmax(q @ k.transpose(-2, -1) / sharpness, dim=-1))
        return (scores @ v).transpose(1, 2).reshape(b, p, c)


class _PFFeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
            nn.Conv3d(ff_dim, ff_dim, 3, padding=1, groups=ff_dim, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
            nn.Conv3d(ff_dim, dim, 1, bias=False),
            nn.BatchNorm3d(dim),
        )

    def forward(self, x: torch.Tensor, grid: tuple) -> torch.Tensor:
        b, _, c = x.shape
        return (
            self.net(x.transpose(1, 2).reshape(b, c, *grid)).flatten(2).transpose(1, 2)
        )


class _PFBlock(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout, theta) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _PFAttention(dim, heads, dropout, theta)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = _PFFeedForward(dim, ff_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, grid, sharpness):
        x = x + self.drop(self.proj(self.attn(self.norm1(x), grid, sharpness)))
        return x + self.drop(self.pwff(self.norm2(x), grid))


def _stem(cin: int, cout: int, kernel, padding) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, kernel, stride=1, padding=padding),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
    )


def _upsample(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=(2, 1, 1)),
        nn.Conv3d(cin, cout, (3, 1, 1), padding=(1, 0, 0)),
        nn.BatchNorm3d(cout),
        nn.ELU(),
    )


class PhysFormer(nn.Module):
    """3-D CNN stem, 4x4x4 tube tokens, 12 temporal-difference transformer blocks, two
    temporal upsamplings: ``(B, T, 128, 128)`` -> ``(B, T)``, T a multiple of 4. The
    published model (``ViT_ST_ST_Compact3_TDC_gra_sharp``)."""

    multiple = 4

    def __init__(
        self,
        dim: int = 96,
        ff_dim: int = 144,
        heads: int = 4,
        layers: int = 12,
        dropout: float = 0.1,
        theta: float = 0.7,
        sharpness: float = 2.0,
        patch: int = 4,
    ) -> None:
        super().__init__()
        self.sharpness = sharpness
        self.stem = nn.Sequential(
            _stem(1, dim // 4, (1, 5, 5), (0, 2, 2)),
            _stem(dim // 4, dim // 2, 3, 1),
            _stem(dim // 2, dim, 3, 1),
        )
        self.patch_embedding = nn.Conv3d(dim, dim, patch, stride=patch)
        self.blocks = nn.ModuleList(
            _PFBlock(dim, heads, ff_dim, dropout, theta) for _ in range(layers)
        )
        self.upsample = nn.Sequential(_upsample(dim, dim), _upsample(dim, dim // 2))
        self.head = nn.Conv1d(dim // 2, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: torch.Tensor):
        x = self.patch_embedding(self.stem(x.unsqueeze(1)))  # (B, dim, T/4, 4, 4)
        b, c, *grid = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens, tuple(grid), self.sharpness)
        x = self.upsample(tokens.transpose(1, 2).reshape(b, c, *grid)).mean((3, 4))
        return self.head(x).squeeze(1)


# --- RhythmFormer -----------------------------------------------------------------------


class _FusionStem(nn.Module):
    """Each frame and its four neighbouring frame differences, fused: ``(N, D, C, H, W)``
    -> ``(N*D, 64, H/4, W/4)``."""

    def __init__(
        self, in_channels: int = 1, alpha: float = 0.5, beta: float = 0.5
    ) -> None:
        super().__init__()

        def down(cin):
            return nn.Sequential(
                nn.Conv2d(cin, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
            )

        def same():
            return nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

        self.stem11, self.stem12 = down(in_channels), down(4 * in_channels)
        self.stem21, self.stem22 = same(), same()
        self.alpha, self.beta = alpha, beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, d, c, h, w = x.shape
        x1 = torch.cat([x[:, :1], x[:, :1], x[:, : d - 2]], 1)
        x2 = torch.cat([x[:, :1], x[:, : d - 1]], 1)
        x4 = torch.cat([x[:, 1:], x[:, d - 1 :]], 1)
        x5 = torch.cat([x[:, 2:], x[:, d - 1 :], x[:, d - 1 :]], 1)
        diff = torch.cat([x2 - x1, x - x2, x4 - x, x5 - x4], 2).reshape(
            n * d, 4 * c, h, w
        )
        diff = self.stem12(diff)
        frame = self.stem11(x.reshape(n * d, c, h, w))
        path1 = self.stem21(self.alpha * frame + self.beta * diff)
        return self.alpha * path1 + self.beta * self.stem22(diff)


def _grid2seq(x: torch.Tensor, region: tuple, heads: int):
    b, c, t, h, w = x.shape
    rt, rh, rw = t // region[0], h // region[1], w // region[2]
    x = x.view(b, heads, c // heads, rt, region[0], rh, region[1], rw, region[2])
    x = torch.einsum("bmdtohpwq->bmthwopqd", x).flatten(2, 4).flatten(-4, -2)
    return x, (rt, rh, rw)


def _seq2grid(x: torch.Tensor, regions: tuple, region: tuple) -> torch.Tensor:
    b, heads, _, _, head_dim = x.shape
    x = x.view(b, heads, *regions, *region, head_dim)
    x = torch.einsum("bmthwopqd->bmdtohpwq", x)
    return x.reshape(b, heads * head_dim, *(n * s for n, s in zip(regions, region)))


def regional_routing_attention(q, k, v, scale: float, graph, region: tuple):
    """BiFormer's attention over regions: each query region attends to the tokens of the
    top-k key regions *graph* routes it to. q, k, v ``(B, C, T, H, W)``."""
    heads, topk = graph.shape[1], graph.shape[3]
    q, regions = _grid2seq(q, region, heads)
    k, _ = _grid2seq(k, region, heads)
    v, _ = _grid2seq(v, region, heads)
    b, _, nregion, size, head_dim = k.shape
    index = graph.view(b, heads, q.shape[2], topk, 1, 1).expand(
        -1, -1, -1, -1, size, head_dim
    )

    def gather(t):
        t = t.view(b, heads, 1, nregion, size, head_dim).expand(
            -1, -1, q.shape[2], -1, -1, -1
        )
        return torch.gather(t, 3, index).flatten(-3, -2)

    attn = torch.softmax((q * scale) @ gather(k).transpose(-1, -2), dim=-1)
    return _seq2grid(attn @ gather(v), regions, region)


class _BRA(nn.Module):
    """Video bi-level routing attention: regions ranked by pooled q-k affinity, then token
    attention within the top-k, plus a depthwise-convolution term (LePE)."""

    def __init__(self, dim, heads, t_patch, topk, side_dwconv) -> None:
        super().__init__()
        self.heads, self.t_patch, self.topk = heads, t_patch, topk
        self.scale = dim**-0.5
        self.lepe = nn.Conv3d(
            dim, dim, side_dwconv, padding=side_dwconv // 2, groups=dim
        )
        self.output_linear = nn.Conv3d(dim, dim, 1)
        self.proj_q = nn.Sequential(CDC_T(dim, dim, 0.2), nn.BatchNorm3d(dim))
        self.proj_k = nn.Sequential(CDC_T(dim, dim, 0.2), nn.BatchNorm3d(dim))
        self.proj_v = nn.Conv3d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, h, w = x.shape
        region = (max(4 // self.t_patch, 1), h // 4, w // 4)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q_r = F.avg_pool3d(q.detach(), region, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool3d(k.detach(), region, ceil_mode=True, count_include_pad=False)
        affinity = q_r.permute(0, 2, 3, 4, 1).flatten(1, 3) @ k_r.flatten(2, 4)
        _, graph = torch.topk(affinity, k=min(self.topk, affinity.shape[-1]), dim=-1)
        graph = graph.unsqueeze(1).expand(-1, self.heads, -1, -1)
        out = regional_routing_attention(q, k, v, self.scale, graph, region)
        return self.output_linear(out + self.lepe(v))


class _BiFormerBlock(nn.Module):
    def __init__(self, dim, heads, t_patch, topk, mlp_ratio, side_dwconv) -> None:
        super().__init__()
        hidden = int(mlp_ratio * dim)
        self.norm1 = nn.BatchNorm3d(dim)
        self.attn = _BRA(dim, heads, t_patch, topk, side_dwconv)
        self.norm2 = nn.BatchNorm3d(dim)
        self.mlp = nn.Sequential(
            nn.Conv3d(dim, hidden, 1),
            nn.BatchNorm3d(hidden),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.BatchNorm3d(hidden),
            nn.GELU(),
            nn.Conv3d(hidden, dim, 1),
            nn.BatchNorm3d(dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _TPTBlock(nn.Module):
    """Temporal periodic transformer stage: time halved log2(t_patch) times, BiFormer
    blocks, time doubled back."""

    def __init__(
        self, dim, depth, heads, t_patch, topk, mlp_ratio, side_dwconv
    ) -> None:
        super().__init__()
        n = int(math.log2(t_patch))
        self.down = nn.Sequential(
            *(
                nn.Sequential(
                    nn.BatchNorm3d(dim),
                    nn.Conv3d(dim, dim, (2, 1, 1), stride=(2, 1, 1)),
                )
                for _ in range(n)
            )
        )
        self.blocks = nn.Sequential(
            *(
                _BiFormerBlock(dim, heads, t_patch, topk, mlp_ratio, side_dwconv)
                for _ in range(depth)
            )
        )
        self.up = nn.Sequential(*(_upsample(dim, dim) for _ in range(n)))

    def forward(self, x):
        return self.up(self.blocks(self.down(x)))


class RhythmFormer(nn.Module):
    """Fusion stem (frames and their differences), 4x4 patches, three temporal periodic
    transformer stages: ``(B, T, 128, 128)`` -> ``(B, T)``, T a multiple of 8. The
    published model, without its unused ``qkv_linear``."""

    multiple = 8

    def __init__(
        self,
        dim: int = 64,
        head_dim: int = 16,
        depth: tuple = (2, 2, 2),
        t_patches: tuple = (2, 4, 8),
        topk: int = 40,
        mlp_ratio: float = 1.5,
        side_dwconv: int = 3,
    ) -> None:
        super().__init__()
        self.stem = _FusionStem()
        self.patch_embedding = nn.Conv3d(64, dim, (1, 4, 4), stride=(1, 4, 4))
        self.stages = nn.Sequential(
            *(
                _TPTBlock(dim, d, dim // head_dim, t, topk, mlp_ratio, side_dwconv)
                for d, t in zip(depth, t_patches)
            )
        )
        self.head = nn.Conv1d(dim, 1, 1)

    def forward(self, x: torch.Tensor):
        n, d, h, w = x.shape
        x = (
            self.stem(x.unsqueeze(2))
            .view(n, d, 64, h // 4, w // 4)
            .permute(0, 2, 1, 3, 4)
        )
        x = self.stages(self.patch_embedding(x)).mean((3, 4))  # (N, dim, D)
        return self.head(x).squeeze(1)


# --- the wrapper zephyr trains ------------------------------------------------------------


class GrayVideoTransformer(nn.Module):
    """A gray-only video network behind :class:`~zephyr.model.BreathingNet`'s interface: the
    crop upsampled to SIZE, time padded to the network's multiple, one output per frame
    resampled to the output grid. No onset head: its onset logits are zeros."""

    def __init__(self, arch: str, core: nn.Module) -> None:
        super().__init__()
        self.arch, self.net = arch, core

    @property
    def receptive_field(self) -> int:
        return CONTEXT

    def forward(self, features, t_in, t_out, chunk=None):
        b, t, _, h, w = features.shape
        x = F.interpolate(
            features[:, :, 0].reshape(b * t, 1, h, w).float(),
            size=(SIZE, SIZE),
            mode="bilinear",
            align_corners=False,
        ).view(b, t, SIZE, SIZE)
        pad = (-t) % self.net.multiple
        if pad:
            x = torch.cat([x, x[:, -1:].expand(-1, pad, -1, -1)], dim=1)
        out = self.net(x)[:, :t]
        signal = resample_embeddings(out.unsqueeze(-1), t_in, t_out).squeeze(-1)
        return signal, torch.zeros_like(signal)


CORES = {
    "physformer": lambda: PhysFormer(),
    "rhythmformer": lambda: RhythmFormer(),
}
"""Each transformer arch and the network it builds."""


def build(arch: str) -> GrayVideoTransformer:
    return GrayVideoTransformer(arch, CORES[arch]())

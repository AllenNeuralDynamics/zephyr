"""The network drawn as an SVG block diagram, read off a loaded model.

Layer sizes, kernels and dilations come from the model itself, so the diagram cannot
drift from the code: blocks are drawn as boxes whose face scales with the feature map's
side and whose depth with its channels (the convention of NN-SVG and PlotNeuralNet).
Coordinates are millimetres; text is Arial at the figures' sizes.
"""

import base64
import io
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from torch import nn

FONT = 2.3  # mm, about 6.5 pt
SMALL = 1.9  # mm, about 5.4 pt
INK, MUTED = "#222222", "#666666"
CONV = ("#d6e6f4", "#a9c8e4", "#7fa9d1")  # front, top, side
DENSE = ("#e8e8e8", "#cfcfcf", "#b5b5b5")
TCN = "#0072b2"


@dataclass(frozen=True)
class Layer:
    """One encoder stage: its output side, channels, and how it was made."""

    side: int
    channels: int
    label: str


def encoder_layers(net: nn.Module, side: int) -> list[Layer]:
    """The frame encoder's stages, from a :class:`zephyr.model.BreathingNet`."""
    encoder = net.encoder
    layers = [Layer(side, encoder_input_channels(net), "Input")]
    convs = []
    if encoder.stem is not None:
        convs.append(encoder.stem[0])
    convs += [m for m in encoder.trunk if isinstance(m, nn.Conv2d)]
    for conv in convs:
        stride = conv.stride[-1]
        side = (side + 2 * conv.padding[-1] - conv.kernel_size[-1]) // stride + 1
        k = "×".join(str(v) for v in conv.kernel_size)
        kind = "Conv3D" if isinstance(conv, nn.Conv3d) else "Conv"
        layers.append(Layer(side, conv.out_channels, f"{kind} {k}, /{stride}"))
    pool = encoder.pool.output_size
    pool = pool if isinstance(pool, int) else pool[0]
    layers.append(Layer(pool, layers[-1].channels, f"Avg pool {pool}×{pool}"))
    return layers


def encoder_input_channels(net: nn.Module) -> int:
    first = (
        net.encoder.stem[0] if net.encoder.stem is not None else net.encoder.trunk[0]
    )
    return first.in_channels


def _png(image: NDArray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _text(x, y, s, size=FONT, color=INK, anchor="middle", weight="normal") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>'
    )


def _arrow(x0, y0, x1, y1) -> str:
    return (
        f'<path d="M{x0:.2f},{y0:.2f} L{x1:.2f},{y1:.2f}" stroke="#333" '
        'stroke-width="0.3" fill="none" marker-end="url(#arrow)"/>'
    )


def _cuboid(x, y, face, depth, colors) -> str:
    """A box whose front face (side *face*) has its lower left corner at (x, y)."""
    dx, dy = depth * 0.7, -depth * 0.5
    front, top, side = colors
    stroke = 'stroke="#4a6f94" stroke-width="0.15"'
    return (
        f'<path d="M{x},{y - face} l{dx},{dy} h{face} l{-dx},{-dy} z" fill="{top}" {stroke}/>'
        f'<path d="M{x + face},{y} l{dx},{dy} v{-face} l{-dx},{-dy} z" fill="{side}" {stroke}/>'
        f'<rect x="{x}" y="{y - face}" width="{face}" height="{face}" fill="{front}" {stroke}/>'
    )


def network_svg(
    net: nn.Module,
    crops: Sequence[NDArray],
    width_mm: float,
    height_mm: float,
    output_hz: float,
) -> str:
    """The whole network, left to right: the input crops, the frame encoder, the
    resampling onto the output grid, the temporal network and its two heads."""
    side = crops[0].shape[-1]
    layers = encoder_layers(net, side)
    embed = net.encoder.embed
    temporal = net.temporal
    dilations = list(temporal.dilations)
    span_s = temporal.receptive_field / output_hz
    base = height_mm * 0.62  # baseline of the blocks
    out = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm}mm" '
            f'height="{height_mm}mm" viewBox="0 0 {width_mm} {height_mm}" '
            'font-family="Arial, Helvetica, sans-serif">'
        ),
        (
            '<defs><marker id="arrow" viewBox="0 0 4 4" refX="3.5" refY="2" '
            'markerWidth="1.5" markerHeight="1.5" markerUnits="userSpaceOnUse" '
            'orient="auto"><path d="M0,0 L4,2 L0,4 z" fill="#333"/></marker></defs>'
        ),
    ]

    # input: consecutive crops, offset like a stack
    img = 13.0
    x = 2.0
    for i, crop in enumerate(crops):
        ox, oy = x + 1.6 * i, base - img - 1.6 * (len(crops) - 1 - i)
        out.append(
            f'<image x="{ox:.2f}" y="{oy:.2f}" width="{img}" height="{img}" '
            f'href="{_png(crop)}" preserveAspectRatio="none"/>'
            f'<rect x="{ox:.2f}" y="{oy:.2f}" width="{img}" height="{img}" '
            'fill="none" stroke="#333" stroke-width="0.15"/>'
        )
    stack_w = img + 1.6 * (len(crops) - 1)
    cx = x + stack_w / 2
    out.append(_text(cx, base + 3.4, "Snout crops"))
    out.append(_text(cx, base + 5.8, f"{side}×{side}, gray", SMALL, MUTED))
    out.append(_text(cx, base + 8.0, "frames t−1, t, t+1", SMALL, MUTED))

    # frame encoder
    x = x + stack_w + 4.0
    enc_x0 = x - 1.5
    out.append(_arrow(x - 3.5, base - img / 2, x - 0.6, base - img / 2))
    for layer in layers[1:]:
        face = 2.2 + 10.0 * (layer.side / side) ** 0.6
        depth = 0.8 + 5.0 * layer.channels / max(lay.channels for lay in layers)
        out.append(_cuboid(x, base, face, depth, CONV))
        w = face + depth * 0.7
        mid = x + w / 2
        out.append(_text(mid, base + 3.4, f"{layer.side}²×{layer.channels}", SMALL))
        # two lines: what the layer is, then its kernel and stride
        split = (
            layer.label.rpartition if "pool" in layer.label else layer.label.partition
        )
        kind, _, detail = split(" ")
        out.append(_text(mid, base + 5.8, kind, SMALL, MUTED))
        out.append(_text(mid, base + 7.9, detail, SMALL, MUTED))
        x += w + 4.2
    # dense embedding as a thin bar
    bar_h = 12.0
    out.append(
        f'<rect x="{x:.2f}" y="{base - bar_h:.2f}" width="1.4" height="{bar_h}" '
        f'fill="{DENSE[0]}" stroke="#555" stroke-width="0.15"/>'
    )
    out.append(_text(x + 0.7, base + 3.4, f"{embed}", SMALL))
    out.append(_text(x + 0.7, base + 5.8, "Linear", SMALL, MUTED))
    enc_x1 = x + 3.5
    out.append(
        f'<rect x="{enc_x0:.2f}" y="2.5" width="{enc_x1 - enc_x0:.2f}" '
        f'height="{base + 6.5:.2f}" fill="none" stroke="#999" stroke-width="0.2" '
        'stroke-dasharray="0.8,0.6" rx="1"/>'
    )
    out.append(_text(enc_x0 + 1.2, 5.6, "Frame encoder", FONT, INK, "start", "bold"))
    out.append(
        _text(enc_x0 + 1.2, 8.2, "each frame, shared weights", SMALL, MUTED, "start")
    )

    # resampling: embeddings at frame times onto the regular output grid
    x = enc_x1 + 3.0
    out.append(_arrow(x - 2.5, base - bar_h / 2, x + 0.5, base - bar_h / 2))
    rs_w = 24.0
    rng = np.random.default_rng(3)
    frames = np.cumsum(rng.uniform(0.75, 1.25, 9))
    frames = x + 1.5 + (frames - frames[0]) / (frames[-1] - frames[0]) * (rs_w - 3)
    grid = np.linspace(x + 1.5, x + rs_w - 1.5, 9)
    y_in, y_out = base - 10.5, base - 2.0
    out.append(_text(x + rs_w / 2, 5.6, "Resample", FONT, INK, "middle", "bold"))
    out.append(_text(x + rs_w / 2, 8.2, "by frame timestamps", SMALL, MUTED))
    for f in frames:
        out.append(
            f'<circle cx="{f:.2f}" cy="{y_in}" r="0.55" fill="{DENSE[2]}" '
            'stroke="#555" stroke-width="0.1"/>'
        )
    for g in grid:
        nearest = frames[np.argmin(np.abs(frames - g))]
        out.append(
            f'<path d="M{nearest:.2f},{y_in + 0.6} L{g:.2f},{y_out - 0.6}" '
            'stroke="#aaa" stroke-width="0.12"/>'
            f'<circle cx="{g:.2f}" cy="{y_out}" r="0.55" fill="{CONV[1]}" '
            'stroke="#4a6f94" stroke-width="0.1"/>'
        )
    out.append(_text(x + rs_w / 2, base + 3.4, f"{output_hz:.0f} Hz × {embed}", SMALL))
    out.append(_text(x + rs_w / 2, base + 5.8, "linear interpolation", SMALL, MUTED))

    # temporal network: dilated convolutions, drawn as in WaveNet
    x = x + rs_w + 4.0
    out.append(_arrow(x - 3.5, base - 6.0, x - 0.6, base - 6.0))
    shown = dilations[:4]
    levels = len(shown) + 1
    tcn_w = 33.0
    n = 1 + 2 * sum(shown)
    xs = np.linspace(x + 1.0, x + tcn_w - 1.0, n)
    ys = np.linspace(base - 1.0, 11.5, levels)
    out.append(
        _text(x + tcn_w / 2, 5.6, "Temporal network", FONT, INK, "middle", "bold")
    )
    out.append(
        _text(
            x + tcn_w / 2,
            8.2,
            f"{len(dilations)} residual blocks, {temporal.project.out_channels} channels",
            SMALL,
            MUTED,
        )
    )
    # nodes the top centre output depends on, from the top level down
    used = [{n // 2}]
    for d in reversed(shown):
        used.append({j for i in used[-1] for j in (i - d, i, i + d) if 0 <= j < n})
    used = used[::-1]  # used[level], level 0 = the input sequence
    for level, d in enumerate(shown, start=1):
        for i in used[level]:
            for j in (i - d, i, i + d):
                out.append(
                    f'<path d="M{xs[j]:.2f},{ys[level - 1]:.2f} '
                    f'L{xs[i]:.2f},{ys[level]:.2f}" stroke="{TCN}" '
                    'stroke-width="0.15" stroke-opacity="0.6"/>'
                )
    for level in range(levels):
        for i in range(n):
            fill = TCN if i in used[level] else "#ffffff"
            out.append(
                f'<circle cx="{xs[i]:.2f}" cy="{ys[level]:.2f}" r="0.42" '
                f'fill="{fill}" stroke="{TCN}" stroke-width="0.1"/>'
            )
        if level:
            out.append(
                _text(
                    x + tcn_w + 0.8,
                    ys[level] + 0.6,
                    f"d={shown[level - 1]}",
                    SMALL,
                    MUTED,
                    "start",
                )
            )
    more = ", ".join(str(d) for d in dilations[len(shown) :])
    out.append(
        _text(
            x + tcn_w / 2,
            base + 3.4,
            f"dilations {', '.join(map(str, shown))}, …, {dilations[-1]}",
            SMALL,
        )
        if more
        else _text(
            x + tcn_w / 2, base + 3.4, f"dilations {', '.join(map(str, shown))}", SMALL
        )
    )
    out.append(
        _text(
            x + tcn_w / 2,
            base + 5.8,
            f"kernel {temporal.kernel}, receptive field {span_s:.1f} s",
            SMALL,
            MUTED,
        )
    )

    # two heads
    x = x + tcn_w + 9.5
    top_y, bottom_y = base - 11.0, base + 1.0
    mid_y = (top_y + bottom_y) / 2
    out.append(_arrow(x - 4.5, mid_y, x - 1.0, top_y + 1.0))
    out.append(_arrow(x - 4.5, mid_y, x - 1.0, bottom_y - 1.0))
    out.append(_text(x, top_y + 1.0, "Trace", FONT, INK, "start", "bold"))
    out.append(_text(x, top_y + 3.6, "1×1 conv", SMALL, MUTED, "start"))
    out.append(_text(x, top_y + 5.8, "Pearson loss", SMALL, MUTED, "start"))
    out.append(_text(x, bottom_y - 1.0, "Onset", FONT, INK, "start", "bold"))
    out.append(_text(x, bottom_y + 1.6, "1×1 conv", SMALL, MUTED, "start"))
    out.append(_text(x, bottom_y + 3.8, "BCE, Gaussian target", SMALL, MUTED, "start"))
    out.append("</svg>")
    return "\n".join(out)

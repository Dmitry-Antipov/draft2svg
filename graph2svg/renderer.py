"""Matplotlib-based renderer for publication-quality graph output."""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch

from .model import Edge, GraphSpec, Node
from .styles import StyleConfig, get_style

# Internal coordinate scale. Node positions (0-1 range) are scaled to this
# range so that node radii and arrow sizes are in comparable units to
# matplotlib's display-point based rendering.
COORD_SCALE = 800


def _label_to_mathtext(label: str | None) -> str:
    """Convert a label with underscore notation to matplotlib mathtext.

    Examples:
        "e_1" -> "$e_1$"
        "x_1" -> "$x_1$"
        "S" -> "$S$"
        "e_10" -> "$e_{10}$"
    """
    if label is None:
        return ""

    # Check if label has underscores (subscript notation)
    m = re.match(r"^([A-Za-z]+)_(\d+)$", label)
    if m:
        base, sub = m.groups()
        if len(sub) > 1:
            return f"${base}_{{{sub}}}$"
        return f"${base}_{sub}$"

    # Single character or word - just italicize
    return f"${label}$"


def _format_weight(weight: str | None, weight_format: str) -> str:
    """Format a weight value according to the graph's weight format."""
    if weight is None:
        return ""
    if weight_format == "multiply":
        return f"{weight}\u00d7"  # multiplication sign ×
    return weight


def _bezier_point(
    src: tuple[float, float],
    tgt: tuple[float, float],
    curvature: float,
    t: float = 0.5,
) -> tuple[float, float]:
    """Compute a point on a quadratic Bezier curve matching matplotlib's arc3."""
    sx, sy = src
    tx, ty = tgt
    mx, my = (sx + tx) / 2, (sy + ty) / 2
    dx, dy = tx - sx, ty - sy
    length = math.hypot(dx, dy)

    if length < 1e-9:
        return mx, my

    px, py = -dy / length, dx / length
    cx = mx + px * curvature * length
    cy = my + py * curvature * length

    x = (1 - t) ** 2 * sx + 2 * (1 - t) * t * cx + t**2 * tx
    y = (1 - t) ** 2 * sy + 2 * (1 - t) * t * cy + t**2 * ty

    return x, y


def _bezier_tangent(
    src: tuple[float, float],
    tgt: tuple[float, float],
    curvature: float,
    t: float = 0.5,
) -> tuple[float, float]:
    """Compute the unit tangent direction at parameter t on the Bezier curve."""
    sx, sy = src
    tx, ty = tgt
    mx, my = (sx + tx) / 2, (sy + ty) / 2
    dx, dy = tx - sx, ty - sy
    length = math.hypot(dx, dy)

    if length < 1e-9:
        return 1.0, 0.0

    px, py = -dy / length, dx / length
    cx = mx + px * curvature * length
    cy = my + py * curvature * length

    tan_x = 2 * (1 - t) * (cx - sx) + 2 * t * (tx - cx)
    tan_y = 2 * (1 - t) * (cy - sy) + 2 * t * (ty - cy)

    tlen = math.hypot(tan_x, tan_y)
    if tlen < 1e-9:
        return 1.0, 0.0

    return tan_x / tlen, tan_y / tlen


def render_graph(
    graph: GraphSpec,
    output_path: str | Path,
    style: StyleConfig | None = None,
    output_format: str | None = None,
) -> Path:
    """Render a graph specification to SVG or PDF."""
    output_path = Path(output_path)
    if style is None:
        style = get_style("default")

    if output_format is None:
        output_format = output_path.suffix.lstrip(".").lower()
        if output_format not in ("svg", "pdf", "png"):
            output_format = "svg"

    # Configure matplotlib for clean text output
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Helvetica", "Arial", "DejaVu Sans", "sans-serif"
    ]

    # Scale node positions from [0,1] to [0, COORD_SCALE] for better
    # compatibility with matplotlib's point-based sizing.
    pos: dict[str, tuple[float, float]] = {}
    for n in graph.nodes:
        pos[n.name] = (n.x * COORD_SCALE, n.y * COORD_SCALE)

    node_radius = style.node_radius * COORD_SCALE

    # Compute plot bounds – include both nodes and Bezier control points so
    # that curved edges don't push outside the axes (which would be clipped
    # or cause bbox_inches="tight" to bloat the vertical extent).
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]

    for edge in graph.edges:
        sp = pos.get(edge.source)
        tp = pos.get(edge.target)
        if not sp or not tp:
            continue
        rad = edge.curvature if abs(edge.curvature) > 0.01 else 0.0
        if abs(rad) < 0.01:
            continue
        # Sample the Bezier curve at several points to find its extent.
        for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            bx, by = _bezier_point(sp, tp, rad, t)
            xs.append(bx)
            ys.append(by)

    margin = node_radius * 4
    plot_x_min = min(xs) - margin
    plot_x_max = max(xs) + margin
    plot_y_min = min(ys) - margin
    plot_y_max = max(ys) + margin

    # Compute aspect ratio from the actual data bounds.  We no longer force a
    # minimum landscape aspect ratio — the output should reflect the graph's
    # natural shape so that square or tall graphs are rendered faithfully.
    data_width = plot_x_max - plot_x_min
    data_height = plot_y_max - plot_y_min
    aspect = data_width / data_height if data_height > 0 else 1.0

    fig_w = style.figure_width
    fig_h = fig_w / max(aspect, 0.3)
    fig_h = max(min(fig_h, 20.0), 4.0)

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_xlim(plot_x_min, plot_x_max)
    ax.set_ylim(plot_y_max, plot_y_min)  # y-down
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if style.background_color and style.background_color != "none":
        fig.patch.set_facecolor(style.background_color)
        ax.set_facecolor(style.background_color)

    # Assign colors to edges
    color_idx = 0
    for edge in graph.edges:
        if edge.color is None:
            edge.color = style.color_palette[color_idx % len(style.color_palette)]
            color_idx += 1

    # Create node patches (used for both drawing and arrow clipping)
    node_patches: dict[str, mpatches.Circle] = {}
    for node in graph.nodes:
        nx, ny = pos[node.name]
        patch = mpatches.Circle(
            (nx, ny),
            radius=node_radius,
            facecolor=style.node_fill_color,
            edgecolor=style.node_border_color,
            linewidth=style.node_border_width,
            zorder=10,
        )
        ax.add_patch(patch)
        node_patches[node.name] = patch

        # Node label
        ax.text(
            nx, ny,
            _label_to_mathtext(node.name),
            ha="center", va="center",
            fontsize=style.node_font_size,
            fontstyle=style.node_font_style,
            fontweight=style.node_font_weight,
            color="#333333",
            zorder=11,
        )

    # Draw edges (using node patches for proper clipping)
    for edge in graph.edges:
        _draw_edge(ax, edge, pos, node_patches, graph.weight_format, style,
                   node_radius)

    # Save
    fig.savefig(
        str(output_path),
        format=output_format,
        bbox_inches="tight",
        dpi=style.figure_dpi,
        transparent=(style.background_color == "none"),
        pad_inches=0.15,
    )
    plt.close(fig)

    return output_path


def _best_label_t(
    src: tuple[float, float],
    tgt: tuple[float, float],
    curvature: float,
    pos: dict[str, tuple[float, float]],
    source_name: str,
    target_name: str,
    node_radius: float,
) -> float:
    """Find the best t-parameter for label placement that avoids nodes.

    Starts at t=0.5 (midpoint) and tries shifting toward source or target
    if the midpoint is too close to any intermediate node.
    """
    avoid_radius = node_radius * 2.0  # clearance zone around each node

    candidates = [0.5, 0.4, 0.6, 0.35, 0.65, 0.3, 0.7, 0.25, 0.75]
    for t in candidates:
        px, py = _bezier_point(src, tgt, curvature, t)
        too_close = False
        for name, (nx, ny) in pos.items():
            if name in (source_name, target_name):
                continue
            dist = math.hypot(px - nx, py - ny)
            if dist < avoid_radius:
                too_close = True
                break
        if not too_close:
            return t

    return 0.5  # fallback


def _draw_edge(
    ax: Axes,
    edge: Edge,
    pos: dict[str, tuple[float, float]],
    node_patches: dict[str, mpatches.Circle],
    weight_format: str,
    style: StyleConfig,
    node_radius: float,
) -> None:
    """Draw a single edge with arrow, label, and weight."""
    src_pos = pos.get(edge.source)
    tgt_pos = pos.get(edge.target)
    if not src_pos or not tgt_pos:
        return

    color = edge.color or "#333333"

    # Arrow style: -|> gives a clean filled triangular arrowhead
    if edge.directed:
        arrowstyle = f"-|>,head_length={style.arrow_head_length},head_width={style.arrow_head_width}"
    else:
        arrowstyle = "-"

    # Connection style
    rad = edge.curvature if abs(edge.curvature) > 0.01 else 0.0
    connectionstyle = f"arc3,rad={rad}"

    # Use patchA/patchB to properly clip arrows at node boundaries
    src_patch = node_patches.get(edge.source)
    tgt_patch = node_patches.get(edge.target)

    arrow = FancyArrowPatch(
        posA=src_pos,
        posB=tgt_pos,
        arrowstyle=arrowstyle,
        connectionstyle=connectionstyle,
        facecolor=color,
        edgecolor=color,
        linewidth=style.edge_width,
        mutation_scale=1,
        patchA=src_patch,
        patchB=tgt_patch,
        zorder=5,
    )
    ax.add_patch(arrow)

    # -- Label placement --
    # Find a t-parameter along the edge that doesn't collide with nodes
    label_t = _best_label_t(
        src_pos, tgt_pos, rad, pos,
        edge.source, edge.target, node_radius,
    )
    mid_x, mid_y = _bezier_point(src_pos, tgt_pos, rad, t=label_t)
    tan_x, tan_y = _bezier_tangent(src_pos, tgt_pos, rad, t=label_t)

    # Compute the tangent angle in degrees (for y-down coords: negate y)
    tangent_angle = math.degrees(math.atan2(-tan_y, tan_x))
    # Normalize rotation so text is always right-side-up (angle in [-90, 90])
    rot = tangent_angle
    if rot > 90:
        rot -= 180
    elif rot < -90:
        rot += 180

    # Normal direction: rotate tangent 90° CCW in y-down coords.
    # In y-down, rotating (tx,ty) 90° CCW gives (ty, -tx).
    # This points to the "left" side of the travel direction (screen-up for
    # rightward travel).
    norm_x, norm_y = tan_y, -tan_x

    # Determine "outer" normal for curved edges.  For rad > 0 the curve bulges
    # in the +normal direction, for rad < 0 it bulges in the −normal direction.
    # We always want the outer_norm to point toward the convex side (outward).
    outer_nx, outer_ny = norm_x, norm_y
    if rad < -0.01:
        outer_nx, outer_ny = -norm_x, -norm_y

    # Offset from edge centreline for the two label slots.
    offset = node_radius * 0.85

    weight_text = _format_weight(edge.weight, weight_format)
    label_text = _label_to_mathtext(edge.label) if edge.label else ""

    if weight_text and label_text:
        # Straddle the edge: weight on the outer (convex) side,
        # label on the inner side.  For straight edges outer == screen-up.
        w_x = mid_x + outer_nx * offset
        w_y = mid_y + outer_ny * offset
        l_x = mid_x - outer_nx * offset
        l_y = mid_y - outer_ny * offset

        ax.text(
            w_x, w_y,
            weight_text,
            ha="center", va="center",
            fontsize=style.weight_font_size,
            color=color,
            fontweight="bold",
            rotation=rot,
            rotation_mode="anchor",
            zorder=8,
        )
        ax.text(
            l_x, l_y,
            label_text,
            ha="center", va="center",
            fontsize=style.edge_label_font_size,
            color=color,
            fontstyle="italic",
            rotation=rot,
            rotation_mode="anchor",
            zorder=8,
        )
    elif weight_text:
        w_x = mid_x + outer_nx * offset
        w_y = mid_y + outer_ny * offset
        ax.text(
            w_x, w_y,
            weight_text,
            ha="center", va="center",
            fontsize=style.weight_font_size,
            color=color,
            fontweight="bold",
            rotation=rot,
            rotation_mode="anchor",
            zorder=8,
        )
    elif label_text:
        l_x = mid_x + outer_nx * offset
        l_y = mid_y + outer_ny * offset
        ax.text(
            l_x, l_y,
            label_text,
            ha="center", va="center",
            fontsize=style.edge_label_font_size,
            color=color,
            fontstyle="italic",
            rotation=rot,
            rotation_mode="anchor",
            zorder=8,
        )

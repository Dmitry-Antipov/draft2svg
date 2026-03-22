"""Layout engine for refining node positions and computing edge routes."""

from __future__ import annotations

import math
from collections import defaultdict

from .model import Edge, GraphSpec, Node


def normalize_positions(graph: GraphSpec, padding: float = 0.1) -> GraphSpec:
    """Normalize node positions to fill [padding, 1-padding] range.

    Ensures the graph uses the available space well while preserving
    both relative positions and the aspect ratio from the LLM extraction.
    The longer axis is scaled to fill the usable range; the shorter axis
    is scaled by the same factor and centred.
    """
    if not graph.nodes:
        return graph

    xs = [n.x for n in graph.nodes]
    ys = [n.y for n in graph.nodes]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_range = x_max - x_min
    y_range = y_max - y_min

    usable = 1.0 - 2 * padding

    if x_range < 1e-12 and y_range < 1e-12:
        # All nodes at the same point — centre them
        for node in graph.nodes:
            node.x = 0.5
            node.y = 0.5
        return graph

    # Use a uniform scale so the aspect ratio is preserved.
    scale = usable / max(x_range, y_range)

    # Centre the shorter axis within the usable range.
    x_offset = padding + (usable - x_range * scale) / 2
    y_offset = padding + (usable - y_range * scale) / 2

    for node in graph.nodes:
        node.x = x_offset + (node.x - x_min) * scale
        node.y = y_offset + (node.y - y_min) * scale

    return graph


def assign_curvatures(graph: GraphSpec) -> GraphSpec:
    """Auto-assign curvatures for edges that share the same node pair.

    Edges between the same pair of nodes get alternating curvatures
    so they don't overlap. Single edges between a pair stay straight
    unless they already have a non-zero curvature assigned.
    """
    # Group edges by their (unordered) node pair
    pair_edges: dict[tuple[str, str], list[Edge]] = defaultdict(list)
    for edge in graph.edges:
        pair = sorted([edge.source, edge.target])
        key = (pair[0], pair[1])
        pair_edges[key].append(edge)

    for pair, edges in pair_edges.items():
        if len(edges) == 1:
            # Single edge: keep its curvature (may be LLM-assigned or 0)
            continue

        # Multiple edges between same nodes: assign curvatures
        n = len(edges)
        for i, edge in enumerate(edges):
            if n == 2:
                # Two edges: curve in opposite directions
                edge.curvature = 0.3 if i == 0 else -0.3
            else:
                # More than 2: spread curvatures
                t = (i - (n - 1) / 2) / max(1, (n - 1) / 2)
                edge.curvature = t * 0.4

    return graph


def _detect_crossings(graph: GraphSpec) -> GraphSpec:
    """Add curvature to straight edges that pass through other nodes.

    If a straight edge from A to B passes close to node C,
    add some curvature to route around it.
    """
    node_positions = {n.name: (n.x, n.y) for n in graph.nodes}

    for edge in graph.edges:
        if abs(edge.curvature) > 0.01:
            # Already curved, skip
            continue

        src = node_positions.get(edge.source)
        tgt = node_positions.get(edge.target)
        if not src or not tgt:
            continue

        # Check if any other node lies close to the line segment
        for node in graph.nodes:
            if node.name in (edge.source, edge.target):
                continue

            dist = _point_to_segment_dist(
                (node.x, node.y), src, tgt
            )
            if dist < 0.06:  # Too close to the edge line
                edge.curvature = 0.25
                break

    return graph


def _point_to_segment_dist(
    point: tuple[float, float],
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
) -> float:
    """Compute minimum distance from a point to a line segment."""
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end

    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq < 1e-12:
        return math.hypot(px - x1, py - y1)

    # Project point onto the line, clamped to [0, 1]
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))

    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return math.hypot(px - proj_x, py - proj_y)


def apply_layout(
    graph: GraphSpec,
    algorithm: str = "original",
    padding: float = 0.1,
) -> GraphSpec:
    """Apply layout refinement to the graph.

    Args:
        graph: The graph specification with LLM-extracted positions.
        algorithm: Layout algorithm to use:
            - "original": Keep LLM positions, just normalize
            - "spring": NetworkX spring layout
            - "kamada_kawai": NetworkX Kamada-Kawai layout
        padding: Padding around the graph (fraction of total space).

    Returns:
        The graph with refined node positions and edge curvatures.
    """
    if algorithm == "original":
        graph = normalize_positions(graph, padding)
    elif algorithm in ("spring", "kamada_kawai"):
        graph = _networkx_layout(graph, algorithm, padding)
    else:
        raise ValueError(f"Unknown layout algorithm: {algorithm}")

    graph = assign_curvatures(graph)
    graph = _detect_crossings(graph)

    return graph


def _networkx_layout(
    graph: GraphSpec, algorithm: str, padding: float
) -> GraphSpec:
    """Apply a networkx layout algorithm."""
    import networkx as nx

    G = nx.DiGraph() if graph.graph_type == "directed" else nx.Graph()

    for node in graph.nodes:
        G.add_node(node.name)
    for edge in graph.edges:
        G.add_edge(edge.source, edge.target)

    if algorithm == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif algorithm == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # NetworkX positions are in [-1, 1] range, normalize to [padding, 1-padding]
    # using uniform scaling to preserve aspect ratio.
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min
    y_range = y_max - y_min

    usable = 1.0 - 2 * padding

    if x_range < 1e-12 and y_range < 1e-12:
        scale = 1.0
    else:
        scale = usable / max(x_range, y_range)

    x_offset = padding + (usable - x_range * scale) / 2
    y_offset = padding + (usable - y_range * scale) / 2

    node_map = {n.name: n for n in graph.nodes}
    for name, (px, py) in pos.items():
        if name in node_map:
            node_map[name].x = x_offset + (px - x_min) * scale
            # Flip y since networkx y increases upward but our y increases downward
            node_map[name].y = y_offset + (y_range - (py - y_min)) * scale

    return graph

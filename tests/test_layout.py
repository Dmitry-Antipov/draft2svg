"""Tests for graph2svg.layout — layout engine."""

from __future__ import annotations

import math

import pytest

from graph2svg.model import Edge, GraphSpec, Node
from graph2svg.layout import (
    _detect_crossings,
    _point_to_segment_dist,
    apply_layout,
    assign_curvatures,
    normalize_positions,
)


# ---------------------------------------------------------------------------
# normalize_positions
# ---------------------------------------------------------------------------

class TestNormalizePositions:
    def test_simple_normalization(self, simple_graph: GraphSpec):
        result = normalize_positions(simple_graph, padding=0.1)
        for node in result.nodes:
            assert 0.1 <= node.x <= 0.9
            assert 0.1 <= node.y <= 0.9

    def test_preserves_relative_order(self, simple_graph: GraphSpec):
        result = normalize_positions(simple_graph, padding=0.1)
        names_x = {n.name: n.x for n in result.nodes}
        assert names_x["S"] < names_x["A"] < names_x["T"]

    def test_fills_range(self, simple_graph: GraphSpec):
        result = normalize_positions(simple_graph, padding=0.1)
        xs = [n.x for n in result.nodes]
        # The extremes should be at the padding boundaries
        assert min(xs) == pytest.approx(0.1)
        assert max(xs) == pytest.approx(0.9)

    def test_single_node_centered(self, single_node_graph: GraphSpec):
        result = normalize_positions(single_node_graph, padding=0.1)
        node = result.nodes[0]
        # With a single node, x_range and y_range are 0 -> fallback to 1.0
        # Position should still be within bounds
        assert 0.1 <= node.x <= 0.9
        assert 0.1 <= node.y <= 0.9

    def test_empty_graph(self, empty_graph: GraphSpec):
        result = normalize_positions(empty_graph, padding=0.1)
        assert result.nodes == []

    def test_custom_padding(self):
        graph = GraphSpec(
            nodes=[
                Node(name="A", x=0.0, y=0.0),
                Node(name="B", x=1.0, y=1.0),
            ],
            edges=[],
        )
        result = normalize_positions(graph, padding=0.2)
        xs = [n.x for n in result.nodes]
        ys = [n.y for n in result.nodes]
        assert min(xs) == pytest.approx(0.2)
        assert max(xs) == pytest.approx(0.8)
        assert min(ys) == pytest.approx(0.2)
        assert max(ys) == pytest.approx(0.8)

    def test_nodes_same_x(self):
        """All nodes on same x -> should not crash (x_range = 0 -> fallback)."""
        graph = GraphSpec(
            nodes=[
                Node(name="A", x=0.5, y=0.0),
                Node(name="B", x=0.5, y=0.5),
                Node(name="C", x=0.5, y=1.0),
            ],
            edges=[],
        )
        result = normalize_positions(graph, padding=0.1)
        ys = [n.y for n in result.nodes]
        assert min(ys) == pytest.approx(0.1)
        assert max(ys) == pytest.approx(0.9)

    def test_preserves_aspect_ratio(self):
        """Uniform scaling must preserve the original aspect ratio.

        A 2:1 rectangle of nodes should remain 2:1 after normalization,
        NOT be stretched to fill both axes independently.
        """
        graph = GraphSpec(
            nodes=[
                Node(name="A", x=0.0, y=0.0),
                Node(name="B", x=2.0, y=0.0),
                Node(name="C", x=2.0, y=1.0),
                Node(name="D", x=0.0, y=1.0),
            ],
            edges=[],
        )
        result = normalize_positions(graph, padding=0.1)
        xs = [n.x for n in result.nodes]
        ys = [n.y for n in result.nodes]
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        # Original was 2:1.  With uniform scaling the longer axis (x) fills
        # the usable range while y is half that.
        assert x_span == pytest.approx(0.8)          # fills usable range
        assert y_span == pytest.approx(0.4)           # half of x_span
        assert x_span / y_span == pytest.approx(2.0)  # aspect preserved

    def test_tall_graph_preserves_aspect(self):
        """A 1:3 tall graph should stay 1:3 after normalization."""
        graph = GraphSpec(
            nodes=[
                Node(name="A", x=0.0, y=0.0),
                Node(name="B", x=1.0, y=3.0),
            ],
            edges=[],
        )
        result = normalize_positions(graph, padding=0.1)
        xs = [n.x for n in result.nodes]
        ys = [n.y for n in result.nodes]
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        # y is the longer axis (3.0) and fills 0.8; x = 1/3 of that
        assert y_span == pytest.approx(0.8)
        assert x_span == pytest.approx(0.8 / 3, abs=1e-6)
        assert y_span / x_span == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# assign_curvatures
# ---------------------------------------------------------------------------

class TestAssignCurvatures:
    def test_single_edge_unchanged(self):
        graph = GraphSpec(
            nodes=[Node(name="A", x=0.0, y=0.0), Node(name="B", x=1.0, y=1.0)],
            edges=[Edge(source="A", target="B", curvature=0.0)],
        )
        result = assign_curvatures(graph)
        assert result.edges[0].curvature == 0.0

    def test_single_edge_preserves_existing_curvature(self):
        graph = GraphSpec(
            nodes=[Node(name="A", x=0.0, y=0.0), Node(name="B", x=1.0, y=1.0)],
            edges=[Edge(source="A", target="B", curvature=0.25)],
        )
        result = assign_curvatures(graph)
        assert result.edges[0].curvature == 0.25

    def test_two_parallel_edges(self, parallel_edges_graph: GraphSpec):
        result = assign_curvatures(parallel_edges_graph)
        curvatures = [e.curvature for e in result.edges]
        # Should get opposite curvatures: 0.3 and -0.3
        assert 0.3 in curvatures
        assert -0.3 in curvatures

    def test_three_parallel_edges(self):
        graph = GraphSpec(
            nodes=[Node(name="A", x=0.0, y=0.0), Node(name="B", x=1.0, y=1.0)],
            edges=[
                Edge(source="A", target="B", label="e_1"),
                Edge(source="B", target="A", label="e_2"),
                Edge(source="A", target="B", label="e_3"),
            ],
        )
        result = assign_curvatures(graph)
        curvatures = sorted([e.curvature for e in result.edges])
        # Spread out: should have negative, zero, positive
        assert curvatures[0] < 0
        assert curvatures[-1] > 0

    def test_independent_pairs(self):
        """Edges between different pairs should be handled independently."""
        graph = GraphSpec(
            nodes=[
                Node(name="A", x=0.0, y=0.0),
                Node(name="B", x=1.0, y=0.0),
                Node(name="C", x=0.5, y=1.0),
            ],
            edges=[
                Edge(source="A", target="B", label="e_1"),
                Edge(source="A", target="C", label="e_2"),
            ],
        )
        result = assign_curvatures(graph)
        # Both are single edges between their pair -> curvature unchanged
        for edge in result.edges:
            # Single edges keep their curvature (default 0.0)
            assert edge.curvature == 0.0


# ---------------------------------------------------------------------------
# _point_to_segment_dist
# ---------------------------------------------------------------------------

class TestPointToSegmentDist:
    def test_point_on_segment(self):
        dist = _point_to_segment_dist((0.5, 0.0), (0.0, 0.0), (1.0, 0.0))
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_point_at_endpoint(self):
        dist = _point_to_segment_dist((0.0, 0.0), (0.0, 0.0), (1.0, 0.0))
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_point_above_midpoint(self):
        dist = _point_to_segment_dist((0.5, 1.0), (0.0, 0.0), (1.0, 0.0))
        assert dist == pytest.approx(1.0, abs=1e-10)

    def test_point_beyond_segment_end(self):
        dist = _point_to_segment_dist((2.0, 0.0), (0.0, 0.0), (1.0, 0.0))
        assert dist == pytest.approx(1.0, abs=1e-10)

    def test_point_before_segment_start(self):
        dist = _point_to_segment_dist((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0))
        assert dist == pytest.approx(1.0, abs=1e-10)

    def test_degenerate_segment(self):
        """Zero-length segment: distance to a point."""
        dist = _point_to_segment_dist((3.0, 4.0), (0.0, 0.0), (0.0, 0.0))
        assert dist == pytest.approx(5.0, abs=1e-10)

    def test_diagonal_segment(self):
        dist = _point_to_segment_dist((0.0, 1.0), (0.0, 0.0), (1.0, 1.0))
        expected = math.sqrt(2) / 2
        assert dist == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# _detect_crossings
# ---------------------------------------------------------------------------

class TestDetectCrossings:
    def test_edge_through_node_gets_curvature(self, crossing_graph: GraphSpec):
        result = _detect_crossings(crossing_graph)
        # e_1 (A->B) passes through C, should get curvature
        e1 = next(e for e in result.edges if e.label == "e_1")
        assert abs(e1.curvature) > 0.01

    def test_no_crossing_unchanged(self, simple_graph: GraphSpec):
        """Edges that don't pass through other nodes stay straight."""
        result = _detect_crossings(simple_graph)
        for edge in result.edges:
            assert edge.curvature == 0.0

    def test_already_curved_edge_unchanged(self):
        """Edges with existing curvature should be skipped."""
        graph = GraphSpec(
            nodes=[
                Node(name="A", x=0.0, y=0.5),
                Node(name="B", x=1.0, y=0.5),
                Node(name="C", x=0.5, y=0.5),
            ],
            edges=[
                Edge(source="A", target="B", curvature=0.4),
            ],
        )
        result = _detect_crossings(graph)
        assert result.edges[0].curvature == 0.4  # unchanged


# ---------------------------------------------------------------------------
# apply_layout (integration)
# ---------------------------------------------------------------------------

class TestApplyLayout:
    def test_original_layout(self, simple_graph: GraphSpec):
        result = apply_layout(simple_graph, algorithm="original")
        # Positions should be normalized
        for node in result.nodes:
            assert 0.0 <= node.x <= 1.0
            assert 0.0 <= node.y <= 1.0

    def test_spring_layout(self, simple_graph: GraphSpec):
        result = apply_layout(simple_graph, algorithm="spring")
        for node in result.nodes:
            assert 0.0 <= node.x <= 1.0
            assert 0.0 <= node.y <= 1.0

    def test_kamada_kawai_layout(self, simple_graph: GraphSpec):
        pytest.importorskip("scipy", reason="kamada_kawai requires scipy")
        result = apply_layout(simple_graph, algorithm="kamada_kawai")
        for node in result.nodes:
            assert 0.0 <= node.x <= 1.0
            assert 0.0 <= node.y <= 1.0

    def test_unknown_layout_raises(self, simple_graph: GraphSpec):
        with pytest.raises(ValueError, match="Unknown layout algorithm"):
            apply_layout(simple_graph, algorithm="nonexistent")

    def test_parallel_edges_get_curvatures(self, parallel_edges_graph: GraphSpec):
        result = apply_layout(parallel_edges_graph, algorithm="original")
        curvatures = [e.curvature for e in result.edges]
        assert any(c != 0.0 for c in curvatures)

    def test_crossing_detection_runs(self, crossing_graph: GraphSpec):
        result = apply_layout(crossing_graph, algorithm="original")
        e1 = next(e for e in result.edges if e.label == "e_1")
        assert abs(e1.curvature) > 0.01

    def test_ground_truth_layout(self, ground_truth_graph: GraphSpec):
        """Apply layout to the ground truth graph without error."""
        result = apply_layout(ground_truth_graph, algorithm="original")
        assert len(result.nodes) == 6
        assert len(result.edges) == 8

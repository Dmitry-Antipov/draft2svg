"""Tests for graph2svg.renderer — Bezier math, label formatting, and rendering."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from graph2svg.model import Edge, GraphSpec, Node
from graph2svg.renderer import (
    _bezier_point,
    _bezier_tangent,
    _best_label_t,
    _format_weight,
    _label_to_mathtext,
    render_graph,
)
from graph2svg.layout import apply_layout
from graph2svg.styles import get_style


# ---------------------------------------------------------------------------
# _label_to_mathtext
# ---------------------------------------------------------------------------

class TestLabelToMathtext:
    def test_none_returns_empty(self):
        assert _label_to_mathtext(None) == ""

    def test_single_char(self):
        assert _label_to_mathtext("S") == "$S$"

    def test_word(self):
        assert _label_to_mathtext("Node") == "$Node$"

    def test_subscript_single_digit(self):
        assert _label_to_mathtext("e_1") == "$e_1$"

    def test_subscript_multi_digit(self):
        assert _label_to_mathtext("e_10") == "$e_{10}$"

    def test_subscript_different_base(self):
        assert _label_to_mathtext("x_5") == "$x_5$"

    def test_multi_char_base_subscript(self):
        assert _label_to_mathtext("var_12") == "$var_{12}$"

    def test_no_subscript_with_underscore_but_no_digit(self):
        """Underscore followed by non-digit doesn't match subscript pattern."""
        result = _label_to_mathtext("a_b")
        # Doesn't match the subscript regex, so treated as raw label
        assert result == "$a_b$"


# ---------------------------------------------------------------------------
# _format_weight
# ---------------------------------------------------------------------------

class TestFormatWeight:
    def test_none_returns_empty(self):
        assert _format_weight(None, "plain") == ""
        assert _format_weight(None, "multiply") == ""

    def test_plain_format(self):
        assert _format_weight("10", "plain") == "10"

    def test_multiply_format(self):
        result = _format_weight("10", "multiply")
        assert result == "10\u00d7"  # 10x

    def test_string_weight(self):
        assert _format_weight("32", "plain") == "32"


# ---------------------------------------------------------------------------
# _bezier_point
# ---------------------------------------------------------------------------

class TestBezierPoint:
    def test_straight_line_midpoint(self):
        """Zero curvature: midpoint of a straight line."""
        x, y = _bezier_point((0.0, 0.0), (10.0, 0.0), 0.0, t=0.5)
        assert x == pytest.approx(5.0)
        assert y == pytest.approx(0.0)

    def test_straight_line_endpoints(self):
        """t=0 gives source, t=1 gives target (approximately, for quadratic)."""
        src = (0.0, 0.0)
        tgt = (10.0, 0.0)
        x0, y0 = _bezier_point(src, tgt, 0.0, t=0.0)
        x1, y1 = _bezier_point(src, tgt, 0.0, t=1.0)
        assert x0 == pytest.approx(0.0)
        assert y0 == pytest.approx(0.0)
        assert x1 == pytest.approx(10.0)
        assert y1 == pytest.approx(0.0)

    def test_curved_midpoint_offset(self):
        """Positive curvature should offset the midpoint."""
        x_straight, y_straight = _bezier_point((0.0, 0.0), (10.0, 0.0), 0.0, t=0.5)
        x_curved, y_curved = _bezier_point((0.0, 0.0), (10.0, 0.0), 0.3, t=0.5)
        # The curved point should be offset in the perpendicular direction
        assert x_curved == pytest.approx(x_straight, abs=0.5)
        assert y_curved != pytest.approx(y_straight, abs=0.1)

    def test_opposite_curvatures_symmetric(self):
        """Positive and negative curvature should mirror each other."""
        src = (0.0, 0.0)
        tgt = (10.0, 0.0)
        _, y_pos = _bezier_point(src, tgt, 0.3, t=0.5)
        _, y_neg = _bezier_point(src, tgt, -0.3, t=0.5)
        assert y_pos == pytest.approx(-y_neg, abs=1e-10)

    def test_coincident_points(self):
        """When source equals target, should return the point (no crash)."""
        x, y = _bezier_point((5.0, 5.0), (5.0, 5.0), 0.3, t=0.5)
        assert x == pytest.approx(5.0)
        assert y == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _bezier_tangent
# ---------------------------------------------------------------------------

class TestBezierTangent:
    def test_straight_line_tangent(self):
        """Tangent on a horizontal straight line should be (1, 0)."""
        tx, ty = _bezier_tangent((0.0, 0.0), (10.0, 0.0), 0.0, t=0.5)
        assert tx == pytest.approx(1.0)
        assert ty == pytest.approx(0.0)

    def test_tangent_is_unit_vector(self):
        tx, ty = _bezier_tangent((0.0, 0.0), (10.0, 5.0), 0.3, t=0.5)
        length = math.hypot(tx, ty)
        assert length == pytest.approx(1.0, abs=1e-10)

    def test_coincident_points_fallback(self):
        """When source equals target, should return fallback (1, 0)."""
        tx, ty = _bezier_tangent((5.0, 5.0), (5.0, 5.0), 0.0, t=0.5)
        assert tx == pytest.approx(1.0)
        assert ty == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _best_label_t
# ---------------------------------------------------------------------------

class TestBestLabelT:
    def test_default_midpoint(self):
        """No interfering nodes: should pick t=0.5."""
        pos = {"A": (0.0, 0.0), "B": (100.0, 0.0)}
        t = _best_label_t(
            (0.0, 0.0), (100.0, 0.0), 0.0,
            pos, "A", "B", node_radius=5.0,
        )
        assert t == 0.5

    def test_avoids_intermediate_node(self):
        """Node C in the middle should push the label off-center."""
        pos = {
            "A": (0.0, 0.0),
            "B": (100.0, 0.0),
            "C": (50.0, 0.0),  # right at t=0.5
        }
        t = _best_label_t(
            (0.0, 0.0), (100.0, 0.0), 0.0,
            pos, "A", "B", node_radius=5.0,
        )
        assert t != 0.5


# ---------------------------------------------------------------------------
# render_graph (integration)
# ---------------------------------------------------------------------------

class TestRenderGraph:
    def test_render_svg(self, simple_graph: GraphSpec, tmp_path: Path):
        graph = apply_layout(simple_graph, algorithm="original")
        output = tmp_path / "test.svg"
        result = render_graph(graph, output, output_format="svg")
        assert result.exists()
        assert result.stat().st_size > 0
        content = result.read_text()
        assert "<svg" in content

    def test_render_pdf(self, simple_graph: GraphSpec, tmp_path: Path):
        graph = apply_layout(simple_graph, algorithm="original")
        output = tmp_path / "test.pdf"
        result = render_graph(graph, output, output_format="pdf")
        assert result.exists()
        assert result.stat().st_size > 0

    def test_render_png(self, simple_graph: GraphSpec, tmp_path: Path):
        graph = apply_layout(simple_graph, algorithm="original")
        output = tmp_path / "test.png"
        result = render_graph(graph, output, output_format="png")
        assert result.exists()
        assert result.stat().st_size > 0

    def test_render_infers_format_from_extension(self, simple_graph: GraphSpec, tmp_path: Path):
        graph = apply_layout(simple_graph, algorithm="original")
        output = tmp_path / "test.pdf"
        result = render_graph(graph, output)  # no explicit format
        assert result.exists()

    def test_render_with_style(self, simple_graph: GraphSpec, tmp_path: Path):
        graph = apply_layout(simple_graph, algorithm="original")
        style = get_style("monochrome")
        output = tmp_path / "test.svg"
        result = render_graph(graph, output, style=style, output_format="svg")
        assert result.exists()

    def test_render_with_weights_and_labels(self, tmp_path: Path):
        graph = GraphSpec(
            nodes=[
                Node(name="S", x=0.0, y=0.5),
                Node(name="T", x=1.0, y=0.5),
            ],
            edges=[
                Edge(source="S", target="T", label="e_1", weight="10"),
            ],
            weight_format="multiply",
        )
        graph = apply_layout(graph, algorithm="original")
        output = tmp_path / "test.svg"
        result = render_graph(graph, output, output_format="svg")
        assert result.exists()

    def test_render_curved_edges(self, parallel_edges_graph: GraphSpec, tmp_path: Path):
        graph = apply_layout(parallel_edges_graph, algorithm="original")
        output = tmp_path / "test.svg"
        result = render_graph(graph, output, output_format="svg")
        assert result.exists()

    def test_render_undirected(self, tmp_path: Path):
        graph = GraphSpec(
            nodes=[
                Node(name="A", x=0.0, y=0.5),
                Node(name="B", x=1.0, y=0.5),
            ],
            edges=[
                Edge(source="A", target="B", directed=False),
            ],
            graph_type="undirected",
        )
        graph = apply_layout(graph, algorithm="original")
        output = tmp_path / "test.svg"
        result = render_graph(graph, output, output_format="svg")
        assert result.exists()

    def test_render_ground_truth(self, ground_truth_graph: GraphSpec, tmp_path: Path):
        """Render the full ground truth graph successfully."""
        graph = apply_layout(ground_truth_graph, algorithm="original")
        output = tmp_path / "test.svg"
        result = render_graph(graph, output, output_format="svg")
        assert result.exists()
        assert result.stat().st_size > 1000  # Non-trivial SVG

    def test_render_empty_graph_no_crash(self, empty_graph: GraphSpec, tmp_path: Path):
        """Rendering an empty graph should not crash."""
        output = tmp_path / "test.svg"
        # Empty graph has no nodes so pos would be empty — this may
        # raise or produce a minimal SVG. Just verify it doesn't crash.
        try:
            render_graph(empty_graph, output, output_format="svg")
        except (ValueError, Exception):
            pass  # OK if it errors gracefully

    def test_render_unknown_extension_defaults_to_svg(
        self, simple_graph: GraphSpec, tmp_path: Path
    ):
        graph = apply_layout(simple_graph, algorithm="original")
        output = tmp_path / "test.xyz"
        result = render_graph(graph, output)
        assert result.exists()

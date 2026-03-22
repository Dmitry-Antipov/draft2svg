"""Tests for graph2svg.model — Pydantic data models and serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph2svg.model import Edge, ExtractionResponse, GraphSpec, Node


# ---------------------------------------------------------------------------
# Node model
# ---------------------------------------------------------------------------

class TestNode:
    def test_create_minimal(self):
        node = Node(name="A", x=0.0, y=1.0)
        assert node.name == "A"
        assert node.x == 0.0
        assert node.y == 1.0
        assert node.shape == "circle"  # default

    def test_create_with_shape(self):
        node = Node(name="B", x=0.5, y=0.5, shape="square")
        assert node.shape == "square"

    def test_serialization_roundtrip(self):
        node = Node(name="C", x=0.3, y=0.7, shape="diamond")
        data = node.model_dump()
        restored = Node.model_validate(data)
        assert restored == node

    def test_json_roundtrip(self):
        node = Node(name="D", x=0.1, y=0.9)
        json_str = node.model_dump_json()
        restored = Node.model_validate_json(json_str)
        assert restored == node


# ---------------------------------------------------------------------------
# Edge model
# ---------------------------------------------------------------------------

class TestEdge:
    def test_create_minimal(self):
        edge = Edge(source="A", target="B")
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.label is None
        assert edge.weight is None
        assert edge.directed is True  # default
        assert edge.color is None
        assert edge.curvature == 0.0  # default

    def test_create_full(self):
        edge = Edge(
            source="S", target="T",
            label="e_1", weight="10",
            directed=True, color="#ff0000",
            curvature=0.3,
        )
        assert edge.label == "e_1"
        assert edge.weight == "10"
        assert edge.color == "#ff0000"
        assert edge.curvature == 0.3

    def test_undirected_edge(self):
        edge = Edge(source="A", target="B", directed=False)
        assert edge.directed is False

    def test_serialization_roundtrip(self):
        edge = Edge(source="X", target="Y", label="e_5", weight="32", curvature=-0.35)
        data = edge.model_dump()
        restored = Edge.model_validate(data)
        assert restored == edge


# ---------------------------------------------------------------------------
# GraphSpec model
# ---------------------------------------------------------------------------

class TestGraphSpec:
    def test_create_minimal(self):
        graph = GraphSpec(nodes=[], edges=[])
        assert graph.nodes == []
        assert graph.edges == []
        assert graph.title is None
        assert graph.graph_type == "directed"
        assert graph.weight_format == "plain"

    def test_create_with_data(self, simple_graph: GraphSpec):
        assert len(simple_graph.nodes) == 3
        assert len(simple_graph.edges) == 2
        assert simple_graph.graph_type == "directed"

    def test_serialization_roundtrip(self, simple_graph: GraphSpec):
        data = simple_graph.model_dump()
        restored = GraphSpec.model_validate(data)
        assert restored == simple_graph

    def test_json_roundtrip(self, simple_graph: GraphSpec):
        json_str = simple_graph.model_dump_json()
        restored = GraphSpec.model_validate_json(json_str)
        assert restored == simple_graph

    def test_to_json_and_from_json(self, simple_graph: GraphSpec, tmp_path: Path):
        json_path = tmp_path / "graph.json"
        simple_graph.to_json(json_path)

        assert json_path.exists()
        loaded = GraphSpec.from_json(json_path)
        assert loaded == simple_graph

    def test_to_json_creates_valid_json(self, simple_graph: GraphSpec, tmp_path: Path):
        json_path = tmp_path / "graph.json"
        simple_graph.to_json(json_path)

        # Verify the file contains valid JSON
        data = json.loads(json_path.read_text())
        assert isinstance(data, dict)
        assert "nodes" in data
        assert "edges" in data

    def test_from_json_ground_truth(self, ground_truth_graph: GraphSpec):
        """Load the actual ground truth file and verify its structure."""
        assert len(ground_truth_graph.nodes) == 6
        assert len(ground_truth_graph.edges) == 8
        assert ground_truth_graph.graph_type == "directed"
        assert ground_truth_graph.weight_format == "multiply"

        node_names = {n.name for n in ground_truth_graph.nodes}
        assert node_names == {"S", "A", "B", "T", "C", "D"}

    def test_from_json_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            GraphSpec.from_json("/nonexistent/path.json")

    def test_weight_format_options(self):
        plain = GraphSpec(nodes=[], edges=[], weight_format="plain")
        assert plain.weight_format == "plain"

        multiply = GraphSpec(nodes=[], edges=[], weight_format="multiply")
        assert multiply.weight_format == "multiply"

    def test_undirected_graph_type(self):
        graph = GraphSpec(nodes=[], edges=[], graph_type="undirected")
        assert graph.graph_type == "undirected"


# ---------------------------------------------------------------------------
# ExtractionResponse model
# ---------------------------------------------------------------------------

class TestExtractionResponse:
    def test_create(self, simple_graph: GraphSpec):
        resp = ExtractionResponse(
            analysis="Found 3 nodes and 2 edges.",
            graph=simple_graph,
        )
        assert resp.analysis == "Found 3 nodes and 2 edges."
        assert resp.graph == simple_graph

    def test_serialization_roundtrip(self, simple_graph: GraphSpec):
        resp = ExtractionResponse(
            analysis="Test analysis",
            graph=simple_graph,
        )
        data = resp.model_dump()
        restored = ExtractionResponse.model_validate(data)
        assert restored == resp

    def test_json_schema_for_api(self):
        schema = ExtractionResponse.json_schema_for_api()
        assert schema["name"] == "extraction_response"
        assert schema["strict"] is True
        assert "schema" in schema
        assert isinstance(schema["schema"], dict)

    def test_json_schema_contains_required_fields(self):
        schema = ExtractionResponse.json_schema_for_api()
        properties = schema["schema"].get("properties", {})
        assert "analysis" in properties
        assert "graph" in properties

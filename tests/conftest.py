"""Shared fixtures for graph2svg tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from graph2svg.model import Edge, GraphSpec, Node


FIXTURES_DIR = Path(__file__).parent / "fixtures"
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.fixture
def simple_graph() -> GraphSpec:
    """A minimal directed graph: S -> A -> T."""
    return GraphSpec(
        nodes=[
            Node(name="S", x=0.0, y=0.5),
            Node(name="A", x=0.5, y=0.5),
            Node(name="T", x=1.0, y=0.5),
        ],
        edges=[
            Edge(source="S", target="A", label="e_1", weight="10", directed=True),
            Edge(source="A", target="T", label="e_2", weight="20", directed=True),
        ],
        graph_type="directed",
        weight_format="plain",
    )


@pytest.fixture
def ground_truth_graph() -> GraphSpec:
    """Load the ground truth graph from examples/input1_ground_truth.json."""
    path = EXAMPLES_DIR / "input1_ground_truth.json"
    return GraphSpec.from_json(path)


@pytest.fixture
def parallel_edges_graph() -> GraphSpec:
    """A graph with multiple edges between the same pair of nodes."""
    return GraphSpec(
        nodes=[
            Node(name="A", x=0.0, y=0.5),
            Node(name="B", x=1.0, y=0.5),
        ],
        edges=[
            Edge(source="A", target="B", label="e_1", directed=True),
            Edge(source="B", target="A", label="e_2", directed=True),
        ],
        graph_type="directed",
    )


@pytest.fixture
def crossing_graph() -> GraphSpec:
    """A graph where a straight edge passes through an intermediate node."""
    return GraphSpec(
        nodes=[
            Node(name="A", x=0.0, y=0.5),
            Node(name="B", x=1.0, y=0.5),
            Node(name="C", x=0.5, y=0.5),  # right on the line A->B
        ],
        edges=[
            Edge(source="A", target="B", label="e_1", directed=True, curvature=0.0),
            Edge(source="A", target="C", label="e_2", directed=True),
        ],
        graph_type="directed",
    )


@pytest.fixture
def empty_graph() -> GraphSpec:
    """An empty graph with no nodes or edges."""
    return GraphSpec(nodes=[], edges=[])


@pytest.fixture
def single_node_graph() -> GraphSpec:
    """A graph with one node and no edges."""
    return GraphSpec(
        nodes=[Node(name="X", x=0.5, y=0.5)],
        edges=[],
    )

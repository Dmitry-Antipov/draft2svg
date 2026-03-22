"""Data models for graph representation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Node(BaseModel):
    """A node in the graph."""

    name: str = Field(description="Display name of the node (e.g. 'S', 'A', 'B')")
    x: float = Field(
        description="Relative x position (0.0 = leftmost, 1.0 = rightmost)"
    )
    y: float = Field(
        description="Relative y position (0.0 = topmost, 1.0 = bottommost)"
    )
    shape: str = Field(
        default="circle",
        description="Node shape: circle, square, diamond",
    )


class Edge(BaseModel):
    """A directed or undirected edge in the graph."""

    source: str = Field(description="Name of the source node")
    target: str = Field(description="Name of the target node")
    label: str | None = Field(
        default=None,
        description="Edge label, e.g. 'e_1', 'x_1'. Use underscore for subscripts.",
    )
    weight: str | None = Field(
        default=None,
        description="Edge weight/capacity as a string, e.g. '32', '10'",
    )
    directed: bool = Field(default=True, description="Whether the edge is directed")
    color: str | None = Field(
        default=None,
        description="Edge color as hex string. Auto-assigned if None.",
    )
    curvature: float = Field(
        default=0.0,
        description="Edge curvature. 0=straight, positive=curves left, negative=curves right.",
    )


class GraphSpec(BaseModel):
    """Complete specification of a graph extracted from an image."""

    nodes: list[Node] = Field(description="List of nodes in the graph")
    edges: list[Edge] = Field(description="List of edges in the graph")
    title: str | None = Field(default=None, description="Optional graph title")
    graph_type: str = Field(
        default="directed",
        description="Graph type: directed, undirected",
    )
    weight_format: str = Field(
        default="plain",
        description="How weights are displayed: 'plain' (just number) or 'multiply' (number followed by x/times sign)",
    )

    def to_json(self, path: str | Path) -> None:
        """Save the graph spec to a JSON file."""
        path = Path(path)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> GraphSpec:
        """Load a graph spec from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        return cls.model_validate(data)


class ExtractionResponse(BaseModel):
    """Wrapper model for structured LLM output.

    Preserves chain-of-thought reasoning alongside the graph data,
    ensuring the LLM can reason step-by-step while still producing
    guaranteed-valid JSON output.
    """

    analysis: str = Field(
        description=(
            "Step-by-step analysis of the hand-drawn graph. "
            "Include: nodes found, edge labels counted, each edge traced, "
            "curvature assignments, and weight format determination."
        )
    )
    graph: GraphSpec = Field(
        description="The extracted graph specification."
    )

    @classmethod
    def json_schema_for_api(cls) -> dict[str, Any]:
        """Return the JSON schema in the format expected by the OpenAI API.

        Produces a schema dict suitable for:
            response_format={"type": "json_schema", "json_schema": schema}
        """
        return {
            "name": "extraction_response",
            "strict": True,
            "schema": cls.model_json_schema(),
        }

"""Tests for graph2svg.extractor — validation, caching, and JSON parsing.

These tests focus on the parts that don't require an LLM API call:
- _parse_json_response
- _validate_graph
- Caching helpers
- _load_and_encode_image
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from graph2svg.extractor import (
    _cache_dir,
    _cache_get,
    _cache_key,
    _cache_put,
    _compose_graph,
    _crop_edge_region,
    _diff_edge_sets,
    _edge_pair,
    _find_dense_edges,
    _format_disputes_for_prompt,
    _format_edges_for_zoom_verify,
    _apply_zoom_verdicts,
    _load_and_encode_image,
    _merge_graph,
    _parse_adjudication_response,
    _parse_json_response,
    _prompt_hash,
    _validate_graph,
    EdgeDispute,
)
from graph2svg.model import Edge, GraphSpec, Node


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_plain_json(self):
        data = {"nodes": [], "edges": []}
        result = _parse_json_response(json.dumps(data))
        assert result == data

    def test_fenced_json(self):
        text = 'Here is the result:\n```json\n{"nodes": [], "edges": []}\n```\n'
        result = _parse_json_response(text)
        assert result == {"nodes": [], "edges": []}

    def test_fenced_without_language(self):
        text = 'Result:\n```\n{"nodes": [], "edges": []}\n```\n'
        result = _parse_json_response(text)
        assert result == {"nodes": [], "edges": []}

    def test_chain_of_thought_then_json(self):
        text = (
            "Let me analyze the image...\n"
            "I found 3 nodes and 2 edges.\n\n"
            '```json\n'
            '{"nodes": [{"name": "A", "x": 0, "y": 0}], "edges": []}\n'
            '```'
        )
        result = _parse_json_response(text)
        assert result["nodes"][0]["name"] == "A"

    def test_multiple_fenced_blocks_uses_last(self):
        text = (
            'First attempt:\n```json\n{"bad": true}\n```\n'
            'Correction:\n```json\n{"good": true}\n```\n'
        )
        result = _parse_json_response(text)
        assert result == {"good": True}

    def test_unfenced_json_block(self):
        text = (
            'Some analysis text\n'
            '{"nodes": [{"name": "B", "x": 0.5, "y": 0.5}], "edges": []}'
        )
        result = _parse_json_response(text)
        assert result["nodes"][0]["name"] == "B"

    def test_nested_json_objects(self):
        data = {
            "nodes": [{"name": "X", "x": 0.0, "y": 0.0, "shape": "circle"}],
            "edges": [{"source": "X", "target": "X", "label": "e_1"}],
        }
        text = f"Result:\n```json\n{json.dumps(data, indent=2)}\n```"
        result = _parse_json_response(text)
        assert result["edges"][0]["source"] == "X"

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_response("This is not JSON at all")

    def test_empty_string_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_response("")


# ---------------------------------------------------------------------------
# _validate_graph
# ---------------------------------------------------------------------------

class TestValidateGraph:
    def test_valid_graph_unchanged(self, simple_graph: GraphSpec):
        result = _validate_graph(simple_graph)
        assert len(result.edges) == 2
        assert len(result.nodes) == 3

    def test_removes_self_loops(self):
        graph = GraphSpec(
            nodes=[Node(name="A", x=0.0, y=0.0)],
            edges=[
                Edge(source="A", target="A", label="loop"),
            ],
        )
        result = _validate_graph(graph)
        assert len(result.edges) == 0

    def test_removes_duplicate_edges(self):
        graph = GraphSpec(
            nodes=[
                Node(name="A", x=0.0, y=0.0),
                Node(name="B", x=1.0, y=1.0),
            ],
            edges=[
                Edge(source="A", target="B", label="e_1"),
                Edge(source="A", target="B", label="e_1"),  # duplicate
            ],
        )
        result = _validate_graph(graph)
        assert len(result.edges) == 1

    def test_keeps_edges_with_different_labels(self):
        graph = GraphSpec(
            nodes=[
                Node(name="A", x=0.0, y=0.0),
                Node(name="B", x=1.0, y=1.0),
            ],
            edges=[
                Edge(source="A", target="B", label="e_1"),
                Edge(source="A", target="B", label="e_2"),
            ],
        )
        result = _validate_graph(graph)
        assert len(result.edges) == 2

    def test_warns_about_unknown_nodes(self, capsys: pytest.CaptureFixture):
        graph = GraphSpec(
            nodes=[Node(name="A", x=0.0, y=0.0)],
            edges=[
                Edge(source="A", target="Z", label="e_1"),  # Z doesn't exist
            ],
        )
        result = _validate_graph(graph, verbose=True)
        captured = capsys.readouterr()
        assert "unknown nodes" in captured.err.lower() or "Z" in captured.err

    def test_warns_about_isolated_nodes(self, capsys: pytest.CaptureFixture):
        graph = GraphSpec(
            nodes=[
                Node(name="A", x=0.0, y=0.0),
                Node(name="B", x=1.0, y=1.0),
                Node(name="Isolated", x=0.5, y=0.5),
            ],
            edges=[
                Edge(source="A", target="B"),
            ],
        )
        result = _validate_graph(graph, verbose=True)
        captured = capsys.readouterr()
        assert "Isolated" in captured.err

    def test_warns_about_duplicate_labels(self, capsys: pytest.CaptureFixture):
        graph = GraphSpec(
            nodes=[
                Node(name="A", x=0.0, y=0.0),
                Node(name="B", x=1.0, y=0.0),
                Node(name="C", x=0.5, y=1.0),
            ],
            edges=[
                Edge(source="A", target="B", label="e_1"),
                Edge(source="A", target="C", label="e_1"),  # same label
            ],
        )
        result = _validate_graph(graph, verbose=True)
        captured = capsys.readouterr()
        assert "e_1" in captured.err

    def test_verbose_no_issues(self, simple_graph: GraphSpec, capsys: pytest.CaptureFixture):
        _validate_graph(simple_graph, verbose=True)
        captured = capsys.readouterr()
        assert "no issues" in captured.err.lower()

    def test_ground_truth_validates(self, ground_truth_graph: GraphSpec):
        result = _validate_graph(ground_truth_graph)
        assert len(result.nodes) == 6
        assert len(result.edges) == 8


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

class TestCaching:
    def test_prompt_hash_deterministic(self):
        h1 = _prompt_hash("test prompt")
        h2 = _prompt_hash("test prompt")
        assert h1 == h2

    def test_prompt_hash_different_inputs(self):
        h1 = _prompt_hash("prompt A")
        h2 = _prompt_hash("prompt B")
        assert h1 != h2

    def test_prompt_hash_length(self):
        h = _prompt_hash("test")
        assert len(h) == 16

    def test_cache_put_and_get(self, tmp_path: Path):
        with patch("graph2svg.extractor._cache_dir", return_value=tmp_path):
            _cache_put("test_key_123", '{"test": "data"}')
            result = _cache_get("test_key_123")
            assert result == '{"test": "data"}'

    def test_cache_get_missing(self, tmp_path: Path):
        with patch("graph2svg.extractor._cache_dir", return_value=tmp_path):
            result = _cache_get("nonexistent_key")
            assert result is None

    def test_cache_get_corrupted(self, tmp_path: Path):
        """Corrupted cache file should return None."""
        cache_file = tmp_path / "bad_key.json"
        cache_file.write_text("not valid json")
        with patch("graph2svg.extractor._cache_dir", return_value=tmp_path):
            result = _cache_get("bad_key")
            assert result is None

    def test_cache_key_deterministic(self, tmp_path: Path):
        """Same inputs produce the same cache key."""
        # Create a dummy image file
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"fake image data")

        k1 = _cache_key(str(img_file), "gpt-4.1", "abc123", "extract")
        k2 = _cache_key(str(img_file), "gpt-4.1", "abc123", "extract")
        assert k1 == k2

    def test_cache_key_varies_with_model(self, tmp_path: Path):
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"fake image data")

        k1 = _cache_key(str(img_file), "gpt-4.1", "abc123", "extract")
        k2 = _cache_key(str(img_file), "gpt-4o", "abc123", "extract")
        assert k1 != k2

    def test_cache_key_varies_with_pass(self, tmp_path: Path):
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"fake image data")

        k1 = _cache_key(str(img_file), "gpt-4.1", "abc123", "extract")
        k2 = _cache_key(str(img_file), "gpt-4.1", "abc123", "verify")
        assert k1 != k2

    def test_cache_dir_creates_directory(self, tmp_path: Path):
        target = tmp_path / "sub" / "graph2svg"
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path / "sub")}):
            result = _cache_dir()
            assert result.exists()
            assert result.name == "graph2svg"


# ---------------------------------------------------------------------------
# _load_and_encode_image
# ---------------------------------------------------------------------------

class TestLoadAndEncodeImage:
    def test_loads_png(self):
        """Load the example PNG file."""
        img_path = Path(__file__).parent.parent / "examples" / "input1.png"
        if not img_path.exists():
            pytest.skip("Example image not available")

        b64_data, media_type = _load_and_encode_image(str(img_path))
        assert media_type == "image/jpeg"
        assert len(b64_data) > 0
        # Verify it's valid base64
        import base64
        decoded = base64.b64decode(b64_data)
        assert len(decoded) > 0

    def test_creates_small_enough_image(self):
        """Ensure large images are resized."""
        from PIL import Image
        import tempfile

        # Create a large test image
        img = Image.new("RGB", (4000, 3000), color=(255, 255, 255))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            tmp_path = f.name

        try:
            b64_data, media_type = _load_and_encode_image(tmp_path)
            assert media_type == "image/jpeg"
            # Decode and check dimensions
            import base64
            from io import BytesIO
            decoded = base64.b64decode(b64_data)
            result_img = Image.open(BytesIO(decoded))
            assert max(result_img.size) <= 2048
        finally:
            os.unlink(tmp_path)

    def test_converts_rgba_to_rgb(self):
        """RGBA images should be converted to RGB."""
        from PIL import Image
        import tempfile

        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            tmp_path = f.name

        try:
            b64_data, media_type = _load_and_encode_image(tmp_path)
            assert media_type == "image/jpeg"
            assert len(b64_data) > 0
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------

def _make_graph(edges_spec: list[tuple], nodes: list[Node] | None = None) -> GraphSpec:
    """Helper to build a GraphSpec from a list of (source, target, label, weight) tuples."""
    if nodes is None:
        # Auto-create nodes from edges
        names = set()
        for src, tgt, *_ in edges_spec:
            names.add(src)
            names.add(tgt)
        nodes = [Node(name=n, x=0.0, y=0.0) for n in sorted(names)]

    edges = []
    for spec in edges_spec:
        src, tgt = spec[0], spec[1]
        label = spec[2] if len(spec) > 2 else None
        weight = spec[3] if len(spec) > 3 else None
        curvature = spec[4] if len(spec) > 4 else 0.0
        edges.append(Edge(
            source=src, target=tgt, label=label, weight=weight,
            directed=True, curvature=curvature,
        ))
    return GraphSpec(nodes=nodes, edges=edges, graph_type="directed")


class TestEdgePair:
    def test_ordered_pair(self):
        e = Edge(source="A", target="B")
        assert _edge_pair(e) == ("A", "B")

    def test_reverse_pair(self):
        e = Edge(source="B", target="A")
        assert _edge_pair(e) == ("A", "B")

    def test_same_pair_different_direction(self):
        e1 = Edge(source="A", target="B")
        e2 = Edge(source="B", target="A")
        assert _edge_pair(e1) == _edge_pair(e2)


class TestDiffEdgeSets:
    def test_identical_graphs_all_agreed(self):
        g1 = _make_graph([("A", "B", "e_1", "10"), ("B", "C", "e_2", "20")])
        g2 = _make_graph([("A", "B", "e_1", "10"), ("B", "C", "e_2", "20")])
        agreed, disputes = _diff_edge_sets(g1, g2)
        assert len(agreed) == 2
        assert len(disputes) == 0

    def test_direction_disagreement(self):
        """Edges between same nodes but different directions → dispute."""
        g1 = _make_graph([("A", "B", "e_1", "10")])
        g2 = _make_graph([("B", "A", "e_1", "10")])
        agreed, disputes = _diff_edge_sets(g1, g2)
        assert len(agreed) == 0
        assert len(disputes) == 1
        assert "direction" in disputes[0].reason

    def test_weight_disagreement(self):
        """Same edge but different weight → dispute."""
        g1 = _make_graph([("A", "B", "e_1", "10")])
        g2 = _make_graph([("A", "B", "e_1", "20")])
        agreed, disputes = _diff_edge_sets(g1, g2)
        assert len(agreed) == 0
        assert len(disputes) == 1
        assert "weight" in disputes[0].reason

    def test_label_disagreement(self):
        """Same edge but different label → dispute."""
        g1 = _make_graph([("A", "B", "e_1", "10")])
        g2 = _make_graph([("A", "B", "e_2", "10")])
        agreed, disputes = _diff_edge_sets(g1, g2)
        assert len(agreed) == 0
        assert len(disputes) == 1
        assert "label" in disputes[0].reason

    def test_pass1_only_edge(self):
        """Edge only in pass 1 → dispute."""
        g1 = _make_graph([("A", "B", "e_1", "10"), ("B", "C", "e_2", "5")])
        g2 = _make_graph([("A", "B", "e_1", "10")])
        agreed, disputes = _diff_edge_sets(g1, g2)
        assert len(agreed) == 1
        assert len(disputes) == 1
        assert "only in Pass 1" in disputes[0].reason
        assert disputes[0].pass1_edge is not None
        assert disputes[0].pass2_edge is None

    def test_pass2_only_edge(self):
        """Edge only in pass 2 → dispute."""
        g1 = _make_graph([("A", "B", "e_1", "10")])
        g2 = _make_graph([("A", "B", "e_1", "10"), ("C", "D", "e_2", "5")])
        agreed, disputes = _diff_edge_sets(g1, g2)
        assert len(agreed) == 1
        assert len(disputes) == 1
        assert "only in Pass 2" in disputes[0].reason
        assert disputes[0].pass1_edge is None
        assert disputes[0].pass2_edge is not None

    def test_mixed_agreements_and_disputes(self):
        """Some edges agree, some dispute — realistic scenario."""
        g1 = _make_graph([
            ("A", "B", "e_1", "10"),  # agrees
            ("B", "C", "e_2", "20"),  # agrees
            ("C", "A", "e_3", "8"),   # direction disagrees
            ("D", "A", None, None),   # pass-1 only (phantom)
        ])
        g2 = _make_graph([
            ("A", "B", "e_1", "10"),  # agrees
            ("B", "C", "e_2", "20"),  # agrees
            ("A", "C", "e_3", "8"),   # direction disagrees
            ("E", "F", None, None),   # pass-2 only
        ])
        agreed, disputes = _diff_edge_sets(g1, g2)
        assert len(agreed) == 2  # A->B and B->C
        assert len(disputes) == 3  # direction, pass-1-only, pass-2-only

    def test_empty_graphs(self):
        g1 = _make_graph([])
        g2 = _make_graph([])
        g1.nodes = []
        g2.nodes = []
        agreed, disputes = _diff_edge_sets(g1, g2)
        assert len(agreed) == 0
        assert len(disputes) == 0

    def test_curvature_averaged(self):
        """When edges agree, curvature should be averaged."""
        g1 = _make_graph([("A", "B", "e_1", "10", 0.2)])
        g2 = _make_graph([("A", "B", "e_1", "10", 0.4)])
        agreed, disputes = _diff_edge_sets(g1, g2)
        assert len(agreed) == 1
        assert len(disputes) == 0
        assert abs(agreed[0].curvature - 0.3) < 0.01

    def test_null_labels_agree(self):
        """Edges with no labels (unlabeled graphs) should still agree."""
        g1 = _make_graph([("A", "B", None, None), ("B", "C", None, None)])
        g2 = _make_graph([("A", "B", None, None), ("B", "C", None, None)])
        agreed, disputes = _diff_edge_sets(g1, g2)
        assert len(agreed) == 2
        assert len(disputes) == 0


class TestEdgeDispute:
    def test_description_both_edges(self):
        d = EdgeDispute(
            dispute_id="dispute_1",
            pass1_edge=Edge(source="A", target="B", label="e_1", weight="10"),
            pass2_edge=Edge(source="B", target="A", label="e_1", weight="10"),
            reason="direction disagrees",
        )
        desc = d.to_description()
        assert "dispute_1" in desc
        assert "Pass 1" in desc
        assert "Pass 2" in desc
        assert "A" in desc and "B" in desc

    def test_description_pass1_only(self):
        d = EdgeDispute(
            dispute_id="dispute_2",
            pass1_edge=Edge(source="X", target="Y"),
            pass2_edge=None,
            reason="only in Pass 1",
        )
        desc = d.to_description()
        assert "not present" in desc

    def test_description_pass2_only(self):
        d = EdgeDispute(
            dispute_id="dispute_3",
            pass1_edge=None,
            pass2_edge=Edge(source="X", target="Y"),
            reason="only in Pass 2",
        )
        desc = d.to_description()
        assert "not present" in desc


class TestFormatDisputesForPrompt:
    def test_formats_multiple_disputes(self):
        disputes = [
            EdgeDispute("d_1", Edge(source="A", target="B"), Edge(source="B", target="A"), "direction"),
            EdgeDispute("d_2", Edge(source="C", target="D"), None, "pass-1 only"),
        ]
        text = _format_disputes_for_prompt(disputes)
        assert "d_1" in text
        assert "d_2" in text
        assert text.count("###") == 2


class TestParseAdjudicationResponse:
    def test_fenced_json_array(self):
        text = 'Analysis...\n```json\n[{"dispute_id": "d_1", "exists": true, "source": "A", "target": "B"}]\n```\n'
        result = _parse_adjudication_response(text)
        assert len(result) == 1
        assert result[0]["dispute_id"] == "d_1"
        assert result[0]["exists"] is True

    def test_multiple_verdicts(self):
        verdicts = [
            {"dispute_id": "d_1", "exists": True, "source": "A", "target": "B"},
            {"dispute_id": "d_2", "exists": False},
        ]
        text = f"Here:\n```json\n{json.dumps(verdicts)}\n```"
        result = _parse_adjudication_response(text)
        assert len(result) == 2
        assert result[1]["exists"] is False

    def test_bare_json_array(self):
        text = 'Some reasoning\n[{"dispute_id": "d_1", "exists": true, "source": "X", "target": "Y"}]'
        result = _parse_adjudication_response(text)
        assert len(result) == 1

    def test_invalid_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_adjudication_response("no JSON here")


class TestMergeGraph:
    def test_merge_agreed_only(self):
        """When there are no disputes, merge uses agreed edges and averaged positions."""
        nodes1 = [Node(name="A", x=0.1, y=0.2), Node(name="B", x=0.8, y=0.9)]
        nodes2 = [Node(name="A", x=0.2, y=0.3), Node(name="B", x=0.7, y=0.8)]
        g1 = GraphSpec(nodes=nodes1, edges=[Edge(source="A", target="B")])
        g2 = GraphSpec(nodes=nodes2, edges=[])
        agreed = [Edge(source="A", target="B")]
        result = _merge_graph(g1, g2, agreed, [])
        assert len(result.edges) == 1
        # Positions should be averaged
        node_a = next(n for n in result.nodes if n.name == "A")
        assert abs(node_a.x - 0.15) < 0.01
        assert abs(node_a.y - 0.25) < 0.01

    def test_merge_with_verdicts(self):
        """Verdicts that exist=True should be added as edges."""
        g1 = _make_graph([("A", "B")])
        g2 = _make_graph([("A", "B")])
        agreed = [Edge(source="A", target="B")]
        verdicts = [
            {"dispute_id": "d_1", "exists": True, "source": "B", "target": "C",
             "label": "e_2", "weight": "5", "directed": True, "curvature": 0.0},
        ]
        # Add node C to both graphs
        g1.nodes.append(Node(name="C", x=1.0, y=1.0))
        g2.nodes.append(Node(name="C", x=1.0, y=1.0))
        result = _merge_graph(g1, g2, agreed, verdicts)
        assert len(result.edges) == 2

    def test_merge_with_nonexistent_verdict(self):
        """Verdicts that exist=False should not produce edges."""
        g1 = _make_graph([("A", "B")])
        g2 = _make_graph([("A", "B")])
        agreed = [Edge(source="A", target="B")]
        verdicts = [
            {"dispute_id": "d_1", "exists": False},
        ]
        result = _merge_graph(g1, g2, agreed, verdicts)
        assert len(result.edges) == 1  # only the agreed edge

    def test_merge_preserves_graph_type(self):
        g1 = GraphSpec(
            nodes=[Node(name="A", x=0, y=0)],
            edges=[], graph_type="directed", weight_format="multiply",
        )
        g2 = GraphSpec(
            nodes=[Node(name="A", x=0, y=0)],
            edges=[],
        )
        result = _merge_graph(g1, g2, [], [])
        assert result.graph_type == "directed"
        assert result.weight_format == "multiply"

    def test_merge_union_of_nodes(self):
        """If one pass finds a node the other doesn't, it should be included."""
        g1 = GraphSpec(nodes=[Node(name="A", x=0.1, y=0.2)], edges=[])
        g2 = GraphSpec(
            nodes=[Node(name="A", x=0.1, y=0.2), Node(name="B", x=0.5, y=0.5)],
            edges=[],
        )
        result = _merge_graph(g1, g2, [], [])
        names = {n.name for n in result.nodes}
        assert names == {"A", "B"}


# ---------------------------------------------------------------------------
# Zoom verification helpers
# ---------------------------------------------------------------------------

def _make_dense_graph() -> GraphSpec:
    """Build a graph with a dense cluster (A, C, D close together) and
    distant nodes (S far left, T far right, B mid-right).

    Mimics the input1 topology.
    """
    return GraphSpec(
        nodes=[
            Node(name="S", x=0.08, y=0.50),
            Node(name="A", x=0.22, y=0.52),
            Node(name="C", x=0.40, y=0.54),
            Node(name="B", x=0.62, y=0.44),
            Node(name="D", x=0.42, y=0.72),
            Node(name="T", x=0.92, y=0.50),
        ],
        edges=[
            Edge(source="S", target="A", label="e_1", weight="10", directed=True),
            Edge(source="B", target="T", label="e_2", weight="10", directed=True),
            Edge(source="A", target="B", label="e_5", weight="32", directed=True),
            Edge(source="A", target="C", label="e_6", weight="6", directed=True),
            Edge(source="D", target="C", label="e_9", weight="10", directed=True),
            Edge(source="C", target="B", label="e_4", weight="6", directed=True),
            Edge(source="B", target="D", label="e_3", weight="12", directed=True),
            Edge(source="A", target="D", label="e_8", weight="10", directed=True),
        ],
        graph_type="directed",
    )


class TestFindDenseEdges:
    def test_finds_dense_edges_in_cluster(self):
        """Edges between close nodes in a dense cluster should be detected."""
        graph = _make_dense_graph()
        dense = _find_dense_edges(graph)
        dense_pairs = {(e.source, e.target) for e in dense}
        # A-C (dist ~0.18), D-C (dist ~0.18) should be dense
        assert ("A", "C") in dense_pairs or ("C", "A") in dense_pairs
        assert ("D", "C") in dense_pairs or ("C", "D") in dense_pairs

    def test_excludes_long_edges(self):
        """Long edges (B→T, A→B) should not be in the dense set."""
        graph = _make_dense_graph()
        dense = _find_dense_edges(graph)
        dense_pairs = {(e.source, e.target) for e in dense}
        assert ("B", "T") not in dense_pairs
        assert ("A", "B") not in dense_pairs

    def test_isolated_short_edge_excluded(self):
        """A short edge between two isolated nodes should NOT be dense."""
        graph = GraphSpec(
            nodes=[
                Node(name="X", x=0.10, y=0.50),
                Node(name="Y", x=0.15, y=0.50),
                # Far away nodes
                Node(name="P", x=0.90, y=0.10),
                Node(name="Q", x=0.95, y=0.10),
            ],
            edges=[
                Edge(source="X", target="Y", label="e_1", directed=True),
                Edge(source="P", target="Q", label="e_2", directed=True),
            ],
            graph_type="directed",
        )
        dense = _find_dense_edges(graph)
        # X-Y is short (0.05) but isolated (no other nodes nearby)
        # P-Q is also short and isolated
        # Neither should qualify as dense (< 3 nearby nodes)
        assert len(dense) == 0

    def test_all_nodes_close_together(self):
        """When all nodes are close, most edges should be dense."""
        graph = GraphSpec(
            nodes=[
                Node(name="A", x=0.50, y=0.50),
                Node(name="B", x=0.55, y=0.50),
                Node(name="C", x=0.52, y=0.55),
                Node(name="D", x=0.48, y=0.52),
            ],
            edges=[
                Edge(source="A", target="B", directed=True),
                Edge(source="B", target="C", directed=True),
                Edge(source="C", target="D", directed=True),
                Edge(source="D", target="A", directed=True),
            ],
            graph_type="directed",
        )
        dense = _find_dense_edges(graph)
        assert len(dense) == 4  # all edges are short and in a dense area

    def test_empty_graph_no_dense(self):
        graph = GraphSpec(nodes=[], edges=[], graph_type="directed")
        dense = _find_dense_edges(graph)
        assert len(dense) == 0

    def test_custom_threshold(self):
        """Lowering threshold should reduce the number of dense edges."""
        graph = _make_dense_graph()
        dense_wide = _find_dense_edges(graph, threshold=0.30)
        dense_narrow = _find_dense_edges(graph, threshold=0.10)
        assert len(dense_narrow) <= len(dense_wide)


class TestCropEdgeRegion:
    def test_crop_produces_base64(self):
        """Cropping around an edge should produce valid base64 JPEG data."""
        img_path = Path(__file__).parent.parent / "examples" / "input1.png"
        if not img_path.exists():
            pytest.skip("Example image not available")

        node_pos = {"A": (0.22, 0.52), "C": (0.40, 0.54)}
        edge = Edge(source="A", target="C", label="e_6", directed=True)
        b64, media_type, rel_pos = _crop_edge_region(
            str(img_path), node_pos, [edge],
        )
        assert media_type == "image/jpeg"
        assert len(b64) > 0
        # Check it's valid base64
        import base64
        decoded = base64.b64decode(b64)
        assert len(decoded) > 100  # not trivially empty

    def test_relative_positions_in_range(self):
        """Node positions in the crop should be in [0, 1] range."""
        img_path = Path(__file__).parent.parent / "examples" / "input1.png"
        if not img_path.exists():
            pytest.skip("Example image not available")

        node_pos = {"A": (0.22, 0.52), "C": (0.40, 0.54)}
        edge = Edge(source="A", target="C", label="e_6", directed=True)
        _, _, rel_pos = _crop_edge_region(
            str(img_path), node_pos, [edge],
        )
        assert "A" in rel_pos
        assert "C" in rel_pos
        for name, (rx, ry) in rel_pos.items():
            assert 0.0 <= rx <= 1.0, f"{name} rel_x={rx} out of range"
            assert 0.0 <= ry <= 1.0, f"{name} rel_y={ry} out of range"

    def test_single_edge_crop_smaller_than_full(self):
        """Crop of a single edge should be smaller than the full image."""
        img_path = Path(__file__).parent.parent / "examples" / "input1.png"
        if not img_path.exists():
            pytest.skip("Example image not available")

        from PIL import Image
        full_img = Image.open(str(img_path))
        full_size = full_img.size[0] * full_img.size[1]

        node_pos = {"D": (0.42, 0.72), "C": (0.40, 0.54)}
        edge = Edge(source="D", target="C", label="e_9", directed=True)
        b64, _, _ = _crop_edge_region(
            str(img_path), node_pos, [edge],
        )

        import base64 as b64_mod
        from io import BytesIO
        decoded = b64_mod.b64decode(b64)
        crop_img = Image.open(BytesIO(decoded))
        crop_size = crop_img.size[0] * crop_img.size[1]
        assert crop_size < full_size

    def test_crop_with_missing_nodes_returns_full(self):
        """If node positions are empty, should return full image."""
        img_path = Path(__file__).parent.parent / "examples" / "input1.png"
        if not img_path.exists():
            pytest.skip("Example image not available")

        edge = Edge(source="X", target="Y", directed=True)
        b64, _, _ = _crop_edge_region(str(img_path), {}, [edge])
        assert len(b64) > 0


class TestFormatEdgesForZoomVerify:
    def test_format_with_positions(self):
        edges = [
            Edge(source="A", target="C", label="e_6", weight="6", directed=True),
        ]
        rel_pos = {"A": (0.17, 0.47), "C": (0.83, 0.53)}
        text = _format_edges_for_zoom_verify(edges, rel_pos)
        assert "A" in text
        assert "C" in text
        assert "e_6" not in text or "A→C" in text  # format may vary
        assert "17%" in text or "0.17" in text  # position info included

    def test_format_multiple_edges(self):
        edges = [
            Edge(source="A", target="B", directed=True),
            Edge(source="C", target="D", directed=True),
        ]
        rel_pos = {"A": (0.1, 0.2), "B": (0.9, 0.8), "C": (0.3, 0.4), "D": (0.7, 0.6)}
        text = _format_edges_for_zoom_verify(edges, rel_pos)
        assert "1." in text
        assert "2." in text


class TestApplyZoomVerdicts:
    def test_direction_correction(self):
        """Zoom verdict should flip edge direction."""
        graph = GraphSpec(
            nodes=[
                Node(name="A", x=0.2, y=0.5),
                Node(name="C", x=0.4, y=0.5),
            ],
            edges=[
                Edge(source="A", target="C", label="e_6", weight="6", directed=True),
            ],
            graph_type="directed",
        )
        verdicts = [{"source": "C", "target": "A", "weight": "8", "confidence": "high"}]
        original_edges = [graph.edges[0]]

        result = _apply_zoom_verdicts(graph, verdicts, original_edges)
        assert result.edges[0].source == "C"
        assert result.edges[0].target == "A"
        assert result.edges[0].weight == "8"

    def test_weight_correction_only(self):
        """Zoom verdict should update weight without changing direction."""
        graph = GraphSpec(
            nodes=[
                Node(name="B", x=0.6, y=0.4),
                Node(name="C", x=0.4, y=0.5),
            ],
            edges=[
                Edge(source="B", target="C", label="e_4", weight="6", directed=True),
            ],
            graph_type="directed",
        )
        verdicts = [{"source": "B", "target": "C", "weight": "12", "confidence": "high"}]
        original_edges = [graph.edges[0]]

        result = _apply_zoom_verdicts(graph, verdicts, original_edges)
        assert result.edges[0].source == "B"
        assert result.edges[0].target == "C"
        assert result.edges[0].weight == "12"

    def test_low_confidence_skipped(self):
        """Low-confidence verdicts should not be applied."""
        graph = GraphSpec(
            nodes=[
                Node(name="A", x=0.2, y=0.5),
                Node(name="B", x=0.8, y=0.5),
            ],
            edges=[
                Edge(source="A", target="B", weight="10", directed=True),
            ],
            graph_type="directed",
        )
        verdicts = [{"source": "B", "target": "A", "weight": "20", "confidence": "low"}]
        original_edges = [graph.edges[0]]

        result = _apply_zoom_verdicts(graph, verdicts, original_edges)
        # Should remain unchanged
        assert result.edges[0].source == "A"
        assert result.edges[0].target == "B"
        assert result.edges[0].weight == "10"

    def test_verdict_for_unverified_edge_ignored(self):
        """Verdicts for edges not in the original_edges list should be ignored."""
        graph = GraphSpec(
            nodes=[
                Node(name="A", x=0.2, y=0.5),
                Node(name="B", x=0.8, y=0.5),
                Node(name="C", x=0.5, y=0.9),
            ],
            edges=[
                Edge(source="A", target="B", weight="10", directed=True),
                Edge(source="B", target="C", weight="5", directed=True),
            ],
            graph_type="directed",
        )
        # Only A-B was zoom-verified, but verdict mentions B-C
        verdicts = [{"source": "C", "target": "B", "weight": "99", "confidence": "high"}]
        original_edges = [graph.edges[0]]  # only A-B

        result = _apply_zoom_verdicts(graph, verdicts, original_edges)
        # B→C should remain unchanged
        assert result.edges[1].source == "B"
        assert result.edges[1].target == "C"
        assert result.edges[1].weight == "5"

    def test_no_change_when_verdict_agrees(self):
        """No corrections when verdict matches current state."""
        graph = GraphSpec(
            nodes=[
                Node(name="S", x=0.1, y=0.5),
                Node(name="A", x=0.3, y=0.5),
            ],
            edges=[
                Edge(source="S", target="A", weight="10", directed=True),
            ],
            graph_type="directed",
        )
        verdicts = [{"source": "S", "target": "A", "weight": "10", "confidence": "high"}]
        original_edges = [graph.edges[0]]

        result = _apply_zoom_verdicts(graph, verdicts, original_edges)
        assert result.edges[0].source == "S"
        assert result.edges[0].target == "A"
        assert result.edges[0].weight == "10"

    def test_weight_none_keeps_original(self):
        """Verdict with weight='none' should keep the original weight."""
        graph = GraphSpec(
            nodes=[
                Node(name="A", x=0.2, y=0.5),
                Node(name="D", x=0.4, y=0.7),
            ],
            edges=[
                Edge(source="A", target="D", weight="10", directed=True),
            ],
            graph_type="directed",
        )
        verdicts = [{"source": "D", "target": "A", "weight": "none", "confidence": "high"}]
        original_edges = [graph.edges[0]]

        result = _apply_zoom_verdicts(graph, verdicts, original_edges)
        assert result.edges[0].source == "D"
        assert result.edges[0].target == "A"
        assert result.edges[0].weight == "10"  # kept from original

    def test_multiple_verdicts_applied(self):
        """Multiple verdicts for different edges should all be applied."""
        graph = GraphSpec(
            nodes=[
                Node(name="A", x=0.2, y=0.5),
                Node(name="B", x=0.6, y=0.4),
                Node(name="C", x=0.4, y=0.5),
            ],
            edges=[
                Edge(source="A", target="C", weight="6", directed=True),
                Edge(source="C", target="B", weight="6", directed=True),
            ],
            graph_type="directed",
        )
        verdicts = [
            {"source": "C", "target": "A", "weight": "8", "confidence": "high"},
            {"source": "B", "target": "C", "weight": "12", "confidence": "high"},
        ]
        original_edges = list(graph.edges)

        result = _apply_zoom_verdicts(graph, verdicts, original_edges)
        ac_edge = next(e for e in result.edges if "A" in (e.source, e.target) and "C" in (e.source, e.target))
        bc_edge = next(e for e in result.edges if "B" in (e.source, e.target) and "C" in (e.source, e.target))
        assert ac_edge.source == "C" and ac_edge.target == "A"
        assert bc_edge.source == "B" and bc_edge.target == "C"


# ---------------------------------------------------------------------------
# Decomposed pipeline: _compose_graph
# ---------------------------------------------------------------------------

class TestComposeGraph:
    """Tests for _compose_graph which merges topology, directions, and weights."""

    def test_basic_composition(self):
        """Compose a simple graph from topology + directions + weights."""
        nodes = [
            Node(name="A", x=0.1, y=0.2),
            Node(name="B", x=0.8, y=0.3),
            Node(name="C", x=0.5, y=0.8),
        ]
        topology = [
            {"node_a": "A", "node_b": "B", "label": "e_1"},
            {"node_a": "A", "node_b": "C", "label": "e_2"},
        ]
        directions = [
            {"source": "A", "target": "B", "label": "e_1", "directed": True, "confidence": "high"},
            {"source": "C", "target": "A", "label": "e_2", "directed": True, "confidence": "high"},
        ]
        weights = [
            {"source": "A", "target": "B", "label": "e_1", "weight": "10"},
            {"source": "C", "target": "A", "label": "e_2", "weight": "5"},
        ]
        graph = _compose_graph(nodes, topology, directions, weights, "plain")

        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert graph.weight_format == "plain"
        assert graph.graph_type == "directed"

        # Check edge directions and weights
        e1 = next(e for e in graph.edges if e.label == "e_1")
        assert e1.source == "A"
        assert e1.target == "B"
        assert e1.weight == "10"

        e2 = next(e for e in graph.edges if e.label == "e_2")
        assert e2.source == "C"
        assert e2.target == "A"
        assert e2.weight == "5"

    def test_composition_without_weights(self):
        """Compose a graph with no weights (unweighted graph)."""
        nodes = [
            Node(name="A", x=0.1, y=0.5),
            Node(name="B", x=0.9, y=0.5),
        ]
        topology = [
            {"node_a": "A", "node_b": "B", "label": None},
        ]
        directions = [
            {"source": "B", "target": "A", "label": None, "directed": True, "confidence": "high"},
        ]
        graph = _compose_graph(nodes, topology, directions, None, "plain")

        assert len(graph.edges) == 1
        assert graph.edges[0].source == "B"
        assert graph.edges[0].target == "A"
        assert graph.edges[0].weight is None

    def test_composition_undirected_graph(self):
        """Compose an undirected graph."""
        nodes = [
            Node(name="X", x=0.0, y=0.5),
            Node(name="Y", x=1.0, y=0.5),
        ]
        topology = [
            {"node_a": "X", "node_b": "Y", "label": None},
        ]
        directions = [
            {"source": "X", "target": "Y", "label": None, "directed": False, "confidence": "high"},
        ]
        graph = _compose_graph(nodes, topology, directions, None, "plain")

        assert graph.graph_type == "undirected"
        assert graph.edges[0].directed is False

    def test_composition_with_label_mismatch_fallback(self):
        """When label doesn't match between passes, fall back to pair matching."""
        nodes = [
            Node(name="A", x=0.1, y=0.5),
            Node(name="B", x=0.9, y=0.5),
        ]
        topology = [
            {"node_a": "A", "node_b": "B", "label": "e_1"},
        ]
        # Direction pass returned a different label (LLM inconsistency)
        directions = [
            {"source": "B", "target": "A", "label": "e_x", "directed": True, "confidence": "high"},
        ]
        graph = _compose_graph(nodes, topology, directions, None, "plain")

        # Should still get direction from fallback pair matching
        assert len(graph.edges) == 1
        assert graph.edges[0].source == "B"
        assert graph.edges[0].target == "A"

    def test_composition_no_direction_fallback(self):
        """When no direction info at all, fallback to alphabetical order."""
        nodes = [
            Node(name="M", x=0.5, y=0.5),
            Node(name="N", x=0.8, y=0.8),
        ]
        topology = [
            {"node_a": "M", "node_b": "N", "label": "e_1"},
        ]
        directions = []  # No direction info at all

        graph = _compose_graph(nodes, topology, directions, None, "plain")
        assert graph.edges[0].source == "M"
        assert graph.edges[0].target == "N"

    def test_composition_multiply_weight_format(self):
        """Compose with multiply weight format."""
        nodes = [
            Node(name="S", x=0.0, y=0.5),
            Node(name="T", x=1.0, y=0.5),
        ]
        topology = [{"node_a": "S", "node_b": "T", "label": "e_1"}]
        directions = [{"source": "S", "target": "T", "label": "e_1", "directed": True, "confidence": "high"}]
        weights = [{"source": "S", "target": "T", "label": "e_1", "weight": "10"}]

        graph = _compose_graph(nodes, topology, directions, weights, "multiply")
        assert graph.weight_format == "multiply"
        assert graph.edges[0].weight == "10"

    def test_composition_full_input1_topology(self):
        """Compose the full input1 ground truth graph from decomposed data."""
        nodes = [
            Node(name="A", x=0.22, y=0.25),
            Node(name="B", x=0.78, y=0.25),
            Node(name="C", x=0.47, y=0.50),
            Node(name="D", x=0.50, y=0.75),
            Node(name="S", x=0.0, y=0.25),
            Node(name="T", x=1.0, y=0.25),
        ]
        topology = [
            {"node_a": "A", "node_b": "S", "label": "e_1"},
            {"node_a": "A", "node_b": "B", "label": "e_5"},
            {"node_a": "B", "node_b": "T", "label": "e_2"},
            {"node_a": "A", "node_b": "C", "label": "e_6"},
            {"node_a": "B", "node_b": "C", "label": "e_4"},
            {"node_a": "B", "node_b": "D", "label": "e_3"},
            {"node_a": "C", "node_b": "D", "label": "e_9"},
            {"node_a": "A", "node_b": "D", "label": "e_8"},
        ]
        directions = [
            {"source": "S", "target": "A", "label": "e_1", "directed": True, "confidence": "high"},
            {"source": "A", "target": "B", "label": "e_5", "directed": True, "confidence": "high"},
            {"source": "B", "target": "T", "label": "e_2", "directed": True, "confidence": "high"},
            {"source": "C", "target": "A", "label": "e_6", "directed": True, "confidence": "high"},
            {"source": "B", "target": "C", "label": "e_4", "directed": True, "confidence": "high"},
            {"source": "B", "target": "D", "label": "e_3", "directed": True, "confidence": "high"},
            {"source": "C", "target": "D", "label": "e_9", "directed": True, "confidence": "high"},
            {"source": "D", "target": "A", "label": "e_8", "directed": True, "confidence": "high"},
        ]
        weights = [
            {"source": "S", "target": "A", "label": "e_1", "weight": "10"},
            {"source": "A", "target": "B", "label": "e_5", "weight": "32"},
            {"source": "B", "target": "T", "label": "e_2", "weight": "10"},
            {"source": "C", "target": "A", "label": "e_6", "weight": "8"},
            {"source": "B", "target": "C", "label": "e_4", "weight": "12"},
            {"source": "B", "target": "D", "label": "e_3", "weight": "9"},
            {"source": "C", "target": "D", "label": "e_9", "weight": "1"},
            {"source": "D", "target": "A", "label": "e_8", "weight": "10"},
        ]

        graph = _compose_graph(nodes, topology, directions, weights, "multiply")

        assert len(graph.nodes) == 6
        assert len(graph.edges) == 8
        assert graph.weight_format == "multiply"
        assert graph.graph_type == "directed"

        # Check all edges match ground truth
        expected = {
            "e_1": ("S", "A", "10"),
            "e_2": ("B", "T", "10"),
            "e_3": ("B", "D", "9"),
            "e_4": ("B", "C", "12"),
            "e_5": ("A", "B", "32"),
            "e_6": ("C", "A", "8"),
            "e_8": ("D", "A", "10"),
            "e_9": ("C", "D", "1"),
        }
        for edge in graph.edges:
            assert edge.label in expected, f"Unexpected edge label: {edge.label}"
            exp_src, exp_tgt, exp_w = expected[edge.label]
            assert edge.source == exp_src, f"{edge.label}: expected source {exp_src}, got {edge.source}"
            assert edge.target == exp_tgt, f"{edge.label}: expected target {exp_tgt}, got {edge.target}"
            assert edge.weight == exp_w, f"{edge.label}: expected weight {exp_w}, got {edge.weight}"

    def test_composition_preserves_all_edges(self):
        """All topology edges should appear in the output, even if directions/weights are incomplete."""
        nodes = [
            Node(name="A", x=0.0, y=0.0),
            Node(name="B", x=1.0, y=0.0),
            Node(name="C", x=0.5, y=1.0),
        ]
        topology = [
            {"node_a": "A", "node_b": "B", "label": "e_1"},
            {"node_a": "A", "node_b": "C", "label": "e_2"},
            {"node_a": "B", "node_b": "C", "label": "e_3"},
        ]
        # Only 2 out of 3 edges have direction info
        directions = [
            {"source": "A", "target": "B", "label": "e_1", "directed": True, "confidence": "high"},
            {"source": "C", "target": "A", "label": "e_2", "directed": True, "confidence": "high"},
        ]
        graph = _compose_graph(nodes, topology, directions, None, "plain")

        assert len(graph.edges) == 3
        # Third edge should fallback to alphabetical
        e3 = next(e for e in graph.edges if e.label == "e_3")
        assert e3.source == "B"
        assert e3.target == "C"

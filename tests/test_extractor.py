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
    _parse_json_response,
    _prompt_hash,
    _validate_graph,
    _load_and_encode_image,
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

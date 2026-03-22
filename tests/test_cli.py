"""Tests for graph2svg.cli — Click CLI integration tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from graph2svg.cli import main


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
GROUND_TRUTH = EXAMPLES_DIR / "input1_ground_truth.json"


@pytest.fixture
def runner():
    return CliRunner()


class TestCLI:
    def test_help_flag(self, runner: CliRunner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Convert a hand-drawn graph image" in result.output

    def test_from_json_to_svg(self, runner: CliRunner, tmp_path: Path):
        """Render from pre-existing JSON (bypasses LLM)."""
        output = tmp_path / "output.svg"
        result = runner.invoke(main, [
            str(EXAMPLES_DIR / "input1.png"),
            "--from-json", str(GROUND_TRUTH),
            "-o", str(output),
            "-f", "svg",
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output.exists()

    def test_from_json_to_pdf(self, runner: CliRunner, tmp_path: Path):
        output = tmp_path / "output.pdf"
        result = runner.invoke(main, [
            str(EXAMPLES_DIR / "input1.png"),
            "--from-json", str(GROUND_TRUTH),
            "-o", str(output),
            "-f", "pdf",
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output.exists()

    def test_from_json_to_png(self, runner: CliRunner, tmp_path: Path):
        output = tmp_path / "output.png"
        result = runner.invoke(main, [
            str(EXAMPLES_DIR / "input1.png"),
            "--from-json", str(GROUND_TRUTH),
            "-o", str(output),
            "-f", "png",
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output.exists()

    def test_from_json_with_style(self, runner: CliRunner, tmp_path: Path):
        output = tmp_path / "output.svg"
        result = runner.invoke(main, [
            str(EXAMPLES_DIR / "input1.png"),
            "--from-json", str(GROUND_TRUTH),
            "-o", str(output),
            "-s", "monochrome",
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output.exists()

    def test_from_json_with_layout(self, runner: CliRunner, tmp_path: Path):
        output = tmp_path / "output.svg"
        result = runner.invoke(main, [
            str(EXAMPLES_DIR / "input1.png"),
            "--from-json", str(GROUND_TRUTH),
            "-o", str(output),
            "--layout", "spring",
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output.exists()

    def test_json_export(self, runner: CliRunner, tmp_path: Path):
        output = tmp_path / "output.svg"
        json_out = tmp_path / "exported.json"
        result = runner.invoke(main, [
            str(EXAMPLES_DIR / "input1.png"),
            "--from-json", str(GROUND_TRUTH),
            "-o", str(output),
            "--json", str(json_out),
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert json_out.exists()
        # Verify exported JSON is loadable
        from graph2svg.model import GraphSpec
        graph = GraphSpec.from_json(json_out)
        assert len(graph.nodes) > 0

    def test_verbose_flag(self, runner: CliRunner, tmp_path: Path):
        output = tmp_path / "output.svg"
        result = runner.invoke(main, [
            str(EXAMPLES_DIR / "input1.png"),
            "--from-json", str(GROUND_TRUTH),
            "-o", str(output),
            "-v",
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"

    def test_infers_format_from_output_extension(self, runner: CliRunner, tmp_path: Path):
        output = tmp_path / "output.pdf"
        result = runner.invoke(main, [
            str(EXAMPLES_DIR / "input1.png"),
            "--from-json", str(GROUND_TRUTH),
            "-o", str(output),
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output.exists()

    def test_default_output_path(self, runner: CliRunner, tmp_path: Path):
        """Without -o, output should be input name with .svg extension."""
        # Copy ground truth to tmp_path to avoid polluting examples/
        import shutil
        input_copy = tmp_path / "test_input.png"
        shutil.copy2(EXAMPLES_DIR / "input1.png", input_copy)
        gt_copy = tmp_path / "gt.json"
        shutil.copy2(GROUND_TRUTH, gt_copy)

        result = runner.invoke(main, [
            str(input_copy),
            "--from-json", str(gt_copy),
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        expected_output = tmp_path / "test_input.svg"
        assert expected_output.exists()

    def test_nonexistent_input(self, runner: CliRunner):
        result = runner.invoke(main, ["/nonexistent/file.png"])
        assert result.exit_code != 0

    def test_nonexistent_from_json(self, runner: CliRunner):
        result = runner.invoke(main, [
            str(EXAMPLES_DIR / "input1.png"),
            "--from-json", "/nonexistent/graph.json",
        ])
        assert result.exit_code != 0

    def test_all_styles_available(self, runner: CliRunner):
        result = runner.invoke(main, ["--help"])
        for style in ["default", "monochrome", "minimal", "colorful"]:
            assert style in result.output

    def test_all_layouts_available(self, runner: CliRunner):
        result = runner.invoke(main, ["--help"])
        for layout in ["original", "spring", "kamada_kawai"]:
            assert layout in result.output

    def test_all_formats_available(self, runner: CliRunner):
        result = runner.invoke(main, ["--help"])
        for fmt in ["svg", "pdf", "png"]:
            assert fmt in result.output

    def test_without_github_token_and_no_from_json(self, runner: CliRunner, tmp_path: Path):
        """Without --from-json and without GITHUB_TOKEN, extract_graph should fail."""
        output = tmp_path / "output.svg"
        with patch.dict("os.environ", {"GITHUB_TOKEN": ""}, clear=False):
            result = runner.invoke(main, [
                str(EXAMPLES_DIR / "input1.png"),
                "-o", str(output),
            ])
            # Should fail because GITHUB_TOKEN is not set
            assert result.exit_code != 0

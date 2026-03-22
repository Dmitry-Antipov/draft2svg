"""Tests for graph2svg.styles — style presets and configuration."""

from __future__ import annotations

import pytest

from graph2svg.styles import (
    PALETTE_COLORFUL,
    PALETTE_MONOCHROME,
    PRESETS,
    StyleConfig,
    get_style,
)


class TestStyleConfig:
    def test_default_values(self):
        style = StyleConfig()
        assert style.node_radius == 0.035
        assert style.node_border_color == "#888888"
        assert style.node_fill_color == "#f5f5f5"
        assert style.edge_width == 3.0
        assert style.background_color == "white"
        assert style.figure_width == 16.0
        assert style.figure_dpi == 150
        assert style.color_palette == PALETTE_COLORFUL

    def test_custom_values(self):
        style = StyleConfig(
            node_radius=0.05,
            edge_width=5.0,
            background_color="#000000",
        )
        assert style.node_radius == 0.05
        assert style.edge_width == 5.0
        assert style.background_color == "#000000"

    def test_get_arrow_style_str(self):
        style = StyleConfig(arrow_head_length=16.0, arrow_head_width=10.0)
        result = style.get_arrow_style_str()
        assert "16.0" in result
        assert "10.0" in result

    def test_color_palette_independence(self):
        """Each StyleConfig instance should have its own palette list."""
        s1 = StyleConfig()
        s2 = StyleConfig()
        s1.color_palette.append("#000000")
        assert len(s2.color_palette) == len(PALETTE_COLORFUL)


class TestPalettes:
    def test_colorful_palette_has_colors(self):
        assert len(PALETTE_COLORFUL) == 12
        for color in PALETTE_COLORFUL:
            assert color.startswith("#")
            assert len(color) == 7  # #RRGGBB

    def test_monochrome_palette_has_colors(self):
        assert len(PALETTE_MONOCHROME) == 6
        for color in PALETTE_MONOCHROME:
            assert color.startswith("#")
            assert len(color) == 7


class TestGetStyle:
    def test_default_preset(self):
        style = get_style("default")
        assert isinstance(style, StyleConfig)

    def test_all_presets_exist(self):
        expected = {"default", "monochrome", "minimal", "colorful"}
        assert set(PRESETS.keys()) == expected

    def test_all_presets_loadable(self):
        for name in PRESETS:
            style = get_style(name)
            assert isinstance(style, StyleConfig)

    def test_monochrome_uses_monochrome_palette(self):
        style = get_style("monochrome")
        assert style.color_palette == PALETTE_MONOCHROME

    def test_minimal_has_smaller_sizes(self):
        default = get_style("default")
        minimal = get_style("minimal")
        assert minimal.node_radius < default.node_radius
        assert minimal.edge_width < default.edge_width
        assert minimal.node_font_size < default.node_font_size

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown style"):
            get_style("nonexistent")

    def test_unknown_preset_lists_available(self):
        with pytest.raises(ValueError, match="default"):
            get_style("bad_name")

    def test_default_no_args(self):
        style = get_style()
        assert style == PRESETS["default"]

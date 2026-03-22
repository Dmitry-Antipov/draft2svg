"""Visual style presets for graph rendering."""

from __future__ import annotations

from dataclasses import dataclass, field


# Curated color palettes for edge coloring
PALETTE_COLORFUL = [
    "#2ca02c",  # green
    "#e91e90",  # pink/magenta
    "#ff8c00",  # orange
    "#00bcd4",  # cyan
    "#888888",  # gray
    "#555555",  # dark gray
    "#9467bd",  # purple
    "#d62728",  # red
    "#1f77b4",  # blue
    "#bcbd22",  # olive
    "#17becf",  # teal
    "#8c564b",  # brown
]

PALETTE_MONOCHROME = [
    "#000000",
    "#333333",
    "#555555",
    "#777777",
    "#999999",
    "#bbbbbb",
]


@dataclass
class StyleConfig:
    """Visual configuration for graph rendering."""

    # Node styling
    node_radius: float = 0.035
    node_border_color: str = "#888888"
    node_fill_color: str = "#f5f5f5"
    node_border_width: float = 2.0
    node_font_size: int = 18
    node_font_style: str = "italic"
    node_font_weight: str = "normal"

    # Edge styling
    edge_width: float = 3.0
    arrow_head_length: float = 16.0
    arrow_head_width: float = 10.0
    arrow_style: str = "->,head_length={head_length},head_width={head_width}"

    # Edge label styling
    edge_label_font_size: int = 16
    weight_font_size: int = 18
    label_offset: float = 0.02  # perpendicular offset from edge

    # Color palette for auto-assigning edge colors
    color_palette: list[str] = field(default_factory=lambda: list(PALETTE_COLORFUL))

    # Figure styling
    background_color: str = "white"
    figure_width: float = 16.0
    figure_height: float = 10.0
    figure_dpi: int = 150
    figure_padding: float = 0.1

    def get_arrow_style_str(self) -> str:
        """Get the formatted arrow style string."""
        return self.arrow_style.format(
            head_length=self.arrow_head_length,
            head_width=self.arrow_head_width,
        )


# Preset styles
PRESETS: dict[str, StyleConfig] = {
    "default": StyleConfig(),
    "monochrome": StyleConfig(
        node_border_color="#000000",
        node_fill_color="#ffffff",
        edge_width=2.5,
        color_palette=list(PALETTE_MONOCHROME),
    ),
    "minimal": StyleConfig(
        node_radius=0.025,
        node_border_width=1.5,
        node_font_size=14,
        edge_width=1.5,
        arrow_head_length=8.0,
        arrow_head_width=5.0,
        edge_label_font_size=12,
        weight_font_size=14,
    ),
    "colorful": StyleConfig(
        node_border_color="#666666",
        node_fill_color="#e8e8e8",
        edge_width=3.5,
        node_font_size=20,
        weight_font_size=20,
    ),
}


def get_style(name: str = "default") -> StyleConfig:
    """Get a style configuration by preset name."""
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown style '{name}'. Available: {available}")
    return PRESETS[name]

"""Command-line interface for graph2svg."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .extractor import extract_graph
from .layout import apply_layout
from .model import GraphSpec
from .renderer import render_graph
from .styles import PRESETS, get_style


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(),
    default=None,
    help="Output file path. Default: input name with .svg extension.",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice(["svg", "pdf", "png"]),
    default=None,
    help="Output format. Default: inferred from output extension, or svg.",
)
@click.option(
    "-m",
    "--model",
    default="gpt-4.1",
    show_default=True,
    help="GitHub Models model name for vision extraction.",
)
@click.option(
    "-s",
    "--style",
    "style_name",
    type=click.Choice(list(PRESETS.keys())),
    default="default",
    show_default=True,
    help="Visual style preset.",
)
@click.option(
    "--json",
    "json_path",
    type=click.Path(),
    default=None,
    help="Save intermediate JSON graph representation to this path.",
)
@click.option(
    "--from-json",
    "from_json_path",
    type=click.Path(exists=True),
    default=None,
    help="Skip vision extraction; load graph from a JSON file.",
)
@click.option(
    "--layout",
    "layout_algo",
    type=click.Choice(["original", "spring", "kamada_kawai"]),
    default="original",
    show_default=True,
    help="Layout algorithm for node positioning.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose output.",
)
@click.option(
    "--no-verify",
    is_flag=True,
    default=False,
    help="Skip the LLM verification pass (faster, but may miss errors).",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Bypass the LLM response cache (always call the API).",
)
def main(
    input_path: str,
    output_path: str | None,
    output_format: str | None,
    model: str,
    style_name: str,
    json_path: str | None,
    from_json_path: str | None,
    layout_algo: str,
    verbose: bool,
    no_verify: bool,
    no_cache: bool,
) -> None:
    """Convert a hand-drawn graph image to publication-quality vector graphics.

    INPUT_PATH is the path to the input image (jpg, png, etc.) or, when used
    with --from-json, any placeholder path.
    """
    # Determine output path
    if output_path is None:
        inp = Path(input_path)
        ext = f".{output_format}" if output_format else ".svg"
        output_path = str(inp.with_suffix(ext))
    elif output_format is None:
        # Infer format from output extension
        ext = Path(output_path).suffix.lstrip(".").lower()
        if ext in ("svg", "pdf", "png"):
            output_format = ext

    if output_format is None:
        output_format = "svg"

    # Load style
    style = get_style(style_name)

    # Step 1: Extract or load graph
    if from_json_path:
        if verbose:
            click.echo(f"Loading graph from {from_json_path}...", err=True)
        graph = GraphSpec.from_json(from_json_path)
    else:
        click.echo("Analyzing image with LLM vision...", err=True)
        graph = extract_graph(
            image_path=input_path,
            model=model,
            verbose=verbose,
            verify=not no_verify,
            use_cache=not no_cache,
        )

    # Save JSON if requested
    if json_path:
        graph.to_json(json_path)
        if verbose:
            click.echo(f"Saved graph JSON to {json_path}", err=True)

    # Step 2: Apply layout
    if verbose:
        click.echo(f"Applying layout: {layout_algo}...", err=True)
    graph = apply_layout(graph, algorithm=layout_algo)

    # Step 3: Render
    if verbose:
        click.echo(f"Rendering to {output_format}...", err=True)

    out = render_graph(
        graph=graph,
        output_path=output_path,
        style=style,
        output_format=output_format,
    )

    click.echo(f"Output saved to {out}", err=True)


if __name__ == "__main__":
    main()

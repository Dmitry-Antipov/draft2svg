# graph2svg

Convert photographs of hand-drawn graph diagrams into publication-quality
vector graphics (SVG, PDF, PNG).

Point `graph2svg` at a whiteboard photo or notebook scan, and it uses LLM
vision to extract the graph structure — nodes, edges, labels, weights,
curvatures — then renders a clean, publication-ready figure with matplotlib.

## Quick start

```bash
# Install
pip install -e .

# Set up GitHub token (needs Models API access)
export GITHUB_TOKEN="ghp_..."

# Convert a photo to SVG
graph2svg photo.jpg -o graph.svg

# Convert to PDF with verbose output
graph2svg photo.jpg -o graph.pdf -v

# Save intermediate JSON for manual editing
graph2svg photo.jpg -o graph.svg --json graph.json

# Render from edited JSON (skip LLM)
graph2svg photo.jpg -o graph.svg --from-json graph.json
```

## Requirements

- Python 3.10+
- A GitHub Personal Access Token with Models API access
  ([create one here](https://github.com/settings/tokens?type=beta))

Set the token as an environment variable:

```bash
export GITHUB_TOKEN="ghp_your_token_here"
```

## Installation

```bash
git clone <repo-url>
cd jpg_to_svg
pip install -e .
```

This installs the `graph2svg` command.

## Usage

```
graph2svg [OPTIONS] INPUT_PATH
```

| Option | Description |
|--------|-------------|
| `-o, --output PATH` | Output file path (default: input name with `.svg`) |
| `-f, --format [svg\|pdf\|png]` | Output format (default: inferred from extension) |
| `-m, --model TEXT` | LLM model name (default: `gpt-4.1`) |
| `-s, --style [default\|monochrome\|minimal\|colorful]` | Visual style preset |
| `--json PATH` | Save the intermediate JSON graph representation |
| `--from-json PATH` | Skip LLM extraction, render from a JSON file |
| `--layout [original\|spring\|kamada_kawai]` | Layout algorithm (default: `original`) |
| `-v, --verbose` | Print detailed progress to stderr |
| `--no-verify` | Skip the LLM verification pass (faster) |

### Examples

**Basic conversion:**
```bash
graph2svg whiteboard.jpg -o network.svg
```

**PDF output with monochrome style:**
```bash
graph2svg whiteboard.jpg -o network.pdf -s monochrome
```

**Extract JSON, edit it, re-render:**
```bash
# Step 1: Extract and save JSON
graph2svg whiteboard.jpg -o draft.png --json graph.json -v

# Step 2: Edit graph.json to fix any errors
# (fix wrong edges, adjust positions, etc.)

# Step 3: Render from corrected JSON
graph2svg whiteboard.jpg -o final.svg --from-json graph.json
```

**Use a different layout algorithm:**
```bash
graph2svg whiteboard.jpg -o graph.svg --layout spring
```

## How it works

1. **Extract** — The input image is sent to an LLM (GPT-4.1 via GitHub
   Models API) with a structured chain-of-thought prompt. The LLM identifies
   nodes, traces edges, reads labels and weights, and produces a JSON graph
   specification.

2. **Verify** — A second LLM call checks the extraction against the image,
   looking for missed edges, wrong directions, phantom edges, and label
   mismatches. This step can be skipped with `--no-verify`.

3. **Validate** — Programmatic checks remove obvious errors: self-loops
   (almost always hallucinated), duplicate edges, and references to
   non-existent nodes.

4. **Layout** — Node positions from the LLM are normalized to fill the
   canvas. Curvatures are assigned for parallel edges, and edges are routed
   around intermediate nodes. Alternatively, NetworkX layouts (spring,
   Kamada-Kawai) can replace the LLM positions entirely.

5. **Render** — Matplotlib produces the final output with proper arrow
   clipping at node boundaries, curved edges, rotated labels, and
   publication-quality typography.

## JSON format

The intermediate JSON can be saved with `--json` and edited manually.
Structure:

```json
{
  "nodes": [
    {"name": "S", "x": 0.0, "y": 0.25, "shape": "circle"},
    {"name": "A", "x": 0.22, "y": 0.25, "shape": "circle"}
  ],
  "edges": [
    {
      "source": "S",
      "target": "A",
      "label": "e_1",
      "weight": "10",
      "directed": true,
      "color": null,
      "curvature": 0.0
    }
  ],
  "title": null,
  "graph_type": "directed",
  "weight_format": "plain"
}
```

**Fields:**

- **Nodes:** `name` (display label), `x`/`y` (0–1 range, top-left origin),
  `shape` (`circle`, `square`, `diamond`)
- **Edges:** `source`/`target` (node names), `label` (use `e_1` for e₁),
  `weight` (bare number as string), `directed` (bool), `color` (hex or null
  for auto), `curvature` (0 = straight, positive = curves left when facing
  target, negative = curves right)
- **weight_format:** `"plain"` (bare numbers) or `"multiply"` (appends ×)
- **graph_type:** `"directed"` or `"undirected"`

## Style presets

| Preset | Description |
|--------|-------------|
| `default` | Colorful edges, light gray nodes, clean sans-serif labels |
| `monochrome` | Grayscale palette, black borders, white node fill |
| `minimal` | Smaller nodes and fonts, thin lines |
| `colorful` | Bolder colors, thicker edges, larger labels |

## Tips

- **Messy drawings:** The LLM works best on drawings with clearly separated
  nodes and edges. Dense areas where many edges overlap will produce errors.
  Use the `--json` workflow to fix these manually.

- **Edge labels:** If your drawing uses subscript notation (e₁, x₁), the
  JSON will represent these as `e_1`, `x_1`, and the renderer will display
  them with proper subscripts.

- **Weight format:** The tool auto-detects whether weights are plain numbers
  or use a multiply notation (10×). You can override this in the JSON.

- **Curvature:** Positive curvature curves the edge to the left (when
  standing at the source facing the target). Negative curves right. Use
  opposite signs for parallel edges between the same nodes.

## License

MIT

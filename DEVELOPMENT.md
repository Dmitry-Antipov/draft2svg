# Development Guide

Detailed documentation for developers working on `graph2svg`.

## Architecture

```
Input Image (.jpg/.png)
        │
        ▼
  ┌─────────────┐     LLM Vision API
  │ extractor.py │     (GitHub Models / GPT-4.1)
  │              │     Pass 1: Chain-of-thought extraction
  │  SYSTEM_PROMPT ──► Pass 2: Verification against image
  │  VERIFY_PROMPT     Pass 3: Programmatic validation
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐     Pydantic v2 models
  │  model.py   │     GraphSpec ← Node[] + Edge[]
  │             │     JSON serialization / deserialization
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐     Position refinement
  │  layout.py  │     Normalization, curvature assignment,
  │             │     crossing detection, NetworkX fallback
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐     Matplotlib rendering
  │ renderer.py │     Bezier curves, arrow clipping,
  │             │     label placement, SVG/PDF/PNG output
  └──────┬──────┘
         │
         ▼
  Output (.svg/.pdf/.png)
```

The pipeline is strictly sequential: extract → layout → render. The
intermediate format (`GraphSpec` JSON) is the contract between stages —
any stage can be replaced or bypassed independently.

## Module Reference

### `model.py` — Data Models

The central data interchange format. Three Pydantic `BaseModel` classes.

#### `Node`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `name` | `str` | required | Case-sensitive display label |
| `x` | `float` | required | 0.0 = left, 1.0 = right |
| `y` | `float` | required | 0.0 = top, 1.0 = bottom (y-down) |
| `shape` | `str` | `"circle"` | Only `circle` is rendered currently |

#### `Edge`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `source` | `str` | required | Must match a node name exactly |
| `target` | `str` | required | Must match a node name exactly |
| `label` | `str?` | `None` | Subscript notation: `e_1` renders as e₁ |
| `weight` | `str?` | `None` | Bare number string, no units |
| `directed` | `bool` | `True` | Controls arrowhead rendering |
| `color` | `str?` | `None` | Hex color; auto-assigned from palette if null |
| `curvature` | `float` | `0.0` | See curvature section below |

#### `GraphSpec`

| Field | Type | Default |
|-------|------|---------|
| `nodes` | `list[Node]` | required |
| `edges` | `list[Edge]` | required |
| `title` | `str?` | `None` |
| `graph_type` | `str` | `"directed"` |
| `weight_format` | `str` | `"plain"` |

Methods: `to_json(path)`, `from_json(path)` (classmethod).

### `extractor.py` — LLM Vision Extraction

The largest module (~525 lines). Handles the full LLM interaction pipeline.

#### Constants

| Name | Value | Purpose |
|------|-------|---------|
| `MAX_IMAGE_DIM` | `2048` | Images larger than this are downscaled |
| `SYSTEM_PROMPT` | ~130 lines | Chain-of-thought extraction prompt |
| `VERIFY_PROMPT` | ~35 lines | Verification/correction prompt |

#### Extraction Pipeline

```
extract_graph(image_path, model, verify, verbose)
    │
    ├── _load_and_encode_image()     Resize + JPEG encode + base64
    │
    ├── Pass 1: _call_llm()          SYSTEM_PROMPT + image
    │   └── _parse_json_response()   Extract JSON from CoT response
    │   └── GraphSpec.model_validate()
    │   └── Retry loop (max 3, exponential backoff on 429)
    │
    ├── Pass 2: _call_llm()          VERIFY_PROMPT + initial JSON + image
    │   └── _parse_json_response()
    │   └── GraphSpec.model_validate()
    │   └── Falls back to Pass 1 result on failure
    │
    └── _validate_graph()            Programmatic sanity checks
        ├── Remove self-loops
        ├── Check node references
        ├── Remove duplicates
        ├── Warn about isolated nodes
        └── Warn about duplicate labels
```

#### Chain-of-Thought Prompt Design

The system prompt instructs the LLM to work through 6 steps, writing out
its reasoning before producing JSON:

1. **Identify nodes** — list names and (x,y) positions
2. **Count edge labels** — enumerate visible labels (e₁, e₂, ...) to set
   expected edge count; warns about gaps in numbering
3. **Trace edges** — for each label, physically follow the line; detailed
   rules for messy drawings (don't assume nearest node, handle crossing
   lines, long arcs)
4. **Assign curvatures** — magnitude and sign convention
5. **Weight format / graph type** — plain vs multiply, directed vs undirected
6. **Output JSON** — in a fenced code block

This chain-of-thought approach dramatically improves accuracy compared to
a "output only JSON" prompt, because the LLM reasons through ambiguous
regions before committing to the final answer.

#### JSON Parser (`_parse_json_response`)

Handles three response formats (tried in order):

1. ` ```json ... ``` ` fenced code block — uses the **last** match
2. Last top-level `{ ... }` that parses as valid JSON (walks backwards
   through text tracking brace depth)
3. Entire response as raw JSON (fallback)

This is necessary because chain-of-thought responses contain analysis text
before the JSON block.

#### Validation (`_validate_graph`)

Programmatic checks that don't require the image:

| Check | Action |
|-------|--------|
| Self-loops (`source == target`) | Remove (almost always hallucinated) |
| Unknown node references | Warn |
| Duplicate edges (same source+target+label) | Remove |
| Isolated nodes (no edges) | Warn |
| Duplicate labels (same label on different edges) | Warn |

#### API Configuration

- **Endpoint:** `https://models.inference.ai.azure.com` (GitHub Models)
- **Auth:** `GITHUB_TOKEN` environment variable
- **Temperature:** 0.1 (low, for deterministic extraction)
- **Max tokens:** 4096 (enough for chain-of-thought + JSON)
- **Image detail:** `"high"` for maximum vision accuracy

### `layout.py` — Layout Engine

Refines node positions and computes edge routing.

#### `apply_layout(graph, algorithm, padding)`

Entry point. Dispatches to the selected algorithm, then runs curvature
assignment and crossing detection.

| Algorithm | Implementation | Notes |
|-----------|---------------|-------|
| `original` | `normalize_positions()` | Keeps LLM positions, scales to fill canvas |
| `spring` | `nx.spring_layout(seed=42)` | Force-directed; ignores LLM positions |
| `kamada_kawai` | `nx.kamada_kawai_layout()` | Energy minimization; ignores LLM positions |

#### `normalize_positions(graph, padding=0.1)`

Scales node coordinates to fill `[padding, 1-padding]` range on both axes.
Preserves relative layout from LLM. Handles degenerate cases (all nodes on
same row/column).

#### `assign_curvatures(graph)`

Groups edges by unordered node pair:
- **1 edge:** keeps existing curvature
- **2 edges:** assigns `+0.3` / `-0.3`
- **3+ edges:** spreads linearly from `-0.4` to `+0.4`

#### `_detect_crossings(graph)`

For straight edges (curvature ≈ 0), checks if any non-endpoint node is
within distance 0.06 of the edge line. If so, adds curvature 0.25 to route
around the obstruction.

### `renderer.py` — Matplotlib Renderer

Renders the final output. ~430 lines.

#### Key Constants

| Name | Value | Purpose |
|------|-------|---------|
| `COORD_SCALE` | `800` | Positions (0-1) scaled to this range for matplotlib compatibility |
| `MIN_ASPECT` | `2.0` | Minimum width:height ratio (enforces landscape output) |

#### Coordinate System

The renderer uses a **y-down** coordinate system throughout:
- `(0, 0)` = top-left
- y-axis is inverted via `ax.set_ylim(y_max, y_min)`
- All positions are multiplied by `COORD_SCALE` (800)

This matches image coordinates and the LLM's position convention.

#### `render_graph(graph, output_path, style, output_format)`

Main rendering function. Flow:

1. Scale positions by `COORD_SCALE`
2. Compute plot bounds including Bezier curve extents (sampled at 7 t-values)
3. Enforce minimum aspect ratio (widen x-range if needed)
4. Create figure with `aspect="equal"`, zero padding
5. Draw nodes as `matplotlib.patches.Circle`
6. Draw edges via `_draw_edge()`
7. Save with `bbox_inches="tight"`, `pad_inches=0.15`

#### `_draw_edge(ax, edge, pos, node_patches, weight_format, style, node_radius)`

Draws a single edge. Key implementation details:

**Arrow rendering:**
- Uses `FancyArrowPatch` with `arrowstyle="-|>"` (filled triangular arrowhead)
- `connectionstyle="arc3,rad={curvature}"` for curved edges
- `patchA`/`patchB` set to the actual node `Circle` patches — this tells
  matplotlib to clip the arrow at the node boundary (not at the node center)
- `mutation_scale=1` for correctly sized arrowheads

**Label placement:**
- `_best_label_t()` finds a t-parameter along the Bezier curve that doesn't
  collide with any intermediate node (clearance = 2× node radius)
- Labels rotate to follow the edge tangent angle (computed via
  `_bezier_tangent()`), normalized to `[-90°, 90°]` so text is always
  right-side-up
- Weight and label **straddle** the edge: weight on the outer (convex) side,
  label on the inner side, offset by `node_radius * 0.85`

**Mathtext conversion (`_label_to_mathtext`):**
- `"e_1"` → `"$e_1$"` (proper subscript in matplotlib)
- `"e_10"` → `"$e_{10}$"` (multi-digit subscript needs braces)

### `styles.py` — Visual Style Configuration

`StyleConfig` is a `@dataclass` with ~20 visual parameters. Four presets
are available:

| Preset | Key characteristics |
|--------|-------------------|
| `default` | 12-color palette, gray node borders, italic labels |
| `monochrome` | 6-shade grayscale, black borders, white fill |
| `minimal` | Smaller everything — nodes, fonts, arrows, lines |
| `colorful` | Bolder — thicker edges, larger fonts, darker borders |

To add a new preset, add an entry to the `PRESETS` dict in `styles.py`.

### `cli.py` — Click CLI

Thin orchestration layer connecting the pipeline stages. All output goes to
stderr via `click.echo(..., err=True)` — stdout is reserved for potential
piping.

## Curvature Convention

Understanding the curvature sign is critical for correct edge rendering.

**Coordinate system:** y-down (0,0 = top-left).

**Sign rule:** Stand at the source node, face the target node.
- **Positive curvature** → edge curves to your **left**
- **Negative curvature** → edge curves to your **right**

**Implementation:** The Bezier control point is offset from the midpoint by
`curvature * edge_length` in the perpendicular direction. The perpendicular
is computed as `(-dy/len, dx/len)` where `(dx, dy)` is the source-to-target
vector.

**Examples (y-down coords):**

| Edge direction | Positive curvature | Negative curvature |
|---------------|-------------------|-------------------|
| Left to right (→) | Curves upward | Curves downward |
| Right to left (←) | Curves downward | Curves upward |
| Top to bottom (↓) | Curves rightward | Curves leftward |
| Bottom to top (↑) | Curves leftward | Curves rightward |

**Parallel edges:** For two edges between the same pair of nodes, use
opposite-sign curvatures (e.g., `+0.3` and `-0.3`) so they bow away from
each other.

## Known Limitations

### LLM Extraction Accuracy

The LLM (GPT-4.1 via GitHub Models) achieves roughly 50-70% edge accuracy
on messy whiteboard photos. Common failure modes:

- **Dense areas:** When many edges converge at one node, the LLM struggles
  to trace individual lines correctly.
- **Phantom edges:** The LLM may invent edges that don't exist (e.g.,
  imagining an e₇ when the drawing skips from e₆ to e₈).
- **Direction reversal:** The LLM sometimes gets arrowhead direction wrong,
  especially for edges that approach a node from an unusual angle.
- **Weight/label swaps:** Labels and weights near dense intersections may be
  assigned to the wrong edge.

**Mitigations:**
- The verification pass catches ~30% of errors (especially phantom edges)
- Programmatic validation removes self-loops and duplicates
- The `--json` workflow lets users fix remaining errors manually

### Rendering

- Only `circle` node shape is implemented (`square`, `diamond` are parsed
  but rendered as circles)
- `bbox_inches="tight"` can slightly reduce the enforced aspect ratio
- Label overlap is possible in dense graphs (no global label de-collision)

## Development Workflow

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export GITHUB_TOKEN="ghp_..."
```

### Testing Extraction

```bash
# Full pipeline with verbose output
graph2svg examples/input1.png -o /tmp/test.png --json /tmp/test.json -v

# Compare JSON against ground truth
diff /tmp/test.json examples/input1_ground_truth.json

# Render from ground truth (bypass LLM)
graph2svg examples/input1.png -o /tmp/reference.png \
  --from-json examples/input1_ground_truth.json
```

### Testing Rendering Only

```bash
# Render from any JSON file (no LLM call)
graph2svg dummy -o output.svg --from-json graph.json

# Try different styles
graph2svg dummy -o mono.svg --from-json graph.json -s monochrome
graph2svg dummy -o minimal.svg --from-json graph.json -s minimal
```

### Adding a New Style Preset

1. Add an entry to `PRESETS` in `styles.py`:

```python
PRESETS["my_style"] = StyleConfig(
    node_radius=0.04,
    edge_width=2.0,
    color_palette=["#ff0000", "#0000ff"],
    # ... override any defaults
)
```

2. The CLI automatically picks it up (populated from `PRESETS.keys()`).

### Adding a New Layout Algorithm

1. Add a branch in `_networkx_layout()` in `layout.py`:

```python
elif algorithm == "circular":
    raw = nx.circular_layout(G)
```

2. Add it to the `apply_layout()` dispatch.
3. Add it to the `--layout` CLI choice list in `cli.py`.

### Improving the LLM Prompt

The system prompt is in `extractor.py` as `SYSTEM_PROMPT`. Key principles:

- **Chain-of-thought:** Let the LLM reason step-by-step before outputting
  JSON. This is critical for accuracy on messy images.
- **Edge label counting:** Force the LLM to count visible labels first,
  establishing an expected edge count.
- **Node degree verification:** After tracing edges, the LLM should check
  that each node's degree matches the number of lines touching it.
- **Don't force JSON-only output:** Allowing reasoning text before the JSON
  block dramatically improves accuracy. The parser handles mixed responses.

### Extending to New Graph Types

Currently focused on directed network/flow graphs. To support new types:

1. Add any new fields to `Node` / `Edge` / `GraphSpec` in `model.py`
2. Update `SYSTEM_PROMPT` in `extractor.py` to describe the new graph type
3. Update rendering in `renderer.py` (e.g., new node shapes, edge styles)
4. Update layout in `layout.py` if needed

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `pyproject.toml` | — | Package metadata, dependencies, entry point |
| `graph2svg/__init__.py` | 3 | Package init, version |
| `graph2svg/model.py` | 76 | Pydantic data models (Node, Edge, GraphSpec) |
| `graph2svg/styles.py` | 110 | StyleConfig dataclass, 4 presets, color palettes |
| `graph2svg/extractor.py` | 525 | LLM extraction, verification, validation |
| `graph2svg/layout.py` | 196 | Position normalization, curvature, NetworkX layouts |
| `graph2svg/renderer.py` | 430 | Matplotlib rendering, Bezier math, label placement |
| `graph2svg/cli.py` | 160 | Click CLI, pipeline orchestration |
| `examples/input1.png` | — | Sample whiteboard photo |
| `examples/input1_ground_truth.json` | 23 | Hand-verified ground truth |
| `examples/output1.pdf` | — | Reference rendered output |

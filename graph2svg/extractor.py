"""LLM vision-based graph extraction from hand-drawn images."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image

from .model import Edge, ExtractionResponse, GraphSpec, Node


# ---------------------------------------------------------------------------
# Response caching
# ---------------------------------------------------------------------------

def _cache_dir() -> Path:
    """Return the cache directory, creating it if needed."""
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = base / "graph2svg"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(image_path: str, model: str, prompt_hash: str, pass_name: str) -> str:
    """Build a deterministic cache key from image content + params.

    Components:
    - SHA-256 of the raw image file bytes
    - model name
    - hash of the system prompt (so prompt edits bust the cache)
    - pass_name ("extract" or "verify")
    """
    h = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    h.update(model.encode())
    h.update(prompt_hash.encode())
    h.update(pass_name.encode())
    return h.hexdigest()


def _prompt_hash(prompt_text: str) -> str:
    """Return a short hash of a prompt string."""
    return hashlib.sha256(prompt_text.encode()).hexdigest()[:16]


def _cache_get(key: str) -> str | None:
    """Look up a cached LLM response. Returns the raw text or None."""
    path = _cache_dir() / f"{key}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return data.get("raw_text")
        except (json.JSONDecodeError, KeyError):
            return None
    return None


def _cache_put(key: str, raw_text: str) -> None:
    """Store a raw LLM response in the cache."""
    path = _cache_dir() / f"{key}.json"
    path.write_text(json.dumps({
        "raw_text": raw_text,
        "timestamp": time.time(),
    }, indent=2))

# Maximum image dimension to send to the API (to stay within token limits)
MAX_IMAGE_DIM = 2048

SYSTEM_PROMPT = """\
You are an expert at analyzing photographs of hand-drawn graphs and network
diagrams on whiteboards, paper, or notebooks. Your task is to extract the
COMPLETE graph structure and produce a JSON representation.

Hand-drawn graphs are often messy: edges overlap, labels crowd together, and
handwriting is imperfect. You must be very careful and methodical.

## Step-by-step process

Work through these steps IN ORDER. Write out your analysis for each step
before producing the final JSON.

### Step 1 — Identify ALL nodes

Scan the entire image for labeled circles, dots, or named intersection points.
List each node with:
- Its exact name (case-sensitive, as written)
- Its (x, y) position as a fraction of the IMAGE dimensions (not the graph
  bounding box): (0, 0) = top-left corner of the image, (1, 1) = bottom-right
  corner of the image.

**Position accuracy is critical.** The rendered output will use these coordinates
directly, so the distances between nodes must match what you see in the drawing.
Follow this procedure:
1. First, identify the pixel-level position of each node relative to the full
   image (top-left = 0,0; bottom-right = 1,1).
2. Nodes that are close together in the drawing MUST have coordinates that are
   close together. If two nodes are 10% of the image apart, their coordinates
   should differ by ~0.10 — do NOT spread them further apart.
3. Verify your positions: compare the distance ratios between pairs of nodes in
   your coordinate list with the visual distances in the image. For example, if
   node A is twice as far from node C as it is from node B in the drawing, that
   ratio must hold in your coordinates.

Write: "Nodes found: [list them with positions]"

### Step 2 — Count edge labels

Before tracing edges, count ALL edge labels visible in the image (e.g. e₁, e₂,
e₃, ...). Write them out. This tells you exactly how many edges to find.

IMPORTANT: Only count labels that are actually written in the image. Edge label
sequences may have gaps (e.g. e₁, e₂, e₃, e₄, e₅, e₆, e₈, e₉ — skipping e₇).
Do NOT assume the sequence is contiguous. Only count what you can see.

Write: "Edge labels visible: [list them]. Total: N edges expected."

### Step 3 — Trace EVERY edge, one by one

This is the most critical step. For EACH edge label you found in Step 2,
find the line it labels and trace that line from one node to the other.

For each edge write out:
- "e_X: [source] → [target], weight=[W], curvature=[describe curve]"

**Tracing rules:**
- Follow each line physically from endpoint to endpoint. Do NOT assume a line
  goes to the nearest node — it may loop around or cross other edges.
- Arrowheads indicate direction. The arrowhead points at the TARGET.
- Labels and weights are usually written alongside the midpoint of the edge
  they belong to. Match each label to the edge whose line it sits next to.
- If two edges connect the same pair of nodes, they curve in opposite
  directions.
- Pay extra attention to dense areas where multiple edges meet at one node.
  Each line entering/leaving a node is a separate edge.
- Long curved arcs may travel far from the straight-line path between nodes.
  An edge labeled far from both endpoints still connects those endpoints —
  trace the line, don't just look at label proximity.

After listing all edges, verify:
- Does your edge count match the label count from Step 2?
- For each node, count the lines touching it in the image and compare with
  your edge list. Write this check out explicitly.
- If a node has a different degree than expected, re-trace its edges.

### Step 4 — Curvature values

For each edge, assign a numeric curvature:
- 0.0 = straight line
- 0.1–0.2 = slight curve
- 0.3–0.5 = strong curve (wide arc)

Sign convention: imagine standing at the source node looking toward the target.
- Edge curves to your LEFT → positive curvature
- Edge curves to your RIGHT → negative curvature

### Step 5 — Weight format & graph type

- If weights are plain numbers (10, 32) → weight_format = "plain"
- If weights have ×/x suffix (10×, 32x) → weight_format = "multiply"
- If edges have arrowheads → graph_type = "directed"

### Step 6 — Output JSON

After completing your analysis, output the final JSON inside a code fence:

```json
{
  "nodes": [
    {"name": "S", "x": 0.0, "y": 0.5, "shape": "circle"}
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

Field notes:
- "shape": always "circle" unless clearly different
- "color": always null
- "label": use underscore for subscripts (e₁ → "e_1"); null if unlabeled
- "weight": bare number as string ("10"); null if no weight visible
- "title": null unless drawing has an explicit title
"""


def _load_and_encode_image(image_path: str) -> tuple[str, str]:
    """Load an image, resize if needed, and return as base64 + media type."""
    img = Image.open(image_path)

    # Convert to RGB if needed (e.g. RGBA, palette mode)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Resize if too large
    max_dim = max(img.size)
    if max_dim > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max_dim
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Encode as JPEG for efficiency
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return b64_data, "image/jpeg"


VERIFY_PROMPT = """\
You are verifying a graph extraction from a hand-drawn image. Below is the JSON
that was extracted. Compare it carefully against the image.

Perform these checks:

1. COUNT: Count the distinct edge labels visible in the image (e_1, e_2, ...).
   The number of edges in the JSON must match. Watch for gaps in the numbering —
   do not assume contiguous sequences (e.g. e_7 may not exist).

2. DIRECTION: For each edge, follow the arrowhead. Source is where the line
   starts (no arrowhead), target is where the arrowhead points.

3. LABEL ASSIGNMENT: Each edge label sits alongside one specific line. Verify
   that each label is assigned to the correct edge (the line it sits next to).

4. WEIGHTS: Similarly, each weight number sits near a specific line. Verify
   correct assignment.

5. MISSED EDGES: Look for lines in the drawing that aren't represented in the
   JSON. Also look for phantom edges in the JSON that don't exist in the drawing
   (this is equally important).

6. NODE DEGREE: For each node, count how many lines touch it in the image.
   Compare this with the number of edges referencing that node in the JSON
   (as source or target). If they don't match, something is wrong.

7. NODE POSITIONS: Verify that node coordinates accurately reflect the actual
   positions in the image. Check:
   - Nodes that are close together in the drawing should have coordinates that
     are close together (not spread apart).
   - Nodes that are roughly at the same height should have similar y values.
   - Nodes that are roughly vertically aligned should have similar x values.
   - The relative distances between node pairs should match the image.
   Coordinates are fractions of the image dimensions (0,0 = top-left corner,
   1,1 = bottom-right corner of the image).

If the extraction is correct, output the same JSON unchanged.
If there are errors, output a CORRECTED JSON with a brief comment before the
JSON explaining what you fixed.

Output the JSON inside a ```json code fence.

Here is the extracted JSON to verify:
"""


REEXTRACT_PROMPT = """\
You are an expert at analyzing photographs of hand-drawn graphs and network
diagrams on whiteboards, paper, or notebooks. Your task is to extract the
COMPLETE graph structure and produce a JSON representation.

This is a SECOND INDEPENDENT extraction. A previous extraction was already done,
but you must NOT rely on it. Instead, analyze the image from scratch.

Hand-drawn graphs are often messy: edges overlap, labels crowd together, and
handwriting is imperfect. You must be very careful and methodical.

## Step-by-step process

Work through these steps IN ORDER. Write out your analysis for each step
before producing the final JSON.

### Step 1 — Identify ALL nodes

Scan the entire image for labeled circles, dots, or named intersection points.
List each node with:
- Its exact name (case-sensitive, as written)
- Its (x, y) position as a fraction of the IMAGE dimensions (not the graph
  bounding box): (0, 0) = top-left corner of the image, (1, 1) = bottom-right
  corner of the image.

**Position accuracy is critical.** The rendered output will use these coordinates
directly, so the distances between nodes must match what you see in the drawing.
Follow this procedure:
1. First, identify the pixel-level position of each node relative to the full
   image (top-left = 0,0; bottom-right = 1,1).
2. Nodes that are close together in the drawing MUST have coordinates that are
   close together. If two nodes are 10% of the image apart, their coordinates
   should differ by ~0.10 — do NOT spread them further apart.
3. Verify your positions: compare the distance ratios between pairs of nodes in
   your coordinate list with the visual distances in the image.

Write: "Nodes found: [list them with positions]"

### Step 2 — Count edge labels

Before tracing edges, count ALL edge labels visible in the image (e.g. e₁, e₂,
e₃, ...). Write them out. This tells you exactly how many edges to find.

IMPORTANT: Only count labels that are actually written in the image. Edge label
sequences may have gaps (e.g. e₁, e₂, e₃, e₄, e₅, e₆, e₈, e₉ — skipping e₇).
Do NOT assume the sequence is contiguous. Only count what you can see.

If the graph has NO edge labels at all, instead count the number of distinct
lines (edges) you can see connecting nodes.

Write: "Edge labels visible: [list them]. Total: N edges expected."

### Step 3 — Trace EVERY edge, one by one

This is the most critical step. For EACH edge, find the line and trace it
from one node to the other.

For each edge write out:
- "e_X: [source] → [target], weight=[W], curvature=[describe curve]"

**Tracing rules:**
- Follow each line physically from endpoint to endpoint. Do NOT assume a line
  goes to the nearest node — it may loop around or cross other edges.
- Arrowheads indicate direction. The arrowhead points at the TARGET.
- If two edges connect the same pair of nodes, they curve in opposite directions.
- Pay EXTRA attention to dense areas where multiple edges meet at one node.
  Each line entering/leaving a node is a separate edge.
- For the direction of each arrow, look VERY CAREFULLY at which end has the
  arrowhead (V or > shape). The plain end is the source; the arrowhead end
  is the target. ZOOM IN mentally on each arrow tip.

After listing all edges, verify:
- Does your edge count match the label count from Step 2?
- For each node, count the lines touching it in the image and compare with
  your edge list.

### Step 4 — Curvature values

For each edge, assign a numeric curvature:
- 0.0 = straight line
- 0.1–0.2 = slight curve
- 0.3–0.5 = strong curve (wide arc)

Sign convention: imagine standing at the source node looking toward the target.
- Edge curves to your LEFT → positive curvature
- Edge curves to your RIGHT → negative curvature

### Step 5 — Weight format & graph type

- If weights are plain numbers (10, 32) → weight_format = "plain"
- If weights have ×/x suffix (10×, 32x) → weight_format = "multiply"
- If edges have arrowheads → graph_type = "directed"

### Step 6 — Output JSON

After completing your analysis, output the final JSON inside a code fence:

```json
{
  "nodes": [
    {"name": "S", "x": 0.0, "y": 0.5, "shape": "circle"}
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

Field notes:
- "shape": always "circle" unless clearly different
- "color": always null
- "label": use underscore for subscripts (e₁ → "e_1"); null if unlabeled
- "weight": bare number as string ("10"); null if no weight visible
- "title": null unless drawing has an explicit title
"""


ADJUDICATE_PROMPT = """\
You are adjudicating disagreements between two independent extractions of a
hand-drawn graph from the same image. Two passes analyzed the same image
independently and produced different results for some edges.

Your job is to look at the image carefully and determine the TRUTH for EACH
disputed edge listed below. The two passes AGREED on the edges not listed
here, so those are settled.

IMPORTANT RULES:
- You are ONLY deciding about the specific disputed edges listed below.
- Do NOT add any other edges beyond what is listed.
- For each dispute, trace the relevant line(s) physically in the image.
- Follow arrowheads to determine direction (arrowhead points at the TARGET).
- If a line does not exist in the drawing, mark it as "exists": false.
- If a line exists, specify the correct source, target, weight, and curvature.
- If the two passes disagree on direction (e.g., Pass 1 says A→B but Pass 2
  says B→A), look VERY CAREFULLY at the arrowhead to determine which is correct.

NODES IN THIS GRAPH (for reference):
{node_list}

For each disputed edge, write your analysis, then output a JSON array of
verdicts inside a ```json code fence:

```json
[
  {{
    "dispute_id": "dispute_1",
    "exists": true,
    "source": "A",
    "target": "B",
    "label": "e_1",
    "weight": "10",
    "directed": true,
    "curvature": 0.0,
    "reasoning": "brief explanation"
  }}
]
```

If "exists" is false, the source/target/label/weight/curvature fields are ignored.

Here are the disputed edges:
"""


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling chain-of-thought + JSON output.

    The response may contain reasoning text followed by a JSON block.
    We try several strategies:
    1. Look for a ```json ... ``` fenced code block (last one wins).
    2. Look for the last { ... } block that parses as valid JSON.
    3. Try parsing the entire text as JSON (fallback).
    """
    text = text.strip()

    # Strategy 1: Find the last ```json ... ``` fenced block.
    import re

    fenced_blocks = re.findall(
        r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL
    )
    if fenced_blocks:
        # Use the last fenced block (most likely the final output)
        return json.loads(fenced_blocks[-1].strip())

    # Strategy 2: Find the last top-level { ... } that parses as JSON.
    # Walk backwards through the text looking for the last '{'.
    last_json = None
    depth = 0
    end = len(text)
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "}":
            if depth == 0:
                end = i + 1
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                candidate = text[i:end]
                try:
                    last_json = json.loads(candidate)
                    break
                except json.JSONDecodeError:
                    depth = 0  # reset and keep searching

    if last_json is not None:
        return last_json

    # Strategy 3: Try parsing the whole text.
    return json.loads(text)


def _validate_graph(graph: GraphSpec, verbose: bool = False) -> GraphSpec:
    """Run programmatic sanity checks on the extracted graph.

    Fixes obvious errors and logs warnings. Returns a cleaned GraphSpec.
    """
    warnings: list[str] = []
    node_names = {n.name for n in graph.nodes}

    # 1. Remove self-loops (almost always hallucinated)
    original_count = len(graph.edges)
    graph.edges = [e for e in graph.edges if e.source != e.target]
    removed = original_count - len(graph.edges)
    if removed:
        warnings.append(f"Removed {removed} self-loop edge(s) (likely hallucinated).")

    # 2. Check for edges referencing non-existent nodes
    bad_edges = []
    for e in graph.edges:
        if e.source not in node_names:
            bad_edges.append(f"Edge {e.label}: source '{e.source}' not in node list")
        if e.target not in node_names:
            bad_edges.append(f"Edge {e.label}: target '{e.target}' not in node list")
    if bad_edges:
        warnings.append("Edges reference unknown nodes: " + "; ".join(bad_edges))

    # 3. Check for duplicate edges (same source, target, and label)
    seen = set()
    deduped = []
    for e in graph.edges:
        key = (e.source, e.target, e.label)
        if key in seen:
            warnings.append(
                f"Removed duplicate edge: {e.source}→{e.target} ({e.label})"
            )
            continue
        seen.add(key)
        deduped.append(e)
    graph.edges = deduped

    # 4. Check for isolated nodes (not connected to any edge)
    connected = set()
    for e in graph.edges:
        connected.add(e.source)
        connected.add(e.target)
    isolated = node_names - connected
    if isolated:
        warnings.append(
            f"Isolated node(s) not connected to any edge: {', '.join(sorted(isolated))}"
        )

    # 5. Check for duplicate labels assigned to different edges
    label_edges: dict[str, list[str]] = {}
    for e in graph.edges:
        if e.label:
            label_edges.setdefault(e.label, []).append(
                f"{e.source}→{e.target}"
            )
    for label, edges in label_edges.items():
        if len(edges) > 1:
            warnings.append(
                f"Label '{label}' assigned to multiple edges: {', '.join(edges)}"
            )

    if verbose and warnings:
        print("Validation warnings:", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)
    elif verbose:
        print("Validation: no issues found.", file=sys.stderr)

    return graph


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------


def _edge_pair(edge: Edge) -> tuple[str, str]:
    """Return the unordered node pair for an edge (for matching across passes).

    We use frozenset-style ordering (sorted) so that A->B and B->A map to the
    same pair. Direction disagreements are handled as disputes.
    """
    a, b = edge.source, edge.target
    return (min(a, b), max(a, b))


def _edge_signature(edge: Edge) -> str:
    """A richer signature for exact matching: source, target, label, weight."""
    return f"{edge.source}->{edge.target}|{edge.label}|{edge.weight}"


class EdgeDispute:
    """Represents a disagreement between two extraction passes about an edge."""

    def __init__(
        self,
        dispute_id: str,
        pass1_edge: Edge | None,
        pass2_edge: Edge | None,
        reason: str,
    ):
        self.dispute_id = dispute_id
        self.pass1_edge = pass1_edge
        self.pass2_edge = pass2_edge
        self.reason = reason

    def to_description(self) -> str:
        """Format this dispute for the adjudication prompt."""
        lines = [f"### {self.dispute_id}: {self.reason}"]
        if self.pass1_edge:
            e = self.pass1_edge
            lines.append(
                f"  Pass 1: {e.source} → {e.target}"
                f" (label={e.label}, weight={e.weight}, curvature={e.curvature})"
            )
        else:
            lines.append("  Pass 1: [edge not present]")
        if self.pass2_edge:
            e = self.pass2_edge
            lines.append(
                f"  Pass 2: {e.source} → {e.target}"
                f" (label={e.label}, weight={e.weight}, curvature={e.curvature})"
            )
        else:
            lines.append("  Pass 2: [edge not present]")
        return "\n".join(lines)


def _diff_edge_sets(
    graph1: GraphSpec,
    graph2: GraphSpec,
    verbose: bool = False,
) -> tuple[list[Edge], list[EdgeDispute]]:
    """Compare two independently extracted graphs and find agreements/disputes.

    Two edges between the same node pair are considered to "agree" if they have
    the same direction, label, and weight. Otherwise they are a dispute.

    Returns:
        agreed: List of edges that both passes agree on (taken from pass 1).
        disputes: List of EdgeDispute objects for edges that differ.
    """
    agreed: list[Edge] = []
    disputes: list[EdgeDispute] = []
    dispute_counter = 0

    # Index pass-2 edges by unordered node pair for matching.
    # Multiple edges can share the same node pair (parallel edges), so we use
    # lists.
    pass2_by_pair: dict[tuple[str, str], list[Edge]] = {}
    for e in graph2.edges:
        pair = _edge_pair(e)
        pass2_by_pair.setdefault(pair, []).append(e)

    # Track which pass-2 edges were matched (so we can find pass-2-only edges).
    matched_pass2: set[int] = set()  # indices into original graph2.edges

    # Build a flat index mapping (pair, list-position) -> original index
    pass2_flat_idx: dict[tuple[tuple[str, str], int], int] = {}
    pair_counters: dict[tuple[str, str], int] = {}
    for orig_idx, e in enumerate(graph2.edges):
        pair = _edge_pair(e)
        pos = pair_counters.get(pair, 0)
        pair_counters[pair] = pos + 1
        pass2_flat_idx[(pair, pos)] = orig_idx

    # For each pass-1 edge, look for a matching pass-2 edge
    for e1 in graph1.edges:
        pair = _edge_pair(e1)
        candidates = pass2_by_pair.get(pair, [])

        best_match: Edge | None = None
        best_match_idx: int = -1
        best_match_list_pos: int = -1

        for list_pos, e2 in enumerate(candidates):
            orig_idx = pass2_flat_idx.get((pair, list_pos))
            if orig_idx is not None and orig_idx in matched_pass2:
                continue  # already consumed

            # Check for exact agreement (same direction, label, weight)
            if (
                e1.source == e2.source
                and e1.target == e2.target
                and e1.label == e2.label
                and e1.weight == e2.weight
            ):
                best_match = e2
                best_match_idx = orig_idx if orig_idx is not None else -1
                best_match_list_pos = list_pos
                break  # perfect match

            # Check for same-direction match (different label or weight)
            if e1.source == e2.source and e1.target == e2.target:
                if best_match is None or (
                    # Prefer same-direction over reverse-direction
                    best_match.source != e1.source
                ):
                    best_match = e2
                    best_match_idx = orig_idx if orig_idx is not None else -1
                    best_match_list_pos = list_pos

            # Accept reverse-direction match (same node pair, opposite direction)
            # — this is a dispute, but we should pair them rather than treating
            # each as an independent "only in pass X" edge.
            if best_match is None:
                best_match = e2
                best_match_idx = orig_idx if orig_idx is not None else -1
                best_match_list_pos = list_pos

        if best_match is not None and best_match_idx >= 0:
            assert best_match is not None
            if (
                e1.source == best_match.source
                and e1.target == best_match.target
                and e1.label == best_match.label
                and e1.weight == best_match.weight
            ):
                # Full agreement — use pass-1's edge (could average curvatures)
                avg_curv = (e1.curvature + best_match.curvature) / 2
                agreed_edge = e1.model_copy(update={"curvature": avg_curv})
                agreed.append(agreed_edge)
            else:
                # Partial match (same node pair, maybe same direction, but
                # different label/weight/direction)
                dispute_counter += 1
                reason_parts = []
                if e1.source != best_match.source or e1.target != best_match.target:
                    reason_parts.append("direction disagrees")
                if e1.label != best_match.label:
                    reason_parts.append(f"label differs ({e1.label} vs {best_match.label})")
                if e1.weight != best_match.weight:
                    reason_parts.append(f"weight differs ({e1.weight} vs {best_match.weight})")
                disputes.append(EdgeDispute(
                    dispute_id=f"dispute_{dispute_counter}",
                    pass1_edge=e1,
                    pass2_edge=best_match,
                    reason="; ".join(reason_parts),
                ))
            matched_pass2.add(best_match_idx)
        else:
            # Pass-1 only edge — might be real or hallucinated
            dispute_counter += 1
            disputes.append(EdgeDispute(
                dispute_id=f"dispute_{dispute_counter}",
                pass1_edge=e1,
                pass2_edge=None,
                reason=f"only in Pass 1 ({e1.source}→{e1.target})",
            ))

    # Find pass-2-only edges (not matched to any pass-1 edge)
    for orig_idx, e2 in enumerate(graph2.edges):
        if orig_idx not in matched_pass2:
            dispute_counter += 1
            disputes.append(EdgeDispute(
                dispute_id=f"dispute_{dispute_counter}",
                pass1_edge=None,
                pass2_edge=e2,
                reason=f"only in Pass 2 ({e2.source}→{e2.target})",
            ))

    if verbose:
        print(
            f"Cross-validation: {len(agreed)} agreed, {len(disputes)} disputed.",
            file=sys.stderr,
        )
        for d in disputes:
            print(f"  {d.dispute_id}: {d.reason}", file=sys.stderr)

    return agreed, disputes


def _format_disputes_for_prompt(disputes: list[EdgeDispute]) -> str:
    """Format dispute list for inclusion in the adjudication prompt."""
    return "\n\n".join(d.to_description() for d in disputes)


def _parse_adjudication_response(text: str) -> list[dict]:
    """Parse the adjudication LLM response into a list of verdict dicts."""
    import re

    # Look for a JSON array in a code fence
    fenced_blocks = re.findall(
        r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL
    )
    if fenced_blocks:
        return json.loads(fenced_blocks[-1].strip())

    # Try to find a bare JSON array
    # Walk backwards looking for ]
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "]":
            # Find matching [
            depth = 0
            for j in range(i, -1, -1):
                if text[j] == "]":
                    depth += 1
                elif text[j] == "[":
                    depth -= 1
                    if depth == 0:
                        candidate = text[j : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break

    raise json.JSONDecodeError("No valid JSON array found in adjudication response", text, 0)


def _merge_graph(
    graph1: GraphSpec,
    graph2: GraphSpec,
    agreed_edges: list[Edge],
    verdicts: list[dict],
    verbose: bool = False,
) -> GraphSpec:
    """Merge agreed edges + adjudication verdicts into a final GraphSpec.

    Uses node positions from pass 1 (could average with pass 2 in the future).
    """
    final_edges = list(agreed_edges)

    for v in verdicts:
        if not v.get("exists", True):
            if verbose:
                print(
                    f"  Adjudication: {v.get('dispute_id', '?')} → does not exist",
                    file=sys.stderr,
                )
            continue

        edge = Edge(
            source=v["source"],
            target=v["target"],
            label=v.get("label"),
            weight=v.get("weight"),
            directed=v.get("directed", True),
            color=None,
            curvature=v.get("curvature", 0.0),
        )
        if verbose:
            print(
                f"  Adjudication: {v.get('dispute_id', '?')} → "
                f"{edge.source}→{edge.target} (label={edge.label}, weight={edge.weight})",
                file=sys.stderr,
            )
        final_edges.append(edge)

    # Average node positions from both passes for better accuracy
    pos1 = {n.name: (n.x, n.y) for n in graph1.nodes}
    pos2 = {n.name: (n.x, n.y) for n in graph2.nodes}
    all_node_names = set(pos1) | set(pos2)

    final_nodes = []
    for name in sorted(all_node_names):
        if name in pos1 and name in pos2:
            x = (pos1[name][0] + pos2[name][0]) / 2
            y = (pos1[name][1] + pos2[name][1]) / 2
        elif name in pos1:
            x, y = pos1[name]
        else:
            x, y = pos2[name]
        final_nodes.append(Node(name=name, x=x, y=y, shape="circle"))

    return GraphSpec(
        nodes=final_nodes,
        edges=final_edges,
        title=graph1.title,
        graph_type=graph1.graph_type,
        weight_format=graph1.weight_format,
    )


# ---------------------------------------------------------------------------
# Decomposed extraction pipeline — focused sub-pass prompts
# ---------------------------------------------------------------------------

PASS_A_PROMPT = """\
You are an expert at analyzing photographs of hand-drawn graphs and network
diagrams on whiteboards, paper, or notebooks.

Your ONLY task is to identify ALL NODES in this image. Ignore edges for now.

## Instructions

1. Scan the ENTIRE image for labeled circles, dots, boxes, or named points.
2. For each node, record:
   - **name**: The exact label (case-sensitive, as written in the drawing).
   - **x, y**: Position as a fraction of the IMAGE dimensions.
     (0, 0) = top-left corner, (1, 1) = bottom-right corner.

3. Position accuracy is CRITICAL:
   - Nodes that are close together in the drawing MUST have close coordinates.
   - Nodes at roughly the same height should have similar y values.
   - Nodes at roughly the same horizontal position should have similar x values.
   - Verify by checking that distance ratios between pairs match the visual.

Output a JSON object inside a ```json code fence:

```json
{
  "nodes": [
    {"name": "A", "x": 0.15, "y": 0.30}
  ]
}
```

Only output the nodes list. Do NOT include edges, title, or any other fields.
"""

PASS_B_PROMPT = """\
You are an expert at analyzing photographs of hand-drawn graphs. Your ONLY task
is to identify the TOPOLOGY — which pairs of nodes are connected by edges.

You do NOT need to determine edge direction (arrows) or weights. Just find
which nodes are connected.

## Known nodes and their approximate positions (fraction of image, 0,0=top-left)

{node_positions}

## Instructions

1. **COUNT edge labels first.** Scan the image for all edge labels (e₁, e₂, …).
   Edge label sequences may have GAPS (e.g. e₁–e₆, e₈, e₉ — no e₇).
   If there are NO edge labels, count distinct lines connecting nodes.
   Record the expected total.

2. **TRACE each edge step by step.** For EACH edge, you MUST:
   a. Find the edge label (or the next untraced line).
   b. Starting from the label, follow the line in one direction until you reach
      a node. Write down WHICH NODE the line reaches.
   c. Go back to the label, follow the line in the OTHER direction until you
      reach a node. Write down WHICH NODE the line reaches.
   d. Record BOTH endpoints.

   **CRITICAL TRACING RULES:**
   - A line may be a LONG CURVED ARC that passes near intermediate nodes
     without connecting to them. KEEP TRACING past any node the line doesn't
     actually ENTER. The line must physically terminate at a node's circle
     (touch it, pierce it, or end with an arrowhead touching it).
   - "Passing near" a node is NOT "connecting to" that node. Only count a
     connection if the line ENDS at that node.
   - Use the node coordinates above to verify: does the line actually reach
     the position where that node is drawn?
   - Two lines may CROSS each other — a crossing is NOT a connection.

3. **Write a "trace" for each edge** describing the path of the line. This is
   mandatory and helps verify correctness. Example traces:
   - "Line from label e₅ goes left and curves upward, bypasses C, continues
     leftward and terminates at A. Other direction: goes right, reaches B."
   - "Line from label e₈ goes right, curves down, passes below D, swings
     back left and up to reach A. Other direction: goes left to D."

4. **Report ALPHABETICAL pairs.** If a line connects B and A, write
   node_a="A", node_b="B". Always put the alphabetically-first node first.
   This says nothing about direction.

5. **PARALLEL EDGES**: Two distinct lines connecting the same pair of nodes
   (usually curving in opposite directions) are listed as separate edges with
   their respective labels.

6. **DEGREE CHECK (mandatory).** After listing all edges, for EACH node:
   - Count how many lines physically touch that node IN THE IMAGE
     (look at the image, don't count from your list).
   - Count how many times that node appears in your edge list.
   - If these numbers differ, ONE OF YOUR EDGES IS WRONG. Re-trace the
     edges for that node and fix errors before producing output.
   Write out the degree check as "degree_check" in your JSON output.

7. If an edge has a label (e₁, e₂, etc.), record it with underscore notation:
   e₁ → "e_1". If unlabeled, use null.

Output a JSON object inside a ```json code fence:

```json
{{
  "edge_count_expected": 8,
  "degree_check": {{
    "A": {{"image": 3, "list": 3}},
    "B": {{"image": 4, "list": 4}}
  }},
  "edges": [
    {{"node_a": "A", "node_b": "B", "label": "e_1", "trace": "from label goes left to A, right to B"}},
    {{"node_a": "A", "node_b": "C", "label": null, "trace": "straight line between A and C"}}
  ]
}}
```

- "node_a" is always the alphabetically-first node of the pair.
- "node_b" is the other node.
- "trace" is your description of the line's path (mandatory).
- Do NOT include direction, weight, curvature, or any other fields.
"""

PASS_C_PROMPT = """\
You are an expert at analyzing photographs of hand-drawn graphs. Your ONLY task
is to determine the DIRECTION of each arrow in this graph.

## Known nodes and their positions

{node_positions}

## Known edges (undirected pairs)

{edge_list}

## Instructions

For EACH edge listed above, look VERY CAREFULLY at the line connecting those
two nodes:

1. Find BOTH ENDS of the line where it meets each node.
2. One end has an arrowhead — a pointed tip shaped like >, V, <, ^, or a small
   triangle. That end is the TARGET (where the arrow goes TO).
3. The other end is a plain line meeting the node circle. That is the SOURCE
   (where the arrow comes FROM).
4. ZOOM IN mentally on BOTH ends of each line. Do not assume direction from
   node names, alphabetical order, or positions — ONLY from the visible
   arrowhead.
5. For edges that are long curved arcs, trace the entire arc to both endpoints
   before determining direction.

If the graph has NO arrowheads on any edge, set "directed" to false for all.

For each edge, report:
- "source": the node the arrow comes FROM (plain end)
- "target": the node the arrow points TO (arrowhead end)
- "label": the edge label (same as provided in the edge list)
- "directed": true if there's an arrowhead, false if not
- "confidence": "high" or "low" — be honest if you can't clearly see the arrowhead

Output a JSON object inside a ```json code fence:

```json
{{
  "edges": [
    {{"source": "A", "target": "B", "label": "e_1", "directed": true, "confidence": "high"}}
  ]
}}
```
"""

PASS_D_PROMPT = """\
You are an expert at analyzing photographs of hand-drawn graphs. Your ONLY task
is to identify the WEIGHT (number) written alongside each edge, and the weight
display format.

## Known nodes and their positions

{node_positions}

## Known edges (with direction)

{edge_list}

## Instructions

1. For EACH edge listed above, look at the line connecting those two nodes.
   Find any number written alongside, near, or on top of that specific line.

2. **Match numbers to edges carefully.** In dense areas, a number might appear
   between multiple edges. Determine which line the number is closest to and
   which line it runs parallel to.

3. Look carefully at whether numbers have a multiplication sign:
   - Plain numbers: 10, 32, 8 → weight_format = "plain"
   - Numbers with ×, x, or a multiplication symbol: 10×, 32x, 8× →
     weight_format = "multiply"
   - Look at ALL weights to determine the format. It should be consistent
     across the graph.
   - The × symbol may look like a small "x" after the number. Look carefully.

4. If an edge has NO visible weight, use null.

5. Record each weight as a bare number string (e.g. "10", "32"), WITHOUT
   any × or x suffix. The weight_format field tells us how to display them.

Output a JSON object inside a ```json code fence:

```json
{{
  "weight_format": "plain",
  "edges": [
    {{"source": "A", "target": "B", "label": "e_1", "weight": "10"}},
    {{"source": "B", "target": "C", "label": "e_2", "weight": null}}
  ]
}}
```
"""


# ---------------------------------------------------------------------------
# Decomposed extraction — helper functions
# ---------------------------------------------------------------------------


def _extract_nodes(
    client,
    model: str,
    image_content: dict,
    image_path: str,
    use_cache: bool,
    verbose: bool,
    max_retries: int,
) -> list[Node]:
    """Pass A: Extract node names and positions from the image.

    Returns a list of Node objects.
    """
    phash = _prompt_hash(PASS_A_PROMPT)
    ckey = _cache_key(image_path, model, phash, "pass_a_nodes") if use_cache else None

    cached = _cache_get(ckey) if (use_cache and ckey) else None
    if cached is not None:
        if verbose:
            print("Pass A (nodes): Using cached response.", file=sys.stderr)
        try:
            data = _parse_json_response(cached)
            return [Node(name=n["name"], x=n["x"], y=n["y"], shape="circle") for n in data["nodes"]]
        except Exception:
            cached = None

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = _call_llm(
                client, model,
                messages=[
                    {"role": "system", "content": PASS_A_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": (
                            "Identify ALL nodes in this hand-drawn graph image. "
                            "Report each node's name and position as fractions of "
                            "the image dimensions (0,0 = top-left, 1,1 = bottom-right)."
                        )},
                        image_content,
                    ]},
                ],
                verbose=verbose,
                temperature=0.1,
            )
            data = _parse_json_response(raw)
            nodes = [Node(name=n["name"], x=n["x"], y=n["y"], shape="circle") for n in data["nodes"]]
            if use_cache and ckey:
                _cache_put(ckey, raw)
            if verbose:
                print(f"Pass A (nodes): Found {len(nodes)} nodes: {[n.name for n in nodes]}", file=sys.stderr)
            return nodes
        except Exception as e:
            last_error = str(e)
            if verbose:
                print(f"Pass A attempt {attempt}/{max_retries}: {last_error}", file=sys.stderr)
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Pass A (nodes) failed after {max_retries} attempts: {last_error}")


def _extract_topology(
    client,
    model: str,
    image_content: dict,
    nodes: list[Node],
    image_path: str,
    use_cache: bool,
    verbose: bool,
    max_retries: int,
) -> list[dict]:
    """Pass B: Extract edge topology (undirected, alphabetical pairs) from the image.

    Returns a list of dicts with keys: node_a, node_b, label.
    node_a is always alphabetically before node_b.
    """
    node_positions = "\n".join(
        f"  {n.name}: ({n.x:.2f}, {n.y:.2f})" for n in sorted(nodes, key=lambda n: n.name)
    )
    prompt = PASS_B_PROMPT.format(node_positions=node_positions)
    phash = _prompt_hash(prompt)
    ckey = _cache_key(image_path, model, phash, "pass_b_topology") if use_cache else None

    cached = _cache_get(ckey) if (use_cache and ckey) else None
    if cached is not None:
        if verbose:
            print("Pass B (topology): Using cached response.", file=sys.stderr)
        try:
            data = _parse_json_response(cached)
            return data["edges"]
        except Exception:
            cached = None

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = _call_llm(
                client, model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": (
                            "Identify ALL edges (connections between nodes) in this "
                            "hand-drawn graph. Trace each line to where it physically "
                            "meets a node circle — use the node positions provided to "
                            "verify endpoints. Report each edge as an alphabetically "
                            "ordered pair. Include the degree check."
                        )},
                        image_content,
                    ]},
                ],
                verbose=verbose,
                temperature=0.1,
            )
            data = _parse_json_response(raw)
            edges = data["edges"]

            # Enforce alphabetical ordering
            for e in edges:
                a, b = e["node_a"], e["node_b"]
                if a > b:
                    e["node_a"], e["node_b"] = b, a

            if use_cache and ckey:
                _cache_put(ckey, raw)
            if verbose:
                print(f"Pass B (topology): Found {len(edges)} edges.", file=sys.stderr)
                for e in edges:
                    print(f"  {e['node_a']}--{e['node_b']} (label={e.get('label')})", file=sys.stderr)
                # Log degree check if present
                dc = data.get("degree_check")
                if dc:
                    print("  Degree check:", file=sys.stderr)
                    for node, counts in sorted(dc.items()):
                        img = counts.get("image", "?")
                        lst = counts.get("list", "?")
                        flag = " *** MISMATCH ***" if img != lst else ""
                        print(f"    {node}: image={img}, list={lst}{flag}", file=sys.stderr)
            return edges
        except Exception as e:
            last_error = str(e)
            if verbose:
                print(f"Pass B attempt {attempt}/{max_retries}: {last_error}", file=sys.stderr)
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Pass B (topology) failed after {max_retries} attempts: {last_error}")


PASS_B_ADJUDICATE_PROMPT = """\
You are an expert at analyzing photographs of hand-drawn graphs. Two separate
analyses produced DIFFERENT topology results for this graph. Your job is to
determine the CORRECT topology by carefully re-examining the image.

## Known nodes and their approximate positions (fraction of image, 0,0=top-left)

{node_positions}

## Analysis 1 found these edges:
{pass1_edges}

## Analysis 2 found these edges:
{pass2_edges}

## DISAGREEMENTS to resolve:
{disagreements}

## Instructions

For each DISPUTED EDGE, carefully trace the line in the image:
1. Find the edge label (if labeled) or the line (if unlabeled).
2. Follow the line from the label in BOTH directions, all the way to where it
   terminates at a node circle.
3. Write a "trace" describing the path.
4. Report the CORRECT pair of nodes for that edge.

**Remember**: Lines can be long curved arcs that pass near intermediate nodes
without connecting to them. Trace ALL THE WAY to the actual endpoints.

**IMPORTANT RULES**:
- Only output edges that ACTUALLY EXIST in the image. Do NOT include entries
  for edges that don't exist — just omit them.
- When two analyses disagree about which nodes a SINGLE line connects
  (e.g., one says D-G, the other says E-G), there is only ONE line — pick
  the CORRECT pair of endpoints. Do NOT output both.
- The total number of edges in the graph should be approximately
  {expected_edge_count}.

Output a JSON object inside a ```json code fence:

```json
{{
  "resolved_edges": [
    {{"node_a": "A", "node_b": "B", "label": "e_5", "trace": "arc curves up from A past C to B"}}
  ]
}}
```
"""


def _cross_validate_topology(
    client,
    model: str,
    image_content: dict,
    nodes: list[Node],
    pass1: list[dict],
    image_path: str,
    use_cache: bool,
    verbose: bool,
    max_retries: int,
) -> list[dict]:
    """Run Pass B a second time and resolve disagreements.

    If both passes agree, return the agreed topology.
    If they disagree, run an adjudication pass to resolve disputes.
    """
    # Run Pass B again with a different cache key and slightly higher temperature
    node_positions_str = "\n".join(
        f"  {n.name}: ({n.x:.2f}, {n.y:.2f})" for n in sorted(nodes, key=lambda n: n.name)
    )
    prompt = PASS_B_PROMPT.format(node_positions=node_positions_str)

    # Use a different cache key for the second pass
    phash = _prompt_hash(prompt + "_pass2")
    ckey = _cache_key(image_path, model, phash, "pass_b_topology_v2") if use_cache else None

    cached = _cache_get(ckey) if (use_cache and ckey) else None
    pass2 = None
    if cached is not None:
        try:
            data = _parse_json_response(cached)
            pass2 = data["edges"]
            for e in pass2:
                a, b = e["node_a"], e["node_b"]
                if a > b:
                    e["node_a"], e["node_b"] = b, a
        except Exception:
            cached = None

    if pass2 is None:
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                raw = _call_llm(
                    client, model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": (
                                "Identify ALL edges in this hand-drawn graph. "
                                "For EACH edge, trace the line step by step from its label "
                                "to BOTH endpoints. Be very careful with long curved arcs — "
                                "trace them all the way to where they terminate at a node."
                            )},
                            image_content,
                        ]},
                    ],
                    verbose=verbose,
                    temperature=0.3,  # slightly higher for diversity
                )
                data = _parse_json_response(raw)
                pass2 = data["edges"]
                for e in pass2:
                    a, b = e["node_a"], e["node_b"]
                    if a > b:
                        e["node_a"], e["node_b"] = b, a
                if use_cache and ckey:
                    _cache_put(ckey, raw)
                break
            except Exception as e:
                last_error = str(e)
                if "rate" in str(e).lower() or "429" in str(e):
                    time.sleep(2 ** attempt)
        if pass2 is None:
            if verbose:
                print("Pass B.2 failed, using Pass B.1 results.", file=sys.stderr)
            return pass1

    if verbose:
        print(f"Pass B.2 (cross-val): Found {len(pass2)} edges.", file=sys.stderr)
        for e in pass2:
            print(f"  {e['node_a']}--{e['node_b']} (label={e.get('label')})", file=sys.stderr)

    # Compare pass1 and pass2
    def edge_key(e):
        return (e["node_a"], e["node_b"], e.get("label"))

    set1 = {edge_key(e) for e in pass1}
    set2 = {edge_key(e) for e in pass2}

    agreed = set1 & set2
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1

    if not only_in_1 and not only_in_2:
        if verbose:
            print("Pass B cross-validation: Both passes AGREE on all edges.", file=sys.stderr)
        return pass1

    if verbose:
        print(f"Pass B cross-validation: {len(agreed)} agreed, "
              f"{len(only_in_1)} only in pass 1, {len(only_in_2)} only in pass 2.",
              file=sys.stderr)
        if only_in_1:
            for k in only_in_1:
                print(f"  Only pass 1: {k[0]}--{k[1]} ({k[2]})", file=sys.stderr)
        if only_in_2:
            for k in only_in_2:
                print(f"  Only pass 2: {k[0]}--{k[1]} ({k[2]})", file=sys.stderr)

    # Build the agreed edges
    agreed_edges = [e for e in pass1 if edge_key(e) in agreed]

    # Build disagreement descriptions — prioritize same-label disputes
    disagreements = []

    # First: same-label different-pair disagreements (most important)
    label_to_pair1 = {e.get("label"): (e["node_a"], e["node_b"]) for e in pass1 if e.get("label")}
    label_to_pair2 = {e.get("label"): (e["node_a"], e["node_b"]) for e in pass2 if e.get("label")}
    disputed_labels = set()
    for label in set(label_to_pair1) & set(label_to_pair2):
        if label_to_pair1[label] != label_to_pair2[label]:
            p1 = label_to_pair1[label]
            p2 = label_to_pair2[label]
            disagreements.append(
                f"- **DISPUTE**: Edge {label}: Analysis 1 says {p1[0]}--{p1[1]}, "
                f"Analysis 2 says {p2[0]}--{p2[1]}. Trace this line and report "
                f"the correct pair."
            )
            disputed_labels.add(label)

    # Second: edges only in one pass (excluding already-disputed labels)
    # For unlabeled edges, try to pair up disagreements that share a node
    # (likely the same physical line with different endpoint identification)
    unlabeled_only1 = [k for k in only_in_1 if k[2] is None and k[2] not in disputed_labels]
    unlabeled_only2 = [k for k in only_in_2 if k[2] is None and k[2] not in disputed_labels]
    labeled_only1 = [k for k in only_in_1 if k[2] is not None and k[2] not in disputed_labels]
    labeled_only2 = [k for k in only_in_2 if k[2] is not None and k[2] not in disputed_labels]

    # Pair up unlabeled disagreements that share a node
    paired_1 = set()
    paired_2 = set()
    for i, k1 in enumerate(unlabeled_only1):
        for j, k2 in enumerate(unlabeled_only2):
            if j in paired_2:
                continue
            shared = {k1[0], k1[1]} & {k2[0], k2[1]}
            if shared:
                disagreements.append(
                    f"- **DISPUTE** (unlabeled): Analysis 1 says {k1[0]}--{k1[1]}, "
                    f"Analysis 2 says {k2[0]}--{k2[1]}. These likely refer to the "
                    f"SAME physical line. Trace it and report the correct pair. "
                    f"Only ONE edge should result."
                )
                paired_1.add(i)
                paired_2.add(j)
                break

    # Any remaining unpaired disagreements
    for i, k in enumerate(unlabeled_only1):
        if i not in paired_1:
            disagreements.append(f"- Analysis 1 says {k[0]}--{k[1]} (unlabeled), Analysis 2 does NOT have this edge. Does this edge exist?")
    for j, k in enumerate(unlabeled_only2):
        if j not in paired_2:
            disagreements.append(f"- Analysis 2 says {k[0]}--{k[1]} (unlabeled), Analysis 1 does NOT have this edge. Does this edge exist?")
    for k in labeled_only1:
        disagreements.append(f"- Analysis 1 says {k[0]}--{k[1]} (label={k[2]}), Analysis 2 does NOT have this edge.")
    for k in labeled_only2:
        disagreements.append(f"- Analysis 2 says {k[0]}--{k[1]} (label={k[2]}), Analysis 1 does NOT have this edge.")

    if not disagreements:
        # Edge sets differ but no specific disagreements to resolve (e.g. different edge counts)
        # Prefer the one with more edges (less likely to miss edges)
        if verbose:
            print("Pass B cross-validation: No specific disagreements, "
                  "using pass with more edges.", file=sys.stderr)
        return pass1 if len(pass1) >= len(pass2) else pass2

    # Format edges for adjudication prompt
    pass1_str = "\n".join(
        f"  {e['node_a']}--{e['node_b']} (label={e.get('label')}, trace={e.get('trace', 'N/A')})"
        for e in pass1
    )
    pass2_str = "\n".join(
        f"  {e['node_a']}--{e['node_b']} (label={e.get('label')}, trace={e.get('trace', 'N/A')})"
        for e in pass2
    )
    disagree_str = "\n".join(disagreements)

    # Use max edge count from either pass as expected
    expected_count = max(len(pass1), len(pass2))

    adj_prompt = PASS_B_ADJUDICATE_PROMPT.format(
        node_positions=node_positions_str,
        pass1_edges=pass1_str,
        pass2_edges=pass2_str,
        disagreements=disagree_str,
        expected_edge_count=expected_count,
    )

    # Run adjudication
    adj_ckey = None
    if use_cache:
        adj_phash = _prompt_hash(adj_prompt)
        adj_ckey = _cache_key(image_path, model, adj_phash, "pass_b_adjudicate")

    cached_adj = _cache_get(adj_ckey) if (use_cache and adj_ckey) else None
    resolved = None

    if cached_adj is not None:
        try:
            data = _parse_json_response(cached_adj)
            resolved = data.get("resolved_edges", [])
        except Exception:
            cached_adj = None

    if resolved is None:
        try:
            raw = _call_llm(
                client, model,
                messages=[
                    {"role": "system", "content": adj_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": (
                            "Two analyses disagree on some edges. Please re-examine "
                            "the image and resolve each disagreement by tracing the "
                            "disputed lines carefully."
                        )},
                        image_content,
                    ]},
                ],
                verbose=verbose,
                temperature=0.1,
            )
            if use_cache and adj_ckey:
                _cache_put(adj_ckey, raw)
            data = _parse_json_response(raw)
            resolved = data.get("resolved_edges", [])
        except Exception as e:
            if verbose:
                print(f"Pass B adjudication failed ({e}), using pass 1.", file=sys.stderr)
            return pass1

    if verbose:
        print(f"Pass B adjudication resolved {len(resolved)} edges:", file=sys.stderr)
        for e in resolved:
            print(f"  {e['node_a']}--{e['node_b']} ({e.get('label')}): {e.get('trace', '')}", file=sys.stderr)

    # Merge: start with agreed edges, add resolved edges
    # Only keep resolved edges that have valid node pairs (filter out "non-edge" entries)
    valid_node_names = {n.name for n in nodes}
    valid_resolved = []
    for e in resolved:
        a, b = e.get("node_a", ""), e.get("node_b", "")
        if not a or not b or a == b:
            continue  # Skip invalid entries
        if a not in valid_node_names or b not in valid_node_names:
            continue  # Skip entries with unknown nodes
        # Skip entries whose trace says "doesn't exist" or similar
        trace = (e.get("trace") or "").lower()
        if "no line" in trace or "doesn't exist" in trace or "does not exist" in trace or "no evidence" in trace:
            if verbose:
                print(f"  Skipping non-edge: {a}--{b} ({e.get('label')}): {e.get('trace', '')}", file=sys.stderr)
            continue
        if a > b:
            e["node_a"], e["node_b"] = b, a
        valid_resolved.append(e)

    # Deduplicate: group resolved edges by label, keep only one per label
    seen_labels = set()
    deduped_resolved = []
    for e in valid_resolved:
        label = e.get("label")
        if label and label in seen_labels:
            continue  # Skip duplicate labels
        if label:
            seen_labels.add(label)
        deduped_resolved.append(e)

    # Replace agreed edges that were re-resolved (by label)
    resolved_labels = {e.get("label") for e in deduped_resolved if e.get("label")}
    final_edges = [e for e in agreed_edges if e.get("label") not in resolved_labels]
    final_edges.extend(deduped_resolved)

    if verbose:
        print(f"Pass B final: {len(final_edges)} edges after cross-validation.", file=sys.stderr)
        for e in final_edges:
            print(f"  {e['node_a']}--{e['node_b']} (label={e.get('label')})", file=sys.stderr)

    return final_edges


def _extract_directions(
    client,
    model: str,
    image_content: dict,
    nodes: list[Node],
    topology: list[dict],
    image_path: str,
    use_cache: bool,
    verbose: bool,
    max_retries: int,
) -> list[dict]:
    """Pass C: Determine direction of each edge from the image.

    Takes known nodes (with positions) and undirected topology from Pass B.
    Returns a list of dicts with: source, target, label, directed, confidence.
    """
    # Format node positions for the prompt
    node_positions = "\n".join(
        f"  {n.name}: ({n.x:.2f}, {n.y:.2f})" for n in sorted(nodes, key=lambda n: n.name)
    )

    # Format edges for the prompt
    edge_lines = []
    for i, e in enumerate(topology, 1):
        label_str = e.get("label") or "unlabeled"
        edge_lines.append(f"  {i}. {e['node_a']} -- {e['node_b']} (label: {label_str})")
    edge_list = "\n".join(edge_lines)

    prompt = PASS_C_PROMPT.format(node_positions=node_positions, edge_list=edge_list)
    phash = _prompt_hash(prompt)
    ckey = _cache_key(image_path, model, phash, "pass_c_directions") if use_cache else None

    cached = _cache_get(ckey) if (use_cache and ckey) else None
    if cached is not None:
        if verbose:
            print("Pass C (directions): Using cached response.", file=sys.stderr)
        try:
            data = _parse_json_response(cached)
            return data["edges"]
        except Exception:
            cached = None

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = _call_llm(
                client, model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": (
                            "For each edge listed, determine the arrow direction by "
                            "looking at which end has the arrowhead. The arrowhead "
                            "points at the TARGET. Report source and target for each edge."
                        )},
                        image_content,
                    ]},
                ],
                verbose=verbose,
                temperature=0.1,
            )
            data = _parse_json_response(raw)
            directions = data["edges"]

            if use_cache and ckey:
                _cache_put(ckey, raw)
            if verbose:
                print(f"Pass C (directions): Got directions for {len(directions)} edges.", file=sys.stderr)
                for d in directions:
                    conf = d.get("confidence", "?")
                    print(f"  {d['source']}→{d['target']} ({d.get('label', '?')}, confidence={conf})", file=sys.stderr)
            return directions
        except Exception as e:
            last_error = str(e)
            if verbose:
                print(f"Pass C attempt {attempt}/{max_retries}: {last_error}", file=sys.stderr)
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Pass C (directions) failed after {max_retries} attempts: {last_error}")


def _extract_directions_with_zoom(
    client,
    model: str,
    image_path: str,
    image_content: dict,
    nodes: list[Node],
    topology: list[dict],
    directions: list[dict],
    use_cache: bool,
    verbose: bool,
    max_retries: int,
) -> list[dict]:
    """Pass C.2: Zoom-verify low-confidence or dense-area edge directions.

    Takes the initial directions from Pass C, identifies edges that need
    zoom verification (low confidence or in dense areas), and runs per-edge
    zoom crops to verify direction.

    Returns updated directions list with corrections applied.
    """
    node_positions = {n.name: (n.x, n.y) for n in nodes}

    # Build a topology trace lookup: (node_a, node_b) -> trace description
    topo_traces: dict[tuple[str, str], str] = {}
    topo_labels: dict[tuple[str, str], str | None] = {}
    for t in topology:
        pair = (min(t["node_a"], t["node_b"]), max(t["node_a"], t["node_b"]))
        topo_traces[pair] = t.get("trace", "")
        topo_labels[pair] = t.get("label")

    # Build a temporary GraphSpec to use _find_dense_edges
    temp_edges = []
    for d in directions:
        temp_edges.append(Edge(
            source=d["source"], target=d["target"],
            label=d.get("label"), directed=d.get("directed", True),
        ))
    temp_graph = GraphSpec(nodes=list(nodes), edges=temp_edges, graph_type="directed")

    # Find edges that need zoom verification
    dense_edges = _find_dense_edges(temp_graph)
    dense_pairs = {(min(e.source, e.target), max(e.source, e.target)) for e in dense_edges}

    # Also include any low-confidence edges
    edges_to_verify: list[int] = []  # indices into directions
    for i, d in enumerate(directions):
        pair = (min(d["source"], d["target"]), max(d["source"], d["target"]))
        if d.get("confidence") == "low" or pair in dense_pairs:
            edges_to_verify.append(i)

    if not edges_to_verify:
        if verbose:
            print("Pass C.2 (zoom): No edges need zoom verification.", file=sys.stderr)
        return directions

    if verbose:
        print(f"Pass C.2 (zoom): Verifying {len(edges_to_verify)} edge(s)...", file=sys.stderr)

    updated = list(directions)
    for idx in edges_to_verify:
        d = directions[idx]
        src, tgt = d["source"], d["target"]
        label = d.get("label")

        if verbose:
            print(f"  Zoom-verifying {src}→{tgt} ({label})...", file=sys.stderr)

        # Create a single-edge object for cropping
        edge_obj = Edge(source=src, target=tgt, label=label, directed=True)
        crop_b64, crop_media, rel_positions = _crop_edge_region(
            image_path, node_positions, [edge_obj],
        )
        crop_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:{crop_media};base64,{crop_b64}",
                "detail": "high",
            },
        }

        src_pos = rel_positions.get(src, (0.5, 0.5))
        tgt_pos = rel_positions.get(tgt, (0.5, 0.5))

        # Build label description for the prompt
        if label:
            label_desc = f"labeled **{label}** "
            label_hint = (
                f"Look for the label \"{label}\" near the line — the line with "
                f"that label is the one you need to examine."
            )
        else:
            label_desc = ""
            label_hint = ""

        # Build trace description from Pass B topology
        pair = (min(src, tgt), max(src, tgt))
        trace = topo_traces.get(pair, "")
        if trace:
            trace_desc = (
                f"\nPath description from topology analysis: \"{trace}\"\n"
                f"Use this description to help identify which physical line "
                f"in the image is this edge."
            )
        else:
            trace_desc = ""

        # Build description of other visible nodes in the crop
        other_nodes = []
        for name, (rx, ry) in sorted(rel_positions.items()):
            if name != src and name != tgt:
                other_nodes.append(f"  {name} is also visible at ({rx:.0%}, {ry:.0%})")
        if other_nodes:
            other_nodes_desc = (
                "\nOther nodes visible in this crop (for reference — "
                "do NOT examine edges to these nodes):\n"
                + "\n".join(other_nodes)
            )
        else:
            other_nodes_desc = ""

        zoom_prompt = ZOOM_VERIFY_PROMPT.format(
            source=src, target=tgt,
            src_x=src_pos[0], src_y=src_pos[1],
            tgt_x=tgt_pos[0], tgt_y=tgt_pos[1],
            label_desc=label_desc,
            label_hint=label_hint,
            trace_desc=trace_desc,
            other_nodes_desc=other_nodes_desc,
        )

        # Cache key per edge
        zoom_ckey = None
        if use_cache:
            zoom_phash = _prompt_hash(zoom_prompt + crop_b64[:64])
            zoom_ckey = _cache_key(image_path, model, zoom_phash, f"pass_c2_zoom_{src}_{tgt}")

        cached_zoom = _cache_get(zoom_ckey) if (use_cache and zoom_ckey) else None
        verdict = None

        if cached_zoom is not None:
            try:
                verdicts = _parse_adjudication_response(cached_zoom)
                if verdicts:
                    verdict = verdicts[0]
            except Exception:
                cached_zoom = None

        if verdict is None:
            try:
                raw = _call_llm(
                    client, model,
                    messages=[
                        {"role": "system", "content": zoom_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": (
                                f"This is a ZOOMED-IN crop showing the region around "
                                f"the edge {label_desc}between {src} and {tgt}. "
                                f"Identify the correct line connecting these two "
                                f"specific nodes and determine which end has the "
                                f"arrowhead."
                            )},
                            crop_content,
                        ]},
                    ],
                    verbose=verbose,
                    temperature=0.1,
                )
                if use_cache and zoom_ckey:
                    _cache_put(zoom_ckey, raw)
                verdicts = _parse_adjudication_response(raw)
                if verdicts:
                    verdict = verdicts[0]
            except Exception as e:
                if verbose:
                    print(f"    Zoom failed ({e}), keeping original.", file=sys.stderr)

        if verdict and verdict.get("confidence") != "low":
            new_src = verdict.get("source", src)
            new_tgt = verdict.get("target", tgt)
            # Validate: the zoom should only return the two nodes of THIS edge.
            # If it returns a node that isn't src or tgt, it saw the wrong edge
            # — discard the verdict.
            expected_nodes = {src, tgt}
            if new_src not in expected_nodes or new_tgt not in expected_nodes:
                if verbose:
                    print(
                        f"    Zoom returned unexpected nodes ({new_src}→{new_tgt}), "
                        f"discarding — likely saw a different edge.",
                        file=sys.stderr,
                    )
                continue

            if new_src != src or new_tgt != tgt:
                if verbose:
                    print(
                        f"    Zoom correction: {src}→{tgt} → {new_src}→{new_tgt}",
                        file=sys.stderr,
                    )
                updated[idx] = {**d, "source": new_src, "target": new_tgt, "confidence": "high"}
            elif verbose:
                print(f"    Zoom confirms: {src}→{tgt}", file=sys.stderr)

    return updated


def _extract_weights(
    client,
    model: str,
    image_content: dict,
    nodes: list[Node],
    directed_edges: list[dict],
    image_path: str,
    use_cache: bool,
    verbose: bool,
    max_retries: int,
) -> tuple[list[dict], str]:
    """Pass D: Extract edge weights and weight_format from the image.

    Takes known nodes and directed edges from Pass C.
    Returns (edge_weights, weight_format) where edge_weights is a list of dicts
    with source, target, label, weight.
    """
    # Format node positions for the prompt
    node_positions = "\n".join(
        f"  {n.name}: ({n.x:.2f}, {n.y:.2f})" for n in sorted(nodes, key=lambda n: n.name)
    )

    # Format edges for the prompt
    edge_lines = []
    for i, e in enumerate(directed_edges, 1):
        label_str = e.get("label") or "unlabeled"
        edge_lines.append(f"  {i}. {e['source']} → {e['target']} (label: {label_str})")
    edge_list = "\n".join(edge_lines)

    prompt = PASS_D_PROMPT.format(node_positions=node_positions, edge_list=edge_list)
    phash = _prompt_hash(prompt)
    ckey = _cache_key(image_path, model, phash, "pass_d_weights") if use_cache else None

    cached = _cache_get(ckey) if (use_cache and ckey) else None
    if cached is not None:
        if verbose:
            print("Pass D (weights): Using cached response.", file=sys.stderr)
        try:
            data = _parse_json_response(cached)
            return data["edges"], data.get("weight_format", "plain")
        except Exception:
            cached = None

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = _call_llm(
                client, model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": (
                            "For each edge, find the weight (number) written alongside "
                            "the line. Also determine if weights use a multiplication "
                            "symbol (×/x). Report each edge's weight."
                        )},
                        image_content,
                    ]},
                ],
                verbose=verbose,
                temperature=0.1,
            )
            data = _parse_json_response(raw)
            weights = data["edges"]
            weight_format = data.get("weight_format", "plain")

            if use_cache and ckey:
                _cache_put(ckey, raw)
            if verbose:
                print(f"Pass D (weights): weight_format={weight_format}", file=sys.stderr)
                for w in weights:
                    print(f"  {w.get('source')}→{w.get('target')}: weight={w.get('weight')}", file=sys.stderr)
            return weights, weight_format
        except Exception as e:
            last_error = str(e)
            if verbose:
                print(f"Pass D attempt {attempt}/{max_retries}: {last_error}", file=sys.stderr)
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Pass D (weights) failed after {max_retries} attempts: {last_error}")


def _compose_graph(
    nodes: list[Node],
    topology: list[dict],
    directions: list[dict],
    weights: list[dict] | None,
    weight_format: str,
) -> GraphSpec:
    """Compose a GraphSpec from the decomposed sub-pass results.

    Merges topology (Pass B), directions (Pass C), and weights (Pass D) into
    final Edge objects with all attributes.
    """
    # Build lookup from (sorted pair, label) → direction info
    dir_lookup: dict[tuple[str, str, str | None], dict] = {}
    for d in directions:
        pair = (min(d["source"], d["target"]), max(d["source"], d["target"]))
        label = d.get("label")
        dir_lookup[(pair[0], pair[1], label)] = d

    # Build lookup from (sorted pair, label) → weight info
    weight_lookup: dict[tuple[str, str, str | None], dict] = {}
    if weights:
        for w in weights:
            src = w.get("source", "")
            tgt = w.get("target", "")
            pair = (min(src, tgt), max(src, tgt))
            label = w.get("label")
            weight_lookup[(pair[0], pair[1], label)] = w

    edges = []
    for topo_edge in topology:
        a, b = topo_edge["node_a"], topo_edge["node_b"]
        label = topo_edge.get("label")
        pair_key = (a, b, label)

        # Get direction from Pass C
        dir_info = dir_lookup.get(pair_key)
        if dir_info:
            source = dir_info["source"]
            target = dir_info["target"]
            directed = dir_info.get("directed", True)
        else:
            # Fallback: try matching by pair only (ignore label mismatch)
            dir_info_nolabel = None
            for dk, dv in dir_lookup.items():
                if dk[0] == a and dk[1] == b:
                    dir_info_nolabel = dv
                    break
            if dir_info_nolabel:
                source = dir_info_nolabel["source"]
                target = dir_info_nolabel["target"]
                directed = dir_info_nolabel.get("directed", True)
            else:
                # Last resort: use alphabetical order
                source, target = a, b
                directed = True

        # Get weight from Pass D
        weight_info = weight_lookup.get(pair_key)
        if not weight_info:
            # Try matching by pair without label
            for wk, wv in weight_lookup.items():
                if wk[0] == a and wk[1] == b:
                    weight_info = wv
                    break
        weight = weight_info.get("weight") if weight_info else None

        edges.append(Edge(
            source=source,
            target=target,
            label=label,
            weight=weight,
            directed=directed,
            color=None,
            curvature=0.0,
        ))

    # Determine graph_type
    any_directed = any(e.directed for e in edges)
    graph_type = "directed" if any_directed else "undirected"

    return GraphSpec(
        nodes=nodes,
        edges=edges,
        title=None,
        graph_type=graph_type,
        weight_format=weight_format,
    )


def extract_graph_decomposed(
    image_path: str,
    model: str = "gpt-4.1",
    verbose: bool = False,
    max_retries: int = 3,
    use_cache: bool = True,
    zoom_directions: bool = True,
) -> GraphSpec:
    """Extract a graph using the decomposed multi-pass pipeline.

    This pipeline breaks extraction into focused sub-passes:
    - Pass A: Node detection (names + positions)
    - Pass B: Edge topology (undirected pairs, alphabetical)
    - Pass C: Edge directions (arrowheads)
    - Pass C.2: Zoom verification for dense/low-confidence edges
    - Pass D: Edge weights + weight_format

    Each pass is a focused LLM call that solves ONE task, reducing errors
    from the monolithic approach where the LLM tries to do everything at once.

    Args:
        image_path: Path to the input image file.
        model: GitHub Models model name.
        verbose: Print progress to stderr.
        max_retries: Retries per pass.
        use_cache: Cache LLM responses.
        zoom_directions: Whether to zoom-verify dense edge directions.

    Returns:
        A GraphSpec object.
    """
    import openai

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise EnvironmentError(
            "GITHUB_TOKEN environment variable is not set.\n"
            "Create a GitHub Personal Access Token with 'models:read' permission:\n"
            "  https://github.com/settings/tokens?type=beta\n"
            "Then: export GITHUB_TOKEN='ghp_your_token_here'"
        )

    client = openai.OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=token,
    )

    if verbose:
        print(f"Loading image: {image_path}", file=sys.stderr)

    b64_data, media_type = _load_and_encode_image(image_path)
    image_content = {
        "type": "image_url",
        "image_url": {
            "url": f"data:{media_type};base64,{b64_data}",
            "detail": "high",
        },
    }

    # --- Pass A: Node detection ---
    if verbose:
        print("=== Pass A: Detecting nodes ===", file=sys.stderr)
    nodes = _extract_nodes(
        client, model, image_content, image_path,
        use_cache, verbose, max_retries,
    )

    # --- Pass B: Topology detection ---
    if verbose:
        print("=== Pass B: Detecting topology (undirected) ===", file=sys.stderr)
    topology = _extract_topology(
        client, model, image_content, nodes, image_path,
        use_cache, verbose, max_retries,
    )

    # --- Pass B.2: Cross-validate topology ---
    if verbose:
        print("=== Pass B.2: Cross-validating topology ===", file=sys.stderr)
    topology = _cross_validate_topology(
        client, model, image_content, nodes, topology, image_path,
        use_cache, verbose, max_retries,
    )

    # --- Pass C: Direction detection ---
    if verbose:
        print("=== Pass C: Detecting edge directions ===", file=sys.stderr)
    directions = _extract_directions(
        client, model, image_content, nodes, topology, image_path,
        use_cache, verbose, max_retries,
    )

    # --- Pass C.2: Zoom verification for dense/low-confidence directions ---
    if zoom_directions:
        if verbose:
            print("=== Pass C.2: Zoom-verifying directions ===", file=sys.stderr)
        directions = _extract_directions_with_zoom(
            client, model, image_path, image_content, nodes, topology,
            directions, use_cache, verbose, max_retries,
        )

    # --- Pass D: Weight detection ---
    # Check if any edges likely have weights (skip for unweighted graphs)
    has_labels = any(e.get("label") for e in topology)
    if verbose:
        print("=== Pass D: Detecting weights ===", file=sys.stderr)
    weights, weight_format = _extract_weights(
        client, model, image_content, nodes, directions, image_path,
        use_cache, verbose, max_retries,
    )

    # Check if all weights are null — if so, it's an unweighted graph
    all_null = all(w.get("weight") is None for w in weights)
    if all_null:
        weights = None
        weight_format = "plain"
        if verbose:
            print("Pass D: All weights null — treating as unweighted graph.", file=sys.stderr)

    # --- Compose final graph ---
    if verbose:
        print("=== Composing final graph ===", file=sys.stderr)
    graph = _compose_graph(nodes, topology, directions, weights, weight_format)

    return _validate_graph(graph, verbose=verbose)


# ---------------------------------------------------------------------------
# Crop-and-zoom edge direction verification
# ---------------------------------------------------------------------------

# Minimum crop dimension in pixels — crops smaller than this are upscaled.
MIN_CROP_DIM = 512

# How close (in fractional image coordinates) two nodes must be for their
# edge to be considered "dense" and worth zoom-verifying.
DENSE_DISTANCE_THRESHOLD = 0.30

# Minimum number of nodes within DENSE_DISTANCE_THRESHOLD of an edge's
# midpoint for the edge to be considered "in a dense region" (not just short).
DENSE_MIN_NEARBY_NODES = 3

ZOOM_VERIFY_PROMPT = """\
You are verifying the direction of a SINGLE arrow in a zoomed-in crop of a
hand-drawn graph diagram.

## The edge to examine

You are looking at the edge {label_desc}connecting node **{source}** and
node **{target}**.
{trace_desc}

## Node positions in this crop

{source} is at approximately ({src_x:.0%}, {src_y:.0%}) in this crop.
{target} is at approximately ({tgt_x:.0%}, {tgt_y:.0%}) in this crop.
{other_nodes_desc}

## How to identify direction

- One end of the line has an arrowhead — a pointed tip shaped like >, V, <,
  or a small triangle. That end is the TARGET (where the arrow points TO).
- The other end is a plain line meeting a node circle. That is the SOURCE
  (where the arrow comes FROM).

## CRITICAL: Identify the CORRECT line

This crop may contain OTHER edges besides the one you are examining.
To find the RIGHT line:
1. Locate node {source} at ({src_x:.0%}, {src_y:.0%}) and node {target} at
   ({tgt_x:.0%}, {tgt_y:.0%}).
2. Find the line that PHYSICALLY CONNECTS these two specific nodes.
   {label_hint}
3. IGNORE any other lines passing through this region that connect to
   different nodes — they are NOT the edge you're examining.
4. Once you've identified the correct line, examine BOTH endpoints to
   determine which end has the arrowhead.

- If a weight (number) is written alongside this specific edge, report it.
- If you cannot clearly see an arrowhead on either end, say confidence "low".

Output your analysis and then a JSON summary inside a ```json code fence:

```json
[
  {{
    "source": "...",
    "target": "...",
    "weight": "...",
    "confidence": "high"
  }}
]
```
"""


def _find_dense_edges(
    graph: GraphSpec,
    threshold: float = DENSE_DISTANCE_THRESHOLD,
    min_nearby: int = DENSE_MIN_NEARBY_NODES,
) -> list[Edge]:
    """Find edges in dense regions of the graph that are worth zoom-verifying.

    An edge qualifies if:
    1. Its endpoints are close together (distance <= threshold), AND
    2. At least one endpoint has ``min_nearby`` other nodes within
       ``threshold * 1.5`` distance (indicating a truly dense/crowded area,
       not just a short edge between isolated nodes).

    These are the edges most likely to have direction errors due to
    small/overlapping arrowheads.
    """
    import math

    pos = {n.name: (n.x, n.y) for n in graph.nodes}

    # Pre-compute how many other nodes are near each node
    # (use a wider radius than the edge-length threshold)
    density_radius = threshold * 1.5
    node_density: dict[str, int] = {}
    for n in graph.nodes:
        if n.name not in pos:
            continue
        nx, ny = pos[n.name]
        count = 0
        for m in graph.nodes:
            if m.name == n.name or m.name not in pos:
                continue
            mx, my = pos[m.name]
            if math.sqrt((nx - mx) ** 2 + (ny - my) ** 2) <= density_radius:
                count += 1
        node_density[n.name] = count

    dense = []
    for e in graph.edges:
        if e.source not in pos or e.target not in pos:
            continue
        sx, sy = pos[e.source]
        tx, ty = pos[e.target]
        dist = math.sqrt((sx - tx) ** 2 + (sy - ty) ** 2)
        if dist > threshold:
            continue

        # At least one endpoint must be in a dense area
        src_dense = node_density.get(e.source, 0) >= min_nearby
        tgt_dense = node_density.get(e.target, 0) >= min_nearby
        if src_dense or tgt_dense:
            dense.append(e)
    return dense


def _crop_edge_region(
    image_path: str,
    node_positions: dict[str, tuple[float, float]],
    edges: list[Edge],
    padding_frac: float = 0.12,
) -> tuple[str, str, dict[str, tuple[float, float]]]:
    """Crop the original image around a set of edges and their nodes.

    For best zoom results, pass a single edge or a small cluster of edges.
    Uses generous padding around the nodes but keeps the crop tight.

    Returns (base64_data, media_type, relative_node_positions) where
    relative_node_positions maps ALL node names visible in the crop to (x, y)
    in the cropped image coordinate system (0-1 fractions).
    """
    img = Image.open(image_path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size

    # Find bounding box of all nodes involved in these edges
    involved_nodes: set[str] = set()
    for e in edges:
        involved_nodes.add(e.source)
        involved_nodes.add(e.target)

    xs = [node_positions[n][0] for n in involved_nodes if n in node_positions]
    ys = [node_positions[n][1] for n in involved_nodes if n in node_positions]

    if not xs or not ys:
        # Fallback: return the full image
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64, "image/jpeg", node_positions

    # Compute padding: ensure we see enough context around the edge
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    # Padding should be proportional to the edge length, with a generous floor
    # to prevent crops that are too tight and miss context
    pad_x = max(padding_frac * max(span_x, 0.08) + 0.04, 0.06)
    pad_y = max(padding_frac * max(span_y, 0.08) + 0.04, 0.06)

    x_min = max(0, min(xs) - pad_x)
    x_max = min(1.0, max(xs) + pad_x)
    y_min = max(0, min(ys) - pad_y)
    y_max = min(1.0, max(ys) + pad_y)

    # Convert to pixels
    px_left = int(x_min * w)
    px_right = int(x_max * w)
    px_top = int(y_min * h)
    px_bottom = int(y_max * h)

    # Crop
    cropped = img.crop((px_left, px_top, px_right, px_bottom))

    # Upscale if the crop is small — we want the arrowheads to be big
    cw, ch = cropped.size
    if max(cw, ch) < MIN_CROP_DIM:
        scale = MIN_CROP_DIM / max(cw, ch)
        new_size = (int(cw * scale), int(ch * scale))
        cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)

    # Compute relative node positions for ALL nodes in the crop bounds
    # (not just the edge's endpoints — this helps the LLM identify which
    # nodes are visible and avoid confusing nearby edges)
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    relative_positions: dict[str, tuple[float, float]] = {}
    for name, (nx, ny) in node_positions.items():
        # Include any node that falls within the crop bounds (with small margin)
        margin = 0.02
        if (x_min - margin <= nx <= x_max + margin and
                y_min - margin <= ny <= y_max + margin):
            rel_x = (nx - x_min) / crop_w if crop_w > 0 else 0.5
            rel_y = (ny - y_min) / crop_h if crop_h > 0 else 0.5
            relative_positions[name] = (rel_x, rel_y)

    # Encode as JPEG
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return b64, "image/jpeg", relative_positions


def _format_edges_for_zoom_verify(
    edges: list[Edge],
    relative_positions: dict[str, tuple[float, float]],
) -> str:
    """Format the edge list for the zoom verification prompt."""
    lines = []
    for i, e in enumerate(edges, 1):
        pos_info = ""
        if e.source in relative_positions:
            sx, sy = relative_positions[e.source]
            pos_info += f" {e.source} is at roughly ({sx:.0%}, {sy:.0%}) in this crop."
        if e.target in relative_positions:
            tx, ty = relative_positions[e.target]
            pos_info += f" {e.target} is at roughly ({tx:.0%}, {ty:.0%}) in this crop."
        lines.append(
            f"{i}. Edge between {e.source} and {e.target}"
            f" (currently extracted as {e.source}→{e.target}"
            f", weight={e.weight}).{pos_info}"
        )
    return "\n".join(lines)


def _apply_zoom_verdicts(
    graph: GraphSpec,
    verdicts: list[dict],
    original_edges: list[Edge],
    verbose: bool = False,
) -> GraphSpec:
    """Apply zoom-verification verdicts to correct edge directions/weights.

    For each verdict, find the matching edge in the graph and update its
    source/target/weight if the verdict disagrees.
    """
    # Build a lookup of original edges by unordered node pair
    edge_lookup: dict[tuple[str, str], list[int]] = {}
    for idx, e in enumerate(graph.edges):
        pair = (min(e.source, e.target), max(e.source, e.target))
        edge_lookup.setdefault(pair, []).append(idx)

    # Also build a set of edges we're verifying (by pair)
    verifying_pairs: set[tuple[str, str]] = set()
    for e in original_edges:
        verifying_pairs.add((min(e.source, e.target), max(e.source, e.target)))

    changes = 0
    new_edges = list(graph.edges)  # shallow copy

    for v in verdicts:
        src = v.get("source")
        tgt = v.get("target")
        weight = v.get("weight")
        confidence = v.get("confidence", "medium")

        if not src or not tgt:
            continue

        # Only apply high/medium confidence verdicts
        if confidence == "low":
            if verbose:
                print(
                    f"  Zoom: skipping low-confidence verdict {src}→{tgt}",
                    file=sys.stderr,
                )
            continue

        pair = (min(src, tgt), max(src, tgt))
        if pair not in verifying_pairs:
            # Verdict for an edge we didn't ask about — ignore
            continue

        indices = edge_lookup.get(pair, [])
        for idx in indices:
            old = new_edges[idx]
            direction_changed = old.source != src or old.target != tgt
            weight_str = str(weight) if weight and weight != "none" else old.weight
            weight_changed = weight_str != old.weight

            if direction_changed or weight_changed:
                new_edges[idx] = old.model_copy(update={
                    "source": src,
                    "target": tgt,
                    "weight": weight_str,
                })
                if verbose:
                    parts = []
                    if direction_changed:
                        parts.append(
                            f"direction {old.source}→{old.target} "
                            f"corrected to {src}→{tgt}"
                        )
                    if weight_changed:
                        parts.append(
                            f"weight {old.weight} corrected to {weight_str}"
                        )
                    print(
                        f"  Zoom: {', '.join(parts)} ({confidence} confidence)",
                        file=sys.stderr,
                    )
                changes += 1
                break  # only fix first matching edge per pair

    if verbose:
        if changes:
            print(f"Zoom verification: {changes} correction(s) applied.", file=sys.stderr)
        else:
            print("Zoom verification: no corrections needed.", file=sys.stderr)

    return graph.model_copy(update={"edges": new_edges})


def _call_llm(
    client,
    model: str,
    messages: list[dict],
    verbose: bool = False,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    response_format: dict | None = None,
) -> str:
    """Make a single LLM API call and return the raw text response.

    Args:
        client: OpenAI client instance.
        model: Model name.
        messages: Chat messages.
        verbose: Print debug info.
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.
        response_format: Optional response format spec (e.g.
            {"type": "json_object"} or {"type": "json_schema", ...}).
    """
    kwargs: dict = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if response_format is not None:
        kwargs["response_format"] = response_format

    response = client.chat.completions.create(**kwargs)
    raw_text = response.choices[0].message.content
    if verbose:
        print(f"LLM response:\n{raw_text}", file=sys.stderr)
    return raw_text


def _run_extraction_pass(
    client,
    model: str,
    image_content: dict,
    system_prompt: str,
    user_text: str,
    pass_name: str,
    image_path: str,
    use_cache: bool,
    verbose: bool,
    max_retries: int,
    temperature: float = 0.1,
) -> GraphSpec:
    """Run a single extraction pass (either pass 1 or pass 2).

    Handles caching and retries. Returns a GraphSpec.
    """
    # Compute cache key
    phash = _prompt_hash(system_prompt)
    ckey: str | None = (
        _cache_key(image_path, model, phash, pass_name) if use_cache else None
    )

    # Check cache
    cached_text = _cache_get(ckey) if (use_cache and ckey) else None
    if cached_text is not None:
        if verbose:
            print(f"{pass_name}: Using cached response.", file=sys.stderr)
        try:
            data = _parse_json_response(cached_text)
            return GraphSpec.model_validate(data)
        except Exception:
            cached_text = None  # stale / corrupt

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            raw_text = _call_llm(
                client,
                model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            image_content,
                        ],
                    },
                ],
                verbose=verbose,
                temperature=temperature,
            )

            data = _parse_json_response(raw_text)
            graph = GraphSpec.model_validate(data)

            # Cache the successful response
            if use_cache and ckey:
                _cache_put(ckey, raw_text)

            if verbose:
                print(
                    f"{pass_name} extracted {len(graph.nodes)} nodes and "
                    f"{len(graph.edges)} edges.",
                    file=sys.stderr,
                )
            return graph

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            if verbose:
                print(
                    f"{pass_name} attempt {attempt}/{max_retries}: {last_error}",
                    file=sys.stderr,
                )
        except Exception as e:
            last_error = str(e)
            if verbose:
                print(
                    f"{pass_name} attempt {attempt}/{max_retries}: {last_error}",
                    file=sys.stderr,
                )
            if "rate" in str(e).lower() or "429" in str(e):
                wait = 2**attempt
                if verbose:
                    print(f"Rate limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)

    raise RuntimeError(
        f"{pass_name} failed after {max_retries} attempts. Last error: {last_error}"
    )


def extract_graph(
    image_path: str,
    model: str = "gpt-4.1",
    verbose: bool = False,
    max_retries: int = 3,
    verify: bool = True,
    use_cache: bool = True,
    pipeline: str = "decomposed",
) -> GraphSpec:
    """Extract a graph specification from a hand-drawn image using LLM vision.

    Supports two pipelines:

    **"decomposed" (default)** — Multi-pass focused extraction:
    - Pass A: Node detection (names + positions)
    - Pass B: Edge topology (undirected, alphabetical pairs)
    - Pass C: Edge directions (arrowheads)
    - Pass C.2: Zoom verification for dense/low-confidence edges
    - Pass D: Edge weights + weight_format

    **"monolithic"** — Legacy cross-validation approach:
    1. Pass 1 (Extract) — independent extraction producing graph_1.
    2. Pass 2 (Re-extract) — second independent extraction.
    3. Diff + Adjudicate disputed edges.
    4. Zoom verification of dense edges.

    Args:
        image_path: Path to the input image file.
        model: GitHub Models model name (e.g., "gpt-4.1").
        verbose: If True, print progress messages to stderr.
        max_retries: Maximum number of retries on failure per pass.
        verify: If True and pipeline="monolithic", run cross-validation.
            Ignored for the decomposed pipeline.
        use_cache: If True, cache LLM responses by image hash to avoid
            repeated API calls. Cache is stored in ~/.cache/graph2svg/.
        pipeline: Pipeline to use: "decomposed" (default) or "monolithic".

    Returns:
        A GraphSpec object representing the extracted graph.

    Raises:
        RuntimeError: If extraction fails after all retries.
        EnvironmentError: If GITHUB_TOKEN is not set.
    """
    if pipeline == "decomposed":
        return extract_graph_decomposed(
            image_path=image_path,
            model=model,
            verbose=verbose,
            max_retries=max_retries,
            use_cache=use_cache,
            zoom_directions=verify,  # use verify flag to control zoom
        )

    # --- Monolithic pipeline (legacy) ---
    # Late import to avoid requiring openai at module level
    import openai

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise EnvironmentError(
            "GITHUB_TOKEN environment variable is not set.\n"
            "Create a GitHub Personal Access Token with 'models:read' permission:\n"
            "  https://github.com/settings/tokens?type=beta\n"
            "Then: export GITHUB_TOKEN='ghp_your_token_here'"
        )

    client = openai.OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=token,
    )

    if verbose:
        print(f"Loading image: {image_path}", file=sys.stderr)

    b64_data, media_type = _load_and_encode_image(image_path)
    image_content = {
        "type": "image_url",
        "image_url": {
            "url": f"data:{media_type};base64,{b64_data}",
            "detail": "high",
        },
    }

    # --- Pass 1: Initial extraction ---
    if verbose:
        print(f"Pass 1: Sending to {model} for analysis...", file=sys.stderr)

    graph1 = _run_extraction_pass(
        client=client,
        model=model,
        image_content=image_content,
        system_prompt=SYSTEM_PROMPT,
        user_text=(
            "Analyze this hand-drawn graph carefully. "
            "Follow the step-by-step process: first identify ALL nodes "
            "with PRECISE positions (as fractions of the image dimensions — "
            "nodes that are close together in the drawing must have close "
            "coordinates), then systematically trace EVERY edge by examining "
            "each node's connections. Pay special attention to dense/overlapping "
            "areas where edges and labels may be hard to distinguish. "
            "Verify your edge count matches the number of edge labels "
            "visible in the image before outputting JSON."
        ),
        pass_name="Pass 1",
        image_path=image_path,
        use_cache=use_cache,
        verbose=verbose,
        max_retries=max_retries,
        temperature=0.1,
    )

    if not verify:
        return _validate_graph(graph1, verbose=verbose)

    # --- Pass 2: Independent re-extraction ---
    if verbose:
        print(f"Pass 2: Independent re-extraction...", file=sys.stderr)

    graph2 = _run_extraction_pass(
        client=client,
        model=model,
        image_content=image_content,
        system_prompt=REEXTRACT_PROMPT,
        user_text=(
            "Analyze this hand-drawn graph from scratch. This is a fresh, "
            "independent extraction — do not rely on any prior analysis. "
            "Be extremely careful with arrow directions: the arrowhead (the "
            "pointed tip, V-shape, or > shape) indicates the TARGET node. "
            "The plain end of the line is the SOURCE. "
            "Trace every edge one by one. Count lines touching each node. "
            "Output the complete JSON."
        ),
        pass_name="Pass 2",
        image_path=image_path,
        use_cache=use_cache,
        verbose=verbose,
        max_retries=max_retries,
        temperature=0.3,  # slightly higher for diversity
    )

    # --- Diff: Find agreed and disputed edges ---
    agreed_edges, disputes = _diff_edge_sets(graph1, graph2, verbose=verbose)

    if not disputes:
        if verbose:
            print(
                "Both passes agree on all edges — no adjudication needed.",
                file=sys.stderr,
            )
        final_graph = _merge_graph(
            graph1, graph2, agreed_edges, [], verbose=verbose,
        )
    else:
        # --- Pass 3: Adjudication of disputed edges ---
        if verbose:
            print(
                f"Pass 3: Adjudicating {len(disputes)} disputed edge(s)...",
                file=sys.stderr,
            )

        node_list = ", ".join(sorted(n.name for n in graph1.nodes))
        dispute_text = _format_disputes_for_prompt(disputes)
        adjudicate_system = ADJUDICATE_PROMPT.format(node_list=node_list) + dispute_text

        # Cache key for adjudication includes the dispute content
        adj_ckey = None
        if use_cache:
            adj_phash = _prompt_hash(adjudicate_system)
            adj_ckey = _cache_key(image_path, model, adj_phash, "adjudicate")

        cached_adj = _cache_get(adj_ckey) if (use_cache and adj_ckey) else None
        verdicts: list[dict] = []

        if cached_adj is not None:
            if verbose:
                print("Pass 3: Using cached response.", file=sys.stderr)
            try:
                verdicts = _parse_adjudication_response(cached_adj)
            except Exception:
                cached_adj = None

        if cached_adj is None:
            try:
                raw_text = _call_llm(
                    client,
                    model,
                    messages=[
                        {"role": "system", "content": adjudicate_system},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Look at the image carefully and resolve each "
                                        "disputed edge. For each dispute, trace the "
                                        "relevant line in the image and determine: "
                                        "(1) does this edge exist? (2) if yes, what is "
                                        "the correct source, target, label, weight, and "
                                        "curvature? Output your verdicts as a JSON array."
                                    ),
                                },
                                image_content,
                            ],
                        },
                    ],
                    verbose=verbose,
                    temperature=0.1,
                )

                if use_cache and adj_ckey:
                    _cache_put(adj_ckey, raw_text)

                verdicts = _parse_adjudication_response(raw_text)

            except Exception as e:
                if verbose:
                    print(
                        f"Adjudication failed ({e}), falling back to Pass 1 result.",
                        file=sys.stderr,
                    )
                return _validate_graph(graph1, verbose=verbose)

        if verbose:
            print(f"Pass 3: Got {len(verdicts)} verdict(s).", file=sys.stderr)

        # --- Merge ---
        final_graph = _merge_graph(
            graph1, graph2, agreed_edges, verdicts, verbose=verbose,
        )

    # --- Pass 4: Zoom verification of dense edges (one edge at a time) ---
    dense_edges = _find_dense_edges(final_graph)
    if dense_edges:
        if verbose:
            print(
                f"Pass 4: Zoom-verifying {len(dense_edges)} edge(s) in dense areas...",
                file=sys.stderr,
            )
            for e in dense_edges:
                print(f"  {e.source}→{e.target} ({e.label})", file=sys.stderr)

        node_positions = {n.name: (n.x, n.y) for n in final_graph.nodes}
        all_zoom_verdicts: list[dict] = []

        for edge_idx, edge in enumerate(dense_edges):
            edge_label = f"{edge.source}→{edge.target} ({edge.label})"
            if verbose:
                print(
                    f"  Pass 4.{edge_idx + 1}: Verifying {edge_label}...",
                    file=sys.stderr,
                )

            # Crop tightly around just this one edge
            crop_b64, crop_media, rel_positions = _crop_edge_region(
                image_path, node_positions, [edge],
            )
            crop_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{crop_media};base64,{crop_b64}",
                    "detail": "high",
                },
            }

            # Build a per-edge prompt with the node positions filled in
            src_pos = rel_positions.get(edge.source, (0.5, 0.5))
            tgt_pos = rel_positions.get(edge.target, (0.5, 0.5))

            # Build label and trace info for the prompt
            if edge.label:
                label_desc = f"labeled **{edge.label}** "
                label_hint = (
                    f"Look for the label \"{edge.label}\" near the line — the line with "
                    f"that label is the one you need to examine."
                )
            else:
                label_desc = ""
                label_hint = ""

            # Build description of other visible nodes in the crop
            other_nodes = []
            for name, (rx, ry) in sorted(rel_positions.items()):
                if name != edge.source and name != edge.target:
                    other_nodes.append(f"  {name} is also visible at ({rx:.0%}, {ry:.0%})")
            if other_nodes:
                other_nodes_desc = (
                    "\nOther nodes visible in this crop (for reference — "
                    "do NOT examine edges to these nodes):\n"
                    + "\n".join(other_nodes)
                )
            else:
                other_nodes_desc = ""

            system_prompt_filled = ZOOM_VERIFY_PROMPT.format(
                source=edge.source,
                target=edge.target,
                src_x=src_pos[0],
                src_y=src_pos[1],
                tgt_x=tgt_pos[0],
                tgt_y=tgt_pos[1],
                label_desc=label_desc,
                label_hint=label_hint,
                trace_desc="",
                other_nodes_desc=other_nodes_desc,
            )

            # Cache key per edge
            zoom_ckey = None
            if use_cache:
                zoom_phash = _prompt_hash(
                    system_prompt_filled + crop_b64[:64]
                )
                zoom_ckey = _cache_key(
                    image_path, model, zoom_phash,
                    f"zoom_{edge.source}_{edge.target}",
                )

            cached_zoom = (
                _cache_get(zoom_ckey)
                if (use_cache and zoom_ckey) else None
            )
            edge_verdicts: list[dict] = []

            if cached_zoom is not None:
                if verbose:
                    print(
                        f"    Using cached response.",
                        file=sys.stderr,
                    )
                try:
                    edge_verdicts = _parse_adjudication_response(cached_zoom)
                except Exception:
                    cached_zoom = None

            if cached_zoom is None:
                try:
                    raw_text = _call_llm(
                        client,
                        model,
                        messages=[
                            {"role": "system", "content": system_prompt_filled},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            "This is a ZOOMED-IN crop showing the "
                                            "region around the edge "
                                            f"{'labeled ' + edge.label + ' ' if edge.label else ''}"
                                            f"between {edge.source} and {edge.target}. "
                                            "Identify the correct line connecting "
                                            "these two specific nodes and determine "
                                            "which end has the arrowhead."
                                        ),
                                    },
                                    crop_content,
                                ],
                            },
                        ],
                        verbose=verbose,
                        temperature=0.1,
                    )

                    if use_cache and zoom_ckey:
                        _cache_put(zoom_ckey, raw_text)

                    edge_verdicts = _parse_adjudication_response(raw_text)

                except Exception as e:
                    if verbose:
                        print(
                            f"    Zoom verification failed ({e}), skipping.",
                            file=sys.stderr,
                        )

            all_zoom_verdicts.extend(edge_verdicts)

        if all_zoom_verdicts:
            if verbose:
                print(
                    f"Pass 4: Got {len(all_zoom_verdicts)} verdict(s) total.",
                    file=sys.stderr,
                )
            final_graph = _apply_zoom_verdicts(
                final_graph, all_zoom_verdicts, dense_edges, verbose=verbose,
            )
    elif verbose:
        print("No dense edges found — skipping zoom verification.", file=sys.stderr)

    return _validate_graph(final_graph, verbose=verbose)

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

from .model import ExtractionResponse, GraphSpec


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
- Its approximate (x, y) position as a fraction of the graph's bounding box:
  (0, 0) = top-left of the graph area, (1, 1) = bottom-right.
  Estimate positions by looking at the node's location relative to the full
  extent of the drawing (not the photo).

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
        img = img.resize(new_size, Image.LANCZOS)

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

If the extraction is correct, output the same JSON unchanged.
If there are errors, output a CORRECTED JSON with a brief comment before the
JSON explaining what you fixed.

Output the JSON inside a ```json code fence.

Here is the extracted JSON to verify:
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


def extract_graph(
    image_path: str,
    model: str = "gpt-4.1",
    verbose: bool = False,
    max_retries: int = 3,
    verify: bool = True,
    use_cache: bool = True,
) -> GraphSpec:
    """Extract a graph specification from a hand-drawn image using LLM vision.

    Uses a two-pass approach:
    1. Initial extraction — analyze the image and produce a JSON graph spec.
    2. Verification pass — send the result back with the image to catch
       missed edges, wrong labels, or other errors.

    Args:
        image_path: Path to the input image file.
        model: GitHub Models model name (e.g., "gpt-4o").
        verbose: If True, print progress messages to stderr.
        max_retries: Maximum number of retries on failure.
        verify: If True, run a verification pass after initial extraction.
        use_cache: If True, cache LLM responses by image hash to avoid
            repeated API calls. Cache is stored in ~/.cache/graph2svg/.

    Returns:
        A GraphSpec object representing the extracted graph.

    Raises:
        RuntimeError: If extraction fails after all retries.
        EnvironmentError: If GITHUB_TOKEN is not set.
    """
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

    # Compute cache keys for both passes
    extract_phash = _prompt_hash(SYSTEM_PROMPT)
    extract_ckey: str | None = _cache_key(image_path, model, extract_phash, "extract") if use_cache else None
    # verify cache key depends on pass-1 result, computed later

    # --- Pass 1: Initial extraction ---
    if verbose:
        print(f"Pass 1: Sending to {model} for analysis...", file=sys.stderr)

    # Check cache first
    graph: GraphSpec | None = None
    cached_text = _cache_get(extract_ckey) if (use_cache and extract_ckey) else None
    if cached_text is not None:
        if verbose:
            print("Pass 1: Using cached response.", file=sys.stderr)
        try:
            data = _parse_json_response(cached_text)
            graph = GraphSpec.model_validate(data)
        except Exception:
            cached_text = None  # cache entry is stale / corrupt

    if cached_text is None:
        last_error = None
        graph = None
        for attempt in range(1, max_retries + 1):
            try:
                raw_text = _call_llm(
                    client,
                    model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Analyze this hand-drawn graph carefully. "
                                        "Follow the step-by-step process: first identify ALL nodes, "
                                        "then systematically trace EVERY edge by examining each node's "
                                        "connections. Pay special attention to dense/overlapping areas "
                                        "where edges and labels may be hard to distinguish. "
                                        "Verify your edge count matches the number of edge labels "
                                        "visible in the image before outputting JSON."
                                    ),
                                },
                                image_content,
                            ],
                        },
                    ],
                    verbose=verbose,
                )

                data = _parse_json_response(raw_text)
                graph = GraphSpec.model_validate(data)

                # Cache the successful response
                if use_cache and extract_ckey:
                    _cache_put(extract_ckey, raw_text)

                if verbose:
                    print(
                        f"Pass 1 extracted {len(graph.nodes)} nodes and "
                        f"{len(graph.edges)} edges.",
                        file=sys.stderr,
                    )
                break

            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
                if verbose:
                    print(
                        f"Attempt {attempt}/{max_retries}: {last_error}",
                        file=sys.stderr,
                    )
            except Exception as e:
                last_error = str(e)
                if verbose:
                    print(
                        f"Attempt {attempt}/{max_retries}: {last_error}",
                        file=sys.stderr,
                    )
                if "rate" in str(e).lower() or "429" in str(e):
                    wait = 2**attempt
                    if verbose:
                        print(f"Rate limited, waiting {wait}s...", file=sys.stderr)
                    time.sleep(wait)

        if graph is None:
            raise RuntimeError(
                f"Failed to extract graph after {max_retries} attempts. "
                f"Last error: {last_error}"
            )

    # At this point graph is guaranteed non-None (either from cache or API)
    assert graph is not None

    # --- Pass 2: Verification ---
    if not verify:
        return _validate_graph(graph, verbose=verbose)

    if verbose:
        print("Pass 2: Verification pass...", file=sys.stderr)

    initial_json = graph.model_dump_json(indent=2)

    # Verification cache key includes hash of the pass-1 result so that
    # a different extraction produces a different verification.
    verify_ckey = None
    if use_cache:
        verify_phash_full = _prompt_hash(VERIFY_PROMPT + initial_json)
        verify_ckey = _cache_key(image_path, model, verify_phash_full, "verify")

    cached_verify = _cache_get(verify_ckey) if (use_cache and verify_ckey) else None
    if cached_verify is not None:
        if verbose:
            print("Pass 2: Using cached response.", file=sys.stderr)
        try:
            data = _parse_json_response(cached_verify)
            verified_graph = GraphSpec.model_validate(data)
            return _validate_graph(verified_graph, verbose=verbose)
        except Exception:
            cached_verify = None  # fall through to API call

    try:
        raw_text = _call_llm(
            client,
            model,
            messages=[
                {
                    "role": "system",
                    "content": VERIFY_PROMPT + initial_json,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Compare the extracted JSON against this image. "
                                "Look for MISSED EDGES (most common error), wrong "
                                "directions, swapped labels, or incorrect weights. "
                                "Output the corrected JSON."
                            ),
                        },
                        image_content,
                    ],
                },
            ],
            verbose=verbose,
        )

        # Cache the successful verification response
        if use_cache and verify_ckey:
            _cache_put(verify_ckey, raw_text)

        data = _parse_json_response(raw_text)
        verified_graph = GraphSpec.model_validate(data)

        if verbose:
            n_diff = len(verified_graph.edges) - len(graph.edges)
            if n_diff != 0:
                print(
                    f"Verification changed edge count by {n_diff:+d} "
                    f"({len(graph.edges)} → {len(verified_graph.edges)}).",
                    file=sys.stderr,
                )
            else:
                print("Verification confirmed extraction.", file=sys.stderr)

        return _validate_graph(verified_graph, verbose=verbose)

    except Exception as e:
        if verbose:
            print(
                f"Verification pass failed ({e}), using initial extraction.",
                file=sys.stderr,
            )
        return _validate_graph(graph, verbose=verbose)

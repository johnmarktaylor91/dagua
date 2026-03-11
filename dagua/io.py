"""Graph construction and export utilities.

from_edges, from_edge_index, from_networkx, from_dict — thin converters
that build Graph instances from various input formats.
to_dot — export to Graphviz DOT format.

JSON import/export: graph_from_json, graph_to_json.
LLM-based construction: graph_from_image, theme_from_image.

Graph.from_* classmethods are thin wrappers over functions here.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from dagua.styles import (
    ClusterStyle, EdgeStyle, GraphStyle, NodeStyle, Theme,
    DEFAULT_THEME_OBJ,
)


# ─── Style dict converters ────────────────────────────────────────────────


def _dict_to_node_style(d: Dict[str, Any]) -> NodeStyle:
    """Convert a dict to NodeStyle, filtering unknown keys and converting padding."""
    valid = {f.name for f in dataclasses.fields(NodeStyle)}
    filtered = {}
    for k, v in d.items():
        if k not in valid:
            continue
        if k == "padding" and isinstance(v, list):
            v = tuple(v)
        if k == "shadow_offset" and isinstance(v, list):
            v = tuple(v)
        filtered[k] = v
    return NodeStyle(**filtered)


def _dict_to_edge_style(d: Dict[str, Any]) -> EdgeStyle:
    """Convert a dict to EdgeStyle, filtering unknown keys."""
    valid = {f.name for f in dataclasses.fields(EdgeStyle)}
    return EdgeStyle(**{k: v for k, v in d.items() if k in valid})


def _dict_to_cluster_style(d: Dict[str, Any]) -> ClusterStyle:
    """Convert a dict to ClusterStyle, filtering unknown keys."""
    valid = {f.name for f in dataclasses.fields(ClusterStyle)}
    filtered = {}
    for k, v in d.items():
        if k not in valid:
            continue
        if k == "label_offset" and isinstance(v, list):
            v = tuple(v)
        filtered[k] = v
    return ClusterStyle(**filtered)


def _dict_to_graph_style(d: Dict[str, Any]) -> GraphStyle:
    """Convert a dict to GraphStyle, filtering unknown keys."""
    valid = {f.name for f in dataclasses.fields(GraphStyle)}
    filtered = {}
    for k, v in d.items():
        if k not in valid:
            continue
        if k in ("max_figsize", "min_figsize") and isinstance(v, list):
            v = tuple(v)
        filtered[k] = v
    return GraphStyle(**filtered)


def _dict_to_theme(d: Dict[str, Any]) -> Theme:
    """Convert a dict to a full Theme object.

    Supports two formats:
    1. Full Theme JSON: {"node_styles": {...}, "edge_styles": {...}, ...}
    2. Legacy flat format: {"default": {...}, "input": {...}} (node styles only)
    """
    # Detect legacy flat format (all values are dicts with style keys)
    if d and "node_styles" not in d and "edge_styles" not in d:
        # Assume legacy: all keys are node type names
        node_styles = {name: _dict_to_node_style(sd) for name, sd in d.items()}
        return Theme(node_styles=node_styles)

    node_styles = {}
    if "node_styles" in d:
        for name, sd in d["node_styles"].items():
            node_styles[name] = _dict_to_node_style(sd)

    edge_styles = {}
    if "edge_styles" in d:
        for name, sd in d["edge_styles"].items():
            edge_styles[name] = _dict_to_edge_style(sd)

    cluster_style = ClusterStyle()
    if "cluster_style" in d:
        cluster_style = _dict_to_cluster_style(d["cluster_style"])

    graph_style = GraphStyle()
    if "graph_style" in d:
        graph_style = _dict_to_graph_style(d["graph_style"])

    name = d.get("name", "custom")
    return Theme(
        name=name,
        node_styles=node_styles,
        edge_styles=edge_styles,
        cluster_style=cluster_style,
        graph_style=graph_style,
    )


# ─── JSON import/export ───────────────────────────────────────────────────


def graph_from_json(data: Union[Dict, str, Path]) -> "DaguaGraph":
    """Build a DaguaGraph from JSON data.

    Args:
        data: A dict, a JSON string, or a file path (str or Path) ending in .json.

    Returns:
        A fully constructed DaguaGraph.
    """
    from dagua.graph import DaguaGraph

    if isinstance(data, (str, Path)):
        s = str(data)
        # Treat as file path if it ends with .json or the file exists
        if s.endswith(".json") or (os.path.exists(s) and not s.strip().startswith("{")):
            with open(s) as f:
                data = json.load(f)
        else:
            data = json.loads(s)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, JSON string, or file path, got {type(data).__name__}")

    # Build theme from top-level "theme" key
    theme = None
    if "theme" in data:
        theme = _dict_to_theme(data["theme"])

    g = DaguaGraph(
        direction=data.get("direction", "TB"),
        _theme=theme if theme is not None else None,
    )
    # If no custom theme was provided, use the default
    if not isinstance(g._theme, Theme):
        from dagua.styles import DEFAULT_THEME
        g._theme = Theme(node_styles=dict(DEFAULT_THEME))

    # 1. Add nodes
    for node_data in data.get("nodes", []):
        node_id = node_data["id"]
        label = node_data.get("label")
        node_type = node_data.get("type", "default")
        style = _dict_to_node_style(node_data["style"]) if "style" in node_data else None
        g.add_node(node_id, label=label, type=node_type, style=style)

    # 2. Add edges
    for edge_data in data.get("edges", []):
        source = edge_data["source"]
        target = edge_data["target"]
        label = edge_data.get("label")
        style = _dict_to_edge_style(edge_data["style"]) if "style" in edge_data else None
        g.add_edge(source, target, label=label, style=style)

    # 3. Add clusters
    for cluster_data in data.get("clusters", []):
        name = cluster_data["name"]
        members = cluster_data.get("members", [])
        label = cluster_data.get("label")
        style = _dict_to_cluster_style(cluster_data["style"]) if "style" in cluster_data else None
        g.add_cluster(name, members, label=label, style=style)

    # 4. Restore back edge mask (cycle support)
    if "back_edges" in data:
        import torch
        g._finalize_edges()
        E = g._edge_index_tensor.shape[1] if g._edge_index_tensor is not None else 0
        if E > 0:
            mask = torch.zeros(E, dtype=torch.bool)
            for idx in data["back_edges"]:
                if 0 <= idx < E:
                    mask[idx] = True
            if mask.any():
                g._back_edge_mask = mask

    return g


def graph_to_json(graph: "DaguaGraph") -> Dict[str, Any]:
    """Serialize a DaguaGraph to a JSON-compatible dict.

    Only includes non-default style fields to keep output compact.
    Serializes Theme if non-default.
    """
    result: Dict[str, Any] = {}

    if graph.direction != "TB":
        result["direction"] = graph.direction

    # Nodes
    _node_defaults = NodeStyle()
    _node_default_dict = dataclasses.asdict(_node_defaults)
    index_to_id = {v: k for k, v in graph._id_to_index.items()}

    nodes = []
    for i in range(graph.num_nodes):
        node: Dict[str, Any] = {"id": index_to_id.get(i, str(i))}
        if i < len(graph.node_labels):
            label = graph.node_labels[i]
            if label != node["id"]:
                node["label"] = label
        if i < len(graph.node_types) and graph.node_types[i] != "default":
            node["type"] = graph.node_types[i]
        if i < len(graph.node_styles) and graph.node_styles[i] is not None:
            style_dict = _style_to_diff_dict(graph.node_styles[i], _node_default_dict)
            if style_dict:
                node["style"] = style_dict
        nodes.append(node)
    if nodes:
        result["nodes"] = nodes

    # Edges
    graph._finalize_edges()
    _edge_defaults = EdgeStyle()
    _edge_default_dict = dataclasses.asdict(_edge_defaults)

    edges = []
    ei = graph._edge_index_tensor
    if ei is not None and ei.numel() > 0:
        for j in range(ei.shape[1]):
            src_idx = ei[0, j].item()
            tgt_idx = ei[1, j].item()
            edge: Dict[str, Any] = {
                "source": index_to_id.get(src_idx, str(src_idx)),
                "target": index_to_id.get(tgt_idx, str(tgt_idx)),
            }
            if j < len(graph.edge_labels) and graph.edge_labels[j] is not None:
                edge["label"] = graph.edge_labels[j]
            if j < len(graph.edge_styles) and graph.edge_styles[j] is not None:
                style_dict = _style_to_diff_dict(graph.edge_styles[j], _edge_default_dict)
                if style_dict:
                    edge["style"] = style_dict
            edges.append(edge)
    if edges:
        result["edges"] = edges

    # Clusters
    _cluster_defaults = ClusterStyle()
    _cluster_default_dict = dataclasses.asdict(_cluster_defaults)

    clusters = []
    for name, members in graph.clusters.items():
        cluster: Dict[str, Any] = {"name": name}
        # Convert indices back to IDs
        if isinstance(members, list):
            cluster["members"] = [index_to_id.get(m, str(m)) for m in members]
        else:
            cluster["members"] = members
        if name in graph.cluster_labels:
            cluster["label"] = graph.cluster_labels[name]
        if name in graph.cluster_styles:
            style_dict = _style_to_diff_dict(graph.cluster_styles[name], _cluster_default_dict)
            if style_dict:
                cluster["style"] = style_dict
        clusters.append(cluster)
    if clusters:
        result["clusters"] = clusters

    # Back edges (cycle support)
    if graph._back_edge_mask is not None and graph._back_edge_mask.any():
        result["back_edges"] = graph._back_edge_mask.nonzero(as_tuple=False).squeeze(1).tolist()

    # Theme (serialize only non-default sections)
    if isinstance(graph._theme, Theme):
        theme_dict = _theme_to_json(graph._theme)
        if theme_dict:
            result["theme"] = theme_dict

    return result


def _theme_to_json(theme: Theme) -> Dict[str, Any]:
    """Serialize a Theme to a JSON dict, only including non-default sections."""
    result: Dict[str, Any] = {}

    _node_defaults = NodeStyle()
    _node_default_dict = dataclasses.asdict(_node_defaults)
    _edge_defaults = EdgeStyle()
    _edge_default_dict = dataclasses.asdict(_edge_defaults)
    _cluster_defaults = ClusterStyle()
    _cluster_default_dict = dataclasses.asdict(_cluster_defaults)
    _graph_defaults = GraphStyle()
    _graph_default_dict = dataclasses.asdict(_graph_defaults)

    if theme.node_styles:
        ns = {}
        for name, style in theme.node_styles.items():
            diff = _style_to_diff_dict(style, _node_default_dict)
            if diff:
                ns[name] = diff
        if ns:
            result["node_styles"] = ns

    if theme.edge_styles:
        es = {}
        for name, style in theme.edge_styles.items():
            diff = _style_to_diff_dict(style, _edge_default_dict)
            if diff:
                es[name] = diff
        if es:
            result["edge_styles"] = es

    cluster_diff = _style_to_diff_dict(theme.cluster_style, _cluster_default_dict)
    if cluster_diff:
        result["cluster_style"] = cluster_diff

    graph_diff = _style_to_diff_dict(theme.graph_style, _graph_default_dict)
    if graph_diff:
        result["graph_style"] = graph_diff

    if theme.name not in ("default", "custom"):
        result["name"] = theme.name

    return result


def _style_to_diff_dict(style: Any, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Return only fields that differ from defaults."""
    current = dataclasses.asdict(style)
    diff = {}
    for k, v in current.items():
        if k.startswith("_") or k == "LEVEL_FILLS" or k == "LEVEL_STROKES":
            continue
        default_v = defaults.get(k)
        if v != default_v:
            # Convert tuples to lists for JSON serialization
            if isinstance(v, tuple):
                v = list(v)
            diff[k] = v
    return diff


# ─── LLM infrastructure ──────────────────────────────────────────────────

_GRAPH_EXTRACTION_PROMPT = """\
You are a graph visualization expert. I will give you a picture of a graph \
(flowchart, DAG, network diagram, etc.). Your job is to produce a JSON object \
that describes the graph structure, preserving connectivity, labels, colors, \
grouping, and flow direction.

## Output Format

Return ONLY a JSON object (no explanation, no code fences) with this schema:

{
  "direction": "TB",
  "nodes": [
    {"id": "unique_id", "label": "Display Label", "type": "default", \
"style": {"shape": "roundrect", "base_color": "#56B4E9"}}
  ],
  "edges": [
    {"source": "id1", "target": "id2", "label": "optional", \
"style": {"color": "#8C8C8C", "style": "dashed"}}
  ],
  "clusters": [
    {"name": "group_name", "members": ["id1", "id2"], \
"label": "Group Label", "style": {"fill": "#E5E5E0"}}
  ],
  "theme": {
    "node_styles": {
      "default": {"base_color": "#56B4E9"},
      "input": {"base_color": "#009E73"}
    },
    "edge_styles": {
      "default": {"color": "#8C8C8C"}
    },
    "graph_style": {
      "background_color": "#FAFAFA"
    }
  }
}

## Rules

1. **Structure first**: Identify all nodes and directed edges. Preserve the exact connectivity.
2. **Labels**: Use the exact text shown on each node. If no label is visible, use a short \
descriptive ID.
3. **Flow direction**: Determine if the graph flows top-to-bottom (TB), bottom-to-top (BT), \
left-to-right (LR), or right-to-left (RL).
4. **Colors**: Match node colors as closely as possible using hex codes. \
Use the base_color field and fill/stroke will be auto-derived. \
Colorblind-safe palette available: sky=#56B4E9, vermillion=#D55E00, \
bluish_green=#009E73, amber=#E69F00, reddish_purple=#CC79A7, blue=#0072B2, yellow=#F0E442.
5. **Shapes**: Match node shapes — rect, roundrect, ellipse, diamond, circle.
6. **Grouping**: If nodes are visually grouped (boxes around groups, shared background), \
add cluster entries.
7. **Edge styles**: If edges have labels, colors, or dashed/dotted styles, include style dicts.
8. **Node types**: Use "input" for input nodes, "output" for output nodes, "default" otherwise.
9. All fields except node "id" and edge "source"/"target" are optional — omit defaults.

Return ONLY the JSON object."""

_THEME_EXTRACTION_PROMPT = """\
You are a design expert. I will give you a picture of a graph visualization. \
Analyze its visual aesthetics and extract a theme specification as a JSON object.

## Output Format

Return ONLY a JSON object (no explanation, no code fences) with this schema:

{
  "name": "custom",
  "node_styles": {
    "default": {"shape": "roundrect", "base_color": "#56B4E9", "font_size": 9.0, \
"font_weight": "regular"},
    "input": {"shape": "ellipse", "base_color": "#009E73"},
    "output": {"base_color": "#D55E00"}
  },
  "edge_styles": {
    "default": {"color": "#666666", "width": 1.0, "style": "solid", "routing": "bezier"}
  },
  "cluster_style": {"fill": "#F0F0F0", "stroke": "#CCCCCC", "corner_radius": 7.0},
  "graph_style": {
    "background_color": "#FFFFFF",
    "margin": 30.0,
    "title_font_size": 10.0
  }
}

## Rules

1. Extract the dominant node shape, colors, and typography from the image.
2. For each distinct node category you see (input, output, default, etc.), create an entry \
in node_styles with the appropriate base_color (hex).
3. Extract edge color, width, and style (solid/dashed/dotted).
4. If clusters/groups are present, extract their fill and stroke colors.
5. Determine the background color and other graph-level settings.
6. All fields are optional — only include what you can confidently determine.

Return ONLY the JSON object."""


def _get_llm_client(
    provider: Optional[str] = None,
) -> Tuple[str, Any]:
    """Auto-detect and return (provider_name, client) from environment.

    Checks ANTHROPIC_API_KEY, then OPENAI_API_KEY.
    """
    if provider is None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise RuntimeError(
                "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, "
                "and install the corresponding SDK: pip install 'dagua[ai]'"
            )

    if provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic SDK not installed. Run: pip install 'dagua[ai]'"
            )
        return "anthropic", anthropic.Anthropic()

    if provider == "openai":
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai SDK not installed. Run: pip install 'dagua[ai]'"
            )
        return "openai", openai.OpenAI()

    raise ValueError(f"Unknown provider: {provider!r}. Use 'anthropic' or 'openai'.")


def _send_image_to_llm(
    client: Any,
    provider: str,
    image_path: str,
    prompt: str,
    model: Optional[str] = None,
) -> str:
    """Send an image + prompt to an LLM and return the text response."""
    # Read and base64-encode the image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Detect media type from extension
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(ext, "image/png")

    if provider == "anthropic":
        model = model or "claude-sonnet-4-20250514"
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.content[0].text

    if provider == "openai":
        model = model or "gpt-4o"
        response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.choices[0].message.content

    raise ValueError(f"Unknown provider: {provider!r}")


def _extract_json_from_response(text: str) -> Dict[str, Any]:
    """Extract a JSON object from LLM response text.

    Handles: pure JSON, ```json code fences, preamble text before JSON.
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code fences
    import re
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    brace_start = text.find("{")
    if brace_start >= 0:
        # Find matching closing brace
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"Could not extract valid JSON from LLM response:\n{text[:200]}")


# ─── Public LLM-based functions ───────────────────────────────────────────


def graph_from_image(
    image_path: Union[str, Path],
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> "DaguaGraph":
    """Construct a DaguaGraph from an image using an LLM.

    Sends the image to an LLM (Anthropic or OpenAI), which returns a JSON
    description of the graph structure. The JSON is then passed to
    graph_from_json() to build the graph.

    Args:
        image_path: Path to the graph image (PNG, JPG, etc.).
        provider: 'anthropic' or 'openai'. Auto-detected from env vars if None.
        model: Model name override. Defaults to claude-sonnet-4-20250514 or gpt-4o.

    Returns:
        A DaguaGraph reconstructed from the image.
    """
    provider_name, client = _get_llm_client(provider)
    response_text = _send_image_to_llm(
        client, provider_name, str(image_path), _GRAPH_EXTRACTION_PROMPT, model
    )
    graph_dict = _extract_json_from_response(response_text)
    return graph_from_json(graph_dict)


def theme_from_image(
    image_path: Union[str, Path],
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Theme:
    """Extract a visual theme from a graph image using an LLM.

    Returns a Theme object containing node_styles, edge_styles,
    cluster_style, and graph_style extracted from the image.

    Args:
        image_path: Path to the graph image (PNG, JPG, etc.).
        provider: 'anthropic' or 'openai'. Auto-detected from env vars if None.
        model: Model name override.

    Returns:
        A Theme object with styles extracted from the image.
    """
    provider_name, client = _get_llm_client(provider)
    response_text = _send_image_to_llm(
        client, provider_name, str(image_path), _THEME_EXTRACTION_PROMPT, model
    )
    raw = _extract_json_from_response(response_text)
    return _dict_to_theme(raw)

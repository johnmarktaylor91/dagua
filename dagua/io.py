"""Graph construction and export utilities.

from_edges, from_edge_index, from_networkx, from_dict — thin converters
that build Graph instances from various input formats.
to_dot — export to Graphviz DOT format.

JSON/YAML import/export: graph_from_json, graph_to_json, graph_from_yaml, graph_to_yaml.
Unified API: load(), save() — auto-detect format from file extension.
LLM-based construction: graph_from_image, theme_from_image.

Graph.from_* classmethods are thin wrappers over functions here.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dagua.styles import (
    ClusterStyle, EdgeStyle, GraphStyle, NodeStyle, Theme,
    DEFAULT_THEME_OBJ,
)


_DIRECT_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_SVG_COMPAT_IMAGE_EXTENSIONS = {".svg"}

_IMAGE_AI_PROVIDER_ENV_KEYS = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
}

_IMAGE_AI_DEFAULT_MODEL = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
}

_PIL_COMPAT_IMAGE_EXTENSIONS = {
    ".bmp",
    ".dib",
    ".gif",
    ".jfif",
    ".jpe",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


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
    """Convert a dict to ClusterStyle, filtering unknown keys.

    Handles member_node_style and member_edge_style as nested dicts.
    """
    valid = {f.name for f in dataclasses.fields(ClusterStyle)}
    filtered = {}
    for k, v in d.items():
        if k not in valid:
            continue
        if k == "label_offset" and isinstance(v, list):
            v = tuple(v)
        if k == "member_node_style" and isinstance(v, dict):
            v = _dict_to_node_style(v)
        if k == "member_edge_style" and isinstance(v, dict):
            v = _dict_to_edge_style(v)
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


# ─── Flex dict converter ─────────────────────────────────────────────────


def _dict_to_flex(d: Dict[str, Any]) -> "Flex":
    """Convert a dict to a Flex value."""
    from dagua.flex import Flex
    if isinstance(d, (int, float)):
        return Flex(target=float(d))
    target = float(d.get("target", 0.0))
    weight = float(d.get("weight", 1.0))
    return Flex(target=target, weight=weight)


def _dict_to_layout_flex(d: Dict[str, Any]) -> "LayoutFlex":
    """Convert a dict to LayoutFlex."""
    from dagua.flex import AlignGroup, Flex, LayoutFlex

    node_sep = _dict_to_flex(d["node_sep"]) if "node_sep" in d else None
    rank_sep = _dict_to_flex(d["rank_sep"]) if "rank_sep" in d else None

    pins = None
    if "pins" in d and isinstance(d["pins"], dict):
        pins = {}
        for node_id, pin_data in d["pins"].items():
            if isinstance(pin_data, dict):
                fx = Flex(target=float(pin_data["x"]), weight=float(pin_data.get("weight", float("inf")))) if "x" in pin_data else None
                fy = Flex(target=float(pin_data["y"]), weight=float(pin_data.get("weight", float("inf")))) if "y" in pin_data else None
                pins[node_id] = (fx, fy)

    align_x = None
    if "align_x" in d and isinstance(d["align_x"], list):
        align_x = []
        for group_data in d["align_x"]:
            nodes = group_data.get("nodes", [])
            weight = float(group_data.get("weight", 5.0))
            align_x.append(AlignGroup(nodes=nodes, weight=weight))

    align_y = None
    if "align_y" in d and isinstance(d["align_y"], list):
        align_y = []
        for group_data in d["align_y"]:
            nodes = group_data.get("nodes", [])
            weight = float(group_data.get("weight", 5.0))
            align_y.append(AlignGroup(nodes=nodes, weight=weight))

    return LayoutFlex(
        node_sep=node_sep,
        rank_sep=rank_sep,
        pins=pins,
        align_x=align_x,
        align_y=align_y,
    )


def _layout_flex_to_dict(flex) -> Dict[str, Any]:
    """Serialize a LayoutFlex to a dict."""
    result: Dict[str, Any] = {}
    if flex.node_sep is not None:
        result["node_sep"] = {"target": flex.node_sep.target, "weight": flex.node_sep.weight}
    if flex.rank_sep is not None:
        result["rank_sep"] = {"target": flex.rank_sep.target, "weight": flex.rank_sep.weight}
    if flex.pins:
        pins_dict = {}
        for node_id, (fx, fy) in flex.pins.items():
            pin_data: Dict[str, Any] = {}
            if fx is not None:
                pin_data["x"] = fx.target
                pin_data["weight"] = fx.weight
            if fy is not None:
                pin_data["y"] = fy.target
                if "weight" not in pin_data:
                    pin_data["weight"] = fy.weight
            pins_dict[str(node_id)] = pin_data
        result["pins"] = pins_dict
    if flex.align_x:
        result["align_x"] = [{"nodes": g.nodes, "weight": g.weight} for g in flex.align_x]
    if flex.align_y:
        result["align_y"] = [{"nodes": g.nodes, "weight": g.weight} for g in flex.align_y]
    return result


# ─── Style-only load/save ────────────────────────────────────────────────


def load_style(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a style-only file (YAML/JSON) into a dict.

    Returns a dict with optional keys: node_style, edge_style, theme, flex.
    """
    p = str(path)
    ext = Path(p).suffix.lower()

    if ext in (".yaml", ".yml"):
        yaml = _ensure_yaml()
        with open(p) as f:
            return yaml.safe_load(f) or {}
    else:
        with open(p) as f:
            return json.load(f)


def save_style(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save a style dict to a YAML/JSON file."""
    p = str(path)
    ext = Path(p).suffix.lower()

    if ext in (".yaml", ".yml"):
        yaml = _ensure_yaml()
        with open(p, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    else:
        with open(p, "w") as f:
            json.dump(config, f, indent=2)


# ─── Core dict → graph converter ──────────────────────────────────────────


def _graph_from_dict(data: Dict) -> "DaguaGraph":
    """Build a DaguaGraph from a plain dict (shared by JSON and YAML paths).

    Theme resolution: if ``theme`` is a string, look up a built-in theme by name.
    If it is a dict, parse it into a Theme object.
    """
    from dagua.graph import DaguaGraph

    # Build theme from top-level "theme" key
    theme = None
    theme_val = data.get("theme")
    if isinstance(theme_val, str):
        from dagua.styles import get_theme
        theme = get_theme(theme_val)
    elif isinstance(theme_val, dict):
        theme = _dict_to_theme(theme_val)

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
        parent = cluster_data.get("parent")
        style = _dict_to_cluster_style(cluster_data["style"]) if "style" in cluster_data else None
        g.add_cluster(name, members, label=label, style=style, parent=parent)

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

    # 5. Parse graph-level defaults
    defaults_data = data.get("defaults")
    if isinstance(defaults_data, dict):
        if "node_style" in defaults_data:
            g.default_node_style = _dict_to_node_style(defaults_data["node_style"])
        if "edge_style" in defaults_data:
            g.default_edge_style = _dict_to_edge_style(defaults_data["edge_style"])

    # 6. Parse flex constraints
    flex_data = data.get("flex")
    if isinstance(flex_data, dict):
        g.flex = _dict_to_layout_flex(flex_data)

    return g


# ─── JSON import/export ───────────────────────────────────────────────────


def graph_from_json(data: Union[Dict, str, Path]) -> "DaguaGraph":
    """Build a DaguaGraph from JSON data.

    Args:
        data: A dict, a JSON string, or a file path (str or Path) ending in .json.

    Returns:
        A fully constructed DaguaGraph.
    """
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

    return _graph_from_dict(data)


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
        if hasattr(graph, 'cluster_parents') and name in graph.cluster_parents and graph.cluster_parents[name] is not None:
            cluster["parent"] = graph.cluster_parents[name]
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

    # Graph-level defaults
    defaults_dict: Dict[str, Any] = {}
    if getattr(graph, "default_node_style", None) is not None:
        diff = _style_to_diff_dict(graph.default_node_style, _node_default_dict)
        if diff:
            defaults_dict["node_style"] = diff
    if getattr(graph, "default_edge_style", None) is not None:
        diff = _style_to_diff_dict(graph.default_edge_style, _edge_default_dict)
        if diff:
            defaults_dict["edge_style"] = diff
    if defaults_dict:
        result["defaults"] = defaults_dict

    # Flex constraints
    if getattr(graph, "flex", None) is not None:
        flex_dict = _layout_flex_to_dict(graph.flex)
        if flex_dict:
            result["flex"] = flex_dict

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


# ─── YAML import/export ───────────────────────────────────────────────────


def _ensure_yaml():
    """Import and return the yaml module, with a clear error if missing."""
    try:
        import yaml  # type: ignore[import-untyped]
        return yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML support. Install it with: "
            "pip install 'dagua[yaml]'"
        )


def graph_from_yaml(data: Union[str, Path]) -> "DaguaGraph":
    """Build a DaguaGraph from YAML data.

    Args:
        data: A YAML string or a file path (str or Path) to a .yaml/.yml file.

    Returns:
        A fully constructed DaguaGraph.
    """
    yaml = _ensure_yaml()

    s = str(data)
    if s.endswith((".yaml", ".yml")) or (
        os.path.exists(s) and not s.strip().startswith("{")
    ):
        with open(s) as f:
            parsed = yaml.safe_load(f)
    else:
        parsed = yaml.safe_load(s)

    if not isinstance(parsed, dict):
        raise TypeError(f"Expected YAML mapping, got {type(parsed).__name__}")

    return _graph_from_dict(parsed)


def graph_to_yaml(graph: "DaguaGraph", path: Optional[Union[str, Path]] = None) -> str:
    """Serialize a DaguaGraph to YAML.

    Args:
        graph: The graph to serialize.
        path: Optional file path to write to. If None, returns YAML string.

    Returns:
        YAML string representation of the graph.
    """
    yaml = _ensure_yaml()

    data = graph_to_json(graph)  # reuse the same dict serialization
    yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)

    if path is not None:
        with open(str(path), "w") as f:
            f.write(yaml_str)

    return yaml_str


# ─── Unified load/save API ────────────────────────────────────────────────


def load(source: Union[Dict, str, Path]) -> "DaguaGraph":
    """Load a DaguaGraph from a file (YAML/JSON), dict, or string.

    Format auto-detection:
    - dict: parsed directly
    - Path ending in .yaml/.yml: YAML
    - Path ending in .json: JSON
    - String starting with '{': JSON
    - Otherwise: try YAML parse (falls back to JSON)

    Args:
        source: A dict, file path, JSON string, or YAML string.

    Returns:
        A fully constructed DaguaGraph.
    """
    if isinstance(source, dict):
        return _graph_from_dict(source)

    s = str(source)

    # File path detection by extension
    if s.endswith((".yaml", ".yml")):
        return graph_from_yaml(s)
    if s.endswith(".json"):
        return graph_from_json(s)

    # File path detection by existence
    if os.path.exists(s):
        ext = Path(s).suffix.lower()
        if ext in (".yaml", ".yml"):
            return graph_from_yaml(s)
        return graph_from_json(s)

    # String content detection
    stripped = s.strip()
    if stripped.startswith("{"):
        return graph_from_json(s)

    # Try YAML parse (YAML is a superset of JSON, so this handles both)
    yaml = _ensure_yaml()
    parsed = yaml.safe_load(s)
    if isinstance(parsed, dict):
        return _graph_from_dict(parsed)

    raise TypeError(f"Cannot load graph from: {type(source).__name__}")


def save(
    graph: "DaguaGraph",
    path: Union[str, Path],
    format: Optional[str] = None,
) -> None:
    """Save a DaguaGraph to a file (YAML or JSON).

    Format auto-detected from file extension. Defaults to YAML when ambiguous.

    Args:
        graph: The graph to save.
        path: Output file path.
        format: Force format — "yaml" or "json". Auto-detected from extension if None.
    """
    p = str(path)

    if format is None:
        ext = Path(p).suffix.lower()
        if ext == ".json":
            format = "json"
        elif ext in (".yaml", ".yml"):
            format = "yaml"
        else:
            format = "yaml"  # default

    data = graph_to_json(graph)

    if format == "json":
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
    elif format == "yaml":
        yaml = _ensure_yaml()
        with open(p, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    else:
        raise ValueError(f"Unknown format: {format!r}. Use 'yaml' or 'json'.")


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
    {"name": "outer_group", "members": ["id1", "id2", "id3"], \
"label": "Outer Group", "style": {"fill": "#E5E5E0"}},
    {"name": "inner_group", "members": ["id2", "id3"], \
"parent": "outer_group", "label": "Inner Group"}
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
6. **Grouping**: If nodes are visually grouped (boxes, shared background, nested regions), \
add cluster entries. Use `parent` for nested groups — add the outermost group first, \
then inner groups with `parent` referencing the outer.
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


# ─── Export functions ──────────────────────────────────────────────────────


@dataclasses.dataclass
class ImageAIConfig:
    """Configuration for image-to-graph/theme AI calls."""

    provider: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None


_IMAGE_AI_CONFIG = ImageAIConfig()


def to_networkx(graph: "DaguaGraph") -> Any:
    """Export a DaguaGraph to a NetworkX DiGraph.

    Node attributes: label, type, cluster (deepest cluster name or None).
    Edge attributes: label, type.

    Requires: ``pip install networkx``
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "NetworkX is required for to_networkx(). Install it with: "
            "pip install networkx"
        )

    G = nx.DiGraph()

    # Reverse mapping for cluster membership (deepest cluster per node)
    node_cluster: Dict[int, Optional[str]] = {i: None for i in range(graph.num_nodes)}
    if graph.clusters:
        node_depth: Dict[int, int] = {}
        for name, members in graph.clusters.items():
            if not isinstance(members, list):
                continue
            depth = graph.cluster_depth(name)
            for idx in members:
                if idx < graph.num_nodes and depth > node_depth.get(idx, -1):
                    node_cluster[idx] = name
                    node_depth[idx] = depth

    index_to_id = {v: k for k, v in graph._id_to_index.items()}
    for i in range(graph.num_nodes):
        node_id = index_to_id.get(i, i)
        attrs: Dict[str, Any] = {
            "label": graph.node_labels[i] if i < len(graph.node_labels) else str(i),
            "type": graph.node_types[i] if i < len(graph.node_types) else "default",
        }
        if node_cluster[i] is not None:
            attrs["cluster"] = node_cluster[i]
        G.add_node(node_id, **attrs)

    graph._finalize_edges()
    ei = graph._edge_index_tensor
    if ei is not None and ei.numel() > 0:
        for e in range(ei.shape[1]):
            src = index_to_id.get(ei[0, e].item(), ei[0, e].item())
            tgt = index_to_id.get(ei[1, e].item(), ei[1, e].item())
            attrs = {}
            if e < len(graph.edge_labels) and graph.edge_labels[e] is not None:
                attrs["label"] = graph.edge_labels[e]
            if e < len(graph.edge_types):
                attrs["type"] = graph.edge_types[e]
            G.add_edge(src, tgt, **attrs)

    return G


def to_igraph(graph: "DaguaGraph") -> Any:
    """Export a DaguaGraph to an igraph.Graph.

    Vertex attributes: name (original ID), label, type.
    Edge attributes: label, type.

    Requires: ``pip install igraph``
    """
    try:
        import igraph
    except ImportError:
        raise ImportError(
            "igraph is required for to_igraph(). Install it with: "
            "pip install igraph"
        )

    index_to_id = {v: k for k, v in graph._id_to_index.items()}

    g = igraph.Graph(directed=True)
    g.add_vertices(graph.num_nodes)

    # Vertex attributes
    g.vs["name"] = [str(index_to_id.get(i, i)) for i in range(graph.num_nodes)]
    g.vs["label"] = [
        graph.node_labels[i] if i < len(graph.node_labels) else str(i)
        for i in range(graph.num_nodes)
    ]
    g.vs["type"] = [
        graph.node_types[i] if i < len(graph.node_types) else "default"
        for i in range(graph.num_nodes)
    ]

    # Edges
    graph._finalize_edges()
    ei = graph._edge_index_tensor
    if ei is not None and ei.numel() > 0:
        edges = [(ei[0, e].item(), ei[1, e].item()) for e in range(ei.shape[1])]
        g.add_edges(edges)

        labels = []
        types = []
        for e in range(ei.shape[1]):
            labels.append(
                graph.edge_labels[e] if e < len(graph.edge_labels) else None
            )
            types.append(
                graph.edge_types[e] if e < len(graph.edge_types) else "normal"
            )
        g.es["label"] = labels
        g.es["type"] = types

    return g


def to_pyg(graph: "DaguaGraph") -> Any:
    """Export a DaguaGraph to a torch_geometric.data.Data object.

    Contains edge_index [2, E] and num_nodes.

    Requires: ``pip install torch-geometric``
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError(
            "torch_geometric is required for to_pyg(). Install it with: "
            "pip install torch-geometric"
        )

    import torch

    graph._finalize_edges()
    ei = graph._edge_index_tensor
    if ei is None or ei.numel() == 0:
        ei = torch.zeros(2, 0, dtype=torch.long)

    return Data(edge_index=ei.clone(), num_nodes=graph.num_nodes)


def to_scipy(graph: "DaguaGraph") -> Any:
    """Export a DaguaGraph to a scipy.sparse.csr_matrix adjacency matrix.

    Returns an [N, N] sparse matrix where entry (i, j) = 1 if edge i→j exists.

    Requires: ``pip install scipy``
    """
    try:
        import scipy.sparse
    except ImportError:
        raise ImportError(
            "scipy is required for to_scipy(). Install it with: "
            "pip install scipy"
        )

    import numpy as np

    N = graph.num_nodes
    graph._finalize_edges()
    ei = graph._edge_index_tensor

    if ei is None or ei.numel() == 0:
        return scipy.sparse.csr_matrix((N, N), dtype=np.int8)

    rows = ei[0].numpy()
    cols = ei[1].numpy()
    data = np.ones(ei.shape[1], dtype=np.int8)
    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(N, N))


# ─── Import functions ──────────────────────────────────────────────────────


def from_igraph(ig_graph, **kwargs) -> "DaguaGraph":
    """Build a DaguaGraph from an igraph.Graph.

    Reads vertex attributes: name (→ node ID), label, type.
    Reads edge attributes: label.

    Args:
        ig_graph: An igraph.Graph instance.
        **kwargs: Extra keyword arguments passed to DaguaGraph().

    Returns:
        A DaguaGraph.
    """
    from dagua.graph import DaguaGraph

    g = DaguaGraph(**kwargs)

    for v in ig_graph.vs:
        node_id = v["name"] if "name" in v.attributes() else v.index
        label = v["label"] if "label" in v.attributes() else None
        node_type = v["type"] if "type" in v.attributes() else "default"
        g.add_node(node_id, label=label, type=node_type)

    for e in ig_graph.es:
        src_v = ig_graph.vs[e.source]
        tgt_v = ig_graph.vs[e.target]
        src_id = src_v["name"] if "name" in src_v.attributes() else e.source
        tgt_id = tgt_v["name"] if "name" in tgt_v.attributes() else e.target
        label = e["label"] if "label" in e.attributes() else None
        g.add_edge(src_id, tgt_id, label=label)

    return g


def from_scipy(adj_matrix, labels=None, **kwargs) -> "DaguaGraph":
    """Build a DaguaGraph from a scipy sparse adjacency matrix.

    Args:
        adj_matrix: A scipy sparse matrix of shape [N, N].
        labels: Optional list of node labels (length N).
        **kwargs: Extra keyword arguments passed to DaguaGraph().

    Returns:
        A DaguaGraph.
    """
    import torch

    from dagua.graph import DaguaGraph

    try:
        import scipy.sparse
    except ImportError:
        raise ImportError(
            "scipy is required for from_scipy(). Install it with: "
            "pip install scipy"
        )

    coo = scipy.sparse.coo_matrix(adj_matrix)
    N = coo.shape[0]

    g = DaguaGraph(**kwargs)
    for i in range(N):
        label = labels[i] if labels is not None and i < len(labels) else None
        g.add_node(i, label=label)

    for src, tgt in zip(coo.row, coo.col):
        g.add_edge(int(src), int(tgt))

    return g


def from_dot(dot_string: str, **kwargs) -> "DaguaGraph":
    """Build a DaguaGraph from a DOT string.

    Parses DOT format via pydot (pure Python, no system Graphviz dependency).

    Args:
        dot_string: A DOT-format string.
        **kwargs: Extra keyword arguments passed to DaguaGraph().

    Returns:
        A DaguaGraph.
    """
    try:
        import pydot
    except ImportError:
        raise ImportError(
            "pydot is required for from_dot(). Install it with: "
            "pip install pydot"
        )

    from dagua.graph import DaguaGraph

    graphs = pydot.graph_from_dot_data(dot_string)
    if not graphs:
        raise ValueError("Could not parse DOT string")
    pg = graphs[0]

    g = DaguaGraph(**kwargs)

    # Detect direction from rankdir
    rankdir = pg.get_rankdir()
    if rankdir:
        rd = rankdir.strip('"').upper()
        if rd in ("TB", "BT", "LR", "RL"):
            g.direction = rd

    # Add nodes
    for node in pg.get_nodes():
        name = node.get_name().strip('"')
        if name in ("node", "edge", "graph", ""):
            continue  # skip DOT defaults
        label = node.get_label()
        if label:
            label = label.strip('"')
        g.add_node(name, label=label)

    # Add edges
    for edge in pg.get_edges():
        src = edge.get_source().strip('"')
        dst = edge.get_destination().strip('"')
        label = edge.get_label()
        if label:
            label = label.strip('"')
        g.add_edge(src, dst, label=label)

    # Add clusters from subgraphs
    for sg in pg.get_subgraphs():
        sg_name = sg.get_name()
        if sg_name.startswith("cluster_"):
            cluster_name = sg_name[len("cluster_"):]
        else:
            cluster_name = sg_name
        cluster_name = cluster_name.strip('"')

        members = []
        for node in sg.get_nodes():
            node_name = node.get_name().strip('"')
            if node_name in g._id_to_index:
                members.append(node_name)

        if members:
            label = sg.get_label()
            if label:
                label = label.strip('"')
            g.add_cluster(cluster_name, members, label=label)

    return g


# ─── LLM infrastructure ──────────────────────────────────────────────────


def configure_image_ai(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ImageAIConfig:
    """Set process-wide defaults for image-to-graph/theme AI calls."""
    global _IMAGE_AI_CONFIG
    _IMAGE_AI_CONFIG = ImageAIConfig(
        provider=provider,
        api_key=api_key,
        api_key_env=api_key_env,
        model=model,
        base_url=base_url,
    )
    return dataclasses.replace(_IMAGE_AI_CONFIG)


def get_image_ai_config() -> ImageAIConfig:
    """Return the current process-wide image AI configuration."""
    return dataclasses.replace(_IMAGE_AI_CONFIG)


def _normalize_provider_name(provider: Optional[str]) -> Optional[str]:
    if provider is None:
        return None
    normalized = provider.lower().strip()
    aliases = {
        "claude": "anthropic",
        "anthropic": "anthropic",
        "openai": "openai",
        "gpt": "openai",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown provider: {provider!r}. Use 'anthropic' or 'openai'."
        )
    return aliases[normalized]


def _resolve_image_ai_config(
    config: Optional[ImageAIConfig] = None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ImageAIConfig:
    """Resolve explicit args, configured defaults, and env variables into one config."""
    base = config or _IMAGE_AI_CONFIG
    provider_name = _normalize_provider_name(
        provider
        or base.provider
        or os.environ.get("DAGUA_IMAGE_AI_PROVIDER")
    )

    if provider_name is None:
        for candidate, env_keys in _IMAGE_AI_PROVIDER_ENV_KEYS.items():
            if any(os.environ.get(key) for key in env_keys):
                provider_name = candidate
                break

    env_name = api_key_env or base.api_key_env or os.environ.get("DAGUA_IMAGE_AI_API_KEY_ENV")
    resolved_api_key = api_key or base.api_key or os.environ.get("DAGUA_IMAGE_AI_API_KEY")
    if resolved_api_key is None and env_name:
        resolved_api_key = os.environ.get(env_name)

    if provider_name and resolved_api_key is None:
        for env_key in _IMAGE_AI_PROVIDER_ENV_KEYS.get(provider_name, ()):
            resolved_api_key = os.environ.get(env_key)
            if resolved_api_key:
                break

    resolved_model = model or base.model or os.environ.get("DAGUA_IMAGE_AI_MODEL")
    resolved_base_url = base_url or base.base_url or os.environ.get("DAGUA_IMAGE_AI_BASE_URL")

    if provider_name is None:
        raise RuntimeError(
            "No image AI provider configured. Pass provider=..., call configure_image_ai(...), "
            "or set DAGUA_IMAGE_AI_PROVIDER / provider-specific API key env vars."
        )
    if resolved_api_key is None:
        provider_envs = ", ".join(_IMAGE_AI_PROVIDER_ENV_KEYS.get(provider_name, ()))
        raise RuntimeError(
            "No image AI API key found. Pass api_key=..., set api_key_env=..., call "
            "configure_image_ai(...), or set one of: DAGUA_IMAGE_AI_API_KEY, "
            f"DAGUA_IMAGE_AI_API_KEY_ENV, {provider_envs}."
        )

    return ImageAIConfig(
        provider=provider_name,
        api_key=resolved_api_key,
        api_key_env=env_name,
        model=resolved_model or _IMAGE_AI_DEFAULT_MODEL[provider_name],
        base_url=resolved_base_url,
    )


def _get_llm_client(
    provider: Optional[str] = None,
    *,
    config: Optional[ImageAIConfig] = None,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[str, Any, ImageAIConfig]:
    """Resolve provider config and return (provider_name, client, resolved_config)."""
    resolved = _resolve_image_ai_config(
        config=config,
        provider=provider,
        api_key=api_key,
        api_key_env=api_key_env,
        model=model,
        base_url=base_url,
    )

    if resolved.provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic SDK not installed. Run: pip install 'dagua[ai]'"
            )
        kwargs = {"api_key": resolved.api_key}
        if resolved.base_url:
            kwargs["base_url"] = resolved.base_url
        return "anthropic", anthropic.Anthropic(**kwargs), resolved

    if resolved.provider == "openai":
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai SDK not installed. Run: pip install 'dagua[ai]'"
            )
        kwargs = {"api_key": resolved.api_key}
        if resolved.base_url:
            kwargs["base_url"] = resolved.base_url
        return "openai", openai.OpenAI(**kwargs), resolved

    raise ValueError(f"Unknown provider: {resolved.provider!r}. Use 'anthropic' or 'openai'.")


def _unpack_llm_client_result(result, *, provider: Optional[str], model: Optional[str], config: Optional[ImageAIConfig], api_key: Optional[str], api_key_env: Optional[str], base_url: Optional[str]) -> Tuple[str, Any, ImageAIConfig]:
    """Accept both legacy 2-tuples and new 3-tuples from _get_llm_client mocks."""
    if len(result) == 3:
        return result
    provider_name, client = result
    provider_name = provider_name or provider or "anthropic"
    try:
        resolved = _resolve_image_ai_config(
            config=config,
            provider=provider_name,
            api_key=api_key,
            api_key_env=api_key_env,
            model=model,
            base_url=base_url,
        )
    except RuntimeError:
        resolved = ImageAIConfig(
            provider=_normalize_provider_name(provider_name),
            api_key=api_key,
            api_key_env=api_key_env,
            model=model or _IMAGE_AI_DEFAULT_MODEL[_normalize_provider_name(provider_name)],
            base_url=base_url,
        )
    return provider_name, client, resolved


def _send_image_to_llm(
    client: Any,
    provider: str,
    image_path: str,
    prompt: str,
    model: str,
) -> str:
    """Send an image + prompt to an LLM and return the text response."""
    image_bytes, media_type = _prepare_image_for_llm(image_path)
    image_data = base64.b64encode(image_bytes).decode("utf-8")

    if provider == "anthropic":
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


def _prepare_image_for_llm(image_path: Union[str, Path]) -> Tuple[bytes, str]:
    """Return image bytes and a provider-safe media type.

    Common web-native formats are passed through directly. Other common raster
    formats are converted to PNG via Pillow so image-to-graph works across the
    usual desktop export formats without surfacing backend-specific surprises.
    """
    path = Path(image_path)
    ext = path.suffix.lower()
    if ext in _DIRECT_IMAGE_MEDIA_TYPES:
        with open(path, "rb") as f:
            return f.read(), _DIRECT_IMAGE_MEDIA_TYPES[ext]

    if ext in _SVG_COMPAT_IMAGE_EXTENSIONS:
        try:
            import cairosvg
        except ImportError as exc:
            raise ImportError(
                "SVG image input requires cairosvg. Install it with: pip install cairosvg"
            ) from exc
        png_bytes = cairosvg.svg2png(url=str(path))
        return png_bytes, "image/png"

    if ext not in _PIL_COMPAT_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {ext or '<none>'}. "
            "Supported input formats include PNG, JPEG, GIF, WebP, BMP, TIFF, and SVG."
        )

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to load and normalize BMP/TIFF and other non-web-native image formats."
        ) from exc

    with Image.open(path) as img:
        normalized = img.convert("RGBA")
        buf = io.BytesIO()
        normalized.save(buf, format="PNG")
        return buf.getvalue(), "image/png"


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


def _python_literal(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_python_literal(v) for v in value) + "]"
    if isinstance(value, tuple):
        return "(" + ", ".join(_python_literal(v) for v in value) + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, dict):
        items = ", ".join(f"{_python_literal(k)}: {_python_literal(v)}" for k, v in value.items())
        return "{" + items + "}"
    return repr(value)


def _constructor_expr(class_name: str, data: Dict[str, Any]) -> str:
    if not data:
        return f"{class_name}()"
    args = ", ".join(f"{key}={_python_literal(value)}" for key, value in data.items())
    return f"{class_name}({args})"


def graph_code_from_dict(graph_dict: Dict[str, Any], function_name: str = "build_graph") -> str:
    """Return canonical Dagua Python code for a graph dict."""
    nodes = graph_dict.get("nodes", [])
    edges = graph_dict.get("edges", [])
    clusters = graph_dict.get("clusters", [])
    imports = ["import dagua", "from dagua import DaguaGraph"]

    needs_node_style = any("style" in node for node in nodes)
    needs_edge_style = any("style" in edge for edge in edges)
    needs_cluster_style = any("style" in cluster for cluster in clusters)
    style_imports: List[str] = []
    if needs_node_style:
        style_imports.append("NodeStyle")
    if needs_edge_style:
        style_imports.append("EdgeStyle")
    if needs_cluster_style:
        style_imports.append("ClusterStyle")
    if style_imports:
        imports.append("from dagua import " + ", ".join(style_imports))

    lines = imports + ["", f"def {function_name}() -> DaguaGraph:", f"    g = DaguaGraph(direction={_python_literal(graph_dict.get('direction', 'TB'))})"]

    for node in nodes:
        parts = [f"    g.add_node({_python_literal(node['id'])}"]
        if "label" in node:
            parts.append(f"label={_python_literal(node['label'])}")
        if "type" in node:
            parts.append(f"type={_python_literal(node['type'])}")
        if "style" in node:
            parts.append(f"style={_constructor_expr('NodeStyle', node['style'])}")
        lines.append(", ".join(parts) + ")")

    if nodes:
        lines.append("")

    for edge in edges:
        parts = [
            f"    g.add_edge({_python_literal(edge['source'])}",
            _python_literal(edge["target"]),
        ]
        if "label" in edge:
            parts.append(f"label={_python_literal(edge['label'])}")
        if "type" in edge:
            parts.append(f"type={_python_literal(edge['type'])}")
        if "style" in edge:
            parts.append(f"style={_constructor_expr('EdgeStyle', edge['style'])}")
        lines.append(", ".join(parts) + ")")

    if clusters:
        lines.append("")
    for cluster in clusters:
        parts = [
            f"    g.add_cluster({_python_literal(cluster['name'])}",
            _python_literal(cluster.get("members", [])),
        ]
        if "label" in cluster:
            parts.append(f"label={_python_literal(cluster['label'])}")
        if "parent" in cluster:
            parts.append(f"parent={_python_literal(cluster['parent'])}")
        if "style" in cluster:
            parts.append(f"style={_constructor_expr('ClusterStyle', cluster['style'])}")
        lines.append(", ".join(parts) + ")")

    lines.extend(["", "    g.compute_node_sizes()", "    return g", "", f"graph = {function_name}()"])
    return "\n".join(lines)


def graph_script_from_dict(
    graph_dict: Dict[str, Any],
    function_name: str = "build_graph",
    output_path: str = "graph.png",
) -> str:
    """Return a ready-to-run Dagua script with layout and export included."""
    builder = graph_code_from_dict(graph_dict, function_name=function_name)
    script_lines = [
        builder,
        "",
        "",
        "if __name__ == '__main__':",
        "    dagua.configure()",
        "    config = dagua.LayoutConfig(",
        "        device='cuda' if __import__('torch').cuda.is_available() else 'cpu',",
        "        steps=90,",
        "        edge_opt_steps=10,",
        "        seed=42,",
        "    )",
        f"    fig, ax = dagua.draw(graph, config, output={_python_literal(output_path)})",
        f"    print('Wrote {output_path}')",
    ]
    return "\n".join(script_lines)


def theme_code_from_dict(theme_dict: Dict[str, Any], variable_name: str = "theme") -> str:
    """Return canonical Dagua Python code for a theme dict."""
    imports = [
        "from dagua.styles import Theme, NodeStyle, EdgeStyle, ClusterStyle, GraphStyle",
        "",
    ]
    node_styles = theme_dict.get("node_styles", {})
    edge_styles = theme_dict.get("edge_styles", {})
    cluster_style = theme_dict.get("cluster_style", {})
    graph_style = theme_dict.get("graph_style", {})
    lines = imports + [f"{variable_name} = Theme("]
    if "name" in theme_dict:
        lines.append(f"    name={_python_literal(theme_dict['name'])},")
    if node_styles:
        lines.append("    node_styles={")
        for key, value in node_styles.items():
            lines.append(f"        {_python_literal(key)}: {_constructor_expr('NodeStyle', value)},")
        lines.append("    },")
    if edge_styles:
        lines.append("    edge_styles={")
        for key, value in edge_styles.items():
            lines.append(f"        {_python_literal(key)}: {_constructor_expr('EdgeStyle', value)},")
        lines.append("    },")
    if cluster_style:
        lines.append(f"    cluster_style={_constructor_expr('ClusterStyle', cluster_style)},")
    if graph_style:
        lines.append(f"    graph_style={_constructor_expr('GraphStyle', graph_style)},")
    lines.append(")")
    return "\n".join(lines)


def graph_dict_from_image(
    image_path: Union[str, Path],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    base_url: Optional[str] = None,
    config: Optional[ImageAIConfig] = None,
) -> Dict[str, Any]:
    """Extract a graph JSON dict from an image using a configured AI provider."""
    provider_name, client, resolved = _unpack_llm_client_result(
        _get_llm_client(
            provider,
            config=config,
            api_key=api_key,
            api_key_env=api_key_env,
            model=model,
            base_url=base_url,
        ),
        provider=provider,
        model=model,
        config=config,
        api_key=api_key,
        api_key_env=api_key_env,
        base_url=base_url,
    )
    response_text = _send_image_to_llm(
        client,
        provider_name,
        str(image_path),
        _GRAPH_EXTRACTION_PROMPT,
        resolved.model,
    )
    return _extract_json_from_response(response_text)


def graph_code_from_image(
    image_path: Union[str, Path],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    base_url: Optional[str] = None,
    config: Optional[ImageAIConfig] = None,
    function_name: str = "build_graph",
    include_demo_script: bool = False,
    output_path: str = "graph.png",
) -> str:
    """Return canonical Dagua code reconstructed from an image."""
    graph_dict = graph_dict_from_image(
        image_path,
        provider=provider,
        model=model,
        api_key=api_key,
        api_key_env=api_key_env,
        base_url=base_url,
        config=config,
    )
    if include_demo_script:
        return graph_script_from_dict(
            graph_dict,
            function_name=function_name,
            output_path=output_path,
        )
    return graph_code_from_dict(graph_dict, function_name=function_name)


def graph_from_image(
    image_path: Union[str, Path],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    base_url: Optional[str] = None,
    config: Optional[ImageAIConfig] = None,
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
    graph_dict = graph_dict_from_image(
        image_path,
        provider=provider,
        model=model,
        api_key=api_key,
        api_key_env=api_key_env,
        base_url=base_url,
        config=config,
    )
    return graph_from_json(graph_dict)


def theme_dict_from_image(
    image_path: Union[str, Path],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    base_url: Optional[str] = None,
    config: Optional[ImageAIConfig] = None,
) -> Dict[str, Any]:
    """Extract a theme JSON dict from an image using a configured AI provider."""
    provider_name, client, resolved = _unpack_llm_client_result(
        _get_llm_client(
            provider,
            config=config,
            api_key=api_key,
            api_key_env=api_key_env,
            model=model,
            base_url=base_url,
        ),
        provider=provider,
        model=model,
        config=config,
        api_key=api_key,
        api_key_env=api_key_env,
        base_url=base_url,
    )
    response_text = _send_image_to_llm(
        client,
        provider_name,
        str(image_path),
        _THEME_EXTRACTION_PROMPT,
        resolved.model,
    )
    return _extract_json_from_response(response_text)


def theme_code_from_image(
    image_path: Union[str, Path],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    base_url: Optional[str] = None,
    config: Optional[ImageAIConfig] = None,
    variable_name: str = "theme",
) -> str:
    """Return canonical Dagua theme code reconstructed from an image."""
    theme_dict = theme_dict_from_image(
        image_path,
        provider=provider,
        model=model,
        api_key=api_key,
        api_key_env=api_key_env,
        base_url=base_url,
        config=config,
    )
    return theme_code_from_dict(theme_dict, variable_name=variable_name)


def theme_from_image(
    image_path: Union[str, Path],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    base_url: Optional[str] = None,
    config: Optional[ImageAIConfig] = None,
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
    raw = theme_dict_from_image(
        image_path,
        provider=provider,
        model=model,
        api_key=api_key,
        api_key_env=api_key_env,
        base_url=base_url,
        config=config,
    )
    return _dict_to_theme(raw)

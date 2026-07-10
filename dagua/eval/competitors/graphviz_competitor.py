"""Graphviz competitor adapters — dot, sfdp, neato, fdp engines."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Set, Tuple

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register
from dagua.eval.size_policy import size_aware_externals

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_CLUSTER_NAME_PATTERN = re.compile(r"[^0-9A-Za-z_]")


def _graphviz_available() -> bool:
    """Check whether Graphviz's `dot` executable is available.

    Returns
    -------
    bool
        ``True`` when `dot` is installed.
    """
    return shutil.which("dot") is not None


def _escape_dot_string(value: str) -> str:
    """Escape a string for DOT quoted-string syntax.

    Parameters
    ----------
    value : str
        Raw string value.

    Returns
    -------
    str
        Escaped string safe for inclusion inside double quotes.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _cluster_id(name: str) -> str:
    """Convert a cluster name into a DOT-safe subgraph identifier.

    Parameters
    ----------
    name : str
        Cluster name from the graph.

    Returns
    -------
    str
        Subgraph identifier with the required ``cluster_`` prefix.
    """
    sanitized = _CLUSTER_NAME_PATTERN.sub("_", name)
    return f"cluster_{sanitized}"


def _cluster_members(graph: DaguaGraph, cluster_name: str) -> List[int]:
    """Return a cluster's flattened member indices.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    cluster_name : str
        Cluster to inspect.

    Returns
    -------
    list[int]
        Flattened node indices for the cluster.
    """
    members = graph.clusters.get(cluster_name, [])
    if isinstance(members, dict):
        from dagua.utils import collect_cluster_leaves

        return [int(index) for index in collect_cluster_leaves(members)]
    return [int(index) for index in members]


def _cluster_children(graph: DaguaGraph) -> Dict[Optional[str], List[str]]:
    """Build the cluster hierarchy indexed by parent name.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.

    Returns
    -------
    dict[Optional[str], list[str]]
        Mapping from parent cluster name to sorted child cluster names. Root
        clusters are stored under ``None``.
    """
    children_by_parent: Dict[Optional[str], List[str]] = {}
    for cluster_name in sorted(graph.clusters):
        parent = graph.cluster_parents.get(cluster_name)
        if parent not in graph.clusters:
            parent = None
        children_by_parent.setdefault(parent, []).append(cluster_name)
    return children_by_parent


def _node_statement(graph: DaguaGraph, index: int, indent: str) -> str:
    """Render a DOT node statement for one graph node.

    When ``graph.node_sizes`` is populated and size-aware externals are
    enabled (see ``dagua.eval.size_policy``), the node is also emitted with
    real ``width``/``height`` (converted to Graphviz's inch convention) and
    ``fixedsize=true`` so Graphviz's ``dot`` engine lays out the node at its
    real label-measured size instead of auto-sizing to the label text --
    this is the same size dagua's own composite score uses to count
    overlaps, closing the size-blind-vs-size-aware scoring mismatch (S1
    HIGH-2). With ``--size-blind-externals`` the old behavior is preserved.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    index : int
        Node index.
    indent : str
        Indentation prefix for the emitted line.

    Returns
    -------
    str
        DOT node declaration.
    """
    label = _escape_dot_string(graph.node_labels[index])
    style = graph.get_style_for_node(index)
    attrs = [
        f'label="{label}"',
        f'fillcolor="{style.fill}"',
        f'fontcolor="{style.font_color}"',
        f"fontsize={style.font_size}",
    ]
    if style.shape == "ellipse":
        attrs.append("shape=ellipse")
    elif style.shape == "circle":
        attrs.append("shape=circle")
    elif style.shape == "diamond":
        attrs.append("shape=diamond")
    else:
        attrs.append("shape=box")
        if style.shape == "roundrect":
            attrs.append('style="filled,rounded"')

    if graph.node_sizes is not None and size_aware_externals():
        # Graphviz uses inches, 72 points/inch (matches dagua's own point scale).
        width_in = max(float(graph.node_sizes[index, 0].item()) / 72.0, 0.01)
        height_in = max(float(graph.node_sizes[index, 1].item()) / 72.0, 0.01)
        attrs.append(f"width={width_in:.4f}")
        attrs.append(f"height={height_in:.4f}")
        attrs.append("fixedsize=true")

    attrs_str = ", ".join(attrs)
    return f"{indent}n{index} [{attrs_str}];"


def _emit_cluster(
    lines: List[str],
    graph: DaguaGraph,
    cluster_name: str,
    children_by_parent: Dict[Optional[str], List[str]],
    emitted_nodes: Set[int],
    depth: int,
) -> None:
    """Append DOT lines for a cluster and its nested children.

    Parameters
    ----------
    lines : list[str]
        Mutable DOT line buffer.
    graph : DaguaGraph
        Source graph.
    cluster_name : str
        Cluster being emitted.
    children_by_parent : dict[Optional[str], list[str]]
        Parent-to-children cluster hierarchy.
    emitted_nodes : set[int]
        Nodes already emitted into a cluster block.
    depth : int
        Current hierarchy depth used for indentation.

    Returns
    -------
    None
        The function mutates ``lines`` in place.
    """
    indent = "  " * (depth + 1)
    cluster_label = _escape_dot_string(graph.cluster_labels.get(cluster_name, cluster_name))
    lines.append(f"{indent}subgraph {_cluster_id(cluster_name)} {{")
    lines.append(f'{indent}  label="{cluster_label}";')
    lines.append(f'{indent}  style=filled; color=lightgrey; fillcolor="#f0f0f0";')

    child_clusters = children_by_parent.get(cluster_name, [])
    for child_name in child_clusters:
        _emit_cluster(lines, graph, child_name, children_by_parent, emitted_nodes, depth + 1)

    nested_members: Set[int] = set()
    for child_name in child_clusters:
        nested_members.update(_cluster_members(graph, child_name))

    for node_index in _cluster_members(graph, cluster_name):
        if node_index in nested_members or node_index in emitted_nodes:
            continue
        lines.append(_node_statement(graph, node_index, f"{indent}  "))
        emitted_nodes.add(node_index)

    lines.append(f"{indent}}}")


def _graph_to_dot(graph: DaguaGraph) -> str:
    """Convert a graph into DOT, including nested cluster subgraphs.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to convert.

    Returns
    -------
    str
        DOT source string suitable for `dot -Tjson`.
    """
    lines = ["digraph G {"]
    lines.append("  rankdir=TB;")
    lines.append('  node [shape=box, style=filled, fontname="Helvetica"];')
    lines.append('  edge [fontname="Helvetica", fontsize=9];')

    emitted_nodes: Set[int] = set()
    if graph.clusters:
        children_by_parent = _cluster_children(graph)
        for cluster_name in children_by_parent.get(None, []):
            _emit_cluster(lines, graph, cluster_name, children_by_parent, emitted_nodes, depth=0)

    for node_index in range(graph.num_nodes):
        if node_index not in emitted_nodes:
            lines.append(_node_statement(graph, node_index, "  "))

    if graph.edge_index.numel() > 0:
        for edge_index in range(graph.edge_index.shape[1]):
            source = int(graph.edge_index[0, edge_index].item())
            target = int(graph.edge_index[1, edge_index].item())
            edge_attrs: List[str] = []
            if edge_index < len(graph.edge_labels) and graph.edge_labels[edge_index]:
                label = _escape_dot_string(str(graph.edge_labels[edge_index]))
                edge_attrs.append(f'label="{label}"')
            edge_style = graph.get_style_for_edge(edge_index)
            edge_attrs.append(f'color="{edge_style.color}"')
            if edge_style.style == "dashed":
                edge_attrs.append("style=dashed")
            elif edge_style.style == "dotted":
                edge_attrs.append("style=dotted")
            attrs_str = f" [{', '.join(edge_attrs)}]" if edge_attrs else ""
            lines.append(f"  n{source} -> n{target}{attrs_str};")

    lines.append("}")
    return "\n".join(lines)


def _graphviz_attribute_value(value: Any) -> str:
    """Convert a Python value to a Graphviz command-line attribute value.

    Parameters
    ----------
    value : Any
        Python scalar from variant parameters.

    Returns
    -------
    str
        String value suitable for ``-Gkey=value``.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _layout_with_dot(
    graph: DaguaGraph,
    timeout: float,
    graph_attributes: Optional[Mapping[str, Any]] = None,
):
    """Run Graphviz `dot` on a graph and parse positions plus edge geometry.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to lay out.
    timeout : float
        Maximum runtime in seconds.
    graph_attributes : Mapping[str, Any] | None, default=None
        Optional Graphviz graph attributes passed as ``-G`` command-line
        overrides.

    Returns
    -------
    tuple
        ``(positions, routes, edge_label_positions)`` where positions has
        shape ``[N, 2]`` and the drawing fields may be ``None``. Callers must
        normalize through :func:`_coerce_layout_capture` because tests stub
        this helper with a bare tensor.
    """
    dot_str = _graph_to_dot(graph)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".dot",
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write(dot_str)
        dot_path = Path(handle.name)

    try:
        command = ["dot", "-Tjson"]
        if graph_attributes is not None:
            for key, value in graph_attributes.items():
                if value is None:
                    continue
                command.append(f"-G{key}={_graphviz_attribute_value(value)}")
        result = subprocess.run(
            [*command, str(dot_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Graphviz failed: {result.stderr}")
        data = json.loads(result.stdout)
    finally:
        dot_path.unlink(missing_ok=True)

    positions = torch.zeros(graph.num_nodes, 2)
    objects = data.get("objects", [])
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            name = str(obj.get("name", ""))
            if not name.startswith("n") or not name[1:].isdigit() or "pos" not in obj:
                continue
            node_index = int(name[1:])
            if node_index >= graph.num_nodes:
                continue
            x_str, y_str = str(obj["pos"]).split(",", maxsplit=1)
            positions[node_index, 0] = float(x_str)
            positions[node_index, 1] = -float(y_str)

    routes, edge_label_positions = _parse_graphviz_json_drawing(data, graph)
    return positions, routes, edge_label_positions


def _parse_graphviz_json_positions(data: dict[str, object], num_nodes: int) -> torch.Tensor:
    """Parse Graphviz JSON node positions into a tensor.

    Parameters
    ----------
    data : dict[str, object]
        Parsed Graphviz JSON payload.
    num_nodes : int
        Number of expected graph nodes.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]`` in Dagua's y-down coordinate
        convention.
    """
    positions = torch.zeros(num_nodes, 2)
    objects = data.get("objects", [])
    if not isinstance(objects, list):
        return positions

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name", ""))
        if not name.startswith("n") or not name[1:].isdigit() or "pos" not in obj:
            continue
        node_index = int(name[1:])
        if node_index >= num_nodes:
            continue
        x_str, y_str = str(obj["pos"]).split(",", maxsplit=1)
        positions[node_index, 0] = float(x_str)
        positions[node_index, 1] = -float(y_str)
    return positions


# ---------------------------------------------------------------------------
# Full-drawing capture (r80-S6): edge splines + edge label positions.
#
# Graphviz "pos" edge attributes carry a cubic B-spline emitted in piecewise
# BEZIER form: an optional arrow endpoint prefix ("e,x,y" -- the true head
# tip) and/or start prefix ("s,x,y"), followed by 3k+1 control points where
# each consecutive (p0,p1,p2,p3), (p3,p4,p5,p6), ... quadruple is one cubic
# bezier segment. We convert to a polyline by sampling each cubic segment
# uniformly, then appending the "s"/"e" endpoints so the polyline covers the
# full drawn path including the arrowhead tip. Y is negated to match dagua's
# y-down convention (same flip as the node-position parser above).
# ---------------------------------------------------------------------------

_SPLINE_SAMPLES_PER_SEGMENT = 8


def _parse_xdot_spline(pos_value: str) -> Optional[List[Tuple[float, float]]]:
    """Parse a Graphviz edge ``pos`` attribute into a polyline.

    Parameters
    ----------
    pos_value : str
        Raw ``pos`` string, e.g. ``"e,39.8,121.1 84.4,173.8 73.7,161.1 ..."``.

    Returns
    -------
    list[tuple[float, float]] | None
        Sampled polyline in GRAPHVIZ coordinates (y-up), or ``None`` when the
        string cannot be interpreted as a spline.
    """
    tokens = str(pos_value).split()
    if not tokens:
        return None

    end_point: Optional[Tuple[float, float]] = None
    start_point: Optional[Tuple[float, float]] = None
    control_points: List[Tuple[float, float]] = []
    try:
        for token in tokens:
            if token.startswith("e,"):
                x_str, y_str = token[2:].split(",", maxsplit=1)
                end_point = (float(x_str), float(y_str))
            elif token.startswith("s,"):
                x_str, y_str = token[2:].split(",", maxsplit=1)
                start_point = (float(x_str), float(y_str))
            else:
                x_str, y_str = token.split(",", maxsplit=1)
                control_points.append((float(x_str), float(y_str)))
    except ValueError:
        return None

    if len(control_points) < 4 or (len(control_points) - 1) % 3 != 0:
        return None

    polyline: List[Tuple[float, float]] = []
    if start_point is not None:
        polyline.append(start_point)

    n_segments = (len(control_points) - 1) // 3
    for seg in range(n_segments):
        p0 = control_points[3 * seg]
        p1 = control_points[3 * seg + 1]
        p2 = control_points[3 * seg + 2]
        p3 = control_points[3 * seg + 3]
        start_k = 0 if seg == 0 else 1  # skip duplicated joint points
        for k in range(start_k, _SPLINE_SAMPLES_PER_SEGMENT):
            t = k / (_SPLINE_SAMPLES_PER_SEGMENT - 1)
            u = 1.0 - t
            x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
            y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
            polyline.append((x, y))

    if end_point is not None:
        polyline.append(end_point)
    return polyline if len(polyline) >= 2 else None


def _edge_label_point(edge_obj: dict) -> Optional[Tuple[float, float]]:
    """Extract an edge-label anchor from a Graphviz JSON edge object.

    Prefers the ``lp`` attribute (label center); falls back to the first
    ``_ldraw_`` op carrying a ``pt`` field.

    Parameters
    ----------
    edge_obj : dict
        One entry of the Graphviz JSON ``edges`` array.

    Returns
    -------
    tuple[float, float] | None
        Label anchor in GRAPHVIZ coordinates (y-up), or ``None``.
    """
    lp = edge_obj.get("lp")
    if isinstance(lp, str) and "," in lp:
        try:
            x_str, y_str = lp.split(",", maxsplit=1)
            return (float(x_str), float(y_str))
        except ValueError:
            pass
    ldraw = edge_obj.get("_ldraw_")
    if isinstance(ldraw, list):
        for op in ldraw:
            if isinstance(op, dict) and isinstance(op.get("pt"), list) and len(op["pt"]) >= 2:
                try:
                    return (float(op["pt"][0]), float(op["pt"][1]))
                except (TypeError, ValueError):
                    continue
    return None


def _parse_graphviz_json_drawing(
    data: dict,
    graph: DaguaGraph,
) -> Tuple[
    Optional[List[Optional[List[Tuple[float, float]]]]],
    Optional[List[Optional[Tuple[float, float]]]],
]:
    """Parse edge splines and label anchors from a Graphviz JSON payload.

    JSON edges reference node objects through ``tail``/``head`` gvids; each
    JSON edge is matched to the next unassigned dagua edge with the same
    (source, target) pair, which preserves input order for parallel edges.

    Parameters
    ----------
    data : dict
        Parsed ``dot -Tjson`` payload.
    graph : DaguaGraph
        Graph the layout was produced for (source of edge order).

    Returns
    -------
    tuple
        ``(routes, edge_label_positions)`` aligned to ``edge_index`` columns
        in DAGUA coordinates (y-down), or ``(None, None)`` when the payload
        carries no usable edge geometry.
    """
    num_edges = int(graph.edge_index.shape[1]) if graph.edge_index.numel() > 0 else 0
    if num_edges == 0:
        return None, None

    objects = data.get("objects", [])
    edges = data.get("edges", [])
    if not isinstance(objects, list) or not isinstance(edges, list) or not edges:
        return None, None

    gvid_to_node: Dict[int, int] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name", ""))
        if name.startswith("n") and name[1:].isdigit() and "_gvid" in obj:
            gvid_to_node[int(obj["_gvid"])] = int(name[1:])

    # Queue of dagua edge indices per (source, target), preserving order.
    pair_queues: Dict[Tuple[int, int], List[int]] = {}
    for e_idx in range(num_edges):
        s = int(graph.edge_index[0, e_idx].item())
        t = int(graph.edge_index[1, e_idx].item())
        pair_queues.setdefault((s, t), []).append(e_idx)

    routes: List[Optional[List[Tuple[float, float]]]] = [None] * num_edges
    labels: List[Optional[Tuple[float, float]]] = [None] * num_edges
    any_route = False
    for edge_obj in edges:
        if not isinstance(edge_obj, dict):
            continue
        tail = gvid_to_node.get(int(edge_obj.get("tail", -1)))
        head = gvid_to_node.get(int(edge_obj.get("head", -1)))
        if tail is None or head is None:
            continue
        queue = pair_queues.get((tail, head))
        if not queue:
            continue
        e_idx = queue.pop(0)
        pos_value = edge_obj.get("pos")
        if isinstance(pos_value, str):
            spline = _parse_xdot_spline(pos_value)
            if spline is not None:
                routes[e_idx] = [(x, -y) for x, y in spline]
                any_route = True
        label_point = _edge_label_point(edge_obj)
        if label_point is not None:
            labels[e_idx] = (label_point[0], -label_point[1])

    if not any_route:
        return None, None
    any_label = any(lbl is not None for lbl in labels)
    return routes, (labels if any_label else None)


_LayoutCapture = Tuple[
    torch.Tensor,
    Optional[List[Optional[List[Tuple[float, float]]]]],
    Optional[List[Optional[Tuple[float, float]]]],
]


def _coerce_layout_capture(result: Any) -> _LayoutCapture:
    """Normalize a layout helper's return value to ``(pos, routes, labels)``.

    Keeps backward compatibility with callers/tests that stub the layout
    helpers to return a bare position tensor.

    Parameters
    ----------
    result : Any
        Either a position tensor or a ``(pos, routes, labels)`` triple.

    Returns
    -------
    _LayoutCapture
        Position tensor plus optional captured routes/labels.
    """
    if isinstance(result, torch.Tensor):
        return result, None, None
    return result


def _layout_with_graphviz_engine(
    graph: DaguaGraph,
    engine: str,
    timeout: float,
    seed: Optional[int],
    graph_attributes: Optional[Mapping[str, Any]] = None,
):
    """Run a Graphviz engine and parse the resulting positions.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to lay out.
    engine : str
        Graphviz layout engine name passed through ``dot -K``.
    timeout : float
        Maximum subprocess runtime in seconds.
    seed : int | None
        Optional stochastic seed. When provided, both ``seed`` and ``start``
        graph attributes are passed because fdp reads ``seed`` while neato and
        sfdp read ``start``.
    graph_attributes : Mapping[str, Any] | None, default=None
        Optional Graphviz graph attributes passed as ``-G`` command-line
        overrides.

    Returns
    -------
    tuple
        ``(positions, routes, edge_label_positions)`` where positions has
        shape ``[N, 2]`` and the drawing fields may be ``None``. Callers must
        normalize through :func:`_coerce_layout_capture` because tests stub
        this helper with a bare tensor.
    """
    from dagua.graphviz_utils import to_dot

    command = ["dot", "-Tjson", f"-K{engine}"]
    if seed is not None:
        command.extend([f"-Gseed={int(seed)}", f"-Gstart={int(seed)}"])
    if graph_attributes is not None:
        for key, value in graph_attributes.items():
            if value is None:
                continue
            command.append(f"-G{key}={_graphviz_attribute_value(value)}")

    node_sizes = None
    if graph.node_sizes is not None and size_aware_externals():
        node_sizes = graph.node_sizes
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".dot",
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write(to_dot(graph, node_sizes=node_sizes))
        dot_path = Path(handle.name)

    try:
        result = subprocess.run(
            [*command, str(dot_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Graphviz failed: {result.stderr}")
        data = json.loads(result.stdout)
    finally:
        dot_path.unlink(missing_ok=True)

    positions = _parse_graphviz_json_positions(data, graph.num_nodes)
    routes, edge_label_positions = _parse_graphviz_json_drawing(data, graph)
    return positions, routes, edge_label_positions


class _GraphvizBase(CompetitorBase):
    """Base class for Graphviz engine variants."""

    engine: str = "dot"

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the configured Graphviz engine for a graph.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            Optional Graphviz stochastic seed. The adapter passes this through
            as both ``seed`` and ``start`` graph attributes because Graphviz
            engines read different names.

        Returns
        -------
        CompetitorResult
            Layout result and timing information.
        """
        start = time.perf_counter()
        # Size-aware fairness (r80-P6 follow-up): spring engines do not remove
        # node overlaps unless overlap removal is requested -- passing real
        # node sizes WITHOUT it makes sfdp/neato strictly worse (grid_20x20
        # overlaps 1000 -> 1774). Graphviz's documented practice for sized
        # nodes is overlap=prism, so size-aware runs get the engine at its
        # documented best ("strongest honest external").
        graph_attributes: Optional[dict[str, Any]] = None
        if self.engine in {"sfdp", "neato", "fdp"} and size_aware_externals():
            graph_attributes = {"overlap": "prism"}
        try:
            pos, routes, edge_label_positions = _coerce_layout_capture(
                _layout_with_graphviz_engine(
                    graph=graph,
                    engine=self.engine,
                    timeout=timeout,
                    seed=seed,
                    graph_attributes=graph_attributes,
                )
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=pos,
                runtime_seconds=elapsed,
                routes=routes,
                edge_label_positions=edge_label_positions,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error="timeout",
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the configured Graphviz engine with variant graph attributes.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            Optional Graphviz stochastic seed.
        variant_params : Mapping[str, Any] | None, default=None
            Graphviz graph attributes forwarded as ``-G`` flags.

        Returns
        -------
        CompetitorResult
            Layout result and timing information.
        """
        start = time.perf_counter()
        try:
            pos, routes, edge_label_positions = _coerce_layout_capture(
                _layout_with_graphviz_engine(
                    graph=graph,
                    engine=self.engine,
                    timeout=timeout,
                    seed=seed,
                    graph_attributes=variant_params,
                )
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=pos,
                runtime_seconds=elapsed,
                routes=routes,
                edge_label_positions=edge_label_positions,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error="timeout",
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )

    def available(self) -> bool:
        """Report whether Graphviz is available.

        Returns
        -------
        bool
            ``True`` when the ``dot`` executable can be resolved.
        """
        return _graphviz_available()


@register
class GraphvizDot(_GraphvizBase):
    name = "graphviz_dot"
    engine = "dot"
    max_nodes = 5_000
    supports_clusters = True
    # hgap/vgap are translated to nodesep/ranksep in layout_with_variant;
    # everything else is forwarded verbatim as a -G graph attribute.
    variant_param_names = frozenset({"hgap", "vgap", "maxiter"})

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run Graphviz dot with nested-cluster awareness.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            Accepted for interface consistency. Graphviz ``dot`` is
            deterministic and does not use stochastic seed attributes.

        Returns
        -------
        CompetitorResult
            Layout result and timing information.
        """
        # dot is deterministic; fdp/sfdp/neato seed plumbing lives in the
        # shared Graphviz engine adapter above.
        del seed

        start = time.perf_counter()
        try:
            pos, routes, edge_label_positions = _coerce_layout_capture(
                _layout_with_dot(graph, timeout=timeout)
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=pos,
                runtime_seconds=elapsed,
                routes=routes,
                edge_label_positions=edge_label_positions,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error="timeout",
            )
        except Exception as error:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(error),
            )

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run Graphviz dot with variant graph attributes.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            Accepted for interface consistency. Graphviz ``dot`` is
            deterministic and does not use stochastic seed attributes.
        variant_params : Mapping[str, Any] | None, default=None
            Graphviz graph attributes forwarded as ``-G`` flags.

        Returns
        -------
        CompetitorResult
            Layout result and timing information.
        """
        del seed

        graph_attributes: dict[str, Any] = {}
        if variant_params is not None:
            for key, value in dict(variant_params).items():
                if key == "vgap":
                    graph_attributes["ranksep"] = value
                elif key == "hgap":
                    graph_attributes["nodesep"] = value
                else:
                    graph_attributes[key] = value

        start = time.perf_counter()
        try:
            pos, routes, edge_label_positions = _coerce_layout_capture(
                _layout_with_dot(
                    graph,
                    timeout=timeout,
                    graph_attributes=graph_attributes or None,
                )
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=pos,
                runtime_seconds=elapsed,
                routes=routes,
                edge_label_positions=edge_label_positions,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error="timeout",
            )
        except Exception as error:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(error),
            )


@register
class GraphvizSfdp(_GraphvizBase):
    name = "graphviz_sfdp"
    engine = "sfdp"
    max_nodes = 100_000
    variant_param_names = frozenset({"K", "maxiter", "repulsiveforce", "theta"})


@register
class GraphvizNeato(_GraphvizBase):
    name = "graphviz_neato"
    engine = "neato"
    max_nodes = 2_000
    variant_param_names = frozenset({"K", "epsilon", "maxiter", "pack"})


@register
class GraphvizFdp(_GraphvizBase):
    name = "graphviz_fdp"
    engine = "fdp"
    max_nodes = 5_000
    variant_param_names = frozenset({"K", "maxiter"})

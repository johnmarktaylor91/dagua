"""Graphviz competitor adapters — dot, sfdp, neato, fdp engines."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

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


def _layout_with_dot(graph: DaguaGraph, timeout: float) -> torch.Tensor:
    """Run Graphviz `dot` on a graph and parse the resulting positions.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to lay out.
    timeout : float
        Maximum runtime in seconds.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.
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
        result = subprocess.run(
            ["dot", "-Tjson", str(dot_path)],
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

    return positions


class _GraphvizBase(CompetitorBase):
    """Base class for Graphviz engine variants."""

    engine: str = "dot"

    def layout(self, graph: DaguaGraph, timeout: float = 300.0) -> CompetitorResult:
        """Run the configured Graphviz engine for a graph.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.

        Returns
        -------
        CompetitorResult
            Layout result and timing information.
        """
        from dagua.graphviz_utils import layout_with_graphviz

        start = time.perf_counter()
        try:
            pos = layout_with_graphviz(graph, engine=self.engine, timeout=timeout)
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
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
        return _graphviz_available()


@register
class GraphvizDot(_GraphvizBase):
    name = "graphviz_dot"
    engine = "dot"
    max_nodes = 5_000
    supports_clusters = True

    def layout(self, graph: DaguaGraph, timeout: float = 300.0) -> CompetitorResult:
        """Run Graphviz dot with nested-cluster awareness.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.

        Returns
        -------
        CompetitorResult
            Layout result and timing information.
        """
        start = time.perf_counter()
        try:
            pos = _layout_with_dot(graph, timeout=timeout)
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
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


@register
class GraphvizNeato(_GraphvizBase):
    name = "graphviz_neato"
    engine = "neato"
    max_nodes = 2_000


@register
class GraphvizFdp(_GraphvizBase):
    name = "graphviz_fdp"
    engine = "fdp"
    max_nodes = 5_000

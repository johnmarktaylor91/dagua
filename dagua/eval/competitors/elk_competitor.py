"""ELK competitor adapter — elkjs via Node.js subprocess.

Size policy (r80-P6): elk_layered natively accepts per-node width/height in
the JSON request, in the same point units dagua uses. When size-aware
externals are enabled (the default; see ``dagua.eval.size_policy``), real
per-node sizes from ``graph.node_sizes`` are submitted instead of the old
hardcoded 120x40 placeholder box. ``--size-blind-externals`` restores the
placeholder for store-compatibility experiments.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register
from dagua.eval.size_policy import size_aware_externals

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_DEFAULT_NODE_WIDTH = 120.0
_DEFAULT_NODE_HEIGHT = 40.0
_ELKJS_NODE_MODULES = Path("/home/jtaylor/projects/dagua/node_modules")
_ELK_SECONDARY_ALGORITHMS: Dict[str, str] = {
    "elk_force": "org.eclipse.elk.force",
    "elk_stress": "org.eclipse.elk.stress",
    "elk_mrtree": "org.eclipse.elk.mrtree",
    "elk_radial": "org.eclipse.elk.radial",
}


def _node_wh(graph: DaguaGraph, node_index: int) -> Tuple[float, float]:
    """Return the width/height ELK should use for one node.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    node_index : int
        Node index.

    Returns
    -------
    Tuple[float, float]
        ``(width, height)`` in point units: the real label-measured size
        when ``graph.node_sizes`` is populated and size-aware externals are
        enabled, otherwise the historical 120x40 placeholder.
    """
    if graph.node_sizes is not None and size_aware_externals():
        return (
            float(graph.node_sizes[node_index, 0].item()),
            float(graph.node_sizes[node_index, 1].item()),
        )
    return (_DEFAULT_NODE_WIDTH, _DEFAULT_NODE_HEIGHT)


_ELK_SCRIPT = r"""
const ELK = require('elkjs');
const elk = new ELK();
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
    const graph = JSON.parse(input);
    elk.layout(graph).then((result) => {
        process.stdout.write(JSON.stringify(result));
    }).catch((err) => {
        process.stderr.write(err.toString());
        process.exit(1);
    });
});
"""


def _node_subprocess_env() -> Dict[str, str]:
    """Return a Node environment that can resolve the repo-local elkjs install.

    Returns
    -------
    dict[str, str]
        Environment variables for the Node subprocess.
    """
    env = dict(os.environ)
    if _ELKJS_NODE_MODULES.exists():
        existing = env.get("NODE_PATH", "")
        paths = [str(_ELKJS_NODE_MODULES)]
        if existing:
            paths.append(existing)
        env["NODE_PATH"] = os.pathsep.join(paths)
    return env


def _build_flat_elk_graph(
    graph: DaguaGraph,
    algorithm_id: str,
    layout_options: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build a flat ELK graph for secondary algorithm references.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    algorithm_id : str
        ELK algorithm identifier.
    layout_options : dict[str, object] or None
        Extra layout options merged after the pinned defaults.

    Returns
    -------
    dict[str, object]
        JSON-compatible ELK graph.
    """
    children: List[Dict[str, object]] = []
    for node_index in range(graph.num_nodes):
        node_w, node_h = _node_wh(graph, node_index)
        children.append({"id": str(node_index), "width": node_w, "height": node_h})

    edges: List[Dict[str, object]] = []
    if graph.edge_index.numel() > 0:
        for e_idx in range(graph.edge_index.shape[1]):
            s = graph.edge_index[0, e_idx].item()
            t = graph.edge_index[1, e_idx].item()
            edges.append({"id": f"e{e_idx}", "sources": [str(s)], "targets": [str(t)]})

    options: Dict[str, object] = {
        "elk.algorithm": algorithm_id,
        "elk.randomSeed": 1,
        "elk.spacing.nodeNode": 80,
        "elk.separateConnectedComponents": False,
    }
    if layout_options is not None:
        options.update(layout_options)
    return {"id": "root", "layoutOptions": options, "children": children, "edges": edges}


def _run_elk_layout_json(
    elk_graph: Dict[str, object],
    timeout: float,
) -> Tuple[Optional[Dict[str, object]], float, Optional[str]]:
    """Run elkjs and parse the output graph.

    Parameters
    ----------
    elk_graph : dict[str, object]
        JSON-compatible ELK graph.
    timeout : float
        Maximum runtime in seconds.

    Returns
    -------
    tuple[dict[str, object] | None, float, str | None]
        Parsed graph, elapsed seconds, and optional error.
    """
    graph_json = json.dumps(elk_graph)
    graph_kb = len(graph_json) // 1024
    heap_mb = min(65536, max(16384, graph_kb * 48))
    start = time.perf_counter()
    try:
        result = subprocess.run(
            ["node", f"--max-old-space-size={heap_mb}", "-e", _ELK_SCRIPT],
            input=graph_json,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_node_subprocess_env(),
        )
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            return None, elapsed, result.stderr[:500]
        return json.loads(result.stdout), elapsed, None
    except subprocess.TimeoutExpired:
        return None, time.perf_counter() - start, "timeout"
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return None, time.perf_counter() - start, str(exc)


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


def _build_elk_children(
    graph: DaguaGraph,
    parent_name: Optional[str],
    children_by_parent: Dict[Optional[str], List[str]],
    emitted_nodes: Set[int],
) -> List[Dict[str, object]]:
    """Build nested ELK children for a cluster subtree.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    parent_name : str or None
        Parent cluster whose children should be emitted. ``None`` refers to
        the root graph.
    children_by_parent : dict[Optional[str], list[str]]
        Parent-to-children cluster hierarchy.
    emitted_nodes : set[int]
        Nodes already assigned to a cluster container.

    Returns
    -------
    list[dict[str, object]]
        ELK child objects for the requested subtree.
    """
    children: List[Dict[str, object]] = []
    for cluster_name in children_by_parent.get(parent_name, []):
        nested_children = _build_elk_children(
            graph,
            cluster_name,
            children_by_parent,
            emitted_nodes,
        )
        descendant_members: Set[int] = set()
        for child_name in children_by_parent.get(cluster_name, []):
            descendant_members.update(_cluster_members(graph, child_name))

        direct_members: List[Dict[str, object]] = []
        for node_index in _cluster_members(graph, cluster_name):
            if node_index in descendant_members or node_index in emitted_nodes:
                continue
            node_w, node_h = _node_wh(graph, node_index)
            direct_members.append({"id": str(node_index), "width": node_w, "height": node_h})
            emitted_nodes.add(node_index)

        cluster_entry: Dict[str, object] = {
            "id": f"cluster_{cluster_name}",
            "children": [*nested_children, *direct_members],
        }
        cluster_label = graph.cluster_labels.get(cluster_name)
        if cluster_label is not None:
            cluster_entry["labels"] = [{"text": cluster_label}]
        children.append(cluster_entry)

    if parent_name is None:
        for node_index in range(graph.num_nodes):
            if node_index not in emitted_nodes:
                node_w, node_h = _node_wh(graph, node_index)
                children.append({"id": str(node_index), "width": node_w, "height": node_h})

    return children


def _collect_elk_positions(
    children: object,
    positions: torch.Tensor,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> None:
    """Collect node positions from nested ELK output.

    Parameters
    ----------
    children : object
        ELK ``children`` field for the current container.
    positions : torch.Tensor
        Output tensor with shape ``[N, 2]``.
    offset_x : float, default=0.0
        X offset accumulated from parent containers.
    offset_y : float, default=0.0
        Y offset accumulated from parent containers.

    Returns
    -------
    None
        The function mutates ``positions`` in place.
    """
    if not isinstance(children, list):
        return

    for child in children:
        if not isinstance(child, dict):
            continue
        child_x = offset_x + float(child.get("x", 0.0))
        child_y = offset_y + float(child.get("y", 0.0))
        child_id = str(child.get("id", ""))
        if child_id.isdigit():
            node_index = int(child_id)
            if node_index < positions.shape[0]:
                positions[node_index, 0] = child_x
                positions[node_index, 1] = child_y
        _collect_elk_positions(child.get("children", []), positions, child_x, child_y)


def _collect_elk_routes(
    data: dict,
    num_edges: int,
) -> Optional[List[Optional[List[Tuple[float, float]]]]]:
    """Parse ELK edge sections into per-edge polylines (r80-S6).

    Every benchmark edge is submitted at root level with id ``e{idx}``, so
    section coordinates are relative to the root and need no offsetting.
    Each section contributes ``startPoint -> bendPoints... -> endPoint``.

    Parameters
    ----------
    data : dict
        Parsed ELK JSON output.
    num_edges : int
        Expected edge count.

    Returns
    -------
    list | None
        Per-edge polylines aligned to ``edge_index`` columns (``None``
        entries for unrouted edges), or ``None`` when no edge carries
        routing sections.
    """
    edges = data.get("edges")
    if not isinstance(edges, list) or num_edges <= 0:
        return None

    routes: List[Optional[List[Tuple[float, float]]]] = [None] * num_edges
    any_route = False
    for edge_obj in edges:
        if not isinstance(edge_obj, dict):
            continue
        edge_id = str(edge_obj.get("id", ""))
        if not edge_id.startswith("e") or not edge_id[1:].isdigit():
            continue
        e_idx = int(edge_id[1:])
        if e_idx >= num_edges:
            continue
        sections = edge_obj.get("sections")
        if not isinstance(sections, list) or not sections:
            continue
        polyline: List[Tuple[float, float]] = []
        for section in sections:
            if not isinstance(section, dict):
                continue
            points = [section.get("startPoint")]
            bend_points = section.get("bendPoints")
            if isinstance(bend_points, list):
                points.extend(bend_points)
            points.append(section.get("endPoint"))
            for point in points:
                if not isinstance(point, dict):
                    continue
                try:
                    xy = (float(point["x"]), float(point["y"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if polyline and polyline[-1] == xy:
                    continue
                polyline.append(xy)
        if len(polyline) >= 2:
            routes[e_idx] = polyline
            any_route = True

    return routes if any_route else None


@register
class ElkLayered(CompetitorBase):
    name = "elk_layered"
    max_nodes = 15_000
    supports_clusters = True

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run ELK layered layout with nested clusters mapped to group nodes.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            ELK random seed. ``None`` pins seed 1.

        Returns
        -------
        CompetitorResult
            Layout result and timing information.
        """
        n = graph.num_nodes
        emitted_nodes: Set[int] = set()
        children = _build_elk_children(graph, None, _cluster_children(graph), emitted_nodes)
        edges = []
        if graph.edge_index.numel() > 0:
            for e_idx in range(graph.edge_index.shape[1]):
                s = graph.edge_index[0, e_idx].item()
                t = graph.edge_index[1, e_idx].item()
                edges.append({"id": f"e{e_idx}", "sources": [str(s)], "targets": [str(t)]})

        elk_graph = {
            "id": "root",
            "layoutOptions": {
                "elk.algorithm": "layered",
                "elk.direction": "DOWN",
                "elk.spacing.nodeNode": "40",
                "elk.layered.spacing.nodeNodeBetweenLayers": "60",
                "elk.layered.thoroughness": "7",
                "elk.randomSeed": 1 if seed is None else int(seed),
            },
            "children": children,
            "edges": edges,
        }

        data, elapsed, error = _run_elk_layout_json(elk_graph, timeout)
        if data is None:
            return CompetitorResult(name=self.name, pos=None, runtime_seconds=elapsed, error=error)

        # Parse positions from ELK output
        pos = torch.zeros(n, 2)
        _collect_elk_positions(data.get("children", []), pos)
        num_edges = graph.edge_index.shape[1] if graph.edge_index.numel() > 0 else 0
        routes = _collect_elk_routes(data, int(num_edges))

        return CompetitorResult(
            name=self.name,
            pos=pos,
            runtime_seconds=elapsed,
            routes=routes,
        )

    def available(self) -> bool:
        try:
            result = subprocess.run(
                ["node", "-e", "require('elkjs')"],
                capture_output=True,
                timeout=10,
                env=_node_subprocess_env(),
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


class _ElkSecondary(CompetitorBase):
    """Base adapter for flat ELK secondary algorithms."""

    name = "elk_secondary"
    algorithm_id = ""
    max_nodes = 15_000
    supports_clusters = False

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run one flat ELK secondary algorithm.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            ELK random seed. ``None`` pins seed 1.

        Returns
        -------
        CompetitorResult
            Layout result and timing information.
        """
        options = {"elk.randomSeed": 1 if seed is None else int(seed)}
        elk_graph = _build_flat_elk_graph(graph, self.algorithm_id, options)
        data, elapsed, error = _run_elk_layout_json(elk_graph, timeout)
        if data is None:
            return CompetitorResult(name=self.name, pos=None, runtime_seconds=elapsed, error=error)

        pos = torch.zeros(graph.num_nodes, 2)
        _collect_elk_positions(data.get("children", []), pos)
        num_edges = graph.edge_index.shape[1] if graph.edge_index.numel() > 0 else 0
        routes = _collect_elk_routes(data, int(num_edges))
        return CompetitorResult(
            name=self.name,
            pos=pos,
            runtime_seconds=elapsed,
            routes=routes,
        )

    def available(self) -> bool:
        """Return whether elkjs is available to Node.

        Returns
        -------
        bool
            ``True`` when the adapter can import ``elkjs``.
        """
        try:
            result = subprocess.run(
                ["node", "-e", "require('elkjs')"],
                capture_output=True,
                timeout=10,
                env=_node_subprocess_env(),
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


@register
class ElkForce(_ElkSecondary):
    """ELK Force reference adapter."""

    name = "elk_force"
    algorithm_id = _ELK_SECONDARY_ALGORITHMS[name]


@register
class ElkStress(_ElkSecondary):
    """ELK Stress reference adapter."""

    name = "elk_stress"
    algorithm_id = _ELK_SECONDARY_ALGORITHMS[name]


@register
class ElkMrTree(_ElkSecondary):
    """ELK MrTree reference adapter."""

    name = "elk_mrtree"
    algorithm_id = _ELK_SECONDARY_ALGORITHMS[name]


@register
class ElkRadial(_ElkSecondary):
    """ELK Radial reference adapter."""

    name = "elk_radial"
    algorithm_id = _ELK_SECONDARY_ALGORITHMS[name]

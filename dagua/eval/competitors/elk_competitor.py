"""ELK competitor adapter — elkjs via Node.js subprocess."""

from __future__ import annotations

import json
import subprocess
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

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
            direct_members.append({"id": str(node_index), "width": 120, "height": 40})
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
                children.append({"id": str(node_index), "width": 120, "height": 40})

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
            Accepted for interface consistency but ignored because ELK layered
            is deterministic for a fixed input.

        Returns
        -------
        CompetitorResult
            Layout result and timing information.
        """
        del seed

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
            },
            "children": children,
            "edges": edges,
        }

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
            )
            elapsed = time.perf_counter() - start

            if result.returncode != 0:
                return CompetitorResult(
                    name=self.name,
                    pos=None,
                    runtime_seconds=elapsed,
                    error=result.stderr[:500],
                )

            # Parse positions from ELK output
            data = json.loads(result.stdout)
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
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name, pos=None, runtime_seconds=elapsed, error="timeout"
            )
        except (FileNotFoundError, json.JSONDecodeError) as e:
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=None, runtime_seconds=elapsed, error=str(e))

    def available(self) -> bool:
        try:
            result = subprocess.run(
                ["node", "-e", "require('elkjs')"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

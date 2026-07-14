"""Dagre competitor adapter — dagre via Node.js subprocess.

Size policy (r80-P6): dagre natively accepts per-node width/height in its
graph payload, in the same point units dagua uses. When size-aware
externals are enabled (the default; see ``dagua.eval.size_policy``), real
per-node sizes from ``graph.node_sizes`` are submitted per node instead of
the old hardcoded 120x40 placeholder box. ``--size-blind-externals``
restores the placeholder for store-compatibility experiments.
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
_DURABLE_NODE_MODULES = Path.home() / "tools" / "dagua-refs" / "node_modules"
_DAGRE_NODE_MODULES = Path("/home/jtaylor/projects/dagua/node_modules")


def _node_subprocess_env() -> Dict[str, str]:
    """Return a Node environment that can resolve repo-local dagre packages.

    Returns
    -------
    dict[str, str]
        Environment variables for Dagre subprocesses.
    """
    env = dict(os.environ)
    paths = [str(path) for path in (_DURABLE_NODE_MODULES, _DAGRE_NODE_MODULES) if path.exists()]
    existing = env.get("NODE_PATH", "")
    if existing:
        paths.append(existing)
    if paths:
        env["NODE_PATH"] = os.pathsep.join(paths)
    return env


def _node_wh(graph: DaguaGraph, node_index: int) -> Tuple[float, float]:
    """Return the width/height dagre should use for one node.

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


_DAGRE_SCRIPT = r"""
const dagre = require('dagre');
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
    const data = JSON.parse(input);
    const g = new dagre.graphlib.Graph({ compound: true });
    g.setGraph({ rankdir: 'TB', nodesep: 40, ranksep: 60 });
    g.setDefaultEdgeLabel(function() { return {}; });
    for (const cluster of data.clusters) {
        g.setNode(cluster.id, { label: cluster.label });
    }
    for (const node of data.nodes) {
        g.setNode(node.id, { width: node.width || 120, height: node.height || 40 });
    }
    for (const parent of data.parents) {
        g.setParent(parent.child, parent.parent);
    }
    for (const edge of data.edges) {
        g.setEdge(edge.source, edge.target);
    }
    dagre.layout(g);
    const result = {};
    g.nodes().forEach(function(v) {
        const n = g.node(v);
        result[v] = { x: n.x, y: n.y };
    });
    process.stdout.write(JSON.stringify(result));
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


def _collect_dagre_cluster_data(
    graph: DaguaGraph,
    cluster_name: str,
    children_by_parent: Dict[Optional[str], List[str]],
    cluster_nodes: List[Dict[str, str]],
    parents: List[Dict[str, str]],
    emitted_nodes: Set[int],
) -> None:
    """Collect compound-graph nodes and parent relations for Dagre.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    cluster_name : str
        Cluster currently being emitted.
    children_by_parent : dict[Optional[str], list[str]]
        Parent-to-children cluster hierarchy.
    cluster_nodes : list[dict[str, str]]
        Output list of synthetic cluster nodes.
    parents : list[dict[str, str]]
        Output parent assignments for nodes and child clusters.
    emitted_nodes : set[int]
        Nodes already assigned to a direct cluster parent.

    Returns
    -------
    None
        The function mutates the provided output lists in place.
    """
    cluster_id = f"cluster_{cluster_name}"
    cluster_nodes.append(
        {
            "id": cluster_id,
            "label": graph.cluster_labels.get(cluster_name, cluster_name),
        }
    )

    parent_name = graph.cluster_parents.get(cluster_name)
    if parent_name in graph.clusters:
        parents.append({"child": cluster_id, "parent": f"cluster_{parent_name}"})

    child_clusters = children_by_parent.get(cluster_name, [])
    for child_name in child_clusters:
        _collect_dagre_cluster_data(
            graph,
            child_name,
            children_by_parent,
            cluster_nodes,
            parents,
            emitted_nodes,
        )

    descendant_members: Set[int] = set()
    for child_name in child_clusters:
        descendant_members.update(_cluster_members(graph, child_name))

    for node_index in _cluster_members(graph, cluster_name):
        if node_index in descendant_members or node_index in emitted_nodes:
            continue
        parents.append({"child": str(node_index), "parent": cluster_id})
        emitted_nodes.add(node_index)


def _build_dagre_input(graph: DaguaGraph) -> Dict[str, object]:
    """Build the Dagre input payload with compound-cluster metadata.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to convert.

    Returns
    -------
    dict[str, object]
        JSON-serializable Dagre input payload.
    """
    nodes = []
    for index in range(graph.num_nodes):
        node_w, node_h = _node_wh(graph, index)
        nodes.append({"id": str(index), "width": node_w, "height": node_h})
    edges: List[Dict[str, str]] = []
    if graph.edge_index.numel() > 0:
        for edge_index in range(graph.edge_index.shape[1]):
            source = int(graph.edge_index[0, edge_index].item())
            target = int(graph.edge_index[1, edge_index].item())
            edges.append({"source": str(source), "target": str(target)})

    cluster_nodes: List[Dict[str, str]] = []
    parents: List[Dict[str, str]] = []
    emitted_nodes: Set[int] = set()
    children_by_parent = _cluster_children(graph)
    for cluster_name in children_by_parent.get(None, []):
        _collect_dagre_cluster_data(
            graph,
            cluster_name,
            children_by_parent,
            cluster_nodes,
            parents,
            emitted_nodes,
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "clusters": cluster_nodes,
        "parents": parents,
    }


@register
class DagreCompetitor(CompetitorBase):
    name = "dagre"
    max_nodes = 1_500  # JS stack overflow at 2000 on dense graphs
    supports_clusters = True

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run Dagre with compound-graph cluster assignments.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            Accepted for interface consistency but ignored because Dagre is
            deterministic for a fixed input.

        Returns
        -------
        CompetitorResult
            Layout result and timing information.
        """
        del seed

        n = graph.num_nodes
        input_data = json.dumps(_build_dagre_input(graph))

        start = time.perf_counter()
        try:
            result = subprocess.run(
                ["node", "-e", _DAGRE_SCRIPT],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_node_subprocess_env(),
            )
            elapsed = time.perf_counter() - start

            if result.returncode != 0:
                return CompetitorResult(
                    name=self.name,
                    pos=None,
                    runtime_seconds=elapsed,
                    error=result.stderr[:500],
                )

            data = json.loads(result.stdout)
            pos = torch.zeros(n, 2)
            for i in range(n):
                node_data = data.get(str(i), {})
                pos[i, 0] = node_data.get("x", 0)
                pos[i, 1] = node_data.get("y", 0)

            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
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
                ["node", "-e", "require('dagre')"],
                capture_output=True,
                timeout=10,
                env=_node_subprocess_env(),
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

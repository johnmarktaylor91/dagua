"""Cytoscape layout competitor adapter via a Node.js subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_DURABLE_NODE_MODULES = Path.home() / "tools" / "dagua-refs" / "node_modules"


def _node_subprocess_env() -> Dict[str, str]:
    """Return a Node environment that can resolve durable Cytoscape packages.

    Returns
    -------
    dict[str, str]
        Environment variables for Cytoscape subprocesses.
    """
    env = dict(os.environ)
    paths = [str(_DURABLE_NODE_MODULES)] if _DURABLE_NODE_MODULES.exists() else []
    existing = env.get("NODE_PATH", "")
    if existing:
        paths.append(existing)
    if paths:
        env["NODE_PATH"] = os.pathsep.join(paths)
    return env


_CYTOSCAPE_SCRIPT = r"""
const cytoscape = require('cytoscape');
const coseBilkent = require('cytoscape-cose-bilkent');
const cise = require('cytoscape-cise');
const avsdf = require('cytoscape-avsdf');
cytoscape.use(coseBilkent);
cytoscape.use(cise);
cytoscape.use(avsdf);

function seededRandom(seed) {
    let state = (Number(seed) >>> 0) || 1;
    return function() {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
    const data = JSON.parse(input);
    const elements = [];
    for (const cluster of data.clusters || []) {
        elements.push({data: {id: cluster.id}});
    }
    for (let i = 0; i < data.num_nodes; i++) {
        const node = {data: {id: String(i)}};
        if (data.parents && data.parents[String(i)]) {
            node.data.parent = data.parents[String(i)];
        }
        if (data.positions && data.positions[String(i)]) {
            node.position = {x: data.positions[String(i)][0], y: data.positions[String(i)][1]};
        }
        elements.push(node);
    }
    for (const [s, t] of data.edges) {
        elements.push({data: {id: 'e_' + s + '_' + t, source: String(s), target: String(t)}});
    }
    const cy = cytoscape({elements, headless: true, styleEnabled: true});
    const layoutOpts = Object.assign(
        {name: data.name, animate: false, fit: false},
        data.options || {}
    );
    if (data.name === 'cise' && layoutOpts.clusters === undefined) {
        layoutOpts.clusters = data.clusterGroups || [];
    }
    const originalRandom = Math.random;
    if (data.seed !== undefined && data.seed !== null) {
        Math.random = seededRandom(data.seed);
    }
    try {
        cy.layout(layoutOpts).run();
    } finally {
        Math.random = originalRandom;
    }
    const positions = {};
    cy.nodes().forEach(n => {
        if (!n.isParent()) {
            positions[n.id()] = [n.position().x, n.position().y];
        }
    });
    process.stdout.write(JSON.stringify(positions));
    process.exit(0);
});
"""


def _graph_to_edge_list(graph: DaguaGraph) -> List[List[int]]:
    """Convert graph edges into a JSON-serializable list.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose ``edge_index`` should be serialized.

    Returns
    -------
    list[list[int]]
        Edge list in ``[[source, target], ...]`` format.
    """
    if graph.edge_index.numel() == 0:
        return []
    edge_index = graph.edge_index.cpu()
    return [
        [int(edge_index[0, edge_idx].item()), int(edge_index[1, edge_idx].item())]
        for edge_idx in range(edge_index.shape[1])
    ]


def _cluster_payload(
    graph: DaguaGraph,
) -> tuple[List[Dict[str, str]], Dict[str, str], List[List[str]]]:
    """Build Cytoscape compound-node payload.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.

    Returns
    -------
    tuple[list[dict[str, str]], dict[str, str], list[list[str]]]
        Cluster elements, node parent mapping, and CiSE cluster groups.
    """
    clusters: List[Dict[str, str]] = []
    parents: Dict[str, str] = {}
    cluster_groups: List[List[str]] = []
    for cluster_name in sorted(getattr(graph, "clusters", {}) or {}):
        cluster_id = f"cluster_{cluster_name}"
        clusters.append({"id": cluster_id})
        members = graph.clusters[cluster_name]
        if isinstance(members, dict):
            continue
        group: List[str] = []
        for member in members:
            node_id = str(int(member))
            parents[node_id] = cluster_id
            group.append(node_id)
        if group:
            cluster_groups.append(group)
    return clusters, parents, cluster_groups


def _positions_payload(pos: Optional[torch.Tensor]) -> Dict[str, List[float]]:
    """Serialize optional warm-start positions.

    Parameters
    ----------
    pos : torch.Tensor | None
        Optional tensor with shape ``[N, 2]``.

    Returns
    -------
    dict[str, list[float]]
        Position mapping keyed by node id.
    """
    if pos is None:
        return {}
    cpu_pos = pos.detach().cpu()
    return {
        str(index): [float(cpu_pos[index, 0].item()), float(cpu_pos[index, 1].item())]
        for index in range(cpu_pos.shape[0])
    }


def _positions_to_tensor(
    raw_positions: Mapping[str, Sequence[float]],
    num_nodes: int,
) -> torch.Tensor:
    """Convert Node.js JSON positions into a tensor.

    Parameters
    ----------
    raw_positions : Mapping[str, Sequence[float]]
        Node-position mapping emitted by the helper.
    num_nodes : int
        Number of nodes expected in the result.

    Returns
    -------
    torch.Tensor
        CPU float tensor with shape ``[N, 2]``.
    """
    pos = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for node_idx in range(num_nodes):
        coords = raw_positions.get(str(node_idx))
        if coords is None or len(coords) < 2:
            continue
        pos[node_idx, 0] = float(coords[0])
        pos[node_idx, 1] = float(coords[1])
    return pos


@register
class CytoscapeCompetitor(CompetitorBase):
    """Cytoscape layout adapter for CoSE-family and circular extensions."""

    name = "cytoscape"
    max_nodes = 10_000
    variant_param_names = frozenset(
        {
            "layout",
            "quality",
            "randomize",
            "nodeRepulsion",
            "idealEdgeLength",
            "edgeElasticity",
            "gravity",
            "gravityRange",
            "gravityCompound",
            "gravityRangeCompound",
            "numIter",
            "nodeSeparation",
        }
    )

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run Cytoscape core CoSE by default.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            Optional seed for patched ``Math.random``.

        Returns
        -------
        CompetitorResult
            Layout result and timing information.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run a Cytoscape layout with optional parameter overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            Optional seed for patched ``Math.random``.
        variant_params : Mapping[str, Any] | None, default=None
            Options merged into the Cytoscape layout object. Use ``layout`` to
            select ``"cose"``, ``"cose-bilkent"``, ``"cise"``, or ``"avsdf"``.

        Returns
        -------
        CompetitorResult
            Layout result and timing information.
        """
        options = dict(variant_params) if variant_params is not None else {}
        layout_name = str(options.pop("layout", "cose"))
        clusters, parents, cluster_groups = _cluster_payload(graph)
        payload: Dict[str, object] = {
            "name": layout_name,
            "num_nodes": graph.num_nodes,
            "edges": _graph_to_edge_list(graph),
            "clusters": clusters,
            "clusterGroups": cluster_groups,
            "parents": parents,
            "positions": _positions_payload(options.pop("positions", None)),
            "seed": seed,
            "options": options,
        }
        start = time.perf_counter()
        try:
            result = subprocess.run(
                ["node", "-e", _CYTOSCAPE_SCRIPT],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                env=_node_subprocess_env(),
                timeout=timeout,
            )
            elapsed = time.perf_counter() - start
            if result.returncode != 0:
                return CompetitorResult(
                    name=self.name,
                    pos=None,
                    runtime_seconds=elapsed,
                    error=(result.stderr or result.stdout)[:500],
                )
            data = json.loads(result.stdout)
            if not isinstance(data, dict):
                raise ValueError("Cytoscape helper returned non-object JSON")
            return CompetitorResult(
                name=self.name,
                pos=_positions_to_tensor(data, graph.num_nodes),
                runtime_seconds=elapsed,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error="timeout",
            )
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(error),
            )

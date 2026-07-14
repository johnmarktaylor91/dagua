"""d3-dag competitor adapter via Node.js subprocess."""

from __future__ import annotations

import json
import subprocess
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register
from dagua.eval.size_policy import size_aware_externals

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_DEFAULT_NODE_WIDTH = 1.0
_DEFAULT_NODE_HEIGHT = 1.0

_D3DAG_SCRIPT = r"""
const {
  graph,
  sugiyama,
  layeringLongestPath,
  decrossOpt,
  decrossDfs,
  coordGreedy
} = require('d3-dag');

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
  try {
    const data = JSON.parse(input);
    const dag = graph();
    const nodes = {};
    for (const node of data.nodes) {
      nodes[String(node.id)] = dag.node(String(node.id));
    }
    for (const edge of data.edges) {
      nodes[String(edge.source)].child(nodes[String(edge.target)]);
    }
    let layout = sugiyama().nodeSize((node) => {
      const size = data.sizes[String(node.data)] || [1, 1];
      return [size[0], size[1]];
    }).gap([data.x_gap, data.y_gap]);
    if (data.layering === 'longestPath') {
      layout = layout.layering(layeringLongestPath());
    }
    if (data.decross === 'opt') {
      layout = layout.decross(decrossOpt());
    } else if (data.decross === 'dfs') {
      layout = layout.decross(decrossDfs());
    }
    if (data.coord === 'greedy') {
      layout = layout.coord(coordGreedy());
    }
    layout(dag);
    const result = {};
    for (const node of dag.nodes()) {
      result[String(node.data)] = { x: node.x, y: node.y };
    }
    process.stdout.write(JSON.stringify({ ok: true, positions: result }));
  } catch (err) {
    process.stdout.write(JSON.stringify({ ok: false, error: String(err && err.stack || err) }));
  }
});
"""


def _node_wh(graph: DaguaGraph, node_index: int) -> Tuple[float, float]:
    """Return the d3-dag node width and height for one graph node.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    node_index : int
        Node index to inspect.

    Returns
    -------
    Tuple[float, float]
        Positive ``(width, height)`` in layout units.
    """
    if graph.node_sizes is not None and size_aware_externals():
        return (
            max(float(graph.node_sizes[node_index, 0].item()), 1.0e-9),
            max(float(graph.node_sizes[node_index, 1].item()), 1.0e-9),
        )
    return (_DEFAULT_NODE_WIDTH, _DEFAULT_NODE_HEIGHT)


def _build_input(
    graph: DaguaGraph,
    layering: str,
    decross: str,
    coord: str,
    x_gap: float,
    y_gap: float,
) -> Dict[str, object]:
    """Build a JSON-serializable d3-dag adapter payload.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    layering : str
        Layering operator name.
    decross : str
        Decross operator name.
    coord : str
        Coordinate operator name.
    x_gap : float
        Horizontal d3-dag gap.
    y_gap : float
        Vertical d3-dag gap.

    Returns
    -------
    dict[str, object]
        Payload consumed by the Node subprocess.
    """
    nodes: List[Dict[str, object]] = []
    sizes: Dict[str, List[float]] = {}
    for index in range(graph.num_nodes):
        width, height = _node_wh(graph, index)
        node_id = str(index)
        nodes.append({"id": node_id})
        sizes[node_id] = [width, height]

    edges: List[Dict[str, str]] = []
    if graph.edge_index.numel() > 0:
        for edge_index in range(graph.edge_index.shape[1]):
            source = int(graph.edge_index[0, edge_index].item())
            target = int(graph.edge_index[1, edge_index].item())
            if source != target:
                edges.append({"source": str(source), "target": str(target)})
    if not edges and graph.num_nodes > 1:
        edges = [{"source": "0", "target": "1"}]

    return {
        "nodes": nodes,
        "sizes": sizes,
        "edges": edges,
        "layering": layering,
        "decross": decross,
        "coord": coord,
        "x_gap": x_gap,
        "y_gap": y_gap,
    }


@register
class D3DagCompetitor(CompetitorBase):
    """Run d3-dag Sugiyama through a local Node.js subprocess."""

    name = "d3dag"
    max_nodes = 1_000
    supports_clusters = False
    variant_param_names = frozenset({"layering", "decross", "coord", "x_gap", "y_gap"})

    def available(self) -> bool:
        """Check whether the local ``d3-dag`` package can be required.

        Returns
        -------
        bool
            ``True`` when Node can load ``d3-dag``.
        """
        try:
            subprocess.run(
                ["node", "-e", "require('d3-dag');"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10.0,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return True

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run d3-dag's default Sugiyama layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime.
        seed : int | None, default=None
            Accepted for competitor API compatibility; d3-dag is deterministic.

        Returns
        -------
        CompetitorResult
            Layout result or error.
        """
        del seed
        return self.layout_with_variant(graph, timeout=timeout, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[object] = None,
    ) -> CompetitorResult:
        """Run d3-dag with optional Sugiyama stage variants.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime.
        seed : int | None, default=None
            Accepted for API compatibility; d3-dag is deterministic.
        variant_params : object | None, optional
            Mapping with optional ``layering``, ``decross``, ``coord``,
            ``x_gap``, and ``y_gap`` entries.

        Returns
        -------
        CompetitorResult
            Layout result or error.
        """
        del seed
        params = variant_params if isinstance(variant_params, dict) else {}
        payload = _build_input(
            graph,
            layering=str(params.get("layering", "simplex")),
            decross=str(params.get("decross", "twoLayer")),
            coord=str(params.get("coord", "simplex")),
            x_gap=float(params.get("x_gap", 1.0)),
            y_gap=float(params.get("y_gap", 1.0)),
        )
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                ["node", "-e", _D3DAG_SCRIPT],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CompetitorResult(
                self.name,
                None,
                time.perf_counter() - start,
                error=f"d3-dag timed out after {timeout:.1f}s",
            )
        runtime = time.perf_counter() - start
        if completed.returncode != 0:
            return CompetitorResult(self.name, None, runtime, error=completed.stderr.strip())
        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            return CompetitorResult(self.name, None, runtime, error=str(exc))
        if not data.get("ok"):
            return CompetitorResult(self.name, None, runtime, error=str(data.get("error")))
        positions = torch.zeros((graph.num_nodes, 2), dtype=torch.float32)
        raw_positions = data["positions"]
        for node in range(graph.num_nodes):
            raw = raw_positions.get(str(node))
            if raw is None:
                continue
            positions[node, 0] = float(raw["x"])
            positions[node, 1] = float(raw["y"])
        return CompetitorResult(self.name, positions, runtime)


__all__ = ["D3DagCompetitor"]

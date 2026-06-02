"""Cytoscape fcose competitor adapter via a Node.js subprocess."""

from __future__ import annotations

import json
import subprocess
import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


_FCOSE_SCRIPT = r"""
const cytoscape = require('cytoscape');
const fcose = require('cytoscape-fcose');
cytoscape.use(fcose);

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
    for (let i = 0; i < data.num_nodes; i++) {
        elements.push({data: {id: String(i)}});
    }
    for (const [s, t] of data.edges) {
        elements.push({data: {id: 'e_' + s + '_' + t, source: String(s), target: String(t)}});
    }
    const cy = cytoscape({elements, headless: true});
    const layoutOpts = Object.assign(
        {name: 'fcose', animate: false, randomize: true},
        data.options || {}
    );
    if (data.seed !== undefined && data.seed !== null) {
        layoutOpts.randomize = true;
    }
    const originalRandom = Math.random;
    if (data.seed !== undefined && data.seed !== null) {
        Math.random = seededRandom(data.seed);
    }
    try {
        const layout = cy.layout(layoutOpts);
        layout.run();
    } finally {
        Math.random = originalRandom;
    }
    const positions = {};
    cy.nodes().forEach(n => {
        positions[n.id()] = [n.position().x, n.position().y];
    });
    process.stdout.write(JSON.stringify(positions));
});
"""


def _graph_to_edge_list(graph: DaguaGraph) -> List[List[int]]:
    """Convert ``edge_index`` into a JSON-serializable edge list.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose edges should be serialized.

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


def _positions_to_tensor(
    raw_positions: Mapping[str, Sequence[float]],
    num_nodes: int,
) -> torch.Tensor:
    """Convert subprocess JSON positions into a ``[N, 2]`` tensor.

    Parameters
    ----------
    raw_positions : Mapping[str, Sequence[float]]
        Node-position mapping emitted by the Node.js helper.
    num_nodes : int
        Number of nodes expected in the output tensor.

    Returns
    -------
    torch.Tensor
        CPU float tensor shaped ``[N, 2]``.

    Raises
    ------
    ValueError
        Raised when the subprocess emits malformed coordinate data.
    """
    pos = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for node_idx in range(num_nodes):
        coords = raw_positions.get(str(node_idx))
        if coords is None:
            continue
        if len(coords) < 2:
            msg = f"missing coordinates for node {node_idx}"
            raise ValueError(msg)
        pos[node_idx, 0] = float(coords[0])
        pos[node_idx, 1] = float(coords[1])
    return pos


@register
class CytoscapeFcose(CompetitorBase):
    """Cytoscape fcose layout adapter."""

    name = "cytoscape_fcose"
    max_nodes = 10_000
    variant_param_names = frozenset(
        {
            "idealEdgeLength",
            "nodeRepulsion",
            "nodeSeparation",
            "numIter",
            "quality",
            "randomize",
        }
    )

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run Cytoscape fcose with default adapter options.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            Optional benchmark seed forwarded to the Node.js helper for
            reproducible option handling when supported by the engine.

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
        """Run Cytoscape fcose with optional parameter overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            Optional benchmark seed forwarded to the helper payload.
        variant_params : Mapping[str, Any] | None, default=None
            Optional fcose layout parameters merged into the Node.js layout
            options object.

        Returns
        -------
        CompetitorResult
            Layout result and timing information.
        """
        payload: Dict[str, object] = {
            "num_nodes": graph.num_nodes,
            "edges": _graph_to_edge_list(graph),
            "seed": seed,
            "options": dict(variant_params) if variant_params is not None else {},
        }
        input_data = json.dumps(payload)

        start = time.perf_counter()
        try:
            result = subprocess.run(
                ["node", "-e", _FCOSE_SCRIPT],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - start

            if result.returncode != 0:
                error_text = result.stderr or result.stdout
                return CompetitorResult(
                    name=self.name,
                    pos=None,
                    runtime_seconds=elapsed,
                    error=error_text[:500],
                )

            data = json.loads(result.stdout)
            if not isinstance(data, dict):
                raise ValueError("fcose helper returned non-object JSON")
            pos = _positions_to_tensor(data, graph.num_nodes)
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name, pos=None, runtime_seconds=elapsed, error="timeout"
            )
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(error),
            )

    def available(self) -> bool:
        """Return whether the Node.js fcose dependency is usable.

        Returns
        -------
        bool
            ``True`` when both Cytoscape and the fcose plugin can be loaded by
            Node.js.
        """
        try:
            result = subprocess.run(
                ["node", "-e", "require('cytoscape'); require('cytoscape-fcose')"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

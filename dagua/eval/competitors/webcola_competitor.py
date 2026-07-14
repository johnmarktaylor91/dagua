"""WebCola competitor adapter via Node.js subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register
from dagua.layout.ops.webcola import webcola_initial_positions

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_DEFAULT_STEPS = 50
_DEFAULT_LINK_DISTANCE = 20.0
_DURABLE_NODE_MODULES = Path.home() / "tools" / "dagua-refs" / "node_modules"


def _node_subprocess_env() -> Dict[str, str]:
    """Return a Node environment that can resolve durable WebCola packages.

    Returns
    -------
    dict[str, str]
        Environment variables for WebCola subprocesses.
    """
    env = dict(os.environ)
    paths = [str(_DURABLE_NODE_MODULES)] if _DURABLE_NODE_MODULES.exists() else []
    existing = env.get("NODE_PATH", "")
    if existing:
        paths.append(existing)
    if paths:
        env["NODE_PATH"] = os.pathsep.join(paths)
    return env


_WEBCOLA_SCRIPT = r"""
const cola = require('webcola');
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
  const data = JSON.parse(input);
  const nodes = data.nodes.map((node) => ({ id: node.id, x: node.x, y: node.y }));
  const links = data.links.map((link) => ({
    source: link.source,
    target: link.target,
    weight: link.weight
  }));
  const layout = new cola.Layout()
    .size([1, 1])
    .nodes(nodes)
    .links(links)
    .linkDistance(data.linkDistance)
    .avoidOverlaps(false)
    .handleDisconnected(false);
  if (data.constraints.length > 0) layout.constraints(data.constraints);
  layout.start(
    data.unconstrainedIterations,
    data.userConstraintIterations,
    data.allConstraintIterations,
    0,
    false,
    false
  );
  const result = {};
  for (const node of nodes) result[node.id] = { x: node.x, y: node.y };
  process.stdout.write(JSON.stringify(result));
});
"""


def _build_webcola_input(
    graph: DaguaGraph,
    steps: int,
    link_distance: float,
    constrained: bool,
    constraints: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the JSON payload for the WebCola Node reference.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    steps : int
        Iteration count.
    link_distance : float
        Constant WebCola link distance.
    constrained : bool
        Whether to run the user-constraint phase instead of the unconstrained
        phase.
    constraints : list[dict[str, Any]], optional
        Explicit WebCola constraints.

    Returns
    -------
    dict[str, Any]
        JSON-serializable payload for the Node script.
    """
    starts = webcola_initial_positions(graph.num_nodes, link_distance).tolist()
    nodes = [
        {"id": str(index), "x": float(starts[index][0]), "y": float(starts[index][1])}
        for index in range(graph.num_nodes)
    ]
    links: List[Dict[str, Any]] = []
    if graph.edge_index.numel() > 0:
        edge_weights = (
            graph.edge_weights.detach().to(device="cpu", dtype=torch.float64).tolist()
            if graph.edge_weights is not None
            else None
        )
        for edge_pos in range(graph.edge_index.shape[1]):
            source = int(graph.edge_index[0, edge_pos].item())
            target = int(graph.edge_index[1, edge_pos].item())
            link: Dict[str, Any] = {"source": source, "target": target}
            if edge_weights is not None:
                link["weight"] = float(edge_weights[edge_pos])
            links.append(link)
    return {
        "nodes": nodes,
        "links": links,
        "linkDistance": float(link_distance),
        "constraints": list(constraints or []),
        "unconstrainedIterations": 0 if constrained else int(steps),
        "userConstraintIterations": int(steps) if constrained else 0,
        "allConstraintIterations": 0,
    }


@register
class WebColaCompetitor(CompetitorBase):
    """Run WebCola through a reproducible Node subprocess."""

    name = "webcola"
    max_nodes = 2_000
    supports_clusters = False
    variant_param_names = frozenset({"steps", "link_distance", "constrained", "constraints"})

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the default unconstrained WebCola reference.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime in seconds.
        seed : int, optional
            Accepted for adapter API compatibility; ignored because starts are
            explicitly pinned.

        Returns
        -------
        CompetitorResult
            Position tensor and runtime, or an error.
        """
        del seed
        return self.layout_with_variant(graph, timeout=timeout, seed=None, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[dict[str, Any]] = None,
    ) -> CompetitorResult:
        """Run WebCola with optional variant parameters.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime in seconds.
        seed : int, optional
            Accepted for adapter API compatibility; ignored.
        variant_params : dict[str, Any], optional
            ``steps``, ``link_distance``, ``constrained``, and ``constraints``.

        Returns
        -------
        CompetitorResult
            Position tensor and runtime, or an error.
        """
        del seed
        params = {} if variant_params is None else dict(variant_params)
        input_data = json.dumps(
            _build_webcola_input(
                graph,
                steps=int(params.get("steps", _DEFAULT_STEPS)),
                link_distance=float(params.get("link_distance", _DEFAULT_LINK_DISTANCE)),
                constrained=bool(params.get("constrained", False)),
                constraints=params.get("constraints"),
            )
        )
        start = time.perf_counter()
        try:
            result = subprocess.run(
                ["node", "-e", _WEBCOLA_SCRIPT],
                input=input_data,
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
                    error=result.stderr[:500],
                )
            data = json.loads(result.stdout)
            pos = torch.zeros((graph.num_nodes, 2), dtype=torch.float64)
            for index in range(graph.num_nodes):
                node_data = data.get(str(index), {})
                pos[index, 0] = float(node_data.get("x", 0.0))
                pos[index, 1] = float(node_data.get("y", 0.0))
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name, pos=None, runtime_seconds=elapsed, error="timeout"
            )
        except (FileNotFoundError, json.JSONDecodeError) as error:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name, pos=None, runtime_seconds=elapsed, error=str(error)
            )

    def available(self) -> bool:
        """Return whether the WebCola Node package is importable.

        Returns
        -------
        bool
            ``True`` when ``require("webcola")`` succeeds.
        """
        try:
            result = subprocess.run(
                ["node", "-e", "require('webcola')"],
                capture_output=True,
                env=_node_subprocess_env(),
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

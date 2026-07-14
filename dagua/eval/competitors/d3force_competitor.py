"""d3-force competitor adapter via Node.js subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, get_runtime_seed, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_DEFAULT_TICKS = 300
_DEFAULT_SEED = 1
_D3FORCE_NODE_MODULES = Path("/home/jtaylor/projects/dagua/node_modules")


def _node_subprocess_env() -> Dict[str, str]:
    """Return a Node environment that can resolve repo-local d3-force.

    Returns
    -------
    dict[str, str]
        Environment variables for d3-force subprocesses.
    """
    env = dict(os.environ)
    if _D3FORCE_NODE_MODULES.exists():
        existing = env.get("NODE_PATH", "")
        paths = [str(_D3FORCE_NODE_MODULES)]
        if existing:
            paths.append(existing)
        env["NODE_PATH"] = os.pathsep.join(paths)
    return env


_D3FORCE_SCRIPT = r"""
const d3 = require('d3-force');
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
  const data = JSON.parse(input);
  const nodes = data.nodes.map((node) => ({ id: node.id }));
  const links = data.links.map((link) => ({ source: link.source, target: link.target }));
  function lcg(seed) {
    let s = seed >>> 0;
    return function() {
      s = (Math.imul(1664525, s) + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }
  const sim = d3.forceSimulation(nodes)
    .randomSource(lcg(data.seed))
    .force('link', d3.forceLink(links).id((d) => d.id))
    .force('charge', d3.forceManyBody().strength(data.manyBodyStrength).theta(data.theta))
    .force('center', data.center ? d3.forceCenter(0, 0) : null)
    .velocityDecay(data.velocityDecay)
    .stop();
  sim.force('link').distance(data.linkDistance).iterations(data.linkIterations);
  sim.tick(data.ticks);
  const result = {};
  for (const node of nodes) result[node.id] = { x: node.x, y: node.y };
  process.stdout.write(JSON.stringify(result));
});
"""


def _build_d3force_input(
    graph: DaguaGraph,
    seed: int,
    ticks: int,
    many_body_strength: float,
    link_distance: float,
    link_iterations: int,
    velocity_decay: float,
    theta: float,
    center: bool,
) -> Dict[str, Any]:
    """Build the JSON payload for the Node d3-force reference.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    seed : int
        LCG seed passed through ``simulation.randomSource``.
    ticks : int
        Number of d3 simulation ticks.
    many_body_strength : float
        Constant ``forceManyBody`` strength.
    link_distance : float
        Constant ``forceLink`` distance.
    link_iterations : int
        Link force iterations per tick.
    velocity_decay : float
        Public d3 velocity-decay parameter.
    theta : float
        Barnes-Hut theta.
    center : bool
        Whether to enable ``forceCenter(0, 0)``.

    Returns
    -------
    dict[str, Any]
        JSON-serializable d3-force input.
    """
    nodes = [{"id": str(index)} for index in range(graph.num_nodes)]
    links: List[Dict[str, str]] = []
    if graph.edge_index.numel() > 0:
        for edge_index in range(graph.edge_index.shape[1]):
            source = int(graph.edge_index[0, edge_index].item())
            target = int(graph.edge_index[1, edge_index].item())
            links.append({"source": str(source), "target": str(target)})
    return {
        "nodes": nodes,
        "links": links,
        "seed": int(seed),
        "ticks": int(ticks),
        "manyBodyStrength": float(many_body_strength),
        "linkDistance": float(link_distance),
        "linkIterations": int(link_iterations),
        "velocityDecay": float(velocity_decay),
        "theta": float(theta),
        "center": bool(center),
    }


@register
class D3ForceCompetitor(CompetitorBase):
    """Run d3-force through a reproducible Node subprocess."""

    name = "d3force"
    max_nodes = 5_000
    supports_clusters = False
    variant_param_names = frozenset(
        {
            "ticks",
            "many_body_strength",
            "link_distance",
            "link_iterations",
            "velocity_decay",
            "theta",
            "center",
        }
    )

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the canonical d3-force triad.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime in seconds.
        seed : int | None, default=None
            LCG seed. ``None`` resolves to the benchmark runtime seed or ``1``.

        Returns
        -------
        CompetitorResult
            Position tensor and runtime, or an error.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[dict[str, Any]] = None,
    ) -> CompetitorResult:
        """Run d3-force with optional parameter overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime in seconds.
        seed : int | None, default=None
            LCG seed. ``None`` resolves to the benchmark runtime seed or ``1``.
        variant_params : dict[str, Any], optional
            d3-force option overrides.

        Returns
        -------
        CompetitorResult
            Position tensor and runtime, or an error.
        """
        params = {} if variant_params is None else dict(variant_params)
        resolved_seed = get_runtime_seed(_DEFAULT_SEED) if seed is None else seed
        input_data = json.dumps(
            _build_d3force_input(
                graph,
                seed=_DEFAULT_SEED if resolved_seed is None else int(resolved_seed),
                ticks=int(params.get("ticks", _DEFAULT_TICKS)),
                many_body_strength=float(params.get("many_body_strength", -30.0)),
                link_distance=float(params.get("link_distance", 30.0)),
                link_iterations=int(params.get("link_iterations", 1)),
                velocity_decay=float(params.get("velocity_decay", 0.4)),
                theta=float(params.get("theta", 0.9)),
                center=bool(params.get("center", True)),
            )
        )

        start = time.perf_counter()
        try:
            result = subprocess.run(
                ["node", "-e", _D3FORCE_SCRIPT],
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
            pos = torch.zeros((graph.num_nodes, 2), dtype=torch.float64)
            for index in range(graph.num_nodes):
                node_data = data.get(str(index), {})
                pos[index, 0] = float(node_data.get("x", 0.0))
                pos[index, 1] = float(node_data.get("y", 0.0))
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error="timeout",
            )
        except (FileNotFoundError, json.JSONDecodeError) as error:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(error),
            )

    def available(self) -> bool:
        """Return whether the d3-force Node package is importable.

        Returns
        -------
        bool
            ``True`` when ``require("d3-force")`` succeeds.
        """
        try:
            result = subprocess.run(
                ["node", "-e", "require('d3-force')"],
                capture_output=True,
                timeout=10,
                env=_node_subprocess_env(),
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

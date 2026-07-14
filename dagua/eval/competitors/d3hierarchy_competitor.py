"""d3-hierarchy tree/cluster competitor adapter via Node.js subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_DURABLE_NODE_MODULES = Path.home() / "tools" / "dagua-refs" / "node_modules"

_D3_HIERARCHY_SCRIPT = r"""
const path = require('path');
const mainCheckout = path.resolve(process.cwd(), '..', '..', '..', 'projects', 'dagua');
const durableRefs = path.join(process.env.HOME || '', 'tools', 'dagua-refs');
const modulePaths = [durableRefs, process.cwd(), mainCheckout];
let d3 = null;
let loadError = null;
for (const base of modulePaths) {
  try {
    d3 = require(require.resolve('d3-hierarchy', { paths: [base] }));
    break;
  } catch (err) {
    loadError = err;
  }
}

function buildHierarchy(data) {
  const childrenByParent = new Map();
  const parentByChild = new Map();
  for (let i = 0; i < data.num_nodes; ++i) childrenByParent.set(i, []);
  for (const edge of data.edges) {
    const source = Number(edge[0]);
    const target = Number(edge[1]);
    if (source === target || source < 0 || target < 0) continue;
    if (source >= data.num_nodes || target >= data.num_nodes) continue;
    if (parentByChild.has(target)) continue;
    parentByChild.set(target, source);
    childrenByParent.get(source).push(target);
  }
  let roots = [];
  for (let i = 0; i < data.num_nodes; ++i) {
    if (!parentByChild.has(i)) roots.push(i);
  }
  if (!roots.length) roots = [0];
  const rootIndex = roots[0];
  for (const extraRoot of roots.slice(1)) {
    if (extraRoot === rootIndex) continue;
    parentByChild.set(extraRoot, rootIndex);
    childrenByParent.get(rootIndex).push(extraRoot);
  }
  const visiting = new Set();
  const seen = new Set();
  function build(index) {
    seen.add(index);
    visiting.add(index);
    const node = { id: String(index), children: [] };
    for (const child of childrenByParent.get(index)) {
      if (visiting.has(child)) continue;
      node.children.push(build(child));
    }
    visiting.delete(index);
    if (!node.children.length) delete node.children;
    return node;
  }
  const root = build(rootIndex);
  if (!root.children) root.children = [];
  for (let i = 0; i < data.num_nodes; ++i) {
    if (!seen.has(i)) root.children.push(build(i));
  }
  if (!root.children.length) delete root.children;
  return root;
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
  try {
    if (!d3) throw loadError || new Error('d3-hierarchy not found');
    const data = JSON.parse(input);
    const root = d3.hierarchy(buildHierarchy(data));
    let layout = data.algorithm === 'cluster' ? d3.cluster() : d3.tree();
    if (data.node_size) {
      layout = layout.nodeSize([data.dx, data.dy]);
    } else {
      layout = layout.size([data.dx, data.dy]);
    }
    layout(root);
    const positions = {};
    root.each((node) => {
      const id = String(node.data.id);
      let x = node.x;
      let y = node.y;
      if (data.radial) {
        const angle = x - Math.PI / 2;
        x = y * Math.cos(angle);
        y = y * Math.sin(angle);
      }
      positions[id] = { x, y };
    });
    process.stdout.write(JSON.stringify({ ok: true, positions }));
  } catch (err) {
    process.stdout.write(JSON.stringify({ ok: false, error: String(err && err.stack || err) }));
  }
});
"""


def _node_subprocess_env() -> Dict[str, str]:
    """Return a Node environment that can resolve durable d3-hierarchy packages.

    Returns
    -------
    dict[str, str]
        Environment variables for d3-hierarchy subprocesses.
    """
    env = dict(os.environ)
    paths = [str(_DURABLE_NODE_MODULES)] if _DURABLE_NODE_MODULES.exists() else []
    existing = env.get("NODE_PATH", "")
    if existing:
        paths.append(existing)
    if paths:
        env["NODE_PATH"] = os.pathsep.join(paths)
    return env


def _build_input(
    graph: DaguaGraph,
    algorithm: str,
    dx: float,
    dy: float,
    node_size: bool,
    radial: bool,
) -> Dict[str, object]:
    """Build a JSON-serializable d3-hierarchy payload.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    algorithm : str
        ``"tree"`` or ``"cluster"``.
    dx : float
        d3 horizontal size or node-size scale.
    dy : float
        d3 vertical size or node-size scale.
    node_size : bool
        Whether to use d3 ``nodeSize`` semantics.
    radial : bool
        Whether the adapter should apply d3's radial transform.

    Returns
    -------
    dict[str, object]
        Payload consumed by the Node subprocess.
    """
    edges: List[List[int]] = []
    if graph.edge_index.numel() > 0:
        for edge_id in range(int(graph.edge_index.shape[1])):
            edges.append(
                [
                    int(graph.edge_index[0, edge_id].item()),
                    int(graph.edge_index[1, edge_id].item()),
                ]
            )
    return {
        "num_nodes": graph.num_nodes,
        "edges": edges,
        "algorithm": algorithm,
        "dx": dx,
        "dy": dy,
        "node_size": node_size,
        "radial": radial,
    }


@register
class D3HierarchyCompetitor(CompetitorBase):
    """Run d3-hierarchy tree or cluster through a local Node subprocess."""

    name = "d3hierarchy"
    max_nodes = 10_000
    supports_clusters = False
    variant_param_names = frozenset({"algorithm", "dx", "dy", "node_size", "radial"})

    def available(self) -> bool:
        """Check whether Node can load the local ``d3-hierarchy`` package.

        Returns
        -------
        bool
            ``True`` when the package resolves from this checkout or the main
            project checkout.
        """
        candidates = [
            Path.home() / "tools" / "dagua-refs",
            Path.cwd(),
            Path.cwd().parents[2] / "projects" / "dagua" if len(Path.cwd().parents) > 2 else None,
        ]
        script = (
            "const r=require('module').createRequire(process.cwd() + '/');"
            "require.resolve('d3-hierarchy', {paths: process.argv.slice(1)});"
        )
        paths = [str(path) for path in candidates if path is not None]
        try:
            subprocess.run(
                ["node", "-e", script, *paths],
                check=True,
                env=_node_subprocess_env(),
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
        """Run d3-hierarchy's default tidy tree layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime.
        seed : int | None, default=None
            Accepted for competitor API compatibility; d3-hierarchy is
            deterministic.

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
        variant_params: Optional[Mapping[str, object]] = None,
    ) -> CompetitorResult:
        """Run d3-hierarchy with optional tree/cluster parameters.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime.
        seed : int | None, default=None
            Accepted for API compatibility; d3-hierarchy is deterministic.
        variant_params : mapping[str, object] | None, optional
            Optional ``algorithm``, ``dx``, ``dy``, ``node_size``, and
            ``radial`` overrides.

        Returns
        -------
        CompetitorResult
            Layout result or error.
        """
        del seed
        params = variant_params if isinstance(variant_params, Mapping) else {}
        algorithm = str(params.get("algorithm", "tree"))
        default_node_size = algorithm != "cluster"
        payload = _build_input(
            graph,
            algorithm=algorithm,
            dx=float(params.get("dx", 1.0)),
            dy=float(params.get("dy", 1.0)),
            node_size=bool(params.get("node_size", default_node_size)),
            radial=bool(params.get("radial", False)),
        )
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                ["node", "-e", _D3_HIERARCHY_SCRIPT],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                env=_node_subprocess_env(),
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CompetitorResult(
                self.name,
                None,
                time.perf_counter() - start,
                error=f"d3-hierarchy timed out after {timeout:.1f}s",
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
        positions = torch.zeros((graph.num_nodes, 2), dtype=torch.float64)
        raw_positions = data["positions"]
        for node in range(graph.num_nodes):
            raw = raw_positions.get(str(node))
            if raw is None:
                continue
            positions[node, 0] = float(raw["x"])
            positions[node, 1] = float(raw["y"])
        return CompetitorResult(self.name, positions, runtime)


__all__ = ["D3HierarchyCompetitor"]

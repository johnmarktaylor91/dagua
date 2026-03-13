"""Dagre competitor adapter — dagre via Node.js subprocess."""

from __future__ import annotations

import json
import subprocess
import time
from typing import TYPE_CHECKING

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


_DAGRE_SCRIPT = r"""
const dagre = require('dagre');
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
    const data = JSON.parse(input);
    const g = new dagre.graphlib.Graph();
    g.setGraph({ rankdir: 'TB', nodesep: 40, ranksep: 60 });
    g.setDefaultEdgeLabel(function() { return {}; });
    for (const node of data.nodes) {
        g.setNode(node.id, { width: 120, height: 40 });
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


@register
class DagreCompetitor(CompetitorBase):
    name = "dagre"
    max_nodes = 2_000

    def layout(self, graph: DaguaGraph, timeout: float = 300.0) -> CompetitorResult:
        n = graph.num_nodes
        nodes = [{"id": str(i)} for i in range(n)]
        edges = []
        if graph.edge_index.numel() > 0:
            for e_idx in range(graph.edge_index.shape[1]):
                s = graph.edge_index[0, e_idx].item()
                t = graph.edge_index[1, e_idx].item()
                edges.append({"source": str(s), "target": str(t)})

        input_data = json.dumps({"nodes": nodes, "edges": edges})

        start = time.perf_counter()
        try:
            result = subprocess.run(
                ["node", "-e", _DAGRE_SCRIPT],
                input=input_data,
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

            data = json.loads(result.stdout)
            pos = torch.zeros(n, 2)
            for i in range(n):
                node_data = data.get(str(i), {})
                pos[i, 0] = node_data.get("x", 0)
                pos[i, 1] = node_data.get("y", 0)

            return CompetitorResult(
                name=self.name, pos=pos, runtime_seconds=elapsed
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name, pos=None, runtime_seconds=elapsed, error="timeout"
            )
        except (FileNotFoundError, json.JSONDecodeError) as e:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name, pos=None, runtime_seconds=elapsed, error=str(e)
            )

    def available(self) -> bool:
        try:
            result = subprocess.run(
                ["node", "-e", "require('dagre')"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

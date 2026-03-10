"""ELK competitor adapter — elkjs via Node.js subprocess."""

from __future__ import annotations

import json
import subprocess
import time
from typing import TYPE_CHECKING

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


@register
class ElkLayered(CompetitorBase):
    name = "elk_layered"
    max_nodes = 50_000

    def layout(self, graph: DaguaGraph, timeout: float = 300.0) -> CompetitorResult:
        n = graph.num_nodes
        children = [{"id": str(i), "width": 120, "height": 40} for i in range(n)]
        edges = []
        if graph.edge_index.numel() > 0:
            for e_idx in range(graph.edge_index.shape[1]):
                s = graph.edge_index[0, e_idx].item()
                t = graph.edge_index[1, e_idx].item()
                edges.append(
                    {"id": f"e{e_idx}", "sources": [str(s)], "targets": [str(t)]}
                )

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
            for child in data.get("children", []):
                idx = int(child["id"])
                if idx < n:
                    pos[idx, 0] = child.get("x", 0)
                    pos[idx, 1] = child.get("y", 0)

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
                ["node", "-e", "require('elkjs')"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

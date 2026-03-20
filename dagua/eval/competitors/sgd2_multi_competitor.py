"""(SGD)^2 multicriteria reference competitor adapter.

Runs the original (SGD)^2 code from github.com/tiga1231/graph-drawing
(cloned to /tmp/graph-drawing) as a reference implementation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_SGD2_REPO = Path("/tmp/graph-drawing")


def _sgd2_multi_available() -> bool:
    """Check if the upstream (SGD)^2 code is available."""
    return (_SGD2_REPO / "gd2.py").exists()


@register
class SGD2MultiRef(CompetitorBase):
    """Reference adapter for the (SGD)^2 multicriteria layout engine."""

    name = "sgd2_multi_ref"
    max_nodes = 5_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the upstream (SGD)^2 multicriteria engine.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float
            Unused.
        seed : int | None
            Random seed.

        Returns
        -------
        CompetitorResult
            Layout result.
        """
        del timeout

        start = time.perf_counter()
        try:
            import numpy as np
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import shortest_path

            # Add repo to path temporarily
            repo_str = str(_SGD2_REPO)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

            from gd2 import GD2  # type: ignore[import-untyped]

            n = graph.num_nodes
            ei = graph.edge_index.cpu().numpy()

            # Build symmetric adjacency
            rows = np.concatenate([ei[0], ei[1]])
            cols = np.concatenate([ei[1], ei[0]])
            data = np.ones(len(rows))
            adj = csr_matrix((data, (rows, cols)), shape=(n, n))

            # Shortest paths
            dist = shortest_path(adj, directed=False)
            if not np.all(np.isfinite(dist)):
                return CompetitorResult(
                    name=self.name,
                    pos=None,
                    runtime_seconds=time.perf_counter() - start,
                    error="graph is disconnected",
                )

            # Build networkx graph for GD2
            import networkx as nx

            G_nx = nx.Graph()
            G_nx.add_nodes_from(range(n))
            for i in range(ei.shape[1]):
                s, t = int(ei[0, i]), int(ei[1, i])
                if s < t:
                    G_nx.add_edge(s, t)

            # Run GD2 with stress criterion
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)

            gd2 = GD2(G_nx, criteria=["stress"], maxiter=2000)
            gd2.optimize()
            positions = gd2.X.detach().cpu()

            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=positions, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )

    def available(self) -> bool:
        """Check if the upstream code is cloned."""
        return _sgd2_multi_available()

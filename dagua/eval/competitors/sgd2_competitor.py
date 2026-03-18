"""s_gd2 competitor adapter wrapping the reference stress-SGD implementation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    import numpy as np

    from dagua.graph import DaguaGraph


def _sgd2_available() -> bool:
    """Return whether the ``s_gd2`` package can be imported.

    Returns
    -------
    bool
        ``True`` when the third-party package is importable in the current
        environment.
    """
    try:
        import s_gd2  # noqa: F401
    except Exception:
        return False
    return True


def _symmetrized_unique_edges(graph: DaguaGraph) -> tuple["np.ndarray", "np.ndarray"]:
    """Build a deduplicated undirected edge list for ``s_gd2``.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose ``edge_index`` is converted.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Source and target arrays with self-loops removed. The result is empty
        for graphs without edges.
    """
    import numpy as np

    edge_index = graph.edge_index.cpu().numpy()
    if edge_index.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    sources = np.concatenate([edge_index[0], edge_index[1]])
    targets = np.concatenate([edge_index[1], edge_index[0]])
    non_self_mask = sources != targets
    filtered = np.stack([sources[non_self_mask], targets[non_self_mask]], axis=1)
    if filtered.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    unique_edges = np.unique(filtered, axis=0)
    return unique_edges[:, 0], unique_edges[:, 1]


@register
class SGD2(CompetitorBase):
    """Competitor adapter for the reference ``s_gd2`` stress-SGD engine."""

    name = "sgd2"
    max_nodes = 50_000

    def layout(self, graph: DaguaGraph, timeout: float = 300.0) -> CompetitorResult:
        """Run ``s_gd2`` and convert its output to a CPU tensor.

        Parameters
        ----------
        graph : DaguaGraph
            Input graph to lay out.
        timeout : float, optional
            Unused adapter timeout in seconds. Included for interface
            compatibility with the benchmark harness.

        Returns
        -------
        CompetitorResult
            Layout result with positions shaped ``[N, 2]`` on CPU, or an error
            payload if the third-party engine fails.
        """
        start = time.perf_counter()
        try:
            import s_gd2

            if graph.num_nodes == 0:
                pos = torch.zeros((0, 2), dtype=torch.float32)
                elapsed = time.perf_counter() - start
                return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)

            if graph.num_nodes == 1:
                pos = torch.zeros((1, 2), dtype=torch.float32)
                elapsed = time.perf_counter() - start
                return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)

            sources, targets = _symmetrized_unique_edges(graph)
            if sources.size == 0:
                pos = torch.zeros((graph.num_nodes, 2), dtype=torch.float32)
                elapsed = time.perf_counter() - start
                return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)

            coordinates = s_gd2.layout(sources.tolist(), targets.tolist())
            pos = torch.tensor(coordinates, dtype=torch.float32) * 100.0

            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )

    def available(self) -> bool:
        """Report whether ``s_gd2`` is usable in the current environment.

        Returns
        -------
        bool
            ``True`` when the dependency imports successfully.
        """
        return _sgd2_available()

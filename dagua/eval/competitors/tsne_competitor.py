"""t-SNE graph layout competitor using sklearn on shortest-path distances."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Mapping, Optional

import torch

from dagua.eval.competitors.base import (
    CompetitorBase,
    CompetitorResult,
    register,
)

if TYPE_CHECKING:
    import numpy as np

    from dagua.graph import DaguaGraph


def _distance_matrix(graph: DaguaGraph) -> "np.ndarray":
    """Compute undirected shortest-path distances for a graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose shortest-path metric is needed.

    Returns
    -------
    numpy.ndarray
        Dense ``[N, N]`` distance matrix with disconnected entries replaced by
        a large finite value.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    num_nodes = graph.num_nodes
    if num_nodes == 0:
        return np.zeros((0, 0), dtype=np.float32)

    edge_index = graph.edge_index.cpu().numpy()
    rows = np.concatenate([edge_index[0], edge_index[1]]) if edge_index.size else np.empty(0, int)
    cols = np.concatenate([edge_index[1], edge_index[0]]) if edge_index.size else np.empty(0, int)
    if graph.edge_weights is not None:
        edge_weights = graph.edge_weights.cpu().numpy().astype(np.float32)
        data = np.concatenate([edge_weights, edge_weights])
    else:
        data = np.ones(rows.shape[0], dtype=np.float32)
    adjacency = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    distances = shortest_path(adjacency, directed=False)

    finite_mask = np.isfinite(distances)
    max_finite = float(np.max(distances[finite_mask])) if np.any(finite_mask) else 1.0
    fill_value = max(max_finite * 2.0, 1.0)
    dense = np.where(np.isinf(distances), fill_value, distances)
    return dense.astype(np.float32, copy=False)


@register
class TSNEGraph(CompetitorBase):
    """Competitor adapter for graph-distance t-SNE."""

    name = "tsne_graph"
    max_nodes = 5_000
    variant_param_names = frozenset({"learning_rate", "max_iter", "perplexity"})

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run sklearn t-SNE on all-pairs graph shortest-path distances.

        Parameters
        ----------
        graph : DaguaGraph
            Input graph to lay out.
        timeout : float, optional
            Unused adapter timeout in seconds. Included for interface
            compatibility with the benchmark harness.
        seed : int | None, default=None
            Random seed for t-SNE initialization. ``None`` preserves the
            adapter's historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result with positions shaped ``[N, 2]`` on CPU, or an error
            payload if sklearn or SciPy fails.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, object]] = None,
    ) -> CompetitorResult:
        """Run sklearn t-SNE on all-pairs graph shortest-path distances.

        Parameters
        ----------
        graph : DaguaGraph
            Input graph to lay out.
        timeout : float, optional
            Unused adapter timeout in seconds. Included for interface
            compatibility with the benchmark harness.
        seed : int | None, default=None
            Random seed for t-SNE initialization. ``None`` preserves the
            adapter's historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result with positions shaped ``[N, 2]`` on CPU, or an error
            payload if sklearn or SciPy fails.
        """
        start = time.perf_counter()
        try:
            from sklearn.manifold import TSNE

            num_nodes = graph.num_nodes
            if num_nodes == 0:
                pos = torch.zeros((0, 2), dtype=torch.float32)
                elapsed = time.perf_counter() - start
                return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)

            if num_nodes == 1:
                pos = torch.zeros((1, 2), dtype=torch.float32)
                elapsed = time.perf_counter() - start
                return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)

            distances = _distance_matrix(graph)
            tsne_kwargs: dict[str, object] = {
                "n_components": 2,
                "metric": "precomputed",
                "init": "random",
                "random_state": seed if seed is not None else 42,
                "perplexity": min(30.0, float(num_nodes - 1)),
            }
            if variant_params is not None:
                tsne_kwargs.update(dict(variant_params))
                perplexity = float(tsne_kwargs.get("perplexity", 30.0))
                tsne_kwargs["perplexity"] = min(perplexity, float(num_nodes - 1))
            tsne = TSNE(**tsne_kwargs)
            coordinates = tsne.fit_transform(distances)
            pos = torch.tensor(coordinates, dtype=torch.float32)

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
        """Report whether sklearn t-SNE is available.

        Returns
        -------
        bool
            ``True`` when sklearn imports successfully.
        """
        try:
            from sklearn.manifold import TSNE  # noqa: F401
        except Exception:
            return False
        return True

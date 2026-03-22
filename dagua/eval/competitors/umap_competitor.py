"""UMAP graph layout competitor using shortest-path distances."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch

from dagua.eval.competitors.base import (
    CompetitorBase,
    CompetitorResult,
    register,
)

if TYPE_CHECKING:
    import numpy as np

    from dagua.graph import DaguaGraph


_NUMBA_CACHE_DIR = "/tmp/dagua-numba-cache"


def _configure_numba_cache() -> None:
    """Point numba's cache to a writable directory before importing UMAP.

    Returns
    -------
    None
        This function mutates ``os.environ`` in-place when necessary.
    """
    os.environ.setdefault("NUMBA_CACHE_DIR", _NUMBA_CACHE_DIR)


def _umap_available() -> bool:
    """Return whether the ``umap`` package can be imported and initialized.

    Returns
    -------
    bool
        ``True`` when the third-party package is importable in the current
        environment.
    """
    try:
        _configure_numba_cache()
        import umap  # noqa: F401
    except Exception:
        return False
    return True


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
class UMAPGraph(CompetitorBase):
    """Competitor adapter for graph-distance UMAP."""

    name = "umap_graph"
    max_nodes = 20_000
    variant_param_names = frozenset(
        {
            "learning_rate",
            "min_dist",
            "n_epochs",
            "n_neighbors",
            "negative_sample_rate",
            "repulsion_strength",
            "spread",
        }
    )

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run UMAP on all-pairs graph shortest-path distances.

        Parameters
        ----------
        graph : DaguaGraph
            Input graph to lay out.
        timeout : float, optional
            Unused adapter timeout in seconds. Included for interface
            compatibility with the benchmark harness.
        seed : int | None, default=None
            Random seed for UMAP initialization. ``None`` preserves the
            adapter's historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result with positions shaped ``[N, 2]`` on CPU, or an error
            payload if UMAP or SciPy fails.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run UMAP on all-pairs graph shortest-path distances.

        Parameters
        ----------
        graph : DaguaGraph
            Input graph to lay out.
        timeout : float, optional
            Unused adapter timeout in seconds. Included for interface
            compatibility with the benchmark harness.
        seed : int | None, default=None
            Random seed for UMAP initialization. ``None`` preserves the
            adapter's historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result with positions shaped ``[N, 2]`` on CPU, or an error
            payload if UMAP or SciPy fails.
        """
        start = time.perf_counter()
        try:
            _configure_numba_cache()
            import umap

            num_nodes = graph.num_nodes
            if num_nodes == 0:
                pos = torch.zeros((0, 2), dtype=torch.float32)
                elapsed = time.perf_counter() - start
                return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)

            if num_nodes <= 3:
                # UMAP's spectral init uses sparse eigensolver which fails
                # when k >= N. Fall back to random placement for tiny graphs.
                generator = torch.Generator(device="cpu")
                generator.manual_seed(42 if seed is None else seed)
                pos = torch.randn((num_nodes, 2), generator=generator, dtype=torch.float32)
                elapsed = time.perf_counter() - start
                return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)

            distances = _distance_matrix(graph)
            reducer_kwargs: dict[str, Any] = {
                "n_components": 2,
                "metric": "precomputed",
                "random_state": seed if seed is not None else 42,
                "n_neighbors": min(15, num_nodes - 1),
                "init": "random" if num_nodes < 10 else "spectral",
            }
            if variant_params is not None:
                reducer_kwargs.update(dict(variant_params))
            reducer_kwargs["n_neighbors"] = min(int(reducer_kwargs["n_neighbors"]), num_nodes - 1)
            reducer = umap.UMAP(**reducer_kwargs)
            coordinates = reducer.fit_transform(distances)
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
        """Report whether UMAP is usable in the current environment.

        Returns
        -------
        bool
            ``True`` when the dependency imports successfully after configuring
            numba's cache directory.
        """
        return _umap_available()

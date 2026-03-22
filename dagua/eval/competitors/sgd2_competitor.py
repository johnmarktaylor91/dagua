"""s_gd2 competitor adapter wrapping the reference stress-SGD implementation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Mapping, Optional

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


def _symmetrized_unique_edges(
    graph: DaguaGraph,
) -> tuple["np.ndarray", "np.ndarray", Optional["np.ndarray"]]:
    """Build a symmetrized edge list and optional weights for ``s_gd2``.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose ``edge_index`` is converted.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray | None]
        Source and target arrays with self-loops removed, plus optional
        per-edge weights aligned with the returned edges. The result is empty
        for graphs without edges.
    """
    import numpy as np

    edge_index = graph.edge_index.cpu().numpy()
    if edge_index.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), None

    sources = np.concatenate([edge_index[0], edge_index[1]])
    targets = np.concatenate([edge_index[1], edge_index[0]])
    non_self_mask = sources != targets
    src_filtered = sources[non_self_mask]
    tgt_filtered = targets[non_self_mask]
    if src_filtered.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), None

    weights_sym: Optional["np.ndarray"]
    if graph.edge_weights is not None:
        edge_weights = graph.edge_weights.cpu().numpy().astype(np.float64)
        weights_sym = np.concatenate([edge_weights, edge_weights])[non_self_mask]
    else:
        weights_sym = None

    filtered = np.stack([src_filtered, tgt_filtered], axis=1)
    unique_edges, inverse = np.unique(filtered, axis=0, return_inverse=True)

    if weights_sym is not None:
        unique_weights = np.zeros(unique_edges.shape[0], dtype=np.float64)
        np.add.at(unique_weights, inverse, weights_sym)
        return unique_edges[:, 0], unique_edges[:, 1], unique_weights

    return unique_edges[:, 0], unique_edges[:, 1], None


def _build_condensed_distances(graph: DaguaGraph) -> tuple["np.ndarray", "np.ndarray"]:
    """Build condensed shortest-path distances and MDS weights.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose undirected shortest-path distances should be computed.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Condensed upper-triangle vectors for graph distances and
        inverse-square weights.

    Raises
    ------
    ValueError
        If the graph is disconnected and therefore has infinite distances.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    n_nodes = graph.num_nodes
    if n_nodes <= 1:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    edge_index = graph.edge_index.cpu().numpy()
    rows = np.concatenate([edge_index[0], edge_index[1]])
    cols = np.concatenate([edge_index[1], edge_index[0]])
    if graph.edge_weights is not None:
        edge_weights = graph.edge_weights.cpu().numpy().astype(np.float64)
        data = np.concatenate([edge_weights, edge_weights])
    else:
        data = np.ones(rows.shape[0], dtype=np.float64)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    distance_matrix = shortest_path(adjacency, directed=False)
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("graph is disconnected")

    distances: list[float] = []
    weights: list[float] = []
    for source in range(n_nodes):
        for target in range(source + 1, n_nodes):
            distance = float(distance_matrix[source, target])
            distances.append(distance)
            weights.append(1.0 / (distance**2))

    return np.array(distances), np.array(weights)


@register
class SGD2(CompetitorBase):
    """Competitor adapter for the reference ``s_gd2`` stress-SGD engine."""

    name = "sgd2"
    max_nodes = 50_000
    variant_param_names = frozenset({"eps", "t_max"})

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run ``s_gd2`` and convert its output to a CPU tensor.

        Parameters
        ----------
        graph : DaguaGraph
            Input graph to lay out.
        timeout : float, optional
            Unused adapter timeout in seconds. Included for interface
            compatibility with the benchmark harness.
        seed : int | None, default=None
            Random seed forwarded to ``s_gd2`` when explicitly requested.

        Returns
        -------
        CompetitorResult
            Layout result with positions shaped ``[N, 2]`` on CPU, or an error
            payload if the third-party engine fails.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run ``s_gd2`` and convert its output to a CPU tensor.

        Parameters
        ----------
        graph : DaguaGraph
            Input graph to lay out.
        timeout : float, optional
            Unused adapter timeout in seconds. Included for interface
            compatibility with the benchmark harness.
        seed : int | None, default=None
            Random seed forwarded to ``s_gd2`` when explicitly requested.

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

            sources, targets, edge_weights = _symmetrized_unique_edges(graph)
            if sources.size == 0:
                pos = torch.zeros((graph.num_nodes, 2), dtype=torch.float32)
                elapsed = time.perf_counter() - start
                return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)

            layout_kwargs: dict[str, Any] = {}
            if seed is not None:
                layout_kwargs["random_seed"] = seed
            if variant_params is not None:
                layout_kwargs.update(dict(variant_params))
            if edge_weights is not None:
                layout_kwargs["V"] = edge_weights.tolist()
            coordinates = s_gd2.layout(sources.tolist(), targets.tolist(), **layout_kwargs)
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


@register
class SGD2MDS(CompetitorBase):
    """Competitor adapter for ``s_gd2.mds_direct`` classical MDS."""

    name = "sgd2_mds"
    max_nodes = 5_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run ``s_gd2.mds_direct`` on graph shortest-path distances.

        Parameters
        ----------
        graph : DaguaGraph
            Input graph to lay out.
        timeout : float, optional
            Unused adapter timeout in seconds. Included for interface
            compatibility with the benchmark harness.
        seed : int | None, default=None
            Random seed forwarded to ``s_gd2.mds_direct``.

        Returns
        -------
        CompetitorResult
            Layout result with positions shaped ``[N, 2]`` on CPU, or an error
            payload if the reference engine fails.
        """
        del timeout

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

            distances, weights = _build_condensed_distances(graph)
            coordinates = s_gd2.mds_direct(
                graph.num_nodes,
                distances,
                weights,
                random_seed=seed if seed is not None else 42,
            )
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

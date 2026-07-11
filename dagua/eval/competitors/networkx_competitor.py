"""NetworkX competitor adapters for selected graph layouts.

Size policy (r80-P6): NetworkX's layout algorithms (spring, etc.) do not
accept per-node size hints, so these adapters are SIZE-BLIND regardless of
``dagua.eval.size_policy.size_aware_externals()``. Their positions are
still scored against dagua's real label-measured ``node_sizes`` (like every
other engine), so their overlap term simply reflects that they were never
told how big the nodes actually are.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

import torch

from dagua.eval.competitors.base import (
    CompetitorBase,
    CompetitorResult,
    register,
)

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


def _graph_to_nx(graph: DaguaGraph, duplicate_policy: str = "sum") -> Any:
    """Convert a ``DaguaGraph`` to a weighted ``networkx.DiGraph``.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph whose topology and optional edge weights should be copied.
    duplicate_policy : {"sum", "last"}, default="sum"
        Policy used when repeated directed edges are copied into the simple
        ``DiGraph`` representation.

    Returns
    -------
    Any
        ``networkx.DiGraph`` with ``weight`` edge attributes when available.
    """
    import networkx as nx

    if duplicate_policy not in {"sum", "last"}:
        raise ValueError("duplicate_policy must be one of 'sum' or 'last'.")

    G = nx.DiGraph()
    G.add_nodes_from(range(graph.num_nodes))
    if graph.edge_index.numel() > 0:
        collapsed_weights: dict[tuple[int, int], float] = {}
        ei = graph.edge_index
        weights = graph.edge_weights
        for e in range(ei.shape[1]):
            source = ei[0, e].item()
            target = ei[1, e].item()
            edge_key = (source, target)
            if weights is not None:
                edge_weight = float(weights[e].item())
            else:
                edge_weight = 1.0
            if duplicate_policy == "sum":
                edge_weight += collapsed_weights.get(edge_key, 0.0)
            collapsed_weights[edge_key] = edge_weight
        for (source, target), edge_weight in collapsed_weights.items():
            G.add_edge(source, target, weight=edge_weight)
    return G


def _normalize_output_dtype(output_dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Normalize user-facing dtype names to torch dtypes.

    Parameters
    ----------
    output_dtype : str or torch.dtype
        Requested floating-point output dtype.

    Returns
    -------
    torch.dtype
        Normalized torch dtype.

    Raises
    ------
    ValueError
        If ``output_dtype`` is not a supported floating-point dtype.
    """
    if isinstance(output_dtype, torch.dtype):
        if output_dtype in {torch.float32, torch.float64}:
            return output_dtype
        raise ValueError("output_dtype must be torch.float32 or torch.float64.")
    if output_dtype in {"float32", "torch.float32"}:
        return torch.float32
    if output_dtype in {"float64", "torch.float64"}:
        return torch.float64
    raise ValueError("output_dtype must be 'float32' or 'float64'.")


def _nx_pos_to_tensor(
    nx_pos: dict,
    num_nodes: int,
    output_scale: float,
    output_dtype: Union[str, torch.dtype] = torch.float32,
) -> torch.Tensor:
    """Convert a NetworkX position mapping to a tensor.

    Parameters
    ----------
    nx_pos : dict
        Mapping from integer node IDs to two-dimensional coordinates.
    num_nodes : int
        Number of nodes ``N`` expected in the output tensor.
    output_scale : float
        Adapter-level multiplier applied after the NetworkX algorithm returns.
    output_dtype : str or torch.dtype, default=torch.float32
        Floating-point dtype for the returned tensor.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    dtype = _normalize_output_dtype(output_dtype)
    pos = torch.zeros(num_nodes, 2, dtype=dtype)
    for node_id, (x, y) in nx_pos.items():
        if node_id < num_nodes:
            pos[node_id, 0] = float(x) * output_scale
            pos[node_id, 1] = float(y) * output_scale
    return pos


def _networkx_center(dim: int) -> Any:
    """Return NetworkX's default spectral-layout center vector.

    Parameters
    ----------
    dim : int
        Layout dimensionality.

    Returns
    -------
    Any
        NumPy zero vector with shape ``[dim]``.
    """
    import numpy as np

    return np.zeros(dim, dtype=float)


def _networkx_spectral_edge_case_positions(G: Any, dim: int) -> Optional[dict[int, Any]]:
    """Return NetworkX spectral special-case positions when applicable.

    Parameters
    ----------
    G : Any
        NetworkX graph.
    dim : int
        Layout dimensionality.

    Returns
    -------
    dict[int, Any] | None
        Position mapping for graphs with at most two nodes, otherwise ``None``.
    """
    import numpy as np

    center = _networkx_center(dim)
    if len(G) > 2:
        return None
    if len(G) == 0:
        pos = np.array([])
    elif len(G) == 1:
        pos = np.array([center])
    else:
        pos = np.array([np.zeros(dim), center * 2.0])
    return dict(zip(G, pos))


def _spectral_laplacian_matrix(A: Any, normalization: str) -> tuple[Any, bool]:
    """Build a NetworkX-style spectral Laplacian.

    Parameters
    ----------
    A : Any
        Symmetric SciPy sparse adjacency matrix.
    normalization : str
        Laplacian normalization mode: ``"unnormalized"`` or ``"random_walk"``.

    Returns
    -------
    tuple[Any, bool]
        Sparse Laplacian matrix and whether it is symmetric.

    Raises
    ------
    ValueError
        If ``normalization`` is not supported by this adapter.
    """
    import numpy as np
    import scipy as sp

    degrees = np.asarray(A.sum(axis=1)).reshape(-1).astype(float, copy=False)
    if normalization == "unnormalized":
        degree_matrix = sp.sparse.dia_array((degrees, 0), shape=A.shape).tocsr()
        return degree_matrix - A, True
    if normalization == "random_walk":
        inv_degree = np.zeros_like(degrees)
        nonzero_mask = degrees > 0.0
        inv_degree[nonzero_mask] = 1.0 / degrees[nonzero_mask]
        normalized = sp.sparse.diags(inv_degree, offsets=0, format="csr")
        identity = sp.sparse.identity(A.shape[0], format="csr", dtype=float)
        return identity - (normalized @ A), False
    raise ValueError("normalization must be one of 'unnormalized' or 'random_walk'.")


def _networkx_laplacian_spectral_array(A: Any, dim: int, normalization: str) -> Any:
    """Compute spectral coordinates from a NetworkX adjacency matrix.

    Parameters
    ----------
    A : Any
        Symmetric SciPy sparse adjacency matrix with shape ``[N, N]``.
    dim : int
        Output dimensionality.
    normalization : str
        Laplacian normalization mode.

    Returns
    -------
    Any
        NumPy coordinate array with shape ``[N, dim]``.
    """
    import numpy as np
    import scipy as sp

    laplacian, symmetric = _spectral_laplacian_matrix(A, normalization=normalization)
    num_nodes = int(A.shape[0])
    if num_nodes < 500:
        dense_laplacian = laplacian.toarray()
        if symmetric:
            eigenvalues, eigenvectors = np.linalg.eigh(dense_laplacian)
        else:
            eigenvalues, eigenvectors = np.linalg.eig(dense_laplacian)
    else:
        k = dim + 1
        ncv = max((2 * k) + 1, int(np.sqrt(num_nodes)))
        if symmetric:
            eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(laplacian, k, which="SM", ncv=ncv)
        else:
            eigenvalues, eigenvectors = sp.sparse.linalg.eigs(laplacian, k, which="SR", ncv=ncv)
    index = np.argsort(np.real(eigenvalues))[1 : dim + 1]
    return np.real(eigenvectors[:, index])


class _NetworkXBase(CompetitorBase):
    """Base for NetworkX layout algorithms."""

    layout_func: str = "spring_layout"
    layout_kwargs: dict[str, Any] = {}
    output_scale: float = 500.0
    output_dtype: torch.dtype = torch.float32
    duplicate_policy: str = "sum"

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the configured NetworkX layout algorithm.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility timeout parameter.
        seed : int | None, default=None
            Random seed forwarded when the underlying NetworkX layout accepts
            a ``seed`` keyword. Deterministic layouts ignore it.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the configured NetworkX layout algorithm.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility timeout parameter.
        seed : int | None, default=None
            Random seed forwarded when the underlying NetworkX layout accepts
            a ``seed`` keyword. Deterministic layouts ignore it.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        import networkx as nx

        G = _graph_to_nx(graph, duplicate_policy=self.duplicate_policy)

        start = time.perf_counter()
        try:
            func = getattr(nx, self.layout_func)
            layout_kwargs = dict(self.layout_kwargs)
            if variant_params is not None:
                layout_kwargs.update(dict(variant_params))
            output_scale = float(layout_kwargs.pop("output_scale", self.output_scale))
            output_dtype = _normalize_output_dtype(
                layout_kwargs.pop("output_dtype", self.output_dtype)
            )
            if seed is not None and "seed" in layout_kwargs:
                layout_kwargs["seed"] = seed
            nx_pos = func(G, **layout_kwargs)
            elapsed = time.perf_counter() - start
            pos = _nx_pos_to_tensor(
                nx_pos,
                graph.num_nodes,
                output_scale=output_scale,
                output_dtype=output_dtype,
            )
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=None, runtime_seconds=elapsed, error=str(e))

    def available(self) -> bool:
        try:
            import networkx  # noqa: F401

            return True
        except ImportError:
            return False


@register
class NetworkXSpring(_NetworkXBase):
    name = "nx_spring"
    max_nodes = 50_000
    layout_func = "spring_layout"
    layout_kwargs = {"seed": 42, "iterations": 50, "method": "force"}
    variant_param_names = frozenset({"gravity", "iterations", "k", "scale", "method"})


@register
class NetworkXKamadaKawai(_NetworkXBase):
    name = "nx_kamada_kawai"
    max_nodes = 5_000
    layout_func = "kamada_kawai_layout"
    layout_kwargs = {}
    output_scale = 1.0
    duplicate_policy = "last"
    variant_param_names = frozenset({"output_dtype", "output_scale"})


@register
class NetworkXSpectral(_NetworkXBase):
    """Competitor adapter for NetworkX's spectral layout."""

    name = "nx_spectral"
    max_nodes = 10_000
    layout_func = "spectral_layout"
    layout_kwargs = {"dim": 2}
    variant_param_names = frozenset({"dim", "scale"})
    output_scale = 1.0
    duplicate_policy = "last"


@register
class NetworkXLaplacianSpectral(_NetworkXBase):
    """Competitor adapter for NetworkX-backed Laplacian spectral variants."""

    name = "nx_spectral_random_walk"
    max_nodes = 10_000
    layout_kwargs = {"dim": 2, "normalization": "random_walk", "scale": 1.0}
    variant_param_names = frozenset(
        {"dim", "normalization", "output_dtype", "output_scale", "scale"}
    )
    output_scale = 1.0
    duplicate_policy = "last"

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the configured Laplacian spectral layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility timeout parameter.
        seed : int | None, default=None
            Unused deterministic-layout seed.
        variant_params : Mapping[str, Any] | None, default=None
            Optional parameters forwarded from ``AlgorithmVariant.original_params``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout, seed

        import networkx as nx

        G = _graph_to_nx(graph, duplicate_policy=self.duplicate_policy)

        start = time.perf_counter()
        try:
            layout_kwargs = dict(self.layout_kwargs)
            if variant_params is not None:
                layout_kwargs.update(dict(variant_params))
            output_scale = float(layout_kwargs.pop("output_scale", self.output_scale))
            output_dtype = _normalize_output_dtype(
                layout_kwargs.pop("output_dtype", self.output_dtype)
            )
            dim = int(layout_kwargs.pop("dim", 2))
            scale = float(layout_kwargs.pop("scale", 1.0))
            normalization = str(layout_kwargs.pop("normalization", "random_walk"))
            edge_case_pos = _networkx_spectral_edge_case_positions(G, dim=dim)
            if edge_case_pos is None:
                A = nx.to_scipy_sparse_array(G, weight="weight", dtype="d")
                if G.is_directed():
                    A = A + A.T
                pos_array = _networkx_laplacian_spectral_array(
                    A,
                    dim=dim,
                    normalization=normalization,
                )
                pos_array = nx.drawing.layout.rescale_layout(pos_array, scale=scale)
                nx_pos = dict(zip(G, pos_array))
            else:
                nx_pos = edge_case_pos
            elapsed = time.perf_counter() - start
            pos = _nx_pos_to_tensor(
                nx_pos,
                graph.num_nodes,
                output_scale=output_scale,
                output_dtype=output_dtype,
            )
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=None, runtime_seconds=elapsed, error=str(e))

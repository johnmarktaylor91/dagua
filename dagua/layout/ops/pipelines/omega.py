"""Omega/RDMDS resistance-distance stress pipeline.

This module ports the core shape of ``likr/egraph-rs`` Omega: an RDMDS
spectral embedding is used to define target distances, then a sparse SGD stress
pass refines edge pairs plus sampled non-edge pairs. The Rust CLI currently
uses ``thread_rng`` and is not seedable, so this port keeps the same stage order
while using Dagua's explicit ``seed`` for deterministic runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_DEFAULT_D = 2
_DEFAULT_K = 30
_DEFAULT_MIN_DIST = 1.0e-3
_DEFAULT_SHIFT = 1.0e-3
_DEFAULT_UNIT_EDGE_LENGTH = 1.0
_DEFAULT_SGD_ITERATIONS = 100
_DEFAULT_SGD_EPS = 0.1


@dataclass(frozen=True)
class OmegaConfig:
    """Configuration for the Omega/RDMDS pipeline.

    Parameters
    ----------
    d : int, default=2
        RDMDS embedding rank. The final layout uses the first two dimensions.
    k : int, default=30
        Number of random node-pair attempts per source node.
    min_dist : float, default=1e-3
        Minimum target distance in the embedding-distance matrix.
    shift : float, default=1e-3
        Positive diagonal shift used by the Rust RDMDS inverse iteration.
        The dense port records this value for API parity; direct eigensolve does
        not require the shifted system.
    unit_edge_length : float, default=1.0
        Weight assigned to every graph edge in the standard Laplacian.
    sgd_iterations : int, default=100
        Number of SparseSGD refinement iterations.
    sgd_eps : float, default=0.1
        Final scheduler epsilon used to derive the minimum learning rate.
    seed : int, default=42
        Deterministic sampler and shuffle seed.
    dtype : torch.dtype, default=torch.float32
        Output tensor dtype.
    """

    d: int = _DEFAULT_D
    k: int = _DEFAULT_K
    min_dist: float = _DEFAULT_MIN_DIST
    shift: float = _DEFAULT_SHIFT
    unit_edge_length: float = _DEFAULT_UNIT_EDGE_LENGTH
    sgd_iterations: int = _DEFAULT_SGD_ITERATIONS
    sgd_eps: float = _DEFAULT_SGD_EPS
    seed: int = 42
    dtype: torch.dtype = torch.float32


@dataclass
class _OmegaPair:
    """One sparse stress pair.

    Parameters
    ----------
    i : int
        First node index.
    j : int
        Second node index.
    distance : float
        Target Euclidean distance.
    weight : float
        Stress weight, equal to ``1 / distance**2``.
    """

    i: int
    j: int
    distance: float
    weight: float


def _undirected_edges(edge_index: torch.Tensor, num_nodes: int) -> List[Tuple[int, int]]:
    """Return valid undirected edges in input order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[tuple[int, int]]
        Unique undirected edges with self-loops and invalid endpoints removed.
    """
    if edge_index.numel() == 0:
        return []
    edges: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for raw_u, raw_v in edge_index.detach().cpu().t().tolist():
        u = int(raw_u)
        v = int(raw_v)
        if u == v or u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            continue
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        edges.append((u, v))
    return edges


def _laplacian(num_nodes: int, edges: List[Tuple[int, int]], unit_edge_length: float) -> np.ndarray:
    """Build the standard graph Laplacian used by egraph-rs RDMDS.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edges : list[tuple[int, int]]
        Unique undirected graph edges.
    unit_edge_length : float
        Edge weight added to the Laplacian degree and adjacency terms.

    Returns
    -------
    numpy.ndarray
        Dense Laplacian matrix with shape ``[N, N]``.
    """
    lap = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    weight = float(unit_edge_length)
    for u, v in edges:
        lap[u, u] += weight
        lap[v, v] += weight
        lap[u, v] -= weight
        lap[v, u] -= weight
    return lap


def _rdmds_embedding(
    num_nodes: int,
    edges: List[Tuple[int, int]],
    config: OmegaConfig,
) -> np.ndarray:
    """Compute the resistance-distance MDS spectral embedding.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edges : list[tuple[int, int]]
        Unique undirected graph edges.
    config : OmegaConfig
        RDMDS configuration.

    Returns
    -------
    numpy.ndarray
        Embedding array with shape ``[N, d]``.
    """
    rank = max(1, int(config.d))
    if num_nodes == 0:
        return np.zeros((0, rank), dtype=np.float64)
    if num_nodes == 1 or not edges:
        return np.zeros((num_nodes, rank), dtype=np.float64)

    lap = _laplacian(num_nodes, edges, config.unit_edge_length)
    eigenvalues, eigenvectors = np.linalg.eigh(lap)
    order = np.argsort(eigenvalues, kind="stable")
    coords = np.zeros((num_nodes, rank), dtype=np.float64)
    for out_dim, eigen_idx in enumerate(order[1 : rank + 1]):
        value = max(float(eigenvalues[eigen_idx]), 0.0)
        if value <= 0.0:
            continue
        coords[:, out_dim] = eigenvectors[:, eigen_idx] / math.sqrt(value)
    return coords


def _embedding_distance(embedding: np.ndarray, i: int, j: int, min_dist: float) -> float:
    """Return the clamped Euclidean distance between two embedding rows.

    Parameters
    ----------
    embedding : numpy.ndarray
        RDMDS embedding with shape ``[N, d]``.
    i : int
        First node index.
    j : int
        Second node index.
    min_dist : float
        Minimum returned distance.

    Returns
    -------
    float
        Euclidean distance clamped to ``min_dist``.
    """
    distance = float(np.linalg.norm(embedding[i] - embedding[j]))
    return max(distance, float(min_dist))


def _build_pairs(
    edges: List[Tuple[int, int]],
    embedding: np.ndarray,
    config: OmegaConfig,
    rng: np.random.Generator,
) -> List[_OmegaPair]:
    """Build edge pairs plus reference-ordered random pairs.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Unique undirected graph edges in input order.
    embedding : numpy.ndarray
        RDMDS embedding with shape ``[N, d]``.
    config : OmegaConfig
        SparseSGD pair configuration.
    rng : numpy.random.Generator
        Deterministic random generator.

    Returns
    -------
    list[_OmegaPair]
        Sparse stress pairs.
    """
    num_nodes = int(embedding.shape[0])
    pairs: List[_OmegaPair] = []
    used: set[Tuple[int, int]] = set()

    for u, v in edges:
        key = (u, v) if u < v else (v, u)
        if key in used:
            continue
        used.add(key)
        distance = _embedding_distance(embedding, u, v, config.min_dist)
        pairs.append(_OmegaPair(u, v, distance, 1.0 / (distance * distance)))

    for i in range(num_nodes):
        for _ in range(max(0, int(config.k))):
            j = int(rng.integers(0, num_nodes))
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            if key in used:
                continue
            used.add(key)
            distance = _embedding_distance(embedding, i, j, config.min_dist)
            pairs.append(_OmegaPair(i, j, distance, 1.0 / (distance * distance)))
    return pairs


def _scheduler_bounds(pairs: List[_OmegaPair], epsilon: float) -> Tuple[float, float]:
    """Compute egraph-rs exponential scheduler bounds.

    Parameters
    ----------
    pairs : list[_OmegaPair]
        Sparse stress pairs.
    epsilon : float
        Scheduler epsilon.

    Returns
    -------
    tuple[float, float]
        ``(eta_min, eta_max)`` bounds.
    """
    weights = [pair.weight for pair in pairs if pair.weight > 0.0]
    if not weights:
        return 0.0, 0.0
    return float(epsilon) / max(weights), 1.0 / min(weights)


def _run_sparse_sgd(
    embedding: np.ndarray,
    pairs: List[_OmegaPair],
    config: OmegaConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run Omega SparseSGD refinement.

    Parameters
    ----------
    embedding : numpy.ndarray
        Initial RDMDS embedding with shape ``[N, d]``.
    pairs : list[_OmegaPair]
        Sparse stress pairs.
    config : OmegaConfig
        SGD configuration.
    rng : numpy.random.Generator
        Deterministic random generator used for per-iteration shuffles.

    Returns
    -------
    numpy.ndarray
        Refined positions with shape ``[N, 2]``.
    """
    pos = np.zeros((embedding.shape[0], 2), dtype=np.float64)
    if embedding.shape[1] > 0:
        pos[:, : min(2, embedding.shape[1])] = embedding[:, :2]
    if not pairs or config.sgd_iterations <= 0:
        return pos

    eta_min, eta_max = _scheduler_bounds(pairs, config.sgd_eps)
    if eta_max <= 0.0:
        return pos
    decay = 0.0
    if config.sgd_iterations > 1 and eta_min > 0.0:
        decay = math.log(eta_max / eta_min) / float(config.sgd_iterations - 1)

    order = np.arange(len(pairs), dtype=np.int64)
    for step in range(config.sgd_iterations):
        eta = eta_max * math.exp(-decay * float(step))
        rng.shuffle(order)
        for pair_idx in order.tolist():
            pair = pairs[pair_idx]
            delta = pos[pair.i] - pos[pair.j]
            norm = float(np.linalg.norm(delta))
            if norm <= 0.0:
                continue
            mu = min(eta * pair.weight, 1.0)
            ratio = 0.5 * (norm - pair.distance) / norm
            move = delta * ratio * mu
            pos[pair.i] -= move
            pos[pair.j] += move
    return pos


@register_op
@dataclass
class ComputeOmegaEmbedding(Op):
    """Compute RDMDS coordinates for the Omega pipeline."""

    config: OmegaConfig
    name: str = "compute_omega_embedding"
    category: OpCategory = OpCategory.INIT
    writes: Tuple[str, ...] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute and cache the RDMDS embedding.

        Parameters
        ----------
        problem : LayoutProblem
            Graph topology and node count.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context; accepted for pipeline compatibility.

        Returns
        -------
        SolveState
            State with ``omega_edges`` and ``omega_embedding`` cached.
        """
        del ctx
        edges = _undirected_edges(problem.edge_index, problem.num_nodes)
        embedding = _rdmds_embedding(problem.num_nodes, edges, self.config)
        state.extras["omega_edges"] = edges
        state.extras["omega_embedding"] = embedding
        return state


@register_op
@dataclass
class BuildOmegaPairs(Op):
    """Build Omega SparseSGD pairs from the RDMDS embedding."""

    config: OmegaConfig
    name: str = "build_omega_pairs"
    category: OpCategory = OpCategory.PREPROCESS
    reads: Tuple[str, ...] = ("extras",)
    writes: Tuple[str, ...] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Sample edge and random pairs in Omega order.

        Parameters
        ----------
        problem : LayoutProblem
            Graph topology and deterministic seed.
        state : SolveState
            Mutable solve state with cached RDMDS embedding.
        ctx : RuntimeContext
            Runtime context; accepted for pipeline compatibility.

        Returns
        -------
        SolveState
            State with ``omega_pairs`` cached.
        """
        del problem, ctx
        embedding = state.extras.get("omega_embedding")
        edges = state.extras.get("omega_edges")
        if not isinstance(embedding, np.ndarray) or not isinstance(edges, list):
            raise RuntimeError("Omega embedding stage must run before pair construction.")
        rng = np.random.default_rng(self.config.seed)
        state.extras["omega_pairs"] = _build_pairs(edges, embedding, self.config, rng)
        return state


@register_op
@dataclass
class RunOmegaSparseSgd(Op):
    """Run the Omega SparseSGD position refinement."""

    config: OmegaConfig
    name: str = "run_omega_sparse_sgd"
    category: OpCategory = OpCategory.OPTIMIZE
    reads: Tuple[str, ...] = ("extras",)
    writes: Tuple[str, ...] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Refine and store Omega positions.

        Parameters
        ----------
        problem : LayoutProblem
            Graph topology and output device.
        state : SolveState
            Mutable solve state with cached embedding and pairs.
        ctx : RuntimeContext
            Runtime context; accepted for pipeline compatibility.

        Returns
        -------
        SolveState
            State with final ``pos`` tensor populated.
        """
        del ctx
        embedding = state.extras.get("omega_embedding")
        pairs = state.extras.get("omega_pairs")
        if not isinstance(embedding, np.ndarray) or not isinstance(pairs, list):
            raise RuntimeError("Omega pair construction must run before SparseSGD.")
        rng = np.random.default_rng(self.config.seed)
        pos = _run_sparse_sgd(embedding, pairs, self.config, rng)
        state.pos = torch.as_tensor(pos, dtype=self.config.dtype, device=problem.edge_index.device)
        return state


def build_omega_pipeline(config: Optional[OmegaConfig] = None) -> Pipeline:
    """Build the Omega/RDMDS pipeline.

    Parameters
    ----------
    config : OmegaConfig, optional
        Pipeline configuration. ``None`` uses Omega reference defaults.

    Returns
    -------
    Pipeline
        RDMDS embedding, pair construction, and SparseSGD refinement stages.
    """
    resolved = OmegaConfig() if config is None else config
    if resolved.d <= 0:
        raise ValueError("d must be positive.")
    if resolved.k < 0:
        raise ValueError("k must be non-negative.")
    if resolved.min_dist <= 0.0:
        raise ValueError("min_dist must be positive.")
    if resolved.unit_edge_length <= 0.0:
        raise ValueError("unit_edge_length must be positive.")
    return Pipeline(
        [
            ComputeOmegaEmbedding(resolved),
            BuildOmegaPairs(resolved),
            RunOmegaSparseSgd(resolved),
        ],
        name="omega_pipeline",
    )


def layout_omega_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    *,
    d: int = _DEFAULT_D,
    k: int = _DEFAULT_K,
    min_dist: float = _DEFAULT_MIN_DIST,
    shift: float = _DEFAULT_SHIFT,
    unit_edge_length: float = _DEFAULT_UNIT_EDGE_LENGTH,
    sgd_iterations: int = _DEFAULT_SGD_ITERATIONS,
    sgd_eps: float = _DEFAULT_SGD_EPS,
    seed: Optional[int] = 42,
    dtype: Union[torch.dtype, str] = torch.float32,
    **kwargs: object,
) -> torch.Tensor:
    """Lay out a graph with Omega/RDMDS.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``. Accepted for pipeline API
        compatibility; Omega does not use node boxes.
    d : int, default=2
        RDMDS embedding rank.
    k : int, default=30
        Number of random pair attempts per node.
    min_dist : float, default=1e-3
        Minimum target distance.
    shift : float, default=1e-3
        RDMDS shift parameter retained for API parity.
    unit_edge_length : float, default=1.0
        Standard Laplacian edge weight.
    sgd_iterations : int, default=100
        Number of SparseSGD refinement iterations.
    sgd_eps : float, default=0.1
        Scheduler epsilon.
    seed : int, optional
        Deterministic sampler seed. ``None`` resolves to ``42``.
    dtype : torch.dtype or str, default=torch.float32
        Output dtype.
    **kwargs : object
        Additional dispatch kwargs accepted for compatibility.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    del node_sizes, kwargs
    resolved_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    config = OmegaConfig(
        d=d,
        k=k,
        min_dist=min_dist,
        shift=shift,
        unit_edge_length=unit_edge_length,
        sgd_iterations=sgd_iterations,
        sgd_eps=sgd_eps,
        seed=42 if seed is None else int(seed),
        dtype=resolved_dtype,
    )
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, seed=config.seed)
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(edge_index.device)))
    final_state = build_omega_pipeline(config).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Omega pipeline did not produce positions.")
    return final_state.pos.to(device=edge_index.device, dtype=resolved_dtype)


__all__ = [
    "BuildOmegaPairs",
    "ComputeOmegaEmbedding",
    "OmegaConfig",
    "RunOmegaSparseSgd",
    "build_omega_pipeline",
    "layout_omega_pipeline",
]

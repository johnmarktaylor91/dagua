"""UMAP graph layout expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.classic.umap_layout import (
    _NEGATIVE_SAMPLE_RATE,
    _all_pairs_shortest_paths,
    _build_undirected_adjacency,
    _fit_ab,
    _knn_from_distances,
    _layout_device,
    _layout_extent,
    _normalize_positions,
    _optimize_embedding,
    _select_positive_edges,
    _smooth_knn_dist,
    _spectral_initialization,
    _symmetrized_fuzzy_graph,
    _undirected_edge_weight_lookup,
    _validate_inputs,
)
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory

# -- extras key constants ---------------------------------------------------
_ADJACENCY_KEY = "umap_adjacency"
_DISTANCES_KEY = "umap_distances"
_KNN_INDICES_KEY = "umap_knn_indices"
_KNN_DISTANCES_KEY = "umap_knn_distances"
_SIGMAS_KEY = "umap_sigmas"
_RHOS_KEY = "umap_rhos"
_FUZZY_HEAD_KEY = "umap_fuzzy_head"
_FUZZY_TAIL_KEY = "umap_fuzzy_tail"
_FUZZY_WEIGHT_KEY = "umap_fuzzy_weight"
_CURVE_A_KEY = "umap_curve_a"
_CURVE_B_KEY = "umap_curve_b"
_POSITIVE_HEAD_KEY = "umap_positive_head"
_POSITIVE_TAIL_KEY = "umap_positive_tail"
_EPOCHS_PER_SAMPLE_KEY = "umap_epochs_per_sample"
_N_EPOCHS_KEY = "umap_n_epochs"
_N_NEIGHBORS_KEY = "umap_n_neighbors"
_LEARNING_RATE_KEY = "umap_learning_rate"
_NEGATIVE_SAMPLE_RATE_KEY = "umap_negative_sample_rate"
_GAMMA_KEY = "umap_gamma"
_MIN_DIST_KEY = "umap_min_dist"
_SPREAD_KEY = "umap_spread"


class _BuildUMAPAdjacency(Op):
    """Build undirected adjacency list from graph edges."""

    name: ClassVar[str] = "umap_build_adjacency"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build the undirected adjacency list exactly like classic UMAP.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with adjacency list stored in ``extras``.
        """
        del ctx

        adjacency = _build_undirected_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        state.extras[_ADJACENCY_KEY] = adjacency
        return state


class _ComputeAllPairsShortestPaths(Op):
    """Compute all-pairs graph distances via BFS or Dijkstra."""

    name: ClassVar[str] = "umap_all_pairs_shortest_paths"
    category: ClassVar[OpCategory] = OpCategory.DISTANCE
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute dense shortest-path distance matrix.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing the adjacency list.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with distance matrix stored in ``extras``.
        """
        del problem, ctx

        adjacency = state.extras[_ADJACENCY_KEY]
        distances = _all_pairs_shortest_paths(adjacency=adjacency)
        state.extras[_DISTANCES_KEY] = distances
        return state


class _ExtractKNN(Op):
    """Extract k-nearest neighbors from the dense distance matrix."""

    name: ClassVar[str] = "umap_extract_knn"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Extract kNN indices and distances from the distance matrix.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing distance matrix and n_neighbors.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with kNN indices and distances stored in ``extras``.
        """
        del problem, ctx

        distances = state.extras[_DISTANCES_KEY]
        n_neighbors = state.extras[_N_NEIGHBORS_KEY]
        knn_indices, knn_distances = _knn_from_distances(
            distances=distances,
            n_neighbors=n_neighbors,
        )
        state.extras[_KNN_INDICES_KEY] = knn_indices
        state.extras[_KNN_DISTANCES_KEY] = knn_distances
        return state


class _SmoothKNNDist(Op):
    """Solve UMAP smooth-kNN bandwidth per node."""

    name: ClassVar[str] = "umap_smooth_knn_dist"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute per-node sigma and rho for the fuzzy simplicial set.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing kNN distances and n_neighbors.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with sigmas and rhos stored in ``extras``.
        """
        del problem, ctx

        knn_distances = state.extras[_KNN_DISTANCES_KEY]
        n_neighbors = state.extras[_N_NEIGHBORS_KEY]
        sigmas, rhos = _smooth_knn_dist(
            knn_distances=knn_distances,
            n_neighbors=n_neighbors,
        )
        state.extras[_SIGMAS_KEY] = sigmas
        state.extras[_RHOS_KEY] = rhos
        return state


class _BuildFuzzySimplicialSet(Op):
    """Build the symmetrized fuzzy simplicial set graph."""

    name: ClassVar[str] = "umap_build_fuzzy_set"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build the symmetrized fuzzy graph with optional edge-weight scaling.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing edge_index and edge_weights.
        state : SolveState
            Mutable solve state containing kNN data and sigma/rho.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with fuzzy graph head/tail/weight stored in ``extras``.
        """
        del ctx

        knn_indices = state.extras[_KNN_INDICES_KEY]
        knn_distances = state.extras[_KNN_DISTANCES_KEY]
        sigmas = state.extras[_SIGMAS_KEY]
        rhos = state.extras[_RHOS_KEY]

        head, tail, weight = _symmetrized_fuzzy_graph(
            knn_indices=knn_indices,
            knn_distances=knn_distances,
            sigmas=sigmas,
            rhos=rhos,
        )

        # Apply edge-weight scaling exactly like the classic implementation.
        if problem.edge_weights is not None and weight.numel() > 0:
            edge_weight_lookup = _undirected_edge_weight_lookup(
                edge_index=problem.edge_index,
                edge_weights=problem.edge_weights,
            )
            scaled_weight = weight.clone()
            for index in range(weight.shape[0]):
                pair = (
                    min(int(head[index].item()), int(tail[index].item())),
                    max(int(head[index].item()), int(tail[index].item())),
                )
                scaled_weight[index] = scaled_weight[index] * edge_weight_lookup.get(pair, 1.0)
            weight = scaled_weight

        state.extras[_FUZZY_HEAD_KEY] = head
        state.extras[_FUZZY_TAIL_KEY] = tail
        state.extras[_FUZZY_WEIGHT_KEY] = weight
        return state


class _SpectralInitialization(Op):
    """Compute spectral initialization for the UMAP embedding."""

    name: ClassVar[str] = "umap_spectral_init"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute spectral initialization from the fuzzy graph.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing num_nodes and seed.
        state : SolveState
            Mutable solve state containing the fuzzy graph.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with initial embedding in ``state.pos``.
        """
        del ctx

        head = state.extras[_FUZZY_HEAD_KEY]
        tail = state.extras[_FUZZY_TAIL_KEY]
        weight = state.extras[_FUZZY_WEIGHT_KEY]

        state.pos = _spectral_initialization(
            num_nodes=problem.num_nodes,
            head=head,
            tail=tail,
            weight=weight,
            seed=problem.seed,
        )
        return state


class _FitCurveParameters(Op):
    """Fit the UMAP a,b curve parameters from min_dist and spread."""

    name: ClassVar[str] = "umap_fit_curve"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Fit UMAP curve parameters for the low-dimensional membership function.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing min_dist and spread in extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with fitted curve_a and curve_b stored in ``extras``.
        """
        del problem, ctx

        min_dist = state.extras[_MIN_DIST_KEY]
        spread = state.extras[_SPREAD_KEY]
        a, b = _fit_ab(min_dist=min_dist, spread=spread)
        state.extras[_CURVE_A_KEY] = a
        state.extras[_CURVE_B_KEY] = b
        return state


class _SelectPositiveEdges(Op):
    """Prune weak edges and build epoch sampling intervals."""

    name: ClassVar[str] = "umap_select_positive_edges"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Select positive edges and compute per-edge epoch intervals.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing the fuzzy graph and epoch count.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with positive edges and epochs_per_sample in ``extras``.
        """
        del problem, ctx

        head = state.extras[_FUZZY_HEAD_KEY]
        tail = state.extras[_FUZZY_TAIL_KEY]
        weight = state.extras[_FUZZY_WEIGHT_KEY]
        n_epochs = state.extras[_N_EPOCHS_KEY]

        positive_head, positive_tail, epochs_per_sample = _select_positive_edges(
            head=head,
            tail=tail,
            weight=weight,
            n_epochs=n_epochs,
        )
        state.extras[_POSITIVE_HEAD_KEY] = positive_head
        state.extras[_POSITIVE_TAIL_KEY] = positive_tail
        state.extras[_EPOCHS_PER_SAMPLE_KEY] = epochs_per_sample
        return state


class _OptimizeUMAPEmbedding(Op):
    """Run the cross-entropy SGD optimization with negative sampling."""

    name: ClassVar[str] = "umap_optimize_embedding"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the UMAP SGD loop over the initial embedding.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing the seed.
        state : SolveState
            Mutable solve state containing the embedding and SGD parameters.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with optimized embedding in ``state.pos``.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_OptimizeUMAPEmbedding requires state.pos to be set.")

        state.pos = _optimize_embedding(
            embedding=state.pos,
            head=state.extras[_POSITIVE_HEAD_KEY],
            tail=state.extras[_POSITIVE_TAIL_KEY],
            epochs_per_sample=state.extras[_EPOCHS_PER_SAMPLE_KEY],
            n_epochs=state.extras[_N_EPOCHS_KEY],
            learning_rate=state.extras[_LEARNING_RATE_KEY],
            negative_sample_rate=state.extras[_NEGATIVE_SAMPLE_RATE_KEY],
            gamma=state.extras[_GAMMA_KEY],
            a=state.extras[_CURVE_A_KEY],
            b=state.extras[_CURVE_B_KEY],
            seed=problem.seed,
        )
        return state


class _FinalizeUMAPPositions(Op):
    """Apply classic UMAP's final normalization and device transfer."""

    name: ClassVar[str] = "umap_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Center, scale, and cast positions like classic UMAP.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing the optimized embedding.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final output positions on the classic return device.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_FinalizeUMAPPositions requires state.pos to be set.")

        device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        extent = _layout_extent(num_nodes=problem.num_nodes, node_sizes=problem.node_sizes)
        normalized = _normalize_positions(positions=state.pos, extent=extent)
        state.pos = normalized.to(dtype=torch.float32, device=device)
        return state


class _StoreUMAPHyperparameters(Op):
    """Store UMAP hyperparameters into extras for downstream ops."""

    name: ClassVar[str] = "umap_store_hyperparameters"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        spread: float = 1.0,
        n_epochs: Optional[int] = None,
        learning_rate: float = 1.0,
        negative_sample_rate: int = _NEGATIVE_SAMPLE_RATE,
        repulsion_strength: float = 1.0,
    ) -> None:
        """Store UMAP hyperparameters for the pipeline.

        Parameters
        ----------
        n_neighbors : int, default=15
            Neighborhood size for the fuzzy simplicial set.
        min_dist : float, default=0.1
            Target minimum distance in the embedding.
        spread : float, default=1.0
            Target spread for the low-dimensional curve fit.
        n_epochs : int, optional
            Number of SGD epochs. Resolved at apply time based on num_nodes.
        learning_rate : float, default=1.0
            Initial SGD learning rate.
        negative_sample_rate : int, default=5
            Negative samples per positive edge.
        repulsion_strength : float, default=1.0
            Repulsive weight gamma for negative samples.

        Returns
        -------
        None
            The op stores the supplied hyperparameters.
        """
        self._n_neighbors = n_neighbors
        self._min_dist = min_dist
        self._spread = spread
        self._n_epochs = n_epochs
        self._learning_rate = learning_rate
        self._negative_sample_rate = negative_sample_rate
        self._repulsion_strength = repulsion_strength

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Write all UMAP hyperparameters into the solve state extras.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Used to resolve default n_epochs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with UMAP hyperparameters stored in ``extras``.
        """
        del ctx

        n_epochs = self._n_epochs
        if n_epochs is None:
            n_epochs = 500 if problem.num_nodes <= 10_000 else 200

        state.extras[_N_NEIGHBORS_KEY] = self._n_neighbors
        state.extras[_MIN_DIST_KEY] = self._min_dist
        state.extras[_SPREAD_KEY] = self._spread
        state.extras[_N_EPOCHS_KEY] = n_epochs
        state.extras[_LEARNING_RATE_KEY] = self._learning_rate
        state.extras[_NEGATIVE_SAMPLE_RATE_KEY] = self._negative_sample_rate
        state.extras[_GAMMA_KEY] = self._repulsion_strength
        return state


def build_umap_layout_pipeline(
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    n_epochs: Optional[int] = None,
    learning_rate: float = 1.0,
    negative_sample_rate: int = _NEGATIVE_SAMPLE_RATE,
    repulsion_strength: float = 1.0,
) -> Pipeline:
    """Build a UMAP pipeline that is bit-identical to classic ``layout_umap``.

    Parameters
    ----------
    n_neighbors : int, default=15
        Neighborhood size for the fuzzy simplicial set.
    min_dist : float, default=0.1
        Target minimum distance in the embedding.
    spread : float, default=1.0
        Target spread for the low-dimensional curve fit.
    n_epochs : int, optional
        Number of SGD epochs. Defaults to 500 when num_nodes <= 10000
        and 200 otherwise.
    learning_rate : float, default=1.0
        Initial SGD learning rate.
    negative_sample_rate : int, default=5
        Negative samples per positive edge.
    repulsion_strength : float, default=1.0
        Repulsive weight gamma for negative samples.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic UMAP's fuzzy simplicial set
        construction, spectral initialization, cross-entropy SGD
        optimization, and final normalization.
    """
    return Pipeline(
        [
            _StoreUMAPHyperparameters(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                spread=spread,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                negative_sample_rate=negative_sample_rate,
                repulsion_strength=repulsion_strength,
            ),
            _BuildUMAPAdjacency(),
            _ComputeAllPairsShortestPaths(),
            _ExtractKNN(),
            _SmoothKNNDist(),
            _BuildFuzzySimplicialSet(),
            _SpectralInitialization(),
            _FitCurveParameters(),
            _SelectPositiveEdges(),
            _OptimizeUMAPEmbedding(),
            _FinalizeUMAPPositions(),
        ],
        name="umap_layout_pipeline",
    )


def layout_umap_layout_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    n_epochs: Optional[int] = None,
    learning_rate: float = 1.0,
    negative_sample_rate: int = _NEGATIVE_SAMPLE_RATE,
    repulsion_strength: float = 1.0,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the UMAP pipeline as a drop-in replacement for classic ``layout_umap``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used only for output scaling.
    n_neighbors : int, default=15
        Neighborhood size for the fuzzy simplicial set.
    min_dist : float, default=0.1
        Target minimum distance in the embedding.
    spread : float, default=1.0
        Target spread for the low-dimensional curve fit.
    n_epochs : int, optional
        Number of SGD epochs. Defaults to 500 when num_nodes <= 10000
        and 200 otherwise.
    learning_rate : float, default=1.0
        Initial SGD learning rate.
    negative_sample_rate : int, default=5
        Negative samples per positive edge.
    repulsion_strength : float, default=1.0
        Repulsive weight gamma for negative samples.
    seed : int, default=42
        Random seed for spectral noise and negative sampling.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_umap``.

    Raises
    ------
    ValueError
        If inputs are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    _validate_inputs(
        edge_index=edge_index,
        num_nodes=num_nodes,
        n_neighbors=n_neighbors,
        edge_weights=edge_weights,
    )
    if min_dist < 0.0:
        raise ValueError("min_dist must be non-negative.")
    if spread <= 0.0:
        raise ValueError("spread must be positive.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if negative_sample_rate < 0:
        raise ValueError("negative_sample_rate must be non-negative.")
    if repulsion_strength < 0.0:
        raise ValueError("repulsion_strength must be non-negative.")
    if n_epochs is not None and n_epochs < 0:
        raise ValueError("n_epochs must be non-negative.")

    # Early exits matching classic layout_umap exactly.
    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    pipeline = build_umap_layout_pipeline(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        negative_sample_rate=negative_sample_rate,
        repulsion_strength=repulsion_strength,
    )
    final_state = pipeline.apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("UMAP pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_umap_layout_pipeline", "layout_umap_layout_pipeline"]

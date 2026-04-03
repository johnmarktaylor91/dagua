"""MaxEnt-Stress expressed as a composable ops pipeline."""

from __future__ import annotations

import math
from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.classic.maxent_stress import (
    _FULL_STRESS_LIMIT,
    _MAJORIZATION_NODE_LIMIT,
    _PIVOT_COUNT,
    _all_pairs_shortest_paths,
    _average_edge_cost,
    _build_undirected_adjacency,
    _entropy_term,
    _full_non_edge_pairs,
    _full_stress_terms,
    _layout_device,
    _layout_extent,
    _majorization_iteration,
    _normalize_positions,
    _pivot_stress_terms,
    _sample_non_edges,
    _stress_term,
)
from dagua.layout.classic.pivot_mds import layout_pivot_mds
from dagua.layout.ops.base import Op, Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory

# ---------------------------------------------------------------------------
# Shared init op -- PivotMDS initialization used by both branches
# ---------------------------------------------------------------------------


class _InitializeMaxentPositions(Op):
    """Initialize positions from PivotMDS, exactly like the classic code."""

    name: ClassVar[str] = "maxent_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def __init__(self, *, for_majorization: bool = False) -> None:
        """Store the branch flag.

        Parameters
        ----------
        for_majorization : bool, default=False
            When ``True``, cast to float64 on CPU for the majorization
            branch. When ``False``, keep float32 for the gradient branch.
        """
        self.for_majorization = for_majorization

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed positions from PivotMDS exactly like classic maxent-stress.

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
            State with ``pos`` populated from PivotMDS.
        """
        del ctx

        positions = layout_pivot_mds(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            node_sizes=problem.node_sizes,
            n_pivots=min(_PIVOT_COUNT, problem.num_nodes),
            seed=problem.seed,
            edge_weights=problem.edge_weights,
        )

        if self.for_majorization:
            state.pos = positions.to(device="cpu", dtype=torch.float64)
        else:
            device = _layout_device(problem.edge_index, problem.node_sizes)
            state.pos = positions.to(device=device, dtype=torch.float32)

        return state


# ---------------------------------------------------------------------------
# Shared preprocess op -- build adjacency and distances
# ---------------------------------------------------------------------------


class _PrepareMaxentState(Op):
    """Compute adjacency, distances, and stress terms for both branches."""

    name: ClassVar[str] = "maxent_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def __init__(self, *, for_majorization: bool = False, use_entropy: bool = False) -> None:
        """Store branch configuration.

        Parameters
        ----------
        for_majorization : bool, default=False
            Prepare full distance matrix and weight matrix for majorization.
        use_entropy : bool, default=False
            Pre-enumerate non-edge pairs for the entropy term.
        """
        self.for_majorization = for_majorization
        self.use_entropy = use_entropy

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build all precomputed data structures mirroring the classic code.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state receiving maxent extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with maxent-specific data in ``extras``.
        """
        del ctx

        weighted = problem.edge_weights is not None
        average_edge_cost = _average_edge_cost(problem.edge_weights)
        adjacency = _build_undirected_adjacency(
            problem.edge_index,
            problem.num_nodes,
            edge_weights=problem.edge_weights,
        )

        if self.for_majorization:
            graph_distances = _all_pairs_shortest_paths(
                adjacency,
                weighted=weighted,
                average_edge_cost=average_edge_cost,
            ).to(dtype=torch.float64)
            weight_matrix = torch.zeros_like(graph_distances)
            off_diagonal_mask = ~torch.eye(problem.num_nodes, dtype=torch.bool)
            weight_matrix[off_diagonal_mask] = (
                graph_distances[off_diagonal_mask].reciprocal().square()
            )
            state.extras["maxent_graph_distances"] = graph_distances
            state.extras["maxent_weight_matrix"] = weight_matrix
        else:
            # Gradient branch: full or pivot stress terms
            if problem.num_nodes <= _FULL_STRESS_LIMIT:
                stress_src, stress_dst, stress_lengths = _full_stress_terms(
                    adjacency,
                    weighted=weighted,
                    average_edge_cost=average_edge_cost,
                )
                pivot_indices = torch.empty((0,), dtype=torch.long)
                pivot_distances = torch.empty((problem.num_nodes, 0), dtype=torch.float32)
                if self.use_entropy:
                    full_ne_src, full_ne_dst = _full_non_edge_pairs(adjacency)
                else:
                    full_ne_src = torch.empty((0,), dtype=torch.long)
                    full_ne_dst = torch.empty((0,), dtype=torch.long)
            else:
                stress_src = torch.empty((0,), dtype=torch.long)
                stress_dst = torch.empty((0,), dtype=torch.long)
                stress_lengths = torch.empty((0,), dtype=torch.float32)
                pivot_indices, pivot_distances = _pivot_stress_terms(
                    adjacency,
                    problem.seed,
                    weighted=weighted,
                    average_edge_cost=average_edge_cost,
                )
                full_ne_src = torch.empty((0,), dtype=torch.long)
                full_ne_dst = torch.empty((0,), dtype=torch.long)

            state.extras["maxent_stress_src"] = stress_src
            state.extras["maxent_stress_dst"] = stress_dst
            state.extras["maxent_stress_lengths"] = stress_lengths
            state.extras["maxent_pivot_indices"] = pivot_indices
            state.extras["maxent_pivot_distances"] = pivot_distances
            state.extras["maxent_full_ne_src"] = full_ne_src
            state.extras["maxent_full_ne_dst"] = full_ne_dst
            state.extras["maxent_adjacency"] = adjacency

        return state


# ---------------------------------------------------------------------------
# Gradient branch ops
# ---------------------------------------------------------------------------


class _InitializeMaxentOptimizer(Op):
    """Create the Adam optimizer with the classic learning rate schedule."""

    name: ClassVar[str] = "maxent_initialize_optimizer"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Enable gradients on positions and create the Adam optimizer.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with ``pos`` already set.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with gradient-enabled positions and optimizer in extras.
        """
        del ctx

        assert state.pos is not None
        state.pos = state.pos.requires_grad_(True)
        initial_lr = min(0.04, 0.8 / float(max(problem.num_nodes, 1)))
        final_lr = max(
            initial_lr * 0.1,
            initial_lr / math.sqrt(float(max(state.total_steps, 1))),
        )
        state.optimizer = torch.optim.Adam([state.pos], lr=initial_lr)
        state.extras["maxent_initial_lr"] = initial_lr
        state.extras["maxent_final_lr"] = final_lr
        return state


class _MaxentGradientStep(Op):
    """One Adam update step of the maxent-stress gradient objective."""

    name: ClassVar[str] = "maxent_gradient_step"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def __init__(self, *, alpha: float = 1.0, use_entropy: bool = False) -> None:
        """Store loss parameters.

        Parameters
        ----------
        alpha : float, default=1.0
            Entropy repulsion weight.
        use_entropy : bool, default=False
            Whether to include the logarithmic entropy repulsion term.
        """
        self.alpha = alpha
        self.use_entropy = use_entropy

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Execute one optimizer step exactly like the classic gradient loop.

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
            State after one gradient update.
        """
        del ctx

        assert state.pos is not None
        assert state.optimizer is not None

        positions = state.pos
        optimizer = state.optimizer
        step = state.step

        stress_src = state.extras["maxent_stress_src"]
        stress_dst = state.extras["maxent_stress_dst"]
        stress_lengths = state.extras["maxent_stress_lengths"]
        pivot_indices = state.extras["maxent_pivot_indices"]
        pivot_distances = state.extras["maxent_pivot_distances"]
        adjacency = state.extras["maxent_adjacency"]

        optimizer.zero_grad(set_to_none=True)

        if problem.num_nodes <= _FULL_STRESS_LIMIT:
            full_ne_src = state.extras["maxent_full_ne_src"]
            full_ne_dst = state.extras["maxent_full_ne_dst"]

            loss = _stress_term(
                positions,
                stress_src,
                stress_dst,
                stress_lengths,
                pivot_indices,
                pivot_distances,
            )
            if self.use_entropy:
                loss = loss + self.alpha * _entropy_term(
                    positions,
                    full_ne_src.to(device=positions.device),
                    full_ne_dst.to(device=positions.device),
                    scale=1.0,
                )
        else:
            loss = _stress_term(
                positions,
                stress_src,
                stress_dst,
                stress_lengths,
                pivot_indices,
                pivot_distances,
            )
            if self.use_entropy:
                sampled_src, sampled_dst, total_non_edges = _sample_non_edges(
                    adjacency,
                    step,
                    problem.seed,
                    positions.device,
                )
                sample_count = max(int(sampled_src.numel()), 1)
                loss = loss + self.alpha * _entropy_term(
                    positions,
                    sampled_src,
                    sampled_dst,
                    scale=(
                        float(total_non_edges) / float(sample_count) if total_non_edges > 0 else 1.0
                    ),
                )

        loss.backward()
        optimizer.step()

        # LR decay exactly like classic
        initial_lr = state.extras["maxent_initial_lr"]
        final_lr = state.extras["maxent_final_lr"]
        fraction = float(step + 1) / float(max(state.total_steps, 1))
        optimizer.param_groups[0]["lr"] = initial_lr + (final_lr - initial_lr) * fraction

        return state


# ---------------------------------------------------------------------------
# Majorization branch ops
# ---------------------------------------------------------------------------


class _MajorizationStep(Op):
    """One in-place Gauss-Seidel stress-majorization iteration."""

    name: ClassVar[str] = "maxent_majorization_step"
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
        """Run one majorization iteration on CPU float64 positions.

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
            State after one majorization update.
        """
        del ctx

        assert state.pos is not None

        _majorization_iteration(
            positions=state.pos,
            graph_distances=state.extras["maxent_graph_distances"],
            weight_matrix=state.extras["maxent_weight_matrix"],
        )

        return state


# ---------------------------------------------------------------------------
# Finalize ops
# ---------------------------------------------------------------------------


class _FinalizeMaxentPositions(Op):
    """Apply classic maxent-stress final normalization and type cast."""

    name: ClassVar[str] = "maxent_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def __init__(self, *, for_majorization: bool = False) -> None:
        """Store the branch flag.

        Parameters
        ----------
        for_majorization : bool, default=False
            When ``True``, cast from float64 to float32 like the majorization
            branch.
        """
        self.for_majorization = for_majorization

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Normalize and cast positions to match classic output exactly.

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
            State with final output positions.
        """
        del ctx

        assert state.pos is not None

        device = _layout_device(problem.edge_index, problem.node_sizes)
        extent = _layout_extent(problem.num_nodes, problem.node_sizes)

        if self.for_majorization:
            state.pos = _normalize_positions(state.pos.to(dtype=torch.float32), extent).to(
                dtype=torch.float32, device=device
            )
        else:
            state.pos = _normalize_positions(state.pos.detach(), extent).to(
                dtype=torch.float32, device=device
            )

        return state


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def build_maxent_stress_majorization_pipeline(steps: int = 200) -> Pipeline:
    """Build a pipeline for the majorization branch of maxent-stress.

    This pipeline matches the classic ``_layout_stress_majorization`` path
    bit-for-bit.

    Parameters
    ----------
    steps : int, default=200
        Number of majorization iterations.

    Returns
    -------
    Pipeline
        Composable pipeline that reproduces majorization-based stress layout.
    """
    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=steps)),
            _InitializeMaxentPositions(for_majorization=True),
            _PrepareMaxentState(for_majorization=True),
            Repeat(
                n=steps,
                ops=[_MajorizationStep()],
            ),
            _FinalizeMaxentPositions(for_majorization=True),
        ],
        name="maxent_stress_majorization_pipeline",
    )


def build_maxent_stress_gradient_pipeline(
    steps: int = 200,
    alpha: float = 1.0,
    use_entropy: bool = False,
) -> Pipeline:
    """Build a pipeline for the gradient branch of maxent-stress.

    This pipeline matches the classic ``_layout_maxent_stress_gradient``
    path bit-for-bit.

    Parameters
    ----------
    steps : int, default=200
        Number of Adam optimization steps.
    alpha : float, default=1.0
        Entropy repulsion weight.
    use_entropy : bool, default=False
        Whether to include the logarithmic entropy repulsion term.

    Returns
    -------
    Pipeline
        Composable pipeline that reproduces gradient-based maxent-stress layout.
    """
    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=steps)),
            _InitializeMaxentPositions(for_majorization=False),
            _PrepareMaxentState(for_majorization=False, use_entropy=use_entropy),
            _InitializeMaxentOptimizer(),
            Repeat(
                n=steps,
                ops=[_MaxentGradientStep(alpha=alpha, use_entropy=use_entropy)],
            ),
            _FinalizeMaxentPositions(for_majorization=False),
        ],
        name="maxent_stress_gradient_pipeline",
    )


def build_maxent_stress_pipeline(
    steps: int = 200,
    alpha: float = 1.0,
    use_entropy: bool = False,
    use_majorization: bool = True,
    num_nodes: int = 0,
) -> Pipeline:
    """Build the correct maxent-stress pipeline matching classic dispatch.

    The classic ``layout_maxent_stress`` chooses majorization when all of:
    - ``use_majorization`` is ``True``
    - ``use_entropy`` is ``False``
    - ``num_nodes <= 5000``
    - ``steps == 200``

    Otherwise it uses the gradient path.

    Parameters
    ----------
    steps : int, default=200
        Number of optimization iterations.
    alpha : float, default=1.0
        Entropy repulsion weight.
    use_entropy : bool, default=False
        Whether to include the entropy repulsion term.
    use_majorization : bool, default=True
        Whether to prefer majorization for eligible problems.
    num_nodes : int, default=0
        Node count, used to choose the dispatch branch.

    Returns
    -------
    Pipeline
        The appropriate composable pipeline.
    """
    if (
        use_majorization
        and not use_entropy
        and num_nodes <= _MAJORIZATION_NODE_LIMIT
        and steps == 200
    ):
        return build_maxent_stress_majorization_pipeline(steps=steps)
    return build_maxent_stress_gradient_pipeline(
        steps=steps,
        alpha=alpha,
        use_entropy=use_entropy,
    )


def layout_maxent_stress_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 200,
    alpha: float = 1.0,
    seed: int = 42,
    use_entropy: bool = False,
    use_majorization: bool = True,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the maxent-stress pipeline as a drop-in replacement for the classic.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes.
    steps : int, default=200
        Number of optimization iterations.
    alpha : float, default=1.0
        Entropy repulsion weight.
    seed : int, default=42
        Random seed.
    use_entropy : bool, default=False
        Include the entropy repulsion term.
    use_majorization : bool, default=True
        Prefer majorization for eligible problems.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``alpha``, or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to produce final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    device = _layout_device(edge_index, node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    pipeline = build_maxent_stress_pipeline(
        steps=steps,
        alpha=alpha,
        use_entropy=use_entropy,
        use_majorization=use_majorization,
        num_nodes=num_nodes,
    )

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = pipeline.apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("MaxEnt-stress pipeline did not produce final positions.")
    return final_state.pos


__all__ = [
    "build_maxent_stress_gradient_pipeline",
    "build_maxent_stress_majorization_pipeline",
    "build_maxent_stress_pipeline",
    "layout_maxent_stress_pipeline",
]

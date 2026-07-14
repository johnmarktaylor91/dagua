"""MulMent multilevel MaxEnt-Stress layout pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.coarsen import HeavyEdgeMatching
from dagua.layout.ops.graph_utils import layout_device
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_DEFAULT_ALPHA = 1.0
_DEFAULT_MIN_ALPHA = 0.008
_DEFAULT_Q = 0.0
_DEFAULT_TOL = 1.0e-4
_DEFAULT_COARSEST_SIZE = 32
_DEFAULT_MAX_LEVELS = 20
_ZERO_DISTANCE_EPSILON = 1.0e-9
_JITTER_SCALE = 1.0e-3


@dataclass(frozen=True)
class MulMentConfig:
    """Configuration for KaDraw-style MulMent layout.

    Parameters
    ----------
    steps : int
        Total refinement budget. The value is split across coarsest and
        uncoarsening levels.
    alpha : float
        Initial MaxEnt repulsion weight.
    min_alpha : float
        Lower bound for the outer-loop alpha schedule.
    q : float
        KaDraw MaxEnt exponent. ``0`` gives inverse-square entropy forces.
    tol : float
        Relative coordinate-change tolerance for each refinement stage.
    coarsest_size : int
        Stop coarsening when the graph reaches this size.
    max_levels : int
        Maximum number of coarsening transitions.
    fidelity_dtype : torch.dtype
        Floating-point dtype used by the local optimizer.
    """

    steps: int = 200
    alpha: float = _DEFAULT_ALPHA
    min_alpha: float = _DEFAULT_MIN_ALPHA
    q: float = _DEFAULT_Q
    tol: float = _DEFAULT_TOL
    coarsest_size: int = _DEFAULT_COARSEST_SIZE
    max_levels: int = _DEFAULT_MAX_LEVELS
    fidelity_dtype: torch.dtype = torch.float32


def _validate_config(config: MulMentConfig) -> None:
    """Validate MulMent configuration values.

    Parameters
    ----------
    config : MulMentConfig
        Configuration to validate.

    Returns
    -------
    None
        The function raises on invalid input.
    """
    if config.steps < 0:
        raise ValueError("steps must be non-negative.")
    if config.alpha < 0.0:
        raise ValueError("alpha must be non-negative.")
    if config.min_alpha < 0.0:
        raise ValueError("min_alpha must be non-negative.")
    if config.tol <= 0.0:
        raise ValueError("tol must be positive.")
    if config.coarsest_size < 1:
        raise ValueError("coarsest_size must be positive.")
    if config.max_levels < 0:
        raise ValueError("max_levels must be non-negative.")
    if config.fidelity_dtype not in (torch.float32, torch.float64):
        raise ValueError("fidelity_dtype must be torch.float32 or torch.float64.")


def _undirected_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    dtype: torch.dtype,
    edge_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unique undirected edges and desired lengths.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    dtype : torch.dtype
        Floating-point dtype for desired lengths.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``. KaDraw's unit-distance graph
        format is modeled by interpreting weights as desired lengths when
        explicitly supplied.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Unique edge tensor with shape ``[2, U]`` and lengths with shape ``[U]``.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_weights length must match edge count.")

    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    weights_cpu = (
        torch.ones((edge_index_cpu.shape[1],), dtype=dtype)
        if edge_weights is None
        else edge_weights.detach().to(device="cpu", dtype=dtype)
    )
    seen: dict[tuple[int, int], list[float]] = {}
    for edge_id, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
            raise ValueError("edge_index contains a node outside [0, num_nodes).")
        if source == target:
            continue
        key = (min(source, target), max(source, target))
        seen.setdefault(key, []).append(float(weights_cpu[edge_id].item()))

    if not seen:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=dtype),
        )

    pairs = sorted(seen)
    lengths = [sum(seen[pair]) / len(seen[pair]) for pair in pairs]
    return (
        torch.tensor(pairs, dtype=torch.long).transpose(0, 1).contiguous(),
        torch.tensor(lengths, dtype=dtype),
    )


def _seeded_initial_positions(num_nodes: int, seed: int, dtype: torch.dtype) -> torch.Tensor:
    """Create deterministic KaDraw-compatible starting coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    seed : int
        Random seed.
    dtype : torch.dtype
        Output dtype.

    Returns
    -------
    torch.Tensor
        Initial coordinates with shape ``[N, 2]``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn((num_nodes, 2), generator=generator, dtype=dtype)


def _run_local_maxent(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_lengths: torch.Tensor,
    config: MulMentConfig,
    outer_iterations: int,
) -> torch.Tensor:
    """Run KaDraw's fixed-point MaxEnt local optimizer.

    Parameters
    ----------
    positions : torch.Tensor
        Initial positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Unique undirected edge tensor with shape ``[2, E]``.
    edge_lengths : torch.Tensor
        Desired edge lengths with shape ``[E]``.
    config : MulMentConfig
        Optimizer parameters.
    outer_iterations : int
        Number of alpha-schedule outer iterations.

    Returns
    -------
    torch.Tensor
        Refined positions with shape ``[N, 2]``.
    """
    num_nodes = int(positions.shape[0])
    if num_nodes <= 1 or edge_index.numel() == 0 or outer_iterations <= 0:
        return positions

    src = edge_index[0].to(dtype=torch.long)
    dst = edge_index[1].to(dtype=torch.long)
    lengths = edge_lengths.to(dtype=positions.dtype).clamp_min(_ZERO_DISTANCE_EPSILON)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for source, target, length in zip(src.tolist(), dst.tolist(), lengths.tolist()):
        adjacency[source].append((target, float(length)))
        adjacency[target].append((source, float(length)))

    pos = positions.clone()
    alpha = float(config.alpha)
    inner_iterations = max(1, int(math.ceil(float(config.steps) / max(1, outer_iterations))))
    q = float(config.q)
    sign_q = 0.0 if q == 0.0 else (-1.0 if q < 0.0 else 1.0)
    all_indices = torch.arange(num_nodes, dtype=torch.long)

    for _ in range(outer_iterations):
        for _inner in range(inner_iterations):
            new_pos = pos.clone()
            for node, neighbors in enumerate(adjacency):
                if not neighbors:
                    continue
                neighbor_ids = torch.tensor([item[0] for item in neighbors], dtype=torch.long)
                neighbor_lengths = torch.tensor(
                    [item[1] for item in neighbors],
                    dtype=pos.dtype,
                )
                rho = torch.reciprocal(torch.sum(torch.reciprocal(neighbor_lengths.square())))
                diff = pos[node].unsqueeze(0) - pos[neighbor_ids]
                dist = torch.linalg.norm(diff, dim=1).clamp_min(_ZERO_DISTANCE_EPSILON)
                scaled = neighbor_lengths / dist
                stress_center = (
                    torch.sum(
                        (pos[neighbor_ids] + scaled.unsqueeze(1) * diff)
                        / neighbor_lengths.square().unsqueeze(1),
                        dim=0,
                    )
                    * rho
                )

                other_mask = all_indices != node
                other = all_indices[other_mask]
                other_diff = pos[node].unsqueeze(0) - pos[other]
                other_dist = torch.linalg.norm(other_diff, dim=1).clamp_min(_ZERO_DISTANCE_EPSILON)
                repulsion = torch.sum(other_diff / other_dist.pow(q + 2.0).unsqueeze(1), dim=0)
                edge_repulsion = torch.sum(diff / dist.pow(q + 2.0).unsqueeze(1), dim=0)
                entropy = (repulsion - edge_repulsion) * (alpha * float(rho.item()))
                new_pos[node] = stress_center + sign_q * entropy

            norm_coords = float(torch.sum(pos.square()).item())
            norm_diff = float(torch.sum((pos - new_pos).square()).item())
            pos = new_pos
            if norm_coords > 0.0 and norm_diff / norm_coords < config.tol:
                return pos
        alpha = max(0.3 * alpha, float(config.min_alpha))
    return pos


@register_op
class MulMentCoarsenAndRefine(Op):
    """Build a hierarchy, optimize the coarsest graph, and unroll levels.

    Parameters
    ----------
    config : MulMentConfig
        MulMent optimizer configuration.
    """

    name = "mulment_coarsen_refine"
    category = OpCategory.COARSEN
    writes = ("pos", "hierarchy")

    def __init__(self, config: MulMentConfig) -> None:
        """Store the MulMent configuration.

        Parameters
        ----------
        config : MulMentConfig
            Validated MulMent configuration.

        Returns
        -------
        None
            The operation stores the configuration.
        """
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply multilevel MaxEnt-Stress layout.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context carrying execution options.

        Returns
        -------
        SolveState
            State with final positions in ``state.pos``.
        """
        HeavyEdgeMatching().apply(problem, state, ctx)
        hierarchy = list(state.hierarchy or [])[: self.config.max_levels]
        while hierarchy and hierarchy[-1].num_nodes < self.config.coarsest_size:
            hierarchy.pop()

        levels = hierarchy
        if not levels:
            edge_index, lengths = _undirected_edges(
                problem.edge_index,
                problem.num_nodes,
                self.config.fidelity_dtype,
                problem.edge_weights,
            )
            state.pos = _run_local_maxent(
                _seeded_initial_positions(
                    problem.num_nodes,
                    problem.seed,
                    self.config.fidelity_dtype,
                ),
                edge_index,
                lengths,
                self.config,
                outer_iterations=max(1, min(self.config.steps, 8)),
            )
            return state

        coarsest = levels[-1]
        assert coarsest.edge_index is not None
        assert coarsest.edge_weights is not None
        pos = _seeded_initial_positions(
            coarsest.num_nodes,
            problem.seed,
            self.config.fidelity_dtype,
        )
        edge_index, lengths = _undirected_edges(
            coarsest.edge_index,
            coarsest.num_nodes,
            self.config.fidelity_dtype,
            coarsest.edge_weights,
        )
        pos = _run_local_maxent(pos, edge_index, lengths, self.config, outer_iterations=4)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(problem.seed) + 1)
        for level_index in range(len(levels) - 1, -1, -1):
            level = levels[level_index]
            assert level.fine_to_coarse is not None
            mapping = level.fine_to_coarse.to(dtype=torch.long)
            jitter = _JITTER_SCALE * torch.randn(
                (level.num_fine, 2),
                generator=generator,
                dtype=self.config.fidelity_dtype,
            )
            pos = pos[mapping] + jitter
            if level_index == 0:
                fine_edge_index = problem.edge_index
                fine_edge_weights = problem.edge_weights
                fine_nodes = problem.num_nodes
            else:
                previous = levels[level_index - 1]
                fine_edge_index = previous.edge_index
                fine_edge_weights = previous.edge_weights
                fine_nodes = previous.num_nodes
            assert fine_edge_index is not None
            edge_index, lengths = _undirected_edges(
                fine_edge_index,
                fine_nodes,
                self.config.fidelity_dtype,
                fine_edge_weights,
            )
            pos = _run_local_maxent(pos, edge_index, lengths, self.config, outer_iterations=2)

        state.pos = pos
        return state


def build_mulment_pipeline(config: Optional[MulMentConfig] = None) -> Pipeline:
    """Build the MulMent pipeline.

    Parameters
    ----------
    config : MulMentConfig, optional
        Pipeline configuration.

    Returns
    -------
    Pipeline
        Pipeline containing the multilevel coarsen/refine operation.
    """
    resolved = config or MulMentConfig()
    _validate_config(resolved)
    return Pipeline([MulMentCoarsenAndRefine(resolved)], name="mulment_pipeline")


def layout_mulment_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 200,
    seed: int = 42,
    alpha: float = _DEFAULT_ALPHA,
    min_alpha: float = _DEFAULT_MIN_ALPHA,
    q: float = _DEFAULT_Q,
    tol: float = _DEFAULT_TOL,
    coarsest_size: int = _DEFAULT_COARSEST_SIZE,
    max_levels: int = _DEFAULT_MAX_LEVELS,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the MulMent layout pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[N, 2]``.
    steps : int, default=200
        Refinement budget.
    seed : int, default=42
        Seed for coarsening and projected jitter.
    alpha : float, default=1.0
        Initial MaxEnt alpha.
    min_alpha : float, default=0.008
        Minimum alpha.
    q : float, default=0.0
        Entropy exponent.
    tol : float, default=1e-4
        Relative convergence tolerance.
    coarsest_size : int, default=32
        Coarsening target.
    max_levels : int, default=20
        Maximum hierarchy depth.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    fidelity_dtype : torch.dtype, optional
        Internal dtype.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    resolved_dtype = torch.float32 if fidelity_dtype is None else fidelity_dtype
    device = layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=resolved_dtype, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=resolved_dtype, device=device)

    config = MulMentConfig(
        steps=steps,
        alpha=alpha,
        min_alpha=min_alpha,
        q=q,
        tol=tol,
        coarsest_size=coarsest_size,
        max_levels=max_levels,
        fidelity_dtype=resolved_dtype,
    )
    _validate_config(config)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    final_state = build_mulment_pipeline(config).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    if final_state.pos is None:
        raise RuntimeError("MulMent pipeline did not produce final positions.")
    return final_state.pos.to(device=device)


__all__ = ["MulMentConfig", "build_mulment_pipeline", "layout_mulment_pipeline"]

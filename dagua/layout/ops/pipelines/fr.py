"""Fruchterman-Reingold force-directed layout pipeline."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from dagua.layout.ops.anneal import InitTemperatureFromExtent, LinearCool
from dagua.layout.ops.base import Conditional, Pipeline, Repeat  # noqa: E402
from dagua.layout.ops.converge import (
    FixedSteps,
    FixedStepsConfig,
    FRConvergenceCheck,
)  # noqa: E402
from dagua.layout.ops.force import ApplyDisplacement, ApplyDisplacementConfig, FRCombinedForce
from dagua.layout.ops.init import RandomUniformInit, RandomUniformInitConfig
from dagua.layout.ops.postprocess import FRFinalizePositions, FRFinalizePositionsConfig
from dagua.layout.ops.preprocess import FRPrepareAdjacency, FRPrepareAdjacencyConfig
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)

_LEGACY_CLASSIC_FR_STEPS = 200
_CANONICAL_NX_SPRING_STEPS = 50
_FR_DAG_DROP_TOLERANCE = 0.1
_FR_SCORE_DROP_TOLERANCE = 1.0e-6


def _dag_consistency_fraction(pos: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Compute the TB directed-edge consistency fraction.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.

    Returns
    -------
    float
        Fraction of edges whose target is not above their source.
    """
    if edge_index.numel() == 0:
        return 1.0
    source = edge_index[0].to(device=pos.device)
    target = edge_index[1].to(device=pos.device)
    self_loops = source == target
    correct = (pos[target, 1] >= pos[source, 1]) | self_loops
    return float(correct.to(dtype=torch.float32).mean().item())


def _quick_directed_composite_score(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> float:
    """Compute the cheap directed composite used by the FR default selector.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` for overlap scoring.

    Returns
    -------
    float
        Directed composite score from Tier-1 metrics only.
    """
    from dagua.metrics import composite, quick

    return float(composite(quick(pos, edge_index, node_sizes=node_sizes, seed=0)))


def _choose_fr_default_layout(
    legacy_pos: torch.Tensor,
    canonical_pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
    dag_drop_tolerance: float = _FR_DAG_DROP_TOLERANCE,
    score_drop_tolerance: float = _FR_SCORE_DROP_TOLERANCE,
) -> torch.Tensor:
    """Choose between legacy 200-step FR and canonical NetworkX-style FR.

    Parameters
    ----------
    legacy_pos : torch.Tensor
        Existing dagua ``classic_fr`` default output with shape ``[N, 2]``.
    canonical_pos : torch.Tensor
        NetworkX-compatible 50-step FR output with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` for overlap scoring.
    dag_drop_tolerance : float, default=0.1
        Maximum allowed drop in TB edge consistency before preserving the
        legacy layout.
    score_drop_tolerance : float, default=1.0e-6
        Maximum allowed Tier-1 composite drop before preserving the legacy
        layout.

    Returns
    -------
    torch.Tensor
        Selected position tensor with shape ``[N, 2]``.
    """
    legacy_dag = _dag_consistency_fraction(legacy_pos, edge_index)
    canonical_dag = _dag_consistency_fraction(canonical_pos, edge_index)
    if canonical_dag + dag_drop_tolerance < legacy_dag:
        return legacy_pos

    legacy_score = _quick_directed_composite_score(legacy_pos, edge_index, node_sizes)
    canonical_score = _quick_directed_composite_score(canonical_pos, edge_index, node_sizes)
    if canonical_score + score_drop_tolerance < legacy_score:
        return legacy_pos
    return canonical_pos


def _normalize_fixed_indices(
    fixed: Optional[Union[Sequence[int], torch.Tensor]],
    num_nodes: int,
) -> tuple[int, ...]:
    """Validate and normalize fixed-node indices.

    Parameters
    ----------
    fixed : sequence of int or torch.Tensor, optional
        Node indices whose FR displacement should be zeroed.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    tuple[int, ...]
        Sorted unique fixed-node indices.

    Raises
    ------
    ValueError
        If any fixed index is outside ``[0, num_nodes)``.
    """
    if fixed is None:
        return ()
    if isinstance(fixed, torch.Tensor):
        raw_indices = fixed.detach().to(device="cpu", dtype=torch.long).flatten().tolist()
    else:
        raw_indices = [int(index) for index in fixed]
    normalized = tuple(sorted(set(int(index) for index in raw_indices)))
    if any(index < 0 or index >= num_nodes for index in normalized):
        raise ValueError("fixed contains a node index outside [0, num_nodes).")
    return normalized


def build_fr_pipeline(
    steps: int = 50,
    networkx_compat: bool = False,
    k: Optional[float] = None,
    fixed_indices: Optional[Sequence[int]] = None,
) -> Pipeline:
    """Build a Fruchterman-Reingold force-directed layout pipeline.

    Reference fidelity
    ------------------
    Targets: NetworkX 3.6.1 ``spring_layout`` / Fruchterman and Reingold
        (1991), "Graph Drawing by Force-directed Placement".
    Fidelity mode: ``networkx_compat=True`` switches final scaling to the
        NetworkX adapter contract; fixed nodes additionally suppress rescaling.
    Verified at: final 100-seed report, strong equivalent; median RMSD 0.088
        to 0.136 across step-count variants.
    Known divergences:
        - Default dagua display scaling remains larger than NetworkX output.
        - Directed graph selection still happens in the wrapper, not this
          builder.

    Parameters
    ----------
    steps : int, default=50
        Maximum number of cooling iterations to run.
    networkx_compat : bool, default=False
        If ``True``, use NetworkX adapter-scale finalization instead of
        dagua's legacy ``50 * sqrt(N)`` display scale.
    k : float, optional
        Explicit NetworkX-style optimal node spacing.
    fixed_indices : sequence of int, optional
        Node indices whose displacement should be zeroed. When provided, final
        centering/scaling is skipped to match NetworkX fixed-node semantics.

    Returns
    -------
    Pipeline
        Pipeline implementing the classical Fruchterman-Reingold algorithm.
        The pipeline produces final node coordinates by sampling unit-square
        initial positions, building adjacency data, setting an initial
        temperature from the current extent, iterating attraction and
        repulsion force updates with displacement and linear cooling, then
        finalizing the coordinates.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=steps)),
            Conditional(
                predicate=lambda problem, state, ctx: state.pos is None,
                op=RandomUniformInit(
                    RandomUniformInitConfig(
                        scale="none",
                        rng_backend="numpy",
                    ),
                ),
            ),
            FRPrepareAdjacency(FRPrepareAdjacencyConfig(k=k)),
            InitTemperatureFromExtent(),
            Repeat(
                n=steps,
                ops=[
                    FRCombinedForce(),
                    ApplyDisplacement(
                        ApplyDisplacementConfig(
                            fixed_indices=tuple(fixed_indices or ()),
                        ),
                    ),
                    FRConvergenceCheck(),
                    LinearCool(),
                ],
            ),
            FRFinalizePositions(
                FRFinalizePositionsConfig(
                    output_scale_factor=500.0 if networkx_compat else 50.0,
                    scale_by_sqrt_num_nodes=not networkx_compat,
                    skip_rescale=bool(fixed_indices),
                ),
            ),
        ],
        name="fr_pipeline",
    )


def layout_fr_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 50,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    pos: Optional[torch.Tensor] = None,
    networkx_compat: bool = False,
    k: Optional[float] = None,
    fixed: Optional[Union[Sequence[int], torch.Tensor]] = None,
) -> torch.Tensor:
    """Run the Fruchterman-Reingold pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only to resolve
        the output device.
    steps : int, default=50
        Maximum number of cooling iterations to run.
    seed : int, default=42
        Random seed for the NumPy-backed unit-square initialization.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    pos : torch.Tensor, optional
        Initial positions with shape ``[N, 2]``. When provided, the pipeline
        starts from these coordinates instead of sampling a random
        initialization.
    networkx_compat : bool, default=False
        If ``True``, use NetworkX-compatible adapter-scale finalization. This
        preserves the force loop while avoiding dagua's legacy display scale.
    k : float, optional
        Explicit NetworkX-style optimal node spacing. ``None`` preserves
        ``sqrt(1 / num_nodes)``.
    fixed : sequence of int or torch.Tensor, optional
        Node indices to hold fixed during displacement. A full ``pos`` tensor
        must also be provided, matching NetworkX's fixed-node requirement.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``edge_weights``, or ``pos`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )
    if pos is not None and pos.shape != (num_nodes, 2):
        raise ValueError(f"pos must have shape ({num_nodes}, 2), got {tuple(pos.shape)}")
    if k is not None and k <= 0.0:
        raise ValueError("k must be positive when provided.")
    fixed_indices = _normalize_fixed_indices(fixed=fixed, num_nodes=num_nodes)
    if fixed_indices and pos is None:
        raise ValueError("fixed nodes require a full pos tensor.")

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    if pos is not None:
        state.pos = pos.detach().clone().to(dtype=torch.float64)
    output_device = (
        edge_index.device
        if edge_index.numel() > 0
        else node_sizes.device
        if node_sizes is not None
        else torch.device("cpu")
    )
    if pos is not None and steps == 0:
        return state.pos.to(device=output_device, dtype=torch.float32)
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(output_device)))
    final_state = build_fr_pipeline(
        steps=steps,
        networkx_compat=networkx_compat,
        k=k,
        fixed_indices=fixed_indices,
    ).apply(
        problem,
        state,
        ctx,
    )
    if final_state.pos is None:
        raise RuntimeError("FR pipeline did not produce final positions.")
    return final_state.pos


def layout_fr_default_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = _LEGACY_CLASSIC_FR_STEPS,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    pos: Optional[torch.Tensor] = None,
    networkx_compat: bool = False,
    k: Optional[float] = None,
    fixed: Optional[Union[Sequence[int], torch.Tensor]] = None,
) -> torch.Tensor:
    """Run the benchmark default FR layout with canonical-fidelity selection.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` for selector scoring.
    steps : int, default=200
        Requested FR iteration count. Non-default values run exactly as
        requested and bypass the selector.
    seed : int, default=42
        Random seed for both default candidates.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    pos : torch.Tensor, optional
        Initial positions with shape ``[N, 2]``. Warm starts run exactly as
        requested and bypass the selector.
    networkx_compat : bool, default=False
        If ``True``, forwarded to :func:`layout_fr_pipeline` for exact
        NetworkX adapter-style output scaling.
    k : float, optional
        Explicit NetworkX-style optimal node spacing.
    fixed : sequence of int or torch.Tensor, optional
        Node indices to hold fixed during displacement.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.
    """
    if steps != _LEGACY_CLASSIC_FR_STEPS or pos is not None or k is not None or fixed is not None:
        return layout_fr_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            steps=steps,
            seed=seed,
            edge_weights=edge_weights,
            pos=pos,
            networkx_compat=networkx_compat,
            k=k,
            fixed=fixed,
        )

    legacy_pos = layout_fr_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        steps=_LEGACY_CLASSIC_FR_STEPS,
        seed=seed,
        edge_weights=edge_weights,
        networkx_compat=networkx_compat,
        k=k,
        fixed=fixed,
    )
    canonical_pos = layout_fr_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        steps=_CANONICAL_NX_SPRING_STEPS,
        seed=seed,
        edge_weights=edge_weights,
        networkx_compat=networkx_compat,
        k=k,
        fixed=fixed,
    )
    return _choose_fr_default_layout(
        legacy_pos=legacy_pos,
        canonical_pos=canonical_pos,
        edge_index=edge_index,
        node_sizes=node_sizes,
    )


__all__ = [
    "build_fr_pipeline",
    "layout_fr_default_pipeline",
    "layout_fr_pipeline",
]

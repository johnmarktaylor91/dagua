"""FM^3 multilevel force-directed layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.fmmm import (
    _FinalizeFMMMPositions,
    _InitializeCoarsestLevel,
    _InitializeFMMMState,
    _InitializeFMMMStateConfig,
    _RefineCoarsestLevel,
    _SingleLevelFallback,
    _UncoarsenLoop,
)
from dagua.layout.ops.graph_utils import layout_device as _layout_device
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


def build_fmmm_pipeline(
    steps: int = 200,
    force_model: str = "ogdf_new",
    reference_mode: bool = False,
    fidelity_mode: bool = False,
) -> Pipeline:
    """Build an FM^3 multilevel force-directed pipeline.

    Reference fidelity
    ------------------
    Targets: Graphviz 7.0.5 fdp / Hachul and Junger (2004), "Drawing Large
        Graphs with a Potential-Field-Based Multilevel Algorithm".
    Fidelity mode: ``reference_mode=True`` or ``fidelity_mode=True`` enables
        OGDF/Graphviz-aligned coarsening, coarsest initialization, and force
        scaling choices used by evaluation competitors.
    Verified at: final 100-seed report, strong equivalent; median RMSD 0.067
        to 0.179 across step-count variants. Round 33 fdp bounded subset
        remained 0.121966.
    Known divergences:
        - Graphviz fdp derived-graph recursion, cluster expansion, and overlap
          removal remain unported.
        - Dagua keeps a fallback single-level solve when multilevel setup is
          unsuitable.

    Parameters
    ----------
    steps : int, default=200
        Total refinement budget distributed across hierarchy levels.
    force_model : str, default="ogdf_new"
        Spring-force model for edge attraction. ``"ogdf_new"`` matches
        OGDF's default; ``"fr"`` preserves Dagua's earlier coefficient for
        benchmark fallback selection.
    reference_mode : bool, default=False
        Use OGDF-aligned coarsening, coarsest initialization, and force
        scaling choices for fidelity comparisons.
    fidelity_mode : bool, default=False
        Alias for ``reference_mode`` used by evaluation competitors.

    Returns
    -------
    Pipeline
        Pipeline implementing the FM^3 algorithm. The pipeline produces final
        node coordinates by constructing a multilevel hierarchy, initializing
        the coarsest graph, refining that level, uncoarsening with per-level
        refinement, falling back to a single-level solve when needed, and
        normalizing the result.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    effective_reference_mode = reference_mode or fidelity_mode
    initialize_state = _InitializeFMMMState(
        config=_InitializeFMMMStateConfig(
            steps=steps,
            force_model=force_model,
            galaxy_choice="lower" if effective_reference_mode else "higher",
            coarsest_init="ogdf_random" if effective_reference_mode else "fr",
            ogdf_force_scaling=effective_reference_mode,
            sum_parallel_weights=not effective_reference_mode,
        )
    )
    initialize_coarsest = _InitializeCoarsestLevel()
    refine_coarsest = _RefineCoarsestLevel()
    uncoarsen_loop = _UncoarsenLoop()
    single_level_fallback = _SingleLevelFallback()
    finalize_positions = _FinalizeFMMMPositions()

    return Pipeline(
        [
            initialize_state,
            initialize_coarsest,
            refine_coarsest,
            uncoarsen_loop,
            single_level_fallback,
            finalize_positions,
        ],
        name="fmmm_pipeline",
    )


def layout_fmmm_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 200,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    force_model: str = "ogdf_new",
    reference_mode: bool = False,
    fidelity_mode: bool = False,
) -> torch.Tensor:
    """Run the FM^3 pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used for extent
        calculation and output-device selection.
    steps : int, default=200
        Total refinement budget distributed across hierarchy levels.
    seed : int, default=42
        Random seed for coarsening, coarse initialization, and prolongation
        jitter.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    force_model : str, default="ogdf_new"
        Spring-force model for edge attraction.
    reference_mode : bool, default=False
        Use OGDF-aligned reference behavior for algorithm fidelity runs.
    fidelity_mode : bool, default=False
        Alias for ``reference_mode`` used by evaluation competitors.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``edge_weights``, or ``force_model`` are
        invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if force_model not in {"ogdf_new", "fr"}:
        raise ValueError("force_model must be either 'ogdf_new' or 'fr'.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge_count {edge_index.shape[1]}"
            )

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
    final_state = build_fmmm_pipeline(
        steps=steps,
        force_model=force_model,
        reference_mode=reference_mode,
        fidelity_mode=fidelity_mode,
    ).apply(
        problem,
        state,
        ctx,
    )
    if final_state.pos is None:
        raise RuntimeError("FM^3 pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_fmmm_pipeline", "layout_fmmm_pipeline"]

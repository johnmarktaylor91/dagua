"""Stress-SGD layout pipeline."""

from __future__ import annotations

from typing import Optional, Union

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.preprocess import BuildAdjacency, BuildAdjacencyConfig
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.stress_sgd import (  # noqa: E402
    InitializeStressSGDState,
    PrepareStressSGDTerms,
    RunStressSGDApproximateSchedule,
    RunStressSGDExactSchedule,
)

_DEFAULT_EPS = 0.01
_DEFAULT_MAX_EXACT_NODES = 10_000


def build_stress_sgd_pipeline(
    steps: int = 30,
    eps: float = _DEFAULT_EPS,
    max_exact_nodes: int = _DEFAULT_MAX_EXACT_NODES,
    sample_size: Union[int, str] = "auto",
    trace_every: int = 0,
    fidelity_mode: bool = False,
) -> Pipeline:
    """Build a Stress-SGD layout pipeline.

    Parameters
    ----------
    steps : int
        Number of optimization epochs.
    eps : float
        Final schedule shrinkage factor.
    max_exact_nodes : int
        Node cutoff for choosing exact versus approximate terms.
    sample_size : int | str
        Sample budget for approximate mode.
    trace_every : int
        Optional snapshot interval.
    fidelity_mode : bool, default=False
        Enable reference-parity preprocessing, exact term precision, and native
        term traversal order for ``classic_stress_sgd`` versus ``s_gd2``
        comparisons.

    Returns
    -------
    Pipeline
        Pipeline implementing the Stress-SGD algorithm. The pipeline produces
        final node coordinates by building weighted adjacency, initializing the
        SGD state, preparing exact or approximate stress terms, and running the
        corresponding annealed optimization schedule with optional traces.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if sample_size != "auto" and (not isinstance(sample_size, int) or sample_size <= 0):
        raise ValueError("sample_size must be a positive integer or 'auto'.")

    return Pipeline(
        [
            BuildAdjacency(
                BuildAdjacencyConfig(
                    weighted=True,
                    dedup="sum" if fidelity_mode else "min",
                    format="list",
                    directed=False,
                )
            ),
            InitializeStressSGDState(
                trace_every=trace_every,
                reference_disconnected_policy=fidelity_mode,
            ),
            PrepareStressSGDTerms(
                max_exact_nodes=max_exact_nodes,
                exact_float64_terms=fidelity_mode,
                reference_term_order=fidelity_mode,
            ),
            RunStressSGDExactSchedule(steps=steps, eps=eps, trace_every=trace_every),
            RunStressSGDApproximateSchedule(
                steps=steps,
                eps=eps,
                sample_size=sample_size,
                trace_every=trace_every,
            ),
        ],
        name="stress_sgd_pipeline",
    )


def layout_stress_sgd_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    init_pos: Optional[torch.Tensor] = None,
    steps: int = 30,
    seed: int = 42,
    sample_size: Union[int, str] = "auto",
    trace_every: int = 0,
    eps: float = _DEFAULT_EPS,
    max_exact_nodes: int = _DEFAULT_MAX_EXACT_NODES,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_mode: bool = False,
) -> Union[torch.Tensor, "tuple[torch.Tensor, list[torch.Tensor]]"]:
    """Run the Stress-SGD pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor | None
        Optional node-size tensor with shape ``[N, 2]``.
    init_pos : torch.Tensor | None
        Optional initial coordinates with shape ``[N, 2]``. When provided,
        exact Stress-SGD starts from these coordinates instead of drawing from
        NumPy.
    steps : int
        Number of optimization epochs.
    seed : int
        RNG seed.
    sample_size : int | str
        Approximate-mode sample size.
    trace_every : int
        Optional trace interval.
    eps : float
        Final schedule shrinkage factor.
    max_exact_nodes : int
        Exact-path cutoff.
    edge_weights : torch.Tensor | None
        Optional edge-weight tensor with shape ``[E]``.
    fidelity_mode : bool, default=False
        Enable reference-parity edge preprocessing, disconnected-graph policy,
        exact ``float64`` term storage, and native term traversal order for
        ``s_gd2`` fidelity runs.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, list[torch.Tensor]]
        Final coordinates with shape ``[N, 2]``. When ``trace_every > 0``,
        periodic trace snapshots are returned alongside the final layout.

    Raises
    ------
    ValueError
        If the public Stress-SGD inputs are invalid.
    RuntimeError
        If the pipeline does not populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if max_exact_nodes < 0:
        raise ValueError("max_exact_nodes must be non-negative")

    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must be shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )
    if init_pos is not None:
        if init_pos.ndim != 2 or init_pos.shape != (num_nodes, 2):
            raise ValueError("init_pos must be shape [N, 2].")

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    output_device = edge_index.device
    prepared_init_pos = (
        init_pos.to(device=output_device, dtype=torch.float32) if init_pos is not None else None
    )
    state = SolveState(pos=prepared_init_pos)
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(output_device)))
    final_state = build_stress_sgd_pipeline(
        steps=steps,
        eps=eps,
        max_exact_nodes=max_exact_nodes,
        sample_size=sample_size,
        trace_every=trace_every,
        fidelity_mode=fidelity_mode,
    ).apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("Stress-SGD pipeline did not produce final positions.")

    traces = final_state.extras.get("stress_sgd_traces", [])
    if trace_every > 0:
        return final_state.pos, traces
    return final_state.pos


__all__ = [
    "build_stress_sgd_pipeline",
    "layout_stress_sgd_pipeline",
]

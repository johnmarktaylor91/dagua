"""ForceAtlas2 force-directed layout pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.force import FA2ForceStep, FA2ForceStepConfig
from dagua.layout.ops.init import (
    FA2InitializePositions,
    FA2InitializePositionsConfig,
    ValidateFA2Inputs,
    ValidateFA2InputsConfig,
)
from dagua.layout.ops.preprocess import FA2PrepareState, FA2PrepareStateConfig
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


@dataclass(frozen=True)
class FA2Config:
    """Configuration for the ForceAtlas2 pipeline.

    Attributes
    ----------
    steps : int
        Number of FA2 iterations.
    gravity : float
        Gravity coefficient.
    scaling_ratio : float
        Repulsion scaling coefficient.
    linlog : bool
        Whether to use log-attraction.
    strong_gravity : bool
        Whether to use strong-gravity mode.
    outbound_attraction_distribution : bool
        Whether to normalize attraction by source mass.
    dissuade_hubs : bool
        Whether to divide attraction by source mass when outbound attraction
        distribution is disabled.
    edge_weight_influence : float
        Exponent applied to edge weights.
    barnes_hut : bool
        Whether to use Barnes-Hut approximation for repulsion.
    barnes_hut_theta : float
        Acceptance threshold for Barnes-Hut.
    jitter_tolerance : float
        Jitter tolerance for adaptive speed control.
    fidelity_mode : bool
        Whether to run FA2 internal tensors in float64 for reference parity.
    """

    steps: int = 100
    gravity: float = 1.0
    scaling_ratio: float = 2.0
    linlog: bool = False
    strong_gravity: bool = False
    outbound_attraction_distribution: bool = True
    dissuade_hubs: bool = False
    edge_weight_influence: float = 1.0
    barnes_hut: bool = False
    barnes_hut_theta: float = 1.2
    jitter_tolerance: float = 1.0
    fidelity_mode: bool = False


def build_fa2_pipeline(config: Optional[FA2Config] = None) -> Pipeline:
    """Build a ForceAtlas2 pipeline.

    Reference fidelity
    ------------------
    Targets: ``fa2`` 1.1.2 / Jacomy et al. (2014), "ForceAtlas2, a Continuous
        Graph Layout Algorithm for Handy Network Visualization".
    Fidelity mode: ``FA2Config.fidelity_mode=True`` uses float64 state and
        reference duplicate-edge overwrite semantics instead of Dagua's summed
        edge weights.
    Verified at: final 100-seed report, strong equivalent for most variants;
        median RMSD 0.048 to 0.173, with dissuade-hubs partial at 0.104.
    Known divergences:
        - Barnes-Hut is native PyTorch/ops code, not the reference Cython tree.
        - Dagua keeps explicit tensor-device handling and optional weighted
          behavior outside fidelity mode.

    Parameters
    ----------
    config : FA2Config, optional
        ForceAtlas2 hyperparameters controlling iteration count, gravity,
        attraction and repulsion variants, and Barnes-Hut acceleration. Uses
        defaults when not provided.

    Returns
    -------
    Pipeline
        Pipeline implementing the classical ForceAtlas2 algorithm. The
        pipeline produces final node coordinates by validating inputs,
        initializing positions, preparing graph-dependent state, and applying
        repeated FA2 force steps with adaptive speed control.

    Raises
    ------
    ValueError
        If ``config.steps`` is negative.
    """
    resolved = config or FA2Config()
    if resolved.steps < 0:
        raise ValueError("steps must be non-negative.")

    dtype = torch.float64 if resolved.fidelity_mode else torch.float32
    return Pipeline(
        [
            ValidateFA2Inputs(
                ValidateFA2InputsConfig(
                    steps=resolved.steps,
                    barnes_hut_theta=resolved.barnes_hut_theta,
                )
            ),
            FixedSteps(FixedStepsConfig(n=resolved.steps)),
            FA2InitializePositions(FA2InitializePositionsConfig(dtype=dtype)),
            FA2PrepareState(
                FA2PrepareStateConfig(
                    outbound_attraction_distribution=resolved.outbound_attraction_distribution,
                    dtype=dtype,
                    duplicate_weight_policy="last" if resolved.fidelity_mode else "sum",
                )
            ),
            Repeat(
                n=resolved.steps,
                ops=[
                    FA2ForceStep(
                        FA2ForceStepConfig(
                            gravity=resolved.gravity,
                            scaling_ratio=resolved.scaling_ratio,
                            linlog=resolved.linlog,
                            strong_gravity=resolved.strong_gravity,
                            outbound_attraction_distribution=resolved.outbound_attraction_distribution,
                            dissuade_hubs=resolved.dissuade_hubs,
                            edge_weight_influence=resolved.edge_weight_influence,
                            barnes_hut=resolved.barnes_hut,
                            barnes_hut_theta=resolved.barnes_hut_theta,
                            jitter_tolerance=resolved.jitter_tolerance,
                        )
                    ),
                ],
            ),
        ],
        name="fa2_pipeline",
    )


def layout_fa2_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 100,
    seed: int = 42,
    gravity: float = 1.0,
    scaling_ratio: float = 2.0,
    linlog: bool = False,
    strong_gravity: bool = False,
    outbound_attraction_distribution: bool = True,
    edge_weights: Optional[torch.Tensor] = None,
    dissuade_hubs: bool = False,
    edge_weight_influence: float = 1.0,
    barnes_hut: bool = False,
    barnes_hut_theta: float = 1.2,
    fidelity_mode: bool = False,
) -> torch.Tensor:
    """Run the ForceAtlas2 pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``. Unused and retained
        for API compatibility.
    steps : int, default=100
        Number of ForceAtlas2 iterations.
    seed : int, default=42
        Random seed for the Python-random initialization.
    gravity : float, default=1.0
        Gravity coefficient applied each iteration.
    scaling_ratio : float, default=2.0
        Repulsion scaling coefficient.
    linlog : bool, default=False
        Whether to use the LinLog attraction variant.
    strong_gravity : bool, default=False
        Whether to use strong-gravity mode.
    outbound_attraction_distribution : bool, default=True
        Whether to normalize attraction by source mass.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    dissuade_hubs : bool, default=False
        Whether to divide attraction by source mass when outbound attraction
        distribution is disabled.
    edge_weight_influence : float, default=1.0
        Exponent applied to edge weights during attraction.
    barnes_hut : bool, default=False
        Whether to use Barnes-Hut approximation for repulsion.
    barnes_hut_theta : float, default=1.2
        Acceptance threshold for Barnes-Hut aggregation.
    fidelity_mode : bool, default=False
        Run FA2 internal tensors in float64 to better match the live
        ForceAtlas2 reference. The default keeps the historical float32 path.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    del node_sizes

    config = FA2Config(
        steps=steps,
        gravity=gravity,
        scaling_ratio=scaling_ratio,
        linlog=linlog,
        strong_gravity=strong_gravity,
        outbound_attraction_distribution=outbound_attraction_distribution,
        dissuade_hubs=dissuade_hubs,
        edge_weight_influence=edge_weight_influence,
        barnes_hut=barnes_hut,
        barnes_hut_theta=barnes_hut_theta,
        fidelity_mode=fidelity_mode,
    )
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(edge_index.device)))
    final_state = build_fa2_pipeline(config=config).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("FA2 pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["FA2Config", "build_fa2_pipeline", "layout_fa2_pipeline"]

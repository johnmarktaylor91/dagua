"""ForceAtlas2 force-directed layout pipeline."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
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
from dagua.layout.ops.pipelines import resolve_fidelity_dtype
from dagua.layout.ops.preprocess import FA2PrepareState, FA2PrepareStateConfig
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _reference_exact_edge_arrays(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Build reference ordered edge, weight, and mass arrays.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    tuple[np.ndarray, Optional[np.ndarray], np.ndarray]
        Undirected edge pairs, optional edge weights, and FA2 node masses.
    """
    collapsed: dict[tuple[int, int], float] = {}
    edges_np = edge_index.detach().cpu().numpy()
    weights_np = None if edge_weights is None else edge_weights.detach().cpu().numpy()
    for edge_offset in range(edges_np.shape[1]):
        source = int(edges_np[0, edge_offset])
        target = int(edges_np[1, edge_offset])
        if source == target:
            continue
        key = (min(source, target), max(source, target))
        collapsed[key] = 1.0 if weights_np is None else float(weights_np[edge_offset])

    ordered_pairs = sorted(collapsed)
    edge_pairs = np.asarray(ordered_pairs, dtype=np.int64).reshape((-1, 2))
    weights = None
    if edge_weights is not None:
        weights = np.asarray([collapsed[pair] for pair in ordered_pairs], dtype=np.float64)
    degree = np.zeros(num_nodes, dtype=np.float64)
    for source, target in ordered_pairs:
        degree[source] += 1.0
        degree[target] += 1.0
    return edge_pairs, weights, degree + 1.0


def _layout_fa2_reference_exact(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    steps: int,
    seed: int,
    gravity: float,
    scaling_ratio: float,
    linlog: bool,
    strong_gravity: bool,
    outbound_attraction_distribution: bool,
    edge_weights: Optional[torch.Tensor],
    dissuade_hubs: bool,
    edge_weight_influence: float,
    fidelity_dtype: torch.dtype,
) -> torch.Tensor:
    """Run the live ``fa2`` exact-loop kernel for fidelity mode.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Number of ForceAtlas2 iterations.
    seed : int
        Python ``random.Random`` seed.
    gravity : float
        Gravity coefficient.
    scaling_ratio : float
        Repulsion scaling coefficient.
    linlog : bool
        Whether to use logarithmic attraction.
    strong_gravity : bool
        Whether to use strong gravity.
    outbound_attraction_distribution : bool
        Whether attraction is divided by source mass.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    dissuade_hubs : bool
        Whether to divide attraction by source mass without outbound
        compensation.
    edge_weight_influence : float
        Edge-weight exponent.

    Returns
    -------
    torch.Tensor
        Final reference-order coordinates with shape ``[N, 2]``.
    """
    if num_nodes == 0:
        return torch.zeros((0, 2), dtype=fidelity_dtype, device=edge_index.device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=fidelity_dtype, device=edge_index.device)

    rng = random.Random(seed)
    pos = np.asarray([[rng.random(), rng.random()] for _ in range(num_nodes)], dtype=np.float64)
    edges, weights, mass = _reference_exact_edge_arrays(edge_index, num_nodes, edge_weights)
    outbound_compensation = float(np.mean(mass)) if outbound_attraction_distribution else 1.0
    old_force = np.zeros_like(pos)
    force = np.zeros_like(pos)
    speed = 1.0
    speed_efficiency = 1.0

    for _ in range(steps):
        old_force[:, :] = force
        force[:, :] = 0.0
        for node_index in range(num_nodes):
            for other_index in range(node_index):
                x_dist = float(pos[node_index, 0] - pos[other_index, 0])
                y_dist = float(pos[node_index, 1] - pos[other_index, 1])
                distance_sq = (x_dist * x_dist) + (y_dist * y_dist)
                if distance_sq > 0.0:
                    factor = scaling_ratio * mass[node_index] * mass[other_index] / distance_sq
                    force[node_index, 0] += x_dist * factor
                    force[node_index, 1] += y_dist * factor
                    force[other_index, 0] -= x_dist * factor
                    force[other_index, 1] -= y_dist * factor

        for node_index in range(num_nodes):
            x_coord = float(pos[node_index, 0])
            y_coord = float(pos[node_index, 1])
            if strong_gravity:
                if x_coord != 0.0 or y_coord != 0.0:
                    factor = scaling_ratio * mass[node_index] * gravity
                    force[node_index, 0] -= x_coord * factor
                    force[node_index, 1] -= y_coord * factor
            else:
                distance = math.sqrt((x_coord * x_coord) + (y_coord * y_coord))
                if distance > 0.0:
                    factor = mass[node_index] * gravity / distance
                    force[node_index, 0] -= x_coord * factor
                    force[node_index, 1] -= y_coord * factor

        for edge_offset in range(edges.shape[0]):
            source = int(edges[edge_offset, 0])
            target = int(edges[edge_offset, 1])
            x_dist = float(pos[source, 0] - pos[target, 0])
            y_dist = float(pos[source, 1] - pos[target, 1])
            edge_factor = 1.0
            if weights is not None:
                weight = float(weights[edge_offset])
                if edge_weight_influence == 1.0:
                    edge_factor = weight
                elif edge_weight_influence != 0.0:
                    edge_factor = weight ** float(edge_weight_influence)
            if linlog:
                distance = math.sqrt((x_dist * x_dist) + (y_dist * y_dist))
                if distance <= 0.0:
                    continue
                factor = -outbound_compensation * edge_factor * math.log1p(distance) / distance
            else:
                factor = -outbound_compensation * edge_factor
            if outbound_attraction_distribution:
                factor /= mass[source]
            if dissuade_hubs and not outbound_attraction_distribution:
                factor /= mass[source]
            force[source, 0] += x_dist * factor
            force[source, 1] += y_dist * factor
            force[target, 0] -= x_dist * factor
            force[target, 1] -= y_dist * factor

        total_swinging = 0.0
        total_effective_traction = 0.0
        for node_index in range(num_nodes):
            diff_x = float(old_force[node_index, 0] - force[node_index, 0])
            diff_y = float(old_force[node_index, 1] - force[node_index, 1])
            sum_x = float(old_force[node_index, 0] + force[node_index, 0])
            sum_y = float(old_force[node_index, 1] + force[node_index, 1])
            total_swinging += mass[node_index] * math.sqrt((diff_x * diff_x) + (diff_y * diff_y))
            total_effective_traction += (
                0.5 * mass[node_index] * math.sqrt((sum_x * sum_x) + (sum_y * sum_y))
            )

        estimated_jitter = 0.05 * math.sqrt(num_nodes)
        min_jitter = math.sqrt(estimated_jitter)
        jitter = min_jitter
        if total_effective_traction > 0.0:
            jitter = max(
                min_jitter,
                min(10.0, estimated_jitter * total_effective_traction / (num_nodes * num_nodes)),
            )
        if total_effective_traction > 0.0 and total_swinging / total_effective_traction > 2.0:
            if speed_efficiency > 0.05:
                speed_efficiency *= 0.5
            jitter = max(jitter, 1.0)
        target_speed = (
            float("inf")
            if total_swinging == 0.0
            else jitter * speed_efficiency * total_effective_traction / total_swinging
        )
        if total_swinging > jitter * total_effective_traction:
            if speed_efficiency > 0.05:
                speed_efficiency *= 0.7
        elif speed < 1000.0:
            speed_efficiency *= 1.3
        speed = speed + min(target_speed - speed, 0.5 * speed)

        for node_index in range(num_nodes):
            diff_x = float(old_force[node_index, 0] - force[node_index, 0])
            diff_y = float(old_force[node_index, 1] - force[node_index, 1])
            swinging = mass[node_index] * math.sqrt((diff_x * diff_x) + (diff_y * diff_y))
            factor = speed / (1.0 + math.sqrt(speed * swinging))
            pos[node_index, 0] += force[node_index, 0] * factor
            pos[node_index, 1] += force[node_index, 1] * factor

    return torch.from_numpy(pos).to(device=edge_index.device, dtype=fidelity_dtype)


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
    fidelity_dtype : torch.dtype
        Internal dtype used when fidelity mode is enabled.
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
    fidelity_dtype: Optional[torch.dtype] = None


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

    dtype = (
        resolve_fidelity_dtype(resolved.fidelity_mode, resolved.fidelity_dtype)
        if resolved.fidelity_mode
        else torch.float32
    )
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
    fidelity_dtype: Optional[torch.dtype] = None,
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

    resolved_dtype = resolve_fidelity_dtype(fidelity_mode, fidelity_dtype)
    if fidelity_mode and not barnes_hut:
        return _layout_fa2_reference_exact(
            edge_index,
            num_nodes,
            steps=steps,
            seed=seed,
            gravity=gravity,
            scaling_ratio=scaling_ratio,
            linlog=linlog,
            strong_gravity=strong_gravity,
            outbound_attraction_distribution=outbound_attraction_distribution,
            edge_weights=edge_weights,
            dissuade_hubs=dissuade_hubs,
            edge_weight_influence=edge_weight_influence,
            fidelity_dtype=resolved_dtype,
        )

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
        fidelity_dtype=resolved_dtype,
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

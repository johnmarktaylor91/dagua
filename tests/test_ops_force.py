"""Tests for primitive force-directed layout operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
import torch

from dagua.layout.classic.fa2 import (
    _adjust_speed_and_apply_forces as _fa2_adjust_speed_and_apply_forces,
)
from dagua.layout.classic.fa2 import _barnes_hut_repulsion as _fa2_barnes_hut_repulsion
from dagua.layout.classic.fa2 import _build_barnes_hut_tree as _fa2_build_barnes_hut_tree
from dagua.layout.classic.fa2 import _compute_degree as _fa2_compute_degree
from dagua.layout.classic.fa2 import _gravity_force as _fa2_gravity_force
from dagua.layout.classic.fa2 import _unique_undirected_edges_with_weights as _fa2_unique_edges
from dagua.layout.classic.gem import _update_node_sequential as _gem_update_node_sequential
from dagua.layout.classic.stress_majorization import _smacof_update as _stress_maj_smacof_update
from dagua.layout.classic.stress_sgd import _apply_pair_update as _stress_sgd_apply_pair_update
from dagua.layout.ops.base import Pipeline, Repeat
from dagua.layout.ops.force import (
    AdaptiveSpeedApply,
    AdaptiveSpeedApplyConfig,
    ApplyDisplacement,
    BarnesHutForce,
    BarnesHutForceConfig,
    CellGridForce,
    DensityGridForce,
    DensityGridForceConfig,
    DesiredLengthSpringAttraction,
    FA2DegreeCompensatedAttraction,
    GEMNodeTick,
    GravityToBarycenter,
    GravityToOrigin,
    GravityToOriginConfig,
    InverseDistanceRepulsion,
    InverseDistanceRepulsionConfig,
    InversePowerRepulsion,
    InversePowerRepulsionConfig,
    InverseSquareRepulsion,
    StressMajNodeSweep,
    StressSGDPairUpdate,
    StressSGDPairUpdateConfig,
    UniformSpringAttraction,
    UniformSpringAttractionConfig,
    ZeroForces,
)
from dagua.layout.ops.init import RandomUniformInit, RandomUniformInitConfig
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState

_GRAPHOPT_COULOMBS_CONSTANT = 8_987_500_000.0
_GRAPHOPT_MIN_DISTANCE = 1.0e-12
_LGL_MIN_DISTANCE = 1.0e-12


@dataclass
class _QuadraticDensityGrid:
    """Simple density proxy with a known analytic gradient.

    Parameters
    ----------
    cell_width : float, default=1.0
        Finite-difference step size exposed to the op.
    """

    cell_width: float = 1.0

    def coarse_density(self, position: torch.Tensor) -> float:
        """Return a quadratic energy with gradient ``[2x, 4y]``.

        Parameters
        ----------
        position : torch.Tensor
            Candidate position with shape ``[2]``.

        Returns
        -------
        float
            Scalar energy value.
        """
        x_value = float(position[0].item())
        y_value = float(position[1].item())
        return (x_value * x_value) + (2.0 * y_value * y_value)


def _make_problem() -> LayoutProblem:
    """Build a deterministic 10-node test graph."""

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 0],
        ],
        dtype=torch.long,
    )
    edge_weights = torch.linspace(1.0, 2.4, steps=edge_index.shape[1], dtype=torch.float32)
    return LayoutProblem(edge_index=edge_index, num_nodes=10, edge_weights=edge_weights)


def _make_positions() -> torch.Tensor:
    """Create a deterministic 10-node position tensor."""

    return torch.tensor(
        [
            [-4.0, -1.5],
            [-3.0, 1.0],
            [-1.5, -2.5],
            [-0.5, 2.0],
            [1.0, -1.0],
            [2.5, 1.5],
            [3.0, -2.0],
            [4.5, 2.5],
            [6.0, -0.5],
            [7.5, 2.0],
        ],
        dtype=torch.float32,
    )


def _make_state() -> SolveState:
    """Build a populated solve state for force-op tests."""

    pos = _make_positions()
    return SolveState(
        pos=pos.clone(),
        forces=torch.zeros_like(pos),
        old_forces=torch.full_like(pos, 0.25),
        temperature=0.5,
        spring_lengths=torch.linspace(1.5, 3.3, steps=pos.shape[0], dtype=pos.dtype),
        extras={"force_area": 1.0},
    )


def _expected_inverse_distance(problem: LayoutProblem, pos: torch.Tensor) -> torch.Tensor:
    """Compute the FR-style exact inverse-distance repulsion."""

    optimal_distance = (1.0 / float(problem.num_nodes)) ** 0.5
    delta = pos.unsqueeze(1) - pos.unsqueeze(0)
    distance = torch.linalg.vector_norm(delta, dim=2).clamp(min=0.01)
    contribution = delta * ((optimal_distance * optimal_distance) / distance.square()).unsqueeze(2)
    diagonal = torch.eye(pos.shape[0], dtype=torch.bool)
    contribution = contribution.masked_fill(diagonal.unsqueeze(2), 0.0)
    return contribution.sum(dim=1)


def _expected_inverse_square(pos: torch.Tensor, charge: float, cutoff: float) -> torch.Tensor:
    """Compute the GraphOpt exact inverse-square repulsion."""

    pair_source, pair_target = torch.triu_indices(pos.shape[0], pos.shape[0], offset=1)
    delta = pos[pair_source] - pos[pair_target]
    distance_sq = delta.square().sum(dim=1)
    mask = (distance_sq > _GRAPHOPT_MIN_DISTANCE) & (distance_sq < (cutoff * cutoff))
    expected = torch.zeros_like(pos)
    if not bool(mask.any().item()):
        return expected

    pair_delta = delta[mask]
    pair_distance_sq = distance_sq[mask]
    pair_distance = torch.sqrt(pair_distance_sq)
    direction = pair_delta / pair_distance.unsqueeze(1)
    magnitude = _GRAPHOPT_COULOMBS_CONSTANT * (charge * charge) / pair_distance_sq
    contribution = direction * magnitude.unsqueeze(1)
    expected.index_add_(0, pair_source[mask], contribution)
    expected.index_add_(0, pair_target[mask], -contribution)
    return expected


def _expected_inverse_power(
    pos: torch.Tensor,
    exponent: float,
    ideal_length: float,
) -> torch.Tensor:
    """Compute the exact SFDP inverse-power repulsion."""

    repulsive_scale = max(ideal_length, 1.0e-9) ** (1.0 - exponent)
    delta = pos[:, None, :] - pos[None, :, :]
    distance_sq = torch.sum(delta * delta, dim=-1).clamp_min(1.0e-9)
    distance = torch.sqrt(distance_sq)
    diagonal = torch.eye(pos.shape[0], dtype=torch.bool)
    distance = distance.masked_fill(diagonal, float("inf"))
    denominator = distance.pow(2.0 - exponent).unsqueeze(-1)
    pairwise_force = repulsive_scale * delta / denominator
    pairwise_force = pairwise_force.masked_fill(diagonal.unsqueeze(-1), 0.0)
    return pairwise_force.sum(dim=1)


def _expected_uniform_area_attraction(problem: LayoutProblem, pos: torch.Tensor) -> torch.Tensor:
    """Compute FR's edge attraction term."""

    optimal_distance = (1.0 / float(problem.num_nodes)) ** 0.5
    src = problem.edge_index[0]
    dst = problem.edge_index[1]
    delta = pos[src] - pos[dst]
    edge_weights = (
        problem.edge_weights if problem.edge_weights is not None else torch.ones(src.shape[0])
    )
    contribution = -delta * (
        edge_weights * torch.linalg.vector_norm(delta, dim=1) / optimal_distance
    ).unsqueeze(1)
    expected = torch.zeros_like(pos)
    expected.index_add_(0, src, contribution)
    expected.index_add_(0, dst, -contribution)
    return expected


def _expected_desired_length_attraction(
    problem: LayoutProblem,
    pos: torch.Tensor,
    spring_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute the GEM-style desired-length attraction term."""

    src = problem.edge_index[0]
    dst = problem.edge_index[1]
    delta = pos[src] - pos[dst]
    distances = torch.linalg.vector_norm(delta, dim=1)
    degrees = torch.zeros((problem.num_nodes,), dtype=pos.dtype)
    degrees.index_add_(0, src, torch.ones_like(src, dtype=pos.dtype))
    degrees.index_add_(0, dst, torch.ones_like(dst, dtype=pos.dtype))
    degree_weights = degrees / 2.5 + 1.0
    source_weights = degree_weights[src].clamp(min=1.0)
    target_weights = degree_weights[dst].clamp(min=1.0)
    source_desired = spring_lengths[src].clamp(min=1.0e-9)
    target_desired = spring_lengths[dst].clamp(min=1.0e-9)
    edge_weights = (
        problem.edge_weights if problem.edge_weights is not None else torch.ones(src.shape[0])
    )
    source_force = -delta * (distances / (source_desired * source_weights)).unsqueeze(1)
    target_force = delta * (distances / (target_desired * target_weights)).unsqueeze(1)
    source_force = source_force * edge_weights.unsqueeze(1)
    target_force = target_force * edge_weights.unsqueeze(1)
    expected = torch.zeros_like(pos)
    expected.index_add_(0, src, source_force)
    expected.index_add_(0, dst, target_force)
    return expected


def _expected_cell_grid_force(problem: LayoutProblem, pos: torch.Tensor) -> torch.Tensor:
    """Compute the LGL sparse-cell repulsion for all nodes."""

    del problem
    num_nodes = int(pos.shape[0])
    area = float(num_nodes * num_nodes)
    cell_size = area**0.25
    repulse_rad = area * float(num_nodes)
    frk = float(num_nodes) ** 0.5
    buckets: dict[tuple[int, int], list[int]] = {}
    for node in range(num_nodes):
        key = (
            int(np.floor(float(pos[node, 0].item()) / cell_size)),
            int(np.floor(float(pos[node, 1].item()) / cell_size)),
        )
        buckets.setdefault(key, []).append(node)

    expected = torch.zeros_like(pos)
    for cell in sorted(buckets):
        nodes_here = buckets[cell]
        for offset_y in (-1, 0, 1):
            for offset_x in (-1, 0, 1):
                neighbor_cell = (cell[0] + offset_x, cell[1] + offset_y)
                if neighbor_cell not in buckets or neighbor_cell < cell:
                    continue
                nodes_there = buckets[neighbor_cell]
                if neighbor_cell == cell:
                    pairs = [
                        (nodes_here[left_index], nodes_here[right_index])
                        for left_index in range(len(nodes_here))
                        for right_index in range(left_index + 1, len(nodes_here))
                    ]
                else:
                    pairs = [(left, right) for left in nodes_here for right in nodes_there]
                for left, right in pairs:
                    delta = pos[left] - pos[right]
                    distance = float(torch.linalg.vector_norm(delta).item())
                    if distance >= cell_size:
                        continue
                    safe_distance = max(distance, _LGL_MIN_DISTANCE)
                    direction = delta / safe_distance
                    magnitude = (frk * frk) * (
                        (1.0 / safe_distance) - ((safe_distance * safe_distance) / repulse_rad)
                    )
                    contribution = direction * magnitude
                    expected[left] += contribution
                    expected[right] -= contribution
    return expected


def _distance_matrix(pos: torch.Tensor) -> torch.Tensor:
    """Build a dense Euclidean distance matrix from positions."""

    return torch.cdist(pos, pos, p=2.0)


def _fresh_ctx() -> RuntimeContext:
    """Create a deterministic runtime context for op tests."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    return RuntimeContext(generator=generator)


def test_force_pipeline_matches_manual_fr_update() -> None:
    """ZeroForces -> repulsion -> attraction -> apply should match FR math."""

    problem = _make_problem()
    initial_pos = _make_positions()
    state = SolveState(pos=initial_pos.clone(), temperature=0.4, extras={"force_area": 1.0})

    pipeline = Pipeline(
        [
            ZeroForces(),
            InverseDistanceRepulsion(config=InverseDistanceRepulsionConfig(k_formula="area")),
            UniformSpringAttraction(config=UniformSpringAttractionConfig(k_formula="area")),
            ApplyDisplacement(),
        ]
    )

    result = pipeline.apply(problem=problem, state=state, ctx=_fresh_ctx())

    expected_force = _expected_inverse_distance(problem=problem, pos=initial_pos)
    expected_force = expected_force + _expected_uniform_area_attraction(
        problem=problem,
        pos=initial_pos,
    )
    expected_length = torch.linalg.vector_norm(expected_force, dim=1).clamp(min=0.01)
    expected_pos = initial_pos + expected_force * (0.4 / expected_length).unsqueeze(1)

    assert result.forces is not None
    torch.testing.assert_close(result.forces, expected_force)
    torch.testing.assert_close(result.pos, expected_pos)


@pytest.mark.parametrize(
    ("name", "runner"),
    [
        (
            "zero_forces",
            lambda: (
                lambda state: (
                    ZeroForces().apply(_make_problem(), state, _fresh_ctx()),
                    torch.zeros_like(_make_positions()),
                    "forces",
                )
            )(_make_state()),
        ),
        (
            "inverse_distance_repulsion",
            lambda: (
                lambda state: (
                    InverseDistanceRepulsion().apply(_make_problem(), state, _fresh_ctx()),
                    _expected_inverse_distance(_make_problem(), _make_positions()),
                    "forces",
                )
            )(_make_state()),
        ),
        (
            "inverse_square_repulsion",
            lambda: (
                lambda state: (
                    InverseSquareRepulsion().apply(_make_problem(), state, _fresh_ctx()),
                    _expected_inverse_square(_make_positions(), charge=0.001, cutoff=500.0),
                    "forces",
                )
            )(_make_state()),
        ),
        (
            "inverse_power_repulsion",
            lambda: (
                lambda state: (
                    InversePowerRepulsion().apply(
                        _make_problem(),
                        SolveState(
                            pos=state.pos,
                            forces=state.forces,
                            extras={"ideal_length": 2.25},
                        ),
                        _fresh_ctx(),
                    ),
                    _expected_inverse_power(_make_positions(), exponent=-1.0, ideal_length=2.25),
                    "forces",
                )
            )(_make_state()),
        ),
        (
            "uniform_spring_attraction",
            lambda: (
                lambda state: (
                    UniformSpringAttraction().apply(_make_problem(), state, _fresh_ctx()),
                    _expected_uniform_area_attraction(_make_problem(), _make_positions()),
                    "forces",
                )
            )(_make_state()),
        ),
        (
            "desired_length_spring_attraction",
            lambda: (
                lambda state: (
                    DesiredLengthSpringAttraction().apply(_make_problem(), state, _fresh_ctx()),
                    _expected_desired_length_attraction(
                        _make_problem(),
                        _make_positions(),
                        torch.linspace(1.5, 3.3, steps=10, dtype=torch.float32),
                    ),
                    "forces",
                )
            )(_make_state()),
        ),
        (
            "fa2_degree_compensated_attraction",
            lambda: (
                lambda state: (
                    FA2DegreeCompensatedAttraction().apply(_make_problem(), state, _fresh_ctx()),
                    _fa2_expected_attraction(_make_problem(), _make_positions()),
                    "forces",
                )
            )(_make_state()),
        ),
        (
            "gravity_to_origin",
            lambda: (
                lambda state: (
                    GravityToOrigin(
                        config=GravityToOriginConfig(strength=1.5, strong_mode=False)
                    ).apply(_make_problem(), state, _fresh_ctx()),
                    _fa2_expected_gravity(_make_problem(), _make_positions(), 1.5, False),
                    "forces",
                )
            )(_make_state()),
        ),
        (
            "gravity_to_barycenter",
            lambda: (
                lambda state: (
                    GravityToBarycenter().apply(_make_problem(), state, _fresh_ctx()),
                    _expected_barycenter_gravity(_make_problem(), _make_positions(), 1.0 / 16.0),
                    "forces",
                )
            )(_make_state()),
        ),
        (
            "barnes_hut_force",
            lambda: (
                lambda state: (
                    BarnesHutForce(config=BarnesHutForceConfig(theta=1.2)).apply(
                        _make_problem(),
                        SolveState(
                            pos=state.pos,
                            forces=state.forces,
                            extras={"quadtree": _fa2_tree_for_positions(_make_positions())},
                        ),
                        _fresh_ctx(),
                    ),
                    _fa2_barnes_hut_repulsion(
                        pos=_make_positions(),
                        mass=_fa2_mass_for_problem(_make_problem()),
                        scaling_ratio=1.0,
                        theta=1.2,
                    ),
                    "forces",
                )
            )(_make_state()),
        ),
        (
            "density_grid_force",
            lambda: (
                lambda state: (
                    DensityGridForce(config=DensityGridForceConfig()).apply(
                        _make_problem(),
                        SolveState(
                            pos=state.pos,
                            forces=state.forces,
                            extras={"density_grid": _QuadraticDensityGrid(cell_width=1.0)},
                        ),
                        _fresh_ctx(),
                    ),
                    torch.stack(
                        [-2.0 * _make_positions()[:, 0], -4.0 * _make_positions()[:, 1]],
                        dim=1,
                    ),
                    "forces",
                )
            )(_make_state()),
        ),
        (
            "cell_grid_force",
            lambda: (
                lambda state: (
                    CellGridForce().apply(_make_problem(), state, _fresh_ctx()),
                    _expected_cell_grid_force(_make_problem(), _make_positions()),
                    "forces",
                )
            )(_make_state()),
        ),
        (
            "apply_displacement",
            lambda: (
                lambda state: (
                    ApplyDisplacement().apply(
                        _make_problem(),
                        SolveState(
                            pos=state.pos,
                            forces=_expected_inverse_distance(_make_problem(), _make_positions()),
                            temperature=0.3,
                        ),
                        _fresh_ctx(),
                    ),
                    _make_positions()
                    + _expected_inverse_distance(_make_problem(), _make_positions())
                    * (
                        0.3
                        / torch.linalg.vector_norm(
                            _expected_inverse_distance(_make_problem(), _make_positions()), dim=1
                        ).clamp(min=0.01)
                    ).unsqueeze(1),
                    "pos",
                )
            )(_make_state()),
        ),
        (
            "adaptive_speed_apply",
            lambda: _run_adaptive_speed_case(),
        ),
        (
            "gem_node_tick",
            lambda: _run_gem_tick_case(),
        ),
        (
            "stress_sgd_pair_update",
            lambda: _run_stress_sgd_case(),
        ),
        (
            "stress_majorization_node_sweep",
            lambda: _run_stress_maj_case(),
        ),
    ],
)
def test_each_force_op_on_ten_node_graph(
    name: str,
    runner: Callable[[], tuple[SolveState, torch.Tensor, str]],
) -> None:
    """Each force op should produce the expected 10-node result."""

    del name
    result, expected, target = runner()
    assert result.pos is None or torch.isfinite(result.pos).all()
    assert result.forces is None or torch.isfinite(result.forces).all()

    if target == "forces":
        assert result.forces is not None
        torch.testing.assert_close(result.forces, expected, rtol=1.0e-5, atol=1.0e-5)
    else:
        assert result.pos is not None
        torch.testing.assert_close(result.pos, expected, rtol=1.0e-5, atol=1.0e-5)


def _fa2_mass_for_problem(problem: LayoutProblem) -> torch.Tensor:
    """Return ForceAtlas2's mass vector for the test graph."""

    return _fa2_compute_degree(problem.edge_index, problem.num_nodes).to(dtype=torch.float32) + 1.0


def _fa2_tree_for_positions(pos: torch.Tensor) -> object:
    """Build a FA2 Barnes-Hut tree for the 10-node test positions."""

    mass = _fa2_mass_for_problem(_make_problem()).detach().cpu().numpy()
    return _fa2_build_barnes_hut_tree(
        pos_np=pos.detach().cpu().numpy(),
        mass_np=mass,
        indices=np.arange(pos.shape[0], dtype=np.int64),
    )


def _fa2_expected_attraction(problem: LayoutProblem, pos: torch.Tensor) -> torch.Tensor:
    """Compute the FA2 attraction term with the reference helper."""

    undirected_edges, undirected_weights = _fa2_unique_edges(
        edge_index=problem.edge_index,
        edge_weights=problem.edge_weights,
    )
    mass = _fa2_mass_for_problem(problem)
    return torch.zeros_like(pos).add(
        _fa2_attraction_reference(
            pos=pos,
            edge_index=undirected_edges,
            mass=mass,
            edge_weights=undirected_weights,
        )
    )


def _fa2_attraction_reference(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    mass: torch.Tensor,
    edge_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Evaluate the FA2 attraction reference helper."""

    from dagua.layout.classic.fa2 import _attraction_force as _fa2_attraction_force

    return _fa2_attraction_force(
        pos=pos,
        edge_index=edge_index,
        mass=mass,
        outbound_att_compensation=float(mass.mean().item()),
        outbound_attraction_distribution=True,
        linlog=False,
        edge_weights=None if edge_weights is None else edge_weights.to(dtype=pos.dtype),
        dissuade_hubs=False,
        edge_weight_influence=1.0,
    )


def _fa2_expected_gravity(
    problem: LayoutProblem,
    pos: torch.Tensor,
    strength: float,
    strong_mode: bool,
) -> torch.Tensor:
    """Compute the FA2 gravity term with the reference helper."""

    mass = _fa2_mass_for_problem(problem)
    return _fa2_gravity_force(
        pos=pos,
        mass=mass,
        gravity=strength,
        strong_gravity=strong_mode,
        scaling_ratio=1.0,
    )


def _expected_barycenter_gravity(
    problem: LayoutProblem,
    pos: torch.Tensor,
    constant: float,
) -> torch.Tensor:
    """Compute the GEM weighted-barycenter gravity term."""

    degrees = torch.zeros((problem.num_nodes,), dtype=pos.dtype)
    src = problem.edge_index[0]
    dst = problem.edge_index[1]
    degrees.index_add_(0, src, torch.ones_like(src, dtype=pos.dtype))
    degrees.index_add_(0, dst, torch.ones_like(dst, dtype=pos.dtype))
    degree_weights = degrees / 2.5 + 1.0
    barycenter = (pos * degree_weights.unsqueeze(1)).sum(dim=0, keepdim=True)
    barycenter = barycenter / float(problem.num_nodes)
    return (barycenter - pos) * constant


def _run_adaptive_speed_case() -> tuple[SolveState, torch.Tensor, str]:
    """Run one deterministic AdaptiveSpeedApply scenario."""

    problem = _make_problem()
    pos = _make_positions()
    force = torch.linspace(-1.5, 1.2, steps=20, dtype=pos.dtype).reshape(10, 2)
    old_force = torch.linspace(1.0, -1.0, steps=20, dtype=pos.dtype).reshape(10, 2)
    state = SolveState(
        pos=pos.clone(),
        forces=force.clone(),
        old_forces=old_force.clone(),
    )
    result = AdaptiveSpeedApply(config=AdaptiveSpeedApplyConfig(jitter_tolerance=1.0)).apply(
        problem,
        state,
        _fresh_ctx(),
    )
    expected_pos, expected_speed, expected_speed_efficiency = _fa2_adjust_speed_and_apply_forces(
        pos=pos,
        force=force,
        old_force=old_force,
        mass=_fa2_mass_for_problem(problem),
        speed=1.0,
        speed_efficiency=1.0,
        jitter_tolerance=1.0,
    )
    assert result.extras["fa2_speed"] == pytest.approx(expected_speed)
    assert result.extras["fa2_speed_efficiency"] == pytest.approx(expected_speed_efficiency)
    return result, expected_pos, "pos"


def _run_gem_tick_case() -> tuple[SolveState, torch.Tensor, str]:
    """Run one deterministic GEM node tick and return the expected position matrix."""

    problem = _make_problem()
    pos = _make_positions()
    forces = torch.linspace(-2.0, 2.0, steps=20, dtype=pos.dtype).reshape(10, 2)
    degrees = torch.zeros((problem.num_nodes,), dtype=pos.dtype)
    src = problem.edge_index[0]
    dst = problem.edge_index[1]
    degrees.index_add_(0, src, torch.ones_like(src, dtype=pos.dtype))
    degrees.index_add_(0, dst, torch.ones_like(dst, dtype=pos.dtype))
    degree_weights = degrees / 2.5 + 1.0
    previous_impulse = torch.zeros_like(forces)
    local_temperatures = torch.full((problem.num_nodes,), 12.0, dtype=pos.dtype)
    skew_gauge = torch.zeros((problem.num_nodes,), dtype=pos.dtype)
    barycenter = (pos * degree_weights.unsqueeze(1)).sum(dim=0)
    expected_pos = pos.clone()
    expected_previous = previous_impulse.clone()
    expected_local = local_temperatures.clone()
    expected_skew = skew_gauge.clone()
    expected_barycenter = barycenter.clone()
    expected_global_temperature = _gem_update_node_sequential(
        node_index=3,
        positions=expected_pos,
        impulse=forces[3],
        previous_impulse=expected_previous,
        local_temperatures=expected_local,
        skew_gauge=expected_skew,
        degree_weights=degree_weights,
        barycenter=expected_barycenter,
        global_temperature=12.0,
    )

    state = SolveState(
        pos=pos.clone(),
        forces=forces.clone(),
        extras={
            "gem_node_index": 3,
            "gem_previous_impulses": previous_impulse.clone(),
            "gem_local_temperatures": local_temperatures.clone(),
            "gem_skew_gauge": skew_gauge.clone(),
            "gem_barycenter": barycenter.clone(),
            "gem_global_temperature": 12.0,
            "gem_degree_weights": degree_weights.clone(),
        },
    )
    result = GEMNodeTick().apply(problem, state, _fresh_ctx())

    torch.testing.assert_close(result.extras["gem_previous_impulses"], expected_previous)
    torch.testing.assert_close(result.extras["gem_local_temperatures"], expected_local)
    torch.testing.assert_close(result.extras["gem_skew_gauge"], expected_skew)
    torch.testing.assert_close(result.extras["gem_barycenter"], expected_barycenter)
    assert result.extras["gem_global_temperature"] == pytest.approx(expected_global_temperature)
    return result, expected_pos, "pos"


def _run_stress_sgd_case() -> tuple[SolveState, torch.Tensor, str]:
    """Run one exact Stress-SGD pair update and return the expected positions."""

    pos = _make_positions()
    expected = pos.detach().cpu().numpy().astype(np.float64, copy=True)
    pair = (2, 8)
    target_distance = 3.5
    eta = 0.8
    weight = 1.0 / (target_distance * target_distance)
    _stress_sgd_apply_pair_update(
        positions=expected,
        source_index=pair[0],
        target_index=pair[1],
        target_distance=target_distance,
        weight=weight,
        eta=eta,
    )

    state = SolveState(
        pos=pos.clone(),
        extras={
            "stress_sgd_pair": pair,
            "stress_sgd_eta": eta,
            "stress_sgd_target_distance": target_distance,
        },
    )
    result = StressSGDPairUpdate(config=StressSGDPairUpdateConfig(clamp_mu=1.0)).apply(
        _make_problem(),
        state,
        _fresh_ctx(),
    )
    expected_tensor = torch.from_numpy(expected).to(dtype=pos.dtype)
    return result, expected_tensor, "pos"


def _run_stress_maj_case() -> tuple[SolveState, torch.Tensor, str]:
    """Run one SMACOF sweep and return the expected positions."""

    pos = _make_positions()
    distance_matrix = _distance_matrix(pos) + torch.eye(pos.shape[0], dtype=pos.dtype)
    target_distances = distance_matrix.detach().cpu().numpy().astype(np.float64, copy=True)
    with np.errstate(divide="ignore"):
        weights = np.where(target_distances > 0.0, 1.0 / np.square(target_distances), 0.0)
    np.fill_diagonal(weights, 0.0)
    laplacian = -weights
    np.fill_diagonal(laplacian, weights.sum(axis=1))
    laplacian_pinv = np.linalg.pinv(laplacian)
    expected = _stress_maj_smacof_update(
        positions=pos.detach().cpu().numpy().astype(np.float64, copy=True),
        target_distances=target_distances,
        weights=weights,
        laplacian_pinv=laplacian_pinv,
    )

    state = SolveState(pos=pos.clone(), distance_matrix=distance_matrix)
    result = StressMajNodeSweep().apply(_make_problem(), state, _fresh_ctx())
    expected_tensor = torch.from_numpy(expected).to(dtype=pos.dtype)
    return result, expected_tensor, "pos"


def test_force_accumulation_sums_contributions() -> None:
    """Two repulsion ops should add into the shared force buffer."""

    problem = _make_problem()
    state = _make_state()
    result = ZeroForces().apply(problem, state, _fresh_ctx())
    result = InverseDistanceRepulsion().apply(problem, result, _fresh_ctx())
    result = InversePowerRepulsion(config=InversePowerRepulsionConfig(exponent=-1.0)).apply(
        problem,
        SolveState(
            pos=result.pos,
            forces=result.forces,
            extras={"ideal_length": 2.25},
        ),
        _fresh_ctx(),
    )

    expected = _expected_inverse_distance(problem, _make_positions()) + _expected_inverse_power(
        _make_positions(), exponent=-1.0, ideal_length=2.25
    )
    assert result.forces is not None
    torch.testing.assert_close(result.forces, expected, rtol=1.0e-5, atol=1.0e-5)


def test_stress_sgd_pair_update_matches_reference_kernel() -> None:
    """StressSGDPairUpdate should match the reference sequential kernel exactly."""

    result, expected, _ = _run_stress_sgd_case()
    assert result.pos is not None
    torch.testing.assert_close(result.pos, expected, rtol=1.0e-6, atol=1.0e-6)


def _problem_from_edges(
    edges: list[tuple[int, int]],
    num_nodes: int,
    edge_weights: torch.Tensor | None = None,
    seed: int = 42,
) -> LayoutProblem:
    """Build a force-op test problem from a Python edge list.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Source-target edge tuples.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor | None, optional
        Optional per-edge weights with shape ``[E]``.
    seed : int, default=42
        Problem seed.

    Returns
    -------
    LayoutProblem
        Problem carrying the requested graph structure.
    """

    if edges:
        src, dst = zip(*edges)
        edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
        seed=seed,
    )


def _force_state(
    pos: torch.Tensor,
    *,
    forces: torch.Tensor | None = None,
    temperature: float = 1.0,
    extras: dict[str, object] | None = None,
) -> SolveState:
    """Build a solve state for force-op contract tests.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    forces : torch.Tensor | None, optional
        Optional force tensor with shape ``[N, 2]``.
    temperature : float, default=1.0
        Temperature used by displacement ops.
    extras : dict[str, object] | None, optional
        Optional extras payload.

    Returns
    -------
    SolveState
        Solve state initialized with the requested payload.
    """

    return SolveState(
        pos=pos.clone(),
        forces=None if forces is None else forces.clone(),
        temperature=temperature,
        extras={} if extras is None else dict(extras),
    )


def test_zero_forces_handles_empty_graph() -> None:
    """ZeroForces should allocate an empty ``[0, 2]`` buffer for ``N=0``."""

    problem = _problem_from_edges([], num_nodes=0)
    state = SolveState()

    result = ZeroForces().apply(problem, state, _fresh_ctx())

    assert result.forces is not None
    assert result.forces.shape == (0, 2)
    assert result.forces.dtype == torch.float32


@pytest.mark.parametrize(
    "op",
    [
        InverseDistanceRepulsion(),
        UniformSpringAttraction(),
    ],
)
def test_core_force_accumulators_handle_single_node_without_motion(op: object) -> None:
    """Single-node force accumulation should be a finite no-op."""

    problem = _problem_from_edges([], num_nodes=1)
    pos = torch.zeros((1, 2), dtype=torch.float32)
    state = _force_state(pos, forces=torch.zeros_like(pos))

    result = op.apply(problem, state, _fresh_ctx())

    assert result.pos is not None
    assert result.forces is not None
    torch.testing.assert_close(result.pos, pos)
    torch.testing.assert_close(result.forces, torch.zeros_like(pos))


def test_inverse_distance_repulsion_identical_positions_stays_finite_and_zero() -> None:
    """Identical coordinates should not create NaNs or infinite repulsion."""

    problem = _problem_from_edges([], num_nodes=2)
    pos = torch.zeros((2, 2), dtype=torch.float32)
    state = _force_state(pos, forces=torch.zeros_like(pos))

    result = InverseDistanceRepulsion().apply(problem, state, _fresh_ctx())

    assert result.forces is not None
    assert torch.isfinite(result.forces).all()
    torch.testing.assert_close(result.forces, torch.zeros_like(pos))


def test_uniform_spring_attraction_zero_length_edges_stays_finite() -> None:
    """Zero-length spring edges should contribute zero finite attraction."""

    problem = _problem_from_edges([(0, 1)], num_nodes=2)
    pos = torch.zeros((2, 2), dtype=torch.float32)
    state = _force_state(pos, forces=torch.zeros_like(pos))

    result = UniformSpringAttraction().apply(problem, state, _fresh_ctx())

    assert result.forces is not None
    assert torch.isfinite(result.forces).all()
    torch.testing.assert_close(result.forces, torch.zeros_like(pos))


def test_stress_sgd_pair_update_zero_distance_is_noop() -> None:
    """StressSGDPairUpdate should leave coincident nodes unchanged."""

    pos = torch.zeros((2, 2), dtype=torch.float32)
    state = SolveState(
        pos=pos.clone(),
        extras={
            "stress_sgd_pair": (0, 1),
            "stress_sgd_eta": 0.8,
            "stress_sgd_target_distance": 3.5,
        },
    )

    result = StressSGDPairUpdate().apply(_problem_from_edges([], num_nodes=2), state, _fresh_ctx())

    assert result.pos is not None
    torch.testing.assert_close(result.pos, pos)


def test_apply_displacement_with_zero_temperature_is_noop() -> None:
    """Zero temperature should clamp all displacement to zero."""

    pos = _make_positions()
    forces = _expected_inverse_distance(_make_problem(), pos)
    state = _force_state(pos, forces=forces, temperature=0.0)

    result = ApplyDisplacement().apply(_make_problem(), state, _fresh_ctx())

    assert result.pos is not None
    torch.testing.assert_close(result.pos, pos)


def test_zero_forces_overwrites_existing_buffer_with_zeros() -> None:
    """ZeroForces should reuse the existing buffer and clear its contents."""

    pos = _make_positions()
    state = SolveState(pos=pos.clone(), forces=torch.full_like(pos, 7.0))

    result = ZeroForces().apply(_make_problem(), state, _fresh_ctx())

    assert result.forces is not None
    torch.testing.assert_close(result.forces, torch.zeros_like(pos))


def test_full_fr_pipeline_with_random_init_matches_manual_update() -> None:
    """Init -> FR force step should equal the manual one-step update."""

    problem = _make_problem()
    pipeline = Pipeline(
        [
            RandomUniformInit(RandomUniformInitConfig(scale="unit", rng_backend="numpy")),
            ZeroForces(),
            InverseDistanceRepulsion(config=InverseDistanceRepulsionConfig(k_formula="area")),
            UniformSpringAttraction(config=UniformSpringAttractionConfig(k_formula="area")),
            ApplyDisplacement(),
        ]
    )

    result = pipeline.apply(
        problem,
        SolveState(temperature=0.4, extras={"force_area": 1.0}),
        _fresh_ctx(),
    )

    initial_pos = torch.from_numpy(np.random.RandomState(problem.seed).rand(problem.num_nodes, 2))
    expected_force = _expected_inverse_distance(problem=problem, pos=initial_pos)
    expected_force = expected_force + _expected_uniform_area_attraction(
        problem=problem,
        pos=initial_pos,
    )
    expected_pos = initial_pos + expected_force * (
        0.4 / torch.linalg.vector_norm(expected_force, dim=1).clamp(min=0.01)
    ).unsqueeze(1)

    assert result.forces is not None
    assert result.pos is not None
    torch.testing.assert_close(result.forces, expected_force)
    torch.testing.assert_close(result.pos, expected_pos)


def test_repeat_iterates_force_pipeline_and_increments_step() -> None:
    """Repeat should execute the inner force ops exactly ``n`` times."""

    problem = _problem_from_edges([], num_nodes=2)
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    state = SolveState(pos=pos.clone(), temperature=0.25, step=3)

    result = Repeat(
        n=5,
        ops=[ZeroForces(), InverseDistanceRepulsion(), ApplyDisplacement()],
    ).apply(problem, state, _fresh_ctx())

    assert result.step == 8
    assert result.forces is not None
    assert result.pos is not None
    assert torch.isfinite(result.pos).all()


def test_repeat_force_pipeline_on_large_fan_out_stays_finite() -> None:
    """Repeated force updates on a 99-leaf hub should remain finite."""

    problem = _problem_from_edges([(0, leaf) for leaf in range(1, 100)], num_nodes=100)
    pos = torch.from_numpy(np.random.RandomState(7).rand(100, 2)).to(dtype=torch.float32)
    state = SolveState(pos=pos.clone(), temperature=0.1, extras={"force_area": 1.0})

    result = Repeat(
        n=5,
        ops=[
            ZeroForces(),
            InverseDistanceRepulsion(),
            UniformSpringAttraction(),
            ApplyDisplacement(),
        ],
    ).apply(problem, state, _fresh_ctx())

    assert result.step == 5
    assert result.pos is not None
    assert result.forces is not None
    assert torch.isfinite(result.pos).all()
    assert torch.isfinite(result.forces).all()


def test_disconnected_pair_fr_step_only_applies_repulsion() -> None:
    """With no edges, a FR step should reduce to pure inverse-distance repulsion."""

    problem = _problem_from_edges([], num_nodes=2)
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    state = SolveState(pos=pos.clone(), temperature=0.2, extras={"force_area": 1.0})

    result = Pipeline(
        [ZeroForces(), InverseDistanceRepulsion(), UniformSpringAttraction(), ApplyDisplacement()]
    ).apply(problem, state, _fresh_ctx())

    expected_force = _expected_inverse_distance(problem, pos)
    expected_pos = pos + expected_force * (
        0.2 / torch.linalg.vector_norm(expected_force, dim=1).clamp(min=0.01)
    ).unsqueeze(1)

    assert result.forces is not None
    assert result.pos is not None
    torch.testing.assert_close(result.forces, expected_force)
    torch.testing.assert_close(result.pos, expected_pos)


@pytest.mark.parametrize(
    ("op", "state"),
    [
        (
            ZeroForces(),
            SolveState(
                pos=_make_positions().clone(),
                forces=torch.ones_like(_make_positions()),
            ),
        ),
        (
            InverseDistanceRepulsion(),
            _force_state(_make_positions(), forces=torch.zeros_like(_make_positions())),
        ),
        (
            UniformSpringAttraction(),
            _force_state(_make_positions(), forces=torch.zeros_like(_make_positions())),
        ),
    ],
)
def test_force_accumulators_write_forces_without_overwriting_positions(
    op: object,
    state: SolveState,
) -> None:
    """Force-accumulation ops should update ``forces`` while preserving ``pos``."""

    before_pos = None if state.pos is None else state.pos.clone()
    before_forces = None if state.forces is None else state.forces.clone()

    result = op.apply(_make_problem(), state, _fresh_ctx())

    assert result.forces is not None
    if before_pos is not None:
        torch.testing.assert_close(result.pos, before_pos)
    if before_forces is not None:
        assert not torch.equal(result.forces, before_forces) or isinstance(op, ZeroForces)


@pytest.mark.parametrize(
    ("op", "state"),
    [
        (
            ApplyDisplacement(),
            SolveState(
                pos=_make_positions().clone(),
                forces=_expected_inverse_distance(_make_problem(), _make_positions()),
                temperature=0.3,
            ),
        ),
        (
            StressSGDPairUpdate(),
            SolveState(
                pos=_make_positions().clone(),
                extras={
                    "stress_sgd_pair": (2, 8),
                    "stress_sgd_eta": 0.8,
                    "stress_sgd_target_distance": 3.5,
                },
            ),
        ),
    ],
)
def test_position_update_ops_write_positions(
    op: object,
    state: SolveState,
) -> None:
    """Position-update ops should move coordinates rather than writing forces."""

    before_pos = state.pos.clone()
    before_forces = None if state.forces is None else state.forces.clone()

    result = op.apply(_make_problem(), state, _fresh_ctx())

    assert result.pos is not None
    assert not torch.equal(result.pos, before_pos)
    if before_forces is None:
        assert result.forces is None
    else:
        torch.testing.assert_close(result.forces, before_forces)


@pytest.mark.parametrize(
    ("op_class", "expected_write", "forbidden_write"),
    [
        (ZeroForces, "forces", "pos"),
        (InverseDistanceRepulsion, "forces", "pos"),
        (InverseSquareRepulsion, "forces", "pos"),
        (InversePowerRepulsion, "forces", "pos"),
        (UniformSpringAttraction, "forces", "pos"),
        (DesiredLengthSpringAttraction, "forces", "pos"),
        (FA2DegreeCompensatedAttraction, "forces", "pos"),
        (GravityToOrigin, "forces", "pos"),
        (GravityToBarycenter, "forces", "pos"),
        (BarnesHutForce, "forces", "pos"),
        (DensityGridForce, "forces", "pos"),
        (CellGridForce, "forces", "pos"),
        (ApplyDisplacement, "pos", "forces"),
        (AdaptiveSpeedApply, "pos", "forces"),
        (GEMNodeTick, "pos", "forces"),
        (StressSGDPairUpdate, "pos", "forces"),
        (StressMajNodeSweep, "pos", "forces"),
    ],
)
def test_force_op_metadata_declares_primary_write_target(
    op_class: type[object],
    expected_write: str,
    forbidden_write: str,
) -> None:
    """Force-op metadata should advertise the correct primary target buffer."""

    assert expected_write in op_class.writes
    assert forbidden_write not in op_class.writes


def test_apply_displacement_metadata_declares_force_read() -> None:
    """ApplyDisplacement should declare that it consumes the force buffer."""

    assert "forces" in ApplyDisplacement.reads


def test_zero_forces_produces_all_zero_buffer() -> None:
    """ZeroForces should always leave the force buffer exactly zero."""

    pos = torch.from_numpy(np.random.RandomState(9).rand(6, 2)).to(dtype=torch.float32)
    state = SolveState(pos=pos.clone(), forces=torch.randn_like(pos))

    result = ZeroForces().apply(_problem_from_edges([], num_nodes=6), state, _fresh_ctx())

    assert result.forces is not None
    torch.testing.assert_close(result.forces, torch.zeros_like(pos))

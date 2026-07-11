"""Regression tests for FM^3 OGDF fidelity controls."""

from __future__ import annotations

import random

import torch

from dagua.layout.ops.fmmm import (
    _GALAXY_CHOICE_LOWER,
    FMMMForceStep,
    _build_hierarchy,
    _RandomNodeSet,
    _unique_edges_with_lengths,
)
from dagua.layout.ops.pipelines.fmmm import (
    _graphviz_fdp_collapse_parallel_edges,
    _graphviz_tile_pack_offsets,
    _layout_ogdf_fmmm_small_fidelity,
    _ogdf_fmmm_build_hierarchy,
    _ogdf_fmmm_max_mult_iter,
    _ogdf_fmmm_nmm_repulsive_forces,
    _ogdf_fmmm_random_placement,
    _ogdf_fmmm_repulsive_forces,
    _ogdf_fmmm_update_box,
    _ogdf_maar_pack_component_transforms,
    _ogdf_maar_pack_offsets,
    _OgdfMt19937,
    layout_fmmm_pipeline,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


def test_fmmm_lower_mass_galaxy_choice_matches_reference_selection() -> None:
    """Verify reference galaxy choice samples the lowest star-mass candidate.

    Returns
    -------
    None
        The assertion validates the selected sun node for a fixed candidate set.
    """
    selectable = _RandomNodeSet.from_star_masses([5, 1, 9])

    selected = selectable.get_random_node_with_lowest_star_mass(random.Random(0), 3)

    assert selected == 1


def test_fmmm_reference_mode_uses_lower_mass_hierarchy() -> None:
    """Verify reference hierarchy construction switches to lower-mass suns.

    Returns
    -------
    None
        The assertion checks that the coarsening result differs from legacy
        higher-mass selection on a graph large enough to coarsen.
    """
    edges = torch.tensor(
        [[node for node in range(59)], [node + 1 for node in range(59)]],
        dtype=torch.long,
    )

    legacy_levels, _ = _build_hierarchy(edges, 60, seed=3)
    reference_levels, _ = _build_hierarchy(
        edges,
        60,
        seed=3,
        galaxy_choice=_GALAXY_CHOICE_LOWER,
    )

    assert not torch.equal(reference_levels[-1].edge_index, legacy_levels[-1].edge_index)


def test_fmmm_reference_mode_averages_parallel_edge_weights() -> None:
    """Verify OGDF fidelity mode does not strengthen reduced parallel edges.

    Returns
    -------
    None
        The assertion checks the base graph collapse used by fidelity mode:
        duplicate unweighted edges become one unit-strength spring.
    """
    edge_index = torch.tensor([[0, 0, 0, 1], [1, 1, 1, 2]], dtype=torch.long)

    _, legacy_lengths, legacy_weights = _unique_edges_with_lengths(edge_index, 3)
    _, reference_lengths, reference_weights = _unique_edges_with_lengths(
        edge_index,
        3,
        sum_parallel_weights=False,
    )

    assert torch.equal(legacy_lengths, reference_lengths)
    assert torch.allclose(legacy_weights, torch.tensor([3.0, 1.0]))
    assert torch.allclose(reference_weights, torch.ones(2))


def test_fmmm_force_step_reference_scaling_records_damped_movement() -> None:
    """Verify OGDF-style force scaling records movement state for damping.

    Returns
    -------
    None
        The assertion validates the reference force path updates both positions
        and oscillation bookkeeping.
    """
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    state = SolveState(
        pos=torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32),
        ideal_length=1.0,
    )
    step = FMMMForceStep(
        edge_index=edge_index,
        ideal_length=1.0,
        edge_lengths=torch.ones(1, dtype=torch.float32),
        ogdf_force_scaling=True,
    )

    result = step.apply(LayoutProblem(edge_index=edge_index, num_nodes=2), state, RuntimeContext())

    assert result.pos is not None
    assert not torch.equal(result.pos, torch.tensor([[0.0, 0.0], [2.0, 0.0]]))
    assert "fmmm_previous_displacement" in result.extras
    assert "fmmm_last_avg_force_norm" in result.extras


def test_fmmm_reference_mode_returns_finite_positions() -> None:
    """Verify the public reference mode produces finite coordinates.

    Returns
    -------
    None
        The assertion validates shape and finiteness for a small graph.
    """
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    positions = layout_fmmm_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        steps=4,
        seed=11,
        reference_mode=True,
    )

    assert positions.shape == (4, 2)
    assert torch.isfinite(positions).all()


def test_fmmm_fidelity_mode_alias_returns_finite_positions() -> None:
    """Verify the evaluation alias enables the OGDF reference path.

    Returns
    -------
    None
        The assertion validates the public ``fidelity_mode`` alias used by
        competitor defaults.
    """
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)

    positions = layout_fmmm_pipeline(
        edge_index=edge_index,
        num_nodes=3,
        steps=4,
        seed=7,
        fidelity_mode=True,
    )

    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()


def test_fmmm_ogdf_max_mult_iter_matches_linearly_decreasing_reference() -> None:
    """Verify OGDF linearly decreasing multilevel iteration budgets.

    Returns
    -------
    None
        The assertion checks coarsest, middle, finest, and small-level floors.
    """
    assert _ogdf_fmmm_max_mult_iter(0, 0, 1000, 20) == 200
    assert _ogdf_fmmm_max_mult_iter(2, 2, 1000, 20) == 200
    assert _ogdf_fmmm_max_mult_iter(1, 2, 1000, 20) == 110
    assert _ogdf_fmmm_max_mult_iter(0, 2, 1000, 20) == 20
    assert _ogdf_fmmm_max_mult_iter(0, 2, 500, 20) == 100


def test_fmmm_fidelity_mode_uses_multilevel_driver_above_coarse_target() -> None:
    """Verify fidelity dispatch no longer uses the single-level path for large graphs.

    Returns
    -------
    None
        The assertion validates finite output and a different result from the
        legacy helper on a graph that crosses OGDF's coarsening threshold.
    """
    edge_index = torch.tensor(
        [[node for node in range(59)], [node + 1 for node in range(59)]],
        dtype=torch.long,
    )

    positions = layout_fmmm_pipeline(
        edge_index=edge_index,
        num_nodes=60,
        steps=4,
        seed=7,
        fidelity_mode=True,
    )
    legacy_positions = _layout_ogdf_fmmm_small_fidelity(
        edge_index=edge_index,
        num_nodes=60,
        steps=4,
        seed=7,
        device=torch.device("cpu"),
    ).to(dtype=torch.float32)

    assert positions.shape == (60, 2)
    assert torch.isfinite(positions).all()
    assert not torch.allclose(positions, legacy_positions)


def test_fmmm_ogdf_hierarchy_uses_reseeded_mt19937_galaxy_stream() -> None:
    """Pin OGDF solar-system selection and coarse edge insertion order.

    Returns
    -------
    None
        The assertions pin the private per-level MT19937 stream and the
        first-surviving-edge behavior of ``makeSimpleUndirected``.
    """
    edge_index = torch.tensor(
        [[node for node in range(59)], [node + 1 for node in range(59)]],
        dtype=torch.long,
    )

    levels, steps = _ogdf_fmmm_build_hierarchy(edge_index, 60, seed=3)

    assert [level.num_nodes for level in levels] == [60, 17]
    assert steps[0].mapping[:20] == [
        0,
        0,
        0,
        13,
        13,
        13,
        3,
        3,
        3,
        3,
        7,
        7,
        7,
        7,
        2,
        2,
        2,
        11,
        11,
        11,
    ]
    assert levels[1].edges[:3] == [(0, 13), (13, 3), (3, 7)]


def test_fmmm_nmm_force_approximates_exact_force_at_dispatch_boundary() -> None:
    """Validate the ported NMM force at OGDF's 175-node boundary.

    Returns
    -------
    None
        The relative force error stays below the order-four expansion budget.
    """
    positions = _ogdf_fmmm_random_placement(175, seed=9)
    boxlength, down_left = _ogdf_fmmm_update_box(positions)

    approximate = torch.tensor(
        _ogdf_fmmm_nmm_repulsive_forces(
            positions,
            boxlength,
            down_left,
            _OgdfMt19937(9),
        ),
        dtype=torch.float64,
    )
    exact = torch.tensor(_ogdf_fmmm_repulsive_forces(positions), dtype=torch.float64)
    relative_error = torch.linalg.norm(approximate - exact) / torch.linalg.norm(exact)

    assert float(relative_error.item()) < 1.0e-3


def test_fmmm_fidelity_mode_decomposes_disconnected_large_graphs() -> None:
    """Verify OGDF fidelity mode lays out disconnected graphs component-wise.

    Returns
    -------
    None
        The assertion guards the benchmark ``random_dag_50`` shape, where many
        isolated nodes previously forced one edgeless coarse graph and produced
        coordinates orders of magnitude larger than OGDF's packed output.
    """
    edge_index = torch.tensor(
        [[node for node in range(50, 96)], [node + 1 for node in range(50, 96)]],
        dtype=torch.long,
    )

    positions = layout_fmmm_pipeline(
        edge_index=edge_index,
        num_nodes=97,
        steps=4,
        seed=7,
        fidelity_mode=True,
    )
    centered = positions.to(dtype=torch.float64) - positions.to(dtype=torch.float64).mean(dim=0)
    dispersion = torch.sqrt(centered.square().sum(dim=1).mean())

    assert positions.shape == (97, 2)
    assert torch.isfinite(positions).all()
    assert float(dispersion.item()) < 1_000.0


def test_fmmm_graphviz_tile_pack_offsets_match_pack_c_golden_vectors() -> None:
    """Verify the bbox tile packer against hand-captured pack.c vectors.

    Returns
    -------
    None
        The assertion validates offsets from Graphviz's ``computeStep``,
        ``genBox``, perimeter sort, and spiral placement logic.
    """
    boxes = [
        (0.0, 0.0, 10.0, 4.0),
        (0.0, 0.0, 6.0, 6.0),
        (0.0, 0.0, 3.0, 12.0),
    ]

    offsets = _graphviz_tile_pack_offsets(boxes)

    assert offsets == [(7.0, -7.0), (7.0, 6.0), (-5.0, -10.0)]


def test_fmmm_graphviz_tile_pack_offsets_handle_nonzero_box_origins() -> None:
    """Verify pack translations preserve Graphviz's rounded lower-left shift.

    Returns
    -------
    None
        The assertion covers non-zero component boxes, including negative
        lower-left coordinates.
    """
    boxes = [
        (-2.0, -1.0, 7.0, 3.0),
        (4.0, -3.0, 8.0, 10.0),
        (0.0, 0.0, 2.0, 2.0),
    ]

    offsets = _graphviz_tile_pack_offsets(boxes)

    assert offsets == [(9.0, -6.0), (-10.0, -7.0), (7.0, 6.0)]


def test_fmmm_ogdf_maar_pack_offsets_match_best_fit_rows() -> None:
    """Verify FMMM disconnected packing follows OGDF MAARPacking rows.

    Returns
    -------
    None
        The assertion covers decreasing-height presort plus the Best-Fit
        shortest-row insertion rule from OGDF ``MAARPacking.cpp``.
    """
    boxes = [
        (2.0, 3.0, 12.0, 33.0),
        (-5.0, 7.0, 45.0, 17.0),
        (0.0, 0.0, 12.0, 12.0),
    ]

    offsets = _ogdf_maar_pack_offsets(boxes)

    assert offsets == [(-2.0, -3.0), (5.0, 23.0), (10.0, 9.0)]


def test_fmmm_ogdf_maar_pack_reports_tipped_components() -> None:
    """Verify MAARPacking exposes OGDF ``NoGrowingRow`` tip-over decisions.

    Returns
    -------
    None
        The assertion guards the export transform needed when MAARPacking tips
        a component rectangle before final placement.
    """
    boxes = [
        (0.0, 0.0, 10.0, 10.0),
        (0.0, 0.0, 8.0, 4.0),
        (0.0, 0.0, 7.0, 4.0),
    ]

    transforms = _ogdf_maar_pack_component_transforms(boxes)

    assert transforms == [(0.0, 0.0, False), (14.0, 1.0, True), (0.0, 10.0, False)]


def test_fmmm_graphviz_fdp_collapses_parallel_edges_to_one_spring() -> None:
    """Verify FDP fidelity uses Graphviz's single spring for multi-edges.

    Returns
    -------
    None
        The assertion checks duplicate undirected pairs are not summed, matching
        Graphviz's multiedge skip in the FDP preprocessing path.
    """
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 1, 0, 3]], dtype=torch.long)
    edge_weights = torch.tensor([2.0, 5.0, 7.0, 11.0], dtype=torch.float64)

    collapsed_edges, collapsed_weights = _graphviz_fdp_collapse_parallel_edges(
        edge_index,
        edge_weights,
    )

    assert torch.equal(collapsed_edges, torch.tensor([[0, 2], [1, 3]], dtype=torch.long))
    assert collapsed_weights is not None
    assert torch.allclose(collapsed_weights, torch.tensor([2.0, 11.0], dtype=torch.float64))


def test_fmmm_fidelity_mode_packs_disconnected_components_only_in_fidelity() -> None:
    """Verify fdp tile packing is gated behind fidelity mode.

    Returns
    -------
    None
        The assertion checks that default behavior is preserved while fidelity
        mode creates separated component boxes.
    """
    edge_index = torch.tensor([[0, 1, 3], [1, 2, 4]], dtype=torch.long)

    default_positions = layout_fmmm_pipeline(
        edge_index=edge_index,
        num_nodes=6,
        steps=4,
        seed=13,
    )
    fidelity_positions = layout_fmmm_pipeline(
        edge_index=edge_index,
        num_nodes=6,
        steps=4,
        seed=13,
        fidelity_mode=True,
    )

    assert default_positions.shape == fidelity_positions.shape == (6, 2)
    assert torch.isfinite(fidelity_positions).all()
    assert not torch.equal(default_positions, fidelity_positions)

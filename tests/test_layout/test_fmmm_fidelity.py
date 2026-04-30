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
from dagua.layout.ops.pipelines.fmmm import layout_fmmm_pipeline
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

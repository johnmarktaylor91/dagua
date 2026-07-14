"""Tests for WebCola VPSC and descent primitives."""

from __future__ import annotations

import math

import torch

from dagua.layout.ops.webcola import (
    project_webcola_constraints,
    solve_vpsc_1d,
    webcola_distance_matrix,
    webcola_initial_positions,
)


def test_solve_vpsc_1d_satisfies_separation_with_minimal_motion() -> None:
    """VPSC should separate two overlapping variables symmetrically."""
    projected = solve_vpsc_1d([0.0, 0.0], [(0, 1, 10.0, False)])

    assert math.isclose(projected[0], -5.0, abs_tol=1.0e-12)
    assert math.isclose(projected[1], 5.0, abs_tol=1.0e-12)
    assert projected[1] - projected[0] >= 10.0


def test_solve_vpsc_1d_honors_equality_constraint() -> None:
    """VPSC equality constraints should collapse variables to a weighted mean."""
    projected = solve_vpsc_1d([0.0, 8.0], [(0, 1, 0.0, True)])

    assert math.isclose(projected[0], 4.0, abs_tol=1.0e-12)
    assert math.isclose(projected[1], 4.0, abs_tol=1.0e-12)


def test_project_webcola_constraints_filters_by_axis() -> None:
    """Projection should apply only constraints declared for the requested axis."""
    constraints = [
        {"axis": "x", "left": 0, "right": 1, "gap": 6.0},
        {"axis": "y", "left": 0, "right": 1, "gap": 100.0},
    ]
    projected = project_webcola_constraints([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], constraints, "x")

    assert math.isclose(projected[1] - projected[0], 6.0, abs_tol=1.0e-12)


def test_webcola_distance_matrix_uses_infinite_disconnected_pairs() -> None:
    """WebCola shortest paths leave disconnected pairs at infinity."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    distances = webcola_distance_matrix(edge_index, num_nodes=4, link_distance=20.0)

    assert distances[0][2] == 40.0
    assert math.isinf(distances[0][3])


def test_webcola_initial_positions_are_deterministic() -> None:
    """The adapter and native pipeline share a deterministic initial circle."""
    first = webcola_initial_positions(5, 20.0)
    second = webcola_initial_positions(5, 20.0)

    assert torch.equal(first, second)
    assert first.shape == (5, 2)

"""Tests for layout equivalence metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dagua.eval.equivalence_metrics import (
    anisotropic_procrustes,
    automorphism_aligned_procrustes,
    component_aligned_procrustes,
    compute_equivalence_metrics,
    normalized_stress,
    spectrum_distance_diagnostic,
)


def _complete_graph_edge_index(num_nodes: int) -> torch.Tensor:
    """Build an undirected complete graph edge index.

    Parameters
    ----------
    num_nodes : int
        Number of complete-graph vertices.

    Returns
    -------
    torch.Tensor
        Edge index with shape ``[2, E]``.
    """
    edges = [
        (source, target) for source in range(num_nodes) for target in range(source + 1, num_nodes)
    ]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_automorphism_aligned_procrustes_removes_complete_graph_relabeling() -> None:
    """Automorphism alignment should remove discrete node relabeling artifacts."""
    pytest.importorskip("igraph")
    edge_index = _complete_graph_edge_index(5)
    reference = np.array(
        [
            [0.0, 0.0],
            [1.7, -0.2],
            [-0.3, 2.1],
            [2.8, 1.3],
            [-1.2, -0.7],
        ],
        dtype=np.float64,
    )
    permutation = np.array([2, 4, 0, 3, 1], dtype=np.int64)
    dagua = reference[permutation]

    metrics = automorphism_aligned_procrustes(dagua, reference, edge_index)

    assert metrics["aut_group_size"] == math.factorial(5)
    assert metrics["plain_procrustes_rmsd"] > 0.1
    assert metrics["aut_procrustes_rmsd"] < 1.0e-12


def test_spectrum_diagnostic_matches_rotated_layout() -> None:
    """Distance and Gram diagnostics should be invariant under rotation."""
    reference = np.array(
        [[0.0, 0.0], [1.0, 0.2], [0.4, 1.4], [-0.5, 0.7], [1.7, 1.1]],
        dtype=np.float64,
    )
    theta = np.deg2rad(37.0)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    rotated = reference @ rotation.T

    metrics = spectrum_distance_diagnostic(rotated, reference)

    assert metrics["dist_matrix_corr"] == pytest.approx(1.0, abs=1.0e-12)
    assert metrics["dist_matrix_rel_frob"] == pytest.approx(0.0, abs=1.0e-12)
    assert metrics["gram_eig_max_absdiff"] == pytest.approx(0.0, abs=1.0e-12)


def test_normalized_stress_is_zero_for_path_at_graph_distances() -> None:
    """A path laid out at exact shortest-path distances should have zero stress."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)

    stress = normalized_stress(positions, edge_index)

    assert stress == pytest.approx(0.0, abs=1.0e-12)


def test_identity_layout_pair_is_practically_equivalent() -> None:
    """Identical layouts should produce perfect equivalence signals."""
    pytest.importorskip("igraph")
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    positions = np.array(
        [[0.0, 0.0], [1.0, 0.2], [2.0, 0.1], [3.0, 0.4]],
        dtype=np.float64,
    )

    metrics = compute_equivalence_metrics(positions, positions.copy(), edge_index)

    assert metrics.plain_procrustes_rmsd == pytest.approx(0.0, abs=1.0e-12)
    assert metrics.aut_procrustes_rmsd == pytest.approx(0.0, abs=1.0e-12)
    assert metrics.stress_rel_delta == pytest.approx(0.0, abs=1.0e-12)
    assert metrics.neighborhood_preservation_delta == pytest.approx(0.0, abs=1.0e-12)
    assert metrics.dist_matrix_corr == pytest.approx(1.0, abs=1.0e-12)
    assert metrics.gram_eig_max_absdiff == pytest.approx(0.0, abs=1.0e-12)
    assert metrics.verdict == "PRACTICALLY_EQUIVALENT"


def test_component_aligned_procrustes_removes_component_packing_drift() -> None:
    """Per-component alignment should ignore disconnected placement drift."""
    pytest.importorskip("igraph")
    edge_index = torch.tensor(
        [[0, 1, 0, 3, 4, 3], [1, 2, 2, 4, 5, 5]],
        dtype=torch.long,
    )
    reference = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.2, 0.8],
            [6.0, 0.0],
            [7.0, 0.0],
            [6.2, 0.8],
        ],
        dtype=np.float64,
    )
    dagua = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.2, 0.8],
            [2.0, 5.0],
            [2.0, 6.0],
            [1.2, 5.2],
        ],
        dtype=np.float64,
    )

    metrics = component_aligned_procrustes(dagua, reference, edge_index)
    combined = compute_equivalence_metrics(dagua, reference, edge_index)

    assert metrics["n_components"] == 2
    assert metrics["component_aligned_rmsd"] == pytest.approx(0.0, abs=1.0e-12)
    assert combined.plain_procrustes_rmsd > 0.1
    assert combined.component_aligned_rmsd == pytest.approx(0.0, abs=1.0e-12)


def test_anisotropic_procrustes_is_allowlisted_by_engine_family() -> None:
    """Sugiyama-family engines should own independent x/y scale invariance."""
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 3], [1, 2, 3, 4, 5, 5]],
        dtype=torch.long,
    )
    reference = np.array(
        [[0.0, 0.0], [1.0, 0.2], [0.3, 1.4], [1.7, 1.1], [2.4, 0.4], [1.1, 2.3]],
        dtype=np.float64,
    )
    dagua = reference * np.array([3.0, 0.25], dtype=np.float64)

    anisotropic = anisotropic_procrustes(dagua, reference)
    allowed = compute_equivalence_metrics(
        dagua,
        reference,
        edge_index,
        engine_name="classic_sugiyama_default",
    )
    blocked = compute_equivalence_metrics(
        dagua,
        reference,
        edge_index,
        engine_name="classic_fr",
    )

    assert anisotropic["anisotropic_rmsd"] == pytest.approx(0.0, abs=1.0e-12)
    assert allowed.plain_procrustes_rmsd > 0.2
    assert allowed.anisotropic_rmsd == pytest.approx(0.0, abs=1.0e-12)
    assert allowed.verdict == "PRACTICALLY_EQUIVALENT"
    assert blocked.anisotropic_rmsd is None
    assert blocked.verdict == "NOT_EQUIVALENT"

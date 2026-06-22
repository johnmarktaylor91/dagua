"""Regression tests for definitive fidelity quality-battery correctness."""

from __future__ import annotations

import numpy as np
import pytest

from dagua.eval.distributional_fidelity import prepare_graph_distances
from dagua.eval.equivalence_metrics import normalized_stress
from scripts import definitive_fidelity_analysis as analysis


def _path_payload(num_nodes: int) -> analysis.ComboPayload:
    """Build a minimal path-graph analysis payload.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the synthetic path graph.

    Returns
    -------
    scripts.definitive_fidelity_analysis.ComboPayload
        Payload with path edges and no benchmark rows.
    """
    edges = tuple((index, index + 1) for index in range(num_nodes - 1))
    return analysis.ComboPayload(
        combo_id=f"synthetic_path_{num_nodes}",
        graph="synthetic_path",
        engine="dagua",
        reference="reference",
        data_dir="",
        reimpl_rows=(),
        ref_rows=(),
        graph_edges=edges,
        graph_n_nodes=num_nodes,
        git_sha="test",
    )


def _path_distances(num_nodes: int) -> np.ndarray:
    """Compute graph distances for a synthetic path graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the synthetic path graph.

    Returns
    -------
    numpy.ndarray
        Shortest-path distance matrix with shape ``[N, N]``.
    """
    edges = np.asarray([(index, index + 1) for index in range(num_nodes - 1)], dtype=np.int64)
    return prepare_graph_distances(edges, num_nodes)


def test_battery_stress_passes_equal_quality_different_scale() -> None:
    """Scale-only layout differences should pass the strict battery stress leg."""
    payload = _path_payload(4)
    dists = _path_distances(4)
    reference = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float64)
    scaled = 7.0 * reference

    raw_stress = normalized_stress(
        scaled,
        analysis.edge_index_array(payload.graph_edges),
        all_pairs_distances=dists,
    )
    metrics = analysis.compute_mode_a_quality_battery(payload, [scaled], [reference], dists)

    assert raw_stress > 1.0
    assert metrics["battery_stress_D_mean"] == pytest.approx(0.0, abs=1.0e-12)
    assert metrics["battery_stress_R_mean"] == pytest.approx(0.0, abs=1.0e-12)
    assert metrics["quality_identical_raw"] is True


def test_battery_stress_rejects_genuinely_worse_layout_after_scale_fit() -> None:
    """Scale fitting should not hide distorted graph-distance geometry."""
    payload = _path_payload(4)
    dists = _path_distances(4)
    reference = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float64)
    distorted = np.asarray([[0.0, 0.0], [0.1, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float64)

    metrics = analysis.compute_mode_a_quality_battery(payload, [distorted], [reference], dists)

    assert metrics["battery_stress_D_mean"] > metrics["battery_stress_margin"]
    assert metrics["battery_stress_direct_equivalent"] is False
    assert metrics["quality_identical_raw"] is False

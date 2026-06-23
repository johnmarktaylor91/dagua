"""Regression tests for definitive fidelity quality-battery correctness."""

from __future__ import annotations

import numpy as np
import pytest

from dagua.eval.distributional_fidelity import prepare_graph_distances
from dagua.eval.equivalence_metrics import normalized_stress
from scripts import definitive_fidelity_analysis as analysis
from scripts import definitive_fidelity_report as report


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


def _finalized_battery_row(
    metrics: dict[str, object], combo_id: str = "synthetic"
) -> dict[str, object]:
    """Finalize one synthetic battery row through report-stage 3Q routing.

    Parameters
    ----------
    metrics : dict[str, object]
        Battery metrics emitted by the analysis stage.
    combo_id : str, default="synthetic"
        Stable row identifier.

    Returns
    -------
    dict[str, object]
        Finalized report row.
    """
    row: dict[str, object] = {
        "spec_version": report.SPEC_VERSION,
        "combo_id": combo_id,
        "graph": "synthetic_path",
        "engine": "dagua",
        "reference": "reference",
        "mode": "A",
    }
    row.update(metrics)
    return report.finalize_rows([row], report.SPEC_VERSION, include_controls=True)[0]


def _repeated_layout(layout: np.ndarray, count: int) -> list[np.ndarray]:
    """Return independent copies of a layout.

    Parameters
    ----------
    layout : numpy.ndarray
        Coordinate array with shape ``[N, 2]``.
    count : int
        Number of copies to return.

    Returns
    -------
    list[numpy.ndarray]
        Copied coordinate arrays.
    """
    return [layout.copy() for _index in range(count)]


def _random_reference_cloud(num_nodes: int, count: int) -> list[np.ndarray]:
    """Build a deliberately non-canonical stochastic reference cloud.

    Parameters
    ----------
    num_nodes : int
        Number of nodes per layout.
    count : int
        Number of layouts.

    Returns
    -------
    list[numpy.ndarray]
        Random coordinate arrays with shape ``[N, 2]``.
    """
    rng = np.random.default_rng(12345)
    return [rng.normal(size=(num_nodes, 2)).astype(np.float64) for _index in range(count)]


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


def test_neighborhood_preservation_passes_when_dagua_is_better() -> None:
    """The NP leg should be one-sided so better dagua preservation passes."""
    payload = _path_payload(20)
    dists = _path_distances(20)
    dagua = np.column_stack([np.arange(20, dtype=np.float64), np.zeros(20, dtype=np.float64)])
    interleaved_order = (0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16, 7, 17, 8, 18, 9, 19)
    interleaved = np.asarray(
        [[index, 0.0] for index in interleaved_order],
        dtype=np.float64,
    )
    metrics = analysis.quality_metric_samples(payload, [dagua], [interleaved], dists)

    np_gate = analysis.quality_np_noninferiority(
        metrics["np_d"],
        metrics["np_r"],
        analysis.QUALITY_NP_ABS_MARGIN,
    )

    assert metrics["np_d"][0] > metrics["np_r"][0]
    assert analysis.metric_equivalent(np_gate) is True
    assert np_gate["noninferior_direct"] is True


def test_chance_layout_still_fails_quality_battery() -> None:
    """Destroyed structure should stay blocked by stress even with one-sided NP."""
    payload = _path_payload(20)
    dists = _path_distances(20)
    reference = np.column_stack([np.arange(20, dtype=np.float64), np.zeros(20, dtype=np.float64)])
    chance = np.asarray(
        [
            [0.34558419, 0.82161814],
            [0.33043708, -1.30315723],
            [0.90535587, 0.44637457],
            [-0.53695324, 0.5811181],
            [0.3645724, 0.2941325],
            [0.02842224, 0.54671299],
            [-0.73645409, -0.16290995],
            [-0.48211931, 0.59884621],
            [0.03972211, -0.29245675],
            [-0.78190846, -0.25719224],
            [0.00814218, -0.27560291],
            [1.29406381, 1.00672432],
            [-2.71116248, -1.88901325],
            [-0.17477209, -0.42219041],
            [0.213643, 0.21732193],
            [2.11783876, -1.11202076],
            [-0.37760501, 2.04277161],
            [0.646703, 0.66306337],
            [-0.51400637, -1.64807517],
            [0.16746474, 0.10901409],
        ],
        dtype=np.float64,
    )

    metrics = analysis.compute_mode_a_quality_battery(payload, [chance], [reference], dists)

    assert metrics["battery_stress_direct_equivalent"] is False
    assert metrics["quality_identical_raw"] is False


def test_variance_tied_canonical_equal_quality_pair_passes_final_3q() -> None:
    """Canonical references should allow equal-quality layouts into final 3Q."""
    payload = _path_payload(20)
    dists = _path_distances(20)
    reference = np.column_stack([np.arange(20, dtype=np.float64), np.zeros(20, dtype=np.float64)])
    metrics = analysis.compute_mode_a_quality_battery(
        payload,
        _repeated_layout(3.0 * reference, 30),
        _repeated_layout(reference, 30),
        dists,
    )
    row = _finalized_battery_row(metrics, "canonical_equal")

    assert metrics["quality_battery_eligible"] is True
    assert metrics["battery_stress_ref_self_spread"] == pytest.approx(0.0)
    assert row["quality_identical"] is True
    assert row["final_rung"] == "3Q"


def test_variance_tied_canonical_worse_layout_fails_final_3q() -> None:
    """Variance-tied margins should not admit genuinely worse canonical layouts."""
    payload = _path_payload(20)
    dists = _path_distances(20)
    reference = np.column_stack([np.arange(20, dtype=np.float64), np.zeros(20, dtype=np.float64)])
    distorted = reference.copy()
    distorted[10, 0] = 0.25
    metrics = analysis.compute_mode_a_quality_battery(
        payload,
        _repeated_layout(distorted, 30),
        _repeated_layout(reference, 30),
        dists,
    )
    row = _finalized_battery_row(metrics, "canonical_worse")

    assert metrics["quality_battery_eligible"] is True
    assert metrics["battery_stress_direct_equivalent"] is False
    assert row["quality_identical"] is False
    assert row["final_rung"] != "3Q"


def test_variance_tied_stochastic_equal_pair_is_exploratory_not_final_3q() -> None:
    """Non-canonical stochastic references should be excluded from final 3Q."""
    payload = _path_payload(20)
    dists = _path_distances(20)
    reference_layouts = _random_reference_cloud(20, 30)
    metrics = analysis.compute_mode_a_quality_battery(
        payload,
        [layout.copy() for layout in reference_layouts],
        reference_layouts,
        dists,
    )
    row = _finalized_battery_row(metrics, "stochastic_equal")

    assert metrics["quality_identical_exploratory"] is True
    assert metrics["quality_battery_eligible"] is False
    assert metrics["quality_reference_plain_mean_W_R"] > 1.0
    assert row["quality_battery_tier"] == analysis.QUALITY_BATTERY_EXPLORATORY_TIER
    assert row["quality_identical"] is False
    assert row["final_rung"] != "3Q"


def test_variance_tied_chance_shuffled_reference_is_not_final_3q() -> None:
    """Shuffled-reference chance layouts must not launder into final 3Q."""
    payload = _path_payload(20)
    dists = _path_distances(20)
    reference_layouts = _random_reference_cloud(20, 30)
    rng = np.random.default_rng(777)
    chance_layouts = [
        layout[rng.permutation(layout.shape[0])].copy() for layout in reference_layouts
    ]
    metrics = analysis.compute_mode_a_quality_battery(
        payload,
        chance_layouts,
        reference_layouts,
        dists,
    )
    row = _finalized_battery_row(metrics, "chance_shuffle")

    assert metrics["quality_battery_eligible"] is False
    assert row["quality_battery_tier"] == analysis.QUALITY_BATTERY_EXPLORATORY_TIER
    assert row["quality_identical"] is False
    assert row["final_rung"] != "3Q"


def test_reference_self_split_remaps_disjoint_seed_halves() -> None:
    """Self-split controls should compare disjoint rows with matched labels."""
    rows = tuple(
        analysis.PositionRow(
            key=f"row_{seed}",
            graph="synthetic_path",
            engine="reference",
            seed=seed,
            status="ok",
            positions_file=f"positions/{seed}.pt",
            runtime_seconds=0.1,
            num_nodes=20,
        )
        for seed in range(100)
    )

    first, second = analysis.split_reference_self_rows(list(rows))

    assert [row.seed for row in first] == list(range(50))
    assert [row.seed for row in second] == list(range(50))
    assert {row.positions_file for row in first}.isdisjoint({row.positions_file for row in second})

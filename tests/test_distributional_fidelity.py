"""Regression tests for the r70 distributional fidelity statistical core."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pytest
from scipy import stats

from dagua.eval import distributional_fidelity as df
from dagua.eval.equivalence_metrics import anisotropic_procrustes
from scripts.fast_fidelity_report import procrustes_rmsd


def _rng(seed: int) -> np.random.Generator:
    """Create a deterministic test RNG.

    Parameters
    ----------
    seed : int
        Seed value.

    Returns
    -------
    numpy.random.Generator
        Deterministic generator.
    """
    return np.random.default_rng(seed)


def _clouds(rng: np.random.Generator, count: int, n_nodes: int) -> list[np.ndarray]:
    """Generate random layout clouds.

    Parameters
    ----------
    rng : numpy.random.Generator
        Deterministic generator.
    count : int
        Number of layouts.
    n_nodes : int
        Number of nodes per layout.

    Returns
    -------
    list[numpy.ndarray]
        Random ``[N, 2]`` layout list.
    """
    return [rng.normal(size=(n_nodes, 2)).astype(np.float64) for _ in range(count)]


def _circle_layouts(count: int, n_nodes: int, phase_step: float = 0.17) -> list[np.ndarray]:
    """Generate deterministic non-degenerate circle-like layouts.

    Parameters
    ----------
    count : int
        Number of layouts.
    n_nodes : int
        Number of points per layout.
    phase_step : float, default=0.17
        Phase increment between layouts.

    Returns
    -------
    list[numpy.ndarray]
        Layout list with shape ``[N, 2]`` entries.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_nodes, endpoint=False)
    layouts: list[np.ndarray] = []
    for index in range(count):
        phase = phase_step * index
        radius = 1.0 + 0.10 * np.sin((index + 1) * theta)
        layouts.append(np.column_stack((radius * np.cos(theta + phase), np.sin(theta - phase))))
    return layouts


def _manual_bh(pvals: Sequence[float]) -> np.ndarray:
    """Compute BH q-values for a small test vector.

    Parameters
    ----------
    pvals : Sequence[float]
        Raw p-values.

    Returns
    -------
    numpy.ndarray
        Manual q-values.
    """
    p_array = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(p_array)
    ranked = p_array[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(p_array)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def test_pairwise_procrustes_agrees_with_project_oracle() -> None:
    """Plain pairwise Procrustes matches the project-wide oracle on random pairs."""
    rng = _rng(1)
    layouts = _clouds(rng, 48, 25)
    matrix = df.pairwise_procrustes_matrix(layouts)
    checked = 0
    for i in range(len(layouts)):
        for j in range(len(layouts)):
            oracle = procrustes_rmsd(layouts[i], layouts[j])
            if oracle > 1.0e-4:
                assert matrix[i, j] == pytest.approx(oracle, abs=1.0e-10)
                checked += 1
    assert checked >= 1000


@pytest.mark.parametrize("distance", [0.0, 1.0e-9, 1.0e-7, 1.0e-5])
def test_pairwise_procrustes_exact_fallback_near_identical(distance: float) -> None:
    """Near-identical pairs below the cutoff use exact SVD residuals."""
    base = np.column_stack((np.linspace(-1.0, 1.0, 12), np.linspace(0.5, -0.5, 12) ** 2))
    perturb = np.zeros_like(base)
    perturb[0, 1] = distance
    layouts = [base, base + perturb]
    matrix = df.pairwise_procrustes_matrix(layouts)
    assert matrix[0, 1] == pytest.approx(procrustes_rmsd(layouts[0], layouts[1]), abs=1.0e-12)


def test_pairwise_procrustes_degenerate_collinear_mirror_and_float32() -> None:
    """Degenerate, collinear, mirrored, two-point, and float32 inputs match the oracle."""
    coincident = np.ones((5, 2), dtype=np.float64)
    collinear = np.column_stack((np.arange(5, dtype=np.float64), np.zeros(5)))
    mirrored = collinear * np.array([-1.0, 1.0])
    layouts = [coincident, collinear, mirrored, collinear.astype(np.float32)]
    five_node_matrix = df.pairwise_procrustes_matrix(layouts)
    for i, first in enumerate(layouts):
        for j, second in enumerate(layouts):
            oracle = procrustes_rmsd(
                np.asarray(first, dtype=np.float64),
                np.asarray(second, dtype=np.float64),
            )
            assert five_node_matrix[i, j] == pytest.approx(
                oracle,
                abs=1.0e-12,
            )
    two_point = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)
    two_node_layouts = [two_point, -two_point]
    two_node_matrix = df.pairwise_procrustes_matrix(two_node_layouts)
    assert two_node_matrix[0, 1] == pytest.approx(
        procrustes_rmsd(two_node_layouts[0], two_node_layouts[1]),
        abs=1.0e-12,
    )


def test_pairwise_procrustes_performance_contract() -> None:
    """The Gram path handles a 200-by-200 matrix of N=2000 layouts quickly."""
    rng = _rng(2)
    layouts = _clouds(rng, 200, 2000)
    start = time.perf_counter()
    matrix = df.pairwise_procrustes_matrix(layouts)
    elapsed = time.perf_counter() - start
    assert matrix.shape == (200, 200)
    assert elapsed < 10.0


def test_energy_distance_matches_hand_computed_toy_case() -> None:
    """The Mode A U-statistic excludes diagonals exactly as specified."""
    w_d = np.array([[0.0, 1.0], [1.0, 0.0]])
    w_r = np.array([[0.0, 2.0], [2.0, 0.0]])
    cross = np.array([[10.0, 4.0], [6.0, 10.0]])
    record = df._mode_a_energy_stats(w_d, w_r, cross)  # noqa: SLF001
    assert record["E"] == pytest.approx((2.0 * 5.0) - 1.0 - 2.0)


def test_synthetic_same_distribution_equivalent_shifted_scaled_not_equivalent() -> None:
    """Split calibration accepts same clouds and rejects shifted/scaled clouds."""
    rng = _rng(3)
    reference = _circle_layouts(34, 20)
    same = [layout + rng.normal(scale=0.01, size=layout.shape) for layout in reference]
    shifted = [layout * np.array([2.0, 0.35]) + np.array([0.2, 0.0]) for layout in reference]
    same_record = df.analyze_mode_a(same, reference, _rng(4))
    shifted_record = df.analyze_mode_a(shifted, reference, _rng(5))
    assert same_record["dist_equivalent"] is True
    assert shifted_record["dist_equivalent"] is False


def test_synthetic_seed_tracking_detected() -> None:
    """Matched noisy copies produce low tracking ratio and tiny tracking p-value."""
    rng = _rng(6)
    reference = _clouds(rng, 32, 18)
    dagua = [layout + rng.normal(scale=0.01, size=layout.shape) for layout in reference]
    record = df.analyze_mode_a(dagua, reference, _rng(7))
    assert record["track_ratio"] <= 0.5
    assert record["p_track"] < 0.001


def test_permutation_p_is_not_extreme_under_true_null() -> None:
    """Paired-swap p-values are non-extreme for an exchangeable null sample."""
    rng = _rng(8)
    pvals = []
    for _case in range(8):
        layouts = _clouds(rng, 24, 12)
        matrix = df.pairwise_procrustes_matrix(layouts)
        n = len(layouts) // 2
        stats_record = df._mode_a_energy_stats(  # noqa: SLF001
            matrix[:n, :n],
            matrix[n:, n:],
            matrix[:n, n:],
        )
        pvals.append(df._paired_swap_pvalue(matrix, stats_record["E"], rng))  # noqa: SLF001
    assert stats.kstest(pvals, "uniform").pvalue > 0.01


def test_conformal_p_exactness_symmetric_score_toy_case() -> None:
    """Mode B conformal p uses the symmetric augmented-score formula."""
    layouts = _circle_layouts(5, 9)
    reference = layouts[0] * np.array([1.2, 0.8])
    record = df.analyze_mode_b(layouts, reference, _rng(9))
    matrix = df.pairwise_procrustes_matrix([*layouts, reference])
    w_d = matrix[:5, :5]
    d_r = matrix[:5, 5]
    scores_d = (w_d.sum(axis=1) + d_r) / 5.0
    score_r = float(np.mean(d_r))
    expected = (1.0 + np.count_nonzero(scores_d >= score_r)) / 6.0
    assert record["p_typ"] == pytest.approx(expected)


def test_near_uniform_cloud_flags_typicality_uninformative() -> None:
    """A near-uniform Mode B cloud voids the typicality verdict."""
    rng = _rng(10)
    layouts = _clouds(rng, 35, 25)
    reference = rng.normal(size=(25, 2))
    record = df.analyze_mode_b(layouts, reference, _rng(11))
    assert record["plain_mean_W_D"] / np.sqrt(2.0) > 0.85
    assert record["typicality_uninformative"] is True


def test_point_mass_mode_b_uses_near_deterministic_route() -> None:
    """Point-mass Mode B clouds skip conformal typicality and use d_R."""
    layout = np.column_stack((np.linspace(0.0, 1.0, 8), np.linspace(1.0, 0.0, 8)))
    record = df.analyze_mode_b([layout.copy() for _ in range(30)], layout.copy(), _rng(12))
    assert record["near_deterministic"] is True
    assert record["typicality_skipped"] is True
    assert record["ref_typical"] is True


def test_mode_b_reference_equal_to_one_draw_is_ref_typical() -> None:
    """A deterministic reference equal to one D draw is typical in an informative cloud."""
    rng = _rng(13)
    layouts = _clouds(rng, 31, 16)
    record = df.analyze_mode_b(layouts, layouts[7].copy(), _rng(14))
    assert record["ref_typical"] is True
    assert record["p_typ"] > 0.05


def test_ladder_tracking_shifted_annotation_and_fallthroughs() -> None:
    """The verdict ladder preserves tracking annotations and guard fall-throughs."""
    rung, annotations = df.assign_rung(
        {
            "mode": "A",
            "dist_equivalent": False,
            "q_track": 0.01,
            "track_ratio": 0.2,
            "quality_equivalent": False,
        }
    )
    assert rung == "4"
    assert "TRACKING_BUT_SHIFTED" in annotations
    rung, annotations = df.assign_rung(
        {"mode": "A", "one_sided_degenerate": True, "quality_equivalent": True}
    )
    assert rung == "3"
    assert "one_sided_degenerate" in annotations
    assert (
        df.assign_rung({"mode": "A", "near_deterministic": True, "mean_diag_B": 1.0e-4})[0] == "1"
    )
    assert (
        df.assign_rung(
            {
                "mode": "A",
                "near_deterministic": True,
                "mean_diag_B": 2.0e-3,
                "quality_equivalent": True,
            }
        )[0]
        == "3"
    )


def test_anisotropic_symmetric_distance_matches_toolkit_directed_residuals() -> None:
    """Free-aspect distance is symmetric and averages toolkit directed residuals."""
    base = np.column_stack((np.linspace(-1.0, 1.0, 9), np.sin(np.linspace(0.0, 2.0, 9))))
    stretched = base * np.array([2.5, 0.4]) + np.array([3.0, -2.0])
    matrix = df.pairwise_procrustes_matrix([base, stretched], free_aspect=True)
    forward = anisotropic_procrustes(base, stretched)["anisotropic_rmsd"]
    backward = anisotropic_procrustes(stretched, base)["anisotropic_rmsd"]
    assert matrix[0, 1] == pytest.approx(matrix[1, 0])
    assert matrix[0, 1] == pytest.approx(0.5 * (forward + backward))


def test_stress_utilities_tost_and_bh() -> None:
    """Stress helpers, TOST branches, and BH adjustment follow registered formulas."""
    edges = np.array([[0, 1], [1, 2], [3, 4]])
    dists = df.prepare_graph_distances(edges, 5)
    pairs = df.sample_pairs(dists, "toy")
    assert {tuple(pair) for pair in pairs.tolist()} == {(0, 1), (0, 2), (1, 2), (3, 4)}
    layout = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert df.stress_per_layout(layout, pairs, dists) == pytest.approx(0.0)
    assert df.paired_tost(np.zeros(5), 1.0e-6)["degenerate_sd"] is True
    assert df.one_sample_tost(np.ones(5), 1.0, 1.0e-6)["equivalent_direct"] is True
    pvals = [0.01, 0.04, 0.03, np.nan]
    qvals = df.bh_fdr(pvals)
    assert qvals[:3] == pytest.approx(_manual_bh(pvals[:3]))
    assert np.isnan(qvals[3])

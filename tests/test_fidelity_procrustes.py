"""Tests for Group A: Procrustes within-vs-between fix and TOST routing."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from fidelity_analysis import (  # noqa: E402
    ALL_QUALITY_METRICS,
    MIN_STOCHASTIC_SEEDS,
    QUALITY_METRICS,
    LayoutRecord,
    PairwiseComparison,
    compute_pairwise_comparisons,
    fidelity_procrustes,
    finalize_group_row,
    initialize_metric_columns,
    tost_pvalue,
)


def _build_layouts(
    transform: Sequence[Sequence[float]],
    noise_scale: float,
    n_per_side: int,
    *,
    side: str,
    n_nodes: int = 24,
    rng_seed: int = 0,
) -> list[LayoutRecord]:
    """Build synthetic layout samples with a controllable shape transform.

    Parameters
    ----------
    transform : Sequence[Sequence[float]]
        Affine transform applied to the shared base skeleton.
    noise_scale : float
        IID Gaussian noise standard deviation.
    n_per_side : int
        Number of layout samples to generate.
    side : str
        Layout side label, either ``"orig"`` or ``"reimpl"``.
    n_nodes : int, optional
        Number of graph nodes in the synthetic skeleton.
    rng_seed : int, optional
        Seed for the NumPy RNG.

    Returns
    -------
    list[LayoutRecord]
        Synthetic layouts suitable for pairwise Procrustes comparisons.
    """
    rng = np.random.default_rng(rng_seed)
    x_coords = np.linspace(-1.0, 1.0, n_nodes, dtype=np.float32)
    y_coords = np.sin(np.linspace(0.0, 3.0 * np.pi, n_nodes, dtype=np.float32))
    base_skeleton = np.stack([x_coords, y_coords], axis=1)
    transform_array = np.asarray(transform, dtype=np.float32)

    layouts: list[LayoutRecord] = []
    for seed in range(n_per_side):
        noise = rng.normal(0.0, noise_scale, size=(n_nodes, 2)).astype(np.float32)
        positions = torch.from_numpy(base_skeleton @ transform_array.T + noise)
        layouts.append(
            LayoutRecord(
                graph_name="fixture_graph",
                variant_id="fixture_variant",
                side=side,
                seed=seed,
                runtime_seconds=1.0,
                positions=positions,
                metrics={},
            )
        )
    return layouts


def _pairwise_rmsd(
    first: Sequence[LayoutRecord],
    second: Sequence[LayoutRecord],
    comparison_type: str,
) -> np.ndarray:
    """Return Procrustes RMSD values for one pairwise comparison family.

    Parameters
    ----------
    first : Sequence[LayoutRecord]
        First layout collection.
    second : Sequence[LayoutRecord]
        Second layout collection.
    comparison_type : str
        Pairwise family label forwarded to the helper.

    Returns
    -------
    numpy.ndarray
        Procrustes RMSD values as ``float64``.
    """
    comparisons: list[PairwiseComparison] = compute_pairwise_comparisons(
        first,
        second,
        comparison_type,
    )
    return np.asarray([item.procrustes_rmsd for item in comparisons], dtype=np.float64)


def _base_stochastic_row() -> dict[str, object]:
    """Build a minimal stochastic per-graph row for verdict testing.

    Returns
    -------
    dict[str, object]
        Row payload compatible with ``finalize_group_row``.
    """
    row: dict[str, object] = {
        "structural_note": "none",
        "_variant_is_stochastic": True,
        "num_orig_seeds": MIN_STOCHASTIC_SEEDS,
        "num_reimpl_seeds": MIN_STOCHASTIC_SEEDS,
        "reflected": False,
        "runtime_ratio": 1.0,
        "scale_ratio_mean": 1.0,
        "max_node_displacement": 0.1,
        "within_vs_between_pvalue": 0.01,
        "verdict": "insufficient_data",
        "anomaly_reason": "",
    }
    initialize_metric_columns(row)
    return row


def test_procrustes_known_good_equivalent() -> None:
    """Equivalent fixtures should pass Procrustes TOST and strong verdict routing."""
    orig = _build_layouts(((1.0, 0.0), (0.0, 1.0)), 0.01, 10, side="orig", rng_seed=1)
    reimpl = _build_layouts(((1.0, 0.0), (0.0, 1.0)), 0.01, 10, side="reimpl", rng_seed=2)

    first_pair_rmsd, _, _, _ = fidelity_procrustes(orig[0].positions, reimpl[0].positions)
    within_orig = _pairwise_rmsd(orig, orig, "orig-orig")
    between = _pairwise_rmsd(orig, reimpl, "orig-reimpl")
    margin = max(float(np.std(within_orig, ddof=1)), 1e-6)
    pvalue = tost_pvalue(within_orig, between, margin)

    assert first_pair_rmsd < 0.02
    assert np.mean(between) < 1.25 * np.mean(within_orig)
    assert pvalue < 0.05

    row = _base_stochastic_row()
    row["procrustes_tost_pvalue_0_5x_bh"] = 0.01
    row["procrustes_tost_pvalue_1x_bh"] = 0.01
    row["procrustes_tost_pvalue_2x_bh"] = 0.01
    for metric_name in ALL_QUALITY_METRICS:
        row[f"{metric_name}_tost_pvalue_1x_bh"] = 0.01
        row[f"{metric_name}_regression_pct"] = 0.0
    finalize_group_row(row)
    assert row["verdict"] == "strong_equivalent"


def test_procrustes_known_bad_divergent() -> None:
    """A sheared reimplementation should fail equivalence and route to divergent."""
    orig = _build_layouts(((1.0, 0.0), (0.0, 1.0)), 0.01, 10, side="orig", rng_seed=1)
    reimpl = _build_layouts(((1.0, 0.35), (0.0, 1.0)), 0.01, 10, side="reimpl", rng_seed=2)

    within_orig = _pairwise_rmsd(orig, orig, "orig-orig")
    between = _pairwise_rmsd(orig, reimpl, "orig-reimpl")
    margin = max(float(np.std(within_orig, ddof=1)), 1e-6)
    pvalue = tost_pvalue(within_orig, between, margin)

    assert np.mean(between) > 5.0 * np.mean(within_orig)
    assert pvalue > 0.95

    row = _base_stochastic_row()
    row["procrustes_tost_pvalue_1x_bh"] = 0.5
    row["procrustes_tost_pvalue_2x_bh"] = 0.5
    for metric_name in QUALITY_METRICS:
        row[f"{metric_name}_tost_pvalue_1x_bh"] = 0.5
    finalize_group_row(row)
    assert row["verdict"] == "divergent"


def test_procrustes_pooled_within_regression() -> None:
    """The orig-only baseline must detect separation that pooled-within can hide."""
    orig = _build_layouts(((1.0, 0.0), (0.0, 1.0)), 0.01, 10, side="orig", rng_seed=1)
    reimpl = _build_layouts(((1.0, 0.08), (0.0, 1.0)), 0.05, 10, side="reimpl", rng_seed=2)

    within_orig = _pairwise_rmsd(orig, orig, "orig-orig")
    within_reimpl = _pairwise_rmsd(reimpl, reimpl, "reimpl-reimpl")
    between = _pairwise_rmsd(orig, reimpl, "orig-reimpl")
    pooled_within = np.concatenate([within_orig, within_reimpl])

    fixed_pvalue = float(mannwhitneyu(between, within_orig, alternative="greater").pvalue)
    pooled_pvalue = float(mannwhitneyu(between, pooled_within, alternative="greater").pvalue)

    assert np.mean(between) > 4.0 * np.mean(within_orig)
    assert np.mean(pooled_within) > 3.0 * np.mean(within_orig)
    assert fixed_pvalue < 0.05
    assert pooled_pvalue > 0.05
    assert pooled_pvalue > fixed_pvalue

    row = _base_stochastic_row()
    row["procrustes_tost_pvalue_1x_bh"] = 0.25
    row["procrustes_tost_pvalue_2x_bh"] = 0.01
    metric_names = list(QUALITY_METRICS)
    row[f"{metric_names[0]}_tost_pvalue_1x_bh"] = 0.01
    row[f"{metric_names[1]}_tost_pvalue_1x_bh"] = 0.3
    row[f"{metric_names[2]}_tost_pvalue_1x_bh"] = 0.4
    finalize_group_row(row)
    assert row["verdict"] == "partial_match"
    assert math.isfinite(fixed_pvalue)

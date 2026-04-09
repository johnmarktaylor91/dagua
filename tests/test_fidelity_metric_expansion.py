"""Tests for Group B metric expansion and Welch-test columns."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fidelity_analysis import (  # noqa: E402
    ALL_QUALITY_METRICS,
    QUALITY_METRICS,
    SAMPLED_QUALITY_METRICS,
    metric_test_columns,
    per_graph_fieldnames,
)


def test_quality_metrics_expanded() -> None:
    """QUALITY_METRICS must include the added quick-surface metrics."""
    expected = {
        "aspect_ratio",
        "dag_consistency",
        "edge_length_cv",
        "edge_straightness_mean_deg",
        "depth_spearman_rho",
        "overlap_count",
    }
    assert set(QUALITY_METRICS) == expected


def test_sampled_quality_metrics() -> None:
    """SAMPLED_QUALITY_METRICS must expose the sampled metric pair."""
    assert set(SAMPLED_QUALITY_METRICS) == {"sampled_stress", "crossing_rate"}


def test_metric_test_columns_include_welch_for_all_metrics() -> None:
    """Each metric-specific column bundle should include Welch raw and BH fields."""
    for metric_name in ALL_QUALITY_METRICS:
        columns = metric_test_columns(metric_name)
        assert f"{metric_name}_welch_pvalue_raw" in columns
        assert f"{metric_name}_welch_pvalue_bh" in columns


def test_per_graph_fieldnames_include_all_welch_columns() -> None:
    """The per-graph CSV schema must surface Welch columns for every metric."""
    columns = per_graph_fieldnames()
    welch_columns = [column for column in columns if "welch_pvalue" in column]
    assert len(welch_columns) == 2 * len(ALL_QUALITY_METRICS)

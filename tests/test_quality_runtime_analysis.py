"""Tests for the quality/runtime analysis pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pandas as pd
import pytest
import torch

from dagua.graph import DaguaGraph

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from quality_runtime_analysis import (  # noqa: E402
    ALL_ANALYSIS_METRICS,
    METRIC_HIGHER_IS_BETTER,
    QR_QUICK_METRICS,
    QR_SAMPLED_METRICS,
    SUMMARY_COLUMNS,
    compute_cache_key,
    compute_family_summary,
    compute_pareto_front,
    derive_graph_family,
    extract_best_of_breed,
    extract_dagua_default_insights,
    load_benchmark_records,
    run_analysis,
    score_engines_on_graph,
)


def make_summary_row(
    *,
    graph_family: str,
    metric_name: str,
    engine_name: str,
    engine_family: str,
    higher_is_better: bool,
    metric_median: float,
    median_rel_best: float,
    median_runtime_rel_fastest: float,
    graphs_in_family_total: int = 3,
    graphs_in_family_available: int = 3,
    graphs_scheduled: int = 3,
    graphs_covered: int = 3,
    coverage_ratio: float = 1.0,
    graphs_ranked: int = 3,
    scorecard_eligible: bool = True,
    metric_p25: float | None = None,
    metric_p75: float | None = None,
    median_graph_rank: float = 1.0,
    win_rate: float = 1.0,
    top3_rate: float = 1.0,
) -> Dict[str, Any]:
    """Build one synthetic family-summary row.

    Parameters
    ----------
    graph_family : str
        Graph family label.
    metric_name : str
        Metric identifier.
    engine_name : str
        Engine name.
    engine_family : str
        Engine family label.
    higher_is_better : bool
        Metric direction.
    metric_median : float
        Median raw metric value.
    median_rel_best : float
        Median rel-best score.
    median_runtime_rel_fastest : float
        Median runtime relative to the fastest engine.
    graphs_in_family_total : int, optional
        Total graph count in the family.
    graphs_in_family_available : int, optional
        Graph count with at least one successful engine.
    graphs_scheduled : int, optional
        Number of graphs scheduled for the engine.
    graphs_covered : int, optional
        Number of graphs successfully covered by the engine.
    coverage_ratio : float, optional
        Successful coverage ratio.
    graphs_ranked : int, optional
        Number of graphs contributing to the metric ranking.
    scorecard_eligible : bool, optional
        Whether the row passes the scorecard gate.
    metric_p25 : float | None, optional
        Lower quartile.
    metric_p75 : float | None, optional
        Upper quartile.
    median_graph_rank : float, optional
        Median graph rank.
    win_rate : float, optional
        Fraction of graph wins.
    top3_rate : float, optional
        Fraction of top-3 finishes.

    Returns
    -------
    Dict[str, Any]
        Summary row payload.
    """
    return {
        "graph_family": graph_family,
        "metric_name": metric_name,
        "engine_name": engine_name,
        "engine_family": engine_family,
        "higher_is_better": higher_is_better,
        "graphs_in_family_total": graphs_in_family_total,
        "graphs_in_family_available": graphs_in_family_available,
        "graphs_scheduled": graphs_scheduled,
        "graphs_covered": graphs_covered,
        "coverage_ratio": coverage_ratio,
        "graphs_ranked": graphs_ranked,
        "scorecard_eligible": scorecard_eligible,
        "metric_median": metric_median,
        "metric_p25": metric_median if metric_p25 is None else metric_p25,
        "metric_p75": metric_median if metric_p75 is None else metric_p75,
        "median_graph_rank": median_graph_rank,
        "win_rate": win_rate,
        "top3_rate": top3_rate,
        "median_rel_best": median_rel_best,
        "median_runtime_rel_fastest": median_runtime_rel_fastest,
    }


def build_fixture_graph(name: str = "tree_fixture_4") -> SimpleNamespace:
    """Build a minimal benchmark graph wrapper for smoke tests.

    Parameters
    ----------
    name : str, optional
        Graph name.

    Returns
    -------
    types.SimpleNamespace
        Namespace with ``name``, ``graph``, and ``tags`` fields matching the
        benchmark registry shape.
    """
    graph = DaguaGraph()
    for node_id in ("a", "b", "c", "d"):
        graph.add_node(node_id, label=node_id)
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("b", "d")
    return SimpleNamespace(name=name, graph=graph, tags={"tree"})


class TestGraphFamilyDerivation:
    """Graph-family derivation tests."""

    def test_hub_spoke_tag(self) -> None:
        """Prefer explicit hub-spoke tags."""
        family, _, bucket = derive_graph_family("hub_spoke_5x50", ["hub-spoke"], 250)
        assert family == "hub_spoke"
        assert bucket == "medium"

    def test_workload_tag_preserved(self) -> None:
        """Preserve workload-specific tags."""
        family, _, _ = derive_graph_family("x", ["linear-deep"], 100)
        assert family == "linear_deep"

    def test_specific_wins_over_generic(self) -> None:
        """Prefer specific tags over generic clustered tags."""
        family, _, _ = derive_graph_family("x", ["compound", "clustered"], 100)
        assert family == "compound"

    def test_name_fallback_no_tags(self) -> None:
        """Fall back to name parsing when no tags are present."""
        family, token, _ = derive_graph_family("grid_50x50", [], 2500)
        assert family == "misc"
        assert token == "50x50"

    def test_size_buckets(self) -> None:
        """Map node counts into the expected size buckets."""
        assert derive_graph_family("x", [], 10)[2] == "tiny"
        assert derive_graph_family("x", [], 50)[2] == "small"
        assert derive_graph_family("x", [], 500)[2] == "medium"
        assert derive_graph_family("x", [], 5000)[2] == "large"
        assert derive_graph_family("x", [], 50000)[2] == "xlarge"


class TestCacheKey:
    """Cache-key tests."""

    def test_stable(self) -> None:
        """Return identical keys for identical inputs."""
        assert compute_cache_key("k1", 32, 128, 50000, "abc", "v1") == compute_cache_key(
            "k1", 32, 128, 50000, "abc", "v1"
        )

    def test_config_sensitivity(self) -> None:
        """Change when the version tag changes."""
        assert compute_cache_key("k1", 32, 128, 50000, "abc", "v1") != compute_cache_key(
            "k1", 32, 128, 50000, "abc", "v2"
        )

    def test_record_key_sensitivity(self) -> None:
        """Change when the record key changes."""
        assert compute_cache_key("k1", 32, 128, 50000, "abc", "v1") != compute_cache_key(
            "k2", 32, 128, 50000, "abc", "v1"
        )


class TestMetricConstants:
    """Metric-constant tests."""

    def test_higher_is_better_mapping(self) -> None:
        """Expose the expected metric directions."""
        assert METRIC_HIGHER_IS_BETTER["dag_consistency"] is True
        assert METRIC_HIGHER_IS_BETTER["sampled_stress"] is False
        assert METRIC_HIGHER_IS_BETTER["overlap_count"] is False

    def test_metric_sets_disjoint(self) -> None:
        """Keep quick and sampled metric sets separate."""
        assert not (QR_QUICK_METRICS & QR_SAMPLED_METRICS)


class TestBenchmarkLoading:
    """Benchmark-loading tests."""

    def test_load_benchmark_records_preserves_all_statuses(self, tmp_path: Path) -> None:
        """Keep all statuses in the records snapshot."""
        input_dir = tmp_path / "bench"
        input_dir.mkdir()
        (input_dir / "results.json").write_text(
            json.dumps(
                {
                    "g::dagua::0": {
                        "graph_name": "graph_a_4",
                        "engine_name": "dagua",
                        "seed": 0,
                        "status": "ok",
                        "runtime_seconds": 1.0,
                        "positions_file": "positions/a.pt",
                        "num_nodes": 4,
                        "num_edges": 3,
                    },
                    "g::other::0": {
                        "graph_name": "graph_a_4",
                        "engine_name": "other",
                        "seed": 0,
                        "status": "error",
                        "runtime_seconds": None,
                        "positions_file": None,
                        "num_nodes": 4,
                        "num_edges": 3,
                    },
                    "g::slow::0": {
                        "graph_name": "graph_a_4",
                        "engine_name": "slow",
                        "seed": 0,
                        "status": "timeout",
                        "runtime_seconds": None,
                        "positions_file": None,
                        "num_nodes": 4,
                        "num_edges": 3,
                    },
                    "g::cap::0": {
                        "graph_name": "graph_a_4",
                        "engine_name": "cap",
                        "seed": 0,
                        "status": "skipped",
                        "runtime_seconds": None,
                        "positions_file": None,
                        "num_nodes": 4,
                        "num_edges": 3,
                    },
                    "g::run::0": {
                        "graph_name": "graph_a_4",
                        "engine_name": "run",
                        "seed": 0,
                        "status": "running",
                        "runtime_seconds": None,
                        "positions_file": None,
                        "num_nodes": 4,
                        "num_edges": 3,
                    },
                }
            ),
            encoding="utf-8",
        )
        (input_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "graphs": [
                        {"name": "graph_a_4", "tags": ["tree"], "num_nodes": 4, "num_edges": 3}
                    ],
                    "engines": [{"name": "dagua"}, {"name": "other"}, {"name": "slow"}],
                }
            ),
            encoding="utf-8",
        )

        records_df = load_benchmark_records(input_dir)

        assert set(records_df["status"]) == {"ok", "error", "timeout", "skipped", "running"}
        assert set(records_df["graph_family"]) == {"tree"}


class TestRanking:
    """Ranking-helper tests."""

    def test_lower_better_rank_and_clamp(self) -> None:
        """Rank lower-better metrics and clamp rel-best at 10."""
        frame = pd.DataFrame(
            [
                {"engine_name": "dagua", "overlap_count": 0.0, "runtime_seconds": 1.0},
                {"engine_name": "alt", "overlap_count": 1.0, "runtime_seconds": 2.0},
                {"engine_name": "bad", "overlap_count": 100.0, "runtime_seconds": 3.0},
            ]
        )

        ranked = score_engines_on_graph(frame, "overlap_count", higher_is_better=False)

        assert ranked["engine_name"].tolist() == ["dagua", "alt", "bad"]
        assert ranked["graph_rank"].tolist() == [1, 2, 3]
        assert ranked.loc[ranked["engine_name"] == "bad", "rel_best"].iloc[0] == pytest.approx(10.0)
        assert ranked.loc[ranked["engine_name"] == "dagua", "runtime_rel_fastest"].iloc[
            0
        ] == pytest.approx(1.0)

    def test_higher_better_negative_values(self) -> None:
        """Handle higher-better metrics even when values are negative."""
        frame = pd.DataFrame(
            [
                {"engine_name": "low", "depth_spearman_rho": -0.4, "runtime_seconds": 1.0},
                {"engine_name": "best", "depth_spearman_rho": 0.2, "runtime_seconds": 2.0},
                {"engine_name": "mid", "depth_spearman_rho": -0.1, "runtime_seconds": 3.0},
            ]
        )

        ranked = score_engines_on_graph(frame, "depth_spearman_rho", higher_is_better=True)

        assert ranked["engine_name"].tolist() == ["best", "mid", "low"]
        assert all(float(value) >= 0.0 for value in ranked["rel_best"])


class TestFamilyAggregation:
    """Family aggregation tests."""

    def test_coverage_denominator_uses_scheduled_rows(self) -> None:
        """Use scheduled rows, including skipped rows, in the denominator."""
        rows: List[Dict[str, Any]] = []
        for metric_name in ALL_ANALYSIS_METRICS:
            default_value = float("nan")
            for payload in (
                {
                    "graph_name": "g1",
                    "engine_name": "dagua",
                    "status": "ok",
                    "runtime_seconds": 1.0,
                },
                {
                    "graph_name": "g2",
                    "engine_name": "dagua",
                    "status": "ok",
                    "runtime_seconds": 1.1,
                },
                {
                    "graph_name": "g3",
                    "engine_name": "dagua",
                    "status": "ok",
                    "runtime_seconds": 1.2,
                },
                {
                    "graph_name": "g1",
                    "engine_name": "limited",
                    "status": "ok",
                    "runtime_seconds": 0.9,
                },
                {
                    "graph_name": "g2",
                    "engine_name": "limited",
                    "status": "ok",
                    "runtime_seconds": 1.0,
                },
                {
                    "graph_name": "g3",
                    "engine_name": "limited",
                    "status": "skipped",
                    "runtime_seconds": None,
                },
            ):
                row = {
                    "record_key": f"{payload['graph_name']}::{payload['engine_name']}",
                    "graph_name": payload["graph_name"],
                    "engine_name": payload["engine_name"],
                    "status": payload["status"],
                    "runtime_seconds": payload["runtime_seconds"],
                    "graph_family": "tree",
                    "engine_family": payload["engine_name"],
                    "dag_consistency": float("nan"),
                }
                for metric in ALL_ANALYSIS_METRICS:
                    row[metric] = default_value
                rows.append(row)

        frame = pd.DataFrame(rows)
        sampled_values = {
            ("g1", "dagua"): 0.20,
            ("g2", "dagua"): 0.25,
            ("g3", "dagua"): 0.30,
            ("g1", "limited"): 0.10,
            ("g2", "limited"): 0.20,
        }
        for (graph_name, engine_name), value in sampled_values.items():
            mask = (frame["graph_name"] == graph_name) & (frame["engine_name"] == engine_name)
            frame.loc[mask, "sampled_stress"] = value

        summary_df, _, _, _ = compute_family_summary(frame)
        row = summary_df[
            (summary_df["graph_family"] == "tree")
            & (summary_df["metric_name"] == "sampled_stress")
            & (summary_df["engine_name"] == "limited")
        ].iloc[0]

        assert row["graphs_scheduled"] == 3
        assert row["graphs_covered"] == 2
        assert row["coverage_ratio"] == pytest.approx(2.0 / 3.0)


class TestParetoFront:
    """Pareto-front tests."""

    def test_dominance_and_roles(self) -> None:
        """Drop dominated points and annotate the expected roles."""
        summary_df = pd.DataFrame(
            [
                make_summary_row(
                    graph_family="tree",
                    metric_name="sampled_stress",
                    engine_name="fast",
                    engine_family="fast",
                    higher_is_better=False,
                    metric_median=0.2,
                    median_rel_best=0.2,
                    median_runtime_rel_fastest=1.0,
                ),
                make_summary_row(
                    graph_family="tree",
                    metric_name="sampled_stress",
                    engine_name="quality",
                    engine_family="quality",
                    higher_is_better=False,
                    metric_median=0.1,
                    median_rel_best=0.0,
                    median_runtime_rel_fastest=2.0,
                ),
                make_summary_row(
                    graph_family="tree",
                    metric_name="sampled_stress",
                    engine_name="dagua",
                    engine_family="dagua",
                    higher_is_better=False,
                    metric_median=0.18,
                    median_rel_best=0.1,
                    median_runtime_rel_fastest=1.5,
                ),
                make_summary_row(
                    graph_family="tree",
                    metric_name="sampled_stress",
                    engine_name="dominated",
                    engine_family="dominated",
                    higher_is_better=False,
                    metric_median=0.5,
                    median_rel_best=0.5,
                    median_runtime_rel_fastest=2.5,
                ),
            ]
        )

        pareto = compute_pareto_front(summary_df)

        assert set(pareto["engine_name"]) == {"fast", "quality", "dagua"}
        roles = {row.engine_name: row.roles for row in pareto.itertuples()}
        assert "fastest" in roles["fast"]
        assert "balanced" in roles["fast"]
        assert "best_quality" in roles["quality"]
        assert "dagua_anchor" in roles["dagua"]


class TestInsights:
    """Insight-extraction tests."""

    def test_extract_all_insight_types(self) -> None:
        """Emit all four requested insight categories."""
        summary_df = pd.DataFrame(
            [
                make_summary_row(
                    graph_family="tree",
                    metric_name="edge_length_cv",
                    engine_name="dagua",
                    engine_family="dagua",
                    higher_is_better=False,
                    metric_median=1.0,
                    median_rel_best=0.3,
                    median_runtime_rel_fastest=1.0,
                ),
                make_summary_row(
                    graph_family="tree",
                    metric_name="edge_length_cv",
                    engine_name="stealer",
                    engine_family="other",
                    higher_is_better=False,
                    metric_median=0.8,
                    median_rel_best=0.0,
                    median_runtime_rel_fastest=1.1,
                ),
                make_summary_row(
                    graph_family="grid",
                    metric_name="sampled_stress",
                    engine_name="dagua",
                    engine_family="dagua",
                    higher_is_better=False,
                    metric_median=1.0,
                    median_rel_best=0.4,
                    median_runtime_rel_fastest=1.0,
                ),
                make_summary_row(
                    graph_family="grid",
                    metric_name="sampled_stress",
                    engine_name="premium",
                    engine_family="other",
                    higher_is_better=False,
                    metric_median=0.6,
                    median_rel_best=0.0,
                    median_runtime_rel_fastest=1.8,
                ),
                make_summary_row(
                    graph_family="community",
                    metric_name="crossing_rate",
                    engine_name="dagua",
                    engine_family="dagua",
                    higher_is_better=False,
                    metric_median=0.5,
                    median_rel_best=0.4,
                    median_runtime_rel_fastest=1.5,
                ),
                make_summary_row(
                    graph_family="community",
                    metric_name="crossing_rate",
                    engine_name="dominator",
                    engine_family="other",
                    higher_is_better=False,
                    metric_median=0.4,
                    median_rel_best=0.2,
                    median_runtime_rel_fastest=1.0,
                ),
                make_summary_row(
                    graph_family="dependency",
                    metric_name="dag_consistency",
                    engine_name="dagua",
                    engine_family="dagua",
                    higher_is_better=True,
                    metric_median=0.95,
                    median_rel_best=0.0,
                    median_runtime_rel_fastest=1.0,
                ),
                make_summary_row(
                    graph_family="dependency",
                    metric_name="dag_consistency",
                    engine_name="laggard",
                    engine_family="other",
                    higher_is_better=True,
                    metric_median=0.80,
                    median_rel_best=0.2,
                    median_runtime_rel_fastest=1.2,
                ),
            ],
            columns=SUMMARY_COLUMNS,
        )

        insights = extract_dagua_default_insights(summary_df)

        expected_types = {
            "steal_from",
            "premium_quality",
            "dagua_dominated",
            "dagua_competitor_winner",
        }
        assert expected_types <= set(insights["insight_type"])


class TestBestOfBreed:
    """Best-of-breed aggregation tests."""

    def test_extract_best_of_breed(self) -> None:
        """Aggregate Pareto appearances across families and metrics."""
        pareto_df = pd.DataFrame(
            [
                {
                    "engine_name": "dagua",
                    "graph_family": "tree",
                    "metric_name": "sampled_stress",
                    "is_best_quality": True,
                    "is_fastest": False,
                    "is_balanced": True,
                    "is_dagua_anchor": True,
                },
                {
                    "engine_name": "dagua",
                    "graph_family": "grid",
                    "metric_name": "crossing_rate",
                    "is_best_quality": False,
                    "is_fastest": True,
                    "is_balanced": False,
                    "is_dagua_anchor": True,
                },
                {
                    "engine_name": "alt",
                    "graph_family": "tree",
                    "metric_name": "sampled_stress",
                    "is_best_quality": False,
                    "is_fastest": True,
                    "is_balanced": False,
                    "is_dagua_anchor": False,
                },
            ]
        )

        best_df = extract_best_of_breed(pareto_df)
        dagua_row = best_df[best_df["engine_name"] == "dagua"].iloc[0]

        assert dagua_row["pareto_appearances"] == 2
        assert dagua_row["pareto_family_count"] == 2
        assert dagua_row["dagua_anchor_count"] == 2


def test_run_analysis_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a small end-to-end analysis without real benchmark data."""
    fixture_graph = build_fixture_graph()
    monkeypatch.setattr("quality_runtime_analysis.get_test_graphs", lambda: [fixture_graph])

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    positions_dir = input_dir / "positions"
    positions_dir.mkdir(parents=True)

    dagua_positions = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [-0.5, 2.0], [0.5, 2.0]],
        dtype=torch.float32,
    )
    alt_positions = torch.tensor(
        [[0.0, 0.0], [0.2, 1.0], [-1.0, 2.2], [1.0, 2.2]],
        dtype=torch.float32,
    )
    torch.save(dagua_positions, positions_dir / "tree_fixture_4__dagua__seed0.pt")
    torch.save(alt_positions, positions_dir / "tree_fixture_4__alt_engine__seed0.pt")

    results_payload = {
        "tree_fixture_4::dagua::seed0": {
            "graph_name": "tree_fixture_4",
            "engine_name": "dagua",
            "seed": 0,
            "status": "ok",
            "runtime_seconds": 0.5,
            "positions_file": "positions/tree_fixture_4__dagua__seed0.pt",
            "num_nodes": 4,
            "num_edges": 3,
            "is_stochastic": False,
            "error": None,
            "skip_reason": None,
        },
        "tree_fixture_4::alt_engine::seed0": {
            "graph_name": "tree_fixture_4",
            "engine_name": "alt_engine",
            "seed": 0,
            "status": "ok",
            "runtime_seconds": 0.6,
            "positions_file": "positions/tree_fixture_4__alt_engine__seed0.pt",
            "num_nodes": 4,
            "num_edges": 3,
            "is_stochastic": False,
            "error": None,
            "skip_reason": None,
        },
        "tree_fixture_4::capped_engine::seed0": {
            "graph_name": "tree_fixture_4",
            "engine_name": "capped_engine",
            "seed": 0,
            "status": "skipped",
            "runtime_seconds": None,
            "positions_file": None,
            "num_nodes": 4,
            "num_edges": 3,
            "is_stochastic": False,
            "error": None,
            "skip_reason": "exceeds max_nodes",
        },
        "tree_fixture_4::error_engine::seed0": {
            "graph_name": "tree_fixture_4",
            "engine_name": "error_engine",
            "seed": 0,
            "status": "error",
            "runtime_seconds": None,
            "positions_file": None,
            "num_nodes": 4,
            "num_edges": 3,
            "is_stochastic": False,
            "error": "boom",
            "skip_reason": None,
        },
        "tree_fixture_4::timeout_engine::seed0": {
            "graph_name": "tree_fixture_4",
            "engine_name": "timeout_engine",
            "seed": 0,
            "status": "timeout",
            "runtime_seconds": None,
            "positions_file": None,
            "num_nodes": 4,
            "num_edges": 3,
            "is_stochastic": False,
            "error": None,
            "skip_reason": None,
        },
        "tree_fixture_4::running_engine::seed0": {
            "graph_name": "tree_fixture_4",
            "engine_name": "running_engine",
            "seed": 0,
            "status": "running",
            "runtime_seconds": None,
            "positions_file": None,
            "num_nodes": 4,
            "num_edges": 3,
            "is_stochastic": False,
            "error": None,
            "skip_reason": None,
        },
    }
    manifest_payload = {
        "graphs": [
            {
                "name": "tree_fixture_4",
                "tags": ["tree"],
                "num_nodes": 4,
                "num_edges": 3,
            }
        ],
        "engines": [
            {"name": "dagua", "available": True, "max_nodes": 0, "is_stochastic": False},
            {"name": "alt_engine", "available": True, "max_nodes": 0, "is_stochastic": False},
            {"name": "capped_engine", "available": True, "max_nodes": 3, "is_stochastic": False},
            {"name": "error_engine", "available": True, "max_nodes": 0, "is_stochastic": False},
            {"name": "timeout_engine", "available": True, "max_nodes": 0, "is_stochastic": False},
            {"name": "running_engine", "available": True, "max_nodes": 0, "is_stochastic": False},
        ],
    }
    (input_dir / "results.json").write_text(json.dumps(results_payload), encoding="utf-8")
    (input_dir / "manifest.json").write_text(json.dumps(manifest_payload), encoding="utf-8")

    result = run_analysis(
        input_dir=input_dir,
        output_dir=output_dir,
        workers=1,
        cache=False,
        write_plots=False,
        max_nodes_for_sampled=100,
    )

    assert result["records"] == 6
    assert result["ok_records"] == 2
    assert result["metric_successes"] == 2
    assert (output_dir / "analysis_records_snapshot.csv").exists()
    assert (output_dir / "family_metric_summary.csv").exists()
    assert (output_dir / "artifact_index.csv").exists()

    snapshot_df = pd.read_csv(output_dir / "analysis_records_snapshot.csv")
    assert set(snapshot_df["status"]) == {"ok", "skipped", "error", "timeout", "running"}

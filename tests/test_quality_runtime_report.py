"""Tests for the QR markdown report renderer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from generate_quality_runtime_report import (  # noqa: E402
    build_best_of_breed_section,
    build_coverage_section,
    build_dagua_insights_section,
    build_dataset_snapshot,
    build_family_scorecards,
    build_markdown_report,
    fmt_num,
)


def test_fmt_num_handles_edge_cases() -> None:
    """`fmt_num` should mask invalid values and respect precision."""
    assert fmt_num(float("nan")) == "-"
    assert fmt_num(float("inf")) == "-"
    assert fmt_num("nan") == "-"
    assert fmt_num(1.234, precision=2) == "1.23"
    assert fmt_num(None) == "-"


def test_dataset_snapshot_empty() -> None:
    """Empty record snapshots should render an explicit placeholder."""
    assert "no records" in build_dataset_snapshot([]).lower()


def test_dataset_snapshot_counts() -> None:
    """Dataset snapshot should count statuses correctly."""
    records = [
        {"status": "ok"},
        {"status": "ok"},
        {"status": "error"},
    ]
    result = build_dataset_snapshot(records)
    assert "ok: 2" in result
    assert "error: 1" in result
    assert "Total records: 3" in result


def test_coverage_section_with_dagua() -> None:
    """Coverage section should render dagua coverage per family."""
    rows = [
        {
            "graph_family": "grid",
            "engine_name": "dagua",
            "graphs_in_family_total": 10,
            "graphs_in_family_available": 8,
            "coverage_ratio": 0.75,
        },
    ]
    result = build_coverage_section(rows)
    assert "grid" in result
    assert "0.750" in result


def test_family_scorecards_sorting() -> None:
    """Lower aggregate median rank should appear first in the scorecard."""
    rows = [
        {
            "graph_family": "tree",
            "metric_name": "sampled_stress",
            "engine_name": "a",
            "median_graph_rank": "2.0",
            "median_rel_best": "0.1",
            "median_runtime_rel_fastest": "1.5",
            "coverage_ratio": "0.8",
            "scorecard_eligible": "true",
        },
        {
            "graph_family": "tree",
            "metric_name": "sampled_stress",
            "engine_name": "b",
            "median_graph_rank": "1.0",
            "median_rel_best": "0.0",
            "median_runtime_rel_fastest": "1.0",
            "coverage_ratio": "1.0",
            "scorecard_eligible": "true",
        },
    ]
    result = build_family_scorecards(rows)
    idx_a = result.find("| a ")
    idx_b = result.find("| b ")
    assert idx_b < idx_a


def test_dagua_insights_empty() -> None:
    """Empty insight sets should render a placeholder."""
    result = build_dagua_insights_section([])
    assert "no actionable" in result.lower()


def test_dagua_insights_formatted() -> None:
    """Insights should use the Wave 2 column names in the rendered table."""
    insights = [
        {
            "insight_type": "steal_from",
            "graph_family": "grid",
            "metric_name": "sampled_stress",
            "competitor_engine_name": "classic_umap",
            "quality_advantage": "0.25",
            "quality_advantage_norm": "0.25",
            "runtime_ratio": "1.1",
            "family_metric_p25": "0.1",
            "family_metric_p50": "0.2",
            "family_metric_p75": "0.3",
        }
    ]
    result = build_dagua_insights_section(insights)
    assert "classic_umap" in result
    assert "steal_from" in result


def test_best_of_breed_section_uses_wave2_columns() -> None:
    """Best-of-breed should read the actual Pareto aggregation columns."""
    rows = [
        {
            "engine_name": "dagua",
            "pareto_family_count": "4",
            "pareto_metric_count": "5",
            "best_quality_count": "2",
            "fastest_count": "1",
            "balanced_count": "3",
        }
    ]
    result = build_best_of_breed_section(rows)
    assert "dagua" in result
    assert "| dagua | 4 | 2 | 1 | 3 |" in result


def test_build_markdown_report_empty_dir(tmp_path: Path) -> None:
    """The report should still render headings for an empty directory."""
    markdown = build_markdown_report(tmp_path, plots_enabled=False)
    assert "Quality/Runtime Analysis" in markdown
    assert "Artifact Index" in markdown


def test_build_markdown_report_with_data(tmp_path: Path) -> None:
    """Minimal sidecar CSVs should produce a readable markdown report."""
    (tmp_path / "analysis_records_snapshot.csv").write_text(
        "status,record_key\nok,a\nok,b\nerror,c\n",
        encoding="utf-8",
    )
    (tmp_path / "family_metric_summary.csv").write_text(
        "graph_family,metric_name,engine_name,median_graph_rank,median_rel_best,"
        "median_runtime_rel_fastest,coverage_ratio,graphs_in_family_total,"
        "graphs_in_family_available,scorecard_eligible\n"
        "grid,sampled_stress,dagua,1.0,0.0,1.0,1.0,5,5,true\n"
        "grid,sampled_stress,classic_fr,2.0,0.1,1.5,1.0,5,5,true\n",
        encoding="utf-8",
    )
    (tmp_path / "dagua_default_insights.csv").write_text(
        "graph_family,metric_name,insight_type,competitor_engine_name,quality_advantage,"
        "quality_advantage_norm,runtime_ratio,family_metric_p25,family_metric_p50,"
        "family_metric_p75\n"
        "grid,sampled_stress,steal_from,classic_fr,0.1,0.2,1.1,0.0,0.1,0.2\n",
        encoding="utf-8",
    )
    (tmp_path / "best_of_breed_configs.csv").write_text(
        "engine_name,pareto_family_count,pareto_metric_count,best_quality_count,"
        "fastest_count,balanced_count\n"
        "dagua,1,1,1,1,1\n",
        encoding="utf-8",
    )

    markdown = build_markdown_report(tmp_path, plots_enabled=False)
    assert "grid" in markdown
    assert "dagua" in markdown
    assert "classic_fr" in markdown

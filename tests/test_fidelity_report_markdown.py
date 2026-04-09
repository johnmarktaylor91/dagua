"""Tests for Cleanup2: markdown fidelity report generator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_fidelity_report import (  # noqa: E402
    build_executive_summary_table,
    build_failures_section,
    build_markdown_report,
    build_procrustes_summary,
    fmt_num,
)


def test_fmt_num_handles_nan() -> None:
    """`fmt_num` should hide NaN and infinity values."""
    assert fmt_num("nan") == "-"
    assert fmt_num(float("nan")) == "-"
    assert fmt_num(float("inf")) == "-"
    assert fmt_num(1.234567, precision=3) == "1.235"
    assert fmt_num("not_a_number") == "-"


def test_executive_summary_empty() -> None:
    """An empty summary should render a placeholder."""
    result = build_executive_summary_table([])
    assert "no algorithm" in result.lower()


def test_executive_summary_one_row() -> None:
    """A single summary row should appear in the markdown table."""
    rows = [
        {
            "algorithm_family": "classic_fr",
            "family_verdict": "strong_equivalent",
            "is_stochastic": "True",
            "num_graphs_paired_ok": "10",
            "num_strong_equivalent": "9",
            "num_weak_equivalent": "1",
            "num_partial_match": "0",
            "num_divergent": "0",
            "procrustes_rmsd_median": "0.012",
        }
    ]
    result = build_executive_summary_table(rows)
    assert "classic_fr" in result
    assert "strong_equivalent" in result
    assert "0.012" in result


def test_failures_section_empty() -> None:
    """The failures section should acknowledge when nothing failed."""
    result = build_failures_section([], [], [])
    assert "No variants" in result


def test_failures_section_with_divergent() -> None:
    """The failures section should surface the dominant rejection reason."""
    rows = [
        {
            "variant_id": "v1",
            "graph_name": "g1",
            "verdict": "divergent",
            "total_rejected": "3",
            "rejection_breakdown_json": json.dumps({"contains_nan": 2, "load_failure": 1}),
        }
    ]
    result = build_failures_section(rows, [], [])
    assert "v1" in result
    assert "contains_nan=2" in result


def test_build_markdown_report_no_data(tmp_path: Path) -> None:
    """The markdown report should render its top-level sections on empty input."""
    out = tmp_path / "report.md"
    markdown = build_markdown_report(tmp_path, out)
    assert "Dagua Fidelity Analysis Report" in markdown
    assert "Methodology" in markdown


def test_build_procrustes_summary_empty() -> None:
    """The Procrustes summary should handle missing per-graph rows."""
    result = build_procrustes_summary([])
    assert "no per-graph data" in result.lower()

"""Render the visual parity v2 dashboard."""

from __future__ import annotations

import argparse
import html
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from scripts.visual_parity.io import read_coverage_matrix, read_ledger
from scripts.visual_parity.ledger import assert_rebase_comparable

RESEARCH_DIR = Path(".project-context/research/sprint_visual_parity_v2")
COVERAGE_PATH = RESEARCH_DIR / "coverage_matrix.json"
LEDGER_PATH = RESEARCH_DIR / "ledger.json"
DASHBOARD_DIR = Path("eval_output/visual_parity_v2/dashboard")


def _sparkline(values: Sequence[float]) -> str:
    """Render an ASCII sparkline.

    Parameters
    ----------
    values
        Numeric values to summarize.

    Returns
    -------
    str
        ASCII-only sparkline.
    """

    if not values:
        return ""
    blocks = "._:-=+*#"
    low = min(values)
    high = max(values)
    if high == low:
        return blocks[-1] * len(values)
    return "".join(
        blocks[int((value - low) / (high - low) * (len(blocks) - 1))] for value in values
    )


def _status_counts(rows: Sequence[Mapping[str, Any]], key: str) -> Counter[str]:
    """Count row values by key.

    Parameters
    ----------
    rows
        Row dictionaries.
    key
        Field to count.

    Returns
    -------
    Counter[str]
        Value counts.
    """

    return Counter(str(row.get(key, "unknown")) for row in rows)


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a markdown table.

    Parameters
    ----------
    headers
        Column headers.
    rows
        Table rows.

    Returns
    -------
    str
        Markdown table.
    """

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def build_markdown(coverage: Mapping[str, Any], ledger: Mapping[str, Any]) -> str:
    """Build dashboard markdown from coverage and ledger stores.

    Parameters
    ----------
    coverage
        Coverage matrix payload.
    ledger
        Ledger payload.

    Returns
    -------
    str
        ASCII-only markdown dashboard.
    """

    assert_rebase_comparable(ledger.get("rounds", []))
    rows = list(ledger.get("rows", []))
    cells = list(coverage.get("cells", []))
    by_subsprint: Dict[str, Counter[str]] = defaultdict(Counter)
    for round_entry in ledger.get("rounds", []):
        subsprint = str(round_entry.get("subsprint", "unknown"))
        tripwires = str(round_entry.get("tripwires", "unknown"))
        by_subsprint[subsprint][tripwires] += 1
    gap_rows = [
        cell
        for cell in cells
        if cell.get("support_status") == "missing"
        and cell.get("parity_status") == "untested"
        and cell.get("priority") in {"P0", "P1"}
    ][:20]
    waivers = [row for row in rows if row.get("waiver")]
    blocked = [cell for cell in cells if cell.get("parity_status") == "blocked_upstream"]
    png_rows = [row for row in rows if row.get("target_kind") == "png_raster"]
    gate_rows = [row for row in rows if row.get("target_kind") != "png_raster"]
    knob_rows = sorted(
        ledger.get("knobs", []),
        key=lambda knob: (str(knob.get("status", "")), -len(knob.get("linked_rows", []))),
    )
    worst_rows = [
        row
        for row in gate_rows
        if row.get("parity_status")
        not in {"in_tolerance", "matched", "waived_improvement", "waived_out_of_scope"}
    ][:6]
    round_values = [
        float(round_entry.get("gates_summary", {}).get("global_in_tol_pct", 0.0))
        for round_entry in ledger.get("rounds", [])
        if round_entry.get("gates_summary", {}).get("global_in_tol_pct") is not None
    ]
    parts = [
        "# Visual Parity v2 Dashboard",
        "",
        "## Headline",
        "",
        _table(
            ["metric", "value"],
            [
                ["ledger_rows", len(rows)],
                ["coverage_cells", len(cells)],
                ["gate_rows", len(gate_rows)],
                ["locked_rows", sum(1 for row in rows if row.get("locked") is True)],
                ["blocked_upstream", len(blocked)],
                ["sparkline", _sparkline(round_values)],
            ],
        ),
        "",
        "## Baseline Rounds",
        "",
        _table(
            [
                "round",
                "track",
                "lane_label",
                "geometry_mode",
                "global_in_tol_pct",
                "mean_l1",
                "mean_ssim",
                "cards",
                "tripwires",
            ],
            [
                [
                    round_entry.get("round_id", ""),
                    round_entry.get("track", ""),
                    round_entry.get("lane_label", ""),
                    round_entry.get("geometry_mode", ""),
                    round_entry.get("gates_summary", {}).get("global_in_tol_pct", ""),
                    round_entry.get("gates_summary", {}).get("pixel_mean_l1_rgb_per_pixel", ""),
                    round_entry.get("gates_summary", {}).get("pixel_mean_ssim", ""),
                    round_entry.get("gates_summary", {}).get("manifest_cards", ""),
                    round_entry.get("tripwires", ""),
                ]
                for round_entry in ledger.get("rounds", [])
            ],
        ),
        "",
        "## Sub-Sprints",
        "",
        _table(
            ["subsprint", "rounds", "tripwire_counts"],
            [
                [
                    subsprint,
                    sum(counter.values()),
                    ", ".join(f"{key}:{value}" for key, value in sorted(counter.items())),
                ]
                for subsprint, counter in sorted(by_subsprint.items())
            ],
        ),
        "",
        "## Knobs",
        "",
        _table(
            ["knob_id", "status", "linked_rows", "values_tried"],
            [
                [
                    knob.get("knob_id", ""),
                    knob.get("status", ""),
                    len(knob.get("linked_rows", [])),
                    ",".join(str(value) for value in knob.get("values_tried", [])),
                ]
                for knob in knob_rows
            ],
        ),
        "",
        "## Worst 6",
        "",
        _table(
            ["row_id", "priority", "status", "target_kind"],
            [
                [
                    row.get("row_id", ""),
                    row.get("priority", ""),
                    row.get("parity_status", ""),
                    row.get("target_kind", ""),
                ]
                for row in worst_rows
            ],
        ),
        "",
        "## Gap Queue",
        "",
        _table(
            ["cell_id", "priority", "support", "notes"],
            [
                [
                    cell.get("cell_id", ""),
                    cell.get("priority", ""),
                    cell.get("support_status", ""),
                    cell.get("notes", ""),
                ]
                for cell in gap_rows
            ],
        ),
        "",
        "## Waivers",
        "",
        _table(
            ["row_id", "status", "reason"],
            [
                [
                    row.get("row_id", ""),
                    row.get("parity_status", ""),
                    row.get("waiver", {}).get("reason", ""),
                ]
                for row in waivers
            ],
        ),
        "",
        "## Blocked Upstream",
        "",
        _table(
            ["status", "count"],
            sorted(_status_counts(blocked, "tool").items()),
        ),
        "",
        "## png_raster Lane",
        "",
        "Reported separately. These rows never gate Track G or Track D.",
        "",
        _table(
            ["row_id", "status", "residual"],
            [
                [row.get("row_id", ""), row.get("parity_status", ""), row.get("residual_class", "")]
                for row in png_rows
            ],
        ),
        "",
    ]
    markdown = "\n".join(parts)
    markdown.encode("ascii")
    return markdown


def build_html(markdown: str) -> str:
    """Build a simple ASCII HTML page for dashboard markdown.

    Parameters
    ----------
    markdown
        Markdown source.

    Returns
    -------
    str
        HTML source.
    """

    escaped = html.escape(markdown)
    return (
        "<!doctype html>\n"
        '<html><head><meta charset="utf-8"><title>Visual Parity v2 Dashboard</title>'
        "<style>body{font-family:Arial,sans-serif;margin:32px;max-width:1200px}"
        "pre{white-space:pre-wrap}table{border-collapse:collapse}"
        "td,th{border:1px solid #bbb;padding:4px 6px}</style>"
        "</head><body><pre>"
        f"{escaped}"
        "</pre></body></html>\n"
    )


def write_dashboard(output_dir: Path = DASHBOARD_DIR) -> None:
    """Render dashboard files.

    Parameters
    ----------
    output_dir
        Destination dashboard directory.

    Returns
    -------
    None
        The function writes ``index.md`` and ``index.html``.
    """

    coverage = read_coverage_matrix(COVERAGE_PATH)
    ledger = read_ledger(LEDGER_PATH)
    markdown = build_markdown(coverage, ledger)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.md").write_text(markdown, encoding="utf-8")
    (output_dir / "index.html").write_text(build_html(markdown), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the dashboard command-line interface.

    Parameters
    ----------
    argv
        Optional command arguments.

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DASHBOARD_DIR)
    args = parser.parse_args(argv)
    write_dashboard(args.out)
    print(f"dashboard: {args.out / 'index.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

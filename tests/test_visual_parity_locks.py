"""Generated visual parity v2 lock tests.

Do not edit by hand. Regenerate with:
    python -m scripts.visual_parity.ledger --generate-lock-tests
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from scripts.visual_parity import ledger
from scripts.visual_parity.io import read_ledger

LEDGER_PATH = Path(".project-context/research/sprint_visual_parity_v2/ledger.json")


def test_generated_lock_file_is_byte_identical() -> None:
    """Regenerating lock tests should reproduce this file byte-for-byte.

    Returns
    -------
    None
        The test asserts deterministic generation.
    """

    data = read_ledger(LEDGER_PATH)
    assert ledger.render_lock_tests(data) == Path(__file__).read_text(encoding="utf-8")


def _locked_rows() -> List[Dict[str, object]]:
    """Return locked ledger rows.

    Returns
    -------
    List[Dict[str, object]]
        Locked rows from the committed ledger.
    """

    data = read_ledger(LEDGER_PATH)
    return [row for row in data["rows"] if row.get("locked") is True]


def test_lock_graphviz_edge_arrowhead_normal_svg_declared() -> None:
    """Protect locked row graphviz.edge.arrowhead.normal.svg_declared.

    Returns
    -------
    None
        The test asserts the committed current values remain locked.
    """

    expected = json.loads("""
        {
            "coverage_cell_id": "graphviz.edge.arrowhead.normal",
            "metrics": [
                {
                    "current": 1.0,
                    "metric_id": "arrow_polygon_iou",
                    "status": "pass",
                    "tolerance": 0.98,
                    "validated_tripwire": true
                },
                {
                    "current": 1.0,
                    "metric_id": "arrow_len_pct",
                    "status": "pass",
                    "tolerance": 0.98,
                    "validated_tripwire": true
                },
                {
                    "current": 0.9807,
                    "metric_id": "arrow_fill_mode",
                    "status": "pass",
                    "tolerance": 0.98,
                    "validated_tripwire": true
                }
            ],
            "parity_status": "in_tolerance",
            "support_status": "supported"
        }
    """)
    by_id = {str(row["row_id"]): row for row in _locked_rows()}
    row = by_id["graphviz.edge.arrowhead.normal.svg_declared"]
    actual = ledger.lock_expectation(row)
    assert actual == expected

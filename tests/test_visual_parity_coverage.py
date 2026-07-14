"""Tests for Lane C visual parity coverage generation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict

from scripts.visual_parity import coverage
from scripts.visual_parity.io import read_coverage_matrix

COVERAGE_PATH = Path(".project-context/research/sprint_visual_parity_v2/coverage_matrix.json")


def _cells_by_id() -> Dict[str, Dict[str, object]]:
    """Return committed coverage cells keyed by id.

    Returns
    -------
    Dict[str, Dict[str, object]]
        Coverage cells by ``cell_id``.
    """

    matrix = read_coverage_matrix(COVERAGE_PATH)
    return {str(cell["cell_id"]): cell for cell in matrix["cells"]}


def test_rebuild_has_external_graphviz_denominator() -> None:
    """Coverage generation should keep a broad Graphviz cosmetic denominator.

    Returns
    -------
    None
        The test asserts the committed matrix satisfies Lane C size floors.
    """

    matrix = read_coverage_matrix(COVERAGE_PATH)
    graphviz_cosmetic = [
        cell
        for cell in matrix["cells"]
        if cell["tool"] == "graphviz"
        and cell["value_group"]
        in {
            "arrow_primitive",
            "arrow_alias",
            "arrow_modifier_expansion",
            "arrow_compound_sample",
            "node_shape",
            "cosmetic_attr",
        }
    ]

    assert len(graphviz_cosmetic) >= 60
    assert {snapshot["id"] for snapshot in matrix["source_snapshots"]} >= {
        "gv-attrs",
        "gv-arrows",
        "gv-shapes",
        "gv-colors",
    }


def test_known_supported_and_gap_cells() -> None:
    """Known present and absent Graphviz cells should be classified correctly.

    Returns
    -------
    None
        The test asserts support state for representative cells.
    """

    cells = _cells_by_id()

    assert cells["graphviz.edge.arrowhead.normal"]["support_status"] == "supported"
    assert cells["graphviz.node.shape.ribosite"]["support_status"] == "missing"
    assert cells["graphviz.node.shape.ribosite"]["parity_status"] == "untested"


def test_arrow_categories_are_distinct_and_generated() -> None:
    """Arrow rows should enumerate the four required atlas categories.

    Returns
    -------
    None
        The test asserts primitive, alias, modifier, and compound groups exist.
    """

    matrix = read_coverage_matrix(COVERAGE_PATH)
    counts = Counter(
        cell["value_group"] for cell in matrix["cells"] if cell["attribute"] == "arrowhead"
    )

    assert counts["arrow_primitive"] >= 23
    assert counts["arrow_alias"] >= 4
    assert counts["arrow_modifier_expansion"] == 42
    assert counts["arrow_compound_sample"] >= 12
    assert coverage.generate_graphviz_modifier_expansions() == sorted(
        coverage.generate_graphviz_modifier_expansions(),
        key=coverage.generate_graphviz_modifier_expansions().index,
    )


def test_adapter_capabilities_seed_verified_facts() -> None:
    """Adapter capability rows should gate nothing until proven.

    Returns
    -------
    None
        The test asserts the verified Lane C adapter facts.
    """

    matrix = read_coverage_matrix(COVERAGE_PATH)
    capabilities = {row["adapter"]: row for row in matrix["adapter_capabilities"]}

    assert all(row["gate_eligible"] is False for row in capabilities.values())
    assert capabilities["gephi"]["fixed_positions"] is False
    assert capabilities["mermaid"]["fixed_positions"] is False
    assert capabilities["graphviz"]["fixed_positions"] is False
    assert capabilities["cytoscape"]["per_element_styles"] is False
    assert capabilities["d3"]["per_element_styles"] is False

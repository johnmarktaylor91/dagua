"""Tests for the competitor adapter capability self-report (Lane D hardening)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.competitor_renderers import capabilities
from scripts.visual_parity.types import AdapterCapability


def test_all_registered_adapters_have_capability_rows() -> None:
    """Every registered competitor adapter should have a capability record."""

    from scripts.competitor_renderers import RENDERERS

    assert set(capabilities.ADAPTER_CAPABILITIES.keys()) == set(RENDERERS.keys())
    for row in capabilities.capability_rows():
        assert isinstance(row, AdapterCapability)
        assert row.evidence


def test_gate_eligible_is_false_for_every_adapter() -> None:
    """No adapter is gate-eligible yet -- capability is not proven per-cell (F3)."""

    assert all(row.gate_eligible is False for row in capabilities.capability_rows())


def test_verified_capability_facts_match_f3() -> None:
    """Capability rows should match the verified adapter facts (correction F3)."""

    rows = capabilities.ADAPTER_CAPABILITIES

    assert rows["gephi"].fixed_positions is False
    assert rows["gephi"].per_element_styles is False

    assert rows["mermaid"].fixed_positions is False

    assert rows["graphviz"].fixed_positions is False

    assert rows["cytoscape"].fixed_positions is True
    assert rows["cytoscape"].per_element_styles is False

    assert rows["d3"].fixed_positions is True


def test_print_versions_writes_json(tmp_path: Path) -> None:
    """--print-versions should write a JSON version map to the requested path."""

    out_path = tmp_path / "versions.json"
    versions = capabilities.print_versions(out_path)

    assert out_path.exists()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk == versions
    assert "dot" in versions

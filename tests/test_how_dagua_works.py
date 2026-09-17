"""Smoke test for the how-dagua-works explainer builder."""

from pathlib import Path

import pytest

from scripts.build_how_dagua_works import build


@pytest.mark.slow
def test_how_dagua_works_build(tmp_path: Path) -> None:
    out = tmp_path / "how_dagua_works"
    build(str(out))

    expected = {
        "pipeline_overview.png",
        "multilevel_hierarchy.png",
        "routing_comparison.png",
        "pinning_constraint.gif",
    }
    assert expected == {p.name for p in out.iterdir() if p.is_file()}

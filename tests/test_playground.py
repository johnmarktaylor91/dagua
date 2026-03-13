"""Tests for the interactive layout playground helpers."""

from __future__ import annotations

import nbformat
import pytest

from dagua.playground import (
    PLAYGROUND_GRAPH_LADDER,
    PLAYGROUND_PANEL_PRESETS,
    _apply_overrides,
    _base_playground_config,
    _graph_catalog,
    _panel_graph_names,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:enable_nested_tensor is True.*:UserWarning"
)


def test_playground_catalog_contains_curated_graphs() -> None:
    catalog = _graph_catalog(max_nodes=2000)
    assert catalog
    assert any(name in catalog for name in PLAYGROUND_GRAPH_LADDER)


def test_panel_presets_resolve_to_available_graphs() -> None:
    catalog = _graph_catalog(max_nodes=2000)
    for preset in PLAYGROUND_PANEL_PRESETS:
        resolved = _panel_graph_names(preset, catalog)
        assert resolved
        assert len(resolved) <= 4
        for name in resolved:
            assert name in catalog


def test_apply_overrides_updates_layout_config() -> None:
    cfg = _base_playground_config(device="cpu")
    updated = _apply_overrides(cfg, {"node_sep": 42.0, "w_crossing": 3.5, "steps": 75})
    assert updated.node_sep == 42.0
    assert updated.w_crossing == 3.5
    assert updated.steps == 75
    assert updated.edge_opt_steps == -1


def test_interactive_playground_notebook_is_valid() -> None:
    path = "/home/jtaylor/projects/dagua/docs/interactive_playground.ipynb"
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    assert nb.cells[0]["cell_type"] == "markdown"
    assert "launch_playground" in "".join(nb.cells[-1]["source"])

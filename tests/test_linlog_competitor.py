"""Tests for the LinLog reference competitor."""

from __future__ import annotations

import torch

from dagua.eval.competitors import get_competitor
from dagua.eval.competitors.linlog_competitor import LinLogReference
from dagua.eval.variants import get_variant, original_variant_name
from dagua.graph import DaguaGraph


def _make_path_graph() -> DaguaGraph:
    """Create a small path graph for LinLog reference tests.

    Returns
    -------
    DaguaGraph
        Four-node path graph with node sizes initialized.
    """
    graph = DaguaGraph.from_edge_list([("a", "b"), ("b", "c"), ("c", "d")])
    graph.compute_node_sizes()
    return graph


def test_linlog_reference_is_registered() -> None:
    """The clean Python LinLog reference should be available by name."""
    competitor = get_competitor("linlog")

    assert competitor is not None
    assert competitor.name == "linlog"


def test_linlog_reference_produces_seeded_positions() -> None:
    """LinLog reference runs should be deterministic for the same seed."""
    graph = _make_path_graph()
    competitor = LinLogReference()

    first = competitor.layout_with_variant(
        graph,
        seed=7,
        variant_params={"attrExponent": 1.0, "repuExponent": 0.0, "steps": 5},
    )
    second = competitor.layout_with_variant(
        graph,
        seed=7,
        variant_params={"attrExponent": 1.0, "repuExponent": 0.0, "steps": 5},
    )
    other = competitor.layout_with_variant(
        graph,
        seed=8,
        variant_params={"attrExponent": 1.0, "repuExponent": 0.0, "steps": 5},
    )

    assert first.error is None
    assert second.error is None
    assert other.error is None
    assert first.pos is not None
    assert second.pos is not None
    assert other.pos is not None
    assert tuple(first.pos.shape) == (graph.num_nodes, 2)
    assert torch.allclose(first.pos, second.pos)
    assert not torch.allclose(first.pos, other.pos)


def test_linlog_reference_accepts_dagua_short_params() -> None:
    """The reference adapter should also accept Dagua's ``a`` and ``r`` names."""
    graph = _make_path_graph()
    result = LinLogReference().layout_with_variant(
        graph,
        seed=11,
        variant_params={"a": 2.0, "r": 0.5, "steps": 3},
    )

    assert result.error is None
    assert result.pos is not None
    assert tuple(result.pos.shape) == (graph.num_nodes, 2)


def test_classic_linlog_variants_pair_to_true_reference() -> None:
    """Classic LinLog variants should compare against the Python reference."""
    variant_ids = (
        "classic_linlog_default",
        "classic_linlog_quadratic",
        "classic_linlog_power",
        "classic_linlog_steps100",
        "classic_linlog_steps500",
    )

    for variant_id in variant_ids:
        variant = get_variant(variant_id)
        assert variant is not None
        assert variant.original_engine == "linlog"
        assert variant.is_true_original is True
        assert original_variant_name(variant) == f"linlog__for__{variant_id}"

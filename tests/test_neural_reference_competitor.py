"""Tests for SmartGD and DeepGD reference benchmark adapters."""

from __future__ import annotations

import pytest
import torch

from dagua.eval.competitors import get_competitor
from dagua.eval.competitors.neural_reference_competitor import DeepGDReference, SmartGDReference
from dagua.eval.variants import (
    base_pairings,
    engine_is_heavy,
    engine_is_stochastic,
    get_variant,
    original_variant_name,
    variant_pairings,
)
from dagua.graph import DaguaGraph

SMARTGD_AVAILABLE = SmartGDReference().available()
DEEPGD_AVAILABLE = DeepGDReference().available()


def _small_graph() -> DaguaGraph:
    """Create a small connected graph for neural reference tests.

    Returns
    -------
    DaguaGraph
        Six-node path graph.
    """
    return DaguaGraph.from_edge_list([(str(node), str(node + 1)) for node in range(5)])


def test_smartgd_deepgd_references_registered() -> None:
    """SmartGD and DeepGD references should be in the competitor registry.

    Returns
    -------
    None
        This test asserts on registered competitor names.
    """
    assert get_competitor("smartgd_reference") is not None
    assert get_competitor("deepgd_reference") is not None


def test_smartgd_deepgd_variant_pairings_registered() -> None:
    """SmartGD and DeepGD reimplementations should pair to true references.

    Returns
    -------
    None
        This test asserts on base and variant pairing metadata.
    """
    smartgd_variant = get_variant("smartgd_reimpl_default")
    deepgd_variant = get_variant("deepgd_reimpl_default")

    assert smartgd_variant is not None
    assert deepgd_variant is not None
    assert smartgd_variant.original_engine == "smartgd_reference"
    assert deepgd_variant.original_engine == "deepgd_reference"
    assert smartgd_variant.is_true_original is True
    assert deepgd_variant.is_true_original is True
    assert base_pairings()["smartgd_reimpl"] == ["smartgd_reference"]
    assert base_pairings()["deepgd_reimpl"] == ["deepgd_reference"]
    assert variant_pairings()["smartgd_reimpl_default"] == [original_variant_name(smartgd_variant)]
    assert variant_pairings()["deepgd_reimpl_default"] == [original_variant_name(deepgd_variant)]
    assert engine_is_stochastic("smartgd_reference") is False
    assert engine_is_stochastic("deepgd_reference") is False
    assert engine_is_heavy("smartgd_reference") is True
    assert engine_is_heavy("deepgd_reference") is True


@pytest.mark.skipif(not SMARTGD_AVAILABLE, reason="SmartGD reference checkpoint unavailable")
def test_smartgd_reference_layout_is_seed_deterministic() -> None:
    """The SmartGD reference adapter should repeat exactly for a fixed seed.

    Returns
    -------
    None
        This test asserts on returned position tensors.
    """
    graph = _small_graph()

    first = SmartGDReference().layout(graph, seed=17)
    second = SmartGDReference().layout(graph, seed=17)

    assert first.error is None
    assert second.error is None
    assert first.pos is not None
    assert second.pos is not None
    assert first.pos.shape == (graph.num_nodes, 2)
    assert torch.equal(first.pos, second.pos)


@pytest.mark.skipif(not DEEPGD_AVAILABLE, reason="DeepGD reference checkpoint unavailable")
def test_deepgd_reference_layout_is_seed_deterministic() -> None:
    """The DeepGD reference adapter should repeat exactly for a fixed seed.

    Returns
    -------
    None
        This test asserts on returned position tensors.
    """
    graph = _small_graph()

    first = DeepGDReference().layout(graph, seed=17)
    second = DeepGDReference().layout(graph, seed=17)

    assert first.error is None
    assert second.error is None
    assert first.pos is not None
    assert second.pos is not None
    assert first.pos.shape == (graph.num_nodes, 2)
    assert torch.equal(first.pos, second.pos)

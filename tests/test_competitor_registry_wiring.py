"""Registry wiring and seed-fallback pins for competitor adapters.

Pins two GLaDOS-prep fixes:

1. ``webcola_competitor`` and ``d3dag_competitor`` carried ``@register``
   decorators but were never imported by ``dagua.eval.competitors.__init__``,
   so both families were silently absent from the engine field.
2. Seed-fallback hygiene: harness-stochastic engines whose certified-pool rows
   all carry explicit seeds (igraph graphopt/drl/lgl/davidson_harel, sgd2,
   fa2_ref) now pin ``seed=None`` to a deterministic default. Engines with
   ``seed=None`` rows in the certified pool (igraph fr/kamada_kawai/mds/rt*/
   sugiyama, sgd2_mds) keep their previous ``seed=None`` behavior untouched.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest
import torch

from dagua.eval.competitors import fa2_competitor, get_competitor
from dagua.eval.competitors.igraph_competitor import (
    IgraphDavidsonHarel,
    IgraphDRL,
    IgraphFR,
    IgraphGraphOpt,
    IgraphKamadaKawai,
    IgraphLGL,
    IgraphMDS,
    IgraphRT,
    IgraphSugiyama,
)
from dagua.graph import DaguaGraph


def test_webcola_and_d3dag_families_are_registered() -> None:
    """The webcola and d3dag adapters must be present in the registry."""
    assert get_competitor("webcola") is not None
    assert get_competitor("d3dag") is not None


def test_igraph_seed_fallbacks_scoped_to_harness_stochastic_engines() -> None:
    """Only pool-safe (always-seeded-in-pool) igraph engines pin a default."""
    assert IgraphGraphOpt.default_seed == 42
    assert IgraphDRL.default_seed == 42
    assert IgraphLGL.default_seed == 42
    assert IgraphDavidsonHarel.default_seed == 42
    # These engines have seed=None rows in the certified pool; their
    # seed=None behavior must stay byte-identical.
    assert IgraphFR.default_seed is None
    assert IgraphKamadaKawai.default_seed is None
    assert IgraphMDS.default_seed is None
    assert IgraphRT.default_seed is None
    assert IgraphSugiyama.default_seed is None


def test_sgd2_seed_none_falls_back_to_42(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sgd2 adapter must pass random_seed=42 when seed is None."""
    from dagua.eval.competitors.sgd2_competitor import SGD2

    captured: dict[str, Any] = {}

    class _StubSGD2:
        @staticmethod
        def layout(sources, targets, **kwargs):
            captured.update(kwargs)
            import numpy as np

            num_nodes = max(max(sources), max(targets)) + 1
            return np.zeros((num_nodes, 2), dtype=np.float64)

    monkeypatch.setitem(sys.modules, "s_gd2", _StubSGD2())

    edge_index = torch.tensor([(0, 1), (1, 2)], dtype=torch.long).t().contiguous()
    graph = DaguaGraph.from_edge_index(edge_index, 3)
    result = SGD2().layout(graph, seed=None)

    assert result.error is None
    assert captured.get("random_seed") == 42


def test_fa2_seed_none_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """fa2_ref with seed=None must not depend on ambient process RNG state."""
    import random

    import numpy as np

    class _StubForceAtlas2:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def forceatlas2_networkx_layout(self, nx_graph, pos=None, iterations=100, **kwargs):
            del pos, iterations, kwargs
            return {
                node: (np.random.uniform(-1, 1), random.uniform(-1, 1)) for node in nx_graph.nodes
            }

    monkeypatch.setattr(fa2_competitor, "_load_forceatlas2", lambda: _StubForceAtlas2)

    edge_index = torch.tensor([(0, 1), (1, 2)], dtype=torch.long).t().contiguous()
    graph = DaguaGraph.from_edge_index(edge_index, 3)
    competitor = fa2_competitor.FA2Reference()

    first = competitor.layout(graph, seed=None)
    # Perturb ambient global RNG state between the two calls.
    np.random.uniform(size=100)
    random.random()
    second = competitor.layout(graph, seed=None)

    assert first.error is None and second.error is None
    assert first.pos is not None and second.pos is not None
    assert torch.equal(first.pos, second.pos)
    # And the pinned fallback matches an explicit seed=42 run.
    explicit = competitor.layout(graph, seed=42)
    assert explicit.pos is not None
    assert torch.equal(first.pos, explicit.pos)


def test_sparse_stress_reimpl_layout_succeeds_without_node_sizes_kwarg() -> None:
    """The adapter must not pass node_sizes to pipelines that reject it.

    layout_sparse_stress_pipeline() has no node_sizes parameter; the adapter
    previously passed it unconditionally, so EVERY sparse_stress_reimpl row
    failed with an unexpected-keyword TypeError (96/104 rows in the certified
    pool carry exactly that error string).
    """
    competitor = get_competitor("sparse_stress_reimpl")
    assert competitor is not None

    edge_index = (
        torch.tensor([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], dtype=torch.long).t().contiguous()
    )
    graph = DaguaGraph.from_edge_index(edge_index, 4)

    result = competitor.layout(graph, seed=7)

    assert result.error is None
    assert result.pos is not None
    assert result.pos.shape == (4, 2)
    assert torch.isfinite(result.pos).all()

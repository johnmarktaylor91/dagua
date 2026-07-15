"""Regression tests for run-scoped seeded reference overrides."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator, Optional

import torch

import dagua.eval.competitors.igraph_competitor as igraph_competitor
from dagua.eval.competitors.igraph_competitor import IgraphSugiyama
from dagua.graph import DaguaGraph
from scripts.run_benchmark import (
    POSITION_DIRNAME,
    build_record_key,
    recover_results_from_positions,
    seeds_for_engine,
)


def test_seeds_for_engine_applies_run_scoped_reference_override() -> None:
    """Overridden reference engines should use the requested seed range."""
    assert seeds_for_engine("graphviz_sfdp", seed_count=3, seed_start=42) == [None]
    assert seeds_for_engine(
        "graphviz_sfdp", seed_count=3, seed_start=42, seed_refs={"graphviz_sfdp"}
    ) == [42, 43, 44]
    assert seeds_for_engine(
        "graphviz_neato", seed_count=3, seed_start=42, seed_refs={"graphviz_sfdp"}
    ) == [None]


def test_seeds_for_engine_routes_synthetic_original_variants_by_original_engine() -> None:
    """Naming a base reference should cover all of its ``__for__`` variants."""
    synthetic_name = "graphviz_sfdp__for__classic_sfdp_default"

    assert seeds_for_engine(synthetic_name, seed_count=2, seed_start=7) == [None]
    assert seeds_for_engine(
        synthetic_name, seed_count=2, seed_start=7, seed_refs={"graphviz_sfdp"}
    ) == [7, 8]
    assert seeds_for_engine(
        synthetic_name, seed_count=2, seed_start=7, seed_refs={synthetic_name}
    ) == [7, 8]


def test_record_keys_reflect_overridden_and_deterministic_seed_shapes() -> None:
    """Record keys should only gain seed suffixes for overridden engines."""
    overridden_seeds = seeds_for_engine(
        "graphviz_sfdp", seed_count=1, seed_start=99, seed_refs={"graphviz_sfdp"}
    )
    deterministic_seeds = seeds_for_engine(
        "graphviz_neato", seed_count=1, seed_start=99, seed_refs={"graphviz_sfdp"}
    )

    assert build_record_key("tiny", "graphviz_sfdp", overridden_seeds[0]) == (
        "tiny::graphviz_sfdp::seed99"
    )
    assert build_record_key("tiny", "graphviz_neato", deterministic_seeds[0]) == (
        "tiny::graphviz_neato::deterministic"
    )


def test_position_recovery_uses_seed_reference_override(tmp_path: Any) -> None:
    """Recovered position records should enumerate overridden reference seeds."""
    graph = DaguaGraph.from_edge_list([("a", "b")])
    test_graph = SimpleNamespace(name="tiny", graph=graph, tags={"unit"})
    competitor = SimpleNamespace(name="graphviz_sfdp")
    positions_dir = tmp_path / POSITION_DIRNAME
    positions_dir.mkdir()
    torch.save(torch.zeros(graph.num_nodes, 2), positions_dir / "tiny__graphviz_sfdp__seed5.pt")

    recovered = recover_results_from_positions(
        output_dir=tmp_path,
        graphs=[test_graph],
        engines=[competitor],
        seed_count=1,
        seed_start=5,
        seed_refs={"graphviz_sfdp"},
        git_sha="test-sha",
    )

    assert sorted(recovered) == ["tiny::graphviz_sfdp::seed5"]
    assert recovered["tiny::graphviz_sfdp::seed5"].seed == 5


def test_igraph_sugiyama_enables_igraph_rng_seed(monkeypatch: Any) -> None:
    """Sugiyama should enter the igraph RNG seed context when seeded."""
    calls: list[tuple[Optional[int], bool]] = []

    class FakeIgraphGraph:
        """Minimal fake ``igraph.Graph`` for the Sugiyama adapter path."""

        def __init__(self, directed: bool = True) -> None:
            """Initialize an empty fake graph.

            Parameters
            ----------
            directed : bool, default=True
                Directed flag accepted for API compatibility.
            """
            del directed
            self.vertex_count = 0
            self.es: dict[str, list[float]] = {}

        def add_vertices(self, count: int) -> None:
            """Record the vertex count requested by the adapter.

            Parameters
            ----------
            count : int
                Number of vertices to add.
            """
            self.vertex_count = count

        def add_edges(self, edges: list[tuple[int, int]]) -> None:
            """Accept edge additions without invoking native igraph.

            Parameters
            ----------
            edges : list[tuple[int, int]]
                Directed edge list requested by the adapter.
            """
            del edges

        def layout(self, algorithm: str, **kwargs: Any) -> list[list[float]]:
            """Return deterministic coordinates for the requested layout.

            Parameters
            ----------
            algorithm : str
                igraph layout algorithm name.
            **kwargs : Any
                Layout keyword arguments.

            Returns
            -------
            list[list[float]]
                Two-dimensional coordinates for every fake vertex.
            """
            del kwargs
            assert algorithm == "sugiyama"
            return [[float(index), float(index + 1)] for index in range(self.vertex_count)]

    @contextmanager
    def spy_igraph_rng_seed(seed: Optional[int], enabled: bool) -> Iterator[None]:
        """Record igraph RNG context arguments.

        Parameters
        ----------
        seed : int | None
            Seed passed by the adapter.
        enabled : bool
            Whether the adapter requested igraph RNG seeding.

        Returns
        -------
        Iterator[None]
            Context manager body.
        """
        calls.append((seed, enabled))
        yield

    fake_igraph = SimpleNamespace(
        Graph=FakeIgraphGraph,
        set_random_number_generator=lambda generator: None,
    )
    monkeypatch.setitem(sys.modules, "igraph", fake_igraph)
    monkeypatch.setattr(igraph_competitor, "_igraph_rng_seed", spy_igraph_rng_seed)

    result = IgraphSugiyama().layout(DaguaGraph.from_edge_list([("a", "b")]), seed=123)

    assert result.error is None
    assert calls == [(123, True)]

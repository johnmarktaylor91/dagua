"""Tests for coarsening ops."""

from __future__ import annotations

from typing import Sequence

import pytest
import torch

from dagua.layout.classic.fmmm import _TYPE_MOON, _TYPE_PLANET, _TYPE_PLANET_WITH_MOONS, _TYPE_SUN
from dagua.layout.ops import coarsen as coarsen_ops
from dagua.layout.ops.coarsen import (
    HeavyEdgeMatching,
    LayerAwareCoarsen,
    LayerAwareCoarsenConfig,
    SolarSystemCoarsen,
    SolarSystemCoarsenConfig,
    StreamingCoarsen,
    StreamingCoarsenConfig,
)
from dagua.layout.ops.state import HierarchyLevel, LayoutProblem, RuntimeContext, SolveState


def _path_problem(num_nodes: int) -> LayoutProblem:
    """Create a path graph for coarsening tests.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the path.

    Returns
    -------
    LayoutProblem
        Path graph with unit node sizes.
    """
    sources = list(range(num_nodes - 1))
    targets = list(range(1, num_nodes))
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=torch.ones((num_nodes, 2), dtype=torch.float32),
        seed=7,
    )


def _disconnected_path_problem(component_sizes: Sequence[int]) -> LayoutProblem:
    """Create a disconnected graph from multiple path components.

    Parameters
    ----------
    component_sizes : sequence[int]
        Path lengths for each connected component.

    Returns
    -------
    LayoutProblem
        Disconnected graph with unit node sizes.
    """
    sources: list[int] = []
    targets: list[int] = []
    offset = 0
    for size in component_sizes:
        for node in range(size - 1):
            sources.append(offset + node)
            targets.append(offset + node + 1)
        offset += size

    num_nodes = sum(component_sizes)
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=torch.ones((num_nodes, 2), dtype=torch.float32),
        seed=17,
    )


def _layered_chain_problem(num_nodes: int) -> tuple[LayoutProblem, torch.Tensor]:
    """Create a chain graph with paired layer assignments.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the chain.

    Returns
    -------
    tuple[LayoutProblem, torch.Tensor]
        Problem and per-node layer assignments.
    """
    problem = _path_problem(num_nodes)
    layers = torch.arange(num_nodes, dtype=torch.long) // 2
    return problem, layers


def _hierarchy_signature(levels: Sequence[HierarchyLevel]) -> list[tuple[int, int, list[int]]]:
    """Serialize hierarchy levels into a comparison-friendly signature.

    Parameters
    ----------
    levels : sequence[HierarchyLevel]
        Hierarchy levels to summarize.

    Returns
    -------
    list[tuple[int, int, list[int]]]
        Per-level ``(num_fine, num_nodes, mapping)`` tuples.
    """
    signature: list[tuple[int, int, list[int]]] = []
    for level in levels:
        assert level.fine_to_coarse is not None
        signature.append((level.num_fine, level.num_nodes, level.fine_to_coarse.tolist()))
    return signature


def _assert_mapping_covers_all_nodes(level: HierarchyLevel) -> None:
    """Assert that a hierarchy mapping is contiguous and total.

    Parameters
    ----------
    level : HierarchyLevel
        Hierarchy level to validate.

    Returns
    -------
    None
        The assertions enforce a total fine-to-coarse mapping.
    """
    assert level.fine_to_coarse is not None
    assert level.fine_to_coarse.shape == (level.num_fine,)
    assert int(level.fine_to_coarse.min().item()) == 0
    assert int(level.fine_to_coarse.max().item()) == level.num_nodes - 1
    assert sorted(set(level.fine_to_coarse.tolist())) == list(range(level.num_nodes))


def _grouped_fine_nodes(level: HierarchyLevel) -> list[list[int]]:
    """Return fine nodes grouped by coarse assignment.

    Parameters
    ----------
    level : HierarchyLevel
        Hierarchy level whose mapping should be inverted.

    Returns
    -------
    list[list[int]]
        Fine-node indices for each coarse node.
    """
    assert level.fine_to_coarse is not None
    groups: list[list[int]] = [[] for _ in range(level.num_nodes)]
    for fine_node, coarse_node in enumerate(level.fine_to_coarse.tolist()):
        groups[int(coarse_node)].append(fine_node)
    return groups


@pytest.mark.parametrize("num_nodes", [12, 20])
def test_heavy_edge_matching_builds_valid_hierarchy_for_multiple_path_sizes(num_nodes: int) -> None:
    """HeavyEdgeMatching should build a strictly shrinking hierarchy on larger paths."""
    problem = _path_problem(num_nodes)

    state = HeavyEdgeMatching().apply(problem, SolveState(), RuntimeContext())

    assert state.hierarchy is not None
    assert state.hierarchy
    assert state.hierarchy[0].num_nodes < problem.num_nodes
    for level in state.hierarchy:
        _assert_mapping_covers_all_nodes(level)
        assert level.node_sizes is not None
        assert level.node_sizes.shape == (level.num_nodes, 2)
        if level.edge_index is not None and level.edge_index.numel() > 0:
            assert int(level.edge_index.min().item()) >= 0
            assert int(level.edge_index.max().item()) < level.num_nodes


def test_heavy_edge_matching_builds_valid_hierarchy_for_20_node_graph() -> None:
    """Heavy-edge matching should produce a valid finest-to-coarsest hierarchy."""
    problem = _path_problem(20)
    state = HeavyEdgeMatching().apply(problem, SolveState(), RuntimeContext())

    assert state.hierarchy is not None
    assert state.hierarchy

    expected_fine_nodes = problem.num_nodes
    previous_coarse_nodes = problem.num_nodes
    for level in state.hierarchy:
        assert level.fine_to_coarse is not None
        assert level.edge_index is not None
        assert level.node_sizes is not None
        assert level.num_fine == expected_fine_nodes
        assert level.fine_to_coarse.shape == (level.num_fine,)
        assert level.node_sizes.shape == (level.num_nodes, 2)
        assert level.num_nodes < previous_coarse_nodes
        assert int(level.fine_to_coarse.min().item()) >= 0
        assert int(level.fine_to_coarse.max().item()) == level.num_nodes - 1

        if level.edge_index.numel() > 0:
            assert int(level.edge_index.min().item()) >= 0
            assert int(level.edge_index.max().item()) < level.num_nodes

        expected_fine_nodes = level.num_nodes
        previous_coarse_nodes = level.num_nodes


def test_heavy_edge_matching_first_level_has_fewer_nodes_than_input() -> None:
    """Heavy-edge matching should coarsen the graph on its first hierarchy level."""

    problem = _path_problem(12)

    result = HeavyEdgeMatching().apply(problem, SolveState(), RuntimeContext())

    assert result.hierarchy is not None
    assert result.hierarchy[0].num_nodes < problem.num_nodes
    assert result.hierarchy[0].fine_to_coarse is not None
    assert result.hierarchy[0].fine_to_coarse.shape == (problem.num_nodes,)


def test_heavy_edge_matching_handles_two_node_graph() -> None:
    """HeavyEdgeMatching should handle the smallest connected graph without invalid output."""
    problem = _path_problem(2)

    state = HeavyEdgeMatching().apply(problem, SolveState(), RuntimeContext())

    assert state.hierarchy == []


def test_heavy_edge_matching_handles_disconnected_graph() -> None:
    """HeavyEdgeMatching should still map every node on disconnected inputs."""
    problem = _disconnected_path_problem((6, 6))

    state = HeavyEdgeMatching().apply(problem, SolveState(), RuntimeContext())

    assert state.hierarchy is not None
    assert state.hierarchy
    _assert_mapping_covers_all_nodes(state.hierarchy[0])


def test_heavy_edge_matching_seeded_rng_is_reproducible() -> None:
    """HeavyEdgeMatching should be deterministic for the same seed."""
    problem = _path_problem(12)

    first = HeavyEdgeMatching().apply(problem, SolveState(), RuntimeContext())
    second = HeavyEdgeMatching().apply(problem, SolveState(), RuntimeContext())

    assert first.hierarchy is not None
    assert second.hierarchy is not None
    assert _hierarchy_signature(first.hierarchy) == _hierarchy_signature(second.hierarchy)


def test_heavy_edge_matching_fine_to_coarse_maps_all_nodes_on_every_level() -> None:
    """HeavyEdgeMatching should cover every fine node at each hierarchy level."""
    problem = _path_problem(20)

    state = HeavyEdgeMatching().apply(problem, SolveState(), RuntimeContext())

    assert state.hierarchy is not None
    for level in state.hierarchy:
        _assert_mapping_covers_all_nodes(level)


@pytest.mark.parametrize("num_nodes", [6, 12])
def test_solar_system_coarsen_builds_valid_hierarchy(num_nodes: int) -> None:
    """SolarSystemCoarsen should build hierarchy levels aligned with prolong steps."""
    problem = _path_problem(num_nodes)

    state = SolarSystemCoarsen(SolarSystemCoarsenConfig(target=1, random_tries=4)).apply(
        problem,
        SolveState(),
        RuntimeContext(),
    )

    assert state.hierarchy is not None
    assert state.hierarchy
    steps = state.extras["solar_system_steps"]
    assert len(steps) == len(state.hierarchy)
    for level, step in zip(state.hierarchy, steps):
        _assert_mapping_covers_all_nodes(level)
        assert torch.equal(step.mapping.to(dtype=torch.long), level.fine_to_coarse)


def test_solar_system_coarsen_builds_sun_planet_moon_structure() -> None:
    """SolarSystemCoarsen should classify a richer path into suns, planets, and moons."""
    problem = _path_problem(6)

    state = SolarSystemCoarsen(SolarSystemCoarsenConfig(target=1, random_tries=3)).apply(
        problem,
        SolveState(),
        RuntimeContext(),
    )

    step = state.extras["solar_system_steps"][0]
    node_types = set(step.node_types)
    assert _TYPE_SUN in node_types
    assert _TYPE_PLANET in node_types or _TYPE_PLANET_WITH_MOONS in node_types
    assert _TYPE_MOON in node_types
    assert any(children for children in step.moon_children)


def test_solar_system_coarsen_handles_tiny_graph() -> None:
    """SolarSystemCoarsen should still produce a valid single-level hierarchy on tiny graphs."""
    problem = _path_problem(3)

    state = SolarSystemCoarsen(SolarSystemCoarsenConfig(target=1, random_tries=2)).apply(
        problem,
        SolveState(),
        RuntimeContext(),
    )

    assert state.hierarchy is not None
    assert len(state.hierarchy) == 1
    _assert_mapping_covers_all_nodes(state.hierarchy[0])


def test_solar_system_coarsen_seed_is_reproducible() -> None:
    """SolarSystemCoarsen should be deterministic for a fixed seed."""
    problem = _path_problem(12)

    first = SolarSystemCoarsen(SolarSystemCoarsenConfig(target=1, random_tries=5)).apply(
        problem,
        SolveState(),
        RuntimeContext(),
    )
    second = SolarSystemCoarsen(SolarSystemCoarsenConfig(target=1, random_tries=5)).apply(
        problem,
        SolveState(),
        RuntimeContext(),
    )

    assert first.hierarchy is not None
    assert second.hierarchy is not None
    assert _hierarchy_signature(first.hierarchy) == _hierarchy_signature(second.hierarchy)
    assert [step.mapping.tolist() for step in first.extras["solar_system_steps"]] == [
        step.mapping.tolist() for step in second.extras["solar_system_steps"]
    ]


def test_solar_system_coarsen_uses_configured_random_tries(monkeypatch: pytest.MonkeyPatch) -> None:
    """SolarSystemCoarsen should pass the configured random-tries count into sun selection."""
    problem = _path_problem(8)
    observed_random_tries: list[int] = []
    original_method = coarsen_ops._RandomNodeSet.get_random_node_with_highest_star_mass

    def _recording_method(
        self: object,
        rng: object,
        random_tries: int,
    ) -> int:
        """Record the configured retry count before delegating to the real method."""
        observed_random_tries.append(random_tries)
        return original_method(self, rng, random_tries)

    monkeypatch.setattr(
        coarsen_ops._RandomNodeSet,
        "get_random_node_with_highest_star_mass",
        _recording_method,
    )

    SolarSystemCoarsen(SolarSystemCoarsenConfig(target=1, random_tries=7)).apply(
        problem,
        SolveState(),
        RuntimeContext(),
    )

    assert observed_random_tries
    assert set(observed_random_tries) == {7}


def test_solar_system_coarsen_fine_to_coarse_maps_all_nodes() -> None:
    """SolarSystemCoarsen should cover every fine node on each generated level."""
    problem = _path_problem(12)

    state = SolarSystemCoarsen(SolarSystemCoarsenConfig(target=1, random_tries=4)).apply(
        problem,
        SolveState(),
        RuntimeContext(),
    )

    assert state.hierarchy is not None
    for level in state.hierarchy:
        _assert_mapping_covers_all_nodes(level)


def test_layer_aware_coarsen_groups_only_within_layers() -> None:
    """LayerAwareCoarsen should never merge fine nodes from different layers."""
    problem, layers = _layered_chain_problem(12)

    state = LayerAwareCoarsen(LayerAwareCoarsenConfig(min_nodes=2, max_levels=4)).apply(
        problem,
        SolveState(layers=layers.clone()),
        RuntimeContext(),
    )

    assert state.hierarchy is not None
    first_level = state.hierarchy[0]
    for group in _grouped_fine_nodes(first_level):
        group_layers = {int(layers[index].item()) for index in group}
        assert len(group_layers) == 1


def test_layer_aware_coarsen_propagates_coarse_layers_from_fine_groups() -> None:
    """LayerAwareCoarsen should assign each coarse node the layer of its fine group."""
    problem, layers = _layered_chain_problem(12)

    state = LayerAwareCoarsen(LayerAwareCoarsenConfig(min_nodes=2, max_levels=4)).apply(
        problem,
        SolveState(layers=layers.clone()),
        RuntimeContext(),
    )

    first_level = state.hierarchy[0]
    assert first_level.coarse_layer_assignments is not None
    for coarse_node, group in enumerate(_grouped_fine_nodes(first_level)):
        expected_layer = max(int(layers[index].item()) for index in group)
        assert int(first_level.coarse_layer_assignments[coarse_node].item()) == expected_layer


def test_layer_aware_coarsen_min_nodes_stops_coarsening() -> None:
    """LayerAwareCoarsen should skip hierarchy construction when already below min_nodes."""
    problem, layers = _layered_chain_problem(12)

    state = LayerAwareCoarsen(LayerAwareCoarsenConfig(min_nodes=20, max_levels=4)).apply(
        problem,
        SolveState(layers=layers.clone()),
        RuntimeContext(),
    )

    assert state.hierarchy == []


def test_layer_aware_coarsen_respects_max_levels_cap() -> None:
    """LayerAwareCoarsen should not build more levels than configured."""
    problem, layers = _layered_chain_problem(12)

    state = LayerAwareCoarsen(LayerAwareCoarsenConfig(min_nodes=2, max_levels=1)).apply(
        problem,
        SolveState(layers=layers.clone()),
        RuntimeContext(),
    )

    assert state.hierarchy is not None
    assert len(state.hierarchy) == 1


def test_layer_aware_coarsen_uses_configured_hub_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """LayerAwareCoarsen should pass its hub threshold through the shared builder."""
    problem, layers = _layered_chain_problem(12)
    recorded: list[float] = []

    def _fake_build_layered_hierarchy(
        problem: LayoutProblem,
        state: SolveState,
        min_nodes: int,
        max_levels: int,
        hub_threshold_percentile: float | None = None,
        streaming_threshold: int | None = None,
    ) -> list[HierarchyLevel]:
        """Capture forwarded configuration without running the real builder."""
        del problem, state, min_nodes, max_levels, streaming_threshold
        recorded.append(float(hub_threshold_percentile))
        return []

    monkeypatch.setattr(coarsen_ops, "_build_layered_hierarchy", _fake_build_layered_hierarchy)

    LayerAwareCoarsen(
        LayerAwareCoarsenConfig(min_nodes=2, max_levels=3, hub_threshold_percentile=12.5)
    ).apply(problem, SolveState(layers=layers.clone()), RuntimeContext())

    assert recorded == [12.5]


def test_layer_aware_coarsen_clears_cached_solar_steps() -> None:
    """LayerAwareCoarsen should discard solar-system prolongation metadata."""
    problem, layers = _layered_chain_problem(12)
    state = SolveState(layers=layers.clone(), extras={"solar_system_steps": ["stale"]})

    result = LayerAwareCoarsen(LayerAwareCoarsenConfig(min_nodes=20)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert "solar_system_steps" not in result.extras


def test_layer_aware_coarsen_mapping_is_contiguous() -> None:
    """LayerAwareCoarsen should assign contiguous coarse ids on the first level."""
    problem, layers = _layered_chain_problem(12)

    state = LayerAwareCoarsen(LayerAwareCoarsenConfig(min_nodes=2, max_levels=4)).apply(
        problem,
        SolveState(layers=layers.clone()),
        RuntimeContext(),
    )

    _assert_mapping_covers_all_nodes(state.hierarchy[0])


def test_streaming_coarsen_matches_layer_aware_coarsen_for_small_coarsenable_graph() -> None:
    """StreamingCoarsen should match the non-streaming builder on a modest graph."""
    problem, layers = _layered_chain_problem(2200)

    layer_aware = LayerAwareCoarsen(LayerAwareCoarsenConfig(min_nodes=2000, max_levels=20)).apply(
        problem, SolveState(layers=layers.clone()), RuntimeContext()
    )
    streaming = StreamingCoarsen(StreamingCoarsenConfig(chunk_size=1)).apply(
        problem,
        SolveState(layers=layers.clone()),
        RuntimeContext(),
    )

    assert layer_aware.hierarchy is not None
    assert streaming.hierarchy is not None
    assert _hierarchy_signature(layer_aware.hierarchy) == _hierarchy_signature(streaming.hierarchy)


def test_streaming_coarsen_handles_chunk_size_larger_than_num_nodes() -> None:
    """StreamingCoarsen should handle a chunk size above the graph size."""
    problem, layers = _layered_chain_problem(12)

    state = StreamingCoarsen(StreamingCoarsenConfig(chunk_size=1000)).apply(
        problem,
        SolveState(layers=layers.clone()),
        RuntimeContext(),
    )

    assert state.hierarchy == []


def test_streaming_coarsen_clears_cached_solar_steps() -> None:
    """StreamingCoarsen should discard solar-system metadata before writing hierarchy output."""
    problem, layers = _layered_chain_problem(12)
    state = SolveState(layers=layers.clone(), extras={"solar_system_steps": ["stale"]})

    result = StreamingCoarsen(StreamingCoarsenConfig(chunk_size=1000)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert "solar_system_steps" not in result.extras


def test_streaming_coarsen_uses_configured_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """StreamingCoarsen should forward its chunk size to the shared builder."""
    problem, layers = _layered_chain_problem(12)
    recorded: list[int] = []

    def _fake_build_layered_hierarchy(
        problem: LayoutProblem,
        state: SolveState,
        min_nodes: int,
        max_levels: int,
        hub_threshold_percentile: float | None = None,
        streaming_threshold: int | None = None,
    ) -> list[HierarchyLevel]:
        """Capture forwarded streaming configuration without running the real builder."""
        del problem, state, min_nodes, max_levels, hub_threshold_percentile
        recorded.append(int(streaming_threshold))
        return []

    monkeypatch.setattr(coarsen_ops, "_build_layered_hierarchy", _fake_build_layered_hierarchy)

    StreamingCoarsen(StreamingCoarsenConfig(chunk_size=321)).apply(
        problem,
        SolveState(layers=layers.clone()),
        RuntimeContext(),
    )

    assert recorded == [321]


def test_streaming_coarsen_mapping_is_contiguous_when_it_coarsens() -> None:
    """StreamingCoarsen should still produce a total fine-to-coarse mapping when active."""
    problem, layers = _layered_chain_problem(2200)

    state = StreamingCoarsen(StreamingCoarsenConfig(chunk_size=1)).apply(
        problem,
        SolveState(layers=layers.clone()),
        RuntimeContext(),
    )

    assert state.hierarchy is not None
    assert state.hierarchy
    _assert_mapping_covers_all_nodes(state.hierarchy[0])

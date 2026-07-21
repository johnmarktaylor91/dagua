"""Off-corpus probe generator tests: seeded determinism and family invariants."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import offcorpus_probe_gen as probe_gen  # noqa: E402


@pytest.mark.smoke
def test_same_seed_regenerates_identical_probe() -> None:
    """Probes are fully seed-determined (regenerable each round)."""
    for family in probe_gen.FAMILIES:
        first = probe_gen.generate_probe(family, 1234)
        second = probe_gen.generate_probe(family, 1234)
        assert first.name == second.name == f"probe_{family}_1234"
        assert first.graph.to_json() == second.graph.to_json()


@pytest.mark.smoke
def test_different_seeds_differ() -> None:
    """Different seeds must not produce the same graph."""
    for family in probe_gen.FAMILIES:
        first = probe_gen.generate_probe(family, 1234)
        second = probe_gen.generate_probe(family, 4321)
        assert first.graph.to_json() != second.graph.to_json()


@pytest.mark.smoke
def test_family_invariants() -> None:
    """Each family carries its structural signature and probe marker."""
    nested = probe_gen.generate_probe("nested_directed_cluster", 7)
    assert nested.graph.clusters, "nested probe must have clusters"
    parents = [parent for parent in nested.graph.cluster_parents.values() if parent]
    assert parents, "nested probe must have child clusters (nesting)"
    assert nested.graph.is_semantically_directed is True

    clustered = probe_gen.generate_probe("clustered_medium", 7)
    assert 4 <= len(clustered.graph.clusters) <= 6
    assert 4 * 15 <= clustered.graph.num_nodes <= 6 * 25

    skinny = probe_gen.generate_probe("deep_skinny_dag", 7)
    assert 400 <= skinny.graph.num_nodes <= 600

    geometric = probe_gen.generate_probe("geometric_random", 7)
    assert geometric.graph.is_semantically_directed is False
    assert "undirected" in geometric.tags

    for probe in (nested, clustered, skinny, geometric):
        assert "offcorpus_probe" in probe.tags
        assert probe.name.startswith("probe_")


@pytest.mark.smoke
def test_probe_json_roundtrip_is_scoreable(tmp_path: Path) -> None:
    """Probe JSON round-trips with clusters intact and node sizes computed."""
    probe = probe_gen.generate_probe("nested_directed_cluster", 42)
    payload = probe_gen.probe_payload(probe, "nested_directed_cluster", 42)
    assert payload["non_holdout"] is True
    path = tmp_path / f"{probe.name}.json"
    path.write_text(__import__("json").dumps(payload))
    loaded = probe_gen.load_probe(path)
    assert loaded.name == probe.name
    assert loaded.graph.num_nodes == probe.graph.num_nodes
    assert loaded.graph.num_edges == probe.graph.num_edges
    assert set(loaded.graph.clusters) == set(probe.graph.clusters)
    assert loaded.graph.cluster_parents == probe.graph.cluster_parents
    # The score_position tripwire requires measured node sizes.
    assert loaded.graph.node_sizes is not None
    assert loaded.graph.is_semantically_directed is True


@pytest.mark.smoke
def test_probe_names_cannot_collide_with_corpus() -> None:
    """The probe_ prefix keeps probes disjoint from corpus graph names."""
    for family in probe_gen.FAMILIES:
        probe = probe_gen.generate_probe(family, 99)
        assert probe.name.startswith("probe_")
    with pytest.raises(ValueError):
        probe_gen.generate_probe("not_a_family", 1)

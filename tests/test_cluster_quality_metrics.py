"""Regression tests for cluster-quality honest-ruler metrics."""

from __future__ import annotations

import pytest
import torch

from dagua.metrics import (
    _CLUSTER_WEIGHTS,
    _COMMON_WEIGHTS,
    cluster_compactness_score,
    cluster_edge_intrusion_score,
    cluster_exclusion_score,
    cluster_label_occlusion_score,
    cluster_nesting_fidelity_score,
    cluster_quality_metrics,
    cluster_sibling_overlap_score,
    composite,
    composite_auto,
    composite_undirected,
)


def _cluster_inputs(pos: torch.Tensor) -> tuple[torch.Tensor, dict[str, list[int]], dict[str, str]]:
    """Return shared node sizes, flat clusters, and labels for small tests.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``. The argument is accepted so tests
        can document alignment between positions and cluster fixtures.

    Returns
    -------
    tuple[torch.Tensor, dict[str, list[int]], dict[str, str]]
        Node sizes, cluster membership, and explicit cluster labels.
    """
    del pos
    return (
        torch.full((5, 2), 2.0),
        {"left": [0, 1], "right": [2, 3]},
        {"left": "Left", "right": "Right"},
    )


def _clean_nested_fixture() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, list[int]],
    dict[str, str],
    dict[str, str],
]:
    """Return a canonical clean labeled nested cluster layout.

    Returns
    -------
    tuple
        Positions, empty edge index, node sizes, cluster membership, parent
        mapping, and labels.
    """
    pos = torch.tensor(
        [
            [-60.0, 0.0],
            [-40.0, 0.0],
            [40.0, 0.0],
            [60.0, 0.0],
            [-15.0, -85.0],
            [0.0, -105.0],
        ]
    )
    sizes = torch.full((6, 2), 10.0)
    edges = torch.zeros((2, 0), dtype=torch.long)
    clusters = {"root": [0, 1, 2, 3, 4, 5], "left": [0, 1], "right": [2, 3]}
    parents = {"left": "root", "right": "root"}
    labels = {"root": "Root", "left": "Left", "right": "Right"}
    return pos, edges, sizes, clusters, parents, labels


def test_cluster_quality_none_applicability_and_composite_invariance() -> None:
    """Absent cluster metadata returns None and leaves composites bit-identical."""
    pos = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    sizes = torch.full((3, 2), 2.0)
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    result = cluster_quality_metrics(pos, edges, sizes, None, None, None)

    for name in _CLUSTER_WEIGHTS:
        assert result[name] is None

    frozen = {name: 0.625 for name in _COMMON_WEIGHTS}
    with_none = {**frozen, **{name: None for name in _CLUSTER_WEIGHTS}}
    assert composite(frozen) == composite(with_none)
    assert composite_undirected(frozen) == composite_undirected(with_none)


def test_cluster_quality_clean_nested_labeled_layout_scores_high() -> None:
    """A canonical clean nested labeled layout gets no synthetic cluster penalties."""
    pos, edges, sizes, clusters, parents, labels = _clean_nested_fixture()

    result = cluster_quality_metrics(pos, edges, sizes, clusters, parents, labels)

    assert result["cluster_nesting_fidelity_score"] == pytest.approx(1.0)
    assert result["cluster_exclusion_score"] == pytest.approx(1.0)
    assert result["cluster_label_occlusion_score"] >= 0.95
    assert result["cluster_compactness_score"] >= 0.95
    assert result["cluster_sibling_overlap_score"] == pytest.approx(1.0)


def test_cluster_quality_asymmetric_membership_clean_layout_scores_high() -> None:
    """Asymmetric descendant placement does not translate the derived parent box."""
    pos, edges, sizes, clusters, parents, labels = _clean_nested_fixture()

    result = cluster_quality_metrics(pos, edges, sizes, clusters, parents, labels)

    assert result["cluster_nesting_fidelity_score"] == pytest.approx(1.0)
    assert result["cluster_nesting_violations"] == 0
    assert result["cluster_exclusion_score"] == pytest.approx(1.0)


def test_cluster_quality_deep_singleton_chain_has_no_nesting_violation() -> None:
    """Bottom-up padding makes equal-member singleton chains structurally nested."""
    pos = torch.tensor([[0.0, 0.0]])
    sizes = torch.full((1, 2), 10.0)
    clusters = {f"c{index}": [0] for index in range(8)}
    parents = {f"c{index}": f"c{index - 1}" for index in range(1, 8)}
    labels = {f"c{index}": "X" for index in range(8)}

    result = cluster_nesting_fidelity_score(pos, sizes, clusters, parents, labels)

    assert result["cluster_nesting_fidelity_score"] == pytest.approx(1.0)
    assert result["cluster_nesting_violations"] == 0


def test_cluster_quality_rename_invariance_for_nested_chain() -> None:
    """Pure cluster renaming does not alter nesting fidelity."""
    pos = torch.tensor([[0.0, 0.0]])
    sizes = torch.full((1, 2), 10.0)
    first_clusters = {"alpha": [0], "beta": [0], "gamma": [0]}
    first_parents = {"beta": "alpha", "gamma": "beta"}
    second_clusters = {"z": [0], "m": [0], "a": [0]}
    second_parents = {"m": "z", "a": "m"}
    labels = {"alpha": "X", "beta": "X", "gamma": "X"}
    renamed_labels = {"z": "X", "m": "X", "a": "X"}

    first = cluster_nesting_fidelity_score(pos, sizes, first_clusters, first_parents, labels)
    second = cluster_nesting_fidelity_score(
        pos, sizes, second_clusters, second_parents, renamed_labels
    )

    assert second["cluster_nesting_fidelity_score"] == first["cluster_nesting_fidelity_score"]
    assert second["cluster_nesting_violations"] == first["cluster_nesting_violations"]


def test_cluster_quality_long_child_label_does_not_break_nesting() -> None:
    """Parent geometry absorbs a child's long label band during bottom-up derivation."""
    pos = torch.tensor([[0.0, 0.0], [5.0, 0.0]])
    sizes = torch.full((2, 2), 10.0)
    clusters = {"parent": [0, 1], "child": [0, 1]}
    parents = {"child": "parent"}
    labels = {"parent": "P", "child": "child-label-that-is-deliberately-long"}

    result = cluster_nesting_fidelity_score(pos, sizes, clusters, parents, labels)

    assert result["cluster_nesting_fidelity_score"] == pytest.approx(1.0)
    assert result["cluster_nesting_violations"] == 0


def test_cluster_sibling_sampling_is_mapping_order_invariant() -> None:
    """Capped sibling sampling is stable when equivalent mappings are reordered."""
    node_count = 210
    positions = torch.tensor(
        [[float(index % 5) * 2.0, float(index // 5) * 40.0] for index in range(node_count)]
    )
    sizes = torch.full((node_count, 2), 20.0)
    clusters = {f"c{index:03d}": [index] for index in range(node_count)}
    reversed_clusters = dict(reversed(list(clusters.items())))
    labels = {name: "X" for name in clusters}
    reversed_labels = dict(reversed(list(labels.items())))

    small = cluster_sibling_overlap_score(positions, sizes, clusters, {}, labels, pair_cap=2)
    small_reordered = cluster_sibling_overlap_score(
        positions, sizes, reversed_clusters, {}, reversed_labels, pair_cap=2
    )
    production = cluster_sibling_overlap_score(positions, sizes, clusters, {}, labels)
    production_reordered = cluster_sibling_overlap_score(
        positions, sizes, reversed_clusters, {}, reversed_labels
    )

    assert small["cluster_sibling_overlap_score"] == pytest.approx(
        small_reordered["cluster_sibling_overlap_score"]
    )
    assert production["cluster_sibling_overlap_pairs"] == 20_000
    assert production_reordered["cluster_sibling_overlap_pairs"] == 20_000
    assert production["cluster_sibling_overlap_score"] == pytest.approx(
        production_reordered["cluster_sibling_overlap_score"]
    )


def test_cluster_intrusion_delta_is_node_count_independent() -> None:
    """A fixed foreign-node and foreign-edge intrusion keeps the same score at scale."""
    base_positions = torch.tensor([[0.0, 0.0], [5.0, 0.0], [2.5, 0.0], [-40.0, 0.0], [40.0, 0.0]])
    clusters = {"center": [0, 1]}
    labels = {"center": "Center"}
    edges = torch.tensor([[3], [4]], dtype=torch.long)

    def score_with_extra_nodes(extra_nodes: int) -> tuple[float, float]:
        """Score the same intrusion after appending unrelated far-away nodes.

        Parameters
        ----------
        extra_nodes : int
            Number of unrelated nodes to append.

        Returns
        -------
        tuple[float, float]
            Exclusion and edge-intrusion scores.
        """
        extras = torch.tensor(
            [[1000.0 + float(index) * 20.0, 1000.0] for index in range(extra_nodes)]
        )
        pos = torch.cat((base_positions, extras), dim=0) if extra_nodes else base_positions
        sizes = torch.full((pos.shape[0], 2), 10.0)
        exclusion = cluster_exclusion_score(pos, sizes, clusters, {}, labels)[
            "cluster_exclusion_score"
        ]
        edge_intrusion = cluster_edge_intrusion_score(pos, edges, sizes, clusters, {}, labels)[
            "cluster_edge_intrusion_score"
        ]
        return float(exclusion), float(edge_intrusion)

    small = score_with_extra_nodes(0)
    large = score_with_extra_nodes(200)

    assert large[0] == pytest.approx(small[0])
    assert large[1] == pytest.approx(small[1])


def test_cluster_degenerate_scale_zeroes_new_cluster_terms() -> None:
    """Point-collapsed layouts do not receive free cluster-quality composite credit."""
    base = {name: 1.0 for name in _COMMON_WEIGHTS}
    base.update({name: 1.0 for name in _CLUSTER_WEIGHTS})
    base["edge_length_mean"] = 0.0
    base["node_diag_mean"] = 10.0
    zeroed = {**base, **{name: 0.0 for name in _CLUSTER_WEIGHTS}}

    assert composite_auto(base) == pytest.approx(composite_auto(zeroed))


def test_cluster_quality_metrics_are_deterministic() -> None:
    """Repeated scoring of the same clustered layout is exactly deterministic."""
    pos = torch.tensor([[-12.0, 0.0], [-8.0, 0.0], [8.0, 0.0], [12.0, 0.0], [0.0, 50.0]])
    sizes, clusters, labels = _cluster_inputs(pos)
    edges = torch.tensor([[0, 2, 4], [1, 3, 0]], dtype=torch.long)

    first = cluster_quality_metrics(pos, edges, sizes, clusters, {}, labels)
    second = cluster_quality_metrics(pos, edges, sizes, clusters, {}, labels)

    assert first == second


def test_cluster_exclusion_drops_for_intruding_foreign_node() -> None:
    """A non-descendant node inside a derived cluster box lowers exclusion."""
    clean_pos = torch.tensor([[-12.0, 0.0], [-8.0, 0.0], [8.0, 0.0], [12.0, 0.0], [0.0, 50.0]])
    bad_pos = clean_pos.clone()
    bad_pos[4] = torch.tensor([-10.0, 0.0])
    sizes, clusters, labels = _cluster_inputs(clean_pos)

    clean = cluster_exclusion_score(clean_pos, sizes, clusters, {}, labels)[
        "cluster_exclusion_score"
    ]
    bad = cluster_exclusion_score(bad_pos, sizes, clusters, {}, labels)["cluster_exclusion_score"]

    assert bad < clean


def test_cluster_sibling_overlap_drops_for_overlapping_siblings() -> None:
    """Same-parent derived boxes that overlap are penalized."""
    clean_pos = torch.tensor([[-20.0, 0.0], [-16.0, 0.0], [16.0, 0.0], [20.0, 0.0], [0.0, 50.0]])
    bad_pos = torch.tensor([[0.0, 0.0], [4.0, 0.0], [3.0, 0.0], [7.0, 0.0], [0.0, 50.0]])
    sizes, clusters, labels = _cluster_inputs(clean_pos)

    clean = cluster_sibling_overlap_score(clean_pos, sizes, clusters, {}, labels)[
        "cluster_sibling_overlap_score"
    ]
    bad = cluster_sibling_overlap_score(bad_pos, sizes, clusters, {}, labels)[
        "cluster_sibling_overlap_score"
    ]

    assert bad < clean


def test_cluster_nesting_fidelity_drops_for_wrong_minimal_enclosure() -> None:
    """A sibling visually enclosing a child beats the declared parent and is penalized."""
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [-15.0, 0.0], [15.0, 0.0], [40.0, 0.0]])
    sizes = torch.full((5, 2), 2.0)
    clusters = {"parent": [0, 1, 2, 3], "child": [0, 1], "sibling": [2, 3]}
    parents = {"child": "parent", "sibling": "parent"}
    labels = {"parent": "Parent", "child": "Child", "sibling": "Sibling"}

    score = cluster_nesting_fidelity_score(pos, sizes, clusters, parents, labels)[
        "cluster_nesting_fidelity_score"
    ]

    assert score < 1.0


def test_cluster_edge_intrusion_drops_for_bypassing_foreign_edge() -> None:
    """An edge with neither endpoint in the cluster crossing its box is penalized."""
    clean_pos = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [40.0, -10.0], [40.0, 10.0], [80.0, 0.0]])
    bad_pos = clean_pos.clone()
    bad_pos[2] = torch.tensor([-40.0, 0.0])
    bad_pos[3] = torch.tensor([40.0, 0.0])
    sizes = torch.full((5, 2), 2.0)
    clusters = {"center": [0, 1]}
    labels = {"center": "Center"}
    edges = torch.tensor([[2], [3]], dtype=torch.long)

    clean = cluster_edge_intrusion_score(clean_pos, edges, sizes, clusters, {}, labels)[
        "cluster_edge_intrusion_score"
    ]
    bad = cluster_edge_intrusion_score(bad_pos, edges, sizes, clusters, {}, labels)[
        "cluster_edge_intrusion_score"
    ]

    assert bad < clean


def test_cluster_label_occlusion_drops_for_foreign_node_in_label_band() -> None:
    """A foreign node overlapping an explicit cluster label band is penalized."""
    clean_pos = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 60.0], [40.0, 0.0], [60.0, 0.0]])
    bad_pos = clean_pos.clone()
    bad_pos[2] = torch.tensor([0.0, 20.0])
    sizes = torch.full((5, 2), 2.0)
    clusters = {"labeled": [0, 1]}
    labels = {"labeled": "Labeled"}

    clean = cluster_label_occlusion_score(clean_pos, sizes, clusters, {}, labels)[
        "cluster_label_occlusion_score"
    ]
    bad = cluster_label_occlusion_score(bad_pos, sizes, clusters, {}, labels)[
        "cluster_label_occlusion_score"
    ]

    assert bad < clean


def test_cluster_compactness_drops_only_above_saturating_band() -> None:
    """Dispersed members are penalized, but tight members both saturate at 1."""
    tight_a = torch.tensor([[0.0, 0.0], [2.0, 0.0], [40.0, 0.0], [42.0, 0.0], [80.0, 0.0]])
    tight_b = torch.tensor([[0.0, 0.0], [1.0, 0.0], [40.0, 0.0], [41.0, 0.0], [80.0, 0.0]])
    loose = torch.tensor([[0.0, 0.0], [100.0, 0.0], [40.0, 0.0], [42.0, 0.0], [80.0, 0.0]])
    sizes, clusters, labels = _cluster_inputs(tight_a)

    first_tight = cluster_compactness_score(tight_a, sizes, clusters, {}, labels)[
        "cluster_compactness_score"
    ]
    second_tight = cluster_compactness_score(tight_b, sizes, clusters, {}, labels)[
        "cluster_compactness_score"
    ]
    dispersed = cluster_compactness_score(loose, sizes, clusters, {}, labels)[
        "cluster_compactness_score"
    ]

    assert first_tight == pytest.approx(1.0)
    assert second_tight == pytest.approx(1.0)
    assert dispersed < first_tight


def test_cluster_quality_randomized_triviality_battery() -> None:
    """Randomized layouts prove every cluster metric is not identically 1.0."""
    generator = torch.Generator().manual_seed(802)
    clusters = {
        "root": [0, 1, 2, 3, 4, 5],
        "left": [0, 1, 2],
        "right": [3, 4, 5],
    }
    parents = {"left": "root", "right": "root"}
    labels = {"root": "Root", "left": "Left", "right": "Right"}
    sizes = torch.full((8, 2), 2.0)
    edges = torch.tensor([[6, 0, 2, 6], [7, 3, 4, 1]], dtype=torch.long)
    ranges: dict[str, tuple[float, float]] = {}
    rows: list[dict[str, float]] = []
    for _ in range(48):
        pos = torch.randn((8, 2), generator=generator) * 24.0
        row = cluster_quality_metrics(pos, edges, sizes, clusters, parents, labels)
        rows.append({name: float(row[name]) for name in _CLUSTER_WEIGHTS if row[name] is not None})

    for name in _CLUSTER_WEIGHTS:
        values = [row[name] for row in rows]
        ranges[name] = (min(values), max(values))
        assert min(values) < max(values)
        assert any(value < 1.0 for value in values)

    assert ranges

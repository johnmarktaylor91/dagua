"""Regression tests for cluster-quality honest-ruler metrics."""

from __future__ import annotations

import pytest
import torch

import dagua.metrics as metrics_module
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

    metrics_module._LAST_CLUSTER_TRIVIALITY_RANGES = ranges

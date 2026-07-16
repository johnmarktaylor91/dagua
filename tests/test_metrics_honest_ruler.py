"""Unit tests for the r83 final honest composite ruler."""

from __future__ import annotations

import math

import pytest
import torch

import dagua.metrics as metrics_module
from dagua.metrics import (
    _COMMON_WEIGHTS,
    _DIRECTED_WEIGHTS,
    _triu_pair_indices_from_linear,
    angular_resolution_score,
    cluster_silhouette_score,
    composite,
    composite_auto,
    composite_large,
    composite_large_auto,
    composite_large_undirected,
    composite_strict,
    composite_undirected,
    crossing_angle_score,
    edge_crossing_score,
    edge_length_deviation_score,
    full,
    gabriel_ksm_correlation,
    gabriel_score,
    isotonic_stress,
    neighborhood_preservation,
    node_occlusion_score,
    path_continuity_score,
    quick,
)


def _common_profile(value: float = 0.5) -> dict[str, float]:
    """Return a complete common-term profile with one shared value."""
    return {name: value for name in _COMMON_WEIGHTS}


def test_frozen_weight_sums() -> None:
    """The common and hierarchy-gated directed tables both sum to 100."""
    assert sum(_COMMON_WEIGHTS.values()) == 100.0
    assert sum(_DIRECTED_WEIGHTS.values()) + 0.75 * sum(_COMMON_WEIGHTS.values()) == 100.0


def test_ksm_invariances_aspect_and_chain_artifact() -> None:
    """KSM has the required similarity invariances but preserves aspect."""
    positions = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    )
    edges = torch.tensor([[0, 1, 3, 4, 0, 1, 2], [1, 2, 4, 5, 3, 4, 5]], dtype=torch.long)
    base = isotonic_stress(positions, edges, n_sources=20, n_targets=20)["ksm_score"]
    angle = math.radians(37.0)
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    variants = (
        positions + torch.tensor([7.0, -3.0]),
        positions @ rotation.T,
        positions * torch.tensor([-1.0, 1.0]),
        positions * 4.25,
    )
    for variant in variants:
        assert isotonic_stress(variant, edges, n_sources=20, n_targets=20)[
            "ksm_score"
        ] == pytest.approx(base, abs=1e-6)
    anisotropic = positions * torch.tensor([3.0, 1.0])
    assert isotonic_stress(anisotropic, edges, n_sources=20, n_targets=20)[
        "ksm_score"
    ] != pytest.approx(base, abs=1e-3)

    chain = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    straight = torch.stack((torch.zeros(5), torch.arange(5, dtype=torch.float32)), dim=1)
    jittered = straight.clone()
    jittered[:, 0] = torch.tensor([0.0, 0.2, -0.2, 0.2, 0.0])
    assert (
        isotonic_stress(jittered, chain)["ksm_score"]
        <= isotonic_stress(straight, chain)["ksm_score"]
    )
    assert isotonic_stress(torch.zeros_like(straight), chain)["ksm_score"] == 0.0


def test_edge_crossing_sqrt_degree_correction_and_k4_order() -> None:
    """EC uses degree-corrected capacity, square-root scaling, and orders K4."""
    edges = torch.tensor([[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]], dtype=torch.long)
    planar = torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0], [1.0, 0.7]])
    crossed = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    good = edge_crossing_score(planar, edges)
    bad = edge_crossing_score(crossed, edges)
    assert good["edge_crossing_score"] > bad["edge_crossing_score"]
    assert bad["crossing_c_max"] == 3
    assert bad["edge_crossing_score"] == pytest.approx(
        1.0 - math.sqrt(min(1.0, bad["crossing_count"] / 3.0))
    )
    star = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    assert edge_crossing_score(torch.rand(4, 2), star)["edge_crossing_score"] == 1.0


def test_triu_pair_decoder_matches_torch_order() -> None:
    """Decoded sampled crossing pairs match ``torch.triu_indices`` order."""
    matrix_size = 37
    first, second = torch.triu_indices(matrix_size, matrix_size, offset=1)
    selected = torch.tensor([0, 1, 35, 36, 37, 100, first.numel() - 1], dtype=torch.long)

    decoded_first, decoded_second = _triu_pair_indices_from_linear(selected, matrix_size)

    assert torch.equal(decoded_first, first[selected])
    assert torch.equal(decoded_second, second[selected])


def test_node_occlusion_saturates_monotonically() -> None:
    """A single overlap is a partial penalty and additional overlaps reduce NO."""
    sizes = torch.ones((3, 2))
    one = node_occlusion_score(torch.tensor([[0.0, 0.0], [0.2, 0.0], [3.0, 0.0]]), sizes)
    three = node_occlusion_score(torch.zeros((3, 2)), sizes)
    assert 0.0 < one["node_occlusion_score"] < 1.0
    assert one["node_occlusion_score"] > three["node_occlusion_score"]


def test_neighborhood_preservation_is_deterministic_and_directional() -> None:
    """Frozen NP rows give perfect cycle neighborhoods and punish scrambling."""
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    square = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    perfect = neighborhood_preservation(square, edges)
    repeat = neighborhood_preservation(square, edges)
    scrambled = neighborhood_preservation(square[[0, 2, 1, 3]], edges)
    assert perfect == repeat
    assert perfect["neighborhood_preservation_score"] == pytest.approx(1.0)
    assert scrambled["neighborhood_preservation_score"] < perfect["neighborhood_preservation_score"]


def test_gabriel_score_and_gate_e_tripwire() -> None:
    """Gabriel intrusions reduce the score and the diagnostic trips above 0.9."""
    edge = torch.tensor([[0], [1]], dtype=torch.long)
    good = gabriel_score(torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]]), edge)
    bad = gabriel_score(torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]]), edge)
    assert 0.0 <= bad["gabriel_score"] < good["gabriel_score"] <= 1.0
    diagnostic = gabriel_ksm_correlation(
        [{"gabriel_score": value, "ksm_score": value} for value in (0.1, 0.4, 0.9)]
    )
    assert diagnostic["gate_e_tripwire"] == 1.0


def test_gabriel_score_vectorization_matches_scalar_reference() -> None:
    """Preserve exact Gabriel counts with edge and node sampling."""
    generator = torch.Generator().manual_seed(83)
    positions = torch.randn((41, 2), generator=generator)
    edges = torch.randint(0, 41, (2, 73), generator=generator)
    n_samples = 37
    node_samples = 23
    measured = gabriel_score(
        positions,
        edges,
        n_samples=n_samples,
        node_samples=node_samples,
    )

    positions64 = positions.to(dtype=torch.float64)
    sampled_edges = edges[:, metrics_module._deterministic_sample_indices(73, n_samples)]
    node_ids = torch.arange(41)[metrics_module._deterministic_sample_indices(41, node_samples)]
    sampled_intrusions = 0
    for source, target in sampled_edges.t():
        source_index, target_index = int(source), int(target)
        midpoint = (positions64[source_index] + positions64[target_index]) / 2.0
        radius_sq = float(
            ((positions64[source_index] - positions64[target_index]) ** 2).sum() / 4.0
        )
        nonincident = (node_ids != source_index) & (node_ids != target_index)
        distances_sq = ((positions64[node_ids[nonincident]] - midpoint) ** 2).sum(dim=1)
        observed = int((distances_sq < radius_sq - 1e-12).sum().item())
        sampled_intrusions += round(observed * 39 / max(1, int(nonincident.sum())))
    expected_intrusions = round(sampled_intrusions * 73 / n_samples)

    assert measured["gabriel_intrusions"] == expected_intrusions
    assert measured["gabriel_score"] == 1.0 - math.sqrt(min(1.0, expected_intrusions / (73 * 39)))


def test_crossing_angle_uses_seventy_degree_knee() -> None:
    """A right-angle crossing beats a shallow crossing and both remain bounded."""
    edges = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    right = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    shallow = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [-1.0, -0.1], [1.0, 0.1]])
    right_score = crossing_angle_score(right, edges)["crossing_angle_score"]
    shallow_score = crossing_angle_score(shallow, edges)["crossing_angle_score"]
    assert right_score == pytest.approx(1.0)
    assert 0.0 <= shallow_score < right_score <= 1.0


def test_path_continuity_direction_and_na() -> None:
    """Straight paths beat corners and graphs without degree-two nodes are NA."""
    chain = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    straight = path_continuity_score(torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), chain)[
        "path_continuity_score"
    ]
    corner = path_continuity_score(torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), chain)[
        "path_continuity_score"
    ]
    star = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    assert straight == pytest.approx(1.0)
    assert 0.0 <= corner < straight
    assert path_continuity_score(torch.rand(4, 2), star)["path_continuity_score"] is None


def test_edge_length_deviation_reciprocal_and_declared_targets() -> None:
    """Equal or target-proportional edges beat uneven edge lengths."""
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    equal = edge_length_deviation_score(torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), edges)
    uneven = edge_length_deviation_score(torch.tensor([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]), edges)
    targeted = edge_length_deviation_score(
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]),
        edges,
        torch.tensor([1.0, 3.0]),
    )
    assert equal["edge_length_deviation_score"] == pytest.approx(1.0)
    assert targeted["edge_length_deviation_score"] == pytest.approx(1.0)
    assert 0.0 <= uneven["edge_length_deviation_score"] < 1.0


def test_angular_resolution_exact_normalization() -> None:
    """Evenly spaced ports beat a cramped fan under degree-relative AR."""
    edges = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]], dtype=torch.long)
    good = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    bad = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.1], [-1.0, 0.0], [0.0, -1.0]])
    good_score = angular_resolution_score(good, edges)["angular_resolution_score"]
    bad_score = angular_resolution_score(bad, edges)["angular_resolution_score"]
    assert good_score == pytest.approx(1.0)
    assert 0.0 <= bad_score < good_score


def test_cluster_silhouette_ground_truth_gate_and_renormalization() -> None:
    """Declared separated labels score higher and absent labels add no free credit."""
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor([[0.0, 0.0], [0.1, 0.0], [3.0, 0.0], [3.1, 0.0]])
    interleaved = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.1, 0.0], [3.1, 0.0]])
    good = cluster_silhouette_score(separated, labels)["cluster_silhouette_score"]
    bad = cluster_silhouette_score(interleaved, labels)["cluster_silhouette_score"]
    assert 0.0 <= bad < good <= 1.0
    profile = _common_profile(1.0)
    profile["cluster_silhouette_score"] = None
    assert composite_undirected(profile) == pytest.approx(100.0)


def test_degenerate_layout_scores_below_random() -> None:
    """The composite guard prevents a collapsed layout from beating random geometry."""
    edges = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 5, 5]], dtype=torch.long)
    sizes = torch.ones((6, 2))
    collapsed = full(torch.zeros((6, 2)), edges, node_sizes=sizes)
    random = full(
        torch.tensor([[0.0, 0.0], [2.0, 0.4], [4.0, -0.3], [6.0, 0.5], [8.0, -0.2], [10.0, 0.1]]),
        edges,
        node_sizes=sizes,
    )
    assert composite_undirected(collapsed) < composite_undirected(random)


def test_composite_routing_strictness_and_large_identity() -> None:
    """Only declared hierarchical digraphs route directed and large wrappers agree."""
    common = _common_profile(0.5)
    hierarchical = {
        **common,
        "declared_hierarchical": True,
        "directed_flow_score": 1.0,
        "depth_order_score": 1.0,
    }
    assert composite_auto(hierarchical, True) == pytest.approx(composite(hierarchical))
    assert composite_auto(hierarchical, False) == pytest.approx(composite_undirected(hierarchical))
    assert composite_auto(common, True) == pytest.approx(composite_undirected(common))
    assert composite_strict(hierarchical) == pytest.approx(composite(hierarchical))
    assert composite_large(hierarchical) == pytest.approx(composite(hierarchical))
    assert composite_large_undirected(common) == pytest.approx(composite_undirected(common))
    assert composite_large_auto(hierarchical, True) == pytest.approx(composite(hierarchical))
    with pytest.raises(ValueError, match="missing required fields"):
        composite_strict({"ksm_score": 1.0})


@pytest.mark.parametrize(
    ("direction", "positions"),
    [
        ("TB", [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]),
        ("BT", [[0.0, 0.0], [0.0, -1.0], [0.0, -2.0]]),
        ("LR", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        ("RL", [[0.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]]),
    ],
)
def test_depth_order_uses_declared_flow_axis(
    direction: str,
    positions: list[list[float]],
) -> None:
    """Perfect and reversed chains score one and zero on every flow axis."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    topo_depth = torch.tensor([0, 1, 2], dtype=torch.long)
    pos = torch.tensor(positions)

    assert quick(pos, edge_index, topo_depth=topo_depth, direction=direction)[
        "depth_order_score"
    ] == pytest.approx(1.0)
    assert quick(pos.flip(0), edge_index, topo_depth=topo_depth, direction=direction)[
        "depth_order_score"
    ] == pytest.approx(0.0)


def test_depth_order_flat_projection_is_inapplicable() -> None:
    """A flat hierarchical layout receives no depth-order credit."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    metrics = quick(
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        edge_index,
        topo_depth=torch.tensor([0, 1, 2]),
        direction="TB",
        declared_hierarchical=True,
    )

    assert metrics["depth_spearman_rho"] is None
    assert metrics["depth_order_score"] is None


def test_large_quick_profile_keeps_sampled_ruler_backbone() -> None:
    """The N>2000 preset includes topology stress and crossings beyond NO+ELD."""
    node_count = 2_001
    edge_index = torch.stack((torch.arange(node_count - 1), torch.arange(1, node_count)))
    pos = torch.stack(
        (torch.arange(node_count, dtype=torch.float32), torch.zeros(node_count)), dim=1
    )

    metrics = quick(pos, edge_index, node_sizes=torch.full((node_count, 2), 0.1), seed=0)

    assert metrics["ksm_n_pairs"] > 0
    assert metrics["crossing_n_samples"] > 0
    assert metrics["neighborhood_n_rows"] >= 128
    assert metrics["gabriel_n_edges"] == 256
    assert {name for name in _COMMON_WEIGHTS if metrics.get(name) is not None} >= {
        "ksm_score",
        "edge_crossing_score",
        "node_occlusion_score",
        "neighborhood_preservation_score",
        "edge_length_deviation_score",
        "gabriel_score",
        "crossing_angle_score",
    }


def test_crossing_angle_large_edge_set_respects_pair_budget() -> None:
    """Crossing-angle scoring never allocates the full large edge-pair space."""
    edge_count = 6_000
    edge_index = torch.stack((torch.arange(edge_count), torch.arange(1, edge_count + 1)))
    pos = torch.stack(
        (
            torch.arange(edge_count + 1, dtype=torch.float32),
            torch.arange(edge_count + 1, dtype=torch.float32).remainder(17),
        ),
        dim=1,
    )

    metrics = crossing_angle_score(pos, edge_index, n_samples=25_000, seed=0)

    assert metrics["crossing_angle_n_pairs"] <= 25_000


def test_full_reuses_one_batched_crossing_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two frozen crossing terms should intersect one shared pair set.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to count calls to the batched intersection predicate.

    Returns
    -------
    None
        This test asserts one predicate evaluation and standalone score parity.
    """
    edges = torch.tensor([[0, 2, 0, 1], [1, 3, 2, 3]], dtype=torch.long)
    positions = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    expected_crossing = edge_crossing_score(positions, edges, seed=0)
    expected_angle = crossing_angle_score(positions, edges, seed=0)
    calls = 0
    original = metrics_module.segments_intersect

    def counting_intersections(
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor,
    ) -> torch.Tensor:
        """Count one batched predicate call while preserving its result.

        Parameters
        ----------
        p1, p2, p3, p4 : torch.Tensor
            Batched segment endpoints with shape ``[P, 2]``.

        Returns
        -------
        torch.Tensor
            Boolean crossing mask with shape ``[P]``.
        """
        nonlocal calls
        calls += 1
        return original(p1, p2, p3, p4)

    monkeypatch.setattr(metrics_module, "segments_intersect", counting_intersections)
    measured = full(positions, edges, crossing_samples=100)

    assert calls == 1
    assert measured["crossing_count"] == expected_crossing["crossing_count"]
    assert measured["edge_crossing_score"] == expected_crossing["edge_crossing_score"]
    assert measured["crossing_angle_count"] == expected_angle["crossing_angle_count"]
    assert measured["crossing_angle_score"] == expected_angle["crossing_angle_score"]

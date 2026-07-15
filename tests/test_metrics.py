"""Tests for aesthetic quality metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import dagua.metrics as metrics_module
from dagua import layout
from dagua.graph import DaguaGraph
from dagua.metrics import (
    _all_pairs_unweighted,
    _bfs_distances,
    _build_csr,
    composite,
    compute_all_metrics,
    compute_dag_fraction,
    compute_edge_straightness,
    compute_x_alignment,
    count_crossings,
    count_overlaps,
    evaluate,
    full,
    layout_similarity,
    neighborhood_preservation,
    sampled_crossing_rate,
    sampled_stress,
)


def _make_small_graph() -> DaguaGraph:
    """Create a small connected graph for metric tests.

    Returns
    -------
    DaguaGraph
        Graph with four nodes and precomputed node sizes.
    """
    graph = DaguaGraph.from_edge_list(
        [
            ("a", "b"),
            ("b", "c"),
            ("a", "d"),
            ("d", "c"),
        ]
    )
    graph.compute_node_sizes()
    if graph.node_sizes is None:
        raise AssertionError("node sizes should be available after compute_node_sizes()")
    return graph


class TestCountCrossings:
    def test_no_crossings(self) -> None:
        """Parallel disjoint edges should not count as crossings."""
        pos = torch.tensor([[0.0, 0.0], [0.0, 100.0], [50.0, 0.0], [50.0, 100.0]])
        ei = torch.tensor([[0, 2], [1, 3]])  # parallel edges
        assert count_crossings(pos, ei) == 0

    def test_one_crossing(self) -> None:
        """Strict interior intersections should count as one crossing."""
        pos = torch.tensor([[0.0, 0.0], [100.0, 100.0], [100.0, 0.0], [0.0, 100.0]])
        ei = torch.tensor([[0, 2], [1, 3]])  # X pattern
        assert count_crossings(pos, ei) == 1

    def test_empty_edges(self) -> None:
        """Empty edge tensors should have no crossings."""
        pos = torch.randn(5, 2)
        ei = torch.zeros(2, 0, dtype=torch.long)
        assert count_crossings(pos, ei) == 0

    def test_collinear_overlap_counts_as_crossing(self) -> None:
        """Exact crossings should match vector semantics for collinear overlap."""
        pos = torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        ei = torch.tensor([[0, 2], [1, 3]])

        assert count_crossings(pos, ei) == 1

    def test_endpoint_touch_does_not_count_as_crossing(self) -> None:
        """Endpoint-only contact should be excluded even without shared node ids."""
        pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        ei = torch.tensor([[0, 2], [1, 3]])

        assert count_crossings(pos, ei) == 0

    def test_near_parallel_non_overlap_does_not_count_as_crossing(self) -> None:
        """Nearly parallel separated segments should remain non-crossing."""
        pos = torch.tensor([[0.0, 0.0], [1.0, 1.0e-9], [0.0, 1.0], [1.0, 1.0 + 1.0e-9]])
        ei = torch.tensor([[0, 2], [1, 3]])

        assert count_crossings(pos, ei) == 0

    def test_exact_and_sampled_predicates_agree_across_edge_cutoff(self) -> None:
        """The 500/501-edge paths should agree on degenerate crossing semantics."""
        pos = torch.zeros((504, 2), dtype=torch.float64)
        pos[0] = torch.tensor([0.0, 0.0])
        pos[1] = torch.tensor([2.0, 0.0])
        pos[2] = torch.tensor([1.0, 0.0])
        pos[3] = torch.tensor([3.0, 0.0])
        edges = [(0, 1), (2, 3)]
        for node in range(4, 502):
            pos[node] = torch.tensor([1000.0 + float(node), 1000.0])
            edges.append((0, node))

        exact_edge_index = torch.as_tensor(edges[:500], dtype=torch.long).T
        sampled_edges = edges + [(0, 502)]
        pos[502] = torch.tensor([2000.0, 1000.0])
        sampled_edge_index = torch.as_tensor(sampled_edges, dtype=torch.long).T
        total_pairs = sampled_edge_index.shape[1] * (sampled_edge_index.shape[1] - 1) // 2
        sampled = sampled_crossing_rate(pos, sampled_edge_index, n_samples=total_pairs, seed=42)

        assert count_crossings(pos, exact_edge_index) == 1
        assert sampled["crossing_estimated_total"] == 1

    def test_sampled_crossing_total_scales_by_eligible_pairs(self) -> None:
        """Sampled totals should not scale valid-pair rates by adjacent pairs."""
        pos = torch.zeros((604, 2), dtype=torch.float64)
        pos[0] = torch.tensor([0.0, 0.0])
        pos[1] = torch.tensor([2.0, 0.0])
        pos[2] = torch.tensor([1.0, 0.0])
        pos[3] = torch.tensor([3.0, 0.0])
        edges = [(0, 1), (2, 3)]
        for node in range(4, 604):
            pos[node] = torch.tensor([1000.0 + float(node), 1000.0])
            edges.append((0, node))
        edge_index = torch.as_tensor(edges, dtype=torch.long).T
        total_pairs = edge_index.shape[1] * (edge_index.shape[1] - 1) // 2

        sampled = sampled_crossing_rate(pos, edge_index, n_samples=total_pairs, seed=42)

        assert sampled["crossing_eligible_pairs"] < total_pairs
        assert sampled["crossing_estimated_total"] == 1


class TestDagFraction:
    def test_perfect_dag(self):
        pos = torch.tensor([[0.0, 0.0], [0.0, 50.0], [0.0, 100.0]])
        ei = torch.tensor([[0, 1], [1, 2]])
        assert compute_dag_fraction(pos, ei) == 1.0

    def test_reversed_dag(self):
        pos = torch.tensor([[0.0, 100.0], [0.0, 50.0], [0.0, 0.0]])
        ei = torch.tensor([[0, 1], [1, 2]])
        assert compute_dag_fraction(pos, ei) == 0.0

    def test_empty(self):
        pos = torch.randn(3, 2)
        ei = torch.zeros(2, 0, dtype=torch.long)
        assert compute_dag_fraction(pos, ei) == 1.0


class TestCountOverlaps:
    def test_overlapping(self):
        pos = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0]])
        assert count_overlaps(pos, ns) == 1

    def test_non_overlapping(self):
        pos = torch.tensor([[0.0, 0.0], [200.0, 0.0]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0]])
        assert count_overlaps(pos, ns) == 0


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        pos = torch.tensor([[0.0, 0.0], [0.0, 50.0]])
        ei = torch.tensor([[0], [1]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0]])
        m = compute_all_metrics(pos, ei, ns)
        assert "num_nodes" in m
        assert "edge_crossings" in m
        assert "dag_fraction" in m
        assert "node_overlaps" in m
        assert "overall_quality" in m

    def test_overall_quality_higher_is_better(self):
        # Good layout
        pos_good = torch.tensor([[0.0, 0.0], [0.0, 50.0], [0.0, 100.0]])
        # Bad layout (reversed, overlapping)
        pos_bad = torch.tensor([[0.0, 100.0], [5.0, 50.0], [0.0, 0.0]])
        ei = torch.tensor([[0, 1], [1, 2]])
        ns = torch.tensor([[40.0, 20.0]] * 3)
        q_good = compute_all_metrics(pos_good, ei, ns)["overall_quality"]
        q_bad = compute_all_metrics(pos_bad, ei, ns)["overall_quality"]
        assert q_good > q_bad


def test_composite_includes_isotonic_stress_weight() -> None:
    """Directed composite should apply the frozen common KSM weight.

    Returns
    -------
    None
        This test asserts the weighted composite formula with non-trivial
        normalized component values.
    """
    metric_values = {name: 0.5 for name in metrics_module._COMMON_WEIGHTS}
    metric_values.update(
        {
            "ksm_score": 0.8,
            "directed_flow_score": 0.5,
            "depth_order_score": 0.5,
            "declared_hierarchical": True,
        }
    )
    expected = 0.75 * (25 * 0.8 + 75 * 0.5) + 16 * 0.5 + 9 * 0.5
    assert composite(metric_values) == pytest.approx(expected)


class TestDirectionAwareMetrics:
    """Test that metrics respect layout direction."""

    def test_dag_fraction_bt(self):
        """BT: edges should go upward (target.y < source.y)."""
        pos = torch.tensor([[0.0, 100.0], [0.0, 50.0], [0.0, 0.0]])
        ei = torch.tensor([[0, 1], [1, 2]])
        assert compute_dag_fraction(pos, ei, direction="BT") == 1.0
        assert compute_dag_fraction(pos, ei, direction="TB") == 0.0

    def test_dag_fraction_lr(self):
        """LR: edges should go rightward (target.x > source.x)."""
        pos = torch.tensor([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
        ei = torch.tensor([[0, 1], [1, 2]])
        assert compute_dag_fraction(pos, ei, direction="LR") == 1.0

    def test_dag_fraction_rl(self):
        """RL: edges should go leftward (target.x < source.x)."""
        pos = torch.tensor([[100.0, 0.0], [50.0, 0.0], [0.0, 0.0]])
        ei = torch.tensor([[0, 1], [1, 2]])
        assert compute_dag_fraction(pos, ei, direction="RL") == 1.0

    def test_edge_straightness_lr(self):
        """LR: deviation from horizontal should be small for horizontal edges."""
        pos = torch.tensor([[0.0, 0.0], [100.0, 0.0]])  # perfectly horizontal
        ei = torch.tensor([[0], [1]])
        angle = compute_edge_straightness(pos, ei, direction="LR")
        assert angle < 1.0  # nearly 0 degrees deviation from horizontal

    def test_edge_straightness_tb(self):
        """TB: deviation from vertical should be small for vertical edges."""
        pos = torch.tensor([[0.0, 0.0], [0.0, 100.0]])  # perfectly vertical
        ei = torch.tensor([[0], [1]])
        angle = compute_edge_straightness(pos, ei, direction="TB")
        assert angle < 1.0

    def test_x_alignment_lr(self):
        """LR: cross-axis displacement is along y."""
        pos = torch.tensor([[0.0, 0.0], [100.0, 10.0]])
        ei = torch.tensor([[0], [1]])
        # LR: cross-axis is y, so alignment = abs(0 - 10) = 10
        assert compute_x_alignment(pos, ei, direction="LR") == pytest.approx(10.0)
        # TB: cross-axis is x, so alignment = abs(0 - 100) = 100
        assert compute_x_alignment(pos, ei, direction="TB") == pytest.approx(100.0)

    def test_compute_all_metrics_with_direction(self):
        """compute_all_metrics passes direction through."""
        pos = torch.tensor([[0.0, 100.0], [0.0, 50.0], [0.0, 0.0]])
        ei = torch.tensor([[0, 1], [1, 2]])
        ns = torch.tensor([[40.0, 20.0]] * 3)
        m_bt = compute_all_metrics(pos, ei, ns, direction="BT")
        m_tb = compute_all_metrics(pos, ei, ns, direction="TB")
        assert m_bt["dag_fraction"] == 1.0
        assert m_tb["dag_fraction"] == 0.0


class TestLayoutSimilarityAndEvaluate:
    def test_layout_similarity_identical(self) -> None:
        """Identical layouts should have near-perfect similarity.

        Returns
        -------
        None
            This test asserts on the returned similarity metrics.
        """
        torch.manual_seed(0)
        pos = torch.randn(20, 2)
        sim = layout_similarity(pos, pos)
        assert sim["procrustes_similarity"] > 0.99

    def test_layout_similarity_different(self) -> None:
        """Independent random layouts should have low Procrustes similarity.

        Returns
        -------
        None
            This test asserts on the returned similarity metrics.
        """
        torch.manual_seed(0)
        pos_a = torch.randn(20, 2)
        torch.manual_seed(1)
        pos_b = torch.randn(20, 2)
        sim = layout_similarity(pos_a, pos_b)
        assert sim["procrustes_similarity"] < 0.5

    def test_evaluate_convenience(self) -> None:
        """The graph-level evaluate helper should return standard metric keys.

        Returns
        -------
        None
            This test asserts on the convenience wrapper output.
        """
        graph = _make_small_graph()
        pos = layout(graph)
        metrics = evaluate(graph, pos)
        assert "dag_consistency" in metrics
        assert "composite_score" in metrics

    def test_stress_is_scale_invariant(self) -> None:
        """Sampled stress should be unchanged by uniform scaling.

        Returns
        -------
        None
            This test asserts on normalized sampled stress values.
        """
        torch.manual_seed(0)
        pos = torch.randn(50, 2)
        edge_index = torch.randint(0, 50, (2, 100))
        stress_a = sampled_stress(pos, edge_index, 50)["sampled_stress"]
        stress_b = sampled_stress(pos * 100.0, edge_index, 50)["sampled_stress"]
        assert abs(stress_a - stress_b) < 0.01 * max(stress_a, stress_b, 1e-6)

    def test_all_pairs_distances_match_truncated_bfs(self) -> None:
        """Vectorized graph distances should match the legacy BFS contract.

        Returns
        -------
        None
            This test asserts all-pairs distances, disconnected pairs, and
            truncation semantics.
        """
        edge_index = torch.tensor([[0, 1, 2, 4], [1, 2, 3, 5]], dtype=torch.long)
        csr_offsets, csr_targets = _build_csr(edge_index, 7)
        all_pairs = _all_pairs_unweighted(csr_offsets, csr_targets, 7, max_dist=2)

        for source in range(7):
            bfs_dist = _bfs_distances(csr_offsets, csr_targets, source, max_dist=2)
            assert all_pairs[source].tolist() == bfs_dist.tolist()

    def test_sampled_metrics_accept_precomputed_all_pairs(self) -> None:
        """Affected sampled metrics should preserve values with cached distances.

        Returns
        -------
        None
            This test compares direct and precomputed-distance metric values.
        """
        torch.manual_seed(0)
        pos = torch.randn(8, 2)
        edge_index = torch.tensor(
            [[0, 1, 2, 2, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]],
            dtype=torch.long,
        )
        csr_offsets, csr_targets = _build_csr(edge_index, 8)
        all_pairs = _all_pairs_unweighted(csr_offsets, csr_targets, 8, max_dist=20)

        stress_direct = sampled_stress(pos, edge_index, 8)
        stress_cached = sampled_stress(pos, edge_index, 8, all_pairs_dist=all_pairs)
        assert stress_cached == pytest.approx(stress_direct)

        torch.manual_seed(123)
        neighborhood_direct = neighborhood_preservation(pos, edge_index, 8, n_samples=8, k=3)
        torch.manual_seed(123)
        neighborhood_cached = neighborhood_preservation(
            pos,
            edge_index,
            8,
            n_samples=8,
            k=3,
            all_pairs_dist=all_pairs,
        )
        assert neighborhood_cached == pytest.approx(neighborhood_direct)

    def test_full_reuses_one_all_pairs_distance_matrix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``full`` should compute graph distances once for shared sampled metrics.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to count helper calls.

        Returns
        -------
        None
            This test asserts that one all-pairs matrix is shared by stress and
            neighborhood preservation.
        """
        calls = 0
        original = metrics_module._all_pairs_unweighted

        def counting_all_pairs(
            csr_offsets: np.ndarray,
            csr_targets: np.ndarray,
            num_nodes: int,
            max_dist: int = 20,
        ) -> np.ndarray:
            """Count all-pairs calls while delegating to the real helper.

            Parameters
            ----------
            csr_offsets : numpy.ndarray
                CSR row offsets.
            csr_targets : numpy.ndarray
                CSR target indices.
            num_nodes : int
                Number of graph nodes.
            max_dist : int, optional
                Maximum retained graph distance.

            Returns
            -------
            numpy.ndarray
                Delegated all-pairs distance matrix.
            """
            nonlocal calls
            calls += 1
            return original(csr_offsets, csr_targets, num_nodes, max_dist=max_dist)

        monkeypatch.setattr(metrics_module, "_all_pairs_unweighted", counting_all_pairs)
        pos = torch.randn(12, 2)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]], dtype=torch.long)

        full(
            pos,
            edge_index,
            stress_sources=4,
            stress_targets=4,
            crossing_samples=10,
            neighborhood_samples=4,
        )

        assert calls == 0

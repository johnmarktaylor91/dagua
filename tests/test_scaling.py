"""Runtime scaling tests: verify Dagua handles large graphs efficiently."""

import time

import pytest
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph


def _make_chain(n: int) -> DaguaGraph:
    """Simple chain graph: 0→1→2→...→n-1."""
    if n == 0:
        return DaguaGraph()
    ei = torch.stack([torch.arange(n - 1), torch.arange(1, n)]) if n > 1 else torch.zeros(2, 0, dtype=torch.long)
    return DaguaGraph.from_edge_index(ei, num_nodes=n)


def _make_random_dag(n: int, edge_ratio: float = 1.5, seed: int = 42) -> DaguaGraph:
    """Random DAG: n nodes, ~n*edge_ratio edges."""
    import random

    rng = random.Random(seed)
    n_edges = int(n * edge_ratio)

    edges = set()
    attempts = 0
    while len(edges) < n_edges and attempts < n_edges * 20:
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, min(i + max(n // 5, 10), n - 1))
        edges.add((i, j))
        attempts += 1

    if edges:
        edge_list = list(edges)
        src = [e[0] for e in edge_list]
        tgt = [e[1] for e in edge_list]
        ei = torch.tensor([src, tgt], dtype=torch.long)
    else:
        ei = torch.zeros(2, 0, dtype=torch.long)

    return DaguaGraph.from_edge_index(ei, num_nodes=n)


class TestDaguaScaling:
    """Verify Dagua handles graphs of increasing size within time bounds."""

    def test_100_nodes(self):
        g = _make_random_dag(100)
        g.compute_node_sizes()
        config = LayoutConfig(steps=50)
        start = time.perf_counter()
        pos = dagua.layout(g, config)
        elapsed = time.perf_counter() - start
        assert pos.shape == (100, 2)
        assert elapsed < 30, f"100 nodes took {elapsed:.1f}s (limit: 30s)"

    def test_500_nodes(self):
        g = _make_random_dag(500)
        g.compute_node_sizes()
        config = LayoutConfig(steps=50)
        start = time.perf_counter()
        pos = dagua.layout(g, config)
        elapsed = time.perf_counter() - start
        assert pos.shape == (500, 2)
        assert elapsed < 60, f"500 nodes took {elapsed:.1f}s (limit: 60s)"

    def test_1000_nodes(self):
        g = _make_random_dag(1000)
        g.compute_node_sizes()
        config = LayoutConfig(steps=50)
        start = time.perf_counter()
        pos = dagua.layout(g, config)
        elapsed = time.perf_counter() - start
        assert pos.shape == (1000, 2)
        assert elapsed < 120, f"1000 nodes took {elapsed:.1f}s (limit: 120s)"

    @pytest.mark.slow
    def test_5000_nodes(self):
        g = _make_random_dag(5000)
        g.compute_node_sizes()
        config = LayoutConfig(steps=50)
        start = time.perf_counter()
        pos = dagua.layout(g, config)
        elapsed = time.perf_counter() - start
        assert pos.shape == (5000, 2)
        assert elapsed < 300, f"5000 nodes took {elapsed:.1f}s (limit: 300s)"

    @pytest.mark.slow
    def test_10000_nodes(self):
        g = _make_random_dag(10000)
        g.compute_node_sizes()
        config = LayoutConfig(steps=50)
        start = time.perf_counter()
        pos = dagua.layout(g, config)
        elapsed = time.perf_counter() - start
        assert pos.shape == (10000, 2)
        assert elapsed < 600, f"10000 nodes took {elapsed:.1f}s (limit: 600s)"


class TestGraphvizComparison:
    """Compare runtime between Dagua and Graphviz at various scales."""

    @pytest.fixture
    def graphviz_available(self):
        import shutil
        if not shutil.which("dot"):
            pytest.skip("Graphviz not installed")

    def _time_graphviz(self, graph: DaguaGraph) -> float:
        import subprocess
        from dagua.graphviz_utils import to_dot

        dot_str = to_dot(graph)
        start = time.perf_counter()
        result = subprocess.run(
            ["dot", "-Tjson"],
            input=dot_str, capture_output=True, text=True, timeout=120,
        )
        elapsed = time.perf_counter() - start
        assert result.returncode == 0, f"Graphviz failed: {result.stderr[:200]}"
        return elapsed

    def test_scaling_comparison_100(self, graphviz_available):
        g = _make_random_dag(100)
        g.compute_node_sizes()
        config = LayoutConfig(steps=50)
        dagua_time = time.perf_counter()
        dagua.layout(g, config)
        dagua_time = time.perf_counter() - dagua_time
        gv_time = self._time_graphviz(g)
        print(f"\n  N=100: dagua={dagua_time:.2f}s, graphviz={gv_time:.2f}s, ratio={dagua_time/gv_time:.1f}x")

    def test_scaling_comparison_500(self, graphviz_available):
        g = _make_random_dag(500)
        g.compute_node_sizes()
        config = LayoutConfig(steps=50)
        dagua_time = time.perf_counter()
        dagua.layout(g, config)
        dagua_time = time.perf_counter() - dagua_time
        gv_time = self._time_graphviz(g)
        print(f"\n  N=500: dagua={dagua_time:.2f}s, graphviz={gv_time:.2f}s, ratio={dagua_time/gv_time:.1f}x")

    def test_scaling_comparison_1000(self, graphviz_available):
        g = _make_random_dag(1000)
        g.compute_node_sizes()
        config = LayoutConfig(steps=50)
        dagua_time = time.perf_counter()
        dagua.layout(g, config)
        dagua_time = time.perf_counter() - dagua_time
        gv_time = self._time_graphviz(g)
        print(f"\n  N=1000: dagua={dagua_time:.2f}s, graphviz={gv_time:.2f}s, ratio={dagua_time/gv_time:.1f}x")

    @pytest.mark.slow
    def test_scaling_comparison_5000(self, graphviz_available):
        """At 5K nodes, Dagua should be competitive or faster than Graphviz."""
        g = _make_random_dag(5000)
        g.compute_node_sizes()
        config = LayoutConfig(steps=50)
        dagua_time = time.perf_counter()
        dagua.layout(g, config)
        dagua_time = time.perf_counter() - dagua_time
        gv_time = self._time_graphviz(g)
        print(f"\n  N=5000: dagua={dagua_time:.2f}s, graphviz={gv_time:.2f}s, ratio={dagua_time/gv_time:.1f}x")


def _make_random_dag_vectorized(n: int, edge_ratio: float = 1.3, seed: int = 42) -> DaguaGraph:
    """Vectorized random DAG for millions of nodes. Forward-only edges."""
    torch.manual_seed(seed)
    n_edges = int(n * edge_ratio)

    src = torch.randint(0, n - 1, (n_edges,))
    # Target must be > source for DAG property; offset by 1..max_skip
    max_skip = max(n // 5, 10)
    skip = torch.randint(1, max_skip + 1, (n_edges,))
    tgt = (src + skip).clamp(max=n - 1)

    # Remove self-loops and deduplicate
    valid = src != tgt
    src, tgt = src[valid], tgt[valid]
    edge_hash = src.long() * n + tgt.long()
    unique_hash = edge_hash.unique()
    src = (unique_hash // n).to(torch.long)
    tgt = (unique_hash % n).to(torch.long)
    ei = torch.stack([src, tgt])

    g = DaguaGraph.from_edge_index(ei, num_nodes=n)
    # Use uniform node sizes to skip expensive text measurement
    g.node_sizes = torch.tensor([[40.0, 20.0]]).expand(n, -1).clone()
    return g


class TestExtremeScale:
    """5M and 10M node tests. Marked 'rare' — NEVER run routinely.

    Run explicitly with: pytest tests/test_scaling.py -m rare -s
    """

    @pytest.mark.rare
    def test_5m_nodes(self):
        """5,000,000 node layout via multilevel V-cycle."""
        n = 5_000_000
        g = _make_random_dag_vectorized(n)
        config = LayoutConfig(
            steps=500,
            device="cpu",
            multilevel_threshold=50000,
            multilevel_min_nodes=2000,
            multilevel_coarse_steps=100,
            multilevel_refine_steps=25,
        )
        start = time.perf_counter()
        pos = dagua.layout(g, config)
        elapsed = time.perf_counter() - start
        assert pos.shape == (n, 2), f"Expected ({n}, 2), got {pos.shape}"
        assert not torch.isnan(pos).any(), "NaN in positions"
        assert not torch.isinf(pos).any(), "Inf in positions"
        print(f"\n  5M nodes: {elapsed:.1f}s")

    @pytest.mark.rare
    def test_10m_nodes(self):
        """10,000,000 node layout via multilevel V-cycle."""
        n = 10_000_000
        g = _make_random_dag_vectorized(n)
        config = LayoutConfig(
            steps=500,
            device="cpu",
            multilevel_threshold=50000,
            multilevel_min_nodes=2000,
            multilevel_coarse_steps=100,
            multilevel_refine_steps=25,
        )
        start = time.perf_counter()
        pos = dagua.layout(g, config)
        elapsed = time.perf_counter() - start
        assert pos.shape == (n, 2), f"Expected ({n}, 2), got {pos.shape}"
        assert not torch.isnan(pos).any(), "NaN in positions"
        assert not torch.isinf(pos).any(), "Inf in positions"
        print(f"\n  10M nodes: {elapsed:.1f}s")

    @pytest.mark.rare
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_5m_nodes_gpu(self):
        """5M nodes on GPU — test VRAM limits."""
        n = 5_000_000
        g = _make_random_dag_vectorized(n)
        config = LayoutConfig(
            steps=500,
            device="cuda",
            multilevel_threshold=50000,
            multilevel_min_nodes=2000,
            multilevel_coarse_steps=100,
            multilevel_refine_steps=25,
        )
        start = time.perf_counter()
        pos = dagua.layout(g, config)
        elapsed = time.perf_counter() - start
        assert pos.shape == (n, 2)
        print(f"\n  5M nodes (GPU): {elapsed:.1f}s")


    @pytest.mark.rare
    def test_20m_nodes(self):
        """20,000,000 node layout via multilevel V-cycle (CPU only — GPU OOM at 11GB)."""
        n = 20_000_000
        g = _make_random_dag_vectorized(n)
        config = LayoutConfig(
            steps=500,
            device="cpu",
            multilevel_threshold=50000,
            multilevel_min_nodes=2000,
            multilevel_coarse_steps=100,
            multilevel_refine_steps=25,
        )
        start = time.perf_counter()
        pos = dagua.layout(g, config)
        elapsed = time.perf_counter() - start
        assert pos.shape == (n, 2), f"Expected ({n}, 2), got {pos.shape}"
        assert not torch.isnan(pos).any(), "NaN in positions"
        assert not torch.isinf(pos).any(), "Inf in positions"
        print(f"\n  20M nodes: {elapsed:.1f}s")


class TestEdgeCases:
    """Test graphs that could cause problems at scale."""

    def test_empty_graph(self):
        g = DaguaGraph()
        pos = dagua.layout(g)
        assert pos.shape == (0, 2)

    def test_single_node(self):
        g = DaguaGraph()
        g.add_node("a")
        pos = dagua.layout(g)
        assert pos.shape == (1, 2)

    def test_disconnected_components(self):
        """Multiple disconnected subgraphs."""
        ei = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        g = DaguaGraph.from_edge_index(ei, num_nodes=6)
        g.compute_node_sizes()
        pos = dagua.layout(g, LayoutConfig(steps=50))
        assert pos.shape == (6, 2)

    def test_self_loop_edge(self):
        """Graph with a self-loop should not crash."""
        g = DaguaGraph()
        g.add_node("a")
        g.add_node("b")
        # Self-loop: a→a
        g._pending_edges.append((0, 0))
        g.edge_labels.append(None)
        g.edge_types.append("normal")
        g.edge_styles.append(None)
        # Normal edge: a→b
        g.add_edge("a", "b")
        g.compute_node_sizes()
        pos = dagua.layout(g, LayoutConfig(steps=50))
        assert pos.shape == (2, 2)

    def test_wide_graph_500_parallel(self):
        """500 parallel nodes — all children of one root."""
        g = DaguaGraph()
        g.add_node("root")
        for i in range(500):
            g.add_edge("root", f"child_{i}")
        g.compute_node_sizes()
        config = LayoutConfig(steps=50)
        start = time.perf_counter()
        pos = dagua.layout(g, config)
        elapsed = time.perf_counter() - start
        assert pos.shape == (501, 2)
        assert elapsed < 60, f"Wide graph (501 nodes) took {elapsed:.1f}s"

    def test_dense_bipartite(self):
        """20→20 bipartite = 400 edges. Heavily connected."""
        edges = []
        for i in range(20):
            for j in range(20):
                edges.append((f"s{i}", f"t{j}"))
        g = DaguaGraph.from_edge_list(edges)
        g.compute_node_sizes()
        pos = dagua.layout(g, LayoutConfig(steps=50))
        assert pos.shape == (40, 2)

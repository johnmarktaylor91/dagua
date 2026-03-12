"""Smoke tests for recent core changes.

Covers: ProgressContext, verbose output, num_workers config, from_edge_list
with num_nodes, _longest_path_layering_vectorized on deep DAGs, and basic
multilevel layout with verbose=True.

All tests marked @pytest.mark.smoke — run with: pytest tests/test_smoke.py -m smoke
"""

import time

import pytest
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.engine import ProgressContext, layout
from dagua.layout import multilevel as _multilevel_mod
from dagua.layout.multilevel import (
    _can_prolong_on_gpu,
    coarsen_once,
    _coarsen_once_streaming,
    _STREAMING_THRESHOLD,
)
from dagua.utils import (
    _longest_path_layering_vectorized,
    _STREAMING_NODE_THRESHOLD,
    longest_path_layering,
)


@pytest.mark.smoke
class TestProgressContext:
    """ProgressContext import and basic usage."""

    def test_import(self):
        assert ProgressContext is not None

    def test_default_indent(self):
        ctx = ProgressContext()
        assert ctx.indent == "  "

    def test_custom_indent(self):
        ctx = ProgressContext(indent="    ")
        assert ctx.indent == "    "


@pytest.mark.smoke
class TestVerboseOutput:
    """Verbose output emits [dagua] prefix in both direct and multilevel paths."""

    def test_direct_layout_verbose(self, capsys):
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c"), ("c", "d")])
        config = LayoutConfig(steps=10, verbose=True)
        dagua.layout(g, config)
        captured = capsys.readouterr()
        assert "[dagua]" in captured.out

    def test_direct_layout_verbose_reports_node_count(self, capsys):
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
        config = LayoutConfig(steps=10, verbose=True)
        dagua.layout(g, config)
        captured = capsys.readouterr()
        assert "3" in captured.out  # 3 nodes

    def test_verbose_off_is_silent(self, capsys):
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
        config = LayoutConfig(steps=10, verbose=False)
        dagua.layout(g, config)
        captured = capsys.readouterr()
        assert captured.out == ""


@pytest.mark.smoke
class TestNumWorkersConfig:
    """num_workers config field exists and defaults correctly."""

    def test_default_value(self):
        config = LayoutConfig()
        assert config.num_workers == 0

    def test_custom_value(self):
        config = LayoutConfig(num_workers=4)
        assert config.num_workers == 4

    def test_field_is_int(self):
        config = LayoutConfig()
        assert isinstance(config.num_workers, int)


@pytest.mark.smoke
class TestFromEdgeListNumNodes:
    """from_edge_list with num_nodes pre-creates nodes correctly."""

    def test_num_nodes_creates_expected_count(self):
        edges = [(0, 1), (1, 2)]
        g = DaguaGraph.from_edge_list(edges, num_nodes=5)
        # Should have 5 nodes even though edges only reference 0, 1, 2
        assert g.num_nodes == 5

    def test_num_nodes_without_edges(self):
        g = DaguaGraph.from_edge_list([], num_nodes=3)
        assert g.num_nodes == 3

    def test_num_nodes_preserves_edges(self):
        edges = [(0, 1), (1, 2), (2, 3)]
        g = DaguaGraph.from_edge_list(edges, num_nodes=4)
        assert g.num_nodes == 4
        assert g.edge_index.shape == (2, 3)

    def test_without_num_nodes_only_referenced(self):
        edges = [("x", "y")]
        g = DaguaGraph.from_edge_list(edges)
        assert g.num_nodes == 2


@pytest.mark.smoke
class TestLongestPathLayeringVectorized:
    """_longest_path_layering_vectorized handles chain graphs efficiently."""

    def test_small_chain_correct(self):
        """Verify correctness on a small chain: 0->1->2->3->4."""
        n = 5
        src = list(range(n - 1))
        tgt = list(range(1, n))
        edge_index = torch.tensor([src, tgt], dtype=torch.long)
        layers = _longest_path_layering_vectorized(edge_index, n)
        assert layers.tolist() == [0, 1, 2, 3, 4]

    def test_agrees_with_scalar_version(self):
        """Vectorized result should match the scalar BFS version."""
        n = 100
        src = list(range(n - 1))
        tgt = list(range(1, n))
        edge_index = torch.tensor([src, tgt], dtype=torch.long)
        scalar_result = longest_path_layering(edge_index, n)
        vector_result = _longest_path_layering_vectorized(edge_index, n)
        scalar_list = scalar_result.tolist() if isinstance(scalar_result, torch.Tensor) else scalar_result
        vector_list = vector_result.tolist() if isinstance(vector_result, torch.Tensor) else vector_result
        assert scalar_list == vector_list

    @pytest.mark.slow
    def test_100k_chain_completes_in_time(self):
        """Deep chain of 100K nodes must complete in <5s."""
        n = 100_000
        src = torch.arange(n - 1, dtype=torch.long)
        tgt = torch.arange(1, n, dtype=torch.long)
        edge_index = torch.stack([src, tgt])

        t0 = time.perf_counter()
        layers = _longest_path_layering_vectorized(edge_index, n)
        elapsed = time.perf_counter() - t0

        assert elapsed < 5.0, f"Took {elapsed:.2f}s, expected <5s"
        assert layers.shape[0] == n
        assert layers[0].item() == 0
        assert layers[-1].item() == n - 1

    def test_diamond_dag(self):
        """Diamond: 0->1, 0->2, 1->3, 2->3 — node 3 should be at layer 2."""
        edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)
        layers = _longest_path_layering_vectorized(edge_index, 4)
        assert layers[0] == 0
        assert layers[1] == 1
        assert layers[2] == 1
        assert layers[3] == 2


@pytest.mark.smoke
class TestMultilevelVerbose:
    """Basic multilevel layout works with verbose=True."""

    def test_multilevel_verbose_output(self, capsys):
        """A graph above multilevel_threshold triggers multilevel path and verbose output."""
        # Use a small threshold to force multilevel without creating a huge graph
        n = 200
        edges = [(f"n{i}", f"n{i+1}") for i in range(n - 1)]
        g = DaguaGraph.from_edge_list(edges)
        config = LayoutConfig(
            steps=10,
            verbose=True,
            multilevel_threshold=100,  # force multilevel
            multilevel_min_nodes=50,
            multilevel_coarse_steps=10,
            multilevel_refine_steps=5,
        )
        pos = dagua.layout(g, config)
        captured = capsys.readouterr()

        assert pos.shape == (n, 2)
        # Multilevel verbose should mention hierarchy and phases
        assert "[dagua]" in captured.out
        assert "hierarchy" in captured.out.lower() or "Phase" in captured.out
        assert "layering" in captured.out.lower()
        assert "coarsen level" in captured.out.lower()

    def test_multilevel_produces_valid_positions(self):
        """Multilevel layout should produce finite, non-NaN positions."""
        n = 150
        edges = [(f"n{i}", f"n{i+1}") for i in range(n - 1)]
        g = DaguaGraph.from_edge_list(edges)
        config = LayoutConfig(
            steps=10,
            multilevel_threshold=100,
            multilevel_min_nodes=50,
            multilevel_coarse_steps=10,
            multilevel_refine_steps=5,
        )
        pos = dagua.layout(g, config)

        assert pos.shape == (n, 2)
        assert torch.isfinite(pos).all(), "Positions contain NaN or Inf"


def _make_layered_dag(n_per_layer: int, n_layers: int):
    """Create a layered DAG with edges from each layer to the next."""
    N = n_per_layer * n_layers
    src_list, tgt_list = [], []
    for layer in range(n_layers - 1):
        base_src = layer * n_per_layer
        base_tgt = (layer + 1) * n_per_layer
        for i in range(n_per_layer):
            # Connect each node to 1-3 nodes in next layer
            for offset in range(min(3, n_per_layer)):
                tgt_node = base_tgt + (i + offset) % n_per_layer
                src_list.append(base_src + i)
                tgt_list.append(tgt_node)
    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    node_sizes = torch.ones(N, 2) * 10.0
    return edge_index, N, node_sizes


def _make_structural_coarsening_case():
    """Small layered graph with two matched pairs and one hub in the same layer."""
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 0, 1, 2, 3, 4, 5, 8, 8, 2],
            [6, 7, 6, 7, 8, 8, 8, 8, 8, 8, 10, 11, 9],
        ],
        dtype=torch.long,
    )
    node_sizes = torch.ones(12, 2) * 10.0
    layers = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2], dtype=torch.long)
    return edge_index, 12, node_sizes, layers


@pytest.mark.smoke
class TestStreamingCoarsenMatchesVectorized:
    """Streaming coarsening produces structurally equivalent results.

    The vectorized and streaming paths may produce different groupings due to
    float32 precision loss in the composite sort key (vectorized path). Both are
    valid coarsenings — we check structural invariants rather than exact equality.
    """

    def test_both_paths_reduce_graph(self):
        """Both paths should materially reduce the graph size."""
        edge_index, N, node_sizes = _make_layered_dag(100, 5)
        layers = longest_path_layering(edge_index, N)
        if isinstance(layers, list):
            layers = torch.tensor(layers, dtype=torch.long)

        result_vec = coarsen_once(edge_index, N, node_sizes, layers)

        old_threshold = _multilevel_mod._STREAMING_THRESHOLD
        try:
            _multilevel_mod._STREAMING_THRESHOLD = 100
            result_stream = coarsen_once(edge_index, N, node_sizes, layers)
        finally:
            _multilevel_mod._STREAMING_THRESHOLD = old_threshold

        assert result_vec.num_nodes < N
        assert result_stream.num_nodes < N
        assert result_vec.num_nodes <= int(N * 0.8)
        assert result_stream.num_nodes <= int(N * 0.5)

    def test_layer_awareness_preserved(self):
        """Streaming path never merges nodes from different layers."""
        edge_index, N, node_sizes = _make_layered_dag(100, 5)
        layers = longest_path_layering(edge_index, N)
        if isinstance(layers, list):
            layers = torch.tensor(layers, dtype=torch.long)

        old_threshold = _multilevel_mod._STREAMING_THRESHOLD
        try:
            _multilevel_mod._STREAMING_THRESHOLD = 100
            result = coarsen_once(edge_index, N, node_sizes, layers)
        finally:
            _multilevel_mod._STREAMING_THRESHOLD = old_threshold

        # Every coarse group must contain nodes from exactly one layer
        ftc = result.fine_to_coarse.tolist()
        groups: dict = {}
        for fine, coarse in enumerate(ftc):
            groups.setdefault(coarse, set()).add(layers[fine].item())
        for coarse_id, layer_set in groups.items():
            assert len(layer_set) == 1, (
                f"Coarse node {coarse_id} merges nodes from layers {layer_set}"
            )

    def test_group_sizes_at_most_three(self):
        """Each coarse group merges at most 3 fine nodes (triple matching)."""
        edge_index, N, node_sizes = _make_layered_dag(100, 5)
        layers = longest_path_layering(edge_index, N)
        if isinstance(layers, list):
            layers = torch.tensor(layers, dtype=torch.long)

        old_threshold = _multilevel_mod._STREAMING_THRESHOLD
        try:
            _multilevel_mod._STREAMING_THRESHOLD = 100
            result = coarsen_once(edge_index, N, node_sizes, layers)
        finally:
            _multilevel_mod._STREAMING_THRESHOLD = old_threshold

        ftc = result.fine_to_coarse.tolist()
        from collections import Counter
        group_sizes = Counter(ftc)
        max_size = max(group_sizes.values())
        assert max_size <= 3, f"Largest coarse group has {max_size} nodes, expected <= 3"

    def test_no_self_loops_in_coarse_edges(self):
        """Coarse edge index should have no self-loops."""
        edge_index, N, node_sizes = _make_layered_dag(100, 5)
        layers = longest_path_layering(edge_index, N)
        if isinstance(layers, list):
            layers = torch.tensor(layers, dtype=torch.long)

        old_threshold = _multilevel_mod._STREAMING_THRESHOLD
        try:
            _multilevel_mod._STREAMING_THRESHOLD = 100
            result = coarsen_once(edge_index, N, node_sizes, layers)
        finally:
            _multilevel_mod._STREAMING_THRESHOLD = old_threshold

        if result.edge_index.numel() > 0:
            src, tgt = result.edge_index[0], result.edge_index[1]
            assert (src != tgt).all(), "Coarse edges contain self-loops"

    def test_reduction_ratio(self):
        """Streaming path achieves ~67% reduction (triple matching)."""
        edge_index, N, node_sizes = _make_layered_dag(100, 5)
        layers = longest_path_layering(edge_index, N)
        if isinstance(layers, list):
            layers = torch.tensor(layers, dtype=torch.long)

        old_threshold = _multilevel_mod._STREAMING_THRESHOLD
        try:
            _multilevel_mod._STREAMING_THRESHOLD = 100
            result = coarsen_once(edge_index, N, node_sizes, layers)
        finally:
            _multilevel_mod._STREAMING_THRESHOLD = old_threshold

        ratio = result.num_nodes / N
        # Triple matching: ceil(N/3) coarse nodes → ratio ~0.34
        assert 0.3 < ratio < 0.5, f"Reduction ratio {ratio:.3f} outside expected range"

    def test_streaming_preserves_float16_node_sizes(self):
        """Streaming coarsening must accept compact float16 node sizes."""
        edge_index, N, node_sizes = _make_layered_dag(100, 5)
        node_sizes = node_sizes.to(torch.float16)
        layers = longest_path_layering(edge_index, N)
        if isinstance(layers, list):
            layers = torch.tensor(layers, dtype=torch.long)

        old_threshold = _multilevel_mod._STREAMING_THRESHOLD
        try:
            _multilevel_mod._STREAMING_THRESHOLD = 100
            result = coarsen_once(edge_index, N, node_sizes, layers)
        finally:
            _multilevel_mod._STREAMING_THRESHOLD = old_threshold

        assert result.node_sizes.dtype == torch.float16
        assert result.node_sizes.shape == (result.num_nodes, 2)


@pytest.mark.smoke
class TestAdaptiveVectorizedCoarsening:
    """Vectorized coarsening uses structural signals without touching streaming path."""

    def test_structurally_similar_pairs_merge_together(self):
        edge_index, N, node_sizes, layers = _make_structural_coarsening_case()
        result = coarsen_once(edge_index, N, node_sizes, layers)

        fine_to_coarse = result.fine_to_coarse.tolist()
        assert fine_to_coarse[6] == fine_to_coarse[7]

    def test_hub_node_can_stay_singleton(self):
        edge_index, N, node_sizes, layers = _make_structural_coarsening_case()
        result = coarsen_once(edge_index, N, node_sizes, layers)

        coarse_id = result.fine_to_coarse[8].item()
        group_size = (result.fine_to_coarse == coarse_id).sum().item()
        assert group_size == 1


@pytest.mark.smoke
class TestChunkedLayeringMatchesOriginal:
    """Chunked layering produces identical layer assignments."""

    def test_layer_assignments_match(self, monkeypatch):
        """Layer assignments should be identical with chunked vs full processing."""
        import dagua.utils as _utils_mod

        edge_index, N, _ = _make_layered_dag(100, 5)

        # Run with full (non-chunked) path
        original_threshold = _utils_mod._STREAMING_NODE_THRESHOLD
        try:
            _utils_mod._STREAMING_NODE_THRESHOLD = N + 1  # force non-chunked
            layers_full = _longest_path_layering_vectorized(edge_index, N)

            _utils_mod._STREAMING_NODE_THRESHOLD = 100  # force chunked
            layers_chunked = _longest_path_layering_vectorized(edge_index, N)
        finally:
            _utils_mod._STREAMING_NODE_THRESHOLD = original_threshold

        assert torch.equal(layers_full, layers_chunked), (
            f"Layer assignments differ: max diff = {(layers_full - layers_chunked).abs().max().item()}"
        )


@pytest.mark.smoke
class TestGpuProlongationGuard:
    def test_disabled_for_cpu_device(self):
        pos = torch.zeros(4, 2)
        fine_to_coarse = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        assert not _can_prolong_on_gpu(pos, fine_to_coarse, 4, "cpu")

    def test_disabled_for_cpu_position_tensor(self, monkeypatch):
        pos = torch.zeros(4, 2)
        fine_to_coarse = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        assert not _can_prolong_on_gpu(pos.to("cpu"), fine_to_coarse, 4, "cuda")

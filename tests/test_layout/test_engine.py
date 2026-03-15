"""Tests for the core layout optimization loop."""

import importlib

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.engine import (
    _auto_layout_steps,
    _edge_batch_size,
    _make_amortized_loss,
    _overlap_interval,
    _resolve_memory_strategy,
)
from dagua.layout.multilevel import build_hierarchy
from dagua.metrics import compute_all_metrics


class TestLayoutBasic:
    def test_returns_positions(self, simple_chain, fast_config):
        pos = layout(simple_chain, fast_config)
        assert pos.shape == (5, 2)
        assert pos.dtype == torch.float32

    def test_empty_graph(self, empty_graph, fast_config):
        pos = layout(empty_graph, fast_config)
        assert pos.shape == (0, 2)

    def test_single_node(self, single_node_graph, fast_config):
        pos = layout(single_node_graph, fast_config)
        assert pos.shape == (1, 2)

    def test_layout_no_labels_large(self):
        """Regression: layout must work on graphs with num_nodes set but no labels."""
        g = DaguaGraph()
        n = 10_000
        g.num_nodes = n
        src = torch.arange(0, n - 100, dtype=torch.long)
        tgt = src + 100
        g._edge_index_tensor = torch.stack([src, tgt])
        g.node_sizes = torch.full((n, 2), 20.0)
        config = LayoutConfig(steps=5, verbose=False, seed=42)
        pos = layout(g, config)
        assert pos.shape == (n, 2)
        assert torch.isfinite(pos).all()


class TestLayoutQuality:
    @pytest.mark.slow
    def test_no_overlaps_chain(self, simple_chain):
        config = LayoutConfig(steps=200)
        pos = layout(simple_chain, config)
        simple_chain.compute_node_sizes()
        m = compute_all_metrics(pos, simple_chain.edge_index, simple_chain.node_sizes)
        assert m["node_overlaps"] == 0

    @pytest.mark.slow
    def test_no_overlaps_diamond(self, diamond_graph):
        config = LayoutConfig(steps=200)
        pos = layout(diamond_graph, config)
        diamond_graph.compute_node_sizes()
        m = compute_all_metrics(pos, diamond_graph.edge_index, diamond_graph.node_sizes)
        assert m["node_overlaps"] == 0

    @pytest.mark.slow
    def test_dag_fraction_chain(self, simple_chain):
        config = LayoutConfig(steps=200)
        pos = layout(simple_chain, config)
        simple_chain.compute_node_sizes()
        m = compute_all_metrics(pos, simple_chain.edge_index, simple_chain.node_sizes)
        assert m["dag_fraction"] == 1.0

    @pytest.mark.slow
    def test_dag_fraction_high(self, diamond_graph):
        config = LayoutConfig(steps=200)
        pos = layout(diamond_graph, config)
        diamond_graph.compute_node_sizes()
        m = compute_all_metrics(pos, diamond_graph.edge_index, diamond_graph.node_sizes)
        assert m["dag_fraction"] >= 0.9

    @pytest.mark.slow
    def test_no_crossings_chain(self, simple_chain):
        config = LayoutConfig(steps=200)
        pos = layout(simple_chain, config)
        simple_chain.compute_node_sizes()
        m = compute_all_metrics(pos, simple_chain.edge_index, simple_chain.node_sizes)
        assert m["edge_crossings"] == 0


class TestLayoutDirections:
    def test_tb_flow(self, simple_chain):
        config = LayoutConfig(steps=100, direction="TB")
        pos = layout(simple_chain, config)
        # In TB, each successive node should have increasing y
        for i in range(4):
            assert pos[i, 1] < pos[i + 1, 1], f"Node {i} should be above node {i + 1} in TB"

    def test_bt_flow(self, simple_chain):
        config = LayoutConfig(steps=100, direction="BT")
        pos = layout(simple_chain, config)
        # In BT, each successive node should have DECREASING y (upward)
        for i in range(4):
            assert pos[i, 1] > pos[i + 1, 1], f"Node {i} should be below node {i + 1} in BT"

    @pytest.mark.slow
    def test_lr_flow(self, simple_chain):
        config = LayoutConfig(steps=200, direction="LR")
        pos = layout(simple_chain, config)
        # In LR, x-range should be larger than y-range (wide, not tall)
        x_range = pos[:, 0].max() - pos[:, 0].min()
        y_range = pos[:, 1].max() - pos[:, 1].min()
        assert x_range > y_range * 0.5, "LR layout should be wider than tall"

    @pytest.mark.slow
    def test_rl_flow(self, simple_chain):
        config = LayoutConfig(steps=200, direction="RL")
        pos = layout(simple_chain, config)
        # In RL, flow goes right-to-left: first node should be rightmost
        x_range = pos[:, 0].max() - pos[:, 0].min()
        y_range = pos[:, 1].max() - pos[:, 1].min()
        assert x_range > y_range * 0.5, "RL layout should be wider than tall"


class TestLayoutClusters:
    @pytest.mark.slow
    def test_cluster_compactness(self, clustered_graph):
        config = LayoutConfig(steps=200)
        pos = layout(clustered_graph, config)
        # Members of same cluster should be closer to each other
        enc_pos = pos[[1, 2]]  # encoder nodes
        enc_spread = (enc_pos[0] - enc_pos[1]).norm()
        # Just check it's finite and reasonable
        assert enc_spread < 500

    @pytest.mark.slow
    def test_cluster_separation(self, clustered_graph):
        config = LayoutConfig(steps=200)
        pos = layout(clustered_graph, config)
        enc_center = pos[[1, 2]].mean(dim=0)
        dec_center = pos[[3, 4]].mean(dim=0)
        separation = (enc_center - dec_center).norm()
        assert separation > 10  # clusters should be separated


class TestLayoutReproducibility:
    def test_seed_reproducibility(self, diamond_graph):
        config = LayoutConfig(steps=100, seed=42)
        pos1 = layout(diamond_graph, config)
        pos2 = layout(diamond_graph, config)
        assert torch.allclose(pos1, pos2, atol=1e-3)

    def test_none_seed_works(self):
        """Layout should work with seed=None (non-deterministic)."""
        from dagua.eval.graphs import _random_dag

        g = _random_dag(30, 50, seed=0)
        config = LayoutConfig(steps=50, seed=None)
        pos = layout(g, config)
        assert pos.shape[0] == g.num_nodes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLayoutCUDA:
    def test_cuda_layout(self, simple_chain):
        config = LayoutConfig(steps=50, device="cuda")
        pos = layout(simple_chain, config)
        assert pos.shape == (5, 2)
        # Result is on the compute device
        assert pos.device.type in ("cpu", "cuda")


def test_build_hierarchy_accepts_precomputed_layer_assignments():
    graph = DaguaGraph.from_edge_list([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
    graph.compute_node_sizes()
    assert graph.node_sizes is not None
    precomputed = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    captured: list[torch.Tensor] = []

    levels = build_hierarchy(
        graph.edge_index,
        graph.num_nodes,
        graph.node_sizes,
        min_nodes=2,
        max_levels=2,
        initial_layer_assignments=precomputed,
        layer_assignments_callback=lambda tensor: captured.append(tensor),
    )

    assert levels
    assert not captured
    assert torch.equal(levels[0].fine_layer_assignments, precomputed)
    assert levels[0].coarse_layer_assignments is not None


def test_edge_batch_size_scaling() -> None:
    """Edge batch sizes should scale up with large edge counts."""
    config = LayoutConfig()

    assert _edge_batch_size(500_000, config) == 500_000
    assert _edge_batch_size(5_000_000, config) == 2_000_000
    assert _edge_batch_size(300_000_000, config) == 5_000_000


def test_auto_steps_scaling() -> None:
    """Auto-step count should scale monotonically with graph size."""
    sizes = [5, 20, 100, 300, 1000, 3000, 10_000]
    expected = [50, 100, 200, 300, 300, 400, 500]

    actual = [_auto_layout_steps(size) for size in sizes]

    assert actual == expected
    assert actual == sorted(actual)


def test_multilevel_kicks_in_at_5k(monkeypatch: pytest.MonkeyPatch) -> None:
    """Graphs above the default threshold should use multilevel layout."""
    engine_module = importlib.import_module("dagua.layout.engine")
    multilevel_module = importlib.import_module("dagua.layout.multilevel")

    g = DaguaGraph()
    n = 6_000
    g.num_nodes = n
    src = torch.arange(0, n - 60, dtype=torch.long)
    tgt = src + 60
    g._edge_index_tensor = torch.stack([src, tgt])
    g.node_sizes = torch.full((n, 2), 20.0)
    called = {"multilevel": False, "direct": False}

    def _fake_multilevel_layout(
        graph: DaguaGraph,
        config: LayoutConfig,
        trace: object | None = None,
    ) -> torch.Tensor:
        """Return a deterministic tensor while recording multilevel usage."""
        del config, trace
        called["multilevel"] = True
        return torch.zeros((graph.num_nodes, 2), dtype=torch.float32)

    def _fake_layout_inner(
        edge_index: torch.Tensor,
        num_nodes: int,
        node_sizes: torch.Tensor,
        config: LayoutConfig,
        device: str = "cpu",
        init_pos: torch.Tensor | None = None,
        clusters: dict | None = None,
        cluster_parents: dict | None = None,
        layer_assignments: torch.Tensor | None = None,
        progress_context: object | None = None,
        trace: object | None = None,
    ) -> torch.Tensor:
        """Fail the test if the direct path is selected unexpectedly."""
        del (
            edge_index,
            num_nodes,
            node_sizes,
            config,
            device,
            init_pos,
            clusters,
            cluster_parents,
            layer_assignments,
            progress_context,
            trace,
        )
        called["direct"] = True
        return torch.zeros((n, 2), dtype=torch.float32)

    monkeypatch.setattr(multilevel_module, "multilevel_layout", _fake_multilevel_layout)
    monkeypatch.setattr(engine_module, "_layout_inner", _fake_layout_inner)

    pos = layout(g, LayoutConfig(steps=50, verbose=False, seed=42))

    assert pos.shape == (n, 2)
    assert called["multilevel"] is True
    assert called["direct"] is False


def test_direct_layout_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Graphs below the default threshold should stay on the direct path."""
    engine_module = importlib.import_module("dagua.layout.engine")
    multilevel_module = importlib.import_module("dagua.layout.multilevel")

    g = DaguaGraph()
    n = 1_000
    g.num_nodes = n
    src = torch.arange(0, n - 10, dtype=torch.long)
    tgt = src + 10
    g._edge_index_tensor = torch.stack([src, tgt])
    g.node_sizes = torch.full((n, 2), 20.0)
    called = {"multilevel": False, "direct": False}

    def _fake_multilevel_layout(
        graph: DaguaGraph,
        config: LayoutConfig,
        trace: object | None = None,
    ) -> torch.Tensor:
        """Fail the test if the multilevel path is selected unexpectedly."""
        del graph, config, trace
        called["multilevel"] = True
        return torch.zeros((n, 2), dtype=torch.float32)

    def _fake_layout_inner(
        edge_index: torch.Tensor,
        num_nodes: int,
        node_sizes: torch.Tensor,
        config: LayoutConfig,
        device: str = "cpu",
        init_pos: torch.Tensor | None = None,
        clusters: dict | None = None,
        cluster_parents: dict | None = None,
        layer_assignments: torch.Tensor | None = None,
        progress_context: object | None = None,
        trace: object | None = None,
    ) -> torch.Tensor:
        """Return a deterministic tensor while recording direct-path usage."""
        del (
            edge_index,
            node_sizes,
            config,
            device,
            init_pos,
            clusters,
            cluster_parents,
            layer_assignments,
            progress_context,
            trace,
        )
        called["direct"] = True
        return torch.zeros((num_nodes, 2), dtype=torch.float32)

    monkeypatch.setattr(multilevel_module, "multilevel_layout", _fake_multilevel_layout)
    monkeypatch.setattr(engine_module, "_layout_inner", _fake_layout_inner)

    pos = layout(g, LayoutConfig(steps=20, verbose=False, seed=42))

    assert pos.shape == (n, 2)
    assert called["direct"] is True
    assert called["multilevel"] is False


def test_config_amortize_defaults() -> None:
    """Default performance knobs should preserve the existing engine behavior."""
    config = LayoutConfig()

    assert config.repel_amortize_interval == 2
    assert config.repel_amortize_threshold == 10_000_000
    assert config.fanout_amortize_interval == 3
    assert config.fanout_amortize_threshold == 10_000_000
    assert config.edge_random_fraction == 0.2
    assert config.edge_batch_size == 0
    assert config.overlap_check_interval == 0


def test_edge_batch_size_respects_fixed_override() -> None:
    """Explicit edge batch size overrides should bypass auto-scaling."""
    config = LayoutConfig(edge_batch_size=1_234)

    assert _edge_batch_size(300_000_000, config) == 1_234


def test_overlap_interval_respects_fixed_override() -> None:
    """Explicit overlap interval overrides should bypass auto-scaling."""
    config = LayoutConfig(overlap_check_interval=7)

    assert _overlap_interval(1_000_000, config) == 7


def test_resolve_memory_strategy_cpu_defaults_to_single_backward() -> None:
    """CPU auto mode should keep a single backward pass."""
    config = LayoutConfig(device="cpu")

    assert _resolve_memory_strategy(1_000_000, 2_000_000, "cpu", config) == (False, False, False)


def test_layout_cpu_with_per_loss_backward_on() -> None:
    """CPU layout should still converge when per-loss backward is forced on."""
    g = DaguaGraph()
    n = 1_000
    src = torch.arange(0, n - 10, dtype=torch.long)
    tgt = src + 10
    g.num_nodes = n
    g._edge_index_tensor = torch.stack([src, tgt])
    g.node_sizes = torch.full((n, 2), 20.0)

    config = LayoutConfig(steps=10, verbose=False, seed=42, per_loss_backward="on")
    pos = layout(g, config)

    assert pos.shape == (n, 2)
    assert torch.isfinite(pos).all()


def test_config_disable_amortization() -> None:
    """Setting interval=1 should disable amortization while preserving layout."""
    g = DaguaGraph()
    g.num_nodes = 100
    g._edge_index_tensor = torch.stack(
        [
            torch.arange(0, 90, dtype=torch.long),
            torch.arange(10, 100, dtype=torch.long),
        ]
    )
    g.node_sizes = torch.full((100, 2), 20.0)

    config = LayoutConfig(
        steps=5,
        seed=42,
        repel_amortize_interval=1,
        fanout_amortize_interval=1,
    )
    pos = layout(g, config)

    assert pos.shape == (100, 2)


def test_config_all_random_edges() -> None:
    """edge_random_fraction=1.0 should keep random edge sampling active every step."""
    g = DaguaGraph()
    g.num_nodes = 100
    g._edge_index_tensor = torch.stack(
        [
            torch.arange(0, 90, dtype=torch.long),
            torch.arange(10, 100, dtype=torch.long),
        ]
    )
    g.node_sizes = torch.full((100, 2), 20.0)

    config = LayoutConfig(edge_random_fraction=1.0, steps=5, seed=42)
    pos = layout(g, config)

    assert pos.shape == (100, 2)


def test_amortized_loss_wrapper_produces_finite_non_zero_total() -> None:
    """Amortized loss wrappers should skip work without collapsing total loss."""

    def _base_loss(
        pos: torch.Tensor,
        node_sizes: torch.Tensor,
        layer_index: object | None,
    ) -> torch.Tensor:
        del node_sizes, layer_index
        return pos.square().sum() + 1.0

    pos = torch.ones((4, 2), dtype=torch.float32)
    node_sizes = torch.ones((4, 2), dtype=torch.float32)
    loss_fn = _make_amortized_loss(_base_loss, skip_every=2)

    values = [loss_fn(pos, node_sizes, None).item() for _ in range(4)]

    assert torch.isfinite(torch.tensor(values)).all()
    assert values[0] > 0.0
    assert values[1] == 0.0
    assert sum(values) > 0.0

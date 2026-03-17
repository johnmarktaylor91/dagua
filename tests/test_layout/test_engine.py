"""Tests for the core layout optimization loop."""

import importlib
import time

import numpy as np
import pytest
import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.constraints import edge_attraction_loss, edge_straightness_loss
from dagua.layout.engine import (
    _auto_layout_steps,
    _create_optimizer,
    _edge_batch_size,
    _estimate_gpu_memory,
    _estimate_hybrid_gpu_memory,
    _layout_inner,
    _make_amortized_loss,
    _overlap_interval,
    _override_for_tree,
    _resolve_memory_strategy,
    _should_use_tiled_gpu,
)
from dagua.layout.graph_classify import GraphFamily, GraphStructure, classify_graph
from dagua.layout.layers import build_layer_index
from dagua.layout.multilevel import (
    _scaled_amortization,
    _scaled_final_refine_steps,
    _scaled_sample_cap,
    _select_refinement_execution,
    build_hierarchy,
    coarsen_once,
)
from dagua.layout.tiled_compute import _partition_edges_by_tile
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


def test_coarsening_reaches_min_nodes() -> None:
    """Hierarchy should coarsen close to min_nodes rather than stopping early."""
    n = 100_000
    width = 100
    src = torch.arange(0, n - width, dtype=torch.long)
    tgt = src + width
    edge_index = torch.stack([src, tgt])
    node_sizes = torch.full((n, 2), 20.0)
    layers = torch.arange(n, dtype=torch.long) // width

    levels = build_hierarchy(
        edge_index,
        n,
        node_sizes,
        min_nodes=2000,
        device="cpu",
        initial_layer_assignments=layers,
    )

    coarsest_n = levels[-1].num_nodes
    assert coarsest_n < 10000, f"Coarsest has {coarsest_n} nodes, expected < 10K"


def test_edge_batch_size_scaling() -> None:
    """Edge batch sizes should scale up with large edge counts."""
    config = LayoutConfig()

    assert _edge_batch_size(100_000, config) == 50000
    assert _edge_batch_size(200_000, config) == 200000
    assert _edge_batch_size(500_000, config) == 200000
    assert _edge_batch_size(1_000_000, config) == 500_000
    assert _edge_batch_size(5_000_000, config) == 2_000_000
    assert _edge_batch_size(300_000_000, config) == 5_000_000


def test_scaled_sample_cap() -> None:
    """Huge final levels should reduce sampled active-node caps."""
    assert _scaled_sample_cap(1_000) == 1_000_000
    assert _scaled_sample_cap(100_000_000) == 500_000
    assert _scaled_sample_cap(1_000_000_000) == 100_000


def test_scaled_amortization_preserves_small_graph_defaults() -> None:
    """Amortization overrides should leave sub-50M graphs unchanged."""
    assert _scaled_amortization(25) == (10, 5)
    assert _scaled_amortization(1_000) == (3, 5)
    assert _scaled_amortization(1_000_000) == (3, 20)
    assert _scaled_amortization(49_999_999) == (3, 40)
    assert _scaled_amortization(100_000_000) == (5, 30)
    assert _scaled_amortization(300_000_000) == (8, 50)
    assert _scaled_amortization(1_000_000_000) == (10, 100)


def test_scaled_final_refine_steps_only_changes_huge_final_level() -> None:
    """Final-level step reduction should start only at the documented thresholds."""
    assert _scaled_final_refine_steps(level_index=0, fine_n=49_999_999, base_refine=15) == 30
    assert _scaled_final_refine_steps(level_index=0, fine_n=100_000_000, base_refine=15) == 18
    assert _scaled_final_refine_steps(level_index=0, fine_n=200_000_000, base_refine=15) == 12
    assert _scaled_final_refine_steps(level_index=0, fine_n=500_000_000, base_refine=15) == 8
    assert _scaled_final_refine_steps(level_index=3, fine_n=500_000_000, base_refine=15) == 7


def test_edge_batch_size_cuda_uses_available_vram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA auto-batching should use the largest safe batch or all edges if they fit."""
    config = LayoutConfig(device="cuda")

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (128_000_000, 256_000_000))

    assert _edge_batch_size(100_000, config) == 0
    assert _edge_batch_size(500_000, config) == 300_000


def test_gpu_memory_estimate_no_phantom_edges() -> None:
    """Streamed edges should not be counted as a full resident GPU edge tensor."""
    num_nodes = 50_000_000
    num_edges = 75_000_000

    resident = _estimate_gpu_memory(num_nodes, num_edges, per_loss_bw=True, edges_on_cpu=False)
    streamed = _estimate_gpu_memory(
        num_nodes,
        num_edges,
        per_loss_bw=True,
        edges_on_cpu=True,
        edge_batch=5_000_000,
    )

    assert streamed < resident - 500_000_000


def test_hybrid_memory_estimate_optimizer_tiers() -> None:
    """Hybrid memory estimates should get cheaper as optimizer state shrinks."""
    num_nodes = 200_000_000
    num_edges = 300_000_000

    adam = _estimate_hybrid_gpu_memory(num_nodes, num_edges, optimizer_type="adam")
    nesterov = _estimate_hybrid_gpu_memory(num_nodes, num_edges, optimizer_type="sgd_nesterov")
    sgd = _estimate_hybrid_gpu_memory(num_nodes, num_edges, optimizer_type="sgd")

    assert adam > nesterov > sgd
    assert pytest.approx((adam - nesterov) / 1024**3, rel=0.0, abs=0.2) == 1.5
    assert pytest.approx((nesterov - sgd) / 1024**3, rel=0.0, abs=0.2) == 1.5


def test_create_optimizer_uses_requested_tier() -> None:
    """Optimizer creation should preserve the requested algorithm and LR scaling."""
    pos = torch.zeros((4, 2), requires_grad=True)
    config = LayoutConfig(lr=0.1)

    adam = _create_optimizer(pos, config, optimizer_type="adam")
    nesterov = _create_optimizer(pos, config, optimizer_type="sgd_nesterov")
    sgd = _create_optimizer(pos, config, optimizer_type="sgd")

    assert isinstance(adam, torch.optim.Adam)
    assert adam.param_groups[0]["lr"] == pytest.approx(0.1)
    assert isinstance(nesterov, torch.optim.SGD)
    assert nesterov.param_groups[0]["lr"] == pytest.approx(0.3)
    assert nesterov.param_groups[0]["nesterov"] is True
    assert nesterov.param_groups[0]["momentum"] == pytest.approx(0.9)
    assert isinstance(sgd, torch.optim.SGD)
    assert sgd.param_groups[0]["lr"] == pytest.approx(0.5)
    assert sgd.param_groups[0]["nesterov"] is False
    assert sgd.param_groups[0]["momentum"] == pytest.approx(0.0)


def test_memory_strategy_selects_plb_not_hybrid_at_50m(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto strategy should keep large streamed-edge layouts on GPU via per-loss backward."""
    engine_module = importlib.import_module("dagua.layout.engine")

    class _FakeBudget:
        """Minimal VRAM budget stub for deterministic strategy selection tests."""

        def remaining(self) -> int:
            """Return the mocked usable VRAM budget in bytes."""
            return 11 * 1024**3

    config = LayoutConfig(device="cuda")

    monkeypatch.setattr(engine_module, "VRAMBudget", _FakeBudget)

    use_plb, use_checkpointing, use_hybrid = _resolve_memory_strategy(
        50_000_000,
        75_000_000,
        "cuda",
        config,
        edges_on_cpu=True,
        edge_batch=5_000_000,
    )

    assert use_plb is True
    assert use_checkpointing is False
    assert use_hybrid is False


def test_refinement_execution_uses_nesterov_fallback_when_adam_does_not_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Huge refinement levels should try cheaper hybrid optimizers before CPU."""
    engine_module = importlib.import_module("dagua.layout.engine")
    multilevel_module = importlib.import_module("dagua.layout.multilevel")

    def _fake_estimate_gpu_memory(
        n: int,
        num_edges: int,
        per_loss_bw: bool = False,
        *,
        edges_on_cpu: bool = False,
        edge_batch: int = 0,
        n_active_override: int | None = None,
    ) -> int:
        """Return large Adam estimates that force the fallback cascade."""
        del n, num_edges, edges_on_cpu, edge_batch, n_active_override
        return 120 if per_loss_bw else 130

    def _fake_estimate_hybrid_gpu_memory(
        n: int,
        num_edges: int,
        *,
        edge_batch: int = 5_000_000,
        optimizer_type: str = "adam",
    ) -> int:
        """Return synthetic hybrid estimates keyed by optimizer tier and batch."""
        del n, num_edges
        lookup = {
            ("adam", 5_000_000): 70,
            ("sgd_nesterov", 5_000_000): 60,
            ("sgd_nesterov", 2_500_000): 45,
            ("sgd", 5_000_000): 40,
        }
        return lookup.get((optimizer_type, edge_batch), 80)

    monkeypatch.setattr(engine_module, "_estimate_gpu_memory", _fake_estimate_gpu_memory)
    monkeypatch.setattr(
        engine_module,
        "_estimate_hybrid_gpu_memory",
        _fake_estimate_hybrid_gpu_memory,
    )
    monkeypatch.setattr(multilevel_module, "_vram_fits", lambda needed_bytes: needed_bytes <= 50)

    force_cpu, force_hybrid, optimizer_type, edge_batch = _select_refinement_execution(
        200_000_000,
        300_000_000,
        LayoutConfig(device="cuda"),
        "cuda",
    )

    assert force_cpu is False
    assert force_hybrid is True
    assert optimizer_type == "sgd_nesterov"
    assert edge_batch == 2_500_000


def test_refinement_execution_keeps_50m_behavior_without_sgd_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new fallback tiers should stay disabled at and below 50M nodes."""
    engine_module = importlib.import_module("dagua.layout.engine")
    multilevel_module = importlib.import_module("dagua.layout.multilevel")

    monkeypatch.setattr(
        engine_module,
        "_estimate_gpu_memory",
        lambda *args, **kwargs: 120,
    )
    monkeypatch.setattr(
        engine_module,
        "_estimate_hybrid_gpu_memory",
        lambda *args, **kwargs: 70 if kwargs.get("optimizer_type", "adam") == "adam" else 40,
    )
    monkeypatch.setattr(multilevel_module, "_vram_fits", lambda needed_bytes: needed_bytes <= 50)

    force_cpu, force_hybrid, optimizer_type, edge_batch = _select_refinement_execution(
        50_000_000,
        75_000_000,
        LayoutConfig(device="cuda"),
        "cuda",
    )

    assert force_cpu is True
    assert force_hybrid is False
    assert optimizer_type == "adam"
    assert edge_batch == 5_000_000


def test_refinement_execution_respects_adam_only_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """adam-only fallback policy should skip SGD tiers and fall back to CPU."""
    engine_module = importlib.import_module("dagua.layout.engine")
    multilevel_module = importlib.import_module("dagua.layout.multilevel")

    monkeypatch.setattr(
        engine_module,
        "_estimate_gpu_memory",
        lambda *args, **kwargs: 120,
    )
    monkeypatch.setattr(
        engine_module,
        "_estimate_hybrid_gpu_memory",
        lambda *args, **kwargs: 70 if kwargs.get("optimizer_type", "adam") == "adam" else 40,
    )
    monkeypatch.setattr(multilevel_module, "_vram_fits", lambda needed_bytes: needed_bytes <= 50)

    force_cpu, force_hybrid, optimizer_type, edge_batch = _select_refinement_execution(
        200_000_000,
        300_000_000,
        LayoutConfig(device="cuda", optimizer_fallback="adam"),
        "cuda",
    )

    assert force_cpu is True
    assert force_hybrid is False
    assert optimizer_type == "adam"
    assert edge_batch == 5_000_000


def test_auto_steps_scaling() -> None:
    """Auto-step count should scale monotonically with graph size."""
    sizes = [5, 20, 100, 300, 1000, 3000, 10_000]
    expected = [50, 100, 150, 200, 250, 300, 400]

    actual = [_auto_layout_steps(size) for size in sizes]

    assert actual == expected
    assert actual == sorted(actual)


def test_classify_tree() -> None:
    """Linear trees should classify as chains."""
    edges = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9]],
        dtype=torch.long,
    )

    result = classify_graph(edges, 10)

    assert result.family == GraphFamily.CHAIN
    assert result.num_components == 1
    assert result.max_degree == 2


def test_classify_general() -> None:
    """Dense cyclic graphs should remain on the general path."""
    edges = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3], [1, 2, 2, 3, 3, 0, 0]],
        dtype=torch.long,
    )

    result = classify_graph(edges, 4)

    assert result.family == GraphFamily.GENERAL


def test_early_exit_forest() -> None:
    """Forest graphs with ``E < N - 1`` should still classify as forests."""
    edges = torch.tensor(
        [[0, 1, 3], [1, 2, 4]],
        dtype=torch.long,
    )

    result = classify_graph(edges, 6)

    assert result.family == GraphFamily.FOREST
    assert result.num_components == 3


def test_classify_wide_layered() -> None:
    """Wide shallow layerings should classify as wide-layered."""
    layers = torch.arange(1000, dtype=torch.long) // 100
    edges_src = torch.arange(0, 900, dtype=torch.long)
    edges_tgt = edges_src + 100

    result = classify_graph(
        torch.stack([edges_src, edges_tgt]),
        1000,
        layer_assignments=layers,
    )

    assert result.family == GraphFamily.WIDE_LAYERED
    assert result.num_layers == 10
    assert result.avg_layer_width == pytest.approx(100.0)


def test_layout_inner_with_prebuilt(monkeypatch: pytest.MonkeyPatch) -> None:
    """_layout_inner should accept prebuilt structure and layer metadata."""
    engine_module = importlib.import_module("dagua.layout.engine")

    edges = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 20.0)
    layers = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    structure = classify_graph(edges, 4, layer_assignments=layers)
    layer_index = build_layer_index(layers)

    monkeypatch.setattr(
        engine_module,
        "classify_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected classify_graph")),
    )
    monkeypatch.setattr(
        engine_module,
        "build_layer_index",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected build_layer_index")
        ),
    )

    pos = _layout_inner(
        edges,
        4,
        node_sizes,
        LayoutConfig(steps=5, seed=42),
        device="cpu",
        layer_assignments=layers,
        graph_structure=structure,
        prebuilt_layer_index=layer_index,
    )

    assert pos.shape == (4, 2)
    assert torch.isfinite(pos).all()


def test_classify_early_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dense non-tree graphs should bypass the Python union-find."""
    graph_classify_module = importlib.import_module("dagua.layout.graph_classify")

    n = 100_000
    src = torch.cat(
        [torch.arange(0, n - 1, dtype=torch.long), torch.arange(1, n, dtype=torch.long)]
    )
    tgt = torch.cat(
        [torch.arange(1, n, dtype=torch.long), torch.arange(0, n - 1, dtype=torch.long)]
    )
    edges = torch.stack([src, tgt])
    layers = torch.arange(n, dtype=torch.long)

    def _unexpected_find_root(*args: object, **kwargs: object) -> int:
        """Fail if the dense-graph early exit falls back to union-find."""
        del args, kwargs
        raise AssertionError("union-find should be skipped")

    monkeypatch.setattr(
        graph_classify_module,
        "_find_root",
        _unexpected_find_root,
    )

    start = time.perf_counter()
    result = classify_graph(edges, n, layer_assignments=layers)
    elapsed = time.perf_counter() - start

    assert result.family == GraphFamily.GENERAL
    assert elapsed < 0.1


def test_sampled_context_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sampled-node context should refresh on the configured cadence."""
    engine_module = importlib.import_module("dagua.layout.engine")

    n = 64
    edges = torch.stack(
        [
            torch.arange(0, n - 8, dtype=torch.long),
            torch.arange(8, n, dtype=torch.long),
        ]
    )
    node_sizes = torch.full((n, 2), 20.0)
    build_calls = 0
    original_builder = engine_module._build_sampled_node_context

    def _counting_builder(
        num_nodes: int,
        layer_index: object,
        device: str | torch.device,
        rvs_nn_k: int,
    ) -> object:
        """Count sampled-context rebuilds while delegating to the real helper."""
        nonlocal build_calls
        build_calls += 1
        return original_builder(num_nodes, layer_index, device, rvs_nn_k)

    monkeypatch.setattr(engine_module, "_build_sampled_node_context", _counting_builder)

    pos = _layout_inner(
        edges,
        n,
        node_sizes,
        LayoutConfig(
            steps=20,
            seed=42,
            exact_repulsion_threshold=8,
            repel_amortize_threshold=0,
            repel_amortize_interval=4,
        ),
        device="cpu",
    )

    assert pos.shape == (n, 2)
    assert torch.isfinite(pos).all()
    assert build_calls == 5


def test_edge_ctx_skipped_under_plb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-loss backward should skip the shared edge-batch context build."""
    engine_module = importlib.import_module("dagua.layout.engine")

    edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 20.0)

    def _unexpected_edge_ctx(*args: object, **kwargs: object) -> object:
        """Fail if the dead edge context is still built under per-loss backward."""
        del args, kwargs
        raise AssertionError("EdgeBatchContext should not be constructed")

    def _edge_attract_without_ctx(
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        x_bias: float = 1.0,
        edge_ctx: object | None = None,
    ) -> torch.Tensor:
        """Assert per-loss backward keeps edge-based losses on the fallback path."""
        del edge_index, x_bias
        assert edge_ctx is None
        return pos.sum() * 0.0

    monkeypatch.setattr(engine_module, "EdgeBatchContext", _unexpected_edge_ctx)
    monkeypatch.setattr(engine_module, "edge_attraction_loss", _edge_attract_without_ctx)

    pos = _layout_inner(
        edges,
        4,
        node_sizes,
        LayoutConfig(
            steps=3,
            seed=42,
            w_dag=0.0,
            w_attract=2.0,
            w_repel=0.0,
            w_overlap=0.0,
            w_cluster=0.0,
            w_cluster_contain=0.0,
            w_crossing=0.0,
            w_straightness=0.0,
            w_length_variance=0.0,
            w_spacing=0.0,
            w_fanout=0.0,
            w_back_edge=0.0,
            per_loss_backward="on",
        ),
        device="cpu",
    )

    assert pos.shape == (4, 2)
    assert torch.isfinite(pos).all()


def test_refinement_skips_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multilevel refinement should not re-run graph classification per level."""
    engine_module = importlib.import_module("dagua.layout.engine")
    multilevel_module = importlib.import_module("dagua.layout.multilevel")

    n = 2_500
    width = 50
    src_primary = torch.arange(0, n - width, dtype=torch.long)
    tgt_primary = src_primary + width
    src_secondary = torch.arange(0, n - width - 1, dtype=torch.long)
    tgt_secondary = src_secondary + width + 1

    g = DaguaGraph()
    g.num_nodes = n
    g._edge_index_tensor = torch.cat(
        [
            torch.stack([src_primary, tgt_primary]),
            torch.stack([src_secondary, tgt_secondary]),
        ],
        dim=1,
    )
    g.node_sizes = torch.full((n, 2), 20.0)

    classify_calls = 0
    classify_impl = classify_graph

    def _counting_classify(*args: object, **kwargs: object) -> GraphStructure:
        """Count structural classification calls while delegating to the real helper."""
        nonlocal classify_calls
        classify_calls += 1
        return classify_impl(*args, **kwargs)

    monkeypatch.setattr(engine_module, "classify_graph", _counting_classify)
    monkeypatch.setattr(multilevel_module, "classify_graph", _counting_classify)

    pos = layout(
        g,
        LayoutConfig(
            steps=1,
            seed=42,
            verbose=False,
            multilevel_threshold=100,
            multilevel_min_nodes=100,
            multilevel_coarse_steps=1,
            multilevel_refine_steps=1,
            w_repel=0.0,
            w_overlap=0.0,
            w_crossing=0.0,
            w_straightness=0.0,
            w_length_variance=0.0,
            w_fanout=0.0,
            w_back_edge=0.0,
        ),
    )

    assert pos.shape == (n, 2)
    assert torch.isfinite(pos).all()
    assert classify_calls <= 2


def test_override_for_tree_disables_tree_irrelevant_losses() -> None:
    """Tree overrides should only zero the tree-irrelevant loss terms."""
    config = LayoutConfig(w_crossing=1.8, w_straightness=2.2, w_length_variance=0.7)

    updated = _override_for_tree(config)

    assert updated is not config
    assert updated.w_crossing == 0.0
    assert updated.w_straightness == 0.0
    assert updated.w_length_variance == 0.0
    assert config.w_crossing == 1.8
    assert config.w_straightness == 2.2
    assert config.w_length_variance == 0.7


def test_multilevel_kicks_in_at_20k(monkeypatch: pytest.MonkeyPatch) -> None:
    """Graphs above the default threshold should use multilevel layout."""
    engine_module = importlib.import_module("dagua.layout.engine")
    multilevel_module = importlib.import_module("dagua.layout.multilevel")

    g = DaguaGraph()
    n = 21_000
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
        *,
        graph_structure: object | None = None,
        prebuilt_layer_index: object | None = None,
        skip_classification: bool = False,
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
            graph_structure,
            prebuilt_layer_index,
            skip_classification,
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
        *,
        graph_structure: object | None = None,
        prebuilt_layer_index: object | None = None,
        skip_classification: bool = False,
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
            graph_structure,
            prebuilt_layer_index,
            skip_classification,
        )
        called["direct"] = True
        return torch.zeros((num_nodes, 2), dtype=torch.float32)

    monkeypatch.setattr(multilevel_module, "multilevel_layout", _fake_multilevel_layout)
    monkeypatch.setattr(engine_module, "_layout_inner", _fake_layout_inner)

    pos = layout(g, LayoutConfig(steps=20, verbose=False, seed=42))

    assert pos.shape == (n, 2)
    assert called["direct"] is True
    assert called["multilevel"] is False


def test_auto_cpu_for_tiny_graphs() -> None:
    """Tiny graphs should complete even when config prefers CUDA."""
    g = DaguaGraph()
    g.add_node("a")
    g.add_node("b")
    g.add_edge("a", "b")

    pos = layout(g, LayoutConfig(device="cuda", steps=10, seed=42))

    assert pos.shape == (2, 2)


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


def test_resolve_memory_strategy_cpu_auto_enables_per_loss_for_large_graphs() -> None:
    """CPU auto mode should enable per-loss backward only above the large-graph threshold."""
    config = LayoutConfig(device="cpu")

    assert _resolve_memory_strategy(50_000, 100_000, "cpu", config) == (False, False, False)
    assert _resolve_memory_strategy(50_001, 100_000, "cpu", config) == (True, False, False)
    assert _resolve_memory_strategy(1_000_000, 2_000_000, "cpu", config) == (True, False, False)


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


def _make_matching_reference_case() -> tuple[
    torch.Tensor,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build a deterministic layered DAG that exercises pair and triple matching.

    Returns
    -------
    tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor]
        Edge index, node count, node sizes, layer assignments, and cluster IDs.
    """
    num_layers = 10
    layer_width = 100
    num_nodes = num_layers * layer_width
    src: list[int] = []
    tgt: list[int] = []
    for layer_idx in range(num_layers - 1):
        base = layer_idx * layer_width
        next_base = (layer_idx + 1) * layer_width
        skip_base = (layer_idx + 2) * layer_width if layer_idx + 2 < num_layers else -1
        for offset in range(layer_width):
            node = base + offset
            bucket = (offset // 4) * 4
            src.append(node)
            tgt.append(next_base + (bucket % layer_width))
            src.append(node)
            tgt.append(next_base + ((bucket + 1 + (offset % 2)) % layer_width))
            if skip_base >= 0 and offset % 7 == 0:
                src.append(node)
                tgt.append(skip_base + ((offset * 3) % layer_width))
        hub_node = base
        for fanout in range(16):
            src.append(hub_node)
            tgt.append(next_base + fanout)

    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    node_sizes = torch.full((num_nodes, 2), 10.0)
    layers = torch.arange(num_nodes, dtype=torch.long) // layer_width
    cluster_ids = torch.full((num_nodes,), -1, dtype=torch.long)
    for layer_idx in range(num_layers):
        base = layer_idx * layer_width
        cluster_ids[base : base + 20] = 0
        cluster_ids[base + 20 : base + 40] = 1
        cluster_ids[base + 40 : base + 60] = 2
        cluster_ids[base + 80 : base + 100] = 3 + (layer_idx % 2)
    return edge_index, num_nodes, node_sizes, layers, cluster_ids


def _reference_fine_to_coarse(
    edge_index: torch.Tensor,
    num_nodes: int,
    layers: torch.Tensor,
    cluster_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reproduce the original Python matching loop for regression comparison.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edges with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    layers : torch.Tensor
        Layer assignment tensor with shape ``[N]``.
    cluster_ids : torch.Tensor | None, optional
        Optional cluster IDs with shape ``[N]``.

    Returns
    -------
    torch.Tensor
        Fine-to-coarse assignment tensor with shape ``[N]``.
    """
    layers_np = layers.cpu().numpy()
    cluster_ids_np = None if cluster_ids is None else cluster_ids.cpu().numpy()
    min_neighbor = np.full(num_nodes, num_nodes, dtype=np.int64)
    min_parent = np.full(num_nodes, num_nodes, dtype=np.int64)
    min_child = np.full(num_nodes, num_nodes, dtype=np.int64)
    in_degree = np.zeros(num_nodes, dtype=np.int64)
    out_degree = np.zeros(num_nodes, dtype=np.int64)
    skip_degree = np.zeros(num_nodes, dtype=np.int64)
    span_sum = np.zeros(num_nodes, dtype=np.float32)

    for src, tgt in edge_index.t().cpu().tolist():
        min_neighbor[src] = min(min_neighbor[src], tgt)
        min_neighbor[tgt] = min(min_neighbor[tgt], src)
        min_parent[tgt] = min(min_parent[tgt], src)
        min_child[src] = min(min_child[src], tgt)
        out_degree[src] += 1
        in_degree[tgt] += 1
        span = abs(int(layers_np[tgt]) - int(layers_np[src]))
        span_sum[src] += span
        span_sum[tgt] += span
        if span > 1:
            skip_degree[src] += 1
            skip_degree[tgt] += 1

    total_degree = in_degree + out_degree
    mean_span = span_sum / np.maximum(total_degree, 1)
    num_layers = int(layers.max().item()) + 1 if num_nodes > 0 else 0
    layer_counts = np.bincount(layers_np, minlength=num_layers)
    layer_offsets = np.zeros(num_layers + 1, dtype=np.int64)
    layer_offsets[1:] = np.cumsum(layer_counts)
    global_order = np.argsort(layers_np, kind="stable")
    fine_to_coarse = np.empty(num_nodes, dtype=np.int64)
    coarse_base = 0

    for layer_idx in range(num_layers):
        start = int(layer_offsets[layer_idx])
        end = int(layer_offsets[layer_idx + 1])
        n_layer = end - start
        if n_layer == 0:
            continue

        layer_nodes = global_order[start:end]
        layer_degree = total_degree[layer_nodes] + 2 * skip_degree[layer_nodes]
        hub_threshold = max(8, int(np.ceil(np.percentile(layer_degree, 90))))
        if cluster_ids_np is not None:
            cluster_key = np.where(
                cluster_ids_np[layer_nodes] >= 0,
                cluster_ids_np[layer_nodes],
                np.iinfo(np.int64).max,
            )
        else:
            cluster_key = np.full(n_layer, np.iinfo(np.int64).max, dtype=np.int64)

        order = np.lexsort(
            (
                np.rint(mean_span[layer_nodes]).astype(np.int64),
                np.clip(total_degree[layer_nodes], 0, 31),
                min_child[layer_nodes],
                min_parent[layer_nodes],
                min_neighbor[layer_nodes],
                -np.clip(skip_degree[layer_nodes], 0, 31),
                cluster_key,
            )
        )
        ordered_nodes = layer_nodes[order]

        local_group_ids: list[int] = []
        local_group = 0
        i = 0
        while i < n_layer:
            current = int(ordered_nodes[i])
            skip_anchor = skip_degree[current] >= 2 and mean_span[current] > 1.5
            if int(total_degree[current]) >= hub_threshold or skip_anchor:
                local_group_ids.append(local_group)
                local_group += 1
                i += 1
                continue

            group_size = 1
            if i + 1 < n_layer:
                nxt = int(ordered_nodes[i + 1])
                if total_degree[nxt] < hub_threshold:
                    same_cluster = (
                        cluster_ids_np is not None
                        and cluster_ids_np[current] >= 0
                        and cluster_ids_np[current] == cluster_ids_np[nxt]
                    )
                    cluster_compatible = (
                        cluster_ids_np is None
                        or cluster_ids_np[current] < 0
                        or cluster_ids_np[nxt] < 0
                        or cluster_ids_np[current] == cluster_ids_np[nxt]
                    )
                    shares_structure = (
                        min_neighbor[current] == min_neighbor[nxt]
                        or min_parent[current] == min_parent[nxt]
                        or min_child[current] == min_child[nxt]
                    )
                    similar_shape = (
                        abs(int(total_degree[current]) - int(total_degree[nxt])) <= 1
                        and abs(float(mean_span[current]) - float(mean_span[nxt])) <= 1.0
                        and abs(int(skip_degree[current]) - int(skip_degree[nxt])) <= 1
                    )
                    if cluster_compatible and (same_cluster or shares_structure or similar_shape):
                        group_size = 2
                        if i + 2 < n_layer:
                            nxt2 = int(ordered_nodes[i + 2])
                            if total_degree[nxt2] < hub_threshold:
                                third_cluster_compatible = (
                                    cluster_ids_np is None
                                    or cluster_ids_np[nxt] < 0
                                    or cluster_ids_np[nxt2] < 0
                                    or cluster_ids_np[nxt] == cluster_ids_np[nxt2]
                                )
                                third_matches = (
                                    min_parent[nxt] == min_parent[nxt2]
                                    or min_child[nxt] == min_child[nxt2]
                                    or min_neighbor[nxt] == min_neighbor[nxt2]
                                )
                                third_shape = (
                                    abs(int(total_degree[nxt]) - int(total_degree[nxt2])) <= 1
                                    and abs(float(mean_span[nxt]) - float(mean_span[nxt2])) <= 1.0
                                    and abs(int(skip_degree[nxt]) - int(skip_degree[nxt2])) <= 1
                                )
                                if third_cluster_compatible and (third_matches or third_shape):
                                    group_size = 3

            local_group_ids.extend([local_group] * group_size)
            local_group += 1
            i += group_size

        fine_to_coarse[ordered_nodes] = np.asarray(local_group_ids, dtype=np.int64) + coarse_base
        coarse_base += local_group

    return torch.from_numpy(fine_to_coarse)


def test_vectorized_matching_matches_original() -> None:
    """Vectorized matching should produce identical groupings to the old loop."""
    edge_index, num_nodes, node_sizes, layers, cluster_ids = _make_matching_reference_case()

    expected = _reference_fine_to_coarse(edge_index, num_nodes, layers, cluster_ids)
    result = coarsen_once(edge_index, num_nodes, node_sizes, layers, cluster_ids=cluster_ids)

    assert torch.equal(result.fine_to_coarse.cpu(), expected)


def test_cluster_sentinel_handling() -> None:
    """Cluster ``-1`` should remain compatible with positive cluster IDs."""
    edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    node_sizes = torch.full((3, 2), 10.0)
    layers = torch.tensor([0, 0, 1], dtype=torch.long)
    cluster_ids = torch.tensor([5, -1, -1], dtype=torch.long)

    result = coarsen_once(edge_index, 3, node_sizes, layers, cluster_ids=cluster_ids)

    assert result.fine_to_coarse[0].item() == result.fine_to_coarse[1].item()


def test_hub_isolation() -> None:
    """High-degree hubs should always remain singleton coarse groups."""
    src = [0] * 12 + [1, 2]
    tgt = list(range(4, 16)) + [16, 17]
    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    node_sizes = torch.full((18, 2), 10.0)
    layers = torch.tensor([0, 0, 0, 0] + [1] * 14, dtype=torch.long)

    result = coarsen_once(edge_index, 18, node_sizes, layers)

    hub_group = result.fine_to_coarse[0].item()
    assert int((result.fine_to_coarse == hub_group).sum().item()) == 1


def test_edge_partitioning() -> None:
    """Edges should split into tile-local tensors plus a cross-tile residual."""
    edge_index = torch.tensor(
        [
            [0, 1, 2, 4, 1],
            [1, 2, 3, 5, 5],
        ],
        dtype=torch.long,
    )

    tile_edges, cross_edges = _partition_edges_by_tile(edge_index, [0, 3], [3, 6])

    assert len(tile_edges) == 2
    assert torch.equal(tile_edges[0], torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
    assert torch.equal(tile_edges[1], torch.tensor([[1], [2]], dtype=torch.long))
    assert torch.equal(cross_edges, torch.tensor([[2, 1], [3, 5]], dtype=torch.long))


def test_tiled_compute_skips_for_small_graphs() -> None:
    """Tiled GPU mode should stay disabled below the huge-graph threshold."""
    assert not _should_use_tiled_gpu("cuda", "cpu", 1_000, 2_000)
    assert not _should_use_tiled_gpu("cuda", "cuda", 60_000_000, 80_000_000)


def test_tiled_compute_allows_cpu_execution_when_cuda_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large CPU solves should still enable tiled GPU when CUDA can accelerate them."""
    engine_module = importlib.import_module("dagua.layout.engine")

    class _FakeBudget:
        """Return a deterministic VRAM fit decision for tiled GPU tests."""

        def fits(self, needed_bytes: int) -> bool:
            """Report that a full resident GPU layout would exceed budget."""
            del needed_bytes
            return False

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(engine_module, "VRAMBudget", _FakeBudget)
    monkeypatch.setattr(engine_module, "_estimate_gpu_memory", lambda *args, **kwargs: 10)

    assert _should_use_tiled_gpu("cpu", "cpu", 60_000_000, 80_000_000)


def test_layout_inner_initializes_tiled_gpu_before_memory_strategy(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tiled GPU activation should happen before standard memory strategy selection."""
    engine_module = importlib.import_module("dagua.layout.engine")
    tiled_compute_module = importlib.import_module("dagua.layout.tiled_compute")

    state = {"tiled_ready": False, "resolved": False}

    class _FakeTiledGPUCompute:
        """Minimal tiled compute stub for ordering tests."""

        def __init__(
            self,
            num_nodes: int,
            edge_index: torch.Tensor,
            node_sizes: torch.Tensor,
            device: str = "cuda",
            vram_budget: int | None = None,
        ) -> None:
            """Record tiled initialization without requiring a real GPU."""
            del num_nodes, edge_index, node_sizes, device, vram_budget
            state["tiled_ready"] = True
            self.current_edge_index: torch.Tensor | None = None
            self.current_edge_ctx: object | None = None
            self.layer_assignments: torch.Tensor | None = None
            self.cross_tile_loss_mask: list[bool] = []
            self.last_unweighted_loss = 0.0

        def compute_step(
            self,
            pos: torch.Tensor,
            loss_fns: list[object],
            loss_weights: list[float],
        ) -> float:
            """Return a no-op tiled loss value for activation tests."""
            del pos, loss_fns, loss_weights
            self.last_unweighted_loss = 0.0
            return 0.0

    def _fake_resolve_memory_strategy(
        n: int,
        num_edges: int,
        device: str,
        config: LayoutConfig,
        edges_on_cpu: bool = False,
        edge_batch: int = 0,
    ) -> tuple[bool, bool, bool]:
        """Assert tiled GPU setup runs before standard strategy selection."""
        del n, num_edges, device, config, edges_on_cpu, edge_batch
        state["resolved"] = True
        assert state["tiled_ready"] is True
        return False, False, False

    monkeypatch.setattr(engine_module, "_should_use_tiled_gpu", lambda *args, **kwargs: True)
    monkeypatch.setattr(engine_module, "_resolve_memory_strategy", _fake_resolve_memory_strategy)
    monkeypatch.setattr(tiled_compute_module, "TiledGPUCompute", _FakeTiledGPUCompute)

    pos = _layout_inner(
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        3,
        torch.full((3, 2), 20.0),
        LayoutConfig(
            steps=1,
            device="cpu",
            seed=42,
            verbose=False,
            w_repel=0.0,
            w_overlap=0.0,
            w_crossing=0.0,
            w_spacing=0.0,
            w_fanout=0.0,
        ),
        device="cpu",
    )

    assert pos.shape == (3, 2)
    assert state["resolved"] is False
    assert "TILED GPU active: 3 nodes in tiles" in capsys.readouterr().out


def test_multilevel_refinement_syncs_config_device_for_cpu_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU refinement levels should pass a CPU config into the inner engine."""
    engine_module = importlib.import_module("dagua.layout.engine")
    multilevel_module = importlib.import_module("dagua.layout.multilevel")

    g = DaguaGraph()
    g.num_nodes = 4
    g._edge_index_tensor = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    g.node_sizes = torch.full((4, 2), 20.0)

    calls: list[tuple[int, str, str]] = []
    original_tensor_to = torch.Tensor.to
    original_randn = torch.randn

    def _fake_tensor_to(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        """Treat mocked CUDA transfers as no-ops so the test can run on CPU."""
        if args:
            target = args[0]
            if isinstance(target, str) and target == "cuda":
                return self
            if isinstance(target, torch.device) and target.type == "cuda":
                return self
        target_device = kwargs.get("device")
        if isinstance(target_device, str) and target_device == "cuda":
            kwargs = {key: value for key, value in kwargs.items() if key != "device"}
            return original_tensor_to(self, *args, **kwargs)
        if isinstance(target_device, torch.device) and target_device.type == "cuda":
            kwargs = {key: value for key, value in kwargs.items() if key != "device"}
            return original_tensor_to(self, *args, **kwargs)
        return original_tensor_to(self, *args, **kwargs)

    def _fake_randn(*args: object, **kwargs: object) -> torch.Tensor:
        """Allocate mocked CUDA random tensors on CPU for the device-sync test."""
        device = kwargs.get("device")
        if isinstance(device, str) and device == "cuda":
            kwargs = {key: value for key, value in kwargs.items() if key != "device"}
        elif isinstance(device, torch.device) and device.type == "cuda":
            kwargs = {key: value for key, value in kwargs.items() if key != "device"}
        return original_randn(*args, **kwargs)

    def _fake_layout_inner(
        edge_index: torch.Tensor,
        num_nodes: int,
        node_sizes: torch.Tensor,
        config: LayoutConfig,
        device: str = "cpu",
        optimizer_type: str = "adam",
        init_pos: torch.Tensor | None = None,
        clusters: dict | None = None,
        cluster_parents: dict | None = None,
        layer_assignments: torch.Tensor | None = None,
        progress_context: object | None = None,
        trace: object | None = None,
        *,
        graph_structure: object | None = None,
        prebuilt_layer_index: object | None = None,
        skip_classification: bool = False,
    ) -> torch.Tensor:
        """Capture engine device inputs while returning deterministic positions."""
        del (
            edge_index,
            node_sizes,
            optimizer_type,
            init_pos,
            clusters,
            cluster_parents,
            layer_assignments,
            progress_context,
            trace,
            graph_structure,
            prebuilt_layer_index,
            skip_classification,
        )
        calls.append((num_nodes, config.device, device))
        return torch.zeros((num_nodes, 2), dtype=torch.float32)

    monkeypatch.setattr(torch.Tensor, "to", _fake_tensor_to, raising=False)
    monkeypatch.setattr(torch, "randn", _fake_randn)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(
        multilevel_module,
        "classify_graph",
        lambda *args, **kwargs: GraphStructure(
            family=GraphFamily.GENERAL,
            num_components=1,
            max_degree=0,
            num_layers=0,
            avg_layer_width=0.0,
            is_planar_hint=False,
        ),
    )
    monkeypatch.setattr(
        multilevel_module,
        "build_hierarchy",
        lambda *args, **kwargs: [
            multilevel_module.CoarseLevel(
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                node_sizes=torch.full((2, 2), 20.0),
                num_nodes=2,
                fine_to_coarse=torch.tensor([0, 0, 1, 1], dtype=torch.long),
                num_fine=4,
                fine_layer_assignments=torch.tensor([0, 0, 1, 1], dtype=torch.long),
                coarse_layer_assignments=torch.tensor([0, 1], dtype=torch.long),
            )
        ],
    )
    monkeypatch.setattr(multilevel_module, "_can_prolong_on_gpu", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        multilevel_module,
        "_select_refinement_execution",
        lambda *args, **kwargs: (True, False, "adam", 0),
    )
    monkeypatch.setattr(engine_module, "_layout_inner", _fake_layout_inner)
    monkeypatch.setattr(engine_module, "_apply_direction", lambda pos, direction: pos)

    pos = multilevel_module.multilevel_layout(
        g,
        LayoutConfig(
            device="cuda",
            steps=1,
            seed=42,
            verbose=False,
            multilevel_threshold=1,
            multilevel_min_nodes=1,
            multilevel_coarse_steps=1,
            multilevel_refine_steps=1,
            offload_to_disk=False,
        ),
    )

    assert pos.shape == (4, 2)
    assert (2, "cuda", "cuda") in calls
    assert (4, "cpu", "cpu") in calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tiled_compute_produces_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tiled GPU compute should match full-GPU gradients for edge-local losses."""
    tiled_compute_module = importlib.import_module("dagua.layout.tiled_compute")

    torch.manual_seed(0)
    num_nodes = 1000
    src = torch.arange(0, num_nodes - 1, dtype=torch.long)
    edge_index = torch.stack([src, src + 1])
    node_sizes = torch.full((num_nodes, 2), 10.0)
    initial_pos = torch.randn(num_nodes, 2, dtype=torch.float32)

    monkeypatch.setattr(tiled_compute_module, "_compute_tile_size", lambda _n, _vram: 500)

    tiled = tiled_compute_module.TiledGPUCompute(
        num_nodes=num_nodes,
        edge_index=edge_index,
        node_sizes=node_sizes,
    )
    tiled.cross_tile_loss_mask = [True, True]

    def tiled_attract(
        pos: torch.Tensor,
        node_sizes_tile: torch.Tensor,
        layer_index: object,
    ) -> torch.Tensor:
        """Evaluate attraction on the active tiled edge batch."""
        del node_sizes_tile, layer_index
        assert tiled.current_edge_index is not None
        return edge_attraction_loss(
            pos,
            tiled.current_edge_index,
            x_bias=1.5,
            edge_ctx=tiled.current_edge_ctx,
        )

    def tiled_straightness(
        pos: torch.Tensor,
        node_sizes_tile: torch.Tensor,
        layer_index: object,
    ) -> torch.Tensor:
        """Evaluate straightness on the active tiled edge batch."""
        del node_sizes_tile, layer_index
        assert tiled.current_edge_index is not None
        return edge_straightness_loss(
            pos,
            tiled.current_edge_index,
            edge_ctx=tiled.current_edge_ctx,
        )

    tiled_pos = initial_pos.clone().requires_grad_(True)
    tiled_loss = tiled.compute_step(tiled_pos, [tiled_attract, tiled_straightness], [1.0, 0.5])
    assert tiled_pos.grad is not None
    tiled_grad = tiled_pos.grad.detach().clone()

    full_pos = initial_pos.to("cuda").requires_grad_(True)
    full_loss = edge_attraction_loss(full_pos, edge_index.to("cuda"), x_bias=1.5)
    full_loss = full_loss + 0.5 * edge_straightness_loss(full_pos, edge_index.to("cuda"))
    full_loss.backward()
    assert full_pos.grad is not None

    assert tiled_loss == pytest.approx(full_loss.item(), rel=1e-5, abs=1e-5)
    assert torch.allclose(tiled_grad, full_pos.grad.cpu(), atol=1e-4, rtol=1e-4)

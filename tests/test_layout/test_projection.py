"""Tests for hard overlap projection."""

import importlib

import pytest
import torch

from dagua.layout.layers import build_layer_index
from dagua.layout.projection import project_overlaps
from dagua.metrics import count_overlaps


def _make_layered_sweep_projection_case(
    num_nodes: int = 520,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a layered overlap case that forces the sweep projector.

    Parameters
    ----------
    num_nodes : int, default=520
        Number of nodes to include. Values above ``500`` force the sweep path.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        CPU position tensor shaped ``[N, 2]``, node sizes shaped ``[N, 2]``,
        and layer IDs shaped ``[N]``.
    """
    x_positions = torch.linspace(0.0, 300.0, steps=num_nodes, dtype=torch.float32)
    y_positions = torch.zeros(num_nodes, dtype=torch.float32)
    pos = torch.stack([x_positions, y_positions], dim=1)
    node_sizes = torch.full((num_nodes, 2), 20.0, dtype=torch.float32)
    layers = torch.zeros(num_nodes, dtype=torch.long)
    return pos, node_sizes, layers


class TestProjectOverlaps:
    def test_resolves_overlaps(self):
        # Create overlapping nodes
        pos = torch.tensor([[0.0, 0.0], [10.0, 5.0], [5.0, 10.0]])
        ns = torch.tensor([[40.0, 20.0]] * 3)
        projected = project_overlaps(pos, ns, iterations=20)
        overlaps = count_overlaps(projected, ns)
        assert overlaps == 0

    def test_preserves_non_overlapping(self):
        # Nodes already well-separated
        pos = torch.tensor([[0.0, 0.0], [200.0, 0.0], [0.0, 200.0]])
        ns = torch.tensor([[40.0, 20.0]] * 3)
        projected = project_overlaps(pos, ns, iterations=10)
        # Should not move much
        assert torch.allclose(pos, projected, atol=1.0)

    def test_single_node(self):
        pos = torch.tensor([[50.0, 50.0]])
        ns = torch.tensor([[40.0, 20.0]])
        projected = project_overlaps(pos, ns)
        assert torch.allclose(pos, projected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_layered_gpu_projection_matches_cpu(self):
        """CUDA layered projection should stay close to the CPU sweep result."""
        pos, ns, layers = _make_layered_sweep_projection_case()
        layer_index = build_layer_index(layers)

        cpu_projected = project_overlaps(
            pos.clone(),
            ns.clone(),
            padding=2.0,
            iterations=8,
            layer_index=layer_index,
        )
        gpu_projected = project_overlaps(
            pos.clone().to("cuda"),
            ns.clone().to("cuda"),
            padding=2.0,
            iterations=8,
            layer_index=build_layer_index(layer_index.node_to_layer.to("cuda"), device="cuda"),
        ).cpu()

        torch.testing.assert_close(gpu_projected, cpu_projected, atol=0.1, rtol=0.0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_projection_logs_cpu_fallback_on_low_vram(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Low-VRAM CUDA projection should fall back to CPU and preserve output."""
        pos, ns, layers = _make_layered_sweep_projection_case()
        layer_index = build_layer_index(layers)
        projection_module = importlib.import_module("dagua.layout.projection")

        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1, 2))

        cpu_projected = projection_module.project_overlaps(
            pos.clone(),
            ns.clone(),
            padding=2.0,
            iterations=8,
            layer_index=layer_index,
        )
        gpu_projected = projection_module.project_overlaps(
            pos.clone().to("cuda"),
            ns.clone().to("cuda"),
            padding=2.0,
            iterations=8,
            layer_index=build_layer_index(layers.to("cuda"), device="cuda"),
        ).cpu()

        torch.testing.assert_close(gpu_projected, cpu_projected, atol=0.1, rtol=0.0)
        assert "[dagua]   Overlap projection: CPU fallback (need " in capsys.readouterr().out

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_projection_logs_oom_cpu_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """CUDA projection OOM should retry on CPU and preserve output."""
        pos, ns, layers = _make_layered_sweep_projection_case()
        projection_module = importlib.import_module("dagua.layout.projection")
        original_run_projection_impl = projection_module._run_projection_impl

        def _oom_once_then_delegate(
            positions: torch.Tensor,
            node_sizes: torch.Tensor,
            padding: float,
            iterations: int,
            layer_index: object,
            convergent: bool = False,
        ) -> None:
            """Raise CUDA OOM only for the initial GPU execution attempt."""
            if positions.device.type == "cuda":
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            original_run_projection_impl(
                positions,
                node_sizes,
                padding,
                iterations,
                layer_index,
                convergent=convergent,
            )

        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (10_000_000_000, 20_000_000_000))
        monkeypatch.setattr(projection_module, "_run_projection_impl", _oom_once_then_delegate)

        cpu_projected = project_overlaps(
            pos.clone(),
            ns.clone(),
            padding=2.0,
            iterations=8,
            layer_index=build_layer_index(layers),
        )
        gpu_projected = project_overlaps(
            pos.clone().to("cuda"),
            ns.clone().to("cuda"),
            padding=2.0,
            iterations=8,
            layer_index=build_layer_index(layers.to("cuda"), device="cuda"),
        ).cpu()

        torch.testing.assert_close(gpu_projected, cpu_projected, atol=0.1, rtol=0.0)
        out = capsys.readouterr().out
        assert "[dagua]   Overlap projection: CUDA (" in out
        assert "[dagua]   Overlap projection: OOM, CPU fallback" in out

    def test_projection_cpu_fallback_without_cuda(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Projection should keep working on CPU when CUDA is unavailable."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        pos = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float32)
        ns = torch.tensor([[20.0, 10.0], [20.0, 10.0]], dtype=torch.float32)
        layer_index = build_layer_index(torch.tensor([0, 0], dtype=torch.long))

        projected = project_overlaps(pos, ns, iterations=4, layer_index=layer_index)

        assert projected.device.type == "cpu"
        assert count_overlaps(projected, ns) == 0


def _make_dense_overlap_clique(
    num_nodes: int = 30,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a dense overlap clique: nodes nearly coincident, real label boxes.

    Parameters
    ----------
    num_nodes : int, default=30
        Number of nodes to place near-coincident.
    seed : int, default=0
        Deterministic RNG seed for the initial jitter.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Position tensor shaped ``[N, 2]`` (all nodes clustered within a
        small radius of the origin) and a label-size node-size tensor
        shaped ``[N, 2]`` (60x20, wider than the initial cluster radius so
        every pair starts overlapping).
    """
    generator = torch.Generator().manual_seed(seed)
    pos = torch.randn(num_nodes, 2, generator=generator) * 0.5
    node_sizes = torch.full((num_nodes, 2), 60.0)
    node_sizes[:, 1] = 20.0
    return pos, node_sizes


class TestExactProjectorConvergence:
    """Convergence proof for the opt-in convergent exact projector (r80-S2b).

    The legacy `_project_exact` per-pass update uses advanced-index `+=`,
    which silently drops repeated node indices (last write wins), so dense
    overlap cliques never converge -- see
    .project-context/research/r79_native/P3B2_STRESS_FORENSICS.md (sbm_4x30
    left 37+ overlaps after 50 iterations). ``convergent=True`` opts into
    the accumulate-then-damp projector that provably reaches zero overlaps
    on an adversarial 30-node clique where every node starts overlapping
    every other node. The DEFAULT stays legacy: the r80-S2 sweep showed the
    convergent trajectory regresses some default-path graphs, so it is
    exposed only to referee-protected callers (portfolio challengers).
    """

    def test_dense_clique_converges_to_zero_overlaps(self) -> None:
        """A 30-node near-coincident clique should fully resolve within budget."""
        pos, node_sizes = _make_dense_overlap_clique(num_nodes=30, seed=0)
        initial_overlaps = count_overlaps(pos, node_sizes)
        assert initial_overlaps > 400  # sanity: this really is a dense clique

        projected = project_overlaps(
            pos.clone(), node_sizes, padding=2.0, iterations=200, convergent=True
        )

        final_overlaps = count_overlaps(projected, node_sizes)
        assert final_overlaps == 0, (
            f"convergent exact projector left {final_overlaps} overlaps unresolved "
            f"(started from {initial_overlaps})"
        )

    def test_dense_clique_converges_across_seeds(self) -> None:
        """Convergence should not depend on the particular initial jitter."""
        for seed in range(10):
            pos, node_sizes = _make_dense_overlap_clique(num_nodes=30, seed=seed)
            projected = project_overlaps(
                pos.clone(), node_sizes, padding=2.0, iterations=200, convergent=True
            )
            final_overlaps = count_overlaps(projected, node_sizes)
            assert final_overlaps == 0, f"seed={seed} left {final_overlaps} overlaps unresolved"

    def test_default_path_preserves_legacy_trajectory(self) -> None:
        """Default (non-convergent) projection must match pre-r80 behavior.

        Regression pin for the r80-S2b revert: the default exact path must
        keep the legacy last-write-wins trajectory bit-for-bit (the r79
        benchmark baselines and every default pipeline's tuning assume it),
        including its known inability to fully resolve dense cliques.
        """
        pos, node_sizes = _make_dense_overlap_clique(num_nodes=30, seed=0)

        default_projected = project_overlaps(pos.clone(), node_sizes, padding=2.0, iterations=50)
        # The legacy trajectory stalls on the dense clique (last-write-wins
        # drops most of the accumulated push) -- residual overlaps expected.
        assert count_overlaps(default_projected, node_sizes) > 0

        from dagua.layout.projection import _project_exact_legacy

        legacy = pos.clone()
        with torch.no_grad():
            _project_exact_legacy(legacy, node_sizes, 2.0, 50)
        torch.testing.assert_close(default_projected, legacy, rtol=0.0, atol=0.0)

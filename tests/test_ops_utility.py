"""Tests for utility ops."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from dagua.layout.ops.postprocess import CenterPositions
from dagua.layout.ops.state import (
    ExecutionPlan,
    HierarchyLevel,
    LayoutProblem,
    MemoryPolicy,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.utility import (
    Checkpoint,
    CheckpointConfig,
    DiskOffload,
    DiskReload,
    GarbageCollect,
    ProgressReport,
    ProgressReportConfig,
    Timer,
    VRAMGuard,
)


def _make_problem() -> LayoutProblem:
    """Create a small problem for utility-op tests.

    Returns
    -------
    LayoutProblem
        Minimal graph problem with three nodes.
    """
    return LayoutProblem(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        num_nodes=3,
        node_sizes=torch.ones((3, 2), dtype=torch.float32),
    )


def _make_hierarchy_level() -> HierarchyLevel:
    """Create a hierarchy level with resident tensors for round-trip tests.

    Returns
    -------
    HierarchyLevel
        Single hierarchy level with several tensor fields populated.
    """
    return HierarchyLevel(
        num_nodes=2,
        num_fine=3,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        node_sizes=torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
        fine_to_coarse=torch.tensor([0, 1, 1], dtype=torch.long),
        fine_layer_assignments=torch.tensor([0, 1, 1], dtype=torch.long),
        coarse_layer_assignments=torch.tensor([0, 1], dtype=torch.long),
        cluster_ids=torch.tensor([0, 0], dtype=torch.long),
    )


def test_checkpoint_and_disk_reload_round_trip_preserves_state(tmp_path: Path) -> None:
    """Checkpoint plus DiskReload should preserve an offloaded solve state."""

    problem = _make_problem()
    original_level = _make_hierarchy_level()
    expected_edge_index = original_level.edge_index.clone()
    expected_node_sizes = original_level.node_sizes.clone()
    expected_fine_to_coarse = original_level.fine_to_coarse.clone()
    expected_fine_layer_assignments = original_level.fine_layer_assignments.clone()
    expected_coarse_layer_assignments = original_level.coarse_layer_assignments.clone()
    expected_cluster_ids = original_level.cluster_ids.clone()
    original_pos = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)
    state = SolveState(
        pos=original_pos.clone(),
        hierarchy=[original_level],
        extras={"marker": "keep-me"},
    )
    ctx = RuntimeContext(
        memory=MemoryPolicy(
            offload_dir=tmp_path / "offload",
            checkpoint_dir=tmp_path / "checkpoints",
        )
    )

    DiskOffload().apply(problem, state, ctx)
    assert state.hierarchy is not None
    assert state.hierarchy[0].edge_index is None
    assert state.hierarchy[0].offload_path is not None

    checkpoint_path = tmp_path / "checkpoint.pt"
    Checkpoint(CheckpointConfig(path=checkpoint_path)).apply(problem, state, ctx)
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert isinstance(restored, SolveState)
    assert restored.hierarchy is not None
    assert restored.hierarchy[0].edge_index is None

    DiskReload().apply(problem, restored, ctx)

    torch.testing.assert_close(restored.pos, original_pos)
    assert restored.extras["marker"] == "keep-me"
    torch.testing.assert_close(restored.hierarchy[0].edge_index, expected_edge_index)
    torch.testing.assert_close(restored.hierarchy[0].node_sizes, expected_node_sizes)
    torch.testing.assert_close(restored.hierarchy[0].fine_to_coarse, expected_fine_to_coarse)
    torch.testing.assert_close(
        restored.hierarchy[0].fine_layer_assignments,
        expected_fine_layer_assignments,
    )
    torch.testing.assert_close(
        restored.hierarchy[0].coarse_layer_assignments,
        expected_coarse_layer_assignments,
    )
    torch.testing.assert_close(restored.hierarchy[0].cluster_ids, expected_cluster_ids)


def test_garbage_collect_runs_without_error() -> None:
    """GarbageCollect should complete on CPU-only test runs."""

    problem = _make_problem()
    state = SolveState()

    result = GarbageCollect().apply(problem, state, RuntimeContext())

    assert "gc_stats" in result.extras
    assert "collected" in result.extras["gc_stats"]


def test_vram_guard_on_cpu_always_passes() -> None:
    """VRAMGuard should be a no-op when the runtime plan targets CPU."""

    problem = _make_problem()
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))

    result = VRAMGuard().apply(problem, state, ctx)

    assert result is state
    assert result.extras["vram_guard"]["checked"] is False


def test_progress_report_writes_progress_json(tmp_path: Path) -> None:
    """ProgressReport should emit a compact JSON payload on cadence."""

    problem = _make_problem()
    progress_path = tmp_path / "progress.json"
    state = SolveState(step=10, total_steps=20, prev_loss=1.25)

    ProgressReport(ProgressReportConfig(file=progress_path, interval=5)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["step"] == 10
    assert payload["total_steps"] == 20
    assert payload["loss"] == 1.25
    assert payload["num_nodes"] == 3


def test_timer_records_wrapped_op_duration() -> None:
    """Timer should time and execute the wrapped op."""

    problem = _make_problem()
    state = SolveState(pos=torch.tensor([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]], dtype=torch.float32))

    result = Timer(op=CenterPositions(), label="center").apply(problem, state, RuntimeContext())

    assert result.pos is not None
    torch.testing.assert_close(result.pos.mean(dim=0), torch.zeros(2), atol=1.0e-6, rtol=0.0)
    assert "center" in result.extras["op_timings"]
    assert result.extras["last_timing_seconds"] >= 0.0

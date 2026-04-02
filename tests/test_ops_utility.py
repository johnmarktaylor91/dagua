"""Tests for utility ops."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from dagua.layout.ops import loss_classic as loss_classic_ops
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.loss_classic import CyclicSampler, CyclicSamplerConfig
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
    VRAMGuardConfig,
)


class _SleepOp(Op):
    """Test helper that sleeps briefly and records its execution."""

    name = "sleep_op"

    def __init__(self, delay_seconds: float) -> None:
        """Store the requested delay.

        Parameters
        ----------
        delay_seconds : float
            Sleep duration used to create a measurable timing interval.

        Returns
        -------
        None
            The helper stores the configured delay.
        """
        self.delay_seconds = delay_seconds

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Sleep for the configured duration and record that the op ran.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this helper.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this helper.

        Returns
        -------
        SolveState
            The same state, annotated in ``extras``.
        """
        del problem, ctx
        time.sleep(self.delay_seconds)
        state.extras["slept"] = True
        return state


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


def _make_state_with_hierarchy() -> SolveState:
    """Create a solve state with a populated hierarchy level.

    Returns
    -------
    SolveState
        State suitable for checkpoint and offload tests.
    """
    return SolveState(
        pos=torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
        hierarchy=[_make_hierarchy_level()],
        extras={"marker": "keep-me"},
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


def test_checkpoint_round_trip_preserves_pos(tmp_path: Path) -> None:
    """Checkpoint should serialize and restore the position tensor."""
    problem = _make_problem()
    state = _make_state_with_hierarchy()
    checkpoint_path = tmp_path / "round_trip.pt"

    Checkpoint(CheckpointConfig(path=checkpoint_path)).apply(problem, state, RuntimeContext())
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert isinstance(restored, SolveState)
    torch.testing.assert_close(restored.pos, state.pos)


def test_checkpoint_round_trip_preserves_extras(tmp_path: Path) -> None:
    """Checkpoint should keep arbitrary extras payloads intact."""
    problem = _make_problem()
    state = _make_state_with_hierarchy()
    checkpoint_path = tmp_path / "extras.pt"

    Checkpoint(CheckpointConfig(path=checkpoint_path)).apply(problem, state, RuntimeContext())
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert restored.extras["marker"] == "keep-me"


def test_checkpoint_round_trip_preserves_none_fields(tmp_path: Path) -> None:
    """Checkpoint should preserve unset optional fields."""
    problem = _make_problem()
    state = SolveState(pos=None, hierarchy=None, adjacency=None)
    checkpoint_path = tmp_path / "none_fields.pt"

    Checkpoint(CheckpointConfig(path=checkpoint_path)).apply(problem, state, RuntimeContext())
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert restored.pos is None
    assert restored.hierarchy is None
    assert restored.adjacency is None


def test_checkpoint_records_resolved_path_in_extras(tmp_path: Path) -> None:
    """Checkpoint should record the saved path in state.extras."""
    problem = _make_problem()
    state = SolveState()
    checkpoint_path = tmp_path / "recorded.pt"

    result = Checkpoint(CheckpointConfig(path=checkpoint_path)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.extras["checkpoint_path"] == str(checkpoint_path.resolve())


def test_checkpoint_cleans_up_temporary_files(tmp_path: Path) -> None:
    """Checkpoint should not leave temporary save files behind."""
    problem = _make_problem()
    checkpoint_path = tmp_path / "cleanup.pt"

    Checkpoint(CheckpointConfig(path=checkpoint_path)).apply(
        problem,
        SolveState(),
        RuntimeContext(),
    )

    assert list(tmp_path.glob(".*.tmp")) == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_disk_offload_clears_hierarchy_tensors_and_writes_payload(tmp_path: Path) -> None:
    """DiskOffload should serialize resident hierarchy tensors and clear them from memory."""
    problem = _make_problem()
    state = _make_state_with_hierarchy()
    ctx = RuntimeContext(memory=MemoryPolicy(offload_dir=tmp_path / "offload"))

    result = DiskOffload().apply(problem, state, ctx)

    assert result.hierarchy is not None
    level = result.hierarchy[0]
    assert level.edge_index is None
    assert level.node_sizes is None
    assert level.fine_to_coarse is None
    assert level.offload_path is not None
    assert level.offload_path.exists()


def test_disk_reload_restores_offloaded_tensors(tmp_path: Path) -> None:
    """DiskReload should repopulate tensor fields from the offload payload."""
    problem = _make_problem()
    state = _make_state_with_hierarchy()
    ctx = RuntimeContext(memory=MemoryPolicy(offload_dir=tmp_path / "offload"))
    expected_level = _make_hierarchy_level()

    DiskOffload().apply(problem, state, ctx)
    reloaded = DiskReload().apply(problem, state, ctx)

    assert reloaded.hierarchy is not None
    torch.testing.assert_close(reloaded.hierarchy[0].edge_index, expected_level.edge_index)
    torch.testing.assert_close(reloaded.hierarchy[0].node_sizes, expected_level.node_sizes)
    torch.testing.assert_close(reloaded.hierarchy[0].fine_to_coarse, expected_level.fine_to_coarse)


def test_disk_offload_and_reload_round_trip_preserves_values(tmp_path: Path) -> None:
    """DiskOffload followed by DiskReload should preserve all serialized values."""
    problem = _make_problem()
    state = _make_state_with_hierarchy()
    ctx = RuntimeContext(memory=MemoryPolicy(offload_dir=tmp_path / "offload"))
    expected_level = _make_hierarchy_level()

    DiskOffload().apply(problem, state, ctx)
    DiskReload().apply(problem, state, ctx)

    assert state.hierarchy is not None
    torch.testing.assert_close(state.pos, torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    torch.testing.assert_close(
        state.hierarchy[0].coarse_layer_assignments,
        expected_level.coarse_layer_assignments,
    )
    torch.testing.assert_close(state.hierarchy[0].cluster_ids, expected_level.cluster_ids)


def test_disk_offload_is_no_op_when_hierarchy_is_missing(tmp_path: Path) -> None:
    """DiskOffload should leave the state unchanged when no hierarchy exists."""
    problem = _make_problem()
    state = SolveState()
    ctx = RuntimeContext(memory=MemoryPolicy(offload_dir=tmp_path / "offload"))

    result = DiskOffload().apply(problem, state, ctx)

    assert result is state
    assert "offload_dir" not in result.extras


def test_disk_reload_is_no_op_when_levels_are_not_offloaded() -> None:
    """DiskReload should succeed even when no offload payloads are present."""
    problem = _make_problem()
    state = _make_state_with_hierarchy()

    result = DiskReload().apply(problem, state, RuntimeContext())

    assert result is state
    assert result.hierarchy is not None
    assert result.hierarchy[0].edge_index is not None


def test_garbage_collect_preserves_positions_and_hierarchy() -> None:
    """GarbageCollect should not mutate resident state tensors."""
    problem = _make_problem()
    state = _make_state_with_hierarchy()
    original_pos = state.pos.clone()
    original_mapping = state.hierarchy[0].fine_to_coarse.clone()

    result = GarbageCollect().apply(problem, state, RuntimeContext())

    torch.testing.assert_close(result.pos, original_pos)
    assert result.hierarchy is not None
    torch.testing.assert_close(result.hierarchy[0].fine_to_coarse, original_mapping)


def test_vram_guard_records_budget_fraction_on_cpu() -> None:
    """VRAMGuard should record the configured budget even on CPU-only runs."""
    problem = _make_problem()
    state = SolveState()

    result = VRAMGuard(VRAMGuardConfig(budget_fraction=0.5)).apply(
        problem,
        state,
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )

    assert result.extras["vram_guard"]["checked"] is False
    assert result.extras["vram_guard"]["budget_fraction"] == 0.5


def test_vram_guard_rejects_invalid_budget_fraction() -> None:
    """VRAMGuard should reject out-of-range budget fractions."""
    with pytest.raises(ValueError, match="budget_fraction"):
        VRAMGuard(VRAMGuardConfig(budget_fraction=0.0)).apply(
            _make_problem(),
            SolveState(),
            RuntimeContext(plan=ExecutionPlan(device="cpu")),
        )


def test_vram_guard_simulated_gpu_passes_when_usage_is_within_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VRAMGuard should pass when simulated GPU usage is below the configured budget."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (700, 1000))

    result = VRAMGuard(VRAMGuardConfig(budget_fraction=0.5)).apply(
        _make_problem(),
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cuda")),
    )

    assert result.extras["vram_guard"]["checked"] is True
    assert result.extras["vram_guard"]["budget_bytes"] == 500
    assert result.extras["vram_guard"]["used_bytes"] == 300


def test_vram_guard_simulated_gpu_raises_when_usage_exceeds_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VRAMGuard should raise MemoryError when simulated GPU usage is too high."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (100, 1000))

    with pytest.raises(MemoryError, match="exceeds"):
        VRAMGuard(VRAMGuardConfig(budget_fraction=0.5)).apply(
            _make_problem(),
            SolveState(),
            RuntimeContext(plan=ExecutionPlan(device="cuda")),
        )


def test_progress_report_skips_non_matching_interval(tmp_path: Path) -> None:
    """ProgressReport should not write a file when the cadence does not match."""
    progress_path = tmp_path / "progress.json"
    state = SolveState(step=3, total_steps=20, prev_loss=1.0)

    ProgressReport(ProgressReportConfig(file=progress_path, interval=5)).apply(
        _make_problem(),
        state,
        RuntimeContext(),
    )

    assert not progress_path.exists()


def test_progress_report_payload_is_valid_json(tmp_path: Path) -> None:
    """ProgressReport should write parseable JSON."""
    progress_path = tmp_path / "progress.json"

    ProgressReport(ProgressReportConfig(file=progress_path, interval=1)).apply(
        _make_problem(),
        SolveState(step=1, total_steps=10, prev_loss=2.5),
        RuntimeContext(),
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload["step"] == 1


def test_progress_report_writes_when_converged_even_off_interval(tmp_path: Path) -> None:
    """ProgressReport should always write once the state is converged."""
    progress_path = tmp_path / "progress.json"

    ProgressReport(ProgressReportConfig(file=progress_path, interval=10)).apply(
        _make_problem(),
        SolveState(step=3, total_steps=20, prev_loss=0.5, converged=True),
        RuntimeContext(),
    )

    assert progress_path.exists()


def test_progress_report_writes_on_final_step_even_off_interval(tmp_path: Path) -> None:
    """ProgressReport should write on the final solver step even if cadence does not match."""
    progress_path = tmp_path / "progress.json"

    ProgressReport(ProgressReportConfig(file=progress_path, interval=10)).apply(
        _make_problem(),
        SolveState(step=20, total_steps=20, prev_loss=0.25),
        RuntimeContext(),
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["step_pct"] == 1.0


def test_timer_measures_nonzero_elapsed_time() -> None:
    """Timer should record a measurable duration for a sleeping op."""
    result = Timer(op=_SleepOp(0.01), label="sleep").apply(
        _make_problem(),
        SolveState(),
        RuntimeContext(),
    )

    assert result.extras["last_timing_seconds"] > 0.0
    assert result.extras["op_timings"]["sleep"][0] > 0.0


def test_timer_noop_leaves_state_unchanged() -> None:
    """Timer without a wrapped op should not mutate positions."""
    original = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    state = SolveState(pos=original.clone())

    result = Timer().apply(_make_problem(), state, RuntimeContext())

    torch.testing.assert_close(result.pos, original)


def test_timer_works_inside_pipeline() -> None:
    """Timer should compose cleanly inside Pipeline."""
    state = SolveState()
    pipeline = Pipeline([Timer(op=_SleepOp(0.001), label="pipeline_sleep")])

    result = pipeline.apply(_make_problem(), state, RuntimeContext())

    assert result.extras["slept"] is True
    assert "pipeline_sleep" in result.extras["op_timings"]
    assert result.ops_applied[-1] == "timer"


def test_cyclic_sampler_creates_sampler_in_extras() -> None:
    """CyclicSampler should populate the sampler store in state.extras."""
    state = SolveState(extras={"sgd2_active_criterion": "ideal_edge_length"})

    result = CyclicSampler(CyclicSamplerConfig(pool_size=4)).apply(
        _make_problem(),
        state,
        RuntimeContext(),
    )

    assert "sgd2_samplers" in result.extras
    assert "ideal_edge_length" in result.extras["sgd2_samplers"]


def test_cyclic_sampler_explicit_pool_size_is_respected() -> None:
    """CyclicSampler should use the configured explicit pool size."""
    state = SolveState(extras={"sgd2_active_criterion": "ideal_edge_length"})

    result = CyclicSampler(CyclicSamplerConfig(pool_size=4)).apply(
        _make_problem(),
        state,
        RuntimeContext(),
    )
    sampler = result.extras["sgd2_samplers"]["ideal_edge_length"]

    assert sampler.sample(10).numel() == 4


def test_cyclic_sampler_samples_a_full_permutation_per_epoch() -> None:
    """CyclicSampler should visit every index exactly once within one epoch."""
    state = SolveState(extras={"sgd2_active_criterion": "ideal_edge_length"})
    result = CyclicSampler(CyclicSamplerConfig(pool_size=4)).apply(
        _make_problem(),
        state,
        RuntimeContext(),
    )
    sampler = result.extras["sgd2_samplers"]["ideal_edge_length"]

    first_epoch = sampler.sample(4)

    assert sorted(first_epoch.tolist()) == [0, 1, 2, 3]


def test_cyclic_sampler_infers_pool_size_from_active_criterion() -> None:
    """CyclicSampler should infer its pool size when configured with zero."""
    state = SolveState(extras={"sgd2_active_criterion": "ideal_edge_length"})

    result = CyclicSampler(CyclicSamplerConfig(pool_size=0)).apply(
        _make_problem(),
        state,
        RuntimeContext(),
    )
    sampler = result.extras["sgd2_samplers"]["ideal_edge_length"]

    assert sampler.sample(10).numel() == _make_problem().edge_index.shape[1]


def test_cyclic_sampler_reshuffles_when_epoch_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CyclicSampler should draw a fresh permutation after consuming a full epoch."""
    permutations = [
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        torch.tensor([3, 2, 1, 0], dtype=torch.long),
    ]

    def _fake_randperm(total: int, device: Any | None = None) -> torch.Tensor:
        """Return deterministic permutations for sampler creation and reshuffle."""
        del device
        return permutations.pop(0).clone()

    monkeypatch.setattr(loss_classic_ops._sgd2.torch, "randperm", _fake_randperm)
    state = SolveState(extras={"sgd2_active_criterion": "ideal_edge_length"})
    result = CyclicSampler(CyclicSamplerConfig(pool_size=4)).apply(
        _make_problem(),
        state,
        RuntimeContext(),
    )
    sampler = result.extras["sgd2_samplers"]["ideal_edge_length"]

    first_epoch = sampler.sample(4)
    second_epoch = sampler.sample(4)

    assert first_epoch.tolist() == [0, 1, 2, 3]
    assert second_epoch.tolist() == [3, 2, 1, 0]

"""Tests for the composable layout operations foundation."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import get_type_hints

import pytest
import torch

from dagua.layout.ops import (
    AnnealingSchedule,
    Conditional,
    ExecutionPlan,
    FlexConstraints,
    GraphStructure,
    HierarchyLevel,
    LayoutProblem,
    Op,
    Pipeline,
    Repeat,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.state import NullTraceSink


class _DoublePositions(Op):
    """Test op that doubles positions in place."""

    name = "double_positions"
    category = "refinement"
    reads = ("pos",)
    writes = ("pos",)
    requires = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Double the position tensor.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context.

        Returns
        -------
        SolveState
            Updated state with doubled positions.
        """
        assert state.pos is not None
        state.pos = state.pos * 2.0
        return state


class _SetLayers(Op):
    """Test op that populates layer assignments from problem size."""

    name = "set_layers"
    category = "structural"
    writes = ("layers",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Set a simple zero layer assignment.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context.

        Returns
        -------
        SolveState
            Updated state with layer assignments.
        """
        state.layers = torch.zeros(problem.num_nodes, dtype=torch.long)
        return state


class _RequiresLayers(Op):
    """Test op that documents a layer precondition."""

    name = "requires_layers"
    category = "metric"
    requires = ("layers",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Return the state unchanged.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context.

        Returns
        -------
        SolveState
            The original state.
        """
        return state


def _make_problem() -> LayoutProblem:
    """Create a minimal layout problem for tests."""

    return LayoutProblem(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        num_nodes=3,
    )


def test_layout_problem_supports_minimal_and_full_fields() -> None:
    """Verify `LayoutProblem` accepts minimal and fully populated inputs."""

    minimal = _make_problem()
    structure = GraphStructure(
        family="tree",
        num_nodes=3,
        num_edges=2,
        max_degree=2,
        depth=2,
        max_layer_width=2,
        num_layers=3,
        avg_layer_width=1.0,
        is_dag=True,
        num_components=1,
    )
    flex = FlexConstraints(
        pin_indices=torch.tensor([0], dtype=torch.long),
        pin_targets=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        pin_weights=torch.tensor([1.0], dtype=torch.float32),
        soft_pin_mask=torch.tensor([True, False, False]),
        hard_pin_mask=torch.tensor([False, False, True]),
        align_groups=[["a", "b"]],
        flex_node_sep=10.0,
        flex_node_sep_weight=2.5,
    )
    full = LayoutProblem(
        edge_index=minimal.edge_index,
        num_nodes=3,
        node_sizes=torch.ones((3, 2), dtype=torch.float32),
        node_labels=["a", "b", "c"],
        direction="TB",
        clusters={"cluster_a": [0, 1]},
        cluster_parents={"cluster_a": "root"},
        structure=structure,
        flex=flex,
        seed=7,
    )

    assert minimal.direction == "BT"
    assert minimal.node_sizes is None
    assert full.structure == structure
    assert full.flex == flex
    assert full.seed == 7


def test_solve_state_allows_missing_positions_for_hierarchy_build() -> None:
    """Verify `SolveState` allows `pos=None` before position initialization."""

    state = SolveState(pos=None, total_steps=25)

    assert state.pos is None
    assert state.total_steps == 25
    assert state.ops_applied == []


def test_hierarchy_level_supports_nullable_payload_fields() -> None:
    """Verify `HierarchyLevel` can represent an offloaded level."""

    level = HierarchyLevel(
        num_nodes=10,
        num_fine=20,
        offload_path=Path("/tmp/level.pt"),
        offload_dir=Path("/tmp"),
        payload_fields=("edge_index", "node_sizes"),
    )

    assert level.edge_index is None
    assert level.node_sizes is None
    assert level.offload_path == Path("/tmp/level.pt")
    assert level.payload_fields == ("edge_index", "node_sizes")


def test_graph_structure_is_frozen() -> None:
    """Verify `GraphStructure` is immutable."""

    structure = GraphStructure(
        family="chain",
        num_nodes=2,
        num_edges=1,
        max_degree=1,
        depth=1,
        max_layer_width=1,
    )

    with pytest.raises(FrozenInstanceError):
        structure.family = "tree"


def test_runtime_context_uses_null_trace_sink_by_default() -> None:
    """Verify `RuntimeContext` defaults to the null trace sink."""

    ctx = RuntimeContext()

    assert isinstance(ctx.trace_sink, NullTraceSink)
    ctx.trace_sink.snapshot(torch.zeros((1, 2)), 0)
    ctx.trace_sink.log("ignored")


def test_execution_plan_supports_multi_device_fields() -> None:
    """Verify execution planning can separate compute and edge devices."""

    plan = ExecutionPlan(device="cuda:0", edge_device="cpu", edges_on_cpu=True)

    assert plan.device == "cuda:0"
    assert plan.edge_device == "cpu"
    assert plan.edges_on_cpu is True


def test_op_apply_signature_uses_three_part_state() -> None:
    """Verify ops expose the expected three-argument `apply` signature."""

    signature = inspect.signature(_DoublePositions().apply)
    hints = get_type_hints(_DoublePositions.apply)

    assert list(signature.parameters) == ["problem", "state", "ctx"]
    assert hints["return"] is SolveState


def test_pipeline_runs_ops_in_sequence_and_tracks_history() -> None:
    """Verify `Pipeline` runs ops in order and records op names."""

    problem = _make_problem()
    state = SolveState(pos=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    pipeline = Pipeline([_DoublePositions(), _SetLayers()], name="sequence")

    result = pipeline(problem, state, RuntimeContext())

    assert torch.equal(
        result.pos,
        torch.tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]),
    )
    assert torch.equal(result.layers, torch.zeros(3, dtype=torch.long))
    assert result.ops_applied == ["double_positions", "set_layers"]


def test_pipeline_lint_reports_missing_best_effort_preconditions() -> None:
    """Verify `Pipeline.lint()` warns about missing documented requirements."""

    pipeline = Pipeline([_RequiresLayers()])
    warnings = pipeline.lint(SolveState())

    assert warnings == ["requires_layers expects 'layers' to be set"]


def test_pipeline_bounds_ops_history_to_one_hundred_entries() -> None:
    """Verify `Pipeline` keeps a bounded operation history."""

    problem = _make_problem()
    state = SolveState(pos=torch.ones((3, 2), dtype=torch.float32))
    pipeline = Pipeline([_DoublePositions()] * 120)

    result = pipeline(problem, state, RuntimeContext())

    assert len(result.ops_applied) == 100
    assert set(result.ops_applied) == {"double_positions"}


def test_repeat_runs_inner_ops_n_times() -> None:
    """Verify `Repeat` executes its inner pipeline the requested number of times."""

    problem = _make_problem()
    state = SolveState(pos=torch.ones((3, 2), dtype=torch.float32))
    repeated = Repeat(3, [_DoublePositions()])

    result = repeated.apply(problem, state, RuntimeContext())

    assert torch.equal(result.pos, torch.full((3, 2), 8.0))
    assert result.ops_applied == ["double_positions", "double_positions", "double_positions"]


def test_conditional_runs_only_when_predicate_is_true() -> None:
    """Verify `Conditional` respects the predicate result."""

    problem = _make_problem()
    false_state = SolveState(pos=torch.ones((3, 2), dtype=torch.float32))
    true_state = SolveState(pos=torch.ones((3, 2), dtype=torch.float32))
    condition = Conditional(lambda _p, state, _c: state.step > 0, _DoublePositions())

    skipped = condition.apply(problem, false_state, RuntimeContext())
    true_state.step = 1
    applied = condition.apply(problem, true_state, RuntimeContext())

    assert torch.equal(skipped.pos, torch.ones((3, 2), dtype=torch.float32))
    assert torch.equal(applied.pos, torch.full((3, 2), 2.0))


def test_annealing_schedule_stores_functions_and_snapshot_values() -> None:
    """Verify `AnnealingSchedule` stores schedule callables and current weights."""

    def decay(step: int, total_steps: int) -> float:
        """Return a simple linear decay value.

        Parameters
        ----------
        step : int
            Current optimization step.
        total_steps : int
            Total optimization steps.

        Returns
        -------
        float
            Decayed weight value.
        """
        return 1.0 - (step / total_steps)

    schedule = AnnealingSchedule(
        weight_fns={"repel": decay},
        current_weights={"repel": 0.75},
        crossing_alpha=2.0,
        crossing_step_counter=3,
        crossing_interval=5,
        crossing_weight_scale=1.5,
    )

    assert schedule.weight_fns["repel"](1, 4) == pytest.approx(0.75)
    assert schedule.current_weights["repel"] == pytest.approx(0.75)
    assert schedule.crossing_interval == 5

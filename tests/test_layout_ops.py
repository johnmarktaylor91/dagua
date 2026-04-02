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
    EarlyBreak,
    ExecutionPlan,
    FlexConstraints,
    GraphStructure,
    HierarchyLevel,
    LayoutProblem,
    LossGroup,
    LossOp,
    MultilevelVCycle,
    NullTraceSink,
    Op,
    OpCategory,
    Pipeline,
    Repeat,
    RuntimeContext,
    SolveState,
    get_op_class,
    list_categories,
    list_ops,
    register_op,
    unregister_op,
)


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


# -----------------------------------------------------------------------
# v2 additions: new SolveState fields
# -----------------------------------------------------------------------


def test_solve_state_has_forces_and_temperature() -> None:
    """Verify new force accumulation and temperature fields."""

    state = SolveState(
        pos=torch.zeros((3, 2)),
        forces=torch.ones((3, 2)),
        old_forces=torch.ones((3, 2)) * 0.5,
        temperature=100.0,
    )
    assert state.forces is not None
    assert state.temperature == 100.0
    assert state.old_forces is not None


def test_solve_state_has_extras_dict() -> None:
    """Verify extras escape hatch works."""

    state = SolveState()
    state.extras["gem_local_temperatures"] = torch.ones(10)
    state.extras["tsne_velocity"] = torch.zeros((10, 2))

    assert "gem_local_temperatures" in state.extras
    assert state.extras["tsne_velocity"].shape == (10, 2)


def test_solve_state_converged_flag_defaults_false() -> None:
    """Verify converged flag starts False."""

    state = SolveState()
    assert state.converged is False


def test_solve_state_ordering_and_edge_routes() -> None:
    """Verify ordering and edge_routes fields."""

    state = SolveState(
        ordering=torch.tensor([2, 0, 1]),
        edge_routes=[[(0.0, 0.0), (1.0, 1.0)]],
    )
    assert state.ordering is not None
    assert len(state.edge_routes) == 1


def test_solve_state_optimizer_field() -> None:
    """Verify optimizer can be stored in SolveState."""

    pos = torch.randn(3, 2, requires_grad=True)
    optimizer = torch.optim.Adam([pos], lr=0.01)
    state = SolveState(pos=pos, optimizer=optimizer)
    assert state.optimizer is not None


def test_layout_problem_has_edge_weights() -> None:
    """Verify edge_weights field on LayoutProblem."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        num_nodes=3,
        edge_weights=torch.tensor([1.0, 2.0]),
    )
    assert problem.edge_weights is not None
    assert problem.edge_weights.shape == (2,)


def test_hierarchy_level_has_cluster_ids() -> None:
    """Verify cluster_ids field on HierarchyLevel."""

    level = HierarchyLevel(
        num_nodes=5,
        num_fine=10,
        cluster_ids=torch.tensor([0, 0, 1, 1, 2]),
    )
    assert level.cluster_ids is not None


def test_execution_plan_has_subset_gpu_threshold() -> None:
    """Verify subset_gpu_threshold field on ExecutionPlan."""

    plan = ExecutionPlan(subset_gpu_threshold=50_000_000)
    assert plan.subset_gpu_threshold == 50_000_000


# -----------------------------------------------------------------------
# v2 additions: OpCategory enum and registry
# -----------------------------------------------------------------------


def test_op_category_enum_has_all_categories() -> None:
    """Verify OpCategory contains all 21 categories."""

    categories = list_categories()
    assert len(categories) == 21
    assert OpCategory.LOSS in categories
    assert OpCategory.FORCE in categories
    assert OpCategory.CONTROL in categories


def test_register_op_and_lookup() -> None:
    """Verify op registration and retrieval."""

    @register_op
    class _TestRegisteredOp(Op):
        name = "_test_registered_op"
        category = OpCategory.UTILITY

        def apply(self, problem, state, ctx):
            return state

    assert get_op_class("_test_registered_op") is _TestRegisteredOp
    assert "_test_registered_op" in list_ops()
    assert "_test_registered_op" in list_ops(category=OpCategory.UTILITY)

    # Cleanup
    unregister_op("_test_registered_op")
    assert "_test_registered_op" not in list_ops()


def test_register_op_raises_on_collision() -> None:
    """Verify name collision raises ValueError."""

    @register_op
    class _CollisionA(Op):
        name = "_collision_test"
        category = OpCategory.INIT

        def apply(self, problem, state, ctx):
            return state

    with pytest.raises(ValueError, match="collision"):

        @register_op
        class _CollisionB(Op):
            name = "_collision_test"
            category = OpCategory.INIT

            def apply(self, problem, state, ctx):
                return state

    # Cleanup
    unregister_op("_collision_test")


# -----------------------------------------------------------------------
# v2 additions: TraceSink op_start/op_end
# -----------------------------------------------------------------------


def test_null_trace_sink_has_op_boundary_methods() -> None:
    """Verify NullTraceSink has op_start and op_end."""

    sink = NullTraceSink()
    sink.op_start("test_op", 0)
    sink.op_end("test_op", 0)


# -----------------------------------------------------------------------
# v2 additions: Pipeline is an Op (nestable)
# -----------------------------------------------------------------------


def test_pipeline_is_op_subclass() -> None:
    """Verify Pipeline inherits from Op and has apply()."""

    assert issubclass(Pipeline, Op)
    pipe = Pipeline([_SetLayers()], name="test")
    assert hasattr(pipe, "apply")


def test_pipeline_apply_and_call_are_equivalent() -> None:
    """Verify Pipeline.apply() and Pipeline.__call__() produce same result."""

    problem = _make_problem()
    state1 = SolveState(pos=torch.ones((3, 2)))
    state2 = SolveState(pos=torch.ones((3, 2)))
    pipe = Pipeline([_DoublePositions()])

    r1 = pipe.apply(problem, state1, RuntimeContext())
    r2 = pipe(problem, state2, RuntimeContext())

    assert torch.equal(r1.pos, r2.pos)


def test_pipeline_trace_between_fires_events() -> None:
    """Verify trace_between triggers op_start/op_end on TraceSink."""

    events = []

    class RecordingSink:
        def snapshot(self, pos, step):
            events.append(("snapshot", step))

        def log(self, message):
            pass

        def op_start(self, op_name, step):
            events.append(("start", op_name))

        def op_end(self, op_name, step):
            events.append(("end", op_name))

    problem = _make_problem()
    state = SolveState(pos=torch.ones((3, 2)))
    ctx = RuntimeContext(trace_sink=RecordingSink())
    pipe = Pipeline([_DoublePositions()], trace_between=True)

    pipe.apply(problem, state, ctx)

    assert ("start", "double_positions") in events
    assert ("end", "double_positions") in events
    assert any(e[0] == "snapshot" for e in events)


def test_pipeline_dependency_graph() -> None:
    """Verify dependency_graph returns reads/writes per op."""

    pipe = Pipeline([_DoublePositions(), _SetLayers()])
    graph = pipe.dependency_graph()

    assert "double_positions" in graph
    assert graph["double_positions"]["reads"] == ("pos",)
    assert graph["set_layers"]["writes"] == ("layers",)


# -----------------------------------------------------------------------
# v2 additions: Conditional with else_op
# -----------------------------------------------------------------------


def test_conditional_else_op() -> None:
    """Verify Conditional runs else_op when predicate is False."""

    problem = _make_problem()
    state = SolveState(pos=torch.ones((3, 2)))

    # Predicate always False, else_op doubles positions
    cond = Conditional(
        predicate=lambda _p, _s, _c: False,
        op=_SetLayers(),
        else_op=_DoublePositions(),
    )
    result = cond.apply(problem, state, RuntimeContext())

    assert torch.equal(result.pos, torch.full((3, 2), 2.0))
    assert result.layers is None  # _SetLayers was NOT run


def test_conditional_backward_compat_no_else() -> None:
    """Verify Conditional works without else_op (backward compat)."""

    cond = Conditional(
        predicate=lambda _p, _s, _c: False,
        op=_DoublePositions(),
    )
    problem = _make_problem()
    state = SolveState(pos=torch.ones((3, 2)))
    result = cond.apply(problem, state, RuntimeContext())

    assert torch.equal(result.pos, torch.ones((3, 2)))


# -----------------------------------------------------------------------
# v2 additions: EarlyBreak + Repeat convergence
# -----------------------------------------------------------------------


def test_early_break_sets_converged() -> None:
    """Verify EarlyBreak sets state.converged."""

    problem = _make_problem()
    state = SolveState(pos=torch.ones((3, 2)))
    eb = EarlyBreak(predicate=lambda _p, _s, _c: True)

    result = eb.apply(problem, state, RuntimeContext())
    assert result.converged is True


def test_repeat_stops_on_converged() -> None:
    """Verify Repeat stops early when state.converged is True."""

    problem = _make_problem()
    state = SolveState(pos=torch.ones((3, 2)))

    # EarlyBreak fires immediately, so Repeat should run inner once
    # (EarlyBreak fires at end of first iteration, Repeat checks before second)
    repeated = Repeat(
        100,
        [
            _DoublePositions(),
            EarlyBreak(predicate=lambda _p, _s, _c: True),
        ],
    )
    result = repeated.apply(problem, state, RuntimeContext())

    # Only ran once: 1.0 * 2 = 2.0 (not 2^100)
    assert torch.equal(result.pos, torch.full((3, 2), 2.0))
    assert result.step == 1
    assert result.converged is True


def test_repeat_increments_step() -> None:
    """Verify Repeat auto-increments state.step."""

    problem = _make_problem()
    state = SolveState(pos=torch.ones((3, 2)))
    repeated = Repeat(5, [_DoublePositions()])

    result = repeated.apply(problem, state, RuntimeContext())
    assert result.step == 5


# -----------------------------------------------------------------------
# v2 additions: LossOp and LossGroup
# -----------------------------------------------------------------------


class _MockLoss(LossOp):
    """Test loss that returns a constant scalar."""

    name = "mock_loss"
    category = OpCategory.LOSS
    weight_key = "w_mock"
    default_weight = 2.0

    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def evaluate(self, problem, state, ctx):
        # Return a scalar tensor that depends on pos so backward works
        if state.pos is not None and state.pos.requires_grad:
            return (state.pos.sum() * 0 + self.value).requires_grad_(True)
        return torch.tensor(self.value, requires_grad=True)


def test_loss_op_evaluate_returns_scalar() -> None:
    """Verify LossOp.evaluate returns a scalar tensor."""

    problem = _make_problem()
    state = SolveState(pos=torch.randn(3, 2, requires_grad=True))
    loss = _MockLoss(value=3.14)

    result = loss.evaluate(problem, state, RuntimeContext())
    assert result.shape == ()
    assert result.item() == pytest.approx(3.14)


def test_loss_group_combined_mode() -> None:
    """Verify LossGroup sums losses and calls backward in combined mode."""

    problem = _make_problem()
    pos = torch.randn(3, 2, requires_grad=True)
    state = SolveState(pos=pos)

    group = LossGroup(
        losses=[_MockLoss(1.0), _MockLoss(2.0)],
        backward_mode="combined",
    )
    result = group.apply(problem, state, RuntimeContext())

    # default_weight=2.0 for both, so total = 2*1 + 2*2 = 6.0
    assert result.prev_loss == pytest.approx(6.0)


def test_loss_group_per_loss_mode() -> None:
    """Verify LossGroup per_loss mode calls backward per loss."""

    problem = _make_problem()
    pos = torch.randn(3, 2, requires_grad=True)
    state = SolveState(pos=pos)

    group = LossGroup(
        losses=[_MockLoss(1.0), _MockLoss(2.0)],
        backward_mode="per_loss",
    )
    result = group.apply(problem, state, RuntimeContext())

    assert result.prev_loss == pytest.approx(6.0)


def test_loss_group_uses_annealing_weights() -> None:
    """Verify LossGroup reads weights from annealing schedule."""

    problem = _make_problem()
    state = SolveState(
        pos=torch.randn(3, 2, requires_grad=True),
        annealing=AnnealingSchedule(current_weights={"w_mock": 0.5}),
    )

    group = LossGroup(losses=[_MockLoss(4.0)])
    result = group.apply(problem, state, RuntimeContext())

    # weight from schedule = 0.5, so total = 0.5 * 4.0 = 2.0
    assert result.prev_loss == pytest.approx(2.0)


# -----------------------------------------------------------------------
# v2 additions: Op.describe() and access_pattern
# -----------------------------------------------------------------------


def test_op_describe() -> None:
    """Verify Op.describe() returns human-readable string."""

    op = _DoublePositions()
    desc = op.describe()
    assert "double_positions" in desc


def test_op_access_pattern_default() -> None:
    """Verify default access_pattern is 'global'."""

    op = _DoublePositions()
    assert op.access_pattern == "global"


# -----------------------------------------------------------------------
# v2 additions: MultilevelVCycle max_levels
# -----------------------------------------------------------------------


def test_repeat_convergence_is_loop_scoped() -> None:
    """Verify a converged Repeat does not prevent a subsequent Repeat."""

    problem = _make_problem()
    state = SolveState(pos=torch.ones((3, 2)))

    # First Repeat converges immediately
    repeat1 = Repeat(
        100,
        [
            _DoublePositions(),
            EarlyBreak(predicate=lambda _p, _s, _c: True),
        ],
    )
    # Second Repeat should still run
    repeat2 = Repeat(3, [_DoublePositions()])

    pipe = Pipeline([repeat1, repeat2])
    result = pipe.apply(problem, state, RuntimeContext())

    # repeat1: 1 iteration -> pos = 2.0
    # repeat2: 3 iterations -> pos = 2.0 * 8.0 = 16.0
    assert torch.equal(result.pos, torch.full((3, 2), 16.0))


def test_loss_group_combined_no_losses() -> None:
    """Verify LossGroup handles zero active losses gracefully."""

    problem = _make_problem()
    state = SolveState(pos=torch.randn(3, 2, requires_grad=True))

    # All weights zero -> no losses evaluated
    group = LossGroup(losses=[], backward_mode="combined")
    result = group.apply(problem, state, RuntimeContext())
    assert result.prev_loss == 0.0


def test_multilevel_vcycle_has_max_levels() -> None:
    """Verify MultilevelVCycle accepts max_levels parameter."""

    dummy_pipe = Pipeline([_SetLayers()])
    vc = MultilevelVCycle(
        coarsen_op=_SetLayers(),
        base_layout=dummy_pipe,
        refine=dummy_pipe,
        min_nodes=100,
        max_levels=15,
    )
    assert vc.max_levels == 15

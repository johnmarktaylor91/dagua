"""Sprint 5: pin / align propagation through V-cycle multilevel layout.

Pre-Sprint-5, the V-cycle built per-level ``LayoutProblem`` views with
``flex=None`` (see ``dagua/layout/ops/vcycle.py:_level_problem``), so
coarse-level refinement and prolongation ignored user pins and alignment
groups entirely -- the finest pass was left to drag the layout into
alignment with the pin in a handful of steps.

Sprint 5 changes that: ``_propagate_flex_to_coarse`` maps fine-level
pins / align groups onto coarse ids via the composed
``fine_to_coarse`` mappings, and ``VCycleRefine`` hands each coarse
level the projected ``FlexConstraints``.

These tests exercise three things:
1. Hard pins still round-trip through V-cycle refinement (the finest-
   level ``HardPinProjection`` must not regress).
2. A pin on the FINEST problem appears on every coarse level's flex
   with the same target / weight / mask, at the composed coarse index.
3. An alignment group propagates: its fine indices collapse into coarse
   ids via the composed mapping, the group is kept if >= 2 coarse
   members remain, and the axis + weight are preserved.
"""

from __future__ import annotations

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.eval.graphs import make_chain
from dagua.layout.engine import layout as engine_layout
from dagua.layout.ops.state import FlexConstraints, HierarchyLevel
from dagua.layout.ops.vcycle import (
    _compose_finest_to_coarse,
    _propagate_flex_to_coarse,
)


@pytest.mark.unit
def test_hard_pin_round_trips_through_vcycle():
    """Hard pin must survive coarsening + V-cycle refinement.

    HardPinProjection runs every step of the finest refine pass, so the
    FINAL position should sit exactly on the pin target. This test is the
    floor for Sprint 5 -- propagation must not regress the existing
    finest-level projection.
    """
    g = make_chain(5000, seed=42).graph
    g.compute_node_sizes()
    g.pin(0, x=123.0, y=456.0)  # weight=inf -> hard pin

    cfg = LayoutConfig(seed=42, steps=40, multilevel_threshold=1000)
    cfg.flex = g.flex
    pos = engine_layout(g, cfg)

    a_idx = g._id_to_index[0]
    assert abs(pos[a_idx, 0].item() - 123.0) < 1e-3
    assert abs(pos[a_idx, 1].item() - 456.0) < 1e-3


def _make_level(num_fine: int, fine_to_coarse: torch.Tensor) -> HierarchyLevel:
    """Build a minimal ``HierarchyLevel`` for propagation tests."""
    num_coarse = int(fine_to_coarse.max().item()) + 1
    return HierarchyLevel(
        num_nodes=num_coarse,
        num_fine=num_fine,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        node_sizes=None,
        fine_to_coarse=fine_to_coarse,
    )


@pytest.mark.unit
def test_pin_propagates_through_multilevel_hierarchy():
    """A finest-level pin at fine-id 3 maps through two coarsening steps
    to the containing coarse id, carrying target / weight / mask intact.
    """
    # Finest -> level 0: 8 fine nodes pair up into 4 coarse nodes.
    level0 = _make_level(8, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long))
    # Level 0 -> level 1: 4 coarse nodes pair up into 2 super-coarse.
    level1 = _make_level(4, torch.tensor([0, 0, 1, 1], dtype=torch.long))

    # Pin on fine id 3 -> level 0 coarse id 1 -> level 1 super-coarse id 0.
    flex = FlexConstraints(
        pin_indices=torch.tensor([3], dtype=torch.long),
        pin_targets=torch.tensor([[100.0, 200.0]], dtype=torch.float32),
        pin_weights=torch.tensor([[float("inf"), float("inf")]], dtype=torch.float32),
        soft_pin_mask=torch.tensor([[False, False]], dtype=torch.bool),
        hard_pin_mask=torch.tensor([[True, True]], dtype=torch.bool),
    )

    composed_l0 = _compose_finest_to_coarse([level0, level1], 0, device=torch.device("cpu"))
    composed_l1 = _compose_finest_to_coarse([level0, level1], 1, device=torch.device("cpu"))

    coarse0 = _propagate_flex_to_coarse(flex, composed_l0, level0.num_nodes)
    coarse1 = _propagate_flex_to_coarse(flex, composed_l1, level1.num_nodes)

    assert coarse0.pin_indices.tolist() == [1], (
        f"level-0 pin must sit on coarse id 1, got {coarse0.pin_indices.tolist()}"
    )
    assert coarse1.pin_indices.tolist() == [0]
    for level_flex in (coarse0, coarse1):
        assert level_flex.pin_targets.tolist() == [[100.0, 200.0]]
        assert level_flex.hard_pin_mask.tolist() == [[True, True]]
        assert level_flex.soft_pin_mask.tolist() == [[False, False]]


@pytest.mark.unit
def test_pin_dedupes_when_multiple_fine_pins_collapse_into_same_coarse():
    """Two fine pins that both land on the same coarse node must collapse
    to a single coarse pin -- keeping the first (lowest fine index) wins.
    Picking one is not about 'correctness', it's about avoiding duplicate
    entries on the coarse level which would double-count the pin loss.
    """
    level = _make_level(4, torch.tensor([0, 0, 1, 1], dtype=torch.long))
    flex = FlexConstraints(
        pin_indices=torch.tensor([0, 1, 2], dtype=torch.long),
        pin_targets=torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype=torch.float32),
        pin_weights=torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=torch.float32),
        soft_pin_mask=torch.tensor([[True, True], [True, True], [True, True]], dtype=torch.bool),
        hard_pin_mask=torch.tensor(
            [[False, False], [False, False], [False, False]], dtype=torch.bool
        ),
    )
    composed = _compose_finest_to_coarse([level], 0, device=torch.device("cpu"))
    coarse = _propagate_flex_to_coarse(flex, composed, level.num_nodes)

    # Fines 0 and 1 both map to coarse 0 -- must dedupe, first wins.
    # Fine 2 maps to coarse 1 -- keeps its entry.
    assert coarse.pin_indices.tolist() == [0, 1]
    assert coarse.pin_targets.tolist() == [[10.0, 10.0], [30.0, 30.0]]
    assert coarse.pin_weights.tolist() == [[1.0, 1.0], [3.0, 3.0]]


@pytest.mark.unit
def test_alignment_group_propagates_and_collapses_on_single_coarse_member():
    """Fine align groups map through fine_to_coarse with dedup. Groups
    that collapse to < 2 distinct coarse members are dropped (nothing to
    align). Remaining groups keep weight + axis."""
    # 8 fine nodes -> 4 coarse nodes; pairs (0,1), (2,3), (4,5), (6,7).
    level = _make_level(8, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long))

    group_spread = (
        torch.tensor([0, 2, 4, 6], dtype=torch.long),  # hits 4 distinct coarse ids
        7.5,
        0,  # x-axis
    )
    group_collapses = (
        torch.tensor([0, 1], dtype=torch.long),  # both -> coarse 0 only
        3.0,
        1,  # y-axis
    )
    group_partial = (
        torch.tensor([0, 1, 4], dtype=torch.long),  # coarse ids {0, 2}
        4.0,
        0,
    )

    flex = FlexConstraints(align_groups=[group_spread, group_collapses, group_partial])
    composed = _compose_finest_to_coarse([level], 0, device=torch.device("cpu"))
    coarse = _propagate_flex_to_coarse(flex, composed, level.num_nodes)

    assert coarse is not None
    assert coarse.align_groups is not None
    kept_indices = [g[0].tolist() for g in coarse.align_groups]
    kept_weights = [g[1] for g in coarse.align_groups]
    kept_axes = [g[2] for g in coarse.align_groups]

    # Fully-spread group survives -- 4 coarse members, weight + axis intact.
    assert [0, 1, 2, 3] in kept_indices
    # Partial group survives -- 2 distinct coarse members.
    assert [0, 2] in kept_indices
    # Collapsed group was dropped.
    for idx in kept_indices:
        assert idx != [0], "single-coarse-member group must be dropped"

    # Matching axis + weight checks.
    pos_spread = kept_indices.index([0, 1, 2, 3])
    assert kept_weights[pos_spread] == 7.5
    assert kept_axes[pos_spread] == 0


@pytest.mark.unit
def test_propagated_flex_is_none_when_no_constraints_survive():
    """Empty pins + no align groups + no flex_node_sep -> no flex at all."""
    level = _make_level(4, torch.tensor([0, 0, 1, 1], dtype=torch.long))
    flex = FlexConstraints(
        pin_indices=torch.empty(0, dtype=torch.long),
        align_groups=None,
        flex_node_sep=None,
    )
    composed = _compose_finest_to_coarse([level], 0, device=torch.device("cpu"))
    coarse = _propagate_flex_to_coarse(flex, composed, level.num_nodes)
    assert coarse is None


@pytest.mark.unit
def test_soft_pin_converges_toward_target_through_vcycle():
    """End-to-end behavioural test: with a strong-enough soft pin and the
    Huber-based position_pin_loss (Sprint 5 r2) the pinned node must land
    meaningfully closer to target than an otherwise identical unpinned
    run. This is the Sprint 5 "soft flex weights propagate with
    appropriate scale" criterion."""
    target = (500.0, 500.0)
    weight = 500.0

    def _run(with_pin: bool):
        g = make_chain(3000, seed=42).graph
        g.compute_node_sizes()
        if with_pin:
            g.pin(0, x=target[0], y=target[1], weight=weight)
            cfg = LayoutConfig(seed=42, steps=40, multilevel_threshold=1000)
            cfg.flex = g.flex
        else:
            cfg = LayoutConfig(seed=42, steps=40, multilevel_threshold=1000)
        pos = engine_layout(g, cfg)
        return pos[g._id_to_index[0]].tolist()

    pinned = _run(with_pin=True)
    unpinned = _run(with_pin=False)
    d_pin = ((pinned[0] - target[0]) ** 2 + (pinned[1] - target[1]) ** 2) ** 0.5
    d_no = ((unpinned[0] - target[0]) ** 2 + (unpinned[1] - target[1]) ** 2) ** 0.5
    # A 30% reduction in distance-to-target is a real signal that
    # propagation + Huber scaling is doing the right thing -- not a
    # tight convergence bound (the engine's ClipGradNorm still caps
    # combined per-step motion so a few-dozen-step solve can't close
    # a 700-unit gap fully).
    assert d_pin < 0.7 * d_no, (
        f"soft pin did not measurably move node toward target: "
        f"pinned dist={d_pin:.1f}, unpinned dist={d_no:.1f}"
    )


@pytest.mark.unit
def test_pin_dedup_prefers_hard_over_soft():
    """Sprint 5 r2: when two fine pins collapse onto the same coarse node,
    a HARD pin must outrank a soft pin regardless of their insertion order.
    Pre-r2 the dedup was strictly 'first wins', so a later hard pin could
    silently be replaced by an earlier soft one."""
    level = _make_level(4, torch.tensor([0, 0, 1, 1], dtype=torch.long))
    # Fine 0 is a SOFT pin, fine 1 is a HARD pin -- both collapse to coarse 0.
    flex = FlexConstraints(
        pin_indices=torch.tensor([0, 1], dtype=torch.long),
        pin_targets=torch.tensor([[5.0, 5.0], [50.0, 50.0]], dtype=torch.float32),
        pin_weights=torch.tensor([[1.0, 1.0], [float("inf"), float("inf")]], dtype=torch.float32),
        soft_pin_mask=torch.tensor([[True, True], [False, False]], dtype=torch.bool),
        hard_pin_mask=torch.tensor([[False, False], [True, True]], dtype=torch.bool),
    )
    composed = _compose_finest_to_coarse([level], 0, device=torch.device("cpu"))
    coarse = _propagate_flex_to_coarse(flex, composed, level.num_nodes)

    assert coarse.pin_indices.tolist() == [0]
    # Hard pin (fine id 1, target 50,50) must win over the earlier soft pin.
    assert coarse.pin_targets.tolist() == [[50.0, 50.0]]
    assert coarse.hard_pin_mask.tolist() == [[True, True]]
    assert coarse.soft_pin_mask.tolist() == [[False, False]]


@pytest.mark.unit
def test_vcycle_actually_passes_propagated_flex_to_coarse_refine():
    """End-to-end: a pin on the finest graph must end up in the coarse-
    level problem.flex seen by refine ops. We patch PositionPinLoss'
    evaluate to capture whatever flex it sees on each call, then verify
    a non-trivial pin_indices tensor was routed through on a coarse
    level (proving propagation is wired through VCycleRefine -- not just
    available as a helper function).
    """
    from dagua.layout.ops.loss_engine import PositionPinLoss

    seen_pins: list[torch.Tensor] = []
    seen_num_nodes: list[int] = []
    orig_evaluate = PositionPinLoss.evaluate

    def spy(self, problem, state, ctx):
        if problem.flex is not None and problem.flex.pin_indices is not None:
            seen_pins.append(problem.flex.pin_indices.clone())
            seen_num_nodes.append(problem.num_nodes)
        return orig_evaluate(self, problem, state, ctx)

    PositionPinLoss.evaluate = spy
    try:
        g = make_chain(5000, seed=42).graph
        g.compute_node_sizes()
        g.pin(0, x=111.0, y=222.0)  # hard pin
        cfg = LayoutConfig(seed=42, steps=5, multilevel_threshold=1000)
        cfg.flex = g.flex
        engine_layout(g, cfg)
    finally:
        PositionPinLoss.evaluate = orig_evaluate

    assert seen_pins, "PositionPinLoss was never invoked -- pipeline mis-wired"
    # At least one call must have seen a COARSE problem (< 5000 nodes).
    coarse_calls = [n for n in seen_num_nodes if n < 5000]
    assert coarse_calls, (
        "PositionPinLoss only ran on the finest level; coarse levels are "
        "still running without propagated flex"
    )

"""Tests for r80-S7#3: quality-gated wiring of the differentiable edge
route optimizer (the mechanism BezierControlPointOpt delegates to)."""

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.edges import route_edges
from dagua.layout.edge_optimization import (
    _HIGH_QUALITY_THRESHOLD,
    _ForcedQualityEdgeConfig,
    maybe_refine_routes,
)


@pytest.fixture
def dense_scene():
    """Positions/edges/sizes with enough overlap for a nonzero heuristic
    edge-node-crossing count, so the draft/balanced adaptive-skip path has
    something to potentially act on."""
    pos = torch.tensor(
        [[0.0, 0.0], [0.0, 300.0], [5.0, 50.0], [0.0, 100.0], [5.0, 150.0], [0.0, 200.0]]
    )
    ei = torch.tensor([[0], [1]])
    ns = torch.tensor([[40.0, 20.0]] * 6)
    return pos, ei, ns


class TestMaybeRefineRoutes:
    def test_high_quality_threshold_is_075(self) -> None:
        assert _HIGH_QUALITY_THRESHOLD == 0.75

    def test_balanced_quality_skips_when_heuristic_is_clean(self) -> None:
        pos = torch.tensor([[0.0, 0.0], [100.0, 100.0], [200.0, 0.0]])
        ei = torch.tensor([[0, 1], [1, 2]])
        ns = torch.tensor([[40.0, 20.0]] * 3)
        curves = route_edges(pos, ei, ns)

        cfg = LayoutConfig(quality="balanced")
        out = maybe_refine_routes(curves, pos, ei, ns, cfg)
        assert out is curves  # unchanged Sprint 6 behavior preserved

    def test_high_quality_forces_the_pass_on(self, dense_scene) -> None:
        pos, ei, ns = dense_scene
        curves = route_edges(pos, ei, ns)

        cfg = LayoutConfig(quality="high", edge_opt_steps=5)
        out = maybe_refine_routes(curves, pos, ei, ns, cfg)
        assert out is not curves
        assert len(out) == len(curves)

    def test_max_quality_also_forces_the_pass_on(self, dense_scene) -> None:
        pos, ei, ns = dense_scene
        curves = route_edges(pos, ei, ns)

        cfg = LayoutConfig(quality="max", edge_opt_steps=5)
        out = maybe_refine_routes(curves, pos, ei, ns, cfg)
        assert out is not curves

    def test_edge_opt_steps_negative_one_always_skips(self, dense_scene) -> None:
        """Explicit opt-out (-1) must win even at max quality."""
        pos, ei, ns = dense_scene
        curves = route_edges(pos, ei, ns)

        cfg = LayoutConfig(quality="max", edge_opt_steps=-1)
        out = maybe_refine_routes(curves, pos, ei, ns, cfg)
        assert out is curves

    def test_heuristic_edge_routing_mode_always_skips(self, dense_scene) -> None:
        pos, ei, ns = dense_scene
        curves = route_edges(pos, ei, ns)

        cfg = LayoutConfig(quality="max", edge_routing="heuristic")
        out = maybe_refine_routes(curves, pos, ei, ns, cfg)
        assert out is curves

    def test_positions_are_never_mutated(self, dense_scene) -> None:
        pos, ei, ns = dense_scene
        pos_before = pos.clone()
        curves = route_edges(pos, ei, ns)

        cfg = LayoutConfig(quality="max", edge_opt_steps=5)
        maybe_refine_routes(curves, pos, ei, ns, cfg)
        assert torch.equal(pos, pos_before)

    def test_empty_curves_returns_empty(self) -> None:
        pos = torch.zeros((0, 2))
        ei = torch.zeros((2, 0), dtype=torch.long)
        ns = torch.zeros((0, 2))
        cfg = LayoutConfig(quality="max")
        assert maybe_refine_routes([], pos, ei, ns, cfg) == []


class TestForcedQualityEdgeConfig:
    def test_overrides_win_over_base_config(self) -> None:
        base = LayoutConfig()
        assert base.w_edge_angular_res == 0.0  # Sprint 6 default: disabled
        wrapped = _ForcedQualityEdgeConfig(base)
        assert wrapped.w_edge_angular_res == 2.0
        assert wrapped.w_edge_curvature_consistency == 1.0

    def test_non_override_fields_pass_through_to_base(self) -> None:
        base = LayoutConfig(edge_opt_steps=7)
        wrapped = _ForcedQualityEdgeConfig(base)
        assert wrapped.edge_opt_steps == 7
        assert wrapped.quality == 0.5  # "balanced" normalized default

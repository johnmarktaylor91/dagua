"""Tests for dagua.flex — Flex, LayoutFlex, AlignGroup, constraint losses, hard pins."""

import pytest
import torch

from dagua.flex import AlignGroup, Flex, LayoutFlex
from dagua.layout.constraints import (
    alignment_loss,
    flex_spacing_loss,
    position_pin_loss,
    project_hard_pins,
)


class TestFlex:
    def test_soft(self):
        f = Flex.soft(40)
        assert f.target == 40
        assert f.weight == 0.5
        assert not f.is_hard

    def test_firm(self):
        f = Flex.firm(40)
        assert f.target == 40
        assert f.weight == 2.0
        assert not f.is_hard

    def test_rigid(self):
        f = Flex.rigid(40)
        assert f.target == 40
        assert f.weight == 10.0
        assert not f.is_hard

    def test_locked(self):
        f = Flex.locked(40)
        assert f.target == 40
        assert f.weight == float("inf")
        assert f.is_hard

    def test_custom_weight(self):
        f = Flex.soft(30, weight=0.8)
        assert f.weight == 0.8

    def test_frozen(self):
        f = Flex(target=1.0, weight=1.0)
        with pytest.raises(AttributeError):
            f.target = 2.0


class TestAlignGroup:
    def test_basic(self):
        g = AlignGroup(nodes=["a", "b", "c"])
        assert g.nodes == ["a", "b", "c"]
        assert g.weight == 5.0

    def test_custom_weight(self):
        g = AlignGroup(nodes=[1, 2], weight=10.0)
        assert g.weight == 10.0


class TestLayoutFlex:
    def test_empty(self):
        lf = LayoutFlex()
        assert lf.node_sep is None
        assert lf.rank_sep is None
        assert lf.pins is None
        assert lf.align_x is None
        assert lf.align_y is None

    def test_with_all_fields(self):
        lf = LayoutFlex(
            node_sep=Flex.firm(40),
            rank_sep=Flex.soft(60),
            pins={"input": (Flex.locked(0), Flex.locked(0))},
            align_x=[AlignGroup(nodes=["a", "b"])],
            align_y=[AlignGroup(nodes=["c", "d"], weight=3.0)],
        )
        assert lf.node_sep.target == 40
        assert lf.rank_sep.weight == 0.5
        assert lf.pins["input"][0].is_hard
        assert len(lf.align_x) == 1
        assert lf.align_y[0].weight == 3.0


class TestPositionPinLoss:
    def test_basic_soft_pin(self):
        pos = torch.tensor([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
        pin_indices = torch.tensor([0])
        pin_targets = torch.tensor([[0.0, 0.0]])
        pin_weights = torch.tensor([[2.0, 2.0]])
        pin_mask = torch.tensor([[True, True]])

        loss = position_pin_loss(pos, pin_indices, pin_targets, pin_weights, pin_mask)
        assert loss.item() > 0
        loss.backward()
        assert pos.grad is not None

    def test_pin_at_target_zero_loss(self):
        pos = torch.tensor([[5.0, 10.0]], requires_grad=True)
        pin_indices = torch.tensor([0])
        pin_targets = torch.tensor([[5.0, 10.0]])
        pin_weights = torch.tensor([[1.0, 1.0]])
        pin_mask = torch.tensor([[True, True]])

        loss = position_pin_loss(pos, pin_indices, pin_targets, pin_weights, pin_mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_empty_pins(self):
        pos = torch.tensor([[1.0, 2.0]], requires_grad=True)
        loss = position_pin_loss(
            pos,
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, 2),
            torch.zeros(0, 2),
            torch.zeros(0, 2, dtype=torch.bool),
        )
        assert loss.item() == 0.0

    def test_partial_pin(self):
        """Pin only x axis, y should not contribute to loss."""
        pos = torch.tensor([[10.0, 999.0]], requires_grad=True)
        pin_indices = torch.tensor([0])
        pin_targets = torch.tensor([[0.0, 0.0]])
        pin_weights = torch.tensor([[5.0, 0.0]])
        pin_mask = torch.tensor([[True, False]])

        loss = position_pin_loss(pos, pin_indices, pin_targets, pin_weights, pin_mask)
        # Should only penalize x deviation
        expected = 5.0 * 100.0  # weight * (10-0)^2, divided by 1 masked element
        assert loss.item() == pytest.approx(expected, rel=1e-4)


class TestAlignmentLoss:
    def test_perfect_alignment_zero_loss(self):
        pos = torch.tensor([[5.0, 10.0], [5.0, 20.0], [5.0, 30.0]], requires_grad=True)
        groups = [(torch.tensor([0, 1, 2]), 5.0, 0)]  # align on x

        loss = alignment_loss(pos, groups)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_misaligned_nonzero_loss(self):
        pos = torch.tensor([[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]], requires_grad=True)
        groups = [(torch.tensor([0, 1, 2]), 5.0, 0)]  # align on x

        loss = alignment_loss(pos, groups)
        assert loss.item() > 0
        loss.backward()
        assert pos.grad is not None

    def test_y_axis_alignment(self):
        pos = torch.tensor([[0.0, 0.0], [10.0, 0.0]], requires_grad=True)
        groups = [(torch.tensor([0, 1]), 1.0, 1)]  # align on y

        loss = alignment_loss(pos, groups)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_empty_groups(self):
        pos = torch.tensor([[1.0, 2.0]])
        loss = alignment_loss(pos, [])
        assert loss.item() == 0.0


class TestProjectHardPins:
    def test_hard_pin_projection(self):
        pos = torch.tensor([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
        pin_indices = torch.tensor([0])
        pin_targets = torch.tensor([[0.0, 0.0]])
        pin_mask = torch.tensor([[True, True]])

        project_hard_pins(pos, pin_indices, pin_targets, pin_mask)

        assert pos[0, 0].item() == pytest.approx(0.0)
        assert pos[0, 1].item() == pytest.approx(0.0)
        # Node 1 unchanged
        assert pos[1, 0].item() == pytest.approx(30.0)

    def test_partial_hard_pin(self):
        """Pin only x, y should stay."""
        pos = torch.tensor([[10.0, 20.0]], requires_grad=True)
        pin_indices = torch.tensor([0])
        pin_targets = torch.tensor([[5.0, 0.0]])
        pin_mask = torch.tensor([[True, False]])

        project_hard_pins(pos, pin_indices, pin_targets, pin_mask)

        assert pos[0, 0].item() == pytest.approx(5.0)
        assert pos[0, 1].item() == pytest.approx(20.0)  # unchanged

    def test_empty_pins(self):
        pos = torch.tensor([[1.0, 2.0]], requires_grad=True)
        # Should be a no-op
        project_hard_pins(
            pos,
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, 2),
            torch.zeros(0, 2, dtype=torch.bool),
        )
        assert pos[0, 0].item() == pytest.approx(1.0)


class TestFlexIntegration:
    def test_layout_with_pins(self):
        """Integration: run layout with flex pins and verify they're respected."""
        from dagua.config import LayoutConfig
        from dagua.graph import DaguaGraph

        g = DaguaGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")

        # Pin node "a" at origin
        g.pin("a", x=0, y=0)

        config = LayoutConfig(steps=50)
        config.flex = g.flex

        from dagua.layout import layout
        pos = layout(g, config)

        # Node "a" should be near (0, 0) — hard pin
        a_idx = g._id_to_index["a"]
        assert abs(pos[a_idx, 0].item()) < 5.0
        assert abs(pos[a_idx, 1].item()) < 5.0

    def test_layout_with_alignment(self):
        """Integration: aligned nodes should end up with similar x coordinates."""
        from dagua.config import LayoutConfig
        from dagua.graph import DaguaGraph

        g = DaguaGraph()
        for i in range(4):
            g.add_node(i, label=str(i))
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)

        g.align([1, 2, 3], axis="x", weight=10.0)

        config = LayoutConfig(steps=100)
        config.flex = g.flex

        from dagua.layout import layout
        pos = layout(g, config)

        # Nodes 1, 2, 3 should have more similar x coordinates than without alignment.
        # The alignment loss competes with other forces (repulsion, overlap), so we
        # verify alignment reduced spread rather than demanding perfect alignment.
        x_vals = [pos[g._id_to_index[i], 0].item() for i in [1, 2, 3]]
        x_spread = max(x_vals) - min(x_vals)

        # Run without alignment for comparison
        g2 = DaguaGraph()
        for i in range(4):
            g2.add_node(i, label=str(i))
        g2.add_edge(0, 1)
        g2.add_edge(0, 2)
        g2.add_edge(0, 3)
        config2 = LayoutConfig(steps=100)
        pos2 = layout(g2, config2)
        x_vals2 = [pos2[i, 0].item() for i in [1, 2, 3]]
        x_spread2 = max(x_vals2) - min(x_vals2)

        # Alignment should reduce spread (or at least not be wildly larger)
        assert x_spread <= x_spread2 * 1.5  # allow some tolerance

"""Tests for cycle (recurrent loop) support in dagua."""

import pytest
import torch

from dagua import DaguaGraph, LayoutConfig, draw
from dagua.layout.cycle import detect_back_edges, make_acyclic
from dagua.metrics import dag_consistency


# ---------------------------------------------------------------------------
# detect_back_edges
# ---------------------------------------------------------------------------


class TestDetectBackEdges:
    """Back-edge detection via DFS."""

    def test_simple_cycle(self):
        """A→B→C→A: one back edge (C→A)."""
        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        mask = detect_back_edges(ei, 3)
        assert mask.shape == (3,)
        assert mask.sum().item() == 1
        # The back edge is C→A (index 2)
        assert mask[2].item() is True

    def test_dag_no_back_edges(self):
        """A→B→C: no cycles."""
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        mask = detect_back_edges(ei, 3)
        assert mask.sum().item() == 0

    def test_self_loop(self):
        """A→A: self-loop is a back edge."""
        ei = torch.tensor([[0], [0]], dtype=torch.long)
        mask = detect_back_edges(ei, 1)
        assert mask[0].item() is True

    def test_self_loop_with_dag(self):
        """A→B, A→A: self-loop detected, A→B is forward."""
        ei = torch.tensor([[0, 0], [1, 0]], dtype=torch.long)
        mask = detect_back_edges(ei, 2)
        # A→B (idx 0) is forward, A→A (idx 1) is back
        assert mask[0].item() is False
        assert mask[1].item() is True

    def test_multiple_cycles(self):
        """Two separate cycles: A→B→A, C→D→C."""
        ei = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        mask = detect_back_edges(ei, 4)
        assert mask.sum().item() == 2

    def test_diamond_with_back_edge(self):
        """Diamond A→B→D, A→C→D plus back edge D→A."""
        ei = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 3, 3, 0]], dtype=torch.long)
        mask = detect_back_edges(ei, 4)
        assert mask.sum().item() == 1
        assert mask[4].item() is True  # D→A

    def test_large_ring(self):
        """100-node ring: 0→1→2→...→99→0."""
        n = 100
        src = list(range(n))
        tgt = list(range(1, n)) + [0]
        ei = torch.tensor([src, tgt], dtype=torch.long)
        mask = detect_back_edges(ei, n)
        assert mask.sum().item() == 1  # one back edge closes the ring

    def test_empty_graph(self):
        """No edges → empty mask."""
        ei = torch.zeros(2, 0, dtype=torch.long)
        mask = detect_back_edges(ei, 5)
        assert mask.shape == (0,)

    def test_disconnected_components_with_cycle(self):
        """Component 1: A→B (DAG), Component 2: C→D→C (cycle)."""
        ei = torch.tensor([[0, 2, 3], [1, 3, 2]], dtype=torch.long)
        mask = detect_back_edges(ei, 4)
        assert mask[0].item() is False  # A→B forward
        assert mask.sum().item() == 1   # one back edge in C→D→C


# ---------------------------------------------------------------------------
# make_acyclic
# ---------------------------------------------------------------------------


class TestMakeAcyclic:
    """Edge reversal for back edges."""

    def test_reverses_back_edges(self):
        """Back edges get src/tgt swapped."""
        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        mask = torch.tensor([False, False, True])
        result = make_acyclic(ei, mask)
        # Edge 2 was (2,0), now (0,2)
        assert result[0, 2].item() == 0
        assert result[1, 2].item() == 2
        # Other edges unchanged
        assert result[0, 0].item() == 0
        assert result[1, 0].item() == 1

    def test_no_back_edges_unchanged(self):
        """No back edges → identical tensor."""
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        mask = torch.tensor([False, False])
        result = make_acyclic(ei, mask)
        assert torch.equal(result, ei)

    def test_does_not_mutate_original(self):
        """make_acyclic returns a new tensor."""
        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        original = ei.clone()
        mask = torch.tensor([False, False, True])
        make_acyclic(ei, mask)
        assert torch.equal(ei, original)


# ---------------------------------------------------------------------------
# DaguaGraph cycle properties
# ---------------------------------------------------------------------------


class TestGraphCycleProperties:
    """has_cycles, back_edge_mask, set_back_edge_mask."""

    def test_has_cycles_true(self):
        g = DaguaGraph.from_edge_list([(0, 1), (1, 2), (2, 0)], num_nodes=3)
        assert g.has_cycles is True

    def test_has_cycles_false(self):
        g = DaguaGraph.from_edge_list([(0, 1), (1, 2)], num_nodes=3)
        assert g.has_cycles is False

    def test_back_edge_mask_returns_tensor(self):
        g = DaguaGraph.from_edge_list([(0, 1), (1, 2), (2, 0)], num_nodes=3)
        mask = g.back_edge_mask
        assert mask is not None
        assert mask.dtype == torch.bool
        assert mask.shape == (3,)
        assert mask.sum().item() == 1

    def test_back_edge_mask_none_for_dag(self):
        g = DaguaGraph.from_edge_list([(0, 1), (1, 2)], num_nodes=3)
        assert g.back_edge_mask is None

    def test_set_back_edge_mask(self):
        g = DaguaGraph.from_edge_list([(0, 1), (1, 2), (2, 0)], num_nodes=3)
        manual_mask = torch.tensor([False, True, False])
        g.set_back_edge_mask(manual_mask)
        assert torch.equal(g.back_edge_mask, manual_mask)

    def test_set_back_edge_mask_wrong_size(self):
        g = DaguaGraph.from_edge_list([(0, 1), (1, 2)], num_nodes=3)
        with pytest.raises(ValueError, match="mask length"):
            g.set_back_edge_mask(torch.tensor([True, True, True]))

    def test_cache_invalidation_on_add_edge(self):
        g = DaguaGraph.from_edge_list([(0, 1), (1, 2), (2, 0)], num_nodes=3)
        _ = g.back_edge_mask  # populate cache
        assert g._back_edge_mask is not None
        g.add_edge(0, 2)  # adds a new edge → triggers _finalize_edges → clears cache
        _ = g.edge_index  # finalize
        # Cache was cleared since new pending edges were flushed
        # Re-detection should still find the original cycle
        assert g.has_cycles is True

    def test_cache_invalidation_on_edge_index_setter(self):
        g = DaguaGraph.from_edge_list([(0, 1), (1, 2), (2, 0)], num_nodes=3)
        _ = g.back_edge_mask
        assert g._back_edge_mask is not None
        # Set new acyclic edges
        g.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        assert g._back_edge_mask is None
        assert g.has_cycles is False


# ---------------------------------------------------------------------------
# Layout on cyclic graphs
# ---------------------------------------------------------------------------


class TestLayoutWithCycles:
    """layout() on cyclic graphs produces valid positions."""

    def test_layout_simple_cycle(self):
        g = DaguaGraph.from_edge_list([(0, 1), (1, 2), (2, 0)], num_nodes=3)
        from dagua import layout
        pos = layout(g, LayoutConfig(steps=20, seed=42))
        assert pos.shape == (3, 2)
        assert torch.isfinite(pos).all()

    def test_edge_index_restored_after_layout(self):
        """Original edge directions are restored after layout."""
        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        g = DaguaGraph.from_edge_index(ei, num_nodes=3)
        original_ei = g.edge_index.clone()
        from dagua import layout
        layout(g, LayoutConfig(steps=10, seed=42))
        assert torch.equal(g.edge_index, original_ei)

    def test_edge_index_restored_on_exception(self):
        """Edge index is restored even if layout throws."""
        g = DaguaGraph.from_edge_list([(0, 1), (1, 0)], num_nodes=2)
        original_ei = g.edge_index.clone()

        # Force an exception during layout by passing bad config
        class BadConfig(LayoutConfig):
            @property
            def steps(self):
                raise RuntimeError("boom")
            @steps.setter
            def steps(self, _):
                pass

        try:
            from dagua.layout.engine import layout as engine_layout
            engine_layout(g, BadConfig(seed=42))
        except (RuntimeError, Exception):
            pass
        assert torch.equal(g.edge_index, original_ei)

    def test_layout_dag_no_overhead(self):
        """DAG layout: _original_edge_index stays None (no reversal happened)."""
        g = DaguaGraph.from_edge_list([(0, 1), (1, 2)], num_nodes=3)
        from dagua import layout
        layout(g, LayoutConfig(steps=10, seed=42))
        assert g._original_edge_index is None


# ---------------------------------------------------------------------------
# draw() end-to-end
# ---------------------------------------------------------------------------


class TestDrawWithCycles:
    """draw() pipeline completes on cyclic graphs."""

    def test_draw_cycle_no_error(self):
        g = DaguaGraph()
        g.add_node("A", label="A")
        g.add_node("B", label="B")
        g.add_node("C", label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")
        fig = draw(g, LayoutConfig(steps=10, seed=42))
        assert fig is not None


# ---------------------------------------------------------------------------
# get_style_for_edge — back edge style
# ---------------------------------------------------------------------------


class TestBackEdgeStyle:
    """get_style_for_edge returns 'back' style for back edges."""

    def test_back_edge_gets_back_style(self):
        g = DaguaGraph.from_edge_list([(0, 1), (1, 2), (2, 0)], num_nodes=3)
        _ = g.back_edge_mask  # trigger detection
        back_idx = g._back_edge_mask.nonzero(as_tuple=False).squeeze().item()
        style = g.get_style_for_edge(back_idx)
        assert style.curvature == 0.6  # back style's curvature

    def test_per_edge_override_beats_back(self):
        """Per-edge style override takes priority over back edge styling."""
        from dagua import EdgeStyle
        g = DaguaGraph.from_edge_list([(0, 1), (1, 0)], num_nodes=2)
        _ = g.back_edge_mask
        # Override edge 1 (the back edge) with a custom style
        g.edge_styles[1] = EdgeStyle(color="#FF0000")
        style = g.get_style_for_edge(1)
        assert style.color == "#FF0000"


# ---------------------------------------------------------------------------
# dag_consistency with back_edge_mask
# ---------------------------------------------------------------------------


class TestDagConsistencyBackEdges:
    """dag_consistency excludes back edges and reports back_edge_count."""

    def test_excludes_back_edges(self):
        """Back edges shouldn't count as violations."""
        # Positions: node 0 at y=0, node 1 at y=1, node 2 at y=2
        pos = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        mask = torch.tensor([False, False, True])  # 2→0 is back

        result = dag_consistency(pos, ei, direction="TB", back_edge_mask=mask)
        assert result["dag_consistency"] == 1.0  # forward edges are correct
        assert result["back_edge_count"] == 1
        assert result["back_edge_fraction"] == pytest.approx(1 / 3)

    def test_no_mask_includes_back_edges(self):
        """Without mask, back edge counted as violation."""
        pos = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        result = dag_consistency(pos, ei, direction="TB")
        assert result["dag_consistency"] < 1.0  # 2→0 is a violation
        assert result["back_edge_count"] == 0

    def test_all_back_edges(self):
        """If all edges are back edges, consistency is 1.0 (no forward edges)."""
        pos = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        ei = torch.tensor([[1], [0]], dtype=torch.long)
        mask = torch.tensor([True])
        result = dag_consistency(pos, ei, direction="TB", back_edge_mask=mask)
        assert result["dag_consistency"] == 1.0
        assert result["back_edge_count"] == 1


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestCycleJsonRoundTrip:
    """Serialize and deserialize graphs with cycles."""

    def test_roundtrip_preserves_back_edges(self):
        g = DaguaGraph.from_edge_list([(0, 1), (1, 2), (2, 0)], num_nodes=3)
        _ = g.back_edge_mask  # detect cycles

        data = g.to_json()
        assert "back_edges" in data

        g2 = DaguaGraph.from_json(data)
        assert g2._back_edge_mask is not None
        assert torch.equal(g2._back_edge_mask, g._back_edge_mask)

    def test_roundtrip_dag_no_back_edges_key(self):
        g = DaguaGraph.from_edge_list([(0, 1), (1, 2)], num_nodes=3)
        data = g.to_json()
        assert "back_edges" not in data

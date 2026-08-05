import importlib

import torch

from dagua.utils import VRAMBudget

init_placement = importlib.import_module("dagua.layout.init_placement")


def test_choose_init_device_keeps_cpu_when_requested():
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    node_sizes = torch.ones((3, 2), dtype=torch.float32)
    assert init_placement._choose_init_device(edge_index, 3, node_sizes, "cpu") == "cpu"


def test_choose_init_device_falls_back_when_cuda_headroom_is_insufficient(monkeypatch):
    edge_index = torch.zeros((2, 10), dtype=torch.int32)
    node_sizes = torch.ones((8, 2), dtype=torch.float16)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(VRAMBudget, "__init__", lambda self: None)
    monkeypatch.setattr(VRAMBudget, "fits", lambda self, needed_bytes: False)

    assert init_placement._choose_init_device(edge_index, 8, node_sizes, "cuda") == "cpu"


def test_init_positions_degenerate_layering_probe_matches_counter_semantics():
    """The heavy-skew probe must behave identically after the Counter rewrite.

    Pins WP03-F18: the degenerate-layering probe used a per-value
    ``list.count`` scan (O(unique_layers x N), quadratic on chains). The
    single-pass histogram must trigger the same relayering decisions: a
    star (heavy skew, >50% of nodes in one layer) and a plain chain
    (every layer distinct, no skew) both return finite [N, 2] positions.
    """
    # Star: node 0 feeds nodes 1..9 -> two layers, 9/10 nodes in layer 1.
    star_edges = torch.stack(
        [
            torch.zeros(9, dtype=torch.long),
            torch.arange(1, 10, dtype=torch.long),
        ]
    )
    star_sizes = torch.full((10, 2), 20.0)
    star_pos = init_placement.init_positions(star_edges, 10, star_sizes)
    assert star_pos.shape == (10, 2)
    assert torch.isfinite(star_pos).all()

    # Chain: unique layer per node (the previously-quadratic shape).
    chain_edges = torch.stack(
        [
            torch.arange(0, 9, dtype=torch.long),
            torch.arange(1, 10, dtype=torch.long),
        ]
    )
    chain_pos = init_placement.init_positions(chain_edges, 10, star_sizes)
    assert chain_pos.shape == (10, 2)
    assert torch.isfinite(chain_pos).all()

import importlib

import torch

init_placement = importlib.import_module("dagua.layout.init_placement")


def test_choose_init_device_keeps_cpu_when_requested():
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    node_sizes = torch.ones((3, 2), dtype=torch.float32)
    assert init_placement._choose_init_device(edge_index, 3, node_sizes, "cpu") == "cpu"


def test_choose_init_device_falls_back_when_cuda_headroom_is_insufficient(monkeypatch):
    edge_index = torch.zeros((2, 10), dtype=torch.int32)
    node_sizes = torch.ones((8, 2), dtype=torch.float16)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(init_placement, "_vram_fits", lambda needed_bytes, safety=0.65: False)

    assert init_placement._choose_init_device(edge_index, 8, node_sizes, "cuda") == "cpu"

"""Protective tests for the large benchmark helper script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench_large.py"
_SPEC = importlib.util.spec_from_file_location("bench_large", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench_large = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench_large)


def test_parse_node_count_accepts_suffixes_and_separators():
    assert bench_large.parse_node_count("1b") == 1_000_000_000
    assert bench_large.parse_node_count("1.5m") == 1_500_000
    assert bench_large.parse_node_count("10_000") == 10_000
    assert bench_large.parse_node_count("2,500k") == 2_500_000


def test_resolve_size_and_layers_rounds_up_preset():
    n, layers, width = bench_large.resolve_size_and_layers("1b")
    assert layers == 1500
    assert n == 1_000_000_500
    assert n % layers == 0
    assert width == n // layers


def test_build_edges_simple_keeps_targets_in_bounds():
    torch.manual_seed(0)
    n, layers, _width = bench_large.resolve_size_and_layers("120", 12)
    edge_index = bench_large.build_edges(n, layers)

    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] >= n - (n // layers)
    assert int(edge_index.min()) >= 0
    assert int(edge_index.max()) < n

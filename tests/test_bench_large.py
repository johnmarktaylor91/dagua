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


def test_graph_checkpoint_round_trip(tmp_path: Path):
    checkpoint_dir = tmp_path / "bench_ckpt"
    paths = bench_large._checkpoint_paths(checkpoint_dir)
    n, layers, _width = bench_large.resolve_size_and_layers("120", 12)
    edge_index = torch.tensor([[0, 1, 2], [10, 11, 12]], dtype=torch.int32)
    node_sizes = torch.full((n, 2), 20.0, dtype=torch.float16)

    bench_large._save_checkpoint_meta(paths["meta"], {"n": n, "layers": layers})
    torch.save(edge_index, paths["edge_index"])
    torch.save(node_sizes, paths["node_sizes"])

    restored = bench_large._load_graph_checkpoint(paths, n, layers)

    assert restored is not None
    restored_edge_index, restored_node_sizes = restored
    assert torch.equal(restored_edge_index, edge_index)
    assert torch.equal(restored_node_sizes, node_sizes)


def test_graph_checkpoint_rejects_mismatched_shape(tmp_path: Path):
    checkpoint_dir = tmp_path / "bench_ckpt"
    paths = bench_large._checkpoint_paths(checkpoint_dir)
    n, layers, _width = bench_large.resolve_size_and_layers("120", 12)

    bench_large._save_checkpoint_meta(paths["meta"], {"n": n + 1, "layers": layers})
    torch.save(torch.zeros((2, 0), dtype=torch.int32), paths["edge_index"])
    torch.save(torch.zeros((n, 2), dtype=torch.float16), paths["node_sizes"])

    assert bench_large._load_graph_checkpoint(paths, n, layers) is None


def test_layer_checkpoint_round_trip(tmp_path: Path):
    checkpoint_dir = tmp_path / "bench_ckpt"
    paths = bench_large._checkpoint_paths(checkpoint_dir)
    n, layers, _width = bench_large.resolve_size_and_layers("120", 12)
    layer_assignments = torch.arange(n, dtype=torch.long) % layers

    bench_large._save_checkpoint_meta(paths["meta"], {"n": n, "layers": layers})
    torch.save(layer_assignments, paths["layer_assignments"])

    restored = bench_large._load_layer_checkpoint(paths, n, layers)

    assert restored is not None
    assert torch.equal(restored, layer_assignments)


def test_duplicate_run_guard_raises_for_live_pid(tmp_path: Path, monkeypatch):
    checkpoint_dir = tmp_path / "bench_ckpt"
    paths = bench_large._checkpoint_paths(checkpoint_dir)
    paths["active_run"].parent.mkdir(parents=True, exist_ok=True)
    paths["active_run"].write_text('{"pid": 999999, "size": "1b"}', encoding="utf-8")
    monkeypatch.setattr(bench_large, "_find_existing_run_pid", lambda size: None)
    monkeypatch.setattr(bench_large, "_pid_alive", lambda pid: pid == 999999)

    import pytest

    with pytest.raises(SystemExit, match="duplicate large benchmark run"):
        bench_large._guard_duplicate_run(paths, "1b", resume=True, force_duplicate_run=False)


def test_duplicate_run_guard_uses_process_scan_without_active_run(tmp_path: Path, monkeypatch):
    checkpoint_dir = tmp_path / "bench_ckpt"
    paths = bench_large._checkpoint_paths(checkpoint_dir)
    monkeypatch.setattr(bench_large, "_find_existing_run_pid", lambda size: 424242)

    import pytest

    with pytest.raises(SystemExit, match="duplicate large benchmark run"):
        bench_large._guard_duplicate_run(paths, "1b", resume=True, force_duplicate_run=False)

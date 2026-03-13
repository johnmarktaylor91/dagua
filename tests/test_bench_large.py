"""Protective tests for the large benchmark helper script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from dagua.layout.multilevel import CoarseLevel


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


def test_load_graph_checkpoint_rejects_bad_shapes(tmp_path: Path):
    checkpoint_dir = tmp_path / "bench_ckpt"
    paths = bench_large._checkpoint_paths(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    bench_large._save_checkpoint_meta(paths["meta"], {"n": 8, "layers": 2})
    torch.save(torch.zeros(8, dtype=torch.int32), paths["edge_index"])
    torch.save(torch.zeros((8, 3), dtype=torch.float16), paths["node_sizes"])

    assert bench_large._load_graph_checkpoint(paths, n=8, layers=2) is None


def test_load_layer_checkpoint_rejects_bad_shape(tmp_path: Path):
    checkpoint_dir = tmp_path / "bench_ckpt"
    paths = bench_large._checkpoint_paths(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    bench_large._save_checkpoint_meta(paths["meta"], {"n": 8, "layers": 2})
    torch.save(torch.zeros((8, 1), dtype=torch.int32), paths["layer_assignments"])

    assert bench_large._load_layer_checkpoint(paths, n=8, layers=2) is None


def test_hierarchy_checkpoint_round_trip(tmp_path: Path):
    checkpoint_dir = tmp_path / "bench_ckpt"
    paths = bench_large._checkpoint_paths(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    bench_large._save_checkpoint_meta(paths["meta"], {"n": 12, "layers": 3})

    levels = [
        CoarseLevel(
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.int32),
            node_sizes=torch.full((4, 2), 20.0, dtype=torch.float16),
            num_nodes=4,
            fine_to_coarse=torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.int32),
            num_fine=6,
            fine_layer_assignments=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32),
            coarse_layer_assignments=torch.tensor([0, 1, 2, 2], dtype=torch.int32),
        )
    ]

    bench_large._save_hierarchy_checkpoint(paths, levels)
    restored = bench_large._load_hierarchy_checkpoint(paths, n=12, layers=3)

    assert restored is not None
    assert len(restored) == 1
    assert restored[0].num_nodes == 4
    assert torch.equal(restored[0].edge_index, levels[0].edge_index)
    assert torch.equal(restored[0].node_sizes, levels[0].node_sizes)
    assert torch.equal(restored[0].fine_to_coarse, levels[0].fine_to_coarse)
    assert torch.equal(restored[0].fine_layer_assignments, levels[0].fine_layer_assignments)
    assert torch.equal(restored[0].coarse_layer_assignments, levels[0].coarse_layer_assignments)


def test_hierarchy_checkpoint_saves_only_newest_level_each_time(tmp_path: Path, monkeypatch):
    checkpoint_dir = tmp_path / "bench_ckpt"
    paths = bench_large._checkpoint_paths(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    written_manifests: list[dict] = []

    def _fake_atomic_torch_save(path: Path, payload: object) -> None:
        saved_paths.append(path.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    def _fake_atomic_write_text(path: Path, payload: str) -> None:
        written_manifests.append(__import__("json").loads(payload))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    monkeypatch.setattr(bench_large, "_atomic_torch_save", _fake_atomic_torch_save)
    monkeypatch.setattr(bench_large, "_atomic_write_text", _fake_atomic_write_text)

    level0 = CoarseLevel(
        edge_index=torch.tensor([[0], [1]], dtype=torch.int32),
        node_sizes=torch.full((4, 2), 20.0, dtype=torch.float16),
        num_nodes=4,
        fine_to_coarse=torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.int32),
        num_fine=6,
        fine_layer_assignments=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32),
        coarse_layer_assignments=torch.tensor([0, 1, 2, 2], dtype=torch.int32),
    )
    level1 = CoarseLevel(
        edge_index=torch.tensor([[0], [1]], dtype=torch.int32),
        node_sizes=torch.full((2, 2), 20.0, dtype=torch.float16),
        num_nodes=2,
        fine_to_coarse=torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        num_fine=4,
        fine_layer_assignments=torch.tensor([0, 1, 2, 2], dtype=torch.int32),
        coarse_layer_assignments=torch.tensor([0, 1], dtype=torch.int32),
    )

    bench_large._save_hierarchy_checkpoint(paths, [level0])
    bench_large._save_hierarchy_checkpoint(paths, [level0, level1])

    assert saved_paths == ["level_00.pt", "level_01.pt"]
    assert written_manifests[0]["levels"] == ["level_00.pt"]
    assert written_manifests[1]["levels"] == ["level_00.pt", "level_01.pt"]


def test_atomic_torch_save_leaves_only_final_file(tmp_path: Path):
    target = tmp_path / "payload.pt"
    bench_large._atomic_torch_save(target, torch.tensor([1, 2, 3]))

    assert target.exists()
    assert torch.equal(torch.load(target), torch.tensor([1, 2, 3]))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["payload.pt"]


def test_positions_checkpoint_round_trip(tmp_path: Path):
    checkpoint_dir = tmp_path / "bench_ckpt"
    paths = bench_large._checkpoint_paths(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    bench_large._save_checkpoint_meta(paths["meta"], {"n": 8, "layers": 2})
    pos = torch.randn(8, 2)
    torch.save(pos, paths["positions"])

    restored = bench_large._load_positions_checkpoint(paths, n=8, layers=2)

    assert restored is not None
    assert torch.equal(restored, pos)


def test_coarsest_positions_checkpoint_round_trip(tmp_path: Path):
    checkpoint_dir = tmp_path / "bench_ckpt"
    paths = bench_large._checkpoint_paths(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    bench_large._save_checkpoint_meta(paths["meta"], {"n": 8, "layers": 2})
    pos = torch.randn(4, 2)
    torch.save(pos, paths["coarsest_positions"])

    restored = bench_large._load_coarsest_positions_checkpoint(paths, n=8, layers=2)

    assert restored is not None
    assert torch.equal(restored, pos)


def test_coarsest_positions_checkpoint_rejects_wrong_row_count(tmp_path: Path):
    checkpoint_dir = tmp_path / "bench_ckpt"
    paths = bench_large._checkpoint_paths(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    bench_large._save_checkpoint_meta(paths["meta"], {"n": 8, "layers": 2})
    levels = [
        CoarseLevel(
            edge_index=torch.tensor([[0], [1]], dtype=torch.int32),
            node_sizes=torch.full((4, 2), 20.0, dtype=torch.float16),
            num_nodes=4,
            fine_to_coarse=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32),
            num_fine=8,
            fine_layer_assignments=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32),
            coarse_layer_assignments=torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        )
    ]
    bench_large._save_hierarchy_checkpoint(paths, levels)
    torch.save(torch.randn(5, 2), paths["coarsest_positions"])

    assert bench_large._load_coarsest_positions_checkpoint(paths, n=8, layers=2) is None


def test_hierarchy_checkpoint_rejects_bad_level_shape(tmp_path: Path):
    checkpoint_dir = tmp_path / "bench_ckpt"
    paths = bench_large._checkpoint_paths(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    bench_large._save_checkpoint_meta(paths["meta"], {"n": 12, "layers": 3})
    paths["hierarchy_dir"].mkdir(parents=True, exist_ok=True)
    paths["hierarchy_meta"].write_text('{"num_levels": 1, "levels": ["level_00.pt"]}', encoding="utf-8")
    torch.save(
        {
            "edge_index": torch.tensor([0, 1], dtype=torch.int32),
            "node_sizes": torch.full((4, 2), 20.0, dtype=torch.float16),
            "num_nodes": 4,
            "fine_to_coarse": torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.int32),
            "num_fine": 6,
            "fine_layer_assignments": torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32),
            "coarse_layer_assignments": torch.tensor([0, 1, 2, 2], dtype=torch.int32),
        },
        paths["hierarchy_dir"] / "level_00.pt",
    )

    assert bench_large._load_hierarchy_checkpoint(paths, n=12, layers=3) is None


def test_hierarchy_checkpoint_rejects_bad_manifest_count(tmp_path: Path):
    checkpoint_dir = tmp_path / "bench_ckpt"
    paths = bench_large._checkpoint_paths(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    bench_large._save_checkpoint_meta(paths["meta"], {"n": 12, "layers": 3})
    paths["hierarchy_dir"].mkdir(parents=True, exist_ok=True)
    paths["hierarchy_meta"].write_text('{"num_levels": 2, "levels": ["level_00.pt"]}', encoding="utf-8")
    torch.save(
        {
            "edge_index": torch.tensor([[0], [1]], dtype=torch.int32),
            "node_sizes": torch.full((4, 2), 20.0, dtype=torch.float16),
            "num_nodes": 4,
            "fine_to_coarse": torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.int32),
            "num_fine": 6,
            "fine_layer_assignments": torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32),
            "coarse_layer_assignments": torch.tensor([0, 1, 2, 2], dtype=torch.int32),
        },
        paths["hierarchy_dir"] / "level_00.pt",
    )

    assert bench_large._load_hierarchy_checkpoint(paths, n=12, layers=3) is None


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


def test_find_existing_run_pid_ignores_shell_wrappers(monkeypatch):
    fake_ps = (
        "123 bash -c cd /repo && python -u scripts/bench_large.py 1b --device cuda --resume\n"
        "456 python -u scripts/bench_large.py 1b --device cuda --resume\n"
    )
    monkeypatch.setattr(bench_large.subprocess, "check_output", lambda *args, **kwargs: fake_ps)
    monkeypatch.setattr(bench_large, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(bench_large.os, "getpid", lambda: 999)

    assert bench_large._find_existing_run_pid("1b") == 456

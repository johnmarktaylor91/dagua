"""Tests for VRAM/RAM-aware layout allocation helpers."""

from __future__ import annotations

import importlib

import pytest
import torch


def test_auto_gpu_alloc_helpers_return_fallbacks_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU auto-sizing helpers should use conservative fallbacks without CUDA."""
    engine_module = importlib.import_module("dagua.layout.engine")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert engine_module._auto_edge_batch_size() == engine_module.DEFAULT_HYBRID_EDGE_BATCH
    assert engine_module._auto_sampled_node_cap() == engine_module.MIN_SAMPLED_ACTIVE_SET
    assert (
        engine_module._auto_subset_gpu_sampled_budget()
        == engine_module.SUBSET_GPU_SAMPLED_BASE_BYTES
    )


def test_auto_edge_batch_size_stays_within_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Edge batch sizing should stay inside the shared VRAM bounds."""
    engine_module = importlib.import_module("dagua.layout.engine")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1_024**4, 2 * 1_024**4))

    batch = engine_module._auto_edge_batch_size()

    assert engine_module._EDGE_BATCH_MIN <= batch <= engine_module._EDGE_BATCH_MAX


def test_auto_sampled_node_cap_stays_within_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sampled-node caps should stay inside the expected VRAM bounds."""
    engine_module = importlib.import_module("dagua.layout.engine")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1_024**4, 2 * 1_024**4))

    cap = engine_module._auto_sampled_node_cap()

    assert engine_module.MIN_SAMPLED_ACTIVE_SET <= cap <= engine_module.SAMPLED_NODE_CONTEXT_CAP


def test_auto_cpu_edge_batch_size_stays_within_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU edge batch sizing should stay inside the host-RAM bounds."""
    multilevel_module = importlib.import_module("dagua.layout.multilevel")

    monkeypatch.setattr(multilevel_module, "_available_ram_bytes", lambda: 1_024**4)

    batch = multilevel_module._auto_cpu_edge_batch_size()

    assert (
        multilevel_module._CPU_FINAL_EDGE_BATCH_CAP
        <= batch
        <= (multilevel_module._CPU_FINAL_EDGE_BATCH_MAX)
    )


def test_auto_cpu_edge_batch_size_returns_fallback_without_ram_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU edge sizing should use its fallback when RAM telemetry is unavailable."""
    multilevel_module = importlib.import_module("dagua.layout.multilevel")

    monkeypatch.setattr(multilevel_module, "_available_ram_bytes", lambda: 0)

    assert multilevel_module._auto_cpu_edge_batch_size() == (
        multilevel_module._CPU_FINAL_EDGE_BATCH_CAP
    )


def test_auto_gpu_alloc_helpers_scale_with_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    """VRAM-aware helpers should scale predictably with mocked free memory."""
    engine_module = importlib.import_module("dagua.layout.engine")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda: (8 * 1024**3, 16 * 1024**3),
    )

    edge_batch = engine_module._auto_edge_batch_size()
    sampled_cap = engine_module._auto_sampled_node_cap()
    sampled_budget = engine_module._auto_subset_gpu_sampled_budget()

    assert edge_batch == 42_949_672
    assert sampled_cap == 2_000_000
    assert sampled_budget == 2 * 1024**3


def test_auto_helpers_log_when_verbose(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verbose auto-sizing should emit the selected allocation decisions."""
    engine_module = importlib.import_module("dagua.layout.engine")
    multilevel_module = importlib.import_module("dagua.layout.multilevel")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda: (6 * 1024**3, 12 * 1024**3),
    )
    monkeypatch.setattr(multilevel_module, "_available_ram_bytes", lambda: 64 * 1024**3)

    engine_module._auto_edge_batch_size(verbose=True)
    engine_module._auto_sampled_node_cap(verbose=True)
    engine_module._auto_subset_gpu_sampled_budget(verbose=True)
    multilevel_module._auto_cpu_edge_batch_size(verbose=True)

    out = capsys.readouterr().out
    assert "Auto edge batch:" in out
    assert "Auto sampled node cap:" in out
    assert "Auto subset-GPU sampled budget:" in out
    assert "Auto CPU edge batch:" in out

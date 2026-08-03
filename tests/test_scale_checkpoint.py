"""Tests for scale checkpoint manifests and resume behavior."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.scale.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointManifest,
    ScaleCheckpointManager,
    checkpoint_is_armed,
)
from dagua.layout.scale.sketch import TopologySketch


def _chain_graph(num_nodes: int) -> DaguaGraph:
    """Build a deterministic chain graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.

    Returns
    -------
    DaguaGraph
        Graph with ``num_nodes - 1`` directed edges.
    """
    edges = [(node, node + 1) for node in range(num_nodes - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).transpose(0, 1).contiguous()
    return DaguaGraph.from_edge_index(edge_index, num_nodes=num_nodes)


def test_checkpoint_default_arms_only_at_10m() -> None:
    """Checkpointing is off below 10M unless explicitly requested."""
    config = LayoutConfig()

    assert not checkpoint_is_armed(config, 9_999_999)
    assert checkpoint_is_armed(config, 10_000_000)
    assert checkpoint_is_armed(LayoutConfig(algorithm_params={"scale_checkpoint": True}), 10)
    disabled = LayoutConfig(algorithm_params={"scale_checkpoint": False})
    assert not checkpoint_is_armed(disabled, 10_000_000)


def test_checkpoint_manifest_round_trip_and_lazy_load(tmp_path: Path) -> None:
    """Manifest schema round-trips and one level can be loaded independently."""
    graph = _chain_graph(8)
    sketch = TopologySketch.from_edge_index(graph.edge_index, graph.num_nodes, depth_cap=16)
    config = LayoutConfig(
        algorithm_params={
            "scale_checkpoint": True,
            "scale_checkpoint_root": str(tmp_path),
        }
    )
    manager = ScaleCheckpointManager.from_config(config, sketch, strategy="LAYERS")
    tensor = torch.arange(16, dtype=torch.float32).reshape(8, 2)

    manager.record(
        phase="final",
        level=0,
        tensors={"pos": tensor},
        telemetry={"purpose": "unit"},
    )

    payload = manager.manifest_path.read_text(encoding="utf-8")
    loaded = CheckpointManifest.from_dict(json.loads(payload))
    assert loaded.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert loaded.last_completed_level == 0
    reloaded = ScaleCheckpointManager.from_config(config, sketch, strategy="LAYERS")
    tensors = reloaded.load_tensors("final", 0)
    assert torch.equal(tensors["pos"], tensor)


def test_layers_checkpoint_resume_matches_uninterrupted(tmp_path: Path) -> None:
    """LAYERS final checkpoint resumes byte-identically to uninterrupted output."""
    graph = _chain_graph(32)
    base_config = LayoutConfig(
        algorithm_params={
            "scale_node_gate": 10,
            "scale_edge_gate": 10_000,
            "scale_checkpoint": True,
            "scale_checkpoint_root": str(tmp_path),
        },
        seed=42,
    )

    uninterrupted = dagua.layout(graph, base_config)
    resumed_graph = _chain_graph(32)
    resumed = dagua.layout(resumed_graph, base_config)

    assert torch.equal(uninterrupted, resumed)
    metadata = getattr(resumed_graph, "_dagua_scale_route_decision")
    assert metadata["layers"]["resumed"]

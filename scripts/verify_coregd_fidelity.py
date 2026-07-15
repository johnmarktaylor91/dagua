"""Verify CoRe-GD port correctness and report layout quality metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
REF_ROOT = Path.home() / "tools" / "dagua-refs" / "coregd"
REF_CONFIG = REF_ROOT / "configs" / "config_rome.json"
REF_CHECKPOINT = REF_ROOT / "checkpoints" / "core_rome.pt"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua import metrics  # noqa: E402
from dagua.layout.ops.pipelines.coregd import (  # noqa: E402
    _resolve_coregd_config,
    build_coregd_model,
    coregd_reference_forward,
    prepare_coregd_data,
)


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a deterministic path graph.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, max(N - 1, 0)]``.
    """
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(
        [list(range(num_nodes - 1)), list(range(1, num_nodes))],
        dtype=torch.long,
    )


def _load_reference_model(config: Any) -> torch.nn.Module:
    """Load the cloned reference model.

    Parameters
    ----------
    config : Any
        Reference-style config namespace.

    Returns
    -------
    torch.nn.Module
        Reference CoRe-GD model with checkpoint weights.
    """
    sys.path.insert(0, str(REF_ROOT))
    from neuraldrawer.network.model import get_model

    model = get_model(config)
    model.load_state_dict(torch.load(REF_CHECKPOINT, map_location=torch.device("cpu")))
    model.eval()
    return model


def _reference_data_from_port_features(
    edge_index: torch.Tensor,
    num_nodes: int,
    config: Any,
) -> Data:
    """Create reference ``Data`` with Dagua-prepared features.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    config : Any
        Dagua CoRe-GD config.

    Returns
    -------
    torch_geometric.data.Data
        Reference-compatible graph data.
    """
    data = prepare_coregd_data(edge_index, num_nodes, config, torch.device("cpu"))
    return Data(
        x=data.x.clone(),
        x_orig=data.x_orig.clone(),
        edge_index=data.edge_index.clone(),
        batch=data.batch.clone(),
        num_nodes=num_nodes,
    )


def _quality_scores(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> dict[str, float]:
    """Compute the requested quality score subset.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    dict[str, float]
        Stress, crossings, and neighborhood-preservation metrics.
    """
    stress = metrics.sampled_stress(pos, edge_index, num_nodes, n_sources=min(20, num_nodes))
    crossings = metrics.count_crossings(pos, edge_index)
    neighborhood = metrics.neighborhood_preservation(pos, edge_index, num_nodes, k=3)
    return {
        "sampled_stress": float(stress["sampled_stress"]),
        "edge_crossings": float(crossings),
        "neighborhood_preservation": float(neighborhood["neighborhood_mean"]),
    }


def main() -> int:
    """Run CoRe-GD port-correctness checks.

    Returns
    -------
    int
        Process exit status.
    """
    if not REF_CONFIG.exists() or not REF_CHECKPOINT.exists():
        print("pretrained_available: false")
        print(f"error: {REF_ROOT} checkpoints/configs are missing")
        return 1

    with REF_CONFIG.open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    reference_namespace = SimpleNamespace(**raw_config)
    port_config = _resolve_coregd_config(
        None,
        seed=42,
        config_path=str(REF_CONFIG),
        checkpoint_path=str(REF_CHECKPOINT),
        coarsen=False,
        iterations=3,
    )
    edge_index = _path_edge_index(24)

    reference_model = _load_reference_model(reference_namespace)
    reference_data = _reference_data_from_port_features(edge_index, 24, port_config)
    port_model = build_coregd_model(port_config)
    port_model.load_state_dict(torch.load(REF_CHECKPOINT, map_location=torch.device("cpu")))
    port_model.eval()

    with torch.no_grad():
        ref_pos = reference_model(
            reference_data,
            int(port_config.iterations),
            transform_to_undirected=True,
        )
        port_pos = port_model(
            reference_data,
            int(port_config.iterations),
            transform_to_undirected=True,
        )
    max_abs = float((ref_pos - port_pos).abs().max().item())
    exact = bool(torch.equal(ref_pos, port_pos))
    close = bool(torch.allclose(ref_pos, port_pos, atol=1.0e-6, rtol=1.0e-6))

    pipeline_pos = coregd_reference_forward(edge_index, 24, port_config, device="cpu")
    quality = _quality_scores(pipeline_pos, edge_index, 24)

    print("pretrained_available: true")
    print(f"checkpoint: {REF_CHECKPOINT}")
    print(f"port_correctness_exact: {exact}")
    print(f"port_correctness_allclose_1e-6: {close}")
    print(f"max_abs_residual: {max_abs:.8g}")
    print("first_divergent_stage: none" if close else "first_divergent_stage: model_forward")
    print("quality_scores:")
    for key, value in quality.items():
        print(f"  {key}: {value:.8g}")
    return 0 if close else 1


if __name__ == "__main__":
    raise SystemExit(main())

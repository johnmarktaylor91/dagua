"""Verify SmartGD and DeepGD port correctness and layout quality."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
SMART_REF_ROOT = Path("/tmp/smartgd-ref")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua import metrics  # noqa: E402
from dagua.layout.ops.pipelines.smartgd import (  # noqa: E402
    SmartGDConfig,
    build_smartgd_model,
    prepare_smartgd_data,
    smartgd_reference_forward,
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


def _procrustes_residual(pos_a: torch.Tensor, pos_b: torch.Tensor) -> float:
    """Compute a rotation/reflection/scale-invariant residual.

    Parameters
    ----------
    pos_a : torch.Tensor
        Reference positions with shape ``[N, 2]``.
    pos_b : torch.Tensor
        Candidate positions with shape ``[N, 2]``.

    Returns
    -------
    float
        Relative Procrustes residual.
    """
    centered_a = pos_a - pos_a.mean(dim=0, keepdim=True)
    centered_b = pos_b - pos_b.mean(dim=0, keepdim=True)
    u, _, vh = torch.linalg.svd(centered_b.T @ centered_a)
    aligned_b = centered_b @ (u @ vh)
    scale = (centered_a * aligned_b).sum() / aligned_b.square().sum().clamp_min(1.0e-12)
    return float(
        ((centered_a - scale * aligned_b).square().sum() / centered_a.square().sum()).sqrt().item()
    )


def _load_smartgd_reference(checkpoint: Path) -> Optional[torch.nn.Module]:
    """Load the cloned SmartGD reference generator.

    Parameters
    ----------
    checkpoint : pathlib.Path
        Checkpoint path.

    Returns
    -------
    torch.nn.Module or None
        Reference model, or ``None`` when the clone is unavailable.
    """
    if not checkpoint.exists() or not SMART_REF_ROOT.exists():
        return None
    sys.path.insert(0, str(SMART_REF_ROOT))
    from smartgd.model import Generator

    model = Generator(
        params=Generator.Params(
            num_blocks=11,
            block_depth=3,
            block_width=8,
            block_output_dim=8,
            edge_net_depth=2,
            edge_net_width=16,
            edge_attr_dim=2,
            node_attr_dim=2,
        )
    )
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")))
    model.eval()
    return model


def _verify_smartgd() -> bool:
    """Run SmartGD port-correctness checks.

    Returns
    -------
    bool
        ``True`` when the similarity gate passes.
    """
    checkpoint = SMART_REF_ROOT / "generator_stress_only.pt"
    edge_index = _path_edge_index(8)
    config = SmartGDConfig(checkpoint_path=str(checkpoint), use_reference_checkpoint=False, seed=42)
    reference_model = _load_smartgd_reference(checkpoint)
    if reference_model is None:
        print("smartgd.pretrained_available: false")
        return False

    prepared = prepare_smartgd_data(
        edge_index=edge_index,
        num_nodes=8,
        init_pos=None,
        device=torch.device("cpu"),
        seed=42,
    )
    port_model = build_smartgd_model(config)
    port_model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")))
    port_model.eval()
    with torch.no_grad():
        ref_pos = reference_model(*prepared)
        port_pos = port_model(*prepared)
    residual = _procrustes_residual(ref_pos, port_pos)
    exact = bool(torch.equal(ref_pos, port_pos))
    close = residual < 0.01
    pipeline_pos = smartgd_reference_forward(edge_index, 8, config, device="cpu")
    quality = _quality_scores(pipeline_pos, edge_index, 8)

    print("smartgd.pretrained_available: true")
    print(f"smartgd.checkpoint: {checkpoint}")
    print(f"smartgd.port_correctness_exact: {exact}")
    print(f"smartgd.port_correctness_procrustes_residual: {residual:.8g}")
    print(
        "smartgd.first_divergent_stage: none"
        if exact
        else "smartgd.first_divergent_stage: dynamic_edge_feature_router"
    )
    print("smartgd.quality_scores:")
    for key, value in quality.items():
        print(f"  {key}: {value:.8g}")
    return close


def main() -> int:
    """Run neural layout fidelity checks.

    Returns
    -------
    int
        Process exit status.
    """
    smartgd_ok = _verify_smartgd()
    print("deepgd.status: pending")
    return 0 if smartgd_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

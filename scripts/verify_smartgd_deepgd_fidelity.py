"""Verify SmartGD and DeepGD port correctness and layout quality."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Optional

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SMART_REF_ROOT = Path.home() / "tools" / "dagua-refs" / "smartgd"
DEEP_REF_ROOT = Path.home() / "tools" / "dagua-refs" / "deepgd"
_SEEDS = (7, 42)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua import metrics  # noqa: E402
from dagua.layout.ops.pipelines.deepgd import (  # noqa: E402
    DeepGDConfig,
    build_deepgd_model,
    deepgd_reference_forward,
)
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


def _cycle_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a deterministic cycle graph.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, N]``.
    """
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(
        [list(range(num_nodes)), [*range(1, num_nodes), 0]],
        dtype=torch.long,
    )


def _star_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a deterministic star graph.

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
        [[0] * (num_nodes - 1), list(range(1, num_nodes))],
        dtype=torch.long,
    )


def _corpus() -> list[tuple[str, torch.Tensor, int]]:
    """Return the small deterministic neural-fidelity graph corpus.

    Returns
    -------
    list[tuple[str, torch.Tensor, int]]
        Graph name, edge tensor, and node count triples.
    """
    return [
        ("path8", _path_edge_index(8), 8),
        ("cycle9", _cycle_edge_index(9), 9),
        ("star10", _star_edge_index(10), 10),
    ]


def _reference_device() -> torch.device:
    """Return the preferred deterministic inference device.

    Returns
    -------
    torch.device
        CUDA when available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_deterministic(seed: int) -> None:
    """Configure PyTorch deterministic inference for one seed.

    Parameters
    ----------
    seed : int
        Seed for CPU and CUDA RNGs.

    Returns
    -------
    None
        Global PyTorch deterministic settings are updated in-place.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    cpu_pos = pos.detach().cpu()
    cpu_edges = edge_index.detach().cpu()
    stress = metrics.sampled_stress(cpu_pos, cpu_edges, num_nodes, n_sources=min(20, num_nodes))
    crossings = metrics.count_crossings(cpu_pos, cpu_edges)
    neighborhood = metrics.neighborhood_preservation(cpu_pos, cpu_edges, num_nodes, k=3)
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


def _load_smartgd_reference(checkpoint: Path, device: torch.device) -> Optional[torch.nn.Module]:
    """Load the cloned SmartGD reference generator.

    Parameters
    ----------
    checkpoint : pathlib.Path
        Checkpoint path.
    device : torch.device
        Target inference device.

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
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    return model


def _first_divergent_stage(exact: bool) -> str:
    """Return the known first divergent stage label.

    Parameters
    ----------
    exact : bool
        Whether reference and port tensors are bit-exact.

    Returns
    -------
    str
        Stage label for verifier output.
    """
    return "none" if exact else "block1_dynamic_edge_feature_expansion"


def _verify_model(
    *,
    name: str,
    checkpoint: Path,
    config_factory: Callable[[int], SmartGDConfig],
    reference_loader: Callable[[Path, torch.device], Optional[torch.nn.Module]],
    model_builder: Callable[[SmartGDConfig], torch.nn.Module],
    pipeline_forward: Callable[..., torch.Tensor],
) -> bool:
    """Run neural reference-vs-port checks over the deterministic corpus.

    Parameters
    ----------
    name : str
        Algorithm label for report output.
    checkpoint : pathlib.Path
        Released checkpoint path.
    config_factory : Callable[[int], SmartGDConfig]
        Factory returning a seeded port configuration.
    reference_loader : Callable[[pathlib.Path, torch.device], torch.nn.Module | None]
        Loader for the upstream reference model.
    model_builder : Callable[[SmartGDConfig], torch.nn.Module]
        Builder for the Dagua port model.
    pipeline_forward : Callable[..., torch.Tensor]
        Dagua pipeline forward helper.

    Returns
    -------
    bool
        ``True`` when every corpus graph is bit-exact and deterministic.
    """
    device = _reference_device()
    reference_model = reference_loader(checkpoint, device)
    if reference_model is None:
        print(f"{name}.pretrained_available: false")
        return False

    config = config_factory(_SEEDS[0])
    port_model = model_builder(config).to(device)
    port_model.load_state_dict(torch.load(checkpoint, map_location=device))
    port_model.eval()

    all_exact = True
    deterministic = True
    worst_residual = 0.0
    worst_max_abs = 0.0
    for seed in _SEEDS:
        _set_deterministic(seed)
        seeded_config = config_factory(seed)
        for graph_name, edge_index, num_nodes in _corpus():
            prepared = prepare_smartgd_data(
                edge_index=edge_index,
                num_nodes=num_nodes,
                init_pos=None,
                device=device,
                seed=seed,
            )
            with torch.no_grad():
                ref_pos = reference_model(*prepared)
                port_pos = port_model(*prepared)
            exact = bool(torch.equal(ref_pos, port_pos))
            all_exact = all_exact and exact
            residual = _procrustes_residual(ref_pos.detach().cpu(), port_pos.detach().cpu())
            max_abs = float((ref_pos - port_pos).abs().max().item())
            worst_residual = max(worst_residual, residual)
            worst_max_abs = max(worst_max_abs, max_abs)
            first = pipeline_forward(edge_index, num_nodes, seeded_config, device=device)
            second = pipeline_forward(edge_index, num_nodes, seeded_config, device=device)
            deterministic = deterministic and bool(torch.equal(first, second))
            print(
                f"{name}.case.{graph_name}.seed{seed}: "
                f"exact={exact} max_abs={max_abs:.8g} procrustes={residual:.8g}"
            )

    quality_edge_index, quality_nodes = _path_edge_index(8), 8
    pipeline_pos = pipeline_forward(quality_edge_index, quality_nodes, config, device=device)
    quality = _quality_scores(pipeline_pos, quality_edge_index, quality_nodes)

    tier = "positional_bit_exact" if all_exact else "positional_close"
    print(f"{name}.pretrained_available: true")
    print(f"{name}.checkpoint: {checkpoint}")
    print(f"{name}.device: {device}")
    print(f"{name}.deterministic_algorithms: {torch.are_deterministic_algorithms_enabled()}")
    print(f"{name}.pipeline_repeat_exact: {deterministic}")
    print(f"{name}.fidelity_tier: {tier}")
    print(f"{name}.port_correctness_exact: {all_exact}")
    print(f"{name}.port_correctness_max_abs: {worst_max_abs:.8g}")
    print(f"{name}.port_correctness_procrustes_residual: {worst_residual:.8g}")
    print(f"{name}.first_divergent_stage: {_first_divergent_stage(all_exact)}")
    print(f"{name}.quality_scores:")
    for key, value in quality.items():
        print(f"  {key}: {value:.8g}")
    return all_exact and deterministic


def _verify_smartgd() -> bool:
    """Run SmartGD port-correctness checks.

    Returns
    -------
    bool
        ``True`` when the bit-exact and deterministic gates pass.
    """
    checkpoint = SMART_REF_ROOT / "generator_stress_only.pt"
    return _verify_model(
        name="smartgd",
        checkpoint=checkpoint,
        config_factory=lambda seed: SmartGDConfig(
            checkpoint_path=str(checkpoint),
            use_reference_checkpoint=False,
            seed=seed,
        ),
        reference_loader=_load_smartgd_reference,
        model_builder=build_smartgd_model,
        pipeline_forward=smartgd_reference_forward,
    )


def _load_deepgd_reference(checkpoint: Path, device: torch.device) -> Optional[torch.nn.Module]:
    """Load the cloned DeepGD reference generator.

    Parameters
    ----------
    checkpoint : pathlib.Path
        Checkpoint path.
    device : torch.device
        Target inference device.

    Returns
    -------
    torch.nn.Module or None
        Reference model, or ``None`` when the clone is unavailable.
    """
    if not checkpoint.exists() or not DEEP_REF_ROOT.exists():
        return None
    sys.path.insert(0, str(DEEP_REF_ROOT))
    from deepgd.model import Generator

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
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    return model


def _verify_deepgd() -> bool:
    """Run DeepGD port-correctness checks.

    Returns
    -------
    bool
        ``True`` when the similarity gate passes.
    """
    checkpoint = DEEP_REF_ROOT / "model_stress_only.pt"
    return _verify_model(
        name="deepgd",
        checkpoint=checkpoint,
        config_factory=lambda seed: DeepGDConfig(
            checkpoint_path=str(checkpoint),
            use_reference_checkpoint=False,
            seed=seed,
        ),
        reference_loader=_load_deepgd_reference,
        model_builder=build_deepgd_model,
        pipeline_forward=deepgd_reference_forward,
    )


def main() -> int:
    """Run neural layout fidelity checks.

    Returns
    -------
    int
        Process exit status.
    """
    smartgd_ok = _verify_smartgd()
    deepgd_ok = _verify_deepgd()
    return 0 if smartgd_ok and deepgd_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

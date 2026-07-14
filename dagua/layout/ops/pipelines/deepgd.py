"""DeepGD neural graph drawing generator pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.smartgd import (
    SmartGDConfig,
    SmartGDGenerator,
    build_smartgd_model,
    smartgd_reference_forward,
)

_DEFAULT_REFERENCE_ROOT = Path.home() / "tools" / "dagua-refs" / "deepgd"


@dataclass(frozen=True)
class DeepGDConfig(SmartGDConfig):
    """Configuration for the DeepGD generator pipeline.

    Parameters
    ----------
    checkpoint_path : str, optional
        Optional generator checkpoint path. When omitted, the cloned reference
        stress-only checkpoint is used when available.
    use_reference_checkpoint : bool, default=True
        Whether to load the cloned reference checkpoint when available.
    seed : int, default=42
        Deterministic seed for random initialization and fallback positions.
    """

    checkpoint_path: Optional[str] = None
    use_reference_checkpoint: bool = True
    seed: int = 42


def _resolve_deepgd_config(
    config: Optional[LayoutConfig],
    *,
    seed: Optional[int] = None,
    **kwargs: object,
) -> DeepGDConfig:
    """Resolve public algorithm parameters into ``DeepGDConfig``.

    Parameters
    ----------
    config : LayoutConfig, optional
        Public layout configuration.
    seed : int, optional
        Explicit seed override.
    **kwargs : object
        Algorithm parameter overrides.

    Returns
    -------
    DeepGDConfig
        Resolved configuration.
    """
    params: dict[str, object] = {}
    if config is not None:
        params.update(getattr(config, "algorithm_params", {}) or {})
        if getattr(config, "seed", None) is not None:
            params["seed"] = config.seed
    params.update(kwargs)
    if seed is not None:
        params["seed"] = seed
    valid = DeepGDConfig.__dataclass_fields__
    filtered = {key: value for key, value in params.items() if key in valid}
    return DeepGDConfig(**filtered)


def _checkpoint_for_config(config: DeepGDConfig) -> Optional[Path]:
    """Resolve the local DeepGD reference checkpoint path.

    Parameters
    ----------
    config : DeepGDConfig
        Pipeline configuration.

    Returns
    -------
    pathlib.Path or None
        Checkpoint path when configured and available.
    """
    if config.checkpoint_path:
        return Path(config.checkpoint_path)
    if not config.use_reference_checkpoint:
        return None
    candidate = _DEFAULT_REFERENCE_ROOT / "model_stress_only.pt"
    return candidate if candidate.exists() else None


def build_deepgd_model(config: DeepGDConfig) -> SmartGDGenerator:
    """Build a DeepGD generator model.

    Parameters
    ----------
    config : DeepGDConfig
        Model configuration.

    Returns
    -------
    SmartGDGenerator
        Generator model.
    """
    return build_smartgd_model(SmartGDConfig(**config.__dict__))


def deepgd_reference_forward(
    edge_index: torch.Tensor,
    num_nodes: int,
    config: DeepGDConfig,
    *,
    init_pos: Optional[torch.Tensor] = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Run the Dagua DeepGD generator forward pass.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input graph edges with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    config : DeepGDConfig
        Generator configuration.
    init_pos : torch.Tensor, optional
        Optional warm-start positions with shape ``[N, 2]``.
    device : str or torch.device, default="cpu"
        Target device.

    Returns
    -------
    torch.Tensor
        Predicted positions with shape ``[N, 2]``.
    """
    checkpoint = _checkpoint_for_config(config)
    smart_config = SmartGDConfig(
        **{
            **config.__dict__,
            "checkpoint_path": str(checkpoint) if checkpoint is not None else None,
            "use_reference_checkpoint": False,
        }
    )
    return smartgd_reference_forward(
        edge_index=edge_index,
        num_nodes=num_nodes,
        config=smart_config,
        init_pos=init_pos,
        device=device,
    )


def layout_deepgd_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    *,
    config: Optional[LayoutConfig] = None,
    seed: Optional[int] = None,
    init_pos: Optional[torch.Tensor] = None,
    device: str | torch.device = "cpu",
    **kwargs: object,
) -> torch.Tensor:
    """Compute a layout with the DeepGD generator.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input graph edges with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``. Unused by DeepGD.
    config : LayoutConfig, optional
        Public layout configuration.
    seed : int, optional
        Seed override.
    init_pos : torch.Tensor, optional
        Optional warm-start positions with shape ``[N, 2]``.
    device : str or torch.device, default="cpu"
        Target device.
    **kwargs : object
        Algorithm parameter overrides.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    del node_sizes
    resolved = _resolve_deepgd_config(config, seed=seed, **kwargs)
    return deepgd_reference_forward(
        edge_index=edge_index,
        num_nodes=num_nodes,
        config=resolved,
        init_pos=init_pos,
        device=device,
    )

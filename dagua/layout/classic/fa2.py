"""ForceAtlas2 — Gephi's continuous force-directed layout.

Key differences from Fruchterman-Reingold:
- Gravity: constant pull toward the center of mass (prevents drift)
- Degree-dependent repulsion: high-degree nodes repel more strongly
- Adaptive speed: step size adjusts based on energy oscillation
- LinLog mode: uses log-attraction for better cluster separation
- No cooling schedule: runs until convergence via adaptive speed

Reference: Jacomy et al., "ForceAtlas2, a Continuous Graph Layout Algorithm
for Handy Network Visualization" (2014), PLoS ONE.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch

_EPSILON = 1e-2
_EXACT_REPULSION_THRESHOLD = 10_000
_EXACT_BLOCK_SIZE = 1_024
_SAMPLED_REPULSION_K = 1_000
_SAMPLED_BLOCK_SIZE = 256
_MAX_SPEED = 10.0


def layout_fa2(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 500,
    seed: int = 42,
    gravity: float = 1.0,
    scaling_ratio: float = 2.0,
    linlog: bool = False,
    strong_gravity: bool = False,
    trace_every: int = 0,
) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
    """Run ForceAtlas2 layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list, shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Unused, accepted for interface compatibility.
    steps : int
        Number of simulation steps.
    seed : int
        Random seed for initial placement.
    gravity : float
        Gravity constant — pulls nodes toward center of mass.
    scaling_ratio : float
        Repulsion scaling. Higher = more spread out.
    linlog : bool
        Use log-attraction for better cluster separation.
    strong_gravity : bool
        Gravity proportional to degree (prevents isolated nodes from drifting).
    trace_every : int
        If > 0, record snapshots every N steps.

    Returns
    -------
    torch.Tensor or tuple
        Final positions ``[N, 2]``, or ``(positions, traces)`` if tracing.
    """
    del node_sizes

    _validate_inputs(
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=steps,
        trace_every=trace_every,
    )

    device = edge_index.device
    if num_nodes == 0:
        empty = torch.zeros((0, 2), dtype=torch.float32, device=device)
        return (empty, []) if trace_every > 0 else empty
    if num_nodes == 1:
        single = torch.zeros((1, 2), dtype=torch.float32, device=device)
        return (single, [single.clone()]) if trace_every > 0 else single

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    degree = _compute_degree(edge_index=edge_index, num_nodes=num_nodes)
    pos = torch.randn(
        (num_nodes, 2),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    pos = pos * 10.0

    speed = 1.0
    speed_efficiency = 1.0
    jitter_tolerance = 1.0
    previous_force = torch.zeros_like(pos)
    traces: list[torch.Tensor] = []

    for step_index in range(steps):
        repulsion_force = _repulsion_force(
            pos=pos,
            degree=degree,
            scaling_ratio=scaling_ratio,
            generator=generator,
        )
        attraction_force = _attraction_force(pos=pos, edge_index=edge_index, linlog=linlog)
        gravity_force = _gravity_force(
            pos=pos,
            degree=degree,
            gravity=gravity,
            strong_gravity=strong_gravity,
        )
        force = repulsion_force + attraction_force + gravity_force

        speed, speed_efficiency = _adaptive_speed(
            force=force,
            previous_force=previous_force,
            speed=speed,
            speed_efficiency=speed_efficiency,
            jitter_tolerance=jitter_tolerance,
        )
        pos = pos + (force * speed)
        previous_force = force

        if trace_every > 0 and step_index % trace_every == 0:
            traces.append(pos.clone())

    if trace_every > 0:
        return pos, traces
    return pos


def _validate_inputs(
    edge_index: torch.Tensor,
    num_nodes: int,
    steps: int,
    trace_every: int,
) -> None:
    """Validate public layout inputs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list tensor.
    num_nodes : int
        Declared number of nodes.
    steps : int
        Number of simulation iterations.
    trace_every : int
        Snapshot cadence.

    Returns
    -------
    None
        Raises ``ValueError`` when inputs are invalid.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative")
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("edge_index must use an integer dtype")
    if edge_index.numel() == 0:
        return

    min_index = int(edge_index.min().item())
    max_index = int(edge_index.max().item())
    if min_index < 0:
        raise ValueError("edge_index cannot contain negative node indices")
    if max_index >= num_nodes:
        raise ValueError("edge_index contains node indices outside num_nodes")


def _compute_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute undirected node degree used as FA2 mass.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list, shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Degree tensor of shape ``[N]`` with minimum value ``1``.
    """
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    if edge_index.numel() == 0:
        return degree.add(1.0)

    ones = torch.ones(edge_index.shape[1], dtype=torch.float32, device=edge_index.device)
    degree.scatter_add_(0, edge_index[0], ones)
    degree.scatter_add_(0, edge_index[1], ones)
    return degree.clamp(min=1.0)


def _repulsion_force(
    pos: torch.Tensor,
    degree: torch.Tensor,
    scaling_ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Compute degree-weighted node repulsion.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions, shape ``[N, 2]``.
    degree : torch.Tensor
        Node degree tensor, shape ``[N]``.
    scaling_ratio : float
        Repulsion strength multiplier.
    generator : torch.Generator
        Random generator for deterministic sampling.

    Returns
    -------
    torch.Tensor
        Repulsive force tensor of shape ``[N, 2]``.
    """
    num_nodes = pos.shape[0]
    if num_nodes <= 1:
        return torch.zeros_like(pos)

    if num_nodes <= _EXACT_REPULSION_THRESHOLD:
        return _repulsion_force_exact(pos=pos, degree=degree, scaling_ratio=scaling_ratio)

    return _repulsion_force_sampled(
        pos=pos,
        degree=degree,
        scaling_ratio=scaling_ratio,
        generator=generator,
    )


def _repulsion_force_exact(
    pos: torch.Tensor,
    degree: torch.Tensor,
    scaling_ratio: float,
) -> torch.Tensor:
    """Compute exact all-pairs repulsion in blocks to control memory.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions, shape ``[N, 2]``.
    degree : torch.Tensor
        Node degree tensor, shape ``[N]``.
    scaling_ratio : float
        Repulsion strength multiplier.

    Returns
    -------
    torch.Tensor
        Repulsive force tensor of shape ``[N, 2]``.
    """
    num_nodes = pos.shape[0]
    mass = degree + 1.0
    force = torch.zeros_like(pos)

    for start in range(0, num_nodes, _EXACT_BLOCK_SIZE):
        end = min(start + _EXACT_BLOCK_SIZE, num_nodes)
        block = pos[start:end]
        delta = block.unsqueeze(1) - pos.unsqueeze(0)
        dist = delta.norm(dim=2).clamp(min=_EPSILON)
        repulsion_mag = scaling_ratio * mass[start:end].unsqueeze(1) * mass.unsqueeze(0) / dist
        force[start:end] = (delta / dist.unsqueeze(2) * repulsion_mag.unsqueeze(2)).sum(dim=1)

    return force


def _repulsion_force_sampled(
    pos: torch.Tensor,
    degree: torch.Tensor,
    scaling_ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Approximate repulsion with deterministic negative sampling.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions, shape ``[N, 2]``.
    degree : torch.Tensor
        Node degree tensor, shape ``[N]``.
    scaling_ratio : float
        Repulsion strength multiplier.
    generator : torch.Generator
        Random generator for deterministic sampling.

    Returns
    -------
    torch.Tensor
        Repulsive force tensor of shape ``[N, 2]``.
    """
    num_nodes = pos.shape[0]
    sample_k = min(_SAMPLED_REPULSION_K, max(num_nodes - 1, 1))
    if sample_k == 0:
        return torch.zeros_like(pos)

    mass = degree + 1.0
    force = torch.zeros_like(pos)
    all_indices = torch.arange(num_nodes, device=pos.device)
    scale = float(num_nodes - 1) / float(sample_k)

    for start in range(0, num_nodes, _SAMPLED_BLOCK_SIZE):
        end = min(start + _SAMPLED_BLOCK_SIZE, num_nodes)
        batch_size = end - start
        self_idx = all_indices[start:end].unsqueeze(1).expand(batch_size, sample_k)
        raw_idx = torch.randint(
            0,
            num_nodes - 1,
            (batch_size, sample_k),
            generator=generator,
            device=pos.device,
        )
        sampled_idx = raw_idx + (raw_idx >= self_idx).to(raw_idx.dtype)

        delta = pos[start:end].unsqueeze(1) - pos[sampled_idx]
        dist = delta.norm(dim=2).clamp(min=_EPSILON)
        repulsion_mag = scaling_ratio * mass[start:end].unsqueeze(1) * mass[sampled_idx] / dist
        force[start:end] = scale * (delta / dist.unsqueeze(2) * repulsion_mag.unsqueeze(2)).sum(
            dim=1
        )

    return force


def _attraction_force(pos: torch.Tensor, edge_index: torch.Tensor, linlog: bool) -> torch.Tensor:
    """Compute edge attraction forces.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions, shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge list, shape ``[2, E]``.
    linlog : bool
        Whether to use the LinLog attraction variant.

    Returns
    -------
    torch.Tensor
        Attraction force tensor of shape ``[N, 2]``.
    """
    force = torch.zeros_like(pos)
    if edge_index.numel() == 0:
        return force

    src, tgt = edge_index
    delta = pos[tgt] - pos[src]
    dist = delta.norm(dim=1).clamp(min=_EPSILON)
    attr_mag = torch.log1p(dist) if linlog else dist
    attr_force = delta / dist.unsqueeze(1) * attr_mag.unsqueeze(1)

    force.scatter_add_(0, src.unsqueeze(1).expand_as(attr_force), attr_force)
    force.scatter_add_(0, tgt.unsqueeze(1).expand_as(attr_force), -attr_force)
    return force


def _gravity_force(
    pos: torch.Tensor,
    degree: torch.Tensor,
    gravity: float,
    strong_gravity: bool,
) -> torch.Tensor:
    """Compute center-seeking gravity.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions, shape ``[N, 2]``.
    degree : torch.Tensor
        Node degree tensor, shape ``[N]``.
    gravity : float
        Gravity constant.
    strong_gravity : bool
        Whether to scale gravity by node degree.

    Returns
    -------
    torch.Tensor
        Gravity force tensor of shape ``[N, 2]``.
    """
    center = pos.mean(dim=0, keepdim=True)
    delta = center - pos
    if strong_gravity:
        return gravity * (degree + 1.0).unsqueeze(1) * delta

    dist = delta.norm(dim=1).clamp(min=_EPSILON)
    return gravity * delta / dist.unsqueeze(1)


def _adaptive_speed(
    force: torch.Tensor,
    previous_force: torch.Tensor,
    speed: float,
    speed_efficiency: float,
    jitter_tolerance: float,
) -> tuple[float, float]:
    """Update the FA2 integration speed from force oscillation.

    Parameters
    ----------
    force : torch.Tensor
        Current force tensor, shape ``[N, 2]``.
    previous_force : torch.Tensor
        Previous force tensor, shape ``[N, 2]``.
    speed : float
        Current integration speed.
    speed_efficiency : float
        Current speed efficiency multiplier.
    jitter_tolerance : float
        Jitter tolerance hyperparameter.

    Returns
    -------
    tuple[float, float]
        Updated ``(speed, speed_efficiency)`` pair.
    """
    swing = (force - previous_force).norm(dim=1).sum().item()
    traction = (0.5 * (force + previous_force)).norm(dim=1).sum().item()
    traction = max(traction, 1e-6)

    estimated_optimal_jitter = 0.05 * traction
    jitter = jitter_tolerance * max(estimated_optimal_jitter, 0.01)
    ratio = swing / traction

    if ratio > 2.0 and speed_efficiency > 0.05:
        speed_efficiency *= 0.5
    elif ratio < 1.0:
        speed_efficiency = min(speed_efficiency * 1.05, 1.5)

    speed = jitter * speed_efficiency / (1.0 + jitter * math.sqrt(max(ratio, 0.0)))
    return min(speed, _MAX_SPEED), speed_efficiency

"""ForceAtlas2 — Gephi's continuous force-directed layout.

Key differences from Fruchterman-Reingold:
- Gravity: pulls nodes toward the origin rather than the centroid
- Degree-dependent repulsion: high-degree nodes repel more strongly
- Adaptive speed: step size adjusts based on energy oscillation
- LinLog mode: uses log-attraction for better cluster separation
- No cooling schedule: runs until convergence via adaptive speed

Reference: Jacomy et al., "ForceAtlas2, a Continuous Graph Layout Algorithm
for Handy Network Visualization" (2014), PLoS ONE, and the ``fa2`` package.
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


def layout_fa2(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 100,
    seed: int = 42,
    gravity: float = 1.0,
    scaling_ratio: float = 2.0,
    linlog: bool = False,
    strong_gravity: bool = False,
    trace_every: int = 0,
    outbound_attraction_distribution: bool = True,
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
    steps : int, default=100
        Number of simulation steps.
    seed : int, default=42
        Random seed for initial placement.
    gravity : float, default=1.0
        Gravity constant.
    scaling_ratio : float, default=2.0
        Repulsion scaling. Higher spreads nodes farther apart.
    linlog : bool, default=False
        Use the LinLog attraction variant.
    strong_gravity : bool, default=False
        Use the strong-gravity mode from the reference implementation.
    trace_every : int, default=0
        If greater than zero, record snapshots every ``trace_every`` steps.
    outbound_attraction_distribution : bool, default=True
        Divide attraction by source-node mass and compensate using the mean
        node mass, matching the reference adapter defaults.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, list[torch.Tensor]]
        Final positions with shape ``[N, 2]``, or ``(positions, traces)`` when
        tracing is enabled.
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

    generator_device = device.type if device.type != "mps" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)

    degree = _compute_degree(edge_index=edge_index, num_nodes=num_nodes)
    mass = degree + 1.0
    attraction_coefficient = float(mass.mean().item()) if outbound_attraction_distribution else 1.0
    pos = torch.rand(
        (num_nodes, 2),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    speed = 1.0
    speed_efficiency = 1.0
    jitter_tolerance = 1.0
    previous_force = torch.zeros_like(pos)
    traces: list[torch.Tensor] = []

    for step_index in range(steps):
        repulsion_force = _repulsion_force(
            pos=pos,
            mass=mass,
            scaling_ratio=scaling_ratio,
            generator=generator,
        )
        attraction_force = _attraction_force(
            pos=pos,
            edge_index=edge_index,
            mass=mass,
            linlog=linlog,
            attraction_coefficient=attraction_coefficient,
            outbound_attraction_distribution=outbound_attraction_distribution,
        )
        gravity_force = _gravity_force(
            pos=pos,
            mass=mass,
            gravity=gravity,
            strong_gravity=strong_gravity,
            scaling_ratio=scaling_ratio,
        )
        force = repulsion_force + attraction_force + gravity_force

        speed, speed_efficiency, node_traction = _adaptive_speed(
            force=force,
            previous_force=previous_force,
            speed=speed,
            speed_efficiency=speed_efficiency,
            jitter_tolerance=jitter_tolerance,
            mass=mass,
            outbound_attraction_distribution=outbound_attraction_distribution,
        )
        node_speed = _node_speed(
            speed=speed,
            node_traction=node_traction,
            force=force,
            previous_force=previous_force,
            mass=mass,
        )
        pos = pos + (force * node_speed.unsqueeze(1))
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
    """Compute deduplicated undirected degree counts.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list, shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Degree tensor of shape ``[N]``.
    """
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    if edge_index.numel() == 0:
        return degree

    source = edge_index[0]
    target = edge_index[1]
    non_self = source != target
    if not bool(non_self.any().item()):
        return degree

    source = source[non_self]
    target = target[non_self]
    lower = torch.minimum(source, target)
    upper = torch.maximum(source, target)
    unique_edges = torch.unique(torch.stack([lower, upper], dim=1), dim=0)
    ones = torch.ones(unique_edges.shape[0], dtype=torch.float32, device=edge_index.device)
    degree.scatter_add_(0, unique_edges[:, 0], ones)
    degree.scatter_add_(0, unique_edges[:, 1], ones)
    return degree


def _repulsion_force(
    pos: torch.Tensor,
    mass: torch.Tensor,
    scaling_ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Compute degree-weighted node repulsion.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions, shape ``[N, 2]``.
    mass : torch.Tensor
        Node masses, shape ``[N]``.
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
        return _repulsion_force_exact(pos=pos, mass=mass, scaling_ratio=scaling_ratio)

    return _repulsion_force_sampled(
        pos=pos,
        mass=mass,
        scaling_ratio=scaling_ratio,
        generator=generator,
    )


def _repulsion_force_exact(
    pos: torch.Tensor,
    mass: torch.Tensor,
    scaling_ratio: float,
) -> torch.Tensor:
    """Compute exact all-pairs repulsion in blocks.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions, shape ``[N, 2]``.
    mass : torch.Tensor
        Node masses, shape ``[N]``.
    scaling_ratio : float
        Repulsion strength multiplier.

    Returns
    -------
    torch.Tensor
        Repulsive force tensor of shape ``[N, 2]``.
    """
    num_nodes = pos.shape[0]
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
    mass: torch.Tensor,
    scaling_ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Approximate repulsion with deterministic negative sampling.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions, shape ``[N, 2]``.
    mass : torch.Tensor
        Node masses, shape ``[N]``.
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


def _attraction_force(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    mass: torch.Tensor,
    linlog: bool,
    attraction_coefficient: float,
    outbound_attraction_distribution: bool,
) -> torch.Tensor:
    """Compute edge attraction forces.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions, shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge list, shape ``[2, E]``.
    mass : torch.Tensor
        Node masses, shape ``[N]``.
    linlog : bool
        Whether to use the LinLog attraction variant.
    attraction_coefficient : float
        Global attraction coefficient.
    outbound_attraction_distribution : bool
        Whether to divide by the source-node mass.

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

    coefficient = torch.full_like(dist, float(attraction_coefficient))
    if outbound_attraction_distribution:
        coefficient = coefficient / mass.index_select(0, src)
    if linlog:
        coefficient = coefficient * (torch.log1p(dist) / dist)

    attr_force = delta * coefficient.unsqueeze(1)
    force.scatter_add_(0, src.unsqueeze(1).expand_as(attr_force), attr_force)
    force.scatter_add_(0, tgt.unsqueeze(1).expand_as(attr_force), -attr_force)
    return force


def _gravity_force(
    pos: torch.Tensor,
    mass: torch.Tensor,
    gravity: float,
    strong_gravity: bool,
    scaling_ratio: float,
) -> torch.Tensor:
    """Compute gravity toward the origin.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions, shape ``[N, 2]``.
    mass : torch.Tensor
        Node masses, shape ``[N]``.
    gravity : float
        Gravity constant.
    strong_gravity : bool
        Whether to use strong gravity mode.
    scaling_ratio : float
        Repulsion scaling used by strong gravity mode in the reference.

    Returns
    -------
    torch.Tensor
        Gravity force tensor of shape ``[N, 2]``.
    """
    dist = pos.norm(dim=1).clamp(min=_EPSILON)
    if strong_gravity:
        factor = scaling_ratio * mass * gravity
    else:
        factor = mass * gravity / dist
    return -pos * factor.unsqueeze(1)


def _adaptive_speed(
    force: torch.Tensor,
    previous_force: torch.Tensor,
    speed: float,
    speed_efficiency: float,
    jitter_tolerance: float,
    mass: torch.Tensor,
    outbound_attraction_distribution: bool,
) -> tuple[float, float, torch.Tensor]:
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
        Current speed-efficiency multiplier.
    jitter_tolerance : float
        Jitter tolerance hyperparameter.
    mass : torch.Tensor
        Node masses, shape ``[N]``.
    outbound_attraction_distribution : bool
        Whether the outbound-attraction heuristic is enabled.

    Returns
    -------
    tuple[float, float, torch.Tensor]
        Updated ``(speed, speed_efficiency, node_traction)`` tuple.
    """
    node_swing = mass * (force - previous_force).norm(dim=1)
    node_traction = 0.5 * mass * (force + previous_force).norm(dim=1)
    swing = max(float(node_swing.sum().item()), _EPSILON)
    traction = max(float(node_traction.sum().item()), _EPSILON)

    num_nodes = float(force.shape[0])
    estimated_optimal_jt = 0.05 * math.sqrt(num_nodes)
    min_jt = math.sqrt(estimated_optimal_jt)
    max_jt = 10.0
    jt = jitter_tolerance * max(
        min_jt,
        min(max_jt, estimated_optimal_jt * traction / (num_nodes * num_nodes)),
    )
    if outbound_attraction_distribution:
        jt = min(jt, 1.0)

    if swing / traction > 2.0 and speed_efficiency > 0.05:
        speed_efficiency *= 0.5
        jt = max(jt, jitter_tolerance)

    target_speed = jt * speed_efficiency * traction / swing
    if swing > jt * traction:
        if speed_efficiency > 0.05:
            speed_efficiency *= 0.7
    elif speed < 1000.0:
        speed_efficiency *= 1.3

    speed = speed + min(target_speed - speed, 0.5 * max(speed, _EPSILON))
    return speed, speed_efficiency, node_traction


def _node_speed(
    force: torch.Tensor,
    previous_force: torch.Tensor,
    speed: float,
    global_traction: Optional[float] = None,
    node_traction: Optional[torch.Tensor] = None,
    mass: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-node movement factors.

    Parameters
    ----------
    force : torch.Tensor
        Current force tensor, shape ``[N, 2]``.
    previous_force : torch.Tensor
        Previous force tensor, shape ``[N, 2]``.
    speed : float
        Global adaptive speed from ``_adaptive_speed()``.
    global_traction : float, optional
        Legacy scalar used by existing reference tests.
    node_traction : torch.Tensor, optional
        Per-node traction magnitudes, shape ``[N]``.
    mass : torch.Tensor, optional
        Node masses, shape ``[N]``.

    Returns
    -------
    torch.Tensor
        Per-node movement factors, shape ``[N]``.
    """
    node_swing = (force - previous_force).norm(dim=1)
    if mass is not None:
        node_swing = node_swing * mass

    if node_traction is None and global_traction is not None:
        return (speed * global_traction) / (node_swing + 1.0)
    factor = speed / (1.0 + torch.sqrt((speed * node_swing).clamp(min=0.0)))
    if node_traction is None:
        return factor
    return torch.where(node_traction > 0, factor, torch.zeros_like(factor))

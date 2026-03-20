"""ForceAtlas2 translated directly from ``fa2_modified`` reference code."""

from __future__ import annotations

import math
from typing import Optional, Union

import torch


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
    """Run a vectorized ForceAtlas2 stepper matching ``fa2_modified``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the layout.
    node_sizes : torch.Tensor, optional
        Unused placeholder kept for API compatibility.
    steps : int, default=100
        Number of ForceAtlas2 iterations.
    seed : int, default=42
        Seed for the uniform ``[0, 1]`` initializer.
    gravity : float, default=1.0
        Gravity coefficient from the reference implementation.
    scaling_ratio : float, default=2.0
        Repulsion scaling coefficient.
    linlog : bool, default=False
        Kept for signature compatibility. The reference backend does not
        implement LinLog mode.
    strong_gravity : bool, default=False
        Whether to use strong-gravity mode.
    trace_every : int, default=0
        Record a snapshot every ``trace_every`` iterations when positive.
    outbound_attraction_distribution : bool, default=True
        Whether to divide attraction by source-node mass and compensate using
        the mean mass.

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
    if linlog:
        raise ValueError("linlog=True is not supported by the fa2_modified reference.")

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

    undirected_edges = _unique_undirected_edges(edge_index)
    masses = _compute_degree(undirected_edges, num_nodes=num_nodes) + 1.0
    outbound_att_compensation = (
        float(masses.mean().item()) if outbound_attraction_distribution else 1.0
    )
    # Match reference: fa2_modified uses Python's random.random() for init,
    # not torch.rand. We must use the same RNG for identical trajectories.
    import random as _random

    _random.seed(seed)
    pos = torch.tensor(
        [[_random.random(), _random.random()] for _ in range(num_nodes)],
        dtype=torch.float32,
        device=device,
    )

    speed = 1.0
    speed_efficiency = 1.0
    jitter_tolerance = 1.0
    old_force = torch.zeros_like(pos)
    traces: list[torch.Tensor] = []

    for step_index in range(steps):
        force = torch.zeros_like(pos)
        force = force + _repulsion_force(pos=pos, mass=masses, scaling_ratio=scaling_ratio)
        force = force + _gravity_force(
            pos=pos,
            mass=masses,
            gravity=gravity,
            strong_gravity=strong_gravity,
            scaling_ratio=scaling_ratio,
        )
        force = force + _attraction_force(
            pos=pos,
            edge_index=undirected_edges,
            mass=masses,
            outbound_att_compensation=outbound_att_compensation,
            outbound_attraction_distribution=outbound_attraction_distribution,
        )
        pos, speed, speed_efficiency = _adjust_speed_and_apply_forces(
            pos=pos,
            force=force,
            old_force=old_force,
            mass=masses,
            speed=speed,
            speed_efficiency=speed_efficiency,
            jitter_tolerance=jitter_tolerance,
        )
        old_force = force

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
    """Validate public layout arguments.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list tensor with shape ``[2, E]``.
    num_nodes : int
        Declared node count.
    steps : int
        Number of optimization steps.
    trace_every : int
        Snapshot cadence.

    Returns
    -------
    None
        Raises ``ValueError`` for invalid inputs.
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


def _unique_undirected_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Collapse a directed edge list into unique undirected edges.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.

    Returns
    -------
    torch.Tensor
        Unique undirected edge list with shape ``[2, E_unique]``.
    """
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    source = edge_index[0].to(dtype=torch.long)
    target = edge_index[1].to(dtype=torch.long)
    non_self = source != target
    if not bool(non_self.any().item()):
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    source = source[non_self]
    target = target[non_self]
    lower = torch.minimum(source, target)
    upper = torch.maximum(source, target)
    unique_pairs = torch.unique(torch.stack([lower, upper], dim=1), dim=0)
    return unique_pairs.transpose(0, 1).contiguous()


def _compute_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute deduplicated undirected degree counts.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Degree tensor with shape ``[N]``.
    """
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    if edge_index.numel() == 0:
        return degree

    undirected_edges = _unique_undirected_edges(edge_index)
    if undirected_edges.numel() == 0:
        return degree

    ones = torch.ones(undirected_edges.shape[1], dtype=torch.float32, device=edge_index.device)
    degree.scatter_add_(0, undirected_edges[0], ones)
    degree.scatter_add_(0, undirected_edges[1], ones)
    return degree


def _repulsion_force(pos: torch.Tensor, mass: torch.Tensor, scaling_ratio: float) -> torch.Tensor:
    """Compute exact all-pairs ForceAtlas2 repulsion.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    mass : torch.Tensor
        Node masses with shape ``[N]``.
    scaling_ratio : float
        Repulsion coefficient.

    Returns
    -------
    torch.Tensor
        Repulsive displacements with shape ``[N, 2]``.
    """
    if pos.shape[0] <= 1:
        return torch.zeros_like(pos)

    delta = pos.unsqueeze(1) - pos.unsqueeze(0)
    distance = torch.cdist(pos, pos, p=2.0)
    distance_sq = distance.square()
    factor = torch.zeros_like(distance_sq)
    valid = distance_sq > 0
    mass_product = mass.unsqueeze(1) * mass.unsqueeze(0)
    factor[valid] = scaling_ratio * mass_product[valid] / distance_sq[valid]
    return (delta * factor.unsqueeze(2)).sum(dim=1)


def _gravity_force(
    pos: torch.Tensor,
    mass: torch.Tensor,
    gravity: float,
    strong_gravity: bool,
    scaling_ratio: float,
) -> torch.Tensor:
    """Compute ForceAtlas2 gravity toward the origin.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    mass : torch.Tensor
        Node masses with shape ``[N]``.
    gravity : float
        Gravity coefficient.
    strong_gravity : bool
        Whether to use strong-gravity mode.
    scaling_ratio : float
        Reference coefficient used only in strong-gravity mode.

    Returns
    -------
    torch.Tensor
        Gravity displacement with shape ``[N, 2]``.
    """
    if strong_gravity:
        factor = scaling_ratio * mass * gravity
        return -pos * factor.unsqueeze(1)

    distance = torch.linalg.vector_norm(pos, dim=1)
    factor = torch.zeros_like(distance)
    valid = distance > 0
    factor[valid] = mass[valid] * gravity / distance[valid]
    return -pos * factor.unsqueeze(1)


def _attraction_force(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    mass: torch.Tensor,
    outbound_att_compensation: float,
    outbound_attraction_distribution: bool,
) -> torch.Tensor:
    """Compute ForceAtlas2 attraction over unique undirected edges.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Unique undirected edge list with shape ``[2, E]``.
    mass : torch.Tensor
        Node masses with shape ``[N]``.
    outbound_att_compensation : float
        Mean-mass compensation used when outbound attraction distribution is on.
    outbound_attraction_distribution : bool
        Whether to divide attraction by the source-node mass.

    Returns
    -------
    torch.Tensor
        Attraction displacement with shape ``[N, 2]``.
    """
    force = torch.zeros_like(pos)
    if edge_index.numel() == 0:
        return force

    source = edge_index[0]
    target = edge_index[1]
    delta = pos.index_select(0, source) - pos.index_select(0, target)
    factor = torch.full(
        (edge_index.shape[1],),
        fill_value=-float(outbound_att_compensation),
        dtype=pos.dtype,
        device=pos.device,
    )
    if outbound_attraction_distribution:
        factor = factor / mass.index_select(0, source)

    attraction = delta * factor.unsqueeze(1)
    index = source.unsqueeze(1).expand_as(attraction)
    force.scatter_add_(0, index, attraction)
    index = target.unsqueeze(1).expand_as(attraction)
    force.scatter_add_(0, index, -attraction)
    return force


def _adjust_speed_and_apply_forces(
    pos: torch.Tensor,
    force: torch.Tensor,
    old_force: torch.Tensor,
    mass: torch.Tensor,
    speed: float,
    speed_efficiency: float,
    jitter_tolerance: float,
) -> tuple[torch.Tensor, float, float]:
    """Translate ``adjustSpeedAndApplyForces`` into vectorized PyTorch.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    force : torch.Tensor
        Current node displacements with shape ``[N, 2]``.
    old_force : torch.Tensor
        Previous node displacements with shape ``[N, 2]``.
    mass : torch.Tensor
        Node masses with shape ``[N]``.
    speed : float
        Current global speed.
    speed_efficiency : float
        Current speed-efficiency coefficient.
    jitter_tolerance : float
        Reference jitter-tolerance hyperparameter.

    Returns
    -------
    tuple[torch.Tensor, float, float]
        Updated ``(positions, speed, speed_efficiency)``.
    """
    swinging = mass * torch.linalg.vector_norm(old_force - force, dim=1)
    effective_traction = 0.5 * mass * torch.linalg.vector_norm(old_force + force, dim=1)

    total_swinging = float(swinging.sum().item())
    total_effective_traction = float(effective_traction.sum().item())

    estimated_optimal_jt = 0.05 * math.sqrt(float(pos.shape[0]))
    min_jt = math.sqrt(estimated_optimal_jt)
    max_jt = 10.0
    jt = jitter_tolerance * max(
        min_jt,
        min(
            max_jt,
            estimated_optimal_jt * total_effective_traction / float(pos.shape[0] * pos.shape[0]),
        ),
    )

    min_speed_efficiency = 0.05
    if total_effective_traction > 0.0 and total_swinging / total_effective_traction > 2.0:
        if speed_efficiency > min_speed_efficiency:
            speed_efficiency *= 0.5
        jt = max(jt, jitter_tolerance)

    if total_swinging == 0.0:
        target_speed = float("inf")
    else:
        target_speed = jt * speed_efficiency * total_effective_traction / total_swinging

    if total_swinging > jt * total_effective_traction:
        if speed_efficiency > min_speed_efficiency:
            speed_efficiency *= 0.7
    elif speed < 1000.0:
        speed_efficiency *= 1.3

    speed = speed + min(target_speed - speed, 0.5 * speed)
    factor = speed / (1.0 + torch.sqrt(speed * swinging))
    return pos + (force * factor.unsqueeze(1)), speed, speed_efficiency

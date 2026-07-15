"""Cytoscape CoSE-Bilkent layout pipeline."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch

from dagua.layout.ops.pipelines.cose import layout_cose_pipeline

_COMPOUND_CENTER_SPACING_FACTOR = 12.0
_COMPOUND_LOCAL_SCALE = 0.5
_COMPOUND_ROTATION_EPSILON = 1.0e-9
_CYTOSCAPE_LCG_MULTIPLIER = 1664525
_CYTOSCAPE_LCG_INCREMENT = 1013904223
_CYTOSCAPE_LCG_MODULUS = 4294967296


def _compound_groups(
    clusters: Optional[dict[str, Any]],
    num_nodes: int,
) -> list[list[int]]:
    """Return deterministic CoSE-Bilkent compound member groups.

    Parameters
    ----------
    clusters : dict[str, Any] | None
        Cluster membership mapping from graph construction.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Sorted member groups followed by singleton unclustered nodes.
    """
    if not clusters:
        return []
    groups: list[list[int]] = []
    assigned: set[int] = set()
    for cluster_id in sorted(clusters):
        members = clusters[cluster_id]
        if isinstance(members, dict):
            continue
        group = sorted(int(member) for member in members if 0 <= int(member) < num_nodes)
        if group:
            groups.append(group)
            assigned.update(group)
    for node in range(num_nodes):
        if node not in assigned:
            groups.append([node])
    return groups


def _induced_edge_index(
    edge_index: torch.Tensor,
    group: list[int],
) -> torch.Tensor:
    """Return the edge tensor induced by one compound group.

    Parameters
    ----------
    edge_index : torch.Tensor
        Global edge tensor with shape ``[2, E]``.
    group : list[int]
        Global node ids in the compound group.

    Returns
    -------
    torch.Tensor
        Local edge tensor with shape ``[2, E_group]``.
    """
    local_index = {node: index for index, node in enumerate(group)}
    local_edges: list[tuple[int, int]] = []
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    edges = edge_index.detach().cpu().long()
    for edge_pos in range(edges.shape[1]):
        source = int(edges[0, edge_pos].item())
        target = int(edges[1, edge_pos].item())
        if source in local_index and target in local_index:
            local_edges.append((local_index[source], local_index[target]))
    if not local_edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(local_edges, dtype=torch.long).t().contiguous()


def _compound_inter_edge_rotation(
    edge_index: torch.Tensor,
    group: list[int],
    group_index: int,
    group_index_by_node: dict[int, int],
    centers: list[tuple[float, float]],
    local: torch.Tensor,
) -> float:
    """Return the inter-edge-driven rotation for one compound member pack.

    Parameters
    ----------
    edge_index : torch.Tensor
        Global edge tensor with shape ``[2, E]``.
    group : list[int]
        Global node ids in one compound member group.
    group_index : int
        Index of ``group`` in the compound group list.
    group_index_by_node : dict[int, int]
        Mapping from global node id to compound group index.
    centers : list[tuple[float, float]]
        Compound representative center coordinates.
    local : torch.Tensor
        Centered local member coordinates with shape ``[len(group), 2]``.

    Returns
    -------
    float
        Rotation angle in radians.
    """
    if edge_index.numel() == 0 or len(group) < 2:
        return 0.0
    local_index_by_node = {node: index for index, node in enumerate(group)}
    center_x, center_y = centers[group_index]
    cos_sum = 0.0
    sin_sum = 0.0
    edges = edge_index.detach().cpu().long()
    for edge_pos in range(edges.shape[1]):
        source = int(edges[0, edge_pos].item())
        target = int(edges[1, edge_pos].item())
        source_group = group_index_by_node.get(source)
        target_group = group_index_by_node.get(target)
        if source_group is None or target_group is None or source_group == target_group:
            continue
        if source_group == group_index:
            local_node = source
            other_group = target_group
        elif target_group == group_index:
            local_node = target
            other_group = source_group
        else:
            continue
        local_index = local_index_by_node[local_node]
        local_x = float(local[local_index, 0])
        local_y = float(local[local_index, 1])
        local_norm = math.hypot(local_x, local_y)
        target_x = centers[other_group][0] - center_x
        target_y = centers[other_group][1] - center_y
        target_norm = math.hypot(target_x, target_y)
        if local_norm <= _COMPOUND_ROTATION_EPSILON or target_norm <= _COMPOUND_ROTATION_EPSILON:
            continue
        local_x /= local_norm
        local_y /= local_norm
        target_x /= target_norm
        target_y /= target_norm
        cos_sum += local_x * target_x + local_y * target_y
        sin_sum += local_x * target_y - local_y * target_x
    if abs(cos_sum) <= _COMPOUND_ROTATION_EPSILON and abs(sin_sum) <= _COMPOUND_ROTATION_EPSILON:
        return 0.0
    return math.atan2(sin_sum, cos_sum)


def _next_cytoscape_random(raw_state: int) -> tuple[int, float]:
    """Advance Cytoscape's seeded random stream by one value.

    Parameters
    ----------
    raw_state : int
        Current unsigned 32-bit LCG state.

    Returns
    -------
    tuple[int, float]
        Updated raw state and the corresponding random value in ``[0, 1)``.
    """
    next_state = (
        _CYTOSCAPE_LCG_MULTIPLIER * int(raw_state) + _CYTOSCAPE_LCG_INCREMENT
    ) & 0xFFFFFFFF
    return next_state, next_state / _CYTOSCAPE_LCG_MODULUS


def _cytoscape_random_initial_positions(
    count: int,
    raw_state: int,
) -> tuple[torch.Tensor, int]:
    """Return random CoSE-Bilkent initial positions from one shared RNG stream.

    Parameters
    ----------
    count : int
        Number of local nodes to initialize.
    raw_state : int
        Current unsigned 32-bit Cytoscape LCG state.

    Returns
    -------
    tuple[torch.Tensor, int]
        Position tensor with shape ``[count, 2]`` and the updated raw state.
    """
    pos = torch.empty((count, 2), dtype=torch.float32)
    for node_index in range(count):
        raw_state, random_x = _next_cytoscape_random(raw_state)
        raw_state, random_y = _next_cytoscape_random(raw_state)
        pos[node_index, 0] = random_x
        pos[node_index, 1] = random_y
    return pos, raw_state


def _layout_compound_groups(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    groups: list[list[int]],
    steps: int,
    seed: int,
    node_repulsion: float,
    ideal_edge_length: float,
    edge_elasticity: float,
    gravity: float,
) -> torch.Tensor:
    """Lay out compound groups with CoSE-style local member placement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Global edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None
        Node-size tensor with shape ``[N, 2]``.
    groups : list[list[int]]
        Compound groups to lay out.
    steps : int
        Number of local CoSE force iterations.
    seed : int
        Seed for deterministic randomized member placement.
    node_repulsion : float
        CoSE-Bilkent node repulsion multiplier.
    ideal_edge_length : float
        Desired CoSE-Bilkent edge length.
    edge_elasticity : float
        CoSE-Bilkent edge elasticity.
    gravity : float
        CoSE-Bilkent gravity strength.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    pos = torch.zeros((num_nodes, 2), dtype=torch.float32)
    center_spacing = max(ideal_edge_length * _COMPOUND_CENTER_SPACING_FACTOR, ideal_edge_length)
    if len(groups) == 2:
        centers = [(-center_spacing / 2.0, 0.0), (center_spacing / 2.0, 0.0)]
    else:
        radius = max(center_spacing, len(groups) * center_spacing / (2.0 * math.pi))
        centers = [
            (
                radius * math.cos(2.0 * math.pi * index / max(len(groups), 1)),
                radius * math.sin(2.0 * math.pi * index / max(len(groups), 1)),
            )
            for index in range(len(groups))
        ]

    group_index_by_node = {
        node: group_index for group_index, group in enumerate(groups) for node in group
    }
    random_state = int(seed) & 0xFFFFFFFF
    if random_state == 0:
        random_state = 1
    for group_index, group in enumerate(groups):
        local_node_sizes = node_sizes[group] if node_sizes is not None else None
        local_initial, random_state = _cytoscape_random_initial_positions(
            count=len(group),
            raw_state=random_state,
        )
        local = layout_cose_pipeline(
            edge_index=_induced_edge_index(edge_index, group),
            num_nodes=len(group),
            node_sizes=local_node_sizes,
            steps=steps,
            seed=seed,
            randomize=False,
            nodeRepulsion=node_repulsion,
            idealEdgeLength=ideal_edge_length,
            edgeElasticity=edge_elasticity,
            gravity=gravity,
            initialTemp=50.0,
            minTemp=0.01,
            pos=local_initial,
        )
        local = (local - local.mean(dim=0, keepdim=True)) * _COMPOUND_LOCAL_SCALE
        rotation = _compound_inter_edge_rotation(
            edge_index=edge_index,
            group=group,
            group_index=group_index,
            group_index_by_node=group_index_by_node,
            centers=centers,
            local=local,
        )
        if rotation != 0.0:
            cos_rotation = math.cos(rotation)
            sin_rotation = math.sin(rotation)
            rotated = torch.empty_like(local)
            rotated[:, 0] = local[:, 0] * cos_rotation - local[:, 1] * sin_rotation
            rotated[:, 1] = local[:, 0] * sin_rotation + local[:, 1] * cos_rotation
            local = rotated
        center = torch.tensor(centers[group_index], dtype=torch.float32)
        for local_index, node in enumerate(group):
            pos[node] = local[local_index] + center
    return pos


def build_cose_bilkent_pipeline(steps: int = 2500, quality: str = "default") -> Any:
    """Return the shared CoSE pipeline builder for Bilkent defaults.

    Parameters
    ----------
    steps : int, default=2500
        Maximum number of force iterations.
    quality : str, default="default"
        CoSE-Bilkent quality tier.

    Returns
    -------
    Any
        Pipeline object built by the core CoSE wrapper.
    """
    from dagua.layout.ops.pipelines.cose import build_cose_pipeline

    cooling = 0.995 if quality == "proof" else 0.99
    return build_cose_pipeline(
        steps=0 if quality == "draft" else steps,
        randomize=True,
        node_repulsion=4500.0,
        ideal_edge_length=50.0,
        edge_elasticity=0.45,
        gravity=0.25,
        initial_temp=50.0,
        cooling_factor=cooling,
        min_temp=0.01,
    )


def layout_cose_bilkent_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 2500,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    quality: str = "default",
    randomize: bool = True,
    nodeRepulsion: float = 4500.0,
    idealEdgeLength: float = 50.0,
    edgeElasticity: float = 0.45,
    gravity: float = 0.25,
    gravityRange: float = 3.8,
    gravityCompound: float = 1.0,
    gravityRangeCompound: float = 1.5,
    tile: bool = True,
    clusters: Optional[dict[str, Any]] = None,
    cluster_parents: Optional[dict[str, Optional[str]]] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the Cytoscape CoSE-Bilkent pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None, optional
        Node-size tensor with shape ``[N, 2]``.
    steps : int, default=2500
        Maximum number of force iterations.
    seed : int, default=42
        Seed for randomized initial placement.
    edge_weights : torch.Tensor | None, optional
        Accepted for API consistency.
    quality : str, default="default"
        Quality tier: ``"draft"``, ``"default"``, or ``"proof"``.
    randomize : bool, default=True
        Whether to randomize initial positions.
    nodeRepulsion : float, default=4500.0
        Node repulsion multiplier.
    idealEdgeLength : float, default=50.0
        Desired edge length.
    edgeElasticity : float, default=0.45
        Edge-force divisor.
    gravity : float, default=0.25
        Gravity force.
    gravityRange : float, default=3.8
        Accepted for option parity; current native step applies global gravity.
    gravityCompound : float, default=1.0
        Accepted for option parity with compound layouts.
    gravityRangeCompound : float, default=1.5
        Accepted for option parity with compound layouts.
    tile : bool, default=True
        Accepted for option parity with disconnected tiling.
    clusters : dict[str, Any] | None, optional
        Cluster membership metadata.
    cluster_parents : dict[str, str | None] | None, optional
        Cluster hierarchy metadata.
    fidelity_dtype : torch.dtype | None, optional
        Optional output dtype override.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    del gravityRange, gravityCompound, gravityRangeCompound, tile, cluster_parents
    if quality not in {"draft", "default", "proof"}:
        raise ValueError("quality must be one of 'draft', 'default', or 'proof'.")
    effective_steps = 0 if quality == "draft" else steps
    cooling = 0.995 if quality == "proof" else 0.99
    groups = _compound_groups(clusters, num_nodes)
    if len(groups) > 1:
        result = _layout_compound_groups(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            groups=groups,
            steps=effective_steps,
            seed=seed,
            node_repulsion=nodeRepulsion,
            ideal_edge_length=idealEdgeLength,
            edge_elasticity=edgeElasticity,
            gravity=gravity,
        )
        if fidelity_dtype is not None:
            return result.to(dtype=fidelity_dtype)
        return result
    return layout_cose_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        steps=effective_steps,
        seed=seed,
        edge_weights=edge_weights,
        randomize=randomize,
        nodeRepulsion=nodeRepulsion,
        idealEdgeLength=idealEdgeLength,
        edgeElasticity=edgeElasticity,
        gravity=gravity,
        initialTemp=50.0,
        coolingFactor=cooling,
        minTemp=0.01,
        fidelity_dtype=fidelity_dtype,
    )


__all__ = ["build_cose_bilkent_pipeline", "layout_cose_bilkent_pipeline"]

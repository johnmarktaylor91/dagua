"""Differentiable surrogate terms for the native W5 finisher."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from dagua.layout.ops.pipelines.native_shape_geometry import (
    NativeShapeGeometry,
    shape_overlap_hinge_loss,
)

_EPS = 1.0e-6


def _zero_like_loss(pos: torch.Tensor) -> torch.Tensor:
    """Return a scalar zero connected to ``pos`` for autograd.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Scalar zero on the same device and dtype as ``pos``.
    """
    return pos.sum() * 0.0


def _edge_pair_indices(
    edge_index: torch.Tensor,
    max_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a bounded deterministic non-incident edge-pair sample.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    max_pairs : int
        Maximum number of edge pairs to return.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Left and right edge ids with matching shape ``[B]``.
    """
    edge_count = int(edge_index.shape[1]) if edge_index.numel() else 0
    if edge_count < 2 or max_pairs <= 0:
        empty = torch.empty(0, dtype=torch.long, device=edge_index.device)
        return empty, empty
    left, right = torch.triu_indices(edge_count, edge_count, offset=1, device=edge_index.device)
    src = edge_index[0]
    tgt = edge_index[1]
    non_incident = (
        (src[left] != src[right])
        & (src[left] != tgt[right])
        & (tgt[left] != src[right])
        & (tgt[left] != tgt[right])
    )
    left = left[non_incident]
    right = right[non_incident]
    if left.numel() <= max_pairs:
        return left, right
    sample = torch.linspace(
        0,
        left.numel() - 1,
        steps=max_pairs,
        device=edge_index.device,
        dtype=torch.float32,
    ).round()
    sample = sample.to(dtype=torch.long).clamp(0, left.numel() - 1)
    return left[sample], right[sample]


def _cross_2d(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return the 2D cross product for paired vectors.

    Parameters
    ----------
    left : torch.Tensor
        Left vectors with shape ``[B, 2]``.
    right : torch.Tensor
        Right vectors with shape ``[B, 2]``.

    Returns
    -------
    torch.Tensor
        Cross-product values with shape ``[B]``.
    """
    return left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]


def soft_crossing_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    max_pairs: int = 4096,
    sharpness: float = 8.0,
) -> torch.Tensor:
    """Estimate segment intersections with a smooth bounded loss.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    max_pairs : int, default=4096
        Deterministic cap on non-incident edge pairs.
    sharpness : float, default=8.0
        Sigmoid slope for the orientation-product tests.

    Returns
    -------
    torch.Tensor
        Scalar loss that increases when sampled edge pairs cross.
    """
    left_ids, right_ids = _edge_pair_indices(edge_index.to(pos.device), max_pairs)
    if left_ids.numel() == 0:
        return _zero_like_loss(pos)
    edges = edge_index.to(device=pos.device, dtype=torch.long)
    a = pos[edges[0, left_ids]]
    b = pos[edges[1, left_ids]]
    c = pos[edges[0, right_ids]]
    d = pos[edges[1, right_ids]]
    ab = b - a
    cd = d - c
    ab_len = torch.linalg.vector_norm(ab, dim=1).clamp(min=_EPS)
    cd_len = torch.linalg.vector_norm(cd, dim=1).clamp(min=_EPS)
    scale = (ab_len * cd_len).square().clamp(min=_EPS)
    o1 = _cross_2d(ab, c - a)
    o2 = _cross_2d(ab, d - a)
    o3 = _cross_2d(cd, a - c)
    o4 = _cross_2d(cd, b - c)
    first_side = torch.sigmoid(-float(sharpness) * o1 * o2 / scale)
    second_side = torch.sigmoid(-float(sharpness) * o3 * o4 / scale)
    return torch.nan_to_num(first_side * second_side, nan=0.0, posinf=1.0).mean()


def crossing_angle_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    max_pairs: int = 4096,
) -> torch.Tensor:
    """Penalize shallow angles on likely crossing edge pairs.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    max_pairs : int, default=4096
        Deterministic cap on non-incident edge pairs.

    Returns
    -------
    torch.Tensor
        Scalar angle loss weighted by soft crossing probability.
    """
    left_ids, right_ids = _edge_pair_indices(edge_index.to(pos.device), max_pairs)
    if left_ids.numel() == 0:
        return _zero_like_loss(pos)
    edges = edge_index.to(device=pos.device, dtype=torch.long)
    left_vec = pos[edges[1, left_ids]] - pos[edges[0, left_ids]]
    right_vec = pos[edges[1, right_ids]] - pos[edges[0, right_ids]]
    left_norm = torch.linalg.vector_norm(left_vec, dim=1).clamp(min=_EPS)
    right_norm = torch.linalg.vector_norm(right_vec, dim=1).clamp(min=_EPS)
    cosine = ((left_vec * right_vec).sum(dim=1).abs() / (left_norm * right_norm)).clamp(0.0, 1.0)
    crossing_weight = soft_crossing_pair_weights(pos, edge_index, left_ids, right_ids)
    return torch.nan_to_num(crossing_weight * cosine.square(), nan=0.0, posinf=1.0).mean()


def soft_crossing_pair_weights(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    left_ids: torch.Tensor,
    right_ids: torch.Tensor,
    *,
    sharpness: float = 8.0,
) -> torch.Tensor:
    """Return soft crossing weights for explicit edge-pair ids.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    left_ids : torch.Tensor
        Left edge ids with shape ``[B]``.
    right_ids : torch.Tensor
        Right edge ids with shape ``[B]``.
    sharpness : float, default=8.0
        Sigmoid slope for orientation products.

    Returns
    -------
    torch.Tensor
        Soft crossing weights with shape ``[B]``.
    """
    if left_ids.numel() == 0:
        return torch.empty(0, dtype=pos.dtype, device=pos.device)
    edges = edge_index.to(device=pos.device, dtype=torch.long)
    a = pos[edges[0, left_ids]]
    b = pos[edges[1, left_ids]]
    c = pos[edges[0, right_ids]]
    d = pos[edges[1, right_ids]]
    ab = b - a
    cd = d - c
    scale = (
        torch.linalg.vector_norm(ab, dim=1).clamp(min=_EPS)
        * torch.linalg.vector_norm(cd, dim=1).clamp(min=_EPS)
    ).square()
    first = torch.sigmoid(-float(sharpness) * _cross_2d(ab, c - a) * _cross_2d(ab, d - a) / scale)
    second = torch.sigmoid(-float(sharpness) * _cross_2d(cd, a - c) * _cross_2d(cd, b - c) / scale)
    return torch.nan_to_num(first * second, nan=0.0, posinf=1.0)


def path_continuity_loss(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Penalize bends through undirected degree-two chain vertices.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    torch.Tensor
        Scalar loss, zero when no degree-two chain vertex exists.
    """
    node_count = int(pos.shape[0])
    if node_count == 0 or edge_index.numel() == 0:
        return _zero_like_loss(pos)
    neighbors: list[set[int]] = [set() for _ in range(node_count)]
    for source, target in edge_index.detach().to(device="cpu", dtype=torch.long).t().tolist():
        if source != target and 0 <= source < node_count and 0 <= target < node_count:
            neighbors[source].add(target)
            neighbors[target].add(source)
    triples = [
        (min(adjacent), vertex, max(adjacent))
        for vertex, adjacent in enumerate(neighbors)
        if len(adjacent) == 2
    ]
    if not triples:
        return _zero_like_loss(pos)
    triples_tensor = torch.tensor(triples, dtype=torch.long, device=pos.device)
    left = pos[triples_tensor[:, 0]] - pos[triples_tensor[:, 1]]
    right = pos[triples_tensor[:, 2]] - pos[triples_tensor[:, 1]]
    cosine = F.cosine_similarity(left, right, dim=1, eps=_EPS).clamp(-1.0, 1.0)
    return ((cosine + 1.0) * 0.5).square().mean()


def signed_flow_score_surrogate(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    direction: str = "TB",
) -> torch.Tensor:
    """Approximate directed-flow score with a smooth edge-alignment term.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    direction : str, default="TB"
        Declared flow direction.

    Returns
    -------
    torch.Tensor
        Scalar score in approximately ``[0, 1]``.
    """
    if edge_index.numel() == 0:
        return _zero_like_loss(pos)
    edges = edge_index.to(device=pos.device, dtype=torch.long)
    vectors = pos[edges[1]] - pos[edges[0]]
    axis_values = {
        "TB": (0.0, 1.0),
        "BT": (0.0, -1.0),
        "LR": (1.0, 0.0),
        "RL": (-1.0, 0.0),
    }.get(direction, (0.0, 1.0))
    axis = torch.tensor(axis_values, dtype=pos.dtype, device=pos.device)
    projection = (vectors * axis).sum(dim=1) / torch.linalg.vector_norm(vectors, dim=1).clamp(
        min=_EPS
    )
    return torch.sigmoid(6.0 * projection).mean()


def depth_order_score_surrogate(
    pos: torch.Tensor,
    topo_depth: torch.Tensor,
    *,
    direction: str = "TB",
    max_pairs: int = 8192,
) -> torch.Tensor:
    """Approximate topological depth ordering with pairwise logistic tests.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    topo_depth : torch.Tensor
        Integer depth tensor with shape ``[N]``.
    direction : str, default="TB"
        Declared flow direction.
    max_pairs : int, default=8192
        Maximum depth-ordered node pairs.

    Returns
    -------
    torch.Tensor
        Scalar score in approximately ``[0, 1]``.
    """
    node_count = int(pos.shape[0])
    if node_count < 2 or topo_depth.numel() < node_count:
        return _zero_like_loss(pos)
    depth = topo_depth.to(device=pos.device, dtype=torch.long)
    left, right = torch.triu_indices(node_count, node_count, offset=1, device=pos.device)
    ordered_mask = depth[left] != depth[right]
    left = left[ordered_mask]
    right = right[ordered_mask]
    if left.numel() == 0:
        return _zero_like_loss(pos)
    if left.numel() > max_pairs:
        sample = torch.linspace(
            0,
            left.numel() - 1,
            steps=max_pairs,
            dtype=torch.float32,
            device=pos.device,
        ).round()
        sample = sample.to(dtype=torch.long).clamp(0, left.numel() - 1)
        left = left[sample]
        right = right[sample]
    axis = 0 if direction in {"LR", "RL"} else 1
    sign = -1.0 if direction in {"BT", "RL"} else 1.0
    coord = pos[:, axis] * sign
    shallow = torch.where(depth[left] < depth[right], left, right)
    deep = torch.where(depth[left] < depth[right], right, left)
    span = (coord.max() - coord.min()).detach().clamp(min=1.0)
    logits = (coord[deep] - coord[shallow]) / (0.05 * span + _EPS)
    return torch.sigmoid(logits).mean()


def edge_length_cv_loss(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Return coefficient-of-variation loss over edge lengths.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    torch.Tensor
        Scalar edge-length CV loss.
    """
    if edge_index.numel() == 0:
        return _zero_like_loss(pos)
    edges = edge_index.to(device=pos.device, dtype=torch.long)
    lengths = torch.linalg.vector_norm(pos[edges[1]] - pos[edges[0]], dim=1)
    mean = lengths.mean().clamp(min=_EPS)
    return torch.nan_to_num(lengths.std(unbiased=False) / mean, nan=0.0, posinf=10.0)


def overlap_hinge_loss(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    *,
    padding: float = 0.0,
    max_nodes: int = 512,
    shape_geometry: Optional[NativeShapeGeometry] = None,
) -> torch.Tensor:
    """Penalize overlapping node boxes with a bounded all-pairs hinge.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    padding : float, default=0.0
        Additional box padding.
    max_nodes : int, default=512
        Deterministic node cap for the all-pairs box check.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box node-shape descriptors. When omitted, this function
        preserves the exact legacy AABB computation.

    Returns
    -------
    torch.Tensor
        Scalar overlap loss.
    """
    node_count = int(pos.shape[0])
    if node_count < 2 or node_sizes.numel() == 0:
        return _zero_like_loss(pos)
    if shape_geometry is not None:
        return shape_overlap_hinge_loss(
            pos,
            node_sizes,
            shape_geometry,
            padding=padding,
            max_nodes=max_nodes,
        )
    if node_count > max_nodes:
        sample = torch.linspace(
            0,
            node_count - 1,
            steps=max_nodes,
            dtype=torch.float32,
            device=pos.device,
        ).round()
        sample = sample.to(dtype=torch.long).clamp(0, node_count - 1)
        work_pos = pos[sample]
        work_sizes = node_sizes.to(device=pos.device, dtype=pos.dtype)[sample]
    else:
        work_pos = pos
        work_sizes = node_sizes.to(device=pos.device, dtype=pos.dtype)
    dx_abs = (work_pos[:, None, 0] - work_pos[None, :, 0]).abs()
    dy_abs = (work_pos[:, None, 1] - work_pos[None, :, 1]).abs()
    min_dx = (work_sizes[:, None, 0] + work_sizes[None, :, 0]) * 0.5 + float(padding)
    min_dy = (work_sizes[:, None, 1] + work_sizes[None, :, 1]) * 0.5 + float(padding)
    overlap = F.relu(min_dx - dx_abs) * F.relu(min_dy - dy_abs)
    mask = ~torch.eye(work_pos.shape[0], dtype=torch.bool, device=pos.device)
    scale = work_sizes.mean().clamp(min=1.0).square()
    return torch.nan_to_num(overlap[mask].mean() / scale, nan=0.0, posinf=10.0)


def soft_knn_neighborhood_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    max_nodes: int = 128,
) -> torch.Tensor:
    """Encourage graph-adjacent nodes to remain geometrically nearby.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    max_nodes : int, default=128
        Deterministic sampled node cap.

    Returns
    -------
    torch.Tensor
        Scalar binary ranking loss over sampled adjacency.
    """
    node_count = int(pos.shape[0])
    if node_count < 2 or edge_index.numel() == 0:
        return _zero_like_loss(pos)
    if node_count > max_nodes:
        sampled = torch.linspace(
            0,
            node_count - 1,
            steps=max_nodes,
            device=pos.device,
            dtype=torch.float32,
        ).round()
        sampled = sampled.to(dtype=torch.long).clamp(0, node_count - 1)
    else:
        sampled = torch.arange(node_count, device=pos.device)
    sample_pos = pos[sampled]
    dist = torch.cdist(sample_pos, sample_pos)
    positive_dist = dist.detach()[dist.detach() > _EPS]
    scale = positive_dist.median() if bool(positive_dist.numel()) else 1.0
    labels = torch.zeros_like(dist)
    sample_id = torch.full((node_count,), -1, dtype=torch.long, device=pos.device)
    sample_id[sampled] = torch.arange(sampled.numel(), device=pos.device)
    edges = edge_index.to(device=pos.device, dtype=torch.long)
    left = sample_id[edges[0]]
    right = sample_id[edges[1]]
    valid = (left >= 0) & (right >= 0) & (left != right)
    labels[left[valid], right[valid]] = 1.0
    labels[right[valid], left[valid]] = 1.0
    mask = ~torch.eye(sampled.numel(), dtype=torch.bool, device=pos.device)
    logits = -dist / torch.as_tensor(scale, dtype=pos.dtype, device=pos.device).clamp(min=1.0)
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask])


def gabriel_intrusion_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    max_edges: int = 256,
    max_nodes: int = 256,
) -> torch.Tensor:
    """Penalize nodes inside sampled Gabriel disks around edges.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    max_edges : int, default=256
        Deterministic edge sample cap.
    max_nodes : int, default=256
        Deterministic node sample cap.

    Returns
    -------
    torch.Tensor
        Scalar Gabriel intrusion loss.
    """
    node_count = int(pos.shape[0])
    edge_count = int(edge_index.shape[1]) if edge_index.numel() else 0
    if node_count < 3 or edge_count == 0:
        return _zero_like_loss(pos)
    edges = edge_index.to(device=pos.device, dtype=torch.long)
    edge_ids = torch.arange(edge_count, device=pos.device)
    if edge_count > max_edges:
        edge_ids = (
            torch.linspace(
                0,
                edge_count - 1,
                steps=max_edges,
                device=pos.device,
                dtype=torch.float32,
            )
            .round()
            .to(dtype=torch.long)
        )
    node_ids = torch.arange(node_count, device=pos.device)
    if node_count > max_nodes:
        node_ids = (
            torch.linspace(
                0,
                node_count - 1,
                steps=max_nodes,
                device=pos.device,
                dtype=torch.float32,
            )
            .round()
            .to(dtype=torch.long)
        )
    src = edges[0, edge_ids]
    tgt = edges[1, edge_ids]
    midpoint = (pos[src] + pos[tgt]) * 0.5
    radius = torch.linalg.vector_norm(pos[tgt] - pos[src], dim=1) * 0.5
    distances = torch.cdist(midpoint, pos[node_ids])
    incident = (node_ids[None, :] == src[:, None]) | (node_ids[None, :] == tgt[:, None])
    intrusion = F.relu(radius[:, None] - distances).masked_fill(incident, 0.0)
    return torch.nan_to_num(intrusion.mean() / radius.detach().mean().clamp(min=1.0), nan=0.0)


def angular_resolution_loss(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Penalize nearly parallel incident edges at shared endpoints.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    torch.Tensor
        Scalar incident-angle loss.
    """
    node_count = int(pos.shape[0])
    if node_count == 0 or edge_index.numel() == 0:
        return _zero_like_loss(pos)
    edges_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    terms: list[torch.Tensor] = []
    for node in range(node_count):
        incident: list[int] = []
        for edge_id, (source, target) in enumerate(edges_cpu.t().tolist()):
            if source == node or target == node:
                incident.append(edge_id)
        if len(incident) < 2:
            continue
        edge_ids = torch.tensor(incident[:8], dtype=torch.long, device=pos.device)
        edges = edge_index.to(device=pos.device, dtype=torch.long)
        other = torch.where(edges[0, edge_ids] == node, edges[1, edge_ids], edges[0, edge_ids])
        vectors = pos[other] - pos[node]
        vectors = F.normalize(vectors, dim=1, eps=_EPS)
        sim = torch.mm(vectors, vectors.T).abs()
        mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
        terms.append(sim[mask].square().mean())
    if not terms:
        return _zero_like_loss(pos)
    return torch.stack(terms).mean()


def barrier_floor_loss(score: torch.Tensor, floor: Optional[float]) -> torch.Tensor:
    """Convert a score floor into a one-sided differentiable penalty.

    Parameters
    ----------
    score : torch.Tensor
        Scalar score to constrain.
    floor : float, optional
        Minimum allowed score. ``None`` disables the barrier.

    Returns
    -------
    torch.Tensor
        Scalar one-sided loss.
    """
    if floor is None:
        return score.sum() * 0.0
    floor_tensor = torch.as_tensor(float(floor), dtype=score.dtype, device=score.device)
    return F.relu(floor_tensor - score).square()

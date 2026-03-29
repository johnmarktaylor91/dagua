"""(SGD)^2 multicriteria graph layout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from dagua.layout.classic._graph_distances import (
    all_pairs_shortest_paths as _shared_all_pairs_shortest_paths,
)
from dagua.layout.classic._graph_distances import (
    build_undirected_adjacency as _shared_build_undirected_adjacency,
)

_EPS = 1.0e-6
_CROSSING_SHARPNESS = 10.0
_DEFAULT_IDEAL_EDGE_LENGTH = 1.0
_DEFAULT_ASPECT_RATIO_TARGET = 0.95
_DEFAULT_VERTEX_RESOLUTION = 1.0
_DEFAULT_GRAPH_K = 8


class SmoothSteps:
    """Piecewise-smooth scalar schedule."""

    def __init__(self, times: list[int], values: list[float]) -> None:
        """Create a schedule from smooth keyframes.

        Parameters
        ----------
        times : list[int]
            Monotone keyframe times.
        values : list[float]
            Keyframe values aligned with ``times``.
        """
        if len(times) != len(values):
            raise ValueError("times and values must have the same length.")
        if len(times) == 0:
            raise ValueError("times and values must be non-empty.")
        if any(later < earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("times must be non-decreasing.")
        self.times = times
        self.values = values

    @staticmethod
    def smooth_step(x: float) -> float:
        """Evaluate the Hermite smooth-step interpolant.

        Parameters
        ----------
        x : float
            Input value.

        Returns
        -------
        float
            Interpolated value in ``[0, 1]``.
        """
        x_clamped = min(max(x, 0.0), 1.0)
        return 3.0 * x_clamped * x_clamped - 2.0 * x_clamped * x_clamped * x_clamped

    def __call__(self, t: int) -> float:
        """Evaluate the schedule at one time step.

        Parameters
        ----------
        t : int
            Iteration index.

        Returns
        -------
        float
            Scheduled weight at ``t``.
        """
        if t <= self.times[0]:
            return self.values[0]
        for index in range(len(self.times) - 1):
            left = self.times[index]
            right = self.times[index + 1]
            if t <= right:
                span = max(right - left, 1)
                frac = float(t - left) / float(span)
                return self.values[index] + self.smooth_step(frac) * (
                    self.values[index + 1] - self.values[index]
                )
        return self.values[-1]


@dataclass(frozen=True)
class _PreparedState:
    """Precomputed graph data needed by the multicriteria objective."""

    device: torch.device
    edges: torch.Tensor
    adjacency: list[list[tuple[int, float]]]
    all_pairs_distances: Optional[torch.Tensor]
    stress_pairs: Optional[torch.Tensor]
    stress_distances: Optional[torch.Tensor]
    stress_weights: Optional[torch.Tensor]
    graph_knn_mask: Optional[torch.Tensor]


def _layout_device(
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.device:
    """Choose the output device for the layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor | None
        Optional node-size tensor.

    Returns
    -------
    torch.device
        Device used for optimization and the returned coordinates.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


def _validate_inputs(
    edge_index: torch.Tensor,
    num_nodes: int,
    steps: int,
    lr: float,
    momentum: float,
    grad_clamp: float,
    batch_size: int,
) -> None:
    """Validate the public (SGD)^2 input arguments.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Number of optimizer iterations.
    lr : float
        SGD learning rate.
    momentum : float
        SGD momentum.
    grad_clamp : float
        Symmetric gradient clamp.
    batch_size : int
        Mini-batch size.

    Returns
    -------
    None
        Raises ``ValueError`` when an argument is invalid.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if lr <= 0.0:
        raise ValueError("lr must be positive.")
    if momentum < 0.0 or momentum >= 1.0:
        raise ValueError("momentum must be in [0, 1).")
    if grad_clamp <= 0.0:
        raise ValueError("grad_clamp must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_index.numel() == 0:
        return

    min_index = int(edge_index.min().item())
    max_index = int(edge_index.max().item())
    if min_index < 0 or max_index >= num_nodes:
        raise ValueError("edge_index contains node indices outside [0, num_nodes).")


def _set_seed(seed: int) -> None:
    """Seed the RNGs used by the multicriteria optimizer.

    Parameters
    ----------
    seed : int
        Requested random seed.

    Returns
    -------
    None
        The global RNG state is updated in-place.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clean_undirected_edges(edge_index: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Collapse the input edge list into unique undirected edges.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    device : torch.device
        Device used by the optimization loop.

    Returns
    -------
    torch.Tensor
        Unique undirected edges with shape ``[2, E_unique]``.
    """
    edges = edge_index.to(device=device, dtype=torch.long)
    if edges.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    src = edges[0]
    dst = edges[1]
    non_self = src != dst
    src = src[non_self]
    dst = dst[non_self]
    if src.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    lower = torch.minimum(src, dst)
    upper = torch.maximum(src, dst)
    unique_pairs = torch.unique(torch.stack([lower, upper], dim=1), dim=0)
    return unique_pairs.transpose(0, 1).contiguous()


def _build_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> list[list[tuple[int, float]]]:
    """Build a deterministic undirected adjacency list.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    list[list[tuple[int, float]]]
        Sorted neighbor lists for each node.
    """
    return _shared_build_undirected_adjacency(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )


def _all_pairs_shortest_paths(
    adjacency: list[list[tuple[int, float]]],
    device: torch.device,
    weighted: bool,
) -> torch.Tensor:
    """Compute the full all-pairs distance matrix.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.
    device : torch.device
        Device used for the returned tensor.
    weighted : bool
        Whether to use Dijkstra instead of BFS.

    Returns
    -------
    torch.Tensor
        Distance matrix with shape ``[N, N]`` and ``inf`` for unreachable
        pairs.
    """
    distances = _shared_all_pairs_shortest_paths(adjacency, weighted=weighted)
    cleaned = distances.astype(np.float64, copy=False)
    if not weighted:
        cleaned = cleaned.copy()
        cleaned[cleaned < 0] = np.inf
    return torch.tensor(cleaned, dtype=torch.float32, device=device)


def _build_stress_terms(distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert the distance matrix into sampled stress pairs and weights.

    Parameters
    ----------
    distances : torch.Tensor
        All-pairs distance matrix with shape ``[N, N]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Pair indices with shape ``[2, P]``, graph distances with shape ``[P]``,
        and inverse-square weights with shape ``[P]``.
    """
    upper = torch.triu_indices(
        distances.shape[0],
        distances.shape[1],
        offset=1,
        device=distances.device,
    )
    upper_distances = distances[upper[0], upper[1]]
    finite_mask = torch.isfinite(upper_distances) & (upper_distances > 0)
    pairs = upper[:, finite_mask]
    positive_distances = upper_distances[finite_mask]
    weights = 1.0 / (positive_distances.square() + _EPS)
    return pairs, positive_distances, weights


def _graph_knn_mask(distances: torch.Tensor, k: int) -> torch.Tensor:
    """Build a binary graph-neighborhood target mask.

    Parameters
    ----------
    distances : torch.Tensor
        All-pairs distance matrix with shape ``[N, N]``.
    k : int
        Number of graph neighbors retained per row.

    Returns
    -------
    torch.Tensor
        Boolean mask with shape ``[N, N]``.
    """
    num_nodes = distances.shape[0]
    mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=distances.device)
    if num_nodes == 0:
        return mask

    safe_distances = distances.clone()
    safe_distances.fill_diagonal_(float("inf"))
    max_neighbors = min(max(k, 1), max(num_nodes - 1, 1))
    for node in range(num_nodes):
        row = safe_distances[node]
        finite_mask = torch.isfinite(row)
        if not bool(finite_mask.any().item()):
            continue
        candidate_indices = torch.nonzero(finite_mask, as_tuple=False).flatten()
        candidate_distances = row[candidate_indices]
        take = min(max_neighbors, int(candidate_indices.numel()))
        _, order = torch.topk(candidate_distances, k=take, largest=False)
        mask[node, candidate_indices[order]] = True
    return mask


def _prepare_state(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    needs_distances: bool,
    edge_weights: Optional[torch.Tensor],
) -> _PreparedState:
    """Precompute the graph structures needed by active criteria.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    device : torch.device
        Optimization device.
    needs_distances : bool
        Whether shortest-path-derived criteria are active.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    _PreparedState
        Precomputed state shared by the loss terms.
    """
    edges = _clean_undirected_edges(edge_index=edge_index, device=device)
    adjacency = _build_adjacency(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    if not needs_distances:
        return _PreparedState(
            device=device,
            edges=edges,
            adjacency=adjacency,
            all_pairs_distances=None,
            stress_pairs=None,
            stress_distances=None,
            stress_weights=None,
            graph_knn_mask=None,
        )

    distances = _all_pairs_shortest_paths(
        adjacency=adjacency,
        device=device,
        weighted=edge_weights is not None,
    )
    stress_pairs, stress_distances, stress_weights = _build_stress_terms(distances)
    graph_knn_mask = _graph_knn_mask(distances=distances, k=_DEFAULT_GRAPH_K)
    return _PreparedState(
        device=device,
        edges=edges,
        adjacency=adjacency,
        all_pairs_distances=distances,
        stress_pairs=stress_pairs,
        stress_distances=stress_distances,
        stress_weights=stress_weights,
        graph_knn_mask=graph_knn_mask,
    )


def _constant_schedule(weight: float) -> Callable[[int], float]:
    """Wrap a static weight in a schedule callable.

    Parameters
    ----------
    weight : float
        Fixed criterion weight.

    Returns
    -------
    Callable[[int], float]
        Schedule returning ``weight`` for every step.
    """
    return lambda _step: weight


def _resolve_schedules(
    criteria: Optional[Dict[str, float]],
    criteria_schedules: Optional[Dict[str, SmoothSteps]],
) -> Dict[str, Callable[[int], float]]:
    """Resolve the active multicriteria weight schedules.

    Parameters
    ----------
    criteria : dict[str, float] | None
        Static criterion weights.
    criteria_schedules : dict[str, SmoothSteps] | None
        Explicit per-criterion schedules.

    Returns
    -------
    dict[str, Callable[[int], float]]
        Criterion schedules keyed by criterion name.
    """
    resolved_criteria = {"stress": 1.0} if criteria is None and criteria_schedules is None else {}
    if criteria is not None:
        resolved_criteria.update(criteria)

    schedules: Dict[str, Callable[[int], float]] = {
        name: _constant_schedule(weight) for name, weight in resolved_criteria.items()
    }
    if criteria_schedules is not None:
        schedules.update(criteria_schedules)
    return schedules


def _sample_indices(total: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample indices with replacement from a finite range.

    Parameters
    ----------
    total : int
        Number of available items.
    batch_size : int
        Requested sample size.
    device : torch.device
        Device used for the returned tensor.

    Returns
    -------
    torch.Tensor
        Index tensor with shape ``[B]``.
    """
    if total <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    return torch.randint(0, total, (batch_size,), device=device)


def _sample_nodes(num_nodes: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample a node mini-batch.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    batch_size : int
        Requested sample size.
    device : torch.device
        Device used for the returned tensor.

    Returns
    -------
    torch.Tensor
        Node-index tensor with shape ``[B]``.
    """
    return _sample_indices(total=num_nodes, batch_size=batch_size, device=device)


def _sample_pairs(pairs: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Sample a mini-batch of node pairs.

    Parameters
    ----------
    pairs : torch.Tensor
        Pair index tensor with shape ``[2, P]``.
    batch_size : int
        Requested sample size.

    Returns
    -------
    torch.Tensor
        Sampled pair indices with shape ``[2, B]``.
    """
    if pairs.numel() == 0:
        return pairs
    indices = _sample_indices(total=pairs.shape[1], batch_size=batch_size, device=pairs.device)
    return pairs[:, indices]


def _sample_edges(edges: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Sample a mini-batch of edges.

    Parameters
    ----------
    edges : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    batch_size : int
        Requested sample size.

    Returns
    -------
    torch.Tensor
        Sampled edges with shape ``[2, B]``.
    """
    return _sample_pairs(pairs=edges, batch_size=batch_size)


def _sample_edge_pairs(edges: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample non-incident edge pairs for crossing-related criteria.

    Parameters
    ----------
    edges : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    batch_size : int
        Requested sample size.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Two edge batches, each with shape ``[2, B]``.
    """
    if edges.shape[1] < 2:
        empty = torch.empty((2, 0), dtype=torch.long, device=edges.device)
        return empty, empty

    edge_count = edges.shape[1]
    left_batches: list[torch.Tensor] = []
    right_batches: list[torch.Tensor] = []
    collected = 0
    while collected < batch_size:
        take = max(batch_size - collected, batch_size)
        left_idx = _sample_indices(edge_count, take, edges.device)
        right_idx = _sample_indices(edge_count, take, edges.device)
        left = edges[:, left_idx]
        right = edges[:, right_idx]
        non_self = left_idx != right_idx
        non_incident = (
            (left[0] != right[0])
            & (left[0] != right[1])
            & (left[1] != right[0])
            & (left[1] != right[1])
        )
        mask = non_self & non_incident
        if not bool(mask.any().item()):
            if edge_count <= 2:
                break
            continue
        left_batches.append(left[:, mask])
        right_batches.append(right[:, mask])
        collected += int(mask.sum().item())
        if edge_count <= 2:
            break

    if len(left_batches) == 0:
        empty = torch.empty((2, 0), dtype=torch.long, device=edges.device)
        return empty, empty

    left_cat = torch.cat(left_batches, dim=1)[:, :batch_size]
    right_cat = torch.cat(right_batches, dim=1)[:, :batch_size]
    return left_cat, right_cat


def _stress_loss(
    pos: torch.Tensor,
    pair_batch: torch.Tensor,
    pair_distances: torch.Tensor,
    pair_weights: torch.Tensor,
) -> torch.Tensor:
    """Evaluate stress over a sampled pair batch.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    pair_batch : torch.Tensor
        Pair indices with shape ``[2, B]``.
    pair_distances : torch.Tensor
        Target graph distances with shape ``[B]``.
    pair_weights : torch.Tensor
        Stress weights with shape ``[B]``.

    Returns
    -------
    torch.Tensor
        Scalar stress loss.
    """
    if pair_batch.numel() == 0:
        return pos.sum() * 0.0
    lengths = torch.linalg.norm(pos[pair_batch[0]] - pos[pair_batch[1]], dim=1)
    return (pair_weights * (lengths - pair_distances).square()).mean()


def _ideal_edge_length_loss(
    pos: torch.Tensor,
    edge_batch: torch.Tensor,
    target: float,
) -> torch.Tensor:
    """Evaluate the ideal-edge-length criterion.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_batch : torch.Tensor
        Sampled edges with shape ``[2, B]``.
    target : float
        Target Euclidean edge length.

    Returns
    -------
    torch.Tensor
        Scalar ideal-edge-length loss.
    """
    if edge_batch.numel() == 0:
        return pos.sum() * 0.0
    lengths = torch.linalg.norm(pos[edge_batch[0]] - pos[edge_batch[1]], dim=1)
    safe_target = max(target, _EPS)
    return (((lengths - safe_target) / safe_target).square()).mean()


def _lovasz_grad(labels_sorted: torch.Tensor) -> torch.Tensor:
    """Compute the Lovasz-extension gradient for binary labels.

    Parameters
    ----------
    labels_sorted : torch.Tensor
        Sorted binary labels with shape ``[M]``.

    Returns
    -------
    torch.Tensor
        Lovasz gradient coefficients with shape ``[M]``.
    """
    positives = labels_sorted.sum()
    intersection = positives - labels_sorted.cumsum(dim=0)
    union = positives + (1.0 - labels_sorted).cumsum(dim=0)
    jaccard = 1.0 - intersection / union.clamp(min=_EPS)
    if labels_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Evaluate the binary Lovasz hinge loss on flattened inputs.

    Parameters
    ----------
    logits : torch.Tensor
        Unnormalized scores with shape ``[M]``.
    labels : torch.Tensor
        Binary labels with shape ``[M]``.

    Returns
    -------
    torch.Tensor
        Scalar Lovasz hinge loss.
    """
    if logits.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, permutation = torch.sort(errors, descending=True)
    labels_sorted = labels[permutation]
    return torch.dot(F.relu(errors_sorted), _lovasz_grad(labels_sorted))


def _neighborhood_preservation_loss(
    pos: torch.Tensor,
    anchor_nodes: torch.Tensor,
    graph_knn_mask: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a Lovasz-hinge neighborhood-preservation loss.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    anchor_nodes : torch.Tensor
        Anchor-node indices with shape ``[B]``.
    graph_knn_mask : torch.Tensor
        Binary graph-neighborhood targets with shape ``[N, N]``.

    Returns
    -------
    torch.Tensor
        Scalar neighborhood-preservation loss.
    """
    if anchor_nodes.numel() == 0:
        return pos.sum() * 0.0

    distances = torch.cdist(pos[anchor_nodes], pos)
    logits = -distances
    logits.scatter_(1, anchor_nodes.unsqueeze(1), -1.0e6)
    labels = graph_knn_mask[anchor_nodes].to(dtype=pos.dtype)
    row_losses = [_lovasz_hinge_flat(logits[row], labels[row]) for row in range(logits.shape[0])]
    return torch.stack(row_losses).mean() if row_losses else pos.sum() * 0.0


def _cross2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute 2D cross products row-wise.

    Parameters
    ----------
    a : torch.Tensor
        First tensor with shape ``[B, 2]``.
    b : torch.Tensor
        Second tensor with shape ``[B, 2]``.

    Returns
    -------
    torch.Tensor
        Cross products with shape ``[B]``.
    """
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _crossing_probability(
    pos: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Estimate crossing likelihood with a smooth orientation proxy.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    left : torch.Tensor
        Left edge batch with shape ``[2, B]``.
    right : torch.Tensor
        Right edge batch with shape ``[2, B]``.

    Returns
    -------
    torch.Tensor
        Crossing probabilities with shape ``[B]``.
    """
    if left.numel() == 0 or right.numel() == 0:
        return torch.empty((0,), dtype=pos.dtype, device=pos.device)

    a = pos[left[0]]
    b = pos[left[1]]
    c = pos[right[0]]
    d = pos[right[1]]
    ab = b - a
    cd = d - c
    orient_abc = _cross2d(ab, c - a)
    orient_abd = _cross2d(ab, d - a)
    orient_cda = _cross2d(cd, a - c)
    orient_cdb = _cross2d(cd, b - c)
    return torch.sigmoid(-_CROSSING_SHARPNESS * orient_abc * orient_abd) * torch.sigmoid(
        -_CROSSING_SHARPNESS * orient_cda * orient_cdb
    )


def _crossings_loss(pos: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Evaluate the smooth crossing proxy loss.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    left : torch.Tensor
        Left edge batch with shape ``[2, B]``.
    right : torch.Tensor
        Right edge batch with shape ``[2, B]``.

    Returns
    -------
    torch.Tensor
        Scalar crossing loss.
    """
    probabilities = _crossing_probability(pos=pos, left=left, right=right)
    if probabilities.numel() == 0:
        return pos.sum() * 0.0
    return probabilities.mean()


def _crossing_angle_loss(
    pos: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the crossing-angle maximization criterion.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    left : torch.Tensor
        Left edge batch with shape ``[2, B]``.
    right : torch.Tensor
        Right edge batch with shape ``[2, B]``.

    Returns
    -------
    torch.Tensor
        Scalar crossing-angle loss.
    """
    probabilities = _crossing_probability(pos=pos, left=left, right=right)
    if probabilities.numel() == 0:
        return pos.sum() * 0.0

    left_vec = pos[left[1]] - pos[left[0]]
    right_vec = pos[right[1]] - pos[right[0]]
    denominator = torch.linalg.norm(left_vec, dim=1).clamp(min=_EPS) * torch.linalg.norm(
        right_vec, dim=1
    ).clamp(min=_EPS)
    cos_sq = (torch.sum(left_vec * right_vec, dim=1) / denominator).square()
    return (probabilities * cos_sq).sum() / probabilities.sum().clamp(min=1.0)


def _aspect_ratio_loss(pos: torch.Tensor, target: float) -> torch.Tensor:
    """Evaluate the aspect-ratio criterion from singular values.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    target : float
        Target minor-to-major singular-value ratio.

    Returns
    -------
    torch.Tensor
        Scalar aspect-ratio loss.
    """
    if pos.shape[0] <= 1:
        return pos.sum() * 0.0
    centered = pos - pos.mean(dim=0, keepdim=True)
    _, singular_values, _ = torch.linalg.svd(centered, full_matrices=False)
    if singular_values.numel() < 2:
        return pos.sum() * 0.0
    ratio = (singular_values[1] / singular_values[0].clamp(min=_EPS)).clamp(_EPS, 1.0 - _EPS)
    target_tensor = torch.tensor(
        float(min(max(target, _EPS), 1.0 - _EPS)),
        device=pos.device,
        dtype=pos.dtype,
    )
    return F.binary_cross_entropy(ratio, target_tensor)


def _angular_resolution_loss(
    pos: torch.Tensor,
    anchor_nodes: torch.Tensor,
    adjacency: list[list[tuple[int, float]]],
) -> torch.Tensor:
    """Evaluate the angular-resolution criterion on sampled nodes.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    anchor_nodes : torch.Tensor
        Anchor-node indices with shape ``[B]``.
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.

    Returns
    -------
    torch.Tensor
        Scalar angular-resolution loss.
    """
    losses: list[torch.Tensor] = []
    for node in anchor_nodes.tolist():
        neighbors = adjacency[node]
        degree = len(neighbors)
        if degree < 2:
            continue
        neighbor_index = torch.tensor(
            [neighbor for neighbor, _ in neighbors],
            dtype=torch.long,
            device=pos.device,
        )
        vectors = pos[neighbor_index] - pos[node]
        vector_count = vectors.shape[0]
        if vector_count < 2:
            continue
        left = vectors.unsqueeze(1).expand(vector_count, vector_count, 2)
        right = vectors.unsqueeze(0).expand(vector_count, vector_count, 2)
        denom = torch.linalg.norm(left, dim=2).clamp(min=_EPS) * torch.linalg.norm(
            right, dim=2
        ).clamp(min=_EPS)
        cosines = torch.clamp((left * right).sum(dim=2) / denom, -1.0 + _EPS, 1.0 - _EPS)
        angles = torch.arccos(cosines)
        upper = torch.triu_indices(vector_count, vector_count, offset=1, device=pos.device)
        pair_angles = angles[upper[0], upper[1]]
        target_angle = 2.0 * math.pi / float(degree)
        losses.append(
            F.relu(torch.full_like(pair_angles, target_angle) - pair_angles).square().mean()
        )
    return torch.stack(losses).mean() if losses else pos.sum() * 0.0


def _vertex_resolution_loss(
    pos: torch.Tensor,
    pair_batch: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Evaluate the vertex-resolution hinge loss.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    pair_batch : torch.Tensor
        Pair indices with shape ``[2, B]``.
    threshold : float
        Minimum desired pairwise distance.

    Returns
    -------
    torch.Tensor
        Scalar vertex-resolution loss.
    """
    if pair_batch.numel() == 0:
        return pos.sum() * 0.0
    distances = torch.linalg.norm(pos[pair_batch[0]] - pos[pair_batch[1]], dim=1)
    return F.relu(torch.full_like(distances, threshold) - distances).square().mean()


def _criterion_loss(
    name: str,
    pos: torch.Tensor,
    state: _PreparedState,
    batch_size: int,
) -> torch.Tensor:
    """Evaluate one named criterion on a sampled mini-batch.

    Parameters
    ----------
    name : str
        Criterion name.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    state : _PreparedState
        Precomputed graph state.
    batch_size : int
        Requested mini-batch size.

    Returns
    -------
    torch.Tensor
        Scalar criterion loss.
    """
    if name == "stress":
        if (
            state.stress_pairs is None
            or state.stress_distances is None
            or state.stress_weights is None
        ):
            return pos.sum() * 0.0
        sample_index = _sample_indices(
            total=state.stress_pairs.shape[1],
            batch_size=batch_size,
            device=state.device,
        )
        return _stress_loss(
            pos=pos,
            pair_batch=state.stress_pairs[:, sample_index],
            pair_distances=state.stress_distances[sample_index],
            pair_weights=state.stress_weights[sample_index],
        )
    if name == "ideal_edge_length":
        return _ideal_edge_length_loss(
            pos=pos,
            edge_batch=_sample_edges(state.edges, batch_size=batch_size),
            target=_DEFAULT_IDEAL_EDGE_LENGTH,
        )
    if name == "neighborhood_preservation":
        if state.graph_knn_mask is None:
            return pos.sum() * 0.0
        return _neighborhood_preservation_loss(
            pos=pos,
            anchor_nodes=_sample_nodes(
                num_nodes=pos.shape[0],
                batch_size=batch_size,
                device=state.device,
            ),
            graph_knn_mask=state.graph_knn_mask,
        )
    if name == "crossings":
        left, right = _sample_edge_pairs(edges=state.edges, batch_size=batch_size)
        return _crossings_loss(pos=pos, left=left, right=right)
    if name == "crossing_angle_maximization":
        left, right = _sample_edge_pairs(edges=state.edges, batch_size=batch_size)
        return _crossing_angle_loss(pos=pos, left=left, right=right)
    if name == "aspect_ratio":
        return _aspect_ratio_loss(pos=pos, target=_DEFAULT_ASPECT_RATIO_TARGET)
    if name == "angular_resolution":
        return _angular_resolution_loss(
            pos=pos,
            anchor_nodes=_sample_nodes(
                num_nodes=pos.shape[0],
                batch_size=batch_size,
                device=state.device,
            ),
            adjacency=state.adjacency,
        )
    if name == "vertex_resolution":
        if state.stress_pairs is None:
            return pos.sum() * 0.0
        return _vertex_resolution_loss(
            pos=pos,
            pair_batch=_sample_pairs(state.stress_pairs, batch_size=batch_size),
            threshold=_DEFAULT_VERTEX_RESOLUTION,
        )
    raise ValueError(f"Unknown (SGD)^2 criterion: {name}")


def layout_sgd2_multi(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    steps: int = 10_000,
    criteria: Optional[Dict[str, float]] = None,
    criteria_schedules: Optional[Dict[str, SmoothSteps]] = None,
    lr: float = 0.01,
    momentum: float = 0.7,
    grad_clamp: float = 5.0,
    batch_size: int = 16,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lay out a graph with the (SGD)^2 multicriteria optimizer.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None, default=None
        Unused placeholder kept for API compatibility.
    seed : int, default=42
        Random seed used for initialization and sampling.
    steps : int, default=10000
        Maximum number of SGD iterations.
    criteria : dict[str, float] | None, default=None
        Static per-criterion weights. ``None`` defaults to pure stress.
    criteria_schedules : dict[str, SmoothSteps] | None, default=None
        Optional piecewise-smooth criterion schedules.
    lr : float, default=0.01
        SGD learning rate.
    momentum : float, default=0.7
        SGD momentum.
    grad_clamp : float, default=20.0
        Symmetric gradient clamp.
    batch_size : int, default=16
        Global mini-batch size shared by the criteria.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``. When provided, shortest
        paths are computed with Dijkstra instead of BFS.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    _validate_inputs(
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=steps,
        lr=lr,
        momentum=momentum,
        grad_clamp=grad_clamp,
        batch_size=batch_size,
    )
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    schedules = _resolve_schedules(criteria=criteria, criteria_schedules=criteria_schedules)
    active_names = set(schedules)
    needs_distances = bool(
        {"stress", "neighborhood_preservation", "vertex_resolution"} & active_names
    )

    _set_seed(seed)
    state = _prepare_state(
        edge_index=edge_index,
        num_nodes=num_nodes,
        device=device,
        needs_distances=needs_distances,
        edge_weights=edge_weights,
    )

    positions = torch.nn.Parameter(
        torch.randn((num_nodes, 2), device=device, dtype=torch.float32)
        * math.sqrt(float(num_nodes))
    )
    optimizer = torch.optim.SGD([positions], lr=lr, momentum=momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.9,
        patience=20_000,
        min_lr=1.0e-5,
    )

    ema_loss: Optional[float] = None
    for step_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        centered = positions - positions.mean(dim=0, keepdim=True)
        loss = centered.sum() * 0.0
        for name, schedule in schedules.items():
            weight = schedule(step_index)
            if weight == 0.0:
                continue
            loss = loss + weight * _criterion_loss(
                name=name,
                pos=centered,
                state=state,
                batch_size=batch_size,
            )

        loss.backward()
        if positions.grad is not None:
            positions.grad.clamp_(-grad_clamp, grad_clamp)
        optimizer.step()
        with torch.no_grad():
            positions.sub_(positions.mean(dim=0, keepdim=True))

        loss_value = float(loss.detach().item())
        ema_loss = loss_value if ema_loss is None else 0.9 * ema_loss + 0.1 * loss_value
        if step_index % 10 == 0:
            scheduler.step(ema_loss)
        if float(optimizer.param_groups[0]["lr"]) <= 1.0e-5:
            break

    detached = positions.detach()
    return (detached - detached.mean(dim=0, keepdim=True)).to(dtype=torch.float32)

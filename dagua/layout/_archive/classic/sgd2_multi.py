"""(SGD)^2 multicriteria graph layout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dagua.layout._archive.classic._graph_distances import (
    all_pairs_shortest_paths as _shared_all_pairs_shortest_paths,
)
from dagua.layout._archive.classic._graph_distances import (
    build_undirected_adjacency as _shared_build_undirected_adjacency,
)

_EPS = 1.0e-6
_SEGMENT_EPS = 1.0e-9
_DEFAULT_IDEAL_EDGE_LENGTH = 1.0
_DEFAULT_ASPECT_RATIO_TARGET = 1.0
_VERTEX_RESOLUTION_SMOOTHNESS = 0.1
_NEIGHBORHOOD_DEPTH_LIMIT = 2
_NEIGHBORHOOD_NEG_SAMPLE_RATE = 0.5
_NEIGHBORHOOD_K_DIST = 1.5
_CROSSING_DETECTOR_TRAIN_STEPS = 2
_CROSSING_DETECTOR_LR = 0.01


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
    incident_edge_pairs: Optional[torch.Tensor]
    non_incident_edge_pairs: Optional[torch.Tensor]


@dataclass
class _VertexResolutionState:
    """State carried across iterations for the vertex-resolution target."""

    prev_target_dist: torch.Tensor
    prev_weight: float


@dataclass
class _CrossingLossState:
    """Persistent neural detector state for the crossing criterion."""

    detector: nn.Module
    optimizer: torch.optim.Optimizer
    train_loss: nn.Module
    position_loss: nn.Module


class _CrossingDetector(nn.Module):
    """Feed-forward crossing detector matching the reference architecture."""

    def __init__(self) -> None:
        """Initialize the detector layers."""
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(8, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the detector on flattened edge-pair coordinates.

        Parameters
        ----------
        x : torch.Tensor
            Edge-pair coordinates with shape ``[B, 8]``.

        Returns
        -------
        torch.Tensor
            Crossing probabilities with shape ``[B, 1]``.
        """
        return self.main(x)


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
    finite_mask = torch.isfinite(upper_distances)
    pairs = upper[:, finite_mask]
    positive_distances = upper_distances[finite_mask]
    weights = 1.0 / (positive_distances.square() + _EPS)
    return pairs, positive_distances, weights


def _build_incident_edge_pairs(
    adjacency: list[list[tuple[int, float]]],
    device: torch.device,
) -> torch.Tensor:
    """Build sampled incident-edge tuples for angular resolution.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.
    device : torch.device
        Device used for the returned tensor.

    Returns
    -------
    torch.Tensor
        Tuple tensor with shape ``[5, P]`` storing
        ``(degree, node_a, neighbor_b, node_c, neighbor_d)``.
    """
    pairs: list[tuple[int, int, int, int, int]] = []
    for node, neighbors in enumerate(adjacency):
        degree = len(neighbors)
        if degree < 2:
            continue
        neighbor_ids = [neighbor for neighbor, _ in neighbors]
        for left_index in range(degree - 1):
            for right_index in range(left_index + 1, degree):
                pairs.append(
                    (
                        degree,
                        node,
                        neighbor_ids[left_index],
                        node,
                        neighbor_ids[right_index],
                    )
                )
    if not pairs:
        return torch.empty((5, 0), dtype=torch.long, device=device)
    return torch.tensor(pairs, dtype=torch.long, device=device).transpose(0, 1).contiguous()


def _build_non_incident_edge_pairs(edges: torch.Tensor) -> torch.Tensor:
    """Build all non-incident undirected edge pairs.

    Parameters
    ----------
    edges : torch.Tensor
        Undirected edge tensor with shape ``[2, E]``.

    Returns
    -------
    torch.Tensor
        Edge-pair tensor with shape ``[4, P]`` storing
        ``(edge0_src, edge0_dst, edge1_src, edge1_dst)``.
    """
    edge_count = edges.shape[1]
    if edge_count < 2:
        return torch.empty((4, 0), dtype=torch.long, device=edges.device)

    pair_indices = torch.triu_indices(edge_count, edge_count, offset=1, device=edges.device)
    left = edges[:, pair_indices[0]]
    right = edges[:, pair_indices[1]]
    non_incident = (
        (left[0] != right[0])
        & (left[0] != right[1])
        & (left[1] != right[0])
        & (left[1] != right[1])
    )
    if not bool(non_incident.any().item()):
        return torch.empty((4, 0), dtype=torch.long, device=edges.device)
    return torch.cat([left[:, non_incident], right[:, non_incident]], dim=0)


def _prepare_state(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    needs_distances: bool,
    needs_incident_edge_pairs: bool,
    needs_non_incident_edge_pairs: bool,
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
    needs_incident_edge_pairs : bool
        Whether angular-resolution sampling tuples are active.
    needs_non_incident_edge_pairs : bool
        Whether crossing-related edge-pair batches are active.
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
    incident_edge_pairs = (
        _build_incident_edge_pairs(adjacency=adjacency, device=device)
        if needs_incident_edge_pairs
        else None
    )
    non_incident_edge_pairs = (
        _build_non_incident_edge_pairs(edges=edges) if needs_non_incident_edge_pairs else None
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
            incident_edge_pairs=incident_edge_pairs,
            non_incident_edge_pairs=non_incident_edge_pairs,
        )

    distances = _all_pairs_shortest_paths(
        adjacency=adjacency,
        device=device,
        weighted=edge_weights is not None,
    )
    stress_pairs, stress_distances, stress_weights = _build_stress_terms(distances)
    return _PreparedState(
        device=device,
        edges=edges,
        adjacency=adjacency,
        all_pairs_distances=distances,
        stress_pairs=stress_pairs,
        stress_distances=stress_distances,
        stress_weights=stress_weights,
        incident_edge_pairs=incident_edge_pairs,
        non_incident_edge_pairs=non_incident_edge_pairs,
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
        Pair index tensor with shape ``[K, P]``.
    batch_size : int
        Requested sample size.

    Returns
    -------
    torch.Tensor
        Sampled pair indices with shape ``[K, B]``.
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


class _CyclicSampler:
    """Epoch-based mini-batch sampler matching the reference DataLoader.

    Each epoch visits every index exactly once in shuffled order.  When the
    epoch is exhausted a fresh permutation is drawn automatically.
    """

    __slots__ = ("_total", "_device", "_perm", "_offset")

    def __init__(self, total: int, device: torch.device) -> None:
        self._total = total
        self._device = device
        self._perm = torch.randperm(total, device=device)
        self._offset = 0

    def sample(self, batch_size: int) -> torch.Tensor:
        """Return the next ``batch_size`` indices, reshuffling on epoch boundary."""
        if self._total <= 0:
            return torch.empty((0,), dtype=torch.long, device=self._device)
        bs = min(batch_size, self._total)
        if self._offset + bs > self._total:
            self._perm = torch.randperm(self._total, device=self._device)
            self._offset = 0
        out = self._perm[self._offset : self._offset + bs]
        self._offset += bs
        return out


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
    adjacency: list[list[tuple[int, float]]],
) -> torch.Tensor:
    """Evaluate the reference BFS-based neighborhood-preservation loss.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    anchor_nodes : torch.Tensor
        Root-node indices with shape ``[B]``.
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.

    Returns
    -------
    torch.Tensor
        Scalar neighborhood-preservation loss.
    """
    sampled_nodes = _sample_neighborhood_nodes(
        adjacency=adjacency,
        root_nodes=anchor_nodes,
        num_nodes=pos.shape[0],
        device=pos.device,
    )
    if sampled_nodes.numel() == 0:
        return pos.sum() * 0.0

    sampled_pos = pos[sampled_nodes]
    logits = -torch.cdist(sampled_pos, sampled_pos) + _NEIGHBORHOOD_K_DIST
    labels = _induced_adjacency_target(
        sampled_nodes=sampled_nodes,
        adjacency=adjacency,
        device=pos.device,
        dtype=pos.dtype,
    )
    return _lovasz_hinge_flat(logits.reshape(-1), labels.reshape(-1))


def _bfs_nodes(
    adjacency: list[list[tuple[int, float]]],
    root: int,
    depth_limit: int,
) -> list[int]:
    """Collect nodes from a bounded BFS rooted at one node.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.
    root : int
        Root node index.
    depth_limit : int
        Maximum BFS depth.

    Returns
    -------
    list[int]
        Visited node indices in discovery order.
    """
    visited = {root}
    queue: list[tuple[int, int]] = [(root, 0)]
    ordered = [root]
    queue_index = 0
    while queue_index < len(queue):
        node, depth = queue[queue_index]
        queue_index += 1
        if depth >= depth_limit:
            continue
        for neighbor, _weight in adjacency[node]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            ordered.append(neighbor)
            queue.append((neighbor, depth + 1))
    return ordered


def _sample_neighborhood_nodes(
    adjacency: list[list[tuple[int, float]]],
    root_nodes: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the sampled subgraph used by neighborhood preservation.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.
    root_nodes : torch.Tensor
        Root-node indices with shape ``[B]``.
    num_nodes : int
        Number of graph nodes.
    device : torch.device
        Device used for the returned tensor.

    Returns
    -------
    torch.Tensor
        Unique sampled node indices with shape ``[S]``.
    """
    if root_nodes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    positive_nodes: list[int] = []
    for root in root_nodes.tolist():
        positive_nodes.extend(
            _bfs_nodes(adjacency=adjacency, root=root, depth_limit=_NEIGHBORHOOD_DEPTH_LIMIT)
        )
    if not positive_nodes:
        return torch.empty((0,), dtype=torch.long, device=device)

    positive = torch.tensor(
        sorted(set(positive_nodes)),
        dtype=torch.long,
        device=device,
    )
    negative_count = int(_NEIGHBORHOOD_NEG_SAMPLE_RATE * int(positive.numel()))
    if negative_count <= 0:
        return positive

    negatives = torch.randint(0, num_nodes, (negative_count,), device=device)
    return torch.unique(torch.cat([positive, negatives]), sorted=True)


def _induced_adjacency_target(
    sampled_nodes: torch.Tensor,
    adjacency: list[list[tuple[int, float]]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the induced adjacency-plus-identity target matrix.

    Parameters
    ----------
    sampled_nodes : torch.Tensor
        Sampled node indices with shape ``[S]``.
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.
    device : torch.device
        Device used for the returned tensor.
    dtype : torch.dtype
        Floating dtype of the returned tensor.

    Returns
    -------
    torch.Tensor
        Dense target matrix with shape ``[S, S]``.
    """
    size = int(sampled_nodes.numel())
    target = torch.eye(size, dtype=dtype, device=device)
    local_index = {int(node): offset for offset, node in enumerate(sampled_nodes.tolist())}
    for row, node in enumerate(sampled_nodes.tolist()):
        for neighbor, _weight in adjacency[node]:
            column = local_index.get(neighbor)
            if column is not None:
                target[row, column] = 1.0
    return target


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


def _edge_pair_positions(
    pos: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Gather flattened coordinates for one batch of edge pairs.

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
        Flattened edge-pair coordinates with shape ``[B, 8]``.
    """
    if left.numel() == 0 or right.numel() == 0:
        return torch.empty((0, 8), dtype=pos.dtype, device=pos.device)
    pair_indices = torch.stack([left[0], left[1], right[0], right[1]], dim=1)
    return pos[pair_indices].reshape(-1, 8)


def _point_on_segment(
    point: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """Test whether points lie on closed 2D segments.

    Parameters
    ----------
    point : torch.Tensor
        Query points with shape ``[B, 2]``.
    start : torch.Tensor
        Segment start points with shape ``[B, 2]``.
    end : torch.Tensor
        Segment end points with shape ``[B, 2]``.

    Returns
    -------
    torch.Tensor
        Boolean mask with shape ``[B]``.
    """
    min_x = torch.minimum(start[:, 0], end[:, 0]) - _SEGMENT_EPS
    max_x = torch.maximum(start[:, 0], end[:, 0]) + _SEGMENT_EPS
    min_y = torch.minimum(start[:, 1], end[:, 1]) - _SEGMENT_EPS
    max_y = torch.maximum(start[:, 1], end[:, 1]) + _SEGMENT_EPS
    return (
        (point[:, 0] >= min_x)
        & (point[:, 0] <= max_x)
        & (point[:, 1] >= min_y)
        & (point[:, 1] <= max_y)
    )


def _are_edge_pairs_crossed(edge_pair_pos: torch.Tensor) -> torch.Tensor:
    """Compute exact geometric crossing labels for disjoint edge pairs.

    Parameters
    ----------
    edge_pair_pos : torch.Tensor
        Flattened edge-pair coordinates with shape ``[B, 8]``.

    Returns
    -------
    torch.Tensor
        Boolean crossing labels with shape ``[B]``.
    """
    if edge_pair_pos.numel() == 0:
        return torch.empty((0,), dtype=torch.bool, device=edge_pair_pos.device)

    segments = edge_pair_pos.reshape(-1, 4, 2)
    a = segments[:, 0]
    b = segments[:, 1]
    c = segments[:, 2]
    d = segments[:, 3]

    orient_abc = _cross2d(b - a, c - a)
    orient_abd = _cross2d(b - a, d - a)
    orient_cda = _cross2d(d - c, a - c)
    orient_cdb = _cross2d(d - c, b - c)

    proper = (
        ((orient_abc > _SEGMENT_EPS) & (orient_abd < -_SEGMENT_EPS))
        | ((orient_abc < -_SEGMENT_EPS) & (orient_abd > _SEGMENT_EPS))
    ) & (
        ((orient_cda > _SEGMENT_EPS) & (orient_cdb < -_SEGMENT_EPS))
        | ((orient_cda < -_SEGMENT_EPS) & (orient_cdb > _SEGMENT_EPS))
    )
    collinear = (
        ((orient_abc.abs() <= _SEGMENT_EPS) & _point_on_segment(c, a, b))
        | ((orient_abd.abs() <= _SEGMENT_EPS) & _point_on_segment(d, a, b))
        | ((orient_cda.abs() <= _SEGMENT_EPS) & _point_on_segment(a, c, d))
        | ((orient_cdb.abs() <= _SEGMENT_EPS) & _point_on_segment(b, c, d))
    )
    return proper | collinear


def _crossings_loss(
    pos: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    crossing_state: _CrossingLossState,
) -> torch.Tensor:
    """Evaluate the reference neural crossing loss.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    left : torch.Tensor
        Left edge batch with shape ``[2, B]``.
    right : torch.Tensor
        Right edge batch with shape ``[2, B]``.
    crossing_state : _CrossingLossState
        Persistent detector state.

    Returns
    -------
    torch.Tensor
        Scalar crossing loss.
    """
    edge_pair_pos = _edge_pair_positions(pos=pos, left=left, right=right)
    if edge_pair_pos.numel() == 0:
        return pos.sum() * 0.0

    labels = _are_edge_pairs_crossed(edge_pair_pos.detach()).to(
        device=pos.device,
        dtype=pos.dtype,
    )
    crossing_state.detector.train()
    for _ in range(_CROSSING_DETECTOR_TRAIN_STEPS):
        preds = crossing_state.detector(edge_pair_pos.detach()).view(-1)
        train_loss = crossing_state.train_loss(preds, labels)
        crossing_state.optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        crossing_state.optimizer.step()

    crossing_state.detector.eval()
    preds = crossing_state.detector(edge_pair_pos).view(-1)
    return crossing_state.position_loss(preds, torch.zeros_like(preds))


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
    edge_pair_pos = _edge_pair_positions(pos=pos, left=left, right=right)
    if edge_pair_pos.numel() == 0:
        return pos.sum() * 0.0

    labels = _are_edge_pairs_crossed(edge_pair_pos).to(device=pos.device, dtype=pos.dtype)
    left_vec = pos[left[1]] - pos[left[0]]
    right_vec = pos[right[1]] - pos[right[0]]
    similarities = F.cosine_similarity(left_vec, right_vec, dim=1)
    similarity_sq = similarities.square()
    return (labels * similarity_sq / (1.0 - similarity_sq + _EPS)).mean()


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
        float(min(max(target, 0.0), 1.0)),
        device=pos.device,
        dtype=pos.dtype,
    )
    return F.binary_cross_entropy(ratio, target_tensor, reduction="sum")


def _angular_resolution_loss(
    pos: torch.Tensor,
    pair_batch: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the reference angular-resolution criterion.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    pair_batch : torch.Tensor
        Incident-edge tuple batch with shape ``[5, B]``.

    Returns
    -------
    torch.Tensor
        Scalar angular-resolution loss.
    """
    if pair_batch.numel() == 0:
        return pos.sum() * 0.0

    degrees = pair_batch[0].to(device=pos.device, dtype=pos.dtype)
    a = pair_batch[1]
    b = pair_batch[2]
    c = pair_batch[3]
    d = pair_batch[4]
    similarities = F.cosine_similarity(pos[b] - pos[a], pos[d] - pos[c], dim=1)
    angles = torch.arccos(similarities.clamp(-0.99, 0.99))
    optimal = 2.0 * math.pi / degrees.clamp(min=1.0)
    normalized = F.relu((-angles + optimal) / optimal.clamp(min=_EPS))
    return F.binary_cross_entropy(normalized, torch.zeros_like(angles))


def _vertex_resolution_loss(
    pos: torch.Tensor,
    pair_batch: torch.Tensor,
    state: _VertexResolutionState,
) -> torch.Tensor:
    """Evaluate the adaptive vertex-resolution loss.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    pair_batch : torch.Tensor
        Pair indices with shape ``[2, B]``.
    state : _VertexResolutionState
        Smoothed target-distance state carried across iterations.

    Returns
    -------
    torch.Tensor
        Scalar vertex-resolution loss.
    """
    if pair_batch.numel() == 0:
        return pos.sum() * 0.0
    distances = torch.linalg.norm(pos[pair_batch[0]] - pos[pair_batch[1]], dim=1)
    dmax = distances.max().detach()
    target = 1.0 / math.sqrt(float(max(pos.shape[0], 1)))
    target_dist = torch.as_tensor(target, device=pos.device, dtype=pos.dtype) * dmax
    previous_target = state.prev_target_dist.to(device=pos.device, dtype=pos.dtype)
    weight = state.prev_weight * _VERTEX_RESOLUTION_SMOOTHNESS + 1.0
    smoothed_target = (
        torch.maximum(target_dist, previous_target)
        + torch.minimum(target_dist, previous_target) * _VERTEX_RESOLUTION_SMOOTHNESS
    ) / weight
    state.prev_target_dist = smoothed_target.detach()
    state.prev_weight = weight
    return F.relu(1.0 - distances / smoothed_target.clamp(min=_EPS)).square().mean()


def _criterion_loss(
    name: str,
    pos: torch.Tensor,
    state: _PreparedState,
    batch_size: int,
    sampler: Optional[_CyclicSampler] = None,
    vertex_resolution_state: Optional[_VertexResolutionState] = None,
    crossing_state: Optional[_CrossingLossState] = None,
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
    sampler : _CyclicSampler | None
        Epoch-based cyclic sampler for this criterion.  Falls back to
        random sampling when ``None``.
    vertex_resolution_state : _VertexResolutionState | None
        Stateful smoothing data for vertex resolution.
    crossing_state : _CrossingLossState | None
        Persistent detector state for the crossing criterion.

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
        if sampler is not None:
            sample_index = sampler.sample(batch_size)
        else:
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
        if sampler is not None:
            idx = sampler.sample(batch_size)
            edge_batch = state.edges[:, idx]
        else:
            edge_batch = _sample_edges(state.edges, batch_size=batch_size)
        return _ideal_edge_length_loss(
            pos=pos,
            edge_batch=edge_batch,
            target=_DEFAULT_IDEAL_EDGE_LENGTH,
        )
    if name == "neighborhood_preservation":
        if sampler is not None:
            anchor_nodes = sampler.sample(batch_size)
        else:
            anchor_nodes = _sample_nodes(
                num_nodes=pos.shape[0],
                batch_size=batch_size,
                device=state.device,
            )
        return _neighborhood_preservation_loss(
            pos=pos,
            anchor_nodes=anchor_nodes,
            adjacency=state.adjacency,
        )
    if name == "crossings":
        if crossing_state is None:
            raise ValueError("crossing_state is required for the crossings criterion.")
        if state.non_incident_edge_pairs is None:
            return pos.sum() * 0.0
        if sampler is not None:
            idx = sampler.sample(batch_size)
            pair_batch = state.non_incident_edge_pairs[:, idx]
        else:
            pair_batch = _sample_pairs(state.non_incident_edge_pairs, batch_size=batch_size)
        left = pair_batch[:2]
        right = pair_batch[2:]
        return _crossings_loss(
            pos=pos,
            left=left,
            right=right,
            crossing_state=crossing_state,
        )
    if name == "crossing_angle_maximization":
        if state.non_incident_edge_pairs is None:
            left, right = _sample_edge_pairs(edges=state.edges, batch_size=batch_size)
        elif sampler is not None:
            idx = sampler.sample(batch_size)
            pair_batch = state.non_incident_edge_pairs[:, idx]
            left = pair_batch[:2]
            right = pair_batch[2:]
        else:
            pair_batch = _sample_pairs(state.non_incident_edge_pairs, batch_size=batch_size)
            left = pair_batch[:2]
            right = pair_batch[2:]
        return _crossing_angle_loss(pos=pos, left=left, right=right)
    if name == "aspect_ratio":
        if sampler is not None:
            idx = sampler.sample(batch_size)
            sampled_pos = pos[idx]
        else:
            sampled_pos = pos
        return _aspect_ratio_loss(pos=sampled_pos, target=_DEFAULT_ASPECT_RATIO_TARGET)
    if name == "angular_resolution":
        if state.incident_edge_pairs is None:
            return pos.sum() * 0.0
        if sampler is not None:
            idx = sampler.sample(batch_size)
            pair_batch = state.incident_edge_pairs[:, idx]
        else:
            pair_batch = _sample_pairs(state.incident_edge_pairs, batch_size=batch_size)
        return _angular_resolution_loss(pos=pos, pair_batch=pair_batch)
    if name == "vertex_resolution":
        if state.stress_pairs is None:
            return pos.sum() * 0.0
        if vertex_resolution_state is None:
            raise ValueError("vertex_resolution_state is required for vertex resolution.")
        if sampler is not None:
            idx = sampler.sample(batch_size)
            pair_batch = state.stress_pairs[:, idx]
        else:
            pair_batch = _sample_pairs(state.stress_pairs, batch_size=batch_size)
        return _vertex_resolution_loss(
            pos=pos,
            pair_batch=pair_batch,
            state=vertex_resolution_state,
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
    lr: float = 1.0,
    momentum: float = 0.7,
    grad_clamp: float = 4.0,
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
    lr : float, default=1.0
        SGD learning rate.
    momentum : float, default=0.7
        SGD momentum.
    grad_clamp : float, default=4.0
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
    needs_distances = bool({"stress", "vertex_resolution"} & active_names)
    needs_incident_edge_pairs = "angular_resolution" in active_names
    needs_non_incident_edge_pairs = bool(
        {"crossings", "crossing_angle_maximization"} & active_names
    )

    _set_seed(seed)
    state = _prepare_state(
        edge_index=edge_index,
        num_nodes=num_nodes,
        device=device,
        needs_distances=needs_distances,
        needs_incident_edge_pairs=needs_incident_edge_pairs,
        needs_non_incident_edge_pairs=needs_non_incident_edge_pairs,
        edge_weights=edge_weights,
    )

    # Build one cyclic sampler per criterion, matching the reference DataLoader
    # pattern: each criterion iterates its pool once per epoch in shuffled order.
    samplers: Dict[str, _CyclicSampler] = {}
    for sname in schedules:
        if sname in ("stress", "vertex_resolution") and state.stress_pairs is not None:
            samplers[sname] = _CyclicSampler(state.stress_pairs.shape[1], device)
        elif sname == "ideal_edge_length" and state.edges.shape[1] > 0:
            samplers[sname] = _CyclicSampler(state.edges.shape[1], device)
        elif sname in ("neighborhood_preservation", "aspect_ratio"):
            samplers[sname] = _CyclicSampler(num_nodes, device)
        elif sname == "angular_resolution" and state.incident_edge_pairs is not None:
            samplers[sname] = _CyclicSampler(state.incident_edge_pairs.shape[1], device)
        elif (
            sname in ("crossings", "crossing_angle_maximization")
            and state.non_incident_edge_pairs is not None
        ):
            samplers[sname] = _CyclicSampler(state.non_incident_edge_pairs.shape[1], device)

    positions = torch.nn.Parameter(
        torch.randn((num_nodes, 2), device=device, dtype=torch.float32)
        * math.sqrt(float(num_nodes))
    )
    crossing_state = None
    if "crossings" in active_names:
        crossing_detector = _CrossingDetector().to(device=device)
        crossing_state = _CrossingLossState(
            detector=crossing_detector,
            optimizer=torch.optim.Adam(crossing_detector.parameters(), lr=_CROSSING_DETECTOR_LR),
            train_loss=nn.BCELoss(),
            position_loss=nn.BCELoss(reduction="sum"),
        )
    vertex_resolution_state = None
    if "vertex_resolution" in active_names:
        vertex_resolution_state = _VertexResolutionState(
            prev_target_dist=torch.tensor(1.0, device=device, dtype=torch.float32),
            prev_weight=0.0,
        )
    optimizer = torch.optim.SGD([positions], lr=lr, momentum=momentum, nesterov=True)
    # Reference uses ReduceLROnPlateau(patience=20000), stepped every 10 iters.
    # With 10K max iterations (= 1000 scheduler steps) and patience 20000,
    # the lr effectively never decays.  We replicate this exactly.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.9,
        patience=20000,
        min_lr=1.0e-5,
    )

    # EMA smoothing matching reference: half-life of 100 iterations.
    _ema_decay = 0.5 ** (1.0 / 100.0)
    ema_weighted_sum = 0.0
    ema_total_weight = 0.0

    for step_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = positions.sum() * 0.0
        for name, schedule in schedules.items():
            weight = schedule(step_index)
            if weight == 0.0:
                continue
            loss = loss + weight * _criterion_loss(
                name=name,
                pos=positions,
                state=state,
                batch_size=batch_size,
                sampler=samplers.get(name),
                vertex_resolution_state=vertex_resolution_state,
                crossing_state=crossing_state,
            )

        loss.backward()
        if positions.grad is not None:
            positions.grad.clamp_(-grad_clamp, grad_clamp)
        optimizer.step()

        # EMA + scheduler step every 10 iterations (reference pattern).
        loss_value = float(loss.detach().item())
        ema_weighted_sum = ema_weighted_sum * _ema_decay + loss_value
        ema_total_weight = ema_total_weight * _ema_decay + 1.0
        if step_index % 10 == 0:
            scheduler.step(ema_weighted_sum / ema_total_weight)
        if float(optimizer.param_groups[0]["lr"]) <= 1.0e-5:
            break

    detached = positions.detach()
    return detached.to(dtype=torch.float32)

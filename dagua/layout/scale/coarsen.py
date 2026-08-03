"""Family-agnostic coarsening substrate for scale strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

_MIN_COARSE_SIZE = 4
_MIN_HEM_REDUCTION = 0.75
_DEFAULT_MIN_SHRINK = 0.50
_DEFAULT_BUCKET_SIZE = 8


@dataclass(frozen=True)
class ScaleGraph:
    """Undirected weighted graph used by scale coarsening.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in this graph level.
    edge_index : torch.Tensor
        Unique undirected edge tensor with shape ``[2, E]`` on CPU.
    edge_weight : torch.Tensor
        Positive edge weights with shape ``[E]`` on CPU.
    adjacency : list[list[tuple[int, float]]]
        Deterministically ordered undirected adjacency list.
    """

    num_nodes: int
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    adjacency: list[list[tuple[int, float]]] = field(default_factory=list)


@dataclass(frozen=True)
class ScaleCoarsenLevel:
    """One fine-to-coarse transition in a scale hierarchy.

    Parameters
    ----------
    fine_num_nodes : int
        Number of nodes in the finer level.
    coarse_num_nodes : int
        Number of nodes in the coarser level.
    fine_to_coarse : torch.Tensor
        Parent map with shape ``[fine_num_nodes]``.
    edge_index : torch.Tensor
        Coarse edge tensor with shape ``[2, E_coarse]``.
    edge_weight : torch.Tensor
        Coarse edge weights with shape ``[E_coarse]``.
    node_sizes : torch.Tensor
        Coarse node sizes with shape ``[coarse_num_nodes, 2]``.
    node_masses : torch.Tensor
        Coarse node masses with shape ``[coarse_num_nodes]``.
    """

    fine_num_nodes: int
    coarse_num_nodes: int
    fine_to_coarse: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    node_sizes: torch.Tensor
    node_masses: torch.Tensor


@dataclass(frozen=True)
class ScaleHierarchy:
    """Finest-to-coarsest scale hierarchy.

    Parameters
    ----------
    finest_graph : ScaleGraph
        Normalized finest graph.
    levels : list[ScaleCoarsenLevel]
        Coarsening transitions ordered from fine to coarse.
    coarsest_graph : ScaleGraph
        Final graph level after all transitions.
    coarsest_node_sizes : torch.Tensor
        Node sizes for the coarsest graph with shape ``[N_coarse, 2]``.
    coarsest_node_masses : torch.Tensor
        Node masses for the coarsest graph with shape ``[N_coarse]``.
    """

    finest_graph: ScaleGraph
    levels: list[ScaleCoarsenLevel]
    coarsest_graph: ScaleGraph
    coarsest_node_sizes: torch.Tensor
    coarsest_node_masses: torch.Tensor


def normalize_scale_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> ScaleGraph:
    """Return a deterministic unique undirected weighted graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    ScaleGraph
        CPU graph with duplicate directed/undirected edges summed.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_weights must have shape [E].")

    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long).contiguous()
    if edge_index_cpu.numel() > 0:
        if int(edge_index_cpu.min().item()) < 0:
            raise ValueError("edge_index cannot contain negative node indices.")
        if int(edge_index_cpu.max().item()) >= int(num_nodes):
            raise ValueError("edge_index references a node outside [0, num_nodes).")
    weights_cpu = (
        torch.ones((edge_index_cpu.shape[1],), dtype=torch.float32)
        if edge_weights is None
        else edge_weights.detach().to(device="cpu", dtype=torch.float32).contiguous()
    )
    return _build_graph_from_edges(edge_index_cpu, int(num_nodes), weights_cpu, topk_per_node=0)


def build_scale_hierarchy(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    edge_weights: Optional[torch.Tensor] = None,
    *,
    target_nodes: int = 2_000,
    max_levels: int = 24,
    seed: int = 42,
    topk_per_node: int = 16,
    min_shrink_ratio: float = _DEFAULT_MIN_SHRINK,
) -> ScaleHierarchy:
    """Build a target-guaranteed family-agnostic coarsening hierarchy.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of finest nodes.
    node_sizes : torch.Tensor or None
        Finest node sizes with shape ``[N, 2]``. Missing sizes use unit boxes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    target_nodes : int, default=2000
        Desired coarsest node cap.
    max_levels : int, default=24
        Maximum fine-to-coarse transitions.
    seed : int, default=42
        Deterministic CPU torch RNG seed used for heavy-edge matching order.
    topk_per_node : int, default=16
        Maximum weighted neighbors retained per coarse node after aggregation.
    min_shrink_ratio : float, default=0.50
        Heavy-edge levels below this shrink escalate to deterministic hub-star
        contraction.

    Returns
    -------
    ScaleHierarchy
        Finest-to-coarsest hierarchy and coarsest payloads.
    """
    if target_nodes < 1:
        raise ValueError("target_nodes must be positive.")
    if max_levels < 0:
        raise ValueError("max_levels must be nonnegative.")
    if min_shrink_ratio < 0.0 or min_shrink_ratio >= 1.0:
        raise ValueError("min_shrink_ratio must be in [0, 1).")

    graph = normalize_scale_graph(edge_index, int(num_nodes), edge_weights)
    sizes = _resolved_node_sizes(node_sizes, int(num_nodes))
    masses = torch.ones((int(num_nodes),), dtype=torch.float32)
    levels: list[ScaleCoarsenLevel] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    while graph.num_nodes > int(target_nodes) and len(levels) < int(max_levels):
        coarsened = _heavy_edge_matching(graph, generator)
        use_star = coarsened is None
        if coarsened is not None:
            fine_to_coarse, coarse_num_nodes = coarsened
            shrink_ratio = 1.0 - float(coarse_num_nodes) / float(graph.num_nodes)
            use_star = shrink_ratio < float(min_shrink_ratio)

        if use_star:
            fine_to_coarse, coarse_num_nodes = _star_contraction_mapping(graph)
            target_floor = max(_MIN_COARSE_SIZE, int(target_nodes) // 2)
            if coarse_num_nodes >= graph.num_nodes or coarse_num_nodes < target_floor:
                fine_to_coarse, coarse_num_nodes = _bucket_contraction_mapping(
                    graph.num_nodes,
                    int(target_nodes),
                )

        if coarse_num_nodes >= graph.num_nodes:
            break

        coarse_graph = _build_graph_from_mapping(
            graph,
            fine_to_coarse,
            coarse_num_nodes,
            topk_per_node=max(0, int(topk_per_node)),
        )
        coarse_sizes = aggregate_node_sizes(sizes, fine_to_coarse, coarse_num_nodes)
        coarse_masses = aggregate_node_masses(masses, fine_to_coarse, coarse_num_nodes)
        levels.append(
            ScaleCoarsenLevel(
                fine_num_nodes=graph.num_nodes,
                coarse_num_nodes=coarse_num_nodes,
                fine_to_coarse=fine_to_coarse.clone(),
                edge_index=coarse_graph.edge_index.clone(),
                edge_weight=coarse_graph.edge_weight.clone(),
                node_sizes=coarse_sizes.clone(),
                node_masses=coarse_masses.clone(),
            )
        )
        graph = coarse_graph
        sizes = coarse_sizes
        masses = coarse_masses

    return ScaleHierarchy(
        finest_graph=normalize_scale_graph(edge_index, int(num_nodes), edge_weights),
        levels=levels,
        coarsest_graph=graph,
        coarsest_node_sizes=sizes,
        coarsest_node_masses=masses,
    )


def aggregate_node_sizes(
    fine_node_sizes: torch.Tensor,
    fine_to_coarse: torch.Tensor,
    num_coarse_nodes: int,
) -> torch.Tensor:
    """Aggregate fine node boxes into coarse boxes by max extent.

    Parameters
    ----------
    fine_node_sizes : torch.Tensor
        Fine sizes with shape ``[N_fine, 2]``.
    fine_to_coarse : torch.Tensor
        Parent map with shape ``[N_fine]``.
    num_coarse_nodes : int
        Number of coarse nodes.

    Returns
    -------
    torch.Tensor
        Coarse sizes with shape ``[N_coarse, 2]``.
    """
    coarse = torch.zeros((int(num_coarse_nodes), 2), dtype=torch.float32)
    if fine_node_sizes.numel() == 0:
        return coarse
    index = fine_to_coarse.to(dtype=torch.long).unsqueeze(1).expand(-1, 2)
    coarse.scatter_reduce_(0, index, fine_node_sizes.to(dtype=torch.float32), reduce="amax")
    return coarse


def aggregate_node_masses(
    fine_node_masses: torch.Tensor,
    fine_to_coarse: torch.Tensor,
    num_coarse_nodes: int,
) -> torch.Tensor:
    """Aggregate fine node masses by parent.

    Parameters
    ----------
    fine_node_masses : torch.Tensor
        Fine masses with shape ``[N_fine]``.
    fine_to_coarse : torch.Tensor
        Parent map with shape ``[N_fine]``.
    num_coarse_nodes : int
        Number of coarse nodes.

    Returns
    -------
    torch.Tensor
        Coarse masses with shape ``[N_coarse]``.
    """
    coarse = torch.zeros((int(num_coarse_nodes),), dtype=torch.float32)
    if fine_node_masses.numel() == 0:
        return coarse
    coarse.scatter_add_(
        0,
        fine_to_coarse.to(dtype=torch.long),
        fine_node_masses.to(dtype=torch.float32),
    )
    return coarse


def prolong_positions(
    coarse_pos: torch.Tensor,
    fine_to_coarse: torch.Tensor,
    *,
    seed: int = 42,
    jitter_scale: float = 1.0,
) -> torch.Tensor:
    """Interpolate fine positions from coarse parents with deterministic offsets.

    Parameters
    ----------
    coarse_pos : torch.Tensor
        Coarse positions with shape ``[N_coarse, 2]``.
    fine_to_coarse : torch.Tensor
        Parent map with shape ``[N_fine]``.
    seed : int, default=42
        Seed folded into the deterministic angular offset.
    jitter_scale : float, default=1.0
        Offset radius in layout units.

    Returns
    -------
    torch.Tensor
        Fine positions with shape ``[N_fine, 2]``.
    """
    parent = fine_to_coarse.to(device=coarse_pos.device, dtype=torch.long)
    fine_pos = coarse_pos[parent].detach().clone()
    if fine_pos.shape[0] <= coarse_pos.shape[0] or float(jitter_scale) <= 0.0:
        return fine_pos.to(dtype=torch.float32)

    ordinals = _sibling_ordinals(fine_to_coarse).to(device=coarse_pos.device, dtype=torch.float32)
    parent_f = parent.to(dtype=torch.float32)
    phase = (ordinals * 12.9898 + parent_f * 78.233 + float(seed) * 0.0174533).remainder(
        6.283185307179586
    )
    radius = float(jitter_scale) * torch.sqrt(ordinals + 1.0) / torch.sqrt(ordinals + 2.0)
    offset = torch.stack((torch.cos(phase), torch.sin(phase)), dim=1) * radius.unsqueeze(1)
    return (fine_pos + offset).to(dtype=torch.float32)


def _resolved_node_sizes(node_sizes: Optional[torch.Tensor], num_nodes: int) -> torch.Tensor:
    """Return CPU float node sizes with shape ``[N, 2]``.

    Parameters
    ----------
    node_sizes : torch.Tensor or None
        Optional input sizes.
    num_nodes : int
        Expected node count.

    Returns
    -------
    torch.Tensor
        CPU float sizes with shape ``[N, 2]``.
    """
    if node_sizes is None:
        return torch.ones((int(num_nodes), 2), dtype=torch.float32)
    sizes = node_sizes.detach().to(device="cpu", dtype=torch.float32)
    if sizes.ndim == 1:
        sizes = sizes.unsqueeze(1).expand(-1, 2).contiguous()
    if tuple(sizes.shape) != (int(num_nodes), 2):
        raise ValueError("node_sizes must have shape [N, 2].")
    return sizes.contiguous()


def _build_graph_from_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor,
    *,
    topk_per_node: int,
) -> ScaleGraph:
    """Aggregate raw edges into a deterministic graph object.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weight : torch.Tensor
        CPU weights with shape ``[E]``.
    topk_per_node : int
        Optional coarse densification cap; ``0`` disables pruning.

    Returns
    -------
    ScaleGraph
        Unique undirected weighted graph.
    """
    edge_totals: dict[tuple[int, int], float] = {}
    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()
    weights = edge_weight.tolist()
    for source, target, weight in zip(sources, targets, weights):
        if source == target:
            continue
        lo = int(min(source, target))
        hi = int(max(source, target))
        edge_totals[(lo, hi)] = edge_totals.get((lo, hi), 0.0) + float(weight)
    if topk_per_node > 0:
        edge_totals = _prune_topk_edges(edge_totals, int(num_nodes), int(topk_per_node))
    return _graph_from_edge_totals(edge_totals, int(num_nodes))


def _build_graph_from_mapping(
    graph: ScaleGraph,
    fine_to_coarse: torch.Tensor,
    coarse_num_nodes: int,
    *,
    topk_per_node: int,
) -> ScaleGraph:
    """Aggregate a coarse graph from a fine-to-coarse parent map.

    Parameters
    ----------
    graph : ScaleGraph
        Fine graph.
    fine_to_coarse : torch.Tensor
        Parent map with shape ``[N_fine]``.
    coarse_num_nodes : int
        Number of coarse nodes.
    topk_per_node : int
        Maximum retained weighted neighbors per coarse node.

    Returns
    -------
    ScaleGraph
        Aggregated coarse graph.
    """
    if graph.edge_index.numel() == 0:
        return _graph_from_edge_totals({}, int(coarse_num_nodes))
    parent = fine_to_coarse.to(dtype=torch.long)
    coarse_src = parent[graph.edge_index[0]]
    coarse_tgt = parent[graph.edge_index[1]]
    cross = coarse_src != coarse_tgt
    if not bool(cross.any()):
        return _graph_from_edge_totals({}, int(coarse_num_nodes))
    lo = torch.minimum(coarse_src[cross], coarse_tgt[cross]).tolist()
    hi = torch.maximum(coarse_src[cross], coarse_tgt[cross]).tolist()
    weights = graph.edge_weight[cross].tolist()
    totals: dict[tuple[int, int], float] = {}
    for source, target, weight in zip(lo, hi, weights):
        key = (int(source), int(target))
        totals[key] = totals.get(key, 0.0) + float(weight)
    if topk_per_node > 0:
        totals = _prune_topk_edges(totals, int(coarse_num_nodes), int(topk_per_node))
    return _graph_from_edge_totals(totals, int(coarse_num_nodes))


def _graph_from_edge_totals(
    edge_totals: dict[tuple[int, int], float],
    num_nodes: int,
) -> ScaleGraph:
    """Build tensor and adjacency payloads from weighted edge totals.

    Parameters
    ----------
    edge_totals : dict[tuple[int, int], float]
        Undirected edge totals keyed by sorted endpoint pair.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    ScaleGraph
        Graph payload with deterministic edge and adjacency ordering.
    """
    ordered = sorted(edge_totals)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(int(num_nodes))]
    if not ordered:
        return ScaleGraph(
            num_nodes=int(num_nodes),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_weight=torch.empty((0,), dtype=torch.float32),
            adjacency=adjacency,
        )
    weights = [float(edge_totals[pair]) for pair in ordered]
    for (source, target), weight in zip(ordered, weights):
        adjacency[source].append((target, weight))
        adjacency[target].append((source, weight))
    for neighbors in adjacency:
        neighbors.sort(key=lambda item: (-item[1], item[0]))
    return ScaleGraph(
        num_nodes=int(num_nodes),
        edge_index=torch.tensor(ordered, dtype=torch.long).transpose(0, 1).contiguous(),
        edge_weight=torch.tensor(weights, dtype=torch.float32),
        adjacency=adjacency,
    )


def _prune_topk_edges(
    edge_totals: dict[tuple[int, int], float],
    num_nodes: int,
    topk_per_node: int,
) -> dict[tuple[int, int], float]:
    """Keep edges selected by either endpoint's top-k weighted neighborhood.

    Parameters
    ----------
    edge_totals : dict[tuple[int, int], float]
        Candidate coarse edges.
    num_nodes : int
        Number of coarse nodes.
    topk_per_node : int
        Weighted-neighbor cap per endpoint.

    Returns
    -------
    dict[tuple[int, int], float]
        Pruned edge totals preserving deterministic endpoint coverage.
    """
    adjacency: list[list[tuple[int, int, float]]] = [[] for _ in range(int(num_nodes))]
    for (source, target), weight in edge_totals.items():
        adjacency[source].append((source, target, float(weight)))
        adjacency[target].append((source, target, float(weight)))
    keep: set[tuple[int, int]] = set()
    for neighbors in adjacency:
        neighbors.sort(key=lambda item: (-item[2], item[0], item[1]))
        for source, target, _weight in neighbors[: int(topk_per_node)]:
            keep.add((source, target))
    return {pair: weight for pair, weight in edge_totals.items() if pair in keep}


def _heavy_edge_matching(
    graph: ScaleGraph,
    generator: torch.Generator,
) -> Optional[tuple[torch.Tensor, int]]:
    """Coarsen one level using seeded random-order heavy-edge matching.

    Parameters
    ----------
    graph : ScaleGraph
        Current graph.
    generator : torch.Generator
        CPU generator used for deterministic visit order.

    Returns
    -------
    tuple[torch.Tensor, int] or None
        Fine-to-coarse mapping and coarse node count, or ``None`` on a weak
        reduction.
    """
    num_nodes = int(graph.num_nodes)
    if num_nodes < _MIN_COARSE_SIZE:
        return None
    order = torch.randperm(num_nodes, generator=generator).tolist()
    matched = bytearray(num_nodes)
    mapping = [-1] * num_nodes
    coarse_node = 0
    for node in order:
        if matched[node]:
            continue
        matched[node] = 1
        partner = -1
        partner_weight = -1.0
        for neighbor, weight in graph.adjacency[node]:
            if matched[neighbor]:
                continue
            if float(weight) > partner_weight:
                partner = int(neighbor)
                partner_weight = float(weight)
        mapping[node] = coarse_node
        if partner >= 0:
            matched[partner] = 1
            mapping[partner] = coarse_node
        coarse_node += 1
    if coarse_node == num_nodes or coarse_node < _MIN_COARSE_SIZE:
        return None
    if coarse_node > int(_MIN_HEM_REDUCTION * float(num_nodes)):
        return None
    return torch.tensor(mapping, dtype=torch.long), int(coarse_node)


def _star_contraction_mapping(graph: ScaleGraph) -> tuple[torch.Tensor, int]:
    """Build a deterministic hub-star contraction mapping.

    Parameters
    ----------
    graph : ScaleGraph
        Current graph.

    Returns
    -------
    tuple[torch.Tensor, int]
        Fine-to-coarse mapping and coarse node count.
    """
    num_nodes = int(graph.num_nodes)
    degrees = [len(neighbors) for neighbors in graph.adjacency]
    order = sorted(range(num_nodes), key=lambda node: (-degrees[node], node))
    mapping = [-1] * num_nodes
    coarse_node = 0
    for node in order:
        if mapping[node] >= 0:
            continue
        mapping[node] = coarse_node
        for neighbor, _weight in graph.adjacency[node]:
            if mapping[neighbor] < 0:
                mapping[neighbor] = coarse_node
        coarse_node += 1
    if graph.edge_index.numel() == 0 and num_nodes > _DEFAULT_BUCKET_SIZE:
        target = max(1, math.ceil(num_nodes / _DEFAULT_BUCKET_SIZE))
        return _bucket_contraction_mapping(num_nodes, target)
    return torch.tensor(mapping, dtype=torch.long), coarse_node


def _bucket_contraction_mapping(num_nodes: int, target: int) -> tuple[torch.Tensor, int]:
    """Build a deterministic contiguous bucket contraction mapping.

    Parameters
    ----------
    num_nodes : int
        Current node count.
    target : int
        Desired maximum coarse node count.

    Returns
    -------
    tuple[torch.Tensor, int]
        Fine-to-coarse mapping and coarse node count.
    """
    if int(num_nodes) <= int(target):
        return torch.arange(int(num_nodes), dtype=torch.long), int(num_nodes)
    bucket = int(math.ceil(float(num_nodes) / float(max(1, int(target)))))
    mapping = torch.div(
        torch.arange(int(num_nodes), dtype=torch.long),
        bucket,
        rounding_mode="floor",
    )
    coarse = int(mapping.max().item()) + 1 if mapping.numel() else 0
    return mapping, coarse


def _sibling_ordinals(fine_to_coarse: torch.Tensor) -> torch.Tensor:
    """Return each fine node's deterministic ordinal within its parent group.

    Parameters
    ----------
    fine_to_coarse : torch.Tensor
        Parent map with shape ``[N_fine]``.

    Returns
    -------
    torch.Tensor
        Zero-based sibling ordinals with shape ``[N_fine]``.
    """
    parent = fine_to_coarse.to(device="cpu", dtype=torch.long)
    counts: dict[int, int] = {}
    ordinals = torch.empty_like(parent)
    for index, raw_parent in enumerate(parent.tolist()):
        value = counts.get(int(raw_parent), 0)
        ordinals[index] = value
        counts[int(raw_parent)] = value + 1
    return ordinals


__all__ = [
    "ScaleCoarsenLevel",
    "ScaleGraph",
    "ScaleHierarchy",
    "aggregate_node_masses",
    "aggregate_node_sizes",
    "build_scale_hierarchy",
    "normalize_scale_graph",
    "prolong_positions",
]

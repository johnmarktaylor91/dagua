"""Community-scaffold native route (native-sprint r2 wave 2).

Two-level layout for graphs with meaningful community structure (SBM /
social / clustered families where generic force layouts blur block
boundaries):

1. **Coarsen**: deterministic label propagation
   (:func:`dagua.layout.graph_classify.label_propagation_communities`)
   partitions nodes into communities. No RNG, no external dependencies.
2. **Metagraph placement**: the community quotient graph (one super-node
   per community, edges weighted by inter-community coupling) is laid out
   with the geodesic-stress core, giving strongly coupled communities
   adjacent slots.
3. **Per-community layout**: each community's induced subgraph is laid out
   independently with geodesic stress in point units.
4. **Compose + separate**: community layouts are translated onto their
   metagraph slots, then the slot frame is uniformly scaled until every
   pair of community disks is separated (content radius + margin), so no
   two communities interleave.

The route is entered into the honest undirected portfolio contest as one
more candidate; the measured-argmax referee decides whether structural
community placement beats the flat incumbent on each graph. No graph names
or corpus-specific constants appear anywhere in this module.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

_LOGGER = logging.getLogger(__name__)

# Communities coarser than this fraction of all nodes mean label propagation
# found no exploitable mesoscale structure (nearly one community per node).
MAX_COMMUNITY_FRACTION = 0.75
# Separation margin applied between community content disks, in units of the
# per-graph spacing target.
COMMUNITY_SEPARATION_MARGIN = 1.5


def _induced_subgraph(
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
    member_mask: torch.Tensor,
    local_index: torch.Tensor,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return the relabeled induced subgraph for one community.

    Parameters
    ----------
    edge_index : torch.Tensor
        Parent edge tensor shaped ``[2, E]``.
    edge_weights : torch.Tensor, optional
        Parent edge weights shaped ``[E]``.
    member_mask : torch.Tensor
        Boolean membership mask shaped ``[N]``.
    local_index : torch.Tensor
        Parent-to-local node map shaped ``[N]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None]
        Local edge tensor and matching weights.
    """
    if edge_index.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long), None
    keep = member_mask[edge_index[0]] & member_mask[edge_index[1]]
    local_edges = local_index[edge_index[:, keep]]
    local_weights = edge_weights[keep] if edge_weights is not None else None
    return local_edges, local_weights


def layout_native_community_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    config: Optional[Any] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    community_labels: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the two-level community-scaffold layout.

    Falls back to the flat geodesic-stress core when label propagation finds
    no exploitable community structure, so the pipeline always returns a
    finite layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]`` (direction ignored).
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node bounding boxes shaped ``[N, 2]``.
    config : Any, optional
        Optional layout configuration carrying ``node_sep``.
    seed : int, default=42
        Deterministic seed.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights shaped ``[E]``.
    community_labels : torch.Tensor, optional
        Precomputed community ids shaped ``[N]`` (dense ``0..K-1``). When
        omitted, deterministic label propagation runs here.
    **kwargs : Any
        Compatibility keywords accepted by generic dispatchers.

    Returns
    -------
    torch.Tensor
        Finite positions shaped ``[N, 2]`` in point units.
    """
    del kwargs
    from dagua.layout.graph_classify import label_propagation_communities
    from dagua.layout.ops.pipelines.native_lattice_grid import (
        _deterministic_fallback_positions,
        _target_edge_length,
        geodesic_dense_work_is_allowed,
        layout_geodesic_stress_pipeline,
    )

    resolved_sep = float(getattr(config, "node_sep", 36.0) or 36.0)
    if num_nodes <= 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    edge_count = int(edge_index.shape[1]) if edge_index.numel() else 0

    def _flat_fallback() -> torch.Tensor:
        """Return the flat fallback without violating dense-work guards.

        Returns
        -------
        torch.Tensor
            Finite fallback positions with shape ``[N, 2]``.
        """
        if not geodesic_dense_work_is_allowed(num_nodes, edge_count):
            spacing = _target_edge_length(node_sizes, resolved_sep)
            return _deterministic_fallback_positions(num_nodes, spacing, seed)
        return layout_geodesic_stress_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            config=config,
            seed=seed,
            edge_weights=edge_weights,
            node_sep=resolved_sep,
        )

    labels = (
        community_labels.detach().to(device="cpu", dtype=torch.long)
        if community_labels is not None
        else label_propagation_communities(edge_index, num_nodes)
    )
    num_communities = int(labels.max().item()) + 1 if labels.numel() else 0
    if num_communities < 2 or num_communities > MAX_COMMUNITY_FRACTION * num_nodes:
        return _flat_fallback()

    spacing = _target_edge_length(node_sizes, resolved_sep)
    cpu_edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    cpu_weights = (
        edge_weights.detach().to(device="cpu", dtype=torch.float32)
        if edge_weights is not None
        else None
    )
    cpu_sizes = (
        node_sizes.detach().to(device="cpu", dtype=torch.float32)
        if node_sizes is not None
        else None
    )

    # Per-community internal layouts (independent geodesic-stress solves).
    positions = torch.zeros((num_nodes, 2), dtype=torch.float32)
    content_radius = torch.zeros(num_communities, dtype=torch.float32)
    local_index = torch.zeros(num_nodes, dtype=torch.long)
    for community in range(num_communities):
        member_mask = labels == community
        members = torch.nonzero(member_mask, as_tuple=False).reshape(-1)
        local_index[members] = torch.arange(int(members.numel()), dtype=torch.long)
        local_edges, local_weights = _induced_subgraph(
            cpu_edges, cpu_weights, member_mask, local_index
        )
        local_sizes = cpu_sizes[members] if cpu_sizes is not None else None
        local_count = int(members.numel())
        local_edge_count = int(local_edges.shape[1]) if local_edges.numel() else 0
        if not geodesic_dense_work_is_allowed(local_count, local_edge_count):
            internal = _deterministic_fallback_positions(local_count, spacing, seed + community)
        else:
            internal = layout_geodesic_stress_pipeline(
                edge_index=local_edges,
                num_nodes=local_count,
                node_sizes=local_sizes,
                config=config,
                seed=seed + community,
                edge_weights=local_weights,
                node_sep=resolved_sep,
            )
        internal = internal - internal.mean(dim=0, keepdim=True)
        positions[members] = internal
        radii = torch.linalg.vector_norm(internal, dim=1)
        half_diagonal = (
            float(torch.linalg.vector_norm(local_sizes, dim=1).max().item()) * 0.5
            if local_sizes is not None and local_sizes.numel() > 0
            else 0.0
        )
        content_radius[community] = float(radii.max().item()) + half_diagonal

    # Metagraph: aggregate inter-community coupling into edge distance costs
    # (heavier coupling -> shorter target distance -> adjacent slots).
    meta_positions = _metagraph_positions(cpu_edges, labels, num_communities, spacing, seed)

    # Uniform slot-frame scale until every community pair is separated.
    meta_positions = _separate_communities(meta_positions, content_radius, spacing)

    composed = positions + meta_positions[labels]
    if not bool(torch.isfinite(composed).all().item()):
        _LOGGER.warning("community scaffold produced non-finite output; flat fallback")
        return _flat_fallback()
    return composed - composed.mean(dim=0, keepdim=True)


def _metagraph_positions(
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    num_communities: int,
    spacing: float,
    seed: int,
) -> torch.Tensor:
    """Return geodesic-stress positions for the community quotient graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        Parent edge tensor shaped ``[2, E]`` (CPU).
    labels : torch.Tensor
        Community ids shaped ``[N]``.
    num_communities : int
        Number of communities.
    spacing : float
        Point-unit spacing target.
    seed : int
        Deterministic seed.

    Returns
    -------
    torch.Tensor
        Metagraph slot centers shaped ``[K, 2]``.
    """
    from dagua.layout.ops.pipelines.native_lattice_grid import (
        _deterministic_fallback_positions,
        geodesic_dense_work_is_allowed,
        layout_geodesic_stress_pipeline,
    )

    if edge_index.numel() == 0:
        return _deterministic_fallback_positions(num_communities, spacing, seed)
    meta_src = labels[edge_index[0]]
    meta_dst = labels[edge_index[1]]
    cross = meta_src != meta_dst
    if not bool(cross.any().item()):
        return _deterministic_fallback_positions(num_communities, spacing, seed)
    lo = torch.minimum(meta_src[cross], meta_dst[cross])
    hi = torch.maximum(meta_src[cross], meta_dst[cross])
    keys = lo * num_communities + hi
    unique_keys, counts = torch.unique(keys, return_counts=True)
    meta_edges = torch.stack(
        [
            torch.div(unique_keys, num_communities, rounding_mode="floor"),
            unique_keys % num_communities,
        ]
    )
    # Heavier coupling means "closer": distance cost 1/sqrt(count) keeps the
    # quotient metric finite and monotone in coupling strength.
    meta_weights = counts.to(dtype=torch.float32).clamp_min(1.0).rsqrt()
    if not geodesic_dense_work_is_allowed(num_communities, int(meta_edges.shape[1])):
        return _deterministic_fallback_positions(num_communities, spacing, seed)
    return layout_geodesic_stress_pipeline(
        edge_index=meta_edges,
        num_nodes=num_communities,
        node_sizes=None,
        seed=seed,
        edge_weights=meta_weights,
        node_sep=spacing,
    )


def _separate_communities(
    meta_positions: torch.Tensor,
    content_radius: torch.Tensor,
    spacing: float,
) -> torch.Tensor:
    """Uniformly scale metagraph slots until community disks are disjoint.

    Parameters
    ----------
    meta_positions : torch.Tensor
        Slot centers shaped ``[K, 2]``.
    content_radius : torch.Tensor
        Community content radii shaped ``[K]``.
    spacing : float
        Point-unit spacing target (margin unit).

    Returns
    -------
    torch.Tensor
        Scaled slot centers shaped ``[K, 2]``.
    """
    num_communities = int(meta_positions.shape[0])
    if num_communities < 2:
        return meta_positions
    centered = meta_positions - meta_positions.mean(dim=0, keepdim=True)
    deltas = centered.unsqueeze(1) - centered.unsqueeze(0)
    pair_distance = torch.linalg.vector_norm(deltas, dim=2)
    required = (
        content_radius.unsqueeze(1)
        + content_radius.unsqueeze(0)
        + COMMUNITY_SEPARATION_MARGIN * spacing
    )
    off_diagonal = ~torch.eye(num_communities, dtype=torch.bool)
    # Degenerate coincident slots cannot be fixed by scaling; spread them on
    # a deterministic circle sized to the largest content disk first.
    min_distance = float(pair_distance[off_diagonal].min().item())
    if min_distance <= 1.0e-6:
        from dagua.layout.ops.pipelines.native_lattice_grid import (
            _deterministic_fallback_positions,
        )

        circle = _deterministic_fallback_positions(
            num_communities, float(required.max().item()), seed=0
        )
        centered = circle - circle.mean(dim=0, keepdim=True)
        deltas = centered.unsqueeze(1) - centered.unsqueeze(0)
        pair_distance = torch.linalg.vector_norm(deltas, dim=2)
    ratios = required[off_diagonal] / pair_distance[off_diagonal].clamp_min(1.0e-9)
    scale = max(float(ratios.max().item()), 1.0)
    return centered * scale


__all__ = [
    "COMMUNITY_SEPARATION_MARGIN",
    "MAX_COMMUNITY_FRACTION",
    "layout_native_community_pipeline",
]

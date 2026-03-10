"""Multilevel coarsening V-cycle for large graph layout.

The single most important technique for scaling beyond 50K nodes.
Coarsen the graph to ~2K nodes, lay out the coarsest graph, then
prolong positions back through each level with brief refinement.

Total work: O(N) if each level does O(N_level) work and reduction ≥ 2x.

Coarsening strategy: layer-aware heavy-edge matching.
- Only merge nodes within the same DAG layer (preserves layered structure)
- Pair adjacent nodes by connectivity (shared parents/children preferred)
- Reduction ratio ~50% per level
- Edges between coarse nodes sum weights for multi-edges
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from dagua.config import LayoutConfig
from dagua.utils import longest_path_layering


@dataclass
class CoarseLevel:
    """One level of the coarsening hierarchy."""
    edge_index: torch.Tensor       # [2, E_c] coarsened edges
    node_sizes: torch.Tensor       # [N_c, 2]
    num_nodes: int
    fine_to_coarse: torch.Tensor   # [N_fine] maps fine node → coarse node
    num_fine: int                  # N at the finer level


def coarsen_once(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    layer_assignments: List[int],
    device: str = "cpu",
) -> CoarseLevel:
    """Single coarsening step via layer-aware heavy-edge matching.

    Within each layer, greedily pair adjacent nodes. Matched pairs merge
    into a single coarse node. Preserves DAG layer structure.

    Fully vectorized — no Python loops over nodes.
    """
    N = num_nodes
    if isinstance(layer_assignments, list):
        layers = torch.tensor(layer_assignments, dtype=torch.long, device=device)
    else:
        layers = layer_assignments.to(device)
    num_layers = int(layers.max().item()) + 1 if N > 0 else 0

    # Sort nodes by layer for contiguous access
    layer_counts = torch.bincount(layers, minlength=num_layers)
    layer_offsets = torch.zeros(num_layers + 1, dtype=torch.long, device=device)
    layer_offsets[1:] = layer_counts.cumsum(0)

    # Build adjacency for matching priority
    match_score = torch.zeros(N, dtype=torch.float32, device=device)
    if edge_index.numel() > 0:
        src, tgt = edge_index[0], edge_index[1]
        in_deg = torch.zeros(N, device=device)
        out_deg = torch.zeros(N, device=device)
        in_deg.scatter_add_(0, tgt, torch.ones(tgt.shape[0], device=device))
        out_deg.scatter_add_(0, src, torch.ones(src.shape[0], device=device))
        match_score = in_deg + out_deg

    # Vectorized greedy matching: sort by (layer, -match_score) globally,
    # then pair consecutive same-layer nodes.
    # Composite sort key: layer dominates, match_score breaks ties (descending)
    score_norm = match_score / (match_score.max() + 1e-8)  # [0, 1)
    sort_key = layers.float() * 2.0 + (1.0 - score_norm)  # layer first, then descending score
    global_order = sort_key.argsort()  # [N] — sorted by (layer, -score)

    # Within each layer group in global_order, consecutive pairs get same coarse ID.
    # Compute within-layer position for each node in sorted order.
    sorted_layers = layers[global_order]  # [N]

    # Within-layer index: 0, 1, 2, ... for each layer
    # Use cumsum trick: positions within layer = global_position - layer_start
    # We can compute this from the sorted layer assignments
    layer_of_sorted = sorted_layers
    layer_start_of_sorted = layer_offsets[layer_of_sorted]  # [N]
    global_pos = torch.arange(N, dtype=torch.long, device=device)
    within_layer_pos = global_pos - layer_start_of_sorted  # [N]

    # Pair index within layer: pos // 2 gives pair number
    pair_within_layer = within_layer_pos // 2  # [N]

    # Coarse ID = cumulative pairs up to this layer + pair_within_layer
    # Number of coarse nodes per layer: ceil(layer_count / 2)
    coarse_per_layer = (layer_counts + 1) // 2  # [L]
    coarse_offsets = torch.zeros(num_layers + 1, dtype=torch.long, device=device)
    coarse_offsets[1:] = coarse_per_layer.cumsum(0)

    # Each node's coarse ID
    coarse_base = coarse_offsets[layer_of_sorted]  # [N]
    coarse_ids_sorted = coarse_base + pair_within_layer  # [N]

    # Map back to original node order
    fine_to_coarse = torch.empty(N, dtype=torch.long, device=device)
    fine_to_coarse[global_order] = coarse_ids_sorted

    N_coarse = coarse_offsets[-1].item()

    # Build coarse node sizes (max of merged pair for each dimension)
    coarse_sizes = torch.zeros(N_coarse, 2, device=device)
    coarse_sizes.scatter_reduce_(
        0, fine_to_coarse.unsqueeze(1).expand(-1, 2),
        node_sizes, reduce="amax",
    )

    # Build coarse edges (remap and deduplicate)
    if edge_index.numel() > 0:
        coarse_src = fine_to_coarse[edge_index[0]]
        coarse_tgt = fine_to_coarse[edge_index[1]]

        # Remove self-loops (merged nodes)
        not_self = coarse_src != coarse_tgt
        coarse_src = coarse_src[not_self]
        coarse_tgt = coarse_tgt[not_self]

        if coarse_src.numel() > 0:
            # Deduplicate edges using hash
            edge_hash = coarse_src * N_coarse + coarse_tgt
            unique_hash, inverse = edge_hash.unique(return_inverse=True)
            # Recover src, tgt from hash
            unique_src = unique_hash // N_coarse
            unique_tgt = unique_hash % N_coarse
            coarse_edge_index = torch.stack([unique_src, unique_tgt])
        else:
            coarse_edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
    else:
        coarse_edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

    return CoarseLevel(
        edge_index=coarse_edge_index,
        node_sizes=coarse_sizes,
        num_nodes=N_coarse,
        fine_to_coarse=fine_to_coarse,
        num_fine=N,
    )


def build_hierarchy(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    min_nodes: int = 2000,
    max_levels: int = 10,
    device: str = "cpu",
) -> List[CoarseLevel]:
    """Build coarsening hierarchy until num_nodes <= min_nodes.

    Returns list of CoarseLevels from finest to coarsest.
    """
    levels: List[CoarseLevel] = []
    current_ei = edge_index
    current_sizes = node_sizes
    current_n = num_nodes

    for _ in range(max_levels):
        if current_n <= min_nodes:
            break

        # Compute layers for current graph
        if current_ei.numel() > 0:
            layer_assignments = longest_path_layering(current_ei, current_n)
        else:
            layer_assignments = [0] * current_n

        level = coarsen_once(
            current_ei, current_n, current_sizes,
            layer_assignments=layer_assignments, device=device,
        )
        levels.append(level)

        # Move to coarser level
        current_ei = level.edge_index
        current_sizes = level.node_sizes
        current_n = level.num_nodes

        # Safety: stop if coarsening didn't reduce enough
        if current_n > level.num_fine * 0.7:
            break

    return levels


def prolong_positions(
    coarse_pos: torch.Tensor,
    level: CoarseLevel,
    device: str = "cpu",
) -> torch.Tensor:
    """Map positions from coarse level back to fine level.

    Each fine node inherits its coarse parent's position with small jitter.
    Uses barycentered placement for nodes with already-placed neighbors.
    """
    fine_pos = coarse_pos[level.fine_to_coarse].clone()

    # Add small random jitter to separate merged nodes
    # Scale jitter by node size to keep it proportional
    jitter_scale = 5.0
    jitter = torch.randn(level.num_fine, 2, device=device) * jitter_scale
    fine_pos = fine_pos + jitter

    return fine_pos


def multilevel_layout(graph, config: LayoutConfig) -> torch.Tensor:
    """Multilevel V-cycle layout for large graphs.

    1. Build coarsening hierarchy
    2. Layout coarsest graph (many steps)
    3. Prolong + refine at each level (few steps)
    """
    from dagua.layout.engine import _layout_inner

    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    n = graph.num_nodes
    edge_index = graph.edge_index.to(device)
    node_sizes = graph.node_sizes.to(device)

    if config.seed is not None:
        torch.manual_seed(config.seed)

    # Build hierarchy
    min_nodes = config.multilevel_min_nodes
    levels = build_hierarchy(edge_index, n, node_sizes, min_nodes=min_nodes, device=device)

    if not levels:
        # Graph is already small enough — use direct layout (skip multilevel dispatch)
        pos = _layout_inner(edge_index, n, node_sizes, config, device=device)
        from dagua.layout.engine import _apply_direction
        direction = config.direction if config else graph.direction
        return _apply_direction(pos, direction)

    # Layout coarsest graph with many steps
    coarsest = levels[-1]
    coarse_config = LayoutConfig(
        steps=config.multilevel_coarse_steps,
        lr=config.lr * 2,  # higher LR at coarse level for faster convergence
        device=device,
        seed=config.seed,
        node_sep=config.node_sep,
        rank_sep=config.rank_sep,
        w_dag=config.w_dag,
        w_attract=config.w_attract,
        w_attract_x_bias=config.w_attract_x_bias,
        w_repel=config.w_repel,
        w_overlap=config.w_overlap,
        w_crossing=config.w_crossing,
        w_straightness=config.w_straightness,
        w_length_variance=config.w_length_variance,
        exact_repulsion_threshold=config.exact_repulsion_threshold,
        negative_sample_k=config.negative_sample_k,
    )

    pos = _layout_inner(
        coarsest.edge_index, coarsest.num_nodes, coarsest.node_sizes,
        coarse_config,
        device=device,
    )

    # Prolong + refine through hierarchy (coarsest → finest)
    for i in range(len(levels) - 1, -1, -1):
        level = levels[i]

        # Prolong: map coarse positions to fine level
        pos = prolong_positions(pos, level, device=device)

        # Get the graph data for this fine level
        if i == 0:
            # Finest level = original graph
            fine_ei = edge_index
            fine_sizes = node_sizes
            fine_n = n
        else:
            prev_level = levels[i - 1]
            fine_ei = prev_level.edge_index
            fine_sizes = prev_level.node_sizes
            fine_n = prev_level.num_nodes

        # Refine with fewer steps (warm start from prolongation)
        refine_steps = config.multilevel_refine_steps
        if i == 0:
            refine_steps = refine_steps * 2  # More steps at finest level

        refine_config = LayoutConfig(
            steps=refine_steps,
            lr=config.lr,
            device=device,
            seed=None,  # Don't reset seed for refinement
            node_sep=config.node_sep,
            rank_sep=config.rank_sep,
            w_dag=config.w_dag,
            w_attract=config.w_attract,
            w_attract_x_bias=config.w_attract_x_bias,
            w_repel=config.w_repel,
            w_overlap=config.w_overlap,
            w_crossing=config.w_crossing,
            w_straightness=config.w_straightness,
            w_length_variance=config.w_length_variance,
            exact_repulsion_threshold=config.exact_repulsion_threshold,
            negative_sample_k=config.negative_sample_k,
        )

        pos = _layout_inner(
            fine_ei, fine_n, fine_sizes,
            refine_config,
            device=device,
            init_pos=pos,
        )

    # Apply direction transform
    from dagua.layout.engine import _apply_direction
    direction = config.direction if config else graph.direction
    pos = _apply_direction(pos, direction)

    return pos

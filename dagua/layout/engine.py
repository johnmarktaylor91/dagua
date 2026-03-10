"""Core optimization loop — the heart of dagua.

Takes a DaguaGraph and LayoutConfig, returns [N, 2] position tensor.
Headless: operates on tensors extracted from the graph.

Scaling strategy (tiered):
- Tier 0 (N < 500): exact O(N^2) repulsion, full overlap check
- Tier 1 (500-5K): scatter sampling repulsion, layer-local overlap
- Tier 2 (5K-50K): RVS repulsion, reduced passes, adaptive batching
- Tier 3 (N > 50K): multilevel coarsening V-cycle

Cross-cutting:
- Pre-compute LayerIndex once, pass to all layer-aware functions
- Stochastic edge batching for O(batch) instead of O(E) per step
- Adaptive overlap projection frequency
- Early stopping on convergence
"""

from __future__ import annotations

from typing import List, Optional

import torch

from dagua.config import LayoutConfig
from dagua.layout.constraints import (
    cluster_compactness_loss,
    cluster_separation_loss,
    crossing_loss,
    dag_ordering_loss,
    edge_attraction_loss,
    edge_length_variance_loss,
    edge_straightness_loss,
    overlap_avoidance_loss,
    repulsion_loss,
    spacing_consistency_loss,
)
from dagua.layout.init_placement import init_positions
from dagua.layout.layers import LayerIndex, build_layer_index
from dagua.layout.projection import project_overlaps
from dagua.utils import longest_path_layering


def layout(graph, config: Optional[LayoutConfig] = None) -> torch.Tensor:
    """Compute layout positions for all nodes.

    Args:
        graph: DaguaGraph instance
        config: LayoutConfig (uses defaults if None)

    Returns:
        [N, 2] tensor of (x, y) positions
    """
    if config is None:
        config = LayoutConfig()

    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Ensure node sizes are computed
    graph.compute_node_sizes()

    n = graph.num_nodes
    if n == 0:
        return torch.zeros(0, 2, device=device)
    if n == 1:
        return torch.zeros(1, 2, device=device)

    # Move data to device
    edge_index = graph.edge_index.to(device)
    node_sizes = graph.node_sizes.to(device)

    # Set seed for determinism
    if config.seed is not None:
        torch.manual_seed(config.seed)
        if device == "cuda":
            torch.cuda.manual_seed(config.seed)

    # Tier 3: Multilevel coarsening for very large graphs
    if n > config.multilevel_threshold:
        from dagua.layout.multilevel import multilevel_layout
        return multilevel_layout(graph, config)

    # Tier 0-2: Direct layout
    pos = _layout_inner(
        edge_index, n, node_sizes, config,
        device=device,
        clusters=graph.clusters if hasattr(graph, 'clusters') else None,
    )

    # Apply direction transform
    direction = config.direction if config else graph.direction
    pos = _apply_direction(pos, direction)

    return pos


def _layout_inner(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    config: LayoutConfig,
    device: str = "cpu",
    init_pos: Optional[torch.Tensor] = None,
    clusters: Optional[dict] = None,
) -> torch.Tensor:
    """Headless optimization loop — operates on raw tensors.

    This is the core engine, usable by both direct layout and multilevel V-cycle.
    No Graph object dependency.

    Args:
        edge_index: [2, E] edge tensor
        num_nodes: number of nodes
        node_sizes: [N, 2] width/height tensor
        config: LayoutConfig with steps, weights, etc.
        device: target device
        init_pos: optional [N, 2] initial positions (for multilevel warm start)
        clusters: optional cluster dict for cluster losses

    Returns:
        [N, 2] position tensor (detached)
    """
    n = num_nodes
    if n == 0:
        return torch.zeros(0, 2, device=device)
    if n == 1:
        return torch.zeros(1, 2, device=device)

    # Apply adaptive spacing based on graph size
    node_sep = config.node_sep
    rank_sep = config.rank_sep
    if config.adaptive_spacing:
        node_sep, rank_sep = _adaptive_spacing(n, node_sep, rank_sep)

    # Step 1: Initialization
    if init_pos is not None:
        pos = init_pos.to(device)
    else:
        pos = init_positions(
            edge_index, n, node_sizes,
            node_sep=node_sep,
            rank_sep=rank_sep,
            device=device,
        )

    # Pre-compute layer structure (used by repulsion, overlap, projection, crossing)
    layer_assignments: Optional[List[int]] = None
    layer_index: Optional[LayerIndex] = None
    if edge_index.numel() > 0:
        layer_assignments = longest_path_layering(edge_index, n)
        layer_index = build_layer_index(layer_assignments, device=device)

    # Determine adaptive parameters based on graph size
    num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
    edge_batch = _edge_batch_size(num_edges, config)
    overlap_interval = _overlap_interval(n, config)

    # Step 2: Set up optimization
    pos = pos.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([pos], lr=config.lr)

    # Step 3: Optimization loop with annealing
    steps = config.steps
    prev_loss = float("inf")
    stall_count = 0

    for step in range(steps):
        t = step / max(steps - 1, 1)  # 0 → 1

        optimizer.zero_grad()

        # Sample edge batch for this step
        if edge_batch > 0 and num_edges > edge_batch:
            perm = torch.randperm(num_edges, device=device)[:edge_batch]
            batch_edges = edge_index[:, perm]
        else:
            batch_edges = edge_index

        # Annealed weights
        w_dag = config.w_dag * (1 - 0.5 * t)
        w_repel = config.w_repel * (1 + 2 * t)
        w_overlap = config.w_overlap * (1 + t)
        w_crossing = config.w_crossing * t

        loss = torch.tensor(0.0, device=device)

        # DAG ordering
        if w_dag > 0:
            loss = loss + w_dag * dag_ordering_loss(pos, batch_edges, node_sizes, rank_sep)

        # Edge attraction
        if config.w_attract > 0:
            loss = loss + config.w_attract * edge_attraction_loss(
                pos, batch_edges, x_bias=config.w_attract_x_bias
            )

        # Repulsion (tiered: exact → scatter → RVS, size-aware per AMD pattern)
        if w_repel > 0:
            loss = loss + w_repel * repulsion_loss(
                pos, n,
                threshold=config.exact_repulsion_threshold,
                sample_k=config.negative_sample_k,
                layer_index=layer_index,
                node_sizes=node_sizes,
                rvs_threshold=config.rvs_threshold,
                rvs_nn_k=config.rvs_nn_k,
            )

        # Overlap avoidance (layer-local for large graphs)
        if w_overlap > 0:
            loss = loss + w_overlap * overlap_avoidance_loss(
                pos, node_sizes, layer_index=layer_index,
            )

        # Clustering
        if config.w_cluster > 0 and clusters:
            loss = loss + config.w_cluster * cluster_compactness_loss(
                pos, clusters, device=pos.device
            )
            loss = loss + config.w_cluster * 0.5 * cluster_separation_loss(
                pos, node_sizes, clusters, device=pos.device
            )

        # Crossing minimization (ramps up over time)
        if w_crossing > 0:
            alpha = 1.0 + 9.0 * t
            loss = loss + w_crossing * crossing_loss(
                pos, batch_edges, alpha=alpha,
                layer_assignments=layer_assignments,
            )

        # Edge straightness
        if config.w_straightness > 0:
            loss = loss + config.w_straightness * edge_straightness_loss(pos, batch_edges)

        # Edge length variance
        if config.w_length_variance > 0:
            loss = loss + config.w_length_variance * edge_length_variance_loss(pos, batch_edges)

        # Spacing consistency (even horizontal rhythm within layers)
        if config.w_spacing > 0 and layer_index is not None:
            loss = loss + config.w_spacing * spacing_consistency_loss(
                pos, node_sizes, layer_index, target_gap=node_sep,
            )

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([pos], max_norm=100.0)

        optimizer.step()

        # Hard overlap projection (adaptive frequency)
        if step % overlap_interval == 0 or step == steps - 1:
            project_overlaps(
                pos, node_sizes, padding=2.0, iterations=5,
                layer_index=layer_index,
            )

        # Early stopping check
        loss_val = loss.item()
        if step > 10 and abs(prev_loss - loss_val) < prev_loss * 1e-4:
            stall_count += 1
            if stall_count >= 5:
                break
        else:
            stall_count = 0
        prev_loss = loss_val

    # Final aggressive overlap projection
    project_overlaps(
        pos, node_sizes, padding=2.0, iterations=20,
        layer_index=layer_index,
    )

    return pos.detach()


def _edge_batch_size(num_edges: int, config: LayoutConfig) -> int:
    """Determine edge batch size based on graph scale.

    Returns 0 for "use all edges" (no batching).
    """
    if hasattr(config, "edge_batch_size") and config.edge_batch_size > 0:
        return config.edge_batch_size

    # Auto-scale: batch at 10K+ edges
    if num_edges <= 10000:
        return 0  # use all edges
    elif num_edges <= 100000:
        return 50000
    elif num_edges <= 1000000:
        return 100000
    else:
        return 200000


def _overlap_interval(num_nodes: int, config: LayoutConfig) -> int:
    """How often to run overlap projection (every N steps)."""
    if hasattr(config, "overlap_check_interval") and config.overlap_check_interval > 0:
        return config.overlap_check_interval

    if num_nodes <= 5000:
        return 5
    elif num_nodes <= 50000:
        return 10
    else:
        return 20


def _adaptive_spacing(
    num_nodes: int,
    base_node_sep: float = 25.0,
    base_rank_sep: float = 50.0,
) -> tuple:
    """Scale spacing based on graph size for density adaptation.

    Small graphs (<20): more breathing room (1.3x)
    Medium (<200): standard (1.0x)
    Large (<1000): slightly tighter (0.85x)
    Very large (1000+): compact (0.7x)
    """
    if num_nodes < 20:
        scale = 1.3
    elif num_nodes < 200:
        scale = 1.0
    elif num_nodes < 1000:
        scale = 0.85
    else:
        scale = 0.7
    return base_node_sep * scale, base_rank_sep * scale


def _apply_direction(pos: torch.Tensor, direction: str) -> torch.Tensor:
    """Transform positions based on layout direction."""
    if direction == "TB":
        return pos
    elif direction == "BT":
        result = pos.clone()
        result[:, 1] = -result[:, 1]
        return result
    elif direction == "LR":
        result = pos.clone()
        result[:, 0] = pos[:, 1]
        result[:, 1] = pos[:, 0]
        return result
    elif direction == "RL":
        result = pos.clone()
        result[:, 0] = -pos[:, 1]
        result[:, 1] = pos[:, 0]
        return result
    return pos

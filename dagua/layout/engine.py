"""Core optimization loop — the heart of dagua.

Takes a DaguaGraph and LayoutConfig, returns [N, 2] position tensor.
Headless: operates on tensors extracted from the graph.
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
)
from dagua.layout.init_placement import init_positions
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

    # Handle direction: internally always use TB (y increases downward)
    # BT: flip y at the end; LR/RL: swap axes

    # Set seed for determinism
    if config.seed is not None:
        torch.manual_seed(config.seed)
        if device == "cuda":
            torch.cuda.manual_seed(config.seed)

    # Step 1: Algorithmic initialization
    pos = init_positions(
        edge_index, n, node_sizes,
        node_sep=config.node_sep,
        rank_sep=config.rank_sep,
        device=device,
    )

    # Compute layer assignments for crossing loss (adjacent-layer proxy)
    layer_assignments: Optional[List[int]] = None
    if edge_index.numel() > 0:
        layer_assignments = longest_path_layering(edge_index, n)

    # Step 2: Set up optimization
    pos = pos.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([pos], lr=config.lr)

    # Step 3: Optimization loop with annealing
    steps = config.steps
    for step in range(steps):
        t = step / max(steps - 1, 1)  # 0 → 1

        optimizer.zero_grad()

        # Annealed weights
        w_dag = config.w_dag * (1 - 0.5 * t)
        w_repel = config.w_repel * (1 + 2 * t)
        w_overlap = config.w_overlap * (1 + t)
        w_crossing = config.w_crossing * t  # only after structure exists

        loss = torch.tensor(0.0, device=device)

        # DAG ordering
        if w_dag > 0:
            loss = loss + w_dag * dag_ordering_loss(pos, edge_index, node_sizes, config.rank_sep)

        # Edge attraction
        if config.w_attract > 0:
            loss = loss + config.w_attract * edge_attraction_loss(
                pos, edge_index, x_bias=config.w_attract_x_bias
            )

        # Repulsion
        if w_repel > 0:
            loss = loss + w_repel * repulsion_loss(
                pos, n,
                threshold=config.exact_repulsion_threshold,
                sample_k=config.negative_sample_k,
            )

        # Overlap avoidance
        if w_overlap > 0:
            loss = loss + w_overlap * overlap_avoidance_loss(pos, node_sizes)

        # Clustering
        if config.w_cluster > 0 and graph.clusters:
            loss = loss + config.w_cluster * cluster_compactness_loss(
                pos, graph.clusters, device=pos.device
            )
            loss = loss + config.w_cluster * 0.5 * cluster_separation_loss(
                pos, node_sizes, graph.clusters, device=pos.device
            )

        # Crossing minimization (ramps up over time)
        if w_crossing > 0:
            alpha = 1.0 + 9.0 * t  # anneal temperature 1 → 10
            loss = loss + w_crossing * crossing_loss(
                pos, edge_index, alpha=alpha,
                layer_assignments=layer_assignments,
            )

        # Edge straightness
        if config.w_straightness > 0:
            loss = loss + config.w_straightness * edge_straightness_loss(pos, edge_index)

        # Edge length variance
        if config.w_length_variance > 0:
            loss = loss + config.w_length_variance * edge_length_variance_loss(pos, edge_index)

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([pos], max_norm=100.0)

        optimizer.step()

        # Hard overlap projection
        if step % 5 == 0 or step == steps - 1:
            project_overlaps(pos, node_sizes, padding=2.0, iterations=5)

    # Final aggressive overlap projection
    project_overlaps(pos, node_sizes, padding=2.0, iterations=20)

    result = pos.detach()

    # Apply direction transform
    direction = config.direction if config else graph.direction
    result = _apply_direction(result, direction)

    return result


def _apply_direction(pos: torch.Tensor, direction: str) -> torch.Tensor:
    """Transform positions based on layout direction."""
    if direction == "TB":
        return pos  # default: y increases downward
    elif direction == "BT":
        # Flip y so y increases upward
        result = pos.clone()
        result[:, 1] = -result[:, 1]
        return result
    elif direction == "LR":
        # Swap x and y
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

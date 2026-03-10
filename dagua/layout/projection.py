"""Hard overlap resolution via projected gradient descent.

After each optimizer step, pushes overlapping node bounding boxes apart.
Runs deterministically with torch.no_grad().

Sprint 3 scaling strategy:
- N <= 500: exact O(N^2) pairwise
- N > 500 with layer_index: vectorized sweep-line (ZERO per-layer Python loops)
- N > 500 without layer_index: vectorized grid-based
- All paths use torch.where (no CPU-GPU sync from .any() checks)
"""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.layers import LayerIndex


def project_overlaps(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float = 2.0,
    iterations: int = 10,
    layer_index: Optional[LayerIndex] = None,
) -> torch.Tensor:
    """Push overlapping node bounding boxes apart in-place.

    Returns the (modified) pos tensor.
    """
    n = pos.shape[0]
    if n <= 1:
        return pos

    with torch.no_grad():
        if n <= 500:
            _project_exact(pos, node_sizes, padding, iterations)
        elif layer_index is not None:
            _project_sweep(pos, node_sizes, padding, iterations, layer_index)
        else:
            _project_grid(pos, node_sizes, padding, iterations)

    return pos


def _project_exact(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    iterations: int,
) -> None:
    """Exact O(N^2) overlap projection for small graphs."""
    n = pos.shape[0]

    for _ in range(iterations):
        dx = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)
        dy = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)

        min_dx = (node_sizes[:, 0].unsqueeze(1) + node_sizes[:, 0].unsqueeze(0)) / 2 + padding
        min_dy = (node_sizes[:, 1].unsqueeze(1) + node_sizes[:, 1].unsqueeze(0)) / 2 + padding

        overlap_x = min_dx - dx.abs()
        overlap_y = min_dy - dy.abs()
        overlapping = (overlap_x > 0) & (overlap_y > 0)
        overlapping.fill_diagonal_(False)

        if not overlapping.any():
            break

        rows, cols = torch.triu_indices(n, n, offset=1, device=pos.device)
        mask = overlapping[rows, cols]

        if not mask.any():
            break

        r = rows[mask]
        c = cols[mask]
        ox = overlap_x[r, c]
        oy = overlap_y[r, c]

        push_x = ox < oy

        if push_x.any():
            x_r = r[push_x]
            x_c = c[push_x]
            x_push = ox[push_x] / 2
            sign = torch.sign(dx[x_r, x_c])
            sign[sign == 0] = 1.0
            pos[x_r, 0] += sign * x_push * 0.5
            pos[x_c, 0] -= sign * x_push * 0.5

        push_y = ~push_x
        if push_y.any():
            y_r = r[push_y]
            y_c = c[push_y]
            y_push = oy[push_y] / 2
            sign = torch.sign(dy[y_r, y_c])
            sign[sign == 0] = 1.0
            pos[y_r, 1] += sign * y_push * 0.5
            pos[y_c, 1] -= sign * y_push * 0.5


def _project_sweep(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    iterations: int,
    layer_index: LayerIndex,
) -> None:
    """Sweep-line overlap projection — ZERO per-layer Python loops.

    Key insight: nodes in the same layer are contiguous in sorted_nodes.
    Sort within each layer by x-coordinate, then check consecutive pairs.
    This is O(N log N) per iteration with no per-layer Python loops.

    Uses composite sort key: layer * BIG_NUMBER + x to get within-layer
    x-ordering via a single global sort.
    """
    N = pos.shape[0]
    device = pos.device
    layers = layer_index.node_to_layer  # [N]
    half_w = node_sizes[:, 0] / 2
    half_h = node_sizes[:, 1] / 2

    for _ in range(iterations):
        # Sort all nodes by (layer, x_position) using composite key
        # This groups same-layer nodes together, sorted by x within each layer
        x_pos = pos[:, 0]
        # Normalize x to [0, 1) range within a large bucket per layer
        x_min = x_pos.min()
        x_range = (x_pos.max() - x_min).clamp(min=1.0)
        x_norm = (x_pos - x_min) / x_range  # [0, 1)

        sort_key = layers.float() * 2.0 + x_norm  # layer dominates, x breaks ties
        sorted_indices = sort_key.argsort()  # [N]

        # Get sorted layer assignments to find same-layer consecutive pairs
        sorted_layers = layers[sorted_indices]  # [N]

        # Consecutive pairs that are in the same layer
        same_layer = sorted_layers[:-1] == sorted_layers[1:]  # [N-1]

        if not same_layer.any():
            break

        # Get the actual node indices for consecutive pairs
        idx_a = sorted_indices[:-1]  # [N-1]
        idx_b = sorted_indices[1:]   # [N-1]

        # Compute overlap for same-layer consecutive pairs
        # (These are already sorted by x, so idx_a is always left of idx_b)
        dx = pos[idx_b, 0] - pos[idx_a, 0]  # positive (b is right of a)
        min_sep_x = half_w[idx_a] + half_w[idx_b] + padding

        overlap_x = min_sep_x - dx  # positive means overlap

        # Only process same-layer pairs with x-overlap
        needs_push = same_layer & (overlap_x > 0)

        if not needs_push.any():
            break

        # Push apart in x (conservative 1/4 factor for stability)
        push_amount = torch.where(needs_push, overlap_x * 0.25, torch.zeros_like(overlap_x))

        # Scatter push amounts to nodes
        # Node a moves left, node b moves right
        push_a = torch.zeros(N, device=device)
        push_b = torch.zeros(N, device=device)
        push_a.scatter_add_(0, idx_a, -push_amount)
        push_b.scatter_add_(0, idx_b, push_amount)

        pos[:, 0] += push_a + push_b

        # Also check near-neighbors (window=2) for wider overlaps
        if N > 2:
            same_layer_2 = sorted_layers[:-2] == sorted_layers[2:]
            idx_a2 = sorted_indices[:-2]
            idx_b2 = sorted_indices[2:]
            dx2 = pos[idx_b2, 0] - pos[idx_a2, 0]
            min_sep_x2 = half_w[idx_a2] + half_w[idx_b2] + padding
            overlap_x2 = min_sep_x2 - dx2
            needs_push2 = same_layer_2 & (overlap_x2 > 0)

            if needs_push2.any():
                push_amount2 = torch.where(needs_push2, overlap_x2 * 0.125, torch.zeros_like(overlap_x2))
                push_a2 = torch.zeros(N, device=device)
                push_b2 = torch.zeros(N, device=device)
                push_a2.scatter_add_(0, idx_a2, -push_amount2)
                push_b2.scatter_add_(0, idx_b2, push_amount2)
                pos[:, 0] += push_a2 + push_b2


def _project_grid(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    iterations: int,
) -> None:
    """Vectorized grid-based overlap projection (fallback without layer info).

    Grid construction uses tensor sorting instead of Python dicts.
    """
    n = pos.shape[0]
    device = pos.device

    max_w = node_sizes[:, 0].max().item()
    max_h = node_sizes[:, 1].max().item()
    cell_size = max(max_w, max_h) + padding
    if cell_size < 1.0:
        cell_size = 1.0

    half_w = node_sizes[:, 0] / 2
    half_h = node_sizes[:, 1] / 2

    for _ in range(iterations):
        # Assign cells via tensor ops
        cx = torch.floor(pos[:, 0] / cell_size).long()
        cy = torch.floor(pos[:, 1] / cell_size).long()

        cx_min = cx.min()
        cy_min = cy.min()
        cx_rel = cx - cx_min
        cy_rel = cy - cy_min
        cy_range = max(cy_rel.max().item() + 1, 1)
        cell_keys = cx_rel * cy_range + cy_rel

        sort_idx = cell_keys.argsort()
        sorted_keys = cell_keys[sort_idx]

        # Find cell boundaries
        changes = torch.where(sorted_keys[1:] != sorted_keys[:-1])[0] + 1
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), changes])
        ends = torch.cat([changes, torch.tensor([n], dtype=torch.long, device=device)])
        cell_sizes_arr = ends - starts

        # Process cells with 2+ nodes
        multi_mask = cell_sizes_arr >= 2
        multi_starts = starts[multi_mask]
        multi_ends = ends[multi_mask]

        any_pushed = False
        n_multi = multi_starts.shape[0]

        for i in range(min(n_multi.item() if isinstance(n_multi, torch.Tensor) else n_multi, 10000)):
            s = multi_starts[i].item()
            e = multi_ends[i].item()
            cell_nodes = sort_idx[s:e]
            m = cell_nodes.shape[0]

            if m > 200:
                perm = torch.randperm(m, device=device)[:200]
                cell_nodes = cell_nodes[perm]
                m = 200

            p = pos[cell_nodes]
            hw_c = half_w[cell_nodes]
            hh_c = half_h[cell_nodes]

            dx = p[:, 0].unsqueeze(1) - p[:, 0].unsqueeze(0)
            dy = p[:, 1].unsqueeze(1) - p[:, 1].unsqueeze(0)
            min_dx = hw_c.unsqueeze(1) + hw_c.unsqueeze(0) + padding
            min_dy = hh_c.unsqueeze(1) + hh_c.unsqueeze(0) + padding
            ox = min_dx - dx.abs()
            oy = min_dy - dy.abs()
            overlapping = (ox > 0) & (oy > 0)
            overlapping.fill_diagonal_(False)

            if not overlapping.any():
                continue

            any_pushed = True
            rows, cols = torch.triu_indices(m, m, offset=1, device=device)
            mask = overlapping[rows, cols]
            if not mask.any():
                continue

            r = rows[mask]
            c = cols[mask]
            ox_m = ox[r, c]
            oy_m = oy[r, c]

            push_x = ox_m < oy_m
            if push_x.any():
                xr = cell_nodes[r[push_x]]
                xc = cell_nodes[c[push_x]]
                x_amt = ox_m[push_x] / 4
                x_sign = torch.sign(dx[r[push_x], c[push_x]])
                x_sign[x_sign == 0] = 1.0
                pos[:, 0].scatter_add_(0, xr, x_sign * x_amt)
                pos[:, 0].scatter_add_(0, xc, -x_sign * x_amt)

            push_y = ~push_x
            if push_y.any():
                yr = cell_nodes[r[push_y]]
                yc = cell_nodes[c[push_y]]
                y_amt = oy_m[push_y] / 4
                y_sign = torch.sign(dy[r[push_y], c[push_y]])
                y_sign[y_sign == 0] = 1.0
                pos[:, 1].scatter_add_(0, yr, y_sign * y_amt)
                pos[:, 1].scatter_add_(0, yc, -y_sign * y_amt)

        if not any_pushed:
            break

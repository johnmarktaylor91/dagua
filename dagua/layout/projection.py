"""Hard overlap resolution via projected gradient descent.

After each optimizer step, pushes overlapping node bounding boxes apart.
Runs deterministically with torch.no_grad().
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import torch


def project_overlaps(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float = 2.0,
    iterations: int = 10,
) -> torch.Tensor:
    """Push overlapping node bounding boxes apart in-place.

    Uses iterative pairwise push-apart. Each iteration finds overlapping pairs
    and pushes them apart along the minimum separation axis.

    For N <= 500: exact O(N^2) pairwise check per iteration.
    For N > 500: grid-based spatial hashing, checking only same + 8 neighbor
    cells. Expected O(N) per iteration for non-pathological layouts.

    Returns the (modified) pos tensor.
    """
    n = pos.shape[0]
    if n <= 1:
        return pos

    with torch.no_grad():
        if n <= 500:
            _project_overlaps_exact(pos, node_sizes, padding, iterations)
        else:
            _project_overlaps_grid(pos, node_sizes, padding, iterations)

    return pos


def _project_overlaps_exact(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    iterations: int,
) -> None:
    """Exact O(N^2) overlap projection for small graphs."""
    n = pos.shape[0]

    for _ in range(iterations):
        # Compute pairwise distances and required separations
        dx = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)  # [N, N]
        dy = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)

        # Required minimum separations
        min_dx = (node_sizes[:, 0].unsqueeze(1) + node_sizes[:, 0].unsqueeze(0)) / 2 + padding
        min_dy = (node_sizes[:, 1].unsqueeze(1) + node_sizes[:, 1].unsqueeze(0)) / 2 + padding

        # Find overlapping pairs (both axes overlapping)
        overlap_x = min_dx - dx.abs()
        overlap_y = min_dy - dy.abs()
        overlapping = (overlap_x > 0) & (overlap_y > 0)

        # Zero out diagonal
        overlapping.fill_diagonal_(False)

        if not overlapping.any():
            break

        # For each overlapping pair, push apart along min overlap axis
        # Use upper triangle only to avoid double-pushing
        rows, cols = torch.triu_indices(n, n, offset=1, device=pos.device)
        mask = overlapping[rows, cols]

        if not mask.any():
            break

        r = rows[mask]
        c = cols[mask]

        ox = overlap_x[r, c]
        oy = overlap_y[r, c]

        # Push along axis with LESS overlap (easier to resolve)
        push_x = ox < oy

        # X-axis push
        if push_x.any():
            x_r = r[push_x]
            x_c = c[push_x]
            x_push = ox[push_x] / 2
            sign = torch.sign(dx[x_r, x_c])
            sign[sign == 0] = 1.0
            pos[x_r, 0] += sign * x_push * 0.5
            pos[x_c, 0] -= sign * x_push * 0.5

        # Y-axis push
        push_y = ~push_x
        if push_y.any():
            y_r = r[push_y]
            y_c = c[push_y]
            y_push = oy[push_y] / 2
            sign = torch.sign(dy[y_r, y_c])
            sign[sign == 0] = 1.0
            pos[y_r, 1] += sign * y_push * 0.5
            pos[y_c, 1] -= sign * y_push * 0.5


def _project_overlaps_grid(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    iterations: int,
) -> None:
    """Grid-based O(N) overlap projection for large graphs.

    Assigns each node to a grid cell. Cell size is 2 * max(node_dimension) + padding
    so that any two overlapping nodes must be in the same or adjacent cells.
    For each iteration, only checks the 3x3 neighborhood of each cell.
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
        # Assign nodes to grid cells
        cx = torch.floor(pos[:, 0] / cell_size).long()
        cy = torch.floor(pos[:, 1] / cell_size).long()

        grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        cx_list = cx.tolist()
        cy_list = cy.tolist()
        for i in range(n):
            grid[(cx_list[i], cy_list[i])].append(i)

        # Collect overlapping pairs from grid neighborhoods
        push_indices_r = []
        push_indices_c = []
        push_ox = []
        push_oy = []

        pos_x = pos[:, 0]
        pos_y = pos[:, 1]

        for (gx, gy), cell_nodes in grid.items():
            if not cell_nodes:
                continue

            # Collect all candidate neighbor nodes
            neighbor_nodes = []
            for ddx in range(-1, 2):
                for ddy in range(-1, 2):
                    nkey = (gx + ddx, gy + ddy)
                    if nkey in grid:
                        if nkey == (gx, gy):
                            continue  # handle same-cell separately
                        if nkey > (gx, gy):
                            neighbor_nodes.extend(grid[nkey])

            # Check all pairs within the same cell
            for a_pos in range(len(cell_nodes)):
                a = cell_nodes[a_pos]
                for b_pos in range(a_pos + 1, len(cell_nodes)):
                    b = cell_nodes[b_pos]
                    dx_val = (pos_x[a] - pos_x[b]).item()
                    dy_val = (pos_y[a] - pos_y[b]).item()
                    min_dx_val = (half_w[a] + half_w[b]).item() + padding
                    min_dy_val = (half_h[a] + half_h[b]).item() + padding
                    ox = min_dx_val - abs(dx_val)
                    oy = min_dy_val - abs(dy_val)
                    if ox > 0 and oy > 0:
                        push_indices_r.append(a)
                        push_indices_c.append(b)
                        push_ox.append(ox)
                        push_oy.append(oy)

            # Check cross-cell pairs
            for a in cell_nodes:
                for b in neighbor_nodes:
                    dx_val = (pos_x[a] - pos_x[b]).item()
                    dy_val = (pos_y[a] - pos_y[b]).item()
                    min_dx_val = (half_w[a] + half_w[b]).item() + padding
                    min_dy_val = (half_h[a] + half_h[b]).item() + padding
                    ox = min_dx_val - abs(dx_val)
                    oy = min_dy_val - abs(dy_val)
                    if ox > 0 and oy > 0:
                        push_indices_r.append(a)
                        push_indices_c.append(b)
                        push_ox.append(ox)
                        push_oy.append(oy)

        if not push_indices_r:
            break

        # Apply pushes
        r = torch.tensor(push_indices_r, dtype=torch.long, device=device)
        c = torch.tensor(push_indices_c, dtype=torch.long, device=device)
        ox_t = torch.tensor(push_ox, dtype=pos.dtype, device=device)
        oy_t = torch.tensor(push_oy, dtype=pos.dtype, device=device)

        dx_sign = torch.sign(pos_x[r] - pos_x[c])
        dy_sign = torch.sign(pos_y[r] - pos_y[c])
        dx_sign[dx_sign == 0] = 1.0
        dy_sign[dy_sign == 0] = 1.0

        push_x_mask = ox_t < oy_t

        # X-axis push
        if push_x_mask.any():
            xr = r[push_x_mask]
            xc = c[push_x_mask]
            x_amount = ox_t[push_x_mask] / 2 * 0.5
            x_sign = dx_sign[push_x_mask]
            # Use scatter_add for concurrent updates
            pos[:, 0].scatter_add_(0, xr, x_sign * x_amount)
            pos[:, 0].scatter_add_(0, xc, -x_sign * x_amount)

        # Y-axis push
        push_y_mask = ~push_x_mask
        if push_y_mask.any():
            yr = r[push_y_mask]
            yc = c[push_y_mask]
            y_amount = oy_t[push_y_mask] / 2 * 0.5
            y_sign = dy_sign[push_y_mask]
            pos[:, 1].scatter_add_(0, yr, y_sign * y_amount)
            pos[:, 1].scatter_add_(0, yc, -y_sign * y_amount)

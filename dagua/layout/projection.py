"""Hard overlap resolution via projected gradient descent.

After each optimizer step, pushes overlapping node bounding boxes apart.
Runs deterministically with torch.no_grad().
"""

from __future__ import annotations

import torch


def project_overlaps(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float = 2.0,
    iterations: int = 10,
) -> None:
    """Push overlapping node bounding boxes apart in-place.

    Uses iterative pairwise push-apart. Each iteration finds overlapping pairs
    and pushes them apart along the minimum separation axis.
    """
    n = pos.shape[0]
    if n <= 1:
        return pos

    with torch.no_grad():
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

    return pos

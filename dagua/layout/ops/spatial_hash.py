"""Uniform spatial hash helpers for local pair queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


def _empty_pairs(device: torch.device) -> torch.Tensor:
    """Return an empty pair tensor on the requested device.

    Parameters
    ----------
    device : torch.device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Empty long tensor with shape ``[2, 0]``.
    """
    return torch.empty((2, 0), dtype=torch.long, device=device)


def _cell_boundaries(sorted_keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return start and end offsets for consecutive sorted cell keys.

    Parameters
    ----------
    sorted_keys : torch.Tensor
        One-dimensional sorted integer cell keys with shape ``[N]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Start and end offsets for each occupied cell.
    """
    num_nodes = int(sorted_keys.shape[0])
    if num_nodes == 0:
        empty = torch.empty((0,), dtype=torch.long, device=sorted_keys.device)
        return empty, empty
    changes = torch.where(sorted_keys[1:] != sorted_keys[:-1])[0] + 1
    starts = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=sorted_keys.device), changes],
    )
    ends = torch.cat(
        [changes, torch.tensor([num_nodes], dtype=torch.long, device=sorted_keys.device)],
    )
    return starts, ends


@dataclass(frozen=True)
class UniformSpatialHash:
    """Uniform grid spatial index for candidate pair generation.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    cutoff_radius : float
        Query radius. Cells have this side length, so any pair within the
        radius must lie in the same or one of the eight adjacent cells.
    """

    positions: torch.Tensor
    cutoff_radius: float

    def __post_init__(self) -> None:
        """Validate construction inputs."""
        if self.positions.ndim != 2 or int(self.positions.shape[1]) != 2:
            raise ValueError("positions must have shape [N, 2].")
        if self.cutoff_radius <= 0.0:
            raise ValueError("cutoff_radius must be positive.")

    def candidate_pairs(self) -> torch.Tensor:
        """Return unique unordered candidate pairs from adjacent grid cells.

        Returns
        -------
        torch.Tensor
            Long tensor with shape ``[2, M]`` containing unique unordered
            candidate pairs. The set has no false negatives for pairs whose
            Euclidean distance is at most ``cutoff_radius``.
        """
        num_nodes = int(self.positions.shape[0])
        if num_nodes <= 1:
            return _empty_pairs(self.positions.device)

        detached_pos = self.positions.detach()
        cell_size = float(self.cutoff_radius)
        cell_xy = torch.floor(detached_pos / cell_size).to(dtype=torch.long)
        min_xy = cell_xy.min(dim=0).values
        rel_xy = cell_xy - min_xy
        y_range = int(rel_xy[:, 1].max().item()) + 1
        y_range = max(y_range, 1)
        cell_keys = rel_xy[:, 0] * y_range + rel_xy[:, 1]

        sort_idx = cell_keys.argsort()
        sorted_keys = cell_keys[sort_idx]
        starts, ends = _cell_boundaries(sorted_keys)
        if int(starts.shape[0]) == 0:
            return _empty_pairs(self.positions.device)

        key_to_span: Dict[int, Tuple[int, int]] = {}
        for key, start, end in zip(sorted_keys[starts].tolist(), starts.tolist(), ends.tolist()):
            key_to_span[int(key)] = (int(start), int(end))

        pair_blocks: List[torch.Tensor] = []
        neighbor_offsets = ((0, 0), (0, 1), (1, -1), (1, 0), (1, 1))
        for key, (start, end) in key_to_span.items():
            cell_nodes = sort_idx[start:end]
            x_rel = key // y_range
            y_rel = key % y_range
            for dx, dy in neighbor_offsets:
                neighbor_y = y_rel + dy
                if neighbor_y < 0 or neighbor_y >= y_range:
                    continue
                neighbor_key = (x_rel + dx) * y_range + neighbor_y
                neighbor_span = key_to_span.get(int(neighbor_key))
                if neighbor_span is None:
                    continue
                if neighbor_key == key:
                    block = self._same_cell_pairs(cell_nodes)
                else:
                    neighbor_nodes = sort_idx[neighbor_span[0] : neighbor_span[1]]
                    block = self._cross_cell_pairs(cell_nodes, neighbor_nodes)
                if int(block.shape[1]) > 0:
                    pair_blocks.append(block)

        if not pair_blocks:
            return _empty_pairs(self.positions.device)
        return torch.cat(pair_blocks, dim=1)

    def candidate_neighbors(self) -> List[torch.Tensor]:
        """Return per-node candidate neighbor lists.

        Returns
        -------
        list[torch.Tensor]
            List of length ``N`` where each tensor contains candidate neighbor
            indices for the corresponding node. Candidate membership is
            non-differentiable; downstream loss math should re-index
            ``positions`` to retain normal autograd flow.
        """
        pairs = self.candidate_pairs()
        neighbors: List[List[torch.Tensor]] = [[] for _ in range(int(self.positions.shape[0]))]
        for pair_index in range(int(pairs.shape[1])):
            source = int(pairs[0, pair_index].item())
            target = int(pairs[1, pair_index].item())
            neighbors[source].append(pairs[1, pair_index])
            neighbors[target].append(pairs[0, pair_index])
        return [
            torch.stack(items).to(dtype=torch.long, device=self.positions.device)
            if items
            else torch.empty((0,), dtype=torch.long, device=self.positions.device)
            for items in neighbors
        ]

    def _same_cell_pairs(self, cell_nodes: torch.Tensor) -> torch.Tensor:
        """Return unique pairs inside one occupied cell.

        Parameters
        ----------
        cell_nodes : torch.Tensor
            Node indices assigned to one cell with shape ``[C]``.

        Returns
        -------
        torch.Tensor
            Long tensor with shape ``[2, M]``.
        """
        cell_count = int(cell_nodes.shape[0])
        if cell_count <= 1:
            return _empty_pairs(self.positions.device)
        row, col = torch.triu_indices(
            cell_count,
            cell_count,
            offset=1,
            device=self.positions.device,
        )
        left = cell_nodes[row]
        right = cell_nodes[col]
        return torch.stack([torch.minimum(left, right), torch.maximum(left, right)])

    def _cross_cell_pairs(
        self,
        left_nodes: torch.Tensor,
        right_nodes: torch.Tensor,
    ) -> torch.Tensor:
        """Return all pairs between two different occupied cells.

        Parameters
        ----------
        left_nodes : torch.Tensor
            Node indices in the first cell with shape ``[L]``.
        right_nodes : torch.Tensor
            Node indices in the neighboring cell with shape ``[R]``.

        Returns
        -------
        torch.Tensor
            Long tensor with shape ``[2, L * R]``.
        """
        left_count = int(left_nodes.shape[0])
        right_count = int(right_nodes.shape[0])
        if left_count == 0 or right_count == 0:
            return _empty_pairs(self.positions.device)
        left = left_nodes.repeat_interleave(right_count)
        right = right_nodes.repeat(left_count)
        return torch.stack([torch.minimum(left, right), torch.maximum(left, right)])


def cell_list_candidate_pairs(positions: torch.Tensor, cutoff_radius: float) -> torch.Tensor:
    """Return candidate pairs from a uniform spatial hash.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    cutoff_radius : float
        Radius used as the uniform grid cell size.

    Returns
    -------
    torch.Tensor
        Long tensor with shape ``[2, M]`` containing candidate pairs.
    """
    return UniformSpatialHash(positions=positions, cutoff_radius=cutoff_radius).candidate_pairs()


__all__ = ["UniformSpatialHash", "cell_list_candidate_pairs"]

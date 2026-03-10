"""Layer index: pre-computed per-layer node groupings for O(1) lookup.

Built once per layout() call. Passed to all layer-aware loss functions,
projection, and overlap detection. Eliminates repeated Python dict-building.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class LayerIndex:
    """Pre-computed layer structure for efficient per-layer operations.

    Attributes:
        node_to_layer: [N] tensor mapping node index -> layer index.
        layer_offsets: [L+1] tensor. Nodes in layer k are at sorted indices
            layer_offsets[k]:layer_offsets[k+1].
        sorted_nodes: [N] tensor. Nodes sorted by layer.
        num_layers: Number of distinct layers.
    """
    node_to_layer: torch.Tensor   # [N] long
    layer_offsets: torch.Tensor   # [L+1] long
    sorted_nodes: torch.Tensor    # [N] long
    num_layers: int

    def nodes_in_layer(self, layer: int) -> torch.Tensor:
        """Return node indices for a specific layer."""
        start = self.layer_offsets[layer].item()
        end = self.layer_offsets[layer + 1].item()
        return self.sorted_nodes[start:end]

    def layer_sizes(self) -> torch.Tensor:
        """Return [L] tensor of nodes per layer."""
        return self.layer_offsets[1:] - self.layer_offsets[:-1]

    def max_layer_width(self) -> int:
        """Return the maximum number of nodes in any single layer."""
        return int(self.layer_sizes().max().item())


def build_layer_index(
    layer_assignments,
    device: str = "cpu",
) -> LayerIndex:
    """Build a LayerIndex from layer assignments (List[int] or torch.Tensor).

    O(N log N) from the sort. All subsequent per-layer operations are O(1) indexed.
    """
    if isinstance(layer_assignments, torch.Tensor):
        node_to_layer = layer_assignments.to(dtype=torch.long, device=device)
        n = node_to_layer.shape[0]
    else:
        n = len(layer_assignments)
        node_to_layer = torch.tensor(layer_assignments, dtype=torch.long, device=device)
    num_layers = int(node_to_layer.max().item()) + 1 if n > 0 else 0

    # Sort nodes by layer
    sorted_indices = node_to_layer.argsort()
    sorted_layers = node_to_layer[sorted_indices]

    # Compute layer boundaries
    # layer_offsets[k] = first position in sorted_nodes where layer == k
    offsets = torch.zeros(num_layers + 1, dtype=torch.long, device=device)
    if n > 0:
        # Count nodes per layer
        counts = torch.bincount(node_to_layer, minlength=num_layers)
        offsets[1:] = counts.cumsum(0)

    return LayerIndex(
        node_to_layer=node_to_layer,
        layer_offsets=offsets,
        sorted_nodes=sorted_indices,
        num_layers=num_layers,
    )

"""Layer index: pre-computed per-layer node groupings for O(1) lookup.

Built once per layout() call. Passed to all layer-aware loss functions,
projection, and overlap detection. Eliminates repeated Python dict-building.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch

from dagua.utils import _cuda_stage_status_message


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

    node_to_layer: torch.Tensor  # [N] int32/long
    layer_offsets: torch.Tensor  # [L+1] long
    sorted_nodes: torch.Tensor  # [N] long
    num_layers: int

    def nodes_in_layer(self, layer: int) -> torch.Tensor:
        """Return node indices for a specific layer."""
        start = int(self.layer_offsets[layer].item())
        end = int(self.layer_offsets[layer + 1].item())
        return self.sorted_nodes[start:end]

    def layer_sizes(self) -> torch.Tensor:
        """Return [L] tensor of nodes per layer."""
        return self.layer_offsets[1:] - self.layer_offsets[:-1]

    def max_layer_width(self) -> int:
        """Return the maximum number of nodes in any single layer."""
        return int(self.layer_sizes().max().item())


def _log_layer_index_status(
    message: str,
    progress: Optional[Callable[[str], None]],
    verbose: bool,
) -> None:
    """Emit a LayerIndex status message when verbose logging is enabled.

    Parameters
    ----------
    message : str
        Status message without the ``[dagua]`` prefix.
    progress : Callable[[str], None] | None
        Optional progress sink used by callers that own log formatting.
    verbose : bool
        Whether fallback and activation logs should be emitted.

    Returns
    -------
    None
    """
    if not verbose:
        return
    if progress is not None:
        progress(message)
    else:
        print(f"[dagua]   {message}", flush=True)


def _layer_index_gpu_sort_required_bytes(num_nodes: int) -> int:
    """Return the conservative scratch estimate for CUDA layer-index sorting.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the layer assignment tensor.

    Returns
    -------
    int
        Estimated bytes for input, scratch, and output tensors during CUDA
        argsort.
    """
    return num_nodes * 8 * 3


def _cuda_layer_argsort(
    node_to_layer: torch.Tensor,
    output_device: str,
) -> torch.Tensor:
    """Run a stable CUDA argsort for layer assignments.

    Parameters
    ----------
    node_to_layer : torch.Tensor
        Layer assignment tensor with shape ``[N]``.
    output_device : str
        Device for the returned permutation tensor.

    Returns
    -------
    torch.Tensor
        Stable layer-sorted node order with shape ``[N]``.
    """
    sort_input = (
        node_to_layer
        if node_to_layer.device.type == "cuda"
        else node_to_layer.to(device="cuda", dtype=node_to_layer.dtype)
    )
    sorted_indices = sort_input.argsort(stable=True)
    return sorted_indices.to(device=output_device)


def build_layer_index(
    layer_assignments: Union[List[int], torch.Tensor],
    device: str = "cpu",
    *,
    verbose: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    enable_cuda_sort: bool = True,
) -> LayerIndex:
    """Build a layer index from node layer assignments.

    Parameters
    ----------
    layer_assignments : list[int] | torch.Tensor
        Layer IDs for each node with shape ``[N]``.
    device : str, default="cpu"
        Device for the returned ``LayerIndex`` tensors.
    verbose : bool, default=False
        Whether to emit CUDA activation and fallback logs.
    progress : Callable[[str], None], optional
        Optional progress sink used by callers that own log formatting.
    enable_cuda_sort : bool, default=True
        Whether CUDA argsort may be used when enough VRAM is available.

    Returns
    -------
    LayerIndex
        Indexed layer structure with stable node ordering inside each layer.

    Notes
    -----
    The sort is always stable so equal-layer nodes keep ascending node-ID
    order, which is required by the streaming coarsening tie-break contract.
    """
    index_dtype = (
        torch.int32 if len(layer_assignments) <= torch.iinfo(torch.int32).max else torch.long
    )
    if isinstance(layer_assignments, torch.Tensor):
        index_dtype = (
            torch.int32
            if layer_assignments.shape[0] <= torch.iinfo(torch.int32).max
            else torch.long
        )
        node_to_layer = layer_assignments.to(dtype=index_dtype, device=device)
        n = node_to_layer.shape[0]
    else:
        n = len(layer_assignments)
        node_to_layer = torch.tensor(layer_assignments, dtype=index_dtype, device=device)
    num_layers = int(node_to_layer.max().item()) + 1 if n > 0 else 0

    sorted_indices: torch.Tensor
    required_bytes = _layer_index_gpu_sort_required_bytes(n)
    free_vram = 0
    use_cuda_sort = False
    if enable_cuda_sort and n > 0 and torch.cuda.is_available():
        try:
            free_vram, _total_vram = torch.cuda.mem_get_info()
        except RuntimeError:
            free_vram = 0
        else:
            use_cuda_sort = required_bytes < int(free_vram * 0.6)

        if use_cuda_sort:
            _log_layer_index_status(
                _cuda_stage_status_message("LayerIndex", "cuda", required_bytes, free_vram),
                progress=progress,
                verbose=verbose,
            )
            try:
                sorted_indices = _cuda_layer_argsort(node_to_layer, output_device=device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                _log_layer_index_status(
                    _cuda_stage_status_message("LayerIndex", "oom_fallback"),
                    progress=progress,
                    verbose=verbose,
                )
                sorted_indices = node_to_layer.argsort(stable=True)
        else:
            _log_layer_index_status(
                _cuda_stage_status_message("LayerIndex", "cpu_fallback", required_bytes, free_vram),
                progress=progress,
                verbose=verbose,
            )
            sorted_indices = node_to_layer.argsort(stable=True)
    else:
        if enable_cuda_sort and verbose and n > 0 and not torch.cuda.is_available():
            _log_layer_index_status(
                _cuda_stage_status_message("LayerIndex", "cpu_fallback", required_bytes, 0),
                progress=progress,
                verbose=verbose,
            )
        sorted_indices = node_to_layer.argsort(stable=True)

    # Compute layer boundaries
    # layer_offsets[k] = first position in sorted_nodes where layer == k
    offsets = torch.zeros(num_layers + 1, dtype=torch.long, device=device)
    if n > 0:
        # Count nodes per layer
        counts = torch.bincount(node_to_layer.to(dtype=torch.long), minlength=num_layers)
        offsets[1:] = counts.cumsum(0)

    return LayerIndex(
        node_to_layer=node_to_layer,
        layer_offsets=offsets,
        sorted_nodes=sorted_indices,
        num_layers=num_layers,
    )

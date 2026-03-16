"""Tiled GPU loss evaluation for layouts that exceed available VRAM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

from dagua.layout.layers import LayerIndex, build_layer_index

LossFn = Callable[[torch.Tensor, torch.Tensor, Optional[LayerIndex]], torch.Tensor]

_BYTES_PER_TILE_NODE = 64
_CUDA_CONTEXT_OVERHEAD_BYTES = 500_000_000
_EDGE_BATCH_BYTES = 64
_EDGE_BUDGET_RATIO = 0.30
_MIN_TILE_SIZE = 1_000_000
_MIN_EDGE_BATCH = 100_000


@dataclass
class _TileEdgeContext:
    """Pre-computed edge tensors for the active tiled edge batch."""

    src: torch.Tensor
    tgt: torch.Tensor
    dx: torch.Tensor
    dy: torch.Tensor
    dist_sq: torch.Tensor


def _ensure_node_sizes_2d(node_sizes: torch.Tensor) -> torch.Tensor:
    """Normalize node sizes to ``[N, 2]``.

    Parameters
    ----------
    node_sizes : torch.Tensor
        Node size tensor shaped ``[N]``, ``[N, 1]``, or ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Contiguous ``[N, 2]`` tensor on CPU.
    """
    if node_sizes.ndim == 1:
        return node_sizes.unsqueeze(1).expand(-1, 2).contiguous().cpu()
    if node_sizes.ndim == 2 and node_sizes.shape[1] == 1:
        return node_sizes.expand(-1, 2).contiguous().cpu()
    return node_sizes.contiguous().cpu()


def _compute_tile_size(num_nodes: int, vram_bytes: int) -> int:
    """Compute the conservative node count for one GPU tile.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the graph.
    vram_bytes : int
        Available VRAM budget in bytes.

    Returns
    -------
    int
        Maximum nodes per tile, clamped to the graph size.
    """
    edge_budget = int(vram_bytes * _EDGE_BUDGET_RATIO)
    node_budget = max(vram_bytes - edge_budget - _CUDA_CONTEXT_OVERHEAD_BYTES, 1)
    tile_size = int(node_budget / _BYTES_PER_TILE_NODE)
    return min(num_nodes, max(tile_size, min(_MIN_TILE_SIZE, num_nodes)))


def _partition_edges_by_tile(
    edge_index: torch.Tensor,
    tile_starts: list[int],
    tile_ends: list[int],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Partition edges into tile-local and cross-tile subsets.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor shaped ``[2, E]``.
    tile_starts : list[int]
        Inclusive tile start offsets.
    tile_ends : list[int]
        Exclusive tile end offsets.

    Returns
    -------
    tuple[list[torch.Tensor], torch.Tensor]
        Local edge tensors reindexed to each tile, plus the residual cross-tile
        edge tensor on CPU.
    """
    if edge_index.numel() == 0:
        empty = torch.zeros((2, 0), dtype=edge_index.dtype)
        return [empty.clone() for _ in tile_starts], empty

    src = edge_index[0]
    dst = edge_index[1]
    tile_edges: list[torch.Tensor] = []
    cross_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)

    for start, end in zip(tile_starts, tile_ends):
        local_mask = (src >= start) & (src < end) & (dst >= start) & (dst < end)
        tile_edges.append((edge_index[:, local_mask] - start).contiguous())
        cross_mask &= ~local_mask

    return tile_edges, edge_index[:, cross_mask].contiguous()


def _remove_self_loops(edge_index: torch.Tensor) -> torch.Tensor:
    """Drop self-loop edges from an edge batch.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.

    Returns
    -------
    torch.Tensor
        Edge tensor with ``src != dst``.
    """
    if edge_index.numel() == 0:
        return edge_index
    keep = edge_index[0] != edge_index[1]
    return edge_index[:, keep].contiguous()


def _compute_edge_batch_size(vram_bytes: int) -> int:
    """Return the conservative number of edges per tiled GPU batch.

    Parameters
    ----------
    vram_bytes : int
        Available VRAM budget in bytes.

    Returns
    -------
    int
        Maximum number of edges per batch.
    """
    edge_budget = max(int(vram_bytes * _EDGE_BUDGET_RATIO), _MIN_EDGE_BATCH * _EDGE_BATCH_BYTES)
    return max(int(edge_budget / _EDGE_BATCH_BYTES), _MIN_EDGE_BATCH)


class TiledGPUCompute:
    """Tiled GPU loss computation for graphs that exceed VRAM.

    Parameters
    ----------
    num_nodes : int
        Total number of graph nodes.
    edge_index : torch.Tensor
        CPU edge tensor shaped ``[2, E]``.
    node_sizes : torch.Tensor
        CPU node size tensor shaped ``[N, 2]``.
    device : str, default="cuda"
        CUDA device used for tile execution.
    vram_budget : int, optional
        Explicit VRAM budget in bytes. When omitted, the current free VRAM is
        queried from CUDA.
    """

    def __init__(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        node_sizes: torch.Tensor,
        device: str = "cuda",
        vram_budget: Optional[int] = None,
    ) -> None:
        self.num_nodes = num_nodes
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("TiledGPUCompute requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        self.edge_index = edge_index.cpu().contiguous()
        self.node_sizes = _ensure_node_sizes_2d(node_sizes)
        self.layer_assignments: Optional[torch.Tensor] = None
        self.cross_tile_loss_mask: list[bool] = []
        self.current_edge_index: Optional[torch.Tensor] = None
        self.current_edge_ctx: Optional[_TileEdgeContext] = None
        self.last_unweighted_loss = 0.0

        self.vram_budget = (
            int(vram_budget)
            if vram_budget is not None
            else int(torch.cuda.mem_get_info(device=self.device)[0])
        )
        self.edge_batch_size = _compute_edge_batch_size(self.vram_budget)

        self._check_cpu_ram_available()
        self.tile_size = self._fit_tile_size_to_vram(
            _compute_tile_size(self.num_nodes, self.vram_budget)
        )
        self.tile_starts, self.tile_ends = self._build_tile_ranges()
        self.tile_edges, self.cross_edges = _partition_edges_by_tile(
            self.edge_index,
            self.tile_starts,
            self.tile_ends,
        )
        self.total_edge_count = (
            int((self.edge_index[0] != self.edge_index[1]).sum().item())
            if self.edge_index.numel() > 0
            else 0
        )

    def _check_cpu_ram_available(self) -> None:
        """Validate that host memory is sufficient for tiled execution.

        Raises
        ------
        MemoryError
            If the available RAM is below the conservative tiled-execution
            budget.
        """
        try:
            import psutil
        except ImportError:
            return

        edge_count = self.edge_index.shape[1] if self.edge_index.numel() > 0 else 0
        cpu_required = self.num_nodes * 20 + edge_count * 16 + self.num_nodes * 16
        available = int(psutil.virtual_memory().available)
        if available < int(cpu_required * 1.3):
            raise MemoryError(
                "Tiled GPU mode needs "
                f"~{cpu_required // 2**30}GB CPU RAM, only {available // 2**30}GB available"
            )

    def _fit_tile_size_to_vram(self, tile_size: int) -> int:
        """Shrink the requested tile size until it fits conservative VRAM headroom.

        Parameters
        ----------
        tile_size : int
            Initial tile size estimate.

        Returns
        -------
        int
            VRAM-safe tile size.
        """
        total_vram = torch.cuda.get_device_properties(self.device).total_memory
        safe_limit = int(total_vram * 0.85)
        adjusted = max(1, min(tile_size, self.num_nodes))
        while (
            adjusted * _BYTES_PER_TILE_NODE
            + min(self.edge_batch_size, max(self.edge_index.shape[1], 1)) * _EDGE_BATCH_BYTES
            + _CUDA_CONTEXT_OVERHEAD_BYTES
            > safe_limit
            and adjusted > 1
        ):
            adjusted = max(adjusted // 2, 1)
        return adjusted

    def _build_tile_ranges(self) -> tuple[list[int], list[int]]:
        """Return contiguous node ranges for tiled execution.

        Returns
        -------
        tuple[list[int], list[int]]
            Inclusive starts and exclusive ends for each tile.
        """
        starts = list(range(0, self.num_nodes, self.tile_size))
        ends = [min(start + self.tile_size, self.num_nodes) for start in starts]
        return starts, ends

    def _iter_edge_batches(self, edge_index: torch.Tensor) -> Sequence[torch.Tensor]:
        """Yield self-loop-free CPU edge batches.

        Parameters
        ----------
        edge_index : torch.Tensor
            CPU edge tensor shaped ``[2, E]``.

        Returns
        -------
        Sequence[torch.Tensor]
            Sequence of edge batches sized to the configured VRAM budget.
        """
        edge_count = edge_index.shape[1] if edge_index.numel() > 0 else 0
        if edge_count == 0:
            return ()
        if edge_count <= self.edge_batch_size:
            return (_remove_self_loops(edge_index),)
        batches = []
        for start in range(0, edge_count, self.edge_batch_size):
            end = min(start + self.edge_batch_size, edge_count)
            batches.append(_remove_self_loops(edge_index[:, start:end]))
        return tuple(batches)

    def _tile_layer_index(self, start: int, end: int) -> Optional[LayerIndex]:
        """Build the layer index for one tile.

        Parameters
        ----------
        start : int
            Inclusive tile start index.
        end : int
            Exclusive tile end index.

        Returns
        -------
        LayerIndex | None
            Tile-local layer index, or ``None`` when no layer assignments are
            available.
        """
        if self.layer_assignments is None:
            return None
        return build_layer_index(self.layer_assignments[start:end], device=self.device.type)

    def _edge_context(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Optional[_TileEdgeContext]:
        """Build the active edge context for the current GPU batch.

        Parameters
        ----------
        pos : torch.Tensor
            Tile-local positions shaped ``[T, 2]`` on CUDA.
        edge_index : torch.Tensor
            Tile-local edge tensor shaped ``[2, E]`` on CUDA.

        Returns
        -------
        _TileEdgeContext | None
            Pre-computed edge tensors, or ``None`` when the batch is empty.
        """
        if edge_index.numel() == 0:
            return None
        src = edge_index[0]
        tgt = edge_index[1]
        src_pos = pos[src]
        tgt_pos = pos[tgt]
        dx = src_pos[:, 0] - tgt_pos[:, 0]
        dy = src_pos[:, 1] - tgt_pos[:, 1]
        return _TileEdgeContext(
            src=src,
            tgt=tgt,
            dx=dx,
            dy=dy,
            dist_sq=dx.square() + dy.square(),
        )

    def _backward_loss_group(
        self,
        positions: torch.Tensor,
        node_sizes: torch.Tensor,
        layer_index: Optional[LayerIndex],
        loss_fns: Sequence[LossFn],
        loss_weights: Sequence[float],
        scale: float = 1.0,
    ) -> tuple[float, float]:
        """Evaluate and backpropagate one independent loss group.

        Parameters
        ----------
        positions : torch.Tensor
            Active position tensor shaped ``[T, 2]`` on CUDA.
        node_sizes : torch.Tensor
            Active node sizes shaped ``[T, 2]`` on CUDA.
        layer_index : LayerIndex, optional
            Tile-local layer structure.
        loss_fns : Sequence[LossFn]
            Loss functions to evaluate for this group.
        loss_weights : Sequence[float]
            Scalar weights matching ``loss_fns``.
        scale : float, default=1.0
            Multiplier used to preserve full-graph mean semantics when a loss is
            evaluated on only part of the graph.

        Returns
        -------
        tuple[float, float]
            Weighted and unweighted loss totals for logging / early stopping.
        """
        total = 0.0
        unweighted = 0.0
        combined: Optional[torch.Tensor] = None

        for loss_fn, weight in zip(loss_fns, loss_weights):
            term = weight * scale * loss_fn(positions, node_sizes, layer_index)
            value = float(term.item())
            total += value
            unweighted += value / weight if weight else 0.0
            combined = term if combined is None else combined + term

        if combined is not None and combined.requires_grad:
            combined.backward()
        return total, unweighted

    def _accumulate_leaf_grad(
        self,
        positions: torch.Tensor,
        grad_buffer: torch.Tensor,
        start: Optional[int] = None,
        node_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Copy a CUDA leaf gradient back into the CPU accumulation buffer.

        Parameters
        ----------
        positions : torch.Tensor
            CUDA leaf tensor whose gradient should be copied.
        grad_buffer : torch.Tensor
            CPU gradient buffer shaped ``[N, 2]``.
        start : int, optional
            Tile start offset for contiguous accumulation.
        node_ids : torch.Tensor, optional
            Explicit CPU node IDs for scatter accumulation.
        """
        if positions.grad is None:
            return
        grad_cpu = positions.grad.detach().cpu()
        if node_ids is not None:
            grad_buffer.index_add_(0, node_ids, grad_cpu)
        else:
            assert start is not None
            grad_buffer[start : start + grad_cpu.shape[0]].add_(grad_cpu)
        positions.grad = None

    def _reset_runtime_state(self) -> None:
        """Clear active edge state between tiled batches."""
        self.current_edge_index = None
        self.current_edge_ctx = None

    def compute_step(
        self,
        positions: torch.Tensor,
        loss_fns: list[LossFn],
        loss_weights: list[float],
    ) -> float:
        """Run one tiled forward/backward pass and accumulate CPU gradients.

        Parameters
        ----------
        positions : torch.Tensor
            CPU position tensor shaped ``[N, 2]`` with ``requires_grad=True``.
        loss_fns : list[LossFn]
            Tile-compatible loss functions.
        loss_weights : list[float]
            Scalar weights matching ``loss_fns``.

        Returns
        -------
        float
            Total weighted loss value for this optimization step.
        """
        if positions.device.type != "cpu":
            raise ValueError("TiledGPUCompute expects CPU positions")
        if len(loss_fns) != len(loss_weights):
            raise ValueError("loss_fns and loss_weights must have identical lengths")

        grad_buffer = torch.zeros_like(positions)
        total_loss = 0.0
        total_unweighted = 0.0
        edge_loss_mask = list(self.cross_tile_loss_mask or [False] * len(loss_fns))

        node_loss_fns = [
            loss_fn for loss_fn, is_edge in zip(loss_fns, edge_loss_mask) if not is_edge
        ]
        node_loss_weights = [
            weight for weight, is_edge in zip(loss_weights, edge_loss_mask) if not is_edge
        ]
        edge_loss_fns = [loss_fn for loss_fn, is_edge in zip(loss_fns, edge_loss_mask) if is_edge]
        edge_loss_weights = [
            weight for weight, is_edge in zip(loss_weights, edge_loss_mask) if is_edge
        ]

        for tile_edges, start, end in zip(self.tile_edges, self.tile_starts, self.tile_ends):
            tile_pos = positions[start:end].detach().to(self.device).requires_grad_(True)
            tile_sizes = self.node_sizes[start:end].to(self.device)
            tile_layer_index = self._tile_layer_index(start, end)

            self._reset_runtime_state()
            if node_loss_fns:
                node_scale = float(end - start) / float(max(self.num_nodes, 1))
                local_total, local_unweighted = self._backward_loss_group(
                    tile_pos,
                    tile_sizes,
                    tile_layer_index,
                    node_loss_fns,
                    node_loss_weights,
                    scale=node_scale,
                )
                total_loss += local_total
                total_unweighted += local_unweighted
                self._accumulate_leaf_grad(tile_pos, grad_buffer, start=start)

            if edge_loss_fns:
                for edge_batch_cpu in self._iter_edge_batches(tile_edges):
                    if edge_batch_cpu.numel() == 0:
                        continue
                    edge_scale = float(edge_batch_cpu.shape[1]) / float(
                        max(self.total_edge_count, 1)
                    )
                    edge_batch = edge_batch_cpu.to(self.device)
                    self.current_edge_index = edge_batch
                    self.current_edge_ctx = self._edge_context(tile_pos, edge_batch)
                    batch_total, batch_unweighted = self._backward_loss_group(
                        tile_pos,
                        tile_sizes,
                        tile_layer_index,
                        edge_loss_fns,
                        edge_loss_weights,
                        scale=edge_scale,
                    )
                    total_loss += batch_total
                    total_unweighted += batch_unweighted
                    self._accumulate_leaf_grad(tile_pos, grad_buffer, start=start)
                    self._reset_runtime_state()

            del tile_pos, tile_sizes, tile_layer_index
            torch.cuda.empty_cache()

        if edge_loss_fns and self.cross_edges.numel() > 0:
            for edge_batch_cpu in self._iter_edge_batches(self.cross_edges):
                if edge_batch_cpu.numel() == 0:
                    continue
                edge_scale = float(edge_batch_cpu.shape[1]) / float(max(self.total_edge_count, 1))
                active_nodes, inverse = torch.unique(
                    edge_batch_cpu.reshape(-1),
                    sorted=True,
                    return_inverse=True,
                )
                edge_batch = inverse.reshape(2, -1).to(self.device)
                cross_pos = positions[active_nodes].detach().to(self.device).requires_grad_(True)
                cross_sizes = self.node_sizes[active_nodes].to(self.device)
                self.current_edge_index = edge_batch
                self.current_edge_ctx = self._edge_context(cross_pos, edge_batch)
                batch_total, batch_unweighted = self._backward_loss_group(
                    cross_pos,
                    cross_sizes,
                    None,
                    edge_loss_fns,
                    edge_loss_weights,
                    scale=edge_scale,
                )
                total_loss += batch_total
                total_unweighted += batch_unweighted
                self._accumulate_leaf_grad(cross_pos, grad_buffer, node_ids=active_nodes)
                self._reset_runtime_state()
                del active_nodes, inverse, edge_batch, cross_pos, cross_sizes
                torch.cuda.empty_cache()

        positions.grad = grad_buffer
        self.last_unweighted_loss = total_unweighted
        self._reset_runtime_state()
        return total_loss

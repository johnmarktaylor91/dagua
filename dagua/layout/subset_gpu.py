"""Subset-resident GPU execution for very large layout solves.

The executor in this module keeps the full position tensor on CPU and gathers
only the nodes needed by one loss term onto the accelerator. This avoids
full-graph CUDA residency while preserving the existing loss-function
signatures used by the layout engine.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional, Sequence, Union

import torch

from dagua.layout.engine import EdgeBatchContext, SampledNodeContext
from dagua.layout.layers import LayerIndex

LossFn = Callable[[torch.Tensor, torch.Tensor, Optional[LayerIndex]], torch.Tensor]
ProgressCallback = Callable[[float], None]
_PROGRESS_LOG_INTERVAL_SECONDS = 30.0


@dataclass(frozen=True)
class PreparedSubsetData:
    """Remapped data required to evaluate one subset loss term.

    Parameters
    ----------
    global_indices : torch.Tensor
        Sorted unique global node IDs with shape ``[M]`` that will be gathered
        from the CPU-resident position tensor.
    local_edges : torch.Tensor, optional
        Edge tensor shaped ``[2, E]`` remapped into the local ``[0, M)``
        coordinate space.
    local_sampled_ctx : SampledNodeContext, optional
        Sampled-node context whose indices have been remapped into the same
        local coordinate space.
    """

    global_indices: torch.Tensor
    local_edges: Optional[torch.Tensor] = None
    local_sampled_ctx: Optional[SampledNodeContext] = None


@dataclass(frozen=True)
class GlobalAccessPattern:
    """Marker for loss terms that must execute against the full CPU graph."""


@dataclass(frozen=True)
class EdgeAccessPattern:
    """Access pattern for losses defined on an edge batch.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]`` containing global node IDs.
    """

    edge_index: torch.Tensor

    def get_indices(self) -> torch.Tensor:
        """Return the unique node IDs touched by the edge batch.

        Returns
        -------
        torch.Tensor
            Sorted unique node IDs shaped ``[M]``.
        """
        if self.edge_index.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=self.edge_index.device)
        return torch.unique(self.edge_index.reshape(-1))

    def remap_to_local(self, global_indices: torch.Tensor) -> PreparedSubsetData:
        """Remap the edge batch into a local node-index space.

        Parameters
        ----------
        global_indices : torch.Tensor
            Sorted unique node IDs shaped ``[M]``.

        Returns
        -------
        PreparedSubsetData
            Local edge tensor referencing ``global_indices``.
        """
        if global_indices.numel() == 0 or self.edge_index.numel() == 0:
            empty_edges = torch.zeros((2, 0), dtype=torch.long, device=global_indices.device)
            return PreparedSubsetData(global_indices=global_indices, local_edges=empty_edges)
        local_edges = torch.searchsorted(global_indices, self.edge_index)
        return PreparedSubsetData(global_indices=global_indices, local_edges=local_edges)


@dataclass(frozen=True)
class SampledAccessPattern:
    """Access pattern for sampled repulsion and overlap losses.

    Parameters
    ----------
    sampled_ctx : SampledNodeContext
        Sampled-node context with global node IDs.
    active_row_cap : int, optional
        Optional cap on the number of active rows to evaluate. When provided,
        the first ``active_row_cap`` rows of the sampled context are used.
    """

    sampled_ctx: SampledNodeContext
    active_row_cap: Optional[int] = None
    _cached_indices: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Cache the sampled union once for repeated subset-execution steps.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        active_idx, sampled = self._view()
        object.__setattr__(
            self,
            "_cached_indices",
            self._compute_cached_indices(active_idx, sampled),
        )

    def get_indices(self) -> torch.Tensor:
        """Return the union of active and sampled node IDs.

        Returns
        -------
        torch.Tensor
            Sorted unique node IDs shaped ``[M]``.
        """
        return self._cached_indices

    def remap_to_local(self, global_indices: torch.Tensor) -> PreparedSubsetData:
        """Remap the sampled-node context into local node coordinates.

        Parameters
        ----------
        global_indices : torch.Tensor
            Sorted unique node IDs shaped ``[M]``.

        Returns
        -------
        PreparedSubsetData
            Local sampled-node context referencing ``global_indices``.
        """
        active_idx, sampled = self._view()
        if global_indices.numel() == 0 or active_idx.numel() == 0:
            empty_idx = torch.zeros(0, dtype=torch.long, device=global_indices.device)
            empty_sampled = torch.zeros((0, 0), dtype=torch.long, device=global_indices.device)
            return PreparedSubsetData(
                global_indices=global_indices,
                local_sampled_ctx=SampledNodeContext(active_idx=empty_idx, sampled=empty_sampled),
            )

        local_active = torch.searchsorted(global_indices, active_idx)
        if sampled.numel() == 0:
            local_sampled = torch.zeros(
                (active_idx.shape[0], 0),
                dtype=torch.long,
                device=global_indices.device,
            )
        else:
            local_sampled = torch.searchsorted(global_indices, sampled.reshape(-1)).reshape(
                sampled.shape
            )
        return PreparedSubsetData(
            global_indices=global_indices,
            local_sampled_ctx=SampledNodeContext(
                active_idx=local_active,
                sampled=local_sampled,
            ),
        )

    def _view(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the capped sampled-node view for this pattern.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Active-node indices shaped ``[A]`` and sampled indices shaped
            ``[A, K]``.
        """
        if (
            self.active_row_cap is None
            or self.active_row_cap >= self.sampled_ctx.active_idx.shape[0]
        ):
            return self.sampled_ctx.active_idx, self.sampled_ctx.sampled
        return (
            self.sampled_ctx.active_idx[: self.active_row_cap],
            self.sampled_ctx.sampled[: self.active_row_cap],
        )

    def _compute_cached_indices(
        self,
        active_idx: torch.Tensor,
        sampled: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the cached global node IDs touched by this sampled pattern.

        Parameters
        ----------
        active_idx : torch.Tensor
            Active-node indices shaped ``[A]``.
        sampled : torch.Tensor
            Sampled-node indices shaped ``[A, K]``.

        Returns
        -------
        torch.Tensor
            Sorted unique node IDs shaped ``[M]``.
        """
        if active_idx.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=active_idx.device)
        if sampled.numel() == 0:
            return torch.unique(active_idx)
        return torch.unique(torch.cat([active_idx, sampled.reshape(-1)]))


@dataclass(frozen=True)
class UnionAccessPattern:
    """Union of multiple subset access patterns.

    Parameters
    ----------
    patterns : Sequence[EdgeAccessPattern | SampledAccessPattern]
        Access patterns that should share one gathered local node subset.
    """

    patterns: Sequence[Union[EdgeAccessPattern, SampledAccessPattern]]

    def get_indices(self) -> torch.Tensor:
        """Return the sorted union of all child pattern indices.

        Returns
        -------
        torch.Tensor
            Sorted unique node IDs shaped ``[M]``.
        """
        parts = [
            pattern.get_indices() for pattern in self.patterns if pattern.get_indices().numel() > 0
        ]
        if not parts:
            return torch.zeros(0, dtype=torch.long)
        if len(parts) == 1:
            return parts[0]
        return torch.unique(torch.cat(parts))

    def remap_to_local(self, global_indices: torch.Tensor) -> PreparedSubsetData:
        """Remap all child patterns into one shared local coordinate space.

        Parameters
        ----------
        global_indices : torch.Tensor
            Sorted unique node IDs shaped ``[M]``.

        Returns
        -------
        PreparedSubsetData
            Combined local edge and sampled-node data.
        """
        local_edges: Optional[torch.Tensor] = None
        local_sampled_ctx: Optional[SampledNodeContext] = None
        for pattern in self.patterns:
            prepared = pattern.remap_to_local(global_indices)
            if prepared.local_edges is not None:
                local_edges = prepared.local_edges
            if prepared.local_sampled_ctx is not None:
                local_sampled_ctx = prepared.local_sampled_ctx
        return PreparedSubsetData(
            global_indices=global_indices,
            local_edges=local_edges,
            local_sampled_ctx=local_sampled_ctx,
        )


@dataclass(frozen=True)
class SubsetGPULossTerm:
    """Loss term metadata for subset-GPU execution.

    Parameters
    ----------
    name : str
        Human-readable loss name used for debugging and tests.
    weight : float
        Current weighted multiplier for the term.
    loss_fn : LossFn
        Loss callable that preserves the existing engine signature.
    access_pattern : GlobalAccessPattern | EdgeAccessPattern | SampledAccessPattern \
            | UnionAccessPattern
        Description of which node subset the loss needs.
    """

    name: str
    weight: float
    loss_fn: LossFn
    access_pattern: Union[
        GlobalAccessPattern, EdgeAccessPattern, SampledAccessPattern, UnionAccessPattern
    ]


class SubsetGPUExecutor:
    """Execute individual loss terms on gathered subsets and scatter gradients.

    Parameters
    ----------
    node_sizes : torch.Tensor
        CPU-resident node-size tensor shaped ``[N, 2]`` aligned with the full
        position tensor.
    layer_index : LayerIndex, optional
        CPU-resident layer structure for global loss execution.
    execution_device : str | torch.device
        Device used for subset acceleration. CUDA is typical, but CPU is useful
        for deterministic tests and fallback execution.
    batch_edges_ref : list[torch.Tensor | None]
        Mutable engine edge reference. The executor temporarily swaps in local
        remapped edges so the existing loss lambdas remain unchanged.
    edge_ctx_ref : list[EdgeBatchContext | None]
        Mutable engine edge-context reference paired with ``batch_edges_ref``.
    sampled_ctx_ref : list[SampledNodeContext | None]
        Mutable engine sampled-context reference. Local remapped sampled
        indices are installed during sampled subset evaluation.
    verbose : bool, default=False
        Whether timed subset progress logs should be emitted.
    """

    def __init__(
        self,
        node_sizes: torch.Tensor,
        layer_index: Optional[LayerIndex],
        execution_device: Union[str, torch.device],
        batch_edges_ref: list[Optional[torch.Tensor]],
        edge_ctx_ref: list[Optional[EdgeBatchContext]],
        sampled_ctx_ref: list[Optional[SampledNodeContext]],
        verbose: bool = False,
    ) -> None:
        self.node_sizes = node_sizes
        self.layer_index = layer_index
        self.execution_device = torch.device(execution_device)
        self.batch_edges_ref = batch_edges_ref
        self.edge_ctx_ref = edge_ctx_ref
        self.sampled_ctx_ref = sampled_ctx_ref
        self.verbose = verbose
        self.last_total_loss = 0.0
        self.last_unweighted_loss = 0.0
        self._grad_buffer: Optional[torch.Tensor] = None

    def compute_step(
        self,
        pos: torch.Tensor,
        loss_terms: Sequence[SubsetGPULossTerm],
        verbose: Optional[bool] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> torch.Tensor:
        """Compute one optimizer-step gradient buffer.

        Parameters
        ----------
        pos : torch.Tensor
            CPU-resident position tensor shaped ``[N, 2]`` with
            ``requires_grad=True``.
        loss_terms : Sequence[SubsetGPULossTerm]
            Weighted loss terms to evaluate for the current optimizer step.
        verbose : bool, optional
            Override for timed subset progress logging. When omitted, the
            executor-level setting is used.
        progress_callback : Callable[[float], None], optional
            Callback invoked when a timed progress heartbeat fires. The engine
            uses this to refresh ``progress.json`` during long subset steps.

        Returns
        -------
        torch.Tensor
            Accumulated gradient buffer shaped ``[N, 2]`` on the same device as
            ``pos``.
        """
        grad_buffer = self._prepare_grad_buffer(pos)
        self.last_total_loss = 0.0
        self.last_unweighted_loss = 0.0
        active_verbose = self.verbose if verbose is None else verbose
        active_terms = sum(1 for term in loss_terms if term.weight > 0.0)
        step_start = time.perf_counter()
        next_progress_time = step_start + _PROGRESS_LOG_INTERVAL_SECONDS
        completed_terms = 0

        for term in loss_terms:
            if term.weight <= 0.0:
                continue
            if isinstance(term.access_pattern, GlobalAccessPattern):
                self._accumulate_global_grad(term, pos, grad_buffer)
            else:
                self._accumulate_subset_grad(term, pos, grad_buffer)
            completed_terms += 1
            now = time.perf_counter()
            if now < next_progress_time:
                continue
            elapsed = now - step_start
            if active_verbose:
                print(
                    f"[dagua]       subset_gpu: {completed_terms}/{active_terms} terms, "
                    f"loss={self.last_unweighted_loss:.1f} ({elapsed:.1f}s elapsed)",
                    flush=True,
                )
            if progress_callback is not None:
                progress_callback(self.last_unweighted_loss)
            next_progress_time = now + _PROGRESS_LOG_INTERVAL_SECONDS

        return grad_buffer

    def _prepare_grad_buffer(self, pos: torch.Tensor) -> torch.Tensor:
        """Return a reusable zeroed gradient buffer aligned with ``pos``.

        Parameters
        ----------
        pos : torch.Tensor
            Full position tensor shaped ``[N, 2]``.

        Returns
        -------
        torch.Tensor
            Zeroed gradient buffer with the same shape, dtype, and device as
            ``pos``.
        """
        needs_realloc = (
            self._grad_buffer is None
            or self._grad_buffer.shape != pos.shape
            or self._grad_buffer.dtype != pos.dtype
            or self._grad_buffer.device != pos.device
        )
        if needs_realloc:
            grad_buffer = torch.zeros_like(pos)
            self._grad_buffer = grad_buffer
        else:
            existing_grad_buffer = self._grad_buffer
            assert existing_grad_buffer is not None
            existing_grad_buffer.zero_()
            grad_buffer = existing_grad_buffer
        return grad_buffer

    def _accumulate_global_grad(
        self,
        term: SubsetGPULossTerm,
        pos: torch.Tensor,
        grad_buffer: torch.Tensor,
    ) -> None:
        """Evaluate a global CPU loss and accumulate its gradient.

        Parameters
        ----------
        term : SubsetGPULossTerm
            Weighted loss term to evaluate.
        pos : torch.Tensor
            CPU-resident position tensor shaped ``[N, 2]``.
        grad_buffer : torch.Tensor
            Gradient accumulator shaped ``[N, 2]``.
        """
        loss = term.weight * term.loss_fn(pos, self.node_sizes, self.layer_index)
        self._record_loss(term.weight, loss)
        grad = self._loss_grad(loss, pos)
        if grad is not None:
            grad_buffer.add_(grad)

    def _accumulate_subset_grad(
        self,
        term: SubsetGPULossTerm,
        pos: torch.Tensor,
        grad_buffer: torch.Tensor,
    ) -> None:
        """Evaluate a subset loss on the execution device and scatter its gradient.

        Parameters
        ----------
        term : SubsetGPULossTerm
            Weighted loss term to evaluate.
        pos : torch.Tensor
            CPU-resident position tensor shaped ``[N, 2]``.
        grad_buffer : torch.Tensor
            Gradient accumulator shaped ``[N, 2]``.
        """
        access_pattern = term.access_pattern
        assert not isinstance(access_pattern, GlobalAccessPattern)
        global_indices = access_pattern.get_indices()
        if global_indices.numel() == 0:
            return

        prepared = access_pattern.remap_to_local(global_indices)
        pos_local = (
            pos.index_select(0, prepared.global_indices)
            .to(self.execution_device)
            .detach()
            .requires_grad_(True)
        )
        node_sizes_local = self.node_sizes.index_select(0, prepared.global_indices).to(
            self.execution_device
        )

        with self._local_refs(prepared, pos_local):
            loss = term.weight * term.loss_fn(pos_local, node_sizes_local, None)
        self._record_loss(term.weight, loss)
        grad_local = self._loss_grad(loss, pos_local)
        if grad_local is None:
            return

        scatter_index = prepared.global_indices.to(device=grad_buffer.device)
        grad_buffer.index_add_(0, scatter_index, grad_local.to(device=grad_buffer.device))

    def _record_loss(self, weight: float, loss: torch.Tensor) -> None:
        """Update the last-step scalar totals from one evaluated loss.

        Parameters
        ----------
        weight : float
            Current weighted multiplier for the term.
        loss : torch.Tensor
            Scalar weighted loss tensor.
        """
        value = float(loss.item())
        self.last_total_loss += value
        self.last_unweighted_loss += value / weight if weight else 0.0

    def _loss_grad(
        self,
        loss: torch.Tensor,
        variable: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return ``d(loss)/d(variable)`` when the loss is connected.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss tensor.
        variable : torch.Tensor
            Tensor to differentiate with respect to.

        Returns
        -------
        torch.Tensor | None
            Gradient tensor with the same shape as ``variable`` when the loss
            depends on it, otherwise ``None``.
        """
        if not loss.requires_grad:
            return None
        grad = torch.autograd.grad(
            loss,
            variable,
            allow_unused=True,
        )[0]
        return grad

    @contextmanager
    def _local_refs(
        self,
        prepared: PreparedSubsetData,
        pos_local: torch.Tensor,
    ) -> Iterator[None]:
        """Install remapped local refs for one subset loss evaluation.

        Parameters
        ----------
        prepared : PreparedSubsetData
            Local remapped indices and per-loss data.
        pos_local : torch.Tensor
            Gathered local position tensor shaped ``[M, 2]``.

        Yields
        ------
        None
            Context where the engine's mutable refs point at local tensors.
        """
        prev_edges = self.batch_edges_ref[0]
        prev_edge_ctx = self.edge_ctx_ref[0]
        prev_sampled_ctx = self.sampled_ctx_ref[0]
        try:
            if prepared.local_edges is not None:
                local_edges = prepared.local_edges.to(device=self.execution_device)
                self.batch_edges_ref[0] = local_edges
                self.edge_ctx_ref[0] = _build_local_edge_context(pos_local, local_edges)
            else:
                self.batch_edges_ref[0] = prev_edges
                self.edge_ctx_ref[0] = prev_edge_ctx

            if prepared.local_sampled_ctx is not None:
                self.sampled_ctx_ref[0] = SampledNodeContext(
                    active_idx=prepared.local_sampled_ctx.active_idx.to(self.execution_device),
                    sampled=prepared.local_sampled_ctx.sampled.to(self.execution_device),
                )
            else:
                self.sampled_ctx_ref[0] = prev_sampled_ctx
            yield
        finally:
            self.batch_edges_ref[0] = prev_edges
            self.edge_ctx_ref[0] = prev_edge_ctx
            self.sampled_ctx_ref[0] = prev_sampled_ctx


def _build_local_edge_context(pos: torch.Tensor, edge_index: torch.Tensor) -> EdgeBatchContext:
    """Build an edge-batch context for a remapped local edge tensor.

    Parameters
    ----------
    pos : torch.Tensor
        Local position tensor shaped ``[M, 2]``.
    edge_index : torch.Tensor
        Local edge tensor shaped ``[2, E]``.

    Returns
    -------
    EdgeBatchContext
        Precomputed edge deltas and squared distances for ``edge_index``.
    """
    if edge_index.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=pos.device)
        empty_float = torch.zeros(0, dtype=pos.dtype, device=pos.device)
        return EdgeBatchContext(
            src=empty,
            tgt=empty,
            dx=empty_float,
            dy=empty_float,
            dist_sq=empty_float,
        )
    src = edge_index[0]
    tgt = edge_index[1]
    src_pos = pos[src]
    tgt_pos = pos[tgt]
    dx = src_pos[:, 0] - tgt_pos[:, 0]
    dy = src_pos[:, 1] - tgt_pos[:, 1]
    dist_sq = dx.square() + dy.square()
    return EdgeBatchContext(src=src, tgt=tgt, dx=dx, dy=dy, dist_sq=dist_sq)

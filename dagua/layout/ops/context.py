"""Shared per-step context operations for layout pipelines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

_MIN_DISTANCE = 1.0e-12
_FINE_REPULSION_SCALE = 1.0e-4


class _DensityGrid:
    """Density proxy used by the DrL energy function."""

    def __init__(self, grid_size: int, view_size: float, radius: int) -> None:
        """Initialize the density grid and its tent kernel."""
        self.grid_size = grid_size
        self.view_size = view_size
        self.radius = radius
        self.cell_width = view_size / float(grid_size)
        self.origin = -0.5 * view_size
        self.density = torch.zeros((grid_size, grid_size), dtype=torch.float64)
        self.node_cells: dict[int, tuple[int, int]] = {}
        self.buckets: dict[tuple[int, int], set[int]] = {}

        axis = torch.arange(-radius, radius + 1, dtype=torch.float64)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        distance = torch.sqrt(xx.square() + yy.square())
        self.kernel = torch.clamp(1.0 - (distance / float(radius)), min=0.0)

    def _cell_index(self, position: torch.Tensor) -> tuple[int, int]:
        """Convert a coordinate to a clamped integer grid cell."""
        x_value = float(position[0].item())
        y_value = float(position[1].item())
        cell_x = int(math.floor((x_value - self.origin) / self.cell_width))
        cell_y = int(math.floor((y_value - self.origin) / self.cell_width))
        return (
            max(0, min(self.grid_size - 1, cell_x)),
            max(0, min(self.grid_size - 1, cell_y)),
        )

    def _apply_kernel(self, cell_x: int, cell_y: int, sign: float) -> None:
        """Add or subtract one tent kernel at the given cell location."""
        x_start = max(0, cell_x - self.radius)
        x_end = min(self.grid_size, cell_x + self.radius + 1)
        y_start = max(0, cell_y - self.radius)
        y_end = min(self.grid_size, cell_y + self.radius + 1)

        kernel_x_start = x_start - (cell_x - self.radius)
        kernel_x_end = kernel_x_start + (x_end - x_start)
        kernel_y_start = y_start - (cell_y - self.radius)
        kernel_y_end = kernel_y_start + (y_end - y_start)

        self.density[y_start:y_end, x_start:x_end] += (
            sign * self.kernel[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end]
        )

    def add_node(self, node: int, position: torch.Tensor) -> None:
        """Insert a node into the coarse grid and fine buckets."""
        cell = self._cell_index(position)
        self.node_cells[node] = cell
        self._apply_kernel(cell[0], cell[1], sign=1.0)
        self.buckets.setdefault(cell, set()).add(node)

    def remove_node(self, node: int) -> None:
        """Remove a node from the coarse grid and fine buckets."""
        cell = self.node_cells.pop(node, None)
        if cell is None:
            return
        self._apply_kernel(cell[0], cell[1], sign=-1.0)
        bucket = self.buckets.get(cell)
        if bucket is None:
            return
        bucket.discard(node)
        if not bucket:
            del self.buckets[cell]

    def coarse_density(self, position: torch.Tensor) -> float:
        """Return the coarse density penalty at one position."""
        cell_x, cell_y = self._cell_index(position)
        value = float(self.density[cell_y, cell_x].item())
        return value * value

    def fine_density(self, node: int, position: torch.Tensor, positions: torch.Tensor) -> float:
        """Return the exact simmer-stage local repulsion penalty."""
        cell_x, cell_y = self._cell_index(position)
        density = 0.0
        for offset_y in (-1, 0, 1):
            for offset_x in (-1, 0, 1):
                neighbor_cell = (cell_x + offset_x, cell_y + offset_y)
                bucket = self.buckets.get(neighbor_cell)
                if not bucket:
                    continue
                for other in bucket:
                    if other == node:
                        continue
                    delta = position - positions[other]
                    distance_sq = float(delta.dot(delta).item()) + _MIN_DISTANCE
                    density += _FINE_REPULSION_SCALE / distance_sq
        return density


def _kdtree_repulsion_pairs(pos: torch.Tensor, query_radius: float) -> np.ndarray:
    """Find nearby node pairs using SciPy's cKDTree."""
    from scipy.spatial import cKDTree

    if pos.shape[0] < 2:
        return np.empty((0, 2), dtype=np.int64)
    tree = cKDTree(pos.detach().cpu().numpy())
    pairs = tree.query_pairs(query_radius, output_type="ndarray")
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return pairs.astype(np.int64)


from dagua.layout.engine import (  # noqa: E402
    SAMPLED_SAME_LAYER_K,
    EdgeBatchContext,
    SampledNodeContext,
    _sampled_node_context_sizes,
)
from dagua.layout.layers import LayerIndex, build_layer_index  # noqa: E402
from dagua.layout.ops.base import Op  # noqa: E402
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState  # noqa: E402
from dagua.layout.ops.taxonomy import OpCategory, register_op  # noqa: E402

_DENSITY_GRID_RADIUS = 10


@dataclass(frozen=True)
class BuildEdgeBatchCtxConfig:
    """Configuration for :class:`BuildEdgeBatchCtx`.

    Parameters
    ----------
    batch_size : int, default=0
        Number of edges per batch. ``0`` means the full edge set.
    """

    batch_size: int = 0


@dataclass(frozen=True)
class RefreshSampledNodeCtxConfig:
    """Configuration for :class:`RefreshSampledNodeCtx`.

    Parameters
    ----------
    interval : int, default=5
        Refresh cadence in optimization steps.
    active_cap : int, default=0
        Optional cap for the active sampled-node set. ``0`` keeps the default
        engine heuristic.
    """

    interval: int = 5
    active_cap: int = 0


@dataclass(frozen=True)
class BuildQuadTreeConfig:
    """Configuration for :class:`BuildQuadTree`.

    Parameters
    ----------
    max_depth : int, default=10
        Maximum recursive subdivision depth.
    """

    max_depth: int = 10


@dataclass(frozen=True)
class BuildDensityGridConfig:
    """Configuration for :class:`BuildDensityGrid`.

    Parameters
    ----------
    grid_size : int, default=1000
        Number of cells along each grid axis.
    view_size : float, default=4000
        Width of the square view window tracked by the density grid.
    """

    grid_size: int = 1000
    view_size: float = 4000.0


@dataclass(frozen=True)
class RefreshKDTreePairsConfig:
    """Configuration for :class:`RefreshKDTreePairs`.

    Parameters
    ----------
    radius : float, default=0.4
        Query radius passed to SciPy's ``cKDTree.query_pairs``.
    interval : int, default=5
        Refresh cadence in optimization steps.
    """

    radius: float = 0.4
    interval: int = 5


@dataclass(frozen=True, slots=True)
class QuadTreeNode:
    """Barnes-Hut quadtree node used by :class:`BuildQuadTree`.

    Parameters
    ----------
    mass_center_x : float
        X coordinate of the cell center of mass.
    mass_center_y : float
        Y coordinate of the cell center of mass.
    mass : float
        Total mass stored in the cell.
    size : float
        Region diameter used by the Barnes-Hut opening test.
    depth : int
        Depth of the node in the quadtree.
    children : tuple[QuadTreeNode, ...] or None
        Child quadrants for internal nodes.
    indices : numpy.ndarray or None
        Particle indices for leaf nodes.
    """

    mass_center_x: float
    mass_center_y: float
    mass: float
    size: float
    depth: int
    children: Optional[tuple["QuadTreeNode", ...]]
    indices: Optional[np.ndarray]


def _validate_positions(state: SolveState) -> torch.Tensor:
    """Return a validated position tensor.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If positions are missing or malformed.
    """
    if state.pos is None:
        raise ValueError("state.pos must be populated before this op runs")
    if state.pos.ndim != 2 or state.pos.shape[1] != 2:
        raise ValueError("state.pos must have shape [N, 2]")
    return state.pos


def _empty_edge_batch_context(device: torch.device, dtype: torch.dtype) -> EdgeBatchContext:
    """Return an empty edge-batch context on the requested device.

    Parameters
    ----------
    device : torch.device
        Device for the returned tensors.
    dtype : torch.dtype
        Floating-point dtype for the distance terms.

    Returns
    -------
    EdgeBatchContext
        Empty edge context with consistent tensor dtypes.
    """
    empty_idx = torch.zeros(0, dtype=torch.long, device=device)
    empty_float = torch.zeros(0, dtype=dtype, device=device)
    return EdgeBatchContext(
        src=empty_idx,
        tgt=empty_idx,
        dx=empty_float,
        dy=empty_float,
        dist_sq=empty_float,
    )


def _select_edge_batch(
    edge_index: torch.Tensor,
    batch_size: int,
    step: int,
    device: torch.device,
) -> torch.Tensor:
    """Select a deterministic contiguous edge batch.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    batch_size : int
        Requested batch size. ``0`` selects the full edge set.
    step : int
        Current optimization step used for chunk rotation.
    device : torch.device
        Device for the returned edge tensor.

    Returns
    -------
    torch.Tensor
        Selected edge tensor with shape ``[2, E_batch]``.

    Raises
    ------
    ValueError
        If ``edge_index`` is malformed or ``batch_size`` is negative.
    """
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("problem.edge_index must have shape [2, E]")

    num_edges = edge_index.shape[1]
    if num_edges == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    edge_index_device = edge_index.detach().to(device=device, dtype=torch.long)
    if batch_size == 0 or batch_size >= num_edges:
        return edge_index_device

    start = (max(step, 0) * batch_size) % num_edges
    end = start + batch_size
    if end <= num_edges:
        return edge_index_device[:, start:end]

    wrap = end - num_edges
    return torch.cat([edge_index_device[:, start:], edge_index_device[:, :wrap]], dim=1)


def _resolve_layer_index(state: SolveState) -> LayerIndex:
    """Return an existing or derived layer index.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    LayerIndex
        Layer index describing the current node set.

    Raises
    ------
    ValueError
        If neither ``state.layer_index`` nor ``state.layers`` is available.
    """
    if state.layer_index is not None:
        return state.layer_index
    if state.layers is None:
        raise ValueError("state.layer_index or state.layers must be populated before sampling")
    return build_layer_index(layer_assignments=state.layers, device="cpu", enable_cuda_sort=False)


def _cpu_generator(
    problem: LayoutProblem,
    state: SolveState,
    ctx: RuntimeContext,
) -> torch.Generator:
    """Return the CPU RNG generator used by sampled-node refresh.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs supplying the fallback seed.
    state : SolveState
        Mutable solve state supplying the current step.
    ctx : RuntimeContext
        Execution infrastructure which may already own a generator.

    Returns
    -------
    torch.Generator
        CPU generator used for ``torch.randint`` and ``torch.rand`` calls.
    """
    if ctx.generator is not None:
        return ctx.generator
    generator = torch.Generator(device="cpu")
    generator.manual_seed(problem.seed + max(state.step, 0))
    return generator


def _build_sampled_node_context(
    num_nodes: int,
    layer_index: LayerIndex,
    device: torch.device,
    n_active: int,
    generator: torch.Generator,
) -> SampledNodeContext:
    """Build sampled-node state using the engine's layer-local sampling layout.

    Parameters
    ----------
    num_nodes : int
        Number of nodes represented by ``layer_index``.
    layer_index : LayerIndex
        Layer index for the current node set.
    device : torch.device
        Target device for the returned tensors.
    n_active : int
        Number of active nodes to sample.
    generator : torch.Generator
        CPU RNG generator used for sampling draws.

    Returns
    -------
    SampledNodeContext
        Active nodes and their same-layer plus adjacent-layer samples.

    Notes
    -----
    Randomness uses ``torch.randint`` followed by two ``torch.rand`` draws on
    the provided CPU generator, then copies the result to ``device``.
    """
    _, n_random, k_same = _sampled_node_context_sizes(
        num_nodes,
        SAMPLED_SAME_LAYER_K,
        n_active_override=n_active,
    )
    if n_active <= 0 or num_nodes == 0:
        empty_idx = torch.zeros(0, dtype=torch.long, device=device)
        empty_sampled = torch.zeros((0, 0), dtype=torch.long, device=device)
        return SampledNodeContext(active_idx=empty_idx, sampled=empty_sampled)

    layers = layer_index.node_to_layer.detach().to(device="cpu", dtype=torch.long)
    offsets = layer_index.layer_offsets.detach().to(device="cpu", dtype=torch.long)
    sorted_nodes = layer_index.sorted_nodes.detach().to(device="cpu", dtype=torch.long)

    active_idx_cpu = torch.randint(
        0,
        num_nodes,
        (n_active,),
        device="cpu",
        generator=generator,
    )
    active_layers = layers[active_idx_cpu]

    same_start = offsets[active_layers]
    same_end = offsets[active_layers + 1]
    same_range = (same_end - same_start).to(dtype=torch.float32)
    same_rand = torch.rand((n_active, k_same), device="cpu", generator=generator)
    same_indices = same_start.unsqueeze(1) + (same_rand * same_range.unsqueeze(1)).long()
    same_indices = same_indices.clamp(min=0, max=max(num_nodes - 1, 0))
    same_sampled = sorted_nodes[same_indices]

    adj_lo = (active_layers - 1).clamp(min=0)
    adj_hi = (active_layers + 2).clamp(max=layer_index.num_layers)
    adj_start = offsets[adj_lo]
    adj_end = offsets[adj_hi]
    adj_range = (adj_end - adj_start).to(dtype=torch.float32)
    adj_rand = torch.rand((n_active, n_random), device="cpu", generator=generator)
    adj_indices = adj_start.unsqueeze(1) + (adj_rand * adj_range.unsqueeze(1)).long()
    adj_indices = adj_indices.clamp(min=0, max=max(num_nodes - 1, 0))
    random_sampled = sorted_nodes[adj_indices]

    sampled_cpu = torch.cat([same_sampled, random_sampled], dim=1)
    return SampledNodeContext(
        active_idx=active_idx_cpu.to(device=device),
        sampled=sampled_cpu.to(device=device),
    )


def _mass_array(state: SolveState, num_nodes: int) -> np.ndarray:
    """Return per-node masses for Barnes-Hut tree construction.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.
    num_nodes : int
        Number of positioned nodes.

    Returns
    -------
    numpy.ndarray
        Mass array with shape ``[N]``.

    Raises
    ------
    ValueError
        If ``state.degree`` is present with the wrong length.
    """
    if state.degree is None:
        return np.ones(num_nodes, dtype=np.float64)

    degree = state.degree.detach().to(device="cpu", dtype=torch.float64)
    if degree.ndim != 1 or degree.shape[0] != num_nodes:
        raise ValueError("state.degree must have shape [N] when provided")
    return degree.numpy() + 1.0


def _leaf_node(
    pos_np: np.ndarray,
    mass_np: np.ndarray,
    indices: np.ndarray,
    depth: int,
) -> QuadTreeNode:
    """Build a quadtree leaf containing one or more particles.

    Parameters
    ----------
    pos_np : numpy.ndarray
        Position array with shape ``[N, 2]``.
    mass_np : numpy.ndarray
        Mass array with shape ``[N]``.
    indices : numpy.ndarray
        Indices stored in the leaf.
    depth : int
        Depth of the leaf.

    Returns
    -------
    QuadTreeNode
        Leaf node with summary mass statistics and retained indices.
    """
    leaf_mass = float(mass_np[indices].sum())
    if leaf_mass > 0.0:
        center_x = float((pos_np[indices, 0] * mass_np[indices]).sum() / leaf_mass)
        center_y = float((pos_np[indices, 1] * mass_np[indices]).sum() / leaf_mass)
    else:
        center_x = float(pos_np[indices, 0].mean())
        center_y = float(pos_np[indices, 1].mean())
    distance = np.sqrt(
        ((pos_np[indices, 0] - center_x) ** 2) + ((pos_np[indices, 1] - center_y) ** 2)
    )
    size = float(2.0 * distance.max()) if distance.size > 0 else 0.0
    return QuadTreeNode(
        mass_center_x=center_x,
        mass_center_y=center_y,
        mass=leaf_mass,
        size=size,
        depth=depth,
        children=None,
        indices=indices.copy(),
    )


def _build_quadtree_node(
    pos_np: np.ndarray,
    mass_np: np.ndarray,
    indices: np.ndarray,
    depth: int,
    max_depth: int,
) -> Optional[QuadTreeNode]:
    """Recursively build one Barnes-Hut quadtree node.

    Parameters
    ----------
    pos_np : numpy.ndarray
        Position array with shape ``[N, 2]``.
    mass_np : numpy.ndarray
        Mass array with shape ``[N]``.
    indices : numpy.ndarray
        Particle indices assigned to the current region.
    depth : int
        Current recursion depth.
    max_depth : int
        Maximum allowed recursion depth.

    Returns
    -------
    QuadTreeNode or None
        Built node, or ``None`` for an empty region.
    """
    if indices.size == 0:
        return None
    if indices.size == 1 or depth >= max_depth:
        return _leaf_node(pos_np=pos_np, mass_np=mass_np, indices=indices, depth=depth)

    cell_mass = float(mass_np[indices].sum())
    if cell_mass > 0.0:
        center_x = float((pos_np[indices, 0] * mass_np[indices]).sum() / cell_mass)
        center_y = float((pos_np[indices, 1] * mass_np[indices]).sum() / cell_mass)
    else:
        center_x = float(pos_np[indices, 0].mean())
        center_y = float(pos_np[indices, 1].mean())

    x = pos_np[indices, 0]
    y = pos_np[indices, 1]
    distance = np.sqrt(((x - center_x) ** 2) + ((y - center_y) ** 2))
    size = float(2.0 * distance.max()) if distance.size > 0 else 0.0

    quadrant_masks = (
        (x < center_x) & (y >= center_y),
        (x < center_x) & (y < center_y),
        (x >= center_x) & (y >= center_y),
        (x >= center_x) & (y < center_y),
    )

    children: list[QuadTreeNode] = []
    for mask in quadrant_masks:
        child_indices = indices[mask]
        if child_indices.size == 0:
            continue
        if child_indices.size == indices.size:
            if depth + 1 > max_depth:
                return _leaf_node(pos_np=pos_np, mass_np=mass_np, indices=indices, depth=depth)
            for child_index in child_indices:
                children.append(
                    _leaf_node(
                        pos_np=pos_np,
                        mass_np=mass_np,
                        indices=np.asarray([child_index], dtype=np.int64),
                        depth=depth + 1,
                    )
                )
            continue

        child = _build_quadtree_node(
            pos_np=pos_np,
            mass_np=mass_np,
            indices=child_indices,
            depth=depth + 1,
            max_depth=max_depth,
        )
        if child is not None:
            children.append(child)

    return QuadTreeNode(
        mass_center_x=center_x,
        mass_center_y=center_y,
        mass=cell_mass,
        size=size,
        depth=depth,
        children=tuple(children),
        indices=None,
    )


@register_op
class BuildEdgeBatchCtx(Op):
    """Precompute edge deltas for the current edge batch."""

    name = "build_edge_batch_ctx"
    category = OpCategory.CONTEXT
    reads = ("pos",)
    writes = ("edge_batch_context",)
    requires = ("pos",)
    access_pattern = "edge"

    def __init__(self, config: Optional[BuildEdgeBatchCtxConfig] = None) -> None:
        """Initialize the op with an optional config.

        Parameters
        ----------
        config : BuildEdgeBatchCtxConfig, optional
            Edge-batching settings. Defaults to the standard config.
        """
        self.config = config or BuildEdgeBatchCtxConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build the shared edge-batch context for edge-based losses.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing current positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            Updated state with ``edge_batch_context`` populated.
        """
        del ctx

        pos = _validate_positions(state)
        batch_edges = _select_edge_batch(
            edge_index=problem.edge_index,
            batch_size=self.config.batch_size,
            step=state.step,
            device=pos.device,
        )
        if batch_edges.numel() == 0:
            state.edge_batch_context = _empty_edge_batch_context(device=pos.device, dtype=pos.dtype)
            return state

        keep = batch_edges[0] != batch_edges[1]
        batch_edges = batch_edges[:, keep]
        if batch_edges.numel() == 0:
            state.edge_batch_context = _empty_edge_batch_context(device=pos.device, dtype=pos.dtype)
            return state

        if int(batch_edges.max().item()) >= pos.shape[0]:
            raise ValueError("problem.edge_index references a node outside state.pos")

        src = batch_edges[0]
        tgt = batch_edges[1]
        src_pos = pos[src]
        tgt_pos = pos[tgt]
        dx = src_pos[:, 0] - tgt_pos[:, 0]
        dy = src_pos[:, 1] - tgt_pos[:, 1]
        dist_sq = dx.square() + dy.square()
        state.edge_batch_context = EdgeBatchContext(
            src=src,
            tgt=tgt,
            dx=dx,
            dy=dy,
            dist_sq=dist_sq,
        )
        return state


@register_op
class RefreshSampledNodeCtx(Op):
    """Refresh the shared sampled-node context on a fixed cadence."""

    name = "refresh_sampled_node_ctx"
    category = OpCategory.CONTEXT
    reads = ("layer_index", "layers", "pos", "step")
    writes = ("sampled_node_context",)
    requires = ("step",)
    access_pattern = "sampled"

    def __init__(self, config: Optional[RefreshSampledNodeCtxConfig] = None) -> None:
        """Initialize the op with an optional config.

        Parameters
        ----------
        config : RefreshSampledNodeCtxConfig, optional
            Sample refresh settings. Defaults to the standard config.
        """
        self.config = config or RefreshSampledNodeCtxConfig()
        if self.config.interval <= 0:
            raise ValueError("interval must be positive")
        if self.config.active_cap < 0:
            raise ValueError("active_cap must be non-negative")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Refresh sampled-node state when the cadence requires it.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs used for fallback RNG seeding.
        state : SolveState
            Mutable solve state with current layers or a prebuilt layer index.
        ctx : RuntimeContext
            Execution infrastructure providing the optional RNG generator.

        Returns
        -------
        SolveState
            Updated state with ``sampled_node_context`` populated or retained.
        """
        if state.sampled_node_context is not None and state.step % self.config.interval != 0:
            return state

        layer_index = _resolve_layer_index(state)
        num_nodes = int(layer_index.node_to_layer.shape[0])
        if state.pos is not None:
            sampled_device = state.pos.device
        else:
            sampled_device = layer_index.node_to_layer.device
        n_active_override = self.config.active_cap if self.config.active_cap > 0 else None
        n_active, _, _ = _sampled_node_context_sizes(
            num_nodes,
            SAMPLED_SAME_LAYER_K,
            n_active_override=n_active_override,
        )
        state.sampled_node_context = _build_sampled_node_context(
            num_nodes=num_nodes,
            layer_index=layer_index,
            device=sampled_device,
            n_active=n_active,
            generator=_cpu_generator(problem, state, ctx),
        )
        return state


@register_op
class BuildQuadTree(Op):
    """Build a Barnes-Hut quadtree over the current positions."""

    name = "build_quad_tree"
    category = OpCategory.CONTEXT
    reads = ("pos", "degree")
    writes = ("extras.quadtree",)
    requires = ("pos",)
    access_pattern = "global"

    def __init__(self, config: Optional[BuildQuadTreeConfig] = None) -> None:
        """Initialize the op with an optional config.

        Parameters
        ----------
        config : BuildQuadTreeConfig, optional
            Quadtree build settings. Defaults to the standard config.
        """
        self.config = config or BuildQuadTreeConfig()
        if self.config.max_depth < 0:
            raise ValueError("max_depth must be non-negative")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build and store a Barnes-Hut quadtree in ``state.extras``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing positions and optional degrees.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            Updated state with ``extras["quadtree"]`` populated.
        """
        del problem, ctx

        pos = _validate_positions(state)
        num_nodes = pos.shape[0]
        if num_nodes == 0:
            state.extras["quadtree"] = None
            return state

        pos_np = pos.detach().to(device="cpu", dtype=torch.float64).numpy()
        mass_np = _mass_array(state, num_nodes)
        indices = np.arange(num_nodes, dtype=np.int64)
        state.extras["quadtree"] = _build_quadtree_node(
            pos_np=pos_np,
            mass_np=mass_np,
            indices=indices,
            depth=0,
            max_depth=self.config.max_depth,
        )
        return state


@register_op
class BuildDensityGrid(Op):
    """Build the DrL-style density grid over the current positions."""

    name = "build_density_grid"
    category = OpCategory.CONTEXT
    reads = ("pos",)
    writes = ("extras.density_grid",)
    requires = ("pos",)
    access_pattern = "global"

    def __init__(self, config: Optional[BuildDensityGridConfig] = None) -> None:
        """Initialize the op with an optional config.

        Parameters
        ----------
        config : BuildDensityGridConfig, optional
            Density-grid settings. Defaults to the standard config.
        """
        self.config = config or BuildDensityGridConfig()
        if self.config.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.config.view_size <= 0.0:
            raise ValueError("view_size must be positive")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build the shared density grid and attach every positioned node.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            Updated state with ``extras["density_grid"]`` populated.
        """
        del problem, ctx

        pos = _validate_positions(state).detach().to(device="cpu", dtype=torch.float64)
        density_grid = _DensityGrid(
            grid_size=self.config.grid_size,
            view_size=float(self.config.view_size),
            radius=_DENSITY_GRID_RADIUS,
        )
        for node in range(pos.shape[0]):
            density_grid.add_node(node=node, position=pos[node])
        state.extras["density_grid"] = density_grid
        return state


@register_op
class RefreshKDTreePairs(Op):
    """Refresh cached local repulsion pairs using SciPy's ``cKDTree``."""

    name = "refresh_kdtree_pairs"
    category = OpCategory.CONTEXT
    reads = ("pos", "step")
    writes = ("extras.kdtree_pairs",)
    requires = ("pos",)
    access_pattern = "sampled"

    def __init__(self, config: Optional[RefreshKDTreePairsConfig] = None) -> None:
        """Initialize the op with an optional config.

        Parameters
        ----------
        config : RefreshKDTreePairsConfig, optional
            KD-tree refresh settings. Defaults to the standard config.
        """
        self.config = config or RefreshKDTreePairsConfig()
        if self.config.radius <= 0.0:
            raise ValueError("radius must be positive")
        if self.config.interval <= 0:
            raise ValueError("interval must be positive")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Refresh nearby-node pairs on the configured cadence.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            Updated state with ``extras["kdtree_pairs"]`` populated or retained.
        """
        del problem, ctx

        if "kdtree_pairs" in state.extras and state.step % self.config.interval != 0:
            return state

        pos = _validate_positions(state)
        state.extras["kdtree_pairs"] = _kdtree_repulsion_pairs(
            pos=pos,
            query_radius=self.config.radius,
        )
        return state

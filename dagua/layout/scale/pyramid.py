"""Deterministic grid-pyramid far-field forces for scale layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

_EPS = 1.0e-6
_MAX_FULL_GATHER_CELLS = 128
_DEFAULT_STREAM_CHUNK_NODES = 2_000_000


@dataclass(frozen=True)
class GridLevel:
    """One density level in a grid pyramid.

    Parameters
    ----------
    mass : torch.Tensor
        Cell mass grid with shape ``[H, W]``.
    centroid : torch.Tensor
        Cell centroid grid with shape ``[H, W, 2]``.
    cell_size : torch.Tensor
        Per-axis cell size with shape ``[2]``.
    origin : torch.Tensor
        Grid origin with shape ``[2]``.
    """

    mass: torch.Tensor
    centroid: torch.Tensor
    cell_size: torch.Tensor
    origin: torch.Tensor


@dataclass(frozen=True)
class GridPyramid:
    """Deterministic multiresolution density pyramid.

    Parameters
    ----------
    levels : list[GridLevel]
        Levels ordered from fine to coarse.
    device_kind : str
        Device kind used to build the tensors.
    """

    levels: list[GridLevel]
    device_kind: str


def build_grid_pyramid(
    pos: torch.Tensor,
    masses: Optional[torch.Tensor] = None,
    *,
    base_cell_size: float = 50.0,
    max_cells_per_axis: int = 256,
    max_levels: int = 6,
    force_cpu: bool = False,
    deterministic_sort: bool = True,
) -> GridPyramid:
    """Build a deterministic density pyramid from node positions.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    masses : torch.Tensor, optional
        Node masses with shape ``[N]``. Missing masses default to one.
    base_cell_size : float, default=50.0
        Desired finest cell size in layout units.
    max_cells_per_axis : int, default=256
        Per-axis grid cap for memory control.
    max_levels : int, default=6
        Maximum number of pooled levels.
    force_cpu : bool, default=False
        Build on CPU even when positions live on CUDA.
    deterministic_sort : bool, default=True
        Use stable cell sorting plus segmented reductions for weighted cell
        accumulations. This avoids CUDA weighted ``bincount`` atomics and makes
        per-device pyramid builds byte-deterministic.

    Returns
    -------
    GridPyramid
        Fine-to-coarse density pyramid.
    """
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("pos must have shape [N, 2].")
    work_pos = pos.detach()
    if force_cpu:
        work_pos = work_pos.to(device="cpu")
    work_pos = work_pos.to(dtype=torch.float32)
    if masses is None:
        work_mass = torch.ones((work_pos.shape[0],), dtype=torch.float32, device=work_pos.device)
    else:
        work_mass = masses.detach().to(device=work_pos.device, dtype=torch.float32)
    if work_mass.shape != (work_pos.shape[0],):
        raise ValueError("masses must have shape [N].")
    if work_pos.shape[0] == 0:
        empty = GridLevel(
            mass=torch.zeros((1, 1), dtype=torch.float32, device=work_pos.device),
            centroid=torch.zeros((1, 1, 2), dtype=torch.float32, device=work_pos.device),
            cell_size=torch.ones((2,), dtype=torch.float32, device=work_pos.device),
            origin=torch.zeros((2,), dtype=torch.float32, device=work_pos.device),
        )
        return GridPyramid(levels=[empty], device_kind=work_pos.device.type)

    min_xy = work_pos.min(dim=0).values
    max_xy = work_pos.max(dim=0).values
    span = torch.clamp(max_xy - min_xy, min=float(base_cell_size))
    desired = torch.clamp(
        torch.ceil(span / max(float(base_cell_size), _EPS)).to(dtype=torch.long),
        min=1,
        max=max(1, int(max_cells_per_axis)),
    )
    cell_size = span / desired.to(device=work_pos.device, dtype=torch.float32)
    origin = min_xy - 0.5 * cell_size
    fine = _build_grid_level(
        work_pos,
        work_mass,
        origin=origin,
        cell_size=cell_size,
        shape=desired,
        deterministic_sort=deterministic_sort,
    )
    levels = [fine]
    current = fine
    while len(levels) < max(1, int(max_levels)):
        if current.mass.shape[0] <= 1 and current.mass.shape[1] <= 1:
            break
        current = _pool_level(current)
        levels.append(current)
    return GridPyramid(levels=levels, device_kind=work_pos.device.type)


def build_grid_pyramid_streaming(
    pos: torch.Tensor,
    masses: Optional[torch.Tensor] = None,
    *,
    base_cell_size: float = 50.0,
    max_cells_per_axis: int = 256,
    max_levels: int = 6,
    chunk_nodes: int = _DEFAULT_STREAM_CHUNK_NODES,
) -> GridPyramid:
    """Build a CPU grid pyramid without materializing per-node sort buffers.

    Parameters
    ----------
    pos : torch.Tensor
        CPU position tensor with shape ``[N, 2]``.
    masses : torch.Tensor, optional
        CPU node masses with shape ``[N]``. Missing masses default to one.
    base_cell_size : float, default=50.0
        Desired finest cell size in layout units.
    max_cells_per_axis : int, default=256
        Per-axis grid cap for memory control.
    max_levels : int, default=6
        Maximum number of pooled levels.
    chunk_nodes : int, default=2_000_000
        Number of nodes processed per streaming chunk.

    Returns
    -------
    GridPyramid
        Fine-to-coarse density pyramid with CPU-resident tensors.

    Notes
    -----
    The regular builder uses stable sorting to make CUDA reductions
    deterministic. At 100M nodes those sort buffers dominate memory, so this
    path uses CPU ``scatter_add_`` chunks against a bounded grid instead.
    """
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("pos must have shape [N, 2].")
    work_pos = pos.detach().to(device="cpu", dtype=torch.float32)
    if masses is None:
        work_mass: Optional[torch.Tensor] = None
    else:
        work_mass = masses.detach().to(device="cpu", dtype=torch.float32)
        if work_mass.shape != (work_pos.shape[0],):
            raise ValueError("masses must have shape [N].")
    if work_pos.shape[0] == 0:
        return build_grid_pyramid(
            work_pos,
            work_mass,
            base_cell_size=base_cell_size,
            max_cells_per_axis=max_cells_per_axis,
            max_levels=max_levels,
            force_cpu=True,
        )

    min_xy = work_pos.min(dim=0).values
    max_xy = work_pos.max(dim=0).values
    span = torch.clamp(max_xy - min_xy, min=float(base_cell_size))
    desired = torch.clamp(
        torch.ceil(span / max(float(base_cell_size), _EPS)).to(dtype=torch.long),
        min=1,
        max=max(1, int(max_cells_per_axis)),
    )
    cell_size = span / desired.to(dtype=torch.float32)
    origin = min_xy - 0.5 * cell_size
    fine = _build_grid_level_streaming(
        work_pos,
        work_mass,
        origin=origin,
        cell_size=cell_size,
        shape=desired,
        chunk_nodes=chunk_nodes,
    )
    levels = [fine]
    current = fine
    while len(levels) < max(1, int(max_levels)):
        if current.mass.shape[0] <= 1 and current.mass.shape[1] <= 1:
            break
        current = _pool_level(current)
        levels.append(current)
    return GridPyramid(levels=levels, device_kind="cpu_streaming")


def far_field_repulsion_force(
    pos: torch.Tensor,
    pyramid: GridPyramid,
    *,
    strength: float = 1.0,
    softening: float = 1.0,
    max_displacement: float = 25.0,
    level_stride: int = 1,
) -> torch.Tensor:
    """Return grid-pyramid repulsion displacement without autograd tracking.

    Each node gathers a 3x3 cell window at the finest level for near-field
    repulsion. Every coarser gathered level contributes only the cells that
    lie fully outside the finer gathered window (no double counting), and the
    coarsest level gathers the full grid so all far mass is felt globally.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    pyramid : GridPyramid
        Density pyramid built from detached positions.
    strength : float, default=1.0
        Repulsion multiplier.
    softening : float, default=1.0
        Per-pair distance softening in layout units.
    max_displacement : float, default=25.0
        Per-node displacement norm cap.
    level_stride : int, default=1
        Gather every ``level_stride``-th level (always including the finest
        and coarsest); skipped mass is simply counted at the next-coarser
        gathered level.

    Returns
    -------
    torch.Tensor
        Repulsive displacement with shape ``[N, 2]`` on ``pos.device``.
    """
    with torch.no_grad():
        work_pos = pos.detach().to(dtype=torch.float32)
        force = torch.zeros_like(work_pos)
        soft2 = float(softening) * float(softening)
        window: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        last = len(pyramid.levels) - 1
        stride = max(1, int(level_stride))
        selected = sorted(set(range(0, last + 1, stride)) | {last})
        for level_index in selected:
            level = pyramid.levels[level_index]
            full = level_index == last and int(level.mass.numel()) <= _MAX_FULL_GATHER_CELLS
            level_force, window = _level_repulsion(
                work_pos,
                level,
                soft2=soft2,
                exclude_window=window,
                gather_full=full,
            )
            force = force + level_force
        force = force * float(strength)
        norm = torch.linalg.norm(force, dim=1, keepdim=True).clamp_min(_EPS)
        capped = force * torch.clamp(float(max_displacement) / norm, max=1.0)
        return capped.to(device=pos.device, dtype=torch.float32)


def density_image(pyramid: GridPyramid, *, level: int = 0) -> torch.Tensor:
    """Return one density level for later LOD rendering.

    Parameters
    ----------
    pyramid : GridPyramid
        Density pyramid.
    level : int, default=0
        Requested level index.

    Returns
    -------
    torch.Tensor
        Detached CPU density image with shape ``[H, W]``.
    """
    index = max(0, min(int(level), len(pyramid.levels) - 1))
    return pyramid.levels[index].mass.detach().to(device="cpu", dtype=torch.float32).clone()


def choose_pyramid_device(num_nodes: int, requested_device: str) -> str:
    """Choose the measured-safe pyramid build device for a rung.

    Parameters
    ----------
    num_nodes : int
        Current node count.
    requested_device : str
        User-requested layout device.

    Returns
    -------
    str
        ``"cuda"`` only when CUDA is requested and available, else ``"cpu"``.
    """
    if str(requested_device).startswith("cuda") and torch.cuda.is_available():
        estimated_peak = _estimate_pyramid_peak_bytes(int(num_nodes))
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
        except RuntimeError:
            return "cpu"
        if estimated_peak > int(free_bytes * 0.70):
            return "cpu"
        return "cuda"
    return "cpu"


def choose_streaming_chunk_device(
    chunk_nodes: int,
    requested_device: str,
    *,
    resident_pyramid_bytes: int = 0,
) -> str:
    """Choose the device for one CPU-resident streaming chunk.

    Parameters
    ----------
    chunk_nodes : int
        Number of positions processed in the chunk.
    requested_device : str
        Requested layout device.
    resident_pyramid_bytes : int, default=0
        Bytes needed to copy the small pyramid payload to the chunk device.

    Returns
    -------
    str
        ``"cuda"`` when the chunk plus pyramid fit under 70% free VRAM,
        otherwise ``"cpu"``.
    """
    if not str(requested_device).startswith("cuda") or not torch.cuda.is_available():
        return "cpu"
    estimated = _estimate_streaming_force_chunk_bytes(
        int(chunk_nodes),
        int(resident_pyramid_bytes),
    )
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
    except RuntimeError:
        return "cpu"
    if estimated > int(free_bytes * 0.70):
        return "cpu"
    return "cuda"


def pyramid_tensor_bytes(pyramid: GridPyramid) -> int:
    """Return resident tensor bytes for a grid pyramid.

    Parameters
    ----------
    pyramid : GridPyramid
        Density pyramid.

    Returns
    -------
    int
        Sum of tensor payload bytes across all levels.
    """
    total = 0
    for level in pyramid.levels:
        total += int(level.mass.numel() * level.mass.element_size())
        total += int(level.centroid.numel() * level.centroid.element_size())
        total += int(level.cell_size.numel() * level.cell_size.element_size())
        total += int(level.origin.numel() * level.origin.element_size())
    return total


def _build_grid_level(
    pos: torch.Tensor,
    masses: torch.Tensor,
    *,
    origin: torch.Tensor,
    cell_size: torch.Tensor,
    shape: torch.Tensor,
    deterministic_sort: bool = True,
) -> GridLevel:
    """Scatter node masses and centroids into one grid level.

    Parameters
    ----------
    pos : torch.Tensor
        Positions with shape ``[N, 2]``.
    masses : torch.Tensor
        Node masses with shape ``[N]``.
    origin : torch.Tensor
        Grid origin with shape ``[2]``.
    cell_size : torch.Tensor
        Cell size with shape ``[2]``.
    shape : torch.Tensor
        Integer ``[W, H]`` cell counts.
    deterministic_sort : bool, default=True
        Whether weighted accumulations use stable sort/segment reductions
        instead of weighted ``bincount``.

    Returns
    -------
    GridLevel
        Scattered grid level.
    """
    width = int(shape[0].item())
    height = int(shape[1].item())
    ij = torch.floor((pos - origin) / cell_size.clamp_min(_EPS)).to(dtype=torch.long)
    x = torch.clamp(ij[:, 0], 0, width - 1)
    y = torch.clamp(ij[:, 1], 0, height - 1)
    cell_id = y * width + x
    cell_count = width * height
    if deterministic_sort:
        flat_mass, flat_x, flat_y = _sorted_cell_sums(
            cell_id,
            masses,
            masses * pos[:, 0],
            masses * pos[:, 1],
            cell_count=cell_count,
        )
    else:
        flat_mass = torch.bincount(
            cell_id,
            weights=masses,
            minlength=cell_count,
        ).to(dtype=torch.float32)
        flat_x = torch.bincount(
            cell_id,
            weights=masses * pos[:, 0],
            minlength=cell_count,
        ).to(dtype=torch.float32)
        flat_y = torch.bincount(
            cell_id,
            weights=masses * pos[:, 1],
            minlength=cell_count,
        ).to(dtype=torch.float32)
    centroid = torch.stack((flat_x, flat_y), dim=1)
    nonzero = flat_mass > 0.0
    centroid[nonzero] = centroid[nonzero] / flat_mass[nonzero].unsqueeze(1)
    centers = _cell_centers(width, height, origin, cell_size, pos.device)
    centroid[~nonzero] = centers[~nonzero]
    return GridLevel(
        mass=flat_mass.reshape(height, width),
        centroid=centroid.reshape(height, width, 2),
        cell_size=cell_size.detach().clone(),
        origin=origin.detach().clone(),
    )


def _build_grid_level_streaming(
    pos: torch.Tensor,
    masses: Optional[torch.Tensor],
    *,
    origin: torch.Tensor,
    cell_size: torch.Tensor,
    shape: torch.Tensor,
    chunk_nodes: int,
) -> GridLevel:
    """Scatter node masses and centroids into one grid using CPU chunks.

    Parameters
    ----------
    pos : torch.Tensor
        CPU positions with shape ``[N, 2]``.
    masses : torch.Tensor or None
        Optional CPU masses with shape ``[N]``.
    origin : torch.Tensor
        Grid origin with shape ``[2]``.
    cell_size : torch.Tensor
        Cell size with shape ``[2]``.
    shape : torch.Tensor
        Integer ``[W, H]`` cell counts.
    chunk_nodes : int
        Maximum node rows per chunk.

    Returns
    -------
    GridLevel
        Scattered CPU grid level.
    """
    width = int(shape[0].item())
    height = int(shape[1].item())
    cell_count = width * height
    flat_mass = torch.zeros((cell_count,), dtype=torch.float32)
    flat_x = torch.zeros_like(flat_mass)
    flat_y = torch.zeros_like(flat_mass)
    chunk = max(1, int(chunk_nodes))
    clamped_cell = cell_size.to(dtype=torch.float32).clamp_min(_EPS)
    for start in range(0, int(pos.shape[0]), chunk):
        end = min(int(pos.shape[0]), start + chunk)
        pos_chunk = pos[start:end].to(dtype=torch.float32)
        mass_chunk = (
            torch.ones((end - start,), dtype=torch.float32)
            if masses is None
            else masses[start:end].to(dtype=torch.float32)
        )
        ij = torch.floor((pos_chunk - origin) / clamped_cell).to(dtype=torch.long)
        x = torch.clamp(ij[:, 0], 0, width - 1)
        y = torch.clamp(ij[:, 1], 0, height - 1)
        cell_id = y * width + x
        flat_mass.scatter_add_(0, cell_id, mass_chunk)
        flat_x.scatter_add_(0, cell_id, mass_chunk * pos_chunk[:, 0])
        flat_y.scatter_add_(0, cell_id, mass_chunk * pos_chunk[:, 1])
    centroid = torch.stack((flat_x, flat_y), dim=1)
    nonzero = flat_mass > 0.0
    centroid[nonzero] = centroid[nonzero] / flat_mass[nonzero].unsqueeze(1)
    centers = _cell_centers(width, height, origin, cell_size, torch.device("cpu"))
    centroid[~nonzero] = centers[~nonzero]
    return GridLevel(
        mass=flat_mass.reshape(height, width),
        centroid=centroid.reshape(height, width, 2),
        cell_size=cell_size.detach().clone(),
        origin=origin.detach().clone(),
    )


def _sorted_cell_sums(
    cell_id: torch.Tensor,
    mass: torch.Tensor,
    weighted_x: torch.Tensor,
    weighted_y: torch.Tensor,
    *,
    cell_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic cell sums by stable sorting nodes by cell.

    Parameters
    ----------
    cell_id : torch.Tensor
        Flattened cell IDs with shape ``[N]``.
    mass : torch.Tensor
        Node masses with shape ``[N]``.
    weighted_x : torch.Tensor
        ``mass * x`` values with shape ``[N]``.
    weighted_y : torch.Tensor
        ``mass * y`` values with shape ``[N]``.
    cell_count : int
        Total number of cells in the grid.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Flat mass, weighted-x, and weighted-y sums with shape ``[cell_count]``.
    """
    device = cell_id.device
    flat_mass = torch.zeros((int(cell_count),), dtype=torch.float32, device=device)
    flat_x = torch.zeros_like(flat_mass)
    flat_y = torch.zeros_like(flat_mass)
    if cell_id.numel() == 0:
        return flat_mass, flat_x, flat_y
    order = cell_id.argsort(stable=True)
    sorted_cell = cell_id[order]
    counts = torch.bincount(sorted_cell, minlength=int(cell_count))
    occupied = torch.nonzero(counts > 0, as_tuple=False).flatten()
    lengths = counts[occupied].to(dtype=torch.long)
    sorted_mass = mass[order].to(dtype=torch.float32)
    sorted_x = weighted_x[order].to(dtype=torch.float32)
    sorted_y = weighted_y[order].to(dtype=torch.float32)
    flat_mass[occupied] = torch.segment_reduce(sorted_mass, "sum", lengths=lengths)
    flat_x[occupied] = torch.segment_reduce(sorted_x, "sum", lengths=lengths)
    flat_y[occupied] = torch.segment_reduce(sorted_y, "sum", lengths=lengths)
    return flat_mass, flat_x, flat_y


def _estimate_pyramid_peak_bytes(num_nodes: int) -> int:
    """Estimate CUDA pyramid temporary bytes for device selection.

    Parameters
    ----------
    num_nodes : int
        Node count.

    Returns
    -------
    int
        Conservative bytes for copied positions, cell IDs, stable sort
        permutation, and reduction values.
    """
    return int(num_nodes) * (2 * 4 + 4 + 8 + 8 + 3 * 4) * 2


def _estimate_streaming_force_chunk_bytes(chunk_nodes: int, resident_pyramid_bytes: int) -> int:
    """Estimate CUDA bytes for one streaming far-field chunk.

    Parameters
    ----------
    chunk_nodes : int
        Number of nodes processed in the chunk.
    resident_pyramid_bytes : int
        Bytes needed for the copied pyramid levels.

    Returns
    -------
    int
        Conservative temporary byte estimate.
    """
    per_node = 2 * 4 + 2 * 4 + 2 * 4 + 2 * 8 + 4
    return int((int(chunk_nodes) * per_node + int(resident_pyramid_bytes)) * 1.5)


def _pool_level(level: GridLevel) -> GridLevel:
    """Average-pool one pyramid level by 2x2 cells.

    Parameters
    ----------
    level : GridLevel
        Fine level.

    Returns
    -------
    GridLevel
        Coarser pooled level.
    """
    mass = level.mass
    height, width = mass.shape
    pad_h = int(height % 2)
    pad_w = int(width % 2)
    padded_mass = (
        F.pad(
            mass.unsqueeze(0).unsqueeze(0),
            (0, pad_w, 0, pad_h),
        )
        .squeeze(0)
        .squeeze(0)
    )
    weighted = level.centroid * mass.unsqueeze(2)
    padded_weighted = F.pad(weighted.permute(2, 0, 1).unsqueeze(0), (0, pad_w, 0, pad_h))
    pooled_mass = F.avg_pool2d(padded_mass.unsqueeze(0).unsqueeze(0), 2, stride=2)[0, 0] * 4.0
    pooled_weighted = F.avg_pool2d(padded_weighted, 2, stride=2)[0].permute(1, 2, 0) * 4.0
    centroid = torch.zeros((*pooled_mass.shape, 2), dtype=torch.float32, device=mass.device)
    nonzero = pooled_mass > 0.0
    centroid[nonzero] = pooled_weighted[nonzero] / pooled_mass[nonzero].unsqueeze(1)
    new_cell = level.cell_size * 2.0
    centers = _cell_centers(
        int(pooled_mass.shape[1]),
        int(pooled_mass.shape[0]),
        level.origin,
        new_cell,
        mass.device,
    ).reshape((*pooled_mass.shape, 2))
    centroid[~nonzero] = centers[~nonzero]
    return GridLevel(
        mass=pooled_mass.contiguous(),
        centroid=centroid.contiguous(),
        cell_size=new_cell,
        origin=level.origin,
    )


def _level_repulsion(
    pos: torch.Tensor,
    level: GridLevel,
    *,
    soft2: float = _EPS,
    exclude_window: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    gather_full: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Gather cell repulsion for every node at one pyramid level.

    Parameters
    ----------
    pos : torch.Tensor
        Positions with shape ``[N, 2]``.
    level : GridLevel
        Pyramid level.
    soft2 : float, default=_EPS
        Squared per-pair distance softening.
    exclude_window : tuple[torch.Tensor, torch.Tensor], optional
        Per-node world-coordinate ``(lo, hi)`` boxes already covered by a
        finer level; cells fully inside are skipped.
    gather_full : bool, default=False
        Gather every grid cell instead of the 3x3 window (coarsest level).

    Returns
    -------
    tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
        Level displacement with shape ``[N, 2]`` and this level's per-node
        covered window boxes for the next-coarser exclusion.
    """
    height, width = level.mass.shape
    origin = level.origin.to(pos.device)
    cell_size = level.cell_size.to(pos.device).clamp_min(_EPS)
    ij = torch.floor((pos - origin) / cell_size).to(dtype=torch.long)
    base_x = torch.clamp(ij[:, 0], 0, width - 1)
    base_y = torch.clamp(ij[:, 1], 0, height - 1)
    force = torch.zeros_like(pos)
    mass = level.mass.to(device=pos.device)
    centroid = level.centroid.to(device=pos.device)

    def _accumulate(x: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        xc = torch.clamp(x, 0, width - 1)
        yc = torch.clamp(y, 0, height - 1)
        cell_mass = mass[yc, xc]
        keep = valid & (cell_mass > 0.0)
        if exclude_window is not None:
            cell_lo = origin + torch.stack((xc, yc), dim=1).to(dtype=pos.dtype) * cell_size
            cell_hi = cell_lo + cell_size
            lo, hi = exclude_window
            inside = (
                (cell_lo[:, 0] >= lo[:, 0] - _EPS)
                & (cell_lo[:, 1] >= lo[:, 1] - _EPS)
                & (cell_hi[:, 0] <= hi[:, 0] + _EPS)
                & (cell_hi[:, 1] <= hi[:, 1] + _EPS)
            )
            keep = keep & ~inside
        delta = pos - centroid[yc, xc]
        dist2 = (delta * delta).sum(dim=1, keepdim=True) + float(soft2)
        contribution = cell_mass.unsqueeze(1) * delta / dist2
        return torch.where(keep.unsqueeze(1), contribution, torch.zeros_like(contribution))

    if gather_full:
        occupied = torch.nonzero(mass > 0.0, as_tuple=False).tolist()
        valid_all = torch.ones_like(base_x, dtype=torch.bool)
        for gy, gx in occupied:
            x = torch.full_like(base_x, int(gx))
            y = torch.full_like(base_y, int(gy))
            force = force + _accumulate(x, y, valid_all)
    else:
        for dy in (-1, 0, 1):
            y = base_y + dy
            for dx in (-1, 0, 1):
                x = base_x + dx
                valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                force = force + _accumulate(x, y, valid)
    window_lo = origin + (torch.stack((base_x, base_y), dim=1) - 1).to(dtype=pos.dtype) * cell_size
    window_hi = window_lo + 3.0 * cell_size
    return force, (window_lo, window_hi)


def _cell_centers(
    width: int,
    height: int,
    origin: torch.Tensor,
    cell_size: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return flattened cell centers for a grid.

    Parameters
    ----------
    width : int
        Grid width.
    height : int
        Grid height.
    origin : torch.Tensor
        Grid origin with shape ``[2]``.
    cell_size : torch.Tensor
        Cell size with shape ``[2]``.
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Cell centers with shape ``[width * height, 2]``.
    """
    xs = torch.arange(int(width), dtype=torch.float32, device=device) + 0.5
    ys = torch.arange(int(height), dtype=torch.float32, device=device) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    return origin.to(device=device) + centers * cell_size.to(device=device)


__all__ = [
    "GridLevel",
    "GridPyramid",
    "build_grid_pyramid",
    "build_grid_pyramid_streaming",
    "choose_pyramid_device",
    "choose_streaming_chunk_device",
    "density_image",
    "far_field_repulsion_force",
    "pyramid_tensor_bytes",
]

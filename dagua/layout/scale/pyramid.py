"""Deterministic grid-pyramid far-field forces for scale layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

_EPS = 1.0e-6


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
    fine = _build_grid_level(work_pos, work_mass, origin=origin, cell_size=cell_size, shape=desired)
    levels = [fine]
    current = fine
    while len(levels) < max(1, int(max_levels)):
        if current.mass.shape[0] <= 1 and current.mass.shape[1] <= 1:
            break
        current = _pool_level(current)
        levels.append(current)
    return GridPyramid(levels=levels, device_kind=work_pos.device.type)


def far_field_repulsion_force(
    pos: torch.Tensor,
    pyramid: GridPyramid,
    *,
    strength: float = 1.0,
    softening: float = 1.0,
    max_displacement: float = 25.0,
) -> torch.Tensor:
    """Return grid-pyramid repulsion displacement without autograd tracking.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    pyramid : GridPyramid
        Density pyramid built from detached positions.
    strength : float, default=1.0
        Repulsion multiplier.
    softening : float, default=1.0
        Distance softening in layout units.
    max_displacement : float, default=25.0
        Per-node displacement norm cap.

    Returns
    -------
    torch.Tensor
        Repulsive displacement with shape ``[N, 2]`` on ``pos.device``.
    """
    with torch.no_grad():
        work_pos = pos.detach().to(dtype=torch.float32)
        force = torch.zeros_like(work_pos)
        for level_index, level in enumerate(pyramid.levels):
            level_force = _level_repulsion(work_pos, level)
            force = force + level_force / float(2**level_index)
        force = force * float(strength)
        norm = torch.linalg.norm(force, dim=1, keepdim=True).clamp_min(float(softening))
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
    del num_nodes
    if str(requested_device).startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_grid_level(
    pos: torch.Tensor,
    masses: torch.Tensor,
    *,
    origin: torch.Tensor,
    cell_size: torch.Tensor,
    shape: torch.Tensor,
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


def _level_repulsion(pos: torch.Tensor, level: GridLevel) -> torch.Tensor:
    """Gather 3x3 cell repulsion for every node at one pyramid level.

    Parameters
    ----------
    pos : torch.Tensor
        Positions with shape ``[N, 2]``.
    level : GridLevel
        Pyramid level.

    Returns
    -------
    torch.Tensor
        Level displacement with shape ``[N, 2]``.
    """
    height, width = level.mass.shape
    ij = torch.floor(
        (pos - level.origin.to(pos.device)) / level.cell_size.to(pos.device).clamp_min(_EPS)
    ).to(dtype=torch.long)
    base_x = torch.clamp(ij[:, 0], 0, width - 1)
    base_y = torch.clamp(ij[:, 1], 0, height - 1)
    force = torch.zeros_like(pos)
    mass = level.mass.to(device=pos.device)
    centroid = level.centroid.to(device=pos.device)
    for dy in (-1, 0, 1):
        y = torch.clamp(base_y + dy, 0, height - 1)
        for dx in (-1, 0, 1):
            x = torch.clamp(base_x + dx, 0, width - 1)
            cell_mass = mass[y, x].unsqueeze(1)
            cell_centroid = centroid[y, x]
            delta = pos - cell_centroid
            dist2 = (delta * delta).sum(dim=1, keepdim=True).clamp_min(_EPS)
            # Remove a node's own cell contribution only when the aggregate is
            # a singleton-like self hit; this avoids exact zero self-forces.
            contribution = cell_mass * delta / dist2
            force = force + contribution
    return force


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
    "choose_pyramid_device",
    "density_image",
    "far_field_repulsion_force",
]

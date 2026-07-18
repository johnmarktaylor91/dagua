"""Shape-aware geometry helpers for native layout finisher terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

_EPS = 1.0e-6
_BOX_KINDS = frozenset({"", "box", "rect", "rectangle", "roundrect", "rounded", "record"})
_ELLIPSE_KINDS = frozenset({"circle", "ellipse", "double_circle"})
_DIAMOND_KINDS = frozenset({"diamond"})
_TRIANGLE_KINDS = frozenset({"triangle"})
_SHAPE_KIND_TO_CODE: Mapping[str, int] = {
    **{name: 0 for name in _BOX_KINDS},
    **{name: 1 for name in _ELLIPSE_KINDS},
    **{name: 2 for name in _DIAMOND_KINDS},
    **{name: 3 for name in _TRIANGLE_KINDS},
}


@dataclass(frozen=True)
class NativeShapeGeometry:
    """Per-node native shape descriptors.

    Parameters
    ----------
    kind_codes : torch.Tensor
        Integer shape-kind tensor with shape ``[N]``. Code ``0`` preserves
        legacy box behavior; nonzero codes activate shape-aware geometry.
    """

    kind_codes: torch.Tensor

    @property
    def has_non_box(self) -> bool:
        """Return whether any node needs the shape-aware path.

        Returns
        -------
        bool
            ``True`` when at least one descriptor is non-rectangular.
        """
        return bool((self.kind_codes != 0).any().item())

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "NativeShapeGeometry":
        """Return descriptors on the requested device.

        Parameters
        ----------
        device : torch.device
            Device for the returned descriptor tensor.
        dtype : torch.dtype
            Unused floating dtype included to mirror tensor-normalization
            call-sites that already carry it.

        Returns
        -------
        NativeShapeGeometry
            Descriptor with integer codes on ``device``.
        """
        del dtype
        return NativeShapeGeometry(kind_codes=self.kind_codes.to(device=device, dtype=torch.long))


def resolve_native_shape_geometry(
    node_shapes: Optional[Sequence[str]],
    node_count: int,
) -> Optional[NativeShapeGeometry]:
    """Resolve optional style shape names into native shape descriptors.

    Parameters
    ----------
    node_shapes : Sequence[str] or None
        Style shape names aligned with graph nodes.
    node_count : int
        Number of graph nodes.

    Returns
    -------
    NativeShapeGeometry or None
        ``None`` when all nodes should keep exact legacy AABB behavior.
    """
    if node_shapes is None or node_count <= 0:
        return None
    codes = [
        int(_SHAPE_KIND_TO_CODE.get(str(node_shapes[index]).strip().lower(), 0))
        if index < len(node_shapes)
        else 0
        for index in range(node_count)
    ]
    tensor = torch.tensor(codes, dtype=torch.long)
    geometry = NativeShapeGeometry(kind_codes=tensor)
    return geometry if geometry.has_non_box else None


def shape_support_radius(
    directions: torch.Tensor,
    sizes: torch.Tensor,
    kind_codes: torch.Tensor,
) -> torch.Tensor:
    """Return each shape's support radius along paired directions.

    Parameters
    ----------
    directions : torch.Tensor
        Unit-ish direction vectors with shape ``[..., 2]``.
    sizes : torch.Tensor
        Node sizes with shape ``[..., 2]``.
    kind_codes : torch.Tensor
        Shape-kind codes broadcastable to ``directions[..., 0]``.

    Returns
    -------
    torch.Tensor
        Support radii with shape ``directions.shape[:-1]``.
    """
    half = sizes.to(device=directions.device, dtype=directions.dtype).clamp_min(_EPS) * 0.5
    abs_dir = directions.abs()
    box = half[..., 0] * abs_dir[..., 0] + half[..., 1] * abs_dir[..., 1]
    ellipse = torch.sqrt(
        (half[..., 0] * directions[..., 0]).square()
        + (half[..., 1] * directions[..., 1]).square()
        + _EPS
    )
    diamond = 1.0 / (
        abs_dir[..., 0] / half[..., 0].clamp_min(_EPS)
        + abs_dir[..., 1] / half[..., 1].clamp_min(_EPS)
    ).clamp_min(_EPS)
    triangle = _triangle_support_radius(directions, half)
    codes = kind_codes.to(device=directions.device, dtype=torch.long)
    return torch.where(
        codes == 1,
        ellipse,
        torch.where(codes == 2, diamond, torch.where(codes == 3, triangle, box)),
    )


def pairwise_shape_signed_gap(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    geometry: NativeShapeGeometry,
    *,
    padding: float = 0.0,
    max_nodes: int = 512,
) -> torch.Tensor:
    """Return sampled pairwise signed gaps between true shape envelopes.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    geometry : NativeShapeGeometry
        Per-node shape descriptors.
    padding : float, default=0.0
        Additional required shape gap.
    max_nodes : int, default=512
        Deterministic cap for all-pairs shape checks.

    Returns
    -------
    torch.Tensor
        Signed gaps for unordered sampled pairs with shape ``[P]``. Negative
        values indicate overlap.
    """
    node_count = int(pos.shape[0])
    if node_count < 2:
        return pos.new_empty(0)
    sample = _sample_nodes(node_count, max_nodes, pos.device)
    work_pos = pos[sample]
    work_sizes = node_sizes.to(device=pos.device, dtype=pos.dtype)[sample]
    work_codes = geometry.kind_codes.to(device=pos.device, dtype=torch.long)[sample]
    left, right = torch.triu_indices(
        work_pos.shape[0],
        work_pos.shape[0],
        offset=1,
        device=pos.device,
    )
    if left.numel() == 0:
        return pos.new_empty(0)
    delta = work_pos[right] - work_pos[left]
    distance = torch.sqrt(delta.square().sum(dim=1) + _EPS)
    direction = delta / distance[:, None].clamp_min(_EPS)
    left_radius = shape_support_radius(direction, work_sizes[left], work_codes[left])
    right_radius = shape_support_radius(-direction, work_sizes[right], work_codes[right])
    return distance - left_radius - right_radius - float(padding)


def shape_overlap_hinge_loss(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    geometry: Optional[NativeShapeGeometry],
    *,
    padding: float = 0.0,
    max_nodes: int = 512,
) -> torch.Tensor:
    """Penalize overlap between sampled true node shapes.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    geometry : NativeShapeGeometry or None
        Shape descriptors. ``None`` returns a connected zero scalar.
    padding : float, default=0.0
        Additional required shape gap.
    max_nodes : int, default=512
        Deterministic cap for all-pairs shape checks.

    Returns
    -------
    torch.Tensor
        Scalar overlap hinge loss.
    """
    if geometry is None:
        return pos.sum() * 0.0
    signed_gap = pairwise_shape_signed_gap(
        pos,
        node_sizes,
        geometry,
        padding=padding,
        max_nodes=max_nodes,
    )
    if signed_gap.numel() == 0:
        return pos.sum() * 0.0
    scale = node_sizes.to(device=pos.device, dtype=pos.dtype).mean().detach().clamp_min(1.0)
    loss = F.relu(-signed_gap).square().mean() / scale.square()
    return torch.nan_to_num(loss, nan=0.0, posinf=10.0, neginf=0.0)


def shape_node_bounds(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    geometry: Optional[NativeShapeGeometry],
    *,
    directions: int = 64,
) -> torch.Tensor:
    """Return derived AABBs that tightly wrap each true supported shape.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    geometry : NativeShapeGeometry or None
        Optional per-node shape descriptors.
    directions : int, default=64
        Unused sampling count retained for future polygonal extensions.

    Returns
    -------
    torch.Tensor
        Bounds tensor with shape ``[N, 4]``.
    """
    del directions
    sizes = node_sizes.to(device=pos.device, dtype=pos.dtype)
    half = sizes * 0.5
    if geometry is None:
        return torch.cat((pos - half, pos + half), dim=1)
    codes = geometry.kind_codes.to(device=pos.device, dtype=torch.long)
    box_half = half
    ellipse_half = half
    diamond_half = half
    triangle_half = half
    x_half = torch.where(
        codes == 1,
        ellipse_half[:, 0],
        torch.where(
            codes == 2,
            diamond_half[:, 0],
            torch.where(codes == 3, triangle_half[:, 0], box_half[:, 0]),
        ),
    )
    y_half = torch.where(
        codes == 1,
        ellipse_half[:, 1],
        torch.where(
            codes == 2,
            diamond_half[:, 1],
            torch.where(codes == 3, triangle_half[:, 1], box_half[:, 1]),
        ),
    )
    half_bounds = torch.stack((x_half, y_half), dim=1)
    return torch.cat((pos - half_bounds, pos + half_bounds), dim=1)


def _sample_nodes(node_count: int, max_nodes: int, device: torch.device) -> torch.Tensor:
    """Return deterministic sampled node ids.

    Parameters
    ----------
    node_count : int
        Total number of nodes.
    max_nodes : int
        Maximum nodes to sample.
    device : torch.device
        Output tensor device.

    Returns
    -------
    torch.Tensor
        Long node ids with shape ``[min(node_count, max_nodes)]``.
    """
    if node_count <= max_nodes:
        return torch.arange(node_count, dtype=torch.long, device=device)
    return (
        torch.linspace(0, node_count - 1, steps=max_nodes, dtype=torch.float32, device=device)
        .round()
        .to(dtype=torch.long)
        .clamp(0, node_count - 1)
    )


def _triangle_support_radius(directions: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    """Return support radius for an upright triangle centered at the origin.

    Parameters
    ----------
    directions : torch.Tensor
        Direction vectors with shape ``[..., 2]``.
    half : torch.Tensor
        Half sizes with shape ``[..., 2]``.

    Returns
    -------
    torch.Tensor
        Support radii with shape ``directions.shape[:-1]``.
    """
    vertices = torch.stack(
        (
            torch.stack((torch.zeros_like(half[..., 0]), half[..., 1]), dim=-1),
            torch.stack((-half[..., 0], -half[..., 1]), dim=-1),
            torch.stack((half[..., 0], -half[..., 1]), dim=-1),
        ),
        dim=-2,
    )
    return (vertices * directions.unsqueeze(-2)).sum(dim=-1).amax(dim=-1).clamp_min(_EPS)

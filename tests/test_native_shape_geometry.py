"""Tests for native shape-aware geometry helpers."""

from __future__ import annotations

import torch
from pytest import MonkeyPatch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.dagua_native import (
    _dot_cluster_bbox,
    layout_dagua_native_pipeline,
)
from dagua.layout.ops.pipelines.native_finisher import (
    _overlap_count,
    _project_checkpoint_for_viability,
)
from dagua.layout.ops.pipelines.native_shape_geometry import (
    NativeShapeGeometry,
    pairwise_shape_signed_gap,
    resolve_native_shape_geometry,
    shape_node_bounds,
    shape_overlap_hinge_loss,
)


def _geometry(shapes: list[str]) -> NativeShapeGeometry:
    """Resolve test shape names into non-null native geometry.

    Parameters
    ----------
    shapes : list[str]
        Shape names aligned to synthetic nodes.

    Returns
    -------
    NativeShapeGeometry
        Resolved geometry descriptor.
    """
    geometry = resolve_native_shape_geometry(shapes, len(shapes))
    assert geometry is not None
    return geometry


def _min_gap(pos: torch.Tensor, sizes: torch.Tensor, shapes: list[str]) -> float:
    """Return the minimum signed true-shape gap for a small layout.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    shapes : list[str]
        Shape names aligned to nodes.

    Returns
    -------
    float
        Minimum signed shape gap.
    """
    gaps = pairwise_shape_signed_gap(pos, sizes, _geometry(shapes), max_nodes=pos.shape[0])
    return float(gaps.min().item()) if gaps.numel() else 0.0


def test_shape_signed_gaps_match_basic_support_geometry() -> None:
    """Circle, ellipse, diamond, and box pairs use true support radii."""
    sizes = torch.tensor([[10.0, 10.0], [10.0, 10.0]], dtype=torch.float32)
    circles = torch.tensor([[0.0, 0.0], [9.0, 0.0]], dtype=torch.float32)
    circle_gap = pairwise_shape_signed_gap(circles, sizes, _geometry(["circle", "circle"]))
    assert torch.allclose(circle_gap, torch.tensor([-1.0]), atol=1.0e-3)

    ellipse_sizes = torch.tensor([[12.0, 4.0], [12.0, 4.0]], dtype=torch.float32)
    ellipses = torch.tensor([[0.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
    ellipse_gap = pairwise_shape_signed_gap(
        ellipses,
        ellipse_sizes,
        _geometry(["ellipse", "ellipse"]),
    )
    assert torch.allclose(ellipse_gap, torch.tensor([-1.0]), atol=1.0e-3)

    diamonds = torch.tensor([[0.0, 0.0], [8.0, 0.0]], dtype=torch.float32)
    diamond_gap = pairwise_shape_signed_gap(diamonds, sizes, _geometry(["diamond", "diamond"]))
    assert torch.allclose(diamond_gap, torch.tensor([-2.0]), atol=1.0e-3)

    boxes = torch.tensor([[0.0, 0.0], [9.0, 0.0]], dtype=torch.float32)
    box_geometry = NativeShapeGeometry(torch.zeros(2, dtype=torch.long))
    box_gap = pairwise_shape_signed_gap(boxes, sizes, box_geometry)
    assert torch.allclose(box_gap, torch.tensor([-1.0]), atol=1.0e-3)


def test_shape_overlap_loss_is_differentiable_by_finite_difference() -> None:
    """Autograd for shape overlap agrees with a centered finite difference."""
    pos = torch.tensor(
        [[0.0, 0.0], [8.5, 0.75], [18.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    sizes = torch.tensor([[10.0, 10.0], [12.0, 6.0], [8.0, 8.0]], dtype=torch.float64)
    geometry = _geometry(["circle", "ellipse", "diamond"])
    loss = shape_overlap_hinge_loss(pos, sizes, geometry)
    loss.backward()
    assert pos.grad is not None
    assert bool(torch.isfinite(pos.grad).all().item())

    epsilon = 1.0e-4
    plus = pos.detach().clone()
    minus = pos.detach().clone()
    plus[1, 0] += epsilon
    minus[1, 0] -= epsilon
    finite_diff = (
        shape_overlap_hinge_loss(plus, sizes, geometry)
        - shape_overlap_hinge_loss(minus, sizes, geometry)
    ) / (2.0 * epsilon)
    assert torch.allclose(pos.grad[1, 0], finite_diff, atol=1.0e-4, rtol=1.0e-3)


def test_shape_overlap_loss_is_nan_safe_at_coincident_positions() -> None:
    """Coincident non-box nodes produce finite loss and finite gradients."""
    pos = torch.zeros((3, 2), dtype=torch.float32, requires_grad=True)
    sizes = torch.tensor([[10.0, 10.0], [8.0, 6.0], [8.0, 8.0]], dtype=torch.float32)
    geometry = _geometry(["circle", "ellipse", "diamond"])
    loss = shape_overlap_hinge_loss(pos, sizes, geometry)
    loss.backward()
    assert bool(torch.isfinite(loss).item())
    assert pos.grad is not None
    assert bool(torch.isfinite(pos.grad).all().item())


def test_shape_projection_resolves_mixed_true_shape_overlaps() -> None:
    """W5 viability projection removes true-shape overlaps in a mixed graph."""
    pos = torch.tensor(
        [[0.0, 0.0], [5.0, 0.0], [8.5, 0.0], [12.0, 0.0]],
        dtype=torch.float32,
    )
    sizes = torch.tensor([[10.0, 10.0], [8.0, 8.0], [10.0, 6.0], [8.0, 8.0]])
    geometry = _geometry(["circle", "diamond", "box", "triangle"])
    projected = _project_checkpoint_for_viability(pos, sizes, geometry)
    assert _overlap_count(projected, sizes, geometry) == 0
    assert _min_gap(projected, sizes, ["circle", "diamond", "box", "triangle"]) >= -1.0e-3


def test_shape_derived_cluster_bounds_use_true_shape_bounds() -> None:
    """Cluster boxes can be derived from true-shape member bounds."""
    pos = torch.tensor([[0.0, 0.0], [20.0, 0.0]], dtype=torch.float32)
    sizes = torch.tensor([[10.0, 10.0], [12.0, 6.0]], dtype=torch.float32)
    geometry = _geometry(["circle", "ellipse"])
    bounds = shape_node_bounds(pos, sizes, geometry)
    cluster_box = _dot_cluster_bbox(pos, sizes, [0, 1], 2.0, bounds)
    assert cluster_box == (-7.0, -7.0, 28.0, 7.0)


def test_box_only_native_pipeline_is_byte_identical_with_shape_metadata(
    monkeypatch: MonkeyPatch,
) -> None:
    """Rectangle metadata keeps the native default path byte-identical."""
    monkeypatch.setenv("DAGUA_NATIVE_DISABLE_W5", "1")
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    sizes = torch.full((4, 2), 10.0, dtype=torch.float32)
    config = LayoutConfig(seed=7, steps=8)
    without_shapes = layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=sizes,
        config=config,
        device="cpu",
    )
    with_shapes = layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=sizes,
        config=LayoutConfig(seed=7, steps=8),
        device="cpu",
        node_shapes=["box", "rectangle", "rect", "roundrect"],
    )
    assert torch.equal(with_shapes, without_shapes)


def test_default_ellipse_cascade_keeps_native_pipeline_byte_identical(
    monkeypatch: MonkeyPatch,
) -> None:
    """Default-resolved ellipse styles do not activate shape-aware native math."""
    monkeypatch.setenv("DAGUA_NATIVE_DISABLE_W5", "1")
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    sizes = torch.full((4, 2), 10.0, dtype=torch.float32)
    without_shapes = layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=sizes,
        config=LayoutConfig(seed=7, steps=8),
        device="cpu",
    )
    with_default_cascade = layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=sizes,
        config=LayoutConfig(seed=7, steps=8),
        device="cpu",
        node_shapes=["ellipse", "ellipse", "ellipse", "ellipse"],
    )
    assert torch.equal(with_default_cascade, without_shapes)

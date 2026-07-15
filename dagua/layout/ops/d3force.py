"""d3-force-compatible composable layout operations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_LCG_A = 1_664_525
_LCG_C = 1_013_904_223
_LCG_M = 4_294_967_296
_INITIAL_RADIUS = 10.0
_INITIAL_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
_JIGGLE_SCALE = 1.0e-6
_DEFAULT_ALPHA_MIN = 0.001
_DEFAULT_ALPHA_TARGET = 0.0
_DEFAULT_MANY_BODY_STRENGTH = -30.0
_DEFAULT_LINK_DISTANCE = 30.0
_DEFAULT_VELOCITY_DECAY_FACTOR = 0.6
_DEFAULT_THETA = 0.9
_D3_DISTANCE_MIN2 = 1.0
_D3_DISTANCE_MAX2 = math.inf


@dataclass
class _D3QuadtreeLeaf:
    """Leaf node in a d3-quadtree-compatible tree.

    Parameters
    ----------
    data : int
        Node index stored in the leaf.
    x : float
        Current x-coordinate for the node.
    y : float
        Current y-coordinate for the node.
    next : _D3QuadtreeLeaf, optional
        Head-linked coincident leaf inserted by d3-quadtree.
    value : float, default=0.0
        Accumulated many-body strength assigned by ``forceManyBody``.
    centroid_x : float, default=0.0
        d3 ``quad.x`` value after charge accumulation.
    centroid_y : float, default=0.0
        d3 ``quad.y`` value after charge accumulation.
    """

    data: int
    x: float
    y: float
    next: Optional["_D3QuadtreeLeaf"] = None
    value: float = 0.0
    centroid_x: float = 0.0
    centroid_y: float = 0.0


@dataclass
class _D3QuadtreeInternal:
    """Internal node in a d3-quadtree-compatible tree.

    Parameters
    ----------
    children : list[_D3QuadtreeNode | None]
        Four child quadrants in d3 order: top-left, top-right, bottom-left,
        bottom-right.
    value : float, default=0.0
        Accumulated many-body strength assigned by ``forceManyBody``.
    centroid_x : float, default=0.0
        d3 ``quad.x`` value after charge accumulation.
    centroid_y : float, default=0.0
        d3 ``quad.y`` value after charge accumulation.
    """

    children: list[Optional["_D3QuadtreeNode"]] = field(
        default_factory=lambda: [None, None, None, None]
    )
    value: float = 0.0
    centroid_x: float = 0.0
    centroid_y: float = 0.0


_D3QuadtreeNode = Union[_D3QuadtreeLeaf, _D3QuadtreeInternal]


@dataclass
class _D3QuadFrame:
    """Traversal frame matching d3-quadtree's ``Quad`` helper.

    Parameters
    ----------
    node : _D3QuadtreeNode
        Quadtree node for this frame.
    x0 : float
        Left bound of the square extent.
    y0 : float
        Top bound of the square extent.
    x1 : float
        Right bound of the square extent.
    y1 : float
        Bottom bound of the square extent.
    """

    node: _D3QuadtreeNode
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class _D3Quadtree:
    """Small Python port of d3-quadtree's addAll, cover, visit, and visitAfter.

    Parameters
    ----------
    x0 : float, default=nan
        Left extent bound.
    y0 : float, default=nan
        Top extent bound.
    x1 : float, default=nan
        Right extent bound.
    y1 : float, default=nan
        Bottom extent bound.
    root : _D3QuadtreeNode, optional
        Root node, or ``None`` for an empty tree.
    """

    x0: float = math.nan
    y0: float = math.nan
    x1: float = math.nan
    y1: float = math.nan
    root: Optional[_D3QuadtreeNode] = None

    @classmethod
    def from_positions(cls, positions: list[tuple[float, float]]) -> "_D3Quadtree":
        """Build a d3-quadtree from positions in array order.

        Parameters
        ----------
        positions : list[tuple[float, float]]
            Node coordinates in d3 node-array order.

        Returns
        -------
        _D3Quadtree
            Tree with the same topology as ``quadtree(nodes, x, y)``.
        """
        tree = cls()
        tree.add_all(positions)
        return tree

    def add_all(self, positions: list[tuple[float, float]]) -> "_D3Quadtree":
        """Add all positions using d3-quadtree's precomputed extent path.

        Parameters
        ----------
        positions : list[tuple[float, float]]
            Node coordinates in d3 node-array order.

        Returns
        -------
        _D3Quadtree
            This tree after insertion.
        """
        x_values: list[float] = [math.nan] * len(positions)
        y_values: list[float] = [math.nan] * len(positions)
        x0 = math.inf
        y0 = math.inf
        x1 = -math.inf
        y1 = -math.inf
        for index, (x, y) in enumerate(positions):
            if math.isnan(x) or math.isnan(y):
                continue
            x_values[index] = x
            y_values[index] = y
            if x < x0:
                x0 = x
            if x > x1:
                x1 = x
            if y < y0:
                y0 = y
            if y > y1:
                y1 = y
        if x0 > x1 or y0 > y1:
            return self
        self.cover(x0, y0).cover(x1, y1)
        for index, x in enumerate(x_values):
            self._add(x, y_values[index], index)
        return self

    def cover(self, x: float, y: float) -> "_D3Quadtree":
        """Expand the tree's square extent to cover one point.

        Parameters
        ----------
        x : float
            X-coordinate to cover.
        y : float
            Y-coordinate to cover.

        Returns
        -------
        _D3Quadtree
            This tree after extent expansion.
        """
        if math.isnan(x) or math.isnan(y):
            return self
        x0 = self.x0
        y0 = self.y0
        x1 = self.x1
        y1 = self.y1
        if math.isnan(x0):
            x0 = math.floor(x)
            y0 = math.floor(y)
            x1 = x0 + 1.0
            y1 = y0 + 1.0
        else:
            z = x1 - x0 or 1.0
            node = self.root
            while x0 > x or x >= x1 or y0 > y or y >= y1:
                index = (int(y < y0) << 1) | int(x < x0)
                parent = _D3QuadtreeInternal()
                parent.children[index] = node
                node = parent
                z *= 2.0
                if index == 0:
                    x1 = x0 + z
                    y1 = y0 + z
                elif index == 1:
                    x0 = x1 - z
                    y1 = y0 + z
                elif index == 2:
                    x1 = x0 + z
                    y0 = y1 - z
                else:
                    x0 = x1 - z
                    y0 = y1 - z
            if isinstance(self.root, _D3QuadtreeInternal):
                self.root = node
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        return self

    def _add(self, x: float, y: float, data: int) -> "_D3Quadtree":
        """Insert one point with d3-quadtree's quadrant and coincident rules.

        Parameters
        ----------
        x : float
            X-coordinate to insert.
        y : float
            Y-coordinate to insert.
        data : int
            Node index associated with the point.

        Returns
        -------
        _D3Quadtree
            This tree after insertion.
        """
        if math.isnan(x) or math.isnan(y):
            return self
        leaf = _D3QuadtreeLeaf(data=data, x=x, y=y)
        node = self.root
        if node is None:
            self.root = leaf
            return self

        parent: Optional[_D3QuadtreeInternal] = None
        child_index = 0
        x0 = self.x0
        y0 = self.y0
        x1 = self.x1
        y1 = self.y1
        while isinstance(node, _D3QuadtreeInternal):
            xm = (x0 + x1) / 2.0
            ym = (y0 + y1) / 2.0
            right = x >= xm
            bottom = y >= ym
            if right:
                x0 = xm
            else:
                x1 = xm
            if bottom:
                y0 = ym
            else:
                y1 = ym
            parent = node
            child_index = (int(bottom) << 1) | int(right)
            child = node.children[child_index]
            if child is None:
                node.children[child_index] = leaf
                return self
            node = child

        xp = node.x
        yp = node.y
        if x == xp and y == yp:
            leaf.next = node
            if parent is None:
                self.root = leaf
            else:
                parent.children[child_index] = leaf
            return self

        while True:
            new_parent = _D3QuadtreeInternal()
            if parent is None:
                self.root = new_parent
            else:
                parent.children[child_index] = new_parent
            parent = new_parent
            xm = (x0 + x1) / 2.0
            ym = (y0 + y1) / 2.0
            right = x >= xm
            bottom = y >= ym
            if right:
                x0 = xm
            else:
                x1 = xm
            if bottom:
                y0 = ym
            else:
                y1 = ym
            child_index = (int(bottom) << 1) | int(right)
            existing_index = (int(yp >= ym) << 1) | int(xp >= xm)
            if child_index != existing_index:
                parent.children[existing_index] = node
                parent.children[child_index] = leaf
                return self

    def visit_after(self, strengths: list[float]) -> None:
        """Run d3-quadtree ``visitAfter`` for many-body charge accumulation.

        Parameters
        ----------
        strengths : list[float]
            Per-node many-body strengths indexed by node index.

        Returns
        -------
        None
            Nodes are mutated with ``value`` and weighted centroid fields.
        """
        quads: list[_D3QuadFrame] = []
        next_frames: list[_D3QuadFrame] = []
        if self.root is not None:
            quads.append(_D3QuadFrame(self.root, self.x0, self.y0, self.x1, self.y1))
        while quads:
            frame = quads.pop()
            node = frame.node
            if isinstance(node, _D3QuadtreeInternal):
                x0 = frame.x0
                y0 = frame.y0
                x1 = frame.x1
                y1 = frame.y1
                xm = (x0 + x1) / 2.0
                ym = (y0 + y1) / 2.0
                child = node.children[0]
                if child is not None:
                    quads.append(_D3QuadFrame(child, x0, y0, xm, ym))
                child = node.children[1]
                if child is not None:
                    quads.append(_D3QuadFrame(child, xm, y0, x1, ym))
                child = node.children[2]
                if child is not None:
                    quads.append(_D3QuadFrame(child, x0, ym, xm, y1))
                child = node.children[3]
                if child is not None:
                    quads.append(_D3QuadFrame(child, xm, ym, x1, y1))
            next_frames.append(frame)
        while next_frames:
            self._accumulate(next_frames.pop().node, strengths)

    def _accumulate(self, node: _D3QuadtreeNode, strengths: list[float]) -> None:
        """Accumulate d3-force many-body charge for one visited node.

        Parameters
        ----------
        node : _D3QuadtreeNode
            Leaf or internal quadtree node to mutate.
        strengths : list[float]
            Per-node many-body strengths indexed by node index.

        Returns
        -------
        None
            The node is updated in place.
        """
        strength = 0.0
        if isinstance(node, _D3QuadtreeInternal):
            x = 0.0
            y = 0.0
            weight = 0.0
            for child in node.children:
                if child is not None:
                    child_weight = abs(child.value)
                    if child_weight:
                        strength += child.value
                        weight += child_weight
                        x += child_weight * child.centroid_x
                        y += child_weight * child.centroid_y
            node.centroid_x = x / weight if weight else math.nan
            node.centroid_y = y / weight if weight else math.nan
        else:
            leaf: Optional[_D3QuadtreeLeaf] = node
            node.centroid_x = node.x
            node.centroid_y = node.y
            while leaf is not None:
                strength += strengths[leaf.data]
                leaf = leaf.next
        node.value = strength


def _d3force_apply_many_body(
    tree: _D3Quadtree,
    positions: list[tuple[float, float]],
    vx: list[float],
    vy: list[float],
    strengths: list[float],
    alpha: float,
    theta2: float,
    rng: D3ForceLCG,
) -> None:
    """Apply d3-force Barnes-Hut many-body updates to velocity arrays.

    Parameters
    ----------
    tree : _D3Quadtree
        Quadtree after ``visit_after`` charge accumulation.
    positions : list[tuple[float, float]]
        Current node coordinates in d3 node-array order.
    vx : list[float]
        Mutable x-velocity array.
    vy : list[float]
        Mutable y-velocity array.
    strengths : list[float]
        Per-node many-body strengths indexed by node index.
    alpha : float
        Current simulation alpha.
    theta2 : float
        Squared Barnes-Hut opening angle.
    rng : D3ForceLCG
        d3-compatible random source for coincident jiggle.

    Returns
    -------
    None
        Velocities are updated in place.
    """
    if tree.root is None:
        return
    for node_index, (node_x, node_y) in enumerate(positions):
        _d3force_apply_many_body_to_node(
            tree=tree,
            node_index=node_index,
            node_x=node_x,
            node_y=node_y,
            vx=vx,
            vy=vy,
            strengths=strengths,
            alpha=alpha,
            theta2=theta2,
            rng=rng,
        )


def _d3force_apply_many_body_to_node(
    tree: _D3Quadtree,
    node_index: int,
    node_x: float,
    node_y: float,
    vx: list[float],
    vy: list[float],
    strengths: list[float],
    alpha: float,
    theta2: float,
    rng: D3ForceLCG,
) -> None:
    """Apply d3-quadtree ``visit(apply)`` for one simulation node.

    Parameters
    ----------
    tree : _D3Quadtree
        Quadtree after charge accumulation.
    node_index : int
        Index of the node receiving force.
    node_x : float
        Current x-coordinate of the receiving node.
    node_y : float
        Current y-coordinate of the receiving node.
    vx : list[float]
        Mutable x-velocity array.
    vy : list[float]
        Mutable y-velocity array.
    strengths : list[float]
        Per-node many-body strengths indexed by node index.
    alpha : float
        Current simulation alpha.
    theta2 : float
        Squared Barnes-Hut opening angle.
    rng : D3ForceLCG
        d3-compatible random source for coincident jiggle.

    Returns
    -------
    None
        ``vx`` and ``vy`` are updated in place for ``node_index``.
    """
    if tree.root is None:
        return
    quads = [_D3QuadFrame(tree.root, tree.x0, tree.y0, tree.x1, tree.y1)]
    while quads:
        frame = quads.pop()
        quad = frame.node
        if not quad.value:
            continue

        x = quad.centroid_x - node_x
        y = quad.centroid_y - node_y
        width = frame.x1 - frame.x0
        length2 = x * x + y * y

        if width * width / theta2 < length2:
            if length2 < _D3_DISTANCE_MAX2:
                x, y, length2 = _d3force_jiggle_and_bound_length(x, y, length2, rng)
                vx[node_index] += x * quad.value * alpha / length2
                vy[node_index] += y * quad.value * alpha / length2
            continue

        if isinstance(quad, _D3QuadtreeInternal):
            _d3force_push_visit_children(quads, quad, frame)
            continue

        if length2 >= _D3_DISTANCE_MAX2:
            continue
        if quad.data != node_index or quad.next is not None:
            x, y, length2 = _d3force_jiggle_and_bound_length(x, y, length2, rng)
        leaf: Optional[_D3QuadtreeLeaf] = quad
        while leaf is not None:
            if leaf.data != node_index:
                scale = strengths[leaf.data] * alpha / length2
                vx[node_index] += x * scale
                vy[node_index] += y * scale
            leaf = leaf.next


def _d3force_push_visit_children(
    quads: list[_D3QuadFrame],
    quad: _D3QuadtreeInternal,
    frame: _D3QuadFrame,
) -> None:
    """Push child quadrants in d3-quadtree ``visit`` stack order.

    Parameters
    ----------
    quads : list[_D3QuadFrame]
        Mutable traversal stack.
    quad : _D3QuadtreeInternal
        Internal node whose children should be visited.
    frame : _D3QuadFrame
        Bounds for ``quad``.

    Returns
    -------
    None
        Child frames are appended to ``quads`` in d3's push order.
    """
    x0 = frame.x0
    y0 = frame.y0
    x1 = frame.x1
    y1 = frame.y1
    xm = (x0 + x1) / 2.0
    ym = (y0 + y1) / 2.0
    child = quad.children[3]
    if child is not None:
        quads.append(_D3QuadFrame(child, xm, ym, x1, y1))
    child = quad.children[2]
    if child is not None:
        quads.append(_D3QuadFrame(child, x0, ym, xm, y1))
    child = quad.children[1]
    if child is not None:
        quads.append(_D3QuadFrame(child, xm, y0, x1, ym))
    child = quad.children[0]
    if child is not None:
        quads.append(_D3QuadFrame(child, x0, y0, xm, ym))


def _d3force_jiggle_and_bound_length(
    x: float,
    y: float,
    length2: float,
    rng: D3ForceLCG,
) -> tuple[float, float, float]:
    """Apply d3-force coincident jiggle and minimum-distance bound.

    Parameters
    ----------
    x : float
        X displacement.
    y : float
        Y displacement.
    length2 : float
        Squared displacement length.
    rng : D3ForceLCG
        d3-compatible random source.

    Returns
    -------
    tuple[float, float, float]
        Possibly jiggled ``x``, ``y``, and bounded squared-length denominator.
    """
    if x == 0.0:
        x = rng.jiggle()
        length2 += x * x
    if y == 0.0:
        y = rng.jiggle()
        length2 += y * y
    if length2 < _D3_DISTANCE_MIN2:
        length2 = math.sqrt(_D3_DISTANCE_MIN2 * length2)
    return x, y, length2


def _d3force_quadtree_accumulation_rows(
    positions: list[tuple[float, float]],
    strength: float = _DEFAULT_MANY_BODY_STRENGTH,
) -> list[tuple[str, int, float, float, float, float, float, float, float]]:
    """Return a deterministic dump of d3-force quadtree accumulated cells.

    Parameters
    ----------
    positions : list[tuple[float, float]]
        Node coordinates in d3 node-array order.
    strength : float, default=-30.0
        Constant many-body strength to assign to each node.

    Returns
    -------
    list[tuple[str, int, float, float, float, float, float, float, float]]
        Rows containing node kind, leaf data index or ``-1``, centroid x,
        centroid y, charge value, and frame bounds ``x0, y0, x1, y1``.
    """
    tree = _D3Quadtree.from_positions(positions)
    tree.visit_after([strength] * len(positions))
    rows: list[tuple[str, int, float, float, float, float, float, float, float]] = []
    if tree.root is None:
        return rows
    _d3force_collect_quadtree_rows(
        rows=rows,
        node=tree.root,
        x0=tree.x0,
        y0=tree.y0,
        x1=tree.x1,
        y1=tree.y1,
    )
    return rows


def _d3force_collect_quadtree_rows(
    rows: list[tuple[str, int, float, float, float, float, float, float, float]],
    node: _D3QuadtreeNode,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> None:
    """Append pre-order quadtree rows for Node reference comparisons.

    Parameters
    ----------
    rows : list[tuple[str, int, float, float, float, float, float, float, float]]
        Mutable row accumulator.
    node : _D3QuadtreeNode
        Current quadtree node.
    x0 : float
        Left frame bound.
    y0 : float
        Top frame bound.
    x1 : float
        Right frame bound.
    y1 : float
        Bottom frame bound.

    Returns
    -------
    None
        Rows are appended in place.
    """
    if isinstance(node, _D3QuadtreeInternal):
        rows.append(("internal", -1, node.centroid_x, node.centroid_y, node.value, x0, y0, x1, y1))
        xm = (x0 + x1) / 2.0
        ym = (y0 + y1) / 2.0
        child = node.children[0]
        if child is not None:
            _d3force_collect_quadtree_rows(rows, child, x0, y0, xm, ym)
        child = node.children[1]
        if child is not None:
            _d3force_collect_quadtree_rows(rows, child, xm, y0, x1, ym)
        child = node.children[2]
        if child is not None:
            _d3force_collect_quadtree_rows(rows, child, x0, ym, xm, y1)
        child = node.children[3]
        if child is not None:
            _d3force_collect_quadtree_rows(rows, child, xm, ym, x1, y1)
        return
    leaf: Optional[_D3QuadtreeLeaf] = node
    while leaf is not None:
        rows.append(
            ("leaf", leaf.data, node.centroid_x, node.centroid_y, node.value, x0, y0, x1, y1)
        )
        leaf = leaf.next


class D3ForceLCG:
    """Linear congruential generator used by d3-force.

    Parameters
    ----------
    seed : int, default=1
        Initial unsigned 32-bit state. d3-force's built-in source starts at
        ``1``; exposing the seed mirrors ``simulation.randomSource``.
    """

    def __init__(self, seed: int = 1) -> None:
        self.state = int(seed) % _LCG_M

    def random(self) -> float:
        """Return the next d3-force LCG value.

        Returns
        -------
        float
            Uniform value in ``[0, 1)`` computed as ``state / 2**32`` after
            the d3-force LCG update.
        """
        self.state = (_LCG_A * self.state + _LCG_C) % _LCG_M
        return self.state / _LCG_M

    def jiggle(self) -> float:
        """Return d3-force's tiny coincident-point perturbation.

        Returns
        -------
        float
            ``(random() - 0.5) * 1e-6``.
        """
        return (self.random() - 0.5) * _JIGGLE_SCALE


def d3force_lcg_values(seed: int = 1, count: int = 20) -> List[float]:
    """Generate d3-force LCG values for tests and verification.

    Parameters
    ----------
    seed : int, default=1
        Initial unsigned 32-bit LCG state.
    count : int, default=20
        Number of values to return.

    Returns
    -------
    list[float]
        First ``count`` generated values.
    """
    rng = D3ForceLCG(seed=seed)
    return [rng.random() for _ in range(count)]


def d3force_phyllotaxis_positions(
    num_nodes: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return d3-force's initial phyllotaxis spiral coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of nodes to initialize.
    dtype : torch.dtype, default=torch.float64
        Output tensor dtype.
    device : torch.device, optional
        Output device. ``None`` uses CPU.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    resolved_device = torch.device("cpu") if device is None else device
    pos = torch.zeros((num_nodes, 2), dtype=dtype, device=resolved_device)
    for index in range(num_nodes):
        radius = _INITIAL_RADIUS * math.sqrt(0.5 + float(index))
        angle = float(index) * _INITIAL_ANGLE
        pos[index, 0] = radius * math.cos(angle)
        pos[index, 1] = radius * math.sin(angle)
    return pos


def _edge_pairs(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    """Convert edge-index tensor to stable Python edge pairs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    list[tuple[int, int]]
        Edge pairs in input column order.
    """
    if edge_index.numel() == 0:
        return []
    return [(int(source), int(target)) for source, target in edge_index.cpu().t().tolist()]


@dataclass(frozen=True)
class D3ForceConfig:
    """Configuration for d3-force-compatible operations.

    Parameters
    ----------
    ticks : int, default=300
        Number of simulation ticks.
    seed : int, default=1
        LCG seed. ``1`` matches d3-force's default source.
    many_body_strength : float, default=-30.0
        Constant charge strength for ``forceManyBody``.
    link_distance : float, default=30.0
        Constant link distance for ``forceLink``.
    link_iterations : int, default=1
        Number of link relaxation passes per tick.
    velocity_decay_factor : float, default=0.6
        Internal multiplier used during velocity Verlet integration. This is
        d3's ``1 - simulation.velocityDecay()`` value.
    theta : float, default=0.9
        Barnes-Hut theta used by d3-force ``forceManyBody``. d3's default is
        ``0.9``, so the squared cutoff is ``0.81``.
    center : bool, default=True
        Whether to apply ``forceCenter(0, 0)``.
    """

    ticks: int = 300
    seed: int = 1
    many_body_strength: float = _DEFAULT_MANY_BODY_STRENGTH
    link_distance: float = 30.0
    link_iterations: int = 1
    velocity_decay_factor: float = _DEFAULT_VELOCITY_DECAY_FACTOR
    theta: float = _DEFAULT_THETA
    center: bool = True


@register_op
class D3ForceInitialize(Op):
    """Initialize positions, velocities, alpha, RNG, and link metadata."""

    name = "d3force_initialize"
    category = OpCategory.INIT
    writes = ("pos", "extras")

    def __init__(self, config: D3ForceConfig, dtype: torch.dtype = torch.float64) -> None:
        self.config = config
        self.dtype = dtype

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Initialize d3-force simulation state.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable state receiving ``pos`` and d3 extras.
        ctx : RuntimeContext
            Execution context providing the requested device.

        Returns
        -------
        SolveState
            State populated with d3-force working fields.
        """
        device = torch.device(ctx.plan.device or "cpu")
        state.pos = d3force_phyllotaxis_positions(problem.num_nodes, self.dtype, device)
        state.extras["d3force_vx"] = [0.0] * problem.num_nodes
        state.extras["d3force_vy"] = [0.0] * problem.num_nodes
        state.extras["d3force_alpha"] = 1.0
        state.extras["d3force_alpha_decay"] = 1.0 - math.pow(_DEFAULT_ALPHA_MIN, 1.0 / 300.0)
        state.extras["d3force_rng"] = D3ForceLCG(seed=self.config.seed)
        state.extras["d3force_edges"] = _edge_pairs(problem.edge_index)
        state.extras["d3force_link_count"] = self._link_counts(problem)
        return state

    def _link_counts(self, problem: LayoutProblem) -> list[int]:
        """Return d3-force link endpoint counts.

        Parameters
        ----------
        problem : LayoutProblem
            Graph inputs containing edge-index tensor.

        Returns
        -------
        list[int]
            Per-node incident link counts.
        """
        counts = [0] * problem.num_nodes
        for source, target in _edge_pairs(problem.edge_index):
            counts[source] += 1
            counts[target] += 1
        return counts


@register_op
class D3ForceUpdateAlpha(Op):
    """Apply d3-force alpha decay for one tick."""

    name = "d3force_update_alpha"
    category = OpCategory.ANNEAL
    reads = ("extras",)
    writes = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Advance simulation alpha.

        Parameters
        ----------
        problem : LayoutProblem
            Unused immutable graph inputs.
        state : SolveState
            State containing d3-force alpha extras.
        ctx : RuntimeContext
            Unused execution context.

        Returns
        -------
        SolveState
            State with updated ``d3force_alpha``.
        """
        del problem, ctx
        alpha = float(state.extras["d3force_alpha"])
        alpha_decay = float(state.extras["d3force_alpha_decay"])
        state.extras["d3force_alpha"] = alpha + (_DEFAULT_ALPHA_TARGET - alpha) * alpha_decay
        return state


@register_op
class D3ForceLink(Op):
    """Apply d3-force ``forceLink`` velocity updates."""

    name = "d3force_link"
    category = OpCategory.FORCE
    reads = ("pos", "extras")
    writes = ("extras",)

    def __init__(self, config: D3ForceConfig) -> None:
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Relax links in d3-force edge order.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            State with positions, velocities, and link metadata.
        ctx : RuntimeContext
            Unused execution context.

        Returns
        -------
        SolveState
            State with updated velocity extras.
        """
        del problem, ctx
        if state.pos is None:
            return state
        pos = state.pos.detach().cpu().numpy()
        vx = state.extras["d3force_vx"]
        vy = state.extras["d3force_vy"]
        rng: D3ForceLCG = state.extras["d3force_rng"]
        alpha = float(state.extras["d3force_alpha"])
        counts = state.extras["d3force_link_count"]
        edges: list[tuple[int, int]] = state.extras["d3force_edges"]
        for _ in range(max(0, int(self.config.link_iterations))):
            for source, target in edges:
                x = float(pos[target, 0]) + vx[target] - float(pos[source, 0]) - vx[source]
                y = float(pos[target, 1]) + vy[target] - float(pos[source, 1]) - vy[source]
                if x == 0.0:
                    x = rng.jiggle()
                if y == 0.0:
                    y = rng.jiggle()
                length = math.sqrt(x * x + y * y)
                strength = 1.0 / float(min(counts[source], counts[target]))
                scale = (length - self.config.link_distance) / length * alpha * strength
                dx = x * scale
                dy = y * scale
                bias = counts[source] / float(counts[source] + counts[target])
                vx[target] -= dx * bias
                vy[target] -= dy * bias
                source_bias = 1.0 - bias
                vx[source] += dx * source_bias
                vy[source] += dy * source_bias
        return state


@register_op
class D3ForceManyBody(Op):
    """Apply d3-force ``forceManyBody`` Barnes-Hut velocity updates."""

    name = "d3force_many_body"
    category = OpCategory.FORCE
    reads = ("pos", "extras")
    writes = ("extras",)

    def __init__(self, config: D3ForceConfig) -> None:
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply Barnes-Hut many-body repulsion in d3 traversal order.

        Parameters
        ----------
        problem : LayoutProblem
            Graph inputs supplying the node count.
        state : SolveState
            State with positions, velocities, alpha, and RNG.
        ctx : RuntimeContext
            Unused execution context.

        Returns
        -------
        SolveState
            State with updated velocity extras.

        """
        del ctx
        if state.pos is None:
            return state
        pos = state.pos.detach().cpu().numpy()
        vx = state.extras["d3force_vx"]
        vy = state.extras["d3force_vy"]
        rng: D3ForceLCG = state.extras["d3force_rng"]
        alpha = float(state.extras["d3force_alpha"])
        strength = float(self.config.many_body_strength)
        positions = [
            (float(pos[node, 0]), float(pos[node, 1])) for node in range(problem.num_nodes)
        ]
        strengths = [strength] * problem.num_nodes
        tree = _D3Quadtree.from_positions(positions)
        tree.visit_after(strengths)
        _d3force_apply_many_body(
            tree=tree,
            positions=positions,
            vx=vx,
            vy=vy,
            strengths=strengths,
            alpha=alpha,
            theta2=float(self.config.theta) * float(self.config.theta),
            rng=rng,
        )
        return state


@register_op
class D3ForceCenter(Op):
    """Apply d3-force ``forceCenter(0, 0)``."""

    name = "d3force_center"
    category = OpCategory.POSTPROCESS
    reads = ("pos",)
    writes = ("pos",)

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Shift the current centroid to the origin.

        Parameters
        ----------
        problem : LayoutProblem
            Graph inputs supplying the node count.
        state : SolveState
            State containing position tensor ``[N, 2]``.
        ctx : RuntimeContext
            Unused execution context.

        Returns
        -------
        SolveState
            State with centered positions when enabled.
        """
        del ctx
        if not self.enabled or state.pos is None or problem.num_nodes == 0:
            return state
        state.pos = state.pos - state.pos.mean(dim=0, keepdim=True)
        return state


@register_op
class D3ForceIntegrate(Op):
    """Apply d3-force velocity Verlet integration."""

    name = "d3force_integrate"
    category = OpCategory.OPTIMIZE
    reads = ("pos", "extras")
    writes = ("pos", "extras")

    def __init__(self, config: D3ForceConfig) -> None:
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Advance positions from velocity state.

        Parameters
        ----------
        problem : LayoutProblem
            Graph inputs supplying the node count.
        state : SolveState
            State with position tensor and velocity lists.
        ctx : RuntimeContext
            Unused execution context.

        Returns
        -------
        SolveState
            State with integrated positions and decayed velocities.
        """
        del ctx
        if state.pos is None:
            return state
        vx = state.extras["d3force_vx"]
        vy = state.extras["d3force_vy"]
        for node in range(problem.num_nodes):
            vx[node] *= self.config.velocity_decay_factor
            vy[node] *= self.config.velocity_decay_factor
            state.pos[node, 0] += vx[node]
            state.pos[node, 1] += vy[node]
        return state

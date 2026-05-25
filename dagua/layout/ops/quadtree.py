"""Graphviz-compatible sparse quadtree utilities.

The implementation in this module ports ``lib/sparse/QuadTree.c`` closely
enough for SFDP/FDP fidelity work.  It intentionally keeps Graphviz's insertion
order, quadrant indexing, leaf linked-list behavior, and cell-level force
accumulation semantics rather than using the faster default Dagua tree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

_GRAPHVIZ_MACHINEACC = 1.0e-16
_GRAPHVIZ_MINDIST = 1.0e-15
_GRAPHVIZ_WIDTH_FLOOR = 1.0e-5
_GRAPHVIZ_WIDTH_SCALE = 0.52
_GRAPHVIZ_REALLOC_INCREMENT = 10
_GRAPHVIZ_SFDP_QUADTREE_SIZE = 45


@dataclass
class GraphvizQuadTreeLeaf:
    """Leaf-list payload matching Graphviz ``node_data``.

    Parameters
    ----------
    node_weight : float
        Weight associated with this point.
    coord : list[float]
        Copied point coordinates with length ``dim``.
    id : int
        Original node index.
    data : list[float], optional
        Per-node force buffer used during Graphviz-style accumulation.
    next : GraphvizQuadTreeLeaf, optional
        Next entry in the leaf linked list. New max-depth points are pushed to
        the head, matching the C implementation.
    """

    node_weight: float
    coord: List[float]
    id: int
    data: Optional[List[float]] = None
    next: Optional["GraphvizQuadTreeLeaf"] = None


@dataclass
class GraphvizQuadTree:
    """Sparse quadtree node matching Graphviz ``QuadTree``.

    Parameters
    ----------
    dim : int
        Coordinate dimensionality.
    center : list[float]
        Center of the square/cube cell with length ``dim``.
    width : float
        Cell radius. The cell bounds are ``center +/- width``.
    max_level : int
        Maximum subdivision depth.
    n : int, default=0
        Number of items in this subtree.
    total_weight : float, default=0.0
        Sum of item weights in this subtree.
    average : list[float], optional
        Graphviz arithmetic average coordinates for this subtree.
    qts : list[GraphvizQuadTree | None], optional
        Child cells in Graphviz quadrant order.
    leaf_head : GraphvizQuadTreeLeaf, optional
        Leaf linked list. Internal nodes retain no leaf payload.
    data : list[float], optional
        Cell-level force accumulator used by ``get_repulsive_force``.
    """

    dim: int
    center: List[float]
    width: float
    max_level: int
    n: int = 0
    total_weight: float = 0.0
    average: Optional[List[float]] = None
    qts: Optional[List[Optional["GraphvizQuadTree"]]] = None
    leaf_head: Optional[GraphvizQuadTreeLeaf] = None
    data: Optional[List[float]] = None

    @classmethod
    def new(
        cls,
        dim: int,
        center: Sequence[float],
        width: float,
        max_level: int,
    ) -> "GraphvizQuadTree":
        """Create an empty Graphviz-compatible quadtree node.

        Parameters
        ----------
        dim : int
            Coordinate dimensionality.
        center : sequence of float
            Center coordinates with length ``dim``.
        width : float
            Cell radius. Must be positive.
        max_level : int
            Maximum subdivision depth.

        Returns
        -------
        GraphvizQuadTree
            Empty quadtree node.

        Raises
        ------
        ValueError
            If ``dim`` or ``width`` is invalid.
        """
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if len(center) != dim:
            raise ValueError("center length must match dim.")
        if width <= 0.0:
            raise ValueError("width must be positive.")
        return cls(
            dim=dim,
            center=[float(value) for value in center],
            width=float(width),
            max_level=max_level,
        )

    @classmethod
    def from_points(
        cls,
        coordinates: torch.Tensor,
        max_level: int,
        weights: Optional[torch.Tensor] = None,
    ) -> Optional["GraphvizQuadTree"]:
        """Build a Graphviz quadtree from a point tensor.

        Parameters
        ----------
        coordinates : torch.Tensor
            Point coordinates with shape ``[N, dim]``.
        max_level : int
            Maximum subdivision depth.
        weights : torch.Tensor, optional
            Point weights with shape ``[N]``. When omitted, Graphviz's point-list
            constructor behavior of unit weights is used.

        Returns
        -------
        GraphvizQuadTree | None
            Root quadtree node, or ``None`` for an empty point set.

        Raises
        ------
        ValueError
            If inputs have incompatible shapes.
        """
        rows = _coordinate_rows(coordinates)
        if not rows:
            return None

        if weights is None:
            weight_values = [1.0] * len(rows)
        else:
            if weights.ndim != 1 or weights.shape[0] != len(rows):
                raise ValueError("weights must have shape [N].")
            weight_values = [float(value) for value in weights.detach().cpu().tolist()]

        dim = len(rows[0])
        xmin = list(rows[0])
        xmax = list(rows[0])
        for row in rows[1:]:
            for axis in range(dim):
                xmin[axis] = min(xmin[axis], row[axis])
                xmax[axis] = max(xmax[axis], row[axis])

        width = xmax[0] - xmin[0]
        center: List[float] = []
        for axis in range(dim):
            center.append((xmin[axis] + xmax[axis]) * 0.5)
            width = max(width, xmax[axis] - xmin[axis])
        width = max(width, _GRAPHVIZ_WIDTH_FLOOR)
        width *= _GRAPHVIZ_WIDTH_SCALE

        tree = cls.new(dim=dim, center=center, width=width, max_level=max_level)
        for node_id, row in enumerate(rows):
            tree.add(coord=row, weight=weight_values[node_id], node_id=node_id)
        return tree

    def add(self, coord: Sequence[float], weight: float, node_id: int) -> "GraphvizQuadTree":
        """Insert one weighted point into the quadtree.

        Parameters
        ----------
        coord : sequence of float
            Point coordinates with length ``dim``.
        weight : float
            Point weight.
        node_id : int
            Original node index.

        Returns
        -------
        GraphvizQuadTree
            The current tree node, matching Graphviz's return convention.
        """
        if len(coord) != self.dim:
            raise ValueError("coord length must match dim.")
        self._add_internal(
            coord=[float(value) for value in coord],
            weight=float(weight),
            node_id=node_id,
            level=0,
        )
        return self

    def get_supernodes(
        self,
        bh: float,
        pt: Sequence[float],
        node_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Collect Graphviz Barnes-Hut supernodes for one query point.

        Parameters
        ----------
        bh : float
            Barnes-Hut opening coefficient.
        pt : sequence of float
            Query point coordinates with length ``dim``.
        node_id : int
            Node id excluded from leaf contributions.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
            ``(centers, weights, distances, counts)`` where centers has shape
            ``[M, dim]`` and counts is the number of recursive calls.
        """
        if len(pt) != self.dim:
            raise ValueError("pt length must match dim.")

        centers: List[List[float]] = []
        weights: List[float] = []
        distances: List[float] = []
        counts = _get_supernodes_internal(
            tree=self,
            bh=float(bh),
            pt=[float(value) for value in pt],
            node_id=node_id,
            centers=centers,
            weights=weights,
            distances=distances,
            counts=0.0,
        )
        center_tensor = torch.tensor(centers, dtype=torch.float64).reshape((-1, self.dim))
        return (
            center_tensor,
            torch.tensor(weights, dtype=torch.float64),
            torch.tensor(distances, dtype=torch.float64),
            counts,
        )

    def get_repulsive_force(
        self,
        coordinates: torch.Tensor,
        bh: float,
        p: float,
        kp: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Graphviz symmetric quadtree repulsive forces.

        Parameters
        ----------
        coordinates : torch.Tensor
            Current coordinates with shape ``[N, dim]``.
        bh : float
            Barnes-Hut opening coefficient.
        p : float
            Repulsive force power.
        kp : float
            Precomputed ``pow(K, 1 - p)`` multiplier.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Force tensor with shape ``[N, dim]`` and normalized Graphviz counts
            with shape ``[4]``.
        """
        rows = _coordinate_rows(coordinates)
        if rows and len(rows[0]) != self.dim:
            raise ValueError("coordinates dim must match quadtree dim.")

        force = [[0.0 for _ in range(self.dim)] for _ in range(self.n)]
        counts = [0.0, 0.0, 0.0, 0.0]
        self._clear_force_data()
        _repulsive_force_interact(
            tree1=self,
            tree2=self,
            rows=rows,
            force=force,
            bh=float(bh),
            p=float(p),
            kp=float(kp),
            counts=counts,
        )
        _repulsive_force_accumulate(tree=self, force=force, counts=counts)
        if self.n > 0:
            counts = [value / float(self.n) for value in counts]
        return (
            torch.tensor(force, dtype=coordinates.dtype, device=coordinates.device),
            torch.tensor(counts, dtype=torch.float64),
        )

    def get_nearest(self, point: Sequence[float]) -> Tuple[torch.Tensor, int, float]:
        """Find the nearest stored point using Graphviz's two-pass traversal.

        Parameters
        ----------
        point : sequence of float
            Query point coordinates with length ``dim``.

        Returns
        -------
        tuple[torch.Tensor, int, float]
            Nearest coordinates, nearest node id, and Euclidean distance. The id
            is ``-1`` and distance is ``-1.0`` when the tree is empty.
        """
        if len(point) != self.dim:
            raise ValueError("point length must match dim.")

        result = _NearestResult(
            coord=[0.0 for _ in range(self.dim)],
            node_id=-1,
            distance=-1.0,
        )
        query = [float(value) for value in point]
        _nearest_internal(tree=self, point=query, result=result, tentative=True)
        _nearest_internal(tree=self, point=query, result=result, tentative=False)
        return torch.tensor(result.coord, dtype=torch.float64), result.node_id, result.distance

    def _add_internal(
        self,
        coord: List[float],
        weight: float,
        node_id: int,
        level: int,
    ) -> None:
        """Insert one point using Graphviz's recursive insertion rules.

        Parameters
        ----------
        coord : list[float]
            Point coordinates with length ``dim``.
        weight : float
            Point weight.
        node_id : int
            Original node index.
        level : int
            Current tree depth.

        Returns
        -------
        None
            The tree is mutated in place.
        """
        _validate_within_graphviz_bounds(tree=self, coord=coord)

        if self.n == 0:
            self.n = 1
            self.total_weight = weight
            self.average = list(coord)
            self.leaf_head = GraphvizQuadTreeLeaf(node_weight=weight, coord=list(coord), id=node_id)
            return

        if level < self.max_level:
            old_n = self.n
            self.total_weight += weight
            if self.average is None:
                raise ValueError("non-empty quadtree is missing average.")
            for axis in range(self.dim):
                self.average[axis] = (self.average[axis] * old_n + coord[axis]) / (old_n + 1)
            if self.qts is None:
                self.qts = [None for _ in range(1 << self.dim)]

            child_index = _get_quadrant(dim=self.dim, center=self.center, coord=coord)
            self._ensure_child(child_index)._add_internal(
                coord=coord,
                weight=weight,
                node_id=node_id,
                level=level + 1,
            )

            if self.leaf_head is not None:
                old_leaf = self.leaf_head
                if self.n != 1:
                    raise ValueError("internal split expected exactly one parent leaf.")
                old_index = _get_quadrant(dim=self.dim, center=self.center, coord=old_leaf.coord)
                self._ensure_child(old_index)._add_internal(
                    coord=old_leaf.coord,
                    weight=old_leaf.node_weight,
                    node_id=old_leaf.id,
                    level=level + 1,
                )
                self.leaf_head = None

            self.n += 1
            return

        if self.qts is not None:
            raise ValueError("max-level Graphviz quadtree node cannot have children.")
        self.n += 1
        self.total_weight += weight
        if self.average is None:
            raise ValueError("non-empty quadtree is missing average.")
        for axis in range(self.dim):
            self.average[axis] = (self.average[axis] * self.n + coord[axis]) / (self.n + 1)
        self.leaf_head = GraphvizQuadTreeLeaf(
            node_weight=weight,
            coord=list(coord),
            id=node_id,
            next=self.leaf_head,
        )

    def _ensure_child(self, quadrant: int) -> "GraphvizQuadTree":
        """Return an existing child or create it in Graphviz quadrant order.

        Parameters
        ----------
        quadrant : int
            Child quadrant index.

        Returns
        -------
        GraphvizQuadTree
            Child quadtree node.
        """
        if self.qts is None:
            self.qts = [None for _ in range(1 << self.dim)]
        child = self.qts[quadrant]
        if child is None:
            child = new_in_quadrant(
                dim=self.dim,
                center=self.center,
                width=self.width / 2.0,
                max_level=self.max_level,
                quadrant=quadrant,
            )
            self.qts[quadrant] = child
        return child

    def _clear_force_data(self) -> None:
        """Clear cached cell and leaf force pointers before a force pass.

        Returns
        -------
        None
            The tree is mutated in place.
        """
        self.data = None
        leaf = self.leaf_head
        while leaf is not None:
            leaf.data = None
            leaf = leaf.next
        if self.qts is None:
            return
        for child in self.qts:
            if child is not None:
                child._clear_force_data()


@dataclass
class _NearestResult:
    """Mutable nearest-neighbor accumulator.

    Parameters
    ----------
    coord : list[float]
        Current nearest coordinate with length ``dim``.
    node_id : int
        Current nearest node id.
    distance : float
        Current nearest distance, or ``-1.0`` before any point is visited.
    """

    coord: List[float]
    node_id: int
    distance: float


def new_in_quadrant(
    dim: int,
    center: Sequence[float],
    width: float,
    max_level: int,
    quadrant: int,
) -> GraphvizQuadTree:
    """Create a child tree in Graphviz's quadrant coordinate convention.

    Parameters
    ----------
    dim : int
        Coordinate dimensionality.
    center : sequence of float
        Parent center coordinates.
    width : float
        Child radius.
    max_level : int
        Maximum subdivision depth.
    quadrant : int
        Graphviz child quadrant index.

    Returns
    -------
    GraphvizQuadTree
        Empty child quadtree.
    """
    child = GraphvizQuadTree.new(dim=dim, center=center, width=width, max_level=max_level)
    remaining = quadrant
    for axis in range(dim):
        if remaining % 2 == 0:
            child.center[axis] -= width
        else:
            child.center[axis] += width
        remaining = (remaining - (remaining % 2)) // 2
    return child


def point_distance(point1: Sequence[float], point2: Sequence[float]) -> float:
    """Compute Euclidean distance between two points.

    Parameters
    ----------
    point1 : sequence of float
        First point coordinates.
    point2 : sequence of float
        Second point coordinates.

    Returns
    -------
    float
        Euclidean distance.
    """
    if len(point1) != len(point2):
        raise ValueError("point dimensions must match.")
    total = 0.0
    for axis, value in enumerate(point1):
        delta = value - point2[axis]
        total += delta * delta
    return math.sqrt(total)


def graphviz_spring_electrical_repulsive_forces(
    positions: torch.Tensor,
    repulsive_scale: float,
    repulsive_exponent: float,
    theta: float,
    max_level: int = 10,
    quadtree_size: int = _GRAPHVIZ_SFDP_QUADTREE_SIZE,
) -> torch.Tensor:
    """Compute Graphviz SFDP repulsive forces with the shared quadtree.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, dim]``.
    repulsive_scale : float
        Global ``K^(1 - p)``-style repulsive multiplier.
    repulsive_exponent : float
        Graphviz repulsive power ``p``.
    theta : float
        Barnes-Hut opening coefficient.
    max_level : int, default=10
        Maximum quadtree subdivision depth.
    quadtree_size : int, default=45
        Node-count threshold above which Graphviz uses the quadtree path.

    Returns
    -------
    torch.Tensor
        Repulsive force tensor with shape ``[N, dim]``.
    """
    rows = _coordinate_rows(positions)
    if len(rows) <= 1:
        return torch.zeros_like(positions)
    if len(rows) < quadtree_size:
        return _graphviz_exact_repulsive_forces(
            positions=positions,
            repulsive_scale=repulsive_scale,
            repulsive_exponent=repulsive_exponent,
        )

    tree = GraphvizQuadTree.from_points(coordinates=positions, max_level=max_level)
    if tree is None:
        return torch.zeros_like(positions)
    force, _ = tree.get_repulsive_force(
        coordinates=positions,
        bh=theta,
        p=repulsive_exponent,
        kp=repulsive_scale,
    )
    return force


def graphviz_supernode_repulsive_force(
    tree: GraphvizQuadTree,
    positions: torch.Tensor,
    node_index: int,
    theta: float,
    repulsive_scale: float,
    repulsive_exponent: float,
) -> torch.Tensor:
    """Compute one Graphviz per-node supernode repulsive force.

    Parameters
    ----------
    tree : GraphvizQuadTree
        Quadtree built from ``positions``.
    positions : torch.Tensor
        Position tensor with shape ``[N, dim]``.
    node_index : int
        Node receiving the force.
    theta : float
        Barnes-Hut opening coefficient.
    repulsive_scale : float
        Global ``K^(1 - p)``-style repulsive multiplier.
    repulsive_exponent : float
        Graphviz repulsive power ``p``.

    Returns
    -------
    torch.Tensor
        Force vector with shape ``[dim]``.
    """
    if node_index < 0 or node_index >= positions.shape[0]:
        raise ValueError("node_index is outside positions.")
    query = positions[node_index].detach().cpu().to(dtype=torch.float64).tolist()
    centers, weights, distances, _ = tree.get_supernodes(
        bh=theta,
        pt=query,
        node_id=node_index,
    )
    force = torch.zeros((positions.shape[1],), dtype=torch.float64)
    for supernode_index in range(centers.shape[0]):
        distance = max(float(distances[supernode_index].item()), _GRAPHVIZ_MINDIST)
        denominator = distance ** (1.0 - repulsive_exponent)
        delta = torch.tensor(query, dtype=torch.float64) - centers[supernode_index]
        force += float(weights[supernode_index].item()) * repulsive_scale * delta / denominator
    return force.to(device=positions.device, dtype=positions.dtype)


def _coordinate_rows(coordinates: torch.Tensor) -> List[List[float]]:
    """Convert a coordinate tensor to CPU Python float rows.

    Parameters
    ----------
    coordinates : torch.Tensor
        Coordinate tensor with shape ``[N, dim]``.

    Returns
    -------
    list[list[float]]
        Python float coordinate rows.

    Raises
    ------
    ValueError
        If ``coordinates`` is not two-dimensional.
    """
    if coordinates.ndim != 2:
        raise ValueError("coordinates must have shape [N, dim].")
    return [
        [float(value) for value in row]
        for row in coordinates.detach().cpu().to(dtype=torch.float64).tolist()
    ]


def _graphviz_exact_repulsive_forces(
    positions: torch.Tensor,
    repulsive_scale: float,
    repulsive_exponent: float,
) -> torch.Tensor:
    """Compute Graphviz's exact small-graph repulsive force loop.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, dim]``.
    repulsive_scale : float
        Global repulsive multiplier.
    repulsive_exponent : float
        Graphviz repulsive power ``p``.

    Returns
    -------
    torch.Tensor
        Repulsive force tensor with shape ``[N, dim]``.
    """
    rows = _coordinate_rows(positions)
    force = [[0.0 for _ in range(positions.shape[1])] for _ in rows]
    for node_index, row in enumerate(rows):
        for other_index, other in enumerate(rows):
            if other_index == node_index:
                continue
            distance = max(point_distance(row, other), _GRAPHVIZ_MINDIST)
            denominator = distance ** (1.0 - repulsive_exponent)
            for axis in range(positions.shape[1]):
                force[node_index][axis] += repulsive_scale * (row[axis] - other[axis]) / denominator
    return torch.tensor(force, dtype=positions.dtype, device=positions.device)


def _validate_within_graphviz_bounds(tree: GraphvizQuadTree, coord: Sequence[float]) -> None:
    """Validate point dimensionality while preserving Graphviz's loose bounds.

    Parameters
    ----------
    tree : GraphvizQuadTree
        Tree receiving the point.
    coord : sequence of float
        Point coordinates with length ``dim``.

    Returns
    -------
    None
        The function raises only for dimensionality errors.
    """
    if len(coord) != tree.dim:
        raise ValueError("coord length must match dim.")
    tolerance = 1.0e5 * _GRAPHVIZ_MACHINEACC * tree.width
    for axis in range(tree.dim):
        lower = tree.center[axis] - tree.width - tolerance
        upper = tree.center[axis] + tree.width + tolerance
        if coord[axis] < lower or coord[axis] > upper:
            return


def _get_quadrant(dim: int, center: Sequence[float], coord: Sequence[float]) -> int:
    """Return Graphviz quadrant id for a point relative to a cell center.

    Parameters
    ----------
    dim : int
        Coordinate dimensionality.
    center : sequence of float
        Cell center coordinates.
    coord : sequence of float
        Point coordinates.

    Returns
    -------
    int
        Quadrant index in Graphviz bit order.
    """
    quadrant = 0
    for axis in range(dim - 1, -1, -1):
        if coord[axis] - center[axis] < 0.0:
            quadrant *= 2
        else:
            quadrant = (2 * quadrant) + 1
    return quadrant


def _get_supernodes_internal(
    tree: Optional[GraphvizQuadTree],
    bh: float,
    pt: Sequence[float],
    node_id: int,
    centers: List[List[float]],
    weights: List[float],
    distances: List[float],
    counts: float,
) -> float:
    """Recursive implementation of Graphviz ``QuadTree_get_supernodes``.

    Parameters
    ----------
    tree : GraphvizQuadTree, optional
        Current quadtree node.
    bh : float
        Barnes-Hut opening coefficient.
    pt : sequence of float
        Query point coordinates.
    node_id : int
        Node id excluded from leaf contributions.
    centers : list[list[float]]
        Output supernode centers.
    weights : list[float]
        Output supernode weights.
    distances : list[float]
        Output query-to-supernode distances.
    counts : float
        Recursive call counter.

    Returns
    -------
    float
        Updated recursive call counter.
    """
    counts += 1.0
    if tree is None:
        return counts

    leaf = tree.leaf_head
    while leaf is not None:
        _maybe_note_realloc_boundary(len(centers))
        if leaf.id != node_id:
            centers.append(list(leaf.coord))
            weights.append(leaf.node_weight)
            distances.append(point_distance(pt, leaf.coord))
        leaf = leaf.next

    if tree.qts is not None:
        distance = point_distance(tree.center, pt)
        if tree.width < bh * distance:
            if tree.average is None:
                raise ValueError("non-empty quadtree is missing average.")
            _maybe_note_realloc_boundary(len(centers))
            centers.append(list(tree.average))
            weights.append(tree.total_weight)
            distances.append(point_distance(tree.average, pt))
        else:
            for child in tree.qts:
                counts = _get_supernodes_internal(
                    tree=child,
                    bh=bh,
                    pt=pt,
                    node_id=node_id,
                    centers=centers,
                    weights=weights,
                    distances=distances,
                    counts=counts,
                )
    return counts


def _maybe_note_realloc_boundary(nsuper: int) -> None:
    """Preserve the reference reallocation boundary as a no-op marker.

    Parameters
    ----------
    nsuper : int
        Number of currently stored supernodes.

    Returns
    -------
    None
        Python lists handle allocation, so no action is required.
    """
    _ = nsuper + _GRAPHVIZ_REALLOC_INCREMENT


def _get_or_assign_node_force(
    force: List[List[float]],
    node_id: int,
    leaf: GraphvizQuadTreeLeaf,
    dim: int,
) -> List[float]:
    """Return the per-node force buffer associated with a leaf.

    Parameters
    ----------
    force : list[list[float]]
        Force rows indexed by node id.
    node_id : int
        Node id.
    leaf : GraphvizQuadTreeLeaf
        Leaf payload storing the cached force pointer.
    dim : int
        Coordinate dimensionality.

    Returns
    -------
    list[float]
        Mutable force row with length ``dim``.
    """
    if leaf.data is None:
        leaf.data = force[node_id]
    if len(leaf.data) != dim:
        raise ValueError("leaf force dimension mismatch.")
    return leaf.data


def _get_or_alloc_force_tree(tree: GraphvizQuadTree, dim: int) -> List[float]:
    """Return the per-cell force accumulator for a quadtree node.

    Parameters
    ----------
    tree : GraphvizQuadTree
        Quadtree node.
    dim : int
        Coordinate dimensionality.

    Returns
    -------
    list[float]
        Mutable cell force row with length ``dim``.
    """
    if tree.data is None:
        tree.data = [0.0 for _ in range(dim)]
    return tree.data


def _repulsive_force_interact(
    tree1: Optional[GraphvizQuadTree],
    tree2: Optional[GraphvizQuadTree],
    rows: Sequence[Sequence[float]],
    force: List[List[float]],
    bh: float,
    p: float,
    kp: float,
    counts: List[float],
) -> None:
    """Recursively accumulate Graphviz cell-cell and leaf-leaf repulsion.

    Parameters
    ----------
    tree1 : GraphvizQuadTree, optional
        First interaction cell.
    tree2 : GraphvizQuadTree, optional
        Second interaction cell.
    rows : sequence[sequence[float]]
        Original coordinate rows indexed by node id.
    force : list[list[float]]
        Per-node force output rows.
    bh : float
        Barnes-Hut opening coefficient.
    p : float
        Repulsive power.
    kp : float
        Repulsive multiplier.
    counts : list[float]
        Graphviz interaction counters with length ``4``.

    Returns
    -------
    None
        Force and count buffers are mutated in place.
    """
    if tree1 is None or tree2 is None:
        return
    if tree1.n <= 0 or tree2.n <= 0:
        raise ValueError("quadtree interactions require non-empty cells.")
    dim = tree1.dim
    if tree1.average is None or tree2.average is None:
        raise ValueError("non-empty quadtree is missing average.")

    distance = point_distance(tree1.average, tree2.average)
    if tree1.width + tree2.width < bh * distance:
        counts[0] += 1.0
        force1 = _get_or_alloc_force_tree(tree=tree1, dim=dim)
        force2 = _get_or_alloc_force_tree(tree=tree2, dim=dim)
        if distance <= 0.0:
            raise ValueError("Graphviz far-cell interaction requires positive distance.")
        _accumulate_pair_force(
            coord1=tree1.average,
            coord2=tree2.average,
            weight1=tree1.total_weight,
            weight2=tree2.total_weight,
            distance=distance,
            p=p,
            kp=kp,
            force1=force1,
            force2=force2,
        )
        return

    if tree1.leaf_head is not None and tree2.leaf_head is not None:
        leaf1 = tree1.leaf_head
        while leaf1 is not None:
            force1 = _get_or_assign_node_force(force=force, node_id=leaf1.id, leaf=leaf1, dim=dim)
            leaf2 = tree2.leaf_head
            while leaf2 is not None:
                if (tree1 is tree2 and leaf2.id < leaf1.id) or leaf1.id == leaf2.id:
                    leaf2 = leaf2.next
                    continue
                force2 = _get_or_assign_node_force(
                    force=force,
                    node_id=leaf2.id,
                    leaf=leaf2,
                    dim=dim,
                )
                counts[1] += 1.0
                distance = _distance_cropped(rows=rows, dim=dim, node1=leaf1.id, node2=leaf2.id)
                _accumulate_pair_force(
                    coord1=leaf1.coord,
                    coord2=leaf2.coord,
                    weight1=leaf1.node_weight,
                    weight2=leaf2.node_weight,
                    distance=distance,
                    p=p,
                    kp=kp,
                    force1=force1,
                    force2=force2,
                )
                leaf2 = leaf2.next
            leaf1 = leaf1.next
        return

    if tree1 is tree2:
        if tree1.qts is None:
            return
        for left_index, child1 in enumerate(tree1.qts):
            for child2 in tree1.qts[left_index:]:
                _repulsive_force_interact(
                    tree1=child1,
                    tree2=child2,
                    rows=rows,
                    force=force,
                    bh=bh,
                    p=p,
                    kp=kp,
                    counts=counts,
                )
        return

    if tree1.width > tree2.width and tree1.leaf_head is None:
        _interact_children_against_tree(
            tree=tree1,
            other=tree2,
            rows=rows,
            force=force,
            bh=bh,
            p=p,
            kp=kp,
            counts=counts,
        )
    elif tree2.width > tree1.width and tree2.leaf_head is None:
        _interact_children_against_tree(
            tree=tree2,
            other=tree1,
            rows=rows,
            force=force,
            bh=bh,
            p=p,
            kp=kp,
            counts=counts,
        )
    elif tree1.leaf_head is None:
        _interact_children_against_tree(
            tree=tree1,
            other=tree2,
            rows=rows,
            force=force,
            bh=bh,
            p=p,
            kp=kp,
            counts=counts,
        )
    elif tree2.leaf_head is None:
        _interact_children_against_tree(
            tree=tree2,
            other=tree1,
            rows=rows,
            force=force,
            bh=bh,
            p=p,
            kp=kp,
            counts=counts,
        )
    else:
        raise ValueError("leaf-leaf interaction should have been handled earlier.")


def _interact_children_against_tree(
    tree: GraphvizQuadTree,
    other: GraphvizQuadTree,
    rows: Sequence[Sequence[float]],
    force: List[List[float]],
    bh: float,
    p: float,
    kp: float,
    counts: List[float],
) -> None:
    """Interact every child of one tree against another tree.

    Parameters
    ----------
    tree : GraphvizQuadTree
        Tree whose children are expanded.
    other : GraphvizQuadTree
        Other interaction tree.
    rows : sequence[sequence[float]]
        Original coordinate rows.
    force : list[list[float]]
        Per-node force output rows.
    bh : float
        Barnes-Hut opening coefficient.
    p : float
        Repulsive power.
    kp : float
        Repulsive multiplier.
    counts : list[float]
        Graphviz interaction counters.

    Returns
    -------
    None
        Force and count buffers are mutated in place.
    """
    if tree.qts is None:
        return
    for child in tree.qts:
        _repulsive_force_interact(
            tree1=child,
            tree2=other,
            rows=rows,
            force=force,
            bh=bh,
            p=p,
            kp=kp,
            counts=counts,
        )


def _accumulate_pair_force(
    coord1: Sequence[float],
    coord2: Sequence[float],
    weight1: float,
    weight2: float,
    distance: float,
    p: float,
    kp: float,
    force1: List[float],
    force2: List[float],
) -> None:
    """Accumulate Graphviz's symmetric repulsive force for one pair.

    Parameters
    ----------
    coord1 : sequence of float
        First coordinate.
    coord2 : sequence of float
        Second coordinate.
    weight1 : float
        First point or cell weight.
    weight2 : float
        Second point or cell weight.
    distance : float
        Pair distance.
    p : float
        Repulsive power.
    kp : float
        Repulsive multiplier.
    force1 : list[float]
        First force buffer.
    force2 : list[float]
        Second force buffer.

    Returns
    -------
    None
        Force buffers are mutated in place.
    """
    if p == -1.0:
        denominator = distance * distance
    else:
        denominator = distance ** (1.0 - p)
    scale = weight1 * weight2 * kp / denominator
    for axis in range(len(coord1)):
        value = scale * (coord1[axis] - coord2[axis])
        force1[axis] += value
        force2[axis] -= value


def _repulsive_force_accumulate(
    tree: GraphvizQuadTree,
    force: List[List[float]],
    counts: List[float],
) -> None:
    """Push Graphviz cell-level force accumulators down to leaves.

    Parameters
    ----------
    tree : GraphvizQuadTree
        Current quadtree node.
    force : list[list[float]]
        Per-node force output rows.
    counts : list[float]
        Graphviz interaction counters.

    Returns
    -------
    None
        Force and count buffers are mutated in place.
    """
    dim = tree.dim
    total_weight = tree.total_weight
    if total_weight <= 0.0:
        raise ValueError("force accumulation requires positive total weight.")
    cell_force = _get_or_alloc_force_tree(tree=tree, dim=dim)
    counts[2] += 1.0

    leaf = tree.leaf_head
    if leaf is not None:
        while leaf is not None:
            node_force = _get_or_assign_node_force(force=force, node_id=leaf.id, leaf=leaf, dim=dim)
            weight_ratio = leaf.node_weight / total_weight
            for axis in range(dim):
                node_force[axis] += weight_ratio * cell_force[axis]
            leaf = leaf.next
        return

    if tree.qts is None:
        return
    for child in tree.qts:
        if child is None:
            continue
        child_force = _get_or_alloc_force_tree(tree=child, dim=dim)
        weight_ratio = child.total_weight / total_weight
        for axis in range(dim):
            child_force[axis] += weight_ratio * cell_force[axis]
        _repulsive_force_accumulate(tree=child, force=force, counts=counts)


def _distance_cropped(
    rows: Sequence[Sequence[float]],
    dim: int,
    node1: int,
    node2: int,
) -> float:
    """Compute Graphviz ``distance_cropped`` for two coordinate rows.

    Parameters
    ----------
    rows : sequence[sequence[float]]
        Coordinate rows.
    dim : int
        Coordinate dimensionality.
    node1 : int
        First node index.
    node2 : int
        Second node index.

    Returns
    -------
    float
        Euclidean distance clamped to Graphviz ``MINDIST``.
    """
    return max(point_distance(rows[node1][:dim], rows[node2][:dim]), _GRAPHVIZ_MINDIST)


def _nearest_internal(
    tree: Optional[GraphvizQuadTree],
    point: Sequence[float],
    result: _NearestResult,
    tentative: bool,
) -> None:
    """Recursive implementation of Graphviz nearest-point lookup.

    Parameters
    ----------
    tree : GraphvizQuadTree, optional
        Current quadtree node.
    point : sequence of float
        Query point coordinates.
    result : _NearestResult
        Mutable nearest result accumulator.
    tentative : bool
        Whether to follow only the best child before the full traversal.

    Returns
    -------
    None
        ``result`` is mutated in place.
    """
    if tree is None:
        return
    leaf = tree.leaf_head
    while leaf is not None:
        distance = point_distance(point, leaf.coord)
        if result.distance < 0.0 or distance < result.distance:
            result.distance = distance
            result.node_id = leaf.id
            result.coord = list(leaf.coord)
        leaf = leaf.next

    if tree.qts is None:
        return
    distance_to_center = point_distance(tree.center, point)
    if result.distance >= 0.0 and (
        distance_to_center - math.sqrt(float(tree.dim)) * tree.width > result.distance
    ):
        return
    if tentative:
        child_index = _nearest_child_index(tree=tree, point=point)
        if child_index >= 0 and tree.qts[child_index] is not None:
            _nearest_internal(
                tree=tree.qts[child_index],
                point=point,
                result=result,
                tentative=tentative,
            )
        return
    for child in tree.qts:
        _nearest_internal(tree=child, point=point, result=result, tentative=tentative)


def _nearest_child_index(tree: GraphvizQuadTree, point: Sequence[float]) -> int:
    """Select the tentative child with nearest average coordinate.

    Parameters
    ----------
    tree : GraphvizQuadTree
        Parent quadtree node.
    point : sequence of float
        Query point coordinates.

    Returns
    -------
    int
        Child index, or ``-1`` when the tree has no populated children.
    """
    if tree.qts is None:
        return -1
    best_distance = -1.0
    best_index = -1
    for index, child in enumerate(tree.qts):
        if child is None:
            continue
        if child.average is None:
            raise ValueError("non-empty quadtree is missing average.")
        distance = point_distance(child.average, point)
        if distance < best_distance or best_distance < 0.0:
            best_distance = distance
            best_index = index
    return best_index


__all__ = [
    "GraphvizQuadTree",
    "GraphvizQuadTreeLeaf",
    "graphviz_spring_electrical_repulsive_forces",
    "graphviz_supernode_repulsive_force",
    "new_in_quadrant",
    "point_distance",
]

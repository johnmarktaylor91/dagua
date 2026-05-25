"""Golden-vector tests for the Graphviz sparse quadtree port."""

from __future__ import annotations

import math
from typing import List

import torch

from dagua.layout.ops.quadtree import (
    GraphvizQuadTree,
    graphviz_spring_electrical_repulsive_forces,
)


def _square_points() -> torch.Tensor:
    """Build a four-corner square point set.

    Returns
    -------
    torch.Tensor
        Point tensor with shape ``[4, 2]``.
    """
    return torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ],
        dtype=torch.float64,
    )


def _leaf_ids(tree: GraphvizQuadTree) -> List[int]:
    """Return max-depth leaf ids in linked-list order.

    Parameters
    ----------
    tree : GraphvizQuadTree
        Tree whose root is a leaf.

    Returns
    -------
    list[int]
        Leaf ids in traversal order.
    """
    ids: List[int] = []
    leaf = tree.leaf_head
    while leaf is not None:
        ids.append(leaf.id)
        leaf = leaf.next
    return ids


def test_from_points_matches_graphviz_bounds_and_quadrants() -> None:
    """Graphviz root width and child quadrant centers should match C."""
    tree = GraphvizQuadTree.from_points(coordinates=_square_points(), max_level=1)

    assert tree is not None
    torch.testing.assert_close(torch.tensor(tree.center), torch.tensor([1.0, 1.0]))
    assert tree.width == 1.04
    assert tree.average == [1.0, 1.0]
    assert tree.qts is not None
    child_centers = [child.center if child is not None else None for child in tree.qts]
    assert child_centers == [
        [0.48, 0.48],
        [1.52, 0.48],
        [0.48, 1.52],
        [1.52, 1.52],
    ]


def test_max_level_leaf_average_and_order_match_graphviz() -> None:
    """Max-depth insertion should preserve Graphviz's head-push leaf list."""
    points = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    tree = GraphvizQuadTree.from_points(coordinates=points, max_level=0)

    assert tree is not None
    assert tree.average == [0.5, 0.5]
    assert _leaf_ids(tree) == [2, 1, 0]


def test_get_supernodes_matches_graphviz_opening_cases() -> None:
    """Supernode selection should match Graphviz's traversal and counts."""
    tree = GraphvizQuadTree.from_points(coordinates=_square_points(), max_level=1)
    assert tree is not None

    centers, weights, distances, counts = tree.get_supernodes(bh=0.5, pt=[0.0, 0.0], node_id=0)

    torch.testing.assert_close(
        centers,
        torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(weights, torch.ones((3,), dtype=torch.float64))
    torch.testing.assert_close(
        distances,
        torch.tensor([2.0, 2.0, math.sqrt(8.0)], dtype=torch.float64),
    )
    assert counts == 5.0

    centers, weights, distances, counts = tree.get_supernodes(bh=2.0, pt=[0.0, 0.0], node_id=0)

    torch.testing.assert_close(centers, torch.tensor([[1.0, 1.0]], dtype=torch.float64))
    torch.testing.assert_close(weights, torch.tensor([4.0], dtype=torch.float64))
    torch.testing.assert_close(distances, torch.tensor([math.sqrt(2.0)], dtype=torch.float64))
    assert counts == 1.0


def test_get_repulsive_force_matches_graphviz_leaf_pair_golden() -> None:
    """Leaf-pair force accumulation should match the C pair order."""
    points = _square_points()
    tree = GraphvizQuadTree.from_points(coordinates=points, max_level=1)
    assert tree is not None

    force, counts = tree.get_repulsive_force(coordinates=points, bh=0.0, p=-1.0, kp=1.0)

    torch.testing.assert_close(
        force,
        torch.tensor(
            [
                [-0.75, -0.75],
                [0.75, -0.75],
                [-0.75, 0.75],
                [0.75, 0.75],
            ],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        counts,
        torch.tensor([0.0, 1.5, 1.25, 0.0], dtype=torch.float64),
    )


def test_get_nearest_uses_graphviz_two_pass_traversal() -> None:
    """Nearest lookup should return the same id and distance as Graphviz."""
    tree = GraphvizQuadTree.from_points(coordinates=_square_points(), max_level=1)
    assert tree is not None

    coord, node_id, distance = tree.get_nearest(point=[1.8, 1.7])

    torch.testing.assert_close(coord, torch.tensor([2.0, 2.0], dtype=torch.float64))
    assert node_id == 3
    assert distance == math.sqrt(0.13)


def test_public_repulsive_wrapper_can_force_quadtree_path() -> None:
    """The public wrapper should expose the Graphviz quadtree force path."""
    force = graphviz_spring_electrical_repulsive_forces(
        positions=_square_points(),
        repulsive_scale=1.0,
        repulsive_exponent=-1.0,
        theta=0.0,
        max_level=1,
        quadtree_size=0,
    )

    torch.testing.assert_close(
        force,
        torch.tensor(
            [
                [-0.75, -0.75],
                [0.75, -0.75],
                [-0.75, 0.75],
                [0.75, 0.75],
            ],
            dtype=torch.float64,
        ),
    )

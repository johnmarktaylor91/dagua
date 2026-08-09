"""Tests for the planarity-guarded polish op (sprint2 W1-B)."""

from __future__ import annotations

import torch

from dagua.layout.ops.planar_polish import (
    _FaceSignGuard,
    drawing_faces,
    exact_crossing_count,
    guarded_descent,
)


def _grid_graph(rows: int, cols: int) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Return edge_index, node count, and a planar unit-grid drawing."""
    edges = []
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            if col < cols - 1:
                edges.append((node, node + 1))
            if row < rows - 1:
                edges.append((node, node + cols))
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    pos = torch.tensor(
        [[float(col), float(row)] for row in range(rows) for col in range(cols)],
        dtype=torch.float64,
    )
    return edge_index, rows * cols, pos


def test_exact_crossing_count_matches_known_layouts() -> None:
    """Grid drawing has zero crossings; a swapped pair introduces some."""
    edge_index, _, pos = _grid_graph(3, 3)
    assert exact_crossing_count(pos, edge_index) == 0
    crossed = pos.clone()
    crossed[[0, 4]] = crossed[[4, 0]]
    assert exact_crossing_count(crossed, edge_index) > 0


def test_drawing_faces_partition_half_edges() -> None:
    """Every half-edge of a planar drawing lands in exactly one face walk."""
    edge_index, num_nodes, pos = _grid_graph(3, 3)
    faces = drawing_faces(pos, edge_index, num_nodes)
    assert faces is not None
    total_half_edges = sum(len(face) for face in faces)
    assert total_half_edges == 2 * edge_index.shape[1]
    # 3x3 grid: 4 inner quads + the outer face.
    assert len(faces) == 5


def test_face_guard_rejects_face_flip() -> None:
    """Reflecting one interior node through the grid flips face windings."""
    edge_index, num_nodes, pos = _grid_graph(3, 3)
    guard = _FaceSignGuard(pos, edge_index, num_nodes)
    assert guard.valid and guard.faces
    assert guard.signs_preserved(pos)
    flipped = pos.clone()
    # Push the center node far outside the grid: incident quads invert.
    flipped[4] = torch.tensor([10.0, 10.0], dtype=torch.float64)
    assert not guard.signs_preserved(flipped)


def test_exact_check_catches_crossing_with_empty_guard_set() -> None:
    """A tree has no guardable faces, so only the exact check certifies.

    Face-winding preservation alone is NOT a sufficient planarity
    certificate: with a bridge-only graph the guard set is empty and every
    layout passes the cheap screen, including a crossing one.
    """
    # Star with two chains: 0-1, 0-2, 1-3, 2-4 (a tree; no cycles).
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]], dtype=torch.long)
    pos = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [1.0, -1.0], [2.0, -1.0], [2.0, 1.0]],
        dtype=torch.float64,
    )
    guard = _FaceSignGuard(pos, edge_index, 5)
    assert guard.valid
    assert not guard.faces  # nothing guardable: cheap screen is blind here
    assert guard.signs_preserved(pos)
    # Edges 1-3 and 2-4 cross in this layout even though the screen passes.
    assert exact_crossing_count(pos, edge_index) > 0


def test_guarded_descent_keeps_zero_crossings_and_improves_objective() -> None:
    """Polish output stays planar and reduces the polish objective."""
    from dagua.layout.ops.pipelines.native_surrogates import edge_length_cv_loss

    edge_index, num_nodes, pos = _grid_graph(3, 3)
    # Distort the grid (keep planarity) so there is something to improve.
    distorted = pos.clone()
    distorted[:, 0] *= 3.0
    distorted[4] = torch.tensor([3.4, 1.4], dtype=torch.float64)
    assert exact_crossing_count(distorted, edge_index) == 0
    polished = guarded_descent(distorted, edge_index, num_nodes, None, steps=60)
    assert polished is not None
    assert exact_crossing_count(polished, edge_index) == 0
    before = float(edge_length_cv_loss(distorted.to(torch.float32), edge_index))
    after = float(edge_length_cv_loss(polished.to(torch.float32), edge_index))
    assert after < before


def test_guarded_descent_fails_closed_on_crossing_input() -> None:
    """A crossing input cannot be certified: polish returns None."""
    edge_index, num_nodes, pos = _grid_graph(3, 3)
    crossed = pos.clone()
    crossed[[0, 4]] = crossed[[4, 0]]
    assert exact_crossing_count(crossed, edge_index) > 0
    assert guarded_descent(crossed, edge_index, num_nodes, None, steps=20) is None


def test_guarded_descent_is_deterministic() -> None:
    """Two identical polish runs produce byte-identical output."""
    edge_index, num_nodes, pos = _grid_graph(3, 4)
    distorted = pos.clone()
    distorted[:, 1] *= 2.5
    first = guarded_descent(distorted, edge_index, num_nodes, None, steps=40)
    second = guarded_descent(distorted, edge_index, num_nodes, None, steps=40)
    assert first is not None and second is not None
    assert torch.equal(first, second)

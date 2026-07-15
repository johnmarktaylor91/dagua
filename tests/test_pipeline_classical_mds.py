"""Exact-fidelity tests for the composable classical MDS pipeline."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, Optional

import pytest
import torch

from dagua.layout.classic.classical_mds import layout_classical_mds
from dagua.layout.ops.pipelines.classical_mds import (
    _IgraphMergeGrid,
    build_classical_mds_pipeline,
    layout_classical_mds_pipeline,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _edge_index_from_edges(edges: Iterable[tuple[int, int]]) -> torch.Tensor:
    """Build a standard ``[2, E]`` edge tensor from edge tuples.

    Parameters
    ----------
    edges : Iterable[tuple[int, int]]
        Directed edges as ``(source, target)`` pairs.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    edge_list = list(edges)
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
    sources, targets = zip(*edge_list)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a directed path graph edge tensor.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Directed path graph edge tensor.
    """
    return _edge_index_from_edges((index, index + 1) for index in range(max(num_nodes - 1, 0)))


def _disconnected_edge_index() -> torch.Tensor:
    """Build a disconnected graph with multiple components and isolates.

    Returns
    -------
    torch.Tensor
        Directed edge tensor for the disconnected graph.
    """
    return _edge_index_from_edges([(0, 1), (1, 2), (3, 4), (6, 7)])


def _complete_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a directed complete graph without self-loops.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Dense directed complete graph edge tensor.
    """
    return _edge_index_from_edges(
        (source, target)
        for source in range(num_nodes)
        for target in range(num_nodes)
        if source != target
    )


def _assert_exact_match(classic: torch.Tensor, pipeline: torch.Tensor) -> None:
    """Assert that two classical MDS outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic classical MDS.
    pipeline : torch.Tensor
        Output from the composable pipeline.

    Returns
    -------
    None
        This helper asserts exact equality.
    """
    assert classic.dtype == pipeline.dtype
    assert classic.device == pipeline.device
    assert torch.equal(classic, pipeline)


def _brute_force_grid_get_sphere(
    grid: _IgraphMergeGrid,
    x_coord: float,
    y_coord: float,
    radius: float,
) -> int:
    """Find a merge-grid collision by scanning all occupied cells.

    Parameters
    ----------
    grid : _IgraphMergeGrid
        Grid populated with already placed component spheres.
    x_coord : float
        Candidate center x coordinate.
    y_coord : float
        Candidate center y coordinate.
    radius : float
        Candidate sphere radius.

    Returns
    -------
    int
        Component id of the first colliding occupied cell, or ``-1``.
    """
    if (
        x_coord - radius <= grid.minx
        or x_coord + radius >= grid.maxx
        or y_coord - radius <= grid.miny
        or y_coord + radius >= grid.maxy
    ):
        return -1
    radius_squared = radius * radius
    for x_index, y_index in zip(grid.occupied_x, grid.occupied_y):
        cell_x = grid.minx + float(x_index) * grid.deltax
        cell_y = grid.miny + float(y_index) * grid.deltay
        delta_x = x_coord - cell_x
        delta_y = y_coord - cell_y
        if delta_x * delta_x + delta_y * delta_y < radius_squared:
            return grid._get_mat(int(x_index), int(y_index)) - 1
    return -1


def _run_pipeline_direct(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute ``build_classical_mds_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor produced by the pipeline.
    """
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_classical_mds_pipeline().apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


def test_layout_runtime_modules_do_not_import_igraph() -> None:
    """Guard against runtime delegation to python-igraph from layout modules."""
    offenders: list[str] = []
    for path in sorted((_PROJECT_ROOT / "dagua" / "layout").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(
                    alias.name == "igraph" or alias.name.startswith("igraph.")
                    for alias in node.names
                ):
                    offenders.append(f"{path.relative_to(_PROJECT_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom) and (
                node.module == "igraph" or (node.module or "").startswith("igraph.")
            ):
                offenders.append(f"{path.relative_to(_PROJECT_ROOT)}:{node.lineno}")

    assert offenders == []


class TestClassicalMDSPipelineFidelity:
    """Bit-exact regression coverage for the classical MDS pipeline."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(0, 42), (1, 42), (2, 42), (5, 42), (5, 99), (20, 42), (50, 7)],
    )
    def test_layout_classical_mds_pipeline_matches_classic_for_requested_sizes(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """The adapter should match classic classical MDS exactly."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_classical_mds(edge_index=edge_index, num_nodes=num_nodes, seed=seed)
        pipeline = layout_classical_mds_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=seed,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_classical_mds_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted shortest-path targets should remain bit-identical."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_classical_mds(
            edge_index=edge_index,
            num_nodes=6,
            seed=17,
            edge_weights=edge_weights,
        )
        pipeline = layout_classical_mds_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            seed=17,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_classical_mds_pipeline_uses_seeded_dla_on_disconnected_graph(self) -> None:
        """Disconnected igraph-compatible layouts should use seeded DLA."""
        edge_index = _disconnected_edge_index()

        first = layout_classical_mds_pipeline(edge_index=edge_index, num_nodes=9, seed=99)
        repeated = layout_classical_mds_pipeline(edge_index=edge_index, num_nodes=9, seed=99)
        different_seed = layout_classical_mds_pipeline(edge_index=edge_index, num_nodes=9, seed=101)
        legacy = layout_classical_mds(edge_index=edge_index, num_nodes=9, seed=99)

        _assert_exact_match(first, repeated)
        assert not torch.equal(first, different_seed)
        assert not torch.equal(first, legacy)

    def test_igraph_merge_grid_lookup_matches_full_occupied_scan(self) -> None:
        """Optimized DLA collision lookup should preserve full-scan decisions."""
        grid = _IgraphMergeGrid(
            minx=-10.0,
            maxx=10.0,
            stepsx=40,
            miny=-10.0,
            maxy=10.0,
            stepsy=40,
        )
        grid.place_sphere(0.0, 0.0, 1.8, 3)
        grid.place_sphere(3.4, -2.2, 1.2, 7)
        grid.place_sphere(-4.5, 4.0, 2.0, 11)

        candidates = [
            (-9.5, 0.0, 0.75),
            (-4.0, 3.6, 0.8),
            (0.9, 0.9, 1.0),
            (2.8, -1.8, 0.65),
            (7.5, 7.5, 0.5),
            (9.8, 0.0, 0.4),
        ]
        for x_coord, y_coord, radius in candidates:
            assert grid.get_sphere(x_coord, y_coord, radius) == _brute_force_grid_get_sphere(
                grid,
                x_coord,
                y_coord,
                radius,
            )

    def test_connected_igraph_fidelity_path_matches_frozen_expectation(self) -> None:
        """Connected graphs should keep the aligned igraph-binding path."""
        edge_index = _edge_index_from_edges([(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (2, 6)])
        expected = torch.tensor(
            [
                [18.665543833577367, -81.42865150765743],
                [11.472505840196021, -12.528514990915745],
                [-41.22107151432031, 8.209953129678556],
                [-81.19011179594563, 18.892882956107357],
                [61.6452504242114, 11.810977488588032],
                [111.81799500822676, 36.150469968091826],
                [-81.19011179594563, 18.892882956107417],
            ],
            dtype=torch.float64,
        )

        positions = layout_classical_mds_pipeline(
            edge_index=edge_index,
            num_nodes=7,
            seed=123,
            igraph_fidelity=True,
            fidelity_dtype=torch.float64,
        )

        assert torch.equal(positions, expected)

    def test_igraph_fidelity_survives_degenerate_top_eigenspace(self) -> None:
        """A 1-50-1 layered graph must not collapse to an all-zeros layout.

        The double-centered Gram matrix of this graph has a top eigenvalue of
        multiplicity 50. LAPACK ``dsyevr`` with ``range='I'`` can silently
        return zero eigenpairs (``m = 0`` with ``info == 0``) on that spectrum
        depending on workspace size, and SciPy's ``eigh`` surfaces that as
        empty arrays without raising. Regression guard for the r78
        ``wide_single_layer_1_50_1`` all-zeros bug.
        """
        hub_edges = [(0, mid) for mid in range(1, 51)] + [(mid, 51) for mid in range(1, 51)]
        edge_index = _edge_index_from_edges(hub_edges)

        positions = layout_classical_mds_pipeline(
            edge_index=edge_index,
            num_nodes=52,
            seed=100,
            igraph_fidelity=True,
            fidelity_dtype=torch.float64,
        )

        assert positions.shape == (52, 2)
        assert torch.isfinite(positions).all()
        norm = torch.linalg.norm(positions)
        # igraph scaling normalizes the raw embedding: two sqrt(2)-length
        # columns give Frobenius norm 2, scaled by 50 -> 100.
        assert torch.isclose(norm, torch.tensor(100.0, dtype=torch.float64))
        # Rank-2: both output columns must be non-degenerate.
        assert torch.linalg.norm(positions[:, 0]) > 1.0
        assert torch.linalg.norm(positions[:, 1]) > 1.0

    def test_build_classical_mds_pipeline_matches_classic_on_complete_graph(self) -> None:
        """The raw pipeline object should match classic classical MDS on a dense graph."""
        edge_index = _complete_edge_index(5)

        classic = layout_classical_mds(edge_index=edge_index, num_nodes=5, seed=7)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=5)

        _assert_exact_match(classic, pipeline)

    def test_layout_classical_mds_pipeline_matches_classic_with_node_sizes(self) -> None:
        """Node-size-driven output extent should remain bit-identical."""
        edge_index = _edge_index_from_edges([(0, 1), (1, 2), (2, 3), (1, 4), (2, 5)])
        node_sizes = torch.tensor(
            [
                [10.0, 12.0],
                [11.0, 8.0],
                [7.0, 9.0],
                [6.0, 6.0],
                [9.0, 10.0],
                [5.0, 7.0],
            ],
            dtype=torch.float32,
        )

        classic = layout_classical_mds(
            edge_index=edge_index,
            num_nodes=6,
            seed=13,
            node_sizes=node_sizes,
        )
        pipeline = layout_classical_mds_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            seed=13,
            node_sizes=node_sizes,
        )

        _assert_exact_match(classic, pipeline)

"""Regression tests for the GRIP pipeline."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import pytest
import torch

from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.grip import (
    GripConfig,
    _build_undirected_adjacency,
    build_grip_pipeline,
    build_mis_filtration,
    intelligent_initial_position,
    layout_grip_pipeline,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import get_op_class, list_ops


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
        Directed path edge tensor with shape ``[2, E]``.
    """
    return _edge_index_from_edges((index, index + 1) for index in range(max(num_nodes - 1, 0)))


def test_grip_pipeline_and_ops_are_registered() -> None:
    """The algorithm and stage ops should resolve through local registries.

    Returns
    -------
    None
        Assertions check pipeline and op registration.
    """
    assert PIPELINE_REGISTRY["grip"] == (
        "dagua.layout.ops.pipelines.grip",
        "layout_grip_pipeline",
    )
    assert get_pipeline_function("grip") is layout_grip_pipeline
    assert get_op_class("grip_build_mis_filtration").__name__ == "GripBuildFiltration"
    assert get_op_class("grip_intelligent_placement").__name__ == "GripIntelligentPlacement"
    assert get_op_class("grip_local_fr_refinement").__name__ == "GripLocalRefinement"
    assert "grip_local_fr_refinement" in list_ops()


def test_grip_mis_filtration_pins_seeded_path_order() -> None:
    """Seeded MIS construction should be deterministic on a path.

    Returns
    -------
    None
        The assertion pins the clean-room greedy draw order and exclusion
        radius behavior for ``V_1``.
    """
    edge_index = _path_edge_index(6)

    levels = build_mis_filtration(edge_index=edge_index, num_nodes=6, seed=42)

    assert levels == [[0, 1, 2, 3, 4, 5], [0, 2, 5]]


def test_grip_intelligent_initial_position_uses_circle_solution() -> None:
    """Two-anchor initialization should solve the circle equations.

    Returns
    -------
    None
        The assertion pins the midpoint solution for a path-distance tie.
    """
    edge_index = _edge_index_from_edges([(0, 1), (1, 2), (2, 3), (3, 0)])
    adjacency = _build_undirected_adjacency(edge_index=edge_index, num_nodes=4)
    positions = torch.zeros((4, 2), dtype=torch.float64)
    positions[0] = torch.tensor([0.0, 0.0], dtype=torch.float64)
    positions[2] = torch.tensor([2.0, 0.0], dtype=torch.float64)

    placed = intelligent_initial_position(
        vertex=1,
        placed=[0, 2],
        adjacency=adjacency,
        positions=positions,
        dtype=torch.float64,
    )

    assert torch.equal(placed, torch.tensor([1.0, 0.0], dtype=torch.float64))


def test_grip_three_round_cycle_pins_clean_room_layout() -> None:
    """The clean-room GRIP stages should remain numerically stable.

    Returns
    -------
    None
        The assertion pins MIS, intelligent init, local FR, and final scaling
        together on a small cycle graph.
    """
    edge_index = _edge_index_from_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])

    positions = layout_grip_pipeline(
        edge_index=edge_index,
        num_nodes=6,
        steps=3,
        seed=42,
        fidelity_dtype=torch.float64,
    )

    expected = torch.tensor(
        [
            [19.1857, -27.2027],
            [4.7180, 37.9373],
            [-50.0000, 17.1431],
            [13.3697, 28.7831],
            [5.8586, -11.8528],
            [6.8680, -44.8081],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(positions, expected, atol=5.0e-4, rtol=0.0)


def test_grip_is_seed_deterministic() -> None:
    """GRIP should repeat exactly for a seed.

    Returns
    -------
    None
        Assertions cover repeatability. The reference runner uses deterministic
        BFS filtration in this mode, so these small graphs are not seed
        sensitive unless a zero-distance force fallback calls ``rand``.
    """
    edge_index = _path_edge_index(8)

    first = layout_grip_pipeline(edge_index, 8, steps=3, seed=7)
    second = layout_grip_pipeline(edge_index, 8, steps=3, seed=7)
    different_seed = layout_grip_pipeline(edge_index, 8, steps=3, seed=9)

    assert torch.equal(first, second)
    assert torch.equal(first, different_seed)
    assert torch.isfinite(first).all()


def test_build_grip_pipeline_matches_public_adapter() -> None:
    """The composable pipeline object should match the public adapter.

    Returns
    -------
    None
        The assertion verifies the ``Op`` composition path.
    """
    edge_index = _edge_index_from_edges([(0, 1), (0, 2), (2, 3), (3, 1)])
    config = GripConfig(rounds=4, fidelity_dtype=torch.float64)
    problem = LayoutProblem(edge_index=edge_index, num_nodes=4, seed=11)

    final_state = build_grip_pipeline(config=config).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    assert final_state.pos is not None
    public = layout_grip_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        steps=4,
        seed=11,
        fidelity_dtype=torch.float64,
    )

    assert torch.equal(final_state.pos, public)


def test_grip_pipeline_does_not_delegate_to_reference_runtime() -> None:
    """Production GRIP code must not call a reference binary or adapter.

    Returns
    -------
    None
        The AST guard rejects imports from competitor adapters, process
        launchers, ctypes/FFI bridges, or OpenGL/Tcl runtime wrappers.
    """
    path = Path("dagua/layout/ops/pipelines/grip.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_roots = {
        "dagua.eval.competitors",
        "ctypes",
        "cffi",
        "jpype",
        "jnius",
        "OpenGL",
        "tkinter",
    }
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)

    assert not any(
        imported == forbidden or imported.startswith(f"{forbidden}.")
        for imported in imports
        for forbidden in forbidden_roots
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": -1},
        {"coarsest_size": 0},
        {"neighbor_factor": 0.0},
        {"min_neighbors": 0},
        {"output_scale": 0.0},
    ],
)
def test_grip_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    """Invalid GRIP parameters should fail before layout work starts.

    Parameters
    ----------
    kwargs : dict[str, float]
        Invalid keyword arguments passed to the public adapter.

    Returns
    -------
    None
        The assertion checks parameter validation.
    """
    with pytest.raises(ValueError):
        layout_grip_pipeline(_path_edge_index(2), 2, **kwargs)

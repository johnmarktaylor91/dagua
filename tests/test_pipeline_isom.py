"""Regression tests for the JUNG ISOM pipeline."""

from __future__ import annotations

import ast
from pathlib import Path

import torch

import dagua
from dagua import LayoutConfig
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.isom import JavaRandom, build_isom_pipeline, layout_isom_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a standard ``[2, E]`` edge tensor.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed edges as ``(source, target)`` pairs.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    sources, targets = zip(*edges)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def _assert_valid_positions(positions: torch.Tensor, num_nodes: int) -> None:
    """Assert that a coordinate tensor is finite and has expected shape.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor to validate.
    num_nodes : int
        Expected number of rows.

    Returns
    -------
    None
        The function asserts on invalid positions.
    """
    assert positions.shape == (num_nodes, 2)
    assert torch.isfinite(positions).all()


def test_isom_registered_for_public_dispatch() -> None:
    """The algorithm name should resolve through the pipeline registry.

    Returns
    -------
    None
        The assertion checks lazy registry import.
    """
    assert PIPELINE_REGISTRY["isom"] == (
        "dagua.layout.ops.pipelines.isom",
        "layout_isom_pipeline",
    )
    assert get_pipeline_function("isom") is layout_isom_pipeline


def test_isom_java_random_sequence_matches_jdk_random() -> None:
    """The local Java RNG port should match ``java.util.Random`` samples.

    Returns
    -------
    None
        The assertion pins the first five ``nextDouble`` values for seed 42.
    """
    rng = JavaRandom(42)
    samples = [rng.next_double() for _ in range(5)]

    assert samples == [
        0.7275636800328681,
        0.6832234717598454,
        0.30871945533265976,
        0.27707849007413665,
        0.6655489517945736,
    ]


def test_isom_zero_steps_matches_seeded_jung_initialization() -> None:
    """Zero steps should expose JUNG's seeded random-location initializer.

    Returns
    -------
    None
        The assertion pins Java ``RandomLocationTransformer`` output.
    """
    edge_index = _edge_index([(0, 1), (1, 2)])

    positions = layout_isom_pipeline(
        edge_index=edge_index,
        num_nodes=3,
        steps=0,
        seed=42,
        fidelity_dtype=torch.float64,
    )

    expected = torch.tensor(
        [
            [436.53820801972086, 409.93408305590725],
            [185.23167319959586, 166.247094044482],
            [399.32937107674417, 542.0233588033069],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(positions, expected, atol=1.0e-12, rtol=0.0)


def test_isom_three_steps_pin_source_port_dynamics() -> None:
    """Short ISOM dynamics should match the source-port oracle.

    Returns
    -------
    None
        The assertion covers random point generation, winner choice, graph
        neighborhood BFS, adaptation, and cooling as one regression pin.
    """
    edge_index = _edge_index([(0, 1), (1, 2)])

    positions = layout_isom_pipeline(
        edge_index=edge_index,
        num_nodes=3,
        steps=3,
        seed=42,
        fidelity_dtype=torch.float64,
    )

    expected = torch.tensor(
        [
            [350.2198693683456, 364.24802840706235],
            [297.32554421702474, 350.68593578619755],
            [399.4901933244124, 532.5723655705046],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(positions, expected, atol=1.0e-12, rtol=0.0)


def test_isom_is_deterministic_and_seed_sensitive() -> None:
    """ISOM should be deterministic for one seed and distinct across seeds.

    Returns
    -------
    None
        The assertions check seeded reproducibility and non-degenerate output.
    """
    edge_index = _edge_index([(0, 1), (0, 2), (2, 3), (3, 4), (4, 1)])

    first = layout_isom_pipeline(edge_index=edge_index, num_nodes=5, steps=25, seed=7)
    second = layout_isom_pipeline(edge_index=edge_index, num_nodes=5, steps=25, seed=7)
    different_seed = layout_isom_pipeline(edge_index=edge_index, num_nodes=5, steps=25, seed=8)

    assert torch.equal(first, second)
    assert not torch.equal(first, different_seed)
    _assert_valid_positions(first, num_nodes=5)


def test_build_isom_pipeline_direct_execution() -> None:
    """The composed ISOM pipeline object should execute directly.

    Returns
    -------
    None
        The test asserts finite output coordinates from direct pipeline use.
    """
    edge_index = _edge_index([(0, 1), (0, 2), (2, 3), (2, 4), (4, 5)])
    problem = LayoutProblem(edge_index=edge_index, num_nodes=6, seed=99)

    final_state = build_isom_pipeline().apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )

    assert final_state.pos is not None
    _assert_valid_positions(final_state.pos, num_nodes=6)


def test_layout_dispatch_accepts_isom_algorithm() -> None:
    """Public layout dispatch should accept ``algorithm='isom'``.

    Returns
    -------
    None
        The test asserts finite public API output coordinates.
    """
    graph = dagua.DaguaGraph()
    for node in range(6):
        graph.add_node(str(node))
    for source, target in [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)]:
        graph.add_edge(str(source), str(target))

    positions = dagua.layout(graph, LayoutConfig(algorithm="isom", steps=10, seed=11))

    _assert_valid_positions(positions, num_nodes=6)


def test_isom_pipeline_has_no_runtime_reference_delegation() -> None:
    """The runtime pipeline must not shell out to JUNG or import competitors.

    Returns
    -------
    None
        The AST assertion keeps reference engines out of production layout.
    """
    source_path = Path("dagua/layout/ops/pipelines/isom.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_imports = {"subprocess", "dagua.eval.competitors", "jpype", "py4j"}
    forbidden_calls = {"run", "Popen", "check_output", "check_call"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = {alias.name for alias in node.names}
            assert forbidden_imports.isdisjoint(imported)
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert all(not module.startswith(name) for name in forbidden_imports)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden_calls

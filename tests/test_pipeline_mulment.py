"""Regression tests for the MulMent pipeline."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import pytest
import torch

import dagua
from dagua.eval.competitors import get_competitor
from dagua.eval.variants import base_pairings, get_variant, original_variant_name
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.mulment import (
    MulMentConfig,
    build_mulment_pipeline,
    layout_mulment_pipeline,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import get_op_class, list_ops


def _edge_index_from_edges(edges: Iterable[tuple[int, int]]) -> torch.Tensor:
    """Build an edge tensor from edge tuples.

    Parameters
    ----------
    edges : Iterable[tuple[int, int]]
        Edge pairs.

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
    """Build a path graph edge tensor.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Path edge tensor with shape ``[2, E]``.
    """
    return _edge_index_from_edges((index, index + 1) for index in range(max(0, num_nodes - 1)))


def test_mulment_pipeline_and_op_are_registered() -> None:
    """MulMent should resolve through pipeline and op registries.

    Returns
    -------
    None
        Assertions cover registry entries.
    """
    assert PIPELINE_REGISTRY["mulment"] == (
        "dagua.layout.ops.pipelines.mulment",
        "layout_mulment_pipeline",
    )
    assert get_pipeline_function("mulment") is layout_mulment_pipeline
    assert get_op_class("mulment_coarsen_refine").__name__ == "MulMentCoarsenAndRefine"
    assert "mulment_coarsen_refine" in list_ops()


def test_mulment_reference_competitor_is_paired() -> None:
    """MulMent reference and reimplementation should be benchmark paired.

    Returns
    -------
    None
        Assertions cover competitor and variant registry wiring.
    """
    variant = get_variant("mulment_reimpl_default")

    assert get_competitor("mulment_reference") is not None
    assert get_competitor("mulment_reimpl") is not None
    assert base_pairings()["mulment_reimpl"] == ["mulment_reference"]
    assert variant is not None
    assert original_variant_name(variant) == "mulment_reference__for__mulment_reimpl_default"


def test_mulment_is_seed_deterministic_and_seed_sensitive() -> None:
    """Seeded MulMent runs should repeat exactly and react to seed changes.

    Returns
    -------
    None
        Assertions pin deterministic RNG behavior.
    """
    edge_index = _path_edge_index(8)

    first = layout_mulment_pipeline(edge_index, 8, steps=4, seed=7, fidelity_dtype=torch.float64)
    second = layout_mulment_pipeline(edge_index, 8, steps=4, seed=7, fidelity_dtype=torch.float64)
    different = layout_mulment_pipeline(
        edge_index,
        8,
        steps=4,
        seed=9,
        fidelity_dtype=torch.float64,
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert torch.isfinite(first).all()


def test_build_mulment_pipeline_matches_public_adapter() -> None:
    """Direct pipeline composition should match the public adapter.

    Returns
    -------
    None
        The assertion compares direct and wrapper execution.
    """
    edge_index = _edge_index_from_edges([(0, 1), (1, 2), (2, 3), (3, 0)])
    config = MulMentConfig(steps=3, fidelity_dtype=torch.float64)
    problem = LayoutProblem(edge_index=edge_index, num_nodes=4, seed=11)

    final_state = build_mulment_pipeline(config).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    assert final_state.pos is not None
    public = layout_mulment_pipeline(edge_index, 4, steps=3, seed=11, fidelity_dtype=torch.float64)

    assert torch.equal(final_state.pos, public)


def test_mulment_builds_label_propagation_hierarchy_in_isolation() -> None:
    """MulMent should expose the KaDraw-style LP hierarchy before layout.

    Returns
    -------
    None
        Assertions pin level sizes and merge maps independently of positions.
    """
    edge_index = _edge_index_from_edges(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (0, 6),
            (1, 7),
            (2, 8),
            (3, 9),
            (4, 10),
            (5, 11),
        ]
    )
    config = MulMentConfig(steps=0, max_levels=4, fidelity_dtype=torch.float64)
    problem = LayoutProblem(edge_index=edge_index, num_nodes=12, seed=7)

    final_state = build_mulment_pipeline(config).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )

    assert final_state.hierarchy is not None
    assert [(level.num_fine, level.num_nodes) for level in final_state.hierarchy] == [
        (12, 12),
        (12, 12),
        (12, 7),
        (7, 6),
    ]
    assert final_state.hierarchy[2].fine_to_coarse is not None
    assert final_state.hierarchy[2].fine_to_coarse.tolist() == [
        0,
        1,
        1,
        2,
        3,
        3,
        4,
        5,
        5,
        2,
        6,
        6,
    ]
    assert final_state.hierarchy[3].fine_to_coarse is not None
    assert final_state.hierarchy[3].fine_to_coarse.tolist() == [
        0,
        1,
        2,
        3,
        0,
        4,
        5,
    ]


def test_mulment_engine_dispatch_returns_finite_positions() -> None:
    """The public engine should dispatch ``algorithm='mulment'``.

    Returns
    -------
    None
        Assertions verify shape and finite output.
    """
    graph = dagua.DaguaGraph()
    for node in range(5):
        graph.add_node(str(node))
    for source in range(4):
        graph.add_edge(str(source), str(source + 1))

    positions = dagua.layout(
        graph,
        dagua.LayoutConfig(
            algorithm="mulment",
            seed=3,
            steps=2,
            algorithm_params={"fidelity_dtype": torch.float64},
        ),
    )

    assert positions.shape == (5, 2)
    assert torch.isfinite(positions).all()


def test_mulment_pipeline_does_not_delegate_to_reference_runtime() -> None:
    """Production MulMent code must not call a reference binary.

    Returns
    -------
    None
        AST assertions reject subprocess and FFI call paths.
    """
    path = Path("dagua/layout/ops/pipelines/mulment.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_roots = {"subprocess", "ctypes", "cffi", "jpype", "jnius", "dagua.eval.competitors"}
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
    [{"steps": -1}, {"alpha": -1.0}, {"tol": 0.0}, {"inner_iterations": -1}, {"max_levels": -1}],
)
def test_mulment_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    """Invalid MulMent parameters should fail before layout work starts.

    Parameters
    ----------
    kwargs : dict[str, float]
        Invalid keyword arguments.

    Returns
    -------
    None
        The assertion checks validation.
    """
    with pytest.raises(ValueError):
        layout_mulment_pipeline(_path_edge_index(3), 3, **kwargs)


def test_mulment_glibc_rand_stream_is_bit_exact() -> None:
    """The glibc ``rand()`` replica must match glibc's TYPE_3 outputs.

    Ground-truth values were captured from a C program calling
    ``srand(seed); rand();`` against glibc on the fidelity host.

    Returns
    -------
    None
        Assertions pin the exact random streams.
    """
    from dagua.layout.ops.pipelines.mulment import _GlibcRand

    expected = {
        13: [1358590890, 733184381, 1941561279, 279246991, 1306448764, 718348024],
        17: [1227918265, 3978157, 263514239, 1969574147, 1833982879, 488658959],
        23: [1562469902, 1039845534, 2025653534, 739593874, 994290584, 1198075102],
    }
    for seed, values in expected.items():
        rng = _GlibcRand(seed)
        assert [rng.rand() for _ in range(len(values))] == values


def test_mulment_engine_default_steps_selects_reference_preset() -> None:
    """``steps == 0`` (engine default) must select the KaDraw fast preset.

    Returns
    -------
    None
        The assertion compares default-dispatch output with the preset run.
    """
    edge_index = _path_edge_index(8)

    preset = layout_mulment_pipeline(edge_index, 8, seed=5, fidelity_dtype=torch.float64)
    zero_steps = layout_mulment_pipeline(
        edge_index,
        8,
        steps=0,
        seed=5,
        fidelity_dtype=torch.float64,
    )

    assert torch.equal(preset, zero_steps)

"""Regression tests for the NNP-NET pipeline."""

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
from dagua.layout.ops.pipelines.nnpnet import (
    NNPNetConfig,
    _pivot_distance_embedding,
    build_nnpnet_pipeline,
    layout_nnpnet_pipeline,
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


def test_nnpnet_pipeline_and_op_are_registered() -> None:
    """NNP-NET should resolve through pipeline and op registries.

    Returns
    -------
    None
        Assertions cover registry entries.
    """
    assert PIPELINE_REGISTRY["nnpnet"] == (
        "dagua.layout.ops.pipelines.nnpnet",
        "layout_nnpnet_pipeline",
    )
    assert get_pipeline_function("nnpnet") is layout_nnpnet_pipeline
    assert get_op_class("nnpnet_project_neighborhood").__name__ == "NNPNetProjectNeighborhood"
    assert "nnpnet_project_neighborhood" in list_ops()


def test_nnpnet_reference_competitor_is_paired() -> None:
    """NNP-NET reference and reimplementation should be benchmark paired.

    Returns
    -------
    None
        Assertions cover competitor and variant registry wiring.
    """
    variant = get_variant("nnpnet_reimpl_default")

    assert get_competitor("nnpnet_reference") is not None
    assert get_competitor("nnpnet_reimpl") is not None
    assert base_pairings()["nnpnet_reimpl"] == ["nnpnet_reference"]
    assert variant is not None
    assert original_variant_name(variant) == "nnpnet_reference__for__nnpnet_reimpl_default"


def test_nnpnet_pivot_embedding_pins_farthest_pivot_order() -> None:
    """Pivot-distance embedding should follow the reference farthest rule.

    Returns
    -------
    None
        The assertion pins normalized path distances.
    """
    distances = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    embedding = _pivot_distance_embedding(distances, embedding_size=2)

    expected = torch.tensor(
        [[0.0, 1.0], [1.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 1.0 / 3.0], [1.0, 0.0]],
        dtype=torch.float64,
    )
    assert torch.equal(embedding, expected)


def test_nnpnet_is_seed_deterministic_and_seed_sensitive() -> None:
    """Seeded NNP-NET runs should repeat exactly and react to teacher seed changes.

    Returns
    -------
    None
        Assertions pin deterministic behavior.
    """
    edge_index = _path_edge_index(6)

    first = layout_nnpnet_pipeline(edge_index, 6, steps=250, seed=7, fidelity_dtype=torch.float64)
    second = layout_nnpnet_pipeline(edge_index, 6, steps=250, seed=7, fidelity_dtype=torch.float64)
    different = layout_nnpnet_pipeline(
        edge_index,
        6,
        steps=250,
        seed=9,
        fidelity_dtype=torch.float64,
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert torch.isfinite(first).all()


def test_build_nnpnet_pipeline_matches_public_adapter() -> None:
    """Direct pipeline composition should match the public adapter.

    Returns
    -------
    None
        The assertion compares direct and wrapper execution.
    """
    edge_index = _edge_index_from_edges([(0, 1), (1, 2), (2, 3), (3, 0)])
    config = NNPNetConfig(steps=250, embedding_size=3, fidelity_dtype=torch.float64)
    problem = LayoutProblem(edge_index=edge_index, num_nodes=4, seed=11)

    final_state = build_nnpnet_pipeline(config).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    assert final_state.pos is not None
    public = layout_nnpnet_pipeline(
        edge_index,
        4,
        steps=250,
        seed=11,
        embedding_size=3,
        fidelity_dtype=torch.float64,
    )

    assert torch.equal(final_state.pos, public)


def test_nnpnet_engine_dispatch_returns_finite_positions() -> None:
    """The public engine should dispatch ``algorithm='nnpnet'``.

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
            algorithm="nnpnet",
            seed=3,
            steps=250,
            algorithm_params={"embedding_size": 3, "fidelity_dtype": torch.float64},
        ),
    )

    assert positions.shape == (5, 2)
    assert torch.isfinite(positions).all()


def test_nnpnet_pipeline_does_not_delegate_to_reference_runtime() -> None:
    """Production NNP-NET code must not call a reference binary.

    Returns
    -------
    None
        AST assertions reject subprocess and FFI call paths.
    """
    path = Path("dagua/layout/ops/pipelines/nnpnet.py")
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
    [
        {"steps": -1},
        {"perplexity": 0.0},
        {"embedding_size": 0},
        {"subgraph_size": 1},
        {"ridge": -1.0},
        {"output_scale": 0.0},
    ],
)
def test_nnpnet_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    """Invalid NNP-NET parameters should fail before layout work starts.

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
        layout_nnpnet_pipeline(_path_edge_index(3), 3, **kwargs)

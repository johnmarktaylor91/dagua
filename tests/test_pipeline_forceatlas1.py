"""Regression tests for the Gephi ForceAtlas1 pipeline."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import pytest
import torch

from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.forceatlas1 import (
    ForceAtlas1Config,
    build_forceatlas1_pipeline,
    layout_forceatlas1_pipeline,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


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
        Directed path graph edge tensor with shape ``[2, E]``.
    """
    return _edge_index_from_edges((index, index + 1) for index in range(max(num_nodes - 1, 0)))


def _run_pipeline_direct(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    config: ForceAtlas1Config,
    seed: int,
    node_sizes: torch.Tensor | None = None,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute the raw ForceAtlas1 pipeline object.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    config : ForceAtlas1Config
        ForceAtlas1 configuration.
    seed : int
        Java-compatible initialization seed.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[N, 2]``.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.
    """
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    final_state = build_forceatlas1_pipeline(config=config).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    assert final_state.pos is not None
    return final_state.pos


def test_forceatlas1_registered_for_public_dispatch() -> None:
    """The algorithm name should resolve through the pipeline registry.

    Returns
    -------
    None
        The assertion checks lazy registry import.
    """
    assert PIPELINE_REGISTRY["forceatlas1"] == (
        "dagua.layout.ops.pipelines.forceatlas1",
        "layout_forceatlas1_pipeline",
    )
    assert get_pipeline_function("forceatlas1") is layout_forceatlas1_pipeline


def test_forceatlas1_zero_steps_matches_gephi_java_random_initialization() -> None:
    """Zero steps should expose Gephi's seeded all-zero init path.

    Returns
    -------
    None
        The assertion pins Java ``Random`` initialization values.
    """
    edge_index = _path_edge_index(3)

    positions = layout_forceatlas1_pipeline(
        edge_index,
        3,
        steps=0,
        seed=42,
        fidelity_dtype=torch.float64,
    )

    expected = torch.tensor(
        [
            [237.56365966796875, 193.22344970703125],
            [-181.28054809570312, -212.9215087890625],
            [175.5489501953125, 413.37225341796875],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(positions, expected, atol=1.0e-5, rtol=0.0)


def test_forceatlas1_three_steps_pins_default_gephi_dynamics() -> None:
    """Default dynamics should match the source-port regression oracle.

    Returns
    -------
    None
        The assertion pins inertia, degree-weighted repulsion, attraction,
        gravity, speed, freeze-balance, and displacement limiting together.
    """
    edge_index = _path_edge_index(3)

    positions = layout_forceatlas1_pipeline(
        edge_index,
        3,
        steps=3,
        seed=42,
        fidelity_dtype=torch.float64,
    )

    expected = torch.tensor(
        [
            [230.40919494628906, 186.32139587402344],
            [-172.42253112792969, -201.14004516601562],
            [170.09187316894531, 403.72854614257812],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(positions, expected, atol=1.0e-5, rtol=0.0)


def test_forceatlas1_variants_are_deterministic_and_distinct() -> None:
    """Outbound distribution and freeze-balance variants should be wired.

    Returns
    -------
    None
        The assertions check determinism and that materially different
        variants do not collapse to the default path.
    """
    edge_index = _path_edge_index(3)

    default = layout_forceatlas1_pipeline(edge_index, 3, steps=3, seed=42)
    default_again = layout_forceatlas1_pipeline(edge_index, 3, steps=3, seed=42)
    outbound = layout_forceatlas1_pipeline(
        edge_index,
        3,
        steps=3,
        seed=42,
        outbound_attraction_distribution=True,
    )
    no_freeze = layout_forceatlas1_pipeline(
        edge_index,
        3,
        steps=3,
        seed=42,
        freeze_balance=False,
    )

    assert torch.equal(default, default_again)
    assert not torch.equal(default, outbound)
    assert not torch.equal(default, no_freeze)
    assert torch.isfinite(outbound).all()
    assert torch.isfinite(no_freeze).all()


def test_forceatlas1_java_style_aliases_match_python_names() -> None:
    """Java-style option aliases should match Pythonic parameter names.

    Returns
    -------
    None
        The assertion covers ``outboundAttractionDistribution``,
        ``adjustSizes``, and ``freezeBalance``.
    """
    edge_index = _path_edge_index(4)
    node_sizes = torch.full((4, 2), 20.0)

    python_names = layout_forceatlas1_pipeline(
        edge_index,
        4,
        node_sizes=node_sizes,
        steps=4,
        seed=11,
        outbound_attraction_distribution=True,
        adjust_sizes=True,
        freeze_balance=False,
    )
    java_names = layout_forceatlas1_pipeline(
        edge_index,
        4,
        node_sizes=node_sizes,
        steps=4,
        seed=11,
        outboundAttractionDistribution=True,
        adjustSizes=True,
        freezeBalance=False,
    )

    assert torch.equal(python_names, java_names)


def test_build_forceatlas1_pipeline_matches_public_adapter() -> None:
    """The composable pipeline object should match the public adapter.

    Returns
    -------
    None
        The assertion verifies the ``Op`` composition path.
    """
    edge_index = _edge_index_from_edges([(0, 1), (0, 2), (2, 3), (3, 1)])
    edge_weights = torch.tensor([1.0, 0.5, 2.0, 1.5], dtype=torch.float64)
    config = ForceAtlas1Config(
        steps=5,
        outbound_attraction_distribution=True,
        adjust_sizes=True,
        fidelity_dtype=torch.float64,
    )
    node_sizes = torch.full((4, 2), 8.0)

    direct = _run_pipeline_direct(
        edge_index,
        4,
        config=config,
        seed=7,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
    )
    public = layout_forceatlas1_pipeline(
        edge_index,
        4,
        node_sizes=node_sizes,
        steps=5,
        seed=7,
        edge_weights=edge_weights,
        outbound_attraction_distribution=True,
        adjust_sizes=True,
        fidelity_dtype=torch.float64,
    )

    assert torch.equal(direct, public)


def test_forceatlas1_pipeline_does_not_delegate_to_reference_runtime() -> None:
    """Production ForceAtlas1 code must not call a Gephi reference runtime.

    Returns
    -------
    None
        The AST guard rejects imports from competitor adapters, subprocesses,
        JPype, Py4J, or Java bridge modules.
    """
    path = Path("dagua/layout/ops/pipelines/forceatlas1.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_roots = {
        "dagua.eval.competitors",
        "subprocess",
        "jpype",
        "py4j",
        "jnius",
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
        {"cooling": 0.0},
        {"max_displacement": 0.0},
    ],
)
def test_forceatlas1_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    """Invalid Gephi parameters should fail before the solve loop.

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
        layout_forceatlas1_pipeline(_path_edge_index(2), 2, **kwargs)

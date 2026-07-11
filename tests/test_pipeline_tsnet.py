"""Exact-fidelity tests for the composable tsNET pipeline."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest
import torch

from dagua.layout.classic.tsnet import layout_tsnet
from dagua.layout.ops.pipelines.tsnet import (
    _joint_probabilities,
    build_tsnet_pipeline,
    layout_tsnet_pipeline,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.tsnet import (
    TsnetGradientStep,
    TsnetGradientStepConfig,
    TsnetInitializePositions,
    TsnetInitializePositionsConfig,
    TsnetPrepareState,
)


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
    """Build a small disconnected graph with two components and isolates.

    Returns
    -------
    torch.Tensor
        Directed edge tensor for the disconnected graph.
    """
    return _edge_index_from_edges([(0, 1), (1, 2), (3, 4)])


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
    """Assert that two tsNET outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic tsNET.
    pipeline : torch.Tensor
        Output from the composable pipeline.

    Returns
    -------
    None
        This helper asserts exact equality.
    """
    assert classic.dtype == pipeline.dtype
    assert classic.device == pipeline.device
    assert torch.equal(classic, pipeline), (
        f"Outputs differ.\n"
        f"  max abs diff: {(classic - pipeline).abs().max().item()}\n"
        f"  classic[:3]:  {classic[:3].tolist()}\n"
        f"  pipeline[:3]: {pipeline[:3].tolist()}"
    )


def _run_pipeline_direct(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    steps: int,
    seed: int,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute ``build_tsnet_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Number of optimization updates.
    seed : int
        Random seed used for initialization.
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
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_tsnet_pipeline(steps=steps).apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


def test_tsnet_prepare_state_populates_typed_distance_matrix() -> None:
    """tsNET preprocessing should cache all-pairs distances on the typed field."""
    problem = LayoutProblem(
        edge_index=_path_edge_index(4),
        num_nodes=4,
        seed=42,
    )
    state = SolveState(extras={"tsnet_perplexity": 3.0})

    prepared = TsnetPrepareState().apply(problem, state, RuntimeContext())

    assert prepared.distance_matrix is not None
    assert prepared.distance_matrix.tolist() == [
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0, 1.0],
        [3.0, 2.0, 1.0, 0.0],
    ]
    assert "tsnet_probabilities" in prepared.extras


def test_tsnet_fidelity_initializer_uses_numpy_random_state() -> None:
    """Fidelity initialization should match sklearn's NumPy MT draws."""
    problem = LayoutProblem(
        edge_index=_path_edge_index(4),
        num_nodes=4,
        seed=17,
    )

    initialized = TsnetInitializePositions(
        TsnetInitializePositionsConfig(fidelity_mode=True)
    ).apply(problem, SolveState(), RuntimeContext())

    assert initialized.pos is not None
    expected = torch.from_numpy(
        (1.0e-4 * np.random.RandomState(17).standard_normal((4, 2))).astype(np.float32)
    )
    torch.testing.assert_close(initialized.pos.detach().cpu(), expected)


def test_tsnet_gradient_step_defaults_to_unit_gradient_scale() -> None:
    """The default TSNET step should keep the native autograd gradient scale."""
    initial_pos = torch.tensor(
        [[-0.1, 0.2], [0.3, -0.4], [0.5, 0.1]],
        dtype=torch.float32,
    )
    probabilities = torch.tensor(
        [
            [1.0e-12, 0.18, 0.22],
            [0.18, 1.0e-12, 0.10],
            [0.22, 0.10, 1.0e-12],
        ],
        dtype=torch.float32,
    )

    explicit_state = _tsnet_gradient_state(initial_pos, probabilities)
    default_state = _tsnet_gradient_state(initial_pos, probabilities)

    explicit_result = TsnetGradientStep(TsnetGradientStepConfig(gradient_scale=1.0)).apply(
        LayoutProblem(edge_index=_path_edge_index(3), num_nodes=3),
        explicit_state,
        RuntimeContext(),
    )
    default_result = TsnetGradientStep().apply(
        LayoutProblem(edge_index=_path_edge_index(3), num_nodes=3),
        default_state,
        RuntimeContext(),
    )

    assert explicit_result.pos is not None
    assert default_result.pos is not None
    torch.testing.assert_close(default_result.pos.detach(), explicit_result.pos.detach())


def _tsnet_gradient_state(initial_pos: torch.Tensor, probabilities: torch.Tensor) -> SolveState:
    """Build the minimal state needed by :class:`TsnetGradientStep`.

    Parameters
    ----------
    initial_pos : torch.Tensor
        Initial positions with shape ``[N, 2]``.
    probabilities : torch.Tensor
        High-dimensional affinity matrix with shape ``[N, N]``.

    Returns
    -------
    SolveState
        State populated with TSNET optimizer extras.
    """
    pos = initial_pos.clone().requires_grad_(True)
    return SolveState(
        pos=pos,
        extras={
            "tsnet_probabilities": probabilities,
            "tsnet_early_exaggeration": 1.0,
            "tsnet_early_exaggeration_steps": 0,
            "tsnet_min_gain": 0.01,
            "tsnet_min_distance": 1.0e-12,
            "tsnet_early_learning_rate": 1.0,
            "tsnet_late_learning_rate": 1.0,
            "tsnet_update": torch.zeros_like(pos),
            "tsnet_gains": torch.ones_like(pos),
            "tsnet_best_error": float("inf"),
            "tsnet_best_iter": 0,
        },
    )


def test_vendored_joint_probabilities_match_classic_reference_math() -> None:
    """Vendored affinities should be bit-exact with the reference primitive."""
    sklearn_tsne = pytest.importorskip("sklearn.manifold._t_sne")
    distances = np.array(
        [
            [0.0, 1.0, 4.0, 9.0, 16.0],
            [1.0, 0.0, 1.0, 4.0, 9.0],
            [4.0, 1.0, 0.0, 1.0, 4.0],
            [9.0, 4.0, 1.0, 0.0, 1.0],
            [16.0, 9.0, 4.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    expected = sklearn_tsne._joint_probabilities(distances, 3.0, 0)
    actual = _joint_probabilities(distances, 3.0)

    assert np.array_equal(actual, expected)
    assert actual.shape == (10,)
    assert actual.dtype == np.float64


def test_tsnet_production_pipeline_has_no_sklearn_import() -> None:
    """The classic tsNET production pipeline must not delegate to sklearn."""
    project_root = Path(__file__).resolve().parents[1]
    production_files = (
        project_root / "dagua" / "layout" / "ops" / "pipelines" / "tsnet.py",
        project_root / "dagua" / "layout" / "ops" / "tsnet.py",
    )
    offenders: list[str] = []
    for path in production_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name == "sklearn" or alias.name.startswith("sklearn.") for alias in node.names
            ):
                offenders.append(f"{path.relative_to(project_root)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom) and (
                node.module == "sklearn" or (node.module or "").startswith("sklearn.")
            ):
                offenders.append(f"{path.relative_to(project_root)}:{node.lineno}")

    assert offenders == []


class TestTsnetPipelineFidelity:
    """Bit-exact regression coverage for the tsNET pipeline."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(0, 42), (1, 42), (2, 42), (5, 42), (5, 99), (10, 42)],
    )
    def test_layout_tsnet_pipeline_matches_classic_for_requested_sizes(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """The adapter should match classic tsNET exactly for the requested cases."""
        edge_index = _path_edge_index(num_nodes)
        # Use fewer steps for test speed -- fidelity is independent of step count.
        steps = 50

        classic = layout_tsnet(edge_index=edge_index, num_nodes=num_nodes, steps=steps, seed=seed)
        pipeline = layout_tsnet_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            steps=steps,
            seed=seed,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_tsnet_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted tsNET distances should remain bit-identical in the pipeline."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_tsnet(
            edge_index=edge_index,
            num_nodes=6,
            steps=50,
            seed=17,
            edge_weights=edge_weights,
        )
        pipeline = layout_tsnet_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            steps=50,
            seed=17,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_tsnet_pipeline_matches_classic_on_disconnected_graph(self) -> None:
        """Disconnected components and isolated nodes should match exactly."""
        edge_index = _disconnected_edge_index()

        classic = layout_tsnet(edge_index=edge_index, num_nodes=7, steps=50, seed=99)
        pipeline = layout_tsnet_pipeline(edge_index=edge_index, num_nodes=7, steps=50, seed=99)

        _assert_exact_match(classic, pipeline)

    def test_build_tsnet_pipeline_matches_classic_on_complete_graph(self) -> None:
        """The raw pipeline object should match classic tsNET on a dense graph."""
        edge_index = _complete_edge_index(5)

        classic = layout_tsnet(edge_index=edge_index, num_nodes=5, steps=50, seed=7)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=5, steps=50, seed=7)

        _assert_exact_match(classic, pipeline)

    def test_layout_tsnet_pipeline_zero_steps(self) -> None:
        """Zero steps should still produce valid output matching classic."""
        edge_index = _path_edge_index(5)

        classic = layout_tsnet(edge_index=edge_index, num_nodes=5, steps=0, seed=42)
        pipeline = layout_tsnet_pipeline(edge_index=edge_index, num_nodes=5, steps=0, seed=42)

        _assert_exact_match(classic, pipeline)

    def test_layout_tsnet_fidelity_mode_matches_sklearn_exact_reference(self) -> None:
        """Fidelity mode should match the sklearn exact t-SNE reference path."""
        sklearn_manifold = pytest.importorskip("sklearn.manifold")
        scipy_csgraph = pytest.importorskip("scipy.sparse.csgraph")
        scipy_sparse = pytest.importorskip("scipy.sparse")
        edge_index = _path_edge_index(8)
        num_nodes = 8
        rows = edge_index[0].numpy()
        cols = edge_index[1].numpy()
        adjacency = scipy_sparse.csr_matrix(
            (
                np.ones(rows.shape[0] * 2, dtype=np.float32),
                (np.concatenate([rows, cols]), np.concatenate([cols, rows])),
            ),
            shape=(num_nodes, num_nodes),
        )
        distances = scipy_csgraph.shortest_path(adjacency, directed=False).astype(
            np.float32,
            copy=False,
        )
        reference = sklearn_manifold.TSNE(
            n_components=2,
            metric="precomputed",
            init="random",
            random_state=3,
            perplexity=7.0,
            method="exact",
            max_iter=250,
        ).fit_transform(distances)

        pipeline = layout_tsnet_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            perplexity=30.0,
            steps=250,
            seed=3,
            fidelity_mode=True,
        )

        torch.testing.assert_close(pipeline.cpu(), torch.tensor(reference, dtype=torch.float32))

"""Regression tests for Pivot-MDS fidelity controls."""

from __future__ import annotations

from typing import Any

import torch

from dagua.eval.variants import VARIANT_REGISTRY
from dagua.graph import DaguaGraph
from dagua.layout.ops.embed import _pivot_mds_coordinates
from dagua.layout.ops.pipelines.pivot_mds import build_pivot_mds_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Build an undirected path edge tensor.

    Parameters
    ----------
    num_nodes : int
        Number of path nodes.

    Returns
    -------
    torch.Tensor
        Edge index tensor with shape ``[2, E]``.
    """
    edges: list[tuple[int, int]] = []
    for node_idx in range(num_nodes - 1):
        edges.append((node_idx, node_idx + 1))
        edges.append((node_idx + 1, node_idx))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_pivot_mds_first_node_mode_starts_at_zero() -> None:
    """Confirm OGDF-compatible pivot selection starts from node zero."""
    edge_index = _path_edge_index(4)
    problem = LayoutProblem(edge_index=edge_index, num_nodes=4, seed=123)
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))

    final_state = build_pivot_mds_pipeline(
        n_pivots=2,
        first_pivot="first_node",
        compute_dtype=torch.float64,
    ).apply(problem, state, ctx)

    assert final_state.pivot_indices is not None
    assert final_state.pivot_indices.tolist() == [0, 3]


def test_pivot_mds_float64_mode_keeps_internal_distances_double() -> None:
    """Confirm fidelity mode stores pivot distances in double precision."""
    edge_index = _path_edge_index(3)
    problem = LayoutProblem(edge_index=edge_index, num_nodes=3, seed=123)
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))

    final_state = build_pivot_mds_pipeline(
        n_pivots=2,
        first_pivot="first_node",
        compute_dtype="float64",
    ).apply(problem, state, ctx)

    assert final_state.pivot_distances is not None
    assert final_state.pivot_distances.dtype == torch.float64
    assert final_state.pos is not None
    assert final_state.pos.dtype == torch.float32


def test_pivot_mds_variants_forward_params_to_reimpl_and_ogdf() -> None:
    """Confirm Pivot-MDS variants align dagua and OGDF pivot counts."""
    pivot_variants = [
        variant for variant in VARIANT_REGISTRY if variant.base_engine == "classic_pivot_mds"
    ]

    assert pivot_variants
    for variant in pivot_variants:
        expected_pivots = variant.reimpl_params["n_pivots"]
        assert variant.reimpl_params["first_pivot"] == "first_node"
        assert variant.reimpl_params["compute_dtype"] == "float64"
        assert variant.reimpl_params["distance_scale"] == 100.0
        assert variant.reimpl_params["ogdf_path_special_case"] is True
        assert variant.original_params == {"n_pivots": expected_pivots}


def test_pivot_mds_ogdf_path_special_case_returns_raw_line() -> None:
    """Confirm OGDF path mode emits a straight line with OGDF edge cost."""
    from dagua.layout.ops.pipelines.pivot_mds import layout_pivot_mds_pipeline

    edge_index = _path_edge_index(4)
    pos = layout_pivot_mds_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        distance_scale=100.0,
        ogdf_path_special_case=True,
    )

    assert pos.tolist() == [[0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [300.0, 0.0]]


def test_pivot_mds_ogdf_fidelity_path_keeps_raw_scale() -> None:
    """Confirm the OGDF fidelity pipeline skips dagua extent normalization."""
    from dagua.layout.ops.pipelines.pivot_mds import layout_pivot_mds_pipeline

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0, 0, 2],
            [1, 0, 2, 1, 3, 2, 0, 3, 2, 0],
        ],
        dtype=torch.long,
    )
    raw_pos = layout_pivot_mds_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        n_pivots=4,
        first_pivot="first_node",
        compute_dtype=torch.float64,
        distance_scale=100.0,
        ogdf_path_special_case=True,
    )
    normalized_pos = layout_pivot_mds_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        n_pivots=4,
        seed=42,
    )

    raw_extent = float(torch.max(raw_pos.max(dim=0).values - raw_pos.min(dim=0).values).item())
    normalized_extent = float(
        torch.max(normalized_pos.max(dim=0).values - normalized_pos.min(dim=0).values).item()
    )

    assert raw_extent > 20.0
    assert normalized_extent <= 20.0 + 1.0e-5


def test_pivot_mds_coordinates_use_ogdf_sqrt_singular_scale() -> None:
    """Confirm Pivot-MDS coordinate recovery matches OGDF's final SVD scale."""
    distance_matrix = torch.tensor(
        [
            [0.0, 1.0, 2.0, 1.0, 2.0, 3.0],
            [3.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 1.0, 0.0, 1.0, 2.0],
            [2.0, 3.0, 2.0, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    coordinates = _pivot_mds_coordinates(distance_matrix, compute_dtype=torch.float64).to(
        dtype=torch.float64
    )
    squared = distance_matrix.square()
    centered = -0.5 * (
        squared
        - squared.mean(dim=1, keepdim=True)
        - squared.mean(dim=0, keepdim=True)
        + squared.mean()
    )
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    expected = vh[:2].transpose(0, 1) * singular_values[:2].sqrt().unsqueeze(0)
    old_scale = vh[:2].transpose(0, 1) * singular_values[:2].unsqueeze(0)

    assert torch.allclose(coordinates, expected, atol=1e-6)
    assert not torch.allclose(coordinates, old_scale, atol=1e-3)


def test_ogdf_pivot_mds_variant_params_reach_runner(monkeypatch: Any) -> None:
    """Confirm OGDF Pivot-MDS forwards ``n_pivots`` as runner JSON options."""
    from dagua.eval.competitors import ogdf_competitor

    captured_options: list[dict[str, Any] | None] = []

    def fake_is_connected_graph(graph: DaguaGraph) -> bool:
        """Pretend the graph is connected for adapter plumbing only.

        Parameters
        ----------
        graph : DaguaGraph
            Graph passed by the adapter.

        Returns
        -------
        bool
            Always ``True`` for this option-forwarding regression test.
        """
        return True

    def fake_run_ogdf(
        graph: DaguaGraph,
        algorithm: str,
        timeout: float,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Capture OGDF runner options without launching the subprocess.

        Parameters
        ----------
        graph : DaguaGraph
            Graph passed by the adapter.
        algorithm : str
            Runner algorithm name.
        timeout : float
            Runner timeout budget.
        seed : int | None, default=None
            Optional benchmark seed forwarded by newer OGDF adapters.
        options : dict[str, Any] | None, default=None
            JSON options that would be forwarded to the runner.

        Returns
        -------
        torch.Tensor
            Dummy position tensor with shape ``[N, 2]``.
        """
        assert seed is None
        assert algorithm == "pivot_mds"
        assert timeout == 300.0
        captured_options.append(options)
        return torch.zeros((graph.num_nodes, 2), dtype=torch.float32)

    monkeypatch.setattr(ogdf_competitor, "_is_connected_graph", fake_is_connected_graph)
    monkeypatch.setattr(ogdf_competitor, "_run_ogdf", fake_run_ogdf)

    graph = DaguaGraph()
    graph.add_node("a")
    graph.add_node("b")
    graph.add_edge("a", "b")
    result = ogdf_competitor.OGDFPivotMDS().layout_with_variant(
        graph,
        variant_params={"n_pivots": 10},
    )

    assert result.error is None
    assert captured_options == [{"numberOfPivots": 10}]

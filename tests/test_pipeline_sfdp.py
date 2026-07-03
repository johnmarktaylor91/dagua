"""Exact-fidelity tests for the composable SFDP pipeline."""

from __future__ import annotations

from typing import Iterable

import pytest
import torch

from dagua.layout.classic.sfdp import layout_sfdp
from dagua.layout.ops.pipelines.sfdp import (
    _decompose_graphviz_supervariables,
    _graphviz_sfdp_coarsen,
    build_sfdp_pipeline,
    layout_sfdp_pipeline,
)
from dagua.layout.ops.sfdp import (
    _GRAPH_KEY,
    _MAPPING_KEY,
    GraphData,
    GraphvizRandom,
    SFDPHierarchyConfig,
    _build_graph,
)
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
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


def _complete_multipartite_graph(groups: list[list[int]]) -> GraphData:
    """Build a weighted undirected complete multipartite graph.

    Parameters
    ----------
    groups : list[list[int]]
        Partition members. Nodes in the same group are not connected to each
        other, and every cross-group pair has unit weight.

    Returns
    -------
    GraphData
        SFDP graph data with unique undirected edges and sorted adjacency lists.
    """
    num_nodes = sum(len(group) for group in groups)
    node_to_group: dict[int, int] = {}
    for group_index, group in enumerate(groups):
        for node in group:
            node_to_group[node] = group_index

    edge_pairs: list[tuple[int, int]] = []
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for source in range(num_nodes):
        for target in range(source + 1, num_nodes):
            if node_to_group[source] == node_to_group[target]:
                continue
            edge_pairs.append((source, target))
            adjacency[source].append((target, 1.0))
            adjacency[target].append((source, 1.0))

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).transpose(0, 1).contiguous()
    edge_weight = torch.ones((len(edge_pairs),), dtype=torch.float32)
    return GraphData(
        num_nodes=num_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        adjacency=adjacency,
    )


def _edge_weight_dict(graph: GraphData) -> dict[tuple[int, int], float]:
    """Convert graph edges to a dictionary for golden-vector assertions.

    Parameters
    ----------
    graph : GraphData
        Graph whose unique undirected edges should be indexed.

    Returns
    -------
    dict[tuple[int, int], float]
        Mapping from ``(source, target)`` edge tuple to edge weight.
    """
    return {
        (int(graph.edge_index[0, edge_id].item()), int(graph.edge_index[1, edge_id].item())): float(
            graph.edge_weight[edge_id].item()
        )
        for edge_id in range(graph.edge_index.shape[1])
    }


def _assert_exact_match(classic: torch.Tensor, pipeline: torch.Tensor) -> None:
    """Assert that two SFDP outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic SFDP.
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


def _run_pipeline_direct(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    steps: int,
    seed: int,
    theta: float = 0.6,
    repulsive_exponent: float = -1.0,
    edge_weights: torch.Tensor | None = None,
    direction: str = "TB",
) -> torch.Tensor:
    """Execute ``build_sfdp_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Maximum number of iterations per level.
    seed : int
        Random seed.
    theta : float
        Barnes-Hut opening angle.
    repulsive_exponent : float
        SFDP repulsion exponent.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    direction : str, default="TB"
        Requested layout flow direction.

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
        direction=direction,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_sfdp_pipeline(
        steps=steps,
        theta=theta,
        repulsive_exponent=repulsive_exponent,
    ).apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


class TestSFDPPipelineFidelity:
    """Bit-exact regression coverage for the SFDP pipeline."""

    def test_graphviz_random_matches_glibc_rand_golden_sequence(self) -> None:
        """Graphviz RNG should match the ``srand(1)``/``rand`` C sequence."""
        generator = GraphvizRandom(seed=1)

        values = [generator.rand() for _ in range(10)]

        assert values == [
            1804289383,
            846930886,
            1681692777,
            1714636915,
            1957747793,
            424238335,
            719885386,
            1649760492,
            596516649,
            1189641421,
        ]

    def test_graphviz_random_permutation_matches_gv_permutation_golden(self) -> None:
        """Graphviz permutation should use ``gv_random`` Fisher-Yates swaps."""
        generator = GraphvizRandom(seed=1)

        order = generator.permutation(8)

        assert order == [2, 6, 5, 1, 0, 3, 4, 7]

    def test_graphviz_supervariable_decomposition_matches_reference_groups(self) -> None:
        """Graphviz supervariables should group identical matrix column patterns."""
        graph = _complete_multipartite_graph([[0, 1], [2, 3], [4, 5], [6, 7]])

        groups = _decompose_graphviz_supervariables(graph)

        assert groups == [[0, 1], [2, 3], [4, 5], [6, 7]]

    def test_graphviz_matrix_coarsening_matches_reference_complete_multipartite(
        self,
    ) -> None:
        """Matrix coarsening should match Graphviz ``R * A * P`` golden values."""
        graph = _complete_multipartite_graph([[0, 1], [2, 3], [4, 5], [6, 7]])
        generator = GraphvizRandom(seed=123)

        coarsened = _graphviz_sfdp_coarsen(
            graph=graph,
            generator=generator,
            config=SFDPHierarchyConfig(),
        )

        assert coarsened is not None
        fine_to_coarse, coarse_graph = coarsened
        assert fine_to_coarse.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
        assert coarse_graph.num_nodes == 4
        assert _edge_weight_dict(coarse_graph) == {
            (0, 1): 4.0,
            (0, 2): 4.0,
            (0, 3): 4.0,
            (1, 2): 4.0,
            (1, 3): 4.0,
            (2, 3): 4.0,
        }

    def test_build_sfdp_pipeline_fidelity_mode_uses_matrix_hierarchy(self) -> None:
        """The public fidelity flag should select the matrix coarsening op."""
        default_names = [op.name for op in build_sfdp_pipeline(steps=0).ops]
        fidelity_names = [op.name for op in build_sfdp_pipeline(steps=0, fidelity_mode=True).ops]

        assert default_names[1] == "sfdp_coarsen_hierarchy"
        assert fidelity_names[1] == "sfdp_graphviz_matrix_coarsen_hierarchy"

    def test_graphviz_matrix_hierarchy_populates_pipeline_extras(self) -> None:
        """Fidelity hierarchy construction should expose Graphviz mappings."""
        graph = _complete_multipartite_graph([[0, 1], [2, 3], [4, 5], [6, 7]])
        edge_index = graph.edge_index
        problem = LayoutProblem(edge_index=edge_index, num_nodes=8, seed=123)
        state = SolveState()
        ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))

        final_state = build_sfdp_pipeline(steps=0, fidelity_mode=True).apply(problem, state, ctx)

        graphs: list[GraphData] = final_state.extras[_GRAPH_KEY]
        mappings: list[torch.Tensor] = final_state.extras[_MAPPING_KEY]
        assert [level.num_nodes for level in graphs] == [8, 4]
        assert mappings[0].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]

    def test_graphviz_order_matches_csr_symmetrization_neighbor_order(self) -> None:
        """Graphviz fidelity graph rows should match 7.0.5 matrix row order."""
        edge_index = _edge_index_from_edges(
            [
                (0, 7),
                (0, 1),
                (10, 11),
                (7, 11),
                (0, 11),
                (11, 12),
            ]
        )

        graph = _build_graph(edge_index=edge_index, num_nodes=13, graphviz_order=True)

        assert graph.adjacency[0] == [(1, 1.0), (7, 1.0), (11, 1.0)]
        assert graph.adjacency[11] == [(12, 1.0), (0, 1.0), (7, 1.0), (10, 1.0)]

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(0, 123), (1, 123), (2, 123), (5, 123), (5, 99), (20, 123), (50, 7)],
    )
    def test_layout_sfdp_pipeline_matches_classic_for_requested_sizes(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """The adapter should match classic SFDP exactly for the requested cases."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_sfdp(edge_index=edge_index, num_nodes=num_nodes, steps=500, seed=seed)
        pipeline = layout_sfdp_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            steps=500,
            seed=seed,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_sfdp_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted SFDP should remain bit-identical in the pipeline."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_sfdp(
            edge_index=edge_index,
            num_nodes=6,
            steps=500,
            seed=17,
            edge_weights=edge_weights,
        )
        pipeline = layout_sfdp_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            steps=500,
            seed=17,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_graphviz_fidelity_ignores_edge_weights_from_dot_reference(self) -> None:
        """Verify Graphviz-fidelity SFDP ignores in-memory edge weights.

        Returns
        -------
        None
            The assertion fails if weighted and unweighted fidelity-mode layouts
            differ.
        """
        edge_index = _path_edge_index(8)
        edge_weights = torch.linspace(1.0, 4.0, edge_index.shape[1], dtype=torch.float64)

        weighted = layout_sfdp_pipeline(
            edge_index=edge_index,
            num_nodes=8,
            steps=25,
            seed=100,
            edge_weights=edge_weights,
            fidelity_mode="graphviz",
        )
        unweighted = layout_sfdp_pipeline(
            edge_index=edge_index,
            num_nodes=8,
            steps=25,
            seed=100,
            fidelity_mode="graphviz",
        )

        _assert_exact_match(weighted, unweighted)

    def test_layout_sfdp_pipeline_matches_classic_on_disconnected_graph(self) -> None:
        """Disconnected components and isolated nodes should match exactly."""
        edge_index = _disconnected_edge_index()

        classic = layout_sfdp(edge_index=edge_index, num_nodes=7, steps=500, seed=99)
        pipeline = layout_sfdp_pipeline(edge_index=edge_index, num_nodes=7, steps=500, seed=99)

        _assert_exact_match(classic, pipeline)

    def test_build_sfdp_pipeline_matches_classic_on_complete_graph(self) -> None:
        """The raw pipeline object should match classic SFDP on a dense graph."""
        edge_index = _complete_edge_index(5)

        classic = layout_sfdp(edge_index=edge_index, num_nodes=5, steps=500, seed=7)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=5, steps=500, seed=7)

        _assert_exact_match(classic, pipeline)

    def test_layout_sfdp_pipeline_matches_classic_with_node_sizes(self) -> None:
        """Node sizes should affect the extent scaling identically."""
        edge_index = _path_edge_index(10)
        node_sizes = torch.rand(10, 2, dtype=torch.float32) * 20.0 + 5.0

        classic = layout_sfdp(
            edge_index=edge_index,
            num_nodes=10,
            node_sizes=node_sizes,
            steps=500,
            seed=42,
        )
        pipeline = layout_sfdp_pipeline(
            edge_index=edge_index,
            num_nodes=10,
            node_sizes=node_sizes,
            steps=500,
            seed=42,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_sfdp_pipeline_matches_classic_with_custom_theta(self) -> None:
        """Custom theta should propagate identically."""
        edge_index = _path_edge_index(15)

        classic = layout_sfdp(
            edge_index=edge_index,
            num_nodes=15,
            steps=500,
            seed=42,
            theta=0.8,
        )
        pipeline = layout_sfdp_pipeline(
            edge_index=edge_index,
            num_nodes=15,
            steps=500,
            seed=42,
            theta=0.8,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_sfdp_pipeline_matches_classic_with_custom_repulsive_exponent(
        self,
    ) -> None:
        """Custom repulsive exponent should propagate identically."""
        edge_index = _path_edge_index(10)

        classic = layout_sfdp(
            edge_index=edge_index,
            num_nodes=10,
            steps=500,
            seed=42,
            repulsive_exponent=-2.0,
        )
        pipeline = layout_sfdp_pipeline(
            edge_index=edge_index,
            num_nodes=10,
            steps=500,
            seed=42,
            repulsive_exponent=-2.0,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_sfdp_pipeline_matches_classic_zero_steps(self) -> None:
        """Zero steps should produce the same output as classic."""
        edge_index = _path_edge_index(8)

        classic = layout_sfdp(edge_index=edge_index, num_nodes=8, steps=0, seed=42)
        pipeline = layout_sfdp_pipeline(edge_index=edge_index, num_nodes=8, steps=0, seed=42)

        _assert_exact_match(classic, pipeline)

    def test_layout_sfdp_pipeline_matches_classic_single_edge(self) -> None:
        """A single edge between two nodes should match exactly."""
        edge_index = _edge_index_from_edges([(0, 1)])

        classic = layout_sfdp(edge_index=edge_index, num_nodes=2, steps=500, seed=42)
        pipeline = layout_sfdp_pipeline(edge_index=edge_index, num_nodes=2, steps=500, seed=42)

        _assert_exact_match(classic, pipeline)

    def test_layout_sfdp_pipeline_matches_classic_star_graph(self) -> None:
        """Star graph topology should match exactly."""
        edges = [(0, i) for i in range(1, 8)]
        edge_index = _edge_index_from_edges(edges)

        classic = layout_sfdp(edge_index=edge_index, num_nodes=8, steps=500, seed=42)
        pipeline = layout_sfdp_pipeline(edge_index=edge_index, num_nodes=8, steps=500, seed=42)

        _assert_exact_match(classic, pipeline)

    def test_layout_sfdp_pipeline_orients_directed_path_to_requested_direction(
        self,
    ) -> None:
        """Final SFDP orientation should respect the requested directed flow."""
        edge_index = _path_edge_index(14)

        top_to_bottom = layout_sfdp_pipeline(
            edge_index=edge_index,
            num_nodes=14,
            steps=500,
            seed=42,
            direction="TB",
        )
        bottom_to_top = layout_sfdp_pipeline(
            edge_index=edge_index,
            num_nodes=14,
            steps=500,
            seed=42,
            direction="BT",
        )

        source = edge_index[0]
        target = edge_index[1]
        assert float((top_to_bottom[target, 1] - top_to_bottom[source, 1]).mean().item()) > 0.0
        assert float((bottom_to_top[target, 1] - bottom_to_top[source, 1]).mean().item()) < 0.0

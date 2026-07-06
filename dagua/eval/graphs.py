"""Test graph collection for evaluation.

Provides a registry of test graphs spanning common and niche structure families:
linear-shallow, linear-deep, wide-parallel, skip-light, skip-heavy, tree,
diamond, nested-shallow, nested-deep, mixed-width, self-loops, multi-edge,
disconnected, large-sparse, and large-dense.

Sources: synthetic generators + TorchLens model traces.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union, overload

import torch

from dagua.graph import DaguaGraph
from dagua.styles import ClusterStyle, EdgeStyle, NodeStyle

_GRAPH_CACHE_DIR = Path(__file__).with_name("_graph_cache")


@dataclass
class TestGraph:
    """A test graph with metadata for evaluation."""

    # Prevent pytest from collecting this dataclass as a test case.
    __test__ = False

    name: str
    graph: DaguaGraph
    tags: Set[str] = field(default_factory=set)
    description: str = ""
    source: str = "synthetic"
    expected_challenges: str = ""


def get_test_graphs(
    tags: Optional[Set[str]] = None,
    max_nodes: Optional[int] = None,
) -> List[TestGraph]:
    """Get test graphs, optionally filtered by tags and size.

    Args:
        tags: If set, only return graphs that have at least one matching tag.
        max_nodes: If set, only return graphs with <= max_nodes nodes.

    Returns:
        List of TestGraph objects.
    """
    all_graphs = _build_all_test_graphs()

    if tags:
        all_graphs = [g for g in all_graphs if g.tags & tags]
    if max_nodes:
        all_graphs = [g for g in all_graphs if g.graph.num_nodes <= max_nodes]

    return all_graphs


def is_semantically_directed(test_graph: TestGraph) -> bool:
    """Return whether edge direction should affect directed layout scoring.

    Parameters
    ----------
    test_graph : TestGraph
        Graph metadata and topology.

    Returns
    -------
    bool
        ``False`` for graphs tagged ``"undirected"`` because their stored edge
        orientation is an implementation artifact rather than graph semantics.
    """
    return "undirected" not in test_graph.tags


def _build_all_test_graphs() -> List[TestGraph]:
    """Build the complete test graph collection.

    Returns
    -------
    List[TestGraph]
        Synthetic, traced, and weighted benchmark graphs.
    """
    graphs = []
    graphs.extend(_synthetic_graphs())
    graphs.extend(_torchlens_graphs())
    graphs.extend(_r79_extension_graphs())
    graphs.append(_make_weighted_chain())
    graphs.append(_make_weighted_clusters())
    graphs.append(make_weighted_karate_graph())
    return graphs


def _finalize_generated_graph(graph: DaguaGraph) -> DaguaGraph:
    """Compute node sizes for a generated graph before returning it.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose node labels should be measured eagerly.

    Returns
    -------
    DaguaGraph
        The same graph instance with ``node_sizes`` populated.
    """
    _ = graph.edge_index
    graph.compute_node_sizes()
    return graph


def _build_named_graph(
    node_ids: Sequence[str],
    edges: Sequence[Tuple[str, str]],
) -> DaguaGraph:
    """Build a graph from explicit node IDs and directed edges.

    Parameters
    ----------
    node_ids : Sequence[str]
        Node IDs and labels to materialize, including isolated nodes.
    edges : Sequence[Tuple[str, str]]
        Directed edges expressed in terms of ``node_ids``.

    Returns
    -------
    DaguaGraph
        Graph with eagerly computed node sizes.
    """
    graph = DaguaGraph()
    for node_id in node_ids:
        graph.add_node(node_id, label=node_id)
    for src, tgt in edges:
        graph.add_edge(src, tgt)
    return _finalize_generated_graph(graph)


def _empty_generated_graph() -> DaguaGraph:
    """Return an empty generated graph with computed node-size tensors.

    Returns
    -------
    DaguaGraph
        Empty graph suitable for edge-case tests.
    """
    return _finalize_generated_graph(DaguaGraph())


def _import_networkx() -> Any:
    """Import NetworkX for optional benchmark graph generation.

    Returns
    -------
    Any
        Imported ``networkx`` module.

    Raises
    ------
    ImportError
        Raised when NetworkX is unavailable in the current environment.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "NetworkX is required for the evaluation graph collection. "
            "Install it with: pip install networkx"
        ) from exc
    return nx


def _undirected_to_dag(graph: DaguaGraph) -> DaguaGraph:
    """Orient an undirected graph into a DAG using ascending node indices.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose undirected edges should be converted into a single acyclic
        orientation.

    Returns
    -------
    DaguaGraph
        The same graph instance with each undirected edge oriented from the
        lower internal node index to the higher one.
    """
    edge_index = graph.edge_index
    if edge_index.numel() == 0:
        graph.edge_labels = []
        graph.edge_types = []
        graph.edge_styles = []
        return _finalize_generated_graph(graph)

    forward_mask = edge_index[0] < edge_index[1]
    reverse_mask = edge_index[0] > edge_index[1]
    forward_edges = edge_index[:, forward_mask]
    reverse_edges = torch.stack([edge_index[1, reverse_mask], edge_index[0, reverse_mask]])
    dag_edges = torch.cat([forward_edges, reverse_edges], dim=1)
    dag_weights: Optional[torch.Tensor]
    weight_dtype: Optional[torch.dtype] = None
    weight_device: Optional[torch.device] = None
    if graph.edge_weights is not None:
        weight_dtype = graph.edge_weights.dtype
        weight_device = graph.edge_weights.device
        forward_weights = graph.edge_weights[forward_mask]
        reverse_weights = graph.edge_weights[reverse_mask]
        dag_weights = torch.cat([forward_weights, reverse_weights], dim=0)
    else:
        dag_weights = None

    unique_edges: list[tuple[int, int]] = []
    unique_weights: list[float] = []
    seen_edges: set[tuple[int, int]] = set()
    for edge_idx in range(int(dag_edges.shape[1])):
        source = int(dag_edges[0, edge_idx].item())
        target = int(dag_edges[1, edge_idx].item())
        edge_pair = (source, target)
        if edge_pair in seen_edges:
            continue
        seen_edges.add(edge_pair)
        unique_edges.append(edge_pair)
        if dag_weights is not None:
            unique_weights.append(float(dag_weights[edge_idx].item()))

    if unique_edges:
        oriented_edges = torch.tensor(unique_edges, dtype=graph.index_dtype).t().contiguous()
    else:
        oriented_edges = torch.zeros((2, 0), dtype=graph.index_dtype)

    graph.edge_index = oriented_edges
    edge_count = int(oriented_edges.shape[1])
    graph.edge_labels = [None] * edge_count
    graph.edge_types = ["normal"] * edge_count
    graph.edge_styles = [None] * edge_count
    if dag_weights is not None:
        graph.edge_weights = torch.tensor(
            unique_weights,
            dtype=weight_dtype if weight_dtype is not None else torch.float32,
            device=weight_device,
        )
    return _finalize_generated_graph(graph)


def _graph_from_undirected_networkx(nx_graph: Any) -> DaguaGraph:
    """Convert an undirected NetworkX graph into a Dagua DAG.

    Parameters
    ----------
    nx_graph : Any
        NetworkX graph object to convert.

    Returns
    -------
    DaguaGraph
        Converted graph with a deterministic acyclic orientation and computed
        node sizes.
    """
    graph = DaguaGraph.from_networkx(nx_graph)
    return _undirected_to_dag(graph)


def _make_sbm_graph(
    sizes: Sequence[int],
    p_in: float,
    p_out: float,
    seed: int = 42,
) -> DaguaGraph:
    """Build a stochastic block model graph and orient it into a DAG.

    Parameters
    ----------
    sizes : Sequence[int]
        Community sizes.
    p_in : float
        Intra-community edge probability.
    p_out : float
        Inter-community edge probability.
    seed : int, default=42
        Random seed for reproducible block-model sampling.

    Returns
    -------
    DaguaGraph
        SBM graph converted into a DAG by orienting each sampled undirected edge
        from lower to higher internal node index.
    """
    nx = _import_networkx()
    n_communities = len(sizes)
    probability_matrix = [
        [p_in if row_idx == col_idx else p_out for col_idx in range(n_communities)]
        for row_idx in range(n_communities)
    ]
    nx_graph = nx.stochastic_block_model(list(sizes), probability_matrix, seed=seed)
    return _graph_from_undirected_networkx(nx_graph)


def make_real_karate_graph() -> TestGraph:
    """Build the classic Zachary karate club social network benchmark.

    Returns
    -------
    TestGraph
        Directed acyclic version of the karate club graph for real-world
        community-structure testing.
    """
    nx = _import_networkx()
    graph = _graph_from_undirected_networkx(nx.karate_club_graph())
    return TestGraph(
        name="real_karate_34",
        graph=graph,
        tags={"social", "community", "real-world", "undirected"},
        description="Zachary's Karate Club social network oriented into a DAG",
        source="networkx",
        expected_challenges="Community separation with hub-heavy social structure",
    )


def _make_weighted_chain(n: int = 20) -> TestGraph:
    """Build a weighted linear chain benchmark.

    Parameters
    ----------
    n : int, default=20
        Number of nodes in the chain.

    Returns
    -------
    TestGraph
        Linear graph with monotonically increasing edge weights.
    """
    graph = DaguaGraph()
    for node_idx in range(n):
        graph.add_node(node_idx, label=str(node_idx))
    for node_idx in range(n - 1):
        weight = 1.0 + (4.0 * float(node_idx) / float(max(n - 2, 1)))
        graph.add_edge(node_idx, node_idx + 1, weight=weight)
    return TestGraph(
        name=f"weighted_chain_{n}",
        graph=_finalize_generated_graph(graph),
        tags={"weighted", "linear", "synthetic"},
        description="Linear chain with weights increasing from 1.0 to 5.0",
        source="synthetic",
        expected_challenges="Asymmetric spacing due to weight gradient",
    )


def _make_weighted_clusters(
    cluster_size: int = 10,
    n_clusters: int = 3,
    seed: int = 42,
) -> TestGraph:
    """Build a clustered weighted-community benchmark.

    Parameters
    ----------
    cluster_size : int, default=10
        Nodes per cluster.
    n_clusters : int, default=3
        Number of clusters to generate.
    seed : int, default=42
        Random seed for reproducible edge sampling.

    Returns
    -------
    TestGraph
        Clustered graph with heavy intra-cluster and light inter-cluster edges.
    """
    rng = random.Random(seed)

    graph = DaguaGraph()
    total_nodes = cluster_size * n_clusters
    for node_idx in range(total_nodes):
        graph.add_node(node_idx, label=str(node_idx))

    for cluster_idx in range(n_clusters):
        cluster_base = cluster_idx * cluster_size
        for source_offset in range(cluster_size):
            for target_offset in range(source_offset + 1, cluster_size):
                if rng.random() < 0.6:
                    graph.add_edge(
                        cluster_base + source_offset,
                        cluster_base + target_offset,
                        weight=5.0,
                    )

    for source_cluster in range(n_clusters):
        for target_cluster in range(source_cluster + 1, n_clusters):
            for _ in range(2):
                source_idx = rng.randint(0, cluster_size - 1)
                target_idx = rng.randint(0, cluster_size - 1)
                graph.add_edge(
                    source_cluster * cluster_size + source_idx,
                    target_cluster * cluster_size + target_idx,
                    weight=0.5,
                )

    return TestGraph(
        name=f"weighted_clusters_{n_clusters}x{cluster_size}",
        graph=_finalize_generated_graph(graph),
        tags={"weighted", "community", "synthetic", "undirected"},
        description="3 clusters of 10 with heavy intra / light inter edges",
        source="synthetic",
        expected_challenges="Cluster separation proportional to weight ratio",
    )


def make_weighted_karate_graph() -> TestGraph:
    """Build a weighted Zachary karate club benchmark.

    Returns
    -------
    TestGraph
        Karate club graph with deterministic heuristic edge weights preserved
        through DAG orientation.
    """
    nx = _import_networkx()
    nx_graph = nx.karate_club_graph()
    for source, target in nx_graph.edges():
        weight = float(nx_graph.degree(source) + nx_graph.degree(target)) / 10.0
        nx_graph[source][target]["weight"] = weight

    graph = DaguaGraph.from_networkx(nx_graph)
    graph = _undirected_to_dag(graph)
    return TestGraph(
        name="weighted_karate_34",
        graph=graph,
        tags={"weighted", "social", "community", "real-world", "undirected"},
        description="Zachary's Karate Club with degree-based interaction weights",
        source="networkx",
        expected_challenges="Weight-aware community separation with hub structure",
    )


def _make_r79_weighted_community_graph(seed: int = 79) -> TestGraph:
    """Build a weighted planted-community graph for the r79 baseline.

    Parameters
    ----------
    seed : int, default=79
        Random seed for deterministic edge sampling.

    Returns
    -------
    TestGraph
        Four-community weighted graph with heavy intra-block and light bridge
        edges.
    """
    rng = random.Random(seed)
    graph = DaguaGraph()
    community_count = 4
    community_size = 18
    for node_idx in range(community_count * community_size):
        graph.add_node(node_idx, label=f"wc_{node_idx}")

    for community_idx in range(community_count):
        base = community_idx * community_size
        for left in range(community_size):
            for right in range(left + 1, community_size):
                if rng.random() < 0.32:
                    graph.add_edge(base + left, base + right, weight=rng.uniform(4.0, 8.0))

    for source_community in range(community_count):
        for target_community in range(source_community + 1, community_count):
            for _ in range(3):
                source = source_community * community_size + rng.randrange(community_size)
                target = target_community * community_size + rng.randrange(community_size)
                graph.add_edge(source, target, weight=rng.uniform(0.15, 0.45))

    return TestGraph(
        name="r79_weighted_community_4x18",
        graph=_finalize_generated_graph(graph),
        tags={"r79_ext", "weighted", "community", "clustered", "synthetic", "undirected"},
        description="R79 weighted planted-community graph with light inter-block bridges",
        source="synthetic",
        expected_challenges="Separating communities while honoring strong weighted ties",
    )


def _make_r79_weighted_mesh_graph(rows: int = 10, cols: int = 12, seed: int = 79) -> TestGraph:
    """Build a weighted rectangular mesh for the r79 baseline.

    Parameters
    ----------
    rows : int, default=10
        Number of mesh rows.
    cols : int, default=12
        Number of mesh columns.
    seed : int, default=79
        Random seed for deterministic diagonal shortcuts.

    Returns
    -------
    TestGraph
        Weighted grid with local mesh edges and a few heavier diagonals.
    """
    rng = random.Random(seed)
    graph = DaguaGraph()
    for row_idx in range(rows):
        for col_idx in range(cols):
            node_idx = row_idx * cols + col_idx
            graph.add_node(node_idx, label=f"wm_{row_idx}_{col_idx}")

    for row_idx in range(rows):
        for col_idx in range(cols):
            node_idx = row_idx * cols + col_idx
            if col_idx + 1 < cols:
                graph.add_edge(node_idx, node_idx + 1, weight=2.0)
            if row_idx + 1 < rows:
                graph.add_edge(node_idx, node_idx + cols, weight=3.0)
            if row_idx + 1 < rows and col_idx + 1 < cols and rng.random() < 0.18:
                graph.add_edge(node_idx, node_idx + cols + 1, weight=5.0)

    return TestGraph(
        name="r79_weighted_mesh_10x12",
        graph=_finalize_generated_graph(graph),
        tags={"r79_ext", "weighted", "grid", "mesh", "synthetic", "undirected"},
        description="R79 weighted 10x12 mesh with deterministic diagonal shortcuts",
        source="synthetic",
        expected_challenges="Maintaining mesh regularity with anisotropic edge weights",
    )


def _make_r79_weighted_skew_dag(num_layers: int = 6, width: int = 10) -> TestGraph:
    """Build a weighted DAG whose weights span more than two orders of magnitude.

    Parameters
    ----------
    num_layers : int, default=6
        Number of topological layers.
    width : int, default=10
        Nodes per layer.

    Returns
    -------
    TestGraph
        Layered weighted DAG with weights from ``0.01`` through ``100.0``.
    """
    graph = DaguaGraph()
    total_nodes = num_layers * width
    for node_idx in range(total_nodes):
        graph.add_node(node_idx, label=f"ws_{node_idx}")

    weight_cycle = [0.01, 0.05, 0.2, 1.0, 8.0, 100.0]
    for layer_idx in range(num_layers - 1):
        layer_base = layer_idx * width
        next_base = (layer_idx + 1) * width
        for offset in range(width):
            graph.add_edge(
                layer_base + offset,
                next_base + offset,
                weight=weight_cycle[(layer_idx + offset) % len(weight_cycle)],
            )
            graph.add_edge(
                layer_base + offset,
                next_base + ((offset + 3) % width),
                weight=weight_cycle[(layer_idx * 2 + offset + 1) % len(weight_cycle)],
            )

    return TestGraph(
        name="r79_weighted_skew_dag_6x10",
        graph=_finalize_generated_graph(graph),
        tags={"r79_ext", "weighted", "skip-heavy", "wide-parallel", "synthetic"},
        description="R79 layered weighted DAG with 0.01 to 100.0 edge weights",
        source="synthetic",
        expected_challenges="Strongly skewed weighted constraints across a wide DAG",
    )


def _make_r79_weighted_hub_spoke(hub_count: int = 4, spokes_per_hub: int = 18) -> TestGraph:
    """Build a weighted hub-and-spoke graph for the r79 baseline.

    Parameters
    ----------
    hub_count : int, default=4
        Number of hub nodes.
    spokes_per_hub : int, default=18
        Number of spokes attached to each hub.

    Returns
    -------
    TestGraph
        Weighted graph with strong hub-spoke links and weaker cross-spoke ties.
    """
    graph = DaguaGraph()
    sink = "sink"
    graph.add_node(sink, label=sink)
    for hub_idx in range(hub_count):
        hub = f"hub_{hub_idx}"
        graph.add_node(hub, label=hub)
        graph.add_edge(hub, sink, weight=12.0)
        for spoke_idx in range(spokes_per_hub):
            spoke = f"h{hub_idx}_spoke_{spoke_idx}"
            graph.add_node(spoke, label=spoke)
            graph.add_edge(hub, spoke, weight=7.0)
            graph.add_edge(spoke, sink, weight=1.0)
            if spoke_idx > 0 and spoke_idx % 6 == 0:
                graph.add_edge(f"h{hub_idx}_spoke_{spoke_idx - 1}", spoke, weight=0.25)

    return TestGraph(
        name="r79_weighted_hub_spoke_4x18",
        graph=_finalize_generated_graph(graph),
        tags={"r79_ext", "weighted", "hub-spoke", "wide-parallel", "synthetic"},
        description="R79 weighted multi-hub spoke graph with weak spoke chains",
        source="synthetic",
        expected_challenges="Balancing dominant hubs without collapsing weak spoke structure",
    )


def _make_r79_weighted_small_world(n: int = 120, seed: int = 79) -> TestGraph:
    """Build a weighted directed small-world graph for the r79 baseline.

    Parameters
    ----------
    n : int, default=120
        Number of nodes.
    seed : int, default=79
        Random seed for Watts-Strogatz rewiring and weights.

    Returns
    -------
    TestGraph
        Weighted cyclic small-world graph with local heavy edges and weak
        shortcuts.
    """
    rng = random.Random(seed)
    graph = DaguaGraph()
    for node_idx in range(n):
        graph.add_node(node_idx, label=f"ww_{node_idx}")

    edge_set: set[tuple[int, int]] = set()
    for source in range(n):
        for offset in (1, 2, 3):
            target = (source + offset) % n
            if rng.random() < 0.12:
                target = rng.randrange(n)
            if source == target or (source, target) in edge_set:
                continue
            edge_set.add((source, target))
            is_shortcut = (target - source) % n not in {1, 2, 3}
            graph.add_edge(source, target, weight=0.4 if is_shortcut else rng.uniform(2.0, 6.0))

    return TestGraph(
        name="r79_weighted_small_world_120",
        graph=_finalize_generated_graph(graph),
        tags={"r79_ext", "weighted", "small-world", "cyclic", "synthetic", "undirected"},
        description="R79 weighted small-world ring with weak rewired shortcuts",
        source="synthetic",
        expected_challenges="Short-path cyclic structure with mixed edge strengths",
    )


def _make_r79_weighted_ladder(rungs: int = 40) -> TestGraph:
    """Build a weighted ladder graph with asymmetric rails.

    Parameters
    ----------
    rungs : int, default=40
        Number of ladder rungs.

    Returns
    -------
    TestGraph
        Two-rail weighted graph with strong upper rail, weak lower rail, and
        medium rungs.
    """
    graph = DaguaGraph()
    for rung_idx in range(rungs):
        graph.add_node(f"top_{rung_idx}", label=f"top_{rung_idx}")
        graph.add_node(f"bot_{rung_idx}", label=f"bot_{rung_idx}")
    for rung_idx in range(rungs - 1):
        graph.add_edge(f"top_{rung_idx}", f"top_{rung_idx + 1}", weight=9.0)
        graph.add_edge(f"bot_{rung_idx}", f"bot_{rung_idx + 1}", weight=0.8)
        graph.add_edge(f"top_{rung_idx}", f"bot_{rung_idx}", weight=3.0)
    graph.add_edge(f"top_{rungs - 1}", f"bot_{rungs - 1}", weight=3.0)

    return TestGraph(
        name="r79_weighted_ladder_40",
        graph=_finalize_generated_graph(graph),
        tags={"r79_ext", "weighted", "linear", "synthetic", "undirected"},
        description="R79 weighted ladder with asymmetric rail strengths",
        source="synthetic",
        expected_challenges="Preserving parallel rails under strongly unequal weights",
    )


def _make_r79_weighted_bipartite(seed: int = 79) -> TestGraph:
    """Build a weighted bipartite graph for the r79 baseline.

    Parameters
    ----------
    seed : int, default=79
        Random seed for deterministic edge sampling.

    Returns
    -------
    TestGraph
        Weighted three-level bipartite graph with dominant and weak cross edges.
    """
    rng = random.Random(seed)
    graph = DaguaGraph()
    left_count = 16
    right_count = 24
    for left_idx in range(left_count):
        graph.add_node(f"left_{left_idx}", label=f"left_{left_idx}")
    for right_idx in range(right_count):
        graph.add_node(f"right_{right_idx}", label=f"right_{right_idx}")
    for left_idx in range(left_count):
        targets = rng.sample(range(right_count), 6)
        for order, right_idx in enumerate(targets):
            weight = 10.0 if order == 0 else rng.uniform(0.5, 3.0)
            graph.add_edge(f"left_{left_idx}", f"right_{right_idx}", weight=weight)

    return TestGraph(
        name="r79_weighted_bipartite_16x24",
        graph=_finalize_generated_graph(graph),
        tags={"r79_ext", "weighted", "bipartite", "wide-parallel", "synthetic", "undirected"},
        description="R79 weighted bipartite graph with one dominant target per source",
        source="synthetic",
        expected_challenges="Crossing pressure under uneven weighted bipartite links",
    )


def _make_r79_nested_cluster_graph(
    name: str,
    top_count: int,
    sub_count: int,
    nodes_per_subcluster: int,
    seed: int,
) -> TestGraph:
    """Build a nested-cluster r79 graph with deterministic cross-cluster edges.

    Parameters
    ----------
    name : str
        Benchmark graph name.
    top_count : int
        Number of top-level clusters.
    sub_count : int
        Number of subclusters in each top-level cluster.
    nodes_per_subcluster : int
        Number of nodes in each subcluster.
    seed : int
        Random seed for intra-subcluster shortcuts and cross-cluster bridges.

    Returns
    -------
    TestGraph
        Nested-cluster graph with two or three hierarchy levels.
    """
    rng = random.Random(seed)
    graph = DaguaGraph()
    node_groups: list[list[list[str]]] = []
    for top_idx in range(top_count):
        top_groups: list[list[str]] = []
        for sub_idx in range(sub_count):
            members = [
                f"c{top_idx}.s{sub_idx}.n{node_idx}" for node_idx in range(nodes_per_subcluster)
            ]
            top_groups.append(members)
            for node_id in members:
                graph.add_node(node_id, label=node_id)
            for node_idx in range(nodes_per_subcluster - 1):
                graph.add_edge(members[node_idx], members[node_idx + 1])
                if node_idx + 2 < nodes_per_subcluster and rng.random() < 0.35:
                    graph.add_edge(members[node_idx], members[node_idx + 2])
        node_groups.append(top_groups)

    for top_idx in range(top_count):
        for sub_idx in range(sub_count - 1):
            graph.add_edge(node_groups[top_idx][sub_idx][-1], node_groups[top_idx][sub_idx + 1][0])
    for top_idx in range(top_count - 1):
        for _ in range(3):
            source_sub = rng.randrange(sub_count)
            target_sub = rng.randrange(sub_count)
            source = rng.choice(node_groups[top_idx][source_sub])
            target = rng.choice(node_groups[top_idx + 1][target_sub])
            graph.add_edge(source, target)

    for top_idx, top_groups in enumerate(node_groups):
        top_name = f"{name}_top_{top_idx}"
        top_members = [node_id for sub_group in top_groups for node_id in sub_group]
        graph.add_cluster(top_name, top_members, label=f"R79 Top {top_idx}")
        for sub_idx, members in enumerate(top_groups):
            sub_name = f"{name}_top_{top_idx}_sub_{sub_idx}"
            graph.add_cluster(
                sub_name,
                members,
                label=f"R79 Top {top_idx} / Sub {sub_idx}",
                parent=top_name,
            )
            if nodes_per_subcluster >= 8:
                inner_name = f"{sub_name}_inner"
                graph.add_cluster(
                    inner_name,
                    members[: max(3, nodes_per_subcluster // 2)],
                    label=f"R79 Inner {top_idx}.{sub_idx}",
                    parent=sub_name,
                )

    return TestGraph(
        name=name,
        graph=_finalize_generated_graph(graph),
        tags={"r79_ext", "clustered", "nested-deep", "synthetic"},
        description="R79 nested cluster graph with top clusters, subclusters, and bridges",
        source="synthetic",
        expected_challenges="Nested containment and routing across cluster hierarchy levels",
    )


def _make_r79_undirected_sbm(
    name: str,
    sizes: Sequence[int],
    p_in: float,
    p_out: float,
    seed: int,
) -> TestGraph:
    """Build an undirected community graph and keep symmetric directed edges.

    Parameters
    ----------
    name : str
        Benchmark graph name.
    sizes : Sequence[int]
        Community sizes for the stochastic block model.
    p_in : float
        Intra-community edge probability.
    p_out : float
        Inter-community edge probability.
    seed : int
        Random seed for deterministic sampling.

    Returns
    -------
    TestGraph
        Symmetric directed representation of an undirected SBM graph.
    """
    nx = _import_networkx()
    block_count = len(sizes)
    probability_matrix = [
        [p_in if row_idx == col_idx else p_out for col_idx in range(block_count)]
        for row_idx in range(block_count)
    ]
    nx_graph = nx.stochastic_block_model(list(sizes), probability_matrix, seed=seed)
    graph = DaguaGraph()
    for node_idx in range(sum(sizes)):
        graph.add_node(node_idx, label=f"sbm_{node_idx}")
    for source, target in sorted(nx_graph.edges()):
        graph.add_edge(int(source), int(target))
        graph.add_edge(int(target), int(source))

    offset = 0
    for block_idx, size in enumerate(sizes):
        members = list(range(offset, offset + size))
        graph.add_cluster(f"{name}_block_{block_idx}", members, label=f"Block {block_idx}")
        offset += size

    return TestGraph(
        name=name,
        graph=_finalize_generated_graph(graph),
        tags={"r79_ext", "community", "clustered", "undirected", "random"},
        description=f"R79 symmetric SBM with p_in={p_in} and p_out={p_out}",
        source="networkx",
        expected_challenges="Undirected community structure at a controlled mixing ratio",
    )


def _make_r79_directed_scc_graph(
    name: str,
    num_nodes: int,
    scc_sizes: Sequence[int],
    seed: int,
) -> TestGraph:
    """Build a directed graph with large SCCs plus DAG-like tails.

    Parameters
    ----------
    name : str
        Benchmark graph name.
    num_nodes : int
        Total number of nodes.
    scc_sizes : Sequence[int]
        Sizes of the strongly connected core components.
    seed : int
        Random seed for deterministic extra edges.

    Returns
    -------
    TestGraph
        Directed graph with 30-40 percent of nodes in two or three SCCs and
        the remaining nodes arranged as DAG-like feeders and exits.
    """
    rng = random.Random(seed)
    graph = DaguaGraph()
    for node_idx in range(num_nodes):
        graph.add_node(node_idx, label=f"scc_{node_idx}")

    offset = 0
    scc_ranges: list[list[int]] = []
    for scc_idx, size in enumerate(scc_sizes):
        members = list(range(offset, offset + size))
        scc_ranges.append(members)
        for member_idx, source in enumerate(members):
            graph.add_edge(source, members[(member_idx + 1) % size])
            if size > 5 and member_idx % 3 == 0:
                graph.add_edge(source, members[(member_idx + 3) % size])
        graph.add_cluster(f"{name}_scc_{scc_idx}", members, label=f"SCC {scc_idx}")
        offset += size

    for scc_idx in range(len(scc_ranges) - 1):
        graph.add_edge(scc_ranges[scc_idx][-1], scc_ranges[scc_idx + 1][0])

    remaining = list(range(offset, num_nodes))
    for node_idx in remaining:
        if node_idx % 5 == 0 and scc_ranges:
            graph.add_edge(rng.choice(scc_ranges[0]), node_idx)
        if node_idx + 1 < num_nodes:
            graph.add_edge(node_idx, node_idx + 1)
        if node_idx + 4 < num_nodes and rng.random() < 0.3:
            graph.add_edge(node_idx, node_idx + 4)
        if scc_ranges and rng.random() < 0.2:
            graph.add_edge(node_idx, rng.choice(scc_ranges[-1]))

    return TestGraph(
        name=name,
        graph=_finalize_generated_graph(graph),
        tags={"r79_ext", "directed", "cyclic", "clustered", "synthetic"},
        description="R79 directed graph with large SCC cores and DAG-like tails",
        source="synthetic",
        expected_challenges="Cycle handling when large SCCs interact with acyclic tails",
    )


def _r79_extension_graphs() -> List[TestGraph]:
    """Build all graph-corpus additions for the r79 native-layout sprint.

    Returns
    -------
    List[TestGraph]
        Deterministic r79 extension graphs, each tagged with ``r79_ext``.
    """
    return [
        _make_r79_weighted_community_graph(),
        _make_r79_weighted_mesh_graph(),
        _make_r79_weighted_skew_dag(),
        _make_r79_weighted_hub_spoke(),
        _make_r79_weighted_small_world(),
        _make_r79_weighted_ladder(),
        _make_r79_weighted_bipartite(),
        _make_r79_nested_cluster_graph("r79_nested_clusters_3x2x10", 3, 2, 10, 79),
        _make_r79_nested_cluster_graph("r79_nested_clusters_2x3x12", 2, 3, 12, 80),
        _make_r79_nested_cluster_graph("r79_nested_clusters_4x2x8", 4, 2, 8, 81),
        _make_r79_undirected_sbm(
            "r79_undirected_sbm_low_mix_4x25", [25, 25, 25, 25], 0.24, 0.01, 79
        ),
        _make_r79_undirected_sbm(
            "r79_undirected_sbm_mid_mix_5x20", [20, 20, 20, 20, 20], 0.20, 0.04, 80
        ),
        _make_r79_undirected_sbm("r79_undirected_sbm_high_mix_3x30", [30, 30, 30], 0.18, 0.08, 81),
        _make_r79_directed_scc_graph("r79_directed_scc_90_3cores", 90, [12, 10, 8], 79),
        _make_r79_directed_scc_graph("r79_directed_scc_120_2cores", 120, [24, 18], 80),
    ]


def make_real_lesmis_graph() -> TestGraph:
    """Build the Les Miserables co-occurrence benchmark graph.

    Returns
    -------
    TestGraph
        Directed acyclic version of a real-world literary character network.
    """
    nx = _import_networkx()
    if hasattr(nx, "les_miserables_graph"):
        nx_graph = nx.les_miserables_graph()
        name = "real_lesmis_77"
        description = "Les Miserables character co-occurrence graph oriented into a DAG"
    else:
        nx_graph = nx.florentine_families_graph()
        name = "real_lesmis_77"
        description = (
            "Florentine families graph used as a fallback social benchmark, oriented into a DAG"
        )
    graph = _graph_from_undirected_networkx(nx_graph)
    return TestGraph(
        name=name,
        graph=graph,
        tags={"social", "community", "real-world", "weighted", "undirected"},
        description=description,
        source="networkx",
        expected_challenges="Dense local communities and a nontrivial social core",
    )


def make_real_football_graph(seed: int = 42) -> TestGraph:
    """Build the football benchmark graph or its synthetic community fallback.

    Parameters
    ----------
    seed : int, default=42
        Random seed used by the synthetic fallback when a built-in football
        graph is unavailable.

    Returns
    -------
    TestGraph
        Football-style community graph oriented into a DAG.
    """
    nx = _import_networkx()
    if hasattr(nx, "football_graph"):
        graph = _graph_from_undirected_networkx(nx.football_graph())
        source = "networkx"
        description = "American college football schedule network oriented into a DAG"
    else:
        # The fallback keeps the benchmark's strong conference structure even
        # when the canonical NetworkX loader is unavailable.
        graph = _make_sbm_graph(
            sizes=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5],
            p_in=0.7,
            p_out=0.05,
            seed=seed,
        )
        source = "synthetic-fallback"
        description = "Football-style synthetic SBM fallback with 12 uneven communities"

    return TestGraph(
        name="real_football_115",
        graph=graph,
        tags={"social", "community", "real-world", "undirected"},
        description=description,
        source=source,
        expected_challenges="Strong communities with enough inter-group edges to stress separation",
    )


def make_erdos_renyi(n: int, p: float = 0.02, seed: int = 42) -> TestGraph:
    """Build an Erdos-Renyi random graph benchmark and orient it into a DAG.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float, default=0.02
        Edge probability for the undirected random graph.
    seed : int, default=42
        Random seed for reproducible sampling.

    Returns
    -------
    TestGraph
        Random graph benchmark with an acyclic orientation.
    """
    nx = _import_networkx()
    nx_graph = nx.erdos_renyi_graph(n, p, seed=seed, directed=False)
    graph = _graph_from_undirected_networkx(nx_graph)
    return TestGraph(
        name=f"er_{n}",
        graph=graph,
        tags={"erdos-renyi", "random", "undirected"},
        description=f"Erdos-Renyi random graph with {n} nodes and p={p}",
        source="networkx",
        expected_challenges="Unstructured connectivity with weak layering cues",
    )


def make_random_geometric(n: int, radius: float = 0.15, seed: int = 42) -> TestGraph:
    """Build a random geometric graph benchmark and orient it into a DAG.

    Parameters
    ----------
    n : int
        Number of nodes.
    radius : float, default=0.15
        Connection radius in the unit square.
    seed : int, default=42
        Random seed for reproducible spatial sampling.

    Returns
    -------
    TestGraph
        Spatial random graph with local neighborhoods and high clustering.
    """
    nx = _import_networkx()
    nx_graph = nx.random_geometric_graph(n, radius, seed=seed)
    graph = _graph_from_undirected_networkx(nx_graph)
    return TestGraph(
        name=f"rgg_{n}",
        graph=graph,
        tags={"geometric", "spatial", "random", "undirected"},
        description=f"Random geometric graph with {n} nodes and radius {radius}",
        source="networkx",
        expected_challenges="Spatial locality, clustered neighborhoods, and irregular density",
    )


def make_sbm(
    n_communities: int = 5,
    community_size: int = 40,
    p_in: float = 0.3,
    p_out: float = 0.01,
    seed: int = 42,
) -> TestGraph:
    """Build a stochastic block model benchmark with planted communities.

    Parameters
    ----------
    n_communities : int, default=5
        Number of communities.
    community_size : int, default=40
        Node count per community.
    p_in : float, default=0.3
        Intra-community edge probability.
    p_out : float, default=0.01
        Inter-community edge probability.
    seed : int, default=42
        Random seed for reproducible block-model sampling.

    Returns
    -------
    TestGraph
        Community benchmark with an acyclic orientation.
    """
    graph = _make_sbm_graph(
        sizes=[community_size] * max(n_communities, 0),
        p_in=p_in,
        p_out=p_out,
        seed=seed,
    )
    return TestGraph(
        name=f"sbm_{n_communities}x{community_size}",
        graph=graph,
        tags={"community", "clustered", "random", "undirected"},
        description=(
            f"Stochastic block model with {n_communities} communities of {community_size} nodes"
        ),
        source="networkx",
        expected_challenges="Planted communities with sparse inter-block bridges",
    )


# ─── Synthetic Graph Generators ───────────────────────────────────────────────


def _synthetic_graphs() -> List[TestGraph]:
    """Generate the synthetic benchmark graph collection.

    Returns
    -------
    List[TestGraph]
        Curated synthetic graphs spanning structural, styling, and routing
        stress cases used by the evaluation stack.
    """
    graphs = []

    # 1. Linear chain (shallow)
    g = DaguaGraph.from_edge_list(
        [
            ("input", "fc1"),
            ("fc1", "relu1"),
            ("relu1", "fc2"),
            ("fc2", "relu2"),
            ("relu2", "output"),
        ]
    )
    graphs.append(
        TestGraph(
            name="linear_3layer_mlp",
            graph=g,
            tags={"linear-shallow"},
            description="Simple 3-layer MLP: input→fc→relu→fc→relu→output",
            expected_challenges="Trivial layout, baseline for comparison",
        )
    )

    # 2. Linear chain (deep)
    edges = []
    prev = "input"
    for i in range(20):
        curr = f"layer_{i}"
        edges.append((prev, curr))
        prev = curr
    edges.append((prev, "output"))
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(
        TestGraph(
            name="deep_chain_20",
            graph=g,
            tags={"linear-deep"},
            description="20-layer deep chain",
            expected_challenges="Very tall layout, edge length consistency",
        )
    )

    # 3. Wide parallel (inception-like)
    edges = [
        ("input", "branch_1x1"),
        ("input", "branch_3x3"),
        ("input", "branch_5x5"),
        ("input", "branch_pool"),
        ("branch_1x1", "concat"),
        ("branch_3x3", "concat"),
        ("branch_5x5", "concat"),
        ("branch_pool", "concat"),
        ("concat", "output"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(
        TestGraph(
            name="inception_block",
            graph=g,
            tags={"wide-parallel"},
            description="Inception-style 4-branch parallel block",
            expected_challenges="Wide layout, parallel branch alignment",
        )
    )

    # 4. Skip connections (light) — simple residual
    edges = [
        ("input", "conv1"),
        ("conv1", "bn1"),
        ("bn1", "relu1"),
        ("relu1", "conv2"),
        ("conv2", "bn2"),
        ("input", "skip"),
        ("skip", "add"),
        ("bn2", "add"),
        ("add", "relu2"),
        ("relu2", "output"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(
        TestGraph(
            name="residual_block",
            graph=g,
            tags={"skip-light", "diamond"},
            description="Single residual block with skip connection",
            expected_challenges="Skip connection routing, merge alignment",
        )
    )

    # 5. Skip connections (heavy) — DenseNet-style
    layers = ["input"] + [f"dense_{i}" for i in range(6)] + ["output"]
    edges = []
    for i in range(1, len(layers) - 1):
        for j in range(i):
            edges.append((layers[j], layers[i]))
    edges.append((layers[-2], layers[-1]))
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(
        TestGraph(
            name="densenet_block",
            graph=g,
            tags={"skip-heavy", "large-dense"},
            description="DenseNet-style fully connected block (6 layers)",
            expected_challenges="Many crossing edges, dense connectivity",
        )
    )

    # 6. Tree (decoder)
    edges = [
        ("root", "left"),
        ("root", "right"),
        ("left", "ll"),
        ("left", "lr"),
        ("right", "rl"),
        ("right", "rr"),
        ("ll", "lll"),
        ("ll", "llr"),
        ("lr", "lrl"),
        ("lr", "lrr"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(
        TestGraph(
            name="binary_tree",
            graph=g,
            tags={"tree"},
            description="Binary tree (depth 4)",
            expected_challenges="Symmetric layout, proper spreading",
        )
    )

    # 7. Diamond / U-Net encoder-decoder
    edges = [
        ("input", "enc1"),
        ("enc1", "enc2"),
        ("enc2", "enc3"),
        ("enc3", "bottleneck"),
        ("bottleneck", "dec3"),
        ("dec3", "dec2"),
        ("dec2", "dec1"),
        ("dec1", "output"),
        # Skip connections
        ("enc1", "dec1"),
        ("enc2", "dec2"),
        ("enc3", "dec3"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(
        TestGraph(
            name="unet_small",
            graph=g,
            tags={"diamond", "skip-light"},
            description="Small U-Net encoder-decoder with skip connections",
            expected_challenges="Symmetric layout, horizontal skip connections",
        )
    )

    # 8. Nested clusters (shallow)
    g = DaguaGraph.from_edge_list(
        [
            ("input", "enc.conv1"),
            ("enc.conv1", "enc.relu"),
            ("enc.relu", "dec.conv2"),
            ("dec.conv2", "dec.relu"),
            ("dec.relu", "output"),
        ]
    )
    g.add_cluster("encoder", [1, 2], label="Encoder")  # enc.conv1, enc.relu
    g.add_cluster("decoder", [3, 4], label="Decoder")  # dec.conv2, dec.relu
    graphs.append(
        TestGraph(
            name="nested_shallow_enc_dec",
            graph=g,
            tags={"nested-shallow"},
            description="Encoder-decoder with 2 clusters",
            expected_challenges="Cluster separation and labeling",
        )
    )

    # 9. Nested clusters (deep) — transformer-like
    edges = [
        ("input", "embed"),
        ("embed", "attn.q_proj"),
        ("embed", "attn.k_proj"),
        ("embed", "attn.v_proj"),
        ("attn.q_proj", "attn.matmul"),
        ("attn.k_proj", "attn.matmul"),
        ("attn.matmul", "attn.softmax"),
        ("attn.v_proj", "attn.softmax"),
        ("attn.softmax", "attn.out_proj"),
        ("attn.out_proj", "add1"),
        ("embed", "add1"),  # residual
        ("add1", "norm1"),
        ("norm1", "ff.fc1"),
        ("ff.fc1", "ff.relu"),
        ("ff.relu", "ff.fc2"),
        ("ff.fc2", "add2"),
        ("norm1", "add2"),  # residual
        ("add2", "norm2"),
        ("norm2", "output"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    # Find indices for cluster members
    node_idx = {name: i for i, name in enumerate(g.node_labels)}
    g.add_cluster(
        "attention",
        [
            node_idx[n]
            for n in [
                "attn.q_proj",
                "attn.k_proj",
                "attn.v_proj",
                "attn.matmul",
                "attn.softmax",
                "attn.out_proj",
            ]
        ],
        label="Multi-Head Attention",
    )
    g.add_cluster(
        "feedforward",
        [
            node_idx[n]
            for n in [
                "ff.fc1",
                "ff.relu",
                "ff.fc2",
            ]
        ],
        label="Feed-Forward",
    )
    graphs.append(
        TestGraph(
            name="transformer_layer",
            graph=g,
            tags={"nested-deep", "wide-parallel", "skip-light"},
            description="Single transformer layer with attention + FFN clusters",
            expected_challenges="Nested clusters, parallel Q/K/V branches, residuals",
        )
    )

    # 10. Mixed width labels
    edges = [
        ("x", "MultiHeadAttention(embed_dim=512, num_heads=8)"),
        ("MultiHeadAttention(embed_dim=512, num_heads=8)", "LayerNorm(normalized_shape=(512,))"),
        ("LayerNorm(normalized_shape=(512,))", "+"),
        ("x", "+"),
        ("+", "ReLU"),
        ("ReLU", "out"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(
        TestGraph(
            name="mixed_width_labels",
            graph=g,
            tags={"mixed-width", "skip-light"},
            description="Graph with very different node label widths",
            expected_challenges="Node sizing, alignment with width variation",
        )
    )

    # 11. Random DAG (medium sparse)
    g = _random_dag(50, 70, seed=42)
    graphs.append(
        TestGraph(
            name="random_dag_50",
            graph=g,
            tags={"large-sparse"},
            description="Random DAG: 50 nodes, ~70 edges",
            expected_challenges="General layout quality at medium scale",
        )
    )

    # 12. Random DAG (large sparse)
    g = _random_dag(200, 300, seed=42)
    graphs.append(
        TestGraph(
            name="random_dag_200",
            graph=g,
            tags={"large-sparse"},
            description="Random DAG: 200 nodes, ~300 edges",
            expected_challenges="Scalability, readability at scale",
        )
    )

    # 13. Grid-like DAG
    edges = []
    rows, cols = 5, 5
    for r in range(rows):
        for c in range(cols):
            node = f"n{r}_{c}"
            if c + 1 < cols:
                edges.append((node, f"n{r}_{c + 1}"))
            if r + 1 < rows:
                edges.append((node, f"n{r + 1}_{c}"))
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(
        TestGraph(
            name="grid_5x5",
            graph=g,
            tags={"diamond", "large-dense", "undirected"},
            description="5x5 grid DAG with horizontal and vertical edges",
            expected_challenges="Grid regularity, many crossing opportunities",
        )
    )

    # 14. Multi-source multi-sink
    edges = []
    for i in range(4):
        for j in range(3):
            edges.append((f"src_{i}", f"mid_{j}"))
    for j in range(3):
        for k in range(4):
            edges.append((f"mid_{j}", f"sink_{k}"))
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(
        TestGraph(
            name="bipartite_4_3_4",
            graph=g,
            tags={"wide-parallel", "large-dense"},
            description="Bipartite: 4 sources → 3 middle → 4 sinks",
            expected_challenges="Edge crossing minimization, alignment",
        )
    )

    # 15. Deep nested clusters with residuals across hierarchy boundaries
    edges = [
        ("input", "stem.conv"),
        ("stem.conv", "stage1.block1.conv1"),
        ("stage1.block1.conv1", "stage1.block1.conv2"),
        ("stage1.block1.conv2", "stage1.add"),
        ("stem.conv", "stage1.add"),
        ("stage1.add", "stage2.block1.conv1"),
        ("stage2.block1.conv1", "stage2.block1.conv2"),
        ("stage2.block1.conv2", "stage2.add"),
        ("stage1.add", "stage2.add"),
        ("stage2.add", "head.norm"),
        ("head.norm", "output"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.add_cluster(
        "encoder", [idx["stem.conv"], idx["stage1.add"], idx["stage2.add"]], label="Encoder"
    )
    g.add_cluster(
        "stage1",
        [idx["stage1.block1.conv1"], idx["stage1.block1.conv2"], idx["stage1.add"]],
        label="Stage 1",
        parent="encoder",
    )
    g.add_cluster(
        "stage2",
        [idx["stage2.block1.conv1"], idx["stage2.block1.conv2"], idx["stage2.add"]],
        label="Stage 2",
        parent="encoder",
    )
    g.add_cluster(
        "head",
        [idx["head.norm"]],
        label="Head",
    )
    g.add_cluster(
        "stage1.block1",
        [idx["stage1.block1.conv1"], idx["stage1.block1.conv2"]],
        label="Stage 1 / Block 1",
        parent="stage1",
    )
    graphs.append(
        TestGraph(
            name="hierarchical_residual_stage",
            graph=g,
            tags={"nested-deep", "skip-light"},
            description="Residual stack with 3-level cluster hierarchy",
            expected_challenges=(
                "Nested cluster containment plus residual edges crossing cluster boundaries"
            ),
        )
    )

    # 16. Recurrent-style feedback with explicit self-loop and cycle
    g = DaguaGraph.from_edge_list(
        [
            ("input", "state_update"),
            ("state_prev", "state_update"),
            ("state_update", "state_proj"),
            ("state_proj", "output"),
            ("output", "state_prev"),
            ("state_proj", "state_proj"),
        ]
    )
    graphs.append(
        TestGraph(
            name="recurrent_feedback_cell",
            graph=g,
            tags={"self-loops", "skip-light"},
            description="Small recurrent cell with feedback edge and explicit self-loop",
            expected_challenges="Cycle breaking, self-loop routing, compact feedback placement",
        )
    )

    # 17. Multi-edge bundle between repeated stages
    g = DaguaGraph()
    for node in ("src", "mid", "dst"):
        g.add_node(node)
    for _ in range(3):
        g.add_edge("src", "mid")
    for _ in range(2):
        g.add_edge("mid", "dst")
    g.add_edge("src", "dst")
    graphs.append(
        TestGraph(
            name="parallel_multiedge_bundle",
            graph=g,
            tags={"multi-edge", "diamond"},
            description="Duplicate edges between the same node pairs plus a direct bypass edge",
            expected_challenges="Multi-edge routing separation and duplicate-edge stability",
        )
    )

    # 18. Disconnected components mixing common motifs
    g = DaguaGraph.from_edge_list(
        [
            ("enc_in", "enc_conv"),
            ("enc_conv", "enc_relu"),
            ("enc_relu", "enc_out"),
            ("res_in", "res_conv1"),
            ("res_conv1", "res_conv2"),
            ("res_in", "res_add"),
            ("res_conv2", "res_add"),
            ("res_add", "res_out"),
        ]
    )
    graphs.append(
        TestGraph(
            name="disconnected_encoder_residual",
            graph=g,
            tags={"disconnected", "skip-light"},
            description="Two disconnected components: a linear encoder and a residual block",
            expected_challenges="Component packing, stable spacing across disconnected subgraphs",
        )
    )

    # 19. Mixture-of-experts style sparse routing
    edges = [
        ("input", "embed"),
        ("embed", "router"),
        ("router", "expert_0"),
        ("router", "expert_3"),
        ("embed", "expert_1"),
        ("embed", "expert_2"),
        ("expert_0", "combine"),
        ("expert_1", "combine"),
        ("expert_2", "combine"),
        ("expert_3", "combine"),
        ("combine", "output"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.add_cluster("experts", [idx[f"expert_{i}"] for i in range(4)], label="Experts")
    graphs.append(
        TestGraph(
            name="moe_router_sparse",
            graph=g,
            tags={"wide-parallel", "large-dense", "nested-shallow"},
            description="Mixture-of-experts routing with sparse fan-out and dense fan-in",
            expected_challenges="Wide expert fan-out, merge alignment, cluster labeling",
        )
    )

    # 20. Ragged multiscale pyramid with lateral skips
    edges = [
        ("input", "p3"),
        ("p3", "p4"),
        ("p4", "p5"),
        ("p5", "top_down_5"),
        ("top_down_5", "merge_4"),
        ("p4", "merge_4"),
        ("merge_4", "top_down_4"),
        ("top_down_4", "merge_3"),
        ("p3", "merge_3"),
        ("merge_3", "detect_small"),
        ("merge_4", "detect_medium"),
        ("top_down_5", "detect_large"),
        ("detect_small", "out"),
        ("detect_medium", "out"),
        ("detect_large", "out"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(
        TestGraph(
            name="ragged_feature_pyramid",
            graph=g,
            tags={"skip-heavy", "wide-parallel", "diamond"},
            description="Feature-pyramid style graph with ragged lateral merges",
            expected_challenges="Long lateral skips, fan-in across scales, uneven branch widths",
        )
    )

    # 21. Kitchen sink model: nested clusters + skips + loops + multi-edges + wide fanout
    g = DaguaGraph()
    for node in [
        "input",
        "stem.conv",
        "stem.norm",
        "stem.act",
        "router",
        "expert_a.0",
        "expert_a.1",
        "expert_b.0",
        "expert_b.1",
        "expert_c.0",
        "expert_c.1",
        "merge",
        "residual_add",
        "memory",
        "memory",
        "feedback_gate",
        "head.norm",
        "classifier",
        "aux_head",
        "output",
    ]:
        if node not in g._id_to_index:
            g.add_node(node)

    for edge in [
        ("input", "stem.conv"),
        ("stem.conv", "stem.norm"),
        ("stem.norm", "stem.act"),
        ("stem.act", "router"),
        ("router", "expert_a.0"),
        ("router", "expert_b.0"),
        ("router", "expert_c.0"),
        ("expert_a.0", "expert_a.1"),
        ("expert_b.0", "expert_b.1"),
        ("expert_c.0", "expert_c.1"),
        ("expert_a.1", "merge"),
        ("expert_b.1", "merge"),
        ("expert_c.1", "merge"),
        ("stem.act", "residual_add"),
        ("merge", "residual_add"),
        ("residual_add", "head.norm"),
        ("head.norm", "classifier"),
        ("classifier", "output"),
        ("head.norm", "aux_head"),
        ("aux_head", "output"),
        ("residual_add", "feedback_gate"),
        ("feedback_gate", "memory"),
        ("memory", "router"),
        ("memory", "memory"),
    ]:
        g.add_edge(*edge)
    # Duplicate routing edge to force multi-edge handling.
    g.add_edge("router", "expert_b.0")

    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.add_cluster(
        "backbone", [idx["stem.conv"], idx["stem.norm"], idx["stem.act"]], label="Backbone"
    )
    g.add_cluster(
        "experts",
        [
            idx["expert_a.0"],
            idx["expert_a.1"],
            idx["expert_b.0"],
            idx["expert_b.1"],
            idx["expert_c.0"],
            idx["expert_c.1"],
        ],
        label="Experts",
    )
    g.add_cluster(
        "expert_a", [idx["expert_a.0"], idx["expert_a.1"]], label="Expert A", parent="experts"
    )
    g.add_cluster(
        "expert_b", [idx["expert_b.0"], idx["expert_b.1"]], label="Expert B", parent="experts"
    )
    g.add_cluster(
        "expert_c", [idx["expert_c.0"], idx["expert_c.1"]], label="Expert C", parent="experts"
    )
    g.add_cluster(
        "expert_b.inner", [idx["expert_b.1"]], label="Expert B / Inner", parent="expert_b"
    )
    g.add_cluster("heads", [idx["head.norm"], idx["classifier"], idx["aux_head"]], label="Heads")
    graphs.append(
        TestGraph(
            name="kitchen_sink_hybrid_net",
            graph=g,
            tags={
                "nested-deep",
                "skip-heavy",
                "wide-parallel",
                "self-loops",
                "multi-edge",
                "mixed-width",
            },
            description=(
                "Overloaded hybrid graph combining experts, residuals, "
                "feedback, nested clusters, aux head, and varied labels"
            ),
            expected_challenges=(
                "Full-stack stress test: nested clusters, loops, duplicate "
                "edges, wide branches, residuals, and mixed label widths"
            ),
        )
    )

    # 22. Kitchen sink system graph: disconnected subsystems + dense handoffs + hierarchy
    g = DaguaGraph.from_edge_list(
        [
            ("api.gateway", "auth.validate"),
            ("auth.validate", "router.dispatch"),
            ("router.dispatch", "svc.search"),
            ("router.dispatch", "svc.reco"),
            ("router.dispatch", "svc.ads"),
            ("svc.search", "join.rank"),
            ("svc.reco", "join.rank"),
            ("svc.ads", "join.rank"),
            ("join.rank", "cache.write"),
            ("cache.write", "response.serialize"),
            ("response.serialize", "response.emit"),
            ("join.rank", "metrics.aggregate"),
            ("metrics.aggregate", "alerts.loop"),
            ("alerts.loop", "metrics.aggregate"),
            ("alerts.loop", "alerts.loop"),
            ("offline.ingest", "offline.train"),
            ("offline.train", "offline.eval"),
            ("offline.eval", "model.registry"),
            ("model.registry", "router.dispatch"),
            ("model.registry", "svc.reco"),
            ("audit.ingest", "audit.report"),
        ]
    )
    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.add_cluster(
        "online",
        [
            idx[n]
            for n in [
                "api.gateway",
                "auth.validate",
                "router.dispatch",
                "svc.search",
                "svc.reco",
                "svc.ads",
                "join.rank",
                "cache.write",
                "response.serialize",
                "response.emit",
            ]
        ],
        label="Online Path",
    )
    g.add_cluster(
        "services",
        [idx[n] for n in ["svc.search", "svc.reco", "svc.ads"]],
        label="Services",
        parent="online",
    )
    g.add_cluster(
        "observability",
        [idx[n] for n in ["metrics.aggregate", "alerts.loop"]],
        label="Observability",
    )
    g.add_cluster(
        "offline",
        [idx[n] for n in ["offline.ingest", "offline.train", "offline.eval", "model.registry"]],
        label="Offline",
    )
    g.add_cluster("audit", [idx[n] for n in ["audit.ingest", "audit.report"]], label="Audit")
    graphs.append(
        TestGraph(
            name="kitchen_sink_platform_graph",
            graph=g,
            tags={
                "nested-deep",
                "disconnected",
                "self-loops",
                "wide-parallel",
                "skip-heavy",
                "mixed-width",
            },
            description=(
                "Platform-style graph mixing online services, offline "
                "training, observability loops, and cross-system handoffs"
            ),
            expected_challenges=(
                "Disconnected subsystems, long handoff edges, nested service "
                "clusters, explicit loops, and varied label widths"
            ),
        )
    )

    # 23. Extreme mixed-width labels with uneven branch depths
    edges = [
        ("x", "TokenEmbedding(vocab_size=50257, embedding_dim=4096)"),
        (
            "TokenEmbedding(vocab_size=50257, embedding_dim=4096)",
            "LayerNorm(normalized_shape=(4096,), eps=1e-05)",
        ),
        ("LayerNorm(normalized_shape=(4096,), eps=1e-05)", "Q"),
        ("LayerNorm(normalized_shape=(4096,), eps=1e-05)", "K"),
        ("LayerNorm(normalized_shape=(4096,), eps=1e-05)", "V"),
        ("Q", "ScaledDotProductAttention(masked=True, causal=True, dropout=0.1)"),
        ("K", "ScaledDotProductAttention(masked=True, causal=True, dropout=0.1)"),
        ("V", "ScaledDotProductAttention(masked=True, causal=True, dropout=0.1)"),
        ("ScaledDotProductAttention(masked=True, causal=True, dropout=0.1)", "+"),
        ("x", "+"),
        ("+", "MLP(hidden_dim=16384, activation=SiLU, gated=True)"),
        ("MLP(hidden_dim=16384, activation=SiLU, gated=True)", "out"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(
        TestGraph(
            name="extreme_mixed_width_transformer",
            graph=g,
            tags={"mixed-width", "wide-parallel", "skip-light"},
            description="Extreme short-vs-long labels inside a transformer-style residual block",
            expected_challenges=(
                "Very uneven node widths, branch alignment, and label-driven spacing"
            ),
        )
    )

    # 24. Hub-and-spoke fanout with uneven fan-in and long labels
    edges = [
        ("gateway", "tiny"),
        ("gateway", "short_branch"),
        ("gateway", "reasonably_sized_processing_stage"),
        ("gateway", "ExtremelyVerboseAndOverlyDescriptiveNormalizationSubsystem"),
        ("gateway", "mid"),
        ("tiny", "merge"),
        ("short_branch", "merge"),
        ("reasonably_sized_processing_stage", "merge"),
        ("ExtremelyVerboseAndOverlyDescriptiveNormalizationSubsystem", "merge"),
        ("mid", "late_side_path"),
        ("late_side_path", "final_merge"),
        ("merge", "final_merge"),
        ("final_merge", "output"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(
        TestGraph(
            name="hub_fanout_label_skew",
            graph=g,
            tags={"mixed-width", "wide-parallel", "diamond"},
            description=(
                "Single hub with strongly uneven branch label widths and an asymmetric late merge"
            ),
            expected_challenges=(
                "Fanout distribution, visual balance, and asymmetric merge routing under width skew"
            ),
        )
    )

    # 25. Nested clusters with long labels and duplicate inter-cluster handoffs
    edges = [
        ("input", "preprocess.tokenize"),
        ("preprocess.tokenize", "encoder.stage_1_attention_projection"),
        ("encoder.stage_1_attention_projection", "encoder.stage_1_feedforward"),
        ("encoder.stage_1_feedforward", "handoff"),
        ("input", "handoff"),
        ("handoff", "decoder.cross_attention_query"),
        ("handoff", "decoder.cross_attention_key_value"),
        ("decoder.cross_attention_query", "decoder.merge"),
        ("decoder.cross_attention_key_value", "decoder.merge"),
        ("decoder.merge", "LongOutputProjectionLayerWithAuxiliaryCalibration"),
        ("LongOutputProjectionLayerWithAuxiliaryCalibration", "output"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    g.add_edge("handoff", "decoder.cross_attention_key_value")
    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.add_cluster(
        "encoder",
        [idx["encoder.stage_1_attention_projection"], idx["encoder.stage_1_feedforward"]],
        label="Encoder",
    )
    g.add_cluster(
        "decoder",
        [
            idx["decoder.cross_attention_query"],
            idx["decoder.cross_attention_key_value"],
            idx["decoder.merge"],
        ],
        label="Decoder",
    )
    g.add_cluster(
        "decoder.cross_attention",
        [idx["decoder.cross_attention_query"], idx["decoder.cross_attention_key_value"]],
        label="Cross Attention",
        parent="decoder",
    )
    graphs.append(
        TestGraph(
            name="clustered_longlabel_handoffs",
            graph=g,
            tags={"mixed-width", "nested-deep", "multi-edge", "skip-light"},
            description="Nested clusters with long labels and repeated inter-cluster handoff edges",
            expected_challenges=(
                "Cluster sizing, duplicate inter-cluster routing, and label-width imbalance"
            ),
        )
    )

    # 26. Stressful disconnected collage of tiny, huge, and cyclic components
    g = DaguaGraph.from_edge_list(
        [
            ("a", "b"),
            ("StandaloneSuperLongLabelForAnOtherwiseTinyChainNode", "tail"),
            ("cycle.start", "cycle.mid"),
            ("cycle.mid", "cycle.end"),
            ("cycle.end", "cycle.start"),
            ("cycle.end", "cycle.end"),
        ]
    )
    graphs.append(
        TestGraph(
            name="disconnected_label_cycle_collage",
            graph=g,
            tags={"mixed-width", "disconnected", "self-loops"},
            description=(
                "Disconnected collage mixing a tiny chain, a huge label, a "
                "directed cycle, and a self-loop"
            ),
            expected_challenges=(
                "Component packing under wildly different local scales and cyclic structures"
            ),
        )
    )

    # 27. Mixed node shapes and routing modes
    g = DaguaGraph.from_edge_list(
        [
            ("input", "ellipse_norm"),
            ("ellipse_norm", "diamond_gate"),
            ("ellipse_norm", "roundrect_path"),
            ("diamond_gate", "merge"),
            ("roundrect_path", "merge"),
            ("merge", "circle_sink"),
        ]
    )
    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.node_styles[idx["input"]] = NodeStyle(shape="rect")
    g.node_styles[idx["ellipse_norm"]] = NodeStyle(shape="ellipse")
    g.node_styles[idx["diamond_gate"]] = NodeStyle(shape="diamond")
    g.node_styles[idx["roundrect_path"]] = NodeStyle(shape="roundrect")
    g.node_styles[idx["circle_sink"]] = NodeStyle(shape="circle")
    g.edge_styles[0] = EdgeStyle(routing="straight", port_style="center", curvature=0.0)
    g.edge_styles[1] = EdgeStyle(routing="ortho", port_style="distributed", curvature=0.2)
    g.edge_styles[2] = EdgeStyle(routing="bezier", port_style="center", curvature=0.7)
    graphs.append(
        TestGraph(
            name="shape_and_routing_matrix",
            graph=g,
            tags={"mixed-width", "diamond", "wide-parallel"},
            description="Mixed node shapes with straight, ortho, and bezier edge routing modes",
            expected_challenges=(
                "Shape-aware ports, mixed routing styles, and asymmetric merge geometry"
            ),
        )
    )

    # 28. Center-port hub with aggressive back-edge and cycle pressure
    g = DaguaGraph.from_edge_list(
        [
            ("hub", "decode_a"),
            ("hub", "decode_b"),
            ("hub", "decode_c"),
            ("decode_a", "join"),
            ("decode_b", "join"),
            ("decode_c", "join"),
            ("join", "head"),
            ("head", "hub"),
            ("decode_b", "decode_b"),
        ]
    )
    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.node_styles[idx["hub"]] = NodeStyle(shape="ellipse")
    for e_idx in range(len(g.edge_styles)):
        g.edge_styles[e_idx] = EdgeStyle(port_style="center", routing="bezier", curvature=0.6)
    graphs.append(
        TestGraph(
            name="center_port_backedge_hub",
            graph=g,
            tags={"self-loops", "skip-heavy", "wide-parallel"},
            description="Center-port fanout hub with explicit back-edge and self-loop",
            expected_challenges=(
                "Cycle breaking around a dominant hub and crowded center-port routing"
            ),
        )
    )

    # 29. Cluster member style stress with inter-cluster edge variety
    g = DaguaGraph.from_edge_list(
        [
            ("ingest", "prep.clean"),
            ("prep.clean", "prep.batch"),
            ("prep.batch", "core.encode"),
            ("core.encode", "core.route"),
            ("core.route", "core.decode"),
            ("core.decode", "post.merge"),
            ("prep.batch", "post.merge"),
            ("post.merge", "serve"),
        ]
    )
    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.add_cluster("prep", [idx["prep.clean"], idx["prep.batch"]], label="Prep")
    g.add_cluster("core", [idx["core.encode"], idx["core.route"], idx["core.decode"]], label="Core")
    g.cluster_styles["core"] = ClusterStyle(
        member_node_style=NodeStyle(shape="diamond"),
        member_edge_style=EdgeStyle(routing="ortho", port_style="center", curvature=0.1),
    )
    graphs.append(
        TestGraph(
            name="cluster_member_style_stress",
            graph=g,
            tags={"nested-shallow", "skip-light", "diamond"},
            description=(
                "Clusters with member style overrides that alter node shapes "
                "and edge routing defaults"
            ),
            expected_challenges=(
                "Cluster-scoped style cascades combined with cross-cluster skip edges"
            ),
        )
    )

    # 30. Dense edge-label braid converging into a central merge
    g = DaguaGraph()
    for node in ["src_a", "src_b", "src_c", "mid_a", "mid_b", "mid_c", "merge", "sink"]:
        g.add_node(node)
    labeled_edges = [
        ("src_a", "mid_a", "token projection"),
        ("src_a", "mid_b", "residual align"),
        ("src_b", "mid_b", "attention weights"),
        ("src_b", "mid_c", "value mixing"),
        ("src_c", "mid_a", "gating score"),
        ("src_c", "mid_c", "context route"),
        ("mid_a", "merge", "left branch"),
        ("mid_b", "merge", "center branch"),
        ("mid_c", "merge", "right branch"),
        ("merge", "sink", "final aggregation output"),
    ]
    for src, tgt, label in labeled_edges:
        g.add_edge(
            src, tgt, label=label, style=EdgeStyle(label_position=0.45, label_avoidance=True)
        )
    graphs.append(
        TestGraph(
            name="edge_label_braid",
            graph=g,
            tags={"wide-parallel", "diamond", "mixed-width"},
            description=(
                "Many competing edge labels packed around a braid of crossings and a central merge"
            ),
            expected_challenges="Edge-label collision avoidance around dense crossing-prone merges",
        )
    )

    # 31. Deep nested cluster labels with long titles and central labeled handoffs
    g = DaguaGraph.from_edge_list(
        [
            ("input", "encoder.stage0"),
            ("encoder.stage0", "encoder.stage1.attention"),
            ("encoder.stage1.attention", "encoder.stage1.mlp"),
            ("encoder.stage1.mlp", "bridge"),
            ("bridge", "decoder.stage0.cross_attention"),
            ("decoder.stage0.cross_attention", "decoder.stage0.ffn"),
            ("decoder.stage0.ffn", "output"),
        ]
    )
    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.add_cluster(
        "Very Long Encoder Cluster Title",
        [idx["encoder.stage0"], idx["encoder.stage1.attention"], idx["encoder.stage1.mlp"]],
        label="Very Long Encoder Cluster Title",
        style=ClusterStyle(label_offset=(10.0, 16.0)),
    )
    g.add_cluster(
        "Encoder Inner Block With Lengthy Subtitle",
        [idx["encoder.stage1.attention"], idx["encoder.stage1.mlp"]],
        label="Encoder Inner Block With Lengthy Subtitle",
        parent="Very Long Encoder Cluster Title",
        style=ClusterStyle(label_offset=(10.0, 14.0)),
    )
    g.add_cluster(
        "Decoder Cluster With Another Long Title",
        [idx["decoder.stage0.cross_attention"], idx["decoder.stage0.ffn"]],
        label="Decoder Cluster With Another Long Title",
        style=ClusterStyle(label_offset=(10.0, 16.0)),
    )
    g.add_edge("encoder.stage1.attention", "bridge", label="attention handoff to bridge")
    g.add_edge("bridge", "decoder.stage0.cross_attention", label="decoder context transfer path")
    graphs.append(
        TestGraph(
            name="nested_cluster_label_stack",
            graph=g,
            tags={"nested-deep", "mixed-width", "skip-light"},
            description=(
                "Long nested cluster labels combined with central inter-cluster edge labels"
            ),
            expected_challenges=(
                "Cluster title stacking and edge-label placement near cluster boundaries"
            ),
        )
    )

    # 32. Label storm: edge labels plus cluster labels competing in a small graph
    g = DaguaGraph.from_edge_list(
        [
            ("input", "prep"),
            ("prep", "branch_left"),
            ("prep", "branch_right"),
            ("branch_left", "join"),
            ("branch_right", "join"),
            ("join", "output"),
        ]
    )
    g.edge_labels = [
        "ingest and normalize",
        "left expert path",
        "right expert path with extra context",
        "left merge contribution",
        "right merge contribution",
        "emit prediction",
    ]
    g.cluster_styles["Prep Cluster"] = ClusterStyle(label_offset=(8.0, 12.0))
    g.cluster_styles["Join Cluster"] = ClusterStyle(label_offset=(8.0, 12.0))
    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.add_cluster(
        "Prep Cluster", [idx["prep"], idx["branch_left"], idx["branch_right"]], label="Prep Cluster"
    )
    g.add_cluster("Join Cluster", [idx["join"]], label="Join Cluster")
    graphs.append(
        TestGraph(
            name="small_label_storm",
            graph=g,
            tags={"mixed-width", "wide-parallel", "nested-shallow"},
            description=(
                "Compact graph where edge labels and cluster labels all compete for the same area"
            ),
            expected_challenges=(
                "Label crowding in a small footprint with both edge and cluster annotations"
            ),
        )
    )

    # 33. Deep ladder with many long residual bridges across stages
    edges = []
    for i in range(8):
        edges.extend(
            [
                (f"stage{i}.main", f"stage{i}.norm"),
                (f"stage{i}.norm", f"stage{i}.act"),
            ]
        )
        if i < 7:
            edges.append((f"stage{i}.act", f"stage{i + 1}.main"))
    for i in range(6):
        edges.append((f"stage{i}.main", f"stage{i + 2}.merge"))
    for i in range(4):
        edges.append((f"stage{i}.norm", f"stage{i + 4}.merge"))
    for i in range(2, 8):
        edges.append((f"stage{i}.merge", f"stage{i}.out"))
    edges.append(("input", "stage0.main"))
    edges.append(("stage7.out", "output"))
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(
        TestGraph(
            name="long_range_residual_ladder",
            graph=g,
            tags={"linear-deep", "skip-heavy", "wide-parallel"},
            description=(
                "Deep ladder with many long residual bridges that leap multiple stages ahead"
            ),
            expected_challenges=(
                "Long-span skip routing, preserving ladder readability, and avoiding bridge tangles"
            ),
        )
    )

    # 34. Interleaved sibling clusters with dense cross-talk
    g = DaguaGraph.from_edge_list(
        [
            ("input", "enc.a0"),
            ("input", "enc.b0"),
            ("enc.a0", "enc.a1"),
            ("enc.a1", "enc.a2"),
            ("enc.b0", "enc.b1"),
            ("enc.b1", "enc.b2"),
            ("enc.a1", "enc.b2"),
            ("enc.b1", "enc.a2"),
            ("enc.a2", "join"),
            ("enc.b2", "join"),
            ("join", "decoder.left"),
            ("join", "decoder.right"),
            ("decoder.left", "decoder.merge"),
            ("decoder.right", "decoder.merge"),
            ("enc.a0", "decoder.right"),
            ("enc.b0", "decoder.left"),
            ("decoder.merge", "output"),
        ]
    )
    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.add_cluster(
        "system",
        [
            idx["enc.a0"],
            idx["enc.a1"],
            idx["enc.a2"],
            idx["enc.b0"],
            idx["enc.b1"],
            idx["enc.b2"],
            idx["decoder.left"],
            idx["decoder.right"],
            idx["decoder.merge"],
        ],
        label="System",
    )
    g.add_cluster(
        "encoder",
        [idx["enc.a0"], idx["enc.a1"], idx["enc.a2"], idx["enc.b0"], idx["enc.b1"], idx["enc.b2"]],
        label="Encoder",
        parent="system",
    )
    g.add_cluster(
        "encoder.path_a",
        [idx["enc.a0"], idx["enc.a1"], idx["enc.a2"]],
        label="Path A",
        parent="encoder",
    )
    g.add_cluster(
        "encoder.path_b",
        [idx["enc.b0"], idx["enc.b1"], idx["enc.b2"]],
        label="Path B",
        parent="encoder",
    )
    g.add_cluster(
        "decoder",
        [idx["decoder.left"], idx["decoder.right"], idx["decoder.merge"]],
        label="Decoder",
        parent="system",
    )
    graphs.append(
        TestGraph(
            name="interleaved_cluster_crosstalk",
            graph=g,
            tags={"nested-deep", "skip-heavy", "wide-parallel"},
            description=(
                "Sibling clusters whose internal paths repeatedly cross-talk "
                "before merging into a decoder"
            ),
            expected_challenges=(
                "Keeping sibling cluster identity legible while routing many "
                "inter-cluster cross-links"
            ),
        )
    )

    # 35. Asymmetric hourglass with a dominant hub and lopsided late merge
    g = DaguaGraph.from_edge_list(
        [
            ("source_a", "pre_a"),
            ("source_b", "pre_b"),
            ("source_c", "pre_c"),
            ("pre_a", "hub"),
            ("pre_b", "hub"),
            ("pre_c", "hub"),
            ("hub", "thin_path"),
            ("hub", "fat_path.0"),
            ("fat_path.0", "fat_path.1"),
            ("fat_path.1", "fat_path.2"),
            ("fat_path.2", "late_join"),
            ("thin_path", "late_join"),
            ("source_a", "late_join"),
            ("late_join", "head"),
            ("head", "output"),
        ]
    )
    graphs.append(
        TestGraph(
            name="asymmetric_hourglass_hub",
            graph=g,
            tags={"diamond", "wide-parallel", "skip-light"},
            description=(
                "Multi-source hourglass with a dominant hub, one thin path, "
                "and one much fatter late branch"
            ),
            expected_challenges=(
                "Visual balance around a hub and preserving hourglass "
                "structure despite strong asymmetry"
            ),
        )
    )

    # 36. Multi-scale skip cascade with cross-resolution handoffs
    g = DaguaGraph.from_edge_list(
        [
            ("input", "stem"),
            ("stem", "p2"),
            ("p2", "p3"),
            ("p3", "p4"),
            ("p4", "p5"),
            ("p5", "topdown4"),
            ("topdown4", "topdown3"),
            ("topdown3", "topdown2"),
            ("p4", "topdown4"),
            ("p3", "topdown3"),
            ("p2", "topdown2"),
            ("p5", "detect_large"),
            ("topdown4", "detect_mid"),
            ("topdown3", "detect_small"),
            ("topdown2", "detect_tiny"),
            ("p2", "detect_large"),
            ("p3", "detect_mid"),
            ("p4", "detect_small"),
            ("detect_large", "fuse"),
            ("detect_mid", "fuse"),
            ("detect_small", "fuse"),
            ("detect_tiny", "fuse"),
            ("fuse", "output"),
        ]
    )
    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.add_cluster(
        "bottom_up", [idx["stem"], idx["p2"], idx["p3"], idx["p4"], idx["p5"]], label="Bottom-Up"
    )
    g.add_cluster("top_down", [idx["topdown4"], idx["topdown3"], idx["topdown2"]], label="Top-Down")
    g.add_cluster(
        "heads",
        [idx["detect_large"], idx["detect_mid"], idx["detect_small"], idx["detect_tiny"]],
        label="Heads",
    )
    graphs.append(
        TestGraph(
            name="multiscale_skip_cascade",
            graph=g,
            tags={"skip-heavy", "nested-shallow", "wide-parallel"},
            description=(
                "Feature-pyramid-style graph with repeated cross-resolution "
                "skips and four detection heads"
            ),
            expected_challenges=(
                "Cross-scale skip routing, head alignment, and preserving the cascade structure"
            ),
        )
    )

    # 37. Near-layered graph with alternating braids and back-pressure
    g = DaguaGraph.from_edge_list(
        [
            ("input", "a0"),
            ("input", "b0"),
            ("a0", "a1"),
            ("b0", "b1"),
            ("a1", "a2"),
            ("b1", "b2"),
            ("a0", "b1"),
            ("b0", "a1"),
            ("a1", "b2"),
            ("b1", "a2"),
            ("a2", "merge"),
            ("b2", "merge"),
            ("merge", "tail0"),
            ("tail0", "tail1"),
            ("tail1", "tail2"),
            ("tail2", "merge"),
            ("tail2", "output"),
        ]
    )
    graphs.append(
        TestGraph(
            name="braided_feedback_tails",
            graph=g,
            tags={"skip-heavy", "diamond", "linear-deep"},
            description=(
                "Alternating braid that looks layered at first, then folds "
                "back through a late feedback tail"
            ),
            expected_challenges=(
                "Crossing-heavy braid alignment plus a late back-pressure loop near the sink"
            ),
        )
    )

    # 38. Very wide front half collapsing into one asymmetric late merge
    g = DaguaGraph.from_edge_list(
        [
            ("input", "w0"),
            ("input", "w1"),
            ("input", "w2"),
            ("input", "w3"),
            ("input", "w4"),
            ("w0", "m0"),
            ("w1", "m1"),
            ("w2", "m2"),
            ("w3", "m3"),
            ("w4", "m4"),
            ("m0", "join.left"),
            ("m1", "join.left"),
            ("m2", "join.center"),
            ("m3", "join.right"),
            ("m4", "join.right"),
            ("w0", "late.merge"),
            ("w2", "late.merge"),
            ("w4", "late.merge"),
            ("join.left", "late.merge"),
            ("join.center", "late.merge"),
            ("join.right", "late.merge"),
            ("late.merge", "head"),
            ("head", "output"),
        ]
    )
    graphs.append(
        TestGraph(
            name="width_skew_late_merge",
            graph=g,
            tags={"wide-parallel", "skip-heavy", "diamond"},
            description=(
                "Very wide early fan-out that collapses into an asymmetric "
                "late merge with long direct skips"
            ),
            expected_challenges=(
                "Balancing a wide first half against a compact sink while "
                "keeping long skips readable"
            ),
        )
    )

    # 39. Two nearly symmetric residual trunks with one broken branch
    g = DaguaGraph.from_edge_list(
        [
            ("input", "left.0"),
            ("input", "right.0"),
            ("left.0", "left.1"),
            ("left.1", "left.2"),
            ("left.2", "merge"),
            ("right.0", "right.1"),
            ("right.1", "right.2"),
            ("right.2", "merge"),
            ("input", "merge"),
            ("left.0", "left.skip"),
            ("left.skip", "merge"),
            ("right.0", "right.skip"),
            ("right.skip", "right.2"),
            ("right.1", "breakout"),
            ("breakout", "merge"),
            ("merge", "output"),
        ]
    )
    graphs.append(
        TestGraph(
            name="broken_symmetry_residual_pair",
            graph=g,
            tags={"skip-heavy", "diamond", "wide-parallel"},
            description=(
                "Two almost-symmetric residual trunks where one arm breaks "
                "pattern just before the merge"
            ),
            expected_challenges=(
                "Preserving the repeated pattern while making the broken "
                "branch obvious instead of messy"
            ),
        )
    )

    # 40. Deep spine with a dominant hub spraying long skips across layers
    g = DaguaGraph.from_edge_list(
        [
            ("input", "s0"),
            ("s0", "s1"),
            ("s1", "s2"),
            ("s2", "s3"),
            ("s3", "s4"),
            ("s4", "s5"),
            ("s5", "output"),
            ("s1", "hub"),
            ("hub", "x0"),
            ("hub", "x1"),
            ("hub", "x2"),
            ("hub", "x3"),
            ("x0", "s3"),
            ("x1", "s4"),
            ("x2", "s5"),
            ("x3", "output"),
            ("hub", "output"),
            ("s0", "x2"),
            ("s2", "x3"),
        ]
    )
    graphs.append(
        TestGraph(
            name="hub_skip_superfan",
            graph=g,
            tags={"linear-deep", "skip-heavy", "wide-parallel"},
            description=(
                "Deep backbone interrupted by a dominant hub that fans long "
                "skips into late layers and the sink"
            ),
            expected_challenges=(
                "Avoiding a central skip knot around a high-degree hub while "
                "preserving backbone flow"
            ),
        )
    )

    # 41. Outerplanar DAG with a non-crossing fan
    graphs.append(
        TestGraph(
            name="outerplanar_dag_20",
            graph=_make_outerplanar_dag_graph(20),
            tags={"planar", "dag", "sparse"},
            description="Outerplanar DAG with a path backbone and non-crossing forward fan",
            expected_challenges=(
                "Preserving outer-face ordering without introducing avoidable bends"
            ),
        )
    )

    # 42. Dense planar graph from nested triangulated cycles
    graphs.append(
        TestGraph(
            name="planar_60",
            graph=_make_planar_nested_cycles_graph(rings=5, nodes_per_ring=12),
            tags={"planar", "dense", "undirected"},
            description=(
                "Planar 60-node graph built from concentric cycles with triangulated spokes"
            ),
            expected_challenges="Dense local structure without any true crossing requirement",
        )
    )

    # 43. Random cubic graph
    graphs.append(
        TestGraph(
            name="regular_3_30",
            graph=_make_regular_graph(degree=3, num_nodes=30, seed=42),
            tags={"regular", "sparse", "undirected"},
            description="Deterministic random 3-regular graph on 30 nodes",
            expected_challenges="Uniform local degree without obvious hierarchy or hubs",
        )
    )

    # 44. Random 4-regular graph
    graphs.append(
        TestGraph(
            name="regular_4_40",
            graph=_make_regular_graph(degree=4, num_nodes=40, seed=42),
            tags={"regular", "undirected"},
            description="Deterministic random 4-regular graph on 40 nodes",
            expected_challenges="Degree-uniform structure with moderate density and few landmarks",
        )
    )

    # 45. Triangular lattice
    graphs.append(
        TestGraph(
            name="triangular_lattice_36",
            graph=_make_triangular_lattice_graph(rows=6, cols=6),
            tags={"grid", "lattice", "planar", "undirected"},
            description="6x6 triangular lattice patch with right/down/down-right links",
            expected_challenges="Maintaining regular spacing across a high-degree planar mesh",
        )
    )

    # 46. Hexagonal lattice
    graphs.append(
        TestGraph(
            name="hexagonal_lattice_42",
            graph=_make_hexagonal_lattice_graph(rows=6, cols=7),
            tags={"grid", "lattice", "planar", "sparse", "undirected"},
            description="6x7 honeycomb lattice patch with degree-3 interior structure",
            expected_challenges="Showing the hexagonal rhythm instead of collapsing rows together",
        )
    )

    # 47. Protein interaction network with clustered modules
    graphs.append(
        TestGraph(
            name="protein_ppi_200",
            graph=_make_protein_ppi_graph(seed=42),
            tags={"scale-free", "clustered", "biological", "undirected"},
            description=(
                "Protein-protein interaction graph with a scale-free core and dense modules"
            ),
            expected_challenges=(
                "Separating biological modules while preserving hub-mediated bridges"
            ),
        )
    )

    # 48. Temporal citation DAG
    graphs.append(
        TestGraph(
            name="citation_dag_300",
            graph=_make_citation_dag_graph(layer_count=10, layer_size=30, seed=42),
            tags={"dag", "layered", "scale-free"},
            description="300-paper citation DAG across 10 temporal layers",
            expected_challenges="Temporal layering with hub-like highly cited early papers",
        )
    )

    # 49. Sierpinski triangle graph
    graphs.append(
        TestGraph(
            name="sierpinski_42",
            graph=_make_sierpinski_graph(depth=3),
            tags={"fractal", "planar", "sparse", "undirected"},
            description="Depth-3 Sierpinski triangle graph",
            expected_challenges="Recursive self-similarity across multiple scales",
        )
    )

    # 50. Chung-Lu power-law random graph with extra triadic closure
    graphs.append(
        TestGraph(
            name="chung_lu_150",
            graph=_make_chung_lu_graph(num_nodes=150, seed=42),
            tags={"scale-free", "clustered", "random", "undirected"},
            description="Chung-Lu style random graph with heavy-tailed expected degrees",
            expected_challenges="Hub dominance combined with locally clustered neighborhoods",
        )
    )

    # 51. Mixed disconnected components
    graphs.append(
        TestGraph(
            name="multi_component_80",
            graph=_make_multi_component_graph(),
            tags={"disconnected", "multi-component", "undirected"},
            description="Disconnected graph with components sized 40, 20, 10, 5, 3, and 2",
            expected_challenges=(
                "Packing very uneven connected components without losing small ones"
            ),
        )
    )

    # 52. Random bipartite graph
    graphs.append(
        TestGraph(
            name="random_bipartite_60",
            graph=_make_random_bipartite_graph(left_size=30, right_size=30, num_edges=90, seed=42),
            tags={"bipartite", "random", "undirected"},
            description="Random 30x30 bipartite graph with ninety cross-partition edges",
            expected_challenges="Crossing minimization under dense two-layer connectivity",
        )
    )

    # 53. Connected weighted graph with log-normal edge weights
    graphs.append(
        TestGraph(
            name="heavy_tail_weights_50",
            graph=_make_heavy_tail_weight_graph(num_nodes=50, seed=42),
            tags={"weighted", "random", "undirected"},
            description="Connected random graph with heavy-tailed log-normal edge weights",
            expected_challenges=(
                "Respecting a few dominant weighted links without collapsing the graph"
            ),
        )
    )

    # 54. Petersen graph
    graphs.append(
        TestGraph(
            name="petersen_10",
            graph=_make_petersen_graph(),
            tags={"regular", "famous", "small", "undirected"},
            description="Classic Petersen graph with deterministic acyclic orientation",
            expected_challenges="Symmetry breaking on a famous small non-planar regular graph",
        )
    )

    graphs.extend(_expanded_structural_graphs())

    return graphs


def _random_dag(n_nodes: int, n_edges: int, seed: int = 42) -> DaguaGraph:
    """Generate a random DAG by sampling edges that respect topological order.

    Parameters
    ----------
    n_nodes : int
        Number of named nodes to materialize.
    n_edges : int
        Target number of sampled forward edges.
    seed : int, default=42
        Random seed for reproducible edge sampling.

    Returns
    -------
    DaguaGraph
        Random DAG with node IDs ``n0`` through ``n{n_nodes - 1}`` and edges
        emitted in sorted string-key order.
    """
    import random

    rng = random.Random(seed)

    node_names = [f"n{i}" for i in range(n_nodes)]
    edges: Set[tuple[str, str]] = set()
    attempts = 0
    while len(edges) < n_edges and attempts < n_edges * 10:
        i = rng.randint(0, n_nodes - 2)
        j = rng.randint(i + 1, n_nodes - 1)
        edges.add((node_names[i], node_names[j]))
        attempts += 1

    return _build_named_graph(node_names, sorted(edges))


# ─── TorchLens Graph Extractors ──────────────────────────────────────────────


def _load_cached_torchlens_graph(cache_path: Path) -> DaguaGraph:
    """Load one cached TorchLens graph from disk.

    Parameters
    ----------
    cache_path : Path
        Cache file to deserialize.

    Returns
    -------
    DaguaGraph
        Cached graph instance.

    Raises
    ------
    TypeError
        If the cache payload is not a ``DaguaGraph``.
    """
    try:
        graph = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        graph = torch.load(cache_path, map_location="cpu")
    if not isinstance(graph, DaguaGraph):
        raise TypeError(f"Unexpected TorchLens cache payload type: {type(graph)!r}")
    return graph


def _cached_torchlens_graph(name: str, graph_builder: Callable[[], DaguaGraph]) -> DaguaGraph:
    """Load or generate a deterministic cached TorchLens graph.

    Parameters
    ----------
    name : str
        Stable benchmark graph name used for the cache filename.
    graph_builder : Callable[[], DaguaGraph]
        Callback that traces the model and constructs the graph on a cache miss.

    Returns
    -------
    DaguaGraph
        Cached or freshly generated graph instance.
    """
    cache_path = _GRAPH_CACHE_DIR / f"{name}.pt"
    if cache_path.exists():
        try:
            return _load_cached_torchlens_graph(cache_path)
        except Exception:
            cache_path.unlink(missing_ok=True)

    graph = graph_builder()
    _GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(graph, cache_path)
    return graph


def _trace_torchlens_graph(name: str, model: Any, inputs: Any, torchlens_module: Any) -> DaguaGraph:
    """Trace a TorchLens model and cache the resulting graph.

    Parameters
    ----------
    name : str
        Stable graph name used for cache lookups.
    model : Any
        PyTorch module to trace.
    inputs : Any
        Input batch forwarded to the model trace.
    torchlens_module : Any
        Imported TorchLens module exposing ``log_forward_pass``.

    Returns
    -------
    DaguaGraph
        Cached graph for the traced model.
    """
    return _cached_torchlens_graph(
        name,
        lambda: DaguaGraph.from_torchlens(
            torchlens_module.log_forward_pass(model, inputs, vis_mode="none")
        ),
    )


def _torchlens_graphs() -> List[TestGraph]:
    """Extract test graphs from TorchLens model traces.

    Returns empty list if TorchLens is not available.
    """
    try:
        import torch.nn as nn
        import torchlens as tl
    except ImportError:
        return []

    graphs = []

    # 1. Simple MLP
    try:
        mlp_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
        )
        x = torch.randn(1, 10)
        g = _trace_torchlens_graph("tl_mlp_3layer", mlp_model, x, tl)
        graphs.append(
            TestGraph(
                name="tl_mlp_3layer",
                graph=g,
                tags={"linear-shallow"},
                source="torchlens",
                description="TorchLens: 3-layer MLP",
            )
        )
    except Exception:
        pass

    # 2. CNN
    try:
        cnn_model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 10),
        )
        x = torch.randn(1, 3, 32, 32)
        g = _trace_torchlens_graph("tl_cnn_small", cnn_model, x, tl)
        graphs.append(
            TestGraph(
                name="tl_cnn_small",
                graph=g,
                tags={"linear-shallow"},
                source="torchlens",
                description="TorchLens: Small CNN (2 conv + fc)",
            )
        )
    except Exception:
        pass

    # 3. ResNet-like block
    try:

        class ResBlock(nn.Module):
            def __init__(self, ch):
                super().__init__()
                self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(ch)
                self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(ch)

            def forward(self, x):
                identity = x
                out = torch.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                return torch.relu(out + identity)

        resnet_like_model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            ResBlock(16),
            ResBlock(16),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        x = torch.randn(1, 3, 32, 32)
        g = _trace_torchlens_graph("tl_resnet_2block", resnet_like_model, x, tl)
        graphs.append(
            TestGraph(
                name="tl_resnet_2block",
                graph=g,
                tags={"skip-light", "nested-shallow"},
                source="torchlens",
                description="TorchLens: 2 residual blocks",
            )
        )
    except Exception:
        pass

    # 4. Simple Transformer encoder layer
    try:
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True
        )
        transformer_model = nn.TransformerEncoder(layer, num_layers=1)
        x = torch.randn(1, 10, 64)
        g = _trace_torchlens_graph("tl_transformer_1layer", transformer_model, x, tl)
        graphs.append(
            TestGraph(
                name="tl_transformer_1layer",
                graph=g,
                tags={"nested-deep", "wide-parallel", "skip-light"},
                source="torchlens",
                description="TorchLens: Single transformer encoder layer",
            )
        )
    except Exception:
        pass

    # ─── Extended TorchLens architectures ────────────────────────────────

    # Import example models from TorchLens test suite
    try:
        from tests.example_models import (
            DiamondLoop,
            LongLoop,
            NestedModules,
            SimpleBranching,
        )

        nested_modules_cls: Any = NestedModules
        simple_branching_cls: Any = SimpleBranching
        diamond_loop_cls: Any = DiamondLoop
        long_loop_cls: Any = LongLoop
    except ImportError:
        # Fall back: mark unavailable if TorchLens test models not importable.
        nested_modules_cls = None
        simple_branching_cls = None
        diamond_loop_cls = None
        long_loop_cls = None

    # 5. Nested module hierarchy (4-level nesting)
    if nested_modules_cls is not None:
        try:
            nested_modules_model = nested_modules_cls()
            x = torch.randn(5)
            g = _trace_torchlens_graph("tl_nested_modules", nested_modules_model, x, tl)
            graphs.append(
                TestGraph(
                    name="tl_nested_modules",
                    graph=g,
                    tags={"nested-deep"},
                    source="torchlens",
                    description="TorchLens: 4-level nested module hierarchy",
                    expected_challenges="Deep module nesting, cluster layout",
                )
            )
        except Exception:
            pass

    # 6. Branching (3-way split and merge)
    if simple_branching_cls is not None:
        try:
            branching_model = simple_branching_cls()
            x = torch.randn(5)
            g = _trace_torchlens_graph("tl_branching", branching_model, x, tl)
            graphs.append(
                TestGraph(
                    name="tl_branching",
                    graph=g,
                    tags={"wide-parallel"},
                    source="torchlens",
                    description="TorchLens: 3-way branching with merge",
                    expected_challenges="Branch alignment, merge point",
                )
            )
        except Exception:
            pass

    # 7. Diamond loop (split → sin/cos → merge, repeated)
    if diamond_loop_cls is not None:
        try:
            diamond_loop_model = diamond_loop_cls()
            x = torch.randn(5)
            g = _trace_torchlens_graph("tl_diamond_loop", diamond_loop_model, x, tl)
            graphs.append(
                TestGraph(
                    name="tl_diamond_loop",
                    graph=g,
                    tags={"diamond", "skip-light"},
                    source="torchlens",
                    description="TorchLens: Diamond pattern with loop iterations",
                    expected_challenges="Repeated diamond pattern, loop detection",
                )
            )
        except Exception:
            pass

    # 8. Long loop (20 iterations of Linear + ReLU)
    if long_loop_cls is not None:
        try:
            long_loop_model = long_loop_cls()
            x = torch.randn(5)
            g = _trace_torchlens_graph("tl_long_loop", long_loop_model, x, tl)
            graphs.append(
                TestGraph(
                    name="tl_long_loop",
                    graph=g,
                    tags={"linear-deep"},
                    source="torchlens",
                    description="TorchLens: 20-iteration loop (Linear+ReLU)",
                    expected_challenges="Very deep chain from loop unrolling",
                )
            )
        except Exception:
            pass

    # 9. ASPP model (multi-scale parallel branches)
    try:
        from tests.example_models import ASPPModel

        aspp_model = ASPPModel(in_channels=3, mid=8, rates=(1, 6, 12))
        x = torch.randn(2, 3, 32, 32)
        g = _trace_torchlens_graph("tl_aspp", aspp_model, x, tl)
        graphs.append(
            TestGraph(
                name="tl_aspp",
                graph=g,
                tags={"wide-parallel", "nested-shallow"},
                source="torchlens",
                description="TorchLens: Atrous Spatial Pyramid Pooling (4 parallel branches)",
                expected_challenges="Multi-scale parallel branches, wide layout",
            )
        )
    except Exception:
        pass

    # 10. Feature Pyramid Network (bidirectional multi-scale)
    try:
        from tests.example_models import FeaturePyramidNet

        fpn_model = FeaturePyramidNet()
        x = torch.randn(2, 3, 32, 32)
        g = _trace_torchlens_graph("tl_fpn", fpn_model, x, tl)
        graphs.append(
            TestGraph(
                name="tl_fpn",
                graph=g,
                tags={"diamond", "skip-light", "nested-shallow"},
                source="torchlens",
                description="TorchLens: Feature Pyramid Network with lateral connections",
                expected_challenges="Bidirectional flow, lateral skip connections",
            )
        )
    except Exception:
        pass

    # 11. Attention block (Q/K/V self-attention)
    try:
        from tests.example_models import _AttentionBlock

        attention_model = _AttentionBlock(dim=64)
        x = torch.randn(2, 10, 64)
        g = _trace_torchlens_graph("tl_attention", attention_model, x, tl)
        graphs.append(
            TestGraph(
                name="tl_attention",
                graph=g,
                tags={"wide-parallel", "nested-shallow"},
                source="torchlens",
                description="TorchLens: Self-attention with Q/K/V projections",
                expected_challenges="Triple parallel branches, matmul merge",
            )
        )
    except Exception:
        pass

    # 12. RandomGraphModel (controllable stress test)
    try:
        from tests.example_models import RandomGraphModel

        random_graph_model = RandomGraphModel(
            target_nodes=50, nesting_depth=2, seed=42, branch_probability=0.3, hidden_dim=64
        )
        x = torch.randn(2, 64)
        g = _trace_torchlens_graph("tl_random_50", random_graph_model, x, tl)
        graphs.append(
            TestGraph(
                name="tl_random_50",
                graph=g,
                tags={"nested-deep", "skip-light", "wide-parallel"},
                source="torchlens",
                description="TorchLens: Random architecture (~50 nodes, mixed patterns)",
                expected_challenges="Unpredictable topology, mixed branch/skip/nest",
            )
        )
    except Exception:
        pass

    return graphs


# ─── Scale Graph Generators ──────────────────────────────────────────────────


def make_chain(n: int, seed: int = 42) -> TestGraph:
    """Build a linear chain graph.

    Parameters
    ----------
    n : int
        Number of nodes in the chain.
    seed : int, default=42
        Unused seed parameter retained for scale-suite API consistency.

    Returns
    -------
    TestGraph
        Linear chain ``0 -> 1 -> ... -> n - 1``.
    """
    del seed
    src = list(range(n - 1))
    tgt = list(range(1, n))
    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    g = DaguaGraph.from_edge_index(edge_index, num_nodes=n)
    return TestGraph(
        name=f"chain_{n}",
        graph=g,
        tags={"linear-deep", "scale"},
        description=f"Linear chain with {n} nodes",
    )


def make_wide_dag(n: int, width: int = 0, seed: int = 42) -> TestGraph:
    """Build a layered DAG with fixed width per layer.

    Parameters
    ----------
    n : int
        Number of nodes to materialize.
    width : int, default=0
        Desired layer width. A non-positive value derives width from ``n``.
    seed : int, default=42
        Random seed for cross-layer target sampling.

    Returns
    -------
    TestGraph
        Layered scale graph with deterministic sorted edge order.
    """
    import random

    rng = random.Random(seed)
    if width <= 0:
        width = max(int(n**0.5), 5)

    n_layers = max(n // width, 2)
    layers: List[List[int]] = []
    node_idx = 0
    for _ in range(n_layers):
        layer_size = min(width, n - node_idx)
        if layer_size <= 0:
            break
        layers.append(list(range(node_idx, node_idx + layer_size)))
        node_idx += layer_size
    if node_idx < n:
        layers[-1].extend(range(node_idx, n))

    edges_set: set[tuple[int, int]] = set()
    for i in range(len(layers) - 1):
        for node in layers[i]:
            k = min(rng.randint(1, 3), len(layers[i + 1]))
            targets = rng.sample(layers[i + 1], k)
            for t in targets:
                edges_set.add((node, t))

    g = DaguaGraph.from_edge_index(torch.zeros(2, 0, dtype=torch.long), num_nodes=n)
    if edges_set:
        el = sorted(edges_set)
        g.edge_index = torch.tensor([[e[0] for e in el], [e[1] for e in el]], dtype=torch.long)
        g.edge_labels = [None] * len(el)
        g.edge_types = ["normal"] * len(el)
        g.edge_styles = [None] * len(el)

    return TestGraph(
        name=f"wide_dag_{n}",
        graph=g,
        tags={"wide-parallel", "scale"},
        description=f"Layered DAG: {n} nodes, width ~{width}",
    )


def make_random_dag(n: int, density: float = 1.5, seed: int = 42) -> TestGraph:
    """Build a random DAG with roughly ``n * density`` edges.

    Parameters
    ----------
    n : int
        Number of nodes to materialize.
    density : float, default=1.5
        Edge multiplier used to choose the target edge count.
    seed : int, default=42
        Random seed for reproducible edge sampling.

    Returns
    -------
    TestGraph
        Random scale DAG with deterministic sorted edge order.
    """
    import random

    rng = random.Random(seed)
    n_edges = int(n * density)

    g = DaguaGraph.from_edge_index(torch.zeros(2, 0, dtype=torch.long), num_nodes=n)
    edges_set: set[tuple[int, int]] = set()
    attempts = 0
    while len(edges_set) < n_edges and attempts < n_edges * 20:
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, min(i + max(n // 5, 10), n - 1))
        edges_set.add((i, j))
        attempts += 1

    if edges_set:
        el = sorted(edges_set)
        g.edge_index = torch.tensor([[e[0] for e in el], [e[1] for e in el]], dtype=torch.long)
        g.edge_labels = [None] * len(el)
        g.edge_types = ["normal"] * len(el)
        g.edge_styles = [None] * len(el)

    return TestGraph(
        name=f"random_dag_{n}",
        graph=g,
        tags={"large-sparse", "scale"},
        description=f"Random DAG: {n} nodes, ~{n_edges} edges",
    )


def make_diamond(n: int, seed: int = 42) -> TestGraph:
    """Diamond/hourglass: fan-out then fan-in."""
    mid = n // 2
    src, tgt = [], []
    # fan-out: node 0 → nodes 1..mid
    for i in range(1, min(mid + 1, n)):
        src.append(0)
        tgt.append(i)
    # fan-in: nodes 1..mid → node n-1
    if n > 2:
        for i in range(1, min(mid + 1, n - 1)):
            src.append(i)
            tgt.append(n - 1)
    # chain the middle section for larger n
    for i in range(mid + 1, n - 1):
        src.append(i - 1)
        tgt.append(i)

    if not src:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.tensor([src, tgt], dtype=torch.long)
    g = DaguaGraph.from_edge_index(edge_index, num_nodes=n)
    return TestGraph(
        name=f"diamond_{n}",
        graph=g,
        tags={"diamond", "scale"},
        description=f"Diamond/hourglass: {n} nodes",
    )


def make_tree(n: int, branching: int = 3, seed: int = 42) -> TestGraph:
    """Balanced tree with given branching factor."""
    src, tgt = [], []
    for i in range(n):
        for b in range(branching):
            child = i * branching + b + 1
            if child < n:
                src.append(i)
                tgt.append(child)

    if not src:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.tensor([src, tgt], dtype=torch.long)
    g = DaguaGraph.from_edge_index(edge_index, num_nodes=n)
    return TestGraph(
        name=f"tree_{n}_b{branching}",
        graph=g,
        tags={"tree", "scale"},
        description=f"Balanced tree: {n} nodes, branching={branching}",
    )


def make_bipartite(n: int, seed: int = 42, max_degree: int = 8) -> TestGraph:
    """Bipartite DAG: sources → middle → sinks in 3 layers.

    For large n, uses sparse random connections (each node connects to at most
    max_degree nodes in the next layer) to avoid O(n²) edge blowup.
    """
    import random

    rng = random.Random(seed)
    third = max(n // 3, 1)
    n_src = third
    n_mid = third
    n_sink = n - n_src - n_mid

    src, tgt = [], []
    # source → mid (sparse for large graphs)
    for i in range(n_src):
        k = min(max_degree, n_mid)
        targets = rng.sample(range(n_src, n_src + n_mid), k)
        for j in targets:
            src.append(i)
            tgt.append(j)
    # mid → sink (sparse for large graphs)
    for j in range(n_src, n_src + n_mid):
        k = min(max_degree, n_sink)
        targets = rng.sample(range(n_src + n_mid, n), k)
        for t in targets:
            src.append(j)
            tgt.append(t)

    if not src:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.tensor([src, tgt], dtype=torch.long)
    g = DaguaGraph.from_edge_index(edge_index, num_nodes=n)
    return TestGraph(
        name=f"bipartite_{n}",
        graph=g,
        tags={"wide-parallel", "large-dense", "scale"},
        description=f"Bipartite 3-layer DAG: {n} nodes",
    )


def _make_scale_grid_test_graph(n: int, seed: int = 42) -> TestGraph:
    """Build the existing scale-suite square-grid test graph.

    Parameters
    ----------
    n : int
        Approximate requested node count.
    seed : int, default=42
        Unused compatibility parameter retained for parity with other scale
        generators.

    Returns
    -------
    TestGraph
        Scale-suite grid benchmark entry.
    """
    del seed
    cols = max(int(math.sqrt(max(n, 1))), 2)
    rows = max(max(n, 1) // cols, 2)
    actual_n = rows * cols

    src: List[int] = []
    tgt: List[int] = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if c + 1 < cols:
                src.append(idx)
                tgt.append(idx + 1)
            if r + 1 < rows:
                src.append(idx)
                tgt.append(idx + cols)

    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    g = DaguaGraph.from_edge_index(edge_index, num_nodes=actual_n)
    return TestGraph(
        name=f"grid_{actual_n}",
        graph=g,
        tags={"diamond", "scale"},
        description=f"Grid DAG: {rows}×{cols} = {actual_n} nodes",
    )


@overload
def make_grid(rows: int, seed: int = 42) -> TestGraph: ...


@overload
def make_grid(rows: int, cols: int, seed: int = 42) -> DaguaGraph: ...


def make_grid(
    rows: int, cols: Optional[int] = None, seed: int = 42
) -> Union[TestGraph, DaguaGraph]:
    """Build either the scale-suite grid or the structural rectangular grid.

    Parameters
    ----------
    rows : int
        Either the approximate node count for the scale-suite form or the row
        count for the structural rectangular grid.
    cols : int, optional
        Column count for the structural rectangular grid. When omitted, the
        legacy scale-suite behavior is preserved.
    seed : int, default=42
        Random seed kept for a consistent generator signature.

    Returns
    -------
    Union[TestGraph, DaguaGraph]
        ``TestGraph`` when called as ``make_grid(n, seed=...)`` for the scale
        suite, otherwise a rectangular ``DaguaGraph`` when ``cols`` is
        provided.
    """
    if cols is None:
        return _make_scale_grid_test_graph(rows, seed=seed)
    return _make_rectangular_grid_graph(rows=rows, cols=cols, seed=seed)


def make_sparse_layered(
    n: int, n_layers: int = 0, edges_per_node: int = 3, seed: int = 42
) -> TestGraph:
    """Layered DAG with controlled sparsity and occasional skip connections.

    Each node connects to `edges_per_node` random nodes in the next layer,
    plus ~10% skip connections that jump 2 layers ahead.
    """
    import random

    rng = random.Random(seed)
    if n_layers <= 0:
        n_layers = max(int(n**0.4), 4)

    # Distribute nodes across layers
    base_size = n // n_layers
    remainder = n % n_layers
    layer_offsets = []
    offset = 0
    for i in range(n_layers):
        size = base_size + (1 if i < remainder else 0)
        layer_offsets.append((offset, offset + size))
        offset += size
    actual_n = offset

    src, tgt = [], []
    for i in range(n_layers - 1):
        lo_s, hi_s = layer_offsets[i]
        lo_t, hi_t = layer_offsets[i + 1]
        layer_size_t = hi_t - lo_t
        if layer_size_t == 0:
            continue
        for node in range(lo_s, hi_s):
            k = min(edges_per_node, layer_size_t)
            targets = rng.sample(range(lo_t, hi_t), k)
            for t in targets:
                src.append(node)
                tgt.append(t)
            # ~10% chance of skip connection (jump 2 layers)
            if i + 2 < n_layers and rng.random() < 0.1:
                lo_skip, hi_skip = layer_offsets[i + 2]
                if hi_skip > lo_skip:
                    src.append(node)
                    tgt.append(rng.randint(lo_skip, hi_skip - 1))

    edge_index = (
        torch.tensor([src, tgt], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    )
    g = DaguaGraph.from_edge_index(edge_index, num_nodes=actual_n)
    return TestGraph(
        name=f"sparse_layered_{actual_n}",
        graph=g,
        tags={"large-sparse", "skip-light", "scale"},
        description=(
            f"Layered DAG: {actual_n} nodes, {n_layers} layers, ~{edges_per_node} edges/node"
        ),
    )


def make_powerlaw_dag(n: int, seed: int = 42) -> TestGraph:
    """DAG with power-law out-degree distribution (realistic network topology).

    Uses preferential attachment: later nodes attach to earlier nodes with
    probability proportional to their current in-degree + 1.
    """
    import random

    rng = random.Random(seed)
    m = 2  # edges per new node

    src, tgt = [], []
    in_degree = [0] * n
    # Weighted pool for preferential attachment (repeat node id by degree+1)
    pool = [0]

    for i in range(1, n):
        targets = set()
        for _ in range(min(m, i)):
            attempts = 0
            while attempts < 50:
                t = pool[rng.randint(0, len(pool) - 1)]
                if t < i and t not in targets:
                    targets.add(t)
                    break
                attempts += 1
            else:
                # fallback: pick any earlier node
                t = rng.randint(0, i - 1)
                targets.add(t)
        for t in targets:
            # Edge direction: earlier → later (maintains DAG property)
            src.append(t)
            tgt.append(i)
            in_degree[i] += 1
            pool.append(i)
        pool.append(i)  # base weight of 1

    edge_index = (
        torch.tensor([src, tgt], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    )
    g = DaguaGraph.from_edge_index(edge_index, num_nodes=n)
    return TestGraph(
        name=f"powerlaw_dag_{n}",
        graph=g,
        tags={"large-sparse", "scale"},
        description=f"Power-law DAG: {n} nodes, preferential attachment",
    )


def make_scale_free(n: int = 120, m: int = 3, seed: int = 42) -> DaguaGraph:
    """Build a preferential-attachment DAG with hub-heavy degree skew.

    Parameters
    ----------
    n : int, default=120
        Number of nodes.
    m : int, default=3
        Maximum number of incoming attachments per new node.
    seed : int, default=42
        Random seed for reproducible attachment choices.

    Returns
    -------
    DaguaGraph
        Scale-free DAG whose earlier nodes attract more later edges. This
        stresses hub placement and edge bundling around high-degree nodes.
    """
    if n <= 0:
        return _empty_generated_graph()

    rng = random.Random(seed)
    node_ids = [f"sf_{idx}" for idx in range(n)]
    edge_list: List[Tuple[str, str]] = []
    weighted_pool: List[int] = [0]
    attachment_count = max(m, 0)

    for node_idx in range(1, n):
        target_count = min(attachment_count, node_idx)
        targets: Set[int] = set()
        while len(targets) < target_count and weighted_pool:
            targets.add(weighted_pool[rng.randrange(len(weighted_pool))])
            if len(targets) == node_idx:
                break
        while len(targets) < target_count:
            targets.add(rng.randrange(node_idx))

        for target_idx in sorted(targets):
            edge_list.append((node_ids[target_idx], node_ids[node_idx]))
            weighted_pool.append(target_idx)
        weighted_pool.extend([node_idx] * (len(targets) + 1))

    return _build_named_graph(node_ids, edge_list)


def _make_rectangular_grid_graph(rows: int, cols: int, seed: int = 42) -> DaguaGraph:
    """Build a rectangular grid DAG with rightward and downward flow.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    seed : int, default=42
        Unused seed parameter retained for API consistency.

    Returns
    -------
    DaguaGraph
        Rectangular grid DAG for testing regular layered structure and local
        crossing pressure.
    """
    del seed
    if rows <= 0 or cols <= 0:
        return _empty_generated_graph()

    node_ids = [f"grid_{r}_{c}" for r in range(rows) for c in range(cols)]
    edge_list: List[Tuple[str, str]] = []
    for r in range(rows):
        for c in range(cols):
            node_id = f"grid_{r}_{c}"
            if c + 1 < cols:
                edge_list.append((node_id, f"grid_{r}_{c + 1}"))
            if r + 1 < rows:
                edge_list.append((node_id, f"grid_{r + 1}_{c}"))
    return _build_named_graph(node_ids, edge_list)


def make_complete_bipartite(a: int = 8, b: int = 12, seed: int = 42) -> DaguaGraph:
    """Build a complete bipartite DAG with left-to-right connectivity.

    Parameters
    ----------
    a : int, default=8
        Node count in the left partition.
    b : int, default=12
        Node count in the right partition.
    seed : int, default=42
        Unused seed parameter retained for API consistency.

    Returns
    -------
    DaguaGraph
        Complete bipartite graph ``K(a, b)`` used to test ordering and
        crossing minimization between two dense layers.
    """
    del seed
    if a <= 0 and b <= 0:
        return _empty_generated_graph()

    left_nodes = [f"left_{idx}" for idx in range(max(a, 0))]
    right_nodes = [f"right_{idx}" for idx in range(max(b, 0))]
    edge_list = [(src, tgt) for src in left_nodes for tgt in right_nodes]
    return _build_named_graph(left_nodes + right_nodes, edge_list)


def make_clustered_medium(
    n_clusters: int = 5,
    nodes_per_cluster: int = 20,
    inter_density: float = 0.05,
    seed: int = 42,
) -> DaguaGraph:
    """Build a medium-sized clustered DAG with sparse cross-cluster handoffs.

    Parameters
    ----------
    n_clusters : int, default=5
        Number of clusters.
    nodes_per_cluster : int, default=20
        Nodes per cluster.
    inter_density : float, default=0.05
        Probability of adding an edge between adjacent clusters.
    seed : int, default=42
        Random seed for reproducible internal and external edges.

    Returns
    -------
    DaguaGraph
        Clustered DAG that tests cluster separation, inter-cluster routing,
        and medium-scale readability.
    """
    if n_clusters <= 0 or nodes_per_cluster <= 0:
        return _empty_generated_graph()

    rng = random.Random(seed)
    graph = DaguaGraph()
    cluster_nodes: List[List[str]] = []
    bridge_density = max(0.0, min(inter_density, 1.0))

    for cluster_idx in range(n_clusters):
        members = [
            f"cluster_{cluster_idx}.node_{node_idx}" for node_idx in range(nodes_per_cluster)
        ]
        cluster_nodes.append(members)
        for member in members:
            graph.add_node(member, label=member)
        for node_idx in range(nodes_per_cluster - 1):
            graph.add_edge(members[node_idx], members[node_idx + 1])
            if node_idx + 2 < nodes_per_cluster and rng.random() < 0.3:
                graph.add_edge(members[node_idx], members[node_idx + 2])

    for cluster_idx in range(n_clusters - 1):
        src_members = cluster_nodes[cluster_idx]
        tgt_members = cluster_nodes[cluster_idx + 1]
        for src in src_members:
            for tgt in tgt_members:
                if rng.random() < bridge_density:
                    graph.add_edge(src, tgt)

    for cluster_idx, members in enumerate(cluster_nodes):
        graph.add_cluster(
            f"cluster_{cluster_idx}",
            members,
            label=f"Cluster {cluster_idx}",
        )

    return _finalize_generated_graph(graph)


def make_hub_and_spoke(
    n_hubs: int = 3,
    spokes_per_hub: int = 20,
    seed: int = 42,
) -> DaguaGraph:
    """Build a hub-and-spoke DAG with dominant high-degree centers.

    Parameters
    ----------
    n_hubs : int, default=3
        Number of central hubs.
    spokes_per_hub : int, default=20
        Number of spoke nodes attached to each hub.
    seed : int, default=42
        Unused seed parameter retained for API consistency.

    Returns
    -------
    DaguaGraph
        Hub-heavy DAG that stresses placement around dominant fan-out and
        fan-in nodes.
    """
    del seed
    if n_hubs <= 0:
        return _empty_generated_graph()

    graph = DaguaGraph()
    graph.add_node("entry", label="entry")
    graph.add_node("exit", label="exit")
    for hub_idx in range(n_hubs):
        hub_id = f"hub_{hub_idx}"
        graph.add_node(hub_id, label=hub_id)
        graph.add_edge("entry", hub_id)
        if hub_idx + 1 < n_hubs:
            graph.add_edge(hub_id, f"hub_{hub_idx + 1}")
        for spoke_idx in range(max(spokes_per_hub, 0)):
            spoke_id = f"{hub_id}.spoke_{spoke_idx}"
            graph.add_node(spoke_id, label=spoke_id)
            graph.add_edge(hub_id, spoke_id)
            graph.add_edge(spoke_id, "exit")
    return _finalize_generated_graph(graph)


def make_wide_single_layer(
    n_sources: int = 1,
    n_wide: int = 50,
    n_sinks: int = 1,
    seed: int = 42,
) -> DaguaGraph:
    """Build a DAG with a very wide middle layer.

    Parameters
    ----------
    n_sources : int, default=1
        Number of source nodes.
    n_wide : int, default=50
        Number of nodes in the wide middle layer.
    n_sinks : int, default=1
        Number of sink nodes.
    seed : int, default=42
        Unused seed parameter retained for API consistency.

    Returns
    -------
    DaguaGraph
        Three-layer DAG with a very wide middle that stresses horizontal
        spacing, ordering, and symmetric fan-in/fan-out.
    """
    del seed
    if n_sources <= 0 and n_wide <= 0 and n_sinks <= 0:
        return _empty_generated_graph()

    source_nodes = [f"src_{idx}" for idx in range(max(n_sources, 0))]
    wide_nodes = [f"mid_{idx}" for idx in range(max(n_wide, 0))]
    sink_nodes = [f"sink_{idx}" for idx in range(max(n_sinks, 0))]
    edge_list = [(src, mid) for src in source_nodes for mid in wide_nodes]
    edge_list.extend((mid, sink) for mid in wide_nodes for sink in sink_nodes)
    return _build_named_graph(source_nodes + wide_nodes + sink_nodes, edge_list)


def make_sparse_dense_pair(n: int = 50, seed: int = 42) -> Tuple[DaguaGraph, DaguaGraph]:
    """Build paired sparse and dense DAGs over the same node set.

    Parameters
    ----------
    n : int, default=50
        Shared node count for both graphs.
    seed : int, default=42
        Random seed for the dense variant.

    Returns
    -------
    Tuple[DaguaGraph, DaguaGraph]
        Sparse then dense graphs sharing identical node labels. This pair tests
        how layout quality changes under edge-density growth alone.
    """
    if n <= 0:
        empty = _empty_generated_graph()
        return empty, _empty_generated_graph()

    rng = random.Random(seed)
    node_ids = [f"pair_{idx}" for idx in range(n)]

    sparse_edges: List[Tuple[str, str]] = []
    for node_idx in range(n - 1):
        sparse_edges.append((node_ids[node_idx], node_ids[node_idx + 1]))
        if node_idx + 3 < n and node_idx % 4 == 0:
            sparse_edges.append((node_ids[node_idx], node_ids[node_idx + 3]))

    dense_edges: List[Tuple[str, str]] = list(sparse_edges)
    for src_idx in range(n - 1):
        for tgt_idx in range(src_idx + 2, min(src_idx + 9, n)):
            if rng.random() < 0.45:
                dense_edges.append((node_ids[src_idx], node_ids[tgt_idx]))

    return _build_named_graph(node_ids, sparse_edges), _build_named_graph(node_ids, dense_edges)


def make_compound_dag(
    n_clusters: int = 5,
    n_per_cluster: int = 30,
    seed: int = 42,
) -> DaguaGraph:
    """Build a clustered DAG whose clusters themselves form a DAG.

    Parameters
    ----------
    n_clusters : int, default=5
        Number of top-level clusters.
    n_per_cluster : int, default=30
        Nodes per cluster.
    seed : int, default=42
        Random seed for intra-cluster shortcuts and inter-cluster skips.

    Returns
    -------
    DaguaGraph
        Compound DAG with explicit clusters arranged in topological order. It
        stresses cluster-aware routing and higher-level DAG flow.
    """
    if n_clusters <= 0 or n_per_cluster <= 0:
        return _empty_generated_graph()

    rng = random.Random(seed)
    graph = DaguaGraph()
    cluster_nodes: List[List[str]] = []

    for cluster_idx in range(n_clusters):
        members = [f"stage_{cluster_idx}.node_{node_idx}" for node_idx in range(n_per_cluster)]
        cluster_nodes.append(members)
        for member in members:
            graph.add_node(member, label=member)
        for node_idx in range(n_per_cluster - 1):
            graph.add_edge(members[node_idx], members[node_idx + 1])
            if node_idx + 3 < n_per_cluster and rng.random() < 0.2:
                graph.add_edge(members[node_idx], members[node_idx + 3])

    for cluster_idx in range(n_clusters - 1):
        src_members = cluster_nodes[cluster_idx][-3:]
        tgt_members = cluster_nodes[cluster_idx + 1][:3]
        for src in src_members:
            for tgt in tgt_members:
                graph.add_edge(src, tgt)
        if cluster_idx + 2 < n_clusters and rng.random() < 0.5:
            graph.add_edge(cluster_nodes[cluster_idx][-1], cluster_nodes[cluster_idx + 2][0])

    for cluster_idx, members in enumerate(cluster_nodes):
        graph.add_cluster(
            f"compound_stage_{cluster_idx}",
            members,
            label=f"Stage {cluster_idx}",
        )

    return _finalize_generated_graph(graph)


def make_long_skip_only(n: int = 20, seed: int = 42) -> DaguaGraph:
    """Build a DAG where every edge jumps at least three layers.

    Parameters
    ----------
    n : int, default=20
        Node count.
    seed : int, default=42
        Unused seed parameter retained for API consistency.

    Returns
    -------
    DaguaGraph
        Skip-heavy DAG with no local edges, stressing long-edge routing and
        layer assignment under pathological span patterns.
    """
    del seed
    if n <= 0:
        return _empty_generated_graph()

    node_ids = [f"skip_{idx}" for idx in range(n)]
    edge_list: List[Tuple[str, str]] = []
    for node_idx in range(n):
        for jump in (3, 4):
            if node_idx + jump < n:
                edge_list.append((node_ids[node_idx], node_ids[node_idx + jump]))
    return _build_named_graph(node_ids, edge_list)


def make_parallel_cycles(
    n_cycles: int = 3,
    cycle_length: int = 5,
    seed: int = 42,
) -> DaguaGraph:
    """Build multiple disconnected directed cycles.

    Parameters
    ----------
    n_cycles : int, default=3
        Number of independent cycles.
    cycle_length : int, default=5
        Nodes per cycle.
    seed : int, default=42
        Unused seed parameter retained for API consistency.

    Returns
    -------
    DaguaGraph
        Disconnected cyclic graph used to test cycle handling and component
        packing.
    """
    del seed
    if n_cycles <= 0 or cycle_length <= 0:
        return _empty_generated_graph()

    node_ids: List[str] = []
    edge_list: List[Tuple[str, str]] = []
    for cycle_idx in range(n_cycles):
        cycle_nodes = [f"cycle_{cycle_idx}.node_{idx}" for idx in range(cycle_length)]
        node_ids.extend(cycle_nodes)
        if cycle_length == 1:
            edge_list.append((cycle_nodes[0], cycle_nodes[0]))
            continue
        for node_idx, node_id in enumerate(cycle_nodes):
            edge_list.append((node_id, cycle_nodes[(node_idx + 1) % cycle_length]))
    return _build_named_graph(node_ids, edge_list)


def make_resnet_block(n_blocks: int = 4, width: int = 16, seed: int = 42) -> DaguaGraph:
    """Build a simple ResNet-style stack of residual blocks.

    Parameters
    ----------
    n_blocks : int, default=4
        Number of residual blocks.
    width : int, default=16
        Channel width encoded into node labels.
    seed : int, default=42
        Unused seed parameter retained for API consistency.

    Returns
    -------
    DaguaGraph
        Residual network DAG that stresses skip routing, repeated motifs, and
        block-level clustering.
    """
    del seed
    if n_blocks <= 0:
        return _build_named_graph([f"input_{width}ch"], [])

    graph = DaguaGraph()
    current = f"input_{width}ch"
    graph.add_node(current, label=current)

    for block_idx in range(n_blocks):
        block_nodes = [
            f"block_{block_idx}.conv1_{width}ch",
            f"block_{block_idx}.bn1",
            f"block_{block_idx}.relu1",
            f"block_{block_idx}.conv2_{width}ch",
            f"block_{block_idx}.bn2",
            f"block_{block_idx}.add",
            f"block_{block_idx}.relu2",
        ]
        for node_id in block_nodes:
            graph.add_node(node_id, label=node_id)
        graph.add_edge(current, block_nodes[0])
        for src, tgt in zip(block_nodes, block_nodes[1:]):
            graph.add_edge(src, tgt)
        graph.add_edge(current, block_nodes[5])
        graph.add_cluster(
            f"resnet_block_{block_idx}",
            block_nodes,
            label=f"Residual Block {block_idx}",
        )
        current = block_nodes[-1]

    output = f"output_{width}ch"
    graph.add_node(output, label=output)
    graph.add_edge(current, output)
    return _finalize_generated_graph(graph)


def make_transformer_full(
    n_heads: int = 4,
    n_layers: int = 2,
    seed: int = 42,
) -> DaguaGraph:
    """Build a multi-layer transformer-style attention stack.

    Parameters
    ----------
    n_heads : int, default=4
        Attention heads per layer.
    n_layers : int, default=2
        Transformer layers.
    seed : int, default=42
        Unused seed parameter retained for API consistency.

    Returns
    -------
    DaguaGraph
        Transformer-style DAG with per-head parallel branches, residual adds,
        and layer-level clusters.
    """
    del seed
    if n_layers <= 0:
        return _build_named_graph(["transformer_input"], [])

    graph = DaguaGraph()
    current = "transformer_input"
    graph.add_node(current, label=current)
    head_count = max(n_heads, 1)

    for layer_idx in range(n_layers):
        norm1 = f"layer_{layer_idx}.norm1"
        concat = f"layer_{layer_idx}.concat"
        proj = f"layer_{layer_idx}.attn_out"
        add1 = f"layer_{layer_idx}.add1"
        norm2 = f"layer_{layer_idx}.norm2"
        ffn1 = f"layer_{layer_idx}.ffn1"
        ffn2 = f"layer_{layer_idx}.ffn2"
        add2 = f"layer_{layer_idx}.add2"
        layer_nodes = [norm1, concat, proj, add1, norm2, ffn1, ffn2, add2]
        for node_id in layer_nodes:
            graph.add_node(node_id, label=node_id)

        head_nodes: List[str] = []
        graph.add_edge(current, norm1)
        for head_idx in range(head_count):
            head_id = f"layer_{layer_idx}.head_{head_idx}.attn"
            head_nodes.append(head_id)
            graph.add_node(head_id, label=head_id)
            graph.add_edge(norm1, head_id)
            graph.add_edge(head_id, concat)

        graph.add_edge(concat, proj)
        graph.add_edge(proj, add1)
        graph.add_edge(current, add1)
        graph.add_edge(add1, norm2)
        graph.add_edge(norm2, ffn1)
        graph.add_edge(ffn1, ffn2)
        graph.add_edge(ffn2, add2)
        graph.add_edge(add1, add2)

        graph.add_cluster(
            f"transformer_layer_{layer_idx}",
            layer_nodes + head_nodes,
            label=f"Transformer Layer {layer_idx}",
        )
        graph.add_cluster(
            f"transformer_layer_{layer_idx}.attention",
            head_nodes + [concat, proj],
            label=f"Attention {layer_idx}",
            parent=f"transformer_layer_{layer_idx}",
        )

        current = add2

    output = "transformer_output"
    graph.add_node(output, label=output)
    graph.add_edge(current, output)
    return _finalize_generated_graph(graph)


def make_dependency_graph(n: int = 100, n_core: int = 5, seed: int = 42) -> DaguaGraph:
    """Build a dependency DAG with a small set of dominant core libraries.

    Parameters
    ----------
    n : int, default=100
        Total node count.
    n_core : int, default=5
        Number of core libraries that attract many dependents.
    seed : int, default=42
        Random seed for preferential dependency assignment.

    Returns
    -------
    DaguaGraph
        Dependency-style DAG with core packages near the sources and a
        power-law fan-out into application packages.
    """
    if n <= 0:
        return _empty_generated_graph()

    rng = random.Random(seed)
    core_count = max(0, min(n_core, n))
    node_ids = [f"core_{idx}" for idx in range(core_count)] + [
        f"pkg_{idx}" for idx in range(max(n - core_count, 0))
    ]
    edge_list: List[Tuple[str, str]] = []

    weighted_sources: List[int] = list(range(core_count)) or [0]
    for node_idx in range(core_count, n):
        dep_candidates: Set[int] = set()
        target_count = min(3, node_idx)
        while len(dep_candidates) < target_count:
            dep_candidates.add(weighted_sources[rng.randrange(len(weighted_sources))] % node_idx)
        for dep_idx in sorted(dep_candidates):
            edge_list.append((node_ids[dep_idx], node_ids[node_idx]))
            weighted_sources.append(dep_idx)
        weighted_sources.extend([node_idx] * (len(dep_candidates) + 1))

    graph = _build_named_graph(node_ids, edge_list)
    if core_count > 0:
        graph.add_cluster(
            "dependency_core",
            [f"core_{idx}" for idx in range(core_count)],
            label="Core Libraries",
        )
    return _finalize_generated_graph(graph)


def make_org_chart(
    levels: int = 4,
    branching: Tuple[int, ...] = (1, 5, 4, 8),
    seed: int = 42,
) -> DaguaGraph:
    """Build a non-uniform tree resembling an organizational chart.

    Parameters
    ----------
    levels : int, default=4
        Number of levels to materialize.
    branching : Tuple[int, ...], default=(1, 5, 4, 8)
        Interpreted as desired node counts per level. When shorter than
        ``levels``, the last value is repeated.
    seed : int, default=42
        Unused seed parameter retained for API consistency.

    Returns
    -------
    DaguaGraph
        Non-uniform tree used to test breadth variation between levels and
        parent-child spacing in organizational layouts.
    """
    del seed
    if levels <= 0:
        return _empty_generated_graph()

    if not branching:
        branching = (1,)
    level_sizes = [branching[min(idx, len(branching) - 1)] for idx in range(levels)]
    level_sizes[0] = max(level_sizes[0], 1)

    node_ids: List[str] = []
    level_nodes: List[List[str]] = []
    for level_idx, level_size in enumerate(level_sizes):
        size = max(level_size, 0)
        nodes = [f"level_{level_idx}.person_{node_idx}" for node_idx in range(size)]
        level_nodes.append(nodes)
        node_ids.extend(nodes)

    edge_list: List[Tuple[str, str]] = []
    for level_idx in range(len(level_nodes) - 1):
        parents = level_nodes[level_idx]
        children = level_nodes[level_idx + 1]
        if not parents or not children:
            continue
        for child_idx, child in enumerate(children):
            parent = parents[child_idx % len(parents)]
            edge_list.append((parent, child))

    return _build_named_graph(node_ids, edge_list)


def make_small_world(n: int = 100, k: int = 4, p: float = 0.1, seed: int = 42) -> DaguaGraph:
    """Build a directed small-world graph using Watts-Strogatz-style rewiring.

    Parameters
    ----------
    n : int, default=100
        Node count.
    k : int, default=4
        Each node connects to ``k / 2`` forward neighbors on the ring.
    p : float, default=0.1
        Rewiring probability for each edge.
    seed : int, default=42
        Random seed for rewiring.

    Returns
    -------
    DaguaGraph
        Directed small-world graph that stresses cyclic structure, short paths,
        and non-layered global geometry.
    """
    if n <= 0:
        return _empty_generated_graph()

    rng = random.Random(seed)
    neighbor_span = max(k // 2, 0)
    node_ids = [f"sw_{idx}" for idx in range(n)]
    edge_set: Set[Tuple[str, str]] = set()
    rewire_prob = max(0.0, min(p, 1.0))

    for src_idx in range(n):
        for offset in range(1, neighbor_span + 1):
            tgt_idx = (src_idx + offset) % n
            if rng.random() < rewire_prob and n > 1:
                candidate_idx = tgt_idx
                while candidate_idx == src_idx:
                    candidate_idx = rng.randrange(n)
                tgt_idx = candidate_idx
            edge_set.add((node_ids[src_idx], node_ids[tgt_idx]))

    return _build_named_graph(node_ids, sorted(edge_set))


def _graph_from_integer_edges(
    num_nodes: int,
    edges: Sequence[Tuple[int, int]],
) -> DaguaGraph:
    """Build a graph from integer node indices and directed edges.

    Parameters
    ----------
    num_nodes : int
        Number of nodes to materialize.
    edges : Sequence[Tuple[int, int]]
        Directed edges expressed in terms of integer node indices.

    Returns
    -------
    DaguaGraph
        Finalized graph with computed node sizes.
    """
    return _finalize_generated_graph(
        DaguaGraph.from_edge_list(list(edges), num_nodes=max(num_nodes, 0))
    )


def _make_outerplanar_dag_graph(num_nodes: int = 20) -> DaguaGraph:
    """Build an outerplanar DAG with a path backbone and non-crossing chords.

    Parameters
    ----------
    num_nodes : int, default=20
        Number of nodes on the outer face.

    Returns
    -------
    DaguaGraph
        Directed outerplanar graph with every edge oriented forward.
    """
    edges = [(node_idx, node_idx + 1) for node_idx in range(max(num_nodes - 1, 0))]
    edges.extend((0, node_idx) for node_idx in range(2, max(num_nodes, 0)))
    return _graph_from_integer_edges(num_nodes=num_nodes, edges=edges)


def _make_planar_nested_cycles_graph(
    rings: int = 5,
    nodes_per_ring: int = 12,
) -> DaguaGraph:
    """Build a planar concentric-ring graph triangulated without crossings.

    Parameters
    ----------
    rings : int, default=5
        Number of concentric cycles.
    nodes_per_ring : int, default=12
        Nodes per ring.

    Returns
    -------
    DaguaGraph
        Deterministically oriented planar graph.
    """
    nx = _import_networkx()
    graph = nx.Graph()
    total_nodes = max(rings, 0) * max(nodes_per_ring, 0)
    graph.add_nodes_from(range(total_nodes))

    for ring_idx in range(max(rings, 0)):
        ring_base = ring_idx * nodes_per_ring
        for slot in range(max(nodes_per_ring, 0)):
            current = ring_base + slot
            nxt = ring_base + ((slot + 1) % nodes_per_ring)
            graph.add_edge(current, nxt)
            if ring_idx + 1 >= rings:
                continue
            lower = (ring_idx + 1) * nodes_per_ring + slot
            lower_diag = (ring_idx + 1) * nodes_per_ring + ((slot + 1) % nodes_per_ring)
            graph.add_edge(current, lower)
            # Consistent diagonals triangulate each annulus while preserving planarity.
            graph.add_edge(current, lower_diag)

    return _graph_from_undirected_networkx(graph)


def _make_regular_graph(degree: int, num_nodes: int, seed: int = 42) -> DaguaGraph:
    """Build a deterministic random regular graph and orient it into a DAG.

    Parameters
    ----------
    degree : int
        Regular degree for every node.
    num_nodes : int
        Number of nodes in the graph.
    seed : int, default=42
        Random seed forwarded to NetworkX.

    Returns
    -------
    DaguaGraph
        Oriented regular graph.
    """
    nx = _import_networkx()
    return _graph_from_undirected_networkx(nx.random_regular_graph(degree, num_nodes, seed=seed))


def _make_triangular_lattice_graph(rows: int = 6, cols: int = 6) -> DaguaGraph:
    """Build a triangular lattice DAG on a rectangular patch.

    Parameters
    ----------
    rows : int, default=6
        Number of lattice rows.
    cols : int, default=6
        Number of lattice columns.

    Returns
    -------
    DaguaGraph
        Directed triangular lattice with right, down, and down-right edges.
    """
    edges: list[tuple[int, int]] = []
    for row_idx in range(max(rows, 0)):
        for col_idx in range(max(cols, 0)):
            node_idx = row_idx * cols + col_idx
            if col_idx + 1 < cols:
                edges.append((node_idx, node_idx + 1))
            if row_idx + 1 < rows:
                below = (row_idx + 1) * cols + col_idx
                edges.append((node_idx, below))
                if col_idx + 1 < cols:
                    edges.append((node_idx, below + 1))
    return _graph_from_integer_edges(num_nodes=rows * cols, edges=edges)


def _make_hexagonal_lattice_graph(rows: int = 6, cols: int = 7) -> DaguaGraph:
    """Build a honeycomb lattice patch with degree-3 interior nodes.

    Parameters
    ----------
    rows : int, default=6
        Number of lattice rows.
    cols : int, default=7
        Number of lattice columns.

    Returns
    -------
    DaguaGraph
        Directed honeycomb-style lattice.
    """
    edges: list[tuple[int, int]] = []
    for row_idx in range(max(rows, 0)):
        for col_idx in range(max(cols, 0)):
            node_idx = row_idx * cols + col_idx
            if row_idx + 1 < rows:
                edges.append((node_idx, (row_idx + 1) * cols + col_idx))
            if col_idx + 1 < cols and col_idx % 2 == row_idx % 2:
                edges.append((node_idx, node_idx + 1))
    return _graph_from_integer_edges(num_nodes=rows * cols, edges=edges)


def _make_protein_ppi_graph(seed: int = 42) -> DaguaGraph:
    """Build a clustered scale-free protein interaction benchmark.

    Parameters
    ----------
    seed : int, default=42
        Random seed for the synthetic interaction sampling.

    Returns
    -------
    DaguaGraph
        Oriented protein-protein interaction style graph.
    """
    nx = _import_networkx()
    rng = random.Random(seed)
    graph = nx.barabasi_albert_graph(50, 2, seed=seed)

    next_node = 50
    module_sizes = [15] * 10
    module_node_sets: list[list[int]] = []

    for module_size in module_sizes:
        module_nodes = list(range(next_node, next_node + module_size))
        module_node_sets.append(module_nodes)
        graph.add_nodes_from(module_nodes)
        for local_idx in range(module_size - 1):
            graph.add_edge(module_nodes[local_idx], module_nodes[local_idx + 1])
        for src_offset in range(module_size):
            for tgt_offset in range(src_offset + 1, module_size):
                if rng.random() < 0.32:
                    graph.add_edge(module_nodes[src_offset], module_nodes[tgt_offset])
        anchors = rng.sample(range(50), k=2)
        for anchor in anchors:
            graph.add_edge(anchor, rng.choice(module_nodes))
            graph.add_edge(anchor, rng.choice(module_nodes))
        for module_node in module_nodes:
            if rng.random() < 0.2:
                graph.add_edge(rng.choice(anchors), module_node)
        next_node += module_size

    for module_idx in range(len(module_node_sets) - 1):
        graph.add_edge(
            rng.choice(module_node_sets[module_idx]),
            rng.choice(module_node_sets[module_idx + 1]),
        )
        if module_idx + 2 < len(module_node_sets) and rng.random() < 0.5:
            graph.add_edge(
                rng.choice(module_node_sets[module_idx]),
                rng.choice(module_node_sets[module_idx + 2]),
            )

    return _graph_from_undirected_networkx(graph)


def _make_citation_dag_graph(
    layer_count: int = 10,
    layer_size: int = 30,
    seed: int = 42,
) -> DaguaGraph:
    """Build a temporal citation DAG with preferential future attachment.

    Parameters
    ----------
    layer_count : int, default=10
        Number of temporal layers.
    layer_size : int, default=30
        Number of papers per layer.
    seed : int, default=42
        Random seed for attachment sampling.

    Returns
    -------
    DaguaGraph
        Directed acyclic citation network.
    """
    rng = random.Random(seed)
    graph = DaguaGraph()
    attractiveness: list[float] = []

    for layer_idx in range(max(layer_count, 0)):
        for local_idx in range(max(layer_size, 0)):
            graph.add_node(layer_idx * layer_size + local_idx)
            attractiveness.append(1.0)

    for layer_idx in range(max(layer_count, 0)):
        layer_start = layer_idx * layer_size
        layer_end = layer_start + layer_size
        for node_idx in range(layer_start, layer_end):
            candidate_sources = list(range(node_idx))
            if not candidate_sources:
                continue

            edge_budget = 2 + rng.randrange(4)
            chosen_sources: set[int] = set()
            while len(chosen_sources) < edge_budget and len(chosen_sources) < len(
                candidate_sources
            ):
                weighted_candidates: list[float] = []
                for source_idx in candidate_sources:
                    source_layer = source_idx // layer_size
                    recency_bonus = 2.0 if layer_idx - source_layer <= 2 else 1.0
                    weighted_candidates.append(attractiveness[source_idx] * recency_bonus)
                source = rng.choices(candidate_sources, weights=weighted_candidates, k=1)[0]
                if source in chosen_sources:
                    continue
                chosen_sources.add(source)
                graph.add_edge(source, node_idx)
                attractiveness[source] += 1.0

            if layer_idx > 0 and node_idx > layer_start and rng.random() < 0.35:
                intra_source = layer_start + rng.randrange(node_idx - layer_start)
                graph.add_edge(intra_source, node_idx)
                attractiveness[intra_source] += 0.5

    return _finalize_generated_graph(graph)


def _build_sierpinski_networkx(depth: int, prefix: str) -> tuple[Any, tuple[str, str, str]]:
    """Recursively build one Sierpinski triangle graph in NetworkX form.

    Parameters
    ----------
    depth : int
        Recursion depth. ``0`` yields a single triangle.
    prefix : str
        Prefix used to keep recursively created node names unique.

    Returns
    -------
    tuple[Any, tuple[str, str, str]]
        NetworkX graph plus ``(top, left, right)`` corner node IDs.
    """
    nx = _import_networkx()
    if depth <= 0:
        top = f"{prefix}top"
        left = f"{prefix}left"
        right = f"{prefix}right"
        graph = nx.Graph()
        graph.add_edges_from(((top, left), (top, right), (left, right)))
        return graph, (top, left, right)

    top_graph, (top_top, top_left, top_right) = _build_sierpinski_networkx(depth - 1, f"{prefix}t.")
    left_graph, (left_top, left_left, left_right) = _build_sierpinski_networkx(
        depth - 1, f"{prefix}l."
    )
    right_graph, (right_top, right_left, right_right) = _build_sierpinski_networkx(
        depth - 1, f"{prefix}r."
    )

    left_graph = nx.relabel_nodes(left_graph, {left_top: top_left}, copy=True)
    right_graph = nx.relabel_nodes(
        right_graph,
        {right_top: top_right, right_left: left_right},
        copy=True,
    )
    graph = nx.compose_all((top_graph, left_graph, right_graph))
    return graph, (top_top, left_left, right_right)


def _make_sierpinski_graph(depth: int = 3) -> DaguaGraph:
    """Build a Sierpinski triangle graph at the requested depth.

    Parameters
    ----------
    depth : int, default=3
        Recursion depth for the gasket construction.

    Returns
    -------
    DaguaGraph
        Oriented Sierpinski graph.
    """
    graph, _ = _build_sierpinski_networkx(depth=depth, prefix="s.")
    return _graph_from_undirected_networkx(graph)


def _make_chung_lu_graph(num_nodes: int = 150, seed: int = 42) -> DaguaGraph:
    """Build a Chung-Lu style graph with extra triadic closure.

    Parameters
    ----------
    num_nodes : int, default=150
        Number of nodes in the graph.
    seed : int, default=42
        Random seed for expected-degree sampling and closure edges.

    Returns
    -------
    DaguaGraph
        Oriented clustered scale-free random graph.
    """
    nx = _import_networkx()
    rng = random.Random(seed)
    raw_weights = [(node_idx + 1) ** -1.2 for node_idx in range(max(num_nodes, 1))]
    target_volume = max(float(num_nodes) * 4.5, 1.0)
    scale = target_volume / sum(raw_weights)
    expected_degrees = [min(scale * weight, float(num_nodes - 1)) for weight in raw_weights]
    graph = nx.expected_degree_graph(expected_degrees, selfloops=False, seed=seed)

    for node_idx in list(graph.nodes()):
        neighbors = list(graph.neighbors(node_idx))
        if len(neighbors) < 2:
            continue
        for _ in range(min(3, len(neighbors) // 2)):
            left, right = rng.sample(neighbors, 2)
            if left != right and not graph.has_edge(left, right) and rng.random() < 0.25:
                graph.add_edge(left, right)

    if graph.number_of_nodes() < num_nodes:
        graph.add_nodes_from(range(graph.number_of_nodes(), num_nodes))
    for isolated in list(nx.isolates(graph)):
        if isolated != 0:
            graph.add_edge(0, isolated)

    return _graph_from_undirected_networkx(graph)


def _make_multi_component_graph() -> DaguaGraph:
    """Build a deterministic graph with six disconnected components.

    Returns
    -------
    DaguaGraph
        Graph containing components of sizes 40, 20, 10, 5, 3, and 2.
    """
    node_ids = [f"mc_{node_idx}" for node_idx in range(80)]
    edges: list[tuple[str, str]] = []

    for node_idx in range(39):
        edges.append((node_ids[node_idx], node_ids[node_idx + 1]))
    for node_idx in range(0, 35, 5):
        edges.append((node_ids[node_idx], node_ids[node_idx + 5]))

    offset = 40
    for node_idx in range(offset, offset + 19):
        edges.append((node_ids[node_idx], node_ids[node_idx + 1]))
    edges.append((node_ids[offset], node_ids[offset + 19]))

    offset = 60
    for node_idx in range(offset + 1, offset + 10):
        edges.append((node_ids[offset], node_ids[node_idx]))

    offset = 70
    for node_idx in range(offset, offset + 4):
        edges.append((node_ids[node_idx], node_ids[node_idx + 1]))

    offset = 75
    edges.extend(
        (
            (node_ids[offset], node_ids[offset + 1]),
            (node_ids[offset + 1], node_ids[offset + 2]),
        )
    )

    return _build_named_graph(node_ids, edges)


def _make_random_bipartite_graph(
    left_size: int = 30,
    right_size: int = 30,
    num_edges: int = 90,
    seed: int = 42,
) -> DaguaGraph:
    """Build a random bipartite graph with edges only across the partition.

    Parameters
    ----------
    left_size : int, default=30
        Size of the left partition.
    right_size : int, default=30
        Size of the right partition.
    num_edges : int, default=90
        Number of cross-partition edges to sample.
    seed : int, default=42
        Random seed for edge sampling.

    Returns
    -------
    DaguaGraph
        Directed bipartite graph.
    """
    rng = random.Random(seed)
    edge_set: set[tuple[int, int]] = set()
    while len(edge_set) < num_edges:
        edge_set.add((rng.randrange(left_size), left_size + rng.randrange(right_size)))
    return _graph_from_integer_edges(num_nodes=left_size + right_size, edges=sorted(edge_set))


def _make_heavy_tail_weight_graph(num_nodes: int = 50, seed: int = 42) -> DaguaGraph:
    """Build a connected weighted graph with log-normal edge weights.

    Parameters
    ----------
    num_nodes : int, default=50
        Number of nodes in the graph.
    seed : int, default=42
        Random seed for topology and weight generation.

    Returns
    -------
    DaguaGraph
        Weighted connected graph.
    """
    rng = random.Random(seed)
    graph = DaguaGraph()
    for node_idx in range(max(num_nodes, 0)):
        graph.add_node(node_idx, label=str(node_idx))

    seen_edges: set[tuple[int, int]] = set()
    for node_idx in range(1, max(num_nodes, 0)):
        parent = rng.randrange(node_idx)
        edge = (parent, node_idx)
        seen_edges.add(edge)
        graph.add_edge(*edge, weight=rng.lognormvariate(0.0, 1.2))

    extra_edges = max(num_nodes // 2, 0)
    while extra_edges > 0:
        source = rng.randrange(max(num_nodes - 1, 1))
        target = rng.randrange(source + 1, max(num_nodes, 1))
        edge = (source, target)
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        graph.add_edge(*edge, weight=rng.lognormvariate(0.0, 1.2))
        extra_edges -= 1

    return _finalize_generated_graph(graph)


def _make_petersen_graph() -> DaguaGraph:
    """Build the classic Petersen graph and orient it acyclically.

    Returns
    -------
    DaguaGraph
        Oriented Petersen graph.
    """
    nx = _import_networkx()
    return _graph_from_undirected_networkx(nx.petersen_graph())


def _expanded_structural_graphs() -> List[TestGraph]:
    """Build the supplemental structural graphs requested for evaluation.

    Returns
    -------
    List[TestGraph]
        New structural coverage graphs wrapped with evaluation metadata.
    """
    sparse_graph, dense_graph = make_sparse_dense_pair(n=50, seed=42)
    powerlaw_500 = make_powerlaw_dag(500, seed=42)
    powerlaw_500.name = "powerlaw_500"
    powerlaw_500.tags = {"scale-free", "large-sparse"}
    powerlaw_500.description = "Power-law DAG benchmark at 500 nodes"
    powerlaw_500.expected_challenges = "Degree skew and sparse long-range attachment patterns"

    powerlaw_2000 = make_powerlaw_dag(2000, seed=42)
    powerlaw_2000.name = "powerlaw_2000"
    powerlaw_2000.tags = {"scale-free", "large-sparse"}
    powerlaw_2000.description = "Power-law DAG benchmark at 2,000 nodes"
    powerlaw_2000.expected_challenges = "High-degree concentration at larger sparse scale"

    graphs = [
        TestGraph(
            name="scale_free_ba_120",
            graph=make_scale_free(n=120, m=3, seed=42),
            tags={"scale-free", "large-sparse", "undirected"},
            description="Barabasi-Albert style preferential-attachment DAG",
            expected_challenges="High-degree hubs, edge bundling, and hub dominance",
        ),
        TestGraph(
            name="grid_rect_6x8",
            graph=make_grid(6, 8, seed=42),
            tags={"grid", "diamond", "undirected"},
            description="Rectangular 6x8 grid DAG with right/down edges",
            expected_challenges="Regular spacing, local ordering, and dense local crossings",
        ),
        TestGraph(
            name="complete_bipartite_8x12",
            graph=make_complete_bipartite(8, 12, seed=42),
            tags={"bipartite", "wide-parallel", "undirected"},
            description="Complete bipartite K(8,12) layered DAG",
            expected_challenges="Crossing minimization between two dense layers",
        ),
        TestGraph(
            name="clustered_medium_5x20",
            graph=make_clustered_medium(5, 20, inter_density=0.05, seed=42),
            tags={"clustered", "nested-shallow"},
            description="Five medium clusters with sparse inter-cluster handoffs",
            expected_challenges="Cluster separation, ordering, and sparse bridge routing",
        ),
        TestGraph(
            name="hub_and_spoke_3x20",
            graph=make_hub_and_spoke(3, 20, seed=42),
            tags={"hub-spoke", "wide-parallel"},
            description="Three dominant hubs each fanning to twenty spokes",
            expected_challenges="Fan-out balance around high-degree hub nodes",
        ),
        TestGraph(
            name="wide_single_layer_1_50_1",
            graph=make_wide_single_layer(1, 50, 1, seed=42),
            tags={"wide-layer", "wide-parallel"},
            description="Single source and sink around a very wide middle layer",
            expected_challenges="Extreme width, horizontal spacing, and ordering stability",
        ),
        TestGraph(
            name="sparse_pair_50",
            graph=sparse_graph,
            tags={"large-sparse"},
            description="Sparse half of a matched sparse-versus-dense DAG pair",
            expected_challenges="Baseline readability for density comparisons",
        ),
        TestGraph(
            name="dense_pair_50",
            graph=dense_graph,
            tags={"large-dense"},
            description="Dense half of a matched sparse-versus-dense DAG pair",
            expected_challenges="Crossing pressure compared against the sparse twin",
        ),
        TestGraph(
            name="compound_dag_5x30",
            graph=make_compound_dag(5, 30, seed=42),
            tags={"compound", "clustered", "nested-shallow"},
            description="Clustered DAG where stages form a higher-level DAG",
            expected_challenges="Compound hierarchy, cluster flow, and skip handoffs",
        ),
        TestGraph(
            name="long_skip_only_24",
            graph=make_long_skip_only(24, seed=42),
            tags={"skip-heavy", "linear-deep"},
            description="Pathological DAG where every edge skips at least three layers",
            expected_challenges="Very long edges without local anchors",
        ),
        TestGraph(
            name="parallel_cycles_4x5",
            graph=make_parallel_cycles(4, 5, seed=42),
            tags={"cyclic", "disconnected"},
            description="Four independent directed 5-cycles",
            expected_challenges="Cycle handling plus disconnected component packing",
        ),
        TestGraph(
            name="resnet_stack_4x16",
            graph=make_resnet_block(4, 16, seed=42),
            tags={"neural-net", "skip-light", "nested-shallow"},
            description="Four-block residual stack with block-level clusters",
            expected_challenges="Repeated residual motifs and merge alignment",
        ),
        TestGraph(
            name="transformer_full_4h_2l",
            graph=make_transformer_full(4, 2, seed=42),
            tags={"neural-net", "wide-parallel", "nested-deep"},
            description="Two-layer transformer-style stack with four attention heads",
            expected_challenges="Parallel heads, residual routing, and nested attention groups",
        ),
        TestGraph(
            name="dependency_graph_100",
            graph=make_dependency_graph(100, 5, seed=42),
            tags={"dependency", "scale-free", "clustered"},
            description="Dependency DAG with dominant core libraries and many packages",
            expected_challenges="Hub-heavy sources and dependency fan-in toward packages",
        ),
        TestGraph(
            name="org_chart_1_5_4_8",
            graph=make_org_chart(4, (1, 5, 4, 8), seed=42),
            tags={"tree"},
            description="Non-uniform organizational tree with changing level widths",
            expected_challenges="Parent-child spacing under uneven breadth",
        ),
        TestGraph(
            name="small_world_100",
            graph=make_small_world(100, 4, 0.1, seed=42),
            tags={"small-world", "cyclic", "undirected"},
            description="Directed Watts-Strogatz small-world graph",
            expected_challenges="Short paths, cycles, and non-layered global geometry",
        ),
    ]

    # Real-world classics
    graphs.extend(
        [
            make_real_karate_graph(),
            make_real_lesmis_graph(),
            make_real_football_graph(seed=42),
        ]
    )

    # Erdos-Renyi random graphs
    graphs.extend(
        [
            make_erdos_renyi(100, p=0.04, seed=42),
            make_erdos_renyi(500, p=0.008, seed=42),
            make_erdos_renyi(2000, p=0.003, seed=42),
        ]
    )

    # Random geometric graphs
    graphs.extend(
        [
            make_random_geometric(100, radius=0.25, seed=42),
            make_random_geometric(500, radius=0.1, seed=42),
            make_random_geometric(2000, radius=0.05, seed=42),
        ]
    )

    # Additional scale-free and community benchmarks
    graphs.extend(
        [
            TestGraph(
                name="ba_500",
                graph=make_scale_free(n=500, m=3, seed=42),
                tags={"scale-free", "large-sparse", "undirected"},
                description="Barabasi-Albert style preferential-attachment DAG at 500 nodes",
                expected_challenges="High-degree hubs at medium sparse scale",
            ),
            TestGraph(
                name="ba_2000",
                graph=make_scale_free(n=2000, m=3, seed=42),
                tags={"scale-free", "large-sparse", "undirected"},
                description="Barabasi-Albert style preferential-attachment DAG at 2,000 nodes",
                expected_challenges="Hub dominance and long sparse spokes at larger scale",
            ),
            TestGraph(
                name="ba_5000",
                graph=make_scale_free(n=5000, m=4, seed=42),
                tags={"scale-free", "large-sparse", "undirected"},
                description="Barabasi-Albert style preferential-attachment DAG at 5,000 nodes",
                expected_challenges="Extremely dominant hubs and bundled fan-in at scale",
            ),
            make_sbm(4, 30, p_in=0.3, p_out=0.01, seed=42),
            make_sbm(5, 50, p_in=0.2, p_out=0.005, seed=42),
            make_sbm(8, 100, p_in=0.1, p_out=0.002, seed=42),
        ]
    )

    # Larger meshes and broader coverage of existing generator families
    graphs.extend(
        [
            TestGraph(
                name="grid_20x20",
                graph=make_grid(20, 20, seed=42),
                tags={"grid", "mesh", "large-dense", "undirected"},
                description="20x20 mesh-like grid DAG",
                expected_challenges="Regular spacing over a moderately large mesh",
            ),
            TestGraph(
                name="grid_50x50",
                graph=make_grid(50, 50, seed=42),
                tags={"grid", "mesh", "large-dense", "undirected"},
                description="50x50 mesh-like grid DAG",
                expected_challenges="Large regular mesh with dense local edge pressure",
            ),
            TestGraph(
                name="wide_1_100_1",
                graph=make_wide_single_layer(1, 100, 1, seed=42),
                tags={"wide-layer", "wide-parallel"},
                description="Single source and sink around a 100-node middle layer",
                expected_challenges="Extreme width with minimal vertical structure",
            ),
            TestGraph(
                name="wide_3_50_3",
                graph=make_wide_single_layer(3, 50, 3, seed=42),
                tags={"wide-layer", "wide-parallel"},
                description="Three sources and sinks around a 50-node middle layer",
                expected_challenges="Ordering stability under broader symmetric fan-in and fan-out",
            ),
            TestGraph(
                name="hub_spoke_5x50",
                graph=make_hub_and_spoke(5, 50, seed=42),
                tags={"hub-spoke", "wide-parallel"},
                description="Five hubs each feeding fifty spokes toward a shared exit",
                expected_challenges="Large fan-out around multiple dominant hub centers",
            ),
            TestGraph(
                name="hub_spoke_10x20",
                graph=make_hub_and_spoke(10, 20, seed=42),
                tags={"hub-spoke", "wide-parallel"},
                description="Ten hubs each feeding twenty spokes toward a shared exit",
                expected_challenges="Competing hub centers with repeated spoke bundles",
            ),
            TestGraph(
                name="compound_10x20",
                graph=make_compound_dag(10, 20, seed=42),
                tags={"compound", "clustered", "nested-shallow"},
                description="Ten-stage compound DAG with twenty nodes per cluster",
                expected_challenges="Cluster flow and inter-stage routing at moderate scale",
            ),
            TestGraph(
                name="small_world_500",
                graph=make_small_world(500, 6, 0.1, seed=42),
                tags={"small-world", "cyclic", "undirected"},
                description="Directed Watts-Strogatz small-world graph at 500 nodes",
                expected_challenges="Short paths and cycles at medium scale",
            ),
            TestGraph(
                name="small_world_2000",
                graph=make_small_world(2000, 4, 0.05, seed=42),
                tags={"small-world", "cyclic", "undirected"},
                description="Directed Watts-Strogatz small-world graph at 2,000 nodes",
                expected_challenges="Sparse cyclic structure with weak global layering cues",
            ),
            TestGraph(
                name="dependency_500",
                graph=make_dependency_graph(500, 10, seed=42),
                tags={"dependency", "scale-free", "clustered"},
                description="Dependency DAG with ten dominant core libraries at 500 nodes",
                expected_challenges="Large dependency fan-in around a compact core",
            ),
            TestGraph(
                name="org_chart_deep",
                graph=make_org_chart(6, (1, 3, 5, 10, 20, 40), seed=42),
                tags={"tree"},
                description="Six-level organizational chart with widening breadth by level",
                expected_challenges="Parent-child spacing across strongly varying level widths",
            ),
            powerlaw_500,
            powerlaw_2000,
        ]
    )

    return graphs


_SCALE_SIZES = {
    "small": [100, 500, 1_000, 2_000],
    "medium": [5_000, 10_000, 50_000],
    "large": [100_000, 500_000, 1_000_000],
    "huge": [5_000_000, 10_000_000, 50_000_000],
}

# Shapes to generate at each tier
_SCALE_SHAPES_FULL = [
    ("random_dag", lambda n, s: make_random_dag(n, density=1.5, seed=s)),
    ("wide_dag", lambda n, s: make_wide_dag(n, seed=s)),
    ("chain", lambda n, s: make_chain(n, seed=s)),
    ("tree", lambda n, s: make_tree(n, branching=3, seed=s)),
    ("grid", lambda n, s: make_grid(n, seed=s)),
    ("sparse_layered", lambda n, s: make_sparse_layered(n, seed=s)),
    ("powerlaw", lambda n, s: make_powerlaw_dag(n, seed=s)),
    ("bipartite", lambda n, s: make_bipartite(n, seed=s)),
]

_SCALE_SHAPES_MINIMAL = [
    ("random_dag", lambda n, s: make_random_dag(n, density=1.5, seed=s)),
    ("sparse_layered", lambda n, s: make_sparse_layered(n, seed=s)),
    ("powerlaw", lambda n, s: make_powerlaw_dag(n, seed=s)),
]


def get_scale_suite(tier: str = "small") -> List[TestGraph]:
    """Get scale test graphs for a given tier.

    Tiers:
      small:  100, 500, 1000, 2000 nodes (all competitors)
      medium: 5000, 10000, 50000 nodes (dagua + sfdp + spring + elk)
      large:  100K, 500K, 1M nodes (dagua + sfdp only)
      huge:   5M, 10M, 50M nodes (dagua only)

    Each tier returns multiple graph shapes at each size.
    """
    if tier not in _SCALE_SIZES:
        raise ValueError(f"Unknown tier {tier!r}. Choose from: {list(_SCALE_SIZES)}")

    graphs = []
    sizes = _SCALE_SIZES[tier]
    # Full variety for small/medium, minimal for large/huge (memory)
    shapes = _SCALE_SHAPES_FULL if tier in ("small", "medium") else _SCALE_SHAPES_MINIMAL

    for n in sizes:
        for _name, gen in shapes:
            graphs.append(gen(n, 42))

    return graphs


def get_scaling_collection(seed: int = 42) -> List[TestGraph]:
    """Get graphs spanning several orders of magnitude for scaling studies.

    Returns graphs at sizes: 50, 200, 1K, 5K, 20K, 100K, 500K, 2M.
    At each size, generates multiple topologies (random DAG, sparse layered,
    power-law, grid, tree) so you can see how both size and topology affect
    metrics.

    The largest graphs (500K, 2M) use only O(n)-edge topologies to keep
    memory reasonable.
    """
    sizes = [50, 200, 1_000, 5_000, 20_000, 100_000, 500_000, 2_000_000]

    # All shapes up to 100K, then just sparse ones for 500K+
    full_shapes = [
        ("random_dag", lambda n, s: make_random_dag(n, density=1.5, seed=s)),
        ("sparse_layered", lambda n, s: make_sparse_layered(n, seed=s)),
        ("powerlaw", lambda n, s: make_powerlaw_dag(n, seed=s)),
        ("grid", lambda n, s: make_grid(n, seed=s)),
        ("tree", lambda n, s: make_tree(n, branching=3, seed=s)),
    ]
    sparse_shapes = [
        ("random_dag", lambda n, s: make_random_dag(n, density=1.5, seed=s)),
        ("sparse_layered", lambda n, s: make_sparse_layered(n, seed=s)),
        ("powerlaw", lambda n, s: make_powerlaw_dag(n, seed=s)),
    ]

    graphs = []
    for n in sizes:
        shapes = full_shapes if n <= 100_000 else sparse_shapes
        for _name, gen in shapes:
            graphs.append(gen(n, seed))

    return graphs


# ─── Utilities ────────────────────────────────────────────────────────────────


def list_tags() -> Set[str]:
    """Return all available tags across test graphs."""
    all_tags = set()
    for tg in get_test_graphs():
        all_tags.update(tg.tags)
    return all_tags

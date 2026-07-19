"""Coverage tests for the benchmark graph corpus."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict, deque
from typing import Callable

import torch
from _pytest.monkeypatch import MonkeyPatch

from dagua.config import LayoutConfig
from dagua.eval.graphs import (
    _expanded_structural_graphs,
    _r8_nested_cluster_graphs,
    _synthetic_graphs,
    get_test_graphs,
    make_clustered_medium,
    make_complete_bipartite,
    make_compound_dag,
    make_dependency_graph,
    make_erdos_renyi,
    make_grid,
    make_hub_and_spoke,
    make_long_skip_only,
    make_org_chart,
    make_parallel_cycles,
    make_random_geometric,
    make_real_football_graph,
    make_real_karate_graph,
    make_real_lesmis_graph,
    make_resnet_block,
    make_sbm,
    make_scale_free,
    make_small_world,
    make_sparse_dense_pair,
    make_transformer_full,
    make_wide_single_layer,
)
from dagua.graph import DaguaGraph
from dagua.layout import layout


def _graph_catalog_snapshot_script() -> str:
    """Return Python code that serializes benchmark graph identity fields.

    Returns
    -------
    str
        Subprocess script that emits one stable JSON object containing each
        graph's node count, edge tensor, node labels, and edge labels.
    """
    return r"""
import json

from dagua.eval.graphs import get_test_graphs

payload = {}
for test_graph in get_test_graphs():
    graph = test_graph.graph
    payload[test_graph.name] = {
        "num_nodes": graph.num_nodes,
        "edge_index": graph.edge_index.cpu().tolist(),
        "node_labels": list(graph.node_labels),
        "edge_labels": list(graph.edge_labels),
    }
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
"""


def _build_graph_catalog_snapshot(hash_seed: str) -> bytes:
    """Build the benchmark graph catalog in a subprocess.

    Parameters
    ----------
    hash_seed : str
        Value assigned to ``PYTHONHASHSEED`` for the subprocess.

    Returns
    -------
    bytes
        JSON bytes emitted by the subprocess.
    """
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = hash_seed
    env["PYTHONPATH"] = os.getcwd()
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", "/tmp/mpl")
    completed = subprocess.run(
        [sys.executable, "-c", _graph_catalog_snapshot_script()],
        check=True,
        cwd=os.getcwd(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _changed_snapshot_names(left: bytes, right: bytes) -> list[str]:
    """Return graph names whose serialized identity differs between snapshots.

    Parameters
    ----------
    left : bytes
        JSON snapshot from the first subprocess.
    right : bytes
        JSON snapshot from the second subprocess.

    Returns
    -------
    list[str]
        Sorted benchmark graph names with differing serialized identity.
    """
    left_payload = json.loads(left)
    right_payload = json.loads(right)
    names = sorted(set(left_payload) | set(right_payload))
    return [name for name in names if left_payload.get(name) != right_payload.get(name)]


def test_benchmark_graphs_are_hash_seed_deterministic() -> None:
    """Every benchmark graph should serialize identically across hash seeds."""
    seed_zero_snapshot = _build_graph_catalog_snapshot("0")
    seed_one_snapshot = _build_graph_catalog_snapshot("1")

    changed_names = _changed_snapshot_names(seed_zero_snapshot, seed_one_snapshot)
    assert seed_zero_snapshot == seed_one_snapshot, changed_names


def _component_count(edge_index, num_nodes: int) -> int:
    neighbors = defaultdict(set)
    if edge_index.numel() > 0:
        src = edge_index[0].tolist()
        tgt = edge_index[1].tolist()
        for s, t in zip(src, tgt):
            neighbors[s].add(t)
            neighbors[t].add(s)

    seen = set()
    count = 0
    for node in range(num_nodes):
        if node in seen:
            continue
        count += 1
        queue = deque([node])
        seen.add(node)
        while queue:
            cur = queue.popleft()
            for nxt in neighbors[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    return count


R8_NESTED_EXPECTATIONS = {
    "r8_nested_chain_depth8_directed": {"nested-depth", "directed"},
    "r8_nested_balanced_3x3x4": {"nested-depth", "fanout", "directed"},
    "r8_nested_mixed_direct_leaf": {"nested-depth", "fanout", "directed"},
    "r8_nested_cross_edges_ladder": {"nested-depth", "fanout", "directed"},
    "r8_nested_edge_labels_compound": {"nested-depth", "mixed-labels", "directed"},
    "r8_nested_wide_labels_shapes": {
        "nested-depth",
        "mixed-labels",
        "mixed-shapes",
        "directed",
    },
    "r8_nested_undirected_communities_depth3": {
        "nested-depth",
        "fanout",
        "directed-undirected",
        "undirected",
    },
    "r8_nested_sbm_overlap_trap": {"fanout", "directed-undirected", "undirected"},
    "r8_nested_parent_child_backedges": {"nested-depth", "directed"},
    "r8_nested_disconnected_forest": {"nested-depth", "fanout", "disconnected", "directed"},
    "r8_nested_singleton_deep": {"nested-depth", "directed"},
    "r8_nested_scale_1k_budget": {
        "nested-depth",
        "fanout",
        "mixed-labels",
        "directed-undirected",
        "scale",
    },
    "r8_nested_lr_direction": {"nested-depth", "mixed-labels", "directed"},
}


def _cluster_depth(graph: DaguaGraph) -> int:
    """Return maximum cluster nesting depth.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose ``clusters`` and ``cluster_parents`` metadata are checked.

    Returns
    -------
    int
        Maximum root-to-leaf cluster depth.
    """
    max_depth = 0
    for cluster_name in graph.clusters:
        depth = 1
        parent = graph.cluster_parents.get(cluster_name)
        while parent is not None:
            depth += 1
            parent = graph.cluster_parents.get(parent)
        max_depth = max(max_depth, depth)
    return max_depth


def _edge_id_pairs(graph: DaguaGraph) -> set[tuple[str, str]]:
    """Return edge endpoint node identifiers for a graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose edge tensor should be converted into stable node IDs.

    Returns
    -------
    set[tuple[str, str]]
        Directed endpoint ID pairs.
    """
    return {
        (str(graph._index_to_id[int(source)]), str(graph._index_to_id[int(target)]))
        for source, target in graph.edge_index.t().tolist()
    }


def test_r8_nested_cluster_corpus_entries_are_registered_and_layoutable(
    monkeypatch: MonkeyPatch,
) -> None:
    """R8 adds exactly the canonical nested-cluster rows to the tuning corpus."""
    monkeypatch.setenv("DAGUA_NATIVE_DISABLE_W5", "1")
    r8_graphs = {test_graph.name: test_graph for test_graph in _r8_nested_cluster_graphs()}
    corpus_graphs = {test_graph.name: test_graph for test_graph in get_test_graphs()}

    assert set(r8_graphs) == set(R8_NESTED_EXPECTATIONS)
    assert set(R8_NESTED_EXPECTATIONS) <= set(corpus_graphs)
    assert len(corpus_graphs) == 133

    for name, required_tags in R8_NESTED_EXPECTATIONS.items():
        test_graph = corpus_graphs[name]
        graph = test_graph.graph

        assert required_tags <= test_graph.tags
        assert {"r8_nested", "clustered", "synthetic"} <= test_graph.tags
        assert graph.clusters
        assert graph.node_sizes is not None
        assert graph.node_sizes.shape == (graph.num_nodes, 2)
        assert all(parent in graph.clusters for parent in graph.cluster_parents.values() if parent)
        assert _cluster_depth(graph) >= (8 if name == "r8_nested_chain_depth8_directed" else 1)

        if name in {"r8_nested_edge_labels_compound", "r8_nested_lr_direction"}:
            assert graph.cluster_labels
        if name == "r8_nested_edge_labels_compound":
            assert any(label is not None for label in graph.edge_labels)
        if name == "r8_nested_wide_labels_shapes":
            shapes = {style.shape for style in graph.node_styles if style is not None}
            assert {"rect", "ellipse", "diamond", "roundrect", "circle"} <= shapes
            assert any("wide label" in label for label in graph.node_labels)
        if name == "r8_nested_lr_direction":
            assert graph.direction == "LR"
        if name == "r8_nested_scale_1k_budget":
            assert 950 <= graph.num_nodes <= 1050
            assert any(cluster.startswith(f"{name}_directed_region_") for cluster in graph.clusters)
            assert any(
                cluster.startswith(f"{name}_undirected_region_") for cluster in graph.clusters
            )
            edge_pairs = _edge_id_pairs(graph)
            assert ("d0_l0_0", "d0_l1_0") in edge_pairs
            assert ("d0_l1_0", "d0_l0_0") not in edge_pairs
            assert ("u0_0", "u0_1") in edge_pairs
            assert ("u0_1", "u0_0") in edge_pairs

        config = LayoutConfig(quality="draft", time_budget_s=0.5, device="cpu", seed=42)
        positions = layout(graph, config)
        assert positions.shape == (graph.num_nodes, 2)
        assert torch.isfinite(positions).all()


def test_synthetic_graphs_cover_common_and_niche_motifs():
    graphs = _synthetic_graphs()
    names = {tg.name for tg in graphs}
    tags = set().union(*(tg.tags for tg in graphs))

    expected_names = {
        "linear_3layer_mlp",
        "transformer_layer",
        "hierarchical_residual_stage",
        "recurrent_feedback_cell",
        "parallel_multiedge_bundle",
        "disconnected_encoder_residual",
        "moe_router_sparse",
        "ragged_feature_pyramid",
        "kitchen_sink_hybrid_net",
        "kitchen_sink_platform_graph",
        "extreme_mixed_width_transformer",
        "hub_fanout_label_skew",
        "clustered_longlabel_handoffs",
        "disconnected_label_cycle_collage",
        "shape_and_routing_matrix",
        "center_port_backedge_hub",
        "cluster_member_style_stress",
        "edge_label_braid",
        "nested_cluster_label_stack",
        "small_label_storm",
        "long_range_residual_ladder",
        "interleaved_cluster_crosstalk",
        "asymmetric_hourglass_hub",
        "multiscale_skip_cascade",
        "braided_feedback_tails",
        "width_skew_late_merge",
        "broken_symmetry_residual_pair",
        "hub_skip_superfan",
    }
    assert expected_names <= names

    expected_tags = {
        "linear-shallow",
        "linear-deep",
        "wide-parallel",
        "skip-light",
        "skip-heavy",
        "tree",
        "diamond",
        "nested-shallow",
        "nested-deep",
        "mixed-width",
        "self-loops",
        "multi-edge",
        "disconnected",
        "large-sparse",
        "large-dense",
    }
    assert expected_tags <= tags


def test_synthetic_graphs_include_final_structural_additions() -> None:
    """The final synthetic additions should be registered with their intended structure."""
    graphs = {tg.name: tg for tg in _synthetic_graphs()}

    assert {"planar", "dag", "sparse"} <= graphs["outerplanar_dag_20"].tags
    assert graphs["outerplanar_dag_20"].graph.num_nodes == 20

    planar = graphs["planar_60"].graph
    assert {"planar", "dense"} <= graphs["planar_60"].tags
    assert planar.num_nodes == 60

    regular_three = graphs["regular_3_30"].graph
    regular_three_degree = torch.bincount(
        regular_three.edge_index.reshape(-1),
        minlength=regular_three.num_nodes,
    )
    assert regular_three.num_nodes == 30
    assert regular_three_degree.tolist() == [3] * 30

    regular_four = graphs["regular_4_40"].graph
    regular_four_degree = torch.bincount(
        regular_four.edge_index.reshape(-1),
        minlength=regular_four.num_nodes,
    )
    assert regular_four.num_nodes == 40
    assert regular_four_degree.tolist() == [4] * 40

    triangular = graphs["triangular_lattice_36"].graph
    hexagonal = graphs["hexagonal_lattice_42"].graph
    assert triangular.num_nodes == 36
    assert hexagonal.num_nodes == 42
    assert {"grid", "lattice", "planar"} <= graphs["triangular_lattice_36"].tags
    assert {"grid", "lattice", "planar", "sparse"} <= graphs["hexagonal_lattice_42"].tags

    protein = graphs["protein_ppi_200"].graph
    citation = graphs["citation_dag_300"].graph
    assert protein.num_nodes == 200
    assert citation.num_nodes == 300
    src = citation.edge_index[0].tolist()
    tgt = citation.edge_index[1].tolist()
    assert all(source < target for source, target in zip(src, tgt))

    assert graphs["sierpinski_42"].graph.num_nodes == 42
    assert graphs["chung_lu_150"].graph.num_nodes == 150
    assert _component_count(graphs["multi_component_80"].graph.edge_index, 80) == 7

    bipartite = graphs["random_bipartite_60"].graph
    src = bipartite.edge_index[0].tolist()
    tgt = bipartite.edge_index[1].tolist()
    assert all(source < 30 for source in src)
    assert all(target >= 30 for target in tgt)

    weighted = graphs["heavy_tail_weights_50"].graph
    assert weighted.edge_weights is not None
    assert weighted.edge_weights.shape[0] == weighted.edge_index.shape[1]
    assert weighted.edge_weights.max().item() > weighted.edge_weights.min().item()

    petersen = graphs["petersen_10"].graph
    petersen_degree = torch.bincount(
        petersen.edge_index.reshape(-1),
        minlength=petersen.num_nodes,
    )
    assert graphs["petersen_10"].tags == {"regular", "famous", "small", "undirected"}
    assert petersen.num_nodes == 10
    assert petersen_degree.tolist() == [3] * 10


def test_synthetic_graphs_include_diverse_sizes_and_hierarchy():
    graphs = _synthetic_graphs()
    node_counts = [tg.graph.num_nodes for tg in graphs]
    assert min(node_counts) <= 3
    assert max(node_counts) >= 200

    assert any(tg.graph.max_cluster_depth >= 1 for tg in graphs)
    assert any(tg.graph.max_cluster_depth >= 2 for tg in graphs)


def test_special_motif_graphs_have_expected_structure():
    graphs = {tg.name: tg for tg in _synthetic_graphs()}

    recurrent = graphs["recurrent_feedback_cell"].graph
    src = recurrent.edge_index[0].tolist()
    tgt = recurrent.edge_index[1].tolist()
    assert any(s == t for s, t in zip(src, tgt))

    multiedge = graphs["parallel_multiedge_bundle"].graph
    edge_pairs = list(zip(multiedge.edge_index[0].tolist(), multiedge.edge_index[1].tolist()))
    counts = Counter(edge_pairs)
    assert max(counts.values()) >= 2

    disconnected = graphs["disconnected_encoder_residual"].graph
    assert _component_count(disconnected.edge_index, disconnected.num_nodes) >= 2


def test_kitchen_sink_graphs_combine_multiple_visual_features():
    graphs = {tg.name: tg for tg in _synthetic_graphs()}

    hybrid = graphs["kitchen_sink_hybrid_net"]
    assert {"nested-deep", "skip-heavy", "wide-parallel", "self-loops", "multi-edge"} <= hybrid.tags
    assert hybrid.graph.max_cluster_depth >= 2

    platform = graphs["kitchen_sink_platform_graph"]
    assert {"nested-deep", "disconnected", "self-loops", "wide-parallel"} <= platform.tags
    assert _component_count(platform.graph.edge_index, platform.graph.num_nodes) >= 2


def test_visual_stress_graphs_cover_label_skew_and_component_extremes():
    graphs = {tg.name: tg for tg in _synthetic_graphs()}

    extreme = graphs["extreme_mixed_width_transformer"]
    label_lengths = [len(label) for label in extreme.graph.node_labels]
    assert max(label_lengths) >= 40
    assert min(label_lengths) <= 3

    clustered = graphs["clustered_longlabel_handoffs"]
    edge_pairs = list(
        zip(clustered.graph.edge_index[0].tolist(), clustered.graph.edge_index[1].tolist())
    )
    counts = Counter(edge_pairs)
    assert max(counts.values()) >= 2
    assert clustered.graph.max_cluster_depth >= 1

    collage = graphs["disconnected_label_cycle_collage"]
    assert _component_count(collage.graph.edge_index, collage.graph.num_nodes) >= 2
    src = collage.graph.edge_index[0].tolist()
    tgt = collage.graph.edge_index[1].tolist()
    assert any(s == t for s, t in zip(src, tgt))


def test_challenge_graphs_cover_long_skips_cluster_crosstalk_and_feedback():
    graphs = {tg.name: tg for tg in _synthetic_graphs()}

    ladder = graphs["long_range_residual_ladder"]
    assert {"linear-deep", "skip-heavy", "wide-parallel"} <= ladder.tags
    ladder_src = ladder.graph.edge_index[0].tolist()
    ladder_tgt = ladder.graph.edge_index[1].tolist()
    assert max(abs(t - s) for s, t in zip(ladder_src, ladder_tgt)) >= 6

    crosstalk = graphs["interleaved_cluster_crosstalk"]
    assert {"nested-deep", "skip-heavy", "wide-parallel"} <= crosstalk.tags
    assert crosstalk.graph.max_cluster_depth >= 2
    cluster_names = set(crosstalk.graph.clusters)
    assert {"encoder", "encoder.path_a", "encoder.path_b", "decoder"} <= cluster_names

    multiscale = graphs["multiscale_skip_cascade"]
    assert {"skip-heavy", "nested-shallow", "wide-parallel"} <= multiscale.tags
    assert len(multiscale.graph.clusters) >= 3

    braid = graphs["braided_feedback_tails"]
    assert {"skip-heavy", "diamond", "linear-deep"} <= braid.tags
    braid_src = braid.graph.edge_index[0].tolist()
    braid_tgt = braid.graph.edge_index[1].tolist()
    assert any(t < s for s, t in zip(braid_src, braid_tgt))

    width_skew = graphs["width_skew_late_merge"]
    assert {"wide-parallel", "skip-heavy", "diamond"} <= width_skew.tags
    assert width_skew.graph.num_nodes >= 10

    broken = graphs["broken_symmetry_residual_pair"]
    assert {"skip-heavy", "diamond", "wide-parallel"} <= broken.tags
    assert any("breakout" in label for label in broken.graph.node_labels)

    superfan = graphs["hub_skip_superfan"]
    assert {"linear-deep", "skip-heavy", "wide-parallel"} <= superfan.tags
    hub_idx = superfan.graph._id_to_index["hub"]
    src = superfan.graph.edge_index[0].tolist()
    assert sum(s == hub_idx for s in src) >= 4


def test_style_and_routing_stress_graphs_exercise_visual_feature_surface():
    graphs = {tg.name: tg for tg in _synthetic_graphs()}

    shape_graph = graphs["shape_and_routing_matrix"].graph
    shape_set = {style.shape for style in shape_graph.node_styles if style is not None}
    assert {"rect", "ellipse", "diamond", "roundrect", "circle"} <= shape_set
    routing_modes = {style.routing for style in shape_graph.edge_styles if style is not None}
    assert {"straight", "ortho", "bezier"} <= routing_modes

    center_port = graphs["center_port_backedge_hub"].graph
    assert all(
        style is not None and style.port_style == "center" for style in center_port.edge_styles
    )
    src = center_port.edge_index[0].tolist()
    tgt = center_port.edge_index[1].tolist()
    assert any(s == t for s, t in zip(src, tgt))
    assert any(t == center_port._id_to_index["hub"] for t in tgt)

    cluster_style_graph = graphs["cluster_member_style_stress"].graph
    core_style = cluster_style_graph.cluster_styles["core"]
    assert core_style.member_node_style is not None
    assert core_style.member_node_style.shape == "diamond"
    assert core_style.member_edge_style is not None
    assert core_style.member_edge_style.routing == "ortho"


def test_label_stress_graphs_cover_edge_and_cluster_annotation_failures():
    graphs = {tg.name: tg for tg in _synthetic_graphs()}

    braid = graphs["edge_label_braid"].graph
    labeled_edges = [label for label in braid.edge_labels if label]
    assert len(labeled_edges) >= 8
    assert max(len(label) for label in labeled_edges) >= 12

    nested = graphs["nested_cluster_label_stack"].graph
    cluster_labels = list(nested.cluster_labels.values())
    assert len(cluster_labels) >= 3
    assert max(len(label) for label in cluster_labels) >= 30
    nested_edge_labels = [label for label in nested.edge_labels if label]
    assert len(nested_edge_labels) >= 2

    storm = graphs["small_label_storm"].graph
    assert sum(label is not None for label in storm.edge_labels) == storm.edge_index.shape[1]
    assert len(storm.cluster_labels) >= 2


def test_all_new_graphs_produce_valid_output() -> None:
    """Every new graph generator should return a valid graph with node sizes."""
    generators: list[Callable[[], DaguaGraph | tuple[DaguaGraph, DaguaGraph]]] = [
        lambda: make_scale_free(),
        lambda: make_grid(4, 5),
        lambda: make_complete_bipartite(),
        lambda: make_clustered_medium(),
        lambda: make_hub_and_spoke(),
        lambda: make_wide_single_layer(),
        lambda: make_sparse_dense_pair(),
        lambda: make_compound_dag(),
        lambda: make_long_skip_only(),
        lambda: make_parallel_cycles(),
        lambda: make_resnet_block(),
        lambda: make_transformer_full(),
        lambda: make_dependency_graph(),
        lambda: make_org_chart(),
        lambda: make_small_world(),
    ]

    for generator in generators:
        produced = generator()
        graphs = produced if isinstance(produced, tuple) else (produced,)
        for graph in graphs:
            assert hasattr(graph, "num_nodes")
            assert graph.num_nodes > 0
            assert graph.edge_index.shape[0] == 2
            assert graph.node_sizes is not None
            assert graph.node_sizes.shape == (graph.num_nodes, 2)


def test_new_graph_collection_entries_are_registered() -> None:
    """The synthetic collection should expose the new structural coverage graphs."""
    source = inspect.getsource(_expanded_structural_graphs)

    expected_name_literals = {
        "scale_free_ba_120",
        "ba_500",
        "ba_2000",
        "ba_5000",
        "grid_rect_6x8",
        "grid_20x20",
        "grid_50x50",
        "complete_bipartite_8x12",
        "clustered_medium_5x20",
        "hub_and_spoke_3x20",
        "wide_single_layer_1_50_1",
        "wide_1_100_1",
        "wide_3_50_3",
        "sparse_pair_50",
        "dense_pair_50",
        "compound_dag_5x30",
        "compound_10x20",
        "long_skip_only_24",
        "parallel_cycles_4x5",
        "resnet_stack_4x16",
        "transformer_full_4h_2l",
        "dependency_graph_100",
        "dependency_500",
        "org_chart_1_5_4_8",
        "org_chart_deep",
        "small_world_100",
        "small_world_500",
        "small_world_2000",
        "powerlaw_500",
        "powerlaw_2000",
    }
    assert all(name in source for name in expected_name_literals)

    expected_generator_calls = {
        "make_real_karate_graph()",
        "make_real_lesmis_graph()",
        "make_real_football_graph(seed=42)",
        "make_erdos_renyi(100, p=0.04, seed=42)",
        "make_erdos_renyi(500, p=0.008, seed=42)",
        "make_erdos_renyi(2000, p=0.003, seed=42)",
        "make_random_geometric(100, radius=0.25, seed=42)",
        "make_random_geometric(500, radius=0.1, seed=42)",
        "make_random_geometric(2000, radius=0.05, seed=42)",
        "make_sbm(4, 30, p_in=0.3, p_out=0.01, seed=42)",
        "make_sbm(5, 50, p_in=0.2, p_out=0.005, seed=42)",
        "make_sbm(8, 100, p_in=0.1, p_out=0.002, seed=42)",
    }
    assert all(call in source for call in expected_generator_calls)


def test_networkx_graph_generators_return_dag_orientations() -> None:
    """Undirected NetworkX imports should be oriented into acyclic graphs."""
    generated = [
        make_real_karate_graph(),
        make_real_lesmis_graph(),
        make_real_football_graph(seed=42),
        make_erdos_renyi(100, p=0.04, seed=42),
        make_random_geometric(100, radius=0.25, seed=42),
        make_sbm(4, 30, p_in=0.3, p_out=0.01, seed=42),
    ]

    for test_graph in generated:
        edge_index = test_graph.graph.edge_index
        assert test_graph.graph.num_nodes > 0
        assert edge_index.shape[0] == 2
        assert test_graph.graph.node_sizes is not None
        assert test_graph.graph.node_sizes.shape == (test_graph.graph.num_nodes, 2)
        if edge_index.shape[1] > 0:
            src = edge_index[0].tolist()
            tgt = edge_index[1].tolist()
            assert all(source < target for source, target in zip(src, tgt))

    football = generated[2]
    assert football.name == "real_football_115"
    assert football.graph.num_nodes == 115


def test_new_graph_generators_handle_edge_cases_and_grid_compatibility() -> None:
    """Edge cases should stay valid, and legacy grid scale calls should still work."""
    assert make_scale_free(0).num_nodes == 0
    assert make_complete_bipartite(1, 0).num_nodes == 1
    assert make_parallel_cycles(1, 1).edge_index.shape[1] == 1

    legacy_grid = make_grid(25, seed=7)
    assert hasattr(legacy_grid, "graph")
    assert legacy_grid.graph.num_nodes >= 4

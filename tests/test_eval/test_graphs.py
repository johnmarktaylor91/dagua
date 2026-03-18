"""Coverage tests for the benchmark graph corpus."""

from __future__ import annotations

import inspect
from collections import Counter, defaultdict, deque
from typing import Callable

from dagua.eval.graphs import (
    _expanded_structural_graphs,
    _synthetic_graphs,
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

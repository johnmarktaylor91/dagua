"""Coverage tests for the benchmark graph corpus."""

from __future__ import annotations

from collections import Counter, defaultdict, deque

from dagua.eval.graphs import _synthetic_graphs


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
    }
    assert expected_names <= names

    expected_tags = {
        "linear-shallow", "linear-deep", "wide-parallel", "skip-light", "skip-heavy",
        "tree", "diamond", "nested-shallow", "nested-deep", "mixed-width",
        "self-loops", "multi-edge", "disconnected", "large-sparse", "large-dense",
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
    edge_pairs = list(zip(clustered.graph.edge_index[0].tolist(), clustered.graph.edge_index[1].tolist()))
    counts = Counter(edge_pairs)
    assert max(counts.values()) >= 2
    assert clustered.graph.max_cluster_depth >= 1

    collage = graphs["disconnected_label_cycle_collage"]
    assert _component_count(collage.graph.edge_index, collage.graph.num_nodes) >= 2
    src = collage.graph.edge_index[0].tolist()
    tgt = collage.graph.edge_index[1].tolist()
    assert any(s == t for s, t in zip(src, tgt))

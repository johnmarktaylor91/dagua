"""Test graph collection for evaluation.

Provides a registry of test graphs spanning common and niche structure families:
linear-shallow, linear-deep, wide-parallel, skip-light, skip-heavy, tree,
diamond, nested-shallow, nested-deep, mixed-width, self-loops, multi-edge,
disconnected, large-sparse, and large-dense.

Sources: synthetic generators + TorchLens model traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import torch

from dagua.graph import DaguaGraph
from dagua.styles import ClusterStyle, EdgeStyle, NodeStyle


@dataclass
class TestGraph:
    """A test graph with metadata for evaluation."""
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


def _build_all_test_graphs() -> List[TestGraph]:
    """Build the complete test graph collection."""
    graphs = []
    graphs.extend(_synthetic_graphs())
    graphs.extend(_torchlens_graphs())
    return graphs


# ─── Synthetic Graph Generators ───────────────────────────────────────────────

def _synthetic_graphs() -> List[TestGraph]:
    """Generate synthetic test graphs covering structural categories."""
    graphs = []

    # 1. Linear chain (shallow)
    g = DaguaGraph.from_edge_list([
        ("input", "fc1"), ("fc1", "relu1"), ("relu1", "fc2"),
        ("fc2", "relu2"), ("relu2", "output"),
    ])
    graphs.append(TestGraph(
        name="linear_3layer_mlp",
        graph=g,
        tags={"linear-shallow"},
        description="Simple 3-layer MLP: input→fc→relu→fc→relu→output",
        expected_challenges="Trivial layout, baseline for comparison",
    ))

    # 2. Linear chain (deep)
    edges = []
    prev = "input"
    for i in range(20):
        curr = f"layer_{i}"
        edges.append((prev, curr))
        prev = curr
    edges.append((prev, "output"))
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(TestGraph(
        name="deep_chain_20",
        graph=g,
        tags={"linear-deep"},
        description="20-layer deep chain",
        expected_challenges="Very tall layout, edge length consistency",
    ))

    # 3. Wide parallel (inception-like)
    edges = [
        ("input", "branch_1x1"), ("input", "branch_3x3"), ("input", "branch_5x5"),
        ("input", "branch_pool"),
        ("branch_1x1", "concat"), ("branch_3x3", "concat"),
        ("branch_5x5", "concat"), ("branch_pool", "concat"),
        ("concat", "output"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(TestGraph(
        name="inception_block",
        graph=g,
        tags={"wide-parallel"},
        description="Inception-style 4-branch parallel block",
        expected_challenges="Wide layout, parallel branch alignment",
    ))

    # 4. Skip connections (light) — simple residual
    edges = [
        ("input", "conv1"), ("conv1", "bn1"), ("bn1", "relu1"),
        ("relu1", "conv2"), ("conv2", "bn2"),
        ("input", "skip"), ("skip", "add"),
        ("bn2", "add"), ("add", "relu2"), ("relu2", "output"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(TestGraph(
        name="residual_block",
        graph=g,
        tags={"skip-light", "diamond"},
        description="Single residual block with skip connection",
        expected_challenges="Skip connection routing, merge alignment",
    ))

    # 5. Skip connections (heavy) — DenseNet-style
    layers = ["input"] + [f"dense_{i}" for i in range(6)] + ["output"]
    edges = []
    for i in range(1, len(layers) - 1):
        for j in range(i):
            edges.append((layers[j], layers[i]))
    edges.append((layers[-2], layers[-1]))
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(TestGraph(
        name="densenet_block",
        graph=g,
        tags={"skip-heavy", "large-dense"},
        description="DenseNet-style fully connected block (6 layers)",
        expected_challenges="Many crossing edges, dense connectivity",
    ))

    # 6. Tree (decoder)
    edges = [
        ("root", "left"), ("root", "right"),
        ("left", "ll"), ("left", "lr"),
        ("right", "rl"), ("right", "rr"),
        ("ll", "lll"), ("ll", "llr"),
        ("lr", "lrl"), ("lr", "lrr"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(TestGraph(
        name="binary_tree",
        graph=g,
        tags={"tree"},
        description="Binary tree (depth 4)",
        expected_challenges="Symmetric layout, proper spreading",
    ))

    # 7. Diamond / U-Net encoder-decoder
    edges = [
        ("input", "enc1"), ("enc1", "enc2"), ("enc2", "enc3"),
        ("enc3", "bottleneck"),
        ("bottleneck", "dec3"), ("dec3", "dec2"), ("dec2", "dec1"),
        ("dec1", "output"),
        # Skip connections
        ("enc1", "dec1"), ("enc2", "dec2"), ("enc3", "dec3"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(TestGraph(
        name="unet_small",
        graph=g,
        tags={"diamond", "skip-light"},
        description="Small U-Net encoder-decoder with skip connections",
        expected_challenges="Symmetric layout, horizontal skip connections",
    ))

    # 8. Nested clusters (shallow)
    g = DaguaGraph.from_edge_list([
        ("input", "enc.conv1"), ("enc.conv1", "enc.relu"),
        ("enc.relu", "dec.conv2"), ("dec.conv2", "dec.relu"),
        ("dec.relu", "output"),
    ])
    g.add_cluster("encoder", [1, 2], label="Encoder")  # enc.conv1, enc.relu
    g.add_cluster("decoder", [3, 4], label="Decoder")  # dec.conv2, dec.relu
    graphs.append(TestGraph(
        name="nested_shallow_enc_dec",
        graph=g,
        tags={"nested-shallow"},
        description="Encoder-decoder with 2 clusters",
        expected_challenges="Cluster separation and labeling",
    ))

    # 9. Nested clusters (deep) — transformer-like
    nodes = [
        "input", "embed",
        "attn.q_proj", "attn.k_proj", "attn.v_proj",
        "attn.matmul", "attn.softmax", "attn.out_proj",
        "ff.fc1", "ff.relu", "ff.fc2",
        "add1", "add2", "norm1", "norm2", "output",
    ]
    edges = [
        ("input", "embed"),
        ("embed", "attn.q_proj"), ("embed", "attn.k_proj"), ("embed", "attn.v_proj"),
        ("attn.q_proj", "attn.matmul"), ("attn.k_proj", "attn.matmul"),
        ("attn.matmul", "attn.softmax"),
        ("attn.v_proj", "attn.softmax"),
        ("attn.softmax", "attn.out_proj"),
        ("attn.out_proj", "add1"),
        ("embed", "add1"),  # residual
        ("add1", "norm1"),
        ("norm1", "ff.fc1"), ("ff.fc1", "ff.relu"), ("ff.relu", "ff.fc2"),
        ("ff.fc2", "add2"),
        ("norm1", "add2"),  # residual
        ("add2", "norm2"),
        ("norm2", "output"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    # Find indices for cluster members
    node_idx = {name: i for i, name in enumerate(g.node_labels)}
    g.add_cluster("attention", [node_idx[n] for n in [
        "attn.q_proj", "attn.k_proj", "attn.v_proj",
        "attn.matmul", "attn.softmax", "attn.out_proj",
    ]], label="Multi-Head Attention")
    g.add_cluster("feedforward", [node_idx[n] for n in [
        "ff.fc1", "ff.relu", "ff.fc2",
    ]], label="Feed-Forward")
    graphs.append(TestGraph(
        name="transformer_layer",
        graph=g,
        tags={"nested-deep", "wide-parallel", "skip-light"},
        description="Single transformer layer with attention + FFN clusters",
        expected_challenges="Nested clusters, parallel Q/K/V branches, residuals",
    ))

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
    graphs.append(TestGraph(
        name="mixed_width_labels",
        graph=g,
        tags={"mixed-width", "skip-light"},
        description="Graph with very different node label widths",
        expected_challenges="Node sizing, alignment with width variation",
    ))

    # 11. Random DAG (medium sparse)
    g = _random_dag(50, 70, seed=42)
    graphs.append(TestGraph(
        name="random_dag_50",
        graph=g,
        tags={"large-sparse"},
        description="Random DAG: 50 nodes, ~70 edges",
        expected_challenges="General layout quality at medium scale",
    ))

    # 12. Random DAG (large sparse)
    g = _random_dag(200, 300, seed=42)
    graphs.append(TestGraph(
        name="random_dag_200",
        graph=g,
        tags={"large-sparse"},
        description="Random DAG: 200 nodes, ~300 edges",
        expected_challenges="Scalability, readability at scale",
    ))

    # 13. Grid-like DAG
    edges = []
    rows, cols = 5, 5
    for r in range(rows):
        for c in range(cols):
            node = f"n{r}_{c}"
            if c + 1 < cols:
                edges.append((node, f"n{r}_{c+1}"))
            if r + 1 < rows:
                edges.append((node, f"n{r+1}_{c}"))
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(TestGraph(
        name="grid_5x5",
        graph=g,
        tags={"diamond", "large-dense"},
        description="5x5 grid DAG with horizontal and vertical edges",
        expected_challenges="Grid regularity, many crossing opportunities",
    ))

    # 14. Multi-source multi-sink
    edges = []
    for i in range(4):
        for j in range(3):
            edges.append((f"src_{i}", f"mid_{j}"))
    for j in range(3):
        for k in range(4):
            edges.append((f"mid_{j}", f"sink_{k}"))
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(TestGraph(
        name="bipartite_4_3_4",
        graph=g,
        tags={"wide-parallel", "large-dense"},
        description="Bipartite: 4 sources → 3 middle → 4 sinks",
        expected_challenges="Edge crossing minimization, alignment",
    ))

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
    g.add_cluster("encoder", [idx["stem.conv"], idx["stage1.add"], idx["stage2.add"]], label="Encoder")
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
    graphs.append(TestGraph(
        name="hierarchical_residual_stage",
        graph=g,
        tags={"nested-deep", "skip-light"},
        description="Residual stack with 3-level cluster hierarchy",
        expected_challenges="Nested cluster containment plus residual edges crossing cluster boundaries",
    ))

    # 16. Recurrent-style feedback with explicit self-loop and cycle
    g = DaguaGraph.from_edge_list([
        ("input", "state_update"),
        ("state_prev", "state_update"),
        ("state_update", "state_proj"),
        ("state_proj", "output"),
        ("output", "state_prev"),
        ("state_proj", "state_proj"),
    ])
    graphs.append(TestGraph(
        name="recurrent_feedback_cell",
        graph=g,
        tags={"self-loops", "skip-light"},
        description="Small recurrent cell with feedback edge and explicit self-loop",
        expected_challenges="Cycle breaking, self-loop routing, compact feedback placement",
    ))

    # 17. Multi-edge bundle between repeated stages
    g = DaguaGraph()
    for node in ("src", "mid", "dst"):
        g.add_node(node)
    for _ in range(3):
        g.add_edge("src", "mid")
    for _ in range(2):
        g.add_edge("mid", "dst")
    g.add_edge("src", "dst")
    graphs.append(TestGraph(
        name="parallel_multiedge_bundle",
        graph=g,
        tags={"multi-edge", "diamond"},
        description="Duplicate edges between the same node pairs plus a direct bypass edge",
        expected_challenges="Multi-edge routing separation and duplicate-edge stability",
    ))

    # 18. Disconnected components mixing common motifs
    g = DaguaGraph.from_edge_list([
        ("enc_in", "enc_conv"), ("enc_conv", "enc_relu"), ("enc_relu", "enc_out"),
        ("res_in", "res_conv1"), ("res_conv1", "res_conv2"), ("res_in", "res_add"),
        ("res_conv2", "res_add"), ("res_add", "res_out"),
    ])
    graphs.append(TestGraph(
        name="disconnected_encoder_residual",
        graph=g,
        tags={"disconnected", "skip-light"},
        description="Two disconnected components: a linear encoder and a residual block",
        expected_challenges="Component packing, stable spacing across disconnected subgraphs",
    ))

    # 19. Mixture-of-experts style sparse routing
    edges = [
        ("input", "embed"), ("embed", "router"),
        ("router", "expert_0"), ("router", "expert_3"),
        ("embed", "expert_1"), ("embed", "expert_2"),
        ("expert_0", "combine"), ("expert_1", "combine"),
        ("expert_2", "combine"), ("expert_3", "combine"),
        ("combine", "output"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.add_cluster("experts", [idx[f"expert_{i}"] for i in range(4)], label="Experts")
    graphs.append(TestGraph(
        name="moe_router_sparse",
        graph=g,
        tags={"wide-parallel", "large-dense", "nested-shallow"},
        description="Mixture-of-experts routing with sparse fan-out and dense fan-in",
        expected_challenges="Wide expert fan-out, merge alignment, cluster labeling",
    ))

    # 20. Ragged multiscale pyramid with lateral skips
    edges = [
        ("input", "p3"), ("p3", "p4"), ("p4", "p5"),
        ("p5", "top_down_5"), ("top_down_5", "merge_4"),
        ("p4", "merge_4"), ("merge_4", "top_down_4"),
        ("top_down_4", "merge_3"), ("p3", "merge_3"),
        ("merge_3", "detect_small"), ("merge_4", "detect_medium"), ("top_down_5", "detect_large"),
        ("detect_small", "out"), ("detect_medium", "out"), ("detect_large", "out"),
    ]
    g = DaguaGraph.from_edge_list(edges)
    graphs.append(TestGraph(
        name="ragged_feature_pyramid",
        graph=g,
        tags={"skip-heavy", "wide-parallel", "diamond"},
        description="Feature-pyramid style graph with ragged lateral merges",
        expected_challenges="Long lateral skips, fan-in across scales, uneven branch widths",
    ))

    # 21. Kitchen sink model: nested clusters + skips + loops + multi-edges + wide fanout
    g = DaguaGraph()
    for node in [
        "input", "stem.conv", "stem.norm", "stem.act",
        "router", "expert_a.0", "expert_a.1", "expert_b.0", "expert_b.1",
        "expert_c.0", "expert_c.1", "merge", "residual_add",
        "memory", "memory", "feedback_gate", "head.norm",
        "classifier", "aux_head", "output",
    ]:
        if node not in g._id_to_index:
            g.add_node(node)

    for edge in [
        ("input", "stem.conv"), ("stem.conv", "stem.norm"), ("stem.norm", "stem.act"),
        ("stem.act", "router"),
        ("router", "expert_a.0"), ("router", "expert_b.0"), ("router", "expert_c.0"),
        ("expert_a.0", "expert_a.1"), ("expert_b.0", "expert_b.1"), ("expert_c.0", "expert_c.1"),
        ("expert_a.1", "merge"), ("expert_b.1", "merge"), ("expert_c.1", "merge"),
        ("stem.act", "residual_add"), ("merge", "residual_add"),
        ("residual_add", "head.norm"), ("head.norm", "classifier"), ("classifier", "output"),
        ("head.norm", "aux_head"), ("aux_head", "output"),
        ("residual_add", "feedback_gate"), ("feedback_gate", "memory"), ("memory", "router"),
        ("memory", "memory"),
    ]:
        g.add_edge(*edge)
    # Duplicate routing edge to force multi-edge handling.
    g.add_edge("router", "expert_b.0")

    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.add_cluster("backbone", [idx["stem.conv"], idx["stem.norm"], idx["stem.act"]], label="Backbone")
    g.add_cluster(
        "experts",
        [idx["expert_a.0"], idx["expert_a.1"], idx["expert_b.0"], idx["expert_b.1"], idx["expert_c.0"], idx["expert_c.1"]],
        label="Experts",
    )
    g.add_cluster("expert_a", [idx["expert_a.0"], idx["expert_a.1"]], label="Expert A", parent="experts")
    g.add_cluster("expert_b", [idx["expert_b.0"], idx["expert_b.1"]], label="Expert B", parent="experts")
    g.add_cluster("expert_c", [idx["expert_c.0"], idx["expert_c.1"]], label="Expert C", parent="experts")
    g.add_cluster("expert_b.inner", [idx["expert_b.1"]], label="Expert B / Inner", parent="expert_b")
    g.add_cluster("heads", [idx["head.norm"], idx["classifier"], idx["aux_head"]], label="Heads")
    graphs.append(TestGraph(
        name="kitchen_sink_hybrid_net",
        graph=g,
        tags={"nested-deep", "skip-heavy", "wide-parallel", "self-loops", "multi-edge", "mixed-width"},
        description="Overloaded hybrid graph combining experts, residuals, feedback, nested clusters, aux head, and varied labels",
        expected_challenges="Full-stack stress test: nested clusters, loops, duplicate edges, wide branches, residuals, and mixed label widths",
    ))

    # 22. Kitchen sink system graph: disconnected subsystems + dense handoffs + hierarchy
    g = DaguaGraph.from_edge_list([
        ("api.gateway", "auth.validate"), ("auth.validate", "router.dispatch"),
        ("router.dispatch", "svc.search"), ("router.dispatch", "svc.reco"), ("router.dispatch", "svc.ads"),
        ("svc.search", "join.rank"), ("svc.reco", "join.rank"), ("svc.ads", "join.rank"),
        ("join.rank", "cache.write"), ("cache.write", "response.serialize"),
        ("response.serialize", "response.emit"),
        ("join.rank", "metrics.aggregate"), ("metrics.aggregate", "alerts.loop"),
        ("alerts.loop", "metrics.aggregate"), ("alerts.loop", "alerts.loop"),
        ("offline.ingest", "offline.train"), ("offline.train", "offline.eval"),
        ("offline.eval", "model.registry"), ("model.registry", "router.dispatch"),
        ("model.registry", "svc.reco"),
        ("audit.ingest", "audit.report"),
    ])
    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.add_cluster("online", [idx[n] for n in [
        "api.gateway", "auth.validate", "router.dispatch", "svc.search", "svc.reco",
        "svc.ads", "join.rank", "cache.write", "response.serialize", "response.emit",
    ]], label="Online Path")
    g.add_cluster("services", [idx[n] for n in ["svc.search", "svc.reco", "svc.ads"]], label="Services", parent="online")
    g.add_cluster("observability", [idx[n] for n in ["metrics.aggregate", "alerts.loop"]], label="Observability")
    g.add_cluster("offline", [idx[n] for n in ["offline.ingest", "offline.train", "offline.eval", "model.registry"]], label="Offline")
    g.add_cluster("audit", [idx[n] for n in ["audit.ingest", "audit.report"]], label="Audit")
    graphs.append(TestGraph(
        name="kitchen_sink_platform_graph",
        graph=g,
        tags={"nested-deep", "disconnected", "self-loops", "wide-parallel", "skip-heavy", "mixed-width"},
        description="Platform-style graph mixing online services, offline training, observability loops, and cross-system handoffs",
        expected_challenges="Disconnected subsystems, long handoff edges, nested service clusters, explicit loops, and varied label widths",
    ))

    # 23. Extreme mixed-width labels with uneven branch depths
    edges = [
        ("x", "TokenEmbedding(vocab_size=50257, embedding_dim=4096)"),
        ("TokenEmbedding(vocab_size=50257, embedding_dim=4096)", "LayerNorm(normalized_shape=(4096,), eps=1e-05)"),
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
    graphs.append(TestGraph(
        name="extreme_mixed_width_transformer",
        graph=g,
        tags={"mixed-width", "wide-parallel", "skip-light"},
        description="Extreme short-vs-long labels inside a transformer-style residual block",
        expected_challenges="Very uneven node widths, branch alignment, and label-driven spacing",
    ))

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
    graphs.append(TestGraph(
        name="hub_fanout_label_skew",
        graph=g,
        tags={"mixed-width", "wide-parallel", "diamond"},
        description="Single hub with strongly uneven branch label widths and an asymmetric late merge",
        expected_challenges="Fanout distribution, visual balance, and asymmetric merge routing under width skew",
    ))

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
    g.add_cluster("encoder", [idx["encoder.stage_1_attention_projection"], idx["encoder.stage_1_feedforward"]], label="Encoder")
    g.add_cluster("decoder", [idx["decoder.cross_attention_query"], idx["decoder.cross_attention_key_value"], idx["decoder.merge"]], label="Decoder")
    g.add_cluster("decoder.cross_attention", [idx["decoder.cross_attention_query"], idx["decoder.cross_attention_key_value"]], label="Cross Attention", parent="decoder")
    graphs.append(TestGraph(
        name="clustered_longlabel_handoffs",
        graph=g,
        tags={"mixed-width", "nested-deep", "multi-edge", "skip-light"},
        description="Nested clusters with long labels and repeated inter-cluster handoff edges",
        expected_challenges="Cluster sizing, duplicate inter-cluster routing, and label-width imbalance",
    ))

    # 26. Stressful disconnected collage of tiny, huge, and cyclic components
    g = DaguaGraph.from_edge_list([
        ("a", "b"),
        ("StandaloneSuperLongLabelForAnOtherwiseTinyChainNode", "tail"),
        ("cycle.start", "cycle.mid"),
        ("cycle.mid", "cycle.end"),
        ("cycle.end", "cycle.start"),
        ("cycle.end", "cycle.end"),
    ])
    graphs.append(TestGraph(
        name="disconnected_label_cycle_collage",
        graph=g,
        tags={"mixed-width", "disconnected", "self-loops"},
        description="Disconnected collage mixing a tiny chain, a huge label, a directed cycle, and a self-loop",
        expected_challenges="Component packing under wildly different local scales and cyclic structures",
    ))

    # 27. Mixed node shapes and routing modes
    g = DaguaGraph.from_edge_list([
        ("input", "ellipse_norm"),
        ("ellipse_norm", "diamond_gate"),
        ("ellipse_norm", "roundrect_path"),
        ("diamond_gate", "merge"),
        ("roundrect_path", "merge"),
        ("merge", "circle_sink"),
    ])
    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.node_styles[idx["input"]] = NodeStyle(shape="rect")
    g.node_styles[idx["ellipse_norm"]] = NodeStyle(shape="ellipse")
    g.node_styles[idx["diamond_gate"]] = NodeStyle(shape="diamond")
    g.node_styles[idx["roundrect_path"]] = NodeStyle(shape="roundrect")
    g.node_styles[idx["circle_sink"]] = NodeStyle(shape="circle")
    g.edge_styles[0] = EdgeStyle(routing="straight", port_style="center", curvature=0.0)
    g.edge_styles[1] = EdgeStyle(routing="ortho", port_style="distributed", curvature=0.2)
    g.edge_styles[2] = EdgeStyle(routing="bezier", port_style="center", curvature=0.7)
    graphs.append(TestGraph(
        name="shape_and_routing_matrix",
        graph=g,
        tags={"mixed-width", "diamond", "wide-parallel"},
        description="Mixed node shapes with straight, ortho, and bezier edge routing modes",
        expected_challenges="Shape-aware ports, mixed routing styles, and asymmetric merge geometry",
    ))

    # 28. Center-port hub with aggressive back-edge and cycle pressure
    g = DaguaGraph.from_edge_list([
        ("hub", "decode_a"),
        ("hub", "decode_b"),
        ("hub", "decode_c"),
        ("decode_a", "join"),
        ("decode_b", "join"),
        ("decode_c", "join"),
        ("join", "head"),
        ("head", "hub"),
        ("decode_b", "decode_b"),
    ])
    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.node_styles[idx["hub"]] = NodeStyle(shape="ellipse")
    for e_idx in range(len(g.edge_styles)):
        g.edge_styles[e_idx] = EdgeStyle(port_style="center", routing="bezier", curvature=0.6)
    graphs.append(TestGraph(
        name="center_port_backedge_hub",
        graph=g,
        tags={"self-loops", "skip-heavy", "wide-parallel"},
        description="Center-port fanout hub with explicit back-edge and self-loop",
        expected_challenges="Cycle breaking around a dominant hub and crowded center-port routing",
    ))

    # 29. Cluster member style stress with inter-cluster edge variety
    g = DaguaGraph.from_edge_list([
        ("ingest", "prep.clean"),
        ("prep.clean", "prep.batch"),
        ("prep.batch", "core.encode"),
        ("core.encode", "core.route"),
        ("core.route", "core.decode"),
        ("core.decode", "post.merge"),
        ("prep.batch", "post.merge"),
        ("post.merge", "serve"),
    ])
    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.add_cluster("prep", [idx["prep.clean"], idx["prep.batch"]], label="Prep")
    g.add_cluster("core", [idx["core.encode"], idx["core.route"], idx["core.decode"]], label="Core")
    g.cluster_styles["core"] = ClusterStyle(
        member_node_style=NodeStyle(shape="diamond"),
        member_edge_style=EdgeStyle(routing="ortho", port_style="center", curvature=0.1),
    )
    graphs.append(TestGraph(
        name="cluster_member_style_stress",
        graph=g,
        tags={"nested-shallow", "skip-light", "diamond"},
        description="Clusters with member style overrides that alter node shapes and edge routing defaults",
        expected_challenges="Cluster-scoped style cascades combined with cross-cluster skip edges",
    ))

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
        g.add_edge(src, tgt, label=label, style=EdgeStyle(label_position=0.45, label_avoidance=True))
    graphs.append(TestGraph(
        name="edge_label_braid",
        graph=g,
        tags={"wide-parallel", "diamond", "mixed-width"},
        description="Many competing edge labels packed around a braid of crossings and a central merge",
        expected_challenges="Edge-label collision avoidance around dense crossing-prone merges",
    ))

    # 31. Deep nested cluster labels with long titles and central labeled handoffs
    g = DaguaGraph.from_edge_list([
        ("input", "encoder.stage0"),
        ("encoder.stage0", "encoder.stage1.attention"),
        ("encoder.stage1.attention", "encoder.stage1.mlp"),
        ("encoder.stage1.mlp", "bridge"),
        ("bridge", "decoder.stage0.cross_attention"),
        ("decoder.stage0.cross_attention", "decoder.stage0.ffn"),
        ("decoder.stage0.ffn", "output"),
    ])
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
    graphs.append(TestGraph(
        name="nested_cluster_label_stack",
        graph=g,
        tags={"nested-deep", "mixed-width", "skip-light"},
        description="Long nested cluster labels combined with central inter-cluster edge labels",
        expected_challenges="Cluster title stacking and edge-label placement near cluster boundaries",
    ))

    # 32. Label storm: edge labels plus cluster labels competing in a small graph
    g = DaguaGraph.from_edge_list([
        ("input", "prep"),
        ("prep", "branch_left"),
        ("prep", "branch_right"),
        ("branch_left", "join"),
        ("branch_right", "join"),
        ("join", "output"),
    ])
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
    g.add_cluster("Prep Cluster", [idx["prep"], idx["branch_left"], idx["branch_right"]], label="Prep Cluster")
    g.add_cluster("Join Cluster", [idx["join"]], label="Join Cluster")
    graphs.append(TestGraph(
        name="small_label_storm",
        graph=g,
        tags={"mixed-width", "wide-parallel", "nested-shallow"},
        description="Compact graph where edge labels and cluster labels all compete for the same area",
        expected_challenges="Label crowding in a small footprint with both edge and cluster annotations",
    ))

    # 33. Deep ladder with many long residual bridges across stages
    edges = []
    for i in range(8):
        edges.extend([
            (f"stage{i}.main", f"stage{i}.norm"),
            (f"stage{i}.norm", f"stage{i}.act"),
        ])
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
    graphs.append(TestGraph(
        name="long_range_residual_ladder",
        graph=g,
        tags={"linear-deep", "skip-heavy", "wide-parallel"},
        description="Deep ladder with many long residual bridges that leap multiple stages ahead",
        expected_challenges="Long-span skip routing, preserving ladder readability, and avoiding bridge tangles",
    ))

    # 34. Interleaved sibling clusters with dense cross-talk
    g = DaguaGraph.from_edge_list([
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
    ])
    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.add_cluster("system", [idx["enc.a0"], idx["enc.a1"], idx["enc.a2"], idx["enc.b0"], idx["enc.b1"], idx["enc.b2"], idx["decoder.left"], idx["decoder.right"], idx["decoder.merge"]], label="System")
    g.add_cluster("encoder", [idx["enc.a0"], idx["enc.a1"], idx["enc.a2"], idx["enc.b0"], idx["enc.b1"], idx["enc.b2"]], label="Encoder", parent="system")
    g.add_cluster("encoder.path_a", [idx["enc.a0"], idx["enc.a1"], idx["enc.a2"]], label="Path A", parent="encoder")
    g.add_cluster("encoder.path_b", [idx["enc.b0"], idx["enc.b1"], idx["enc.b2"]], label="Path B", parent="encoder")
    g.add_cluster("decoder", [idx["decoder.left"], idx["decoder.right"], idx["decoder.merge"]], label="Decoder", parent="system")
    graphs.append(TestGraph(
        name="interleaved_cluster_crosstalk",
        graph=g,
        tags={"nested-deep", "skip-heavy", "wide-parallel"},
        description="Sibling clusters whose internal paths repeatedly cross-talk before merging into a decoder",
        expected_challenges="Keeping sibling cluster identity legible while routing many inter-cluster cross-links",
    ))

    # 35. Asymmetric hourglass with a dominant hub and lopsided late merge
    g = DaguaGraph.from_edge_list([
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
    ])
    graphs.append(TestGraph(
        name="asymmetric_hourglass_hub",
        graph=g,
        tags={"diamond", "wide-parallel", "skip-light"},
        description="Multi-source hourglass with a dominant hub, one thin path, and one much fatter late branch",
        expected_challenges="Visual balance around a hub and preserving hourglass structure despite strong asymmetry",
    ))

    # 36. Multi-scale skip cascade with cross-resolution handoffs
    g = DaguaGraph.from_edge_list([
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
    ])
    idx = {name: i for i, name in enumerate(g.node_labels)}
    g.add_cluster("bottom_up", [idx["stem"], idx["p2"], idx["p3"], idx["p4"], idx["p5"]], label="Bottom-Up")
    g.add_cluster("top_down", [idx["topdown4"], idx["topdown3"], idx["topdown2"]], label="Top-Down")
    g.add_cluster("heads", [idx["detect_large"], idx["detect_mid"], idx["detect_small"], idx["detect_tiny"]], label="Heads")
    graphs.append(TestGraph(
        name="multiscale_skip_cascade",
        graph=g,
        tags={"skip-heavy", "nested-shallow", "wide-parallel"},
        description="Feature-pyramid-style graph with repeated cross-resolution skips and four detection heads",
        expected_challenges="Cross-scale skip routing, head alignment, and preserving the cascade structure",
    ))

    # 37. Near-layered graph with alternating braids and back-pressure
    g = DaguaGraph.from_edge_list([
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
    ])
    graphs.append(TestGraph(
        name="braided_feedback_tails",
        graph=g,
        tags={"skip-heavy", "diamond", "linear-deep"},
        description="Alternating braid that looks layered at first, then folds back through a late feedback tail",
        expected_challenges="Crossing-heavy braid alignment plus a late back-pressure loop near the sink",
    ))

    # 38. Very wide front half collapsing into one asymmetric late merge
    g = DaguaGraph.from_edge_list([
        ("input", "w0"), ("input", "w1"), ("input", "w2"), ("input", "w3"), ("input", "w4"),
        ("w0", "m0"), ("w1", "m1"), ("w2", "m2"), ("w3", "m3"), ("w4", "m4"),
        ("m0", "join.left"), ("m1", "join.left"), ("m2", "join.center"), ("m3", "join.right"), ("m4", "join.right"),
        ("w0", "late.merge"), ("w2", "late.merge"), ("w4", "late.merge"),
        ("join.left", "late.merge"), ("join.center", "late.merge"), ("join.right", "late.merge"),
        ("late.merge", "head"), ("head", "output"),
    ])
    graphs.append(TestGraph(
        name="width_skew_late_merge",
        graph=g,
        tags={"wide-parallel", "skip-heavy", "diamond"},
        description="Very wide early fan-out that collapses into an asymmetric late merge with long direct skips",
        expected_challenges="Balancing a wide first half against a compact sink while keeping long skips readable",
    ))

    # 39. Two nearly symmetric residual trunks with one broken branch
    g = DaguaGraph.from_edge_list([
        ("input", "left.0"), ("input", "right.0"),
        ("left.0", "left.1"), ("left.1", "left.2"), ("left.2", "merge"),
        ("right.0", "right.1"), ("right.1", "right.2"), ("right.2", "merge"),
        ("input", "merge"),
        ("left.0", "left.skip"), ("left.skip", "merge"),
        ("right.0", "right.skip"), ("right.skip", "right.2"),
        ("right.1", "breakout"), ("breakout", "merge"),
        ("merge", "output"),
    ])
    graphs.append(TestGraph(
        name="broken_symmetry_residual_pair",
        graph=g,
        tags={"skip-heavy", "diamond", "wide-parallel"},
        description="Two almost-symmetric residual trunks where one arm breaks pattern just before the merge",
        expected_challenges="Preserving the repeated pattern while making the broken branch obvious instead of messy",
    ))

    # 40. Deep spine with a dominant hub spraying long skips across layers
    g = DaguaGraph.from_edge_list([
        ("input", "s0"), ("s0", "s1"), ("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "s5"), ("s5", "output"),
        ("s1", "hub"), ("hub", "x0"), ("hub", "x1"), ("hub", "x2"), ("hub", "x3"),
        ("x0", "s3"), ("x1", "s4"), ("x2", "s5"), ("x3", "output"),
        ("hub", "output"), ("s0", "x2"), ("s2", "x3"),
    ])
    graphs.append(TestGraph(
        name="hub_skip_superfan",
        graph=g,
        tags={"linear-deep", "skip-heavy", "wide-parallel"},
        description="Deep backbone interrupted by a dominant hub that fans long skips into late layers and the sink",
        expected_challenges="Avoiding a central skip knot around a high-degree hub while preserving backbone flow",
    ))

    return graphs


def _random_dag(n_nodes: int, n_edges: int, seed: int = 42) -> DaguaGraph:
    """Generate a random DAG by sampling edges that respect topological order."""
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

    return DaguaGraph.from_edge_list(list(edges), num_nodes=n_nodes)


# ─── TorchLens Graph Extractors ──────────────────────────────────────────────

def _torchlens_graphs() -> List[TestGraph]:
    """Extract test graphs from TorchLens model traces.

    Returns empty list if TorchLens is not available.
    """
    try:
        import torchlens as tl
        import torch.nn as nn
    except ImportError:
        return []

    graphs = []

    # 1. Simple MLP
    try:
        model = nn.Sequential(
            nn.Linear(10, 20), nn.ReLU(),
            nn.Linear(20, 10), nn.ReLU(),
            nn.Linear(10, 5),
        )
        x = torch.randn(1, 10)
        log = tl.log_forward_pass(model, x, vis_mode="none")
        g = DaguaGraph.from_torchlens(log)
        graphs.append(TestGraph(
            name="tl_mlp_3layer",
            graph=g,
            tags={"linear-shallow"},
            source="torchlens",
            description="TorchLens: 3-layer MLP",
        ))
    except Exception:
        pass

    # 2. CNN
    try:
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 10),
        )
        x = torch.randn(1, 3, 32, 32)
        log = tl.log_forward_pass(model, x, vis_mode="none")
        g = DaguaGraph.from_torchlens(log)
        graphs.append(TestGraph(
            name="tl_cnn_small",
            graph=g,
            tags={"linear-shallow"},
            source="torchlens",
            description="TorchLens: Small CNN (2 conv + fc)",
        ))
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

        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            ResBlock(16),
            ResBlock(16),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        x = torch.randn(1, 3, 32, 32)
        log = tl.log_forward_pass(model, x, vis_mode="none")
        g = DaguaGraph.from_torchlens(log)
        graphs.append(TestGraph(
            name="tl_resnet_2block",
            graph=g,
            tags={"skip-light", "nested-shallow"},
            source="torchlens",
            description="TorchLens: 2 residual blocks",
        ))
    except Exception:
        pass

    # 4. Simple Transformer encoder layer
    try:
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        transformer_model = nn.TransformerEncoder(layer, num_layers=1)
        x = torch.randn(1, 10, 64)
        log = tl.log_forward_pass(transformer_model, x, vis_mode="none")
        g = DaguaGraph.from_torchlens(log)
        graphs.append(TestGraph(
            name="tl_transformer_1layer",
            graph=g,
            tags={"nested-deep", "wide-parallel", "skip-light"},
            source="torchlens",
            description="TorchLens: Single transformer encoder layer",
        ))
    except Exception:
        pass

    # ─── Extended TorchLens architectures ────────────────────────────────

    # Import example models from TorchLens test suite
    try:
        from tests.example_models import (
            NestedModules,
            SimpleBranching,
            DiamondLoop,
            LongLoop,
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
            model = nested_modules_cls()
            x = torch.randn(5)
            log = tl.log_forward_pass(model, x, vis_mode="none")
            g = DaguaGraph.from_torchlens(log)
            graphs.append(TestGraph(
                name="tl_nested_modules",
                graph=g,
                tags={"nested-deep"},
                source="torchlens",
                description="TorchLens: 4-level nested module hierarchy",
                expected_challenges="Deep module nesting, cluster layout",
            ))
        except Exception:
            pass

    # 6. Branching (3-way split and merge)
    if simple_branching_cls is not None:
        try:
            model = simple_branching_cls()
            x = torch.randn(5)
            log = tl.log_forward_pass(model, x, vis_mode="none")
            g = DaguaGraph.from_torchlens(log)
            graphs.append(TestGraph(
                name="tl_branching",
                graph=g,
                tags={"wide-parallel"},
                source="torchlens",
                description="TorchLens: 3-way branching with merge",
                expected_challenges="Branch alignment, merge point",
            ))
        except Exception:
            pass

    # 7. Diamond loop (split → sin/cos → merge, repeated)
    if diamond_loop_cls is not None:
        try:
            model = diamond_loop_cls()
            x = torch.randn(5)
            log = tl.log_forward_pass(model, x, vis_mode="none")
            g = DaguaGraph.from_torchlens(log)
            graphs.append(TestGraph(
                name="tl_diamond_loop",
                graph=g,
                tags={"diamond", "skip-light"},
                source="torchlens",
                description="TorchLens: Diamond pattern with loop iterations",
                expected_challenges="Repeated diamond pattern, loop detection",
            ))
        except Exception:
            pass

    # 8. Long loop (20 iterations of Linear + ReLU)
    if long_loop_cls is not None:
        try:
            model = long_loop_cls()
            x = torch.randn(5)
            log = tl.log_forward_pass(model, x, vis_mode="none")
            g = DaguaGraph.from_torchlens(log)
            graphs.append(TestGraph(
                name="tl_long_loop",
                graph=g,
                tags={"linear-deep"},
                source="torchlens",
                description="TorchLens: 20-iteration loop (Linear+ReLU)",
                expected_challenges="Very deep chain from loop unrolling",
            ))
        except Exception:
            pass

    # 9. ASPP model (multi-scale parallel branches)
    try:
        from tests.example_models import ASPPModel
        aspp_model = ASPPModel(in_channels=3, mid=8, rates=(1, 6, 12))
        x = torch.randn(2, 3, 32, 32)
        log = tl.log_forward_pass(aspp_model, x, vis_mode="none")
        g = DaguaGraph.from_torchlens(log)
        graphs.append(TestGraph(
            name="tl_aspp",
            graph=g,
            tags={"wide-parallel", "nested-shallow"},
            source="torchlens",
            description="TorchLens: Atrous Spatial Pyramid Pooling (4 parallel branches)",
            expected_challenges="Multi-scale parallel branches, wide layout",
        ))
    except Exception:
        pass

    # 10. Feature Pyramid Network (bidirectional multi-scale)
    try:
        from tests.example_models import FeaturePyramidNet
        fpn_model = FeaturePyramidNet()
        x = torch.randn(2, 3, 32, 32)
        log = tl.log_forward_pass(fpn_model, x, vis_mode="none")
        g = DaguaGraph.from_torchlens(log)
        graphs.append(TestGraph(
            name="tl_fpn",
            graph=g,
            tags={"diamond", "skip-light", "nested-shallow"},
            source="torchlens",
            description="TorchLens: Feature Pyramid Network with lateral connections",
            expected_challenges="Bidirectional flow, lateral skip connections",
        ))
    except Exception:
        pass

    # 11. Attention block (Q/K/V self-attention)
    try:
        from tests.example_models import _AttentionBlock
        attention_model = _AttentionBlock(dim=64)
        x = torch.randn(2, 10, 64)
        log = tl.log_forward_pass(attention_model, x, vis_mode="none")
        g = DaguaGraph.from_torchlens(log)
        graphs.append(TestGraph(
            name="tl_attention",
            graph=g,
            tags={"wide-parallel", "nested-shallow"},
            source="torchlens",
            description="TorchLens: Self-attention with Q/K/V projections",
            expected_challenges="Triple parallel branches, matmul merge",
        ))
    except Exception:
        pass

    # 12. RandomGraphModel (controllable stress test)
    try:
        from tests.example_models import RandomGraphModel
        random_graph_model = RandomGraphModel(
            target_nodes=50, nesting_depth=2, seed=42, branch_probability=0.3, hidden_dim=64
        )
        x = torch.randn(2, 64)
        log = tl.log_forward_pass(random_graph_model, x, vis_mode="none")
        g = DaguaGraph.from_torchlens(log)
        graphs.append(TestGraph(
            name="tl_random_50",
            graph=g,
            tags={"nested-deep", "skip-light", "wide-parallel"},
            source="torchlens",
            description="TorchLens: Random architecture (~50 nodes, mixed patterns)",
            expected_challenges="Unpredictable topology, mixed branch/skip/nest",
        ))
    except Exception:
        pass

    return graphs


# ─── Scale Graph Generators ──────────────────────────────────────────────────


def make_chain(n: int, seed: int = 42) -> TestGraph:
    """Linear chain: 0→1→2→...→n-1."""
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
    """Layered DAG with fixed width per layer."""
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

    edges_set: set = set()
    for i in range(len(layers) - 1):
        for node in layers[i]:
            k = min(rng.randint(1, 3), len(layers[i + 1]))
            targets = rng.sample(layers[i + 1], k)
            for t in targets:
                edges_set.add((node, t))

    g = DaguaGraph.from_edge_index(
        torch.zeros(2, 0, dtype=torch.long), num_nodes=n
    )
    if edges_set:
        el = list(edges_set)
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
    """Random DAG with ~n*density edges."""
    import random

    rng = random.Random(seed)
    n_edges = int(n * density)

    g = DaguaGraph.from_edge_index(
        torch.zeros(2, 0, dtype=torch.long), num_nodes=n
    )
    edges_set: set = set()
    attempts = 0
    while len(edges_set) < n_edges and attempts < n_edges * 20:
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, min(i + max(n // 5, 10), n - 1))
        edges_set.add((i, j))
        attempts += 1

    if edges_set:
        el = list(edges_set)
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


def make_grid(n: int, seed: int = 42) -> TestGraph:
    """Grid DAG: √n × √n grid with edges pointing right and down."""
    import math

    cols = max(int(math.sqrt(n)), 2)
    rows = max(n // cols, 2)
    actual_n = rows * cols

    src, tgt = [], []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            # right edge
            if c + 1 < cols:
                src.append(idx)
                tgt.append(idx + 1)
            # down edge
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

    edge_index = torch.tensor([src, tgt], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    g = DaguaGraph.from_edge_index(edge_index, num_nodes=actual_n)
    return TestGraph(
        name=f"sparse_layered_{actual_n}",
        graph=g,
        tags={"large-sparse", "skip-light", "scale"},
        description=f"Layered DAG: {actual_n} nodes, {n_layers} layers, ~{edges_per_node} edges/node",
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

    edge_index = torch.tensor([src, tgt], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    g = DaguaGraph.from_edge_index(edge_index, num_nodes=n)
    return TestGraph(
        name=f"powerlaw_dag_{n}",
        graph=g,
        tags={"large-sparse", "scale"},
        description=f"Power-law DAG: {n} nodes, preferential attachment",
    )


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

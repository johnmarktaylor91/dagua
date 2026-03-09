"""Test graph collection for evaluation.

Provides a registry of test graphs covering 14 structural categories:
linear-shallow, linear-deep, wide-parallel, skip-light, skip-heavy,
tree, diamond, nested-shallow, nested-deep, mixed-width, self-loops,
multi-edge, large-sparse, large-dense.

Sources: synthetic generators + TorchLens model traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch

from dagua.graph import DaguaGraph


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

    return graphs


def _random_dag(n_nodes: int, n_edges: int, seed: int = 42) -> DaguaGraph:
    """Generate a random DAG by sampling edges that respect topological order."""
    import random
    rng = random.Random(seed)

    node_names = [f"n{i}" for i in range(n_nodes)]
    edges = set()
    attempts = 0
    while len(edges) < n_edges and attempts < n_edges * 10:
        i = rng.randint(0, n_nodes - 2)
        j = rng.randint(i + 1, n_nodes - 1)
        edges.add((node_names[i], node_names[j]))
        attempts += 1

    return DaguaGraph.from_edge_list(list(edges))


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
        model = nn.TransformerEncoder(layer, num_layers=1)
        x = torch.randn(1, 10, 64)
        log = tl.log_forward_pass(model, x, vis_mode="none")
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
    except ImportError:
        # Fall back: define inline if TorchLens test models not importable
        NestedModules = SimpleBranching = DiamondLoop = LongLoop = None

    # 5. Nested module hierarchy (4-level nesting)
    if NestedModules is not None:
        try:
            model = NestedModules()
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
    if SimpleBranching is not None:
        try:
            model = SimpleBranching()
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
    if DiamondLoop is not None:
        try:
            model = DiamondLoop()
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
    if LongLoop is not None:
        try:
            model = LongLoop()
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
        model = ASPPModel(in_channels=3, mid=8, rates=(1, 6, 12))
        x = torch.randn(2, 3, 32, 32)
        log = tl.log_forward_pass(model, x, vis_mode="none")
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
        model = FeaturePyramidNet()
        x = torch.randn(2, 3, 32, 32)
        log = tl.log_forward_pass(model, x, vis_mode="none")
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
        model = _AttentionBlock(dim=64)
        x = torch.randn(2, 10, 64)
        log = tl.log_forward_pass(model, x, vis_mode="none")
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
        model = RandomGraphModel(target_nodes=50, nesting_depth=2, seed=42, branch_probability=0.3, hidden_dim=64)
        x = torch.randn(2, 64)
        log = tl.log_forward_pass(model, x, vis_mode="none")
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


# ─── Utilities ────────────────────────────────────────────────────────────────

def list_tags() -> Set[str]:
    """Return all available tags across test graphs."""
    all_tags = set()
    for tg in get_test_graphs():
        all_tags.update(tg.tags)
    return all_tags

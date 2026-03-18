"""Synthetic graph generators for dagua.

Usage::

    import dagua

    g = dagua.generate_graph("scale_free", 1_000_000)
    g = dagua.generate_graph("wide_dag", 1_000_000, layers=100)
    g = dagua.generate_graph("fractal", 100_000, branching=4)
    g = dagua.generate_graph("tree", 50_000, branching=3)

All generators return a :class:`DaguaGraph` and are deterministic via
``seed``.  Large graphs (> 10M nodes) use chunked tensor construction
to avoid OOM.
"""

from __future__ import annotations

import math

import torch

from dagua.graph import DaguaGraph

# Chunk size for edge construction on large graphs.
_EDGE_CHUNK = 50_000_000


# ─── Public API ──────────────────────────────────────────────────────────────


STRUCTURES = [
    "wide_dag",
    "scale_free",
    "fractal",
    "tree",
    "chain",
    "grid",
    "small_world",
    "clustered",
]


def generate_graph(
    structure: str,
    num_nodes: int,
    *,
    seed: int = 42,
    **kwargs: object,
) -> DaguaGraph:
    """Generate a synthetic graph with the given structure and size.

    Parameters
    ----------
    structure : str
        One of: ``wide_dag``, ``scale_free``, ``fractal``, ``tree``,
        ``chain``, ``grid``, ``small_world``, ``clustered``.
    num_nodes : int
        Target node count (actual count may differ slightly to satisfy
        structural constraints).
    seed : int, default=42
        Random seed for reproducibility.
    **kwargs
        Structure-specific parameters (see individual generator docs).

    Returns
    -------
    DaguaGraph
        Generated graph with node sizes pre-computed.
    """
    generators = {
        "wide_dag": generate_wide_dag,
        "scale_free": generate_scale_free,
        "fractal": generate_fractal,
        "tree": generate_tree,
        "chain": generate_chain,
        "grid": generate_grid,
        "small_world": generate_small_world,
        "clustered": generate_clustered,
    }
    if structure not in generators:
        raise ValueError(f"Unknown structure {structure!r}. Choose from: {', '.join(generators)}")
    return generators[structure](num_nodes, seed=seed, **kwargs)


# ─── Wide DAG ────────────────────────────────────────────────────────────────


def generate_wide_dag(
    num_nodes: int,
    *,
    seed: int = 42,
    layers: int = 0,
    cross_prob: float = 0.5,
) -> DaguaGraph:
    """Layered DAG with backbone edges and random cross-layer connections.

    Parameters
    ----------
    num_nodes : int
        Target node count.
    seed : int
        Random seed.
    layers : int
        Number of layers. 0 = auto (sqrt(N)/10, minimum 10).
    cross_prob : float
        Probability of cross-connection per eligible pair within a layer.
    """
    if layers <= 0:
        layers = max(int(num_nodes**0.5 / 10) * 10, 10)
    w = num_nodes // layers
    num_nodes = w * layers  # round to exact multiple

    torch.manual_seed(seed)

    # Backbone: each node connects to the node one layer ahead
    backbone_n = num_nodes - w
    src_ids = torch.arange(0, backbone_n, dtype=torch.long)
    tgt_ids = src_ids + w

    # Cross-connections within each layer
    cross_edges = []
    for layer_start in range(0, num_nodes - w, w):
        layer_end = layer_start + w
        next_start = layer_end
        next_end = min(next_start + w, num_nodes)
        if next_start >= num_nodes:
            break
        n_cross = int(w * cross_prob)
        if n_cross > 0:
            cross_src = torch.randint(layer_start, layer_end, (n_cross,), dtype=torch.long)
            cross_tgt = torch.randint(next_start, next_end, (n_cross,), dtype=torch.long)
            cross_edges.append(torch.stack([cross_src, cross_tgt]))

    parts = [torch.stack([src_ids, tgt_ids])]
    if cross_edges:
        parts.extend(cross_edges)
    edge_index = torch.cat(parts, dim=1)

    g = DaguaGraph.from_edge_index(edge_index, num_nodes=num_nodes)
    g.compute_node_sizes()
    return g


# ─── Scale-Free DAG ─────────────────────────────────────────────────────────


def generate_scale_free(
    num_nodes: int,
    *,
    seed: int = 42,
    m: int = 3,
) -> DaguaGraph:
    """Scale-free DAG via batched preferential attachment.

    Produces a power-law degree distribution with visible hub nodes.
    Edges are directed from higher-index to lower-index nodes (DAG).

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    seed : int
        Random seed.
    m : int
        Edges per new node (controls density and hub strength).
    """
    torch.manual_seed(seed)
    n = num_nodes
    if n <= m:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        g = DaguaGraph.from_edge_index(edge_index, num_nodes=n)
        g.compute_node_sizes()
        return g

    # Start with a small complete DAG of m+1 nodes
    init_edges_src = []
    init_edges_tgt = []
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            init_edges_src.append(i)
            init_edges_tgt.append(j)

    # Degree array for preferential attachment (in-degree + out-degree)
    degree = torch.zeros(n, dtype=torch.float32)
    for i in range(m + 1):
        degree[i] = m  # initial clique

    all_src = [torch.tensor(init_edges_src, dtype=torch.long)]
    all_tgt = [torch.tensor(init_edges_tgt, dtype=torch.long)]

    # Batched preferential attachment — process nodes in chunks
    batch_size = min(100_000, n)
    for batch_start in range(m + 1, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_n = batch_end - batch_start

        # Sample m targets for each new node, weighted by degree
        # Each new node connects to m existing nodes (indices < node_id)
        probs = degree[:batch_start].clone()
        probs = probs / probs.sum()

        # Multinomial sampling: batch_n * m samples from existing nodes
        targets = torch.multinomial(probs, batch_n * m, replacement=True).reshape(batch_n, m)

        # Source = new node ids (higher index → lower index = DAG)
        sources = torch.arange(batch_start, batch_end, dtype=torch.long).unsqueeze(1).expand(-1, m)

        all_src.append(sources.reshape(-1))
        all_tgt.append(targets.reshape(-1))

        # Update degrees
        degree[batch_start:batch_end] += m
        tgt_flat = targets.reshape(-1)
        degree.scatter_add_(0, tgt_flat, torch.ones_like(tgt_flat, dtype=torch.float32))

    edge_index = torch.stack([torch.cat(all_src), torch.cat(all_tgt)])

    g = DaguaGraph.from_edge_index(edge_index, num_nodes=n)
    g.compute_node_sizes()
    return g


# ─── Fractal DAG ─────────────────────────────────────────────────────────────


def generate_fractal(
    num_nodes: int,
    *,
    seed: int = 42,
    branching: int = 4,
) -> DaguaGraph:
    """Self-similar fractal DAG built by recursive motif expansion.

    At each level, every leaf is replaced by a copy of the base motif
    (a star with ``branching`` children). The result is self-similar at
    every zoom level — ideal for powers-of-10 visualizations.

    Parameters
    ----------
    num_nodes : int
        Target node count (actual is the nearest motif-compatible size).
    seed : int
        Random seed.
    branching : int
        Children per hub node in the base motif.
    """
    torch.manual_seed(seed)

    # Determine depth to reach target size
    # Each level multiplies node count by ~branching
    # Level k has 1 + b + b^2 + ... + b^k = (b^(k+1) - 1) / (b - 1)
    depth = 1
    while (branching ** (depth + 1) - 1) // (branching - 1) < num_nodes:
        depth += 1

    # Build iteratively: start with root, expand leaves
    edges_src = []
    edges_tgt = []
    next_id = 1
    # Current leaves to expand
    leaves = [0]

    for _level in range(depth):
        new_leaves = []
        for parent in leaves:
            for _child in range(branching):
                if next_id >= num_nodes:
                    break
                edges_src.append(parent)
                edges_tgt.append(next_id)
                new_leaves.append(next_id)
                next_id += 1
            if next_id >= num_nodes:
                break
        leaves = new_leaves
        if next_id >= num_nodes:
            break

    actual_n = next_id
    edge_index = torch.tensor([edges_src, edges_tgt], dtype=torch.long)

    # Add some cross-branch edges for visual interest
    n_cross = actual_n // 10
    if n_cross > 0:
        cross_src = torch.randint(0, actual_n, (n_cross,))
        cross_tgt = torch.randint(0, actual_n, (n_cross,))
        # Ensure DAG: src < tgt
        lo = torch.minimum(cross_src, cross_tgt)
        hi = torch.maximum(cross_src, cross_tgt)
        valid = lo != hi
        cross_edge = torch.stack([lo[valid], hi[valid]])
        edge_index = torch.cat([edge_index, cross_edge], dim=1)

    g = DaguaGraph.from_edge_index(edge_index, num_nodes=actual_n)
    g.compute_node_sizes()
    return g


# ─── Tree ────────────────────────────────────────────────────────────────────


def generate_tree(
    num_nodes: int,
    *,
    seed: int = 42,
    branching: int = 3,
) -> DaguaGraph:
    """Balanced k-ary tree DAG.

    Parameters
    ----------
    num_nodes : int
        Target node count.
    seed : int
        Random seed.
    branching : int
        Children per internal node.
    """
    torch.manual_seed(seed)
    n = num_nodes

    # Build tree level by level
    src_list = []
    tgt_list = []
    next_id = 1
    parents = [0]

    while next_id < n and parents:
        new_parents = []
        for p in parents:
            for _ in range(branching):
                if next_id >= n:
                    break
                src_list.append(p)
                tgt_list.append(next_id)
                new_parents.append(next_id)
                next_id += 1
        parents = new_parents

    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    g = DaguaGraph.from_edge_index(edge_index, num_nodes=next_id)
    g.compute_node_sizes()
    return g


# ─── Chain ───────────────────────────────────────────────────────────────────


def generate_chain(
    num_nodes: int,
    *,
    seed: int = 42,
) -> DaguaGraph:
    """Linear chain 0 → 1 → 2 → ... → N-1.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    seed : int
        Random seed (unused, included for API consistency).
    """
    n = num_nodes
    src = torch.arange(0, n - 1, dtype=torch.long)
    tgt = torch.arange(1, n, dtype=torch.long)
    edge_index = torch.stack([src, tgt])
    g = DaguaGraph.from_edge_index(edge_index, num_nodes=n)
    g.compute_node_sizes()
    return g


# ─── Grid ────────────────────────────────────────────────────────────────────


def generate_grid(
    num_nodes: int,
    *,
    seed: int = 42,
    aspect: float = 1.0,
) -> DaguaGraph:
    """2D grid DAG with edges pointing right and down.

    Parameters
    ----------
    num_nodes : int
        Target node count (rounded to rows × cols).
    seed : int
        Random seed.
    aspect : float
        Width/height ratio. 1.0 = square, 2.0 = twice as wide.
    """
    rows = max(int(math.sqrt(num_nodes / aspect)), 1)
    cols = max(int(rows * aspect), 1)
    n = rows * cols

    src_list = []
    tgt_list = []

    # Right edges
    for r in range(rows):
        offset = r * cols
        right_src = torch.arange(offset, offset + cols - 1, dtype=torch.long)
        right_tgt = right_src + 1
        src_list.append(right_src)
        tgt_list.append(right_tgt)

    # Down edges
    for r in range(rows - 1):
        offset = r * cols
        down_src = torch.arange(offset, offset + cols, dtype=torch.long)
        down_tgt = down_src + cols
        src_list.append(down_src)
        tgt_list.append(down_tgt)

    edge_index = torch.stack([torch.cat(src_list), torch.cat(tgt_list)])
    g = DaguaGraph.from_edge_index(edge_index, num_nodes=n)
    g.compute_node_sizes()
    return g


# ─── Small-World DAG ────────────────────────────────────────────────────────


def generate_small_world(
    num_nodes: int,
    *,
    seed: int = 42,
    k: int = 6,
    rewire_prob: float = 0.1,
) -> DaguaGraph:
    """Watts-Strogatz small-world adapted as a DAG.

    Starts with a ring lattice where each node connects to k nearest
    forward neighbors, then randomly rewires edges.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    seed : int
        Random seed.
    k : int
        Number of forward neighbors in the base ring lattice.
    rewire_prob : float
        Probability of rewiring each edge to a random target.
    """
    torch.manual_seed(seed)
    n = num_nodes
    half_k = max(k // 2, 1)

    # Ring lattice: each node connects to next half_k nodes (forward only = DAG)
    src_parts = []
    tgt_parts = []
    for offset in range(1, half_k + 1):
        src = torch.arange(0, n, dtype=torch.long)
        tgt = (src + offset) % n
        # Only keep forward edges (src < tgt) for DAG property
        valid = src < tgt
        src_parts.append(src[valid])
        tgt_parts.append(tgt[valid])

    all_src = torch.cat(src_parts)
    all_tgt = torch.cat(tgt_parts)

    # Rewire
    num_edges = all_src.shape[0]
    rewire_mask = torch.rand(num_edges) < rewire_prob
    n_rewire = int(rewire_mask.sum().item())
    if n_rewire > 0:
        new_tgt = torch.randint(0, n, (n_rewire,))
        rewire_src = all_src[rewire_mask]
        # Ensure DAG: src < tgt
        lo = torch.minimum(rewire_src, new_tgt)
        hi = torch.maximum(rewire_src, new_tgt)
        valid = lo != hi
        all_tgt[rewire_mask] = all_tgt[rewire_mask]  # reset
        # Replace valid rewired edges
        rewire_indices = rewire_mask.nonzero(as_tuple=True)[0][valid]
        all_src[rewire_indices] = lo[valid]
        all_tgt[rewire_indices] = hi[valid]

    edge_index = torch.stack([all_src, all_tgt])
    g = DaguaGraph.from_edge_index(edge_index, num_nodes=n)
    g.compute_node_sizes()
    return g


# ─── Clustered DAG ──────────────────────────────────────────────────────────


def generate_clustered(
    num_nodes: int,
    *,
    seed: int = 42,
    num_clusters: int = 0,
    intra_density: float = 0.01,
    inter_density: float = 0.001,
) -> DaguaGraph:
    """DAG with dense intra-cluster edges and sparse inter-cluster edges.

    Each cluster is a mini-DAG. Clusters are connected in a chain
    (cluster 0 → cluster 1 → ...) with sparse random edges between them.

    Parameters
    ----------
    num_nodes : int
        Target node count.
    seed : int
        Random seed.
    num_clusters : int
        Number of clusters. 0 = auto (sqrt(N), clamped to [4, 100]).
    intra_density : float
        Edge density within each cluster.
    inter_density : float
        Edge density between adjacent clusters.
    """
    torch.manual_seed(seed)
    n = num_nodes

    if num_clusters <= 0:
        num_clusters = max(4, min(int(math.sqrt(n)), 100))

    cluster_size = n // num_clusters
    actual_n = cluster_size * num_clusters

    all_src = []
    all_tgt = []

    # Intra-cluster edges (random DAG within each cluster)
    for c in range(num_clusters):
        offset = c * cluster_size
        n_intra = max(int(cluster_size * cluster_size * intra_density), cluster_size - 1)
        # Random edges within cluster, ensuring DAG (src < tgt within cluster)
        local_src = torch.randint(0, cluster_size - 1, (n_intra,))
        local_tgt = torch.randint(1, cluster_size, (n_intra,))
        lo = torch.minimum(local_src, local_tgt)
        hi = torch.maximum(local_src, local_tgt)
        valid = lo != hi
        all_src.append(lo[valid] + offset)
        all_tgt.append(hi[valid] + offset)

        # Backbone within cluster
        chain_src = torch.arange(offset, offset + cluster_size - 1, dtype=torch.long)
        chain_tgt = chain_src + 1
        all_src.append(chain_src)
        all_tgt.append(chain_tgt)

    # Inter-cluster edges (between adjacent clusters, directed forward)
    for c in range(num_clusters - 1):
        offset_a = c * cluster_size
        offset_b = (c + 1) * cluster_size
        n_inter = max(int(cluster_size * cluster_size * inter_density), 1)
        inter_src = torch.randint(offset_a, offset_a + cluster_size, (n_inter,))
        inter_tgt = torch.randint(offset_b, offset_b + cluster_size, (n_inter,))
        all_src.append(inter_src)
        all_tgt.append(inter_tgt)

    edge_index = torch.stack([torch.cat(all_src), torch.cat(all_tgt)])
    g = DaguaGraph.from_edge_index(edge_index, num_nodes=actual_n)
    g.compute_node_sizes()
    return g

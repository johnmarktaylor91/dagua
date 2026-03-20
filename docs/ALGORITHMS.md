# Graph Layout Algorithms in Dagua

Dagua includes reimplementations of 14 classic graph layout algorithms, plus its
own differentiable layout engine. Each reimplementation is a faithful translation
of the original reference code — matching formulas, parameters, and optimization
procedures as exactly as possible.

This document describes each algorithm: what it does, where it comes from, how
we validated it, and how to use it in dagua.

---

## Dagua's Own Engine

**What it does.** Dagua treats node positions as learnable parameters and layout
aesthetics as differentiable loss functions. The solver is gradient descent:
`loss.backward(); optimizer.step()`. Multiple objectives (DAG ordering, node
repulsion, edge attraction, crossing minimization, overlap avoidance) are
composed into a single loss and optimized jointly.

**Key properties.** GPU-accelerated via PyTorch. Scales to billions of nodes
via multilevel coarsening. Respects DAG hierarchy by default. Supports
user constraints (pins, alignments, spacing) as soft loss terms.

**When to use it.** This is dagua's default and recommended engine. Best for
directed acyclic graphs, neural network architectures, and any graph where
hierarchical structure matters.

---

## Force-Directed Algorithms

Force-directed algorithms model the graph as a physical system: nodes repel each
other (like charged particles), edges attract connected nodes (like springs), and
the layout is the equilibrium state.

### Fruchterman-Reingold (FR)

**Paper.** Fruchterman & Reingold, 1991. "Graph drawing by force-directed placement."

**How it works.** All nodes repel each other with force proportional to `k^2/d`
(where `k` is the ideal distance and `d` is the current distance). Connected
nodes attract with force proportional to `d^2/k`. A temperature parameter starts
high (allowing large movements) and cools linearly, producing a coarse-to-fine
annealing effect.

**Our implementation.** Exact translation of NetworkX's `spring_layout`. Verified
at 0.000000 Procrustes disparity on unweighted graphs with matching seeds.

**Reference.** `networkx.spring_layout`, `igraph.layout_fruchterman_reingold`

**Usage.** `dagua.layout(graph, algorithm="fr")`

---

### ForceAtlas2 (FA2)

**Paper.** Jacomy et al., 2014. "ForceAtlas2, a continuous graph layout algorithm
for handy network visualization."

**How it works.** An evolution of FR designed for real-world network visualization.
Key innovations: degree-weighted repulsion (heavier nodes repel more), adaptive
speed control (the algorithm measures "swinging" vs "traction" and adjusts step
size dynamically), gravity toward the origin (prevents drift), and optional
hub-dissuading attraction (divides attraction by source node mass).

**Our implementation.** Line-by-line translation of the `fa2-modified` Python
package, which itself mirrors the Gephi Java implementation. Verified at 0.000005
Procrustes disparity with matching seeds.

**Reference.** `fa2-modified` package (Python)

**Usage.** `dagua.layout(graph, algorithm="fa2")`

---

### GEM

**Paper.** Frick, Ludwig & Mehldau, 1995. "A fast adaptive layout algorithm for
undirected graphs."

**How it works.** A force-directed algorithm with per-node adaptive temperature.
Each node has its own "temperature" (step size) that adapts based on whether the
node is oscillating (temperature decreases), rotating (temperature decreases via
a skew gauge), or moving consistently (temperature increases). This makes GEM
converge faster than FR on graphs with mixed structure.

**Our implementation.** Matched to the OGDF C++ source (`GEMLayout.cpp`).
Degree-weighted attraction and gravity, continuous angle-based temperature
adaptation.

**Reference.** OGDF (`ogdf::GEMLayout`, run via subprocess)

**Usage.** `dagua.layout(graph, algorithm="gem")`

---

### LinLog

**Paper.** Noack, 2009. "Modularity clustering is force-directed layout."

**How it works.** A force-directed layout where the energy function is designed
so that the optimal layout reveals community structure. Attraction between
connected nodes is proportional to their edge distance. Repulsion between
non-adjacent nodes is proportional to the logarithm of their distance. The
theoretical result: the energy minimum corresponds to the modularity-optimal
clustering of the graph.

**Our implementation.** Follows the paper. Repulsion computed over non-edge
pairs only (critical for the modularity correspondence). This is the only
algorithm in dagua with no runnable external reference — Noack's original Java
applet is lost, and no maintained library implements it.

**Reference.** Paper only (no runnable original exists anywhere)

**Usage.** `dagua.layout(graph, algorithm="linlog")`

---

## Stress-Based Algorithms

Stress-based algorithms compute the shortest-path distances between all node
pairs, then optimize node positions so that Euclidean distances match graph
distances as closely as possible. The "stress" is the weighted sum of squared
differences between target and actual distances.

### Kamada-Kawai (KK)

**Paper.** Kamada & Kawai, 1989. "An algorithm for drawing general undirected
graphs."

**How it works.** Computes all-pairs shortest paths, then minimizes the stress
energy using L-BFGS-B (a quasi-Newton optimizer). Spring strengths are
proportional to `1/d^2`. Initialization is a circular layout. Produces
high-quality layouts for small-to-medium graphs but is O(N^2) in distance
computation.

**Our implementation.** Exact translation of NetworkX's `kamada_kawai_layout`,
including the scipy L-BFGS-B solver, circular initialization, and centering
term. Verified at 0.000000 Procrustes disparity.

**Reference.** `networkx.kamada_kawai_layout`, `igraph.layout_kamada_kawai`

**Usage.** `dagua.layout(graph, algorithm="kk")`

---

### Stress-SGD

**Paper.** Zheng, Pawar & Goodman, 2018. "Graph drawing by stochastic gradient
descent."

**How it works.** Instead of optimizing stress with majorization (the classical
approach), this algorithm uses SGD: pick a random pair of nodes, compute how far
their Euclidean distance is from their graph distance, and nudge both nodes to
reduce the error. The learning rate follows an exponential decay schedule. Simple,
fast, and reaches lower stress than majorization on many graphs.

**Our implementation.** Faithful translation of the s_gd2 C++ source code.
Sequential (Gauss-Seidel) pair updates, exponential schedule, step clamping.
Achieves 0.993 stress ratio vs s_gd2 (statistically identical objective values).
Exact position match is impossible because the C library's internal shuffle RNG
cannot be reproduced from Python.

**Reference.** `s_gd2` package (C++ with Python bindings)

**Usage.** `dagua.layout(graph, algorithm="stress_sgd")`

---

### Maxent-Stress

**Paper.** Gansner, Hu & North, 2013. "Maxent-stress model for graph layout."

**How it works.** Extends stress minimization with a maximum-entropy repulsion
term. The stress term pulls connected nodes to their target distances; the
entropy term pushes non-adjacent nodes apart via a logarithmic potential. The
balance between stress and entropy is controlled by an alpha parameter. Produces
layouts with better-separated clusters than pure stress.

**Our implementation.** The stress objective matches OGDF's `StressMinimization`.
The entropy term follows the paper. We use Adam optimization rather than OGDF's
stress majorization — same objective, different solver. Supports both pure stress
mode (matching OGDF) and maxent-stress mode (matching the paper).

**Reference.** OGDF (`ogdf::StressMinimization`, run via subprocess)

**Usage.** `dagua.layout(graph, algorithm="maxent_stress")`

---

## Spectral and Dimensionality Reduction

These algorithms embed the graph into low-dimensional space using matrix
decomposition or manifold learning. They don't simulate forces — they compute
positions analytically or via embedding optimization.

### Spectral Layout

**Paper.** Hall, 1970. "An r-dimensional quadratic placement algorithm." Also
Koren, 2003. "On spectral graph drawing."

**How it works.** Computes the graph Laplacian (degree matrix minus adjacency
matrix), then takes the eigenvectors corresponding to the smallest non-zero
eigenvalues as node coordinates. The result places connected nodes close together
and reveals the graph's spectral structure. Deterministic — no randomness.

**Our implementation.** Exact translation of NetworkX's `spectral_layout`.
Same Laplacian construction, same eigensolver (numpy for dense, scipy ARPACK for
sparse), same eigenvector selection. Verified at 0.000000 Procrustes disparity.

**Reference.** `networkx.spectral_layout`

**Usage.** `dagua.layout(graph, algorithm="spectral")`

---

### Pivot-MDS

**Paper.** Brandes & Pich, 2007. "Eigensolver methods for progressive
multidimensional scaling of large data."

**How it works.** Classical MDS (multidimensional scaling) embeds nodes so that
Euclidean distances approximate graph distances, but requires O(N^2) memory for
the full distance matrix. Pivot-MDS selects a small set of "pivot" nodes,
computes distances only from pivots to all nodes (O(P*N)), and uses SVD on this
rectangular matrix to produce coordinates. Much faster than full MDS with similar
quality.

**Our implementation.** Max-min pivot selection, BFS distances, double-centering
followed by SVD. Coordinates are V*S (right singular vectors scaled by singular
values).

**Reference.** OGDF (`ogdf::PivotMDS`), `s_gd2.mds_direct`, `igraph.layout_mds`

**Usage.** `dagua.layout(graph, algorithm="pivot_mds")`

---

### tsNET

**Paper.** Kruiger et al., 2017. "Graph layouts by t-SNE."

**How it works.** Computes all-pairs shortest-path distances, converts them to
affinities using the t-SNE perplexity-matching procedure, then minimizes the
KL divergence between high-dimensional affinities and low-dimensional Student-t
similarities. The result preserves local neighborhood structure and tends to
produce tight clusters separated by whitespace.

**Our implementation.** Follows the paper's optimization: SGD with momentum
and per-parameter gains (the classic t-SNE trick where gains increase when the
gradient direction changes and decrease when it's consistent). Two-phase momentum
(0.5 during early exaggeration, 0.8 after). PivotMDS initialization.

**Reference.** `sklearn.manifold.TSNE` (closest available — uses the same
objective but different optimizer details)

**Usage.** `dagua.layout(graph, algorithm="tsnet")`

---

## Hierarchical / Layered Algorithms

Layered algorithms assign nodes to discrete horizontal layers (respecting edge
direction), then arrange nodes within each layer to minimize edge crossings.
The canonical framework is Sugiyama's.

### Sugiyama

**Paper.** Sugiyama, Tagawa & Toda, 1981. "Methods for visual understanding of
hierarchical system structures."

**How it works.** Four phases: (1) break cycles by reversing some edges,
(2) assign nodes to layers based on topological depth, (3) order nodes within
each layer to minimize edge crossings (barycenter heuristic), (4) assign x-coordinates
placing each node at the average position of its neighbors.

**Our implementation.** Matched to igraph's `layout_sugiyama`. Includes layer
promotion (pushing nodes toward successors to minimize dummy edges) and
barycenter-based x-positioning with refinement sweeps.

**Reference.** `igraph.layout_sugiyama`, `graphviz dot`, OGDF (`ogdf::SugiyamaLayout`)

**Usage.** `dagua.layout(graph, algorithm="sugiyama")`

---

## Simulated Annealing

### Davidson-Harel

**Paper.** Davidson & Harel, 1996. "Drawing graphs nicely using simulated
annealing."

**How it works.** Defines an energy function with five terms: node distribution
(spread nodes evenly), border repulsion (keep nodes away from edges of the
drawing area), edge length (prefer short edges), edge crossings (penalize
crossings), and node-edge proximity (keep nodes away from non-incident edges).
Uses simulated annealing: randomly perturb one node at a time, accept or reject
based on the Metropolis criterion.

**Our implementation.** All five energy terms from the paper. Geometric cooling
schedule. Energy terms use sum (not mean) to match paper scaling.

**Reference.** `igraph.layout_davidson_harel`, OGDF (`ogdf::DavidsonHarelLayout`)

**Usage.** `dagua.layout(graph, algorithm="davidson_harel")`

---

## Multilevel

### FM^3

**Paper.** Hachul & Junger, 2004. "Drawing large graphs with a potential-field-based
multilevel algorithm."

**How it works.** A multilevel force-directed algorithm. First, the graph is
repeatedly coarsened using a "solar system" decomposition: high-degree nodes
become "suns," their neighbors become "planets," and neighbors-of-neighbors
become "moons." At the coarsest level, a simple layout is computed. Then the
graph is progressively un-coarsened, with positions inherited from the coarser
level and refined using force-directed simulation with Barnes-Hut or multipole
approximation for fast repulsion.

**Our implementation.** Solar-system coarsening, Barnes-Hut repulsion (OGDF uses
full multipole expansion — a documented approximation difference), FR-style
forces, jitter-based prolongation.

**Reference.** OGDF (`ogdf::FMMMLayout`, run via subprocess)

**Usage.** `dagua.layout(graph, algorithm="fmmm")`

---

## Validation

Every algorithm with a runnable reference has been tested for fidelity:

| Algorithm | Reference | Match Type | Evidence |
|---|---|---|---|
| FR | NetworkX | Exact | 0.000000 Procrustes (unweighted) |
| KK | NetworkX | Exact | 0.000000 Procrustes (unweighted) |
| Spectral | NetworkX | Exact | 0.000000 Procrustes (unweighted) |
| FA2 | fa2-modified | Exact | 0.000005 Procrustes (same seed) |
| Stress-SGD | s_gd2 | Same objective | 0.993 stress ratio |
| Sugiyama | igraph | Structural | Same layers + ordering |
| tsNET | sklearn TSNE | Same family | Statistical comparison |
| GEM | OGDF | Code-matched | Formulas from GEMLayout.cpp |
| FM^3 | OGDF | Code-matched | Structure from FMMMLayout.cpp |
| Maxent-Stress | OGDF | Same objective | Stress term matches |
| Davidson-Harel | igraph + OGDF | Paper-matched | Energy terms from paper |
| Pivot-MDS | OGDF + s_gd2 + igraph | Code-matched | SVD from paper |
| LinLog | (none) | Paper only | No runnable reference exists |

"Exact" means Procrustes disparity < 0.00001 on unweighted test graphs with
matching seeds. "Same objective" means the algorithms minimize the same cost
function to statistically indistinguishable values, but optimization trajectories
differ due to C-level RNG differences. "Code-matched" means our formulas are
translated from the reference source code. "Paper-matched" means we follow the
published algorithm description.

---

## Using Classic Algorithms

```python
import dagua

# Use dagua's own engine (default, recommended)
pos = dagua.layout(graph)

# Use a specific classic algorithm
pos = dagua.layout(graph, algorithm="fr")
pos = dagua.layout(graph, algorithm="kk")
pos = dagua.layout(graph, algorithm="fa2")
pos = dagua.layout(graph, algorithm="sugiyama")

# Compare multiple algorithms on the same graph
for algo in ["fr", "kk", "fa2", "spectral", "stress_sgd"]:
    pos = dagua.layout(graph, algorithm=algo)
    dagua.render(graph, pos, output=f"layout_{algo}.png")
```

All classic algorithms accept a `seed` parameter for reproducibility:
```python
pos = dagua.layout(graph, algorithm="fr", seed=42)
```

# SFDP and UMAP: Complete Algorithm Specifications for Reimplementation

Sources:
- sfdp: Graphviz source code (gitlab.com/graphviz/graphviz, lib/sfdpgen/*)
- UMAP: umap-learn 0.5.x source code (umap/umap_.py, umap/layouts.py, umap/spectral.py)
- Date: 2026-03-20

---

## 1. SFDP (Scalable Force-Directed Placement — Hu 2005)

### 1.1 Algorithm Overview

Multilevel force-directed layout. Three phases:
1. **Coarsen** the graph into a hierarchy of progressively smaller graphs
2. **Layout** the coarsest graph with force-directed placement
3. **Prolong** positions to finer levels and refine with more force-directed iterations

### 1.2 Force Model

The spring-electrical model uses two forces parameterized by exponent `p` (default -1):

**Attractive force** (edges only):
```
F_attract(i,j) = C^((2-p)/3) / K * ||x_i - x_j|| * (x_j - x_i)
```
Where:
- `C = 0.2` (force scaling constant)
- `K` = ideal edge length (computed as average edge length in current layout)
- `CRK = C^((2-p)/3) / K` (precomputed)

In code: `f[k] -= CRK * (x[i*dim+k] - x[ja[j]*dim+k]) * dist`

Note: the attractive force is **linear in distance** (Hooke's law with the CRK prefactor).

**Repulsive force** (all pairs, approximated via Barnes-Hut):
```
F_repulse(i,j) = K^(1-p) / ||x_i - x_j||^(1-p) * (x_i - x_j) / ||x_i - x_j||
```
Where `KP = K^(1-p)` (precomputed).

In code: `f[k] += KP * (x[i*dim+k] - x[j*dim+k]) / pow(dist, 1 - p)`

With default `p = -1`:
- Attractive: `F_a ~ C * ||d|| * direction` (linear spring)
- Repulsive: `F_r ~ K^2 / ||d||^2 * direction` (inverse-square, like gravity)

For power-law graphs, `p` is set to `-1.8` for stronger long-range repulsion.

### 1.3 Barnes-Hut Quadtree Approximation

**Threshold**: `theta = 0.6`

**Decision rule**: For quadtree cell `c` and point `i`:
```
if width(c) / dist(i, center(c)) < theta:
    treat c as single supernode at its center-of-mass
else:
    recurse into children
```

**Quadtree parameters**:
- `max_qtree_level = 10`
- Hybrid mode: use quadtree only when `n > QUAD_TREE_HYBRID_SIZE (10000)`
- Adaptive level optimization via `oned_optimizer_train` with cost weighting:
  `cost = count_level0 + 0.85 * count_level1 + 3.3 * count_level2`

### 1.4 Movement and Convergence

**Force normalization**: Each node's total force vector is normalized to unit length.
The node moves `step` distance in the direction of the net force:
```
F_total = sum of all attractive + repulsive forces on node i
direction = F_total / ||F_total||
x_i += step * direction
```

This is **not** a standard gradient step — it's a fixed-distance step in the force direction.

**Adaptive cooling** (step size update per iteration):
```python
def update_step(step, Fnorm, Fnorm0):
    if Fnorm >= Fnorm0:        # no progress
        step *= 0.90           # cool = 0.90
    elif Fnorm > 0.95 * Fnorm0:  # modest progress
        pass                   # keep step
    else:                      # good progress
        step = 0.99 * step / 0.90  # increase ~= step * 1.1
    return step
```

**Convergence**: `while step > tol and iter < maxiter`
- `tol = 0.001` (relative to K)
- `maxiter = 500`

### 1.5 Coarsening (Heavy Edge Matching)

From `Multilevel.c`:

**Algorithm**: Maximal independent edge set via heaviest-edge-per-node matching.

```
1. Identify supervariables (groups of nodes with identical adjacency)
2. Process nodes in random permutation order
3. For each unmatched node i:
   a. Find heaviest-weight neighbor j that is also unmatched
   b. Match (i, j) — they become one coarse node
   c. Mark both as MATCHED
4. Unmatched nodes become singleton coarse nodes
```

**Coarse graph construction**:
- Build projection matrix P (fine → coarse mapping): P[coarse_id, fine_id] = 1
- Restriction matrix R = P^T
- Coarse adjacency: A_coarse = R * A * P (matrix triple product)

**Stopping criteria**:
```
minsize = 4
min_coarsen_factor = 0.75
stop when: nc == n OR nc < minsize OR nc > min_coarsen_factor * n
```
(i.e., stop if coarsening reduced size by less than 25%, or graph is tiny)

### 1.6 Multilevel Orchestration

```
MULTILEVEL_LAYOUT(A, ctrl):
    # Phase 1: Build coarsening hierarchy
    hierarchy = []
    A_current = A
    while can_coarsen(A_current):
        P, R, A_coarse = coarsen(A_current)
        hierarchy.append((A_current, P, R))
        A_current = A_coarse

    # Phase 2: Layout coarsest level
    x = random_init(A_current.n, dim)  # uniform [0,1]
    K = -1  # will be computed from avg edge length
    spring_electrical_embedding(A_current, ctrl, x)

    # Phase 3: Prolong and refine
    for (A_fine, P, R) in reversed(hierarchy):
        # Interpolate positions: x_fine = P * x_coarse
        x_fine = P @ x_coarse

        # Smooth: each node → 50% self + 50% avg of neighbors
        for i in range(n_fine):
            x_fine[i] = 0.5 * x_fine[i] + 0.5 * mean(x_fine[neighbors(i)])

        # Add small perturbation to break symmetry
        # (only for nodes that share a coarse representative)
        for each coarse node c:
            for each fine node j mapped to c (beyond the first):
                x_fine[j] += K * 0.001 * (rand() - 0.5)

        # Refine at this level
        ctrl.random_start = False
        ctrl.K *= 0.75  # reduce ideal edge length for finer levels
        ctrl.adaptive_cooling = False
        ctrl.step = 0.1
        spring_electrical_embedding(A_fine, ctrl, x_fine)

        x_coarse = x_fine
```

### 1.7 K (Ideal Edge Length) Computation

```
K = average_edge_length(A, dim, x)
  = sum(||x_i - x_j|| for (i,j) in edges) / num_edges
```
Computed once at the coarsest level. Multiplied by 0.75 at each finer level.

### 1.8 Default Parameters Summary

| Parameter | Value | Description |
|-----------|-------|-------------|
| C | 0.2 | Force scaling constant |
| p | -1.0 | Repulsive exponent (AUTOP = -1.0001234) |
| K | auto | Ideal edge length (avg edge length) |
| theta (bh) | 0.6 | Barnes-Hut opening angle |
| cool | 0.90 | Cooling factor |
| tol | 0.001 | Convergence tolerance |
| maxiter | 500 | Max iterations per level |
| step_init | 0.1 | Initial step size |
| max_qtree_level | 10 | Quadtree depth limit |
| QUAD_TREE_HYBRID_SIZE | 10000 | Switch to quadtree above this |
| minsize | 4 | Min coarse graph size |
| min_coarsen_factor | 0.75 | Stop if coarsening ratio > this |
| random_seed | 123 | Default seed |
| K_decay_per_level | 0.75 | K multiplier at each finer level |
| perturbation | K * 0.001 | Noise magnitude during prolongation |
| interp_alpha | 0.5 | Self-weight in neighbor averaging |

### 1.9 Post-Processing (Optional Smoothing)

After multilevel layout, optional smoothing methods:
- **Stress majorization** (graph dist, avg dist, or power dist): solves `(Lw + lambda*I)x = Lwdd*y + lambda*x0`
- **Triangle smoothing**: adds Delaunay triangulation edges, weights by `dist^0.6`
- **Spring smoothing**: 20 more iterations with halved step size
- **RNG smoothing**: relative neighborhood graph edges

Default sfdp uses `SMOOTHING_SPRING` or no smoothing.

---

## 2. UMAP (Uniform Manifold Approximation and Projection — McInnes et al. 2018)

### 2.1 Algorithm Overview

Three phases:
1. **Build fuzzy simplicial set** from distance data (the high-d graph)
2. **Initialize embedding** (spectral or random)
3. **Optimize embedding** via SGD to minimize cross-entropy between high-d and low-d fuzzy sets

### 2.2 Phase 1: Fuzzy Simplicial Set Construction

#### 2.2.1 Smooth k-NN Distances

For each point `i`, find sigma_i (bandwidth) and rho_i (connectivity distance):

```python
# rho_i = distance to nearest neighbor (local connectivity adjustment)
# For local_connectivity=1: rho_i = distance to 1st nearest neighbor
rho_i = knn_dists[i, 0]  # (simplified; interpolation for non-integer local_connectivity)

# sigma_i found by binary search such that:
# sum_{j in kNN(i)} exp(-(d(i,j) - rho_i) / sigma_i) = log2(k)
target = log2(k)
sigma_i = binary_search(
    f = lambda sigma: sum(exp(-max(0, d(i,j) - rho_i) / sigma) for j in kNN(i)),
    target = target,
    n_iter = 64,
    tol = 1e-5
)

# Floor: sigma_i >= 1e-3 * mean(knn_dists[i])
```

#### 2.2.2 Membership Strengths

For each point `i` and its neighbor `j`:
```python
if d(i,j) - rho_i <= 0 or sigma_i == 0:
    w_ij = 1.0
else:
    w_ij = exp(-(d(i,j) - rho_i) / sigma_i)
```

This creates a directed weighted graph (sparse matrix) where each row is a local fuzzy set.

#### 2.2.3 Symmetrization (Fuzzy Union)

Convert directed graph to undirected via probabilistic t-conorm:
```python
# A = directed membership matrix
# With set_op_mix_ratio = 1.0 (default, pure fuzzy union):
B = A + A^T - A * A^T
# Equivalent to: P(a or b) = P(a) + P(b) - P(a)*P(b)
# (product t-norm for intersection, probabilistic sum for union)

# General form with mix ratio r:
B = r * (A + A^T - A*A^T) + (1-r) * (A * A^T)
```

### 2.3 Phase 2: Low-Dimensional Curve (a, b Parameters)

UMAP uses a smooth approximation to the step function:
```python
# Target: f(d) = 1 if d < min_dist, else exp(-(d - min_dist) / spread)
# Approximation: phi(d) = 1 / (1 + a * d^(2b))

# Fit a, b by least-squares curve fitting:
xv = linspace(0, 3*spread, 300)
yv = where(xv < min_dist, 1.0, exp(-(xv - min_dist) / spread))
a, b = curve_fit(lambda x, a, b: 1/(1 + a*x**(2*b)), xv, yv)
```

Default `min_dist=0.1, spread=1.0` gives approximately `a ≈ 1.93, b ≈ 0.79`.

The low-d membership strength between embedded points at distance d is:
```
phi(d) = 1 / (1 + a * d^(2b))
```

### 2.4 Phase 3: Embedding Initialization

**Spectral** (default):
```python
# Normalized Laplacian of the fuzzy graph:
D_sqrt = diag(1 / sqrt(degree))
L = I - D_sqrt @ graph @ D_sqrt

# Find smallest k+1 eigenvectors (excluding trivial eigenvector)
eigenvalues, eigenvectors = eigsh(L, k=dim+1, which='SM')
embedding = eigenvectors[:, 1:dim+1]  # skip first (constant) eigenvector

# Scale to [-10, 10] and add small noise (std=0.0001)
embedding = 10 * (embedding - min) / (max - min)
```

For multiple connected components: spectral embed each component separately,
arrange components via spectral embedding of their centroids.

### 2.5 Phase 4: SGD Optimization

#### 2.5.1 Edge Sampling Schedule

Edges are sampled proportional to their membership weight:
```python
epochs_per_sample = n_epochs / (weights / weights.max())
# High-weight edges: sampled every epoch
# Low-weight edges: sampled less frequently
# Edges with weight < max_weight / n_epochs are pruned (set to 0)
```

Negative samples per positive sample: `negative_sample_rate = 5`
```python
epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
```

#### 2.5.2 The Loss Function (Cross-Entropy)

The theoretical objective is fuzzy set cross-entropy:
```
CE = sum_e [ mu(e) * log(mu(e)/phi(e)) + (1-mu(e)) * log((1-mu(e))/(1-phi(e))) ]
```
Where:
- `mu(e)` = high-d membership strength (from fuzzy simplicial set)
- `phi(e)` = low-d membership: `1 / (1 + a * ||y_i - y_j||^(2b))`

In practice, optimized via edge sampling (like word2vec negative sampling):

#### 2.5.3 Attractive Gradient (Positive Samples)

For edge (i,j) with `dist_sq = ||y_i - y_j||^2`:
```python
grad_coeff = -2 * a * b * dist_sq^(b-1) / (a * dist_sq^b + 1)
# Per dimension:
grad_d = clip(grad_coeff * (y_i[d] - y_j[d]), -4.0, 4.0)
y_i[d] += alpha * grad_d
y_j[d] -= alpha * grad_d  # (if move_other=True)
```

This is the gradient of `log(phi(d))` = `log(1/(1+a*d^(2b)))` w.r.t. y_i.

#### 2.5.4 Repulsive Gradient (Negative Samples)

For each positive sample, draw `negative_sample_rate` random nodes k:
```python
dist_sq = ||y_i - y_k||^2
if dist_sq > 0:
    grad_coeff = 2 * gamma * b / ((0.001 + dist_sq) * (a * dist_sq^b + 1))
    # Per dimension:
    grad_d = clip(grad_coeff * (y_i[d] - y_k[d]), -4.0, 4.0)
    y_i[d] += alpha * grad_d
    # Note: y_k is NOT moved (asymmetric negative sampling)
```

This approximates the gradient of `log(1 - phi(d))` = `log(1 - 1/(1+a*d^(2b)))`.

The `0.001` in the denominator prevents division by zero for coincident points.

#### 2.5.5 Learning Rate Schedule

Linear decay:
```python
alpha = initial_alpha * (1.0 - epoch / n_epochs)
```

Default `initial_alpha = 1.0`.

#### 2.5.6 Gradient Clipping

All per-dimension gradient contributions are clipped to `[-4.0, 4.0]`.

### 2.6 Default Parameters Summary

| Parameter | Default | Description |
|-----------|---------|-------------|
| n_neighbors | 15 | k for k-NN graph |
| n_components | 2 | Embedding dimension |
| min_dist | 0.1 | Min distance in embedding |
| spread | 1.0 | Scale of embedding |
| n_epochs | None | Auto: 500 if n<=10000, else 200 |
| learning_rate | 1.0 | Initial SGD alpha |
| negative_sample_rate | 5 | Neg samples per positive |
| repulsion_strength (gamma) | 1.0 | Weight on negative samples |
| local_connectivity | 1.0 | Assumed local connectivity |
| set_op_mix_ratio | 1.0 | 1.0 = fuzzy union, 0.0 = intersection |
| init | "spectral" | Initialization method |
| random_state | None | RNG seed |
| SMOOTH_K_TOLERANCE | 1e-5 | Binary search tolerance |
| MIN_K_DIST_SCALE | 1e-3 | Min sigma relative to mean dist |
| CLIP_RANGE | [-4.0, 4.0] | Gradient clipping bounds |
| a, b | fit from min_dist, spread | Curve parameters |

### 2.7 For Graph Layout (Precomputed Distances)

When using UMAP for graph layout with `metric='precomputed'`:
1. Compute all-pairs shortest-path distances (or approximate)
2. Feed as distance matrix to UMAP
3. UMAP finds k-NN from the distance matrix (argsort each row)
4. Rest of algorithm proceeds identically

Key: the "high-dimensional distances" ARE the graph-theoretic distances.

---

## 3. PyTorch Reimplementation Notes

### 3.1 sfdp in PyTorch

**Straightforward to port**:
- Force computation is fully differentiable (spring + inverse-square)
- Quadtree can be replaced by `torch.cdist` for small graphs
  or a batch-friendly tree implementation for large ones
- Coarsening is graph manipulation (not differentiable, but doesn't need to be)

**Key design choices**:
- Use `torch.cdist` for all-pairs repulsion if n < 10000
- Implement Barnes-Hut via recursive cell partitioning on GPU for n >= 10000
- Coarsening: implement heavy-edge matching on CPU, transfer positions to GPU
- The unit-direction + fixed-step movement is unusual — can also try standard
  gradient descent on energy E = sum(attractive_potential) + sum(repulsive_potential)

**Energy form** (if we want to use autograd instead of explicit forces):
```
E_attract = sum over edges: CRK * ||x_i - x_j||^2 / 2
E_repulse = sum over pairs: -KP * ||x_i - x_j||^(2-p) / (2-p)  [for p != 2]
```
With p=-1: E_repulse = -K^2 * log(||x_i - x_j||) / ... (need care with the integral)

### 3.2 UMAP in PyTorch

**The core optimization loop is naturally expressible in PyTorch**:

```python
# Given: head, tail (edge indices), epochs_per_sample, a, b
# Positive step:
dist_sq = ((embedding[head] - embedding[tail]) ** 2).sum(-1)
phi = 1 / (1 + a * dist_sq ** b)
attractive_grad = -2 * a * b * dist_sq ** (b-1) / (a * dist_sq ** b + 1)

# Negative sampling:
neg_indices = torch.randint(0, n, (len(head), neg_rate))
dist_sq_neg = ((embedding[head].unsqueeze(1) - embedding[neg_indices]) ** 2).sum(-1)
repulsive_grad = 2 * gamma * b / ((0.001 + dist_sq_neg) * (a * dist_sq_neg ** b + 1))
```

**Key differences from standard implementation**:
- Original uses edge-level epoch scheduling; PyTorch version can batch all edges
  per epoch and weight by membership strength instead
- Original clips gradients per dimension; PyTorch can use `torch.clamp`
- Negative sampling is trivially parallel on GPU
- Spectral init via `torch.linalg.eigh` on normalized Laplacian

**The fuzzy set construction** (smooth_knn_dist) is a pre-processing step that
runs once on CPU — no need to port to PyTorch. It produces the graph weights
that feed into the optimization loop.

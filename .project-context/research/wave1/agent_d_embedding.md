Assumption: “configurable parameters” means the public `layout_*` function arguments; internal constants and unexposed heuristics are listed under “Hardcoded candidates”.

## [spectral.py](/home/jtaylor/projects/dagua/dagua/layout/classic/spectral.py)
**Execution order**
1. Validate `num_nodes` and `edge_weights` shape.
2. Choose output device from `edge_index`, else `node_sizes`, else CPU.
3. Return empty/zero tensors for `N=0/1`.
4. Build directed CSR adjacency from `edge_index` and optional `edge_weights`; default edge value is `1.0`.
5. Symmetrize with `A + A^T` if needed.
6. Build Laplacian: `D-A`, `I-D^{-1/2}AD^{-1/2}`, or `I-D^{-1}A`.
7. Choose dense eigensolve for `N < 500`, sparse ARPACK solve otherwise.
8. Sort eigenpairs by ascending eigenvalue and keep the first nontrivial 2 coordinates.
9. If no nontrivial eigenvector exists, fall back to `x = linspace(-1,1,N)`, `y = 0`.
10. Center and rescale so max absolute coordinate is `1.0`.
11. Convert to `float32` on the chosen device.

**Configurable parameters**
- `edge_index: torch.Tensor`
- `num_nodes: int`
- `node_sizes: Optional[torch.Tensor] = None`
- `seed: int = 42`
- `edge_weights: Optional[torch.Tensor] = None`
- `normalization: str = "symmetric"`

**Data structures**
- `edge_index`: `[2, E]`
- `adjacency`: `scipy.sparse.csr_matrix [N, N]`
- `degrees`: `np.ndarray [N]`
- `laplacian`: `csr_matrix [N, N]`
- `eigenvalues`: `np.ndarray [K]`
- `eigenvectors`: `np.ndarray [N, K]`
- `positions`: `np.ndarray [N, 2]`
- No `layers` or `forces`.

**Solver / optimizer**
- Dense: `np.linalg.eigh` for symmetric Laplacians, `np.linalg.eig` for random-walk.
- Sparse: `scipy.sparse.linalg.eigsh` or `eigs` with `which="SM"` / `"SR"` and heuristic `ncv`.

**Convergence**
- No outer iterative convergence.
- Dense path is direct.
- Sparse path uses SciPy/ARPACK internal convergence defaults.

**Shared steps**
- `BuildAdjacency` with tsNET / UMAP / SGD2.
- Spectral eigensolve motif with UMAP initialization.
- Center+scale normalization with tsNET / UMAP.

**Ops catalog cross-reference**
- Matches: `BuildAdjacency`, `Eigendecomposition`, `CenterPositions`, `ScalePositions`.
- Missing: explicit `SymmetrizeAdjacency`, explicit `BuildLaplacian`, seeded sparse eigensolver control.
- Split suggestion: `SpectralInit` should be `SymmetrizeAdjacency -> BuildLaplacian -> Eigendecomposition`.
- Merge suggestion: the postpass is really one `NormalizePositions` op here.

**Hardcoded candidates**
- `SPARSE_EIGEN_THRESHOLD = 500`
- `_EIGENVALUE_TOLERANCE = 1e-9`
- Output dimension fixed at `2`
- Rescale target `1.0`
- Sparse `k` and `ncv` heuristics
- Linear fallback `[-1,1]`

**Exact RNG usage**
1. The `seed` argument is assigned to `_` and otherwise ignored.
2. Dense path uses no RNG.
3. Sparse `eigsh` / `eigs` omit `v0` and `rng`; per local SciPy docs that means a random start vector is generated internally with a new NumPy generator from OS entropy. That randomness is not controlled by `seed`.

## [tsnet.py](/home/jtaylor/projects/dagua/dagua/layout/classic/tsnet.py)
**Execution order**
1. Validate `num_nodes`, `perplexity`, and `steps`.
2. Choose device and return early for `N=0/1`.
3. Build sorted undirected adjacency with the shared helper.
4. Compute all-pairs shortest paths with BFS or Dijkstra.
5. Replace unreachable distances row-wise with `max_finite + 1`.
6. For each node, binary-search Gaussian precision `beta` to match target perplexity.
7. Symmetrize conditional probabilities into global `P`.
8. Initialize positions from tiny Gaussian noise.
9. For each step, apply early exaggeration if `step < 250`, compute exact t-SNE KL loss, backprop, then do the gains+momentum update.
10. Center and scale to a stable extent.

**Configurable parameters**
- `edge_index: torch.Tensor`
- `num_nodes: int`
- `node_sizes: Optional[torch.Tensor] = None`
- `perplexity: float = 30`
- `steps: int = 1000`
- `seed: int = 42`
- `edge_weights: Optional[torch.Tensor] = None`

**Data structures**
- `adjacency`: `list[list[tuple[int, float]]]`
- `distances`: `torch.Tensor [N, N]`
- `probabilities`: `torch.Tensor [N, N]`
- `positions`: `torch.Tensor [N, 2]`
- `update`: `torch.Tensor [N, 2]`
- `gains`: `torch.Tensor [N, 2]`
- No `layers` or persistent `forces`.

**Solver / optimizer**
- Exact autograd KL objective.
- Manual t-SNE gains+momentum step, not `torch.optim`.
- Momentum is `0.5` during exaggeration, `0.8` afterward.

**Convergence**
- Per-row perplexity search stops at `abs(error) < 1e-5` or 100 iterations.
- Outer optimization is fixed-step only.

**Shared steps**
- `BuildAdjacency` and `AllPairsShortestPaths` with UMAP / SGD2.
- Random normal init with SGD2.
- Normalize positions with UMAP.

**Ops catalog cross-reference**
- Matches: `RandomNormalInit`, `AllPairsShortestPaths`, `PerplexityMatch`, `KLDivergenceLoss`, `EarlyExaggeration`, `NormalizePositions`, `FixedSteps`.
- Missing: dedicated `TSNEGainsMomentumStep`.
- Split suggestion: keep `KLDivergenceLoss` separate from the custom update op.
- Merge suggestion: none beyond the normalize pass.

**Hardcoded candidates**
- `_MIN_DISTANCE = 1e-12`
- Beta search cap `100`
- Beta tolerance `1e-5`
- Init std `1e-4`
- Early exaggeration `12.0`
- Early exaggeration steps `250`
- `min_gain = 0.01`
- Learning-rate rule `max(N/48, 50)`
- Output dimension fixed at `2`

**Exact RNG usage**
1. For `N >= 2`, create `torch.Generator(device="cpu")`.
2. Call `generator.manual_seed(seed)`.
3. Call `torch.randn(num_nodes, 2, generator=generator, dtype=torch.float32)` once for initialization.
4. No later RNG calls occur.

## [umap_layout.py](/home/jtaylor/projects/dagua/dagua/layout/classic/umap_layout.py)
**Execution order**
1. Validate graph inputs and scalar hyperparameters.
2. Choose device and return early for `N=0/1`.
3. Build undirected adjacency; weighted mode converts each edge to cost `1 / max(weight, eps)`.
4. Compute all-pairs shortest paths with repeated BFS or Dijkstra.
5. Fill unreachable distances with `max(2 * max_finite, 1)`.
6. Extract `k` nearest neighbors per node.
7. Solve smooth-kNN bandwidths `sigma` and `rho` per node.
8. Build the directed fuzzy simplicial set, then symmetrize it with `w_ij + w_ji - w_ij w_ji`.
9. If original `edge_weights` exist, rescale fuzzy weights by summed undirected input weights.
10. Build spectral initialization, with special cases for `N=0/1/2` and empty fuzzy graphs.
11. Fit low-dimensional curve parameters `(a, b)` from `min_dist` and `spread`.
12. Pick epoch count, prune weak positive edges, and compute `epochs_per_sample`.
13. Run UMAP’s pairwise SGD with linearly decaying `alpha` and negative sampling.
14. Center and scale the final embedding.

**Configurable parameters**
- `edge_index: torch.Tensor`
- `num_nodes: int`
- `node_sizes: Optional[torch.Tensor] = None`
- `n_neighbors: int = 15`
- `min_dist: float = 0.1`
- `spread: float = 1.0`
- `n_epochs: Optional[int] = None`
- `learning_rate: float = 1.0`
- `negative_sample_rate: int = 5`
- `repulsion_strength: float = 1.0`
- `seed: int = 42`
- `edge_weights: Optional[torch.Tensor] = None`

**Data structures**
- `adjacency`: `list[list[int]]` or `list[list[tuple[int, float]]]`
- `distances`: `torch.Tensor [N, N]`
- `knn_indices`, `knn_distances`: `[N, K]`
- `sigmas`, `rhos`: `[N]`
- `head`, `tail`, `weight`: positive fuzzy graph edges `[E']`
- `embedding`: `torch.Tensor [N, 2]`
- `next_sample_epoch`, `next_negative_epoch`: `[E']`
- No `layers` or persistent `forces`.

**Solver / optimizer**
- Spectral initializer via dense `eigh` or sparse `eigsh`.
- `scipy.optimize.curve_fit` for `(a, b)`.
- Manual pairwise SGD with in-place positive and negative updates; no autograd optimizer.

**Convergence**
- Smooth-kNN search stops at tolerance `1e-5` or 64 iterations.
- `curve_fit` caps at `maxfev=10000`, else falls back.
- Outer embedding optimization is fixed-epoch only.

**Shared steps**
- APSP preprocessing with tsNET / SGD2.
- Spectral eigensolve with spectral.py.
- Normalize positions with tsNET.

**Ops catalog cross-reference**
- Matches: `BuildAdjacency`, `AllPairsShortestPaths`, `SmoothKNNBandwidth`, `FuzzySimplicialSet`, `CurveFit_ab`, `SpectralInit`, `UMAPCrossEntropyLoss`, `LRDecay`, `FixedSteps`.
- Missing: explicit `UMAPPairSGD`, `PositiveEdgeSamplingSchedule`, and weighted-cost inversion inside `BuildAdjacency`.
- Split suggestion: separate fuzzy-graph construction from pairwise SGD.
- Merge suggestion: none, except the postpass again behaves like one `NormalizePositions`.

**Hardcoded candidates**
- `_EPSILON = 1e-9`
- `_MIN_SPAN = 1e-6`
- `_MIN_SIGMA_SCALE = 1e-3`
- `_SMOOTH_K_TOLERANCE = 1e-5`
- `_SMOOTH_K_BINARY_SEARCH_STEPS = 64`
- `_SPECTRAL_SPARSE_THRESHOLD = 512`
- `_GRADIENT_CLIP_VALUE = 4.0`
- Default epoch rule `500 if N <= 10000 else 200`
- Curve-fit start/fallback `(1.93, 0.79)`
- Special-case init range `[-10, 10]`
- Negative-gradient denominator offset `0.001`

**Exact RNG usage**
1. `_spectral_initialization` is path-dependent:
2. If `num_nodes <= 2`, no RNG.
3. If `head.numel() == 0`, create local CPU `torch.Generator`, `manual_seed(seed)`, then call `torch.rand((num_nodes, 2), generator=generator, dtype=torch.float32)` once.
4. Else, after spectral coordinates are computed, create a new local CPU `torch.Generator`, `manual_seed(seed)`, then call `torch.randn((num_nodes, 2), generator=generator, dtype=torch.float32)` once for noise.
5. If the sparse `eigsh` branch is taken, SciPy also draws its own random start vector because `v0` / `rng` are not supplied; that randomness is outside `seed`.
6. In `_optimize_embedding`, if `head.numel() > 0` and `n_epochs > 0`, create another fresh local CPU `torch.Generator`, `manual_seed(seed)`.
7. Every negative sample then consumes one `torch.randint(0, num_nodes, (1,), generator=generator)` in `(epoch, edge_id, while-loop)` order.

## [neulay.py](/home/jtaylor/projects/dagua/dagua/layout/classic/neulay.py)
**Execution order**
1. Validate public arguments.
2. Choose device and return early for `N=0/1`.
3. Resolve `magnitude`; default is `100 * N^(1/3) * radius`.
4. Seed PyTorch, NumPy, and CUDA.
5. Remove self-loops and move edges to the optimization device.
6. If `use_gcn and gcn_steps > 0`, build the normalized adjacency and initialize `_ResGCN`.
7. Run the GCN phase with RMSprop on `elastic + KD-tree Gaussian repulsion`, refreshing KD-tree pairs every 5 steps.
8. Otherwise, create Xavier-uniform initial positions directly.
9. Compute remaining direct-phase budget.
10. If direct budget is positive, optimize positions with RMSprop on the same loss, again refreshing KD-tree pairs every 5 steps.
11. Return detached coordinates.

**Configurable parameters**
- `edge_index: torch.Tensor`
- `num_nodes: int`
- `node_sizes: Optional[torch.Tensor] = None`
- `seed: int = 42`
- `steps: int = 20_000`
- `gcn_steps: int = 2_000`
- `use_gcn: bool = True`
- `dim: int = 2`
- `lr: float = 0.01`
- `radius: float = 0.4`
- `magnitude: Optional[float] = None`
- `edge_weights: Optional[torch.Tensor] = None`

**Data structures**
- `cleaned_edge_index`: `[2, E_clean]`
- `pos`: `[N, dim]`
- KD-tree `pairs`: `np.ndarray [M, 2]`
- Normalized adjacency: sparse COO `[N, N]`
- `_ResGCN` tensors: `weight1 [N,100]`, `h1 [N,100]`, `h2 [N,3]`, `weight2 [203, dim]`
- Sliding `loss_window`: Python `list[float]` of length 10
- No `layers` or persistent `forces`.

**Solver / optimizer**
- Two RMSprop optimizers:
- GCN phase uses `torch.optim.RMSprop(model.parameters(), lr=0.01)`.
- Direct phase uses `torch.optim.RMSprop([pos], lr=lr)`.
- Loss is `elastic_loss + kdtree_repulsion_loss`.

**Convergence**
- GCN phase breaks when relative loss-window range `< 1e-4 * sqrt(N)`.
- Direct phase breaks when relative loss-window range `< 1e-8 * sqrt(N)`.
- Both otherwise run to their step budgets.

**Shared steps**
- Xavier init is unique among these files.
- Autograd optimization loop motif overlaps tsNET / SGD2.
- KD-tree refresh is unique here but matches the planned op catalog.

**Ops catalog cross-reference**
- Matches: `XavierInit`, `ElasticLoss`, `KDTreeRepulsionLoss`, `RefreshKDTreePairs`, `RMSpropStep`, `GCNForward`, `SlidingWindowRelative`.
- Missing: explicit `BuildNormalizedAdjacency`.
- Split suggestion: represent the GCN phase as a short pipeline, not one monolith.
- Merge suggestion: none.

**Hardcoded candidates**
- `_PATIENCE = 10`
- `_GCN_REL_TOL = 1e-4`
- `_LINEAR_REL_TOL = 1e-8`
- `_GNN_LR = 0.01`
- `_PAIR_QUERY_RADIUS_FACTOR = 4.0`
- `_PAIR_REFRESH_INTERVAL = 5`
- Adaptive magnitude formula
- GCN hidden sizes `100` and `3`
- `edge_weights` are accepted but unused

**Exact RNG usage**
1. `_set_seed(seed)` calls `torch.manual_seed(seed)`, `np.random.seed(seed)`, and `torch.cuda.manual_seed_all(seed)` if CUDA is available.
2. If the GCN phase runs, model construction consumes global PyTorch RNG in this order:
3. `nn.init.xavier_uniform_` for `weight1`.
4. `nn.init.xavier_uniform_` for `gcn1.weight`.
5. `nn.init.xavier_uniform_` for `gcn2.weight`.
6. `nn.init.xavier_uniform_` for `weight2`.
7. If the GCN phase is skipped, `_initial_positions` performs one `nn.init.xavier_uniform_` on `[N, dim]`.
8. No later stochastic calls appear; KD-tree queries and both optimizers are deterministic after initialization.
9. NumPy’s RNG is seeded but not consumed elsewhere in this file.

## [sgd2_multi.py](/home/jtaylor/projects/dagua/dagua/layout/classic/sgd2_multi.py)
**Execution order**
1. Validate arguments, choose device, return early for `N=0/1`.
2. Resolve criterion schedules; default is pure stress.
3. Determine which precomputes are needed from the active criteria.
4. Seed PyTorch and CUDA.
5. Prepare shared graph state: unique undirected edges, adjacency, optional APSP stress terms, optional incident-edge tuples, optional non-incident edge pairs.
6. Build one cyclic sampler per active criterion pool.
7. Initialize positions from Gaussian noise scaled by `sqrt(N)`.
8. Optionally initialize the neural crossing detector and its Adam optimizer.
9. Optionally initialize vertex-resolution state.
10. Build Nesterov SGD for positions and `ReduceLROnPlateau`.
11. For each outer step, zero grads and accumulate weighted criterion losses in schedule order.
12. Each criterion samples its own batch and evaluates one of: stress, ideal edge length, neighborhood preservation, crossings, crossing-angle maximization, aspect ratio, angular resolution, or vertex resolution.
13. Backprop, clamp gradient values, and step SGD.
14. Update EMA-smoothed loss, call the LR scheduler every 10 iterations, and stop if LR reaches `1e-5`.
15. Return detached positions.

**Configurable parameters**
- `edge_index: torch.Tensor`
- `num_nodes: int`
- `node_sizes: Optional[torch.Tensor] = None`
- `seed: int = 42`
- `steps: int = 10_000`
- `criteria: Optional[Dict[str, float]] = None`
- `criteria_schedules: Optional[Dict[str, SmoothSteps]] = None`
- `lr: float = 1.0`
- `momentum: float = 0.7`
- `grad_clamp: float = 4.0`
- `batch_size: int = 16`
- `edge_weights: Optional[torch.Tensor] = None`

**Data structures**
- `_PreparedState`: `edges`, `adjacency`, `all_pairs_distances`, `stress_pairs`, `stress_distances`, `stress_weights`, `incident_edge_pairs`, `non_incident_edge_pairs`
- `edges`: `[2, E_unique]`
- `stress_pairs`: `[2, P]`
- `incident_edge_pairs`: `[5, P]`
- `non_incident_edge_pairs`: `[4, P]`
- `positions`: `[N, 2]`
- `_CyclicSampler`: permutation tensor + offset
- `_CrossingLossState`: detector, optimizer, BCE losses
- `_VertexResolutionState`: previous target distance + smoothing weight
- No `layers`; `forces` are implicit in gradients, not stored.

**Solver / optimizer**
- Main layout optimizer: `torch.optim.SGD(..., momentum=momentum, nesterov=True)`.
- Crossing detector inner optimizer: `torch.optim.Adam(..., lr=0.01)`.
- LR scheduler: `ReduceLROnPlateau(factor=0.9, patience=20000, min_lr=1e-5)`.

**Convergence**
- Formal stop is `optimizer.lr <= 1e-5`.
- With default `steps=10000` and scheduler patience `20000`, LR usually never decays; practically this is fixed-step.
- Crossing detector always trains exactly 2 inner steps per outer iteration.

**Shared steps**
- Random normal init with tsNET.
- APSP preprocessing with tsNET / UMAP.
- Batch SGD loop motif with NeuLay / tsNET.
- Smooth schedule and plateau scheduler are unique here.

**Ops catalog cross-reference**
- Matches: `RandomNormalInit`, `AllPairsShortestPaths`, `StressLoss`, `MultiCriteriaLoss`, `SGDNesterovStep`, `SmoothStepsSchedule`, `ReduceLROnPlateau`, `LRThreshold`.
- Missing: `GradientValueClamp`, `CyclicSampler`, `CrossingDetectorTrainStep`, and most named constituent criteria.
- Split suggestion: `MultiCriteriaLoss` should be split into separate criterion ops plus a combiner.
- Merge suggestion: none.

**Hardcoded candidates**
- `_DEFAULT_IDEAL_EDGE_LENGTH = 1.0`
- `_DEFAULT_ASPECT_RATIO_TARGET = 1.0`
- `_VERTEX_RESOLUTION_SMOOTHNESS = 0.1`
- Neighborhood depth `2`
- Neighborhood negative sample rate `0.5`
- `_NEIGHBORHOOD_K_DIST = 1.5`
- Crossing detector train steps `2`
- Crossing detector LR `0.01`
- Detector widths `128/512/64`
- Init scale `sqrt(N)`
- Scheduler factor `0.9`, patience `20000`, min LR `1e-5`
- EMA half-life `100`
- Scheduler interval `10`

**Exact RNG usage**
1. `_set_seed(seed)` calls `torch.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)` if CUDA is available.
2. Sampler construction consumes RNG first: each created `_CyclicSampler` calls `torch.randperm(total, device=device)` once, in `for sname in schedules` order.
3. Position initialization then calls `torch.randn((num_nodes, 2), device=device, dtype=torch.float32)`.
4. If `"crossings"` is active, `_CrossingDetector()` construction consumes CPU PyTorch RNG through the default `nn.Linear.reset_parameters()` calls for its 4 Linear layers, in declaration order.
5. During optimization, `_CyclicSampler.sample()` consumes another `torch.randperm(...)` whenever a sampler exhausts its current permutation.
6. `neighborhood_preservation` also consumes one `torch.randint(0, num_nodes, (negative_count,), device=device)` per evaluation when `negative_count > 0`.
7. `crossing_angle_maximization` only falls back to extra random sampling if `state.non_incident_edge_pairs is None`, via repeated `_sample_indices -> torch.randint(...)`.
8. No local `torch.Generator` objects are used; all draws go through the global generator seeded in step 1.

## Cross-Algorithm Patterns
- None of these five algorithms use `layers`; only SGD2 stores persistent auxiliary state, and none stores a persistent `forces` buffer.
- Reused motifs: device selection from `edge_index`/`node_sizes`, `N=0/1` short-circuits, undirected adjacency preprocessing, center+scale normalization, and fixed outer step budgets with smaller inner convergence checks.
- Distance-based preprocessing is shared by tsNET, UMAP, and SGD2; spectral eigensolves are shared by `spectral.py` and UMAP initialization.
- Direct catalog matches are strongest for `BuildAdjacency`, `AllPairsShortestPaths`, `Eigendecomposition`, `PerplexityMatch`, `SmoothKNNBandwidth`, `FuzzySimplicialSet`, `CurveFit_ab`, `ElasticLoss`, `KDTreeRepulsionLoss`, `RMSpropStep`, `RandomNormalInit`, `SGDNesterovStep`, `ReduceLROnPlateau`, `SlidingWindowRelative`, `LRThreshold`, and `NormalizePositions`.
- The catalog is weakest around seeded sparse eigensolvers, t-SNE’s gains/momentum update, UMAP’s pairwise SGD with negative sampling, NeuLay’s normalized-adjacency build, and SGD2’s criterion-specific losses/samplers.
- Best split candidates: `MultiCriteriaLoss`, spectral preprocessing vs eigensolve, and UMAP fuzzy-graph construction vs pairwise SGD.
- Best merge candidate for these implementations: `CenterPositions + ScalePositions` into one `NormalizePositions` op.

Codex session ID: 019d4fcd-cc1e-7ee3-826f-d72aa009abf1
Resume in Codex: codex resume 019d4fcd-cc1e-7ee3-826f-d72aa009abf1

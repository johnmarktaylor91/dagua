# Wave 1: Primitive Operations Implementation Plan

## Overview

Implement ~150 concrete Op subclasses across 20 category files in `dagua/layout/ops/`.
These ops are the atomic vocabulary sufficient to express the ALGORITHM LOGIC of all
24 classic algorithms and the native engine. Engine execution infrastructure (tiled GPU,
subset GPU, hybrid CPU/GPU) stays in engine.py for now -- ops must be callable FROM
those strategies but do not replace them.

Research basis: 7 Codex agents crawled all algorithm code (`.project-context/research/wave1/`).
DESIGN.md's original ~80-op taxonomy validated at ~85% accuracy; this plan refines it.

## Ground Rules

- Every op subclasses `Op` or `LossOp` from `base.py`
- Every op has `@register_op` from `taxonomy.py`
- Every configurable op has a companion frozen `@dataclass` config
- Every op sets `name`, `category`, `reads`, `writes`, `requires`
- Every op that uses randomness MUST match the RNG backend of the classic/ code:
  - torch.Generator ops: use ctx.generator or create local Generator(device="cpu")
  - numpy ops: use np.random.default_rng(seed) or np.random.RandomState(seed)
  - Python random ops: use random.Random(seed) (private instance, never module-global)
  - SciPy-internal: document as "uncontrolled" in op docstring
  The op's docstring MUST specify which RNG backend and exact call sequence.
- DO NOT modify `base.py`, `state.py`, `taxonomy.py`, or any `classic/` file
- Algorithm-specific transient state goes in `state.extras` dict with key convention
  "algo_field" (e.g. "tsne_gains", "umap_head"). Reads/writes metadata for extras
  keys is advisory only -- this is the documented tradeoff from DESIGN.md.
- Every op has unit tests in `tests/test_ops_{category}.py`

---

## Final Op Catalog

### INIT (9 ops) -- `ops/init.py`

| Op Name | Config | Used By | RNG | Key Behavior |
|---------|--------|---------|-----|-------------|
| RandomUniformInit | scale:str="sqrt_n", range:tuple=(0,1) | FR,GraphOpt,DRL,LGL,DH,SFDP,FA2 | torch.rand or random.Random | Uniform [0,1]^2 scaled |
| RandomNormalInit | std:float=1e-4, mean:float=0.0, scale:str="none" | tsNET,SGD2,LinLog,GEM | torch.randn | Normal N(0,std)^2 |
| CircularInit | scale:float=1.0 | KK | none | Deterministic linspace circle |
| SpectralInit | normalization:str="symmetric", sparse_threshold:int=500 | UMAP,engine(Fiedler) | scipy internal | Laplacian eigenvectors |
| ClassicalMDSInit | unreachable_fill:str="max_plus_1" | ClassicalMDS,StressMaj(warm) | none | Double-center + eigh |
| PivotMDSInit | n_pivots:int=50 | MaxEnt,PivotMDS | torch.randint(1 draw) | Pivot SVD embedding |
| XavierInit | gain_fn:str="default" | NeuLay | torch xavier_uniform_ | Xavier uniform init |
| FromAlgorithmInit | algorithm:str="fr", config:dict={} | FMMM | delegates to inner algo | Run another layout as init |
| DeterministicInit | method:str="barycenter" | Sugiyama,RT,engine | none | Layer-based coordinate assignment |

### PREPROCESS (5 ops) -- `ops/preprocess.py`

| Op Name | Config | Used By | Key Behavior |
|---------|--------|---------|-------------|
| DetectCycles | method:str="dfs_then_greedy" | engine,Sugiyama | DFS back-edge detection + greedy fallback |
| MakeAcyclic | (none) | engine,Sugiyama | Reverse detected back-edges |
| ClassifyGraph | large_graph_cutoff:int=10_000_000 | engine | Classify family: tree/chain/forest/wide/general |
| BuildAdjacency | directed:bool=False, weighted:bool=False, dedup:str="min", format:str="list", keep_multiplicity:bool=False, weight_transform:str="none" | 20+ algos | edge_index -> adjacency (list/dense/csr). dedup="min"/"sum"/"keep_all"; format="list"/"dense"/"csr"; weight_transform="none"/"inverse" (for UMAP cost conversion) |
| DetectComponents | (none) | engine,LGL,MaxEnt | Union-find connected components |

### DISTANCE (6 ops) -- `ops/distance.py`

| Op Name | Config | Used By | Key Behavior |
|---------|--------|---------|-------------|
| BFSDistances | unreachable:int=-1 | KK,MDS,PivotMDS,StressSGD,Maj,ME,tsNET,UMAP | Single-source BFS |
| DijkstraDistances | unreachable:float=inf | same (weighted) | Single-source Dijkstra |
| AllPairsShortestPaths | method:str="auto", unreachable_fill:str="max_plus_1" | 8 algos | Repeated BFS/Dijkstra for all sources |
| PivotSelection | n_pivots:int=50, method:str="maxmin" | PivotMDS,StressSGD(lg),ME | Seeded first pivot + maxmin continuation |
| PivotDistanceQueries | (none -- uses pivot_indices from state) | PivotMDS,StressSGD(lg),ME | BFS/Dijkstra from each pivot |
| ConnectivityCheck | (none) | StressSGD,engine | BFS from node 0, return bool |

### LAYERING (4 ops) -- `ops/layering.py`

| Op Name | Config | Used By | Key Behavior |
|---------|--------|---------|-------------|
| LongestPathLayering | (none) | Sugiyama,engine,multi | Kahn-based topological layering |
| LayerPromotion | (none) | Sugiyama | Push nodes to deepest legal layer |
| BuildLayerIndex | enable_cuda_sort:bool=True | engine | Sorted node index by layer |
| InsertDummyNodes | (none) | Sugiyama | Expand multi-layer edges with dummy nodes |

### ORDERING (4 ops) -- `ops/ordering.py`

| Op Name | Config | Used By | Key Behavior |
|---------|--------|---------|-------------|
| BarycenterSweep | passes:int=24, direction:str="both", use_weights:bool=True | Sugiyama,engine init | Weighted barycenter ordering |
| MedianSweep | passes:int=24 | Sugiyama variant | Median-based ordering |
| TransposeHeuristic | passes:int=8 | engine init | Adjacent swap to reduce crossings |
| SpectralOrder | (none) | engine init(Fiedler) | LOBPCG Fiedler vector ordering |

### COORDINATE (2 ops) -- `ops/coordinate.py`

| Op Name | Config | Used By | Key Behavior |
|---------|--------|---------|-------------|
| BrandesKopf4Pass | node_sep:float=1.0, rank_sep:float=1.0 | Sugiyama | 4-orientation x-coordinate assignment |
| BucheimWalkerTree | sibling_sep:float=1.0, layer_sep:float=1.5, component_gap:float=2.0 | Reingold-Tilford | Buchheim linear-time tree layout |

### COARSEN (4 ops) -- `ops/coarsen.py`

| Op Name | Config | Used By | Key Behavior |
|---------|--------|---------|-------------|
| HeavyEdgeMatching | (none) | SFDP | Random-order heavy-edge matching |
| SolarSystemCoarsen | random_tries:int=20, target:int=50 | FMMM | Sun/planet/moon hierarchy |
| LayerAwareCoarsen | hub_threshold_percentile:float=90, min_nodes:int=2000, max_levels:int=20 | engine multilevel | Layer-compatible pair/triple grouping |
| StreamingCoarsen | chunk_size:int=100_000_000 | engine(100M+) | Streaming segmented-sort coarsening |

### PROLONG (3 ops) -- `ops/prolong.py`

| Op Name | Config | Used By | Key Behavior |
|---------|--------|---------|-------------|
| DirectMapping | jitter_scale:float=5.0 | SFDP,engine | Copy coarse pos by mapping + jitter |
| LambdaInterpolation | waggle_factor:float=0.05 | FMMM | Weighted interpolation + random waggle |
| NeighborSmoothing | blend_factor:float=0.5 | SFDP | Smooth toward neighbor mean |

### FORCE (17 ops) -- `ops/force.py`

| Op Name | Config | Used By | Key Behavior |
|---------|--------|---------|-------------|
| ZeroForces | (none) | FR,GEM,FA2,GraphOpt,SFDP,FMMM,LGL | Zero state.forces buffer |
| InverseDistanceRepulsion | k_formula:str="area" | FR,GEM,FA2(exact),LinLog(exact) | k^2/d repulsion -> forces |
| InverseSquareRepulsion | charge:float=0.001, cutoff:float=500 | GraphOpt | Coulomb 1/d^2 -> forces |
| InversePowerRepulsion | exponent:float=-1.0 | SFDP | General k/d^p repulsion |
| UniformSpringAttraction | k_formula:str="area", spring_length:float=0.0, spring_constant:float=1.0 | FR,GraphOpt | Hooke spring -> forces. spring_length/constant for GraphOpt's explicit params |
| DesiredLengthSpringAttraction | (none -- uses spring_lengths from state) | FMMM,GEM | Spring with per-edge desired length |
| FA2DegreeCompensatedAttraction | linlog:bool=False, dissuade_hubs:bool=False, outbound_compensation:bool=True | FA2 | FA2's degree-adjusted attraction |
| GravityToOrigin | strength:float=1.0, strong_mode:bool=False | FA2 | Pull toward (0,0) -> forces |
| GravityToBarycenter | constant:float=1/16 | GEM | Pull toward weighted center of mass |
| BarnesHutForce | theta:float=1.2 | FA2,SFDP,FMMM | Quadtree approximated repulsion |
| DensityGridForce | grid_size:int=1000, view_size:int=4000, radius:int=10 | DRL | Density-grid energy evaluation |
| CellGridForce | cell_size:float=auto, repulse_rad:float=auto | LGL | Cell-bucket local repulsion |
| ApplyDisplacement | (none) | FR,GraphOpt,SFDP,FMMM,LGL | Read forces, clamp, update pos |
| AdaptiveSpeedApply | jitter_tolerance:float=1.0 | FA2 | FA2 global adaptive speed + apply |
| GEMNodeTick | (none) | GEM | Sequential single-node force+move+temp update |
| StressSGDPairUpdate | clamp_mu:float=1.0 | StressSGD | Sequential pair stress-SGD update |
| StressMajNodeSweep | (none -- uses laplacian from state) | StressMaj,MaxEnt(small) | SMACOF node-wise majorization sweep |

### LOSS -- Engine losses (16 ops) -- `ops/loss_engine.py`
### LOSS -- Classic algorithm losses (8 ops) -- `ops/loss_classic.py`

Combined listing (split into two files to avoid Codex agent conflicts):

### LOSS (24 ops) -- `ops/loss_engine.py` + `ops/loss_classic.py`

| Op Name | Config | Used By | Key Behavior |
|---------|--------|---------|-------------|
| DagOrderingLoss | rank_sep:float=50.0 | engine | Penalize upward edges |
| EdgeAttractionLoss | x_bias:float=4.0 | engine | Weighted edge distance |
| EdgeStraightnessLoss | (none) | engine | Penalize horizontal edge span |
| EdgeLengthVarianceLoss | (none) | engine | Edge length variance |
| RepulsionLoss | threshold:int=2000, sample_k:int=128, rvs_threshold:int=5000, rvs_nn_k:int=20 | engine | Inverse-distance repulsion (5 strategies) |
| OverlapAvoidanceLoss | padding:float=2.0, rvs_threshold:int=100000 | engine | Bbox overlap penalty (5 strategies) |
| CrossingLoss | alpha:float=5.0, max_pairs:int=2000 | engine | Sigmoid crossing proxy |
| ClusterCompactnessLoss | (none) | engine | Mean distance to cluster centroid |
| ClusterSeparationLoss | padding:float=10.0 | engine | Cluster bbox overlap |
| ClusterContainmentLoss | padding:float=18.0 | engine | Child within parent bbox |
| SpacingConsistencyLoss | target_gap:float=25.0 | engine | Consecutive same-layer gap variance |
| FanoutDistributionLoss | degree_threshold:int=5 | engine | Hub child angular distribution |
| BackEdgeCompactnessLoss | (none) | engine | Horizontal span of back-edges |
| PositionPinLoss | (none) | engine | Distance to pinned targets |
| AlignmentLoss | (none) | engine | Variance along alignment axis |
| FlexSpacingLoss | (none) | engine | Delegated spacing with flex weight |
| ExactPairStressLoss | weight_fn:str="inverse_sq" | StressSGD,StressMaj,ME,SGD2 | Sum (d_ij - ||p_i-p_j||)^2 * w_ij |
| PivotApproxStressLoss | (none -- uses pivot_distances from state) | ME(large),SGD2 | Pivot-approximated stress |
| KLDivergenceLoss | exaggeration:float=12.0, exaggeration_steps:int=250 | tsNET | t-SNE KL divergence |
| UMAPCrossEntropyLoss | neg_rate:int=5, repulsion_strength:float=1.0 | UMAP | UMAP attraction + negative sampling repulsion |
| LinLogAttractionLoss | exponent_a:float=1.0 | LinLog | log(1+d) * w attraction |
| LinLogRepulsionLoss | exponent_r:float=0.0 | LinLog | d^(r-1) repulsion |
| EntropyLoss | alpha:float=1.0 | MaxEnt | Non-edge entropy regularizer |
| DavidsonHarelEnergyLoss | w_distribution:float=1.0, w_border:float=0.1, w_edge_length:float=0.2, w_crossing:float=2.0, w_node_edge:float=0.5 | Davidson-Harel | 5-term SA energy function |
| SGD2CrossingDetectorStep | inner_steps:int=2, detector_lr:float=0.01 | SGD2 | Train crossing detector + return position loss |
| SGD2CriterionLoss | criterion:str="stress", batch_size:int=16 | SGD2 | Single criterion evaluation with cyclic sampling |
| CyclicSampler | pool_size:int=auto | SGD2 | Cyclic permutation-based batch sampler (stored in extras) |

### EMBED (11 ops) -- `ops/embed.py`

| Op Name | Config | Used By | Key Behavior |
|---------|--------|---------|-------------|
| SymmetrizeAdjacency | (none) | spectral,UMAP | A + A^T |
| BuildLaplacian | normalization:str="symmetric" | spectral,engine init | D-A or I-D^{-1/2}AD^{-1/2} |
| BuildNormalizedAdjacency | add_self_loops:bool=True | NeuLay | D^{-1/2}(A+I)D^{-1/2} sparse COO (self-loops per reference) |
| Eigendecomposition | sparse_threshold:int=500, k:int=2 | Spectral,ClassicalMDS | eigh/eigsh for k smallest |
| SVD | (none) | PivotMDS | torch.linalg.svd |
| Pseudoinverse | (none) | StressMaj | np.linalg.pinv |
| GCNForward | hidden_sizes:tuple=(100,3), output_dim:int=2 | NeuLay | ResGCN forward pass (stores model in extras) |
| PerplexityMatch | perplexity:float=30, tol:float=1e-5, max_iter:int=100 | tsNET | Binary-search Gaussian precisions |
| SmoothKNNBandwidth | n_neighbors:int=15, tol:float=1e-5, max_iter:int=64 | UMAP | Smooth kNN sigma/rho per node |
| FuzzySimplicialSet | (none) | UMAP | Directed->symmetric fuzzy graph |
| CurveFit_ab | min_dist:float=0.1, spread:float=1.0 | UMAP | scipy.optimize.curve_fit for UMAP a,b |

### OPTIMIZE (7 ops) -- `ops/optimize.py`

| Op Name | Config | Used By | Key Behavior |
|---------|--------|---------|-------------|
| CreateOptimizer | optimizer_type:str="adam", lr:float=0.05, target:str="pos", key:str="default" | engine,LinLog,ME,NeuLay,SGD2 | Create torch.optim; target="pos" uses state.pos, target="extras.X" uses extras[X]. Stored at state.optimizer (key="default") or extras["optimizer_<key>"] |
| OptimizerStep | key:str="default" | engine,LinLog,ME,NeuLay,SGD2 | optimizer.step() for the named optimizer |
| ClipGradNorm | max_norm:float=100.0 | engine,SGD2 | torch.nn.utils.clip_grad_norm_ |
| ClipGradValue | max_value:float=4.0 | SGD2 | Clamp gradient element-wise |
| LBFGSStep | maxiter:int=None | KK | SciPy L-BFGS-B minimize |
| TSNEGainsMomentumStep | lr_rule:str="N/48", min_gain:float=0.01 | tsNET | t-SNE gains+momentum custom update |
| UMAPPairSGD | neg_rate:int=5, clip:float=4.0 | UMAP | UMAP pairwise SGD with negative sampling |

### PROJECT (5 ops) -- `ops/project.py`

| Op Name | Config | Used By | Key Behavior |
|---------|--------|---------|-------------|
| OverlapProjection | padding:float=2.0, iterations:int=10 | engine | Multi-strategy overlap removal |
| HardPinProjection | (none -- uses flex from problem) | engine | torch.where on pinned axes |
| BoundaryClamp | extent:float=auto | Davidson-Harel | Clamp positions to bounding box |
| MovementClamp | mode:str="temperature" | FR,GraphOpt,SFDP,FMMM | Per-node displacement cap |
| MonotoneSafeguard | max_bisections:int=8, tolerance:float=1e-8 | StressMaj | Blend to prevent stress increase |

### ANNEAL (10 ops) -- `ops/anneal.py`

| Op Name | Config | Used By | Key Behavior |
|---------|--------|---------|-------------|
| LinearCool | rate:float=auto | FR | temperature -= cooling_step |
| ExponentialCool | factor:float=0.99 | FMMM,SFDP | temperature *= factor |
| AdaptiveCool | up_factor:float=1.1, down_factor:float=0.9 | SFDP | Adapt step from force progress |
| PerNodeTemperature | init_temp:float=12.0, min_temp:float=0.005 | GEM | Per-node oscillation/rotation cooling |
| PhaseSchedule | phases:list[PhaseConfig] | DRL | 6-phase temperature/attraction/damping |
| SmoothStepsSchedule | keyframes:dict | SGD2 | Smooth weight interpolation over steps |
| WeightAnnealing | (none -- uses annealing schedule from state) | engine | Update current_weights from schedule_fns |
| LRDecay | mode:str="linear", start_lr:float=auto, end_lr:float=auto | LinLog,ME,UMAP | Learning rate schedule |
| EarlyExaggeration | multiplier:float=12.0, until_step:int=250 | tsNET | Multiply P matrix during early steps |
| ReduceLROnPlateau | factor:float=0.9, patience:int=20000, min_lr:float=1e-5 | SGD2 | torch.optim.lr_scheduler wrapper |
| IdealLengthDecay | decay_factor:float=0.75 | SFDP | Decay ideal_length between multilevel levels |

### CONTEXT (5 ops) -- `ops/context.py`

| Op Name | Config | Used By | Key Behavior |
|---------|--------|---------|-------------|
| BuildEdgeBatchCtx | batch_size:int=auto | engine | Sample or chunk edge batch + precompute dx/dy/dist |
| RefreshSampledNodeCtx | interval:int=5, active_cap:int=auto | engine | Sample active nodes + same-layer/adjacent-layer peers |
| BuildQuadTree | max_depth:int=10 | FA2,SFDP,FMMM | Quadtree for Barnes-Hut |
| BuildDensityGrid | grid_size:int=1000, view_size:int=4000 | DRL | Density grid for DRL energy |
| RefreshKDTreePairs | radius:float=0.4, interval:int=5 | NeuLay | scipy.spatial.cKDTree pair refresh |

### CONVERGE (6 ops) -- `ops/converge.py`

| Op Name | Config | Used By | Key Behavior |
|---------|--------|---------|-------------|
| FixedSteps | n:int | most algos | Set state.total_steps, no convergence check |
| DisplacementThreshold | threshold:float=1e-4 | FR,LGL | Converge when mean displacement < threshold |
| TemperatureThreshold | min_temp:float=0.005 | GEM,SFDP | Converge when temperature drops |
| SlidingWindowRelative | window:int=10, tol:float=1e-4 | NeuLay | Converge when loss window range < tol*sqrt(N) |
| StallCount | limit:int=5, rel_threshold:float=1e-4 | engine | Converge after limit stalls |
| LRThreshold | min_lr:float=1e-5 | SGD2 | Converge when LR drops below threshold |

### POSTPROCESS (6 ops) -- `ops/postprocess.py`

| Op Name | Config | Used By | Key Behavior |
|---------|--------|---------|-------------|
| CenterPositions | (none) | 10+ algos | Subtract mean |
| ScalePositions | method:str="max_abs", factor:float=1.0 | 10+ algos | Scale to factor |
| NormalizePositions | extent_fn:str="sqrt_n_times_5" | MDS,PivotMDS,ME,SFDP,GEM,LinLog | Center + scale to computed extent |
| DirectionTransform | direction:str="TB" | engine,Sugiyama | Rotate/flip for TB/BT/LR/RL |
| StripDummyNodes | (none) | Sugiyama | Remove dummy nodes from positions |
| SpreadFanoutChildren | hub_threshold:int=8, widening:float=1.5 | engine init | Redistribute high-fanout children |

### EDGE_ROUTE (2 ops) -- `ops/edge_route.py`

| Op Name | Config | Used By | Key Behavior |
|---------|--------|---------|-------------|
| BezierControlPointOpt | lr:float=0.1, steps:int=auto, 6 weights | engine | Gradient-based Bezier optimization |
| ReconstructEdgeRoutes | (none) | Sugiyama | Build polylines from dummy node chains |

### UTILITY (7 ops) -- `ops/utility.py`

| Op Name | Config | Used By | Key Behavior |
|---------|--------|---------|-------------|
| Checkpoint | path:str=auto | engine multilevel | Serialize state to disk |
| DiskOffload | (none) | engine multilevel | Offload hierarchy level tensors |
| DiskReload | (none) | engine multilevel | Reload hierarchy level tensors |
| GarbageCollect | (none) | engine multilevel | gc.collect + empty_cache + malloc_trim |
| VRAMGuard | budget_fraction:float=0.85 | engine | Check VRAM before GPU ops |
| ProgressReport | file:str=None, interval:int=10 | engine | Write progress.json |
| Timer | (none) | (new) | Measure op execution time |

---

## File Assignments Summary

| File | Category | Op Count | Priority |
|------|----------|----------|----------|
| init.py | INIT | 9 | P1 (simple inits) / P2 (spectral/MDS inits) |
| preprocess.py | PREPROCESS | 5 | P1 |
| distance.py | DISTANCE | 6 | P1 |
| layering.py | LAYERING | 4 | P1 |
| ordering.py | ORDERING | 4 | P2 |
| coordinate.py | COORDINATE | 2 | P2 |
| coarsen.py | COARSEN | 4 | P2 |
| prolong.py | PROLONG | 3 | P2 |
| force.py | FORCE | 17 | P1 (biggest non-loss file) |
| loss_engine.py | LOSS (engine) | 16 | P1 |
| loss_classic.py | LOSS (classic) | 8 | P1 |
| embed.py | EMBED | 11 | P2 |
| optimize.py | OPTIMIZE | 7 | P1 |
| project.py | PROJECT | 5 | P2 |
| anneal.py | ANNEAL | 10 | P2 |
| context.py | CONTEXT | 5 | P1 (BuildQuadTree needed by force ops) |
| converge.py | CONVERGE | 6 | P1 |
| postprocess.py | POSTPROCESS | 6 | P1 |
| edge_route.py | EDGE_ROUTE | 2 | P3 |
| utility.py | UTILITY | 7 | P3 |

**Total: ~150 ops across 20 files** (revised after adversarial review round 1)

---

## Implementation Priority

**P1 (foundational -- implement first, others depend on these):**
init, preprocess, distance, layering, force, loss, optimize, converge, postprocess

**P2 (secondary -- depend on P1 ops):**
ordering, coordinate, coarsen, prolong, embed, project, anneal, context

**P3 (infrastructure -- can be stubs initially):**
edge_route, utility

---

## Testing Strategy

Each test file `tests/test_ops_{category}.py` must include:

1. **Per-op unit tests** on small graphs (5-20 nodes)
   - Verify correct SolveState field writes
   - Verify reads/writes metadata accuracy
   - Test config variations
   - Test edge cases: empty graph, single node, disconnected
   - Test device respect (ctx.plan.device)

2. **RNG fidelity tests** for ops with randomness
   - Run the op with seed=42
   - Run the equivalent classic/ code path with seed=42
   - Assert torch.allclose() on outputs
   - This is the CRITICAL correctness criterion

3. **Composition tests** (at least one per file)
   - Pipeline([op1, op2]) produces valid state
   - Repeat(n=3, ops=[op1]) increments step correctly

---

## Codex Agent Split for Implementation

### Batch 1 (P1 foundations -- dispatch first)

| Agent | Files | Ops | Est. Complexity |
|-------|-------|-----|-----------------|
| C1 | init.py (simple inits: RandomUniform, RandomNormal, Circular, Xavier, Deterministic) + tests | 5 | Medium (RNG fidelity critical) |
| C2 | preprocess.py + distance.py + tests | 11 | Medium (graph algorithms) |
| C3 | layering.py + context.py + tests | 9 | Medium (BuildQuadTree needed by force ops) |
| C4 | force.py + tests | 17 | HIGH (biggest file, force glue ops critical) |
| C5 | loss_engine.py + tests | 16 | HIGH (engine losses) |
| C6 | loss_classic.py + tests | 8 | HIGH (classic algo losses) |
| C7 | optimize.py + converge.py + tests | 13 | Medium |
| C8 | postprocess.py + tests | 6 | Low |

### Batch 2 (P2 secondary -- dispatch after Batch 1)

| Agent | Files | Ops | Est. Complexity |
|-------|-------|-----|-----------------|
| C9 | init.py (complex inits: SpectralInit, ClassicalMDSInit, PivotMDSInit, FromAlgorithmInit) + tests | 4 | Medium (depends on embed ops + other pipelines) |
| C10 | ordering.py + coordinate.py + tests | 6 | Medium (discrete algorithms) |
| C11 | coarsen.py + prolong.py + tests | 7 | HIGH (multilevel machinery) |
| C12 | embed.py + tests | 11 | HIGH (linear algebra + UMAP/NeuLay) |
| C13 | project.py + anneal.py + tests | 15 | Medium |

### Batch 3 (P3 infra -- dispatch last)

| Agent | Files | Ops | Est. Complexity |
|-------|-------|-----|-----------------|
| C14 | edge_route.py + utility.py + tests | 9 | Low-medium |

**Pre-dispatch checklist:**
- No two agents write the same file (VERIFIED after splitting loss.py into loss_engine.py + loss_classic.py)
- init.py split across C1 (simple) and C9 (complex): C9 runs after embed ops exist. C9 prompt includes C1's output.
- Shared imports from base, state, taxonomy are read-only (no conflicts)
- Test files are separate per agent
- C3 now includes context.py (BuildQuadTree) since force ops in C4 depend on it

---

## Key Design Decisions Baked Into This Plan

1. **CenterPositions and ScalePositions kept separate from NormalizePositions**.
   NormalizePositions = center + scale-to-computed-extent. The simple ops exist for
   algorithms that only need one or the other.

2. **Execution strategies are NOT ops**. Tiled GPU, subset GPU, per-loss backward,
   hybrid CPU/GPU stay in engine.py. Ops must be callable from those strategies.

3. **Unreachable fill is a config on distance ops**, not a separate op.

4. **Force pipeline pattern**: ZeroForces -> [force ops accumulate into state.forces]
   -> ApplyDisplacement (reads forces, clamps, updates pos). GEMNodeTick is the
   exception: sequential single-node force+move+temp-update for GEM's Gauss-Seidel.

5. **GCNForward added** to EMBED. NeuLay IS expressible. Model weights live in extras.
   CreateOptimizer supports targeting extras parameters via target="extras.X".

6. **loss.py split into two files**: loss_engine.py (16 engine losses) and
   loss_classic.py (8 classic algorithm losses). No file conflicts between agents.

7. **RNG contract loosened**: Ops match the RNG backend of their classic/ source
   (torch.Generator, numpy, Python random). Not unified to torch.Generator only.

8. **Multiple optimizers**: SGD2 and NeuLay need >1 optimizer. CreateOptimizer
   supports a `key` param; non-default optimizers stored in extras["optimizer_<key>"].

9. **Algorithm-specific state** uses extras dict (e.g. "tsne_gains", "umap_head",
   "sgd2_samplers"). Advisory reads/writes metadata -- documented tradeoff.

10. **InversePowerRepulsion added** for SFDP's general exponent repulsion.

---

## Verification Plan

After all implementation agents complete:

1. Run: `pytest tests/test_ops_*.py -x --tb=short`
2. Update `dagua/layout/ops/__init__.py` to import ALL category modules (init, preprocess,
   distance, layering, ordering, coordinate, coarsen, prolong, force, loss_engine,
   loss_classic, embed, optimize, project, anneal, context, converge, postprocess,
   edge_route, utility) so @register_op decorators fire on import.
3. Verify all ops are registered: `python -c "from dagua.layout.ops import list_ops; print(len(list_ops()))"`
   Expected: ~155
4. Update `dagua/layout/ops/DESIGN.md` with final catalog
5. Dispatch Codex review of full ops/ directory
6. Commit: `feat(ops): implement complete primitive operation library`

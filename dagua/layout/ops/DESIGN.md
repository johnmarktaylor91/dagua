# Composable Layout Operations -- Design Document

## Overview

Dagua's layout system is being restructured around composable operations.
Every graph layout algorithm -- from Fruchterman-Reingold to the native
billion-node engine -- is expressible as a Pipeline of typed, configurable,
documented operations.

This document defines the data structures that make this possible.

## Architecture

```
                       +------------------+
                       |   User / CLI     |
                       +--------+---------+
                                |
                    dagua.layout(graph, config)
                                |
                       +--------v---------+
                       |  LayoutProblem   |  <-- immutable inputs
                       |  (graph topology,|      (edge_index, node_sizes,
                       |   constraints,   |       clusters, flex, direction)
                       |   structure)     |
                       +--------+---------+
                                |
                       +--------v---------+
                       |   SolveState     |  <-- mutable working data
                       |  (pos, layers,   |      (positions, hierarchy,
                       |   hierarchy,     |       cached precomputes,
                       |   extras, ...)   |       convergence counters)
                       +--------+---------+
                                |
               +----------------+----------------+
               |                                 |
      +--------v---------+              +--------v---------+
      |  RuntimeContext   |              |    Pipeline      |
      |  (ExecutionPlan,  |              |  [Op, Op, ...]   |
      |   MemoryPolicy,   |              |                  |
      |   TraceSink, RNG) |              |  Composition:    |
      +-------------------+              |  - Sequential    |
                                         |  - Repeat(n)     |
                                         |  - Conditional   |
                                         |  - LossGroup     |
                                         |  - MultilevelVC  |
                                         +------------------+
```

## Three-Part State Model (Revised)

### LayoutProblem (immutable)

Read-only graph structure and user constraints. Created once, shared across
all ops. No changes from v1 except documentation clarifications.

Fields:
- edge_index: [2, E] -- graph topology
- num_nodes: int
- node_sizes: [N, 2] or None -- width, height per node
- node_labels: list[str] or None
- direction: str -- TB, BT, LR, RL
- clusters: dict or None -- cluster membership
- cluster_parents: dict or None -- cluster hierarchy
- structure: GraphStructure or None -- cached classification
- flex: FlexConstraints or None -- pins, alignments, spacing
- seed: int -- base RNG seed
- edge_weights: Tensor[E] or None -- **NEW** edge weight vector

Rationale for edge_weights: 15 of 25 classic algorithms accept edge_weights.
Putting it in LayoutProblem avoids passing it through every Op's config.

### SolveState (mutable)

Working data that ops read and write. Changes from v1:

NEW FIELDS:
- temperature: float or None -- global temperature (used by FR, SFDP,
  GEM, Davidson-Harel, DRL, FMMM, native engine annealing)
- ordering: Tensor[N] or None -- within-layer node ordering indices
  (used by Sugiyama, init_placement, crossing minimization)
- edge_routes: list or None -- polyline/curve routes per edge
  (used by Sugiyama dummy-node routes, edge optimization)
- extras: dict[str, Any] -- **escape hatch** for algorithm-specific
  transient state that does not belong in the core schema.
  Examples: local_temperatures (GEM), velocity (t-SNE), gains (t-SNE),
  density_grid (DRL), crossing_detector (SGD^2), walker_tree (RT)
- converged: bool -- flag set by convergence-checking ops

EXISTING FIELDS (kept as-is):
- pos, layers, layer_index, back_edge_mask, hierarchy
- adjacency, distance_matrix, pivot_indices, pivot_distances
- component_ids, degree, laplacian, affinity_matrix
- spring_lengths, spring_strengths
- sampled_node_context, edge_batch_context
- annealing, step, total_steps, prev_loss, stall_count, ops_applied

### HierarchyLevel (revised)

NEW FIELD:
- cluster_ids: Tensor[num_nodes] or None -- propagated cluster assignments
  (needed by multilevel coarsening to maintain cluster-compatible grouping)

### ExecutionPlan (revised)

NEW FIELD:
- subset_gpu_threshold: int = 10_000_000 -- node count above which
  subset_gpu mode is forced

### RuntimeContext (unchanged)

Same as v1: plan, memory, trace_sink, progress_file, generator, log_prefix.

### TraceSink Protocol (revised)

NEW METHOD:
- op_start(op_name: str, state: SolveState) -> None
- op_end(op_name: str, state: SolveState) -> None

These fire at Pipeline op boundaries, enabling per-op video frames.
NullTraceSink implements both as no-ops. The snapshot() and log() methods
remain for backward compatibility.

## Op Interface (Revised)

```python
class Op(ABC):
    # Class-level metadata
    name: str = "unnamed_op"
    category: OpCategory = OpCategory.UNKNOWN
    reads: tuple[str, ...] = ()    # SolveState fields read
    writes: tuple[str, ...] = ()   # SolveState fields written
    requires: tuple[str, ...] = () # fields that should already be set

    @abstractmethod
    def apply(self, problem, state, ctx) -> SolveState: ...

    def describe(self) -> str:
        """One-line human-readable description."""
        return f"{self.name} ({self.category.value})"
```

Changes from v1:
- category is now OpCategory enum instead of free-form string
- describe() method for user-facing documentation
- No other signature changes -- the three-argument apply() is correct

## OpConfig Pattern

Every configurable Op has a companion frozen dataclass:

```python
@dataclass(frozen=True)
class RepulsionConfig:
    """Configuration for repulsion force computation."""
    exact_threshold: int = 2000
    sample_k: int = 128
    rvs_threshold: int = 5000
    rvs_nn_k: int = 20

class RepulsionLoss(Op):
    name = "repulsion"
    category = OpCategory.LOSS

    def __init__(self, config: RepulsionConfig | None = None):
        self.config = config or RepulsionConfig()

    def apply(self, problem, state, ctx):
        # uses self.config.exact_threshold, etc.
        ...
```

Design rules:
1. Config is a frozen dataclass (immutable after creation).
2. Every Op constructor takes config=None and uses defaults.
3. Config field names match the algorithm's original parameter names
   where possible (e.g., FR's "cooling_factor", not "decay_rate").
4. Configs are serializable to/from dict for persistence.
5. There is NO OpConfig base class -- each config is independent.
   Type checking uses isinstance(x, dataclass) if needed.

Rationale for no base class: configs have nothing in common except being
frozen dataclasses. A base class would add ceremony without value.

## Operation Taxonomy

Every distinct operation found across all 25 classic algorithms and the
native engine, grouped by category.

### OpCategory Enum

```
INIT          Position initialization
PREPROCESS    Graph structure prep (cycles, classify, adjacency)
DISTANCE      Graph-theoretic distance computation
LAYERING      Layer/rank assignment (DAG structure)
ORDERING      Within-layer node ordering
COORDINATE    Coordinate assignment from ordering
COARSEN       Multilevel hierarchy building
PROLONG       Multilevel prolongation/refinement setup
FORCE         Per-step force computation (non-differentiable)
LOSS          Differentiable loss evaluation
EMBED         Spectral/dimensionality reduction
OPTIMIZE      Parameter update (optimizer step)
PROJECT       Hard constraint enforcement (non-differentiable)
ANNEAL        Weight/temperature/schedule evolution
CONTEXT       Per-step shared state building
CONVERGE      Convergence/stopping criteria
POSTPROCESS   Centering, scaling, direction transforms
EDGE_ROUTE    Post-layout edge curve optimization
UTILITY       Checkpoint, offload, trace, timing
CONTROL       Pipeline, Repeat, Conditional, LossGroup, VCycle
```

### Full Taxonomy Table

| Category | Operation | Used By | Key Config |
|----------|-----------|---------|------------|
| INIT | RandomUniformInit | FR, GraphOpt, DRL, LGL, DH, SFDP | scale, range |
| INIT | RandomNormalInit | tsNET, SGD^2, LinLog, GEM | std, mean |
| INIT | CircularInit | KK | scale |
| INIT | SpectralInit | UMAP, engine(Fiedler) | normalization, k |
| INIT | MDS/PivotMDSInit | MaxEnt, StressMaj(warm) | n_pivots |
| INIT | XavierInit | NeuLay | gain_fn |
| INIT | FromAlgorithmInit | FMMM(uses FR) | algorithm, config |
| INIT | DeterministicInit | Sugiyama, RT | (none) |
| PREPROCESS | DetectCycles | engine, Sugiyama | method(dfs,greedy) |
| PREPROCESS | MakeAcyclic | engine, Sugiyama | (reverse back-edges) |
| PREPROCESS | ClassifyGraph | engine | threshold |
| PREPROCESS | BuildAdjacency | 15+ algorithms | directed, weighted |
| PREPROCESS | DetectComponents | engine, LGL | (none) |
| DISTANCE | BFSDistances | KK, MDS, PivotMDS, StressSGD, Maj, ME, tsNET, UMAP | (source node) |
| DISTANCE | DijkstraDistances | same (when weighted) | (source node) |
| DISTANCE | AllPairsShortestPaths | 8 algorithms | method(bfs,dijkstra) |
| DISTANCE | PivotDistances | PivotMDS, StressSGD(lg), ME | n_pivots, selection |
| DISTANCE | TriangleApprox | StressSGD(large) | (from pivot dists) |
| LAYERING | LongestPathLayering | Sugiyama, engine, multi | (none) |
| LAYERING | LayerPromotion | Sugiyama | (none) |
| LAYERING | BuildLayerIndex | engine | device |
| LAYERING | InsertDummyNodes | Sugiyama | (none) |
| ORDERING | BarycenterSweep | Sugiyama, engine init | passes, direction |
| ORDERING | MedianSweep | Sugiyama variant | passes |
| ORDERING | TransposeHeuristic | engine init | passes |
| ORDERING | SpectralOrder | engine init(Fiedler) | k |
| COORDINATE | BrandesKopf4Pass | Sugiyama | node_sep, rank_sep |
| COORDINATE | BucheimWalkerTree | Reingold-Tilford | sibling_sep, layer_sep |
| COARSEN | HeavyEdgeMatching | SFDP | reduction_target |
| COARSEN | SolarSystemCoarsen | FMMM | random_tries, target |
| COARSEN | LayerAwareCoarsen | engine multilevel | hub_threshold, triple |
| COARSEN | StreamingCoarsen | engine(100M+) | chunk_size |
| PROLONG | DirectMapping | SFDP, engine | jitter_scale |
| PROLONG | LambdaInterpolation | FMMM | waggle_factor |
| PROLONG | NeighborSmoothing | SFDP | blend_factor |
| FORCE | CoulombRepulsion | FR, GEM, FMMM, GraphOpt | k, exponent |
| FORCE | SpringAttraction | FR, GEM, FMMM, GraphOpt, FA2 | k, formula |
| FORCE | GravityToOrigin | FA2 | strength, strong_mode |
| FORCE | GravityToBarycenter | GEM | constant |
| FORCE | BarnesHutForce | FA2, SFDP, FMMM | theta |
| FORCE | DensityGridForce | DRL | grid_size, view, radius |
| FORCE | CellGridForce | LGL | cell_size, repulse_rad |
| FORCE | AdaptiveSpeedApply | FA2 | jitter_tolerance |
| FORCE | GaussSeidelPairUpdate | StressSGD, StressMaj, ME | clamp_mu |
| FORCE | GuttmanTransform | StressMaj | (none) |
| FORCE | SMACOFUpdate | StressMaj | monotone_safeguard |
| LOSS | DagOrderingLoss | engine | rank_sep |
| LOSS | EdgeAttractionLoss | engine | x_bias |
| LOSS | EdgeStraightnessLoss | engine | (none) |
| LOSS | EdgeLengthVarianceLoss | engine | (none) |
| LOSS | RepulsionLoss | engine | threshold, sample_k |
| LOSS | OverlapAvoidanceLoss | engine | padding |
| LOSS | CrossingLoss | engine | alpha, max_pairs |
| LOSS | ClusterCompactnessLoss | engine | (none) |
| LOSS | ClusterSeparationLoss | engine | padding |
| LOSS | ClusterContainmentLoss | engine | padding |
| LOSS | SpacingConsistencyLoss | engine | target_gap |
| LOSS | FanoutDistributionLoss | engine | degree_threshold |
| LOSS | BackEdgeCompactnessLoss | engine | (none) |
| LOSS | PositionPinLoss | engine | (none) |
| LOSS | AlignmentLoss | engine | (none) |
| LOSS | FlexSpacingLoss | engine | (none) |
| LOSS | StressLoss | StressSGD, StressMaj, ME, SGD^2 | weight_fn |
| LOSS | ElasticLoss | NeuLay | (none) |
| LOSS | KDTreeRepulsionLoss | NeuLay | radius, magnitude |
| LOSS | KLDivergenceLoss | tsNET | exaggeration |
| LOSS | UMAPCrossEntropyLoss | UMAP | a, b, neg_rate |
| LOSS | LinLogAttractionLoss | LinLog | exponent_a |
| LOSS | LinLogRepulsionLoss | LinLog | exponent_r |
| LOSS | EntropyLoss | MaxEnt | alpha |
| LOSS | MultiCriteriaLoss | SGD^2 | criteria dict |
| LOSS | EnergyFn(5-term) | Davidson-Harel | 5 weights |
| EMBED | Eigendecomposition | Spectral, ClassicalMDS | normalization |
| EMBED | SVD | PivotMDS | (none) |
| EMBED | Pseudoinverse | StressMaj | (none) |
| EMBED | FiedlerVector | engine init | max_iter |
| EMBED | PerplexityMatch | tsNET | perplexity, tol |
| EMBED | SmoothKNNBandwidth | UMAP | n_neighbors |
| EMBED | FuzzySimplicialSet | UMAP | (none) |
| EMBED | CurveFit_ab | UMAP | min_dist, spread |
| EMBED | GCNForward | NeuLay | architecture |
| OPTIMIZE | AdamStep | LinLog, ME(grad), engine | lr |
| OPTIMIZE | SGDNesterovStep | SGD^2 | lr, momentum |
| OPTIMIZE | LBFGSStep | KK | maxiter |
| OPTIMIZE | RMSpropStep | NeuLay | lr |
| OPTIMIZE | OptimizerStep | engine | (from ExecutionPlan) |
| OPTIMIZE | ClipGradNorm | engine, SGD^2 | max_norm |
| PROJECT | OverlapProjection | engine | padding, iters, method |
| PROJECT | HardPinProjection | engine | (none) |
| PROJECT | BoundaryClamp | Davidson-Harel | extent |
| PROJECT | MovementClamp | FR, GraphOpt, SFDP, FMMM | max_delta |
| PROJECT | MonotoneSafeguard | StressMaj | max_bisections |
| ANNEAL | LinearCool | FR | rate |
| ANNEAL | ExponentialCool | FMMM, SFDP | factor |
| ANNEAL | AdaptiveCool | SFDP | up_factor, down_factor |
| ANNEAL | PerNodeTemperature | GEM | init_temp, min_temp |
| ANNEAL | PhaseSchedule | DRL | phase_configs |
| ANNEAL | SmoothStepsSchedule | SGD^2 | keyframes |
| ANNEAL | WeightAnnealing | engine | schedule_fns |
| ANNEAL | LRDecay | LinLog, ME, UMAP | mode, start, end |
| ANNEAL | EarlyExaggeration | tsNET | multiplier, until_step |
| ANNEAL | ReduceLROnPlateau | SGD^2 | factor, patience |
| CONTEXT | BuildEdgeBatchCtx | engine | batch_size |
| CONTEXT | RefreshSampledNodeCtx | engine | interval, cap |
| CONTEXT | BuildQuadTree | FA2, SFDP, FMMM | (none) |
| CONTEXT | BuildDensityGrid | DRL | grid_size, view |
| CONTEXT | RefreshKDTreePairs | NeuLay | radius, interval |
| CONVERGE | FixedSteps | most algorithms | n |
| CONVERGE | DisplacementThreshold | FR | threshold |
| CONVERGE | TemperatureThreshold | GEM, SFDP | min_temp |
| CONVERGE | SlidingWindowRelative | NeuLay | window, tol |
| CONVERGE | StallCount | engine | limit, rel_threshold |
| CONVERGE | LRThreshold | SGD^2 | min_lr |
| POSTPROCESS | CenterPositions | 10+ algorithms | (none) |
| POSTPROCESS | ScalePositions | 10+ algorithms | method, factor |
| POSTPROCESS | NormalizePositions | MDS, PivotMDS, ME, SFDP, GEM, LinLog | extent_fn |
| POSTPROCESS | DirectionTransform | engine, Sugiyama | direction |
| POSTPROCESS | StripDummyNodes | Sugiyama | (none) |
| POSTPROCESS | HorizontalFlip | Reingold-Tilford | (none) |
| EDGE_ROUTE | BezierControlPointOpt | engine | lr, steps, 6 weights |
| EDGE_ROUTE | ReconstructEdgeRoutes | Sugiyama | (none) |
| UTILITY | Checkpoint | engine multilevel | path |
| UTILITY | DiskOffload | engine multilevel | (none) |
| UTILITY | DiskReload | engine multilevel | (none) |
| UTILITY | GarbageCollect | engine multilevel | (none) |
| UTILITY | VRAMGuard | engine | budget_fraction |
| UTILITY | ProgressReport | engine | file, interval |
| UTILITY | Snapshot | Pipeline trace | (none) |
| UTILITY | Timer | (new) | (none) |
| CONTROL | Pipeline | -- | name, trace_between |
| CONTROL | Repeat | -- | n |
| CONTROL | Conditional | -- | predicate, else_op |
| CONTROL | LossGroup | -- | losses, backward_mode |
| CONTROL | MultilevelVCycle | -- | coarsen, base, refine |
| CONTROL | EarlyBreak | -- | predicate |

## Composition Patterns

### 1. Pipeline (sequential)

```python
Pipeline([
    ClassifyGraph(),
    InitPositions(),
    OptimizationLoop,
    DirectionTransform(),
])
```

REVISED: Pipeline now calls ctx.trace_sink.op_start/op_end at each
boundary when trace_between=True.

### 2. Repeat (fixed iteration loop)

```python
Repeat(n=50, ops=[
    ComputeForces(),
    ApplyDisplacement(),
    LinearCool(),
])
```

No changes from v1.

### 3. Conditional (predicate branching)

REVISED: Conditional now supports else_op.

```python
Conditional(
    predicate=lambda p, s, c: p.num_nodes > 20000,
    op=MultilevelPipeline,
    else_op=DirectPipeline,
)
```

### 4. LossGroup (NEW -- shared-context loss accumulation)

Evaluates multiple loss ops sharing the same autograd graph, then
calls backward() once (or per-loss if configured).

```python
LossGroup(
    losses=[
        DagOrderingLoss(config=...),
        RepulsionLoss(config=...),
        OverlapAvoidanceLoss(config=...),
    ],
    backward_mode="combined",  # or "per_loss"
)
```

This captures the engine's _backward_standard_loss_terms pattern.
In combined mode: sum weighted losses, single backward().
In per_loss mode: each loss backward() separately, intermediates freed.

### 5. MultilevelVCycle (revised skeleton)

```python
MultilevelVCycle(
    coarsen_op=LayerAwareCoarsen(config=...),
    base_layout=Pipeline([...]),  # coarsest level
    refine=Pipeline([...]),       # per-level refinement
    min_nodes=2000,
    max_levels=20,
)
```

REVISED: add max_levels parameter. The skeleton remains a skeleton
for this sprint -- concrete implementation comes in the migration sprint.

### 6. EarlyBreak (NEW -- loop escape)

Used inside Repeat to break out of the loop:

```python
Repeat(n=500, ops=[
    ComputeForces(),
    ApplyDisplacement(),
    EarlyBreak(predicate=lambda p, s, c: s.converged),
])
```

EarlyBreak sets state.converged = True. Repeat checks this flag
after each iteration and stops if set.

## Example Pipelines

### Fruchterman-Reingold

```python
Pipeline([
    RandomUniformInit(config=RandomUniformInitConfig(scale="sqrt_n")),
    Repeat(n=50, ops=[
        CoulombRepulsion(config=CoulombConfig(k_formula="area")),
        SpringAttraction(config=SpringConfig(k_formula="area")),
        MovementClamp(config=ClampConfig(mode="temperature")),
        LinearCool(config=LinearCoolConfig()),
        EarlyBreak(predicate=mean_displacement_below(1e-4)),
    ]),
    CenterAndScale(config=ScaleConfig(factor="sqrt_n_times_50")),
], name="fruchterman_reingold")
```

### Sugiyama (Hierarchical DAG)

```python
Pipeline([
    MakeAcyclic(),
    LongestPathLayering(),
    LayerPromotion(),
    InsertDummyNodes(),
    Repeat(n=24, ops=[
        BarycenterSweep(config=SweepConfig(direction="down")),
        BarycenterSweep(config=SweepConfig(direction="up")),
    ]),
    BrandesKopf4Pass(config=BrandesKopfConfig(node_sep=1.0, rank_sep=1.0)),
    StripDummyNodes(),
    DirectionTransform(),
], name="sugiyama")
```

### Native Dagua Engine

```python
Pipeline([
    ClassifyGraph(),
    BuildLayerIndex(),
    Conditional(
        predicate=is_tree_or_chain,
        op=OverrideTreeWeights(),
    ),
    InitPositions(config=InitPosConfig(use_spectral_for_large=True)),
    Conditional(
        predicate=exceeds_multilevel_threshold,
        op=MultilevelVCycle(
            coarsen_op=LayerAwareCoarsen(),
            base_layout=optimization_loop(steps="coarse"),
            refine=optimization_loop(steps="refine"),
        ),
        else_op=optimization_loop(steps="total"),
    ),
    Conditional(predicate=has_relax_steps, op=RelaxPass()),
    DirectionTransform(),
], name="dagua_native")
```

Where optimization_loop is:

```python
def optimization_loop(steps):
    return Repeat(n=steps, ops=[
        BuildEdgeBatchCtx(),
        RefreshSampledNodeCtx(config=SampleConfig(interval=5)),
        WeightAnnealing(),
        LossGroup(
            losses=[
                DagOrderingLoss(), RepulsionLoss(),
                OverlapAvoidanceLoss(), CrossingLoss(),
                EdgeAttractionLoss(), EdgeStraightnessLoss(),
                # ... all 16 losses
            ],
            backward_mode="auto",  # per_loss when N > 50K
        ),
        ClipGradNorm(config=ClipConfig(max_norm=100.0)),
        OptimizerStep(),
        HardPinProjection(),
        Conditional(
            predicate=is_projection_step,
            op=OverlapProjection(),
        ),
        StallCount(config=StallConfig(limit=5, rel_threshold=1e-4)),
        EarlyBreak(predicate=lambda p, s, c: s.converged),
    ])
```

## Registry and Discovery

```python
# taxonomy.py

_OP_REGISTRY: dict[str, type[Op]] = {}

def register_op(cls: type[Op]) -> type[Op]:
    """Class decorator. Registers an Op by its name."""
    _OP_REGISTRY[cls.name] = cls
    return cls

def get_op_class(name: str) -> type[Op]:
    """Look up a registered Op class by name."""
    return _OP_REGISTRY[name]

def list_ops(category: OpCategory | None = None) -> list[str]:
    """List registered op names, optionally filtered by category."""
    ...

def list_categories() -> list[OpCategory]:
    """List all OpCategory values."""
    ...
```

Usage:
```python
@register_op
class RepulsionLoss(Op):
    name = "repulsion_loss"
    category = OpCategory.LOSS
    ...
```

Adding a new op = one file, one class, @register_op decorator.

## Dependency System

The v1 advisory system (reads/writes/requires + Pipeline.lint()) is
retained and enhanced:

1. reads/writes/requires remain documentation-level metadata.
2. Pipeline.lint() checks requires vs. available writes (unchanged).
3. NEW: Pipeline.dependency_graph() returns a dict mapping op names
   to their declared reads/writes, enabling visualization.
4. Enforcement remains advisory -- runtime dispatch is too dynamic
   for static guarantees (e.g., Conditional branches, extras dict).

Example:
```python
class RepulsionLoss(Op):
    reads = ("pos", "sampled_node_context")
    writes = ()
    requires = ("pos",)
```

If RepulsionLoss is placed before any init op in a Pipeline,
lint() will warn: "repulsion_loss expects 'pos' to be set".

## The extras Dict

SolveState.extras is a dict[str, Any] escape hatch for algorithm-specific
transient state. Guidelines:

1. Use extras for state needed by only 1-2 algorithms.
2. Use typed SolveState fields for state shared by 3+ ops.
3. Key naming convention: "algo_field" (e.g., "gem_local_temperatures",
   "tsne_velocity", "drl_density_grid").
4. Ops that write to extras should declare it in writes:
   writes = ("extras.gem_local_temperatures",)
5. Ops that read from extras should declare it in reads:
   reads = ("extras.gem_local_temperatures",)

## Migration Path

### Current state -> Ops v2 (this sprint)
- Revise state.py with new fields (temperature, ordering, extras, etc.)
- Revise base.py with LossGroup, EarlyBreak, Conditional.else_op,
  Pipeline.trace_between
- Add taxonomy.py with OpCategory enum and registry
- Add DESIGN.md (this document)
- All existing tests continue to pass

### Ops v2 -> Concrete ops (next sprint)
- Implement ~80 concrete Op subclasses from the taxonomy table
- Each classic algorithm becomes a named Pipeline
- Native engine becomes a named Pipeline
- All reimplemented algorithms verified against classic/ references

### Concrete ops -> Engine migration (future)
- engine.py _layout_inner becomes Pipeline.__call__
- multilevel.py becomes MultilevelVCycle.apply()
- LayoutConfig maps to op configs
- Public API unchanged: dagua.layout() still works

## Adversarial Review Findings and Resolutions

Round 1 review by Claude architect agent found 30 issues (6 CRITICAL).
All CRITICAL and HIGH issues resolved in this revision:

### CRITICAL (all resolved)

1. **Force accumulation gap** -- SolveState had no forces field.
   FIX: Added `forces: Tensor | None` and `old_forces: Tensor | None`.
   Force ops accumulate into forces buffer. ApplyDisplacement reads it.

2. **LossOp backward gap** -- Op.apply() returns SolveState, but loss
   evaluation needs a scalar tensor return channel.
   FIX: Added `LossOp` subclass with `evaluate() -> torch.Tensor`.
   `LossGroup` calls evaluate(), sums, calls backward().

3. **Missing SolveState fields** -- DESIGN.md described fields not in code.
   FIX: Implemented all: temperature, ordering, edge_routes, extras,
   converged, forces, old_forces, optimizer.

4. **Missing LayoutProblem.edge_weights** -- 15 algorithms need it.
   FIX: Added `edge_weights: Tensor | None` to LayoutProblem.

5. **OpCategory enum missing** -- Referenced but not implemented.
   FIX: Created taxonomy.py with OpCategory enum and registry.

6. **Pipeline not an Op** -- Cannot nest in Conditional.
   FIX: Pipeline now subclasses Op with apply() method.

### HIGH (all resolved)

7. **EarlyBreak + Repeat** -- Repeat had no convergence check.
   FIX: Added `converged` field to SolveState. Repeat now checks it.
   Added EarlyBreak op. Repeat auto-increments state.step.

8. **Conditional.else_op missing** -- Design described it, code lacked it.
   FIX: Added else_op parameter (default None, backward compatible).

9. **Optimizer state** -- No home for Adam moments, SGD momentum.
   FIX: Added `optimizer: Any | None` to SolveState.

10. **TraceSink missing op_start/op_end** -- No visualization hooks.
    FIX: Added to TraceSink protocol and NullTraceSink.

11. **HierarchyLevel missing cluster_ids** -- Needed for coarsening.
    FIX: Added field.

12. **ExecutionPlan missing subset_gpu_threshold** -- Needed for scaling.
    FIX: Added field with default 10M.

### Design decisions (from review, documented but not blocking)

- **Gauss-Seidel ops**: StressSGD and UMAP process pairs sequentially.
  These are implemented as monolithic ops (not decomposed into per-pair
  sub-ops). LossGroup is for batch gradient, not Gauss-Seidel.

- **Neural sub-models**: SGD^2's CrossingDetector and NeuLay's ResGCN
  live in `extras` dict. Inner training loops are acceptable inside
  evaluate() for ops that wrap learning procedures.

- **Edge cutting** (DRL): DRL dynamically modifies the graph.
  Uses `extras["drl_adjacency"]` for a mutable copy.
  LayoutProblem.edge_index remains immutable (original topology).

- **Hybrid CPU/GPU backward**: LossGroup's combined mode handles this
  internally when RuntimeContext.plan.hybrid_cpu_gpu is True. The
  _GradBridge pattern lives inside LossGroup, not as a separate op.

- **Tiled GPU compute**: Tiling is a LossGroup execution strategy
  (activated when plan.mode == "tiled_gpu"), not a separate op.

- **Step counter**: Repeat auto-increments state.step after each
  iteration. Ops that need step count read state.step.

- **NumPy interop**: Some classic algorithms operate in NumPy.
  Ops handle conversion internally. Large matrices (pseudoinverse)
  live in extras.

## Constraints Verification

C1  USER CLARITY: OpCategory enum, typed configs, describe() method,
    Pipeline repr with op names.
C2  GENERALITY: 20 categories, 80+ operations covering all algorithms.
C3  DOCUMENTATION: All classes get docstrings. This DESIGN.md.
C4  COMPOSABILITY: Pipeline, Repeat, Conditional, LossGroup, VCycle, EarlyBreak.
C5  COMPLETENESS: Every operation from every algorithm represented.
    Cross-verified against audit of all 25 classic + native engine.
C6  UTILITY OPS: Checkpoint, DiskOffload, GarbageCollect, Timer, Snapshot.
C7  VISUALIZATION: TraceSink.op_start/op_end + Pipeline.trace_between.
C8  EXTENSIBILITY: @register_op + one file + one class.
C9  INTEGRATION: LayoutProblem/SolveState/RuntimeContext unchanged in spirit.
    Engine headless contract preserved.
C10 DEPENDENCY: reads/writes/requires + lint() + dependency_graph().
C11 UNIT COHERENCE: Each op is one algorithmic step (verified against audit).
C12 CONFIGURABILITY: Frozen dataclass configs per op. Field names match
    original algorithm parameters. No missing options.

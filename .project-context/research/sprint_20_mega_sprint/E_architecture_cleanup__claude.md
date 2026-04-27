# E -- Architecture Cleanup & Topology-Dispatch Design

Agent: Claude Opus 4.7 (1M context), sprint-20 research agent E.
Scope: **Independent** second opinion on how to de-Frankenstein the native
pipeline while giving each topology bucket a sub-pipeline tailored to it.

## TL;DR

- The native pipeline file is 1,336 lines / 22 functions / 10 ops in one giant
  `build_dagua_pipeline` body plus an adapter that runs its own
  component-decomposition loop before the pipeline even starts.
  (`dagua/layout/ops/pipelines/dagua_native.py:1336`,
  `dagua/layout/ops/pipelines/dagua_native.py:939-1177`,
  `dagua/layout/ops/pipelines/dagua_native.py:1272-1333`.)
  Every sprint-19 patch is gated by a different predicate (`is_acyclic`,
  `num_layers>=2`, `num_components>1`, `max_degree<=4`, `lattice_like`...),
  and those predicates are scattered between the adapter, `resolve.py`, and
  the pipeline body. This is the Frankenstein smell.
- I propose splitting the native engine into **four named sub-pipelines**
  (`native_tree`, `native_layered_dag`, `native_force_directed`,
  `native_hybrid`), with a single topology classifier deciding which one
  runs. All four share a reusable `GradientCore` and postprocessing tail, so
  this is **not** a code duplication move -- it is a cleanup that removes
  ~300 lines of conditionals from the default path.
- Every sprint-19 op survives, but each one is hosted by exactly the
  sub-pipeline where it makes sense. BK refinement and dummy-node insertion
  become unconditional members of `native_layered_dag` and disappear from
  the force-directed path entirely; median+transpose likewise.
- Four config flags (`decompose_components`, `insert_dummy_nodes`,
  `brandes_koepf_refine`, `use_native_median_transpose`) collapse into a
  single `native_pipeline` enum with `auto` as the default. Users who want
  fine-grained override keep getting it through a new `native_overrides`
  dataclass; the top-level `LayoutConfig` surface becomes smaller.
- `classify_graph` already has the bones we need, but needs four additions:
  a `family` value for genuinely non-hierarchical graphs
  (`FORCE_DIRECTED`), a `has_dominant_component` flag, a `cyclicity_ratio`
  (fraction of edges participating in cycles), and a `num_layers_effective`
  that collapses spurious long-path layerings into their 3-5 useful layers.
- Migration: one new `dispatch.py` module + four new pipeline modules, done
  in a single PR. Regression safety comes from a gated rollout: keep
  `native_pipeline="legacy_monolith"` available for one sprint, bench
  every protected graph on both paths.
- Risk: two of the strongest wins (`org_chart_deep +22.67`,
  `random_dag_200 +20.88`) are produced by the sprint-19 stack on layered
  DAGs. They must land in `native_layered_dag` bit-for-bit. Adversarial
  regression tests (not just benchmark composites) are the right gate.

The rest of this report works through the audit, the proposed taxonomy, the
four sub-pipeline sketches, the refactor diff, and the migration plan.

---

## 1. Current-State Audit

### 1.1 File sizes and cyclomatic smell

```
dagua/layout/engine.py                         3,476 lines   (legacy body + dispatcher)
dagua/layout/resolve.py                          501 lines   (config-time resolvers)
dagua/layout/graph_classify.py                   523 lines   (topology classifier)
dagua/layout/ops/pipelines/dagua_native.py     1,336 lines   (the default pipeline)
dagua/layout/ops/pipelines/sugiyama.py           187 lines   (clean)
dagua/layout/ops/pipelines/fr.py                 162 lines   (clean)
median across other 22 pipelines                ~170 lines   (all clean)
```

Source of truth: `wc -l` on the tree.

The mean non-dagua pipeline is ~170 lines; the dagua native pipeline is
**~8x larger**. That is the Frankenstein number. It's not just the
pipeline body -- it's also the six private helper functions that the
adapter walks through before the pipeline even runs.

### 1.2 What's inside the 1,336 lines

Using real line anchors from the file I just read:

| Segment | Lines | Responsibility |
|---|---|---|
| Imports + constants | 1-92 | Ops, types, packing constants |
| `_resolve_native_layer_assignments` | 95-124 | Shim: longest-path layering or use cached |
| `_has_long_layer_edges` | 126-148 | Predicate for dummy-node gate |
| `_should_use_native_dummy_nodes` | 151-189 | 5-clause gate for dummy insertion |
| `_stress_pivot_prep` | 192-207 | Conditional BFS-pivot ops |
| `build_gradient_core` | 210-277 | **The reusable chunk** -- Adam + loss + overlap |
| `_build_refine_pipeline_factory` | 280-336 | V-cycle refine factory |
| `_build_coarse_init_pipeline_factory` | 339-375 | V-cycle coarse factory |
| `_prepare_native_config` | 378-447 | Shim: resolve + stash private attrs |
| `_run_native_problem` | 450-507 | Per-problem runner w/ tree fast-path |
| `_has_pins` / `_has_cross_component_flex` | 510-559 | Component-decomp gates |
| `_should_decompose_components` | 562-613 | 7-clause gate for component split |
| `_subset_flex` | 616-684 | Project flex to a child |
| `_extract_component_problem` | 687-770 | Build child problem |
| `_grid_dimensions` / `_choose_component_grid` / `_row_major_offsets` / `_tile_component_positions` | 773-936 | **Component packer** |
| `build_dagua_pipeline` | 939-1177 | The actual op pipeline (with a V-cycle fork) |
| `layout_dagua_native_pipeline` | 1180-1333 | The adapter loop |

Only **`build_gradient_core` and `build_dagua_pipeline`** are actual op
composition. Everything else is gating, relabeling, packing, or
adapter-orchestration. That's the asymmetry -- the pipeline body itself is
reasonable; the surrounding scaffolding has silently outgrown the
abstraction.

### 1.3 The five sprint-19 additions, as gating predicates

`dagua_native.py:989-1021` is the crux:

```python
is_acyclic = bool(structure.is_directed_acyclic)                  # sprint-19f,g,h gate
enable_native_median_transpose = bool(config.use_native_median_transpose)
enable_brandes_koepf_refine = bool(config.brandes_koepf_refine)
crossing_reduction_ops = [BarycenterReorder(...)]
if enable_native_median_transpose and is_acyclic:                  # silent no-op on cyclic
    crossing_reduction_ops.extend([MedianSweep(...), TransposeHeuristic(...)])
crossing_reduction_ops.append(BrandesKoepfHorizontalRefine(...))   # sprint-19g
```

And `dagua_native.py:1110-1117` is:

```python
*([InsertDummyNodes(), ActivateExpandedGraphState()] if resolved_use_dummy_nodes else []),
```

The adapter runs its own component-decomp loop at
`dagua_native.py:1272-1333`. Each predicate is a separate branch:

| Sprint patch | Predicate location | Predicate content |
|---|---|---|
| Per-component decomp (19d) | `_should_decompose_components` | 7 clauses: `num_components>1` and not dominant and no cluster and no pin and no cross-component flex |
| Dummy-node split (19h) | `_should_use_native_dummy_nodes` | 5 clauses: enabled, acyclic, single component, layers>1, N>=20, not dense, has long edge |
| Median+transpose (19f) | `dagua_native.py:1006` inline | `use_native_median_transpose and is_acyclic` |
| BK x-refine (19g) | `BrandesKoepfHorizontalRefineConfig(enabled=..., structure=...)` | Op reads structure internally |
| Topology-aware aspect (19e) | `resolve_topology_aware_aspect` | Walks `structure.topology_tags` |

Every predicate asks one of these: "is this a DAG?", "is this layered?",
"is this disconnected?", "is this dense?". **That's the signal: there are
only a handful of real topology types, and we should classify once and
dispatch to a matched sub-pipeline.**

### 1.4 Config sprawl

`dagua/config.py` exposes these native-only flags today:

```
insert_dummy_nodes:            bool = True      # config.py:148
decompose_components:          bool = True      # config.py:152
brandes_koepf_refine:          bool = True      # config.py:144
use_native_median_transpose:   bool = True      # config.py:138
use_tree_fast_path:            bool = False     # config.py:134
native_median_passes:          int  = 4         # config.py:139
native_transpose_passes:       int  = 8         # config.py:140
multilevel_threshold:          int  = 20000     # config.py:155 (effectively disabled)
w_stress, w_stress_n_pivots:   opt-ins          # config.py:111-115
```

That's 9 algorithmic dials exposed on the top-level LayoutConfig, all
documented as "pipeline tuning". Four of them (`insert_dummy_nodes`,
`decompose_components`, `brandes_koepf_refine`,
`use_native_median_transpose`) are currently "on by default, silently
no-ops when topology doesn't match". They are the config-level shadow of
the Frankenstein -- they tell the user that they can turn knobs that, in
practice, only do anything for certain graph families.

### 1.5 Classifier inventory

`graph_classify.py` already yields a rich `GraphStructure` at
`graph_classify.py:31-46`:

```python
family: GraphFamily            # GENERAL/TREE/FOREST/CHAIN/BIPARTITE_DAG/WIDE_LAYERED/GRID (unused)
num_components, max_degree
num_layers, avg_layer_width, max_layer_width, layer_width_cv
is_planar_hint, is_acyclic, is_directed_acyclic
edge_to_node_ratio
topology_tags: tuple[str, ...]   # lattice_like, planar_dag, wide_layered, dense_dag, bipartite_dag
```

The enum has `GRID` but no code path emits it
(`graph_classify.py:19-28`). Likewise there is no family for
**genuinely cyclic / non-hierarchical** graphs -- small_world, Erdos-Renyi,
regular graphs, complete graphs all fall into `GENERAL`, which is exactly
the bucket where dagua still loses
(`small_world_100 -8.51`, `small_world_500 -4.82`, `regular_3_30 -3.86`).
This is precisely the hole a force-directed sub-pipeline should fill.

## 2. Proposed Topology Dispatch

### 2.1 Categories the engine should dispatch on

I propose four runtime buckets (enums below are concrete identifiers I'd
add to `graph_classify.GraphFamily`):

| Bucket | New / existing | Detection rule | What wins it today |
|---|---|---|---|
| `TREE` (inc. CHAIN / FOREST) | existing fast-path | `num_components>=1 and is_acyclic and E == N - num_components` | Reingold-Tilford |
| `LAYERED_DAG` | new **umbrella** for existing BIPARTITE_DAG + WIDE_LAYERED + "general acyclic with useful layering" | `is_directed_acyclic and num_layers_effective >= 2` | Sugiyama-family ops: dummy nodes, median+transpose, BK x-refine |
| `FORCE_DIRECTED` | new (fills `GENERAL ∧ not is_directed_acyclic` case) | `not is_directed_acyclic or (is_directed_acyclic and num_layers_effective <= 1)` | Stress / FR / SGD^2 |
| `HYBRID` | new (semi-layered, partially cyclic) | DAG skeleton exists but cyclicity_ratio > 0.05 OR num_layers_effective in {2,3} AND max_layer_width_cv > 0.8 | Mixed approach: layer the DAG core, spring-relax the cycles |

Secondary tags (carry through inside each bucket, inherited from the
existing topology tags):

- `lattice_like`, `planar_dag`, `dense_dag`, `bipartite_dag`, `wide_layered`
  -- already in `topology_tags`, keep them. Used by aspect-ratio policy
  and by fine-tuning switches inside a bucket.

### 2.2 New classifier fields

Add to `GraphStructure` (`graph_classify.py:31-46`):

```python
num_layers_effective: int         # Hide the Sugiyama-style long-path
                                  # layering when it's just a contrived
                                  # chain. Defined as the number of layers
                                  # that contain >= 2 nodes when layers
                                  # are collapsed so singleton-chains
                                  # merge with their neighbor.
cyclicity_ratio: float            # fraction of edges NOT in a valid
                                  # DAG skeleton -- i.e., how back-edge
                                  # heavy is this graph?
has_dominant_component: bool      # largest_comp >= 0.85 * N
                                  # (moves the gate out of the native
                                  # adapter into the classifier)
family: GraphFamily               # Extend enum with LAYERED_DAG,
                                  # FORCE_DIRECTED, HYBRID.
                                  # GENERAL becomes a fallback for
                                  # degenerate / empty graphs.
```

`cyclicity_ratio` and `num_layers_effective` change nothing about the cost
of classification -- both fall out of the acyclic check and the bincount
that already happens (`graph_classify.py:286-322`).

### 2.3 Decision tree (ASCII)

```
layout(graph, config)
  +- classify_graph(edge_index, num_nodes, layer_assignments)
  |    returns GraphStructure{family, tags, ...}
  |
  +- dispatch_native(structure)
       |
       +- TREE / CHAIN / FOREST      -> native_tree
       |                                 (Reingold-Tilford fast-path)
       |
       +- LAYERED_DAG                -> native_layered_dag
       |   (is_directed_acyclic AND    (always: dummy nodes, median,
       |    num_layers_effective>=2)    transpose, BK x-refine)
       |
       +- FORCE_DIRECTED             -> native_force_directed
       |   (cycles OR flat)              (always: Force2DInit,
       |                                   stress / repulsion / attraction,
       |                                   NO barycenter / BK)
       |
       +- HYBRID                     -> native_hybrid
           (DAG + small cycle density)   (DAG skeleton get layered,
                                          back-edges handled as springs)
```

The adapter's current component-decomposition wraps this entire block --
it applies orthogonally to the bucket choice, because "has multiple
weakly-connected components" is a preprocessing concern, not a topology
identity. Pull it out of `dagua_native.py` into its own
`component_decompose.py` wrapper op that runs the dispatcher once per
child.

Component packing stays where it is (in a dedicated module). That's
another 150 lines that has no business living next to the pipeline body.

## 3. Sub-Pipeline Sketches

Every sub-pipeline uses three shared building blocks:
- `GradientCore` (already extracted, `dagua_native.py:210-277`)
- `NativeSolveTail` (new): the "crossing_reduction_ops + OverlapProjection +
  StripDummyNodes + AspectRatioFit + ClusterGridArrange" block. Today that
  block is inline at `dagua_native.py:1149-1174`; promoting it to a
  named builder lets each sub-pipeline opt into the parts it needs.
- `NativeInitCore` (new): `NativeEngineInit` + optional `Force2DInitIfFlat`
  + optional `_stress_pivot_prep`. Currently inline at
  `dagua_native.py:1086-1124`.

### 3.1 `native_tree`

```
NativeInitCore(force_2d=False, stress_pivot=False)
|
+- ReingoldTilfordTree()   # fast-path, already in _run_native_problem:497
|
+- AspectRatioFit()        # keep final aspect
```

Already implemented as the tree fast-path at
`dagua_native.py:492-499`. This just promotes the early-return into an
explicit sub-pipeline the dispatcher can return to. No algorithmic
change.

### 3.2 `native_layered_dag`

```
NativeInitCore(force_2d=False, stress_pivot=(w_stress>0))
|
+- InsertDummyNodes()              # ALWAYS (was conditional)
+- ActivateExpandedGraphState()
|
+- build_gradient_core(...)        # unchanged -- Adam + losses + overlap
|
+- BarycenterReorder()             # ALWAYS
+- MedianSweep(passes=4)           # ALWAYS (was is_acyclic-gated)
+- TransposeHeuristic(passes=8)    # ALWAYS
+- BrandesKoepfHorizontalRefine()  # ALWAYS (was enabled/structure-gated)
|
+- OverlapProjection()
+- StripDummyNodes()
+- AspectRatioFit(topology-aware)
+- ClusterGridArrange()
```

Every sprint-19 layered-DAG improvement lives here **unconditionally**,
because by construction this sub-pipeline only runs on layered DAGs. All
the `if is_acyclic and num_layers>=2 and not dense_dag` gates evaporate
because the dispatcher already certified those facts. The dispatcher
writes the precondition once (in the classifier), not in every op's
config.

Losses: the current loss bundle is correct for this bucket; nothing to
change. `DagOrderingLoss` is guaranteed safe because the family
certifies acyclicity. The `is_acyclic` gate at `resolve.py:422` can
drop the check -- the dispatcher replaces it.

### 3.3 `native_force_directed`

This is the big new one, for the bucket where dagua lags competitors
today (`small_world_100 -8.51`, `small_world_500 -4.82`,
`parallel_cycles_4x5 -4.49`, `regular_3_30 -3.86`,
`disconnected_label_cycle_collage -4.95`).

```
NativeInitCore(force_2d=True, stress_pivot=True)
|
+- build_gradient_core(
|     losses=[
|         RepulsionLoss(...),                 # force-directed bread and butter
|         EdgeAttractionLoss(...),
|         OverlapAvoidanceLoss(...),
|         PivotApproxStressLoss(),            # w_stress set auto > 0 for this bucket
|         # NO DagOrderingLoss
|         # NO CrossingLoss (too expensive, low impact on flat graphs)
|         # NO EdgeStraightnessLoss (no rank direction to straighten to)
|     ],
|     overlap_interval=..., stall=...
| )
|
+- OverlapProjection()
+- AspectRatioFit(target_aspect=1.0)   # undo the 0.25 narrow default
+- ClusterGridArrange()
```

Key differences from `native_layered_dag`:
1. `Force2DInitIfFlat` unconditional -- we're here *because* the graph is
   flat.
2. `PivotApproxStressLoss` defaults on (currently opt-in at w_stress=0).
3. DagOrderingLoss, CrossingLoss, EdgeStraightnessLoss dropped. These
   three are what currently collapse cyclic graphs into a narrow
   vertical strip. Sprint-17 already half-patched this by gating
   DagOrderingLoss at `resolve.py:422`; this finishes the job.
4. No dummy nodes, no median/transpose, no BK. They are not just
   useless here -- they actively distort the layout.
5. Aspect target defaults to 1.0 instead of 0.25. Force-directed layouts
   are roughly square; narrowing them forces repulsion and attraction
   into an unnatural rectangle.

Projected impact: this is where the 4 listed regressions (-4 to -8 each)
live. A correct force-directed path should close those, give or take --
igraph_kamada_kawai scores 30.6 below dagua on average but wins
`small_world_100` by 8.5. The gap is topological, not implementation
quality.

### 3.4 `native_hybrid`

Graphs that are mostly DAGs but have a handful of cycles (partial
recurrent nets, transformer-with-residual-loops,
`parallel_cycles_4x5`). Small but real: 3-6 protected graphs sit
near the DAG/non-DAG boundary.

```
NativeInitCore(force_2d=False, stress_pivot=True)
|
+- MakeAcyclic (reverse back-edges)   # already in engine; use classifier's
|                                      # DAG skeleton, not separate pass
+- InsertDummyNodes() (on DAG skeleton)
+- ActivateExpandedGraphState()
|
+- build_gradient_core(
|     losses=[DagOrdering, EdgeAttraction, Repulsion,
|             Overlap, BackEdgeCompactness, PivotStress]
|     # Include BackEdgeCompactness to make the reversed edges draw well
| )
|
+- BarycenterReorder() + MedianSweep + TransposeHeuristic
|
+- # Skip BK -- back-edges violate its strict-DAG assumption,
+- # Barycenter is more robust here.
|
+- OverlapProjection()
+- StripDummyNodes()
+- AspectRatioFit(topology-aware)
```

This is where the existing `back_edge_compactness_loss` (in
`dagua/layout/constraints.py`) finally earns its keep. Today it's
a loss term that fires iff `w_back_edge>0`, which nobody sets because
`HYBRID` isn't a routable bucket.

### 3.5 Shared tail summary

```
NativeSolveTail(
    include_strip_dummy=<bool>,
    include_barycenter=<bool>,
    include_median_transpose=<bool>,
    include_bk=<bool>,
    target_aspect=<float | None>,
    include_cluster_grid=<bool>,
)
```

Each sub-pipeline passes its own constant bools. They aren't branches on
runtime state any more -- they're identity-of-sub-pipeline constants.
That's the key refactor: the conditional logic that was inside
`build_dagua_pipeline` becomes compile-time-constant when you know which
sub-pipeline you're building.

## 4. Config Flag Evolution

### 4.1 Collapse, don't delete

The four flags I'd collapse:

```
decompose_components           -> folded into engine pre-dispatch step; always on
insert_dummy_nodes             -> folded into native_layered_dag (always on there)
brandes_koepf_refine           -> folded into native_layered_dag (always on there)
use_native_median_transpose    -> folded into native_layered_dag + native_hybrid
native_median_passes           -> MOVES to native_overrides.median_passes
native_transpose_passes        -> MOVES to native_overrides.transpose_passes
use_tree_fast_path             -> folded into dispatcher (tree family always takes fast path)
```

The new user-facing surface:

```python
@dataclass
class NativeOverrides:
    """Advanced overrides; users rarely touch these."""
    force_pipeline: Literal["auto","tree","layered_dag","force_directed","hybrid"] = "auto"
    median_passes: int = 4
    transpose_passes: int = 8
    w_stress_auto: bool = True     # let dispatcher raise w_stress for force_directed
    skip_component_decompose: bool = False

class LayoutConfig:
    ...
    native: NativeOverrides | None = None   # None -> all defaults
```

The `force_pipeline` knob lets anyone benchmark a specific sub-pipeline
or force a choice when classifier confidence is low. Every other
sprint-19 knob disappears from the top-level config.

### 4.2 Back-compat

Keep the four deprecated flags on `LayoutConfig` for one release,
translate at `layout()` entry:

```python
if config.insert_dummy_nodes is False:
    warnings.warn("insert_dummy_nodes is deprecated; use "
                  "NativeOverrides(force_pipeline='force_directed') to "
                  "disable dummy-node insertion.", DeprecationWarning)
    # silent degrade: flag has no effect on auto dispatch
```

This is the cheapest deprecation path: the flags still parse, they just
become no-ops on the new dispatcher. Users who truly depended on them
are vanishingly rare (they're sprint-19 internal tuning).

## 5. Refactor Plan -- Concrete Diff Sketch

### 5.1 Files that die or shrink

| Path | Change | After |
|---|---|---|
| `dagua/layout/ops/pipelines/dagua_native.py` | Shrink to **adapter + dispatcher only** | ~300 lines |
| `dagua/layout/engine.py` (legacy body) | Unchanged this sprint (separate cleanup) | same |

### 5.2 New files (net add ~600 lines, minus 900 removed = -300 lines)

| Path | Responsibility | Approx LOC |
|---|---|---|
| `dagua/layout/ops/pipelines/native_tree.py` | Tree/forest fast-path (Reingold-Tilford + aspect fit) | ~80 |
| `dagua/layout/ops/pipelines/native_layered_dag.py` | All sprint-19 layered-DAG ops, unconditional | ~180 |
| `dagua/layout/ops/pipelines/native_force_directed.py` | New sub-pipeline with stress + force losses | ~160 |
| `dagua/layout/ops/pipelines/native_hybrid.py` | DAG skeleton + back-edge compactness | ~170 |
| `dagua/layout/ops/pipelines/_native_shared.py` | `NativeInitCore`, `NativeSolveTail`, `GradientCore` (moved) | ~220 |
| `dagua/layout/ops/pipelines/_native_dispatch.py` | `dispatch_native(structure, config) -> Pipeline` | ~90 |
| `dagua/layout/ops/component_decompose.py` | Move the 400 lines of component tiling out of pipelines/dagua_native.py | ~400 |
| `dagua/layout/graph_classify.py` | Add LAYERED_DAG / FORCE_DIRECTED / HYBRID families + 3 new fields | +60 |

### 5.3 What `dagua_native.py` looks like after

```python
"""dagua_native adapter: decompose components, dispatch per component."""

from dagua.layout.ops.component_decompose import decompose_and_run
from dagua.layout.ops.pipelines._native_dispatch import dispatch_native

def build_dagua_pipeline(config):
    return dispatch_native(config)   # returns whichever sub-pipeline matches

def layout_dagua_native_pipeline(edge_index, num_nodes, node_sizes, ..., config, ...):
    # 20 lines of torch setup -> LayoutProblem build ->
    # decompose_and_run(problem, state, ctx, dispatch_native) ->
    # return pos
    ...
```

That's it for the top file. Every "is this DAG? is this dense? is this
flat?" branch moves to the classifier, and every "run this op if..."
branch moves to the sub-pipeline identity.

### 5.4 Dispatcher body (the single source of truth)

```python
def dispatch_native(config: LayoutConfig) -> Pipeline:
    structure = get_structure(config)
    overrides = config.native or NativeOverrides()

    if overrides.force_pipeline != "auto":
        return _SUB_PIPELINES[overrides.force_pipeline](config)

    family = structure.family
    if family in (GraphFamily.TREE, GraphFamily.CHAIN, GraphFamily.FOREST):
        return build_native_tree(config)

    if family == GraphFamily.LAYERED_DAG:
        return build_native_layered_dag(config)

    if family == GraphFamily.FORCE_DIRECTED:
        return build_native_force_directed(config)

    if family == GraphFamily.HYBRID:
        return build_native_hybrid(config)

    # GENERAL only reached for degenerate / empty graphs.
    # Fall back to force_directed which is the most robust.
    return build_native_force_directed(config)
```

Single 20-line function. Replacing 240 lines of inline conditional
logic. This is the shape of the cleanup.

## 6. Migration -- Phased or One PR?

**One PR, behind a config flag.** Reasoning:

1. The change spans classifier, resolver, pipeline body, and config
   surface. Phasing creates temporary states where half the dispatch is
   new and half is old -- worst of both worlds.
2. Reversibility: keep the old `build_dagua_pipeline` body under
   `force_pipeline="legacy_monolith"` for one sprint. If benchmarks
   regress unexpectedly, flip the default back to `"legacy_monolith"`
   without reverting the structural work.
3. Scope: Codex can do the mechanical split in one dispatched pass
   given a line-accurate spec. The hard thinking (which ops belong to
   which bucket) is what this report is for.

Recommended sequence **within** the one PR:

```
Step A: Add LAYERED_DAG / FORCE_DIRECTED / HYBRID to GraphFamily.
        Add num_layers_effective, cyclicity_ratio, has_dominant_component
        to GraphStructure. Write unit tests for every protected graph
        confirming family assignment.  [~150 lines + tests]

Step B: Extract NativeInitCore, NativeSolveTail, component_decompose
        from dagua_native.py into shared modules. No behavior change,
        just moves. Existing tests pass.  [~500 LOC refactor]

Step C: Write the four sub-pipelines. Each one is a ~150-line file
        that composes shared builders. Wire them through
        dispatch_native.                                [~590 LOC add]

Step D: Update LayoutConfig. Keep deprecated flags as no-op shims
        with DeprecationWarning. Add NativeOverrides.  [~40 LOC]

Step E: Default remains "auto"; add force_pipeline="legacy_monolith"
        escape hatch that runs the old body unchanged for one sprint.

Step F: Regression run -- full 93-graph h2h on both paths
        (auto + legacy_monolith). Protected wins must match within
        0.2 composite; targets should close their gaps.
```

Steps A and B are dispatchable to Codex verbatim. Steps C and D need
the human-in-the-loop review because they're where the loss-set and
aspect-ratio choices land. Step F is pure measurement.

## 7. Risk and Regression Analysis

### 7.1 Protected wins at risk

From CONTEXT.md:

| Graph | dagua | Best competitor | Bucket | Risk |
|---|---|---|---|---|
| org_chart_deep 91.64 | elk 68.98, +22.67 | LAYERED_DAG | **HIGH** -- needs exact sprint-19 stack |
| random_dag_200 65.21 | dagre 44.33, +20.88 | LAYERED_DAG | **HIGH** |
| hub_fanout_label_skew 92.67 | dot 76.43, +16.24 | LAYERED_DAG | **HIGH** |
| org_chart_1_5_4_8 95.89 | dot 80.26, +15.64 | LAYERED_DAG | MEDIUM |
| random_dag_50 61.30 | dagre 45.80, +15.50 | LAYERED_DAG | MEDIUM |
| random_bipartite_60 80.39 | elk 65.97, +14.42 | LAYERED_DAG (bipartite_dag tag) | MEDIUM |
| edge_label_braid 91.96 | dagre 79.35, +12.61 | LAYERED_DAG | MEDIUM |
| bipartite_4_3_4 80.68 | dot 68.07, +12.61 | LAYERED_DAG | MEDIUM |
| weighted_karate_34 71.68 | dot 59.37, +12.31 | **HYBRID** (has cycles) | **HIGH** -- currently benefits from sprint-19f,g running |
| real_karate_34 71.68 | dot 59.37, +12.31 | **HYBRID** | **HIGH** -- same |

Nine of ten protected wins are LAYERED_DAG; that entire stack must
transfer over unchanged. Two karate wins are the interesting case --
classify_graph today assigns them to GENERAL because they have cycles,
so they currently get median+transpose **only because** the sprint-19
gate `use_native_median_transpose and is_acyclic` keeps median off... but
then `BarycenterReorder` still fires for them. Getting the HYBRID bucket
right is load-bearing for those.

### 7.2 Tests I'd want before shipping

1. **Family-assignment test** (unit-level). Parametrize on every graph
   in the 93-graph protected suite. Assert the family choice. Golden
   file: ship one, fail loudly on any silent drift.
2. **Sub-pipeline smoke test** (unit). For each sub-pipeline, build it
   with a minimal LayoutConfig and run it on a 20-node synthetic
   graph from the matching family. Assert convergence + no NaN +
   positions shaped `[N, 2]`.
3. **Protected-graph composite test** (integration). For each of the
   top 15 protected wins, run auto path, assert composite is within
   0.5 of the value recorded in CONTEXT.md. `tests/layout/
   test_protected_wins.py` style.
4. **Target-graph composite test** (integration). Same for the 10
   sprint-20 targets; initial run can just record the baseline, then
   each subsequent implementation step must not regress **any** target
   below its baseline.
5. **Regression on `trace_fallback` and `relax_fallback`** in
   `engine.py:913-935`. Neither should be affected, but the refactor
   touches config and could accidentally change the default `algorithm`
   string. One-line smoke test each.

### 7.3 What the refactor buys even if every number stays identical

- **Readability**: a new contributor can look at
  `native_force_directed.py` and understand all of dagua's flat-graph
  strategy in 160 lines. Today that story is fragmented across four
  sprint-19 commits and six private gating functions.
- **Orthogonality**: adding a new technique (GraphSAGE init, constrained
  stress, etc.) means adding a new sub-pipeline, not threading another
  predicate into `build_dagua_pipeline`. The next 5 sprints' worth of
  ideas doesn't grow the default pipeline's complexity.
- **Testability**: every sub-pipeline is independently testable with
  synthetic inputs in its bucket. Today the test coverage for
  sprint-19g's BK x-refine is, effectively, "did the benchmark
  regress?" -- a slow, high-variance signal.
- **Concurrency with the directed/undirected split**: JMT's sprint-20
  directive explicitly calls out different handling for directed vs
  undirected. `native_force_directed` is the undirected path;
  `native_layered_dag` and `native_hybrid` are the directed ones. The
  dispatcher is the hook.

### 7.4 What I'd NOT do

- **Don't rewrite `GradientCore`.** It's already clean
  (`dagua_native.py:210-277`) and it's the shared heart of three of the
  four sub-pipelines. Touching it invites accidental regressions.
- **Don't move ops between modules.** The 268-op registry is the other
  thing that actually works. Ops don't need to know which sub-pipeline
  composes them.
- **Don't re-think `resolve.py`.** It's the right layer for
  config-time resolution; after the refactor it just has fewer
  conditional clauses (acyclic/family gates move to dispatcher).
- **Don't touch `classify_graph`'s O(V+E) discipline.** The new fields
  are all read from tensors we already traverse once.

## 8. Notes for codex agent E running in parallel

If we diverge, the forks to be careful about:

1. **Where `component_decompose` lives.** I prefer a dedicated module
   sibling to `pipelines/`. Codex might fold it into a control op. Both
   work. The important constraint is: it wraps *the dispatcher*, not a
   specific sub-pipeline. A graph with 5 components of different
   families (rare but possible) should classify each component
   separately.

2. **Whether `native_hybrid` exists as a separate sub-pipeline or is a
   config variant of `native_layered_dag`.** I lean separate because
   the loss set differs materially (BackEdgeCompactnessLoss, no BK
   refine). Codex might argue merge-with-flags; both tenable.

3. **Whether `force_pipeline` in NativeOverrides is the right name.**
   Could be `sub_pipeline` or `variant`. Low stakes.

4. **Whether `legacy_monolith` hides behind a private flag or is
   public.** Private (underscore prefix), documented internally only.
   We want users using `auto`.

---

## 9. Bottom Line

The sprint-19 work was correct. Five improvements, five wins. The
problem isn't the improvements -- it's that they were each grafted onto
the same monolithic pipeline body with its own predicate, when the real
structure is "the pipeline is a four-way disjoint union of sub-pipelines
keyed on topology family." Separating those four paths takes a
1,336-line file down to ~300, fixes the conceptual muddle, and opens
the door cleanly for the sprint-20 ambitious additions (force-directed
path, directed/undirected split, constrained stress, etc.) without
another round of "is_acyclic-gated" predicates.

One PR. One default flip (to "auto"). One escape hatch
(`legacy_monolith`) for one sprint. Four named sub-pipelines, each the
right size to reason about in one screenful.

End of report.

# Sprint 20 Research E -- Topology-Dispatch Architecture

Agent: Opus 4.7 (architect role). Scope: architectural cleanup of the default
dagua pipeline after sprint-19's patch wave, with a proposed
topology-dispatch refactor. Read-only; markdown-only deliverable.

## TL;DR

- **The "one default pipeline" story is already a lie.** `build_dagua_pipeline`
  in `dagua/layout/ops/pipelines/dagua_native.py:939-1177` reads five
  runtime-resolved booleans (`use_vcycle`, `resolved_use_dummy_nodes`,
  `enable_native_median_transpose and is_acyclic`, `enable_brandes_koepf_refine`,
  plus a cluster-grid predicate inside `ClusterGridArrange`) and branches the
  op list accordingly. On top of that the `layout_dagua_native_pipeline`
  adapter wraps the builder in an explicit per-weak-component decomposition
  (`dagua_native.py:1272-1331`) and an up-front tree fast-path
  (`dagua_native.py:492-499`). That is at minimum 7 orthogonal switches
  folded into a single call site. It is a hidden dispatcher in denial.
- **Frankenstein risk is real and measurable.** Six distinct public config
  flags were introduced during sprint-19 alone
  (`config.py:134,138-140,144,148,152`) to gate the new phases, every one of
  them paired with an internal `_dagua_native_*` mirror set by
  `prepare_pipeline_config` (`resolve.py:296-404`). Flag interaction is
  undocumented but non-trivial: dummy-node insertion depends on the classifier
  family being acyclic and having long edges (`dagua_native.py:151-189`); BK
  refine is also DAG-only; median/transpose is also DAG-only; component
  decomposition silently suppresses dummies, BK, and median/transpose on the
  parent before recursing. Multiple flags gate the same code paths.
- **The correct architecture is an explicit topology dispatcher** at
  `engine.layout()` that classifies once, then routes to one of ~5
  topology-specialised sub-pipelines (layered_dag, tree, cyclic_with_skeleton,
  connected_flat, disconnected). Each sub-pipeline is a pure composition of
  already-registered ops. The global config flags collapse into per-branch
  defaults. No flag deletions are needed for parity -- the flags become
  internal branch selectors hidden behind a clean public API.
- **We should migrate incrementally, not big-bang.** Step 1 is the smallest
  change that pays: extract the existing branching inside
  `build_dagua_pipeline` into a named sub-pipeline table keyed by
  (is_directed_acyclic, is_connected, has_long_edges, family). Step 2 moves
  component decomposition out of the adapter and into an `OuterDispatch` op.
  Step 3 promotes "force_directed_flat" as a real sibling sub-pipeline rather
  than a config-knob-neutered DAG pipeline, unlocking the small_world /
  dense-random losses identified by the context doc. Each step ships with
  a before/after composite so we can revert surgically.
- **Three pipelines and one wrapper can be deleted after the migration:**
  the standalone `sugiyama.py` pipeline (subsumed by the layered_dag branch,
  reuses the same BK and transpose ops we already export), the `_legacy`
  engine body in `engine.py:1419-2800` that the sprint-0 docstring already
  scheduled for archive, and the `Force2DInitIfFlat` op
  (`dagua_native.py:1109`) which becomes dead code once cyclic graphs route
  to their own sub-pipeline. That is ~3500 LOC of subsumed surface area.
- **Risk is bounded.** Today's 7-way switch runs inside one pipeline builder
  with no isolation between branches. An explicit dispatcher actually
  *reduces* risk because each branch is independently testable, swappable,
  and benchmarkable. The regression risk during migration comes from two
  places: (a) the per-component wrapper re-classifies each subproblem at
  `dagua_native.py:749-757`, so we must preserve that behaviour when the
  dispatcher moves up a level; (b) the tree fast-path
  (`dagua_native.py:492-499`) currently ships default-off
  (`config.py:134`), so moving it into the dispatcher must keep
  `use_tree_fast_path` defaulted-off to avoid silent regressions on the
  composite where R-T loses on edge-length-CV.

---

## 1. Current-state audit

### 1.1 The default pipeline is a 7-way switch statement

`build_dagua_pipeline` (`dagua/layout/ops/pipelines/dagua_native.py:939-1177`)
reads the following decision variables before constructing the op list:

```
# From build_dagua_pipeline body, line numbers refer to dagua_native.py
resolved_use_dummy_nodes   L961  -> InsertDummyNodes + ActivateExpandedGraphState
enable_native_median_transpose L999 + is_acyclic L994  -> MedianSweep + TransposeHeuristic
enable_brandes_koepf_refine L1002 -> BrandesKoepfHorizontalRefine (always appended
                                    but with enabled= flag internal to the op)
use_vcycle                 L1023 -> entirely different op list (HeavyEdgeMatching
                                    + VCycleRefine)
w_stress > 0               L199  -> pivot prep ops prepended
structure.family == TREE + use_tree_fast_path  dagua_native.py:492-499
                                 -> ReingoldTilfordTree bypass of the whole builder
problem.clusters populated -> ClusterGridArrange is always in the op list but
                              only fires when its own internal predicate matches
                              (cluster_arrange op body)
```

Plus the *outer* adapter `layout_dagua_native_pipeline`
(`dagua_native.py:1180-1333`) does:

```
_should_decompose_components  L562-613 -> weak-component tiling wrapper
```

That is eight runtime decision points. The "default pipeline" is a switch
statement whose cases are spread across
`build_dagua_pipeline`, `_run_native_problem`, and
`layout_dagua_native_pipeline`. The classifier output
(`graph_classify.classify_graph`, `graph_classify.py:401-522`) is re-consumed
in at least four places (`resolve.py:323,350`, `dagua_native.py:179-188,489`)
and sometimes re-computed (`dagua_native.py:749-757` for each child
component).

### 1.2 Flag matrix

All flags below are in `dagua/config.py`:

| Public flag                    | Default | Line | Gate in pipeline                                            |
|--------------------------------|---------|------|-------------------------------------------------------------|
| `use_tree_fast_path`           | False   | 134  | `dagua_native.py:494`                                       |
| `use_native_median_transpose`  | True    | 138  | `dagua_native.py:999`, also gated by `is_acyclic` L994     |
| `native_median_passes`         | 4       | 139  | `dagua_native.py:1009`                                      |
| `native_transpose_passes`      | 8       | 140  | `dagua_native.py:1010`                                      |
| `brandes_koepf_refine`         | True    | 144  | `dagua_native.py:1002`; BK op has its own structure gate   |
| `insert_dummy_nodes`           | True    | 148  | `dagua_native.py:175-188` (plus 6 other conditions)        |
| `decompose_components`         | True    | 152  | `dagua_native.py:584`                                       |
| `multilevel_threshold`         | 20000   | 155  | `resolve.py:395-403` -> `_dagua_native_use_vcycle`          |
| `adaptive_spacing`             | True    | 75   | `resolve.py:344`                                            |
| `w_stress`                     | 0.0     | 114  | `dagua_native.py:199`, `resolve.py:481`                     |

Interaction table (pairs where flag A's on-state is silently dominated or
subsumed by flag B's gate):

- `use_native_median_transpose` and `brandes_koepf_refine` and
  `insert_dummy_nodes` all additionally require `is_directed_acyclic` +
  connected + multi-layer. On a cyclic graph, flipping any of the three
  to False has zero effect; the gate in `_should_use_native_dummy_nodes`
  (`dagua_native.py:151-189`) already blocked them. This is three
  user-visible knobs that collapse into one DAG branch.
- `decompose_components` effectively neuters the single-component branches
  of all three DAG-only flags: the parent graph, if decomposable, runs each
  child through a fresh `_prepare_native_config` (`dagua_native.py:1304-1314`)
  which re-classifies per component. The parent top-level
  `build_dagua_pipeline` call is then run only on components that fail the
  decomposition gate, but the outer adapter still builds a parent-level
  pipeline that is never used when `_should_decompose_components` is true
  (`dagua_native.py:1282-1331` returns tiled positions directly, never
  invokes the parent `build_dagua_pipeline`).
- `use_tree_fast_path` is default False and gated on `family == TREE`, so it
  only applies to the 1 component slice of whatever graph slice made it to
  `_run_native_problem`. In practice the flag is dead because the default is
  off and we never hit the branch on the benchmark suite. But it is still
  wired and maintained.
- `multilevel_threshold`/`_dagua_native_use_vcycle` subsumes all other
  flags: the v-cycle path (`dagua_native.py:1024-1072`) builds a completely
  different pipeline with different ops, ignoring median/transpose/BK/dummy
  entirely. This is effectively a parallel pipeline masquerading as a
  conditional branch.
- `w_stress` adds three prep ops at L199 before the optimizer loop and a
  loss at `resolve.py:481-484`. Unrelated to topology -- a pure weight
  feature. Doesn't belong in a topology-dispatch discussion but illustrates
  how cleanly-isolated feature flags *should* look.

### 1.3 ASCII interaction diagram

```
layout(graph, config) [engine.py:883]
    |
    +-- algorithm == None ------> alias to "dagua_native"  [engine.py:941]
    +-- algorithm == "_legacy" -> _layout_inner()          [engine.py:937, 1419]
    +-- else -----------------> get_pipeline_function(name) [engine.py:950]
                                         |
                                         v
                   layout_dagua_native_pipeline()   [dagua_native.py:1180]
                                         |
                                         +-- DetectComponents     [L1279]
                                         |
                                         +-- _should_decompose?   [L1282]
                                         |    yes -> for each component:
                                         |              _prepare_native_config
                                         |              _run_native_problem
                                         |              tile + AspectRatioFit
                                         |    no  -> _run_native_problem
                                         |
                                         v
                   _run_native_problem                 [dagua_native.py:450]
                                         |
                                         +-- family == TREE &&
                                         |   use_tree_fast_path
                                         |     -> ReingoldTilfordTree  [L497]
                                         |
                                         +-- build_dagua_pipeline      [L501]
                                                       |
                                                       +-- use_vcycle? -> V-cycle branch [L1024-1072]
                                                       +-- else -> default branch
                                                                      (5 toggles pick ops)
```

The adapter layer silently owns two of the seven decisions. The pipeline
builder owns four. The classifier is re-read at every layer.

---

## 2. Proposed topology-dispatch architecture

### 2.1 Principle

**Classify once at `engine.layout()`. Dispatch to one of N topology-named
sub-pipelines. Each sub-pipeline is a pure composition of already-registered
ops. Flags become branch-internal defaults, not public switches.**

The classifier already lives at `dagua/layout/graph_classify.py:401-522` and
runs in O(V+E). It already returns everything we need: `family`,
`num_components`, `num_layers`, `is_acyclic`, `is_directed_acyclic`,
`topology_tags` (including `lattice_like`, `planar_dag`, `wide_layered`,
`dense_dag`). Sprint-19e extended `topology_tags` specifically for this
purpose; we just never wired it as a dispatcher.

### 2.2 Dispatch tree

```
engine.layout(graph, config)
    |
    +-- classify once: structure = classify_graph(edge_index, num_nodes)
    |
    v
dispatch_topology(structure) -> sub_pipeline_name
    |
    +-- num_components > 1 AND not(pins/clusters cross components)
    |       -> "disconnected"  (recurse into sub-dispatch per component,
    |                           then tile)
    |
    +-- family == TREE         -> "tree"      (Reingold-Tilford)
    +-- family == CHAIN        -> "tree"      (degenerate R-T)
    |
    +-- is_directed_acyclic AND num_layers >= 2
    |       -> "layered_dag"   (Sugiyama-family: dummy + BK + median/transpose)
    |
    +-- not is_acyclic AND num_layers >= 3
    |       -> "cyclic_skeleton" (cycle-reversal pre-pass + layered_dag inner,
    |                              the sprint-19a path done explicitly)
    |
    +-- not is_acyclic OR num_layers <= 1
    |       -> "force_flat"    (stress-majorization / KK / FR family for
    |                            small_world, social-net, dense random;
    |                            currently a dagua weakness)
    |
    +-- num_nodes >= multilevel_threshold
    |       -> "multilevel"    (v-cycle; orthogonal to the above, wraps any
    |                            inner sub-pipeline)
```

Six branches, all on classifier output. One is a recursion (disconnected),
one is a wrapper (multilevel). The inner four are flat and mutually
exclusive.

### 2.3 Per-branch composition (all reuse existing ops)

Each sub-pipeline below is notation, not code. File references are to ops
already present in the repo.

**Branch: `tree`** -- subsumes current tree fast-path and the sugiyama-lite
tree path. Subsumes `use_tree_fast_path` flag (which disappears).

```
Pipeline(name="tree"):
    ReingoldTilfordTree  [ops/coordinate.py, already in use at dagua_native.py:481-499]
    AspectRatioFit
    OverlapProjection(iterations=5)
```

Rationale: R-T is the Graphviz-dot tree path; zero crossings by
construction. Composite trade-off (edge-length CV) is offset by winning on
`dag_consistency` + `overlap_count` + `depth_spearman`. We should measure
before shipping it as default; current default is off
(`config.py:134`).

**Branch: `layered_dag`** -- the current default but extracted into its own
named pipeline. Identical to today's `build_dagua_pipeline` minus the
conditional gates (gates are now part of the dispatcher, not the
pipeline).

```
Pipeline(name="layered_dag"):
    FixedSteps
    NativeEngineInit                               [ops/init.py]
    InsertDummyNodes + ActivateExpandedGraphState  [ops/layering.py]
    # Sprint 15 pivot prep (optional)
    InitAnnealingSchedule
    CreateOptimizer
    build_gradient_core(losses, ...)
    BarycenterReorder
    MedianSweep                                    [ops/ordering.py]
    TransposeHeuristic                             [ops/ordering.py]
    BrandesKoepfHorizontalRefine                   [ops/coordinate.py]
    OverlapProjection
    StripDummyNodes
    AspectRatioFit(target_aspect=resolve_topology_aware_aspect)
    ClusterGridArrange
```

Flags collapse: `use_native_median_transpose`, `brandes_koepf_refine`,
`insert_dummy_nodes` all default-true inside this branch, exposed only as
debug switches. Delete their public surface.

**Branch: `cyclic_skeleton`** -- for mostly-acyclic graphs with a small back
arc set. The sprint-19a cycle reversal pre-pass becomes an explicit op in
this branch.

```
Pipeline(name="cyclic_skeleton"):
    ReverseBackEdges      [cycle.py, sprint-19a function]
    Pipeline(name="layered_dag")(...)   # inner
    RestoreBackEdgeDirection  # for metrics/rendering
```

**Branch: `force_flat`** -- NEW. For cyclic + flat graphs (small_world,
social_net, recurrent_feedback_cell). Currently routed through layered_dag
with `Force2DInitIfFlat` trying to rescue a collapsed 1D init -- one of the
messier patches in sprint-19 and contextualized by area A findings
(`area_A_algorithm_core__claude.md:275-295`).

```
Pipeline(name="force_flat"):
    CircularInit       # or SpectralInit for larger graphs
    InitAnnealingSchedule (DAG weights zeroed)
    CreateOptimizer
    build_gradient_core(losses_without_w_dag, ...)
    OverlapProjection
    AspectRatioFit(target_aspect=1.0)
```

Register the proposed `CircularInit` + `SpectralInit` as new ops. These are
trivial (a few dozen lines each) and remove the need for
`Force2DInitIfFlat` (`ops/force_2d_init.py`), which can then be archived.
Subsumes area A finding D2 and CONTEXT.md's "small_world / planar_60"
losses.

**Branch: `disconnected`** -- recurse.

```
Wrapper(name="disconnected"):
    DetectComponents
    for each component:
        child_structure = classify_graph(component.edge_index, component.num_nodes)
        child_sub_pipeline = dispatch_topology(child_structure)  # RECURSE
        child_pos = child_sub_pipeline.apply(...)
    tile_components(...)  # existing _tile_component_positions
    AspectRatioFit
```

Same behaviour as today's per-component wrapper
(`dagua_native.py:1282-1331`), but moved up one level and dispatching into
whichever sub-pipeline fits the component. Currently every component gets
the same default pipeline -- a tiny cycle next to a layered DAG both run the
full gradient loop. With per-component dispatch, small cycles route to
force_flat, layered components route to layered_dag, trees route to
tree.

**Branch: `multilevel`** -- wrapper. Kept intact as today's v-cycle path,
but lifted out of `build_dagua_pipeline` into the dispatcher so it can wrap
any inner sub-pipeline. The current implementation hard-codes the default
inner pipeline (`dagua_native.py:1024-1072`).

### 2.4 Code-shape of the dispatcher

Pseudocode, at `engine.py:layout()`:

```
def layout(graph, config=None, trace=None):
    ... (existing setup, skipped)

    structure = classify_graph(graph.edge_index, graph.num_nodes)

    if graph.num_nodes >= config.multilevel_threshold:
        sub_pipeline = "multilevel"
    elif structure.num_components > 1 and not graph.has_cross_component_flex:
        sub_pipeline = "disconnected"
    elif structure.family in (TREE, CHAIN) and config.use_tree_fast_path:
        sub_pipeline = "tree"
    elif structure.is_directed_acyclic and structure.num_layers >= 2:
        sub_pipeline = "layered_dag"
    elif not structure.is_acyclic and structure.num_layers >= 3:
        sub_pipeline = "cyclic_skeleton"
    else:
        sub_pipeline = "force_flat"

    pipeline = build_sub_pipeline(sub_pipeline, config, structure)
    return pipeline.apply(problem, state, ctx)
```

That is 15 lines. It replaces four layers of gating spread across three
files (~1000 LOC of conditional logic in `dagua_native.py`,
`resolve.py`, and `engine.py`).

### 2.5 How existing flags collapse

| Today's flag                     | Post-migration fate                                                            |
|----------------------------------|--------------------------------------------------------------------------------|
| `use_tree_fast_path`             | Internal to `tree` branch selection. Public API unchanged (still can disable). |
| `use_native_median_transpose`    | Default True inside `layered_dag` branch. Public API: still a kill switch.     |
| `native_median_passes`           | Tunable inside `layered_dag` branch. Unchanged API.                            |
| `native_transpose_passes`        | Tunable inside `layered_dag` branch. Unchanged API.                            |
| `brandes_koepf_refine`           | Default True inside `layered_dag`. Kill-switch public API.                     |
| `insert_dummy_nodes`             | Default True inside `layered_dag`. Kill-switch public API.                     |
| `decompose_components`           | Eliminated. The `disconnected` branch is always selected when applicable.      |
| `multilevel_threshold`           | Unchanged; triggers `multilevel` branch.                                       |
| `w_stress`                       | Unchanged; orthogonal feature.                                                 |
| `adaptive_spacing`               | Unchanged; orthogonal.                                                         |

Net change: public flag surface is unchanged for parity, but all the
flag-flag *interactions* are gone because they are now topology-branch
selections made up-front.

---

## 3. Migration plan

Three steps, each independently shippable with a before/after composite
benchmark. Roughly ordered by risk-reward ratio.

### Step 1 -- "Extract sub-pipelines within the builder" (lowest risk)

**What:** In `build_dagua_pipeline` (`dagua_native.py:939-1177`), replace
the inline `if` branches with a named-sub-pipeline lookup. The outer
function becomes a 20-line dispatcher on the already-classified structure;
each branch is a new function like `_build_layered_dag_pipeline(config)`,
`_build_force_flat_pipeline(config)`, `_build_tree_pipeline(config)`.

**Why first:** Zero behaviour change if we map every combination of today's
flags to the same resulting op list. This is pure refactoring; we prove it
by running the 93-graph benchmark and confirming <0.1 composite drift.

**Concretely:**
1. Add `dagua/layout/ops/pipelines/_subpipes.py` (name TBD). Expose
   `build_layered_dag_pipeline`, `build_force_flat_pipeline`,
   `build_tree_pipeline` as named factories.
2. Rewrite `build_dagua_pipeline` as a dispatcher that reads
   `structure.family`, `is_directed_acyclic`, `num_layers`, and picks.
3. Keep all public config flags honoured during dispatch. A user who sets
   `insert_dummy_nodes=False` still disables dummy insertion inside
   `build_layered_dag_pipeline`.
4. Delete the `Force2DInitIfFlat` special-case (`dagua_native.py:1109`).
   Cyclic + flat graphs now route to `build_force_flat_pipeline` which
   uses `CircularInit` directly.

**Effort:** 6-10 hours. **Risk:** very low (refactor only). **Value:** the
codebase finally reflects its actual logic. Later steps become easier.

### Step 2 -- "Lift the outer dispatcher to `engine.layout()`" (low risk)

**What:** Move the component-decomposition loop from the adapter
(`dagua_native.py:1282-1331`) into a new `engine.layout()` level
dispatcher. Also move the tree fast-path check out of `_run_native_problem`
(`dagua_native.py:492-499`) and into the dispatcher. The
`dagua_native` pipeline-name registry entry becomes a thin "build the
layered_dag sub-pipeline" alias for backwards compat.

**Why second:** This requires touching `engine.py:layout()`. We keep the
existing `algorithm="dagua_native"` path functional as an alias. Users
calling `dagua.layout(g)` get the new dispatcher; users calling
`dagua.layout(g, LayoutConfig(algorithm="dagua_native"))` get the same
thing.

**Effort:** 8-12 hours. **Risk:** low-medium (touches the entry point, but
behaviour is preserved by construction). **Value:** per-component
dispatch now routes mixed graphs correctly; a graph with one DAG component
and one cycle component runs layered_dag on the first and force_flat on
the second. The CONTEXT.md `disconnected_label_cycle_collage` -5 loss
disappears (each small cycle gets force_flat instead of being shoved into
layered_dag with `Force2DInitIfFlat`).

### Step 3 -- "Promote force_flat as a first-class branch" (medium risk, highest reward)

**What:** Build out `force_flat` with a real stress-majorization inner loop
(see area A finding D2 and CONTEXT.md big-bet notes). Today we just strip
the `w_dag` weight and rely on existing attraction/repulsion losses; this
is not competitive with KK or stress-majorization on genuinely
hierarchy-free graphs.

**Concretely:**
1. Add `CircularInit` op. ~30 LOC.
2. Add `StressMajorizationStep` op if we do not already have one
   reusable from `ops/stress.py`. (Pipelines already exist for stress-sgd
   and stress-majorization, but as standalone pipelines, not composable
   ops. Extract.)
3. `build_force_flat_pipeline` composes: `CircularInit` ->
   `stress_majorization_loop` -> `OverlapProjection` ->
   `AspectRatioFit(1.0)`.
4. Benchmark on the small_world / recurrent / planar_60 losses.

**Effort:** 15-25 hours. **Risk:** medium (we're changing behaviour on ~10
benchmark graphs). **Value:** the context doc identifies 3 of the top 10
open losses as "no hierarchy" graphs (`small_world_100` -8,
`small_world_500` -5, `planar_60` -9). This is the biggest outstanding
category. If force_flat works, expected composite +2 to +5 on those graphs
individually, 0 regression on layered graphs (they do not route here).

---

## 4. What to delete / archive

After migration, these become dead code or clear redundancy:

1. **`dagua/layout/ops/pipelines/sugiyama.py`** (`dagua_native.py` and ops
   already export the pieces). The standalone entry
   `algorithm="sugiyama"` is useful for debugging but should be a thin
   alias for `build_layered_dag_pipeline(force=True)`. ~188 LOC saved as a
   module deletion; the underlying ops stay. The `_` -prefixed functions
   in `ops/sugiyama.py` that are Sugiyama-specific wrappers around the
   shared BK / dummy-node / barycenter ops can also go; callers inside
   `ops/sugiyama.py:1451-2000` are a single pipeline builder away from
   dispatching directly into the shared ops.

2. **`dagua/layout/engine.py::_layout_inner`** (`engine.py:1419-2800`).
   Already scheduled for archive in the file's sprint-0 docstring
   (`engine.py:10-17`). 1380 LOC. The `algorithm="_legacy"` path also
   goes. Move to `dagua/layout/_archive/legacy_engine/`. The legacy path's
   only surviving feature (trace-enabled animation) needs op-level snapshot
   hooks landed first (planned for sprint 6+ per `engine.py:910`). Do
   this after animation is ported.

3. **`dagua/layout/ops/force_2d_init.py::Force2DInitIfFlat`** -- dead code
   once cyclic/flat routing moves to `force_flat`. ~50-80 LOC. The op is
   currently a symptom of not dispatching on cyclicity up front
   (dagua_native.py:1100-1109).

4. **Redundant internal flags in `LayoutConfig`** (`config.py:138-140,148`):
   `native_median_passes`, `native_transpose_passes`, `insert_dummy_nodes`,
   maybe `use_native_median_transpose`, `brandes_koepf_refine`. These can
   stay in the public API for tuning, but the internal mirror attrs
   (`_dagua_native_use_dummy_nodes`, etc. set in `resolve.py:424-446`)
   become unnecessary because the dispatcher already made the decision.
   ~40 LOC of config plumbing.

5. **`dagua/layout/resolve.py::prepare_pipeline_config`**
   (`resolve.py:296-404`). Today sets ~15 private `_dagua_native_*` attrs.
   After migration, most become sub-pipeline-local; the function collapses
   to just computing layered spacing + aspect + steps. ~100 LOC saved.

**Cumulative:** ~3500 LOC of code can be archived or shrunk. The
public API surface of `dagua.layout()` does not change.

---

## 5. Risk analysis

### 5.1 What we could break

- **Tree fast-path regression.** Today it is off by default
  (`config.py:134`). If the migration accidentally flips it on (e.g. by
  routing TREE family to `tree` branch without checking
  `use_tree_fast_path`), we regress on edge-length-CV. *Mitigation:*
  preserve the `use_tree_fast_path` check in the dispatcher exactly as
  `_run_native_problem` does today. Add a test that asserts a hand-crafted
  tree routed with `use_tree_fast_path=False` takes the layered_dag branch.

- **Component decomposition safety gates.** Today
  `_should_decompose_components` (`dagua_native.py:562-613`) has 7
  safety predicates: clusters, pins, cross-component flex, dominance
  fraction, singleton patterns. All must move to the
  dispatcher-level `disconnected` check. *Mitigation:* lift the function
  verbatim; do not simplify its gates in step 2.

- **Per-component re-classification cost.** Each child problem today
  re-runs `classify_graph` (`dagua_native.py:749-757`). The new
  dispatcher recurses through `dispatch_topology`, which does the same,
  so no change. Cost is bounded by the BFS O(V+E) classifier.

- **Force_flat branch quality.** Worst-case scenario: stress-majorization
  on small_world_100 still loses. *Mitigation:* ship step 3 as an
  opt-in branch first (a config flag `force_flat_algorithm="stress" |
  "kk" | "fr"`), benchmark, then promote to default once measured.

- **V-cycle interaction.** Today `_dagua_native_use_vcycle` is a
  `build_dagua_pipeline` branch; the whole builder emits a different op
  list. Under the new architecture, v-cycle becomes a wrapper that can
  wrap any inner sub-pipeline. This is more flexible but has to preserve
  today's default inner pipeline for N >= threshold (large graphs). That
  inner pipeline is currently a simplified layered_dag variant (no
  BK/median/transpose, `dagua_native.py:1025-1072`). Preserve that
  behaviour explicitly or large-graph quality regresses.

### 5.2 What we protect

The CONTEXT.md "per-graph wins to protect" list (CONTEXT.md:36-48) includes
`org_chart_deep`, `random_dag_200`, `hub_fanout_label_skew`. All are DAGs.
All will route to `layered_dag`. The layered_dag pipeline is today's
`build_dagua_pipeline` modulo the conditional gates, which we preserve as
the branch's default-on set. If step 1 is a pure refactor, the 93-graph
benchmark composite should drift <0.2 points. We validate that by running
`/tmp/h2h_wins.py` before and after.

### 5.3 What we gain

- Adding a new topology branch becomes mechanical (add a file, register in
  dispatcher). Compare today: adding a branch requires touching at least
  `dagua_native.py`, `config.py`, `resolve.py`, and whatever
  per-flag-mirror plumbing exists.
- Flag-flag interaction bugs (sprint-19b's "relax cycle-reversal gate for
  small graphs" retro) become structurally impossible because each
  topology branch reads its inputs once from the classifier.
- Per-component heterogeneous dispatch (CONTEXT.md
  `disconnected_label_cycle_collage` loss) happens "for free" in step 2.

### 5.4 Anti-Frankenstein acid test

After migration, a hypothetical "sprint-21 follow-up" that wants to add an
"orthogonal routing refinement for wide bipartite DAGs" should touch
exactly one file: add an op, then either add it to
`layered_dag`'s post-ordering section or create a new branch
(`wide_bipartite`). Today the same change would require a new config flag,
a new private mirror, a new gate inside `build_dagua_pipeline`, a new
classifier tag, and another pass over `resolve.py`. That lived experience
is the diagnostic for whether the refactor worked.

---

## 6. Ambition check

This proposal is deliberately conservative on scope. It does not add new
algorithms; it cleans up the arrangement of what already exists. The
CONTEXT.md mandate says "strike a balance -- avoid Frankenstein patchwork.
Be ambitious." The big-bet energy should go into **new algorithms for
force_flat** (stress majorization, constrained stress, modern
post-2022 techniques like GraphSAGE-init). The architectural refactor is
the *enabler* for that ambition -- we cannot honestly claim a new
force_flat branch is "different handling for directed vs undirected" if
there is no explicit directed-vs-undirected dispatch. The dispatcher is
the prerequisite. Ship it in parallel with or slightly ahead of
algorithmic work.

Agent E recommendation: land step 1 (pure refactor) as a standalone commit
gate before any sprint-20 algorithm work. Then land algorithm work in the
now-obvious sub-pipelines, which are by construction isolated and
benchmarkable. Steps 2 and 3 can run concurrently with algorithm work or
after. The net effect is that sprint-20 ships both the architectural
cleanup and the ambitious small_world / planar_60 wins, without making the
codebase worse.

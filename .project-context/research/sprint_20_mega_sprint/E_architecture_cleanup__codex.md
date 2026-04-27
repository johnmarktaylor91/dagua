# Sprint 20 Research E - Architecture cleanup and topology dispatch

## TL;DR

- Split `dagua_native` into a topology-dispatch shell plus four named native sub-pipelines: `native_layered_dag`, `native_flat_force`, `native_tree_forest`, and `native_component_pack`. Keep the differentiable `gradient_core` as shared infrastructure, but stop pretending every graph wants the same layered-DAG polish stack.
- Extend `classify_graph` from family tags used mostly for aspect policy into an explicit dispatch descriptor. The current classifier knows tree, forest, chain, bipartite, wide layered, DAG-ness, components, layer widths, degree, edge ratio, and a few aspect tags, but it does not express hierarchy strength, cycle rank, reciprocal/undirected signal, long-edge burden, component-size distribution, or planar-undirected/regular/small-world style flat topology.
- Move topology policy out of `build_dagua_pipeline()`. Today policy is scattered across `resolve.py`, `layout_dagua_native_pipeline()`, `_run_native_problem()`, and `build_dagua_pipeline()`: aspect ratio in `resolve.py`, component decomposition in the adapter, tree fast path in `_run_native_problem()`, dummy nodes and DAG polish in `build_dagua_pipeline()`. That is the Frankenstein seam JMT called out.
- Treat sprint-19 features as conditional building blocks, not defaults. Component decomposition is a wrapper. Dummy insertion, median/transpose, and Brandes-Koepf are DAG-only. Topology-aware aspect is a final stage for all layouts, but with different policies for layered, flat force-directed, and component-packed output.
- Keep public flags temporarily as overrides, but add a single `topology_dispatch="auto"` policy and migrate the flags to debug/kill-switch status. Existing flags should not keep driving default architecture after sprint 20.
- Use a phased migration, not one big PR. First add the descriptor and parity-preserving dispatcher, then extract sub-pipelines without behavior changes, then introduce flat force/stress fallback behind topology gates, then retire legacy flags and fallback pathways after benchmark parity.

## Current-state audit

The default native pipeline is too large for the kind of topology-specific work sprint 20 needs. In this checkout, `dagua/layout/ops/pipelines/dagua_native.py` is 1,336 lines. It is 5.5x the next largest pipeline file, `maxent_stress.py` at 241 lines, and much larger than the force-directed pipelines that are already available: `fa2.py` is 220 lines, `sfdp.py` is 183 lines, `fr.py` is 162 lines, and `sugiyama.py` is 187 lines. The registry lists 24 named pipeline algorithms in `dagua/layout/ops/pipelines/__init__.py:12`, but the default path still funnels all default behavior through one `dagua_native` registry entry at `dagua/layout/ops/pipelines/__init__.py:22`.

The complexity inside the native file is not just raw length. A quick AST audit of `dagua_native.py` shows 20 top-level functions. The two largest are `build_dagua_pipeline()` at 239 lines and `layout_dagua_native_pipeline()` at 154 lines. The current adapter and pipeline builder together decide config resolution, classification, component decomposition, tree fast path, dummy-node expansion, stress prep, optimizer creation, gradient loop, crossing reduction, BK x-refine, overlap projection, dummy stripping, aspect fit, and cluster grid arrangement. The source confirms this control stack:

- Config prep annotates private `_dagua_native_*` attrs and decides dummy-node eligibility at `dagua/layout/ops/pipelines/dagua_native.py:378` through `dagua/layout/ops/pipelines/dagua_native.py:447`.
- The tree fast path lives inside `_run_native_problem()` at `dagua/layout/ops/pipelines/dagua_native.py:492` through `dagua/layout/ops/pipelines/dagua_native.py:499`, not in the classifier or dispatcher.
- Component decomposition has separate gates at `dagua/layout/ops/pipelines/dagua_native.py:562` through `dagua/layout/ops/pipelines/dagua_native.py:613`, component extraction at `dagua/layout/ops/pipelines/dagua_native.py:687`, and component tiling at `dagua/layout/ops/pipelines/dagua_native.py:890`.
- `build_dagua_pipeline()` assembles the main op sequence from `dagua/layout/ops/pipelines/dagua_native.py:939` through `dagua/layout/ops/pipelines/dagua_native.py:1177`.
- The pipeline always builds crossing-reduction ops with barycenter, optional median/transpose, and BK refine at `dagua/layout/ops/pipelines/dagua_native.py:999` through `dagua/layout/ops/pipelines/dagua_native.py:1021`, then appends them after the gradient core at `dagua/layout/ops/pipelines/dagua_native.py:1143`.

This is difficult to reason about because the file is simultaneously an adapter, policy engine, and pipeline factory. The comment at the top says the "pipeline body here is pure composed ops" and has "no inline helpers" at `dagua/layout/ops/pipelines/dagua_native.py:1` through `dagua/layout/ops/pipelines/dagua_native.py:5`, but the file now contains many inline topology helpers. The drift is not a quality failure by any single sprint-19 patch; the problem is accumulation. Each patch needed a gate, and those gates landed near the code they affected instead of in a single topology-dispatch layer.

`engine.py` is also carrying transitional complexity. The public `layout()` entry point remaps `algorithm=None` to `"dagua_native"` at `dagua/layout/engine.py:936` through `dagua/layout/engine.py:944`, forwards graph state and config only for the remapped default at `dagua/layout/engine.py:966` through `dagua/layout/engine.py:980`, then invokes the registry pipeline at `dagua/layout/engine.py:984` through `dagua/layout/engine.py:994`. The old legacy path still exists below it, including multilevel direct layout and relaxation fallback. Trace and `relax_steps` still force the legacy path with warnings at `dagua/layout/engine.py:913` through `dagua/layout/engine.py:935`. `_layout_inner()` still delegates to `_layout_inner_pipeline()` only when `config.use_pipeline` is true at `dagua/layout/engine.py:1488`, while default `algorithm=None` avoids that path by becoming `algorithm="dagua_native"` earlier. That is two dispatch mechanisms in one file.

`resolve.py` is valuable but overloaded with topology policy. It resolves steps, spacing, aspect, overlap interval, projection iterations, stall behavior, optimizer type, vcycle threshold, and structure storage in one function at `dagua/layout/resolve.py:296` through `dagua/layout/resolve.py:404`. It already uses classification to override tree weights and chain step count at `dagua/layout/resolve.py:323` through `dagua/layout/resolve.py:336`, then applies topology-aware aspect at `dagua/layout/resolve.py:350`. The aspect policy itself is small and hard-coded: lattice-like maps to 0.05, planar DAG to 0.08, wide/bipartite to 0.85, dense DAG to 0.05, and default to 0.25 at `dagua/layout/resolve.py:129` through `dagua/layout/resolve.py:157`. That policy is useful, but it should be attached to a chosen layout plan, not be the only topology-specific plan choice.

`graph_classify.py` is the right foundation but too narrow for sprint 20 dispatch. The current family enum has `GENERAL`, `TREE`, `FOREST`, `CHAIN`, `BIPARTITE_DAG`, `WIDE_LAYERED`, and `GRID` at `dagua/layout/graph_classify.py:19` through `dagua/layout/graph_classify.py:29`. `GRID` is never assigned in the final family choice at `dagua/layout/graph_classify.py:485` through `dagua/layout/graph_classify.py:496`. The returned `GraphStructure` stores components, max degree, layer count, average and max layer width, layer width CV, planar hint, edge-to-node ratio, undirected acyclicity, directed acyclicity, and topology tags at `dagua/layout/graph_classify.py:31` through `dagua/layout/graph_classify.py:46`. Tags are only derived for directed acyclic graphs. `_derive_topology_tags()` returns no tags for cyclic graphs at `dagua/layout/graph_classify.py:365` through `dagua/layout/graph_classify.py:366`, and suppresses tags for tree/forest/chain at `dagua/layout/graph_classify.py:367` through `dagua/layout/graph_classify.py:368`. That means the exact bucket where sprint 20 wants force-directed fallback, such as small-world, regular, parallel cycles, and cyclic disconnected collage, is structurally under-described.

The existing op design document supports a better architecture. It defines categories for preprocessing, layering, ordering, coordinate assignment, force, loss, project, postprocess, edge routing, and control at `dagua/layout/ops/DESIGN.md:190` through `dagua/layout/ops/DESIGN.md:213`. It explicitly lists `ClassifyGraph`, `DetectComponents`, `InsertDummyNodes`, `BarycenterSweep`, `MedianSweep`, `TransposeHeuristic`, `BrandesKopf4Pass`, `OverlapProjection`, and force/stress operations as composable pieces at `dagua/layout/ops/DESIGN.md:227` through `dagua/layout/ops/DESIGN.md:352`. The design sketch already shows `ClassifyGraph()` and conditional pipeline control for native layout at `dagua/layout/ops/DESIGN.md:481` through `dagua/layout/ops/DESIGN.md:503`. The code should finally make that design real.

Benchmark context justifies acting now. The sprint-20 context says Dagua leads every competitor on average, with graphviz_dot closest at +4.11, but the remaining losses cluster in topologies that the current layered-native path handles poorly: `small_world_100` is -8.51, `small_world_500` is -4.82, `parallel_cycles_4x5` is -4.49, `regular_3_30` is -3.86, and `planar_60` is -9.25. The same context says sprint-19 introduced per-component decomposition, topology-aware aspect ratio, median/transpose, BK x-refine, and dummy nodes. Those are exactly the features now interleaved in one default path.

## Findings

Severity: high. The default native pipeline has become a policy monolith. Evidence: `dagua_native.py` is 1,336 lines, while every other registered pipeline is at most 241 lines; the main pipeline builder alone spans `dagua/layout/ops/pipelines/dagua_native.py:939` through `dagua/layout/ops/pipelines/dagua_native.py:1177`. Proposed change: split native into a dispatch shell and named sub-pipeline builders. This directly addresses the Frankenstein risk without discarding the working layered path.

Severity: high. Flat and cyclic graph topology is under-modeled. Evidence: `_derive_topology_tags()` exits immediately for non-DAGs at `dagua/layout/graph_classify.py:365` through `dagua/layout/graph_classify.py:366`, but sprint-20 losses include `small_world_100` at -8.51, `small_world_500` at -4.82, `parallel_cycles_4x5` at -4.49, and `regular_3_30` at -3.86. Proposed change: add explicit flat-force and planar-undirected dispatch categories, plus classifier fields for hierarchy strength, cycle rank, reciprocal signal, and degree distribution.

Severity: high. Sprint-19 ops are correct locally but globally over-broad. Evidence: median/transpose and BK are assembled in the default crossing-reduction list at `dagua/layout/ops/pipelines/dagua_native.py:999` through `dagua/layout/ops/pipelines/dagua_native.py:1021`, then appended to the main path at `dagua/layout/ops/pipelines/dagua_native.py:1143` through `dagua/layout/ops/pipelines/dagua_native.py:1149`; tests explicitly require cyclic skip for median/transpose at `tests/test_layout/test_native_median_transpose.py:250` through `tests/test_layout/test_native_median_transpose.py:273` and BK at `tests/test_layout/test_brandes_koepf_native.py:145` through `tests/test_layout/test_brandes_koepf_native.py:164`. Proposed change: make ordering polish and BK plan-selected DAG sub-pipeline stages.

Severity: medium. Component decomposition is a good feature in the wrong layer. Evidence: decomposition safety policy is embedded in the adapter at `dagua/layout/ops/pipelines/dagua_native.py:562` through `dagua/layout/ops/pipelines/dagua_native.py:613`, while actual weak-component detection exists as an op at `dagua/layout/ops/preprocess.py:1338`. Proposed change: move component packing into an outer wrapper that recursively dispatches child components.

Severity: medium. Config flags are now architecture controls rather than override controls. Evidence: the public flags live together in `LayoutConfig` at `dagua/config.py:135` through `dagua/config.py:152`, and each is read directly in native policy code. Proposed change: add `topology_dispatch="auto"` and use the old flags as scoped kill switches under the chosen plan.

Severity: medium. Existing classic pipelines are underused by the default architecture. Evidence: FA2, SFDP, and Stress-SGD already expose clean composable pipelines at `dagua/layout/ops/pipelines/fa2.py:88`, `dagua/layout/ops/pipelines/sfdp.py:65`, and `dagua/layout/ops/pipelines/stress_sgd.py:63`. Proposed change: use them as the first implementation substrate for `native_flat_force` rather than implementing new force code inside `dagua_native.py`.

## Proposed topology dispatch design

The engine should dispatch on a two-level descriptor: graph constraints first, topology second. "Directed vs undirected" alone is not enough because `DaguaGraph` is fundamentally directed (`dagua/graph.py:67` through `dagua/graph.py:76`), while many graph families are semantically undirected even when encoded as directed edge lists. The descriptor should answer: is there a reliable hierarchy to preserve, or is hierarchy a metric artifact that hurts the drawing?

Recommended dispatch categories:

1. `TRIVIAL`: zero or one node. Existing adapter already returns zeros at `dagua/layout/ops/pipelines/dagua_native.py:1211` through `dagua/layout/ops/pipelines/dagua_native.py:1214`.
2. `COMPONENT_PACK`: more than one weak component and safe to solve independently. This is not a layout algorithm; it is an outer wrapper. Existing gates exclude clusters, pins, dominant-component cases, and cross-component flex at `dagua/layout/ops/pipelines/dagua_native.py:584` through `dagua/layout/ops/pipelines/dagua_native.py:612`.
3. `TREE_CHAIN`: connected tree or chain. Current classifier already detects chain and tree at `dagua/layout/graph_classify.py:458` through `dagua/layout/graph_classify.py:468`.
4. `FOREST`: acyclic underlying graph with multiple components. This usually routes through `COMPONENT_PACK` first, then each component as `TREE_CHAIN` or small general.
5. `LAYERED_DAG`: directed acyclic graph with meaningful layer depth. This is the current native strength and should remain the default for org charts, dependencies, random DAGs, layered feature pyramids, and transformer-like graphs.
6. `LONG_EDGE_DAG`: a `LAYERED_DAG` with enough long edge span to justify dummy expansion. The current `_should_use_native_dummy_nodes()` requires directed acyclicity, one component, more than one layer, at least 20 nodes, not dense DAG, and at least one edge spanning two or more layers at `dagua/layout/ops/pipelines/dagua_native.py:151` through `dagua/layout/ops/pipelines/dagua_native.py:189`. Promote this into classifier metadata: `long_edge_count`, `long_edge_fraction`, `max_layer_span`.
7. `WIDE_SHALLOW_DAG`: bipartite and wide-layered graphs. Existing family detection covers two-layer DAGs and wide layered graphs at `dagua/layout/graph_classify.py:475` through `dagua/layout/graph_classify.py:480`, and aspect policy already treats this separately at `dagua/layout/resolve.py:153`.
8. `DENSE_NARROW_DAG`: high edge-to-node ratio, narrow layers, many layers. Existing `"dense_dag"` tag is assigned at `dagua/layout/graph_classify.py:392` through `dagua/layout/graph_classify.py:396`. It should stay DAG-native but skip dummy nodes and be careful with crossing losses.
9. `PLANAR_LAYERED_DAG` and `LATTICE_LAYERED_DAG`: existing `"planar_dag"` and `"lattice_like"` tags at `dagua/layout/graph_classify.py:378` through `dagua/layout/graph_classify.py:390`. These need DAG polish and topology-aware aspect, but also likely benefit from stress or planar-preserving x/y relaxation after layer assignment.
10. `RAGGED_PYRAMID_DAG`: new category for strongly varying layer widths, fan-in/fan-out pyramids, and feature pyramids. The classifier already has `layer_width_cv`; use high CV plus DAG depth and asymmetric width profile. This directly targets `ragged_feature_pyramid`, the largest sprint-20 loss at -10.04.
11. `CYCLIC_HIERARCHICAL`: directed graph with cycles but a strong condensation DAG or mostly forward edges after cycle reversal. The current native path skips DAG ordering only when `is_acyclic` is false in `build_loss_ops()` at `dagua/layout/resolve.py:414` through `dagua/layout/resolve.py:423`, but still starts from layered init plus `Force2DInitIfFlat()` at `dagua/layout/ops/pipelines/dagua_native.py:1100` through `dagua/layout/ops/pipelines/dagua_native.py:1109`. A better classifier should estimate feedback-edge ratio or SCC condensation depth. Low feedback with deep condensation stays near layered-native.
12. `FLAT_FORCE_SMALL`: cyclic or weakly hierarchical graphs under a small threshold, including small-world, regular, dense random, social, and parallel cycles. This category should not run median/transpose, BK, or dummy nodes.
13. `FLAT_FORCE_LARGE`: same semantics as `FLAT_FORCE_SMALL`, but chooses scalable force or stress approximations such as SFDP, FA2 Barnes-Hut, or pivot stress.
14. `PLANAR_UNDIRECTED`: cyclic but sparse, low-degree, near-planar graph with no DAG hierarchy. This targets `planar_60`, hex/mesh variants that are not real DAGs, and cycle lattices. It should use stress/SFDP or planar-aware force, not DAG compaction.
15. `CLUSTERED`: clusters present. This is an overlay modifier. It can combine with DAG or flat force, but cluster losses and `ClusterGridArrange` should be enabled only when clusters exist.
16. `FLEX_CONSTRAINED`: pins/align/flex present. This is another modifier. It should suppress unsafe component decomposition and ensure hard-pin projection is in every optimizing path.
17. `SCALE_HUGE`: above memory/classifier thresholds. Current `classify_graph()` uses a special path above 10,000,000 nodes to avoid degree allocation at `dagua/layout/graph_classify.py:425` through `dagua/layout/graph_classify.py:450`. The dispatch plan should preserve that guard and choose streaming/multilevel behavior before expensive classification.

The classifier changes should be additive. Keep `GraphFamily` for coarse compatibility, but add a `DispatchTopology` enum or `LayoutPlan` dataclass rather than overloading `GraphFamily`. Add cheap fields to `GraphStructure`: `avg_degree`, `degree_cv`, `cycle_rank`, `reciprocal_edge_fraction`, `long_edge_fraction`, `max_layer_span`, `component_size_cv`, `largest_component_fraction`, `hierarchy_score`, and `flatness_score`. For large graphs, compute only fields available without large allocations. For directed/undirected split, use `reciprocal_edge_fraction` and edge source metadata if it exists later; in current Dagua, semantic undirectedness must be inferred.

The dispatch diagram should look like this:

```text
engine.layout()
  |
  +-- normalize graph state, node sizes, flex ids
  |
  +-- NativeDispatch.resolve(problem, config)
        |
        +-- classify_graph_plus()
        +-- choose LayoutPlan(topology, modifiers, postprocess)
        |
        +-- COMPONENT_PACK? ------------------------------+
        |        |                                        |
        |        +-- split weak components                |
        |        +-- dispatch each child recursively      |
        |        +-- tile + final aspect                  |
        |                                                 |
        +-- TREE_CHAIN ------> native_tree_forest         |
        +-- LAYERED_DAG -----> native_layered_dag         |
        +-- CYCLIC_HIER -----> native_hybrid_condensation |
        +-- FLAT_FORCE ------> native_flat_force          |
        +-- SCALE_HUGE ------> native_scale_multilevel    |
        |
        +-- shared final project/aspect/direction/cache
```

My position versus the expected Opus architecture-purist answer: I agree with the purist goal of a clean policy layer, but I would not rewrite native into a fully declarative graph of tiny ops in one PR. The pragmatic path is to extract named sub-pipelines around existing, tested sequences first. Dagua already has strong default wins to protect, including `org_chart_deep` +22.67, `random_dag_200` +20.88, `hub_fanout_label_skew` +16.24, and `random_bipartite_60` +14.42 from the sprint context. A pure rewrite risks burning those wins before the flat-graph story exists. The architecture should be ambitious, but the migration should be incremental and benchmark-gated.

## Sub-pipeline sketches

`native_component_pack` should own component decomposition. It should wrap any child pipeline, not be embedded in `layout_dagua_native_pipeline()`. The existing pieces are good: `DetectComponents` exists as an op at `dagua/layout/ops/preprocess.py:1338`, the safety gate exists at `dagua/layout/ops/pipelines/dagua_native.py:562`, and tiling exists at `dagua/layout/ops/pipelines/dagua_native.py:890`. Move those into a component module and make the child call `NativeDispatch` recursively. This keeps disconnected cyclic collages from forcing all components through one global layered solve. Regression coverage already checks decomposition improves `disconnected_label_cycle_collage` at `tests/test_layout/test_component_decomposition.py:228` through `tests/test_layout/test_component_decomposition.py:240`.

`native_layered_dag` should be the current high-performing core for DAGs. Its sequence:

```text
NativeEngineInit
optional InsertDummyNodes + ActivateExpandedGraphState
optional stress pivot prep
InitAnnealingSchedule
CreateOptimizer
gradient_core(DAG-aware losses)
BarycenterReorder
optional MedianSweep + TransposeHeuristic
BrandesKopfHorizontalRefine
OverlapProjection
StripDummyNodes
AspectRatioFit
ClusterGridArrange when clustered
```

This is basically today's `build_dagua_pipeline()` at `dagua/layout/ops/pipelines/dagua_native.py:1074` through `dagua/layout/ops/pipelines/dagua_native.py:1177`, but moved into a named sub-pipeline and gated by topology. The `gradient_core` extraction at `dagua/layout/ops/pipelines/dagua_native.py:210` through `dagua/layout/ops/pipelines/dagua_native.py:277` should stay shared. The DAG polish stack should be assembled only for acyclic layered plans. Median/transpose has a regression test proving cyclic graphs should skip and produce identical positions at `tests/test_layout/test_native_median_transpose.py:250` through `tests/test_layout/test_native_median_transpose.py:273`. BK tests also assert cyclic skip at `tests/test_layout/test_brandes_koepf_native.py:145` through `tests/test_layout/test_brandes_koepf_native.py:164`.

`native_tree_forest` should choose between Reingold-Tilford and native gradient by metric objective. Current config has `use_tree_fast_path: bool = False` with comments saying R-T gives zero crossings but worse edge-length variance under current composite weights at `dagua/config.py:124` through `dagua/config.py:134`. That is a sign the tree path is a topology-specific choice, not a random fast path inside `_run_native_problem()`. For sprint 20, keep default as current native-gradient tree behavior unless a tree-specific benchmark shows RT wins on the protected metric. But move the decision into the plan: `TREE_CHAIN` can choose `reingold_tilford` when `optimize_for_crossings` or very large tree, and `native_layered_dag` with tree weights otherwise.

`native_flat_force` is the missing pipeline. It should not reuse `NativeEngineInit` unless there is a real hierarchy signal. For flat graphs, initializers should be spectral, FA2/FR random, pivot-MDS, or stress warm start. Existing alternatives are already registered: FA2 validates, initializes, prepares undirected state, and repeats force steps at `dagua/layout/ops/pipelines/fa2.py:88` through `dagua/layout/ops/pipelines/fa2.py:123`; SFDP builds a multilevel graph and refines levels at `dagua/layout/ops/pipelines/sfdp.py:65` through `dagua/layout/ops/pipelines/sfdp.py:82`; Stress-SGD builds undirected adjacency and chooses exact or approximate stress terms at `dagua/layout/ops/pipelines/stress_sgd.py:63` through `dagua/layout/ops/pipelines/stress_sgd.py:84`. The first implementation does not need to invent new physics. Compose:

```text
FlatInit: SpectralInit or PivotMDSInit
FlatPreprocess: BuildAdjacency(directed=False) or FA2PrepareState
FlatCore: FA2ForceStep or StressSGD schedule
NativeConstraintRefine: optional short gradient_core without DagOrderingLoss, CrossingLoss off or low
OverlapProjection
AspectRatioFit(target from flat policy, usually near 1.0 or data-driven)
```

This path targets `small_world_100`, `small_world_500`, `regular_3_30`, `parallel_cycles_4x5`, and `planar_60`. It should drop `DagOrderingLoss`, `MedianSweep`, `TransposeHeuristic`, `BrandesKopfHorizontalRefine`, and dummy nodes. `build_loss_ops()` already skips `DagOrderingLoss` on cyclic graphs at `dagua/layout/resolve.py:414` through `dagua/layout/resolve.py:423`, but that is only a partial fix; the rest of the pipeline still acts like a layered pipeline that had a bad day.

`native_hybrid_condensation` is a later-stage pipeline for directed cyclic graphs with hierarchy. It should find SCCs, layout the condensation DAG with `native_layered_dag`, and layout nodes within SCCs with `native_flat_force`, then expand. This is the right answer for recurrent cells, feedback control graphs, and dependency graphs with small cycles. Do not build it before the flat force path exists. It depends on classifier fields such as SCC count, largest SCC fraction, condensation depth, and feedback edge fraction.

`native_scale_multilevel` should remain separate from this sprint's small benchmark work. The existing V-cycle path is present but intentionally disabled by raising the default threshold to 1,000,000 in `prepare_pipeline_config()` because previous V-cycle results were catastrophic on chains and random DAGs at `dagua/layout/resolve.py:384` through `dagua/layout/resolve.py:403`. Do not revive it as part of topology cleanup unless scale-specific tests are included.

## Big-bet proposals

Big bet 1: make topology dispatch the default native architecture. Projected impact is mostly risk reduction plus enabling future wins: the layered path remains protected, and the flat path can improve the current cyclic/undirected losses without adding more gates to `dagua_native.py`. The cost is a larger one-time test update, especially around default dispatch tests that currently assert `build_dagua_pipeline()` is called.

Big bet 2: add `native_flat_force` and let it compete for low-hierarchy graphs. Projected impact is highest on `small_world_100`, `small_world_500`, `parallel_cycles_4x5`, `regular_3_30`, and `planar_60`, where the current deficit ranges from -3.86 to -9.25. The tradeoff is runtime: FA2/SFDP/stress may cost more than the current native gradient on small graphs. The enabling rule should require a measured composite win above 2 points on multiple target graphs before default activation.

Big bet 3: build SCC condensation dispatch for cyclic directed graphs. Projected impact is cleaner handling for recurrent feedback, state-machine, and dependency-cycle graphs, where some hierarchy exists but not as a pure DAG. The cost is implementation complexity and test surface. This should follow the flat-force path, because SCC internals need a good flat layout.

Big bet 4: replace scattered boolean flags with a typed `LayoutPlan`. Projected impact is maintainability and debuggability: benchmark reports can say "selected native_flat_force because hierarchy_score=0.12" instead of requiring code archaeology. The cost is public API transition. Existing flags should remain until at least one release after plan selection is documented.

## Which sprint-19 ops belong where

Component decomposition belongs outside every child pipeline as a wrapper. It should run before topology dispatch when safe, then dispatch each component. It is useless inside a connected child solve and actively wrong when clusters, pins, or cross-component flex bind components together. The current gate already encodes that, including cluster and pin suppression at `dagua/layout/ops/pipelines/dagua_native.py:588` through `dagua/layout/ops/pipelines/dagua_native.py:591`.

Dummy-node expansion belongs only in `LONG_EDGE_DAG`, not all DAGs and never flat force. The current gate is conservative and should become a plan rule. The tests show it must skip cyclic graphs (`tests/test_layout/test_engine.py:239` through `tests/test_layout/test_engine.py:252`), improve hexagonal lattice composite (`tests/test_layout/test_engine.py:255` through `tests/test_layout/test_engine.py:266`), and keep random DAG near baseline (`tests/test_layout/test_engine.py:269` through `tests/test_layout/test_engine.py:279`). Preserve those tests, but relocate the decision.

Median sweep and transpose belong in layered DAG ordering polish only. They are not meaningful on flat cyclic layouts because there is no stable layer order. Current code already checks `is_acyclic` before adding them at `dagua/layout/ops/pipelines/dagua_native.py:1006` through `dagua/layout/ops/pipelines/dagua_native.py:1012`, and the cyclic preservation test enforces the skip. Keep them out of `native_flat_force` and SCC-internal layouts.

Brandes-Koepf x-refine belongs in layered DAG coordinate compaction only. The op's own config has an `enabled` flag at `dagua/layout/ops/pipelines/dagua_native.py:1014` through `dagua/layout/ops/pipelines/dagua_native.py:1019`, but the better gate is the plan. BK is useless after force-directed layout and likely harmful on undirected cycles. Keep it for `LAYERED_DAG`, `LONG_EDGE_DAG`, `WIDE_SHALLOW_DAG`, `DENSE_NARROW_DAG`, and maybe `CYCLIC_HIERARCHICAL` only at the condensation level.

Topology-aware aspect belongs in final postprocess for every path, but the policy should be topology-specific. Current aspect values are DAG-biased and very tall by default. Flat force should probably target near-square or metric-driven aspect, not `0.25`. Planar undirected and lattice-like flat graphs should preserve isotropy unless labels require widening.

Force2DInitIfFlat belongs only as a defensive fallback, not a primary flat-graph story. The comment says longest-path layering collapses cyclic graphs to y=0 and `Force2DInitIfFlat` randomizes y at `dagua/layout/ops/pipelines/dagua_native.py:1100` through `dagua/layout/ops/pipelines/dagua_native.py:1109`. That is exactly the wrong abstraction for sprint 20: flat graphs should enter a flat initializer directly.

## Config flag evolution

Do not remove the current flags immediately. They are valuable benchmark kill switches and tests rely on them. But demote them from policy to overrides.

Add a single high-level option:

```python
topology_dispatch: Literal["auto", "layered", "flat_force", "tree", "legacy_native"] = "auto"
```

Then reinterpret existing flags:

- `decompose_components`: keep as a wrapper override. `True` means allow auto decomposition; `False` disables it. It remains user-visible because component packing can change global geometry.
- `insert_dummy_nodes`: keep as a kill switch for `LONG_EDGE_DAG`. It should no longer be checked deep in `_prepare_native_config()`; the plan should say `use_dummy_nodes = auto_rule and config.insert_dummy_nodes`.
- `brandes_koepf_refine`: keep as a kill switch for layered coordinate refine. It should not be consulted by flat pipelines.
- `use_native_median_transpose`: keep as a kill switch for layered ordering polish. Rename later to `layered_ordering_refine` if public API churn is acceptable.

New internal plan fields should replace scattered `_dagua_native_*` attrs: `plan.topology`, `plan.use_component_pack`, `plan.use_dummy_nodes`, `plan.use_ordering_polish`, `plan.use_bk`, `plan.aspect_policy`, `plan.loss_profile`, `plan.init_profile`, `plan.scale_profile`. `prepare_pipeline_config()` should still resolve numeric config values, but it should not decide the algorithm class.

## Concrete refactor plan

What stays:

- `build_gradient_core()` stays and becomes a shared native sub-pipeline. Its memory-conscious per-loss backward rationale at `dagua/layout/ops/pipelines/dagua_native.py:251` through `dagua/layout/ops/pipelines/dagua_native.py:254` remains important.
- `resolve.py` keeps numeric config resolution: adaptive spacing, auto steps, overlap interval, projection iterations, stall config, flex conversion, and loss construction. It should stop owning topology dispatch.
- `graph_classify.py` stays as the cheap structural scanner, but grows dispatch-relevant fields and tags.
- Existing classic pipelines stay registered and available. `fa2`, `sfdp`, and `stress_sgd` should be used as ingredients or references, not duplicated.

What moves:

- Component decomposition helpers move from `dagua_native.py` to a native dispatch/component module, for example `dagua/layout/ops/pipelines/native_components.py`.
- `_should_use_native_dummy_nodes()` moves into plan construction, probably `dagua/layout/native_dispatch.py` or `dagua/layout/resolve_dispatch.py`.
- Tree fast path moves out of `_run_native_problem()` into the dispatch plan.
- Crossing polish assembly moves into `build_layered_ordering_refine()` or `native_layered_dag`.
- Aspect policy moves from a standalone resolver into `plan.aspect_policy`, while preserving `resolve_topology_aware_aspect()` as the implementation function initially.

What dies:

- The idea that `build_dagua_pipeline()` is the single default pipeline for every topology. Keep a compatibility wrapper with that name during migration, but it should call `build_native_dispatch_pipeline()` or build only the layered DAG sub-pipeline.
- The "flat cyclic graph" hack as primary behavior. `Force2DInitIfFlat` can remain as a safety op, but flat dispatch should not depend on layered init failing first.
- Private `_dagua_native_*` attrs as the main policy transport. They can exist during migration, but the target should be a typed plan object.
- The duplicate direct dispatch inside `engine.py`. Long term, `algorithm=None`, explicit `"dagua_native"`, and `_layout_inner(use_pipeline=True)` should converge on one native dispatch entrypoint.

Dead-code watchlist after refactor: `_run_native_problem()` may collapse to a thin "execute selected child pipeline" helper; `_prepare_native_config()` may lose dummy-node policy and become mostly a wrapper around `prepare_pipeline_config()`; `_layout_inner_pipeline()` may be unnecessary once default and direct internal paths share the same adapter; `Force2DInitIfFlat` may become unused by default after `native_flat_force` lands.

## Migration path

Use phases. A one-big-PR rewrite is too risky because the current default wins are real and the regression surface is broad.

Phase 1: add topology descriptor without behavior change. Extend `GraphStructure` and add `LayoutPlan`, but make every current graph choose equivalent current behavior. Add tests for new classifier fields on chain, tree, disconnected components, layered DAG with long edges, cyclic small-world, reciprocal undirected cycle, planar-ish sparse cycle, dense DAG, and ragged pyramid. No benchmark behavior should change.

Phase 2: extract wrappers and sub-pipelines with parity. Move component packing, layered DAG pipeline assembly, and ordering polish into named functions/modules. Keep `build_dagua_pipeline()` as a compatibility facade that returns the current layered path. Run existing tests plus h2h winners. Expected behavior should be identical or within floating deterministic tolerance.

Phase 3: route default through `NativeDispatch` but map all categories to existing current path except tree/component special cases already present. This centralizes policy without introducing flat force yet. Update `tests/test_layout_default_dispatch.py`, whose current contract says default calls `build_dagua_pipeline()` exactly once at `tests/test_layout_default_dispatch.py:46` through `tests/test_layout_default_dispatch.py:53`; after refactor it should assert default calls native dispatch and selected child plan.

Phase 4: introduce `native_flat_force` behind `topology_dispatch="auto"` but initially only for explicit `topology_dispatch="flat_force"` or an experiment flag. Compare on sprint-20 losses and protected winners. The threshold to enable by default should be concrete: improve at least three of `small_world_100`, `small_world_500`, `parallel_cycles_4x5`, `regular_3_30`, `planar_60` by more than 2 composite points, with no protected winner regressing by more than 0.5.

Phase 5: enable flat dispatch for low-risk categories. Start with cyclic, non-clustered, non-flex, connected graphs where `hierarchy_score` is low. Exclude planar/lattice DAG tags and any directed acyclic graph. Then widen to planar-undirected and regular graphs after benchmarks.

Phase 6: add SCC hybrid for directed cyclic hierarchical graphs. This is a second big feature and should not block flat force. Use it only after SCC and condensation tests exist.

Phase 7: deprecate old flags as public architecture controls. Keep them as kill switches for at least one release. Document `topology_dispatch` and plan override behavior.

## Risk and regression analysis

The highest risk is regressing Dagua's current layered strengths. The sprint context says `org_chart_deep`, `random_dag_200`, `hub_fanout_label_skew`, `org_chart_1_5_4_8`, `random_dag_50`, `random_bipartite_60`, `edge_label_braid`, `bipartite_4_3_4`, and karate graphs are strong wins. These graphs should remain on `native_layered_dag` or existing specialized behavior unless a new pipeline demonstrates a measured win.

The second risk is misclassification. The current classifier has a large-graph cutoff for a reason: above 10,000,000 nodes, it avoids degree and union-find allocations at `dagua/layout/graph_classify.py:425` through `dagua/layout/graph_classify.py:450`. Any new descriptor fields must have scale guards. For example, reciprocal edge fraction can be sampled on large graphs; exact cycle rank and component distributions can be skipped or estimated when memory cost is too high.

The third risk is changing public flag semantics. Tests currently instantiate `LayoutConfig(use_native_median_transpose=False)`, `brandes_koepf_refine=False`, `insert_dummy_nodes=False`, and `decompose_components=False`. Preserve those flags. The migration should change where they are interpreted, not whether they work.

The fourth risk is postprocess mismatch. Overlap projection and aspect fit are currently at the tail of the native path. Force-directed paths must still handle node sizes, labels, hard pins, clusters, and final direction transforms. `engine.layout()` applies direction and caches after default native returns at `dagua/layout/engine.py:984` through `dagua/layout/engine.py:991`; a new dispatch path must preserve this.

Required regression surface:

- Unit tests for `classify_graph` and new plan selection: trivial, tree, chain, forest, bipartite DAG, wide DAG, dense DAG, long-edge DAG, ragged pyramid, cyclic flat, reciprocal undirected, planar-undirected, clustered modifier, flex modifier, huge-graph guard.
- Existing feature tests: dummy-node skip/improve/no-regress, median/transpose acyclic improvement and cyclic skip, BK preserve-y/skip-cyclic/no-regress, component decomposition tiling and skip gates.
- Dispatch contract tests: default `algorithm=None`, explicit `algorithm="dagua_native"`, `_legacy`, trace fallback, relax fallback, other algorithms. These need updating from "build_dagua_pipeline called" to "native dispatch selected expected plan."
- Benchmark smoke set: sprint-20 losses plus protected winners. Minimum set: `ragged_feature_pyramid`, `planar_60`, `small_world_100`, `small_world_500`, `parallel_cycles_4x5`, `regular_3_30`, `hexagonal_lattice_42`, `dependency_500`, `org_chart_deep`, `random_dag_200`, `hub_fanout_label_skew`, `random_bipartite_60`.
- Runtime/memory tests for classification on large synthetic graphs. Any new O(E log E) or O(N) tensor allocation must be explicit and guarded.

## Final recommendation

Be ambitious on architecture, conservative on landing. The right end state is not "one smarter `if` inside `dagua_native.py`." It is a native topology dispatcher where classification produces a typed plan and each plan names a composition of existing ops. The first implementation should split the default into at least four named sub-pipelines: component packing, layered DAG, flat force, and tree/forest. The layered DAG path should initially be behavior-preserving, because it is where Dagua is already ahead. The new energy should go into the flat-force path and the classifier fields needed to route to it.

This is a pragmatic middle position between patchwork and a purist rewrite. It removes the main Frankenstein risk by centralizing policy, while preserving the sprint-19 wins as reusable modules. It also gives sprint 20 a clean place to add directed/undirected split and force-directed fallback without another 500 lines in `dagua_native.py`.

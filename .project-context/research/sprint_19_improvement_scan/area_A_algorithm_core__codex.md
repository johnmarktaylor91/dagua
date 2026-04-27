# Area A: Algorithm Core Review

## TL;DR

- The default `dagua_native` path is still a generic gradient solver with layered initialization, not a full layered layout pipeline. Compared to `dot`, `dagre`, ELK layered, and `igraph_sugiyama`, the biggest missing phases are topology-gated layered dispatch, network-simplex ranking, full discrete crossing reduction, and exact horizontal coordinate assignment. Those are the highest-leverage changes.
- Dagua already contains part of the answer in-tree: a separate `sugiyama` pipeline with dummy-node expansion and Brandes-Koepf coordinate assignment, plus standalone `MedianSweep` and `TransposeHeuristic` ops. The main failure is composition: the default path does not use them for DAG-like graphs, and the dedicated Sugiyama path still stops short of competitor-grade ranking and ordering.
- Runtime is being spent in the wrong places. For layered graphs, Dagua pays for 100-300 Adam steps plus global differentiable losses, while competitors spend most of their work in near-linear or sweep-based discrete phases. There are also avoidable local inefficiencies: duplicated layering, repeated CPU/Python conversions, and sampled soft losses where an exact discrete phase would be both cheaper and better.

## Baseline Comparison

Today’s default path is `algorithm=None -> "dagua_native"` in [dagua/layout/engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py:904) and [dagua/layout/engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py:943). The default `dagua_native` pipeline is `NativeEngineInit -> Force2DInitIfFlat -> optional stress prep -> anneal -> Adam gradient core -> BarycenterReorder -> OverlapProjection -> AspectRatioFit -> ClusterGridArrange` in [dagua/layout/ops/pipelines/dagua_native.py](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/dagua_native.py:360).

That is materially different from the classic layered competitors:

- `dot`: rank assignment, crossing minimization with transpositions, coordinate assignment, spline routing (Gansner et al. 1993; Brandes-Koepf 2001).
- `dagre`: network-simplex ranking, median/barycenter ordering with transpose-style improvement, Brandes-Koepf placement (Gansner et al. 1993; Brandes-Koepf 2001).
- ELK layered: greedy cycle breaking, layered ranking, stronger crossing minimization, Brandes-Koepf or linear-segments node placement, port-aware routing (Sugiyama et al. 1981; Brandes-Koepf 2001; Sander linear segments).
- `igraph_sugiyama`: classic Sugiyama layering, crossing minimization, dummy-node bends, layered coordinates (Sugiyama et al. 1981).

Dagua does have a separate Sugiyama pipeline with dummy expansion and BK assignment in [dagua/layout/ops/pipelines/sugiyama.py](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/sugiyama.py:59), [dagua/layout/ops/sugiyama.py](/home/jtaylor/projects/dagua/dagua/layout/ops/sugiyama.py:1500), and [dagua/layout/ops/sugiyama.py](/home/jtaylor/projects/dagua/dagua/layout/ops/sugiyama.py:1733), but that path is not the default and it still uses longest-path layering plus barycenter-only ordering.

## Findings

### 1. Missing topology-gated dispatch from the default path into a true layered pipeline

- Severity: High
- Evidence: The default engine remaps `algorithm=None` to `"dagua_native"` and only special-cases trees; there is no analogous fast path for general DAGs or layered families in [dagua/layout/engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py:904), [dagua/layout/ops/pipelines/dagua_native.py](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/dagua_native.py:529), and [dagua/layout/resolve.py](/home/jtaylor/projects/dagua/dagua/layout/resolve.py:292). `classify_graph()` already distinguishes `BIPARTITE_DAG`, `WIDE_LAYERED`, `num_components`, and `is_planar_hint`, but native composition only uses tree/chain overrides and acyclicity gating in [dagua/layout/graph_classify.py](/home/jtaylor/projects/dagua/dagua/layout/graph_classify.py:269) and [dagua/layout/resolve.py](/home/jtaylor/projects/dagua/dagua/layout/resolve.py:379).
- Proposed intervention: Add a family-dispatch phase before `dagua_native`: for acyclic or near-layered graphs, run an upgraded Sugiyama pipeline first, then optionally apply a short continuous refine pass. The existing `sugiyama` pipeline is the right base; it should become the default for DAG-heavy families rather than an opt-in algorithm.
- Expected impact: On layered DAG losses such as `dependency_500`, `dense_pair_50`, `extreme_mixed_width_transformer`, and the planar DAGs, this is the single biggest likely gain: roughly +2 to +6 composite on affected graphs, with a likely positive suite mean as well. Runtime should also drop materially on those graphs because discrete layered phases replace most of the 150-250 step gradient solve.
- Rough effort: 8-14 hours

### 2. Missing network-simplex rank assignment

- Severity: High
- Evidence: Both the native initializer and the dedicated Sugiyama path use longest-path layering, not network simplex: [dagua/layout/init_placement.py](/home/jtaylor/projects/dagua/dagua/layout/init_placement.py:58), [dagua/layout/ops/init.py](/home/jtaylor/projects/dagua/dagua/layout/ops/init.py:2405), and [dagua/layout/ops/sugiyama.py](/home/jtaylor/projects/dagua/dagua/layout/ops/sugiyama.py:1451). Promotion in [dagua/layout/ops/sugiyama.py](/home/jtaylor/projects/dagua/dagua/layout/ops/sugiyama.py:249) reduces dummy count, but it is still not the Gansner et al. 1993 objective of minimizing total weighted edge length under rank constraints.
- Proposed intervention: Implement a `NetworkSimplexLayering` op and use it for layered families. Longest-path can remain as a fallback or warm start. Feed edge weights and `minLen` semantics through the ranking stage so rank assignment directly minimizes long-edge slack.
- Expected impact: Strongest direct effect is on `edge_length_cv`, `depth_spearman`, and `dag_consistency`, especially for deep sparse DAGs and dense DAGs with uneven slack. I would expect +1 to +4 composite on affected DAGs, and a visible improvement in long-edge behavior before any crossing pass runs.
- Rough effort: 16-30 hours

### 3. Default layered graphs are missing exact horizontal coordinate assignment

- Severity: High
- Evidence: The native default uses force-based optimization plus post-hoc aspect-ratio fitting; it never runs exact layered coordinate assignment in [dagua/layout/ops/pipelines/dagua_native.py](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/dagua_native.py:372). The separate Sugiyama path already has Brandes-Koepf compaction in [dagua/layout/ops/sugiyama.py](/home/jtaylor/projects/dagua/dagua/layout/ops/sugiyama.py:1733) and the shared BK implementation in [dagua/layout/ops/coordinate.py](/home/jtaylor/projects/dagua/dagua/layout/ops/coordinate.py:50).
- Proposed intervention: Promote Brandes-Koepf 2001 node placement into the default DAG path. The most defensible composition is `rank -> dummy expansion -> order -> BK -> optional short continuous relax`, not `gradient -> barycenter reorder`.
- Expected impact: This directly targets the weighted composite terms Dagua still loses on: `edge_straightness`, `crossing_rate`, and indirectly `edge_length_cv`. On DAGs with long chains and skip edges, I would expect +2 to +6 composite and a runtime win because exact placement is linear-ish in the layered graph size.
- Rough effort: 8-16 hours

### 4. Crossing reduction is incomplete in both the default path and the dedicated Sugiyama path

- Severity: High
- Evidence: The native path only applies a late `BarycenterReorder` in [dagua/layout/ops/pipelines/dagua_native.py](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/dagua_native.py:421), and that op only permutes existing `x` coordinates by barycenter order in [dagua/layout/ops/barycenter.py](/home/jtaylor/projects/dagua/dagua/layout/ops/barycenter.py:115). The dedicated Sugiyama path uses repeated barycenter sweeps, but not median sweeps or transpose refinement, in [dagua/layout/ops/sugiyama.py](/home/jtaylor/projects/dagua/dagua/layout/ops/sugiyama.py:1617). Yet Dagua already has unused `MedianSweep` and `TransposeHeuristic` ops in [dagua/layout/ops/ordering.py](/home/jtaylor/projects/dagua/dagua/layout/ops/ordering.py:687) and [dagua/layout/ops/ordering.py](/home/jtaylor/projects/dagua/dagua/layout/ops/ordering.py:766).
- Proposed intervention: Upgrade ordering to a proper layered stack on the dummy-expanded graph: initialize from median or weighted barycenter, alternate down/up sweeps, run transpose after each sweep block, stop when crossings stop improving. This should be used in the Sugiyama path and then exposed to the default DAG dispatch.
- Expected impact: This is a pure quality lever. Expect +1 to +3 composite on crossing-heavy DAGs and on small pathological cases where local swaps matter disproportionately. It is also likely to reduce dependence on the soft differentiable crossing proxy.
- Rough effort: 8-14 hours

### 5. Long-edge dummy-node treatment exists, but it is absent from the default DAG path

- Severity: High
- Evidence: The explicit Sugiyama pipeline inserts dummy nodes for multi-rank edges in [dagua/layout/ops/sugiyama.py](/home/jtaylor/projects/dagua/dagua/layout/ops/sugiyama.py:1500), but the default `dagua_native` path never expands the graph before ordering or coordinate placement in [dagua/layout/ops/pipelines/dagua_native.py](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/dagua_native.py:360). The soft crossing loss emulates segments on the fly in [dagua/layout/constraints.py](/home/jtaylor/projects/dagua/dagua/layout/constraints.py:975), but that does not give node ordering or x-placement any explicit long-edge structure.
- Proposed intervention: For DAG-like graphs, make dummy expansion an early phase, not just an optional separate algorithm. Use dummy nodes for ordering, BK alignment, and optional route reconstruction. Preserve only original-node positions at output.
- Expected impact: Strongest gains should appear in `dag_consistency`, `depth_spearman`, and `edge_straightness` on deep graphs with skip connections. I would expect +1 to +4 composite on affected graphs.
- Rough effort: 8-12 hours

### 6. Component decomposition and packing are missing for general disconnected graphs

- Severity: High
- Evidence: `classify_graph()` computes `num_components` in [dagua/layout/graph_classify.py](/home/jtaylor/projects/dagua/dagua/layout/graph_classify.py:255), but the default native pipeline does not branch on it at all; its full composition is still a single global solve in [dagua/layout/ops/pipelines/dagua_native.py](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/dagua_native.py:360). The only obvious component-aware coordinate logic in-tree is tree-specific Reingold-Tilford handling in [dagua/layout/ops/coordinate.py](/home/jtaylor/projects/dagua/dagua/layout/ops/coordinate.py:1393).
- Proposed intervention: Add a connected-components preprocess phase that lays out each component independently with the family-appropriate subpipeline, then packs component bounding boxes with size-aware spacing. This should happen before global gradient losses so disconnected components do not waste repulsion budget against one another.
- Expected impact: This directly targets `disconnected_label_cycle_collage`-style failures and should improve both quality and runtime. On disconnected graphs, +1 to +3 composite is plausible, and runtime could improve near-linearly with component count because each subproblem is smaller.
- Rough effort: 10-18 hours

### 7. Planar graph handling is effectively absent

- Severity: Medium
- Evidence: `GraphStructure` exposes only an `is_planar_hint`, derived from the trivial `E < 3V - 6` test, in [dagua/layout/graph_classify.py](/home/jtaylor/projects/dagua/dagua/layout/graph_classify.py:278). That hint is only stashed on config in [dagua/layout/resolve.py](/home/jtaylor/projects/dagua/dagua/layout/resolve.py:342); the native loss construction only consumes acyclicity, not planarity, in [dagua/layout/resolve.py](/home/jtaylor/projects/dagua/dagua/layout/resolve.py:379).
- Proposed intervention: For small graphs, add an exact planarity test and a planar fast path. For planar DAGs, combine planar embedding or Tutte-style straight-line placement with layered constraints when compatible; for non-layered planar graphs, a dedicated planar embedder is still preferable to generic gradient descent.
- Expected impact: This is targeted rather than universal, but the target cases are already in the loss table: `hexagonal_lattice_42` and `sierpinski_42`. Expect +1 to +3 composite on those families, mostly via zero or near-zero crossings and better angular regularity.
- Rough effort: 16-32 hours

### 8. Edge-length normalization is too soft and too late

- Severity: Medium
- Evidence: The current native path relies on `EdgeLengthVarianceLoss` and an end-of-pipeline `AspectRatioFit` in [dagua/layout/resolve.py](/home/jtaylor/projects/dagua/dagua/layout/resolve.py:427) and [dagua/layout/ops/pipelines/dagua_native.py](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/dagua_native.py:434). There is no exact compaction or target-length phase keyed to rank differences. In the vectorized initializer for `N > 100`, x positions are even spaced using average width rather than exact per-layer widths in [dagua/layout/init_placement.py](/home/jtaylor/projects/dagua/dagua/layout/init_placement.py:293).
- Proposed intervention: Add a post-ordering compaction/normalization phase that explicitly equalizes adjacent-layer edge lengths subject to node sizes and separations. In a layered path, this should sit after BK as a lightweight local adjustment, not as a global differentiable objective.
- Expected impact: Mostly `edge_length_cv`, with smaller positive spillover into readability. Expect roughly +0.5 to +2 composite on affected graphs; larger if rank assignment is also improved.
- Rough effort: 6-12 hours

### 9. Port constraints and upward-planarity-aware preferences are missing

- Severity: Medium
- Evidence: The default native path signature forwards clusters, flex, layers, and weights, but there is no port-side or port-order phase in [dagua/layout/ops/pipelines/dagua_native.py](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/dagua_native.py:453). The Sugiyama pipeline phases are fixed to validate, acyclicize, layer, dummy-expand, barycenter-order, coordinate-assign, and optionally build routes in [dagua/layout/ops/pipelines/sugiyama.py](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/sugiyama.py:59). By contrast ELK layered explicitly exposes port constraints and upward-layout controls.
- Proposed intervention: Add optional port constraints to the layered path: side assignment, fixed port order, and upward bias during cycle breaking and coordinate alignment. For graphs with semantically ordered fan-outs, use these constraints during ordering instead of trying to recover them from soft losses.
- Expected impact: Moderate but real on transformer-like small DAGs, dense pairings, and user-authored dependency diagrams. Likely +0.5 to +1.5 composite on affected graphs, especially in `edge_straightness` and crossings.
- Rough effort: 20-40 hours

### 10. Iteration budget and convergence are not topology-aware enough

- Severity: Medium
- Evidence: Auto step count is just a size ladder in [dagua/layout/resolve.py](/home/jtaylor/projects/dagua/dagua/layout/resolve.py:68), and early stopping is a short relative-loss stall check in [dagua/layout/resolve.py](/home/jtaylor/projects/dagua/dagua/layout/resolve.py:142) and [dagua/layout/ops/converge.py](/home/jtaylor/projects/dagua/dagua/layout/ops/converge.py:477). The pipeline does not use displacement-based convergence even though the op exists in [dagua/layout/ops/converge.py](/home/jtaylor/projects/dagua/dagua/layout/ops/converge.py:150).
- Proposed intervention: Split convergence policy by family. For layered graphs, discrete ordering convergence and BK completion should end the main solve, followed by at most a short refine stage. For force-based families, combine loss stall with displacement threshold and maybe learning-rate floor. Do not give all graph families the same 50-500 step ladder.
- Expected impact: Mostly runtime, with some quality improvement from avoiding over-optimization after discrete structure is fixed. I would expect 20-50% runtime reduction on many DAG-like graphs with no quality loss, and occasionally small quality gains by preserving a cleaner discrete solution.
- Rough effort: 4-8 hours

### 11. There is avoidable hot-path overhead in the default initializer and layered utilities

- Severity: Medium
- Evidence: `NativeEngineInit` computes positions with `init_positions()` and then recomputes longest-path layering again for `state.layers` when no prebuilt assignments were passed, so layering work is duplicated in [dagua/layout/init_placement.py](/home/jtaylor/projects/dagua/dagua/layout/init_placement.py:58), [dagua/layout/ops/init.py](/home/jtaylor/projects/dagua/dagua/layout/ops/init.py:2364), and [dagua/layout/ops/init.py](/home/jtaylor/projects/dagua/dagua/layout/ops/init.py:2405). The small-graph initializer and Sugiyama helpers repeatedly use `.tolist()` and Python dict/list loops in [dagua/layout/init_placement.py](/home/jtaylor/projects/dagua/dagua/layout/init_placement.py:133), [dagua/layout/init_placement.py](/home/jtaylor/projects/dagua/dagua/layout/init_placement.py:486), [dagua/layout/ops/sugiyama.py](/home/jtaylor/projects/dagua/dagua/layout/ops/sugiyama.py:166), [dagua/layout/ops/sugiyama.py](/home/jtaylor/projects/dagua/dagua/layout/ops/sugiyama.py:345), and [dagua/layout/ops/sugiyama.py](/home/jtaylor/projects/dagua/dagua/layout/ops/sugiyama.py:490). The layered crossing proxy also forces several `.item()` synchronizations per step in [dagua/layout/constraints.py](/home/jtaylor/projects/dagua/dagua/layout/constraints.py:1019) and [dagua/layout/constraints.py](/home/jtaylor/projects/dagua/dagua/layout/constraints.py:1072).
- Proposed intervention: Cache layer assignments produced during init; stop recomputing them in `NativeEngineInit`. Move small-graph layered preprocessing onto tensor-first utilities where practical. If the upgraded DAG path uses discrete ordering/BK, the soft crossing proxy can often be disabled entirely for those graphs, which removes one of the noisier global losses.
- Expected impact: Runtime-only or runtime-first. Expect a modest win on small/medium graphs from removing duplicated work, and a larger win once layered DAGs stop paying for the sampled crossing proxy inside the gradient loop.
- Rough effort: 8-16 hours

### 12. Multilevel exists on paper but is effectively disabled in production composition

- Severity: Low to Medium
- Evidence: `prepare_pipeline_config()` raises the default V-cycle threshold to `1_000_000`, explicitly disabling the path unless the user opts in, in [dagua/layout/resolve.py](/home/jtaylor/projects/dagua/dagua/layout/resolve.py:349). That means the default path for large layered graphs is still the flat gradient core, not a scalable hierarchy.
- Proposed intervention: After the layered-family dispatch is fixed, rehabilitate multilevel only for the remaining force-directed families or for very large layered graphs with DAG-aware coarsening. Do not send general DAGs into the current flat gradient path at large `N`.
- Expected impact: Mostly runtime and scalability, not benchmark-full mean. Important for future growth, but behind the layered-quality phases above.
- Rough effort: 20-40 hours

## Runtime Observations

- The biggest algorithmic runtime waste is strategic, not micro-level: layered graphs are still paying for the generic gradient core in [dagua/layout/ops/pipelines/dagua_native.py](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/dagua_native.py:124) instead of mostly discrete phases.
- The soft crossing loss is still a poor cost-quality trade. It samples or caps pairs in [dagua/layout/ops/loss_engine.py](/home/jtaylor/projects/dagua/dagua/layout/ops/loss_engine.py:318) and [dagua/layout/constraints.py](/home/jtaylor/projects/dagua/dagua/layout/constraints.py:944), but it does not replace real layered ordering. It is paying global differentiable cost for an inferior signal.
- `init_positions()` does meaningful work that `NativeEngineInit` then repeats: the code computes longest-path layers once for placement and again for `state.layers`. That is unnecessary per-solve overhead.
- The dedicated Sugiyama utilities are still CPU/Python heavy. That is acceptable for small graphs, but once this path becomes the default for DAGs it will be worth vectorizing the neighbor-building and dummy-expansion utilities.
- GPU/CPU fallback in overlap projection is defensible, but it still copies full tensors when GPU sweep-line projection cannot fit in memory in [dagua/layout/projection.py](/home/jtaylor/projects/dagua/dagua/layout/projection.py:134). That is another reason to reduce reliance on a long gradient loop for layered graphs.

## Recommended Action Queue

1. Make layered-family dispatch real: route DAG-like graphs from `dagua_native` into the Sugiyama path, then optionally run a short continuous refine stage.
2. Add network-simplex ranking to the Sugiyama path and make it the default ranker for layered families.
3. Upgrade layered ordering to `median/barycenter + transpose`, using the already existing ordering ops instead of only late `BarycenterReorder`.
4. Promote Brandes-Koepf placement to the default DAG path; stop treating exact coordinate assignment as an opt-in alternate algorithm.
5. Add component decomposition and packing before any global force solve.
6. Remove duplicated layering work from `NativeEngineInit` and reduce Python/CPU churn in the layered utilities.
7. Add topology-aware stopping rules so discrete layered solves do not burn generic gradient steps.
8. Add a small-graph planar fast path for exact or near-exact planar embeddings.
9. Add port constraints and upward-layout options after the layered backbone is stabilized.
10. Revisit multilevel only after the layered and disconnected-graph phases above are fixed.

## Bottom Line

The highest-value conclusion is simple: Dagua already has enough building blocks to stop losing layered graphs for avoidable reasons. The missing work is not another soft loss. It is the classical layered skeleton that competitors still rely on: better ranking, better dummy-node handling, better discrete ordering, and exact coordinate assignment, all selected automatically for the graph families that need them.

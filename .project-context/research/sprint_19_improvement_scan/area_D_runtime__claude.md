# Area D: Runtime -- Claude Performance Audit

Sprint 19 improvement scan. Target: reduce dagua layout runtime on CPU
without quality regression. All measurements taken 2026-04-24 on the
Linux workstation with `CUDA_VISIBLE_DEVICES=""`.

## 1. TL;DR -- Top 3 Speedups

1. **`_project_exact` final overlap projection is catastrophically slow on
   mid-size graphs.** At n=300 a single call takes **14.9 seconds** out
   of a 27.5 s total layout (54%). The op allocates a dense
   `[N,N]` overlap matrix per iteration and then builds `torch.triu_indices`
   and masks rather than just iterating the upper triangle of a small
   resolved-pairs set. Dropping the exact path below some modest ceiling
   (e.g. n<=100) and switching to the sweep-line/layered projection for
   100<n<=500 will recover ~10-14 s on every 200-500 node graph.
   **Estimated impact: 40-55% speedup on graphs in the 200-500 node
   range**. Effort 2-4h.

2. **`_exact_repulsion_loss` and `_exact_overlap_loss` are O(N^2) in the
   autograd forward and dominate steps for 50 <= N <= 500**. On
   dag_chain_100 the exact repulsion path is 345 ms of 1.0 s (34%) and
   exact overlap is another 129 ms (13%). Both allocate full `[N,N]`
   tensors and a `torch.eye(N)` boolean mask per step. The existing
   sampled path (`k=128`) is gated at `threshold=2000` / `exact_threshold=2000`;
   lowering the threshold to ~300 retains accuracy for small N (sample_k
   >= N covers all pairs) while avoiding the O(N^2) forward graph for
   larger. **Estimated 25-35% speedup on 100-500 node graphs**. Effort 1h
   (threshold change + test sweep).

3. **`_crossing_loss_layered` dominates small-graph optimization** (80% of
   hex_42 runtime = 361 ms of 447 ms; 42% of dag_chain_100 = 434 ms of
   1 s). Each step calls `torch.repeat_interleave` 3x, allocates a full
   segment tensor for every (src,tgt) pair's span, then globally sorts
   and groups. The op runs on tiny graphs where the *legacy fallback*
   (simple O(E^2) pair sigmoid) would be one order of magnitude cheaper.
   Route through `_crossing_loss_fallback` when `num_edges <= ~200`
   regardless of `layer_assignments`, and cache the per-iteration
   segment decomposition across gradient steps (it only depends on
   `edge_index` + `layers`, which are constant during gradient_core).
   **Estimated 50-70% speedup on graphs with layered continuous
   crossing**. Effort 3-5h for a proper cache.

Aggregate: mixing these three changes should take typical 100-500 node
layout from ~1-35 s to ~0.5-10 s on CPU.

## 2. Baseline Wall-Clock

Measured with `time.perf_counter()` over a warmed-up process, mean of 3
consecutive layouts (except the large runs where wall-clock was captured
from a single run after warmup). `LayoutConfig(seed=42)` default on
every graph.

| Graph              | N   | E    | Wall/run | Notes                          |
|--------------------|-----|------|----------|--------------------------------|
| hex_42 (grid 7x6)  |  42 |   71 |  447 ms  | DAG, planar                    |
| cycle_100          | 100 |  100 |  529 ms  | cyclic flat                    |
| tree_127 (bin D7)  | 127 |  126 |  582 ms  | TREE classified but fast path not firing (see hotspot #9) |
| dag_chain_20       |  20 |   25 |  508 ms  | small DAG                      |
| dag_chain_50       |  50 |   65 |  546 ms  | small DAG                      |
| dag_chain_100      | 100 |  132 | 1.02 s   | DAG (100 layers)               |
| dag_chain_200      | 200 |  265 | 14.8 s   | huge cliff vs 100              |
| dag_chain_300      | 300 |  398 | 10.9 s   | cliff continues                |
| sparse_300 (k=3)   | 300 |  851 | 27.5 s   | overlap-projection dominated   |

**Key observation**: the 100 -> 200 node cliff is ~15x, driven almost
entirely by `_project_exact` crossing its `N^2` inflection where
`torch.eye`, `torch.triu_indices`, and the full `[N,N]` boolean mask
thrash CPU cache. `sparse_300` is dominated by the same call in the
final projection (14.9 s in one call, 20 iterations).

Early-stop is firing aggressively on small graphs: LossGroup.apply shows
only 3-4 ncalls per layout on hex_42 and cycle_100. That means
`StallCount` with `rel_threshold=5e-4` and `stall_limit=3` for N<=200
kills optimization after ~4 steps. Many small-graph runs are already
spending the majority of wall-clock on init + final projection and post
passes, not gradient steps.

## 3. Hotspot Table (Top 20 by Cumulative Time)

Aggregated from `cProfile.sort_stats('cumulative')` across
`hex_42`, `cycle_100`, `tree_127`, `dag_chain_100`, and `sparse_300`.

| #  | Function | File:Line | Share (cycle_100) | Share (sparse_300) | Notes |
|----|----------|-----------|-------------------|--------------------|-------|
|  1 | `_project_exact`                | projection.py:192      | 0%        | **54%** (14.9s) | O(N^2) full matrix + triu_indices per iteration |
|  2 | `_exact_repulsion_loss`         | loss_engine.py:166     | **64%**   | 8%  | `pos.unsqueeze(0) - pos.unsqueeze(1)` + `~eye(N)` mask |
|  3 | `_exact_overlap_loss`           | loss_engine.py:204     | 30%       | 6%  | 5 dense `[N,N]` tensors per call |
|  4 | `_crossing_loss_layered`        | constraints.py:975     | 0%        | 1%  | 80% of hex_42; 42% of dag_chain_100 |
|  5 | `{run_backward}`                | torch autograd         | 1.5%      | 16% | Backward pass through O(N^2) forward graphs |
|  6 | `torch.repeat_interleave`       | constraints.py:1033-1112 | 0% | 0.8% | Called 9x per crossing_loss step |
|  7 | `tensor.abs`                    | projection.py:208      | 0%        | 6.5% | 40 calls, 1.8s in sparse_300 (part of project_exact) |
|  8 | `tensor.any`                    | projection.py:213,219 | 0%        | 6.2% | 107 calls, 1.7s in sparse_300 |
|  9 | `ReingoldTilfordTree` fast-path | dagua_native.py:537-555 | 0% (miss)|0% (miss) | `use_tree_fast_path` defaults to **False** on `LayoutConfig`; bin-tree_127 pays full pipeline |
| 10 | `BarycenterReorder.apply`       | barycenter.py:137      | 0%        | 0.05% | 48 ms on hex_42 (11%); scales with num_layers * iters=8 |
| 11 | `layer_index.nodes_in_layer`    | layers.py:34           | 0%        | 0.07% | 796 calls on dag_chain_100 (21 ms); each call does two `.item()` CPU syncs |
| 12 | `init_positions`                | init_placement.py:23   | 1%        | 0.1% | 3-16 ms on small graphs, mostly cheap |
| 13 | `classify_graph`                | graph_classify.py:210  | 0%        | 0.06% | 17 ms on dag_chain_100; repeated work if pipeline runs twice |
| 14 | `_cluster_cache` build          | loss_engine.py:131     | 0%        | 0%  | Rebuilds on V-cycle level change even when unchanged |
| 15 | `OverlapProjection.apply` (final) | project.py:208 (wrap) | 0.2%    | 54% | Delegates to `_project_exact` |
| 16 | `make_acyclic_robust`           | cycle.py:196           | 0%        | 0.5% | ~12 ms on tree_127; DFS in Python |
| 17 | `edge_attraction_loss`          | constraints.py:91      | 0.1%      | 0.1% | Fine, just per-edge |
| 18 | `spacing_consistency_loss`      | constraints.py:1693    | 0.1%      | 0.3% | |
| 19 | `.item()` scalar sync           | various                | 0.1%      | 1%  | 1861 calls on dag_chain_100 |
| 20 | `PeriodicOverlapProjection` (intra-loop) | project.py:285 | 0% (skipped due to stall) | minor | `overlap_interval=5` so runs at step 5, 10, ...; often short-circuited by early-stop |

Top hotspots cluster into three families:
1. Dense O(N^2) forward/backward graphs in repulsion and overlap.
2. Dense O(N^2) projection loops, especially `_project_exact`.
3. Per-step Python-visible torch ops with `.item()` syncs
   (`nodes_in_layer`, `cum_spans` etc.)

## 4. Optimization Opportunities

### 4.1 HIGH severity

#### H1. Lower `_project_exact` ceiling; use sweep for 100 < N <= 500
- **File:** `dagua/layout/projection.py:75-85`.
- **Observation:** `_project_exact` unconditionally handles `N <= 500`.
  On sparse_300 the final projection is 14.9 s for 20 iterations
  (~750 ms/iter) because `_project_exact` materializes `[N,N]` f32
  tensors (360 KB each) four times plus `triu_indices` + bool masks.
  For layered graphs, `_project_sweep` uses `torch.argsort` of a
  composite key (O(N log N)) and has no per-layer Python loops.
- **Proposed fix:** Change dispatch to prefer `_project_sweep` when
  `layer_index is not None and N > 100` (keep exact path for
  singletons / small layer-free graphs). Non-layered graphs with
  N > 200 should use `_project_grid` (already exists, also O(N log N)).
- **Risk:** Sweep is layer-only; if graph has no `layer_index`
  (unlikely in native pipeline), grid path handles it.  Different
  numerical trajectory but both converge to an overlap-free state.
  Regression gate: overlap_count metric. Low risk.
- **Effort:** 2-4h (dispatch + regression sweep).
- **Impact:** 40-55% speedup on 200-500 node graphs. Kills the
  100->200 cliff.

#### H2. Drop exact repulsion / overlap thresholds from 2000 to ~200
- **File:** `dagua/layout/ops/loss_engine.py:298, 313`.
- **Observation:** On cycle_100 the exact repulsion forward + backward
  builds a `[100,100,2]` diff tensor + `[100,100]` dist_sq + `eye(N)`
  mask at every step. The sampled path with `sample_k=128` covers
  every pair at N<=128 and still takes O(N*k) instead of O(N^2). For
  N=200, sample_k=128 means 128*200 = 25 600 pair evaluations per
  step vs 40 000 for exact; forward graph is ~1.5x smaller and has
  no `torch.eye` allocation.
- **Proposed fix:** Set `RepulsionLossConfig.threshold = 200`,
  `OverlapAvoidanceLossConfig.exact_threshold = 200`. Also consider a
  small-N special case: when N <= 64, materialize the
  upper-triangular indices ONCE (cached in state.extras) and reuse,
  avoiding `torch.eye` allocation per step.
- **Risk:** Sampled repulsion has higher gradient variance. Could
  slow convergence slightly, but the target graphs already early-stop
  in 3-5 steps so it likely affects nothing. Low-medium risk.
- **Effort:** 1-2h (config change + benchmark sweep).
- **Impact:** 25-35% speedup on 100-500 node graphs.

#### H3. Route small-N crossing through the simple fallback; cache
    segment decomposition
- **File:** `dagua/layout/constraints.py:929, 975`.
- **Observation:** `_crossing_loss_layered` is 80% of hex_42 runtime.
  Per-call work: 3 `torch.repeat_interleave` (358 ms on hex_42), plus
  sort + unique_consecutive + group-offset index math. The *fallback*
  path is a simple sigmoid on randomly sampled pairs -- much cheaper,
  and for tiny graphs (71 edges) the quality difference is
  negligible.
- **Proposed fix part A:** Raise `num_edges < 20` to
  `num_edges <= 200` in the fallback dispatch. (Current is
  `or num_edges < 20`.)
- **Proposed fix part B:** Cache the per-edge segment decomposition
  in `state.extras["_crossing_segments"]` keyed on
  `(id(edge_index), id(layers))`. The segment indices, spans, and
  offsets are pure functions of topology and layer assignments,
  neither of which change during gradient_core. Only the
  `src_x`/`tgt_x` gather needs to re-run each step.
- **Risk:** Fallback has slightly different gradient shape
  (pairs of entire edges vs layer-local virtual segments). For
  hex_42 / dag_chain_100 both produce near-zero crossings anyway.
  Medium risk if the cached decomposition is stale across V-cycle
  level boundaries; invalidation is straightforward via identity check.
- **Effort:** 3-5h for robust cache.
- **Impact:** 50-70% speedup on small layered graphs; 20% on
  medium.

#### H4. Tree fast-path opt-in: flip default or fix the config check
- **File:** `dagua/layout/ops/pipelines/dagua_native.py:548-555`,
  `dagua/config.py` (LayoutConfig).
- **Observation:** `classify_graph` returns `GraphFamily.TREE` for
  our bin-tree_127 sample, but `getattr(config, "use_tree_fast_path",
  True)` evaluates to `False` because `LayoutConfig` has a
  `use_tree_fast_path` attribute defaulting to `False`. The
  `getattr(..., True)` default is a dead branch.
- **Proposed fix:** Either (a) flip the LayoutConfig default to
  `True` (per the code comment intent), or (b) remove the attribute
  from LayoutConfig so the getattr default kicks in.
- **Risk:** R-T output differs from the gradient pipeline, may
  regress specific tree graphs where the continuous optimizer
  happened to produce a better cluster_separation or aesthetic.
  Verify on `binary_tree_127`, `org_chart_deep`, any tree-family
  benchmark graphs. Medium risk -- need regression sweep before
  shipping.
- **Effort:** 1h + regression sweep (2-4h).
- **Impact:** **95%+ speedup on trees** (from ~600 ms to <5 ms per
  layout). Would immediately turn all tree-family graphs into near-
  instant operations.

### 4.2 MEDIUM severity

#### M1. BarycenterReorder: amortize per-iteration Python work
- **File:** `dagua/layout/ops/barycenter.py:137-213`, also
  `dagua/layout/layers.py:34`.
- **Observation:** At `iterations=8`, the op runs 8 sweeps x (L-1)
  inner iterations. For dag_chain_100 with ~100 layers this is
  796 `nodes_in_layer` calls (21 ms), each doing two `.item()` CPU
  syncs. Plus each inner iteration rebuilds
  `member_to_idx = torch.full((N,), -1)` (O(N) scatter per layer).
- **Proposed fixes:**
  - Replace `nodes_in_layer` body with `self.sorted_nodes[slice]`
    after reading `layer_offsets` as a `.tolist()` ONCE outside the
    loop.
  - Precompute the per-layer `member_to_idx` ONCE at the start of
    `apply()` (a single `[N]` tensor indexed by `node_to_layer` plus
    a per-layer offset).
  - Per-iteration, only recompute barycenters -- the `mask =
    (src_layer==k) & (tgt_layer==adj_layer_idx)` computation is a
    pure function of `layers` + `edge_index` -- both frozen within
    one `apply()` call -- so the full list of
    `(k, neighbour_src, neighbour_dst)` tuples can be precomputed
    once.
- **Risk:** None, pure refactor. Low.
- **Effort:** 2-3h.
- **Impact:** ~40% reduction in barycenter time. Modest absolute
  savings (hex_42: 48 ms -> ~30 ms). Adds up across benchmark
  (93 graphs * 20 ms = ~2 s).

#### M2. StallCount early-stop is aggressive -- document + verify
- **File:** `dagua/layout/ops/converge.py` (StallCount),
  `dagua/layout/resolve.py:142` (stall_config).
- **Observation:** `stall_limit=3, rel_threshold=5e-4` on
  `N<=200` causes gradient_core to exit after ~4 steps on every
  small graph I profiled (hex_42, cycle_100, dag_chain_100). This
  is NOT a speed problem per se -- it's actually the primary
  reason CPU layouts finish in under a second -- but means all
  per-loss machinery (`anneal`, `clip_grad_norm`, backward, ...)
  runs ~4x, and *most* of the wall-clock is in init + final
  projection + post-passes. Confirm this is intended and that
  disabling it (`stall_limit=999`) doesn't unlock meaningful
  composite-score gains that justify the extra compute.
- **Risk:** If early-stop is masking an under-tuned optimizer, the
  real speedup lever is tightening the stall tolerance even
  further on small graphs where the gradient norm rapidly
  collapses.
- **Effort:** 1h profile sweep.
- **Impact:** Diagnostic. Not a direct fix.

#### M3. `_spread_fanout_children` uses `.item()` in a Python loop
- **File:** `dagua/layout/init_placement.py:571-622`.
- **Observation:** For every hub with degree >= 8, the post-pass
  does `positions[hub, 0].item()`, `node_sizes[c, 0].item()` for
  every child, and `positions[c, 0].item()` during sort. On a
  hub graph with many fan-out nodes (org_chart_deep, hub_fanout_*
  graphs) this is a lot of CPU syncs. The loop could vectorize by
  gathering once: `torch.sort(positions[children, 0])`.
- **Risk:** None, pure refactor.
- **Effort:** 1h.
- **Impact:** 1-5% on hub-heavy graphs.

#### M4. `_init_positions_vectorized` drops back to CPU via `.item()`
  in its barycenter loop
- **File:** `dagua/layout/init_placement.py:436-481`.
- **Observation:** Inner `for L in range(num_layers)` loop at L436
  unpacks `offsets[L].item()` and `offsets[L+1].item()` per layer.
  For high-depth DAGs (dag_chain_100 has 100 layers) this is 100
  sync points.
- **Proposed fix:** Precompute `offsets.tolist()` once before the
  loop.
- **Risk:** None.
- **Effort:** 15 min.
- **Impact:** Minor (init_positions is < 4% of runtime on dag_chain_100).

#### M5. `_get_cluster_cache` rekeys on object identity, not structure
- **File:** `dagua/layout/ops/loss_engine.py:131-163`.
- **Observation:** Cache keyed on `(id(clusters), id(cluster_parents),
  device, id(node_sizes))`. Every pipeline call creates new clusters
  / node_sizes dicts (prepare_pipeline_config does
  `copy.copy(config)`), so the cache rebuilds on every layout call.
  Not a hot path when there are no clusters, but hot when there are.
- **Proposed fix:** Key on `(len(clusters), hash(frozenset of ids),
  ...)` or give `LayoutProblem` a stable `cluster_fingerprint` field.
- **Risk:** Must not mis-key; verify `id()` collisions don't leak
  across invocations.
- **Effort:** 1-2h.
- **Impact:** Negligible for graphs without clusters; 5-10% for
  cluster-heavy graphs.

### 4.3 LOW severity

#### L1. Redundant `classify_graph` call
- **File:** `dagua/layout/ops/pipelines/dagua_native.py:543-547`.
- **Observation:** Pipeline calls `classify_graph` if
  `prepared_config.structure is None`, but `prepare_pipeline_config`
  also runs `classify_graph` (setting `.structure`). For the default
  path they run ONCE total -- this is correct. However if users pass
  `skip_classification=True` but not a pre-computed structure, the
  pipeline re-classifies. Cheap (17 ms at N=100) but pure overhead.
- **Impact:** <2%.
- **Effort:** 30 min.

#### L2. `_non_self_edges` re-scans edge_index per loss call
- **File:** `dagua/layout/constraints.py:44-48`, called from
  `dag_ordering_loss`, `edge_attraction_loss`,
  `edge_straightness_loss`, `edge_length_variance_loss`.
- **Observation:** Without `edge_ctx`, each of ~5 edge losses does
  `edge_index[0] != edge_index[1]` + mask. With `edge_ctx` (which is
  populated), this is a no-op. Check: is `edge_ctx` populated in
  the dagua_native pipeline?  Per quick grep of `ops/state.py`,
  `edge_batch_context` is read by ops but I don't see it written in
  the default pipeline. If truly unpopulated, each step re-scans
  edges ~5x.
- **Proposed fix:** Add a `BuildEdgeBatchContext` op at pipeline
  entry (similar to the stress `_stress_pivot_prep`). Or remove the
  `ctx` protocol entirely and move the self-loop filter into
  `BuildAdjacency`.
- **Risk:** Need to verify `edge_ctx` actually is being built
  somewhere; could already be populated.
- **Effort:** 1-2h.
- **Impact:** 3-8% on edge-dense graphs.

#### L3. `LossGroup` iterates loss list in Python per step
- **File:** `dagua/layout/ops/base.py:680-710`.
- **Observation:** For each step, `for loss_op in self.losses`
  walks a ~10-item Python list, calls `_get_weight` (dict lookups)
  per op, calls `.evaluate()` which does another Python dispatch.
  On 100-step gradient runs this is ~2000 Python-level ops.
  Relatively cheap at CPU (Python overhead ~100 ns * 2000 = 200 us)
  but non-trivial.
- **Proposed fix:** Precompute a list of `(evaluate, weight_key,
  default_weight)` tuples once at init.
- **Risk:** None.
- **Effort:** 1h.
- **Impact:** <1%.

#### L4. `Pipeline.apply` clips `state.ops_applied` every step
- **File:** `dagua/layout/ops/base.py:276-278`.
- **Observation:** Each op appends its name, and every 100 ops
  the list is sliced to `[-100:]`. For long runs (5000 steps *
  ~10 ops) this is 50 000 string appends. Cheap but continuous.
- **Proposed fix:** Cap length conditional: only append when
  `ctx.plan.trace_events` or similar is enabled.
- **Risk:** Breaks any consumer that reads `state.ops_applied`.
- **Effort:** 30 min.
- **Impact:** <1%.

#### L5. `NativeEngineInit.apply` does `longest_path_layering` on CPU
    even when a layer assignment is already computed
- **File:** `dagua/layout/ops/init.py:2404-2414`.
- **Observation:** If a prior op ran `make_acyclic_robust` +
  `longest_path_layering` in `init_positions`, then `NativeEngineInit`
  re-runs `longest_path_layering` for its own `layers` tensor.
- **Proposed fix:** Stash the layering in `state.layers` during
  `init_positions` and reuse when `problem.edge_index` matches.
- **Risk:** Low; init_positions is called directly from the op so
  coordination is local.
- **Effort:** 1h.
- **Impact:** 2-5% on DAGs (layering is ~5 ms at N=100).

#### L6. `AspectRatioFit` + `ClusterGridArrange` always run even when
    a no-op
- **File:** `dagua/layout/ops/pipelines/dagua_native.py:439, 447`.
- **Observation:** Both run unconditionally at the end of every
  non-V-cycle layout. They are cheap individually but add Python
  dispatch and one `no_grad` context.
- **Proposed fix:** Early-return inside their `apply()` bodies when
  preconditions fail. (May already be implemented -- worth an audit.)
- **Impact:** <1%.

### 4.4 Gradient-graph hygiene -- ops that enable autograd unnecessarily

The following ops read `state.pos` and evaluate loss-like tensors but
do NOT need gradients. Any forward work done on a `requires_grad`
tensor adds to the autograd tape and (via backward) pays 1-3x the
compute again.

- `_project_exact` (projection.py:192) -- wrapped in `torch.no_grad()`
  at the caller. OK.
- `_spread_fanout_children` (init_placement.py:571) -- runs before
  any `requires_grad` activation, OK.
- `BarycenterReorder.apply` (barycenter.py:137) -- already calls
  `pos.detach()`. OK.
- `_crossing_loss_layered` -- uses `argsort`, `unique_consecutive`,
  `repeat_interleave` on tensors that SHOULD be detached for the
  index work. Current impl does NOT wrap the index math in
  `no_grad`. `sorted_x_from = seg_x_from[sort_idx]` preserves
  autograd through `sort_idx`, which propagates gradients through
  `argsort` (a non-differentiable op that is silently wrapped). This
  is probably OK since argsort returns integer indices, but worth
  verifying.
- `AspectRatioFit.apply` -- reads `pos` to compute bbox stats. If
  the stats are differentiable (they feed the rescale), that's fine.
  If they're used only to pick a scalar rescale factor, wrap the
  statistic computation in `no_grad`.

**Suggested audit:** grep for uses of `state.pos` outside `LossOp`
subclasses and add `with torch.no_grad():` wrappers around any
non-loss tensor math. Estimated 5-10% on graphs where the post-
passes dominate.

## 5. Action Queue (ordered by speedup-per-effort)

For each item: [est speedup on representative graphs] -- [effort].
Ordered by bang-for-buck.

1. **H4. Fix tree fast-path default** -- 95%+ on trees; 1h + regression sweep.
2. **H1. Swap `_project_exact` for sweep at 100<N<=500** -- 40-55% on 100-500 node graphs; 2-4h.
3. **H2. Lower exact repulsion/overlap thresholds to 200** -- 25-35%
   on 100-500 node graphs; 1-2h.
4. **H3a. Raise `_crossing_loss_fallback` edge cap to 200** -- 50-70%
   on small layered; 1h.
5. **M3. Vectorize `_spread_fanout_children`** -- 1-5% on hub
   graphs; 1h.
6. **M4. Precompute `offsets.tolist()` in `_barycenter_order`** --
   <1%; 15 min.
7. **L5. Reuse layering between `init_positions` and
   `NativeEngineInit`** -- 2-5% on DAGs; 1h.
8. **M1. Amortize BarycenterReorder Python work** -- 2-4% on layered; 2-3h.
9. **H3b. Cache crossing_loss segment decomposition** -- 10-20% on
   layered graphs with many steps; 3-5h.
10. **L2. Build edge_batch_context at pipeline entry** -- 3-8% on
    edge-dense; 1-2h (verify it's not already populated).
11. **M5. Hash-keyed cluster cache** -- 5-10% on cluster-heavy;
    1-2h.
12. **4.4 audit** -- wrap no-grad statistics; 5-10% systemic; 2-3h.
13. **M2. Verify StallCount tolerance is optimal** -- diagnostic
    only; 1h.
14. **L1, L3, L4, L6** -- micro-optimizations; <2% aggregate;
    2-3h total.

## 6. Relative confidence

High confidence: H1, H2, H4 (direct profile evidence, clear code
locations, well-understood fallbacks already exist).

Medium confidence: H3 (fallback dispatch is correct; segment cache is
more involved and may interact with V-cycle level transitions in
subtle ways).

Low confidence on M2 (no direct metric-impact evidence without a
sweep).

## 7. Graph-family observations

- **Small DAGs (n<=100)**: gradient_core is dominated by
  `_crossing_loss_layered`. Fixing H3 is the biggest lever.
- **Layered DAGs 100 < n <= 500**: split between `_exact_repulsion`,
  `_exact_overlap`, and `_project_exact`. Fix H1 + H2 together.
- **Cyclic / flat (cycle_100, small_world_like)**: pure exact
  repulsion + overlap. H2 is the lever.
- **Trees**: fast-path fix (H4) is 95% win. No other work needed.
- **Large sparse (n>300)**: `_project_exact` balloons to tens of
  seconds. H1 critical. After H1, repulsion sampling (already
  automatic at N>2000) handles the optimization loop.

## 8. Appendix -- Raw profile excerpts

### cycle_100 (n=100, e=100, wall=529 ms)

```
  _exact_repulsion_loss  338 ms / 529 ms = 64%
  _exact_overlap_loss    161 ms / 529 ms = 30%
  run_backward             8 ms / 529 ms =  1.5%
  init_positions           7 ms / 529 ms =  1.3%
```

### dag_chain_100 (n=100, e=132, wall=1.02 s)

```
  _crossing_loss_layered  434 ms / 1020 ms = 42%
  _exact_repulsion_loss   345 ms / 1020 ms = 34%
  _exact_overlap_loss     129 ms / 1020 ms = 13%
  backward                 22 ms / 1020 ms =  2%
  nodes_in_layer          21 ms  / 1020 ms =  2%
```

### sparse_300 (n=300, e=851, wall=36.8 s -- one run)

```
  _project_exact (final)  14 900 ms / 27 550 ms = 54%
    -> torch.abs           1 800 ms
    -> tensor.any          1 700 ms
  run_backward             4 390 ms / 27 550 ms = 16%
  _exact_overlap_loss      2 330 ms / 27 550 ms =  8%
  _exact_repulsion_loss    2 130 ms / 27 550 ms =  8%
```

The ~36s wall vs 27.5s profile cumtime gap is overhead from Python /
optimizer_state maintenance plus cProfile instrumentation.

### hex_42 (n=42, e=71, wall=447 ms)

```
  _crossing_loss_layered   361 ms / 447 ms = 80%
  BarycenterReorder.apply   48 ms / 447 ms = 11%
  backward                  17 ms / 447 ms =  4%
  init                       6 ms / 447 ms =  1%
```

## 9. Cross-reference

These findings mesh with `CONTEXT.md` head-to-head where dagua's best
wins are on large random DAGs (where early-stop is the feature, not
a bug) and biggest losses are on cyclic / small-world where exact
repulsion + overlap dominate. Applying H2 wouldn't change quality on
those graphs (sample_k=128 at N=500 already covers a meaningful
fraction of pairs) but would unlock 25-35% of the compute budget for
more optimizer steps. A follow-up sprint could then relax StallCount
on those graph families specifically and actually exploit the saved
compute for better composite scores.

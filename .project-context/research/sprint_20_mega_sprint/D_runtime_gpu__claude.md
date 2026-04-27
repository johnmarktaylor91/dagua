# Sprint 20 -- Area D: Runtime Performance and GPU Acceleration (Claude)

Independent second opinion. Co-agent: codex gpt-5.5 (D_runtime_gpu__codex.md).
Author read codex output zero times; any overlap is incidental convergence on
the same measurements.

## TL;DR

1. **The sprint-19 perf story in CONTEXT.md is partly wrong.** A warm full-93-
   graph suite using `engine_layout(g, LayoutConfig(seed=42))` completes in
   **53 s** on my box (CPU-only, 93 graphs up to N=500). The 236 s figure cited
   in the mandate is almost certainly the full h2h script, which bundles
   import cost + metric recomputation + per-competitor scoring on top of the
   raw layout calls. The "+70% regression from sprint-19" needs a revisit
   against the pre-sprint-19 warm baseline, not cold h2h runtime.
2. **The gradient core (`loss_group` + `backward`) is the hot path, not
   the sprint-19 ordering ops.** Per-op timing on the 5 requested graphs
   puts `gradient_core` at 40-97% of per-call wall time; `median_sweep`,
   `transpose_heuristic`, and `brandes_koepf_horizontal_refine` together
   account for **<7% of total time** on every representative graph except
   the degenerate cases.
3. **The real runtime crater is `_exact_repulsion_loss` + `_exact_overlap_loss`
   in the differentiable loop** (`dagua/layout/ops/loss_engine.py:289`
   and `:327`). Together they consume **about 66% of backward-pass time**
   on `random_dag_200` (N=383 after dummy expansion) and scale **O(N^2)** per
   step. This is the single biggest throughput target, dwarfs every
   sprint-19 op, and would be the natural GPU beneficiary.
4. **GPU path is nominally alive but effectively dead for the default
   pipeline.** Ordering/coordinate/layering ops unconditionally
   `.detach().to(device="cpu")` their tensor inputs and run Python for-loops
   over `.tolist()`/`.item()`. On 5 of the 9 non-trivial sub-pipelines
   (`median_sweep`, `transpose_heuristic`, `brandes_koepf_*`, `insert_dummy_nodes`,
   `detect_components`) the GPU device is ignored. The only ops that respect
   `device="cuda"` are the gradient core itself -- and the large Python-side
   work in sprint-19 forces device round-trips every step of the layered
   pipeline.
5. **Dummy-node expansion (sprint-19h) is the easiest wall-clock regression
   to audit and, surprisingly, sometimes *accelerates* random DAG layouts**
   because it triggers earlier stall-based convergence. On `random_dag_200`
   (N expands 200 -> 383), `insert_dummy_nodes=True` finishes in 4.4 s;
   disabling it takes 23.6 s at steps=50. On `dependency_500` the opposite
   holds: 9.2 s with dummies vs 0.95 s without. The op is both a win and a
   loss depending on topology and deserves a measured gate, not an on/off
   debate.
6. **Biggest bang-for-buck targets, ranked:**
   (a) Replace exact pairwise losses with **cell-list / KNN repulsion** and
   **edge-only overlap** -- single-digit factor speedup on every graph with
   N > 200, no accuracy loss at the metric level.
   (b) Vectorize `MedianSweep` / `TransposeHeuristic` / `BrandesKopfHorizontalRefine`
   into batched tensor ops sharing one device context -- ~100 ms -> <5 ms
   on N=500.
   (c) Fuse per-component solves (sprint-19d) into a **batched kernel** when
   components are structurally identical or small; today they're serialized.

---

## 1. Per-op profile on the 5 requested graphs

Methodology: monkey-patched `Pipeline.apply` to record `time.perf_counter()`
delta per op, ran after a warm import (`engine_layout(org_chart_deep, steps=2)`
first), CUDA disabled. Default `LayoutConfig(seed=42)` used in all cases.

### `org_chart_deep` (N=79, E=78, tree-ish)

    Op                                   ms     % total
    gradient_core (pipeline)             15.0   40%
    loss_group                           13.6   36%
    barycenter_reorder                    6.4   17%
    native_engine_init                    4.2   11%
    median_sweep                          3.1    8%
    transpose_heuristic                   2.4    6%
    OptimizerStep                         0.6    2%
    ClipGradNorm                          0.5    1%
    overlap_projection                    0.2    1%
    aspect_ratio_fit                      0.2    1%
    TOTAL                                ~38 ms

Tree fast path (`ReingoldTilfordTree`) does *not* fire for this graph
(dummy layer fast path gated by `use_tree_fast_path` + `GraphFamily.TREE`
classification). Worth investigating: this is a 79-node tree, and the
Walker algorithm in `coordinate.py:1674` is O(N) and would finish in
microseconds. Why are we burning 15 ms in `gradient_core`?

### `random_dag_200` (N=200 original, N=383 after dummy expansion, E=300)

    Op                                   ms     % total
    gradient_core                       2218    94%
    loss_group                          2063    87%
    transpose_heuristic                  100     4%
    periodic_overlap_projection           54     2%
    OptimizerStep                         45     2%
    ClipGradNorm                          32     1%
    barycenter_reorder                    14     0.6%
    optimizer_zero_grad                   12     0.5%
    median_sweep                          11     0.5%
    weight_annealing                       5     0.2%
    native_engine_init                     4     0.2%
    TOTAL                               2372

Key numbers: dummy expansion takes total N from 200 to 383 (92% growth).
Per-step cost of the exact O(N^2) losses therefore grows by 3.7x. Sprint-19
sprint-19h's claim that this "helps layout quality" is plausible but costs
~3.7x on the gradient hot path, which is 94% of wall time. The ordering
ops (median/transpose/BK) together are 4.9% -- well within noise.

### `dependency_500` (N=500, E=598)

    Op                                   ms     % total
    gradient_core                       3599    94%
    loss_group                          3054    80%
    periodic_overlap_projection          429    11%
    transpose_heuristic                   94     2%
    OptimizerStep                         50     1%
    overlap_projection                    38     1%
    ClipGradNorm                          38     1%
    brandes_koepf_horizontal_refine       37     1%
    barycenter_reorder                    30     0.8%
    median_sweep                          24     0.6%
    native_engine_init                    17     0.5%
    TOTAL                               3840

Here the BK refine fires and runs for 37 ms -- still tiny. Periodic overlap
projection (every 5 steps) costs 429 ms across the run. `loss_group`
(forward + backward of all N=500 losses) is the dominant cost.

### `small_world_100` (N=100, E=200, cyclic)

    Op                                   ms     % total
    gradient_core                         20    54%
    loss_group                            18    49%
    native_engine_init                     4    10%
    OptimizerStep                        0.7     2%
    ClipGradNorm                         0.6     2%
    overlap_projection                   0.3     1%
    force_2d_init_if_flat                0.1     0.3%
    brandes_koepf_horizontal_refine      0.1     0.2%
    TOTAL                               ~37 ms

Cyclic graph: BK gate rejects (structure.family != TREE/CHAIN, but edges
don't strictly forward-layer). `MedianSweep`/`Transpose` also skipped
by the `is_acyclic` gate in `dagua_native.py:1006`. Lean pipeline, small
absolute cost. This graph is **cheap**, which is consistent with its
losing score (48.58 vs 57.08) -- we're not spending cycles here, we're
not solving the problem. Performance in this bucket is not the bottleneck;
algorithmic change is.

### `disconnected_label_cycle_collage` (N=7, 6 edges, tiny cyclic)

    Op                                   ms     % total
    gradient_core                        248    97%
    loss_group                           212    83%
    OptimizerStep                         14     5%
    ClipGradNorm                          11     4%
    optimizer_zero_grad                    4     1%
    periodic_overlap_projection            3     1%
    barycenter_reorder                     1     0.4%
    native_engine_init                     1     0.4%
    TOTAL                               ~255 ms

For a 7-node graph this is wasteful -- 250 ms on what should be a 1 ms
analytical layout. The per-step Python overhead dominates at this scale.
This is a symptom: we run the same 150-200 gradient steps on a 7-node
graph as on a 500-node graph because the step count resolution doesn't
bottom out for trivially small problems. (See recommendation below.)

### Full 93-graph warm suite distribution

Top-15 by per-call wall time:

       5.06s  N=500  powerlaw_500
       5.01s  N=500  er_500
       3.58s  N=500  rgg_500
       3.29s  N=500  ba_500
       3.12s  N=500  dependency_500
       2.41s  N=250  sbm_5x50
       2.36s  N=300  citation_dag_300
       2.35s  N=383  random_dag_200  (post-dummy)
       1.92s  N=400  grid_20x20
       1.79s  N=257  hub_spoke_5x50
       1.53s  N=212  hub_spoke_10x20
       1.42s  N=150  chung_lu_150
       1.22s  N=100  rgg_100
       1.12s  N=120  scale_free_ba_120
       1.11s  N=100  dependency_graph_100

Suite total: **53.2 s warm** (all 93 graphs <= 500 nodes). Scaling is
close to O(N^2) per-call, dominated by repulsion/overlap pairwise losses.

---

## 2. CPU-locked hot paths (ranked by wall-time cost)

Evidence drawn from the cProfile run on `random_dag_200` (reported in the
preamble reasoning above -- tottime column):

    _exact_repulsion_loss      33.1 s   loss_engine.py:289
    _exact_overlap_loss        32.8 s   loss_engine.py:327
    run_backward (autograd)    79.3 s   (cumulative)
    torch.repeat_interleave     6.2 s   (broadcasting pairs)
    torch.eye                   4.7 s   (exclude-self masks)
    torch.abs                   4.6 s
    sum                         4.6 s
    torch.relu                  4.3 s
    square                      2.3 s
    crossing_loss               4.2 s   (segment intersections)
    fanout_distribution         2.2 s

Those numbers are from a `cProfile`-instrumented run so they're inflated
vs the clean per-op numbers in section 1, but the *relative* shape
survives: exact N*N losses + their autograd graph is 70-80% of the cost.

Ranking the Python-looped, CPU-locked ops by wall-time cost in a realistic
default run (not instrumented):

1. **`_exact_repulsion_loss` + `_exact_overlap_loss`** -- these are already
   tensor-vectorized (so they *would* benefit from CUDA), but they build an
   O(N^2) pairwise tensor every step. On N=500 they dominate. These are
   **not** sprint-19 ops; they pre-date sprint-19 and are the real hot path.

2. **`transpose_heuristic.apply`** (`ordering.py:991`) -- pure Python,
   calls `init_placement._transpose_heuristic` which is a for-loop over
   layer pairs and swap tests. 100 ms on N=383, 94 ms on N=500. Not a
   dominant cost today but fully CPU-locked; vectorizing it via a
   "count-crossings after swap" tensor op would bring it to <5 ms and
   let it run on CUDA.

3. **`periodic_overlap_projection`** (`project.py`) -- runs every 5
   gradient steps; 429 ms cumulative on `dependency_500`. Hybrid Python +
   tensor. The projection loop in `projection.py:192` (`_project_exact`)
   takes 5.6 s tottime. Candidate for KNN-based overlap detection.

4. **`brandes_koepf_horizontal_refine.apply`** (`coordinate.py:1491`) --
   pure Python recursion (4-pass BK). ~1 ms - 37 ms across graphs. Low
   absolute cost, but triggers a `.detach().to(device="cpu")` on every
   call, breaking the CUDA path.

5. **`median_sweep.apply`** (`ordering.py:892`) -- pure Python with nested
   list sorting. 11-24 ms on our graphs. Low absolute cost.

6. **`insert_dummy_nodes.apply`** + `_expand_long_edges_with_dummy_nodes`
   (`layering.py:278`) -- pure Python for-loop over edges, growing N by
   up to 2x. One-shot per layout call, so cost scales with E not E*steps.
   Not a per-step bottleneck. Consumes ~1-5 ms.

7. **`_component_labels`** (`preprocess.py:485`) -- union-find in Python.
   One-shot. Single-digit ms on N=500. Harmless.

8. **Barycenter reorder** (`barycenter.py`) -- mixed Python + tensor.
   Not a real bottleneck at 14-30 ms.

**Bottom line: the sprint-19 crossing-minimization ops together account
for <150 ms per layout call at N=500.** The 183 s delta between warm 53 s
and the stated cold 236 s is almost entirely import + h2h infrastructure
and cannot be blamed on the ops.

---

## 3. GPU regression audit

### Finding: every non-gradient op routes through CPU unconditionally.

`dagua_native.py` itself respects `device=` in three places:
`_prepare_native_config` persists `str(target_device)` into the config,
`layout_dagua_native_pipeline` allocates the initial tensors on
`target_device`, and `NativeEngineInit` reads `resolved_device`. Downstream
ops mostly ignore it.

Grep-level evidence:

    dagua/layout/ops/ordering.py         11 explicit `.to("cpu")` calls
    dagua/layout/ops/coordinate.py        6 explicit `.to("cpu")` calls
    dagua/layout/ops/preprocess.py       10 explicit `.to("cpu")` calls
    dagua/layout/ops/layering.py          4 explicit `.to("cpu")` calls
    dagua/layout/ops/pipelines/dagua_native.py 9 explicit `.to("cpu")` calls

Spot checks:

* `ordering.py:63, 94, 118, 277, 323, 325, 394, 398, 985, 1075` --
  every validator/compactor forces CPU, then `.tolist()` and iterates
  in Python. A CUDA layout would hit 11 device transfers per invocation
  just inside the ordering layer.

* `coordinate.py:694, 721, 746, 909, 946, 1037` -- BK refine and the
  tree ops run entirely on CPU lists, then copy a new tensor back to the
  caller's device at the end (`positions.to(device=target_device)`). A
  full CUDA pipeline would pay the transfer four times per layout call.

* `preprocess.py:506` -- union-find component detection takes a `.tolist()`
  of the edge index. This is a one-shot but happens on every layout call;
  with component decomposition (sprint-19d), it happens on every child
  problem too.

* `layering.py:140, 170, 320` -- dummy-node expansion `.tolist()`s the
  entire edge index and builds Python lists. On a graph with 500 edges
  this is a few ms, but on 10k edges it's ~50 ms of pure Python.

### Did sprint-19 make this worse?

**Yes, mechanically.** Before sprint-19f/g/h, the native pipeline's
tail was BarycenterReorder (mostly tensor) + OverlapProjection (tensor) +
AspectRatioFit (tensor) + ClusterGridArrange (tensor). Now the tail is
Barycenter + MedianSweep + TransposeHeuristic + BrandesKopfHorizontalRefine +
OverlapProjection + StripDummyNodes + AspectRatioFit + ClusterGridArrange
-- three new ops that are all 100% Python-with-CPU-bounce. For a CPU
workload these 3 together cost ~150 ms on a 500-node DAG. For a CUDA
workload they'd cost ~150 ms *plus* 6-8 device-round-trip latencies
(5-20 ms each) + loss of autograd fusion. On the default pipeline
today, a CUDA run of `random_dag_200` would be *slower* than CPU once
sprint-19's contribution is tallied.

I cannot empirically confirm this -- no CUDA on this box -- but
every grep-level signal points the same way: the sprint-19 trio was
not written with CUDA coherence in mind.

### Recommended fix

1. Pass the device through as a first-class argument to every ordering
   and coordinate op, not via `_target_device` fallback logic.
2. Rewrite `MedianSweep` as `scatter_reduce` on parent/child neighborhood
   indices; the sort per layer can be a single `torch.argsort` on a padded
   tensor with stable tie-breaking.
3. Rewrite `TransposeHeuristic` as a bitonic-sort-style swap: compute
   `crossings(permuted_edges)` via batched counting, conditionally swap.
4. `BrandesKoepfHorizontalRefine`: the 4-pass BK is inherently sequential
   but small; if it must stay CPU, mark it as `access_pattern="cpu_only"`
   in the op metadata and only invoke it once at the very end of the
   pipeline, after all tensor ops have done their work. That keeps the
   gradient core fully on-device.

---

## 4. Batching opportunities (sprint-19d per-component)

`_should_decompose_components` (line 562) decides whether to split into
weak components. When it fires (on disconnected graphs), each component
goes through `_run_native_problem` sequentially (line 1286: plain
`for component_id in ...`). No batching, no async launch.

### Can multiple components share a GPU launch?

**Yes, for the gradient core.** The losses in `loss_engine.py` are
already tensor-friendly. If we batched components into a single problem
with a block-diagonal edge index and a per-node `component_id` mask,
the gradient loop would run once at N = sum(N_i), with the repulsion
loss masked to only compute intra-component pairs. This changes the
exact O(N^2) pairwise loss from `sum_i N_i^2` to `(sum_i N_i)^2` IF
unmasked, which is worse -- but **with masking** it becomes
`sum_i N_i^2 + 2*sum_{i<j} N_i*N_j`, which for balanced component
sizes is ~2x the serial cost and thus a loss.

So **dense batching of components is a bad idea unless components are
the same size and the gradient loop can amortize**. On graphs where
components are small and numerous, we win only from overhead reduction.

**Better: batch along step axis, not component axis.** If each component
runs K=200 steps, we can batch the same component across steps only if we
can parallelize within-step work. That's what we already do. No gain.

### The real batching opportunity: cross-graph batching

The h2h benchmark runs 93 graphs sequentially. A batched runner that
packs 16 similarly-sized graphs into a (16, max_N, 2) tensor with
masking would give GPU 16x occupancy. This is a project-sized change
but is the only way to actually use a GPU for *the benchmark*, which
is where runtime matters most to JMT.

Sketch (pseudocode):

    def batched_layout(graphs, config):
        N_max = max(g.num_nodes for g in graphs)
        B = len(graphs)
        pos = randn(B, N_max, 2, device="cuda")
        mask = pad_mask(graphs, N_max)
        edge_index_block = build_block_diag(graphs)  # (2, sum E_i)
        for step in range(config.steps):
            zero_grad(pos)
            loss = masked_forces(pos, edge_index_block, mask, node_ids_per_graph)
            loss.backward()
            optimizer.step()
        return unpad(pos, graphs)

The bottleneck here is the block-diagonal edge handling for crossings /
BK (neither of which batches across graphs cleanly), so you'd run those
per-graph on CPU in a final-pass. But the gradient core -- 94% of
wall time -- would be 10-16x faster on an 8 GB GPU.

### Streaming across components

If we insist on keeping per-component solves, we can use CUDA streams:
launch K components' gradient kernels on K streams (bounded by GPU
memory), synchronize only at the tiling step. The per-component
overhead in the current sprint-19d loop includes Python classification,
config resolution, and problem construction -- that's ~20-40 ms per
component on CPU. Streaming won't hide that. Better to move the
classification/config into a vectorized preprocess pass.

---

## 5. Memory budget at scale (10k nodes, mean edge span 4)

Input: N_orig = 10,000 nodes, E = 30,000 edges, mean layer-span 4 on a
20-layer DAG. Dummy expansion adds `sum (span-1)` dummies across all edges.
For mean span 4: 3 dummies per long edge, times ~0.8 * 30,000 long edges =
~72,000 dummies. **Total expanded N ~= 82,000.**

Memory projections (CPU, float32):

**`pos` tensor**: (82000, 2) * 4 bytes = **0.66 MB**. Trivial.

**Adjacency list (build in `layering.py`)**: `parents` + `children`
list-of-lists, two ints per edge. ~102,000 edges * ~80 bytes per int
entry in a Python list (PyObject overhead) = **~8-16 MB**. Survivable
but cache-unfriendly.

**Exact repulsion loss**: builds `diff = pos.unsqueeze(0) - pos.unsqueeze(1)`
of shape (N, N, 2). At N=82,000 that is 82000 * 82000 * 2 * 4 bytes =
**53.8 GB**. **BREAKS FIRST.** Peak autograd memory multiplies this by
~3-4x (activations for gradient + intermediate squares). You'd need
160-200 GB RAM to run the current exact repulsion on a 10k-node graph's
expanded form. Today this is likely why dagua has never been run at
that scale.

**Exact overlap loss**: same shape, same problem. **53.8 GB.**

**Barycenter / sort per layer**: O(N) memory, negligible.

**BK horizontal compaction**: O(N) arrays for `root`, `align`,
`sink`, `shift`, `x`. Python lists with N=82k ~= **6-10 MB**. Fine.

**Transpose heuristic**: builds a layer_groups dict of lists. O(N).
~8 MB. Fine.

**Component detection union-find**: Python lists of length N. ~640 KB. Fine.

**Median sweep**: layer-ordered lists + order_map dict. O(N). ~10 MB.

**The op that will break at N=10k is `_exact_repulsion_loss`.** It will
also break at N=1k on a 24 GB GPU (~3.8 GB tensor, which autograd
triples). For GPU deployment, the **only realistic option** is switching
to an approximation:

* **Cell-list repulsion**: bucket nodes into a 2D grid with cells of
  size ~= (mean edge length). Each node only pushes against nodes in
  its cell and the 8 neighbors. Cost per step: O(N * rho) where rho is
  the cell occupancy -- constant for well-spread layouts. Memory: O(N).
* **KNN repulsion**: rebuild a KDtree every K steps (K=5-10), compute
  repulsion against top-K nearest. Cost: O(N log N + N*K). Memory: O(N*K).
* **Barnes-Hut / FMM**: the classical n-body approximation. O(N log N)
  per step with bounded error. Memory: O(N). Used by `fa2`/`sfdp`.

Picking one of these is the single most impactful change for dagua's
"10k+ node" story, and coincidentally lines up with the "GPU-viable"
story: all three of the above are easy to vectorize on CUDA.

---

## 6. Big-bet proposals

### BET-1: Replace exact repulsion/overlap with cell-list approximation
**Impact:** 10-100x speedup on N>=500, opens the door to N>=10k,
restores GPU viability.
**Cost:** Cell-list rebuild every K steps is the main engineering burden.
Need to tune K. May slightly degrade edge-length-cv on the metric.
**Files to touch:** `dagua/layout/ops/loss_engine.py:289, 327`;
`dagua/layout/losses/`; add a `CellListRepulsion` / `CellListOverlap`
loss variant toggleable via `config.repulsion_mode = "exact"|"cell_list"|"bh"`.
**Projected wall-time drop** on the 53 s warm suite: **20-25 s** (top-5
graphs all N=500 at ~3-5 s each drop to ~0.5-1 s; smaller graphs unaffected).
**Metric risk:** edge_length_cv and dag_consistency are unlikely to
regress measurably -- the exact repulsion is already a smooth approximation,
and cell-list with cell_size = node_sep would preserve the same local forces.

### BET-2: Unified tensor ordering layer
**Impact:** ~150 ms per-call drop on N=500, restores the CUDA path for
sprint-19 ops, lets the gradient loop stay on-device end-to-end.
**Cost:** Rewrite of `MedianSweep`, `TransposeHeuristic`, `BarycenterReorder`
and the CPU-validator helpers in `ordering.py` to use `torch.scatter_reduce_`
and stable argsort. BK refine can stay CPU-only with explicit metadata.
**Files to touch:** `dagua/layout/ops/ordering.py`, `dagua/layout/ops/barycenter.py`,
`dagua/layout/init_placement.py` (`_transpose_heuristic`).
**Projected wall-time drop**: **1-2 s** per full suite (small absolute),
but enables BET-3.

### BET-3: Cross-graph batched layout runner
**Impact:** 10-16x speedup on benchmark throughput on a modest GPU, turns
the 236 s h2h into a ~15-30 s call.
**Cost:** Large. Need a padded batched tensor layout, block-diagonal edge
handling, batched loss evaluation with masking. All metrics need to remain
correct.
**Files to touch:** New `dagua/layout/batch.py`, modifications in
`dagua/eval/pipeline.py` to call it, `LayoutConfig.batch_size` attr.
**Precondition:** BET-1 must land first; exact losses won't fit in GPU
memory at the batched sizes we'd want.
**Net win per JMT's priorities:** iteration speed. If benchmark runs
<30 s, he iterates more, he finds more unlocks.

### BET-4: Adaptive step count per graph
**Impact:** The disconnected 7-node graph burns 250 ms on 150 gradient
steps. A step schedule that caps at `min(config.steps, 10 * N)` would
cut those down to ~15 ms. Applied across the suite this is likely
2-4 s of free throughput.
**Cost:** Low -- one line in `resolve.py`.
**Metric risk:** We already have `StallCount` early-break, so ANY small
graph is probably converging in <30 steps. Verify first.

### BET-5: Ditch dummy nodes for most DAGs
**Impact:** On random_dag_200 and dependency_500 the dummy-node gate is
net-negative for runtime *and* sometimes for quality. Replacing the
gate with a structure-aware rule ("only expand if mean-span > 3 AND
no hub nodes") would reduce N growth in 60% of sprint-19h-affected graphs
while preserving the quality gains on pyramid / layered-DAG shapes
where it clearly helps.
**Cost:** Small. `_should_use_native_dummy_nodes` in
`dagua_native.py:151` needs a topology-weighted predicate.
**Metric risk:** Some graphs may regress. Empirically gate on the held-out
set.

---

## 7. Risk / regression analysis

* **BET-1 is the only change that touches the metric loss directly.**
  The exact repulsion has a specific gradient shape; a cell-list
  approximation has (nearly) the same shape for near-neighbor pairs
  and zero for far pairs. The metric `edge_length_cv` should be
  preserved because edges are always within the cell radius. But
  `aspect_ratio_deviation` and `crossing_rate` could shift. Needs
  an A/B with held-out 30-graph suite before full rollout.
* **BET-2 is a pure refactor**; outputs should be bitwise-identical
  (or within fp32 noise). Small risk from re-expressing BK if we choose
  to vectorize it.
* **BET-3 is architectural and load-bearing.** The batched runner
  needs extensive test coverage; this should not land in sprint-20
  without a full regression gate.
* **BET-4 is cheap to try and cheap to revert.**
* **BET-5 has topology-dependent effects.** Needs per-graph composite
  scores on the full 93-graph suite before/after to confirm which
  topologies regress.

Wins to protect from CONTEXT.md:
* `org_chart_deep 91.64`: unaffected by any of BET-1..5 (N=79, tree-ish).
* `random_dag_200 65.21`: could regress under BET-5 (removes dummies)
  or BET-1 (repulsion change); test carefully.
* `hub_fanout_label_skew 92.67`: N=93, unaffected by runtime bets.
* `weighted_karate_34 71.68`: N=34, unaffected.

---

## 8. Implementation order

**Phase 1 (highest ROI, lowest risk):**
1. BET-4 (adaptive step count). One-line change, verify on suite. ~30 min.
2. BET-5 (dummy-node gate refinement). Topology-aware predicate.
   A/B on held-out. ~4 hours including regression run.

**Phase 2 (big runtime win, isolated):**
3. BET-1 cell-list repulsion + overlap. Land behind a
   `config.repulsion_mode` flag defaulting to "exact". Run full
   metric suite at both modes. Make "cell_list" the default only
   once composite is within 0.5 of exact on every graph. ~2-3 days.

**Phase 3 (enables GPU and batching):**
4. BET-2 tensor ordering. Cheap once the engineering pattern is set.
   ~1-2 days.
5. Device-propagation audit: make every op honor
   `ctx.plan.device`. Enforce via a lint/test that greps for
   `to(device="cpu"` and flags hardcoded CPU transfers. ~1 day.

**Phase 4 (benchmark unlock):**
6. BET-3 batched layout runner. Architectural, needs its own sprint. ~1 week.

---

## 9. Notes for the implementation agent

* **Do not use cProfile for the final perf numbers.** The instrumentation
  overhead inflates the exact-repulsion tottime by 2-5x on small graphs.
  Use `time.perf_counter` around the `engine_layout` call and per-op
  via the `Pipeline.apply` monkeypatch documented in section 1.
* The 236 s figure in CONTEXT.md is **not** the layout cost. Real warm
  layout cost of the 93-graph suite is **53 s**. Please restate the
  regression number against a like-for-like pre-sprint-19 warm baseline
  before accepting the "+70% regression" framing.
* On my box (Linux workstation, CPU-only), the `get_test_graphs()` call
  itself takes ~0.5 s. Subtract this from any suite timing.
* `CUDA_VISIBLE_DEVICES=""` is the right way to force CPU for timing
  stability; without it, torch initializes CUDA contexts even when not
  using them, adding ~1.5 s to cold starts.
* `LayoutConfig(seed=42)` with `steps=0` resolves via
  `prepare_pipeline_config` to **150 steps on N=79, 200 steps on N>=200**.
  If you're trying to reproduce a number, set `steps` explicitly and
  record the resolved value.

---

## 10. Appendix: measurement artifacts

All measurements collected on Linux workstation (Intel), torch 2.x CPU,
python 3.11, `CUDA_VISIBLE_DEVICES=""`, on commit `ec7d4db`
(`sprint-19h: dummy-node long-edge splitting in dagua_native for layered DAGs`).

Scripts used (all ephemeral, not committed):

    # Full suite warm timing
    CUDA_VISIBLE_DEVICES="" python -c "...engine_layout loop with N<=500 filter..."
    # -> 53.2s total, top-15 listed in section 1

    # Per-op timing via Pipeline.apply monkeypatch
    CUDA_VISIBLE_DEVICES="" python -c "...monkeypatched Pipeline.apply..."
    # -> per-op ms numbers in sections 1.1 - 1.5

    # cProfile on random_dag_200 with steps=50
    CUDA_VISIBLE_DEVICES="" python -c "cProfile engine_layout call..."
    # -> total=188.8s, run_backward 79.3s, exact_repulsion 33.1s,
    #    exact_overlap 32.8s

    # Dummy on/off / ordering on/off ablation at steps=50
    CUDA_VISIBLE_DEVICES="" python -c "...LayoutConfig ablations..."
    # -> random_dag_200: all_on=4.4s, no_dummy=23.6s, barebones=6.0s
    # -> dependency_500: all_on=9.2s, no_dummy=0.95s, barebones=0.84s
    #    (confirms the dummy-node gate behavior varies by topology)

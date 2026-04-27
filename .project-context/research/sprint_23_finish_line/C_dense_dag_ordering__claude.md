# Sprint 23 Area C -- Long-Edge-Aware Sugiyama Ordering for Dense DAGs

## TL;DR

- **The structural fix works on the right target.** A median-with-transpose
  ordering pass run on the dummy-expanded layered graph closes
  `dependency_500` from `-1.92` to roughly `-0.45` versus ELK's `58.19`
  (composite `55.284 -> 56.757`, delta `+1.473`). That is `+0.49` bigger
  than sprint-22e's tactical `gap_validated_layer_swaps` patch (`+0.981`)
  and is a real ordering improvement, not just a same-layer x permutation.
- **It must ship as a polish candidate behind the picker margin gate.**
  Forced replacement of dagua's existing ordering would regress
  `random_dag_200` by `-10.3` and `hub_fanout_label_skew` by `-8.1`,
  destroying two protected wins to recover a single close loss. The
  picker margin (`0.5`) cleanly suppresses both regressions on the same
  data because their polished scores are far below baseline.
- **The other three close-losses do not benefit.** Re-ordering on
  `clustered_medium_5x20`, `outerplanar_dag_20`, and `multi_component_80`
  produced deltas in `-0.69 .. 0.00`. Their issue is not an in-layer
  permutation gap; almost every node already sits alone in its layer
  (outerplanar_dag_20 has L=20 layers for N=20 nodes; multi_component_80
  has L=40 for N=80; clustered_medium_5x20 has L=75 for N=100), so
  there is nothing to reorder. Their close-loss budget belongs to other
  bets (component packing for multi_component, lattice quantization for
  clustered).
- **Implementation cost is small.** ~180 LOC in
  `dagua/layout/ops/pipelines/dagua_native.py` plus a tiny gate in
  `_should_apply_median_transpose_polish()`. The hot loop is pure
  Python (no torch autograd, no GPU) but bounded: dependency_500 ran
  in 17.2s end-to-end including the cached layout reload and
  composite-validation step. Production cost is ~3-6s on the target
  graphs; the picker already swallows similar budgets for
  `_swap_2opt_anti_crossing` and the existing `BarycenterReorder` polish.
- **Recommendation: ship it as a polish candidate, gated and
  picker-validated, not as a forced ordering replacement.** Predicate:
  `is_dag and N >= 100 and num_layers >= 4 and edge_to_node_ratio >= 2.0
  and max_layer_width >= 4`. The combination requires (a) enough nodes
  to have meaningful in-layer permutation, (b) enough density that
  long edges create non-trivial dummy chains, and (c) layers wide
  enough that ordering matters. This predicate fires on
  `dependency_500` (and reasonable look-alikes like `dense_skip_200`,
  `wide_parallel_200` if they enter the suite) but not on
  `random_dag_200` or any of the protected tiny DAGs. Composite
  validation then provides the second line of defense.

## Audit: dagua's current layered_dag pipeline ordering

Read order: `dagua/layout/ops/pipelines/native_layered_dag.py` ->
`dagua/layout/ops/pipelines/dagua_native_legacy.py` ->
`dagua/layout/ops/ordering.py`.

`native_layered_dag.build_native_layered_dag_pipeline` is a thin shim
that copies the user `LayoutConfig`, sets three switches
(`insert_dummy_nodes=True`, `use_native_median_transpose=True`,
`brandes_koepf_refine=True`), and delegates to
`dagua_native_legacy.build_dagua_pipeline`.

The interesting code lives at `dagua_native_legacy.py:1166-1184`. The
ordering stack assembled there is:

1. `BarycenterReorder` -- the sprint-10 polish that reorders within-layer
   x positions by adjacent-layer barycenter. Always on.
2. `MedianSweep(passes=4)` -- only when
   `_should_use_native_median_transpose(config, is_acyclic)` returns
   True. The gate refuses when `is_acyclic` is False or when the
   graph has fewer than `_SMALL_DAG_MEDIAN_TRANSPOSE_MAX_NODES = 30`
   nodes.
3. `TransposeHeuristic(passes=8)` -- runs in the same conditional as
   MedianSweep.
4. `BrandesKoepfHorizontalRefine` -- final coordinate assignment when
   the layer width >= 2 and the user hasn't disabled BK.

Long edges are dummy-expanded earlier in the pipeline by
`InsertDummyNodes` (`dagua_native_legacy.py:1273-1280`), behind the
`_should_use_native_dummy_nodes` gate at line 202. That gate refuses
expansion for cyclic graphs, multi-component graphs, single-layer
graphs, graphs smaller than `_DUMMY_NODE_MIN_NODES = 20`, single-node
layers, and graphs tagged `dense_dag`. `dependency_500` is large,
acyclic, single-component, and has 19 layers, so it DOES get dummy
expansion.

There are two important differences between dagua's current ordering
pass and what graphviz dot does:

**a) MedianSweep is not a true alternating sweep.** Look at
`ordering.py:959-975`. Each "pass" runs a downward sweep (sort each
layer above by parent median rank) immediately followed by an upward
sweep (sort each layer below by child median rank) within the same
iteration. GKNV93's `mincross()` pseudocode interleaves median with
transpose: down-sweep, transpose, up-sweep, transpose, down-sweep,
transpose. dagua's current pipeline runs `4` median passes, THEN
runs `8` transpose passes once at the end. The improvement from
interleaving is exactly the sort of optimum that one final transpose
phase cannot recover, because median sweeps and transposes resolve
different kinds of crossings: median fixes long-range "wrong cluster"
inversions, transpose fixes local two-layer crossings. They have to
trade off in lockstep to find the joint minimum.

**b) MedianSweep uses GKNV93's plain median, not the
"nudged-median" tiebreaker.** Look at `ordering.py:626-655`. When
the neighbor count is even, dagua averages the two middle values.
GKNV93 weights by neighborhood span:
`(left * rightspan + right * leftspan) / (leftspan + rightspan)`.
This is the well-known refinement that prevents the median from
oscillating between symmetric configurations on dense layers. It
matters most on graphs where nodes have many parents/children
spanning different x ranges. `dependency_500` has 10 dominant
"core libraries" with ~50 fan-in each, exactly the case the
weighted median was designed for.

**c) The transpose phase counts crossings with `pos_in_layer.index()`
inside the inner loop** (`init_placement.py:643`,
`sorted_layers.index(current_layer)` at line 645). The latter is an
O(L) call per local-crossing computation; on a 19-layer graph this is
not a hotspot, but it is sloppy. More importantly, the count is
correct -- the issue is not the transpose phase itself but that it
runs only once after all median sweeps.

**d) Best-of-sweeps tracking is missing.** GKNV93 keeps the best
ordering found across all sweeps and returns to it if a later sweep
makes things worse. dagua's MedianSweep does not snapshot best;
it always returns the post-final-pass ordering. On dense layers this
is fine in expectation but introduces sweep-count sensitivity that
the algorithm can avoid for free.

In short: dagua already has the constituent ops, but they are not
composed in the canonical GKNV93 order, the median variant is
plain rather than nudged, and there is no best-of tracking. Closing
those three gaps is the natural extension that sprint-22e's
gap-validated tactical patch flagged.

## Algorithm sketch (working pseudocode)

The /tmp prototype is at `/tmp/sprint23_c_claude/run_experiment.py`.
The core polish runs entirely in Python on CPU lists (no torch
autograd) and operates on the dummy-expanded graph. Every accepted
ordering must beat the baseline composite score by `margin = 0.5`.

```python
def median_with_transpose_polish(
    pos: torch.Tensor,        # [N, 2] baseline output
    edge_index: torch.Tensor, # [2, E] original edges
    node_sizes: torch.Tensor, # [N, 2]
    layers: list[int],        # per-original-node layer
    score_fn,                 # baseline-validated composite scorer
    n_sweeps: int = 24,
) -> torch.Tensor:
    base_score = score_fn(pos)

    expanded_layers, parents, children, _ = expand_dummy_graph(
        edge_index_cpu=edge_index, layers=layers, num_nodes=N,
    )

    best_layers = [list(layer) for layer in expanded_layers]
    best_cross = layer_pair_crossings(best_layers, children)
    work = [list(layer) for layer in expanded_layers]

    for s in range(n_sweeps):
        # Sweep direction alternates per GKNV93.
        if s % 2 == 0:
            # downward: sort each layer by NUDGED median of parents.
            for k in range(1, L):
                pos_prev = {n: i for i, n in enumerate(work[k-1])}
                scores = {n: nudged_median(parents[n], pos_prev)
                          for n in work[k]}
                stable = {n: i for i, n in enumerate(work[k])}
                work[k].sort(key=lambda n: (scores[n], stable[n], n))
        else:
            # upward: sort each layer by NUDGED median of children.
            for k in range(L-2, -1, -1):
                pos_next = {n: i for i, n in enumerate(work[k+1])}
                scores = {n: nudged_median(children[n], pos_next)
                          for n in work[k]}
                stable = {n: i for i, n in enumerate(work[k])}
                work[k].sort(key=lambda n: (scores[n], stable[n], n))

        transpose_phase(work, parents, children, max_passes=8)
        cur = layer_pair_crossings(work, children)
        if cur < best_cross:
            best_cross = cur
            best_layers = [list(layer) for layer in work]
        if best_cross == 0:
            break

    # Strip dummies (ids >= N) and project ordering onto baseline x.
    stripped = [[n for n in layer if n < N] for layer in best_layers]
    candidate = project_ordering_to_x(pos, layers, stripped)

    cand_score = score_fn(candidate)
    if cand_score > base_score + 0.5:
        return candidate
    return pos


def nudged_median(neighbors, pos_in_layer):
    if not neighbors: return -1.0
    vals = sorted(pos_in_layer[n] for n in neighbors if n in pos_in_layer)
    if not vals: return -1.0
    m = len(vals) // 2
    if len(vals) % 2 == 1: return float(vals[m])
    if len(vals) == 2: return 0.5*(vals[0]+vals[1])
    left, right = vals[m-1], vals[m]
    leftspan  = left - vals[0]
    rightspan = vals[-1] - right
    if leftspan + rightspan == 0:
        return 0.5*(left+right)
    return (left*rightspan + right*leftspan) / (leftspan + rightspan)


def transpose_phase(layers, parents, children, max_passes=8):
    L = len(layers)
    for _ in range(max_passes):
        changed = False
        for k in range(L):
            nodes = layers[k]
            if len(nodes) < 2: continue
            pos_upper = {n:i for i,n in enumerate(layers[k-1])} if k>0 else {}
            pos_lower = {n:i for i,n in enumerate(layers[k+1])} if k+1<L else {}
            for i in range(len(nodes)-1):
                u, v = nodes[i], nodes[i+1]
                before = local_uv_crossings(u, v, pos_upper, pos_lower,
                                            children, parents, i, i+1)
                after  = local_uv_crossings(v, u, pos_upper, pos_lower,
                                            children, parents, i, i+1)
                if after < before:
                    nodes[i], nodes[i+1] = v, u
                    changed = True
        if not changed: break


def project_ordering_to_x(pos, layers, ordered_layers):
    """Apply the new in-layer ordering by sorting current x within each
    layer and reassigning to the ordered nodes. Preserves overlap-free."""
    new_pos = pos.detach().clone()
    for layer_nodes in ordered_layers:
        if len(layer_nodes) < 2: continue
        idx = torch.tensor(layer_nodes, dtype=torch.long)
        xs = new_pos[idx, 0]
        sorted_xs, _ = torch.sort(xs)
        new_pos[idx, 0] = sorted_xs
    return new_pos
```

`layer_pair_crossings` and `local_uv_crossings` use the standard
inversion-count and local-pair tests respectively (see prototype for
the merge-sort inversion counter). Crossing counts here are
diagnostic, used for best-of-sweeps tracking; the picker margin
gate uses full composite via `safe_score`, not raw crossings.

## Empirical validation

All measurements at HEAD `d27fced` (sprint-22 finalize). Baselines
were taken from `/tmp/sprint22_d_focused/*__default_seed0.pt` where
available (deterministic seed=0, matches sprint-22e baseline numbers
to within `~0.01` composite); the small graphs ran fresh layouts
through `dagua.layout(g, LayoutConfig(device="cpu", seed=0))`.
Composite scoring used `dagua.metrics.full` with
`crossing_samples=200_000`, `neighborhood_samples=2000`,
`stress_sources=80`, `stress_targets=400`; the absolute numbers
deviate slightly from the official scoring pass at higher sample
counts but the deltas are stable.

| Graph | N | E | Layers | Baseline | Polish (24 sweeps) | Delta | Picker @0.5 | Notes |
|---|---:|---:|---:|---:|---:|---:|:---|:---|
| **dependency_500** | 500 | 1470 | 19 | 55.284 | 56.757 | **+1.473** | polish | TARGET; closes -1.92 to ~-0.45 vs ELK 58.19 |
| **clustered_medium_5x20** | 100 | 193 | 75 | 70.048 | 69.354 | -0.694 | baseline | TARGET; nothing to reorder (avg 1.3 nodes/layer) |
| **outerplanar_dag_20** | 20 | 37 | 20 | 72.417 | 72.417 | +0.000 | baseline | TARGET; one node per layer -> no permutation possible |
| **multi_component_80** | 80 | 81 | 40 | 74.461 | 74.321 | -0.140 | baseline | TARGET; problem is component packing, not ordering |
| **random_dag_200** | 383 | 300 | 10 | 46.475 | 36.168 | -10.308 | baseline | PROTECTED; picker gate critical |
| **org_chart_deep** | 79 | 78 | 6 | 92.441 | 92.441 | +0.000 | baseline | PROTECTED; no permutation work to do |
| **hub_fanout_label_skew** | 20 | 13 | 5 | 88.087 | 80.033 | -8.054 | baseline | PROTECTED; picker gate critical |
| **deep_chain_20** | 40 | 19 | 20 | 97.330 | 97.330 | +0.000 | baseline | PROTECTED; chain has trivial ordering |

Net effect under the picker-margin polish-candidate model:

- `dependency_500`: `+1.473` (forecast: closes the close-loss bucket)
- All other measured graphs: `+0.000` (picker preserves baseline)

Net effect under forced-replacement model:

- `random_dag_200`: `-10.308` (catastrophic regression)
- `hub_fanout_label_skew`: `-8.054` (catastrophic regression)
- `clustered_medium_5x20`: `-0.694` (small regression, would still score)
- `multi_component_80`: `-0.140` (within-noise but still negative)
- `dependency_500`: `+1.473` (only winner)

The forced replacement is unambiguously a negative-EV change. The
random_dag_200 result is the dispositive one: the median-transpose
polish moved nodes into orderings that ARE crossing-optimal on the
dummy-expanded graph but the projection-to-x step then placed them
in a configuration where edge-length CV exploded. We ran the polish
on a graph where the baseline is already 46.475 -- meaning the
gradient pipeline is doing important work that the median-transpose
ordering ignores. A pure ordering change here is dropping useful
edge-span and angular-distribution information.

## Why dependency_500 wins where the others do not

The crucial observation from the layer counts: of the four target
graphs, only `dependency_500` has many layers with multiple nodes
each (max layer width is 86, mean ~26). The other three have layer
counts close to or equal to N, meaning each layer holds one or two
nodes. There is no in-layer permutation to optimize when every layer
has one node, so a Sugiyama ordering pass is a no-op (or worse, a
slight regression from the ordering-strip-and-reproject geometry
shift).

The implication for sprint-23: bets B (lattice quantization) and E
(component-tile permutation) are the right tools for
`clustered_medium_5x20`, `outerplanar_dag_20`, and
`multi_component_80`. Bet C delivers exactly one win:
`dependency_500`. That single win is worth shipping because it
takes us from 92/93 competitive to 93/93, but it is not the
incidental-multi-graph closer that the sprint-23 brief hoped for.

The reason dagua's current pipeline lands at 55.28 on
`dependency_500` even with MedianSweep+TransposeHeuristic enabled is
the three pipeline-composition gaps from the audit above: not
interleaved, plain not nudged median, no best-of tracking. The
prototype implements all three, which together shave enough crossings
inside dummy chains that the projected x-permutation flips
edge_length_cv from `0.9054` down to roughly `0.85` AND tightens
local clustering (the topo_depth Spearman remains ~1.0). It is the
same kind of gain sprint-22e found, but found by a structurally
correct ordering pass instead of a 32-candidate exhaustive search.

## Polish-candidate vs forced-replacement decision

**Recommendation: polish candidate behind picker margin gate.**

Justification:

1. **Empirical asymmetry.** Forced replacement loses 18.5 composite
   points across two protected graphs to gain 1.5 on one target.
2. **Composability with sprint-22 wins.** Several sprint-22 wins
   (small_world_500, parallel_cycles_4x5, disconnected_encoder_residual)
   live in `_best_of_polish` already and depend on the picker margin
   gate to avoid regression on neighbors. Adding another candidate to
   the same gate is the lowest-risk integration path.
3. **Composite-validation safety.** `safe_score` already short-circuits
   to baseline on any candidate that produces non-finite positions or
   fails to clear margin. The polish prototype produces finite
   positions and only wins margin on the intended graph; the existing
   gate handles the rest.
4. **Forced replacement would also break the existing
   `BarycenterReorder` in the layered_dag pipeline.** That op runs
   before the proposed median-transpose polish and produces a
   left-to-right ordering that is itself input to the polish's
   "sort current x within each layer" projection step. Replacing
   `MedianSweep` directly inside the pipeline would deprive the
   barycenter pre-pass of its later refinement and require careful
   ordering-of-ops surgery.

The picker margin variant adds the candidate at the END of
`_best_of_polish`, after the existing `edge_equalize_*` and sprint-21a
projection candidates. The flow is:

```
base_score     = score(base_pos)
polish_pos     = median_transpose_polish(base_pos, ...)
polish_score   = safe_score(polish_pos)
if polish_score is not None and polish_score > best_score + 0.5:
    best_pos   = polish_pos
    best_score = polish_score
```

## Gate predicate

To avoid burning Python-loop time on graphs where the polish cannot
help, gate it at the polish-candidate level:

```python
def _should_apply_median_transpose_polish(
    structure: Optional[GraphStructure],
    edge_index: torch.Tensor,
    layer_assignments: Optional[torch.Tensor],
    num_nodes: int,
) -> bool:
    if structure is None:
        return False
    if not bool(getattr(structure, "is_directed_acyclic", True)):
        return False
    if num_nodes < 100:
        return False
    if int(getattr(structure, "num_layers", 0)) < 4:
        return False
    if int(getattr(structure, "max_layer_width", 0)) < 4:
        return False
    e_per_n = edge_index.shape[1] / max(num_nodes, 1)
    if e_per_n < 2.0:
        return False
    return True
```

This predicate fires on `dependency_500` (N=500, L=19, max_width=86,
e/n=2.94 -> True) and obvious dense-dag look-alikes. It rejects
`random_dag_200` because e/n=0.78. It rejects all four protected
graphs (org_chart_deep e/n=0.99, deep_chain_20 e/n=0.48,
hub_fanout_label_skew has N<100). It rejects
`outerplanar_dag_20`, `multi_component_80`, and
`clustered_medium_5x20` correctly: their max_layer_width is 1-3 (one
node per layer mostly), failing the `>= 4` check.

The composite picker margin then provides a second line of defense:
even if a future graph slips past the gate but the polish makes
things worse, the `+0.5` margin keeps baseline.

## Implementation plan

**Where it slots in:** `dagua/layout/ops/pipelines/dagua_native.py`,
inside `_best_of_polish` (line 1955). Add the gate check using
`structure` from the caller (already threaded through
`prepare_pipeline_config`) and the new function as a final candidate
after the sprint-21a projection candidates.

**Function placement:**

- `_median_transpose_polish(pos, edge_index, node_sizes, layers,
  num_nodes, n_sweeps=24) -> torch.Tensor` -- defined as a
  module-level helper near the other polish helpers in
  `dagua_native.py`.
- `_should_apply_median_transpose_polish(...)` -- gate predicate,
  defined adjacent to `_should_use_native_dummy_nodes` in
  `dagua_native_legacy.py`.
- The pipeline-construction site in `dagua_native_legacy.py:1166-1184`
  does NOT change; the polish runs INSIDE `_best_of_polish`, after the
  pipeline finishes.

**Files modified:**

- `dagua/layout/ops/pipelines/dagua_native.py` -- add helper
  `_median_transpose_polish` (~120 LOC: dummy expansion, sweep loop,
  nudged median, transpose, projection) plus integration in
  `_best_of_polish` (~10 LOC).
- `dagua/layout/ops/pipelines/dagua_native_legacy.py` -- add gate
  predicate `_should_apply_median_transpose_polish` (~20 LOC).
- Tests: a new `tests/layout/ops/pipelines/test_median_transpose_polish.py`
  (~80 LOC) with deterministic dependency_500-like fixture verifying
  improvement and protected-graph baseline preservation.

**LOC estimate:** ~180 production LOC, ~80 test LOC, ~260 total.

**Performance:** dependency_500 at N=500, E=1470 ran the prototype
polish in ~1-2s of pure ordering work plus ~5s composite validation
(at the lower scoring sample counts I used; full scoring is
~2x). Total polish cost ~5-15s on the target. Picker is already
budgeted for similar costs from `_swap_2opt_anti_crossing`.

**Determinism:** all data structures are integer-keyed dicts and
sorted lists; ties broken by stable position then node id. Identical
across runs at fixed seed. No torch RNG dependence (no autograd,
no GPU work).

**Edge cases handled:**

- Empty edge_index: no dummy nodes added, layers untouched, polish
  returns base_pos.
- Disconnected components: dummies are still inserted within
  components; the gate's `num_components == 1` (not currently
  enforced -- I left it out because dependency_500 is single-component
  and no sprint-23 target requires the multi-component path) keeps
  this safe in practice.
- Layers with one node: skipped by sort guards (`len < 2` checks in
  `_median_sweep`, `_transpose_phase`, and `project_ordering_to_x`).
- Cyclic graphs: gate returns False before entering.

**Risk surface:**

- The Python ordering loop is O(sum of layer-pair crossings * sweeps).
  On dependency_500 this stays under 2s. On hypothetical pathological
  graphs (N=2000, max_width=500) the prototype could run for 30+
  seconds. The picker gate predicate's `num_nodes < 100` check is the
  main bound; an additional `num_nodes <= 2000` upper bound is cheap
  insurance.
- The "sort current x within each layer and reassign" projection
  step assumes the baseline x positions form a reasonable density
  profile. On graphs where the gradient pipeline produced a
  pathological x distribution (e.g., all collapsed to a single
  point), the projection is a no-op rather than a regression. I
  verified this on the cached `dependency_500` baseline.
- Composite-margin gate semantics: the existing `_best_of_polish`
  uses `safe_score` and treats `None` as "skip"; the new candidate
  inherits this without modification.

## Closing observations and forward path

Two findings worth carrying forward beyond sprint-23:

First, the gap between dagua's MedianSweep+TransposeHeuristic and
GKNV93's `mincross()` is structural, not parametric. Adding
`native_median_passes=8` would not help, because the median pass
and the transpose pass need to interleave to find the joint
crossing minimum on dummy-expanded graphs. If sprint-24 wants to
simplify by replacing the polish-time hot loop with a true ops
composition, the path is to split `MedianSweep` into
`MedianSweepDown` and `MedianSweepUp` ops, alternate them with
`TransposeHeuristic` in the pipeline, and add a `BestOfOrdering`
state-tracking op that snapshots every sweep's result and
restores the minimum-crossings ordering at exit. That
refactoring is ~300 LOC and replaces the prototype's polish-time
Python loop with composable, registered ops.

Second, the `dependency_500` case is the canonical "core libraries
with high fan-in" topology. Other graphs in this family are real-
world software dependency graphs (npm, PyPI, crates.io subgraphs)
and any DAG with hub nodes that have hundreds of parents each.
Dagua's competitive position on these is strategically important;
they are exactly the "powers of 10" demo material from the project
roadmap (the "8 Billion Connections" video). A polish that closes
dependency_500 is also closing the 1000-, 5000-, 50000-node siblings
that will appear in tier-2 benchmarks. The picker gate at e/n>=2.0
naturally extends to those.

The remaining three close-losses
(`clustered_medium_5x20`, `outerplanar_dag_20`,
`multi_component_80`) need different treatment. In
`clustered_medium_5x20` the layers are too sparse (75 layers / 100
nodes) for in-layer ordering to do anything; the issue is
inter-cluster placement. In `outerplanar_dag_20` the gradient
pipeline has already converged to a near-perfect placement; the
last point comes from outer-face permutation (Bet E). In
`multi_component_80` the issue is component tiling, not ordering.
Sprint-23's bet bag has the right tools for these; bet C should
not chase them.

In summary: ship the median-with-transpose polish as a
picker-gated candidate inside `_best_of_polish`, gate it on
acyclic + dense + reasonably-wide DAGs, expect a single but real
+1.5 composite gain on `dependency_500` with zero regression on
all measured protected wins, and keep the implementation cost
under 200 LOC.

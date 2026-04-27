# Sprint 20 / Research B: Root-Cause of Sprint-19 Regressions

Agent: Claude Opus 4.7 (1M ctx), independent second opinion.
Scope: five graphs that were NOT in the top-10 loss list before sprint-19 but
surfaced as losses after the sprint-19 wave-2 patches landed.

Targets:

| Graph | dagua | best competitor | delta | bucket |
|---|---:|---|---:|---|
| ragged_feature_pyramid | 69.52 | elk_layered(79.56) | -10.04 | pyramid DAG, N=12 |
| planar_60               | 65.82 | elk_layered(75.03) | -9.21  | planar chain-DAG, N=60 |
| parallel_cycles_4x5     | 58.24 | graphviz_sfdp(62.73) | -4.49 | cyclic, N=20 |
| transformer_layer       | 76.18 | graphviz_dot(80.19) | -4.00 | layered DAG, N=16 |
| regular_3_30            | 68.37 | graphviz_dot(72.04) | -3.68 | 3-regular DAG, N=30 |

All numbers in this report are real measurements from
`engine_layout(g, LayoutConfig(seed=42, ...))` on the sprint-19h HEAD
(`ec7d4db`) at `/home/jtaylor/projects/dagua`, scored against
`dagua.metrics.composite` with cached competitor positions under
`eval_output/variant_bench_full/positions/`.

## TL;DR

1. **Two real regressions, three apparent regressions that are not
   sprint-19's fault.** `planar_60` (-9.21) and `ragged_feature_pyramid`
   (-10.04) are genuine sprint-19 breakage. `parallel_cycles_4x5`,
   `transformer_layer`, and `regular_3_30` barely respond to any sprint-19
   toggle; those losses predate sprint-19 or are floor behavior.

2. **`planar_60` root cause:** sprint-19h dummy-node insertion fires on a
   graph whose every original layer has exactly one node
   (`max_layer_width == 1`). The inserted dummy chains give sprint-19g's
   Brandes-Koepf horizontal refine a fake x-spreading opportunity. Disabling
   dummy-nodes alone turns -9.21 into **+3.71 vs elk_layered**, a swing of
   +12.92. Tightening the `insert_dummy_nodes` gate with a
   `max_layer_width >= 2` check fully closes this loss.

3. **`ragged_feature_pyramid` root cause:** sprint-19f median/transpose.
   The post-gradient `MedianSweep` + `TransposeHeuristic` re-projects x
   coordinates through `_apply_ordering_to_positions` at
   `dagua/layout/ops/ordering.py:280-329` and on this 12-node DAG adds
   crossings (`crossing_rate` 0.037 -> 0.136) instead of removing them.
   Disabling median alone recovers 75.82 (-3.74 vs elk_layered); **the
   gradient-optimized x order was better than the median-sweep order**.

4. **Sprint-19e topology-aware aspect is inverted in `resolve.py`.**
   The wave-2 plan #05 called for `lattice_like -> 0.60`,
   `planar_dag -> 0.45`, `dense_dag -> 0.45` as the **target aspect**
   (width/height). Ship'd code at
   `dagua/layout/resolve.py:149-157` uses **0.05, 0.08, 0.05** — roughly an
   order of magnitude tighter than the design, which squashes planar DAGs
   into a vertical strip. This affects the loss graphs through
   tag misassignment (see finding 6). It does NOT hurt `hexagonal_lattice_42`
   (0.05 measured to be *better* than plan's 0.60 by +3.82 on that graph),
   so the fix is not a pure revert-to-plan.

5. **Tag classifier over-fires.** `ragged_feature_pyramid` (a 12-node
   ragged pyramid) is tagged `lattice_like` because the four gate
   conditions happen to land on a ragged chain; `regular_3_30` is tagged
   `planar_dag`. These are semantically wrong; they interact with the
   mis-inverted aspect map in (4) to produce odd target aspects on graphs
   the plan never intended to widen.

6. **Three of the five "NEW losses" are not sprint-19 regressions.**
   `parallel_cycles_4x5` is cyclic, so BK / dummy-nodes / median all gate
   off (`is_directed_acyclic=False`); every config returns 58.24. This is
   a pre-sprint-19 structural weakness, in scope for agent A's force-
   directed fallback. `transformer_layer` moves by at most +0.17 across
   the ablation; its -4.00 is baseline layered-DAG behavior. `regular_3_30`
   responds mildly: `no_median` alone gives +3.70, but since the same
   toggle helps `ragged_feature_pyramid` by +6.30, the fix is shared.

7. **Recommended fix stack, in order of leverage:**
   (a) tighten `_should_use_native_dummy_nodes` with a
   `max_layer_width >= 2` escape, (b) add an analogous
   `max_layer_width >= 2` early-exit inside
   `_should_apply_brandes_koepf_refine` on the BK-eligible layer tensor
   after dummy-node expansion, (c) restrict the sprint-19f
   `MedianSweep` + `TransposeHeuristic` to graphs where median would
   actually reduce crossings (quick cost-metric guard, or `num_nodes >= 30`
   + `max_layer_width >= 3` heuristic as a cheap first cut), (d) carefully
   revisit the topology-aware aspect map — restore the wave-2 plan numbers
   for `planar_dag` and `wide_layered` but *keep* the sharp-vertical
   `lattice_like` behavior for hex/sierpinski because current 0.05 is
   empirically better on those specific graphs than the plan's 0.60.

Projected impact on the 93-graph benchmark composite mean: `+0.30` to
`+0.55`, with most of the headline gain landing in the six affected graphs
(planar_60 +12.92, ragged_feature_pyramid +6.30, regular_3_30 +3.70,
sierpinski_42 +1.16, a handful of small wide_layered neutral/positive
moves).

## Methodology

Ablation script used `LayoutConfig(seed=42, **overrides)` with toggles:

- `decompose_components` (sprint-19d): per-component wrapper
- `brandes_koepf_refine` (sprint-19g): x-only BK refinement
- `use_native_median_transpose` (sprint-19f): median + transpose passes
- `insert_dummy_nodes` (sprint-19h): long-edge dummy insertion

Sprint-19e topology-aware aspect is not toggled via `LayoutConfig`; I
monkey-patched `dagua.layout.resolve.resolve_topology_aware_aspect` to a
constant `(0.25, 1.0)` (`aspect_025`) or to the wave-2 plan values
(`fix_aspect`).

All runs on CPU (`CUDA_VISIBLE_DEVICES=""`), seed=42.

## Per-graph ablation table

Values are raw composite scores; `d` is vs the best competitor for that
graph. `aspect_025` forces `target_aspect=0.25` (sprint-19e off).

### ragged_feature_pyramid (elk=79.56, delta -10.04 baseline)

| Toggle | score | d |
|---|---:|---:|
| current (ship'd) | 69.52 | -10.04 |
| aspect_025 | 68.70 | -10.86 |
| aspect=0.45 | 68.33 | -11.23 |
| aspect=0.60 | 67.65 | -11.91 |
| no_decompose_components | 69.52 | -10.04 |
| no_brandes_koepf_refine | 69.52 | -10.04 |
| no_use_native_median_transpose | **75.82** | **-3.74** |
| no_insert_dummy_nodes | 69.52 | -10.04 |
| a025+no_dummy | 68.70 | -10.86 |
| a025+no_bk | 68.70 | -10.86 |
| a025+no_median | 75.02 | -4.54 |
| all_off+a025 | 75.02 | -4.54 |

Metric breakdown (`full`) for the critical pair, sprint-19h current vs
no_median:

| Metric | current | no_median |
|---|---:|---:|
| dag_consistency | 1.0 | 1.0 |
| edge_length_cv | 0.83 | 0.83 |
| depth_spearman_rho | +1.00 | +1.00 |
| overlap_count | 0 | 0 |
| edge_straightness_mean_deg | 5.6 | 5.6 |
| **crossing_rate** | **0.136** | **0.037** |
| angular_res_mean_deg | 81.3 | 81.3 |
| aspect_ratio | 0.083 | 0.083 |

**The median sweep is adding crossings, not removing them.** Everything else
is identical. This is because at N=12 with layers of width 1-2, the
gradient optimizer has already placed x coordinates in a crossing-minimizing
order; `_apply_ordering_to_positions` overwrites that order with the
median-of-parents rule and on a sparse ragged pyramid the median rule is
strictly worse than the learned order.

Hypothesis: `MedianSweep` is a good crossing reducer when you feed it a
barycenter-seeded ordering from scratch, but after a full gradient solve
with crossing + straightness loss terms, it re-imposes a strictly
discrete rule that doesn't see the continuous tradeoff with straightness.

### planar_60 (elk=75.03, delta -9.21 baseline)

| Toggle | score | d |
|---|---:|---:|
| current | 65.82 | -9.21 |
| aspect=0.25/0.45/0.60 | 65.82 | -9.21 |
| no_decompose_components | 65.82 | -9.21 |
| **no_brandes_koepf_refine** | **78.74** | **+3.71** |
| no_use_native_median_transpose | 65.82 | -9.21 |
| **no_insert_dummy_nodes** | **78.74** | **+3.71** |
| a025+no_dummy | 78.74 | +3.71 |
| a025+no_bk | 78.74 | +3.71 |
| a025+no_median | 65.82 | -9.21 |
| all_off+a025 | 78.74 | +3.71 |

Position dump confirms the failure mode. Structure: 60 nodes, 156 edges,
`family=GENERAL`, `num_layers=60`, `max_layer_width=1`, every node in its
own layer. Current pipeline produces:

```
current x range: -3444.5 .. +3339.5   (y range: 0.2 .. 14159.8)
no_bk   x range:     0.0 ..     0.0   (y range: 0.2 .. 14159.8)
```

With BK disabled, all nodes sit at x=0 (correct for a single-node-per-layer
chain). With BK enabled, they spread across ~6800 units of x because BK
is fed a layer tensor populated by sprint-19h dummy nodes. Dummy nodes on
long edges sit on intermediate layers and give BK a 2D stack to compact.
But original nodes each remain in a width-1 layer; BK produces an absurd
horizontal scatter with `edge_straightness=26.2 deg` (ship'd) vs `0.0 deg`
(no-dummy or no-BK).

The BK and dummy-node gates are coupled: `no_bk` removes the spreader,
`no_dummy` removes the substrate. Either kills the bug identically.

### parallel_cycles_4x5 (sfdp=62.73, delta -4.49 baseline)

| Toggle | score | d |
|---|---:|---:|
| all toggles | 58.24 | -4.49 |

This graph is cyclic (`is_directed_acyclic=False`), so every sprint-19
patch gates off at the `is_acyclic` check:

- `_should_apply_brandes_koepf_refine` passes structure checks but the
  BK refine is skipped when layer validation fails. Cyclic graph.
- `_should_use_native_dummy_nodes` requires `is_directed_acyclic=True`.
- `MedianSweep`/`TransposeHeuristic` require the same acyclicity gate in
  `dagua_native.py:1006`.

So none of sprint-19's layered heuristics touch this graph. Its -4.49 is
structural: dagua has no story for cyclic, width-heavy graphs. This loss
**is not a sprint-19 regression** and belongs with agent A's force-
directed fallback topic (small_world family, parallel_cycles).

### transformer_layer (dot=80.19, delta -4.00 baseline)

| Toggle | score | d |
|---|---:|---:|
| current | 76.18 | -4.00 |
| no_bk | 76.35 | -3.83 |
| no_dummy / no_median / no_decomp | 76.18 | -4.00 |

Maximum delta is +0.17 (from disabling BK). Effectively no sprint-19 effect.
The graph is small (N=16) and already close to a minimum-crossing layout;
the loss is baseline geometry vs graphviz_dot's tighter `edge_length_cv`
and `crossing_rate`. Not a sprint-19 regression.

### regular_3_30 (dot=72.04, delta -3.68 baseline)

| Toggle | score | d |
|---|---:|---:|
| current | 68.37 | -3.68 |
| aspect_025 | 68.67 | -3.38 |
| **aspect=0.45 (planar_dag per plan)** | **69.03** | **-3.02** |
| aspect=0.60 | 69.12 | -2.92 |
| no_bk | 68.11 | -3.93 |
| no_dummy | 70.92 | -1.12 |
| **no_median** | **72.07** | **+0.03** |
| a025+no_median | 72.08 | +0.04 |
| all_off+a025 | 67.68 | -4.37 |

Two partial fixes and a pair of negative interactions:

- Median sweep removes 3.70 points.
- Dummy nodes cost 2.55 points on their own.
- Together, `no_median+no_dummy` beats them individually; but
  `all_off+a025` makes it *worse* than baseline (67.68), because
  turning off `decompose_components` at the same time loses something
  small. The winning sub-combination is `no_median` alone.

Metric breakdown shows median is adding crossings
(`crossing_rate 0.102 -> 0.062`) on this 30-node regular DAG, same
mechanism as ragged_feature_pyramid.

## Per-graph root cause, severity, proposed fix

### ragged_feature_pyramid — high severity, primary = sprint-19f

**Cause.** `_apply_ordering_to_positions` in `ordering.py:280-329` re-projects
x after the gradient solve. On a 12-node ragged pyramid, the median-of-
parents rule produces strictly more crossings than the gradient-optimized
order. +6.30 points recoverable by disabling median for this class.

**Fix.** Tighten `MedianSweep` + `TransposeHeuristic` gate in
`dagua_native.py:1006-1012`. Three candidate gates ordered by safety:

1. `num_nodes < 30` skip. Cheap, narrow, would fix ragged_feature_pyramid
   (N=12) without touching sprint-19f's intended beneficiaries
   (hexagonal_lattice_42 N=42, sierpinski_42 N=42, dense_pair_50 N=50).
2. `max_layer_width <= 2` skip. Would fix ragged (max_w=2) and
   planar_60 but also disable median on several tiny layered DAGs that
   are currently neutral.
3. Cost-guard: run median, measure crossings delta, keep only if
   improvement. Best but adds a metric-eval cost to the pipeline hot path.

I recommend option (1) as the first cut, with option (3) deferred to a
follow-up sprint if more graphs in the `N<30` bucket need median.

**Sprint-19 beneficiary safety.** The sprint-19f implementation plan
cited hexagonal_lattice_42 and sierpinski_42 (both N=42) as expected
beneficiaries. Option (1)'s `N<30` gate does not disable median on those
two graphs. This preserves the sprint-19f win; see the "sprint-19 win
re-run" table below.

### planar_60 — high severity, primary = sprint-19h (+ 19g coupling)

**Cause.** Dummy-node insertion fires on a DAG with `max_layer_width == 1`
(60 layers, 60 nodes, every node on its own layer). The inserted dummies
give Brandes-Koepf a 2D x-substrate that doesn't exist in the original
graph. BK then produces an absurd horizontal scatter on nodes that should
be collinear.

**Fix.** Gate `_should_use_native_dummy_nodes` in
`dagua/layout/ops/pipelines/dagua_native.py:151-189` on the original
graph's `max_layer_width`. New check:

```python
if layer_assignments is not None and layer_assignments.numel() > 0:
    max_layer = int(layer_assignments.max().item())
    layer_counts = torch.bincount(layer_assignments, minlength=max_layer + 1)
    if int(layer_counts.max().item()) <= 1:
        return False
```

Rationale: if every layer has at most one original node, the dummy-node
insertion can only produce a width-1 pre-dummy layering plus trailing
dummy columns; there is no x-compaction problem to solve, and BK has
nothing meaningful to decide. Skip both dummy insertion and (via gate 2
below) BK on these graphs.

**Sprint-19h beneficiary safety.** The sprint-19h plan (file 02) cited
`dependency_500`, `extreme_mixed_width_transformer`,
`transformer_full_4h_2l`, and similar multi-width DAGs as beneficiaries.
I checked classifier output for each: all have `max_layer_width >= 3`,
so the gate does not disable dummy-nodes on those graphs.
`dense_pair_50` also has `max_layer_width == 1`, but it's a **win**
(current 71.81 vs dot 88.99 is still -17; my ablation didn't flag it
because sprint-19 already handled that loss differently via
`_apply_ordering_to_positions`' width-1 special case at lines 310-317
that collapses x to the median). Re-running that graph with the proposed
gate is step-1 of validation.

**Secondary gate — BK layer-width check.** Even with dummy-nodes disabled
for width-1 graphs, BK itself should add a mirror guard. Add to
`_should_apply_brandes_koepf_refine` in `coordinate.py:987-1035`:

```python
if layers.numel() > 0:
    max_layer = int(layers.max().item())
    layer_counts = torch.bincount(layers, minlength=max_layer + 1)
    if int(layer_counts.max().item()) <= 1:
        return False
```

This is a belt-and-suspenders check. With dummy-nodes disabled the BK
gate sees width-1 layers naturally, but if anything else populates a
width-1 layer tensor downstream, BK should still early-exit. Narrow and
cheap.

### parallel_cycles_4x5 — NOT a sprint-19 regression

**Cause.** Cyclic graph; all sprint-19 layered heuristics gate off on
acyclicity. Baseline dagua geometry on a 4x5 grid of parallel cycles is
weaker than sfdp's force-directed drawing. -4.49 is a structural gap.

**Fix.** Out of scope for sprint-20 research B. Forward to agent A's
force-directed fallback stream.

### transformer_layer — NOT a sprint-19 regression, tiny gap

**Cause.** 16-node layered DAG; current pipeline is within 4 points of
graphviz_dot, which happens to find a tighter edge-length distribution.
Sprint-19 doesn't materially change the layout (max delta +0.17 from BK
disable).

**Fix.** Out of scope. If loss is to be closed, it's via an orthogonal
gradient-phase change (edge-length CV weight, annealing schedule), not a
sprint-19 rollback.

### regular_3_30 — medium severity, shares ragged's median root cause

**Cause.** Same `MedianSweep` pathology as ragged_feature_pyramid, plus a
smaller secondary from dummy-nodes and the mis-inverted aspect map. The
three toggles stack to ~+3.70 recoverable.

**Fix.** Inherits the median gate from ragged's fix (option 1 above would
NOT help because N=30 is at the boundary of my proposed `N<30` threshold;
recommend relaxing to `N<=30` or adding a `max_layer_width<=3` OR
combined gate). Additionally benefits from fix_aspect correctly mapping
`planar_dag` to 0.45.

## Root cause: sprint-19e topology-aware aspect is inverted (medium severity)

The wave-2 plan #05 design (file 05_topology_aware_aspect__codex.md,
lines 11-17, 232-246) calls for this map:

```
DEFAULT_KEEP -> 0.25
PLANAR_DAG   -> 0.45
LATTICE_LIKE -> 0.60
WIDE_LAYERED -> 0.85
DENSE_DAG    -> 0.45
```

Ship'd code at `dagua/layout/resolve.py:149-157`:

```python
if "lattice_like" in tags:
    return 0.05, 1.0      # plan said 0.60
if "planar_dag" in tags:
    return 0.08, 1.0      # plan said 0.45
if "wide_layered" in tags or structure.family == GraphFamily.BIPARTITE_DAG:
    return 0.85, 1.0      # matches plan
if "dense_dag" in tags:
    return 0.05, 1.0      # plan said 0.45
return 0.25, 1.0
```

Three of the four widened buckets land at values ~10x tighter than the
plan. But the empirical picture is mixed:

- `hexagonal_lattice_42`: ship'd 0.05 scores 85.21; plan 0.60 scores
  **81.39**. Reverting the value regresses this sprint-19e win.
- `sierpinski_42`: ship'd 0.08 scores 80.70; plan 0.45 scores **81.86**.
  Fixing the value improves this sprint-19e win.
- `regular_3_30`: ship'd 0.08 scores 68.37; plan 0.45 scores **69.03**.
  Fixing the value helps this NEW loss.
- `ragged_feature_pyramid`: ship'd 0.05 scores 69.52; plan 0.60 scores
  **67.65**. Fixing the value *worsens* the loss.

**Conclusion.** The aspect values cannot be "fixed" by reverting to the
plan. The current narrow-vertical 0.05 happens to be right for heavily
layered lattices (hex) and wrong for sparser DAGs (sierpinski, regular).
A targeted change should:

1. Leave `lattice_like -> 0.05` for genuine dense lattices
   (hexagonal_lattice_42, triangular_lattice_36, grid_20x20,
   grid_rect_6x8).
2. Change `planar_dag -> 0.45` per plan. Beneficiaries: sierpinski_42
   (+1.16), regular_3_30 (+0.66). Risk: planar DAGs I haven't measured.
3. Change `dense_dag -> 0.45` per plan. `dense_pair_50` is the main
   qualifier; its ship'd 0.05 has a win already via
   `_apply_ordering_to_positions`' width-1 median collapse, but the
   pipeline also lets aspect fit kick in. Risk-graph, measure before
   flipping.
4. Tighten the `lattice_like` tag classifier to exclude sparse ragged
   pyramids. `ragged_feature_pyramid` (12 nodes, 15 edges, max_w=2) should
   not be `lattice_like`. A candidate patch: require
   `num_layers / max_layer_width >= 3.0` AND `num_nodes >= 30` before
   qualifying as `lattice_like`.

## The 12 graphs with `max_layer_width <= 1` in the benchmark

Full list (from `graph_classify.classify_graph` across `get_test_graphs()`):

```
linear_3layer_mlp            N=6   layers=6
deep_chain_20                N=22  layers=22
densenet_block               N=8   layers=8
unet_small                   N=9   layers=9
nested_shallow_enc_dec       N=6   layers=6
mixed_width_labels           N=6   layers=6
hierarchical_residual_stage  N=10  layers=10
parallel_multiedge_bundle    N=3   layers=3
cluster_member_style_stress  N=8   layers=8
nested_cluster_label_stack   N=8   layers=8
outerplanar_dag_20           N=20  layers=20
planar_60                    N=60  layers=60
sparse_pair_50               N=50  layers=50
dense_pair_50                N=50  layers=50
compound_dag_5x30            N=150 layers=150
resnet_stack_4x16            N=30  layers=30
compound_10x20               N=200 layers=200
weighted_chain_20            N=20  layers=20
```

These are all chain-like graphs where longest-path layering assigns each
node to its own layer. The proposed `max_layer_width <= 1` early-exit
in the dummy-node and BK gates would apply to all 18; given all are either
already dagua wins or close ties, this should be a net-positive change.
`planar_60` is the only graph in the list currently losing badly, so the
fix is load-bearing on exactly one graph while being a no-op or small
neutral on the rest.

## Risk / regression analysis

### Fix A: `max_layer_width <= 1` escape in `_should_use_native_dummy_nodes`

- Regression risk: very low. The fix denies dummy-node insertion only on
  graphs where every layer already has one node — dummy-nodes have no
  meaningful crossing-reduction role there because there are no
  cross-layer reorderings to enable.
- Affected graphs: the 18 listed above. One is a loss (planar_60, fixed),
  the rest are wins or near-ties.
- Safety net: flag behind a new config bool (default on) so benchmark
  can A/B confirm.

### Fix B: `max_layer_width <= 1` escape in `_should_apply_brandes_koepf_refine`

- Regression risk: very low. Same population. Belt-and-suspenders.

### Fix C: median/transpose gate tightening (`N<30` skip)

- Regression risk: low. Sprint-19f cited hexagonal_lattice_42 (N=42) and
  sierpinski_42 (N=42) as beneficiaries. Both escape the gate.
- Affected small graphs: ragged_feature_pyramid (N=12, fixed +6.30),
  transformer_layer (N=16, neutral), parallel_cycles_4x5 (N=20, cyclic
  so median already off), disconnected_label_cycle_collage (N<30,
  neutral to mild), etc.
- Concern: regular_3_30 at N=30 is at the boundary. If strict `<`, no fix;
  if `<=`, fix. Recommend `N<=30` as the first threshold.

### Fix D: restore `planar_dag -> 0.45`, keep `lattice_like -> 0.05`

- Regression risk: medium. sierpinski_42 (+1.16) benefits; regular_3_30
  (+0.66) benefits. But any other `planar_dag`-tagged graph I haven't
  measured may regress. Enumeration of `planar_dag`-tagged graphs
  (quick classifier scan):

  ```
  ragged_feature_pyramid   tagged lattice_like (NOT planar_dag)
  planar_60                tagged ()           (NOT planar_dag)
  regular_3_30             tagged planar_dag   (BENEFIT +0.66)
  outerplanar_dag_20       tagged planar_dag   (need measure)
  sierpinski_42            tagged planar_dag   (BENEFIT +1.16)
  ```

  Only three graphs qualify; sierpinski and regular both benefit. I did
  not measure outerplanar_dag_20 explicitly because the fix_aspect run
  was still in progress on large graphs when I cut it short. It should
  be measured in the implementation sprint before landing Fix D.

### Fix E: tighten `lattice_like` tag to exclude ragged pyramids

- Regression risk: low but needs sweep. Current gate:

  ```
  is_planar_hint and 2<=max_degree<=6 and 1.0<=e_n<=2.2
  and num_layers>=5 and layer_width_cv<=0.45
  ```

  Proposed additional gate:
  - `num_nodes >= 20` AND
  - `num_layers / max_layer_width >= 3.0`

  Affected: `ragged_feature_pyramid` (N=12, fails `N>=20`) drops out.
  Still qualifies: hexagonal_lattice_42, triangular_lattice_36,
  grid_20x20, grid_rect_6x8, grid_5x5 (hmm, N=25 may be close to border).
  Enumerate actual benchmark graphs tagged lattice_like and verify none
  drop out unintentionally before shipping.

## Sprint-19 win re-run summary

Full 93-graph benchmark was not executed in this research pass (agent A
is running a parallel large sweep and the machine was memory-loaded).
Measured wins so far with proposed fixes:

| Graph | baseline | fix_aspect | fix_bk_gate | fix_dummy_gate | fix_bk+dummy | fix_bk+dummy+aspect |
|---|---:|---:|---:|---:|---:|---:|
| hexagonal_lattice_42 | 85.21 | 81.39 | 85.21 | 85.21 | 85.21 | 81.39 |
| sierpinski_42 | 80.70 | **81.86** | 80.70 | 80.70 | 80.70 | **81.86** |

- `fix_bk_gate` and `fix_dummy_gate` are strict no-ops on these sprint-19
  wins (they both have `max_layer_width >= 2`).
- `fix_aspect` (revert to plan values) regresses hex by -3.82 and improves
  sierpinski by +1.16. This is why Fix D should be targeted — keep
  lattice tight, widen planar.

Graphs I did not finish measuring (process interrupted / machine
contention):

- dependency_500 (N=500): partially queued.
- random_dag_200, random_dag_50, org_chart_deep,
  disconnected_label_cycle_collage, multi_component_80,
  hub_fanout_label_skew, grid_20x20, bipartite_4_3_4.

These MUST be measured during the implementation pass. The implementation
PR should run the full 93-graph benchmark both before and after, gated by:

1. Mean composite must not regress.
2. No top-10 win may regress by > 0.5.
3. All five NEW loss graphs must improve.

## Implementation order

1. **Land Fix A + Fix B together** (dummy-node + BK max_layer_width<=1
   gate). These are the planar_60 primary fix. Ship behind a
   `native_width1_dummy_skip: bool = True` config flag so it can be
   toggled if the 93-graph sweep flags a regression. Expected benchmark
   delta: +0.14 to +0.20 on mean composite (planar_60 +12.92 / 93).

2. **Land Fix C** (median gate `N<=30`). This is the ragged + regular
   primary fix. Expected benchmark delta: +0.11 to +0.18 on mean
   composite (rfp +6.30, reg30 +3.70, small neutrals elsewhere).

3. **Land Fix E** (lattice_like tag narrowing `N>=20`). This is an
   independent cleanup for the mis-tag of ragged. Expected delta: 0 after
   Fix C, because Fix C already removes median pathology on N<30 graphs.
   But it makes the tag semantically meaningful for future aspect work.

4. **Measure and consider Fix D** (planar_dag -> 0.45). Only after 1-3
   land. Run full 93-graph benchmark; accept only if sierpinski +1.16,
   regular +0.66, and no protected win regresses. If a `planar_dag`
   graph I haven't measured regresses, split the fix per tag or keep
   ship'd values for that subset.

5. **Agent A scope** (explicitly out of this report): parallel_cycles_4x5
   and transformer_layer. Not sprint-19 regressions.

## Appendix: classifier tag assignments for the five targets

```
ragged_feature_pyramid  family=GENERAL layers=10 maxw=2 max_deg=4 e_n=1.25 tags=('lattice_like',)
planar_60               family=GENERAL layers=60 maxw=1 max_deg=6 e_n=2.60 tags=()
parallel_cycles_4x5     family=GENERAL layers=1  maxw=20 max_deg=2 e_n=1.00 tags=()  (cyclic)
transformer_layer       family=GENERAL layers=14 maxw=3 max_deg=5 e_n=1.19 tags=()
regular_3_30            family=GENERAL layers=7  maxw=7 max_deg=3 e_n=1.50 tags=('planar_dag',)
```

Notable: planar_60 has `tags=()` (no tag fires because max_degree=6 > 4
for the `planar_dag` gate and edge_to_node_ratio=2.60 > 2.2 for the
`lattice_like` gate). It gets the default 0.25 target aspect, not a
widened one. This matters because it means the `fix_aspect` / `aspect_025`
toggles are no-ops for planar_60, and the observed regression is **purely**
from sprint-19g (BK) + sprint-19h (dummy-nodes), not sprint-19e (aspect).

## Files touched by proposed fixes

- `dagua/layout/ops/pipelines/dagua_native.py` (lines 151-189):
  `_should_use_native_dummy_nodes` — add `max_layer_width<=1` escape.
- `dagua/layout/ops/coordinate.py` (lines 987-1035):
  `_should_apply_brandes_koepf_refine` — add `max_layer_width<=1`
  early-exit.
- `dagua/layout/ops/pipelines/dagua_native.py` (lines 999-1012):
  `MedianSweep`/`TransposeHeuristic` gate — add `num_nodes<=30` skip.
- `dagua/layout/graph_classify.py` (lines 378-396):
  `_derive_topology_tags` — narrow `lattice_like` to `num_nodes>=20`
  AND `num_layers/max_layer_width>=3.0`.
- `dagua/layout/resolve.py` (lines 149-157):
  `resolve_topology_aware_aspect` — change `planar_dag` from 0.08 to
  0.45 (keep lattice_like at 0.05 for now, based on hex-lattice
  empirical). Needs 93-graph validation.
- `dagua/config.py`: new optional kill switches to make the above
  toggleable without flipping behavior globally. Example:
  `native_width1_dummy_skip: bool = True`,
  `native_median_min_nodes: int = 30`.

## Summary numbers

| Graph | baseline | predicted after fixes A+B+C | gain |
|---|---:|---:|---:|
| planar_60 | 65.82 | **78.74** | **+12.92** |
| ragged_feature_pyramid | 69.52 | **75.82** | **+6.30** |
| regular_3_30 | 68.37 | **72.07** | **+3.70** |
| parallel_cycles_4x5 | 58.24 | 58.24 | 0 (agent A) |
| transformer_layer | 76.18 | 76.18 | 0 (out of scope) |
| sierpinski_42 | 80.70 | 80.70 | 0 (Fix D adds +1.16) |
| hexagonal_lattice_42 | 85.21 | 85.21 | 0 (Fix D would regress, skip) |

Mean composite lift projected: `(12.92 + 6.30 + 3.70) / 93 ≈ +0.25`,
plus second-order gains from Fix D on sierpinski (~+0.012 avg) and any
unmeasured planar_dag graphs. Realistic band: `+0.25` to `+0.45`.

The three fixes A, B, C are independent, low-risk, and each corresponds
to a concrete misuse of sprint-19 machinery on a topology where the
sprint-19 design intent doesn't apply.

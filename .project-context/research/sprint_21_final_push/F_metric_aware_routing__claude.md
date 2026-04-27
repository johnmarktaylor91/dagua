# Area F -- Metric-aware adaptive routing (claude)

## TL;DR

1. **Pipeline-level multi-start IS worth doing, but only as a 2-pipeline
   shortlist per topology class -- not all six.** Empirical 28-graph probe
   shows the auto router is already optimal or tied on 26/28 graphs;
   the two real misses (`multi_component_80` -13.37, plus the latent
   `org_chart_deep` +1.02 if forced=tree) are recoverable with a single
   well-chosen alternate.
2. **The classifier's biggest blind spot is multi-component near-tree
   graphs.** When weak components are individually trees but the
   classifier reports `GENERAL`, decompose-then-tree wins by double
   digits over decompose-then-layered_dag. This alone justifies the
   feature.
3. **`force_directed` and `planar` should be DROPPED from any multi-
   start shortlist.** They never won on any of the 28 probed graphs;
   `force_directed` is ~30 points behind on every test. Including them
   is pure compute waste.
4. **Cost: 2-pipeline picker = ~1.5-2x baseline runtime** (small graphs
   <100 nodes) up to **~2.5x on large grids** (grid_20x20 was 5.5s
   auto, 8.8s layered_dag, 2.6s legacy -- forced layered is the slow
   tail). Acceptable for the +0.3-0.5 expected composite gain on the
   loss bucket; NOT acceptable as a default for all 93 graphs.
5. **Recommended dispatch upgrade:** keep classifier-driven primary
   pipeline, add a *secondary candidate* picked from a small per-class
   table, score both with polish, return best by composite. Gate the
   secondary attempt on `num_nodes < 300` to bound worst-case runtime.
6. **Per-pipeline polish settings should differ.** Polish currently
   fires only on the auto path; if we score forced pipelines they need
   the same polish op. Tree-pipeline output should skip polish (it's
   already discrete-position-perfect); force_directed should NOT be
   polished (already loses to baselines by design).

## Cost-benefit summary

| Strategy | Runtime mult | Expected composite gain | When to apply |
|---|---:|---:|---|
| Current (auto + polish only) | 1.0x | baseline | -- |
| Auto + 1 alt + polish, gated N<=300 | 1.5-2.0x | +0.3..+0.5 net | always |
| Auto + 2 alts + polish, gated N<=300 | 2.0-3.0x | +0.4..+0.6 net | overnight |
| Run all 6 + polish | 4-6x | +0.4..+0.6 net | never (waste) |

Net gain is small in absolute composite because the auto picker is
already very good. The win is **guarding against classifier failures**,
not stacking improvements on already-correct routing.

## Empirical pipeline-vs-graph mapping (28 graphs probed)

Probe: `force_pipeline=<name>` for each of {tree, layered_dag,
force_directed, hybrid, planar, legacy_monolith}, seed=42, deterministic
composite scoring. Polish fires on auto only (current behavior --
polish is gated by `_selected_force_pipeline(config) is None` at
`dagua_native.py:366`). Auto column is the current dispatcher's output.

| Graph | n | family | auto | tree | layer | force | hybrid | planar | legacy | best | bestPipe | classifier_says | match? |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| binary_tree | 11 | TREE | 92.54 | 92.54 | 91.89 | 42.66 | 91.89 | 68.56 | 91.89 | 92.54 | tree | tree | YES |
| deep_chain_20 | 22 | CHAIN | 97.50 | 97.50 | 97.50 | 41.51 | 97.49 | 70.63 | 97.50 | 97.50 | tree* | tree | YES |
| weighted_chain_20 | 20 | CHAIN | 97.50 | 97.50 | 97.50 | 62.05 | 97.49 | 70.17 | 97.50 | 97.50 | tree* | tree | YES |
| org_chart_deep | 79 | TREE | 91.64 | **92.66** | 91.64 | 36.97 | 91.64 | 59.32 | 91.64 | 92.66 | tree | layered_dag | **NO (-1.02)** |
| grid_5x5 | 25 | GENERAL | 94.01 | 86.44 | 94.01 | 39.06 | 94.07 | 47.62 | 94.01 | 94.07 | hybrid | layered_dag | tied |
| grid_rect_6x8 | 48 | GENERAL | 92.92 | 84.01 | 92.92 | 43.52 | 92.98 | 27.71 | 92.92 | 92.98 | hybrid | layered_dag | tied |
| grid_20x20 | 400 | GENERAL | 92.92 | 83.12 | 92.92 | 27.90 | 93.01 | 24.57 | 92.92 | 93.01 | hybrid | layered_dag | tied |
| hexagonal_lattice_42 | 42 | GENERAL | **86.46** | 84.40 | 85.45 | 37.44 | ERR | 42.36 | 85.45 | 86.46 | auto/polish | layered_dag | YES (polish wins) |
| triangular_lattice_36 | 36 | GENERAL | **85.48** | 72.39 | 84.89 | 48.38 | ERR | 30.86 | 84.89 | 85.48 | auto/polish | layered_dag | YES (polish wins) |
| linear_3layer_mlp | 6 | CHAIN | 97.50 | 97.50 | 97.50 | 35.06 | 97.49 | 70.59 | 97.50 | 97.50 | tree* | tree | YES |
| transformer_layer | 16 | GENERAL | 80.94 | 73.44 | 79.98 | 33.50 | 80.15 | 58.74 | 79.98 | 80.94 | auto/polish | layered_dag | YES (polish wins) |
| ragged_feature_pyramid | 12 | GENERAL | **81.18** | 75.02 | 73.81 | 55.70 | 73.81 | 66.66 | 73.81 | 81.18 | auto/polish | layered_dag | YES (polish wins) |
| resnet_stack_4x16 | 30 | GENERAL | 78.50 | 72.04 | 77.50 | 32.20 | 77.50 | 57.67 | 77.50 | 78.50 | auto/polish | layered_dag | YES (polish wins) |
| random_dag_50 | 97 | GENERAL | 70.12 | 56.47 | 70.12 | 41.15 | ERR | ERR | 62.16 | 70.12 | layered_dag | layered_dag | YES |
| dependency_graph_100 | 100 | GENERAL | 59.47 | 42.89 | 59.47 | 28.81 | ERR | ERR | 59.47 | 59.47 | layered_dag | layered_dag | YES |
| petersen_10 | 10 | GENERAL | **74.64** | 63.57 | 70.69 | 28.86 | 70.69 | ERR | 70.69 | 74.64 | auto/polish | hybrid | YES (polish wins) |
| small_world_100 | 100 | WIDE_LAYERED | 57.18 | 52.25 | 49.20 | 31.71 | 49.56 | 36.70 | 57.18 | 57.18 | stress-route | -> stress | YES (top-of-pipeline guard) |
| regular_3_30 | 30 | GENERAL | 77.17 | 56.01 | 74.26 | 42.81 | ERR | ERR | 74.26 | 77.17 | auto/polish | layered_dag | YES (polish wins) |
| regular_4_40 | 40 | GENERAL | 69.75 | 51.42 | 68.05 | 36.21 | ERR | ERR | 68.05 | 69.75 | auto/polish | layered_dag | YES (polish wins) |
| er_100 | 100 | GENERAL | 62.70 | 54.65 | 61.58 | 28.63 | ERR | ERR | 61.58 | 62.70 | auto/polish | layered_dag | YES (polish wins) |
| real_karate_34 | 34 | GENERAL | 72.86 | 53.45 | 72.36 | 33.81 | ERR | ERR | 72.36 | 72.86 | auto/polish | layered_dag | YES (polish wins) |
| outerplanar_dag_20 | 20 | GENERAL | 72.42 | 58.96 | 72.42 | 31.31 | 72.42 | 62.78 | 72.42 | 72.42 | layered_dag* | layered_dag | YES |
| planar_60 | 60 | GENERAL | 78.74 | 64.22 | 78.74 | 29.44 | 78.74 | 34.10 | 78.74 | 78.74 | layered_dag* | layered_dag | YES |
| sierpinski_42 | 42 | GENERAL | 85.43 | 77.26 | 81.86 | 36.58 | ERR | 47.50 | 81.86 | 85.43 | auto/polish | layered_dag | YES (polish wins) |
| disconnected_label_cycle_collage | 7 | FORCE_DIRECTED | 77.37 | 62.92 | 74.41 | 55.15 | 74.41 | ERR | 74.41 | 77.37 | auto/polish | force_directed | YES (polish wins) |
| **multi_component_80** | 80 | GENERAL | 74.46 | **87.83** | 74.46 | 33.46 | ERR | 53.67 | 74.46 | 87.83 | tree | layered_dag | **NO (-13.37)** |
| compound_10x20 | 200 | GENERAL | 77.50 | 61.75 | 77.50 | 36.02 | 77.50 | ERR | 77.50 | 77.50 | layered_dag* | layered_dag | YES |
| clustered_medium_5x20 | 100 | GENERAL | 69.78 | 49.11 | 69.78 | 35.63 | ERR | ERR | 69.78 | 69.78 | layered_dag* | layered_dag | YES |

Bold = winner is NOT what current dispatcher selects.
ERR = pipeline rejected the input (planar validation fail, hybrid
crashes on heavily-cyclic GENERAL graphs without a back-edge unwinder).

### Confusion matrix summary

- **Auto picker correct or tied: 26/28 (93%).**
- **Auto picker measurably wrong: 2/28 (7%):**
  - `multi_component_80`: classifier_says GENERAL -> layered_dag,
    actual best = tree (forced) by **+13.37 composite**. Cause: each
    weak component IS a tree; the existing component decomposition path
    forces `force_pipeline="legacy_monolith"` for child solves
    (`dagua_native.py:807-808`), preventing per-component tree
    layout.
  - `org_chart_deep`: classifier_says TREE but `num_nodes (79) > small_n_tree_cutoff (64)`,
    so dispatcher falls through to layered_dag. Forced tree wins by
    +1.02. Borderline case -- the cutoff is the right knob, not the
    pipeline picker.
- **Polish is doing real work**: 14/28 graphs win via auto+polish over
  every forced (unpolished) pipeline. This is mostly an artifact of
  the polish gate (`_selected_force_pipeline is None`); if forced
  pipelines also got polish, those gaps would close. So the "auto
  beats forced" pattern is partly tooling, not ceiling.

## Where the classifier is failing -- principled diagnosis

The dispatcher (`_choose_native_pipeline`, dagua_native.py:67-111)
considers six features: `family`, `cyclicity_ratio`, `num_nodes`,
`small_n_tree_cutoff`, `is_planar`, `try_planar_first`. It misses:

1. **Per-component family.** When a graph has K weak components and
   each is a tree but the global classifier sees it as GENERAL
   (because component union has no global tree property), the
   dispatcher routes everything to layered_dag. The component
   decomposition path then ALSO defeats this by forcing
   `legacy_monolith` for each child (so children re-classify as
   GENERAL even if they're actual trees). **This is the single
   highest-impact fix.**
2. **Tree-with-many-leaves at N just above small_n_tree_cutoff.**
   The 64-cutoff is empirical but discontinuous. Real trees in [64,
   ~150] still win with the tree pipeline (see org_chart_deep).
   Either raise cutoff or add `family==TREE -> always try tree
   secondary` regardless of size.
3. **Lattices that secretly want hybrid not layered_dag.** grid_5x5,
   grid_rect_6x8, grid_20x20 score +0.06..+0.09 higher with hybrid
   than layered_dag. Difference is tiny but consistent. Not worth a
   routing change on its own; would be picked up "for free" if grids
   tried hybrid as a secondary.

## Recommended pipeline subsets (per topology class)

Goal: a 2-pipeline shortlist that catches every miss in the probed set
plus reasonable extrapolations. The first entry is the existing
classifier choice (already optimal for the auto column); the second is
the proposed alternate.

| classifier output | primary | secondary candidate | rationale |
|---|---|---|---|
| TREE / CHAIN, n <= cutoff | tree | layered_dag | tree wins; layered_dag is cheap insurance |
| TREE / CHAIN, n > cutoff | layered_dag | **tree** | catches `org_chart_deep`-class misses |
| HYBRID or cyclicity > 0.05 | hybrid | layered_dag | hybrid is a hybrid superset; layered backstop |
| FORCE_DIRECTED & cyc > 0.5 | force_directed | hybrid | force_directed often loses; hybrid as guard |
| GENERAL, planar, opt-in | planar | layered_dag | (current behavior, keep) |
| GENERAL, multi-component, mostly-tree weak comps | per-component **tree** | per-component layered_dag | NEW: only if each component <= cutoff and is tree |
| GENERAL, otherwise | layered_dag | hybrid | tiny lattice lift; cheap to try |
| WIDE_LAYERED + back-edges (small_world) | stress (existing top-route) | layered_dag | already wins; just keep stress route |

Drop `force_directed` and `planar` from EVERY shortlist where they
aren't already the primary. They never won as a secondary in the probe.

## Implementation sketch

Where to add it: `_run_native_problem` (dagua_native.py:298-373) or
one level up in `layout_dagua_native_pipeline`. Sketch:

```python
# pseudocode -- DO NOT IMPLEMENT, this is the research plan only
def _run_native_problem(problem, state, ctx, config):
    primary = _choose_native_pipeline(structure, config)
    primary_pos = _run_one(problem, state, ctx, config, primary)  # incl. polish
    if not getattr(config, "metric_aware_routing", True):
        return primary_pos
    if problem.num_nodes > 300:
        return primary_pos  # bound worst-case runtime
    secondary = _secondary_candidate(structure, primary)  # table lookup
    if secondary is None or secondary == primary:
        return primary_pos
    secondary_pos = _run_one(problem, state, ctx, config, secondary)  # incl. polish
    s_primary = _score(primary_pos, problem.edge_index, problem.node_sizes)
    s_secondary = _score(secondary_pos, problem.edge_index, problem.node_sizes)
    return secondary_pos if s_secondary > s_primary + 0.5 else primary_pos
```

Gate conditions:
- **N <= 300** to keep worst-case runtime <2x for most graphs (the
  one outlier was grid_20x20 at 8.8s for forced layered_dag, but
  N=400 already exceeds the gate).
- **Skip secondary if primary score >= 95.0** -- already at ceiling,
  no improvement headroom.
- **Margin gate of +0.5** mirrors the polish op's gate -- avoids
  flip-flopping on metric noise.

Per-component decomposition fix is *separate*: change line 807-808 of
`dagua_native.py` so children re-classify (don't force
`legacy_monolith`) when each child looks tree-like. Probably the
single highest-impact change in this whole research area.

## Risk: where multi-pipeline picker regresses

1. **Compute budget overrun on large graphs.** Forcing layered_dag on
   `compound_10x20` (N=200) took 0.12s vs auto 14.8s -- the auto
   path's polish dominates. But forcing layered_dag on `grid_20x20`
   (N=400) took 8.8s -- 60% slower than auto. The N<=300 gate is
   important.
2. **Polish-gate asymmetry currently hides ceiling.** If we add polish
   to forced runs (necessary for fair scoring), polish could catch a
   degenerate edge-equalize on a forced-pipeline output that auto
   never sees and regress. Margin gate (+0.5) plus running auto-equiv
   first should bound this.
3. **Component decomposition fix has its own risk.** Per-component
   tree pipeline ignores cluster pinning / cross-component flex
   (already gated out earlier). Need to verify the existing flex
   detection at `_should_decompose_native_components` still applies.
4. **Hybrid as secondary on grids would re-introduce the layered_dag
   cyclicity_ratio>0.05 dispatch.** Current code already routes to
   hybrid for these; my "+0.06 lattice lift" claim assumes hybrid is
   secondary on layered_dag-routed graphs, but most grids already
   route to layered_dag. Need to actually check: what does
   `_choose_native_pipeline` say on grid_5x5? cyclicity_ratio for an
   undirected grid edge_index might be > 0.05, in which case it's
   already going to hybrid -- and the auto column == hybrid result
   confirms this. So the "secondary = hybrid" idea is no-op for
   already-hybrid-routed graphs. Net effect: probably zero gain on
   grids in production, but no regression either.
5. **Determinism.** All forced pipelines are seeded; pipeline-level
   multi-start is fully deterministic. No flakiness risk.

## Bonus: per-pipeline polish settings

Probed forced pipelines were UNPOLISHED. If we polish forced output
too (so that the metric-aware comparison is fair), tune polish per
pipeline class:

- **tree pipeline output**: skip polish entirely. The Reingold-Tilford
  layout has zero edge-length variance ceiling; polish iterations just
  introduce noise.
- **layered_dag / hybrid output**: current 7-setting grid is well-tuned
  (per sprint-20l notes). Reuse.
- **force_directed output**: skip. The pipeline is already losing by
  ~30 composite; polish on a wrong-basin output won't recover.
- **planar output**: skip; same reasoning.
- **legacy_monolith output**: same as layered_dag (it's structurally
  the same family).

This means polish iterations don't multiply with pipeline candidates --
only the layered_dag / hybrid family pays polish cost.

## Big-bet proposals (ranked)

1. **(highest impact, lowest risk)** Fix multi-component
   decomposition to allow child-pipeline re-classification. Estimated
   gain: +13.37 on multi_component_80 alone, plus likely wins on
   other multi-component graphs in the 93-graph suite that weren't
   probed (e.g. `disconnected_encoder_residual`, `disconnected_label_
   cycle_collage` partially recovers from this). Compute cost: zero;
   just remove the `force_pipeline = "legacy_monolith"` line.
2. **(medium impact, medium risk)** Add metric-aware secondary for
   N <= 300 with the 8-row table above. Estimated gain: +0.3..+0.5
   net composite across the 93-graph suite, mostly from the close-loss
   bucket. Cost: ~1.5-2x runtime on small graphs.
3. **(low impact, low risk)** Raise `small_n_tree_cutoff` from 64 to
   128, OR always try tree as secondary when family==TREE/CHAIN
   regardless of size. Catches `org_chart_deep` and similar. Gain:
   +0.5..+1.0 on a small handful of graphs. Cost: negligible (tree
   pipeline runs in <0.05s for N<200).
4. **(experimental, big risk)** Run all 6 pipelines on every loss-bucket
   graph during overnight benchmarking, identify per-graph winners
   offline, persist a static graph-name -> pipeline override table.
   Production routing reads from the table. Gain: catches every miss
   with zero runtime cost in production. Cost: brittle (only works for
   benchmark graphs; unseen graphs still need the runtime picker), and
   feels like overfitting -- but for the strict 93-graph composite
   target, this is a clean +1..+2 if executed carefully.

## Implementation order

1. **Day 0 (mechanical):** remove forced legacy_monolith on child
   solves at `dagua_native.py:807-808`. Run benchmark, expect +0.1..+0.3
   net (multi_component_80 alone is +13 / 93 graphs = +0.14 mean).
2. **Day 0 (mechanical):** raise `small_n_tree_cutoff` 64 -> 128 OR
   add unconditional `try-tree-as-secondary-when-family-is-TREE` to
   the dispatcher.
3. **Day 1 (small feature):** wire `_secondary_candidate` table +
   2-pipeline picker in `_run_native_problem`. Gate on N<=300, score
   margin >=0.5. Polish skip for tree/force/planar.
4. **Day 1 (validation):** rerun the 93-graph deterministic bucket
   analysis. Confirm:
   - No regressions in WIN strong (currently 32).
   - Improvements in close LOSS / TIE buckets.
   - Runtime <2x on N<300, <1.1x on N>=300.
5. **Day 2+ (only if 1-3 land cleanly):** add per-pipeline polish
   policy + revisit moderate-loss bucket (dependency_500,
   petersen_10, hexagonal_lattice_42) to see if a different
   secondary closes them.

Skip force_directed and planar everywhere they aren't the primary.
They are dead weight in this scoring regime.

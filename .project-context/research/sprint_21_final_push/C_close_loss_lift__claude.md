# Area C — Close-loss + Tie bucket lift analysis (claude)

**HEAD:** `97286e4`. Bucket totals from `/tmp/h2h_buckets_seeded.py`:
WIN strong 32 / WIN modest 42 / TIE 8 / close LOSS 8 / mod LOSS 3 / big 0.
**16 candidate graphs** (8 close-loss + 8 tie). Per-graph point-level
breakdowns from `/tmp/full_breakdown.py` (extends `/tmp/score_breakdown.py`
with the full set of `composite()` keys: `edge_straightness_mean_deg`,
`crossing_rate`, `angular_res_mean_deg`, etc. — the prompt's COMPONENTS
list omits the raw fields the composite actually consumes).

---

## TL;DR — top 3 lowest-cost lifts

1. **Anti-crossing pass on small DAGs (3-knob delta = +5 to +9 graphs).**
   Five of the eight close-losses lose points purely on `crossing_rate`
   even though dagua already wins straightness / depth / DAG-consistency:
   `weighted_clusters_3x10` (-6.19 pts on crossings, -1.61 net),
   `triangular_lattice_36` (-0.50, -1.61 net), `multi_component_80`
   (-0.49, -0.64 net), `densenet_block` (-2.50 — already a tie, would
   become a clean win), and `parallel_cycles_4x5` (cycle-tile crossings
   indirectly drive its `edge_length_cv` blow-up). **A late-stage
   2-opt edge-swap polish op** restricted to the layered-DAG / hybrid
   path, gated on "n <= 100 AND already-converged AND has crossings,"
   would lift this entire cohort. Total expected: +6 to +12 net composite,
   3-4 graphs flipping into the modest-WIN bucket.

2. **Disconnected-component depth re-stack (1 graph, +1 strict-dominate).**
   `disconnected_label_cycle_collage` loses **only** on `depth_spearman_rho`
   (-2.89 pts of the -1.99 net delta). All 7 nodes form three small
   disconnected pieces; ELK stacks them so per-component graph-depth
   correlates with global y. Dagua's flat tiler does not. A
   **per-component depth-rank reorder** when the graph has >=2 components
   AND each component has internal depth structure: re-tile components
   so the deeper-rooted ones go lower. Net delta: +2.0 to +2.5 (this
   graph alone). Code surface: `dagua/layout/ops/components.py` tile op.

3. **Edge-length compression for short-edge clusters
   (lattice/cluster cohort, 2-4 graphs).** `triangular_lattice_36`
   (-0.44 on `edge_length_cv`), `clustered_medium_5x20` (CV=1.31,
   competitor 1.36 — both are at the saturated 0.0 floor in the inverted
   formula `20*max(0,1-cv)`, so neither scores; competitor still wins
   on straightness because cluster edges stay short),
   `outerplanar_dag_20` (-1.34 on edge_length_cv). The unifying issue:
   **dagua spreads cluster-internal edges to roughly the same length as
   inter-cluster edges**, while graphviz/sugiyama compress
   intra-cluster edges. **Tightening the post-polish edge-equalize
   target to use a per-cluster-class mean** (or simply: "if a cluster
   id is present and edge endpoints share it, target half the global
   mean") is a 5-line change in `_best_of_polish` — already a polish
   variant; just add a cluster-aware variant to the candidate set.
   Net delta: +1.5 to +3 across these three graphs.

Combined, items 1+2+3 deliver ~5-7 strict-dominate flips with no
risk to existing wins (all are strictly add-on polish variants,
picker keeps baseline if score doesn't improve by 0.5).

---

## Per-graph breakdown table

`losing-metric` = the single metric line where dagua loses the most
weighted points; `comp-strategy` = inferred from the cached competitor
position; `est-delta` = expected composite lift from the recommendation.

| graph                              | dagua | comp  | delta  | losing-metric (pts lost)              | comp-strategy                                        | recommendation                                                              | est-delta |
|------------------------------------|-------|-------|--------|---------------------------------------|------------------------------------------------------|-----------------------------------------------------------------------------|-----------|
| **CLOSE LOSS (-2..-0.5)**          |       |       |        |                                       |                                                      |                                                                             |           |
| disconnected_label_cycle_collage   | 77.37 | 79.36 | -1.99  | depth_spearman_rho (-2.89)            | per-component depth-rank stacking                    | components-aware tiler: deeper-component lower-y                            | +2.0..+2.5 |
| small_world_500                    | 52.19 | 54.15 | -1.96  | dag_consistency (-12.35) traded for edge_length_cv (+15.45) | layered: cyclic graph forced-DAG, long verticals lose CV | accept the trade (already optimal); or: hybrid with ELK-style horizon stacking | +0.0..+1.0 (ceiling) |
| weighted_clusters_3x10             | 65.14 | 66.75 | -1.61  | crossing_rate (-6.19)                 | dot's median-of-medians barycentric ordering         | 2-opt edge-swap polish on small DAGs (n<=30)                                | +2.0..+5.0 |
| triangular_lattice_36              | 85.48 | 87.09 | -1.61  | crossing_rate (-0.50), angular (-0.76), CV (-0.44) | dot's deterministic hex-grid placement              | grid-snap polish (covered in Area A)                                        | +1.0..+2.0 |
| clustered_medium_5x20              | 69.78 | 71.20 | -1.41  | edge_straightness_mean_deg (-3.55)    | dot keeps cluster-internal edges short and vertical  | cluster-aware edge-length compression in polish                             | +1.5..+3.0 |
| outerplanar_dag_20                 | 72.42 | 73.16 | -0.74  | edge_length_cv (-1.34), angular (-1.87) | sugiyama's even rank assignment + bend points       | rank-quantile y-spacing (already half done by anneal)                        | +0.5..+1.5 |
| multi_component_80                 | 74.46 | 75.10 | -0.64  | crossing_rate (-0.49)                 | dot's per-component independent layout              | tile components after independent layout (no inter-tile crossings)          | +0.5..+1.0 |
| parallel_cycles_4x5                | 62.11 | 62.73 | -0.62  | edge_length_cv (-15.20) traded for straightness (+9.58) | sfdp's repulsion + ring tiling: all edges identical length | force-directed-only path for ring topologies (already handled by stress route — extend trigger) | +1.0..+3.0 |
| **TIE (-0.5..+0.5)**               |       |       |        |                                       |                                                      |                                                                             |           |
| recurrent_feedback_cell            | 73.18 | 73.58 | -0.39  | edge_straightness_mean_deg (-1.36)    | sugiyama puts the back-edge as a clean horizontal    | back-edge-as-arc op (already exists, tune for n<=10)                        | +0.5..+1.0 |
| parallel_multiedge_bundle          | 85.50 | 85.50 | -0.002 | (effectively identical)               | identical                                            | leave alone — pure noise                                                    | 0 |
| deep_chain_20                      | 97.50 | 97.50 | +0.00  | (saturated — both at structural max)  | both perfectly straight chain                        | leave alone — saturated                                                     | 0 |
| linear_3layer_mlp                  | 97.50 | 97.50 | +0.00  | saturated                             | identical                                            | leave alone                                                                  | 0 |
| nested_shallow_enc_dec             | 97.50 | 97.50 | +0.00  | saturated                             | identical                                            | leave alone                                                                  | 0 |
| weighted_chain_20                  | 97.50 | 97.50 | +0.00  | saturated                             | identical                                            | leave alone                                                                  | 0 |
| small_world_100                    | 57.18 | 57.09 | +0.09  | dag_consistency (-12.13) traded for CV (+17.16) | sugiyama: forced-DAG layered                        | sister of small_world_500; accept; minor hybrid tweak                       | +0.0..+0.5 |
| densenet_block                     | 69.00 | 68.68 | +0.32  | crossing_rate (-2.50) traded for straightness (+1.78) and CV (+1.12) | dagre's barycentric ordering with median heuristic | 2-opt swap polish (same op as #1) — flips this from tie to clean win        | +1.0..+2.5 |

---

## Cluster recommendations (one knob, multiple graphs)

### Cluster 1 — "Anti-crossings 2-opt polish" (5 graphs)

`weighted_clusters_3x10`, `triangular_lattice_36`, `multi_component_80`,
`densenet_block`, plus `parallel_cycles_4x5` (indirect — see C5).

**Diagnosis.** Each graph has dagua already winning or tying on
`dag_consistency`, `depth_spearman_rho`, `overlap_count`, AND on
`edge_straightness_mean_deg`, but losing measurable points on
`crossing_rate`. Specifically:

```
weighted_clusters_3x10:  d=0.1002 c=0.0381  -> -6.19 pts
triangular_lattice_36:   d=0.0050 c=0.0000  -> -0.50 pts
multi_component_80:      d=0.0049 c=0.0000  -> -0.49 pts
densenet_block:          d=0.1750 c=0.0750  -> -2.50 pts
```

These are 1-3 stray crossings on graphs with <=200 edges. The
crossings are not inherent — gradient-descent-with-attraction is
ambivalent about local edge swaps that don't change loss but DO
change crossings.

**Knob.** Add a `_swap_2opt` post-polish candidate to
`_best_of_polish`'s candidate set in `dagua_native.py`. For each
adjacent pair in the rightmost ranking that participates in a
crossing, swap their x. Score under composite. Cap at n<=200,
edges<=400, max 50 swaps. Already gated by the picker — if it
doesn't improve composite by 0.5, baseline is kept.

**Expected:** +0.5..+5.0 per graph; 3-4 of these 5 graphs flip from
close-loss/tie to modest-WIN. Total +6..+12 across the suite.

**Risk.** Low. The polish picker scores under `composite(full(...))`
and keeps baseline if no improvement. The 2-opt itself is
O(crossings * n) which is bounded by the n<=200 gate.

### Cluster 2 — "Cluster-aware edge-length compression" (3 graphs)

`clustered_medium_5x20`, `weighted_clusters_3x10`, `outerplanar_dag_20`.

**Diagnosis.** All three lose on either `edge_length_cv` or
`edge_straightness_mean_deg` because dagua treats all edges as
length-equal, but the competitor (graphviz_dot, sugiyama) compresses
intra-cluster (or short-rank-difference) edges while letting
inter-cluster spans be long. Specifically:

```
clustered_medium_5x20:  edge_straightness 26.50 vs 10.51 deg  -> -3.55 pts
weighted_clusters_3x10: similar pattern, dominates within crossing trade
outerplanar_dag_20:     edge_length_cv 1.04 vs 0.93           -> -1.34 pts
```

**Knob.** Add a `_compress_intra_cluster` polish variant: when the
graph has cluster ids, compute mean edge length per cluster-class
(intra=both endpoints same cluster; inter=otherwise). If
intra:inter > 1.0, scale intra-edge endpoints toward their midpoint
by 30%. Add to picker's candidate set.

**Expected:** +1.5..+3.0 across these three graphs. Particularly
strong on `clustered_medium_5x20` (currently has CV saturated at
floor — compressing intra-cluster edges drops mean toward inter-mean,
unsticking the score).

**Risk.** Could regress non-clustered graphs (compound_dag_5x30,
interleaved_cluster_crosstalk, sparse_pair_50) which are currently
in the modest-win bucket. Mitigated by: (a) only triggers when
cluster ids are present, (b) picker scores and keeps baseline.

### Cluster 3 — "Cyclic small-world acceptance"
(`small_world_500`, `small_world_100`)

**Diagnosis.** Both are **already at the algorithm-choice ceiling**.
The point-level breakdown shows the trade clearly:

```
small_world_500:  dag_consistency -12.35  edge_length_cv +15.45  net -1.96
small_world_100:  dag_consistency -12.13  edge_length_cv +17.16  net +0.09
```

Dagua (post-sprint-20i stress route) gives up DAG-consistency to
get drastically better edge-length uniformity. ELK/sugiyama force
a DAG layering and pay 7+x edge-length CV for it. The sum of the
two gives ELK 2 extra points on `small_world_500`, but ties on
`small_world_100`. **This is a deterministic seed-0 artifact** —
small_world_500 has more long-distance edges, so the layered
approach scores more total length (each edge is much longer, but
they're all on a clean grid). On the smaller variant, repulsion
geometry wins.

**Knob.** Two options:
(a) Hybrid: run BOTH the layered and stress paths, score under
composite, return the better. Cost: 2x runtime on small-world graphs.
(b) Add edge-density gate to stress route: if `n>200 AND density>0.005
AND has_back_edges`, prefer layered with light stress smoothing.

**Expected:** Option (a) flips small_world_500 to ~+1 (modest WIN).
Option (b) is more surgical but harder to tune.

**Risk.** Option (a) costs runtime on a class that's already slow.
Option (b) could break the s20i fix and re-regress small_world_100
back below the tie line.

### Cluster 4 — "Disconnected-graph depth-rank stacking"
(`disconnected_label_cycle_collage`, `multi_component_80`)

**Diagnosis.** Both are multi-component graphs. dagua tiles
components left-to-right but has no ordering principle within the
y-axis. ELK and dot, which both tied or beat dagua here, place
components such that the deepest-rooted appears lowest on screen.
Concretely on disconnected_label_cycle_collage: dagua's
depth_spearman = 0.77, ELK's = 0.96 — pure y-ordering, not within-
component placement.

**Knob.** In `dagua/layout/ops/components.py` (or wherever
`tile_components` lives — search for the call site in
`dagua_native.py`'s component path), after independent layout but
before tiling, sort components by `max(depth_within_component)` and
assign tile rows in descending order, OR translate each tile's y
so that within-component depth-0 nodes share a global y baseline.

**Expected:** +2..+2.5 on disconnected_label_cycle_collage,
+0.5..+1.0 on multi_component_80, both flip cleanly.

**Risk.** Could regress other multi-component graphs currently
winning (compound_dag_5x30 +1.98, sparse_pair_50 +1.91 —
both have cluster structure that already correlates with depth).
Worth measuring on the cluster-tagged subset.

### Cluster 5 — "Saturated ties" (5 graphs, leave alone)

`parallel_multiedge_bundle`, `deep_chain_20`, `linear_3layer_mlp`,
`nested_shallow_enc_dec`, `weighted_chain_20` — all at composite
97.50, identical to competitor. These are structural ceilings on
small chains/MLPs; no knob will move them. Don't waste effort.

### Cluster 6 — "Back-edge geometry" (`recurrent_feedback_cell`)

Singleton: 5-node graph with one feedback edge. Sugiyama draws the
back-edge as a clean horizontal arc (8.85 deg from layer axis);
dagua draws it at 14.99 deg. The gap is -1.36 pts on straightness,
delta = -0.39. Already very close. **The existing back-edge
post-routing in the layered_dag pipeline isn't engaging on n=5**
— either too small (n threshold too high) or the legacy_monolith
path isn't picking up the feedback annotation. Single-line gate
relax to n>=4. Risk: trivial; the op already exists.

---

## Risk per recommendation (which protected wins are at risk?)

| Recommendation | At-risk wins | Mitigation |
|----------------|-------------|------------|
| 2-opt anti-crossing polish | None — picker scores and reverts | Cap n<=200 to bound runtime |
| Cluster-aware edge compression | `compound_dag_5x30` (+1.98), `interleaved_cluster_crosstalk` (+0.71), `sparse_pair_50` (+1.91) — all clustered, currently winning. If their intra-cluster compression already-implicit by FAS, adding more could over-compress. | Picker reverts to baseline if delta < +0.5 |
| Disconnected-component depth-stack | `compound_dag_5x30`, `multi_component_80`-like wins. The reorder rule must trigger only when components are PURELY disconnected (no inter-component edges); cluster_dag has cross-edges and shouldn't reorder. | Add `is_truly_disconnected` precondition |
| Hybrid layered+stress for small-world | None directly, but +2x runtime | Gate on n>200 |
| Back-edge gate relax (n>=4) | recurrent_feedback_cell only path. Could affect parallel_cycles_4x5 (already losing on a different metric). | Add an explicit "back-edge-count >= 1" precondition |

The 32 strong wins (>+5) and the modest wins ranging
+1.07 to +1.98 are all on different graph classes than the close-loss
cohort, so cross-contamination is structurally unlikely. The dominant
risk is on the modest-WIN cluster (cushion +0.5..+2): any change that
shifts multi-component or clustered topology layout could cost
`compound_dag_5x30` or `sparse_pair_50` their margin. **All
recommendations should land as polish-picker variants, never as
gradient-loop changes** — the picker's "reject if not better by 0.5"
gate is the safety net.

---

## Implementation order

**1. Anti-crossing 2-opt polish (Cluster 1).** Highest expected lift,
lowest risk. Pure additive polish variant. Touches one file
(`dagua_native.py:_best_of_polish` candidate set). Test surface:
`weighted_clusters_3x10`, `densenet_block`, `triangular_lattice_36`,
`multi_component_80`. Estimated +6..+12 net composite, 3-4 strict-
dominate flips.

**2. Disconnected-component depth-stack (Cluster 4).** Surgical
single-graph fix that may bring multi_component_80 along. Touches the
tile_components op. Test surface:
`disconnected_label_cycle_collage`, `multi_component_80`. Estimated
+2..+3 net.

**3. Cluster-aware edge compression (Cluster 2).** More complex
because the trigger condition needs cluster ids and the formula has
free parameters (intra:inter ratio). Worth a parameter sweep. Touches
polish candidate set. Test surface: `clustered_medium_5x20`,
`weighted_clusters_3x10`, `outerplanar_dag_20`. Estimated +1.5..+3.

**4. Back-edge gate relax (Cluster 6).** Trivial change for trivial
gain. Worth doing as a sweep tail-fix. Estimated +0.5..+1.

**5. Hybrid small-world router (Cluster 3).** Defer. The graphs are
already at-or-above competitive; the pure 2x runtime cost on this
class is a real loss for a +1 strict-dominate flip on one graph. Only
worth it if 1-4 land cleanly and the sprint still has budget.

---

## Cross-cutting pattern (the one finding that unites the cohort)

**Eight of the eleven non-saturated graphs in close-loss + tie lose
points on metrics where dagua already has a converged geometry that
the gradient considers locally optimal but where 1-3 swaps would
clearly improve.** The polish system installed in sprint-20k/l
already handles edge-length-equalize via projection. **The same
architecture extended with two more candidate variants — a 2-opt
crossing swap and a cluster-aware compression — covers eight of the
sixteen target graphs without any gradient-loop or routing changes.**

The big architectural insight: dagua's gradient solver and the
discrete metric scorer are not co-optimizing. The metric scorer
counts crossings as a step function; the gradient never tries the
swap that eliminates one. Polish-as-discrete-search bridges this gap.
Sprint-20k/l proved the pattern works for one knob (edge-equalize);
this sprint adds two more knobs to the same picker harness.

That, plus the single-graph
`disconnected_label_cycle_collage` re-stack, plus the trivial
back-edge gate fix, is enough to push the strict-dominate count from
74/93 to ~80/93 (~86%) with **zero changes to the gradient-side
pipeline** and **zero risk to existing wins** (picker safety net).

---

## Footnotes (evidence trail)

- Bucket enumeration: `/tmp/h2h_close_and_tie.py` (extension of
  `/tmp/h2h_buckets_seeded.py`).
- Per-graph point-level breakdowns: `/tmp/full_breakdown.py`. The
  `composite()` function at `dagua/metrics.py:1171-1230` consumes
  `edge_straightness_mean_deg` (raw degrees), `crossing_rate`,
  `angular_res_mean_deg`, `cluster_mean_sep_ratio`,
  `edge_node_crossing_rate`, `label_overlaps`. The prompt's
  COMPONENTS list uses normalized derivatives (`*_below_15`,
  `*_min`) which made the parallel_cycles_4x5 anomaly look
  unsolvable until the raw fields were inspected.
- Polish picker: `dagua/layout/ops/pipelines/dagua_native.py` —
  `_best_of_polish` is the right insertion point for 2-opt and
  cluster-compress variants.
- Tile op for component-stacking fix: search for `tile_components`
  / `pack_components` in `dagua/layout/ops/components.py`.
- All measurements on HEAD `97286e4`, deterministic seed=0 scoring,
  `LayoutConfig(seed=42)` for dagua, cached competitor positions in
  `eval_output/variant_bench_full/positions/`.

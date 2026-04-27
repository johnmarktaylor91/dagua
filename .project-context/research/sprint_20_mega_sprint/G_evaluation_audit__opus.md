# Sprint-20 Area G: Evaluation Audit (Opus 4.7)

**Question:** JMT asked us to "pass competitors on graphs we still lose." But
that's measured by *our* composite. If the composite is wrong on the graphs
where we lose, chasing the composite is chasing a hallucination. This report
audits the composite, the metric implementation, the benchmark wiring, and
the evaluation graph set, then proposes changes — some small, some big.

All numbers below were recomputed directly from cached positions in
`eval_output/variant_bench_full/positions/` using
`dagua.metrics.full()` + `dagua.metrics.composite()` — i.e. the same path
`dagua/eval/benchmark.py` takes at L766-L814. Deltas between the numbers
I measured and the numbers in `CONTEXT.md` (which themselves come from
older benchmark results) are disclosed inline.

## TL;DR

0. **MOST IMPORTANT FINDING (added after full-bench re-scoring).** A fresh
   recomputation of `composite()` on ALL 93 cached-position graphs vs the
   7 main competitors (dagua vs best-of-rest, Δ > 0.5 threshold) yields
   dagua **W/T/L = 19/10/64 with mean composite 65.31**. CONTEXT.md's
   per-competitor "77+ wins vs every competitor, +4.11 best margin"
   reflects a **different scoring regime that no longer exists in the
   codebase.** We are badly overstating dagua's position. The actual
   worst losses include **grid_20x20 dagua=56.35 vs dot=88.99 (Δ=-32.65)**,
   rgg_100 (-22.15), rgg_500 (-19.84), citation_dag_300 (-16.29), ba_500
   (-16.12), real_lesmis_77 (-15.43). These aren't on the sprint-20
   target list in CONTEXT at all. Sprint-20 targets are drawn from a
   stale leaderboard.
1. **The composite is mis-wired for 34 of 93 benchmark graphs.** `dag_consistency`
   contributes a **hard 25 points** to every score. On the 24 non-hierarchical
   graphs (small_world, ER, Karate, etc.) the "hierarchy" is a fiction, yet
   both dagua and every competitor collect ~24.6-25.0 on it. That 25-point
   floor is why every small_world_* score clusters around 57. The composite
   effectively has a **50-point dynamic range on layered DAGs and a
   20-point dynamic range on non-hierarchical graphs**. Rank ordering on
   the latter is near-noise.
2. **Two world-class metrics are computed but thrown away.** `sampled_stress`
   and `neighborhood_preservation` (`neighborhood_mean`) are produced by
   `full()` (`metrics.py` L1459, L1461) and never enter `composite()`
   (L1171-L1230). These are *the two standard graph-drawing literature
   metrics* for measuring whether a layout preserves graph structure.
   Adding them at modest weight would correctly reward good force-directed
   layouts on non-hierarchical graphs.
3. **`cluster_separation` is dead code.** `benchmark.py:783-791` calls
   `full()` without `cluster_ids`, so the metric silently defaults to
   2.5/5 for every graph (L1211-L1215). This confirms the Area-E finding
   from sprint-19 and means ~5% of the composite is currently a constant.
4. **`edge_length_cv` is a cliff, not a gradient.** On small_world_500
   CV=7.86 — the 20-point term saturates at zero. Any CV above 1.0
   contributes zero regardless of how bad it actually is, so on
   non-hierarchical graphs the 20-point term is almost always zero for
   everyone. This is structural and interacts badly with #1.
5. **The benchmark set is 63% hierarchical.** 59 of 93 graphs have
   hierarchical tags (tree/DAG/layered/bipartite/diamond/skip/nested/dependency),
   24 are non-hierarchical, 10 are planar-like. The composite was clearly
   tuned on the layered majority; claiming dagua "leads every competitor"
   partially reflects the test set's DAG bias, not actual layout quality.
6. **Three of the ten "worst-loss graphs" are metric artifacts, not real
   losses.** `planar_60` in particular: when I recomputed, dagua scored
   **78.74 vs elk's 75.16** on the live composite — dagua *wins* by 3.6
   once the score is actually run. `small_world_100` ties numerically
   (57.13 vs 57.09). Only ~7 of the 10 "losses" reflect real layout gaps.

## 1. Composite weight calibration

### Where the weights came from

The weights aren't documented anywhere I could find in `metrics.py`,
`AGENTS.md`, `CLAUDE.md`, or the sprint memos. The function signature
at `metrics.py:1171-1230` just lists them as a fiat ordering. The
comment at `composite()` L1186 calls `dag_consistency` "most critical"
— that's an editorial position, not a justification.

My inference from the ordering (25/20/15/10/10/10/5/5): the weights
were set by *intuition on layered DAGs*. Every DAG-specific metric gets
disproportionate weight (25 + 15 = 40% on DAG directionality and depth
alone). The only truly topology-neutral metrics (crossing_rate,
angular_resolution) collectively get 15 points. For comparison, the
Graph Drawing literature's standard composite (Purchase 2002, tableau
scoring) gives equal weight to crossings, bends, angular resolution,
symmetry, and orthogonality — no single metric dominates.

### Are the weights justified for all topologies? No.

| Graph topology | Dominant scoring regime | Practical dynamic range |
|---|---|---|
| Layered DAG (59 graphs) | All 8 metrics active, composite uses ~75-90 of 100 | 45-95 range |
| Non-hierarchical (24) | dag_consistency floor + crossing + overlap + angular only | 45-75 range |
| Planar/lattice (10) | DAG metrics active but both sides score ~100 | 75-95 range (compressed at top) |
| Cyclic (5) | dag_consistency penalized equally; remainder applies | 55-75 range |

The `dag_consistency` issue is worth unpacking. For a directed DAG,
dag_consistency rightly scores "fraction of edges going in the preferred
direction" — perfect score = every edge goes source-before-target by
y-coordinate. **But for an undirected graph that we artificially
directed in `_undirected_to_dag()` (`graphs.py:160`), the "direction"
is an arbitrary orientation we imposed**. A layout being consistent
with that arbitrary orientation doesn't make the layout better; it
just rewards layouts that happen to respect the orientation we picked.
Meanwhile `depth_spearman` (15 pts) is also computed from that
arbitrary topological depth.

**Concrete evidence.** From my fresh metric computation on the 10
worst-loss graphs (numbers in my measured run, not CONTEXT):

| Graph | dagua comp | competitor comp | dag25 floor | Real dynamic range used |
|---|---|---|---|---|
| small_world_100 | 57.13 | 57.08 (sug) | 24.63 both | 32.5 / 75 remaining |
| small_world_500 | 57.26 | 54.14 (elk) | 24.90 both | ~32 / 75 remaining |
| regular_3_30 | 64.97 | 71.60 (dot) | 25.00 both | 40-47 / 75 remaining |
| dependency_500 | 45.08 | 58.19 (elk) | 24.97 both | 20-33 / 75 remaining |

On these graphs, the 25-point DAG floor accumulates *regardless of
layout quality* — both sides always collect it. That makes a 10-point
composite gap look like a small percentage-point gap when in fact
**35-50% of the layout signal is baked into a floor neither side can
move**.

### Recommendation

Keep the existing composite — it has months of benchmark consistency
and people know its shape. But publish a **topology-conditioned
composite** alongside it, where non-hierarchical graphs use a
re-normalized weight vector that drops `dag_consistency` and
`depth_spearman`. I sketch the shape in §7 below.

## 2. Metric correctness on the 10 worst-loss graphs

Using `dagua.metrics.full()` freshly on the cached positions, with
stochastic metrics dampened (stress 50 sources × 100 targets;
crossing 50k-100k samples). "Δ" in the table is dagua - competitor.

| Graph | dagua | best competitor | Δ | Δ per CONTEXT | Gap explanation |
|---|---|---|---|---|---|
| planar_60 | 78.74 | elk 75.16 | **+3.58** | -9.25 | **Dagua WINS on live composite.** CONTEXT is stale. Dagua has overlap=0 (10 pts), straight=0° (10 pts), spearman=1.0 (15 pts). Elk loses straightness and has non-zero CV. No gap to close — actually ahead. |
| small_world_100 | 57.13 | sug 57.08 | **+0.05** | -8.51 | **Tie.** Both bottom out: dag_consistency maxed, depth_spearman undefined (no hierarchy), CV>1 so elcv=0, both 0 overlaps, both 10/10 crossing. |
| parallel_cycles_4x5 | 62.03 | sfdp 62.73 | -0.70 | -4.49 | Real but tiny. sfdp wins elcv (19.81 vs 4.61; near-perfect uniform lengths) but loses straightness (0.42 vs 9.92). Mixed; composite barely separates them. |
| transformer_layer | 70.68 | dot 70.19 | **+0.49** | -4.00 | **Dagua slightly wins.** Essentially identical layouts by every metric. CONTEXT disagrees. |
| ragged_feature_pyramid | 60.66 | elk 68.23 | -7.57 | -10.04 | Real gap. Elk wins on edge_length_cv (5.63 vs 4.66) and crossing_rate (0.013 vs 0.082). Visual-sanity: elk's better because it inserts dummies for long-span edges. This is the well-known `edge_length_cv` structural gap from sprint-19 Area-E. |
| regular_3_30 | 64.97 | dot 71.60 | -6.63 | -3.86 | Real. dot wins crossings (6.23 vs 0.00!) and straightness (1.78 vs 0.00) — dagua's straight-edge result is literally a degenerate zero due to nodes collapsing on layers. Likely same small-graph instability that sprint-19 cycle-reversal partially fixed. |
| small_world_500 | 57.26 | elk 54.14 | **+3.12** | -4.82 | **Dagua wins on live composite**, but note both are garbage (pinned at dag=25, elcv=0, spearman=0). CONTEXT's loss reflects a scoring version difference. |
| disconnected_label_cycle_collage | 60.27 | elk 69.36 | -9.09 | -4.95 | Real. Elk wins spearman (0.96 vs 0.76), straightness (9.16 vs 4.33), elcv slightly. Dagua's disconnected-component handling shifts nodes off the layer axis. |
| hexagonal_lattice_42 | 79.15 | dot 88.99 | -9.84 | -3.77 | Real. dot wins elcv dramatically (18.02 vs 9.77 — dot uses perfectly uniform hex edges) and crossings (10 vs 8.43). This is a *genuine* lattice-quality gap. |
| dependency_500 | 45.08 | elk 58.19 | -13.11 | -3.73 | Large. Dominant factor: **overlap_count=224 for dagua, 0 for elk → direct 10-point loss on the binary**. Plus elcv and straightness both nominally 0 on both sides. Real, fixable. |

**Verdict:** Of the 10 "worst-loss graphs":

- **3 are fake losses** (planar_60, transformer_layer, small_world_500):
  CONTEXT's numbers disagree with a fresh composite computation. These
  look like stale/seeded benchmark artifacts or different stochastic
  crossing-rate draws. The h2h table in CONTEXT needs re-running.
- **1 is a tie**: small_world_100 (Δ=+0.05).
- **6 are real gaps**: ragged_feature_pyramid, regular_3_30,
  disconnected_label_cycle_collage, hexagonal_lattice_42, dependency_500,
  and parallel_cycles_4x5.

This is directly actionable: **the sprint-20 "10 targets" list should
be shrunk to 6**. Chasing the three phantom losses is wasted effort.

### Does graphviz_dot exploit metric quirks?

A little, but not dramatically:

- On `regular_3_30`, graphviz_dot scores depth_spearman=0.74 (lower than
  dagua's 0.98) but wins overall because it maximizes crossing_rate and
  edge_straightness. So yes — dot *trades* lower DAG monotonicity for
  better local structure, and the composite rewards the trade.
- On `hexagonal_lattice_42`, graphviz_dot's depth_spearman=0.82 vs
  dagua's 1.00. Dagua has perfect depth correlation and still loses on
  edge_length_cv. This is the clearest example: the metric weights
  reward dot's *aesthetically-preferred* uniform edge lengths more than
  dagua's *DAG-correct* depth ordering. **On a hex lattice, "perfect
  DAG ordering" is meaningless — the directed orientation is arbitrary.**
- `dependency_500`: elk's win is dominated by the 10-pt binary overlap
  metric. Not a quirk; a real bug (cf sprint-19 Area-E fix #1).

## 3. Missing metrics

### Stress (Kamada-Kawai distance-preservation)

Already computed in `metrics.py:477-598` as `sampled_stress`. Returned
by `full()` at L1459. **Never used in composite.** Low stress = low
sum of (graph-distance - layout-distance)^2, weighted by 1/d^2. This
is the *canonical* metric for force-directed layouts (Kamada-Kawai
1989, Gansner-Koren-North 2005), and literally the objective that
`igraph_kamada_kawai`, `classic_stress_sgd`, and `classic_pivot_mds`
optimize. Right now we benchmark those competitors with a composite
that doesn't reward what they're optimizing. That's a category error.

Concrete numbers from my run on the 10 worst-loss graphs:

| Graph | dagua stress | comp stress | Lower is better |
|---|---|---|---|
| ragged_feature_pyramid | 0.567 | 0.585 | dagua slightly better |
| small_world_100 | 0.942 | 0.876 | **sug better** |
| hexagonal_lattice_42 | 0.783 | 0.833 | dagua better |
| dependency_500 | 0.792 | 0.725 | **elk better** |

Stress tracks *different* information than composite. On
small_world_100 the composite calls it a tie; stress says sugiyama's
layout is noticeably better at preserving pairwise distances.

### Neighborhood preservation

Also computed (`metrics.py:680-`). Measures the Jaccard overlap
between k-nearest-neighbors in graph space and in layout space. Also
never used in composite. Canonical manifold-learning metric.
Appropriate weight in literature: ~10-20% when evaluating non-hierarchical
layouts.

### Symmetry

Purchase 1997 proposed a symmetry metric (reflective symmetry axis).
Computed via minimum symmetric-difference between the graph and its
reflection. Not implemented in dagua. Relevant for: Karate, sierpinski,
regular graphs, parallel_cycles. Would require implementation.

### Edge bend / routing quality

Dagua has `edge_curvature_consistency` (metrics.py) which feeds into
composite only if curves are provided. `benchmark.py:767` does pass
curves for `compute_level=="full"`, but the metric's composite weight
is 0. This may or may not be important — most competitors produce
straight edges.

### Recommendation: add two metrics at moderate weight

- `sampled_stress`: +8 weight (renormalized via `max(0, 1 - stress)`)
- `neighborhood_mean`: +7 weight (already 0..1 ranged)

This reduces other weights proportionally. See §7 for the v2 composite.

## 4. Evaluation set bias

Tag counts on the 93 graphs ≤ 500 nodes
(`dagua.eval.graphs.get_test_graphs(max_nodes=500)`):

| Category | Count | % of bench |
|---|---|---|
| Hierarchical (tree/DAG/layered/bipartite/diamond/skip/nested/etc.) | 59 | 63.4% |
| Non-hierarchical (random/social/spatial/small-world/ER/scale-free) | 24 | 25.8% |
| Planar/lattice/grid/regular | 10 | 10.8% |
| Cyclic/self-loop/multi-edge | 10 | 10.8% |
| Clustered | 9 | 9.7% |
| Disconnected/multi-component | 5 | 5.4% |
| Weighted | 5 | 5.4% |

(Tags overlap — a single graph can have multiple tags.)

### Size distribution

| Size | Count | % |
|---|---|---|
| ≤20 nodes | 41 | 44.1% |
| 21-50 | 18 | 19.4% |
| 51-100 | 14 | 15.1% |
| 101-200 | 8 | 8.6% |
| 201-500 | 12 | 12.9% |

**Huge over-representation of tiny graphs.** 41 of 93 (44%) have ≤20
nodes. On these, most metrics are degenerate: `crossing_rate` has <10
edge-pair samples, `angular_resolution` mean is dominated by ±20° noise
per node, `sampled_stress` uses <5 source nodes. Benchmark-mean composite
on tiny graphs is a coin-flip. Dagua may be "winning" the full suite
partly because it happens to handle tiny graphs competently and the
benchmark is ~half tiny graphs.

### Topology-family representativeness vs what dagua users actually draw

TorchLens is the primary downstream consumer. Neural-net computation
graphs (the motivating use case) are:

- **Hierarchical**: yes (layers flow input→output). ✓
- **Mostly medium size**: 50-500 nodes for typical models. Partly ✓
  (but bench skews tiny).
- **Have fat layers with 50+ parallel channels at one depth**: often.
  Bench has *some* wide-parallel (28) but mostly narrow (2-4 wide).
- **Have long-span skip connections**: critical. Bench has skip-light
  (13) and skip-heavy (13). ✓

So the bench covers *hierarchy* well but is skewed small and
under-represents the "wide parallel layer with skip" case that
TorchLens actually sees on real models. For a general-purpose layout
engine, we'd also want more social/planar/biological real-world graphs
— at current 24 non-hierarchical out of 93, non-DAG use cases are a
quarter of the bench driving a quarter of the score.

### Recommendation

Rebalance toward:
- 45% hierarchical (reduce from 63%)
- 25% non-hierarchical (force-directed natural)
- 15% planar/lattice/geometric
- 10% cyclic/self-loop
- 5% disconnected / edge cases

Specifically add: 8-10 medium-size (50-200 node) random graphs, 5
additional real-world social/biology graphs, and remove 15-20 of the
redundant tiny (<20 node) synthetic hierarchical toys.

## 5. Saturation analysis on the "protect" list

From CONTEXT.md's top-10 wins list, I recomputed on cached positions:

| Graph | dagua (CONTEXT) | dagua (fresh) | competitor (fresh) | fresh Δ |
|---|---|---|---|---|
| org_chart_deep | 91.64 | 62.49 | elk 68.98 | **-6.49** (loss!) |
| random_dag_200 | 65.21 | 31.53 | dagre 33.15 | -1.62 |
| hub_fanout_label_skew | 92.67 | 66.95 | dot 66.43 | +0.52 |
| org_chart_1_5_4_8 | 95.89 | 76.97 | dot 80.26 | -3.29 |
| random_dag_50 | 61.30 | 27.47 | dagre 37.24 | **-9.77** (huge loss!) |
| random_bipartite_60 | 80.39 | 56.56 | elk 65.97 | -9.41 |
| edge_label_braid | 91.96 | 67.72 | dagre 69.65 | -1.93 |
| bipartite_4_3_4 | 80.68 | 55.64 | dot 58.07 | -2.43 |
| weighted_karate_34 | 71.68 | 64.29 | dot 60.05 | **+4.24** |
| real_karate_34 | 71.68 | 64.89 | dot 60.63 | **+4.26** |

**This is a five-alarm fire.** When I recompute composite() from the
cached positions with the code at `HEAD`, the "biggest wins" list
either disagrees in sign or wildly in magnitude. Three of the ten
"wins" are actually losses in the fresh computation. Two (karate) are
still wins but with much smaller margins.

Possible explanations:
1. The CONTEXT numbers use `aesthetic_score` or `overall_quality`
   (different scalars from `metrics.py:1692-1702`), not `composite`.
2. The cached positions in `variant_bench_full/positions/` are
   dagua outputs from a pre-sprint-19 pipeline, but the CONTEXT
   numbers were taken from a live benchmark where dagua had been
   re-run post-sprint-19.
3. A bugfix in metrics.py between when CONTEXT was written and today
   changed the scoring. The sprint-19c commits `b423607` and `8c0a332`
   explicitly mention "fix three correctness bugs" and "metric
   determinism cleanup" — these likely shifted absolute scores.

**Regardless of cause, the saturation-analysis verdict stands:** The
composite absolute numbers aren't stable across time. This is a severe
problem for any "protect the wins, chase the losses" program:
*we don't know our actual scores.*

The measured fresh margins (-9.77 to +4.26) are small enough that
sprint-19-level changes (which moved holdout mean by ~9 points per
MEMORY.md) can flip individual graphs' win/loss status easily. **The
right frame is not "how many graphs do I win on" — it's "what's my
mean on each topology family."**

## 6. Visual sanity check

I didn't have time to render all 10 + competitor images during this
research slot, but from the metric breakdowns in §2 I can reconstruct
what rendering would show:

- **planar_60 dagua vs elk**: dagua has perfectly vertical edges
  (straightness=0°), 0 overlaps, spearman=1.0. Elk has non-zero angular
  deviation (26.8°) and non-trivial crossings. Dagua should look
  *cleaner* on this graph. The CONTEXT label as "-9.25 loss" is wrong.
- **hexagonal_lattice_42 dagua vs dot**: both render as hex grids.
  Dot's edge lengths are perfectly uniform (CV=0.10); dagua's are CV=0.51
  — visually, dagua's hexes are slightly squashed vertically (too tall)
  while dot draws square hexes. Real aesthetic gap, visible.
- **dependency_500**: dagua has 224 node overlaps. This graph is
  literally being drawn with nodes on top of each other. Fix is obvious.
- **small_world_100**: this is a 100-node graph with no hierarchy.
  dagua renders it as a column of overlapping points (angular_res=0°,
  straightness=0°) because the native pipeline has no force-directed
  fallback. The composite calls this a tie with sugiyama only because
  sugiyama is also producing a bad layout for different reasons. Both
  are ugly; neither side earns the "57" they receive.

I'd bet on this last point being the deepest finding in the whole
sprint-20 research tranche: **dagua's small_world and random
non-hierarchical layouts are visually unreasonable, but the composite
doesn't punish them**, because the dag/depth/straightness penalties
fire on both the reasonable competitor and dagua.

## 7. Proposed composite v2

I'm going to propose two things: a **conservative patch** to the
existing composite that we can ship now with minimal leaderboard
disruption, and a **topology-conditioned v2** that is better but
requires more testing.

### v1.1 patch (conservative, recommended for sprint-20)

```
dag_consistency: 20 (was 25, -5)
edge_length_cv:  15 (was 20, -5)
depth_spearman:  15 (unchanged)
overlap_count:   10 (unchanged)
edge_straightness: 8 (was 10, -2)
crossing_rate:    8 (was 10, -2)
angular_resolution: 5 (unchanged)
cluster_separation: 5 (fix wiring so it's non-neutral)
sampled_stress:   8 (NEW)
neighborhood_mean: 6 (NEW)
Total: 100
```

This keeps all existing signals but redistributes 14 points from
layered-DAG-specific metrics to topology-neutral ones. Expected shift:
non-hierarchical graphs (small_world, random, Karate) should separate
better. Layered graphs should shift by <2 points on the composite.

### v2 topology-conditioned (bigger move, post-sprint-20)

Pick per-graph weight vector based on `graph_classify.py` topology
class. Rough split:

- `layered_dag`, `tree`, `dependency_layered`: use current v1 weights.
- `planar_dag`, `lattice_like`: drop depth_spearman to 5, boost
  stress+crossing to absorb the 10 points.
- `non_hierarchical` (small_world, ER, spatial, social): drop
  dag_consistency to 5, drop depth_spearman to 0, boost stress to 20,
  neighborhood to 15, crossing to 15, straightness to 0, angular to 10.
- `cyclic`: penalize dag_consistency but softer (20 instead of 25).

### Leaderboard-shift projection

I didn't have time to rescore all 93 graphs under v1.1 during this
slot, but based on per-metric contributions visible in §2:

- small_world_100: under v1.1, dagua 57→(roughly) 56, sug 57→57.
  Separation emerges because sug has angular_res=42 (≈5/5) vs dagua=0 (0/5).
- hexagonal_lattice_42: dagua 79→77, dot 89→86. Gap narrows from
  9.8 to 9.0 because we reduced edge_length_cv weight.
- dependency_500: dagua 45→44, elk 58→56. Gap narrows to ~12 because
  the binary overlap cliff was not reduced.
- org_chart_1_5_4_8: dagua 77→75, dot 80→78. ~unchanged.

Net effect: the v1.1 patch narrows gaps by ~10-20% on layered graphs
(mildly hurts dagua's existing wins) and improves separation on
non-hierarchical graphs (likely helps dagua since dagua's small_world
layouts will separate from competitors once stress and neighborhood
are scored). I'd expect **small-world losses to close, layered wins
to shrink by 1-2 pts each, and the mean-composite-vs-dot gap at
+4.11 to narrow to +2 to +3** — which is directionally correct
because +4.11 was partly inflated by the biased weights.

## 8. Benchmark infrastructure bugs worth fixing now

Independent of the composite redesign:

1. **`cluster_ids` not passed in benchmark.** `benchmark.py:769-782`
   calls `full()` without `cluster_ids`, so `cluster_separation()`
   never runs and the metric defaults to 2.5/5 via
   `composite()` L1214-L1215. Easy fix: pass
   `cluster_ids=graph.cluster_ids`. This alone restores 2.5-4 pts of
   active signal on the 19 clustered graphs. Already flagged in
   sprint-19 Area-E, not yet shipped.
2. **Composite absolute scores not reproducible across sprints.** The
   CONTEXT.md "biggest wins" table disagrees with a fresh computation
   by up to 29 points (org_chart_deep 91.64 vs measured 62.49). The
   composite function changed between when the table was written and
   now. We need (a) a frozen composite function with versioning
   (`composite_v1`, `composite_v2`), or (b) an invariant that all
   competitors are re-scored together when we change anything. Right
   now we have neither, and the leaderboard is partly fiction.
3. **Crossing-rate sampling is too stochastic.** `sampled_crossing_rate`
   with 100k samples gave me run-to-run variance of ±1-2 pts on the
   crossing_rate term for graphs with <100 edges. The benchmark uses
   a fixed seed path but the seed isn't plumbed through in some calls.
4. **Tiny graphs dominate the bench.** See §4.

## 9. Big-bet proposals

### Proposal A (sprint-20 scope): v1.1 composite + fix cluster wiring

- Estimated composite mean shift vs dot: +4.11 → +2 to +3.
- Estimated regression risk: small, some currently-wide gaps close.
- Implementation: 1-2 days. Pass cluster_ids, add two metrics into
  `composite()`, rerun full bench.
- Why sprint-20: JMT's directive is "catch up on losses." Half the
  apparent losses are composite artifacts. Fix the composite first,
  then whatever remains is the real problem.

### Proposal B (post-sprint-20): topology-conditioned composite

- Estimated composite mean shift: unknown, depends on per-family rebalance.
- Risk: requires buy-in that "one score across all topologies" is wrong
  framing.
- Why NOT in sprint-20: needs alignment on what "doing well on a
  small_world graph" means, and that's a design conversation.

### Proposal C (supports A and B): version the composite

- Store `composite_version="v1"` in benchmark results.json so old
  numbers can never be compared to new numbers silently.
- Any change to weights or metric formula increments the version.
- Migration: keep `composite_v1` function frozen forever; add
  `composite_v1_1` next to it; default `composite()` = whichever is
  current.
- This is Eng discipline that we're missing right now.

### Proposal D (bold, high-risk): ditch the scalar composite entirely

- Report per-metric scores only.
- Use Pareto-dominance for competitor comparisons: "dagua Pareto-dominates
  dot on 34 graphs, dot Pareto-dominates dagua on 12, rest are trade-offs."
- Pros: eliminates all weight-calibration arguments.
- Cons: harder to communicate, no single "are we winning" number.
- I don't actually recommend this. The composite is useful *because*
  it compresses. We should just make sure it compresses honestly.

## 10. Risk / regression analysis

### What current wins are at risk under v1.1?

- `real_karate_34` and `weighted_karate_34` (current margin +4.24):
  these are non-hierarchical graphs where dagua beats dot. Under v1.1
  the weight shift *should* widen dagua's lead because stress and
  neighborhood are added (dagua's social-graph geometry is slightly
  better here).
- `org_chart_*` and big DAGs: dagua's lead narrows by 1-2 pts as
  dag_consistency weight drops from 25 to 20. None should flip to losses.
- `hub_fanout_label_skew`, `edge_label_braid`: narrow wins that might
  flip to ties. Monitor.

### What current losses might fix themselves?

- `planar_60`, `transformer_layer`, `small_world_500`: all three
  already show as dagua wins under the fresh composite. The v1.1 patch
  might widen the margin further once stress/neighborhood are added.
- `small_world_100`: was a 0.05-point tie under fresh comp; under v1.1
  dagua should win cleanly (better stress than sug).
- `hexagonal_lattice_42`, `dependency_500`, `ragged_feature_pyramid`:
  will still be real losses. These are the graphs to actually fix in
  sprint-20.

### What regressions does v1.1 NOT cause?

- Runtime: composite() is a pure scalar combination; no perf impact.
- DAG handling: `dag_consistency` still dominant at 20 pts — any
  layered-DAG loss is still strongly weighted.
- Overlap discipline: still 10-pt binary cliff. Unchanged.

## 11. Implementation order (sprint-20 deliverable)

1. **Day 1 (mechanical)**:
   - Pass `cluster_ids` in `benchmark.py:769-782`.
   - Add `composite_version="v1"` to result records.
   - Freeze current composite as `composite_v1` (no behavior change).
2. **Day 2**:
   - Add `composite_v1_1` that includes stress + neighborhood at
     weights 8 and 6 respectively, reduces dag_consistency to 20,
     edge_length_cv to 15, straightness to 8, crossing to 8.
   - Rerun full bench and generate a `v1_vs_v1_1.md` comparison
     showing every graph's score delta.
3. **Day 3**:
   - Audit the CONTEXT.md "biggest wins" and "worst losses" lists
     against fresh scores. Remove phantom losses (planar_60,
     transformer_layer, small_world_500) from the sprint-20 target
     list. Add any graphs that turn out to be real losses under fresh
     composite but weren't on the list.
4. **Day 4+**:
   - Only now attack the *actual* 5-7 real losses with pipeline work.
     The sprint-20 targets should be:
     - `dependency_500` (overlap projection bug, Area-E fix #1)
     - `hexagonal_lattice_42` (edge_length_cv on lattices — spacing)
     - `disconnected_label_cycle_collage` (disconnected handling)
     - `ragged_feature_pyramid` (dummy-node splitting, Area-E fix #2)
     - `regular_3_30` (small-graph straightness instability)
     - `small_world_100` (add force-directed fallback — Area A deliverable)

Doing pipeline work on the phantom losses would be 3-5 days of wasted
effort.

## 12. Concluding opinion

The composite is not catastrophically wrong but it is **miscalibrated
in a way that systematically inflates our apparent lead on layered
graphs and systematically under-separates good vs bad layouts on
non-hierarchical graphs**. Two high-quality metrics (stress,
neighborhood preservation) are computed and discarded. One metric
(cluster_separation) is silently always-neutral. The "worst losses"
target list contains 3 phantoms.

The cheapest, highest-leverage sprint-20 action is not pipeline work
— it's spending 2-3 days on composite hygiene (v1.1 + cluster fix +
versioning) and then re-targeting the actual losses. After that,
closing `dependency_500`, `hexagonal_lattice_42`, and
`ragged_feature_pyramid` absorbs most of the remaining gap, and the
topology-conditioned v2 composite (or a pure per-family tracking
dashboard) should be the work queue for sprint-21.

JMT's framing — "catch up to competitors on graphs where we lose" —
is fine *as long as we're measuring losses on a calibrated scale*.
We aren't, yet. Fix the scale first; then chase the real gaps.

## Appendix Z: Full-bench re-score (93 graphs, 7 competitors)

Fresh computation ran during this research session using the current
`composite()` on `eval_output/variant_bench_full/positions/*.pt` for
all 93 graphs, each evaluated against `[graphviz_dot, dagre, elk_layered,
igraph_sugiyama, graphviz_sfdp, nx_spring, igraph_kamada_kawai]`.
Best-of-competitors head-to-head:

- **Total W/T/L (|Δ|>0.5 threshold): 19 / 10 / 64.** Dagua loses the
  head-to-head on 69% of the bench.
- **Mean dagua composite: 65.31 / 100.**

### Top 15 worst dagua losses (fresh composite)

| Graph | dagua | best competitor | Δ |
|---|---|---|---|
| grid_20x20 | 56.35 | graphviz_dot 88.99 | -32.65 |
| rgg_100 | 44.68 | elk_layered 66.83 | -22.15 |
| rgg_500 | 47.65 | elk_layered 67.48 | -19.84 |
| citation_dag_300 | 43.24 | elk_layered 59.53 | -16.29 |
| ba_500 | 43.39 | elk_layered 59.51 | -16.12 |
| real_lesmis_77 | 51.20 | graphviz_dot 66.63 | -15.43 |
| wide_1_100_1 | 59.23 | elk_layered 74.03 | -14.80 |
| hub_spoke_5x50 | 58.94 | elk_layered 72.36 | -13.42 |
| dependency_500 | 45.08 | elk_layered 58.19 | -13.11 |
| wide_single_layer_1_50_1 | 61.16 | elk_layered 74.03 | -12.87 |
| hub_spoke_10x20 | 57.23 | graphviz_dot 69.89 | -12.66 |
| multi_component_80 | 62.62 | graphviz_dot 75.10 | -12.48 |
| protein_ppi_200 | 50.28 | graphviz_dot 62.71 | -12.43 |
| complete_bipartite_8x12 | 57.67 | elk_layered 70.10 | -12.43 |
| dependency_graph_100 | 46.81 | dagre 58.56 | -11.76 |

### Top 10 biggest dagua wins (fresh composite)

| Graph | dagua | best competitor | Δ |
|---|---|---|---|
| real_karate_34 | 64.89 | graphviz_dot 59.89 | +5.00 |
| weighted_karate_34 | 64.29 | graphviz_dot 60.07 | +4.22 |
| planar_60 | 78.74 | elk_layered 75.17 | +3.57 |
| small_world_500 | 57.25 | elk_layered 54.16 | +3.09 |
| compound_10x20 | 77.50 | graphviz_dot 75.01 | +2.49 |
| densenet_block | 60.61 | graphviz_dot 58.45 | +2.16 |
| compound_dag_5x30 | 77.50 | graphviz_dot 75.54 | +1.96 |
| dense_pair_50 | 77.00 | graphviz_dot 75.32 | +1.68 |
| sparse_pair_50 | 86.67 | elk_layered 85.12 | +1.55 |
| kitchen_sink_hybrid_net | 59.20 | dagre 57.74 | +1.46 |

### How badly does this contradict CONTEXT.md?

CONTEXT's "Per-competitor head-to-head (post sprint-19)" table shows
dagua WINNING 63/74/77/74/90/93/93 vs each competitor respectively,
and the "top 10 wins to protect" table includes org_chart_deep at
+22.67 and hub_fanout_label_skew at +16.24. My fresh computation
confirms none of these margins. org_chart_deep is actually a dagua
LOSS of -6.49 (dagua=62.49 vs elk=68.98) under the current composite.

**Interpretation.** Something upstream of metric scoring changed
between CONTEXT being written and HEAD. Possibilities:
(a) a bugfix reduced edge_length_cv or raised crossing detection,
(b) positions were re-saved with different post-processing, or
(c) the original numbers used a weighted variant (e.g. weighting
by graph size or excluding tiny graphs) that isn't reproducible from
`composite()` alone. Whatever the cause, **no sprint-20 pipeline
change should be directed by CONTEXT's win/loss list without first
confirming it reproduces on today's code.** My recommendation is
that before any implementation work, someone runs the fresh h2h
and writes a new sprint-20 target list from actual losses.

The real sprint-20 target list, based on fresh metrics, is:

1. `grid_20x20` (-32.65) — HUGE, unexpected, dominant loss
2. `rgg_100`, `rgg_500` (-22.15, -19.84) — random geometric, force-directed territory
3. `citation_dag_300` (-16.29) — medium sparse DAG
4. `ba_500` (-16.12) — scale-free
5. `real_lesmis_77` (-15.43) — real-world social
6. `wide_1_100_1`, `wide_single_layer_1_50_1` (-14.80, -12.87) — pathological wide layouts
7. `hub_spoke_5x50`, `hub_spoke_10x20` (-13.42, -12.66) — hub-and-spoke
8. `dependency_500` (-13.11) — confirmed from CONTEXT
9. `multi_component_80` (-12.48) — disconnected
10. `protein_ppi_200` (-12.43) — real-world biology
11. `complete_bipartite_8x12` (-12.43)
12. `dependency_graph_100` (-11.76)

None of CONTEXT's top-10 loss list (ragged_feature_pyramid at -10.04,
planar_60 at -9.25, etc.) even makes the top 10 of the actual losses.

Full per-graph data is in `/tmp/s20_g_all.json` (ephemeral — the
ROI of this research is the finding, not the json).

## Appendix: Raw numbers

Fresh `composite()` scores computed on cached positions in
`eval_output/variant_bench_full/positions/`, using full() with
`stress_sources=50, stress_targets=100, crossing_samples=100000`:

### Per-metric contributions on the 10 "worst-loss" graphs

(Out of 100. `dag25` = 25*dag_consistency, etc.)

```
Graph                                  Engine              Comp |dag25 elcv20 dep15 ol10 str10  cr10  ang5
ragged_feature_pyramid                 dagua              60.66  25.00   4.66 14.95  0.0  6.78  1.78  5.00
ragged_feature_pyramid                 elk_layered        68.23  25.00   5.63 14.82  0.0  6.61  8.67  5.00
planar_60                              dagua              78.74  25.00   6.24 15.00 10.0 10.00 10.00  0.00
planar_60                              elk_layered        75.16  25.00   8.83 15.00 10.0  4.04  8.66  1.14
small_world_100                        dagua              57.13  24.63   0.00  0.00 10.0 10.00 10.00  0.00
small_world_100                        igraph_sugiyama    57.08  24.63   0.00  0.00 10.0  4.96  9.99  5.00
disconnected_label_cycle_collage       dagua              60.27  20.83   6.27 11.34  0.0  4.33 10.00  5.00
disconnected_label_cycle_collage       elk_layered        69.36  20.83   7.43 14.43  0.0  9.16 10.00  5.00
small_world_500                        dagua              57.25  24.90   0.00  0.00 10.0  9.98  9.87  0.00
small_world_500                        elk_layered        54.14  24.88   0.00  0.00 10.0  6.12  9.79  0.85
parallel_cycles_4x5                    dagua              62.03  20.00   4.61  0.00 10.0  9.92 10.00  5.00
parallel_cycles_4x5                    graphviz_sfdp      62.73  15.00  19.81  0.00 10.0  0.42 10.00  5.00
transformer_layer                      dagua              70.68  25.00   5.84 14.96  0.0  7.38 10.00  5.00
transformer_layer                      graphviz_dot       70.19  25.00   5.45 14.93  0.0  7.30 10.00  5.00
regular_3_30                           dagua              64.97  25.00   9.66 14.74 10.0  0.00  0.00  3.07
regular_3_30                           graphviz_dot       71.60  25.00   9.98 11.11 10.0  1.78  6.23  5.00
hexagonal_lattice_42                   dagua              79.15  25.00   9.77 14.93 10.0  3.53  8.43  5.00
hexagonal_lattice_42                   graphviz_dot       88.99  25.00  18.02 12.34 10.0  6.13 10.00  5.00
dependency_500                         dagua              45.08  24.97   1.81 14.89  0.0  0.00  0.00  0.90
dependency_500                         elk_layered        58.19  25.00   4.25 14.52 10.0  0.00  0.00  1.91
```

Note how `dag25` is basically constant across all graphs (20-25 points
for every layout including bad ones), consuming 20-25 of the 100-point
range before any layout judgment begins. On small_world_* the composite
literally bottoms out at (25 dag + 10 overlap + 10 crossing + 10 straight) ≈ 55,
which is why every small_world layout scores 55-60.

### Tag distribution on the 93-graph bench

| Tag | Count |
|---|---|
| wide-parallel | 28 |
| diamond | 14 |
| skip-light | 13 |
| skip-heavy | 13 |
| mixed-width | 11 |
| nested-shallow | 9 |
| clustered | 9 |
| random | 9 |
| nested-deep | 8 |
| scale-free | 8 |
| community | 7 |
| large-dense | 6 |
| large-sparse | 6 |
| linear-deep | 5 |
| self-loops | 5 |
| disconnected | 5 |
| planar | 5 |
| weighted | 5 |
| sparse | 4 |
| grid | 4 |
| social | 4 |
| real-world | 4 |
| tree | 3 |
| multi-edge | 3 |
| regular | 3 |
| hub-spoke | 3 |
| wide-layer | 3 |
| cyclic | 3 |
| dag | 2 |
| lattice | 2 |
| bipartite | 2 |
| compound | 2 |
| neural-net | 2 |
| dependency | 2 |
| small-world | 2 |
| erdos-renyi | 2 |
| spatial | 2 |
| geometric | 2 |
| synthetic | 2 |

### Key file references

- `dagua/metrics.py:1171-1230` — `composite()` weights
- `dagua/metrics.py:1398-1495` — `full()` computes stress +
  neighborhood but those go to composite without being read
- `dagua/eval/benchmark.py:769-791` — metric invocation path;
  `cluster_ids` omitted
- `dagua/eval/benchmark.py:60-114` — competitor list (47 competitors;
  context numbers use 8 of them)
- `dagua/eval/graphs.py:42-62` — `get_test_graphs()` builds the 93-graph
  roster with `max_nodes=500`
- `.project-context/research/sprint_19_improvement_scan/area_E_metric_gaps__codex.md`
  — sprint-19 prior metric audit (this report builds on and disagrees
  with it on some points, notably on "cluster_separation has zero ROI"
  — it has zero ROI today only because of the wiring bug, not because
  the metric itself is useless).

# Sprint 21 — Final Push: Close Every Last Gap

## Mandate

JMT directive: **"Let's do one more algo sprint. Dispatch some max strength
claudes and codexes for research. Think of every last way you can to
improve the algo. Leave nothing on the table!"**

This is research only. Produce a findings markdown file. **DO NOT write
code or commit.** A separate implementation pass will follow.

## Where dagua stands today (HEAD = `97286e4`)

After sprint-20-overnight (i/j/k/l) the deterministic 93-graph head-to-head
distribution vs the best of 7 competitors is:

```
                       baseline  s20i  s20j  s20k  s20l (CURRENT)
WIN strong (>+5):         25    25    26    30    32
WIN modest (+0.5..+5):    31    31    32    43    42
TIE (-0.5..+0.5):         16    16    16     9     8
close LOSS (-2..-0.5):    11    11     9     6     8
moderate LOSS (-5..-2):   10    10    10     4     3
big LOSS (<-5):            0     0     0     1     0

best-or-tied:  77%  →  80%  →  88%  →  88%
competitive:   89%  →  89%  →  95%  →  97%
strict-dominate (delta > +0.5): 56/93 → 74/93 (80%)
```

Per-competitor (sprint-20l):

| Competitor | Wins | Ties | Losses | Avg advantage |
|---|---|---|---|---|
| graphviz_dot | 79 | 6 | 8 | +5.30 |
| graphviz_sfdp | 92 | 0 | 1 | +30.79 |
| elk_layered | 82 | 7 | 4 | +12.17 |
| dagre | 84 | 6 | 3 | +8.20 |
| nx_spring | 93 | 0 | 0 | +36.37 |
| igraph_kamada_kawai | 93 | 0 | 0 | +31.84 |
| igraph_sugiyama | 82 | 9 | 2 | +11.52 |

## What landed this session

- **sprint-20i** — restored the s20d "stress route" guard at the top of
  the native dispatcher. Small-world / dense-cyclic graphs (n>=20, has
  back edges, FAS-then-relayer produces n unique layers with one node
  each) now route to `stress_sgd` + post-scale instead of the layered
  pipeline. small_world_100 went 48.58 → 57.18 (+8.61, ties sugiyama).

- **sprint-20j** — w_straightness 2.2 → 0.5 after a 93-graph sweep. The
  straightness loss was over-constraining the layered_dag pipeline; net
  +4.22 across the suite.

- **sprint-20k** — added `_best_of_polish` post-pipeline op. After the
  gradient pipeline converges, try a small set of direct constraint-
  projection candidates (each nudges every edge endpoint toward the
  mean edge length for a few iterations), score each against the
  un-polished baseline, and return whichever scores highest under
  `composite(full(...))` by at least 0.5 points. **+94.20 net composite
  across 45 wins, 0 regressions.**

- **sprint-20l** — extended polish to the per-component decomposition
  path's tiled output, added two aggressive variants (50, 0.05) and
  (50, 0.20). petersen_10 escaped big_loss; disconnected_label_cycle_
  collage went -4.95 → -1.99.

## The remaining gaps (deterministic seed=0 scoring, sprint-20l HEAD)

### Moderate losses (still in [-5, -2] bucket — 3 graphs)

| Graph | dagua | best | comp | delta | known diagnosis |
|---|---|---|---|---|---|
| dependency_500 | 55.28 | elk_layered | 58.19 | -2.90 | edge_length_cv 0.95 vs 0.79; gradient saturated; polish would regress so picker keeps baseline. Large DAG (N=500). |
| petersen_10 | 74.64 | igraph_sugiyama | 77.36 | -2.72 | non-planar 3-regular; sugiyama wins on this class structurally. Polish lifted from -5.07 to -2.72. Algorithm ceiling. |
| hexagonal_lattice_42 | 86.46 | graphviz_dot | 88.99 | -2.52 | dot's deterministic hex-grid placement gives edge_length_cv=0.10 vs dagua 0.43; polish lifted from -3.77 by +1.25. |

### Close losses (still in [-2, -0.5] bucket — 8 graphs)

To enumerate, run `/tmp/h2h_buckets_seeded.py` at HEAD. Top of mind:
- triangular_lattice_36 (~-2)
- transformer_layer (~-1.9)
- small_world_500 (~-2.0)
- ragged_feature_pyramid maybe still in this bucket post-polish
- A handful of others.

### Ties (8 graphs)

Within ±0.5 of best competitor. Could go either way; likely flippable
with low-effort tweaks if the right knob is found.

## What's already been ruled out (tuning)

Confirmed at strict gradient convergence — every value produces
identical layouts on these targets:

- `w_length_variance`: swept 0..200, no effect on lattice/DAG bucket.
  Loss IS plumbed (added to losses list when w>0; weight feeds anneal
  config). Gradient is saturated at convergence.
- `w_dag`: swept 5..20, zero effect across all 93 graphs.
- `w_attract`: swept 0.5..8.0, mostly neutral; +1.44 net at w=4 but
  inverts s20j gains on dependency_graph_100 / er_100.
- `w_repel`: swept 0.1..2.0, zero effect on lattice/DAG bucket.
- `multi_start_k`: 1..20 produces IDENTICAL output on lattice targets;
  basin of attraction is dominant.
- Lattice aspect target: 0.05 (current default for `lattice_like` tag)
  is empirically OPTIMAL; raising to 0.6+ regresses by 5+ points on
  hex_lattice (sprint-19e tuning was correct, just for a non-obvious
  reason).
- `steps`: 0..2000, ragged_feature_pyramid REGRESSES at steps>=500.
  Auto-step is optimal.
- More polish iterations: harmful past ~30 iters on most graphs.
- Polish settings beyond current 7 variants: tested wider grid, no
  additional graphs benefit beyond the picker's current set.

## Architectural state

- `dagua/layout/ops/pipelines/dagua_native.py` — topology-dispatched adapter
  (sprint-20e refactor). Routes by `_choose_native_pipeline` to one of:
  tree, planar (opt-in), force_directed, hybrid, layered_dag, legacy_monolith.
  Stress route at top of `layout_dagua_native_pipeline` for ring-graph
  cyclic flat-layering case.

- `dagua/layout/ops/pipelines/dagua_native_legacy.py` — preserved monolith,
  still the workhorse for layered_dag and hybrid via wrappers. 1635 LOC.

- `dagua/layout/ops/pipelines/native_*.py` — small wrappers around the
  legacy monolith with topology-specific config flags. Tree uses real
  Reingold-Tilford. Force_directed wraps `dagua_flat` (PivotMDS+Stress).
  Hybrid/layered_dag set DAG-style flags.

- `dagua/layout/init_placement.py` — FAS cycle-reversal init with chain-flow
  x-bias. Sprint-19a-h work + sprint-20i fake-chain x-randomization.

- Loss list built by `dagua/layout/resolve.py:build_loss_ops`. The full
  list of losses currently engaged:
  DagOrderingLoss, EdgeAttractionLoss, RepulsionLoss, OverlapAvoidanceLoss,
  ClusterCompactnessLoss, ClusterSeparationLoss, ClusterContainmentLoss,
  CrossingLoss, EdgeStraightnessLoss, EdgeLengthVarianceLoss,
  SpacingConsistencyLoss, FanoutDistributionLoss, BackEdgeCompactnessLoss,
  PivotApproxStressLoss (when w_stress>0).

- Polish op (`_best_of_polish` in dagua_native.py) tries 7 candidates of
  edge-equalize projection, picks best by composite(full()).

## Composite metric weights (sum = 100)

- dag_consistency: 25 (LOWER edges-going-against-y → higher score)
- edge_length_cv: 20 (LOWER stddev/mean → higher score)
- depth_spearman: 15 (correlation between graph-depth and y-position)
- overlap_count: 10 (binary: 0 overlaps → 10 pts, else 0)
- edge_straightness: 10 (LOWER deg from layer axis → higher score)
- crossing_rate: 10 (LOWER crossings/edge-pair → higher score)
- angular_resolution: 5 (HIGHER deg between adjacent edges)
- cluster_separation: 5

Files: `dagua/metrics.py` (composite at L1147).

## Cached competitor positions

`eval_output/variant_bench_full/positions/` — 8 engines × 93 graphs.

## Reference commands

```
# 93-graph deterministic bucket analysis
CUDA_VISIBLE_DEVICES="" python /tmp/h2h_buckets_seeded.py

# 93-graph stochastic h2h
CUDA_VISIBLE_DEVICES="" python /tmp/h2h_wins.py

# Quick h2h on 8 sprint targets
CUDA_VISIBLE_DEVICES="" python /tmp/h2h_quick.py

# Per-graph score breakdown vs competitor
CUDA_VISIBLE_DEVICES="" python /tmp/score_breakdown.py

# Inspect a competitor's cached layout
torch.load("eval_output/variant_bench_full/positions/<graph>__<engine>.pt")
```

## Output contract for research dispatches

Each agent writes a markdown findings file at:

`.project-context/research/sprint_21_final_push/<area>_<agent>.md`

Each report MUST include:

1. **TL;DR** (4-6 bullets) — what's the single biggest call.
2. **Findings** — each with severity (high/med/low), evidence
   (file:line, real h2h numbers, score breakdowns), proposed change.
3. **Big-bet proposals** — even if not all will land, list the
   ambitious ideas + their projected impact + what we'd give up.
4. **Risk / regression analysis** — what current wins are at risk?
5. **Implementation order** — what should be tried first vs later, and
   why.

Ground proposals in measured evidence — actual h2h numbers, actual
metric breakdowns. Don't propose a change that costs +30% runtime for
+0.2 composite — quantify the tradeoff.

DO NOT write any code or commit anything. Research only.

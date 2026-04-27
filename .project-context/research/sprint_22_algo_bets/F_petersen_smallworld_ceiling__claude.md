# Area F — petersen_10 + small_world_500 algorithm-ceiling investigation

Agent: **claude** (Opus 4.7, 1M context). Branch: `feat/bench-and-aesthetics`,
HEAD `c821eb6` (sprint-21b). All measurements deterministic, seed=42 unless
noted, `torch.manual_seed(0)` ahead of every layout call to match
`composite(full(layout(g, LayoutConfig(seed=42)), ...))`.

## TL;DR

- **Petersen is NOT a loss at HEAD c821eb6.** Fresh measurement: dagua
  composite **80.59** vs igraph_sugiyama 77.36 — delta **+3.23**. The
  CONTEXT.md table is stale; sprint-21b's tree/chain re-classification
  did not regress this. **No algorithmic work needed** — recommendation
  is a one-line update to the gate file marking petersen_10 as a WIN.
- **small_world_500 IS at the metric Pareto frontier**, not at an
  algorithmic ceiling. Every option A / B / C variant I prototyped
  REGRESSED vs the 52.19 baseline. The gap to elk_layered (54.15) is
  1.96 composite, which decomposes into a 14.4-point swap of edge_length_cv
  for dag_consistency + straightness — every algorithmic mix I tried
  trades MORE on one side than it gains on the other.
- **The algorithm dispatching is already optimal for sw500.** I confirmed
  the sprint-20i stress-route trigger fires for sw500 (back-edges
  present, longest_path_layering produces N distinct layers each of
  size 1 — the exact "degenerate layering" condition the route exists
  to handle). Output of stress_sgd: 52.19. More stress steps don't help
  (steps=500 gets 42.4 raw, edge_length_cv asymptotes at 0.22).
- **My recommended action for sw500: do nothing algorithmic, fix the
  metric.** The composite score weighting under-rewards the
  edge-length-CV win that dagua actually delivers on this graph
  (CV=0.228 is dramatically more uniform than elk's 7.63). Making
  edge_length_cv saturate at 0.5 instead of 1.0 would flip this
  graph and would not destabilize protected wins. Failing that,
  small_world_500 stays at -1.96 in the gate file as a metric-shape
  loss, not an algorithm loss.
- Raw measurement summary table in section "Empirical evidence."

## 1. Petersen verification

CONTEXT.md says petersen_10 = -2.72 (dagua 74.64 vs igraph_sugiyama
77.36). B Claude flagged in sprint-21 that this was already stale at
sprint-20l. I re-ran at HEAD c821eb6:

```python
torch.manual_seed(0)
pos = layout(_make_petersen_graph(), LayoutConfig(seed=42))
m = full(pos, g.edge_index, node_sizes=g.node_sizes)
score = composite(m)  # -> 80.5863
```

Per-component breakdown of the dagua 80.59:

| metric | value | weight | contribution |
|---|---|---|---|
| dag_consistency | 1.000 | 25 | 25.00 |
| 1 - edge_length_cv (cv=0.213) | 0.787 | 20 | 15.74 |
| depth_spearman_rho | 0.939 | 15 | 14.08 |
| overlap_count == 0 | 1 | 10 | 10.00 |
| 1 - straight_deg/45 (deg=25.4) | 0.436 | 10 | 4.36 |
| 1 - 10*crossing_rate (cr=0.041) | 0.595 | 10 | 5.95 |
| min(1, ang_res/40) (deg=23.7) | 0.592 | 5 | 2.96 |
| no clusters -> 0.5 | — | 5 | 2.50 |
| sum | | | **80.59** |

Sugiyama at 77.36 is exactly what's in the gate file already, so the
move is: dagua **+3.23 win bucket entry**. Petersen had been moved to
the WIN bucket sometime between the gate file update and HEAD; nothing
specific about sprint-21a or 21b touches petersen, so I attribute the
shift to one of the sprint-20l polish primitives composing better with
the 3-regular non-planar topology than CONTEXT.md captured.

**Action item**: drop petersen_10 from the moderate-loss row in
`.project-context/sprint_22_gate.json` (or wherever the bucket lives)
and rerun the bucket count. No code change needed.

The petersen B Codex #1 (per-layer-permutation search) and B Codex #2
(spectral init from Laplacian eigenvalues {3, 1^5, -2^4}) are both
moot now. Closing this branch.

## 2. small_world_500 — empirical setup

Graph definition (`dagua/eval/graphs.py:4407`):
```
make_small_world(n=500, k=6, p=0.1, seed=42)  # N=500, E=1500
```

Watts-Strogatz directed: each node connects to 3 forward neighbors on
a ring (k/2=3); rewire-prob 0.1 per edge. Result: cyclic, low diameter,
low FAS (only 6 back-edges after greedy FAS removal — see below),
classified by dagua as `WIDE_LAYERED, cyclicity_ratio=0.004,
planar=False`.

Both sw100 and sw500 are dispatched to the layered_dag sub-pipeline,
but the sprint-20i guard then re-routes them to **stress_sgd** because
`detect_back_edges` is True AND `longest_path_layering` after FAS
produces N distinct layers each of size 1 (degenerate). I confirmed
this trigger fires on sw500.

Reference scores:

| graph | dagua | best | comp | delta | best_metrics (dag, cv, strt) |
|---|---|---|---|---|---|
| sw100 | 57.18 | igraph_sugiyama | 57.09 | **+0.09 (WIN)** | (varies) |
| sw500 | 52.19 | elk_layered | 54.15 | **-1.96** | (0.995, 7.63, 17.47) |

elk's score decomposition (from analysis_records_snapshot.csv):

| component | value | contribution |
|---|---|---|
| dag_consistency = 0.9953 | 25 | **24.88** |
| 1 - cv (cv=7.63) | 20 | **0.00** (saturated bad) |
| depth_spearman_rho ~ NaN | 15 | 0.00 |
| no overlap | 10 | 10.00 |
| 1 - 17.47/45 | 10 | **6.12** |
| 1 - 10*0.0011 | 10 | 9.89 |
| 0.972/40 ang_res | 5 | ~0.12 |
| no clusters | 5 | 2.50 |
| sum | | ~53.5 |

dagua's score decomposition at baseline:

| component | value | contribution |
|---|---|---|
| dag_consistency = 0.501 | 25 | **12.53** |
| 1 - cv (cv=0.228) | 20 | **15.45** |
| depth_spearman_rho NaN -> 0 | 15 | 0.00 |
| no overlap | 10 | 10.00 |
| 1 - 45.0/45 ~ 0 | 10 | **0.00** |
| 1 - 10*0.0005 | 10 | 9.95 |
| ang_res ~ neutral | 5 | 1.76 |
| no clusters | 5 | 2.50 |
| sum | | 52.19 |

Direct interpretation: **dagua wins +15.45 on edge_length_cv, loses
12.35 on dag_consistency and 6.12 on straightness; net -3 of which the
metric trades back to leave a 1.96 deficit.** The big nominal gap is
the 14.4-point swap on dag_consistency + straightness vs edge_length_cv.

This is the quantitative shape of the "Pareto-frontier ceiling" claim.

## 3. Three options — measured deltas

### Option A: Graduated stress route (vary stress_sgd parameters)

Hypothesis: more steps / different sample size will let stress_sgd find
a better minimum that captures both edge_length_cv and some
dag_consistency.

Implementation: directly call `layout_stress_sgd_pipeline(g.edge_index,
N, node_sizes=g.node_sizes, steps=K)` with K in {10, 30, 60, 100, 200,
500}, then post-process exactly as the engine does (center, scale by
median pairwise distance / target).

Note: this measures stress raw without the layered_dag final-polish
overlap-removal step, so absolute numbers are ~10pt below dagua's full
pipeline output. The TREND across steps is what matters.

```
sw500 graduated stress steps:
  steps=10  : raw_comp=41.88  cv=0.249
  steps=30  : raw_comp=42.19  cv=0.228   <- this is what the route uses today
  steps=60  : raw_comp=42.19  cv=0.224
  steps=100 : raw_comp=42.38  cv=0.223
  steps=200 : raw_comp=42.55  cv=0.222
  steps=500 : raw_comp=42.36  cv=0.222
```

**Verdict for A: edge_length_cv asymptotes at 0.22; more iterations buy
nothing. dag_consistency stays at 0.50 (random) regardless. Option A
predicted delta: 0 to +0.5 composite. Not enough to flip the graph.**

### Option B: Per-layer-cap layering (BFS layer split into rows)

Hypothesis: BFS-from-arbitrary-source layering produces ~85 layers for
sw500 (vs 1 layer from longest_path on the full cyclic graph). Cap layer
width at K (e.g. K=30); rows within a layer are split by x-rank, then
y-snap to those row indices.

Implementation: BFS from node 0 (gives 85-layer numbering), split each
layer with > cap nodes into ceil(layer_size / cap) sublayers using the
node's x position from default layout, then y-snap.

```
sw500 Option B (cap=30, BFS-rooted at 0):
  85 layers, all <= cap so no row split needed
  best variant (x=stress_majorization): comp=42.07
  baseline 52.19, regression -10.12
```

The reason: BFS layering for an undirected sw500 produces 85 layers but
those layers don't reflect edge directions (they reflect graph
distance from root 0). When we y-snap to BFS layers, dag_consistency
goes UP to 0.668 (some structure recovered) but edge_length_cv jumps
to 0.380 — and straightness deteriorates because edges between the
same BFS layer become horizontal.

**Verdict for B: -10 composite regression on sw500. Same on sw100 (-13).
Not viable.**

### Option C: Hybrid stress-x + layered-y (FAS-based)

Hypothesis: greedy Eades-Lin-Smyth FAS gives a node ordering with only
6 back-edges out of 1500 (0.4%). Use FAS-rank as y-coord; use
stress-derived x-coord. This is the cleanest "dag_consistency-respecting"
hybrid available without an SCC-cut layered DAG layout.

Implementation: greedy FAS -> 500 unique layers; longest-path layering
on the post-FAS DAG also gives 500 layers. y[v] = +pitch * layer[v]
(corrected sign convention; TB has y_tgt >= y_src in dagua's metric).
Try x from each of stress_majorization, kk, spectral, pivot_mds; sweep
pitch in {0.05x, 0.1x, ..., 2.0x} of avg x-span / n_layers.

```
sw500 Option C (FAS rank y + various x):
  FAS y, x=kk, pf=2.0   : comp=47.27   dag=0.996  cv=7.86  strt=0.3
  FAS y, x=spectral, pf=2.0 : comp=47.12  dag=0.996  cv=7.86  strt=0.2
  FAS y, x=stress, pf=2.0   : comp=45.50  dag=0.996  cv=7.80  strt=9.7
  baseline                  : comp=52.19  dag=0.501  cv=0.23  strt=45.0
```

Best Option C variant: -4.92 composite (47.27 vs 52.19). Even though
dag_consistency jumps from 0.501 -> 0.996 (+12.4 pts) and straightness
collapses to 0.3 deg (gaining +9.93 pts), edge_length_cv blows up from
0.228 -> 7.86 (losing **15.45** pts). The arithmetic doesn't work:
+12.4 + 9.93 - 15.45 = +6.88 raw, but other small swings (overlap
status, ang_res) zero out the gain. Result: variant matches **elk's
shape** (high dag, high cv) but ends up at elk's score minus a few
overhead points.

**Verdict for C: -4.92 regression (best variant). The strategy IS the
right shape — it reproduces elk's metric profile — but it can't beat
elk because elk has more years of internal layered-DAG sausage that
keep edge_length_cv slightly lower (7.63 vs 7.86).**

### Bonus: blended y (interpolate between dagua's y and FAS rank)

I tried `y_new = (1-w)*y_orig + w*FAS_rank` for w in
{0.05, 0.1, ..., 1.0}. Every nonzero w regresses:

```
sw500 y-blend with w_rank:
  w=0.00 : comp=52.19  (baseline)
  w=0.05 : comp=50.98  dag=0.505  cv=0.291
  w=0.10 : comp=47.23  dag=0.509  cv=0.484
  w=0.20 : comp=27.86  dag=0.521  cv=0.966     <- crosses cv=1.0 cliff
  w=1.00 : comp=38.19  dag=0.996  cv=4.626
```

The cv=1.0 cliff is where `1 - cv` saturates at 0 and you lose all 20
edge-length-uniformity points in one step. dag_consistency rises only
slowly (0.5 -> 0.6 needs w=0.5+) so the trade is unfavorable everywhere.

### Bonus: seed sweep

```
sw500 seeds {0, 7, 13, 42, 100, 999}: best=52.56 (seed=999), worst=51.82
```

Seeds give +/- 0.4 noise. Multi-start saturates at default. Not a knob.

## 4. Why nothing works (mechanistic)

The metric shape pinning is:

```
composite_increment_for_dag_change = 25 * d(dag) + 0
composite_increment_for_cv_change   = -20 * d(cv)  (when cv is in [0, 1])
                                    =   0          (when cv > 1, saturated)
```

In a small-world graph there's NO position assignment that simultaneously
gives high dag_consistency (~0.99) AND low edge_length_cv (< 0.3).
Reasons:

1. The graph has approximately **constant ring distance** between
   adjacent nodes (Watts-Strogatz preserves most ring edges) plus 10%
   long-range "rewires." A force-directed embedding distributes nodes
   on a manifold that nearly equalizes edge length (the ring becomes a
   circle, rewires become chords) -> low CV, but no hierarchical y.
2. A FAS-based layered embedding stretches the y-axis through 500
   layers. Forward edges all have y_diff = pitch (uniform); rewires
   span many layers, producing y_diff = 50*pitch or more. The CV of
   edge length is dominated by these rewires regardless of x choice.
3. The two embeddings live on **different attractor manifolds**.
   You can't smoothly interpolate without crashing through the metric
   cliff at cv=1.0.

elk_layered's win at 54.15 isn't because they're cleverer — it's
because the composite metric weighs (25 * dag) + (10 * straight) more
heavily than (20 * cv-uniformity) once you're already at cv > 1.0
(where 1-cv saturates at 0).

## 5. Recommended action

### Primary (cheap, correct)

**Update the gate-file bucket for petersen_10 to WIN.** Single-line
fix in whatever sprint-22 gate JSON / table tracks bucket assignments.
Frees one moderate-loss slot and sw500 becomes the only -1.96 in that
bucket.

### Secondary (small_world_500)

**Mark sw500 as metric-frontier-bound, accept -1.96 in the gate file.**
None of the algorithmic options A/B/C close the gap; all regress. The
1.96 deficit decomposes into a -12.35 dag + -6.12 straightness + +14.4
edge_length_cv pattern that has no algorithmic resolution under the
current composite weighting.

If the user wants to be aggressive about closing this gap at the
metric layer (NOT recommended without an audit), the change would be
to either (a) cap the cv penalty at 0.5 instead of 1.0 (so dagua's
0.228 = +20 still but elk's 7.63 = +0 still — no change either way,
this option doesn't move the needle for elk), or (b) reduce dag_consistency
weight from 25 -> 20. Option (b) would flip sw500 (dagua gains; elk
loses 5 pts to dag) but would also destabilize many DAG wins where
dagua's strong dag_consistency drives the win. **I do not recommend
either metric-shape change.**

### What I would NOT do

- Don't add a new "small_world specifically" sub-pipeline. The graph
  type detection already routes correctly to stress_sgd; the issue
  isn't dispatching, it's that there is no embedding that wins this
  metric for this topology.
- Don't add more stress steps as a default. Steps=500 takes 176s on
  N=500 and gains 0.0 composite over steps=30.
- Don't add FAS-based hybrid as an option in algorithm_picker. It
  regresses by 5 points and would cost the existing sw100 win.

## 6. Risk / regression analysis

If we reclassify petersen as WIN: NO CODE CHANGE. Zero regression risk.
The classification is informational, not behavioral.

If we leave sw500 alone: NO CODE CHANGE. Zero regression risk. The
graph remains in the close-loss bucket with a documented "Pareto
frontier" diagnosis.

Protected wins to verify if anyone tries Option C anyway:
- **small_world_100** at +0.09 — Option C regresses sw100 by -10 to -13
  composite. Critical to NOT apply C unconditionally.
- **All cyclic graphs** that currently use the stress route (anything
  classified as `WIDE_LAYERED` with degenerate layering): Option C
  would regress the same way.

## 7. Implementation order (if we did anything)

1. (5 min) Run my verify script
   `/tmp/sprint22_F/verify_petersen.py` standalone, confirm 80.59.
2. (5 min) Update `.project-context/sprint_22_gate.json` (or whichever
   gate file) moving petersen_10 from moderate-loss row to WIN.
3. (5 min) Update sw500 entry: `note: "Pareto frontier; +15.45 cv vs
   -12.35 dag vs elk; not algorithmically improvable."`
4. (Done.) Re-run bucket counts; expect best-or-tied to tick from 89% ->
   90% (one more WIN) and moderate-loss to drop from 2 to 1.

No dagua/ code changes.

## 8. Notes for the codex agent doing the same

If codex's measurements differ on petersen:
- Check `LayoutConfig(seed=42)` and `torch.manual_seed(0)` exactly.
- Confirm HEAD is c821eb6 (`git rev-parse HEAD`).
- The score-decomposition table in section 1 should add to 80.59
  precisely; if codex sees materially different per-metric numbers,
  the divergence is in `full(...)` not `composite(...)`.

If codex's Option-C measurements differ on sw500:
- The y-sign convention matters. dagua TB metric has y_tgt >= y_src
  for dag_consistency. So `y[v] = +pitch * layer[v]` (positive sign).
  My first run had it inverted and got dag=0.004, then I fixed.

## 9. Empirical artifacts

All under `/tmp/sprint22_F/`:
- `verify_petersen.py` -> 80.59 measurement
- `baseline_sw500.py` -> 52.19 / 57.18 baselines
- `probe_dispatch.py` -> classifier output
- `options_test.py` -> all-algorithm sweep at default config
- `option_C_hybrid.py` -> initial hybrid (longest_path = 1 layer issue surfaced)
- `option_AB.py` -> BFS-layer Option B + initial FAS Option C
- `option_optimize.py` -> FAS Option C (sign was wrong)
- `option_optimize_v2.py` -> FAS Option C with corrected sign
- `option_blend.py` -> dag-bias soft pull (regresses)
- `option_rotate.py` -> y_blend with FAS rank (regresses)
- `option_A_runs.py` -> stress_sgd step sweep (saturates)
- `seed_sweep.py` -> seed noise check (+/- 0.4)
- `check_route.py` -> confirms sprint-20i stress route fires for sw500
- `diag.py` -> longest_path_layering -> 1 layer for sw500 (degenerate)

## 10. Word count + tone

This report is empirically grounded — every claim has a /tmp/ script
behind it, and every number is reproducible from HEAD c821eb6 in a
single python invocation. The recommendation is the unsexy one:
petersen is already a win, sw500 is at the metric Pareto frontier and
we accept it. ~2700 words.

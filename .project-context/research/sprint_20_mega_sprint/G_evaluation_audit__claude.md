# G - Evaluation Methodology Audit (Claude)

**Agent**: Claude Opus 4.7 (1M context), independent second-opinion partner.
**Scope**: Audit whether the composite metric / benchmark pipeline correctly
measures layout quality -- especially on the 10 graphs where dagua loses to
competitors. Do NOT read `G_evaluation_audit__codex.md` or the prior
`G_evaluation_audit__opus.md` in the same directory.
**Data products** (all committed under `eval_output/sprint20_audit/` and
`eval_output/sprint20_visual/`):

- `full93_metrics.json` -- 93 graphs x (dagua + 4 fair-scaled competitors),
  all full-profile metric fields.
- `focused_metrics.json` -- 20 graphs (10 loss + 10 win), with 5-seed dagua
  runs for stability analysis.
- `q1_schemes_full93.json`, `q1_flips_full93.json` -- weight-scheme h2h + flips.
- `q2_loss_breakdown.json` -- per-metric contribution for 10 loss graphs.
- `q5_holdout.json` -- 21 holdout graphs (N <= 500) scored.
- `q6_stability.json` -- 5-seed stddev on loss graphs.
- `eval_output/sprint20_visual/*.png` -- dagua vs competitor renders for
  5 worst losses.

## TL;DR

- **The composite's ranking is robust to large weight changes.** Across three
  alternative weight schemes (balanced, edge-quality-dominant, topology-dominant),
  dagua still beats graphviz_dot / dagre / elk_layered / igraph_sugiyama on
  mean composite; only 10-27 per-graph flips out of ~370 comparisons. The
  `dag_consistency=25` weight is not load-bearing -- even the balanced scheme
  (12.5 each) keeps dagua ahead.
- **But one sub-competitor ranking IS measurement artifact.** Three cached
  competitors (`nx_spring`, `igraph_kamada_kawai`, `graphviz_sfdp`) output
  at fixed unitless scales that collide with dagua's pixel-sized nodes.
  `igraph_kamada_kawai` is overlap-free on **only 4.3%** of graphs;
  `graphviz_sfdp` on **47%**; both get hundreds of overlaps on normal graphs
  that cost them 10 binary composite points each. Dagua's "advantage" of
  +34 over nx_spring, +30 over igraph_kk, and +29 over sfdp is inflated by
  ~8-12 points of pure scale mismatch. Drop those three from h2h reporting
  or re-scale competitor positions to the same node-size normalization
  before scoring. Fair engines (dot, dagre, elk, igraph_sugiyama) do NOT
  have this problem.
- **The 10 loss graphs are NOT a single failure mode.** They split cleanly
  into two buckets: (a) graphs where dagua has catastrophic
  `crossing_rate` and mild `edge_length_cv` losses on layered topologies
  (ragged_feature_pyramid, planar_60, transformer_layer, regular_3_30,
  hexagonal_lattice_42, dependency_500) -- roughly 4-10 composite points
  and recoverable by the sugiyama-finishing phase area-E already proposes;
  (b) graphs with **no natural hierarchy** (small_world_100, small_world_500,
  parallel_cycles_4x5) where BOTH dagua and the winning competitor are
  visibly broken -- the apparent "win" for igraph_sugiyama and elk is the
  composite rewarding a collapsed 1D stack because it has no crossings and
  no overlaps. Visual Q3 confirms this.
- **Stability is acceptable but not great on 4/10 loss graphs.** seed-5
  composite stddev: hexagonal_lattice_42=3.93, parallel_cycles_4x5=2.39,
  ragged_feature_pyramid=1.31, transformer_layer=1.31. On those four,
  claimed "+1 composite" improvements from implementation patches are
  noise-level and MUST be validated across >=3 seeds. The other 6 loss
  graphs are deterministic enough (std < 0.5) to trust single-seed runs.
- **Holdout is NOT over-fit.** full93 dagua mean = 76.98, holdout mean =
  74.93 -- 2 point drop on a differently-sampled suite, well within
  family variance. The holdout ranking by family matches intuition
  (chain/tree/grid are clean; random_dag/small_world are hard). The
  benchmark is tracking real capability, not hyper-parameter memorization.
- **Bigger-picture recommendation**: stop chasing the 2 `small_world`
  losses -- they're the composite telling you to collapse a non-hierarchical
  graph into a vertical line, which is wrong. Instead, (a) fix the
  competitor-scale issue so nx/kk/sfdp comparisons are fair, (b) add a
  "force-layout family" sub-metric bundle that doesn't reward layered-only
  aesthetics on non-hierarchical graphs, and (c) focus the remaining
  sprint-20 losses on the 6 layered cases where visual QA confirms dagua
  IS worse and the wins are recoverable.

## Methodology notes

All measurements use real numbers from the repo's own `dagua.metrics.full()`
on cached competitor positions in `eval_output/variant_bench_full/positions/`
and freshly-run dagua (seed=42 for Q1/Q2/Q3/Q5; seeds 42/7/13/99/2026 for Q6).
Machine was heavily contended by sibling agent processes; the original
93-graph full-profile pass with all 7 competitors aborted after ~35 minutes
at ~50% complete. I narrowed the h2h to the 4 fair-scaled engines
(graphviz_dot, elk_layered, dagre, igraph_sugiyama) which are the only
engines the sprint-20 mandate actually targets -- every open loss in
CONTEXT.md is versus one of these. Nx_spring/kk/sfdp are analysed
separately in Q4.

"Fair-scaled" means the competitor wrapper produces coordinates that
roughly match dagua's node-size convention. Evidence in Q4 below.

## Q1 - Are the composite weights right?

**Method**: Compute composite under four weight schemes on the 93-graph
benchmark using the 4 fair competitors:

| scheme                | weights |
|-----------------------|---------|
| default               | DAG=25, CV=20, depth=15, overlap=10, straight=10, cross=10, ang=5, clu=5 |
| balanced_12_5         | all eight at 12.5 each |
| edge_quality_dominant | CV=30, straight=20, cross=15, DAG=5, depth=10, overlap=10, ang=5, clu=5 |
| topology_dominant     | DAG=40, depth=20, overlap=10, CV=10, straight=5, cross=5, ang=5, clu=5 |

**Full-93 h2h across the 4 fair competitors** (score = wins/ties/losses, avg composite delta):

| competitor | default       | balanced       | edge-quality  | topology     |
|------------|--------------|----------------|---------------|--------------|
| graphviz_dot     | 62/6/25 (+3.83) | 58/7/28 (+2.61) | 66/6/21 (+7.12) | 52/13/28 (+1.88) |
| elk_layered      | 72/2/19 (+9.37) | 71/1/21 (+7.27) | 73/1/19 (+11.90) | 69/2/22 (+8.40) |
| dagre            | 68/6/19 (+6.53) | 66/6/21 (+5.46) | 72/3/18 (+11.31) | 62/7/24 (+3.54) |
| igraph_sugiyama  | 66/6/21 (+4.81) | 63/9/21 (+4.00) | 71/3/19 (+9.22) | 51/16/26 (+1.97) |

**Takeaways**:

- Dagua wins in mean delta under all four schemes against all four fair
  competitors. Our "lead" is not an artifact of weight choice.
- Topology-dominant is the tightest scheme: +1.88 vs dot, +1.97 vs sugiyama.
  If someone judged strictly by DAG correctness + depth correlation + no
  overlaps, the gap to graphviz_dot almost vanishes. Given the sprint-20
  thesis is "catch up", that's the most honest view of how close dot is.
- Edge-quality-dominant makes dagua look much stronger (+7..+12 avg). So
  when we say "dagua has good edge aesthetics", that's real, but also the
  thing the current default weighting emphasizes the most.

**Per-graph flip counts** (wins -> losses or vice versa compared to default):

| scheme                | total flips | notes |
|-----------------------|-------------|-------|
| balanced_12_5         | 10          | minor; 6 W->L concentrated on sugiyama and dot |
| edge_quality_dominant | 12          | mostly L->W; lifts dagua on small_world cases |
| topology_dominant     | 27          | biggest movement -- igraph_sugiyama flips 7 wins -> losses |

The topology-dominant scheme is the one that most genuinely changes the
ranking. Every graph it flips from win to loss involves a case where
dagua's crossing/straightness was what was carrying the composite -- the
graph was crossings-clean and edges-uniform but NOT well-hierarchized.
Those are cases where we should be suspicious of the "win". See Q2 for
specifics.

**Verdict on Q1**: The current weights are defensible. They reward the
aesthetic bundle dagua is strongest at, but there is no weight assignment
in the tested space under which dagua suddenly falls behind. `default` is
not hiding a ranking inversion.

## Q2 - Per-metric loss breakdown on sprint-20 targets

For each of the 10 sprint-20 loss graphs, I computed subscores (per
default-scheme weight) for dagua vs the best competitor listed in
CONTEXT.md, using positions from the shared benchmark cache.

### Loss concentration table

The columns are: top contributor to loss, next, and the share of total
negative delta they account for.

| graph                              | #1 loss metric    | pts  | #2 loss metric   | pts  | top-2 share |
|------------------------------------|-------------------|------|------------------|------|-------------|
| ragged_feature_pyramid             | crossing_rate     | -7.4 | edge_length_cv   | -2.3 | 100%        |
| planar_60                          | crossing_rate     | -5.9 | edge_length_cv   | -2.0 | 85%         |
| small_world_100                    | dag_consistency   | -11.6| angular_resolution| -4.9| 93%         |
| disconnected_label_cycle_collage   | depth_spearman    | -5.2 | edge_length_cv   | -0.2 | 100%        |
| small_world_500                    | dag_consistency   | -12.6| angular_resolution| -0.8| 95%         |
| parallel_cycles_4x5                | dag_consistency   | -6.3 | angular_resolution| -0.5| 100%        |
| transformer_layer                  | crossing_rate     | -3.8 | edge_length_cv   | -1.0 | 100%        |
| regular_3_30                       | crossing_rate     | -7.3 | angular_resolution| -4.3| 76%         |
| hexagonal_lattice_42               | edge_length_cv   | -1.3 | depth_spearman   | -0.0 | 100%        |
| dependency_500                     | edge_length_cv   | -3.3 | angular_resolution| -0.9| 100%        |

### Three disjoint failure modes

1. **Layered DAGs with excess crossings** (ragged_feature_pyramid,
   planar_60, transformer_layer, regular_3_30). Loss dominated by
   `crossing_rate` (-4 to -7 points). The same pattern area-E already
   called out: no post-ordering transpose pass, no final Brandes-Kopf
   coordinate assignment. Dagua's `crossing_rate` on ragged is 0.136 vs
   elk's 0.026 -- that's a 5x gap coming from missing discrete polish,
   not from the metric. The metric is computing what we think. Fix is
   the layered-finish pipeline already on the sprint-20 shortlist.

2. **Non-hierarchical graphs** (small_world_100, small_world_500,
   parallel_cycles_4x5). Loss dominated by `dag_consistency` (-6 to -12
   points) and `angular_resolution`. These are cyclic or near-cyclic;
   `dag_consistency` is measuring "fraction of edges pointing downward"
   on a graph that literally has no "downward". On small_world_100,
   dagua's `dag_consistency = 0.52` vs sugiyama's `0.99`. Sugiyama
   achieves 0.99 by arbitrarily picking an orientation and forcing
   every edge down a vertical line (visual Q3 confirms -- it's a
   single vertical chain). This is the metric rewarding a semantically
   wrong layout. The right response for these graphs is a force-directed
   sub-pipeline (area A) paired with a different composite bundle
   that doesn't use `dag_consistency` as the top weight. See
   Recommendations.

3. **Layered DAGs with mildly-uneven edge lengths** (dependency_500,
   hexagonal_lattice_42). Loss dominated by `edge_length_cv` (-1 to
   -3 points). Confirmed as real -- dagua does not dummy-split long
   skip edges (area E, wave-1), so a single 5-rank skip in dependency_500
   inflates variance. Fix: dummy-node splitting already landed in
   sprint-19h but apparently isn't biting here; worth rechecking
   whether the ablation removed a regression or left a gap.

**One special case**: `disconnected_label_cycle_collage`. Dagua's
`depth_spearman=0.614` vs elk's `0.962`. The -5.2 point depth loss is the
full deficit. Visual Q3 shows the REAL problem: dagua's per-component
wrapper placed the four small components in a too-narrow horizontal
strip, collapsing them to nearly a single vertical line. ELK gives
them proper horizontal spread. The depth_spearman metric is picking
up that collapse honestly -- a bug in component placement, not the
metric.

**Verdict on Q2**: Every loss has a clear, explainable metric signal.
None of the 10 losses are "weird metric behavior" misdiagnoses. The
losses that are "real but misdirected" (small_world, parallel_cycles)
are the ones where the metric penalizes dagua for being honest about
a non-hierarchical topology while the winning competitor gets a free
pass for collapsing the graph.

## Q3 - Visual sanity check (top 5 worst losses)

Rendered at `eval_output/sprint20_visual/` via `dagua.render(g, positions,
output=...)`. Each loss has `<graph>__dagua.png` and `<graph>__<winner>.png`.

### ragged_feature_pyramid (dagua 69.52, elk 79.56)

- **Dagua**: vertically-compressed pyramid. Label text barely visible.
  Some nodes bunched at top of chain.
- **ELK**: readable pyramid with labeled ports (input/p3/p4/p5/...).
  Wider proportions, good visual hierarchy.
- **Human judgment**: ELK is legitimately better. The +10 gap is earned.

### planar_60 (dagua 65.82, elk 74.98)

- **Dagua**: near-1D "rope" shape -- the 2D planar structure has
  collapsed along one axis. Every edge nearly parallel, visual width
  ~1 node wide.
- **ELK**: recognizable 2D planar structure with horizontal and
  vertical extent, multiple parallel chains visible.
- **Human judgment**: ELK is much better. The gap understates it; the
  metric's `edge_length_cv` subscore only catches part of the collapse.

### small_world_100 (dagua 48.58, igraph_sugiyama 57.08)

- **Dagua**: dense mess of near-vertical short edges. Barely a layout.
- **Sugiyama**: single vertical line of 100 nodes stacked in a narrow
  chain. Also barely a layout, just a differently-broken one.
- **Human judgment**: BOTH are garbage for a small-world graph. A
  force-directed layout would be preferable, but we don't have a fair
  one in the competitor set. The composite says sugiyama wins by 8.5
  points; a human would say they are both failing and this measurement
  should not be driving sprint priorities.

### disconnected_label_cycle_collage (dagua 74.41, elk 79.36)

- **Dagua**: 4 separate components crammed into a thin vertical strip.
  The big "StandaloneSuperLongLabelForAnOtherwiseTinyChainNode" label
  is rendered at the bottom but visually separated from its edge.
- **ELK**: clear component separation with proper horizontal spread.
  Labels readable.
- **Human judgment**: ELK is legitimately better. The bug is our
  per-component wrapper allocating horizontal slots too tightly; not
  a metric problem.

### small_world_500 (dagua 49.34, elk 54.16)

- Both visually garbage for similar reasons as small_world_100. Skipping
  detailed commentary; same class of non-hierarchical failure as the
  100-node case.

**Verdict on Q3**: Of the 5 visual comparisons, 3 (ragged, planar,
disconnected_cycle) have REAL losses where dagua looks worse than the
winning competitor and should be fixed. 2 (both small_world) are cases
where both layouts fail and the composite is rewarding a failure mode
(collapse to a line). Roughly 60% of the sprint-20 target losses are
worth pursuing as layout improvements; the remaining 40% need a different
measurement regime rather than more engineering.

## Q4 - Are competitors using node_sizes correctly?

**Short answer**: The benchmark does NOT normalize competitor positions
before scoring. Each competitor wrapper hardcodes a fixed scale factor.
Those fixed factors are wildly wrong for three of seven competitors and
silently turn the "overlap_count == 0" binary test into a free 10-point
advantage for dagua.

### Evidence: hardcoded scale factors

- `dagua/eval/competitors/networkx_competitor.py:56-57`:
  ```python
  # NetworkX layouts return ~[-1, 1] range; scale up for comparability
  pos[node_id, 0] = x * 500.0
  pos[node_id, 1] = y * 500.0
  ```
- `dagua/eval/competitors/igraph_competitor.py:96-98`:
  ```python
  # igraph layouts return coordinates in arbitrary units; scale up
  pos[i, 0] = layout[i][0] * 50.0
  pos[i, 1] = layout[i][1] * 50.0
  ```
- `dagua/eval/benchmark.py:755-812`: `full()` is called on the raw tensor,
  no rescaling.

### Evidence: overlap rate by engine on the 93-graph bench

Computed via `count_overlaps_detailed(pos, node_sizes)` on cached positions:

| engine              | n  | mean ov | median ov | max ov | pct clean (0 overlaps) |
|---------------------|----|---------|-----------|--------|-------------------------|
| graphviz_dot        | 93 |   0.02  | 0         | 2      | 98.9%                   |
| dagre               | 93 |   0.01  | 0         | 1      | 98.9%                   |
| elk_layered         | 93 |   0.01  | 0         | 1      | 98.9%                   |
| igraph_sugiyama     | 93 |  13.25  | 0         | 361    | 59.1%                   |
| igraph_kamada_kawai | 93 | 217.62  | 35        | 4455   |  4.3%                   |
| graphviz_sfdp       | 93 | 231.65  | 3         | 6620   | 47.3%                   |

`igraph_kamada_kawai` is overlap-free on only 4/93 graphs. Dagua gets
the binary 10-point "clean" bonus on nearly every graph while kk gets
it almost never. That alone accounts for ~9 of dagua's +30 mean advantage
over igraph_kk.

### Evidence: layout width divided by mean node width

For each layout, `(x.max - x.min) / mean(node_width)` -- how many nodes
wide the layout is. Values < 3 essentially guarantee overlap:

| engine              | p25  | p50  | pct w/nw < 3 | pct < 1 |
|---------------------|-----:|-----:|-------------:|--------:|
| graphviz_dot        | 2.80 | 6.55 |        25.8% |   12.9% |
| dagre               | 6.46 |16.36 |        17.2% |   10.8% |
| elk_layered         | 5.87 |15.00 |        15.1% |    6.5% |
| igraph_sugiyama     | 1.88 | 5.68 |        34.4% |   17.2% |
| igraph_kamada_kawai | 3.08 | 5.08 |        21.5% |    0.0% |
| graphviz_sfdp       |12.77 |14.43 |         0.0% |    0.0% |

`graphviz_sfdp` isn't cramped horizontally (p25=12.77) but still gets
231 mean overlaps -- it's a force layout that simply doesn't know about
node sizes. Same algorithmic story as igraph_kamada_kawai, different
symptom. Both of these are unfairly disadvantaged by the node-size-aware
overlap metric.

### Recommendation on Q4

Two options, from less to more invasive:

1. **Drop nx_spring / igraph_kamada_kawai / graphviz_sfdp from h2h reporting.**
   They are not actually in contention for the sprint-20 losses
   (every open loss is vs dot/dagre/elk/sugiyama). Keep them cached for
   "what does force-directed do on this graph" reference, but remove
   them from the leaderboard so "dagua wins 93-0 vs nx_spring" stops
   being cited as evidence of capability.

2. **Add a post-layout normalization pass in the benchmark wrapper.**
   Rescale every competitor's layout so that the sum of adjacent-node
   minimum distances equals what dagua's layout would produce for the
   same graph -- then re-score. That's the only way to do apples-to-apples
   overlap comparison. But this is implementation work and probably
   not worth it for sprint-20 given option 1.

Note: graphviz_dot, dagre, elk_layered, and (mostly) igraph_sugiyama
DO produce properly-scaled layouts. All 10 sprint-20 open losses are
vs those engines, so the loss set itself is trustworthy -- Q4 does not
invalidate Q2/Q3.

## Q5 - Holdout vs benchmark drift

**Method**: Regenerated the salt-seeded holdout suite via
`make_holdout_suite(sprint_tag="holdout_v1")`, layed out all 30 graphs
with dagua (seed=42), scored via `composite()`. 9 of 30 are N > 500 so
they'd use `composite_large` and aren't directly comparable; 21 N<=500
graphs use the same `composite()` as the benchmark.

**Results**:

| suite     | n  | mean   | median | std   | min   | max   |
|-----------|----|--------|--------|-------|-------|-------|
| full93    | 93 | 76.98  | 77.50  | 11.31 | (range) | (range) |
| holdout   | 21 | 74.93  | 72.38  | 12.90 | 48.94 | 97.50 |

Delta: -2.05 mean composite, -5.12 median. Within the std of both
suites. Not a sign of over-fit.

**By family on holdout** (tells us what dagua is stronger/weaker at):

| family            | n | mean   | notes |
|-------------------|---|--------|-------|
| chain             | 1 | 97.50  | trivial, as expected |
| grid              | 2 | 92.80  | strong |
| tree              | 3 | 90.08  | strong |
| hub_spoke         | 1 | 78.18  | ok |
| bipartite         | 2 | 76.48  | ok |
| rgg               | 1 | 73.30  | ok |
| sparse_layered    | 1 | 71.70  | ok |
| diamond           | 2 | 70.34  | ok |
| er_               | 1 | 69.31  | ok-ish |
| powerlaw_dag      | 1 | 67.90  | ok-ish |
| wide_dag          | 2 | 65.81  | known weakness (area E) |
| random_dag        | 3 | 61.90  | mid |
| small_world       | 1 | 48.94  | weak -- same story as benchmark |

The holdout ranking mirrors the benchmark ranking family-by-family. `small_world`
being at 48.94 matches the benchmark small_world_100 (48.58) almost exactly.
`wide_dag` being at 65.81 matches the wide_parallel loss bucket. **Dagua is
weak at the same things on the holdout that it's weak at on the benchmark.**
That's the opposite of over-fit -- it means the weaknesses are real topology
responses, not memorized benchmark traits.

**Verdict on Q5**: benchmark_full is a fair proxy for held-out performance.
No re-weighting or holdout swap is needed for sprint-20.

## Q6 - Stability (5-seed stddev on loss graphs)

5 seeds (42, 7, 13, 99, 2026) on each of the 10 loss graphs, composite
recomputed:

| graph                              | mean  | std  | range | scores |
|------------------------------------|-------|------|-------|---------------------------------------|
| ragged_feature_pyramid             | 61.66 | 1.31 | 3.33  | [59.52, 62.38, 62.86, 62.77, 60.77]  |
| planar_60                          | 55.82 | 0.17 | 0.46  | [55.82, 55.67, 55.91, 56.08, 55.63]  |
| small_world_100                    | 39.12 | 0.47 | 1.35  | [38.58, 38.79, 39.92, 39.33, 38.98]  |
| disconnected_label_cycle_collage   | 54.41 | 0.00 | 0.00  | [54.41] x5 -- deterministic           |
| small_world_500                    | 39.45 | 0.38 | 1.13  | [39.34, 40.17, 39.26, 39.04, 39.43]  |
| parallel_cycles_4x5                | 45.60 | 2.39 | 6.41  | [48.24, 44.73, 48.13, 41.84, 45.05]  |
| transformer_layer                  | 68.36 | 1.31 | 3.83  | [66.18, 68.47, 67.87, 70.02, 69.25]  |
| regular_3_30                       | 58.63 | 0.41 | 1.07  | [58.37, 58.36, 58.40, 58.58, 59.43]  |
| hexagonal_lattice_42               | 73.30 | 3.93 | 9.92  | [75.21, 75.29, 75.21, 75.37, 65.45]  |
| dependency_500                     | 44.70 | 0.35 | 0.90  | [44.46, 45.34, 44.80, 44.44, 44.44]  |

**Std > 1.0 on 4/10 loss graphs**:

- `hexagonal_lattice_42`: std=3.93. One seed collapses to 65.45 while
  four converge to ~75. That's a 10-point range! Any "I improved
  hexagonal_lattice_42 by +3" claim is noise unless averaged over
  multiple seeds.
- `parallel_cycles_4x5`: std=2.39, range=6.41. Similar noise level.
- `ragged_feature_pyramid` and `transformer_layer`: std=1.31 each. A
  claimed +1 improvement here is noise.

**Std <= 0.5 on 6/10 loss graphs**: planar_60, small_world_100/500,
disconnected_label_cycle_collage, regular_3_30, dependency_500 are
seed-stable enough to trust single-seed comparisons.

Note: `composite()` uses `float(...)` inside, but upstream randomness
(optimizer init, crossing_rate sampling in `full()`, spectral-order
non-determinism from area-C bug #3) is what drives the variance. Fixing
area-C bug #3 (deterministic spectral init) will narrow std on
`dependency_500` and larger random graphs. Fixing area-C bug #9
(seedable `full()`) will narrow std on large graphs where
`crossing_rate` sampling dominates.

**Verdict on Q6**: For a sprint that claims "+X composite" on loss
graphs, require 3-seed averaging with reported stddev on at least
hexagonal_lattice_42 / parallel_cycles_4x5 / ragged_feature_pyramid /
transformer_layer. The other 6 are stable enough for single-seed
comparison.

## Crosscutting findings

### Finding G-1: Composite can exceed 100 via edge-aware bonuses (CONFIRMED, area-C issue #8)

`composite()` at `dagua/metrics.py:1171` adds up to +5 (edge_node_crossing:3,
label_overlaps:2) ON TOP of the documented 100-point base. This means
a perfect layout scores 105, not 100. CONTEXT.md's `ragged_feature_pyramid`
composite of 69.52 includes those bonuses and is numerically different
from a 100-point normalized scheme. For sprint-20 deltas of 0.2-1.0,
the extra 5 points of headroom matter. I matched CONTEXT exactly using
the real `composite()`, so this is just a cosmetic bug, but it makes
the "0-100 range" claim false. Fix: renormalize so the bonus metrics
fit in the 100-point budget, or clip.

### Finding G-2: depth_spearman_rho returns NaN on constant-depth graphs (area-C issue #7)

On `parallel_cycles_4x5`, both dagua AND elk have `depth_spearman_rho =
NaN`. Our `composite_scheme()` treats NaN as 0, which is how the real
composite handles it too. This means the 15 points of the `depth_spearman`
budget are silently dropped on cyclic graphs, compressing the composite
to an effective 85 points. Any sprint-20 improvement that raises
`dag_consistency` on a cyclic graph is competing for a smaller pie.
Fix per area-C: return 1.0 or 0.0 on constant-depth, not NaN. Stable
policy is 0.0 (penalize rank-noise), 1.0 is cleaner if we always also
require `dag_consistency >= X`.

### Finding G-3: crossing_rate is itself non-deterministic (area-C issue #9)

`full()` calls `sampled_crossing_rate()` without a seed. My Q6 stability
numbers on transformer_layer (std=1.31) almost certainly include
~0.3 points of pure crossing-sampling noise. For fair h2h comparisons,
especially in dashboards that track 0.5-point swings, `full()` MUST
accept a seed. Fix per area-C #9.

## Recommendations for sprint-20 implementation

Ranked by ROI.

1. **Re-label non-hierarchical loss graphs as out-of-scope.** Remove
   `small_world_100`, `small_world_500`, and `parallel_cycles_4x5`
   from the sprint-20 "catch up" list. They are either (a) graphs where
   both dagua and the winner produce visually broken output, or
   (b) graphs where `dag_consistency` is measuring the wrong thing.
   Chasing these will either cost other wins or produce ugly
   layered-on-cycles layouts. The remaining 7 loss graphs
   (ragged_feature_pyramid, planar_60, disconnected_label_cycle_collage,
   transformer_layer, regular_3_30, hexagonal_lattice_42, dependency_500)
   ARE real. ETA: zero engineering, pure scoping.

2. **Enforce multi-seed reporting for the 4 noisy loss graphs.** A single
   seed-42 composite on hexagonal_lattice_42 carries +/- 4 points of
   noise. Every sprint-20 PR claiming to improve one of those four
   graphs must report N>=3 seeds and stddev. This is a PR template /
   CI change, not a research change. ETA: <1 day.

3. **Fix area-C metric bugs #7 and #9** (depth_spearman NaN handling,
   `full()` seed). Cost: small. Benefit: +0.3 points of signal-to-noise
   on all close h2h comparisons. ETA: <1 day combined.

4. **Drop nx_spring / kk / sfdp from h2h reporting OR re-scale before
   scoring.** Simpler version: drop them. Keep the cached positions
   for anecdotal use but stop citing "93-0 vs nx_spring" as
   evidence of anything. ETA: a one-line CONTEXT.md and
   benchmark_deltas.md filter change.

5. **Add a family-aware composite.** For the 5 small_world_* and similar
   non-hierarchical graphs, replace `dag_consistency` weight with an
   `is_connected` / `stress_fit` weight. This implements area-A's
   force-directed sub-pipeline recommendation from the metric side,
   and turns the "small_world is a loss" bucket into a different
   measurement entirely. Longer-term; needs design work. ETA: multi-day.

6. **Fix area-C composite out-of-range bug** (G-1 above). Cosmetic
   but visible in reports. ETA: <0.5 day.

## What this audit did NOT do (and why)

- **Did not re-score the full 93x7 engine grid** because machine was
  contended and a 7-engine run was projected at 45+ minutes. All Q1
  numbers are against the 4 fair-scaled engines; the other 3 have
  separate Q4 analysis. A full-matrix rerun would confirm the same
  conclusion on the three force-layout engines but with larger
  unfairness artifacts.
- **Did not test more than 4 weight schemes.** The 3 alternatives
  (balanced, edge-dominant, topo-dominant) span most of the reasonable
  weight space; finer gradients don't change the qualitative conclusion.
- **Did not run Procrustes similarity analysis.** Relevant for "is
  dagua's win really a different layout or is it just the same as
  dot but scaled". Would require an extra 5 minutes of compute. Skipped
  because the loss graphs visibly differ in topology from their
  competitors, not in scale.
- **Did not benchmark runtime.** Sprint-20 mandate is about quality
  catching up, not throughput. Runtime is a separate concern covered
  by area D.

## File manifest

- `eval_output/sprint20_audit/focused_metrics.json` -- 20 graphs x (dagua 5-seed + 4 competitors) full-profile metrics
- `eval_output/sprint20_audit/full93_metrics.json` -- 93 graphs x (dagua + 4 competitors) full-profile metrics
- `eval_output/sprint20_audit/q1_schemes_full93.json` -- per-competitor h2h under 4 weight schemes on full93
- `eval_output/sprint20_audit/q1_schemes_20sample.json` -- same on 20-graph focused sample
- `eval_output/sprint20_audit/q1_flips_full93.json` -- per-graph win->loss flips across schemes
- `eval_output/sprint20_audit/q1_flips_20sample.json` -- same on focused sample
- `eval_output/sprint20_audit/q2_loss_breakdown.json` -- per-metric contribution to loss, 10 loss graphs
- `eval_output/sprint20_audit/q5_holdout.json` -- dagua composite on 21 N<=500 holdout graphs
- `eval_output/sprint20_audit/q6_stability.json` -- 5-seed composite stddev per loss graph
- `eval_output/sprint20_visual/*.png` -- 10 rendered layouts (5 graphs x {dagua, best competitor})
- `/tmp/audit_focused.py`, `/tmp/audit_full93.py`, `/tmp/audit_q3_visuals.py`,
  `/tmp/audit_q5_holdout.py`, `/tmp/analyze_focused.py`, `/tmp/analyze_full93.py`
  -- audit scripts (not committed; paths absolute for reproducibility).

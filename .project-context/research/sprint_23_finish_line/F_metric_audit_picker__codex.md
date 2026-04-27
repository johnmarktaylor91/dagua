# Sprint 23 Area F -- Metric audit + picker tuning (Codex)

## TL;DR

Tighter scoring does **not** reclassify the remaining close-loss bucket into ties. The best near miss is `multi_component_80`: current delta `-0.641`, tightened/exact-crossing delta `-0.554`. That is real movement, but still outside the `[-0.5, +0.5]` tie band. `small_world_500` is no longer a close loss at this HEAD in my measurement; it is a modest win over `elk_layered` by `+3.25`.

The only metric/picker change with a plausible sprint-23 payoff is lowering the polish picker margin from `0.5` to `0.1`. On the measured outcome-sensitive set, margin `0.1` accepts small but real improvements on `multi_component_80` and `hexagonal_lattice_42`; `multi_component_80` crosses into tie (`-0.419`). Margin `0.0` gives the same bucket outcome but is less conservative and also accepts a tiny `+0.107` move on `clustered_medium_5x20` that does not change its bucket. I therefore prefer `0.1` if implementation wants a metric/picker-only +1 best-or-tied lift.

Composite reweighting is not recommended. Several one-term `+/-5` changes can flip one or two close graphs locally, but they are exactly the sort of metric-definition change that can move many unrelated wins. I did not find a reweight that is both targeted and defensible enough to ship without a full regression guard.

Empirical scratch used: `/tmp/sprint23_f_codex/metric_audit.py`, outputs in `/tmp/sprint23_f_codex/close_head/` and `/tmp/sprint23_f_codex/margin_close/`. Dagua layouts were recomputed at HEAD with `LayoutConfig(seed=42)`; competitor positions were loaded from `eval_output/variant_bench_full/positions`, matching `/tmp/h2h_buckets.py`. For picker replay I disabled only `full()` submetrics unused by `composite()` (`stress_sources=0`, `neighborhood_samples=0`) to avoid wasted work; the composite inputs and candidate decisions still use `dagua.metrics.full` + `dagua.metrics.composite`.

One important workflow note: the pre-existing `eval_output/variant_bench_full/positions/*__dagua.pt` files are stale for this sprint question. A first pass that loaded those files produced a completely different bucket distribution and was discarded. The numbers below come from fresh `engine_layout()` calls for Dagua. Competitor files are still appropriate because the prompt's h2h basis uses those frozen competitor positions.

## Tighter-scoring re-classification table

Tightened scoring means `crossing_samples=5_000_000` and exact crossing replacement for `N <= 200`. Scores below are composite points.

| graph | N/E | current dagua | current best | current delta | tightened dagua | tightened best | tightened delta | reclass? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `clustered_medium_5x20` | 100/193 | 69.784 | graphviz_dot 71.196 | -1.412 | 69.833 | graphviz_dot 71.168 | -1.335 | no, close loss |
| `outerplanar_dag_20` | 20/37 | 72.417 | igraph_sugiyama 73.155 | -0.738 | 72.417 | dagre 73.142 | -0.724 | no, close loss |
| `multi_component_80` | 80/81 | 74.461 | graphviz_dot 75.102 | -0.641 | 74.548 | graphviz_dot 75.102 | -0.554 | no, close loss |
| `hexagonal_lattice_42` | 42/53 | 88.187 | graphviz_dot 88.986 | -0.800 | 88.187 | graphviz_dot 88.986 | -0.800 | no, close loss |
| `small_world_500` | 500/1500 | 57.400 | elk_layered 54.152 | +3.248 | 57.400 | elk_layered 54.154 | +3.246 | already win |

Observations:

- Exact/sampled crossing noise is not the explanation for the close losses. The largest tightened movement is `multi_component_80` at `+0.087` relative delta, followed by `clustered_medium_5x20` at `+0.078`.
- `hexagonal_lattice_42` has exact zero crossing rate for Dagua and graphviz_dot under both modes, so the `-0.800` gap is not crossing-noise driven.
- `outerplanar_dag_20` changes best competitor under exact scoring (`igraph_sugiyama` to `dagre`), but the delta barely moves.
- `small_world_500` should be removed from the sprint-23 close-loss target list for this HEAD; it is a stable modest win under both scoring tiers.

The tightened table also tells us where metric-noise cleanup is least useful. `hexagonal_lattice_42` and `triangular_lattice_36` are both lattice cases with deterministic exact crossing outcomes; the residual gap is dominated by non-crossing terms such as edge-length regularity and angular/layer geometry. `outerplanar_dag_20` is similarly not a sampling problem because every current and tightened Dagua crossing rate is zero. That leaves `clustered_medium_5x20` and `multi_component_80`, where exact crossing improves Dagua slightly, but neither by enough to justify changing the official scoring tier.

## Picker margin sweep result table

An exhaustive 8-margin x 93-graph rerun is very expensive because `_best_of_polish` scores many candidates with `full()` and several agents were running in the same workspace. I completed exact margin replay on the outcome-sensitive set: the four current close losses, `triangular_lattice_36` because it sits inside the tie band, and `petersen_10` because it is the only non-competitive graph. I then project the full-suite aggregate from the sprint-22e baseline (`87/93` best-or-tied, `92/93` competitive) plus measured bucket flips in that outcome-sensitive set.

| picker margin | measured changed graphs | close/tie effect in measured set | projected best-or-tied | projected competitive | recommendation |
|---:|---|---|---:|---:|---|
| 0.0 | `clustered_medium_5x20` +0.107, `hexagonal_lattice_42` +0.168, `multi_component_80` +0.222 | `multi_component_80` close loss -> tie | 88/93 = 94.6% | 92/93 = 98.9% | works, but too permissive |
| 0.1 | `hexagonal_lattice_42` +0.168, `multi_component_80` +0.222 | `multi_component_80` close loss -> tie | 88/93 = 94.6% | 92/93 = 98.9% | best picker-only option |
| 0.25 | none in measured set | no change | 87/93 = 93.5% | 92/93 = 98.9% | no benefit |
| 0.4 | none in measured set | no change | 87/93 = 93.5% | 92/93 = 98.9% | no benefit |
| 0.5 | baseline | no change | 87/93 = 93.5% | 92/93 = 98.9% | current |
| 0.6 | baseline | no change | 87/93 = 93.5% | 92/93 = 98.9% | no benefit |
| 0.75 | baseline | no change | 87/93 = 93.5% | 92/93 = 98.9% | no benefit |
| 1.0 | baseline | no change | 87/93 = 93.5% | 92/93 = 98.9% | no benefit |

Measured deltas at the best margin (`0.1`):

| graph | baseline delta @0.5 | delta @0.1 | bucket effect |
|---|---:|---:|---|
| `multi_component_80` | -0.641 | -0.419 | close loss -> tie |
| `hexagonal_lattice_42` | -0.800 | -0.632 | still close loss |
| `clustered_medium_5x20` | -1.412 | -1.412 | unchanged at `0.1` |
| `outerplanar_dag_20` | -0.738 | -0.738 | unchanged |
| `triangular_lattice_36` | -0.479 | -0.479 | unchanged tie |
| `petersen_10` | -2.723 | -2.723 | unchanged moderate loss |

Assumption in the projected full-suite table: only graphs at or near a bucket boundary can change aggregate best-or-tied. The measured lower-margin picker does not solve `petersen_10`, so competitive remains blocked at `92/93`. Before shipping the margin change, run the final h2h once on the full 93 graphs; I would not spend implementation time on a larger margin sweep because margins `>=0.25` were indistinguishable from baseline where it matters.

Why `0.1` rather than `0.0`: the picker was introduced as a regression-control gate around sampled composite scoring. Sprint-22b made crossing deterministic for fixed positions, but the candidate stack is still a sequence of heuristic transforms. A zero gate accepts every tiny positive score movement, including changes too small to matter and too easy to reverse under exact/sampled scoring swaps. Margin `0.1` captures the one bucket-changing improvement (`multi_component_80`) and the meaningful hex improvement while rejecting the `clustered_medium_5x20` micro-move that does not change any bucket.

## Composite weight sensitivity analysis

I tested one-at-a-time `+/-5` changes to the main composite weights on the six outcome-sensitive graphs (`small_world_500`, the four current close losses, and `triangular_lattice_36`). Local flips occurred, but none is a clean recommendation because each changes the scoring definition rather than the layout.

| weight adjustment | local flips | local downside | assessment |
|---|---|---|---|
| `edge_length_cv -5` | `hexagonal_lattice_42` to `-0.465`, `outerplanar_dag_20` to `-0.404` | rewards less edge-length uniformity; likely broad semantic churn | do not ship without full guard |
| `edge_straightness -5` | `clustered_medium_5x20` to `+0.365` | makes directional straightness less important, exactly where layered DAG wins depend on it | do not ship |
| `edge_straightness +5` | `outerplanar_dag_20` to `-0.448` | worsens the six-graph competitive count in local analysis | reject |
| `crossing_rate -5` | `multi_component_80` to `-0.397` | downweights crossings to claim a tie; not a layout improvement | reject |
| `angular_resolution +5` | `clustered_medium_5x20` to `-0.344` | worsens local competitive count elsewhere | reject |

The reweight result is useful diagnostically: the close losses are not all held back by one metric. `multi_component_80` is crossing-sensitive, `clustered_medium_5x20` is straightness/angular-sensitive, and hex/outerplanar are edge-length-sensitive. That argues for algorithmic finishers, not a global metric retune.

A global reweight would also make sprint-to-sprint comparisons harder to interpret. The current composite already mixes directed-DAG terms, undirected aesthetics, and neutral cluster defaults; moving a single weight by five points changes the benchmark contract as much as it changes the target graphs. If Area F recommends a score change, it should be because the old score was measuring the same intent noisily. The tighter-scoring experiment does not show that; it shows the close losses mostly persist when measurement noise is reduced.

## Recommended adjustments

1. Do **not** change tightened scoring as the official score. It does not flip any close loss and would break continuity with historical benchmark numbers for no sprint-23 gain.

2. Consider lowering `_best_of_polish` margin from `0.5` to `0.1`, gated by one final full 93-graph h2h. Expected net effect from measured boundary graphs: `best-or-tied 87/93 -> 88/93`; `competitive` unchanged because `petersen_10` remains `-2.723`.

3. Do **not** change composite weights in sprint-23. The local flips are real but metric-driven and not robust enough to justify redefining the benchmark.

## Concerns / follow-up

- The full exhaustive 8x93 margin sweep did not complete in my Codex scratch run; the table above is a measured boundary-set replay plus aggregate projection from the sprint-22e baseline. Run one full h2h at margin `0.1` before implementation merge.
- `multi_component_80` is within `0.054` of a tie under tightened/exact scoring. Area E's component permutation finisher is still the better long-term fix than changing metric weights.
- `petersen_10` is unaffected by margin changes and metric tightening; Area A remains mandatory for 100% competitive.

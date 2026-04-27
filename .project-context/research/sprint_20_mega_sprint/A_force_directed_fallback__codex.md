# Force-directed fallback for no-hierarchy graphs

Agent: sprint-20 research agent A, Codex GPT-5.5
Date: 2026-04-24
Scope: read-only empirical research. No code edits.

## TL;DR

- Do not replace `dagua_native` with the existing standalone FR, FA2, SFDP, or
  stress-SGD pipelines for small-world graphs. Measured standalone scores are
  worse than current default: `small_world_100` default 48.58 vs best requested
  standalone `kk` 37.92; `small_world_500` default 49.34 vs best requested
  standalone `kk` 40.94; `parallel_cycles_4x5` default 58.24 vs best requested
  standalone `kk` 45.73.
- The promising fallback is not raw force-directed layout. It is
  **stress-majorization or KK seed, then deterministic node-box extent scaling,
  then normal overlap/aspect post-processing**. A no-code prototype that scales
  stress-majorization to `overlap_count == 0` scored 56.98 on
  `small_world_100` and 52.32 on `small_world_500`, vs current 48.58 and 49.34.
  Five stress-majorization seeds raised `small_world_100` to 57.59, slightly
  above the cached `igraph_sugiyama` 57.09.
- The dispatch gate should be narrow: connected, cyclic, collapsed layering
  (`is_directed_acyclic == False`, `num_layers <= 1`), non-tree, no clusters or
  pins, and probably `N >= 20`. This catches `small_world_100`,
  `small_world_500`, `small_world_2000`, and `parallel_cycles_4x5`; it does not
  fire on protected layered wins like `org_chart_deep`, `random_dag_200`,
  `random_bipartite_60`, or the protected karate layouts.
- `parallel_cycles_4x5` is not a good target for the force fallback. Current
  native is 58.24; scaled stress-majorization is 54.46, scaled KK is 53.87, and
  UMAP is 53.05. Keep it layered/component-cyclic for now, or handle it with a
  tiny-cycle-specific circular or ladder layout.
- Projected gain for a first implementation: +8.4 to +9.0 on
  `small_world_100`, +3.0 on `small_world_500`, and no intended change on
  protected DAG wins. Runtime cost is acceptable only if gated: about 0.7 to
  1.3s for stress-majorization at 100 nodes, 11 to 25s at 500 nodes on this
  CPU-only workstation under concurrent load. Multi-start is useful at 100
  nodes, not worth it at 500 nodes unless cached or parallelized.

## Findings

### Finding 1 - Raw existing force pipelines do not beat default under current composite

Severity: high.

The requested measurement template was run exactly with CPU forced by
`CUDA_VISIBLE_DEVICES=""` and seed 42. Results:

| graph | default | fr | fa2 | kk | stress_sgd | sfdp | best requested standalone |
|---|---:|---:|---:|---:|---:|---:|---|
| `small_world_100` | 48.58 | 27.74 | 28.32 | 37.92 | 27.74 | 30.64 | `kk` 37.92 |
| `small_world_500` | 49.34 | 26.69 | 26.28 | 40.94 | 26.69 | 24.42 | `kk` 40.94 |
| `parallel_cycles_4x5` | 58.24 | 30.10 | 31.18 | 45.73 | 34.06 | 37.20 | `kk` 45.73 |

This answers question 1 directly: among `fr`, `fa2`, `kk`, `stress_sgd`,
`sfdp`, and the requested standalone set, `kk` is the best completed pipeline
on all three target graphs, but it is still worse than current default on all
three.

UMAP was measured separately because it is slow and not in the exact template.
It scored 36.40 on `small_world_100` in 62.122s, timed out without a
`small_world_500` score at 180s, and scored 52.43 to 53.05 on
`parallel_cycles_4x5` in 15.6 to 17.7s. That is still below current native
58.24 for `parallel_cycles_4x5`. I do not recommend UMAP for the default
fallback.

The core reason raw force layouts lose is visible in the metric breakdown. For
`small_world_500`, raw KK has good crossing rate 0.0006 and reasonable edge
length CV 0.280, but `overlap_count == 124750`, so it loses the full 10-point
overlap term from `metrics.py:1195`. Raw stress-majorization has the same
problem: 42.31 composite, crossing 0.0006, CV 0.233, but 72453 overlaps.
Standalone force pipelines produce coordinates in abstract unit-like space;
Dagua's composite scores actual node boxes.

Proposed change: never dispatch directly to raw FR/FA2/KK/stress as the final
layout. Any force fallback must include a box-aware scaling/projection
post-stage before computing final positions.

### Finding 2 - Box-aware scale normalization turns stress-majorization into a real competitor

Severity: high.

I prototyped a read-only transform outside the codebase: center the force
positions, compute the minimum uniform scale needed so every node box pair is
separated on at least one axis, multiply by that scale, then score with
`full()` and `composite()`. This does not change edge crossings, angular
ordering, or edge length CV; it only makes force coordinates compatible with
Dagua's node extents and recovers the 10-point no-overlap term.

Measured results:

| graph | algo | raw | scaled to no overlaps | delta | scaled details |
|---|---:|---:|---:|---:|---|
| `small_world_100` | KK | 38.00 | 47.87 | +9.87 | `overlap_count` 4950 -> 0 |
| `small_world_100` | stress_majorization | 46.99 | 56.98 | +9.99 | CV 0.158, crossing 0.0002, angle 43.18 |
| `small_world_100` | SFDP | 30.58 | 40.65 | +10.07 | still weak CV/angle |
| `small_world_500` | KK | 40.94 | 50.94 | +10.00 | CV 0.280, crossing 0.0005 |
| `small_world_500` | stress_majorization | 42.32 | 52.32 | +10.00 | CV 0.233, crossing 0.0005 |
| `small_world_500` | SFDP | 24.42 | 34.41 | +9.99 | CV 0.972, weak |
| `parallel_cycles_4x5` | KK | 45.78 | 53.87 | +8.09 | still below default 58.24 |
| `parallel_cycles_4x5` | stress_majorization | 45.73 | 54.46 | +8.73 | still below default 58.24 |
| `parallel_cycles_4x5` | UMAP | 53.05 | 52.44 | -0.61 | already overlap-free |

This is the main empirical result. It changes the story from "force-directed
fallback is bad" to "force-directed fallback must be box-aware." The current
pipeline already has `OverlapProjection` late in `dagua_native.py:1150`, but
the standalone force pipelines do not run this path. The fallback should either
compose force output into a small postprocess pipeline or call a shared
`ScaleToNodeBoxes` op before `OverlapProjection`.

Proposed change: implement a `ScaleToNodeBoxes` or `NormalizeForceExtent` op
that runs after force/stress layout and before final overlap projection. The op
should compute a conservative uniform scale from pairwise node extents for
`N <= 1000`, with a grid or sample approximation above that. This is
non-differentiable, deterministic, cheap relative to stress-majorization, and
fits the project direction of using non-differentiable wins where useful.

### Finding 3 - Stress-majorization beats pure FR/FA2 after scaling; stress-SGD does not

Severity: high.

Question 4 asks whether stress-majorization (`stress_sgd` in the prompt) beats
pure FR/FA2 and what it costs at 500 nodes. The answer is split:

- `stress_sgd` as implemented behaves like FR on the targets. At
  `small_world_500`, `fr` scored 26.69 in 0.001s and `stress_sgd` scored 26.69
  in 0.565s. Both had crossing around 0.2365 and poor angular resolution.
  `stress_sgd` is not the fallback candidate.
- `stress_majorization` is the useful existing pipeline. At
  `small_world_500`, raw score was 42.32 in 21 to 25s, and scaled score was
  52.32. It beats scaled KK 50.94 and raw default 49.34, but still trails
  cached `elk_layered` 54.16.

Five-seed stress-majorization results:

| graph | seed | scaled score | layout time | selected metrics |
|---|---:|---:|---:|---|
| `small_world_100` | 1 | 57.59 | 1.347s | dag 0.505, CV 0.131, crossing 0.0001, angle 48.68 |
| `small_world_100` | 2 | 57.08 | 0.949s | dag 0.495, CV 0.144, crossing 0.0001, angle 45.76 |
| `small_world_100` | 3 | 56.83 | 0.787s | dag 0.500, CV 0.158, crossing 0.0002, angle 43.19 |
| `small_world_100` | 42 | 56.98 | 0.730s | dag 0.505, CV 0.158, crossing 0.0003, angle 43.18 |
| `small_world_100` | 99 | 57.14 | 0.824s | dag 0.510, CV 0.158, crossing 0.0005, angle 43.06 |
| `small_world_500` | 1 | 52.26 | 10.934s | dag 0.502, CV 0.231, crossing 0.0005, angle 14.86 |
| `small_world_500` | 2 | 52.28 | 21.838s | dag 0.503, CV 0.232, crossing 0.0005, angle 14.99 |
| `small_world_500` | 3 | 52.18 | 22.203s | dag 0.499, CV 0.233, crossing 0.0006, angle 15.29 |
| `small_world_500` | 42 | 52.32 | 24.730s | dag 0.504, CV 0.233, crossing 0.0005, angle 14.91 |
| `small_world_500` | 99 | 52.18 | 23.617s | dag 0.501, CV 0.233, crossing 0.0005, angle 14.74 |

Multi-start is worth it for `small_world_100`: best seed 57.59 beats the
cached `igraph_sugiyama` score 57.09 and current default by +9.01. It is not
worth it for `small_world_500`: all five seeds cluster tightly between 52.18
and 52.32, so the extra 40 to 90 seconds would buy almost nothing.

Proposed change: for `N <= 150`, run 3 stress-majorization seeds and choose
the best cheap proxy score after scaling. For `N > 150`, run one
stress-majorization seed, or use KK as a faster fallback if runtime budget is
tight. Do not use `stress_sgd` for this gate unless it is reworked.

### Finding 4 - The topology signature is collapsed cyclic layering, not "undirected-looking"

Severity: high.

`graph_classify.py` already computes the right low-cost ingredients:
`is_directed_acyclic`, `is_acyclic`, component count, degree, edge/node ratio,
layer count, layer width, and topology tags (`graph_classify.py:31-46`,
`graph_classify.py:452-521`). The measured signatures:

| graph | N | E | family | components | layers | max layer width | E/N | max degree | directed DAG? | undirected acyclic? |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| `small_world_100` | 100 | 200 | WIDE_LAYERED | 1 | 1 | 100 | 2.00 | 4 | False | False |
| `small_world_500` | 500 | 1500 | WIDE_LAYERED | 1 | 1 | 500 | 3.00 | 6 | False | False |
| `small_world_2000` | 2000 | 4000 | WIDE_LAYERED | 1 | 1 | 2000 | 2.00 | 4 | False | False |
| `parallel_cycles_4x5` | 20 | 20 | GENERAL | 1 | 1 | 20 | 1.00 | 2 | False | False |
| `disconnected_label_cycle_collage` | 7 | 6 | GENERAL | 3 | 3 | 3 | 0.86 | 4 | False | False |
| `regular_3_30` | 30 | 45 | GENERAL | 1 | 7 | 7 | 1.50 | 3 | True | False |
| `weighted_karate_34` | 34 | 78 | GENERAL | 1 | 7 | 9 | 2.29 | 17 | True | False |
| `real_karate_34` | 34 | 78 | GENERAL | 1 | 7 | 9 | 2.29 | 17 | True | False |

The protected wins matter. `weighted_karate_34` and `real_karate_34` are
semantically undirected, but current Dagua is a protected win at 71.68 vs best
competitor around 59.37 in the shared context. Their measured classifier state
is directed-acyclic under the input orientation, seven layers, max degree 17.
A broad "underlying graph has cycles" or "social-looking graph" gate would
risk these wins. Similarly, many DAG benchmarks are undirected-cyclic in the
weak sense because skip connections make undirected cycles; `grid_5x5`,
`hexagonal_lattice_42`, `sierpinski_42`, `random_bipartite_60`, and
`regular_3_30` all have `is_acyclic == False` but should not automatically be
force-directed.

Recommended gate:

```text
needs_no_hierarchy_force =
    config.algorithm is None
    and no clusters
    and no pins / cross-component flex
    and structure.num_components == 1
    and structure.num_nodes >= 20
    and not structure.is_directed_acyclic
    and structure.num_layers <= 1
    and structure.max_layer_width >= 0.80 * num_nodes
    and 0.75 <= structure.edge_to_node_ratio <= 3.25
    and structure.max_degree <= max(12, 0.05 * num_nodes)
```

This gate catches the small-world family and `parallel_cycles_4x5`, but the
implementation should additionally exclude tiny/simple cycles from
stress-majorization unless testing shows a win. I would initially route only
`N >= 50` through stress-majorization. For `20 <= N < 50`, measure a
cycle-specific handler first, because `parallel_cycles_4x5` lost 3.78 points
with the best scaled stress result.

Where to dispatch: not inside `engine.layout()` as a new public algorithm
override. `engine.py:936-950` remaps `algorithm=None` to `dagua_native` and
then calls a registered pipeline. Keep that path clean. Put the gate either in
`dagua_native.layout_dagua_native_pipeline()` after `_prepare_native_config()`
has a `GraphStructure`, or in a small `DispatchByTopology` op/factory called by
`build_dagua_pipeline()`. The current native pipeline already has a limited
cyclic init fallback at `dagua_native.py:1100-1109`; this sprint-20 fallback is
a stronger sibling for the case where the whole solved graph has no hierarchy.

### Finding 5 - Competitors win small-world by metric-aligned hierarchy, not force aesthetics

Severity: medium.

Cached competitor layouts show why the target is tricky. For
`small_world_100`, the top cached competitors are:

| engine | composite | dag | CV | straight | crossing | angle | overlaps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `igraph_sugiyama` | 57.09 | 0.985 | 3.727 | 22.67 | 0.0000 | 42.29 | 0 |
| `graphviz_dot` | 56.70 | 0.985 | 3.830 | 18.60 | 0.0076 | 35.78 | 0 |
| `elk_layered` | 56.38 | 0.985 | 3.918 | 13.60 | 0.0062 | 23.21 | 0 |
| `dagre` | 52.31 | 0.985 | 2.676 | 38.11 | 0.0018 | 30.73 | 0 |

For `small_world_500`, top competitors:

| engine | composite | dag | CV | straight | crossing | angle | overlaps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `elk_layered` | 54.16 | 0.995 | 7.630 | 17.47 | 0.0019 | 6.82 | 0 |
| `graphviz_dot` | 53.05 | 0.996 | 7.441 | 22.31 | 0.0014 | 6.00 | 0 |
| `igraph_sugiyama` | 51.53 | 0.996 | 2.542 | 29.31 | 0.0008 | 5.81 | 0 |
| `dagre` | 47.56 | 0.996 | 1.377 | 50.81 | 0.0014 | 2.46 | 0 |

The competitors sacrifice the 20-point edge-length-CV term almost completely
on small-world graphs, but they gain nearly all 25 points of DAG consistency,
keep overlaps at zero, and mostly avoid crossings. Scaled stress-majorization
takes the opposite tradeoff: DAG consistency stays around 0.50, but edge length
CV and angular resolution are much better. That is why scaled stress is enough
to close `small_world_100`, but only partially closes `small_world_500`.

Implication: there are two viable directions. The conservative implementation
is scaled stress fallback. The ambitious implementation is a directed ordering
stage for cyclic graphs that gets DAG consistency near 0.98 while keeping
stress-like edge uniformity. That would require a circular/spectral ordering or
minimum feedback arc ordering followed by constrained stress, not plain force.

## Big-bet proposals

### Proposal A - `cyclic_stress_native` fallback pipeline

Build a new internal pipeline, not a public user-facing algorithm name at first:

1. Run `stress_majorization` for collapsed cyclic graphs.
2. Uniformly scale positions to node extents with padding, using the measured
   exact pairwise scale for small graphs.
3. Run `OverlapProjection` for final collision cleanup.
4. Run `AspectRatioFit` with square-ish target, not layered 0.25.
5. Optionally run a local edge-crossing or angular-resolution polish that does
   not re-layer.

Expected impact from measured prototype:

| graph | current | prototype evidence | projected after real op |
|---|---:|---:|---:|
| `small_world_100` | 48.58 | scaled stress seed 42 = 56.98, best of 5 = 57.59 | 56.5 to 57.5 |
| `small_world_500` | 49.34 | scaled stress seed 42 = 52.32 | 52.0 to 53.0 |
| `parallel_cycles_4x5` | 58.24 | scaled stress = 54.46 | no dispatch |

Runtime: +0.7 to 1.3s at 100 nodes, +11 to 25s at 500 nodes on CPU. This is
large compared with current native `small_world_500` layout time around 0.08s
in the standalone timing run, so the gate must be narrow. If runtime budget is
tight, use scaled KK at 500 nodes: score 50.94 and measured layout time 12.776s
in the scaled run, still +1.60 over current but weaker than stress.

### Proposal B - Multi-start only for small collapsed cyclic graphs

Use multi-start stress-majorization for `50 <= N <= 150`. Five measured seeds
on `small_world_100` ranged 56.83 to 57.59, with best seed 1 beating the
cached best competitor by +0.50. For 500 nodes, five seeds ranged only 52.18 to
52.32; multi-start is wasted there.

Implementation detail: score candidates with a cheap proxy before running
`full()`: no-overlap scale factor, sampled crossing estimate, edge length CV,
and angular resolution. The public composite is too slow to run repeatedly
inside layout for 500-node graphs.

### Proposal C - Constrained cyclic ordering as the real long-term win

The remaining gap on `small_world_500` is DAG consistency. Stress layouts sit
near 0.50 because directions are visually arbitrary. Competitors get 0.995 by
finding a near-total vertical ordering. A modern high-reward path is:

1. Compute an order from spectral/Fiedler coordinates or minimum-feedback-arc
   heuristic.
2. Use that order as a soft y constraint.
3. Run stress/KK in x with weak y monotonicity, not full layered y ranks.
4. Scale/project boxes.

This could keep stress's CV advantage while recovering 10 to 12 of the lost
DAG-consistency points. It is riskier because it may recreate the current
over-layering failure if y is too strong.

### Proposal D - Modern techniques worth considering

- Maxent-stress exists (`maxent_stress` pipeline) but measured lower than
  stress-majorization here: `small_world_500` raw 36.74 and many overlaps.
  Keep it as a later candidate if its extent handling is fixed.
- ForceAtlas2 LinLog exists as `linlog`, but measured weak:
  `small_world_100` 27.11, `small_world_500` 25.58, `parallel_cycles_4x5`
  44.06. Do not prioritize.
- UMAP is too slow and not high enough: 36.40 on `small_world_100`, timeout at
  180s on `small_world_500`, about 53 on `parallel_cycles_4x5`.
- Literature directions like constrained stress (IPSep-CoLa style), PRISM-like
  overlap removal, and topology-aware untangling are more relevant than
  another raw force model. Gorochowski 2022 did not surface as a clearly
  actionable graph-drawing technique in this pass; if the intended reference is
  a specific paper, it should be evaluated by the modern-techniques agent.

## Risk / regression analysis

The protected wins in `CONTEXT.md` are primarily layered or tree/DAG shapes:
`org_chart_deep`, `random_dag_200`, `hub_fanout_label_skew`,
`org_chart_1_5_4_8`, `random_dag_50`, `random_bipartite_60`,
`edge_label_braid`, `bipartite_4_3_4`, `weighted_karate_34`, and
`real_karate_34`.

Measured classifier state for protected examples:

| protected graph | directed DAG? | layers | family/tags | gate fires? |
|---|---|---:|---|---|
| `org_chart_deep` | True | 6 | TREE | no |
| `random_dag_200` | True | 10 | GENERAL, 202 components | no |
| `hub_fanout_label_skew` | True | 5 | GENERAL | no |
| `org_chart_1_5_4_8` | True | 4 | TREE | no |
| `random_dag_50` | True | 12 | GENERAL, 52 components | no |
| `random_bipartite_60` | True | 2 | BIPARTITE_DAG, wide_layered | no |
| `edge_label_braid` | True | 4 | GENERAL | no |
| `bipartite_4_3_4` | True | 3 | GENERAL | no |
| `weighted_karate_34` | True | 7 | GENERAL, max degree 17 | no |
| `real_karate_34` | True | 7 | GENERAL, max degree 17 | no |

The risky cases are not the protected wins; they are cyclic small graphs such
as `recurrent_feedback_cell`, `center_port_backedge_hub`,
`disconnected_label_cycle_collage`, and `parallel_cycles_4x5`. The proposed
`N >= 50` first implementation avoids those. For `parallel_cycles_4x5`, current
native is 58.24 and the best scaled force measurement was 54.46, so dispatching
there would be a regression.

Another risk is runtime. Current default on `small_world_500` measured 0.076s
in the broad timing run, while stress-majorization seed 42 measured 24.730s in
the five-seed run. This is a very large cost for +2.98 composite. The first
implementation should either:

- gate to `small_world_100`-scale graphs only, or
- use a budget flag such as `force_fallback_max_nodes=200` initially, or
- implement faster pivot stress / KK fallback before enabling 500-node stress.

Finally, do not broaden `graph_classify` semantics to label all
undirected-cyclic graphs as force-directed. Many Dagua benchmark DAGs have
undirected cycles but meaningful directed layering. The measured karate
examples are especially dangerous because they are semantically undirected but
protected by the current metric.

## Implementation order

1. Add a read-only benchmark script first. Capture current default,
   raw force, scaled force, and competitor cached scores for
   `small_world_100`, `small_world_500`, `parallel_cycles_4x5`, and the ten
   protected wins. This prevents tuning against only the two small-world
   graphs.
2. Implement `ScaleToNodeBoxes` as a standalone postprocess op. It should take
   `state.pos` and `problem.node_sizes`, compute the conservative uniform scale
   measured in this report, and preserve the center. Unit-test it directly:
   synthetic overlapping force positions should become `overlap_count == 0`.
3. Build an internal `cyclic_stress_native` pipeline: `stress_majorization`
   seed -> `ScaleToNodeBoxes` -> `OverlapProjection` -> square-ish
   `AspectRatioFit`. Do not wire it to default yet.
4. Add a conservative dispatch gate in native adapter code using the measured
   topology fields: connected, non-DAG, `num_layers <= 1`, giant single layer,
   `N >= 50`, no clusters, no pins. Initially set `max_nodes=150` so only the
   high-confidence `small_world_100` class gets the multi-start win.
5. Run protected-win regression measurements. The gate should not fire on any
   protected graph; if it does, tighten before comparing composites.
6. Extend to 500 nodes only after runtime is addressed. Options: use scaled KK
   for 150 to 1000 nodes, implement pivot stress, or add a runtime budget that
   skips fallback when a single stress solve exceeds a threshold.
7. Later, research constrained cyclic ordering to recover DAG consistency on
   `small_world_500`. That is the path to beating 54.16 rather than merely
   narrowing the gap.

## Assumptions and concerns

- Measurements were CPU-only with `CUDA_VISIBLE_DEVICES=""`. Several other
  sprint agents were active in the same workspace, so absolute wall times are
  noisy. Composite numbers are the important result; timings should be
  rechecked in an isolated run before landing a runtime-sensitive default.
- UMAP `small_world_500` has no composite number from this pass because it did
  not finish within a 180s timeout. I recorded that as a timeout rather than
  inventing a number.
- The box-scaling prototype is not an existing op. Its scores are real
  composite measurements on real positions, but they are projected as
  implementation evidence until the op lands.
- I did not modify code, did not run ruff, and did not run tests because this
  task was read-only markdown research.

## Knowledge worth carrying forward

- The current default already has a cyclic flat init fallback
  (`Force2DInitIfFlat`) at `dagua_native.py:1100-1109`. Sprint 20 should not
  duplicate that; it should add a final-layout fallback when the entire graph
  has no hierarchy.
- Raw force layouts are misleadingly bad under Dagua's composite because
  coordinate scale is not box-aware. Always evaluate force candidates after
  extent normalization or overlap projection.
- `stress_majorization`, not `stress_sgd`, is the high-quality existing stress
  pipeline for the small-world target.
- `parallel_cycles_4x5` should stay out of the first force fallback despite
  being cyclic and collapsed; current native is better than all measured force
  variants.

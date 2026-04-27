# Area F -- Metric-aware Adaptive Routing (codex)

## TL;DR

- Do **not** run all native pipelines by default. The current dispatcher plus
  polish is already the winner or tied winner on 26/28 probed graphs; all-six
  routing buys little for a 3.8x median-ish cost and a 4-6x worst-case mental
  model that is real on larger layouts.
- Add a **two-candidate metric-aware guard**, not a full tournament: run the
  classifier-selected primary, run one topology-specific alternate only when
  the graph is small enough and has headroom, polish both eligible outputs,
  score with `composite(full())`, and accept only with a +0.5 margin.
- The highest-value miss is not "try force-directed more"; it is
  **multi-component near-tree structure**. In the probe, `multi_component_80`
  scores 74.46 under current auto routing but 87.83 with the tree pipeline
  (+13.37). The child component path currently forces `legacy_monolith`, which
  prevents component-local tree routing.
- `force_directed` and `planar` should almost never be secondary candidates.
  In the measured sample, neither won once; `force_directed` was usually down
  by 20-40 composite points.
- Make polish a candidate-scoring policy, not an auto-route side effect.
  Today `_run_native_problem` only calls `_best_of_polish` when
  `force_pipeline is None`, so forced-pipeline measurements are lower bounds
  for "pipeline + polish" unless the implementation explicitly polishes forced
  candidates.

## Current Dispatch Surface

The native dispatcher is a one-shot choice. `_choose_native_pipeline` accepts a
forced override first, then maps classified topology to `tree`, `planar`,
`force_directed`, `hybrid`, or `layered_dag` using a small decision table
(`dagua/layout/ops/pipelines/dagua_native.py:67`). `build_dagua_pipeline`
then materializes exactly one selected pipeline (`dagua_native.py:114`).

The polish step is downstream, but it is not pipeline-agnostic today. In
`_run_native_problem`, polish is gated on `edge_equalize_polish`, eligible
pipeline names, `N >= 6`, non-empty edges, node sizes, and critically
`_selected_force_pipeline(config) is None` (`dagua_native.py:364-372`). The
per-component tiled output has the same auto-only polish gate
(`dagua_native.py:1373-1380`). Therefore an implementation that compares
candidate pipelines must either:

1. run candidates without setting public `force_pipeline`, or
2. invoke the same polish scorer inside the metric-aware candidate runner.

Otherwise the primary auto route receives polish and forced alternates do not,
which biases the tournament.

One more routing blind spot is visible in the component path. When native
component decomposition is active, child configs are prepared per component,
but a child with no explicit force override is then forced to
`legacy_monolith` (`dagua_native.py:1347-1352`). That preserves an older
component-packing win, but it blocks per-component reclassification into
`tree`, which is exactly the case the pipeline-level picker is meant to catch.

## Cost-Benefit Analysis

Evidence source: `/tmp/probe_pipelines_results.csv`, generated on 28 benchmark
graphs across trees, chains, grids, lattices, DAGs, cyclic graphs, planar
graphs, real-world graphs, compound graphs, and disconnected graphs. It scores
`auto` plus forced `tree`, `layered_dag`, `force_directed`, `hybrid`, `planar`,
and `legacy_monolith` by deterministic `composite(full())`, seed 42. Important
measurement caveat: current `auto` includes polish; forced variants are raw
pipeline outputs because the current code suppresses polish when
`force_pipeline` is set. Treat forced numbers as lower bounds for a future fair
"pipeline + polish" tournament.

Aggregate results:

| Strategy | Runtime evidence | Composite evidence | Recommendation |
|---|---:|---:|---|
| Current auto + polish | 41.84s total over 28 graphs; 1.49s mean, 0.52s median | Winner/tied on 26/28 | Keep as primary |
| All valid pipelines | Mean multiplier 3.79x over auto on the sample; outlier multiplier 22x on tiny/fast auto cases | Only 2 material auto misses > +0.5 | Too expensive as default |
| Auto + one topology alternate | Mean multiplier 1.42x, median 1.25x in a simple simulated table | Catches the tree-family miss and can be extended for multi-component trees | Best production shape |

The headline "3-6x cost" is directionally correct, but the measured cost is
more nuanced. Some forced paths are nearly free (`tree` is effectively 0.00s
on small samples), while others are very expensive: `planar_60` took 2.69s
auto, 3.54s planar; `dependency_graph_100` took 2.44s auto and 11.61s
legacy; `grid_20x20` took 5.51s auto, 8.76s layered, 11.07s force-directed,
and 8.12s planar. The cost risk is not the second candidate itself; it is
choosing the wrong second candidate on medium graphs.

The benefit is also concentrated. In the 28-graph probe, auto was beaten by
more than 0.5 only on:

- `org_chart_deep`: auto 91.64, tree 92.66, +1.02. The classifier family is
  `TREE`, but `N=79` exceeds `small_n_tree_cutoff=64`, so the dispatcher falls
  through to `layered_dag`.
- `multi_component_80`: auto 74.46, tree 87.83, +13.37. The global family is
  `GENERAL`, but component-local tree structure is the right model.

The expected suite-average gain from a safe two-candidate picker is therefore
small but worthwhile: roughly +0.1 to +0.3 mean composite if only the known
misses exist, with higher value as an insurance policy against classifier
edge cases. Full all-pipeline routing is not justified unless run in an
offline diagnostic mode.

## Empirical Pipeline-vs-Graph Mapping

Scores below are current auto-with-polish versus forced raw pipeline scores.
They are still useful because they identify which pipeline families are even
competitive before fair forced-candidate polish is wired.

| Graph | Classifier family | Auto | Tree | Layered | Force | Hybrid | Planar | Legacy | Winner |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| binary_tree | TREE | 92.54 | 92.54 | 91.89 | 42.66 | 91.89 | 68.56 | 91.89 | auto/tree |
| deep_chain_20 | CHAIN | 97.50 | 97.50 | 97.50 | 41.51 | 97.49 | 70.63 | 97.50 | tie |
| org_chart_deep | TREE | 91.64 | 92.66 | 91.64 | 36.97 | 91.64 | 59.32 | 91.64 | tree |
| grid_5x5 | GENERAL | 94.01 | 86.44 | 94.01 | 39.06 | 94.07 | 47.62 | 94.01 | hybrid tiny |
| grid_rect_6x8 | GENERAL | 92.92 | 84.01 | 92.92 | 43.52 | 92.98 | 27.71 | 92.92 | hybrid tiny |
| grid_20x20 | GENERAL | 92.92 | 83.12 | 92.92 | 27.90 | 93.01 | 24.57 | 92.92 | hybrid tiny |
| hexagonal_lattice_42 | GENERAL | 86.46 | 84.40 | 85.45 | 37.44 | ERR | 42.36 | 85.45 | auto polish |
| triangular_lattice_36 | GENERAL | 85.48 | 72.39 | 84.89 | 48.38 | ERR | 30.86 | 84.89 | auto polish |
| transformer_layer | GENERAL | 80.94 | 73.44 | 79.98 | 33.50 | 80.15 | 58.74 | 79.98 | auto polish |
| ragged_feature_pyramid | GENERAL | 81.18 | 75.02 | 73.81 | 55.70 | 73.81 | 66.66 | 73.81 | auto polish |
| resnet_stack_4x16 | GENERAL | 78.50 | 72.04 | 77.50 | 32.20 | 77.50 | 57.67 | 77.50 | auto polish |
| random_dag_50 | GENERAL | 70.12 | 56.47 | 70.12 | 41.15 | ERR | ERR | 62.16 | auto/layered |
| dependency_graph_100 | GENERAL | 59.47 | 42.89 | 59.47 | 28.81 | ERR | ERR | 59.47 | tie |
| petersen_10 | GENERAL | 74.64 | 63.57 | 70.69 | 28.86 | 70.69 | ERR | 70.69 | auto polish |
| small_world_100 | WIDE_LAYERED | 57.18 | 52.25 | 49.20 | 31.71 | 49.56 | 36.70 | 57.18 | auto/stress |
| regular_3_30 | GENERAL | 77.17 | 56.01 | 74.26 | 42.81 | ERR | ERR | 74.26 | auto polish |
| regular_4_40 | GENERAL | 69.75 | 51.42 | 68.05 | 36.21 | ERR | ERR | 68.05 | auto polish |
| er_100 | GENERAL | 62.70 | 54.65 | 61.58 | 28.63 | ERR | ERR | 61.58 | auto polish |
| real_karate_34 | GENERAL | 72.86 | 53.45 | 72.36 | 33.81 | ERR | ERR | 72.36 | auto polish |
| outerplanar_dag_20 | GENERAL | 72.42 | 58.96 | 72.42 | 31.31 | 72.42 | 62.78 | 72.42 | tie |
| planar_60 | GENERAL | 78.74 | 64.22 | 78.74 | 29.44 | 78.74 | 34.10 | 78.74 | tie |
| sierpinski_42 | GENERAL | 85.43 | 77.26 | 81.86 | 36.58 | ERR | 47.50 | 81.86 | auto polish |
| disconnected_label_cycle_collage | FORCE_DIRECTED | 77.37 | 62.92 | 74.41 | 55.15 | 74.41 | ERR | 74.41 | auto polish |
| multi_component_80 | GENERAL | 74.46 | 87.83 | 74.46 | 33.46 | ERR | 53.67 | 74.46 | tree |
| compound_10x20 | GENERAL | 77.50 | 61.75 | 77.50 | 36.02 | 77.50 | ERR | 77.50 | tie |
| clustered_medium_5x20 | GENERAL | 69.78 | 49.11 | 69.78 | 35.63 | ERR | ERR | 69.78 | tie |

Confusion summary: `auto` wins or ties 23 rows outright and is within 0.5 of
the best on 26/28. `tree` wins two material cases. `hybrid` wins three grid
cases by only +0.06 to +0.09, below the recommended acceptance margin.
`layered_dag`, `force_directed`, `planar`, and `legacy_monolith` have zero
material wins as forced raw secondaries in this sample.

## Recommended Candidate Subsets

Use the current classifier result as primary. Add at most one secondary:

| Topology condition | Primary | Secondary | Gate |
|---|---|---|---|
| `TREE` or `CHAIN`, `N <= small_n_tree_cutoff` | tree | layered_dag | Skip if primary score >= 95 |
| `TREE` or `CHAIN`, `N > small_n_tree_cutoff` | layered_dag | tree | Always try while `N <= 300`; catches `org_chart_deep` |
| `GENERAL`, weak components mostly tree-like | current auto/per-component | per-component tree | Highest priority; require no clusters, no pins, no cross-component flex |
| `GENERAL`, lattice/grid tags or low cyclicity planar-ish | layered_dag/hybrid | hybrid or layered_dag opposite | Accept only +0.5; measured gains are tiny |
| `HYBRID` or cyclicity > 0.05 | hybrid | layered_dag | Avoid force-directed secondary |
| `FORCE_DIRECTED` with high cyclicity | force_directed only if current rule chooses it | hybrid | Current rule rarely reaches this; score guard required |
| Planar opt-in | planar | layered_dag | Keep existing fallback behavior |
| Stress-route flat cyclic graphs | stress route | none initially | The route exists outside `build_dagua_pipeline`; do not perturb first pass |

I would not include `planar` or `force_directed` in generic shortlists. They
are useful as explicit user overrides and diagnostic algorithms, not as
default tournament candidates under the current composite.

## Implementation Sketch

Add an internal candidate runner near `_run_native_problem`, not in
`build_dagua_pipeline`. `build_dagua_pipeline` should remain the factory for a
single selected pipeline; the metric-aware feature is an orchestration layer
that builds and scores multiple single-pipeline runs.

Sketch:

```python
primary = _choose_native_pipeline(structure, config)
primary_pos = _run_one_pipeline(problem, state, ctx, config, primary)
primary_pos = _polish_for_candidate(primary, primary_pos, problem, config)
primary_score = _score_native_result(primary_pos, problem.edge_index, problem.node_sizes)

if not _should_try_metric_aware_alt(problem, structure, primary_score, config):
    return primary_pos

secondary = _secondary_native_candidate(problem, structure, primary, config)
secondary_pos = _run_one_pipeline(problem, state, ctx, config, secondary)
secondary_pos = _polish_for_candidate(secondary, secondary_pos, problem, config)
secondary_score = _score_native_result(secondary_pos, problem.edge_index, problem.node_sizes)
return secondary_pos if secondary_score > primary_score + 0.5 else primary_pos
```

Recommended gates:

- `metric_aware_routing=True`, default initially false or "auto-small" until a
  93-graph timing pass confirms cost.
- `num_nodes <= 300` for the general two-candidate guard. Consider a lower
  default such as 150 if CI/runtime sensitivity is high.
- Skip when `primary_score >= 95.0`; there is little headroom and many trivial
  chains already hit 97.50.
- Skip with clusters, pins, or cross-component flex unless the existing
  component-decomposition safety gates already allow it.
- Use a +0.5 acceptance margin, matching polish's anti-regression policy.
- Log candidate name and score only under verbose/debug; do not expose
  benchmark-specific routing tables.

The first implementation step should actually be the component fix: allow
child problems to use their classified pipeline when their component family is
`TREE` or `CHAIN`, instead of blindly forcing `legacy_monolith`. This may solve
the biggest observed miss without paying a second full-graph layout cost.

## Regression Risks

- **Runtime regressions on medium graphs.** `dependency_graph_100` and
  `grid_20x20` show that the wrong alternate can be many seconds slower.
  The secondary table must be conservative.
- **Metric overfitting.** Routing by `composite(full())` optimizes exactly the
  benchmark metric. That is intentional here, but use a margin and avoid graph
  name lookup tables so unseen graphs still route structurally.
- **Polish asymmetry.** If forced alternates are compared without polish, the
  metric-aware picker will under-select them. If every alternate gets every
  polish candidate, runtime can multiply again. Solution: a per-pipeline polish
  policy.
- **Component-layout behavioral changes.** Removing the child
  `legacy_monolith` force could disturb protected Sprint-19d component packing.
  Keep existing decomposition gates, and only enable component-local tree
  routing when every nontrivial component classifies as `TREE` or `CHAIN`.
- **Tiny score flips on grids.** Hybrid beats auto/layered by +0.06 to +0.09 on
  three grids. Do not accept those; they are below meaningful margin and could
  churn across future polish changes.

## Bonus: Adaptive Polish By Candidate

Polish should be attached to the candidate scorer:

- `layered_dag`, `hybrid`, `legacy_monolith`: use the current `_best_of_polish`
  set. These are where auto-polish wins most of the sample.
- `tree`: score base first; run polish only if the graph is not a clean
  tree/chain or if base score is below 90. Tree layouts are often already
  metric-ceiling layouts, and unnecessary nudges risk damaging symmetry.
- `force_directed`: skip polish by default. It starts too far behind on this
  composite for edge equalization to rescue it.
- `planar`: skip except in explicit planar experiments. Current measured
  planar scores are consistently below layered/hybrid on the benchmark sample.
- Component-tiled output: keep polishing the final tiled result, because
  `disconnected_label_cycle_collage` gained from that path, but consider
  component-local polish only after the child routing fix is validated.

## Final Recommendation

Ship this in three passes. First, fix component-local tree routing behind the
existing decomposition gates. Second, add a two-candidate metric-aware guard
for `N <= 300` with a +0.5 score margin and no `force_directed`/`planar`
generic secondaries. Third, move polish into a per-candidate policy so future
pipeline-level comparisons are fair. Keep all-six pipeline tournaments as an
offline research command, not production dispatch.

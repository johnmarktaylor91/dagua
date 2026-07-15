# R74 CX6 Findings: cross-cutting methodology, divergent tails, strict 3Q audit

Read-only scope honored. No repository files were modified.

## Executive verdict

I do **not** find one global scale/aspect/RNG/reference-adapter bug that explains many divergent
families at once. The highest-repeat graphs are real hot spots, especially `random_dag_50`
and disconnected/multicomponent graphs, but their symptoms split by family and by metric direction.
The best cross-family conclusion is negative: disconnected/complex graph structure amplifies known
family-specific gaps, but it is not itself a single shared convention bug in the r73 data.

Under the existing strict 3Q gate, there are **0 legitimate rung-3 -> 3Q promotions** beyond the
current 36. I did not loosen the rule.

## Source and gate evidence

- `dagua/eval/distributional_fidelity.py:343-412` assigns rungs. The 3Q path is only
  `quality_identical` or `q_battery < 0.05`; rung 3 is only the looser stress-equivalent fallback.
- `scripts/definitive_fidelity_report.py:410-458` applies BH to `battery_p_iut -> q_battery`
  before rung assignment.
- `scripts/definitive_fidelity_report.py:2447-2449` defines the 3Q battery: stress 2%,
  crossings 2%/0.5 floor, kNN 0.02, with IUT max-p.
- `scripts/definitive_fidelity_report.py:2353-2385` implements the negative+chance anti-laundering
  gate: 3Q controls must be <= 5%.
- UMAP reference adapter computes a SciPy CSR graph-distance matrix at
  `dagua/eval/competitors/umap_competitor.py:76-85` and calls `umap.UMAP(metric="precomputed")`
  at `dagua/eval/competitors/umap_competitor.py:185-198`.
- UMAP native port sums unweighted duplicate edges into adjacency costs at
  `dagua/layout/ops/umap.py:123-183`, builds membership strengths at
  `dagua/layout/ops/umap.py:382-430`, prunes positive edges at `dagua/layout/ops/umap.py:904-945`,
  derives per-source Tausworthe states at `dagua/layout/ops/umap.py:1012-1043`, and updates sampled
  SGD edges at `dagua/layout/ops/umap.py:1169-1278`.
- UMAP upstream uses `fuzzy_simplicial_set` at
  `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/umap_.py:565-600`
  and sampled Euclidean optimization at
  `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/layouts.py:238-329`.
- DrL dagua presets are in `dagua/layout/ops/drl.py:185-230`, and the pipeline always creates
  `RuntimeContext(plan=ExecutionPlan(device="cpu"))` at `dagua/layout/ops/pipelines/drl.py:147-160`.
  Igraph DrL validates positive weights and constructs `drl::graph` at
  `/home/jtaylor/projects/_references/igraph/src/layout/drl/drl_layout.cpp:435-478`, initializes
  phase schedules at
  `/home/jtaylor/projects/_references/igraph/src/layout/drl/drl_graph.cpp:126-176`, and reads seeded
  coordinates only when `use_seed` is true at
  `/home/jtaylor/projects/_references/igraph/src/layout/drl/drl_layout.cpp:470-477`.
- Neato dagua splits components, solves each with `seed + component_index`, packs, then converts
  observable coordinates at `dagua/layout/ops/pipelines/neato.py:1368-1438`.
  Graphviz neato defaults to component packing via `getPackModeInfo`, `pinfo.mode = l_node`, and
  `packGraphs` at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatoinit.c:1371-1410`.
  Graphviz packs graph components through `packGraphs -> putGraphs -> shiftGraphs` at
  `/home/jtaylor/projects/_references/graphviz/lib/pack/pack.c:1092-1100`; pack mode comments show
  `l_node`/`l_graph` semantics at
  `/home/jtaylor/projects/_references/graphviz/lib/pack/pack.c:761-786`.

## Mission 1: systematic-pattern sweep

Input: `eval_output/fidelity_definitive_r73/per_combo.json`, 3,955 rows.

Rung counts:

| Rung | Count |
|---|---:|
| 1 | 2227 |
| 2 | 531 |
| 3 | 279 |
| 3Q | 36 |
| 4 | 574 |
| 2' | 51 |
| INSUFFICIENT_DATA | 257 |

Rung-4 by family:

| Family | Rung-4 rows |
|---|---:|
| sugiyama | 231 |
| sfdp | 184 |
| fmmm | 85 |
| classical_mds | 34 |
| gem | 22 |
| umap | 8 |
| drl | 5 |
| maxent | 3 |
| neato | 2 |

Graphs divergent across the most families:

| Graph | Divergent families | Rung-4 rows | Family split |
|---|---:|---:|---|
| `random_dag_50` | 8 | 20 | classical_mds 2, fmmm 4, gem 2, maxent 3, neato 1, sfdp 5, sugiyama 2, umap 1 |
| `random_dag_200` | 6 | 13 | classical_mds 2, fmmm 1, gem 1, sfdp 6, sugiyama 2, umap 1 |
| `parallel_cycles_4x5` | 5 | 11 | classical_mds 2, fmmm 1, neato 1, sfdp 6, sugiyama 1 |
| `kitchen_sink_platform_graph` | 4 | 15 | classical_mds 2, fmmm 1, sfdp 6, sugiyama 6 |
| `real_lesmis_77` | 4 | 14 | drl 2, gem 1, sfdp 6, sugiyama 5 |
| `multi_component_80` | 4 | 13 | classical_mds 2, fmmm 4, sfdp 6, sugiyama 1 |
| `disconnected_encoder_residual` | 4 | 13 | classical_mds 2, fmmm 4, sfdp 6, sugiyama 1 |
| `disconnected_label_cycle_collage` | 4 | 12 | classical_mds 2, fmmm 3, sfdp 6, sugiyama 1 |

Negatives ruled out:

- **Global scale/aspect bug: not supported.** Rung-4 stress direction is mixed: Dagua stress lower
  than reference in 312 rows and higher in 262 rows. By family, directions are also mixed:
  sfdp 87 lower / 97 higher, fmmm 40 / 45, sugiyama 143 / 88, neato 1 / 1. A global scale/aspect
  convention should not produce this balanced direction split, and stress is scale-optimized in
  `distributional_fidelity.py:241-269`.
- **Global disconnected-component bug: not supported.** Disconnected rows are enriched in rung 4
  but not dominant: rung-4 disconnected rate is 145/574 = 25.3%, versus rung-3 52/279 = 18.6%,
  rung-2 43/531 = 8.1%, rung-1 153/2227 = 6.9%. Some tail failures are disconnected
  (`neato` 2/2, `umap` 2/8), but DrL is 0/5 disconnected and the largest families include many
  connected failures.
- **Global RNG-stream bug: not supported.** Rung-4 has Mode A and Mode B populations:
  339 Mode A, 235 Mode B. Sugiyama is entirely Mode B in rung 4, while sfdp/fmmm/gem/umap/drl/neato
  are Mode A. One RNG-stream bug cannot explain deterministic-reference Mode B rows.
- **Global edge-multiplicity bug: not supported.** `parallel_multiedge_bundle` is only 7 rung-4
  rows total, 6 UMAP + 1 SFDP. It is high leverage for UMAP but not broad across families.

Best single cross-cutting fix, if forced:

- **Component/reference adapter audit for disconnected graph distances and packing.**
  It would plausibly affect a subset of `random_dag_50`, `random_dag_200`, `parallel_cycles_4x5`,
  and multi-component graphs, but the maximum honest scope is limited: neato 2 rows, some UMAP
  disconnected rows, and family-specific SFDP/FMMM/classical-MDS packing or distance rows. It is
  not a one-line global flip.

## Mission 2: divergent tails

### UMAP: 8 Mode A divergent rows

Rows:

| Graph | Engine rows | Signature |
|---|---:|---|
| `parallel_multiedge_bundle` | 6 | all UMAP variants except only by parameter; stress D is much lower than R in 5/6, crossings/kNN identical at 0/1 |
| `random_dag_50` | 1 | disconnected; e_rel 3.548; stress and kNN differ |
| `random_dag_200` | 1 | disconnected; e_rel 1.254; crossings close, kNN close |

The six `parallel_multiedge_bundle` rows are the clear UMAP lever. Dagua and the reference adapter
both construct graph-distance inputs, but this is exactly where multiplicity conventions matter:
the reference adapter builds a SciPy CSR from duplicated rows/cols
(`umap_competitor.py:76-85`), while dagua sums duplicate unweighted edges into path costs
(`umap.py:123-183`). Because these rows have perfect crossing and kNN means (`cross_D/R=0/0`,
`np_D/R=1/1`) yet fail distributional layout equivalence (`p_diff=9.999e-05`), this is not a
quality bug; it is a layout distribution/RNG or graph-distance convention mismatch.

Best achievable tier: likely rung 2 if adapter-path parity of CSR duplicate coalescing and UMAP
RNG state can be matched; otherwise 3Q would be invalid because current `q_battery=1.0` even when
quality looks visually identical for multiedge rows.

Concrete next experiment/fix:

1. On the benchmark path, for `parallel_multiedge_bundle`, dump the reference dense distance matrix
   from `umap_competitor._distance_matrix` and the dagua native distances after
   `ComputeAllPairsShortestPaths`.
2. Test duplicate-edge policies: summed cost, min cost/unit simple graph, and multiplicity as
   attraction weight after distances. The current code claims summed CSR parity; verify that with
   actual SciPy `shortest_path` on duplicate CSR.
3. If distances match, instrument initial spectral/random embedding and `_make_tau_state`
   (`umap.py:1012-1043`) against upstream `optimize_layout_euclidean` state
   (`umap/layouts.py:238-329`).

Confidence: high for `parallel_multiedge_bundle` as the UMAP root bucket; medium on exact fix until
distance dumps prove which convention is wrong. Effort: 0.5-1 day.

### DrL: 5 Mode A divergent rows

Rows:

| Graph | Variant | e_rel | Main battery gap |
|---|---|---:|---|
| `real_karate_34` | `classic_drl_coarsen` | 0.0136 | crossings 139.8 vs 135.3 |
| `real_karate_34` | `classic_drl_default` | 0.0219 | crossings 143.3 vs 138.4 |
| `real_karate_34` | `classic_drl_refine` | 0.00766 | crossings 338.1 vs 414.4 |
| `real_lesmis_77` | `classic_drl_coarsen` | 0.229 | stress/cross/kNN all differ |
| `real_lesmis_77` | `classic_drl_refine` | 0.0356 | crossings 3797 vs 4505 |

This is a tiny, real-network tail, not disconnected or multiedge. Three rows have very small e_rel
but still fail distributional equivalence. The likely residual is in DrL’s procedural C++ mechanics:
phase schedules/presets (`dagua/layout/ops/drl.py:185-230`), seeded coordinate handling
(`igraph drl_layout.cpp:470-477`), and density/grid boundary behavior. The dagua pipeline comment
itself says full-suite parity depends on C++ float rounding and density-grid boundary behavior
(`dagua/layout/ops/pipelines/drl.py:39-41`).

Best achievable tier: rung 2 for `real_karate_34` rows if trace parity identifies a small schedule
or RNG mismatch; `real_lesmis_77::coarsen` may remain rung 4 or rung 3 without deeper C++ trace
matching because e_rel is 0.229 and all quality metrics move.

Concrete next experiment/fix:

1. Add temporary trace-only benchmark instrumentation for first 5 iterations per phase on
   `real_karate_34` and `real_lesmis_77`.
2. Compare igraph `options` values from `igraph_layout_drl_options_init` with dagua presets;
   do not assume preset names are identical.
3. Compare candidate jump RNG sequence and density-grid cell assignment near boundaries.

Confidence: medium. Effort: 1-2 days because it requires C++ trace comparison.

### Neato: 2 Mode A divergent rows

Rows:

| Graph | e_rel | Signature |
|---|---:|---|
| `parallel_cycles_4x5` | 0.0824 | disconnected; kNN D/R 0.833 vs 0.964 |
| `random_dag_50` | 0.0784 | disconnected; stress D lower, crossings/kNN differ |

Both neato divergences are disconnected. Dagua splits weak components, solves each component with
`seed + component_index`, packs components, and only then converts to Graphviz-observable positions
(`neato.py:1368-1438`). Graphviz neato also splits and packs, but the exact reference path is
`getPackModeInfo`, default `pinfo.mode = l_node`, per-component `neatoLayout`, overlap removal,
splines, and `packGraphs` (`neatoinit.c:1371-1410`). This makes component packing and component RNG
seeding the most likely residual.

Best achievable tier: rung 2 for both if component seed/order/packing is ported exactly; 3Q is not
currently available (`q_battery=1.0` for both).

Concrete fix:

1. Remove or gate `seed + component_index` if Graphviz does not reseed per component that way.
2. Verify component order from `pccomps` vs dagua `_weak_components`.
3. Align the Graphviz `l_node` polyomino packing path, including spline-aware coverage, not only
   bounding-box packing.

Confidence: high on root bucket, medium on exact seed-vs-pack split. Effort: 1 day for trace and
component order; 2-3 days for full pack fidelity.

## Mission 3: genuine 3Q audit

Strict calculation on `per_combo.json`:

- Current `3Q` count: 36.
- Rung-3 count: 279.
- Rung-3 rows with `q_battery < 0.05`: 0.
- Rung-3 rows with `quality_identical_raw=True`: 0.
- Rung-3 rows with raw `battery_p_iut < 0.05`: 0.
- Rung-3/4 rows with raw `battery_p_iut < 0.05`: 0.

Therefore the legitimate rung-3 -> 3Q promotion count under the existing strict gate is:

**0**

Control rerun:

Command:

```bash
python scripts/definitive_fidelity_report.py --controls \
  --controls-dir eval_output/fidelity_definitive/controls \
  --output-dir /tmp/r74_controls_check
```

Important output:

- `gate_5_quality_identical_laundering`: passed.
- scored: 40.
- missing battery: 0.
- 3Q controls: 0.
- 3Q control pass rate: 0.0%.
- limit: 5.0%.

The overall controls payload has `all_passed=false`, but not because of 3Q laundering. The failing
gate is `gate_3_negative`, with `non_primary_percent=90.0`; the quality-identical anti-laundering
gate itself is clean at 0/40.

No “parity-looking” reclassification under the strict battery exists to test further: there are zero
rung-3/4 rows with `battery_p_iut < 0.05`. The independent control pass-rate for the actual strict
3Q gate is therefore 0/40 = 0.0%.

## ROI-ordered fix avenues

1. **UMAP multiedge distance/RNG parity**
   - Impact: up to 6 of 8 UMAP divergent rows, concentrated on one graph.
   - Best tier: rung 2 if distance and optimizer streams match; no 3Q relabel without battery.
   - Evidence: `parallel_multiedge_bundle` is 6/8 UMAP divergences; reference CSR distance path at
     `umap_competitor.py:76-85`; native duplicate policy at `umap.py:123-183`.
   - Confidence: high bucket, medium exact fix.

2. **Neato disconnected component order/seed/packing**
   - Impact: 2/2 neato divergent rows.
   - Best tier: rung 2.
   - Evidence: both rows disconnected; dagua component solve/pack at `neato.py:1368-1438`;
     Graphviz pack path at `neatoinit.c:1371-1410`.
   - Confidence: high bucket, medium exact split.

3. **DrL trace parity on real-network tail**
   - Impact: 5 rows.
   - Best tier: rung 2 for karate rows; uncertain for `real_lesmis_77::coarsen`.
   - Evidence: small connected real graphs only; dagua presets `drl.py:185-230`; igraph execution
     path `drl_layout.cpp:435-478`.
   - Confidence: medium.

4. **Broad disconnected/component audit**
   - Impact: cross-family but limited and non-uniform. It may help parts of `random_dag_50`,
     `random_dag_200`, `parallel_cycles_4x5`, and multi-component graphs, but the data rules out
     a single global fix.
   - Best tier: family-dependent.
   - Confidence: medium as a triage theme, low as a single fix.

## Assumptions and concerns

- I treated family names by substring normalization of `engine` names; this is sufficient for the
  requested aggregate sweep but not a replacement for report-owned family tables.
- I did not call direct layout pipelines, per guardrail. All quantitative conclusions are from the
  stored benchmark-path `per_combo.json` and the official controls command.
- The UMAP multiedge diagnosis needs actual distance-matrix dumps before code changes; current code
  already comments that duplicate sums are intentional, so the next step is to prove whether the
  benchmark reference actually sees the same matrix.

## Executive summary

1. CONFIRMED: r74 crossing equivalence has a calibrated-floor problem on exact-count small graphs: floor is 0.5 mean crossings, 115/235 crossing-failing rows have zero reference self-spread, and 75/235 are within 1 crossing on average.
2. CONFIRMED: none of the 235 crossing-failing r74 Phase-2 rows used sampled crossings; sampling noise is a future/large-graph criteria risk, not the source of those 235 failures.
3. CONFIRMED: `count_crossings` exact path uses a different intersection predicate from `segments_intersect`: exact path ignores collinear overlaps, while vector sampled path counts them. This is an eval consistency bug.
4. CONFIRMED: the prompt's "~165 huge divergent combos" was not reproducible from the checked files. Both the project helper and benchmark metadata found 9 unrescored `>300` divergent rows, all `random_dag_200` with 383 reported nodes and 300 edges.
5. CONFIRMED: better-on-all-failing-legs count is 93/337, matching sprint lead context. Current stress/crossing criteria are two-sided; NP is one-sided non-inferiority.
6. Recommendation: first fix crossing criteria and estimator consistency, then require gate_5 0/40 negative+chance and a positive reference-self-split control before reclassifying anything.

## Findings ranked by expected impact

### 1. CONFIRMED: crossing margin is too brittle for discrete exact counts

Evidence command:

```bash
python3 - <<'PY' > /tmp/r75_metrics_stats.txt
import json, statistics
rows=[json.loads(l) for l in open('eval_output/fidelity_definitive/r74_phase2_rescore.jsonl')]
target=[r for r in rows if not r.get('quality_identical_raw')]
cf=[r for r in target if not r.get('cross_direct_equivalent', True)]
vals=[]
for r in cf:
    d=abs(float(r.get('cross_D_mean',0))-float(r.get('cross_R_mean',0)))
    vals.append((d,float(r.get('cross_margin',0)),float(r.get('cross_ref_self_spread',0) or 0),r.get('cross_sampled')))
print('rows',len(rows),'targets_not_quality_identical',len(target),'cross_failing',len(cf))
for name, arr in [('abs_delta',[v[0] for v in vals]),('margin',[v[1] for v in vals]),('ref_self_spread',[v[2] for v in vals])]:
    arr=sorted(arr)
    print(name,'min',arr[0],'p10',arr[int(.1*(len(arr)-1))],'p25',arr[int(.25*(len(arr)-1))],'median',statistics.median(arr),'p75',arr[int(.75*(len(arr)-1))],'p90',arr[int(.9*(len(arr)-1))],'max',arr[-1])
for t in [0.5,1,2,3,5,10,20,50,100]:
    print('delta<=',t,sum(1 for v in vals if v[0]<=t))
for t in [0,0.5,1,2,5,10]:
    print('margin<=',t,sum(1 for v in vals if v[1]<=t))
print('ref_self_spread_zero',sum(1 for v in vals if v[2]==0),'sampled',sum(1 for v in vals if v[3]))
PY
cat /tmp/r75_metrics_stats.txt
```

Output:

```text
rows 409 targets_not_quality_identical 337 cross_failing 235
abs_delta min 0.016666666666666666 p10 0.1 p25 0.9166666666666661 median 4.0 p75 24.0 p90 76.26666666666665 max 6001.0
margin min 0.5 p10 0.5 p25 0.5 median 1.6899883449481559 p75 8.88 p90 60.6726095990238 max 1068.239051812927
ref_self_spread min 0.0 p10 0.0 p25 0.0 median 0.09999999999999999 p75 3.9389277113386476 p90 50.75006991088383 max 1068.239051812927
delta<= 0.5 48
delta<= 1 75
delta<= 2 89
delta<= 3 113
delta<= 5 134
delta<= 10 155
delta<= 20 173
delta<= 50 200
delta<= 100 215
margin<= 0 0
margin<= 0.5 94
margin<= 1 105
margin<= 2 125
margin<= 5 155
margin<= 10 189
ref_self_spread_zero 115 sampled 0
```

Code claims:

- Dagua exact/sampled crossing metric entry point is `dagua/metrics.py:2038`; exact path is selected at `E <= 500` on `dagua/metrics.py:2068`, sampled path at `dagua/metrics.py:2079`.
- The fidelity battery records `cross_sampled = edge_index.shape[1] > 500` at `scripts/definitive_fidelity_analysis.py:1362`.
- Crossing TOST uses paired diffs and `quality_cross_margin` at `scripts/definitive_fidelity_analysis.py:1208` and one-sample deterministic mode at `scripts/definitive_fidelity_analysis.py:1260`.
- The margin is `max(2% * mean(reference), 0.5, reference self-spread)` at `scripts/definitive_fidelity_analysis.py:1598`.
- Reference self-spread is sample standard deviation, returning `0.0` for fewer than two finite values at `dagua/eval/equivalence_metrics.py:94`.

Reference source context:

- OGDF's intersection statistics use a sweep-line intersection graph, with segment construction from graph attributes at `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/LayoutStatistics_intersect.cpp:400` and intersection scheduling at `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/LayoutStatistics_intersect.cpp:347`.
- Graphviz edgepaint's line segment intersection notes explicitly use inclusive `0 <= t <= 1` / `0 <= u <= 1`, and also treat "very close" lines as intersections at `/home/jtaylor/projects/_references/graphviz/lib/edgepaint/intersection.c:58` and `/home/jtaylor/projects/_references/graphviz/lib/edgepaint/intersection.c:161`.

Fix sketch:

- Replace absolute count margin for exact crossings with a discrete-aware margin, e.g. `max(2% * mean_ref, ref_self_spread, 1.0 / sqrt(n_pairs_or_battery_n), exact_count_floor)`, where the floor is pre-registered as either 1.0 crossing on the sample mean or a graph-size-normalized crossing-rate margin.
- Prefer crossing rate for equivalence (`count / eligible_edge_pairs`) and report raw counts as diagnostics. This avoids a 1-crossing delta meaning the same thing on a 20-edge and 500-edge graph.
- Decisive experiment: recompute only battery decisions from existing `cross_*` fields under candidate margins, then run controls. Runtime: seconds for field replay, minutes for controls.

Expected impact:

- Upper bound from current data: 75 rows are within 1 crossing on average; 89 within 2; 115 have zero reference self-spread. Actual reclassification impact is lower because stress and NP must also pass.

Risk:

- High laundering risk if a larger floor is blanket-applied. Must pass gate_5 exactly: 0/40 negative+chance controls in 3Q (`scripts/definitive_fidelity_report.py:2482`) and the persisted gate must block report rendering if absent/failed (`scripts/definitive_fidelity_report.py:3234`).

### 2. CONFIRMED: sampled crossing estimator is not margin-calibrated and has a denominator bug for large graphs

Evidence:

- All current Phase-2 crossing failures are exact (`sampled 0` in the command above), but the `E > 500` path is still unsafe for S2.
- `sampled_crossing_rate` excludes edge pairs sharing a node at `dagua/metrics.py:803`, computes rate over valid sampled pairs at `dagua/metrics.py:825`, but estimates total crossings as `rate * E * (E - 1) / 2` at `dagua/metrics.py:831`. That scales a valid-pair conditional rate by all unordered edge pairs, including ineligible adjacent-edge pairs.
- For `total_pairs > 10_000_000`, sampling falls back to random edge ids with replacement/dedup only through equality filtering at `dagua/metrics.py:794`; the comment admits with-replacement bias for huge E at `dagua/metrics.py:798`.
- Fidelity passes the same `cross_seed` to Dagua and reference layouts (`scripts/definitive_fidelity_analysis.py:1323`, `scripts/definitive_fidelity_analysis.py:1346`, `scripts/definitive_fidelity_analysis.py:1350`), so common random numbers reduce paired noise, but `cross_ref_self_spread` does not include sampling uncertainty explicitly (`scripts/definitive_fidelity_analysis.py:1399`, `scripts/definitive_fidelity_analysis.py:1423`).

Fix sketch:

- Return `(estimate, se, n_valid, eligible_pair_estimate_or_exact)` from the metric used by fidelity, not just an int.
- Use the same estimator for Dagua, reference, and reference-self-spread. Margin should be `max(reference_self_spread_of_estimator, z * sqrt(se_D^2 + se_R^2), pre-registered practical rate floor)`.
- For medium large graphs, sample uniformly from eligible non-adjacent edge pairs, or estimate the eligible denominator separately with the same random stream. Do not scale by all `E choose 2` after filtering adjacent pairs.

Expected impact:

- Direct impact on current 409-row Phase-2 set: none, because `cross_sampled=False` for all 235 crossing failures.
- Required for the bounded-time rescoring path below.

Risk:

- Changes large-graph counts and may move previously bit/quality rows. Gate with a synthetic graph where many edge pairs share endpoints, plus a complete exact-vs-sampled convergence test at `E` just above 500.

### 3. CONFIRMED: exact and sampled crossing predicates disagree on collinear/near-degenerate cases

Evidence:

- Vector predicate counts collinear overlap as a crossing by design at `dagua/metrics.py:146` and `dagua/metrics.py:182`, returning `proper | collinear_overlap` at `dagua/metrics.py:201`.
- Exact path calls `_segments_intersect_scalar` at `dagua/metrics.py:2075`; scalar predicate only tests strict orientation sign changes and excludes endpoint allclose at `dagua/metrics.py:2083`. It does not count collinear overlap.
- This creates a discontinuity at `E=500/501` and a possible scale/degeneracy artifact for layouts with near-collinear edges.
- Reference context: OGDF has explicit handling/comments for overlapping segments in the sweep structure at `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/LayoutStatistics_intersect.cpp:305` and `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/LayoutStatistics_intersect.cpp:573`; Graphviz edgepaint treats close parallel lines as intersections at `/home/jtaylor/projects/_references/graphviz/lib/edgepaint/intersection.c:151`.

Fix sketch:

- Make exact and sampled fidelity paths use the same predicate. Conservative choice: use the vector `segments_intersect` in exact batches, because it has documented degeneracy behavior.
- Add regression tests for `E=500` vs `E=501` on the same graph plus collinear-overlap examples.

Expected impact:

- Unknown without replay. Cheapest decisive experiment: recompute crossings for the 235 failing rows with vector exact batches and count changed decisions. Runtime likely under 30 minutes because rows are <=300 nodes and `E<=500`.

Risk:

- Medium. Counting collinear overlaps can penalize collapsed but otherwise reference-like layouts. Anti-laundering and positive controls are mandatory.

### 4. CONFIRMED: actual unrescored `>300` divergent worklist is 9 rows, not ~165 from the prompt

Evidence command:

```bash
python3 - <<'PY' > /tmp/r75_metrics_worklist.txt
import json, importlib.util, sys, os
os.environ['MPLCONFIGDIR']='/tmp/matplotlib-r75'
spec=importlib.util.spec_from_file_location('dfa','scripts/definitive_fidelity_analysis.py')
dfa=importlib.util.module_from_spec(spec); sys.modules['dfa']=dfa; spec.loader.exec_module(dfa)
graph_data=dfa.load_graph_data()
phase={json.loads(l)['combo_id'] for l in open('eval_output/fidelity_definitive/r74_phase2_rescore.jsonl')}
rows=[]
for l in open('eval_output/fidelity_definitive/r74_analysis.jsonl'):
    r=json.loads(l); cid=r['combo_id']; graph=r['graph']; engine=r['engine']
    if cid in phase: continue
    if r.get('quality_identical_raw') or r.get('quality_identical'): continue
    if r.get('dist_equivalent') or r.get('bit_exact'): continue
    n,e = graph_data.get(graph,(None,()))
    if n is not None and n>300:
        rows.append((cid, graph, engine, n, len(e)))
print('huge divergent not in phase2',len(rows))
for x in sorted(rows, key=lambda x:(x[3],x[1],x[2])):
    print('\t'.join(map(str,x)))
PY
cat /tmp/r75_metrics_worklist.txt
```

Output:

```text
huge divergent not in phase2 9
random_dag_200::classic_fmmm_graphviz_fdp_fidelity	random_dag_200	classic_fmmm_graphviz_fdp_fidelity	383	300
random_dag_200::classic_fmmm_steps100	random_dag_200	classic_fmmm_steps100	383	300
random_dag_200::classic_fmmm_steps200	random_dag_200	classic_fmmm_steps200	383	300
random_dag_200::classic_maxent_stress_steps50	random_dag_200	classic_maxent_stress_steps50	383	300
random_dag_200::classic_sugiyama_default	random_dag_200	classic_sugiyama_default	383	300
random_dag_200::classic_sugiyama_graphviz_fidelity	random_dag_200	classic_sugiyama_graphviz_fidelity	383	300
random_dag_200::classic_sugiyama_passes4	random_dag_200	classic_sugiyama_passes4	383	300
random_dag_200::classic_sugiyama_tight	random_dag_200	classic_sugiyama_tight	383	300
random_dag_200::classic_umap_nn30	random_dag_200	classic_umap_nn30	383	300
```

Cross-check command using benchmark metadata also returned `all not phase >300 9` and `not quality/dist/bit false among >300 9`.

Bounded-time scoring design:

- Stress: use landmark/pivot APSP with a fixed graph-seeded landmark set. Metric and margin must both use the same landmark estimator. Report a bootstrap CI over landmarks; if CI half-width exceeds the practical margin, return `APPROX_UNRESOLVED`, not divergent or identical.
- Crossings: use fixed-seed common random eligible edge-pair sampling. Compare crossing rates, not extrapolated counts. Margin is `max(reference estimator self-spread, paired sampling SE term, practical rate floor)`.
- NP: use approximate kNN with the same approximate backend/settings for Dagua and reference, and include estimator self-spread in the margin. For N under a cheap exact cutoff, exact `torch.cdist` remains preferable.
- Time budget: per combo cap, e.g. 60 seconds wall clock for all three metrics, with per-leg timeout. Fallback semantics: no 3Q if any required leg is `TIMEOUT` or `APPROX_UNRESOLVED`; row remains divergent/unscored with explicit reason.

### 5. HYPOTHESIS: engine-level population equivalence can certify stochastic-reference quality, but only as an aggregate claim

Design:

- Unit of analysis: per-combo per-seed paired metric samples already produced by `quality_metric_samples` (`scripts/definitive_fidelity_analysis.py:1297`).
- For each engine and graph class, aggregate paired deltas for stress, crossing rate, and NP. Use TOST on mean paired deltas for stress/crossing and one-sided non-inferiority for NP; add a KS or Wasserstein equivalence check to catch distribution-shape mismatches.
- Pre-register margins by graph class and metric estimator. Use BH correction across engine x graph-class x metric families via existing `bh_fdr` at `dagua/eval/distributional_fidelity.py:312`.
- Controls: reference-self-split positive controls must pass; chance and negative controls must remain 0/40 for 3Q-equivalent aggregate claims.

Licensed headline:

- "Engine X is population-quality-equivalent to reference Y on graph class Z under the registered quality estimators."

Not licensed:

- It does not certify every `(engine, graph)` combo as bit-identical, distributionally identical, or 3Q identical.
- It must not count toward the north-star identical headline for individual combos.

Cheapest decisive experiment:

- Replay existing r74 per-seed metric arrays where available; otherwise rerun analysis with metric arrays persisted. Runtime: minutes for replay, under one benchmark-analysis pass if arrays are not persisted.

### 6. CONFIRMED: Dagua-better rows need a separate non-identical tier, not 3Q

Evidence command output:

```text
target 337 better_all_failing_legs 93 mixed_has_good 89
better by engine [('classic_sfdp_p_neg2', 22), ('classic_sugiyama_passes4', 8), ('classic_sfdp_theta08', 7), ...]
```

Code claims:

- Stress and crossings are symmetric two-sided TOSTs: `df.paired_tost(metrics["stress_d"] - metrics["stress_r"], ...)` at `scripts/definitive_fidelity_analysis.py:1201` and crossings at `scripts/definitive_fidelity_analysis.py:1208`.
- NP is one-sided non-inferiority: `quality_np_noninferiority` documents `mean(np_d) >= mean(np_r) - margin` at `scripts/definitive_fidelity_analysis.py:1642` and returns direct success if so at `scripts/definitive_fidelity_analysis.py:1662`.
- Rung assignment currently has only bit/dist/quality/divergent routes and no "superior but distinct" tier (`dagua/eval/distributional_fidelity.py:343`).

Policy recommendation:

- Keep root-cause-first: "better than reference" is evidence of comparison mismatch until proven otherwise.
- If proven benign, create a non-identical tier such as `QUALITY_SUPERIOR_DISTINCT`. It can be useful product evidence but must not increment bit-identical, distributionally-equivalent, quality-identical, or north-star identical counts.
- Require both: (1) parameter/extraction audit proving the reference comparison is fair, and (2) controls showing the superiority rule does not launder chance/negative rows.

Risk:

- High reputational risk if "better" is presented as identical. Better-on-good-side rows are exactly where parameter mismatch, reference post-processing, or metric defects tend to hide.

## Per-root-cause controls and tests

- Crossing floor/rate change: replay r74 rows; add tests for zero-spread small integer counts; require gate_5 0/40 negative+chance and reference-self-split positive pass.
- Sampled crossing estimator: test denominator on a star-like graph with many adjacent edge pairs; exact-vs-sampled convergence on a graph with known crossings; common-random-number reproducibility by seed.
- Predicate consistency: tests for collinear overlap, endpoint touch, near-parallel non-overlap, and the same graph evaluated just below/above `E=500`.
- Huge-graph approximate path: tests where exact <=500 result and approximate >500 estimator agree within registered CI; timeout test that returns unresolved instead of pass/fail.
- Population equivalence: synthetic positive self-split and negative/chance aggregate controls with BH correction.
- Dagua-better tier: fixture where Dagua is strictly better on stress/cross/NP but position distribution differs; assert not 3Q/identical.

## Target combos not explained

- I did not root-cause algorithm-specific divergences. This bucket was criteria-only.
- The S2 prompt says approximately 165 huge divergent combos; I could not reproduce that from `r74_analysis.jsonl` minus `r74_phase2_rescore.jsonl`. The authoritative project helper and an independent benchmark-metadata cross-check both returned exactly 9 `>300` divergent rows, all on `random_dag_200`.

## Knowledge

- `r74_phase2_rescore.jsonl` has 409 rows; 337 remain non-3Q after filtering `quality_identical_raw`.
- Crossing failures dominate: 235/337 current targets fail crossings, but all are exact-count rows.
- The crossing metric has two separate predicates: scalar exact and vector sampled. They are not equivalent on collinear overlap.
- The hard 3Q anti-laundering gate is implemented in report code and should be treated as mandatory for any criteria relaxation.

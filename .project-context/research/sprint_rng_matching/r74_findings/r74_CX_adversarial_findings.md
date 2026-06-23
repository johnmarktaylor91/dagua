# r74 CX adversarial findings

Read-only review of `.project-context/research/sprint_rng_matching/r74_PLAN.md`,
`/tmp/r74_O{1..6}_findings.md`, `/tmp/r74_CX{1..6}_findings.md`, reference sources,
and `eval_output/fidelity_definitive_r73/per_combo.json`.

## 1. A1 p_neg2

Verdict: the Graphviz clamp is real, but the plan's counting/gate needs tightening.

Source evidence:
- `/home/jtaylor/projects/_references/graphviz/lib/sfdpgen/sfdpinit.c:211-212` parses
  `K` and `repulsiveforce`; `repulsiveforce` is `ctrl->p = -1.0 * late_double(..., -AUTOP, 0.0)`.
- `/home/jtaylor/projects/_references/graphviz/lib/common/utils.c:55-68` implements
  `late_double`; line 66 returns `minimum` if `rv < minimum`. Therefore graph attr
  `repulsiveforce=-2.0` is clamped to `0.0` before the negation.
- `/home/jtaylor/projects/_references/graphviz/lib/sfdpgen/spring_electrical.c:284-289`
  then forces any `p >= 0` to `p = -1` and computes `KP = pow(K, 1 - p)`.
- `/home/jtaylor/projects/_references/graphviz/lib/sparse/QuadTree.c:158-162` and
  `:190-194` special-case `p == -1` as `dist*dist`; non-`-1` uses `pow(dist, 1-p)`.
- Dagua does not clamp: `dagua/eval/variants.py:1616-1621` maps
  `repulsive_exponent=-2.0` to Graphviz `repulsiveforce=-2.0`, while
  `dagua/layout/ops/pipelines/sfdp.py:406-426` uses `repulsive_exponent` directly in
  force scales and `:529-540` uses `pow(distance, 1.0 - self.repulsive_exponent)`.

Data evidence:
- Raw benchmark-path 5-seed reference artifacts under
  `eval_output/benchmark_5seed_fidelity/positions` are byte/tensor identical for all
  105 available `graphviz_sfdp__for__classic_sfdp_default.pt` vs
  `graphviz_sfdp__for__classic_sfdp_p_neg2.pt` graph pairs. This supports the clamp.
- In `per_combo.json`, aggregate reference summary fields are not all identical:
  only 23/101 `classic_sfdp_p_neg2` rows exactly match `classic_sfdp_default` on
  `stress_R_mean`, `cross_R_mean`, `np_R_mean`, `mean_W_R`, `plain_mean_W_R`.
  This is likely overlay/provenance/seed aggregation noise, not semantic evidence
  against the clamp. The plan's "byte-identical reference rows" gate should be
  implemented against raw coordinate artifacts for the same benchmark run, not
  against per-combo summary aggregates.

Honest count from `per_combo.json`:
- `classic_sfdp_p_neg2`: 101 rows; 73 rung 4, 5 insufficient.
- If `p_neg2` is normalized to default semantics, 53/73 divergent rows inherit a
  currently rung1-3 default sibling. Of those, 52 are connected and 1 is disconnected.
- The 20 non-flips are 12 connected rows whose default sibling is already divergent
  and 8 disconnected rows whose default sibling is also divergent.
- Therefore the honest A1 force-law-only flip estimate is 52 connected rows, not
  a blanket 73 and not a separate 52 plus disconnected wins. If the review excludes
  "also disconnected" and "default/floor sibling still divergent", A1 = 52.

Concrete examples:
- `ba_500::classic_sfdp_p_neg2`: rung 4, connected; default sibling rung 2.
- `citation_dag_300::classic_sfdp_p_neg2`: rung 4, connected; default sibling rung 3.
- `random_dag_200::classic_sfdp_p_neg2`: rung 4, disconnected; default sibling rung 4.

## 2. A1/B1 double-counting

Counts from `per_combo.json`:
- A1 naive predicted flips: 53 p_neg2 rows whose default sibling is rung1-3.
- A1 honest force-law-only flips: 52 connected rows.
- A1 p_neg2 disconnected divergent rows: 9.
- B1 non-p_neg2 disconnected sfdp divergent rows: 48:
  `classic_sfdp_default` 9, `classic_sfdp_graphviz_fidelity` 9,
  `classic_sfdp_steps200` 9, `classic_sfdp_theta04` 9, `classic_sfdp_theta08` 12.
- B1 disconnected graph set:
  `dependency_500`, `dependency_graph_100`, `disconnected_encoder_residual`,
  `disconnected_label_cycle_collage`, `er_100`, `er_500`,
  `kitchen_sink_platform_graph`, `multi_component_80`, `parallel_cycles_4x5`,
  `random_bipartite_60`, `random_dag_200`, `random_dag_50`.
- A1 p_neg2 disconnected graph set:
  `dependency_graph_100`, `disconnected_encoder_residual`,
  `disconnected_label_cycle_collage`, `er_100`, `kitchen_sink_platform_graph`,
  `multi_component_80`, `parallel_cycles_4x5`, `random_bipartite_60`,
  `random_dag_200`.

Net unique sfdp estimate:
- A1 honest 52 + B1 48 = 100 candidate row wins.
- If p_neg2 disconnected rows are also made packable after B1, add up to 9 later,
  but they are not A1 force-law wins. Do not sell A1+B1 as 52+48+9 or as 73+48.
- Realistic net after rebench should be lower than 100 because B1 has mixed
  quality direction and connected default residuals remain. Use 80-95 until
  benchmark-path evidence proves all 48 B1 rows flip.

## 3. Regression risk and required guards

### B1 sfdp disconnected packing

Yes, it can regress. Graphviz source confirms component handling:
`/home/jtaylor/projects/_references/graphviz/lib/sfdpgen/sfdpinit.c:268-288` calls
`ccomps`; for multiple components it runs `sfdpLayout` on each subgraph and then
`packSubgraphs`.

Risk from data:
- All 48 B1 rows have positive `e_rel`, but 18/48 have Dagua lower raw stress than
  reference (`stress_D_mean < stress_R_mean`). Examples:
  `disconnected_label_cycle_collage::classic_sfdp_default` has
  `stress_D_mean=0.00014101184277223554` vs `stress_R_mean=0.0889834858507234`;
  `kitchen_sink_platform_graph::classic_sfdp_default` has
  `0.020903390661697427` vs `0.030517640178993784`;
  `random_bipartite_60::classic_sfdp_default` has
  `0.0930033631540412` vs `0.10029922039922122`.
- Since the goal is reference fidelity, becoming more graphviz-like can intentionally
  worsen Dagua's quality metrics. That is acceptable only if the rung improves and
  no already-good combos regress.

Required guard:
- Implement only behind Graphviz-fidelity SFDP path and only when `num_components > 1`.
- Rebench every `classic_sfdp*` variant on the benchmark path, not direct calls.
- Hard fail the change if any current rung1-3 sfdp row becomes rung4/insufficient, or
  if any currently D-better disconnected row becomes worse without a rung improvement.
- Track per-component RNG reset/order and pack mode (`l_node`, `CL_OFFSET`) against
  Graphviz source; otherwise this becomes a new divergent layout, not a fidelity fix.

### B2 classical_mds disconnected packing

Risk is lower but not zero.

Data:
- `classic_classical_mds_default` and `classic_classical_mds_igraph_fidelity` have
  20 disconnected rung4 rows, 10 per engine.
- All 20 have Dagua stress worse than reference. Examples:
  `disconnected_encoder_residual::classic_classical_mds_default` has
  `stress_D_mean=0.4212898431204042` vs `stress_R_mean=0.011342511471706453`;
  `multi_component_80::classic_classical_mds_default` has
  `0.5228020873337964` vs `0.03919609955825177`.
- But there are already 4 disconnected MDS rows at rung 3:
  `dependency_500::{default,igraph_fidelity}` and
  `dependency_graph_100::{default,igraph_fidelity}`. These are at risk if a
  component-packing branch is applied too broadly.

Required guard:
- Gate on disconnected graphs only, but preserve the current connected path byte-for-byte.
- For disconnected rows, compare before/after on all MDS variants; hard fail any current
  rung3 disconnected row regressing to rung4.
- Do not promise rung1/2 unless the igraph DLA merge and reference seeding are ported.
  Deterministic TileToRows is a rung3 approximation, not a bit-exact igraph merge.

### C1 sugiyama LP objective/gating

Yes, this can regress a lot.

Source evidence:
- Dagua currently uses a zero LP objective in `_igraph_glpk_layer_assignments`:
  `dagua/layout/ops/sugiyama.py:378-384`.
- The Dagua docstring says the igraph objective is effectively zero:
  `dagua/layout/ops/sugiyama.py:344-349`; this is false for igraph 1.0.0.
- igraph gates GLPK to directed graphs with `no_of_nodes <= 1000`:
  `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:563-565`.
- igraph objective coefficients are `outdegs - indegs`:
  `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:608-615`.
- igraph falls back to Eades/undirected feedback for other cases:
  `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:661-670`.

Data risk:
- 151 Sugiyama rows are already rung 3 and 32 are `2'`; these can regress.
- 231 Sugiyama rows are rung4, including 166 igraph-reference and 65 graphviz-reference.
- 143/231 divergent Sugiyama rows have Dagua lower raw stress than reference; 123/231
  have lower crossings than reference. Examples:
  `ba_500::classic_sugiyama_default` has `stress_D_mean=0.3397065420245803` vs
  `stress_R_mean=0.4747330738426796`, crossings `117932.0` vs `140276.0`;
  `citation_dag_300::classic_sugiyama_default` has `0.3433019166160319` vs
  `0.4850303735899797`, crossings `78174.0` vs `89829.0`.

Required guard:
- Apply the igraph LP objective/gating only on igraph-reference Sugiyama variants.
  Do not touch `classic_sugiyama_graphviz_fidelity`, which targets Graphviz dot.
- Rebench all Sugiyama variants because code paths share `ops/sugiyama.py`.
- Hard fail if any current rung3 or `2'` row regresses to rung4/insufficient.
- Report quality-worse-but-reference-closer explicitly; do not market this as a
  quality improvement. This is a fidelity change.

## 4. D1 sgd2_multi_with_crossing

Verdict: a real reference exists, but it is not completable under the current benchmark
cap. This is not structural-by-design, and a blind rerun will likely reproduce
`no_reference_rows`.

Data:
- `per_combo.json` has 81 `classic_sgd2_multi_with_crossing` rows, all
  `INSUFFICIENT_DATA` with `insufficient_reason="no_reference_rows"`.
- Siblings are strong controls:
  `classic_sgd2_multi_default` 81/81 rung1,
  `stress_only` 81/81 rung1,
  `lr001` 81/81 rung1,
  `batch8` 85/85 rung1,
  `with_aspect` 85/85 rung1, etc.
- The merged store `/tmp/r74_cx5_joined_merged.json` shows the crossing reference
  exists as `sgd2_multi_ref__for__classic_sgd2_multi_with_crossing`.
  Across 81 rows it has status totals: `error=5208`, `skipped=2854`, `ok=38`;
  Dagua totals: `error=6164`, `skipped=1928`, `ok=8`. Errors are timeouts.
- Some reference rows did finish individual seeds, e.g.
  `hierarchical_residual_stage` reference `ok=10`, max runtime about 41.85s;
  `kitchen_sink_hybrid_net` reference `ok=9`, max runtime about 44.03s;
  `ragged_feature_pyramid` reference `ok=4`. None reaches 30 matched seeds.

Conclusion:
- Do not relabel STRUCTURAL_NA.
- Do not spend a standard rerun expecting 81 flips. First either optimize the crossing
  criterion path or intentionally raise per-layout/watchdog caps enough to collect at
  least 30 matched seeds per combo. A rerun with the same caps will reproduce "no
  reference rows".

## 5. Ordering/collision

Shared files and serialization:
- A1 and B1 both touch `dagua/layout/ops/pipelines/sfdp.py` and must serialize.
- B2 likely touches `dagua/layout/ops/embed.py`,
  `dagua/layout/ops/pipelines/classical_mds.py`, and possibly shared component pack
  helpers from `dagua/layout/ops/gem.py`. It can proceed separate from SFDP only if
  helper movement is avoided.
- A4 maxent_stress and B2 both want component splitting/packing helpers, so serialize
  if extracting or changing shared pack code.
- C1 and A3 both touch `dagua/layout/ops/sugiyama.py`; serialize.
- D1 touches `dagua/layout/ops/sgd2_multi.py`,
  `dagua/layout/ops/pipelines/sgd2_multi.py`, and possibly
  `dagua/eval/competitors/sgd2_multi_competitor.py`; keep isolated from layout-core
  changes but benchmark scheduling must wait for perf/cap decision.
- UMAP nn30 touches `dagua/layout/ops/umap.py` and
  `dagua/eval/competitors/umap_competitor.py`; independent.

Safe order:
1. A1 p_neg2 clamp in SFDP, then benchmark all SFDP variants. This reduces noise before
   evaluating disconnected packing.
2. B1 SFDP disconnected packing, then benchmark all SFDP variants and compare against
   the A1 result, not r73 directly.
3. B2 classical_mds disconnected branch and A4 maxent_stress disconnected branch, but
   serialize any shared helper changes; benchmark each family separately.
4. A2 UMAP nn30 reference/native clamp or rerun after current clamp; independent.
5. A3 Sugiyama recursion fix, then C1 Sugiyama LP objective/gating in a separate change.
   Benchmark all Sugiyama variants after each.
6. D1 SGD2 crossing only after deciding "perf fix" vs "longer timeout"; do not mix with
   semantic layout changes.

## 6. Kill list

KEEP:
- A1 SFDP p_neg2 clamp: source-proven, raw reference artifacts identical; honest 52
  force-law flips.
- A2 UMAP nn30 rerun/clamp: small and reference-side incompletion is known.
- A3 Sugiyama recursion: one explicit crash path; keep narrow.
- A4 maxent_stress disconnected: small but source-plausible; verify OGDF runner flags.
- B2 classical_mds disconnected: keep but revise expected tier to rung3 unless DLA is ported.

REVISE:
- B1 SFDP disconnected: real root cause, but mixed D-better rows mean "48 flips" is not
  guaranteed. Needs strict no-regression guards and per-combo verification.
- C1 Sugiyama LP objective: source-real but high regression risk; apply only to igraph
  variants and benchmark all Sugiyama. Do not touch graphviz_fidelity.
- D1 SGD2 crossing: not structural, but not a blind rerun. Requires performance work or
  much larger caps before rebench.
- D2 compute-frontier relabel: keep only for true large-graph timeouts; do not hide
  small-graph perf bugs or overlay/seed-key mismatches.
- D3 small-graph perf vectorize: useful but too broad for a single fidelity sprint patch;
  split per engine.
- C3 SFDP gv_random: revise to experiment only after A1/B1; do not sell count without
  single-iteration benchmark-path evidence.

KILL / DEFER:
- C2 FMMM component rotation as broad win: cap to disconnected subset only; connected FMMM
  remains unproven floor/systematic residual.
- Any "floor" relabel without FP-chaos evidence: violates the floor guardrail.
- Any 3Q promotion work: current strict controls are 0/40; do not launder.
- Blind SGD2 crossing rerun under current caps: likely reproduces no-reference rows.

## Guardrail audit

- NEVER LAUNDER: plan mostly honors this; the danger is language around "floor" and
  "quality-identical" without FP-chaos/3Q evidence.
- NO RUNTIME DELEGATION: no proposed fix should call references at runtime; ensure B2 does
  not call igraph DLA and D1 does not reintroduce `s_gd2` calls in native pipeline.
- VERIFY ON BENCHMARK PATH: critical for all counts; direct pipeline calls are insufficient.
- MATCH params+seed: especially SFDP component RNG reset/order and SGD2 crossing caps.
- FLOOR evidence: SFDP/FMMM/GEM/classical-MDS floor claims still need actual perturbation
  or trace evidence before relabeling.

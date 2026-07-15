# r74 PLAN -- Opus Adversarial Critique (read-only; benchmark-data + source verified)

Data source: `eval_output/fidelity_definitive_r73/per_combo.json` (3,955 rows; `final_rung` is a
STRING -- e.g. `"4"` -- not an int; analysts who compared `==4` will silently get 0). Verified rung
totals: 1=2227, 2=531, 2'=51, 3=279, 3Q=36, 4=574, INSUFFICIENT_DATA=257. Matches CX6.

## 1. A1 p_neg2 -- premise REAL, count INFLATED, regression risk UNNAMED

**Source confirms the clamp (KEEP the premise):**
- `graphviz/lib/sfdpgen/sfdpinit.c:212`: `ctrl->p = -1.0*late_double(g, ...repulsiveforce..., -AUTOP, 0.0)`.
  The `0.0` is `late_double`'s MIN bound; `lib/common/utils.c` returns `minimum` when `rv < minimum`.
  So `repulsiveforce=-2` -> clamped to `0.0` -> negated to `-0.0`.
- `spring_electrical.c:287/437/574`: `if (p >= 0) ctrl->p = p = -1;`. Force denominator `pow(dist, 1.-p)`
  = `pow(dist, 2)` for p=-1. dagua runs `repulsive_exponent=-2` unclamped -> `pow(dist, 3)`
  (`pipelines/sfdp.py:529-541`). The bug is real and one-line. **Premise = TRUE, verified in source.**

**Reference-row identity CONFIRMED in the definitive data:** p_neg2 vs default `stress_R_mean` is
byte-identical on **89/96** shared graphs. The 7 that differ are all N>=200 (`ba_500`, `er_500`,
`powerlaw_500`, `random_dag_200`, `random_dag_50`, plus rgg_500 within 1e-9) -- large-graph
Barnes-Hut/SHA-contamination, NOT the clamp (CX2 sec.1 reaches the same conclusion via raw `.pt`).

**HONEST flip count -- NOT 52, NOT 73, ~53 with 20 residual:**
- p_neg2 divergent graphs: 73. default divergent graphs: 21. p_neg2-ONLY divergent: 53.
- Of those 53: 1 disconnected (B1 overlap), 0 degenerate -> "52 connected non-degenerate" (plan's 52).
- BUT CX2's stronger projection: the default-rung of the 73 p_neg2-divergent graphs is 24x rung1,
  23x rung2, 6x rung3, **20 still rung4**. So clamping p_neg2->default collapses ~53 and **20 stay
  divergent inheriting default residuals.** Realistic A1 win = **~53 combos**, not 73.
- REGRESSION TRAP the plan never states: among the 52 honest candidates, **25 have dagua WORSE (D>R)
  and 27 have dagua BETTER (D<R).** The clamp makes dagua's repulsion WEAKER (p3->p2). On the 27 D<R
  combos dagua currently beats graphviz; matching graphviz's weaker law moves dagua TOWARD R. That is
  fine IF it lands inside equivalence, but it is NOT guaranteed and the plan asserts a flip without
  per-combo direction check. GUARD: re-bench all p_neg2; assert no rung1-3 default-sibling combo
  regresses and flag any D<R combo that fails to reach >=rung3.

## 2. DOUBLE-COUNTING A1 (p_neg2) x B1 (sfdp disconnected)

- A1 p_neg2 divergent combos: 73; of which disconnected (also B1 territory): **9**. So A1's pure-A1
  contribution is the 64 connected p_neg2 combos (of which ~52 flip).
- B1 target = ALL sfdp disconnected divergent across variants = **57 combos** (NOT 48; "48" is the
  non-p_neg2 subset -- CX2 line 13 confirms: 73+48+63=184 by primary-fix, 57 raw disconnected).
- **Net unique sfdp divergent = 184** (127 connected + 57 disconnected). A1+B1 are largely disjoint:
  overlap is the 9 p_neg2-AND-disconnected combos. Summing "A1 52 + B1 48" = 100 is the misleading
  number; the honest *combined* sfdp recovery ceiling is bounded by 184, realistic ~52(A1)+~25-34(B1
  after direction filter) ~= 80-85, NOT 100.

## 3. REGRESSION RISK by fix (direction split from data; D<R = dagua already better)

- **B1 sfdp disconnected** (57 divergent): **34 D>R (safe to fix), 23 D<R (REGRESSION RISK).**
  Packing graphviz-style forces dagua toward R on the 23 it already beats. GUARD: per-combo, abort if
  a D<R combo's e_rel worsens or its rung drops; keep current layout where dagua already <=R.
- **B2 classical_mds disconnected** (20 divergent): **20/20 D>R.** Clean direction, NO regression risk
  on the divergent set. The only risk is the ~current rung1-3 mds combos -- guard those don't regress.
- **A4 maxent disconnected** (3 divergent): plan calls this "deterministic -> rung 1." DATA REFUTES the
  framing: 2 of 3 are dagua-BETTER -- `maxent_stress_default` D=0.357<R=0.400, `steps400` D=0.307<R=0.366;
  only `steps50` is D>R. Forcing OGDF ComponentSplitter packing pushes the 2 D<R toward R. NOT a free
  rung-1 win. GUARD: per-combo; expect maybe 1-2 flips, not 3.
- **C1 sugiyama LP objective** (231 divergent, 183 currently rung1-3): the change adds a real objective
  + gates LP to directed graphs <=1000 nodes (else Eades). dagua currently runs HiGHS unconditionally
  with ZERO objective. RISK: any of the 183 currently-matching combos whose match depends on the
  current zero-objective x-ordering can regress 1/2/3 -> 4. GUARD: re-bench ALL sugiyama variants (not
  just graphviz_fidelity); hard-assert zero rung1-3 -> rung4 regressions; the LP gate (directed,
  <=1000) must exactly mirror igraph or it silently changes which graphs use Eades.
- **C2 fmmm rotation** (85 divergent: 67 connected, 18 disconnected): CX3 is RIGHT -- rotation only
  touches the 18 disconnected; the 67 connected are an unproven force residual. Of the 18, 8 are D<R.
  Cap expectation at ~10, not 12-15.

## 4. SGD2 (D1) -- FUTILE RERUN. Relabel, do not burn compute.

Data is unambiguous: all 81 `classic_sgd2_multi_with_crossing` combos -- final_rung
INSUFFICIENT_DATA, insufficient_reason `no_reference_rows`, **`n_ref_seeded_ok=0` on ALL 81**, and
**`n_reimpl_ok` max=3 (sum=8 across 81)**. CX5's per-combo table shows BOTH D and R are `error:100`
or `skipped:97` -- the neural crossing detector (`/tmp/graph-drawing/gd2.py:253-288`, trains a net
per step) times out on BOTH sides at the 120s cap (`variants.py:2147-2152`). A reference EXISTS by
design (siblings batch8/default/with_aspect are 81/81 rung1) but is **NOT completable at current
cost.** Plan's "re-bench to completion -> likely rung 1, biggest count win" would re-time-out 81
combos and waste >=2,430 seeds for ~0 recovery. VERDICT: **relabel COMPUTE_FRONTIER_NA (or
STRUCTURAL_NA);** only rerun AFTER a multi-day crossing-path perf fix lands -- out of sprint scope.

## 5. FILE COLLISIONS / SERIALIZE ORDER

Shared files force serialization:
- **A1, B1, C3 ALL edit `dagua/layout/ops/sfdp.py` + `pipelines/sfdp.py`** (and A1 also `eval/variants.py`).
  These three CANNOT run in parallel worktrees against each other -- serialize within one sfdp lane.
- **A2 umap, A3 sugiyama-recursion, A4 maxent, B2 mds, B3 fmmm-perf, C1 sugiyama-LP, C2 fmmm-rotation**
  each touch distinct engine files -> safe to parallelize ACROSS engines, serialize WITHIN engine
  (A3 sugiyama-recursion vs C1 sugiyama-LP both edit `ops/sugiyama.py`; B3 fmmm-perf vs C2
  fmmm-rotation both edit `ops/fmmm.py`/`pipelines/fmmm.py`).

Safe implement+verify order:
1. **A1 sfdp clamp** (isolated semantic, re-bench sfdp, capture direction guard) ->
2. **B1 sfdp disconnected packing** ON TOP of A1 (same files, sequential), re-bench sfdp again ->
3. **C3 gv_random** only if B1 leaves connected floor unproven (same files, last in sfdp lane).
   In PARALLEL with the sfdp lane (different files): A2 umap, A4 maxent, B2 mds, A3 sugiyama-recursion.
4. **B3 fmmm-perf**, then **C2 fmmm-rotation** (same fmmm files, sequential).
5. **C1 sugiyama-LP** LAST, after A3 sugiyama-recursion, isolated re-bench, heaviest regression guard.
6. **D2 relabel** + **D1 sgd2 relabel** are metadata-only, no code, do anytime.

## 6. KILL LIST

| Fix | Verdict | One-line reason |
|---|---|---|
| A1 sfdp p_neg2 clamp | **KEEP** (high-conf) | Source-verified clamp; 89/96 refs identical; ~53 flips (not 73), guard the 27 D<R |
| A2 umap_nn30 clamp | **KEEP** | D already clamps; refs are BrokenProcessPool; rerun refs recovers ~9-10 INSUFFICIENT cheaply |
| A3 sugiyama recursion | **KEEP** | 1 crash combo, isolated, pure correctness; serialize before C1 |
| A4 maxent disconnected | **REVISE** | 2/3 are dagua-BETTER; not free rung-1; per-combo guard, expect 1-2 not 3 |
| B1 sfdp disconnected pack | **REVISE** | 57 not 48; 23/57 are D<R regression-risk; per-combo direction guard mandatory |
| B2 classical_mds pack | **KEEP** (clean dir) | 20/20 D>R; safe direction; only guard non-divergent mds rows |
| B3 fmmm fdp vectorize | **KEEP** | Pure perf; recovers INSUFFICIENT timeouts; low semantic risk |
| C1 sugiyama LP objective | **REVISE/GATE** | Biggest lever but 183 matching combos at risk; LP gate must mirror igraph exactly; full-variant re-bench |
| C2 fmmm rotation | **REVISE** | Cap at disconnected subset (~18, 8 of them D<R) per CX3; connected 67 unproven |
| C3 sfdp gv_random | **KEEP-as-experiment** | Real source mismatch (modulo vs rejection); test, don't assume; gate before floor claim |
| D1 sgd2 rerun | **KILL (rerun) / KEEP (relabel)** | Both sides time out; rerun futile; relabel COMPUTE_FRONTIER_NA |
| D2 relabel COMPUTE_FRONTIER | **KEEP** | Legit large-graph timeouts; metadata only |
| D3 small-graph vectorize | **KEEP** | Real scalar-loop perf bug (davidson 0/100 on 42 nodes); bounded |
| PROVE-FLOOR (gem/mds/fmmm/sfdp) | **KEEP w/ EVIDENCE** | gem 269/309 already bit-exact; floor claim needs the 1-ULP perturbation RUN, not assertion |
| 3Q NO-PROMOTIONS | **KEEP** | 0/279 rung-3 have battery_p_iut<0.05; controls 0/40; line held (CX6 sec.3) |

## Honest NET-combo estimate (realistic, NOT summed)
- sfdp: A1 ~52 + B1 ~25-34 (after dropping 23 D<R) = ~77-86, capped by 184 total.
- mds B2 ~20, maxent A4 ~1-2, umap A2 ~9-10, fmmm C2 ~8-10 + B3 perf-recoveries, sugiyama C1 25-60
  (HIGH variance, regression-gated), D3 ~50-80 INSUFFICIENT recoveries.
- **Realistic divergent reduction this sprint: ~120-170 (574 -> ~400-450), NOT the ~250+ a naive sum
  implies.** sgd2's 81 are NOT in the divergent bucket (they're INSUFFICIENT) and are a relabel, not a
  recovery. The summed-headline overcounts by double-counting sfdp and over-crediting A4/C2.

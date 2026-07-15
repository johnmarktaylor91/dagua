# r75 metrics/criteria bucket -- sonnet findings

## 1. Executive summary

- S1 CONFIRMED: crossings fails 163/337 divergent combos via the TOST-gated `metric_equivalent`
  path (235/337 show *any* nonzero delta; 163 actually gate the IUT). Root cause is NOT primarily
  discreteness-collapses-margin-to-floor as hypothesized -- 59.5% of cross-failing rows DO have
  `cross_ref_self_spread == 0` (deterministic references), but inspection shows these are mostly
  genuine, small, systematic crossing-count differences between two near-deterministic algorithms
  (e.g. sugiyama variants: dagua consistently draws 1 more crossing than igraph on the same
  graph), not noise. The floor (`QUALITY_CROSS_ABS_FLOOR = 0.5`) is doing its job correctly on
  these; the miscalibration is elsewhere (S1b).
- S1 CONFIRMED (separate, real bug): `sampled_crossing_rate()` in `dagua/metrics.py:826` computes
  a proper sampling standard error (`crossing_se`) but `crossing_count()` at
  `scripts/definitive_fidelity_analysis.py:1726-1745` discards it, keeping only the point
  estimate. The crossings margin never incorporates sampling noise. Currently this affects **zero**
  combos in the current 163-row cross-failing set (`cross_sampled=False` for all 163 -- the E>500
  sampled path never fires in Phase-2's <=300-node scope), but it is a live landmine for S2's
  huge-graph worklist, where E>500 is common.
- S2 CONFIRMED, brief's premise wrong: `r74_analysis.jsonl` (the file the brief names as the
  source for the >300-node worklist) contains scored rows for only 1 of the 16 graphs with
  >300 nodes (`random_dag_200`, 383 nodes) -- the other 15 (`ba_500/2000/5000`, `er_500/2000`,
  `grid_20x20/50x50`, `powerlaw_500/2000`, `rgg_500/2000`, `small_world_500/2000`,
  `dependency_500`, `sbm_8x100`) have **no rows at all**, not "dropped for size." The r74 round
  simply never re-ran the big-graph tier. Using the most recent full-coverage snapshot instead
  (`per_combo_r73.jsonl`), the honest >300-node divergent worklist is **238 combos across all 16
  big graphs** (not ~165), enumerated in section 4.
- S4 CONFIRMED: 93/337 divergent combos are dagua-better on every failing leg (my independent
  recount matches the brief's number exactly using "all failing legs strictly better" as the
  definition). `classic_sfdp_p_neg2` alone accounts for 22-34 of these (depending on strictness),
  overwhelmingly via the stress leg, and the reference-side param (`repulsiveforce: -2.0`) is
  correctly matched in `dagua/eval/variants.py:1615-1622` -- so this is NOT the classic
  param-mismatch bug pattern; needs a stress-bucket investigation, out of this bucket's scope but
  flagged here since S4 asked for root-cause-first triage.
- S4 CONFIRMED: `np`'s one-sided non-inferiority test (`quality_np_noninferiority`,
  `scripts/definitive_fidelity_analysis.py:1642-1669`) structurally cannot register a "dagua
  worse" failure when dagua is better -- confirmed 0/337 divergent combos have `np` as a failing
  leg in the dagua-better set, vs stress/cross's symmetric two-sided TOST which flags both
  directions. This is the template S4 should generalize into an explicit "quality-superior" tier
  policy.
- Ranked by expected reclassification impact: (1) S1b sampling-noise fix protects future huge-
  graph work more than it reclassifies current combos (0 today, blocks S2); (2) S4 tiering policy
  change affects up to 93 combos' *headline classification* without changing any pass/fail math
  (pure bookkeeping, zero regression risk); (3) S1a crossings-floor is mostly *not* a bug --
  expect only a handful of combos (bucket estimate: <10) where the systematic 1-edge-crossing
  difference is itself a genuine near-tie artifact worth a floor bump, not a wholesale relaxation.

## 2. Findings (ranked by expected combo-count impact)

### F1. [S2] `r74_analysis.jsonl` never covers 15 of 16 >300-node graphs -- CONFIRMED

The brief instructs enumerating the >300-node worklist from
`eval_output/fidelity_definitive/r74_analysis.jsonl`. That file has 1447 rows spanning 72 distinct
graphs; cross-referencing against node counts pulled from `results.json` position-mapping (
`num_nodes` field, e.g. `benchmark_100seed_r74_fixes/results.json` key
`asymmetric_hourglass_hub::classic_classical_mds_default::seed100` -> `"num_nodes": 14`), only
`random_dag_200` (383 nodes) among the 16 big graphs (`ba_500/2000/5000`, `er_500/2000`,
`grid_20x20/50x50`, `powerlaw_500/2000`, `rgg_500/2000`, `small_world_500/2000`,
`dependency_500`, `sbm_8x100`) has ANY rows in `r74_analysis.jsonl`, `r74_evalrescore.jsonl`, or
`r74_phase2_rescore.jsonl`. All three r74-vintage files show identical single-graph coverage:
```
$ python3 -c "... graphs present in r74_analysis.jsonl ..."
big graphs present: ['random_dag_200']
big graphs ABSENT: ['ba_2000','ba_500','ba_5000','dependency_500','er_2000','er_500',
  'grid_20x20','grid_50x50','powerlaw_2000','powerlaw_500','rgg_2000','rgg_500','sbm_8x100',
  'small_world_2000','small_world_500']
```
This is not a scope note buried in a comment -- it means the r74 fixes/rescore round silently
regressed coverage of the entire big-graph tier back to nothing scored, rather than "scored but
skipped/hung." The most recent snapshot with full 16-graph coverage is
`eval_output/fidelity_definitive/per_combo_r73.jsonl` (3955 rows, all 16 big graphs present, 400
rows on big graphs total). Using that as the worklist source (see section 4 for the full
enumeration): **238 of 400 big-graph combos are divergent** (`quality_identical_raw=False`) as of
r73 -- an order of magnitude closer to the brief's "~165" guess than r74_analysis.jsonl's "9",
but still a materially different number the sprint lead should know before scoping S2's follow-up
task. Some of the 238 will already be fixed by r72/r73/r74 changes that landed after this snapshot
was taken (their small-graph analogues show up fixed in `r74_phase2_rescore.jsonl`'s 72
newly-reclassified rows) -- so 238 is an upper bound, not a current-truth count; a fresh capped-
timeout run against current `develop` is needed to get the real number, which is exactly S2's
proposed follow-up task.

### F2. [S1b] Sampled-crossing standard error computed then discarded -- CONFIRMED

`dagua/metrics.py:826-833` (`sampled_crossing_rate`) returns `crossing_se` (a binomial proportion
SE: `(rate*(1-rate)/n_valid)**0.5`), but the only caller in the analysis pipeline,
`scripts/definitive_fidelity_analysis.py:1726-1745` (`crossing_count`), does:
```python
result = sampled_crossing_rate(pos, ei, n_samples=125000, seed=seed)
return int(result["crossing_estimated_total"])
```
(via `dagua/metrics.py:2078-2080`'s `count_crossings` dispatch for E>500) -- only the point
estimate survives; `crossing_se` never reaches `quality_cross_margin`
(`scripts/definitive_fidelity_analysis.py:1598-1619`), which only ever sees
`2%*mean(ref) vs 0.5 floor vs ref_self_spread` (seed-to-seed variance across positions, not
sampling variance of one position's crossing estimate). Today this affects **0** of the 163
cross-failing combos in `r74_phase2_rescore.jsonl` because `cross_sampled=False` on all of them
(confirmed: `cf['cross_sampled'].value_counts()` == `{False: 163}`) -- Phase-2's <=300-node scope
happens to keep every current cross-failing combo under the E<=500 exact-count threshold
(`dagua/metrics.py:2068`). It becomes load-bearing the moment S2's big-graph worklist is
processed: many of the 238 big-graph combos in F1 will have E>500 (`ba_5000`, `grid_50x50` etc are
guaranteed to), so this bug should be fixed BEFORE S2's follow-up task runs, not after.

The seeding *is* correctly calibrated on one axis: `cross_seed = stable_int_seed(f"{combo_id}::
r70::crossings")` (`scripts/definitive_fidelity_analysis.py:1323`) is shared between the dagua and
reference calls to `crossing_count` for the same combo, so both sides sample the identical
edge-pair subset -- that part is a legitimate paired comparison, not two independently-noisy
estimators being compared. The gap is narrower than "sampling noise ignored entirely": it's
"sampling noise not incorporated into the MARGIN," so a small true difference amplified by a
noisy 125k-sample estimate on a 5000-edge graph could produce a false-positive divergence that a
wider margin would correctly absorb.

### F3. [S1a] Crossings floor is mostly doing its job; systematic near-ties are the real residual -- MOSTLY NOT A BUG, partial fix warranted

Quantified over the 163 TOST-gated cross-failing combos in `r74_phase2_rescore.jsonl`:
```
abs_delta:  min 0.35  p25 2.23  median 7.0  p75 39.0  max 6001.0
cross_margin: min 0.5  p25 0.5  median 2.2  p75 9.96  max 425.9
ref_self_spread==0 (deterministic ref): 59.5% of the 163
margin sits exactly at the 0.5 floor: 36.8% of the 163
abs_delta<=1: 24/163 (14.7%);  abs_delta<=2: 34/163 (20.9%);  abs_delta<=3: 56/163 (34.4%)
```
The brief's hypothesis was that discreteness collapses `ref_self_spread` to 0 on small/sparse
graphs, making the 0.5 floor bite on noise. Inspecting the 34 combos with `abs_delta<=2` AND
`ref_self_spread==0` (`hub_skip_superfan`::sugiyama-family, `multiscale_skip_cascade`::sugiyama-
family, `center_port_backedge_hub`::classical_mds, `petersen_10`::sugiyama_graphviz_fidelity,
etc.) shows these are `mode: "B"` combos (deterministic reference) where dagua's OWN seed-to-seed
crossing count is ALSO essentially constant (`stress_D_sd` ~1e-16, `near_deterministic: true`
flag set). Example: `hub_skip_superfan::classic_sugiyama_default` -- dagua draws exactly 3
crossings on all 100 seeds, igraph's sugiyama draws exactly 2 on its single deterministic layout,
every time. This is a real, reproducible, systematic 1-crossing difference between two
deterministic algorithms, not floor-collapsed noise -- the IUT test is behaving correctly by
flagging it as not equivalent. A human reviewing "1 crossing off on a ~10-15 node DAG, always" may
or may not call that "identical" depending on taste, but it is NOT the "any honest reading calls
it identical" case the brief hypothesized; it is a genuine (probably crossing-minimization
heuristic) discrepancy that belongs in the sugiyama bucket's queue, not a metrics-layer floor
fix. Recommend NOT relaxing the floor to cover this population -- doing so would also swallow
real algorithmic differences.

Where a floor tweak IS defensible: the 5 combos with `abs_delta<=0.5` (already passing under
current settings, since 0.5 is both the delta and the floor -- edge case at exact equality) and a
handful with `abs_delta` in [0.5, 1] straddling the floor by less than one full crossing on
graphs where BOTH sides show `ref_self_spread>0` (i.e. genuinely noisy, not deterministic) --
e.g. `deep_chain_20::classic_fmmm_steps10` (D=0.467, R=0.033, margin=0.5, ref_spread=0.197: the
noisy averaged crossing count itself is sub-1, this is a fractional-seed-average artifact of
`metric_mean` over a 0/1-heavy small sample, not a real 1-crossing gap). These fractional-count
near-zero cases (crossing counts averaged over 60-100 seeds where individual seeds mostly draw 0
crossings) are the closest match to the brief's hypothesis and number roughly 10-15 of the 163,
concentrated in `classic_fmmm_steps10` and `classic_sfdp_p_neg2` on small/medium graphs.

### F4. [S4] Dagua-better set is real and dominated by one engine's stress leg -- CONFIRMED, root cause NOT found (adjacent bucket)

Independently recomputing "dagua strictly better on every metric leg that fails the IUT test"
over the 337-combo target list gives **93** combos (matches the brief's number exactly), with 89
more "mixed" (some legs better, some worse). Breakdown by engine (`/tmp/r75_better_stats.txt`):
`classic_sfdp_p_neg2` (22), `classic_sugiyama_passes4` (8), `classic_sfdp_theta08` (7),
`classic_sfdp_graphviz_fidelity` (6), `classic_sfdp_theta04` (6), `classic_sugiyama_wide` (6), and
a long tail. Of the 93, `stress` is the failing-and-better leg in the overwhelming majority (108
stress-leg failures across the union set, vs 47 cross-leg, 0 np-leg -- np structurally cannot
appear here, see F5). Checked whether `classic_sfdp_p_neg2` is a parameter-mismatch bug per the
guardrail: `dagua/eval/variants.py:1615-1622` shows
`{"steps": 500, "theta": 0.6, "repulsive_exponent": -2.0, "fidelity_mode": "graphviz"}` on the
dagua side matched against `{"maxiter": 500, "theta": 0.6, "repulsiveforce": -2.0}` on the
graphviz reference side -- steps/theta/repulsive-force all matched, ruling out the simple
param-mismatch pattern. The stress deltas are large and one-directional (e.g.
`disconnected_label_cycle_collage`: dagua stress 0.0008 vs reference 0.089, ~110x "better";
`multi_component_80`: dagua 0.0648 vs reference 0.2036, ~3x "better") which smells more like a
reference-side quality problem specific to very-negative repulsive exponents (graphviz's sfdp
with `repulsiveforce=-2.0` is a known-unstable regime that can produce poorly-converged or
irregular layouts) than a dagua superiority claim. This needs a dedicated sfdp-bucket
investigation (positions inspection, not just metrics) -- out of scope for the metrics/criteria
bucket, but flagged per S4's root-cause-first mandate: **do not tier this set as "confirmed
quality-superior" until the sfdp-bucket investigation clears it**, since 22-34 of the 93-112
dagua-better combos trace to this one unresolved engine.

### F5. [S4] np's one-sided design vs stress/cross's two-sided design -- CONFIRMED, exactly the tiering template needed

`quality_np_noninferiority` (`scripts/definitive_fidelity_analysis.py:1642-1669`) computes:
```python
noninferior = d_mean >= r_mean - margin
return {"p_tost": 0.0 if noninferior else 1.0, ..., "equivalent_direct": noninferior}
```
This is a pure one-sided threshold check: dagua passes whenever it is not-worse-by-more-than-
margin, with NO penalty and NO flag for being much better. Compare `quality_stress_margin` /
`quality_cross_margin` callers, which route into `paired_tost`/`one_sample_tost`
(`dagua/eval/distributional_fidelity.py:272-309`, `_tost_values` at 917-973) -- genuine two-sided
TOST: `p_tost = max(p_low, p_high)`, and the degenerate-SD branch checks `abs(mean) <= margin`
(symmetric around zero), so a large negative deviation (dagua much lower/better) is judged
identically to a large positive deviation (dagua much worse). This IS why 0/337 divergent combos
have np as a failing-and-better leg: the test cannot produce that outcome by construction. This
is the correct behavior for np (a "recall" metric where more is unambiguously better and there is
no equivalent-but-different concept), and is exactly the model S4 should adopt for a policy
change on stress/cross's "dagua much better" case -- but converting stress/cross to one-sided
tests would be wrong (unlike np, "dagua's stress/crossings is far below the reference's" is
usually a comparison-basis red flag per the sprint's asymmetry note, not evidence of quality; a
one-sided test would silently launder those bugs into a passing headline).

## 3. Fix sketches, expected impact, and risk

### Fix A (S1b, HIGH priority for S2 unblock): propagate `crossing_se` into the cross margin
- **Sketch**: `crossing_count()` (`scripts/definitive_fidelity_analysis.py:1726`) should return
  `(count, se)` instead of just `count` for the sampled path (se=0 for the exact E<=500 path).
  `quality_metric_samples` / `quality_reference_diagnostics` should carry a
  `cross_sampling_se` array alongside `cross_d`/`cross_r`, and `quality_cross_margin` should add
  a `max(existing_margin, k * sampling_se_pooled)` term (k~2 for ~95% CI) ONLY when
  `cross_sampled=True`, leaving the E<=500 exact-count path (163/163 of today's failures)
  completely untouched.
- **Expected impact**: 0 reclassifications in the current 337-combo divergent set (no sampled
  combo is currently in it). Prevents false divergences appearing when S2's big-graph rescore
  runs (F1's 238-combo worklist will include many E>500 graphs). Framing this as prerequisite
  infrastructure for S2, not a combo-count win today.
- **Risk**: low -- purely additive to sampled-path margin, gated by `cross_sampled` flag so it
  cannot touch any of the 163 currently-failing exact-count combos or any bit-exact combo (which
  by definition isn't in the quality-battery path at all). Must re-run gate_5 (0/40
  quality-identical-laundering) and gate_4 (chance-control KS) after the change since both use
  sampled crossings on some control graphs -- verify `eval_output/fidelity_definitive/controls/
  gate_results.json` stays `gate_5_quality_identical_laundering.passed: true` (currently 0/40,
  `three_q_percent: 0.0`) and `gate_4_chance.passed: true` (currently `ks_p: 0.82`).

### Fix B (S1a, LOW-MEDIUM priority, narrow scope): fractional near-zero crossing floor for noisy sub-graphs only
- **Sketch**: Do NOT touch the deterministic (`ref_self_spread==0`) systematic-difference
  population (F3's sugiyama/mds cases -- those are real, route to algorithm buckets). For the
  ~10-15 combos where `ref_self_spread > 0` (genuinely noisy across seeds) AND
  `cross_R_mean < 1.0` (sub-single-crossing average, meaning most seeds draw 0), consider a
  count-floor of `max(0.5, 1 seed-flip-equivalent)`: since a single seed flipping 0->1 crossings
  changes the 60-100-seed mean by ~1/60 to 1/100, and the CURRENT floor of 0.5 already tolerates
  ~30-50 such flips, this floor is almost certainly already generous enough -- the actual fix
  needed is likely in how `metric_mean` handles near-zero fractional crossing counts (verify the
  TOST isn't being fooled by a fractional mean that no single seed ever actually produces), not
  the floor value itself. Cheapest decisive experiment (~10 min): for the 15 candidate combos,
  histogram the per-seed `cross_d`/`cross_r` samples (not just the mean) to confirm they're
  genuinely 0/1-valued Bernoulli-like distributions before proposing any margin change.
- **Expected impact**: at most 10-15 combos, and only if the per-seed histogram experiment
  confirms a genuine near-tie population distinct from F3's systematic-difference population.
- **Risk**: MEDIUM if scoped wrong -- any floor increase applied broadly (not gated to the noisy
  subset) would repeat the r74 stress-scale-artifact failure mode by masking real per-engine
  crossing regressions on other combos. Must re-run gate_5 and gate_3 (negative control) after any
  change; gate_3 is currently already failing (`passed: false`, `non_primary_percent: 90.0` --
  a pre-existing issue unrelated to crossings, worth flagging to the sprint lead but out of this
  bucket's fix scope).

### Fix C (S4, LOW risk, pure bookkeeping): add a "quality-superior, distributionally-distinct" tier
- **Sketch**: for combos where ALL failing legs are dagua-better (the 93-combo set), add a new
  tier field (e.g. `quality_superior_not_identical: bool`) computed exactly like
  `metric_identical` but checking `d_better_than_r` instead of `equivalent`. This tier must NOT
  feed into `quality_identical_raw` (the north-star identical headline) -- it's a reporting/
  triage label only, distinguishing "divergent because worse" from "divergent because
  suspiciously better" so the latter gets root-cause priority (per the sprint's own asymmetry
  finding that "dagua better" usually means a comparison bug). Model directly on how np's
  one-sided test already treats "better" as a pure pass (F5) -- but do NOT convert stress/cross to
  one-sided; keep the two-sided TOST as the source of truth for `quality_identical_raw`, and layer
  the better/worse tag on top only for combos that already fail the two-sided test.
- **Expected impact**: 93 combos get a corrected triage label (helps focus the sfdp_p_neg2
  investigation from F4); zero combos change pass/fail status, zero regression risk to bit-exact
  or quality-identical combos since this is purely additive metadata.
- **Risk**: near-zero -- it's a new field, not a change to any existing gating logic. The only
  discipline required is to NOT let this tier count toward the "everything identical" north-star
  metric in any dashboard/report rollup (explicit ask in the bucket brief).

## 4. Big-graph (>300 node) divergent worklist for S2's follow-up task

Source: `eval_output/fidelity_definitive/per_combo_r73.jsonl` (most recent full-coverage snapshot;
`r74_analysis.jsonl` has no rows for 15/16 of these graphs, see F1). Node counts from
`results.json` `num_nodes` field.

Big graphs (16, node count in parens): ba_500(500), ba_2000(2000), ba_5000(5000), er_500(500),
er_2000(2000), grid_20x20(400), grid_50x50(2500), powerlaw_500(500), powerlaw_2000(2000),
random_dag_200(383), rgg_500(500), rgg_2000(2000), sbm_8x100(800), small_world_500(500),
small_world_2000(2000), dependency_500(500).

Divergent-combo counts by graph (238 total, r73 snapshot, likely an overestimate of current-develop
truth since r72/r73/r74 fixes landed after this snapshot for some engines -- treat as upper bound
pending a fresh capped-timeout rescore):
```
random_dag_200      31   small_world_500   24   rgg_500        23
grid_20x20          16   powerlaw_500      16   er_500         15
sbm_8x100           15   ba_500            14   powerlaw_2000  14
rgg_2000            14   dependency_500    13   er_2000        11
small_world_2000    11   ba_2000            9   ba_5000         6
grid_50x50           6
```
Top engines by divergent-combo count across these 16 graphs: classic_sfdp_theta08 (15),
classic_sugiyama_{default,passes4,passes48,tight,wide} (13 each), classic_sfdp_{default,
graphviz_fidelity,p_neg2,steps200} (12 each), classic_fmmm_steps10 (11), classic_drl_refine (10),
classic_sfdp_theta04 (10), classic_fr_steps500 (7), classic_fmmm_graphviz_fdp_fidelity (7).
This gives S2's follow-up task a concrete, sized target: expect on the order of 150-240 combos
(pending fresh rescore) concentrated in sfdp/sugiyama/fmmm, the SAME engine families already
dominant in the <=300-node crossings-failing set (F3) -- suggesting whatever root causes exist in
those engine buckets are size-independent, not scale-specific artifacts.

## 5. Target combos I could NOT explain

- The `parallel_cycles_4x5::classic_neato` and `parallel_cycles_4x5::classic_sfdp_graphviz_fidelity`
  rows show unusually LARGE stress values for dagua on a tiny graph (D=1.0 vs R=0.018 for neato;
  a 55x gap) that don't fit either the "dagua systematically better" or "simple noise" pattern --
  this looks like a possible degenerate/collapsed layout on one specific small graph, worth a
  positions-file visual check, but I did not have budget to pull `positions/parallel_cycles_4x5__*`
  and render it within the 45-minute cap.
- `disconnected_label_cycle_collage`'s `E`/`E_cross` fields (visible in the full per-combo record)
  carry ~0.9 values that look like a DIFFERENT equivalence metric (possibly the Procrustes-family
  distributional check, not the 3Q battery) -- I did not have time to trace what `E`/`E_cross`
  independently measure relative to `battery_stress`/`cross`/`np_*`; flagging so the rival report
  or a follow-up bucket can clarify whether that field family needs its own audit pass (it wasn't
  in this bucket's brief scope, but its presence alongside the 3Q fields in every row suggests it
  feeds SOME classification decision upstream of `quality_identical_raw`).
- Could not fully explain why `gate_3_negative` is currently failing (`non_primary_percent: 90.0`,
  target implied ~100% or a stated threshold) -- outside this bucket's brief (S1-S4 are all about
  the 3Q battery / crossings specifically), but it's a live red flag in the same controls
  infrastructure gate_5 depends on, and any crossings-margin change (Fix A/B) should re-verify
  gate_3 didn't get worse, even though this bucket doesn't own fixing it.

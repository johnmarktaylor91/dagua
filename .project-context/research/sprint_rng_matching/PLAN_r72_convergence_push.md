# PLAN r72: Convergence Push (research-grounded)

Version: 2 (post adversarial review -- 10 findings, 2 CRITICAL; resolutions in Appendix R).
Builds on r71 (705->463 divergent). 5 research agents delivered
verified findings (2026-06-13). JMT decisions locked: new tier = quality BATTERY (stress +
crossings + k-NN neighborhood) at ~1-2% tight tolerance; FP push = port-parity hard,
libm-emulation = principled stop.

## Research headlines (what changed the picture)

1. **UMAP is ALREADY BIT-EXACT** (verified 0.0 diff, kernel + end-to-end). The ~24 "residual"
   combos are an ADAPTER artifact: the reference runs umap-learn on FEATURES while dagua does
   graph-UMAP on APSP distances. FIX: reference adapter -> `metric='precomputed'` on the same
   APSP matrix. Easy win.
2. **FMMM port is tractable**: dagua ALREADY has the bit-exact single-level OGDF force kernel
   AND the solar-system multilevel coarsening -- they're just NOT WIRED. Port = wiring + 5
   corrections (biggest: per-level `get_max_mult_iter` budget). ~194 combos. 2-3 codex rounds.
3. **sgd2_multi + neato weighted divergence is FIXABLE** (17 combos, one scoped edit): native
   gets edge weights via _quick_classic blanket-passing, but the GD2/graphviz-neato REFERENCES
   use UNWEIGHTED distances. Match the reference -> exclude {classic_sgd2_multi, classic_neato}
   from weight-passing. (Consistent with the drl fix: match each reference's weight semantics.)
4. **sfdp is genuinely irreducible** (verified Lyapunov ~0.8/iter; 1-ULP -> basin by iter 100).
   Port is source-complete. Route to the new quality-identical tier. Principled STOP confirmed.
5. **gem 23 / drl 5 / neato 10 are chaos-floor** (cross-divergence <= reference self-variation),
   route to quality tier. **davidson_harel already resolved** (0 divergent; 11 insufficient =
   reseed).
6. **New tier "3Q" QUALITY_IDENTICAL** spec complete: battery {stress@2%, crossings@2%/floor0.5,
   neighborhood-preservation@0.02-abs}, combined as INTERSECTION-UNION test (AND of 3 TOSTs, NO
   multiplicity correction -- Berger-Hsu), BH the IUT p across combos. All 3 metrics already in
   equivalence_metrics.py. Insert rung "3Q" between rung-3 (kept as looser stress-only fallback)
   and rung-4. Est. ~140-220 combos land here, dominated by sfdp + sugiyama.
7. **P3 gaps**: 248 insufficient -> 196 STRUCTURAL_NA (compute frontier), ~52 recoverable
   (26 cheap @900s, 26 giant-graph expensive). All timeouts, no logic bugs.

## Work plan (phases)

### Phase I -- parallel codex fixes (disjoint files)
- **I-A FMMM multilevel port** (`dagua/layout/ops/pipelines/fmmm.py` + import `ops/fmmm.py`).
  Round 1: wire bit-exact single-level kernel into existing coarsening hierarchy +
  corrections D (exact get_max_mult_iter per-level budget) + E (per-edge desired-length
  attractive force). Target: dispersion ratio 1.4x -> 1.0 on 4 anchors (random_dag_50,
  transformer_layer, multiscale_skip_cascade, rgg_100=CANARY don't over-correct). Round 2
  (only if dispersion matches but per-seed RMSD high): corrections A+B+C (mt19937 sun-selection
  RNG + Advanced prolongation + waggle RNG form) for bit-exactness. HIGH effort.
- **I-B new tier "3Q"** (`dagua/eval/distributional_fidelity.py` + `scripts/definitive_fidelity_
  analysis.py` + `scripts/definitive_fidelity_report.py`). Per the quality-battery spec:
  per-seed crossings + neighborhood-preservation alongside existing stress; 3 paired TOSTs;
  IUT (max-p) + BH -> q_battery; rung "3Q" between 3 and 4; keep rung-3; PASSING_RUNGS += 3Q;
  MODE_A_PASS_RUNGS unchanged (3Q is NOT distributionally-matched); 5-tier report + breakdown
  section. Cost guard: reuse per-combo dists, build edge tensor once, cap battery seeds if slow.
- **I-C weight fix** (`dagua/eval/competitors/classic_competitor.py` only): exclude
  {layout_sgd2_multi_pipeline, layout_neato_pipeline} from _quick_classic weight-passing
  (line ~1702), keyed on fn_name (NOT name substring); unit test asserts sgd2_multi/neato get
  NO edge_weights while stress_maj/maxent/stress_sgd still DO (they use separate dedicated
  sites, unaffected). Medium. **UMAP ADAPTER FIX DROPPED** (review CRITICAL-2: umap_competitor
  already uses metric='precomputed' on APSP -- the research "fix" is a no-op). umap's 24
  residual combos are handled by the 3Q tier (I-B): if SGD-basin but quality-identical, they
  land in 3Q automatically. No umap code change.

### Phase B -- re-benchmark fixed engines (provenance-stamped, fresh dirs)
- fmmm (after I-A verified): fmmm variants on failing-map graphs.
- umap reference (after I-C): umap_graph__for__* on umap failing graphs (reference changed).
- sgd2_multi + neato (after I-C): native on the 17 weighted graphs (native weight path changed).
- P3 recoverable-cheap: the 26 combos @ timeout 900 + umap_nn30 ref (10) + davidson reseed (11).

### Phase A -- union re-analysis + report v3
- Re-run the producer (analysis) with the NEW battery code over the union store (all overlay
  dirs: escalation + seeded_refs + fmmm_realfix + umap_realfix + sgd2neato_realfix + drlref +
  gem_realfix + P3 recovery). This computes 3Q for ALL combos.
- Report v3: 5-tier (1/1b/2/3/3Q/4) + quality-identical breakdown per family. Final scorecard.
- Supersession (r71 -> r72), file-for-review, text JMT.

## Expected outcome (projection)
- Fixes close: umap ~24, sgd2/neato ~17, fmmm ~194 (if port lands), = ~235 combos.
- New tier reclassifies ~140-220 sfdp/sugiyama "divergent" -> "quality-identical" (honest:
  equally-good drawing, different basin).
- Remaining genuine DIFFERENT (fails distributional AND quality battery): the hard chaos floor,
  likely <150 combos, ~3% of all pairs -- most being disconnected-component packing chaos.
- Honest 100%-framing: every pair either bit/stat-equivalent, quality-identical, or
  documented-irreducible (FP floor / structural compute limit).

## Appendix R -- adversarial review resolutions (2026-06-13)

10 findings (2 CRIT, 4 HIGH, 3 MED, 1 LOW), all incorporated:
- CRIT-1 (3Q laundering): I-B MUST add an anti-laundering gate -- compute the full battery on
  negative-control + chance-control rows; assert 3Q rate on them <= 5% (ideally ~0); HARD gate
  in the r72 gate file BEFORE 3Q enters the report. If negatives launder, tighten margins.
- CRIT-2 (umap no-op): umap adapter fix DROPPED; umap residual -> 3Q tier (above).
- HIGH-3 (rung-set sites): I-B enumerates ALL ~7 hardcoded rung-set literals -- add 3Q to
  `usable` filter (report ~line 1400), PASSING_RUNGS; KEEP 3Q OUT of MODE_A/MODE_B_PASS_RUNGS
  (so it doesn't inflate DISTRIBUTIONALLY_MATCHED -- verified correct); assign_rung BOTH
  mode-A and mode-B branches (ordered above "4", below "3"); gate counters; rung4 detection.
- HIGH-4 (FMMM regression): capture current passing-fmmm combo set; assert no currently-passing
  combo regresses to rung 4. `get_max_mult_iter` is NET-NEW code (not wiring) -- budget it.
- HIGH-5 (FMMM Round-1 criterion): Round-1 success = the TWO-SAMPLE distributional test on the
  4 anchors (NOT just dispersion ratio); pre-authorize Round 2 (corrections A+B+C: mt19937
  sun-selection RNG + Advanced prolongation + waggle) so codex doesn't stop at dispersion-match.
  ~194 projection is CONDITIONAL on the port landing distributionally.
- HIGH-6 (complete re-analysis): Phase A re-analysis runs over the COMPLETE combo set (no
  restrictive --combos-file); gate: every non-insufficient non-bit-exact combo has finite
  q_battery before the report runs.
- MED-7 (rung-3 vs 3Q clarity): report rung-3 (loose stress-only @5%) and 3Q (strict battery)
  as DISTINCT lines -- "stress-only-equivalent (loose)" vs "quality-identical (strict)" --
  never blurred; both in Tier 3 accounting but separately counted. (CC decision, not asked.)
- MED-8 (crossings determinism): pin a FIXED seed for the sampled-crossings path (E>500);
  document large-E crossings as estimates.
- MED-9 (provenance): add classic_fmmm*, classic_sgd2_multi, classic_neato to fixed_engines.json
  with pre_fix_dirs + new fixed_sha before the v3 report asserts provenance.
- LOW-10 (weight fix scoping): key the exclusion on fn_name with a unit test (above).

## Principled stops (leave nothing on the table EXCEPT these)
- sfdp last-ULP libm bit-emulation (verified chaos; port is source-complete).
- 196 structural-NA insufficient combos (compute frontier, not bugs).
- gem/drl/neato-structural chaos floor (reference non-reproducible across seeds).
- FMMM bit-exactness IF Round 1 dispersion-match suffices for distributional equivalence
  (don't chase bit-exact RNG if the ensemble matches).

## Execution discipline
- Verify every fix on the get_competitor() BENCHMARK path (NOT direct pipeline -- the false
  umap fix lesson). Per-seed Procrustes vs reference for bit-exact claims.
- Parallel codexes own DISJOINT files (fmmm-pipeline / distributional+report / competitors).
- BLISS/toolkit calls -> hard-killed subprocess. kill -9 process GROUP for orphans.
- Provenance: stamp git_sha; fixed_engines.json; report asserts no pre-fix rows for fixed engines.
- State r72_convergence_push_STATE.md + gate. Anti-flail: 3 rounds/family then accept residual.

## Open risks (for the reviewer)
- FMMM RNG call-order parity (A+B+C) is the make-or-break for bit-exact; dispersion-match
  (Round 1) may suffice for distributional equivalence -- verify before chasing bit-exact.
- 3Q crossings cost O(E^2) exact for E<=500; neighborhood O(N^2 logN)/seed -- the new heavy
  cost; cap battery seeds / reuse dists.
- sgd2_multi/neato weight exclusion: is matching the unweighted reference the right fidelity
  call, or should weighted be a dagua feature? (Surface; default = match reference for fidelity.)
- IUT no-multiplicity-correction is correct (Berger-Hsu) but counterintuitive -- document WHY.
- Union-store merge order (freshest last) -- the r71 merge-order bug must not recur.

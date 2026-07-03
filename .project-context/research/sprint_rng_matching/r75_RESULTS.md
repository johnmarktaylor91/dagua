# r75 -- Fidelity Endgame Sprint: RESULTS (2026-07-01/02, Fable-led)

## Headline
Entering divergent set (409 combos, <=300 nodes): resolved to
- **111 quality-identical** (certified, gate_5 0/40 held)
- **55 no-canonical-reference** (theta04/theta08/steps200: graphviz 7.0.5 provably cannot express
  those knobs -- JMT-approved exclusion tier; incl. 8 insufficient-data rows carrying the flag)
- **79 divergent-but-superior-distinct** (dagua measurably better on every failing leg; explicitly
  labeled DIFFERENT, never counted as identical)
- **163 still divergent** (sugiyama 121, sfdp 44, fmmm 33, classical_mds 22, gem 7, umap 7,
  maxent 3, drl 3, neato 2) -- every one carries a named root cause or evidenced floor
- 1 insufficient-data
Honest divergent 337 (stale accounting) -> 242 (163 + 79). Direct flips vs r74-phase2: 50
(sfdp 35, mds 8, sugiyama 7). Controls: gates 1/2/4/5/6 GREEN (gate_6 newly green); gate_3
pre-existing calibration note only. 3 "regressions" are honesty corrections (2 gem + 1 fmmm --
old passes were stale-binary artifacts).

## What r75 discovered (the systemic defects -- the sprint's real value)
1. **STALE REFERENCE BINARY (fmmm+gem root cause).** Committed scripts/ogdf_runner predated the
   gemRounds/fmmmFixedIterations plumbing -- it silently IGNORED those params, so all
   ogdf_fmmm/ogdf_gem references ran at OGDF defaults while dagua ran matched counts. Proven:
   identical output at 10-vs-200 iterations; dagua fmmm matches the rebuilt source-true runner to
   RMSD 0.000-0.0014 on every previously-diverging probe seed. Stress byte-identical between
   binaries (maxent/stress claims unaffected). Rebuilt binary committed (0817427); all fmmm/gem
   references regenerated. gem's honest state is WORSE than previously believed (r71 port matched
   the stale binary) -- r76 work item.
2. **SEED-ERA FRANKENSTEIN (overlay defect).** load_results_multi unioned rows per seed-key
   across dirs, mixing layouts from different code eras into one battery (4,502 of 8,756 combos
   affected in the 11-dir chain). Fixed: per-combo freshest-dir-wins resolution (7fa972e).
   This defect masked the DLA port's success (0/30 flips -> 8 real flips after fix).
3. **STALE-VINTAGE BASELINE.** The r74 "337 divergent" scored p_neg2 on pre-clamp layouts (52
   rows). 35 flipped when scored on current positions.
4. **NO-CANONICAL-REFERENCE variants.** graphviz sfdp ignores theta/maxiter graph attrs
   (compile-time constants); 47 scored rows moved to a documented non-counting tier.
5. **Benchmark batch-timeout semantics.** --timeout applies to the whole 100-seed batch per
   combo; slow engine/graph pairs need batch-scale ceilings (caused mds/fmmm/sugiyama false
   errors; topped up; 9 sugiyama_graphviz combos remain unmeasured -- stage-A simplex too slow at
   that scale, r76 perf item).

## What r75 shipped (develop, 10 branches merged)
- fix(eval): sampled-crossings estimator (denominator + SE propagation) + exact/sampled predicate
  unification + quality_superior_distinct metadata (205129e)
- feat(classical_mds): igraph-faithful disconnected path -- per-component MDS + literal DLA merge
  port (ec24b05) + collision-lookup perf fix, bit-identity 9/9 (8266c77)
- feat(sugiyama): graphviz network-simplex x-coordinates, stage A -- binary_tree x matches
  graphviz to 1.5e-16 (e6ba3db)
- fix(sugiyama): igraph 1.0.0 LP objective quirk port (IN/IN), runtime-verified regression
  (bb72c8b)
- fix(bench): rebuilt ogdf_runner + provenance (0817427)
- feat(eval): no-canonical tier + superior-distinct report line (01e589f)
- fix(eval): per-combo freshest-dir overlay (7fa972e)
- fix(eval): gate_6 reference-self-split control data (0065174); controls_full/ holds full data
- docs: mincross attempts 1+2 failure analysis (00a2893; code parked in codex transcripts)

## Honest failures / parked
- **mincross phase 1: 2 attempts, ladder not passed.** Attempt 2 matched dot -v ordering counts
  on 3/4 calibration graphs; residual = GD_nlist install order + representative-chain ED_xpenalty
  merge. Parked with full analysis + recoverable patches (r75_findings/r75_cx_mincross*.log).
- FMMM's 33 remaining divergent rows: references now honest; residual needs triage (steps100/200
  parity + crossings-leg discreteness) -- r76.
- Process incident: a forced worktree removal after unverified commits destroyed attempt-2
  working code (recoverable); lesson banked -- verify git log before worktree remove.

## r76 queue (THE FINAL SPRINT -- JMT standing authorization, codex-first routing)
mincross GD_nlist + chain-merge port; stage-A x-simplex perf >~100-node graphs (unblocks 9
unmeasured combos + ba_500 tail); sugiyama stages B-D (flat/labels/clusters); igraph GLPK tie
parity + BK ordinal-edge + qsort ties; big-graph tier hang-safe scoring + rescore; fmmm residual
triage; gem re-port vs honest refs; umap numba trace; drl/neato perturbation dispositions;
population-equivalence aggregate tier; final ledger: every combo = identical | evidenced floor |
no-canonical | superior-distinct | aggregate-equivalent.

## Key artifacts
- Final data: eval_output/fidelity_definitive/r75_final.jsonl (409 rows, fixed loader);
  per_combo_r75.jsonl (full 3,955-combo store); official report eval_output/fidelity_definitive_r75/
- Research + verdicts: r75_findings/ (10 research reports, adversarial verdicts, probe results,
  impl notes); full iteration log: r75_endgame_STATE.md
- Benchmark dirs added: benchmark_100seed_r75_fixes, _r75_mds_topup, _r75_topup2

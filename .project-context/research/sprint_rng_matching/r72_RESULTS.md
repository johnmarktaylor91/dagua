# r72 Convergence Push -- RESULTS (DEFINITIVE, supersedes r71)

**Date:** 2026-06-15
**Status:** COMPLETE. This is the current authoritative fidelity verdict. Supersedes r71
(`project_r71_fidelity_completion`, `r71` report dir) and all earlier (r70, WHERE_WE_STAND,
ALLGRAPHS_SUMMARY).

## Headline

Escalation-divergent (Mode-A stochastic, seed-tracked vs seeded references):

| | r70 | r71 | r72 preR3 | r72 fmmm-R3 | **r72 FINAL** |
|---|---|---|---|---|---|
| divergent | 705 | 463 | 418 | 381 | **331** |

**-374 combos (-53%) from the r70 baseline.** New "quality-identical" (3Q) tier: **32**
(directional-but-drawing-quality-equal divergences; tight 2% IUT battery; anti-laundering
gate clean 0/40).

Final Mode-A distribution: rung1=2181, rung2=536, rung3=129, **3Q=32**, divergent=331.

## What landed this sprint

1. **FMMM multilevel port (R1-R3, commit b0fc1e8).** Wired dagua's existing bit-exact OGDF
   force kernel + solar-system coarsening; connected-component decomposition (OGDF
   DIVIDE_ET_IMPERA) was the over-dispersion root cause; OGDF-fidelity ignores cluster forces
   (matches plain-graph OGDF reference). OGDF-steps variants: divergent 114->77, 3Q 14->31.

2. **FMMM fdp-routing fix (commit 22817c2).** `layout_fmmm_pipeline` gated the graphviz_fdp
   branch by `and clusters`; unclustered graphs (truthy string `fidelity_mode`) fell through to
   the OGDF FM3 path -- so `classic_fmmm_graphviz_fdp_fidelity` **never ran fdp on plain
   graphs**, producing systematic 0.48x under-dispersion vs the real fdp binary. Routed
   unclustered graphviz_fdp to the fdp emulator + threaded steps->fdp maxiter. fdp_fidelity
   divergent **61 -> 11** (42+18 now distributionally equivalent).

3. **sgd2_multi bit-exact (commit af614aa).** Emulated DataLoader epoch-shuffle + ideal-edge
   RNG stream; 0.04-0.46 -> ~1e-6 at matched seeds.

4. **neato weight fix (commit 98a7264).**

5. **3Q QUALITY_IDENTICAL tier (commit 02146c2).** Battery {normalized stress + edge crossings
   + k-NN neighborhood} as intersection-union TOST (Berger-Hsu, max-p, no multiplicity
   correction), BH-corrected; tight 2% tolerance; sits between Tier 3 and Tier 4. Anti-laundering
   gate: 0/40 negative+chance controls launder in.

## Remaining 331 divergent -- the floor

| family | n | nature |
|---|---|---|
| sfdp | 184 | irreducible FP-chaos (Lyapunov ~0.8/iter verified; 1-ULP -> different basin by iter 100) -> principled STOP |
| fmmm | 88 | steps10 low-iter FP-divergence + disconnected component-placement ambiguity + 11 fdp local-neighborhood residual + steps100/200 chaotic tail |
| umap | 26 | downstream UMAP-SGD basin |
| gem | 22 | chaos floor |
| drl/neato/maxent | 11 | chaotic tails |

These are dominated by **FP-basin chaos** -- matching them would require bit-level libm/RNG
emulation (last-ULP transcendental matching), which is the **principled STOP** per JMT decision
#2 (2026-06-13). Directional-but-quality-equal cases among them are captured by the 3Q tier;
the rest are genuinely different layouts at the same drawing quality OR honestly unmeasurable.

## Honest caveats (compute-frontier)

- **fdp_fidelity on 12 big graphs (200-500 nodes) -> INSUFFICIENT.** Pure-Python fdp emulation is
  O(N^2)/iteration and exceeds the 300s benchmark timeout. They were *spuriously* divergent
  before the fix (OGDF-vs-fdp comparison); insufficient is the honest verdict. Same precedent as
  sgd2_multi_with_crossing (round 8 compute-frontier).
- **gate3_negative** still "fails" identically to r71: `non_primary 90%` is the ModeB
  typicality-power escape; the ModeA false-tracking sub-gate is perfect (0/limit-3). Documented
  carryover, not r72-introduced.

## Artifacts

- Report: `eval_output/fidelity_definitive_r72/DEFINITIVE_FIDELITY_REPORT.md` +
  `FOUR_TIER_CATEGORIZATION.md` + `per_combo.json` + `controls/gate_results.json`
- Merged verdicts: `eval_output/fidelity_definitive/per_combo_r72_final.jsonl` (3955 combos)
- Scorecards: `r72_scorecard_final.json` (+ `_r3`, `_preR3` snapshots)
- Provenance: `eval_output/fidelity_definitive/fixed_engines.json`
- Benchmarks: `benchmark_100seed_{r72_fixes,fmmm_r3,fdp_fix}`

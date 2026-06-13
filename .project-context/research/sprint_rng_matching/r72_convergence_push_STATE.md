---
run: r72_convergence_push
created: 2026-06-13
state: PHASE_RESEARCH
current_phase: R
gate_file: .project-context/autonomous_gate_r72.json
builds_on: R71_FIDELITY_COMPLETION_SUMMARY.md (705->463 divergent)
---

# r72 -- Convergence Push (autonomous, Opus + codex)

JMT directive (2026-06-13): "launch another sprint pushing as close as possible to full
convergence -- full research agents, make a plan, codex-iterate to convergence, dispatch,
execute. Add a tier 'statistically different but quality-identical' for small DIRECTIONAL
chaotic divergences that don't affect quality. Push hard to get as identical as possible.
Leave nothing on the table unless STRONG principled reasons."

## JMT decisions (locked 2026-06-13)
1. NEW TIER bar = quality BATTERY {normalized stress + edge crossings + k-NN neighborhood
   preservation}, all equal within TIGHT tolerance (~1-2%, STRICTER than the existing 5%
   QUALITY_EQUIVALENT). "Quality-identical" = indistinguishable across all three. Sits as a
   new band BETWEEN Tier 3 (statistically-equivalent) and Tier 4 (different): combos that
   FAIL the distributional/energy test but pass the quality battery.
2. FP push = port-level parity HARD (identical init, coarsening, force model, summation/
   reduction order, same math lib if linkable); true libm bit-emulation (last-ULP
   transcendental matching) is the PRINCIPLED STOP -> remainder routes to the new
   quality-identical tier.

## r71 residual to attack (the targets)
- fmmm (~194): single-level -> OGDF MULTILEVEL port. BIGGEST FIXABLE LEVER.
- sfdp (~185): FP libm basin chaos -> port-parity push, then quality-identical tier.
- umap (~24): downstream SGD basin -> port-parity push, then quality-identical tier.
- gem residual (23), sgd2_multi (18), davidson/neato/drl chaotic tails (~30): findable gaps?
- classical_mds: deterministic-vs-stochastic (handled: r70 deterministic verdict).
- P3 insufficient (248): mostly structural (big-graph timeouts, slow sgd2_multi_ref).

## Phase ladder
| Phase | What | Done when |
|---|---|---|
| R | Research salvo (5 threads, parallel Opus agents) | all 5 reports in; synthesized |
| P | Plan from research; adversarial review (codex/Opus critics) iterate to convergence | reviewer PASS or rounds exhausted |
| I | Dispatch codexes: FMMM multilevel port, FP-parity (sfdp/umap), new-tier impl, chaotic-tail fixes | each committed + verified |
| B | Re-benchmark fixed engines; re-analyze union store | data merged |
| A | Report v3 with new tier; final scorecard; supersession; file-for-review; text JMT | gate all-pass; DONE |

## Research threads (launched phase R)
1. FMMM multilevel port spec (OGDF FMMM internals: coarsening, get_max_mult_iter, multipole).
2. FP-determinism for cross-impl force-layout reproducibility (sfdp/umap; what's port-fixable).
3. Chaotic-tail engines (sgd2_multi/davidson_harel/drl-plain/neato-tail): findable RNG/init gaps?
4. Quality-battery design for the new tier (stress/crossings/neighborhood equivalence testing).
5. P3 structural gaps: which 248 insufficient are recoverable vs genuinely structural.

## Wake-up routing
- research agent returns -> collect; when all 5 in -> synthesize -> draft plan -> review.
- codex DONE -> verify on benchmark path (NOT direct pipeline), commit, re-benchmark.
- benchmark DONE -> re-analyze union, update scorecard.
- BLISS/toolkit calls -> hard-killed subprocess pattern. kill -9 process GROUP for orphans.

## Iteration log
| Round | Phase | When | Result |
|---|---|---|---|
| 0 | R | 2026-06-13 | scaffold; JMT tier+FP decisions locked; research salvo launching |
| 1 | R DONE | 2026-06-13 | All 5 research agents delivered (verified findings). HEADLINES: (1) UMAP ALREADY BIT-EXACT -- residual is adapter artifact (ref runs umap-learn on features not graph-APSP); fix adapter metric='precomputed'. (2) FMMM port TRACTABLE -- dagua has both bit-exact kernel + coarsening, unwired; port=wiring+5 corrections (get_max_mult_iter biggest). (3) sgd2_multi+neato 17 combos FIXABLE -- native uses weights, refs don't; exclude from weight-passing. (4) sfdp IRREDUCIBLE (Lyapunov 0.8/iter verified) -> quality tier. (5) gem23/drl5/neato10 chaos-floor; davidson resolved (reseed 11). (6) new tier 3Q spec complete (battery IUT, Berger-Hsu). (7) P3: 196 structural-NA, 52 recoverable. Plan PLAN_r72_convergence_push.md written. -> adversarial review then dispatch. |
| 2 | P review + dispatch | 2026-06-13 | Adversarial review (Opus): FAIL, 10 findings (2 CRIT: 3Q laundering gate needed; umap "fix" is no-op -- adapter already metric='precomputed'). ALL incorporated -> plan v2 (Appendix R), committed 305196d. umap fix DROPPED (3Q tier handles its residual). Phase I dispatched 3 PARALLEL codexes (disjoint files, verified): I-A FMMM multilevel port pid 2808625 watcher bfbq066o6 (HIGH; wire existing kernel+coarsening + get_max_mult_iter + regression guard + Round1=distributional); I-B 3Q quality tier pid 2808845 watcher b6ekalimq (HIGH; battery IUT + anti-laundering gate + all rung-set sites); I-C weight fix pid 2809072 watcher b5e421h4w (MED; sgd2_multi/neato by fn_name). ON each DONE: verify on benchmark path, commit, re-benchmark; then Phase A union re-analysis (COMPLETE combo set) + report v3. |
| 3 | I-C done (partial) | 2026-06-13 | I-C committed 98a7264. Benchmark-path verify: **neato FIXED** (weighted_chain 42-79 -> 0.0001) but **sgd2_multi NOT fixed** (still 0.55-0.65). Diagnosis: sgd2_multi pipeline only uses the kwarg (suppressed correctly) -> residual is NOT weights but the SGD EPOCH-SHUFFLE RNG mismatch (research agent over-claimed "0.015 unweighted"; sgd2 is stochastic SGD, basin differs at matched seed unless RNG stream matches). 4 broad-test failures (maxent/davidson/pivot/sugiyama weighted-pipeline) are PRE-EXISTING/orthogonal (I-C diff narrowly scoped to 2 fn_names, verified). Dispatched sgd2_multi RNG-match codex pid 2854779 watcher btmeh3gbf (HIGH, disjoint sgd2_multi.py). I-A (2808625) + I-B (2808845) STILL running. ON sgd2 DONE: if RNG matches -> bit-exact; else 3Q tier. LESSON: research-agent benchmark claims need benchmark-path verification (the 0.015 was not the get_competitor path). |
| 4 | I-A r1 done -> r2 | 2026-06-13 | FMMM Round 1 committed 9abd10a. PARTIAL: transformer_layer MATCHED PERFECTLY (disp 1.47->1.00, E->-0.0004 -- proves multilevel wiring+budget right), but inconsistent: random_dag_50 still over (1.39), multiscale UNDER (0.44 over-corrected), real_lesmis mild regression. + PERFORMANCE problem (multilevel path CPU-bound, broad test >11min -> benchmark-timeout risk). Round-1 author flagged controversial length/coordinate-unit scaling = likely root of the over/under inconsistency (systematic, not RNG noise). Round 2 dispatched pid 2873684 watcher byo2di6m5 (HIGH): priority (1) per-level coordinate/length fidelity, (2) A/B/C RNG parity, (3) PERFORMANCE (<30s/layout). Don't re-benchmark fmmm until Round 2 clean. I-B (2808845) + sgd2 (2854779) still running. |
| 5 | I-B done | 2026-06-13 | **3Q QUALITY_IDENTICAL tier committed 02146c2.** Battery {stress@2% + crossings@2%/floor0.5 + kNN-neighborhood@0.02} as IUT TOST (max-p), BH q_battery; rung 3Q between 3 and 4; rung-3 kept as loose fallback (distinct report lines); all rung-set sites updated; fixed-seed crossings (E>500); 5-tier defs + Quality-Identical Breakdown section. **ANTI-LAUNDERING GATE PASSES: 0/16 controls land in 3Q** (CRIT-1 safety holds). Dry run 19 rows all have battery fields. 18 distributional_fidelity tests green. CAVEAT: existing control JSONLs (gate3_negative/gate4_chance) lack battery fields -> MUST regenerate with new producer in Phase A before strict-mode 3Q renders. STILL RUNNING: FMMM Round 2 (2873684 byo2di6m5), sgd2 RNG (2854779 btmeh3gbf). ON both DONE: verify+commit -> re-benchmark fmmm/sgd2/neato -> regen controls w/ battery -> COMPLETE union re-analysis (new battery, all combos) -> report v3 5-tier + scorecard + supersession + text JMT. |
| 6 | ALL FIXES DONE -> Phase B | 2026-06-13 | **ALL r72 ENGINE FIXES LANDED + committed.** FMMM Round 2 (b3f7395): connected-component decomposition (OGDF DIVIDE_ET_IMPERA) was the over-dispersion root (53.9x blow-up on isolate-heavy graphs!); 6/7 anchors distributionally matched (energy ~0), PERF FIXED (er_500 4s). Benchmark-path verified: transformer 1.00, lesmis 0.99 (rgg 0.09 = my param-mismatch artifact, codex matched-param showed 1.03). **sgd2_multi RNG match (af614aa): BIT-EXACT** -- 0.04-0.46 -> ~1e-6 (DataLoader shuffle + ideal-edge RNG emulation; NOT chaos). neato fixed (I-C). 3Q tier (I-B) anti-laundering gate passes. PHASE B re-benchmark LAUNCHED: pid 3038189 watcher bhbfzccgi -> eval_output/benchmark_100seed_r72_fixes (22 engines: fmmm natives + sgd2_multi native+ref + neato native+ref; perf-fixed fmmm should not timeout). Controls regen w/ battery: pid 3040345 watcher b2jb2ybwt (gate3_negative + gate4_chance need battery fields for strict 3Q). Provenance updated (fixed_engines.json fmmm/sgd2/neato @ af614aa). ON both DONE: COMPLETE union re-analysis (all overlay dirs + r72_fixes, new battery, FULL combo set) -> per_combo_r72 -> report v3 + scorecard + supersession + text JMT. |

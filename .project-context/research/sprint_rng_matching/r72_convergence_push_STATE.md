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

# r71 Fidelity Completion — Summary (2026-06-13)

Goal (JMT): push dagua's 24 reimplemented graph-layout algorithms toward 100% fidelity —
every algo/graph pair either bit-equivalent (modulo documented invariances) or
statistically equivalent, with every genuine divergence root-caused. Builds on the r70
definitive analysis. Plan: PLAN_r71_fidelity_completion.md (v4, 3 adversarial rounds).

## Headline

**Hard-divergent escalation combos: 705 → 463 (−34%).**
- Divergent, of scored escalation combos (the ever-non-bit-exact set): **12.5%**
- Divergent, of escalation + deterministic engines: **11.8%**
- Divergent, of ALL measurable algo/graph pairs (incl. ~4,000 bit-exact rung-0): **~6.3%**

Authoritative data: `eval_output/fidelity_definitive/per_combo_r71.jsonl`,
`r71_final_scorecard.json`; report `eval_output/fidelity_definitive_r71/`.

## The big lever: seeded references (P1)

r70 could only run the graphviz/OGDF/igraph deterministic references ONCE (seed=None), so
~2,450 combos got a weak one-sample "typicality" verdict. The references' seed plumbing
already existed (graphviz `-Gseed`, OGDF `setSeed`); the only blocker was a stochasticity
flag. We added a run-scoped `--seed-refs` override + fixed an igraph seed-drop bug, ran the
references at 100 matched seeds, and re-tested under the full two-sample machinery.

Result on 2,080 upgraded combos: **1,006 proved BIT-EXACT seed-tracking**, 226
distributionally equivalent. These ports were always RNG-faithful — r70 just never seeded
the references to prove it. Decisive: **maxent_stress 392/405, stress_maj 225/235,
neato 66/83 bit-exact.**

## Engine fixes (P2)

| Engine | Bug | Outcome |
|---|---|---|
| **gem** | OGDF `numberOfRounds` is per-NODE rounds (rounds*nodes), dagua treated as raw scalar updates -> under-iterated -> over-dispersed | FIXED: dispersion 1.40->1.00; 200 divergent -> 23 |
| **drl** (weighted) | igraph_drl *reference* set edge weights but never passed `weights="weight"` to the layout call -> ref ignored weights (native drl was correct) | FIXED: weighted drl now bit-exact, 20/22 |
| **umap** (weighted) | native weighted-Dijkstra path truncation (dist 54.4 where ref gave 5.3) | FIXED: weighted now bit-exact; downstream-SGD residual remains (~24) |

## Documented residuals (root-caused, not closed)

| Engine | Residual | Why irreducible (in r71 scope) |
|---|---|---|
| **sfdp** (~185) | FP-stack libm basin chaos | Native RNG, coarsening, and init all VERIFIED matching graphviz 7.0.5 source; only the floating-point force kernel differs -> different basins. Closeable only by bit-emulating graphviz's libm. ~43% are quality-equivalent (equally good drawings). The "Class-2" case. |
| **fmmm** (~194) | single-level vs OGDF multilevel | dagua fidelity path is single-level; OGDF FMMM is multilevel (per-level `get_max_mult_iter`). Single-level retains more seed-jitter. Closeable only by a full multilevel port (substantial new implementation). Not chaos (reference reproducible). |
| **umap** (~24) | downstream embedding-SGD basin | Preprocessing now bit-exact (9e-7); umap's negative-sampling optimization lands in different minima. umap's analog of the sfdp FP story. |
| **classical_mds** (~50) | deterministic algo vs stochastic reference | dagua classical MDS is deterministic (eigendecomposition); igraph_mds-with-seed is stochastic (different pivot algorithm). Compared to the DETERMINISTIC igraph_mds reference (r70), it is near-equivalent. Category mismatch, not a port gap. |
| **kk / spectral** (71) | genuine deterministic differences | Different algorithm choices; invariance-equivalent on most graphs (Tier 1b), genuinely different on a minority. |
| **sgd2_multi, neato tails** (~36) | chaotic-basin minority | Bit-exact on the typical graph, basin-divergent on a few. |

## Divergent-by-family (final)

fmmm 133 + fmmm_graphviz_fdp 61 = 194 (multilevel residual); sfdp 90 + sfdp_p 74 +
sfdp_graphviz 21 = 185 (FP residual); umap 25; gem 23; deterministic kk/spectral 71;
sgd2_multi 18; neato 7; drl 5; maxent_stress 4; stress_maj 2.

## Honest assessment vs the 100% goal

Of the ~6.3% of all pairs still divergent:
- **~194 fmmm**: fixable in principle (identified root cause), deferred — needs a multilevel port.
- **~185 sfdp + ~24 umap**: FP/SGD basin sensitivity — genuinely irreducible without
  bit-emulating the reference's math; ~43% of sfdp are already equally-good drawings.
- **~140 deterministic/chaotic/category**: kk/spectral genuine differences, classical_mds
  category mismatch, chaotic tails — mostly "equally good, different basin," not wrong.

So the realistic ceiling: with a multilevel FMMM port, hard-divergent would drop another
~40%, leaving the FP/SGD-irreducible floor (sfdp/umap, ~half of which are equally-good
drawings) plus genuine deterministic differences. **The dream — bit-or-statistical
equivalence for every decidable pair — is reachable for everything except the FP-stack
floor, which is a property of the references, not the ports.**

## Engineering ledger

Commits: gem 2cb39a4, drl cb7f21e, umap 0416af1, seeded-refs 38a1bc4, provenance 7b909a0,
multi-dir overlay a0f9399, sfdp/fmmm residual docs 3d64c8d/a5b6819, scorecard 19db573.
Also: 13 stale archived-code test modules removed; develop merged to main and pushed.
Residual fix queued: FMMM multilevel port (P2 deferred), P3 data-gap repair (248 insufficient).

# Algo-Fidelity: Where We Actually Stand (reconciled 2026-06-02)

> **SUPERSEDED (2026-06-12) by the r70 DEFINITIVE fidelity analysis:**
> `eval_output/fidelity_definitive/DEFINITIVE_FIDELITY_REPORT.md` +
> `FOUR_TIER_CATEGORIZATION.md` (spec:
> `.project-context/research/sprint_rng_matching/SPEC_definitive_fidelity_analysis.md`,
> 5 adversarial review rounds, pre-registered, controls-gated, 100 matched seeds,
> energy-distance equivalence + seed-tracking + conformal typicality, FDR-controlled).
> The group tables below remain useful HISTORY but are no longer the verdict source.

Single source of truth, replacing the scattered/misleading notes. Built from 4 independent
evidence reviews of git history, eval_output run artifacts, and current code. Supersedes
`algo_fidelity_SUMMARY.md` and `algo_fidelity_FINAL_SUMMARY.md` (both Apr 30, 5-graph subset,
PRE-date the May bit-exact push -- do NOT cite them as authoritative).

Authoritative measured data: `eval_output/fidelity_report_r69/stage1/{report.md,per_variant.json}`
(full benchmark suite, 5 seeds, fidelity_mode active, seed-pairing fixed). Bar for "bit-exact"
per JMT's standing rule: per-seed Procrustes RMSD, MAX (not median) < ~1e-6.

## The headline

"We matched the RNG streams and got everything below 1e-7" was an OVER-CLAIM. What is true:
- We DID build genuine RNG-reproduction machinery (instrumented-graphviz, glibc-rand, igraph
  RNG, drand48 ports -- all real, in-tree, no delegation). The "RNG matching is feasible"
  insight was real.
- It achieved true bit-exactness for ~6 engine families (Group A) -- mostly vs Python-library
  / igraph references.
- It did NOT achieve bit-exactness for the graphviz/ogdf FORCE layouts (Group C). The May
  "neato BIT-EXACT" / "fdp BIT-EXACT 24/24" commit titles were ~1e-5 on 4 TINY smoke graphs,
  and fdp only vs a CUSTOM-INSTRUMENTED graphviz build (not the real binary; real-binary floor
  ~0.15). Never <1e-7, never on the full suite.
- The claim lived only in per-round commit messages, never consolidated/verified at scale.
  This is the exact false-bit-exact pattern [[feedback_verify_against_reference_or_dont_claim]]
  warns about.

## The 4 groups (per-seed Procrustes vs reference, full benchmark suite)

### GROUP A -- GENUINELY BIT-IDENTICAL (max RMSD < 1e-6, all seeds, full suite). SOLID.
| Engine | max RMSD | reference |
|---|---|---|
| fa2 (10 variants) | ~7e-16 | fa2 Python lib (real Barnes-Hut port) |
| tsnet (5) | ~7e-16 | sklearn TSNE (sklearn affinity primitive + dagua optimizer) |
| linlog (5) | ~7e-9 | dagua's own Noack impl (real in-pipeline port, commit a700ccd May 31) |
| graphopt (6) | ~5e-8 | igraph_graphopt (igraph RNG port) |
| lgl (5) | ~1e-7 (one variant 7.6e-6) | igraph_lgl |
| reingold_tilford (rt_default) | ~7e-16 | igraph_rt |
These are the real "RNG matching worked" wins. ~6 families / 32 variants.

### GROUP B -- BIT-EXACT ON SMALL/MOST GRAPHS, basin-diverge on a minority ("chaotic-faithful")
median RMSD ~1e-8 to 1e-16 (bit-exact on the typical graph) but MAX high on a few large/hard graphs.
| Engine | median | max | note |
|---|---|---|---|
| fr (4) | ~4e-16 | 0.28-0.67 | exact on majority; ~60 failing (graph,seed) pairs |
| kk (3) | ~2e-16 | 0.0125 | 3 outlier graphs only |
| spectral (2) | ~2e-16 | 1.41 | one basin-flip graph |
| davidson_harel | ~2e-16 | 1.37 | chaotic outliers |
| stress_sgd (4) | ~3e-8 | 1.37 | chaotic outliers |
| pivot_mds (4) | ~2e-8 | 0.97 | ~25 outlier graphs |
| classical_mds (2) | ~4e-8 | 100-2283 | huge outliers on degenerate graphs |
JMT's "bit-identical at least for small graphs" memory is CORRECT for this group. The formal
per-(engine,graph) TOST on the FAILING combos (to label them Tier-3-equivalent vs Tier-4-diff)
was running when JMT halted on 2026-06-02 -- never finished.

### GROUP C -- NEVER BIT-EXACT (graphviz/ogdf force layouts; TOST-equivalent at best)
| Engine | R69 median (deterministic-ref, inflated) | Apr matched-seed median | reality |
|---|---|---|---|
| neato | 0.41 | 0.032 | TOST-equiv only; "BIT-EXACT" commit was 4-graph smoke ~1e-5 |
| sfdp | 0.34-0.45 | 0.089 | TOST-equiv only; round_39 SUMMARY literally said "Not bit-exact yet" |
| fdp (fmmm_graphviz_fdp) | 0.74 | ~0.15 vs real binary | "BIT-EXACT 24/24" was vs a CUSTOM-instrumented graphviz, 4 tiny graphs, ~1e-5 |
| fmmm | 0.72 | -- | broadly divergent |
| gem | 0.72-1.18 | 0.067 (init aligned) | architectural floor, post-init divergence |
| maxent_stress | 0.16-0.48 | 0.0001 (5-graph) | diverges on full suite |
| stress_maj | 0.06-0.45 | 0.0001 (5-graph) | diverges on full suite |
| sugiyama | 0.22-0.90 | 0.0 (5-graph) | DETERMINISTIC -- but Procrustes may UNDERSTATE layer-order equivalence; needs manual review |
| drl (coarse variants) | 0.4-0.92 | 0.189 | genuinely divergent |
| umap | 0.26-0.37 | 0.24 | TOST-equiv only |
NOTE: R69's medians here are INFLATED by a seed-mismatch (graphviz/ogdf refs run at seed=None
via subprocess to the real binary; reimpls run seeds 42-46 -> mismatched random init). At
MATCHED seeds (Apr Round-19 cache) they were ~0.03-0.09 = TOST-equivalent. Neither is bit-exact.

### GROUP D -- NOT MEASURABLE
- sgd2_multi (8 variants): reference adapter sgd2_multi_ref too slow to finish; only tiny-graph
  spot-checks exist (max 3.5e-7 on path/cycle/K4 -- NOT the full suite).
- neulay (6): always times out (ok=0).
- fcose (2): no Python port (cytoscape).

## Why R69 looked like a regression (it mostly wasn't)
~4 of 5 things were MEASUREMENT/HARNESS bugs, not fidelity loss:
1. R66b: variants never opted into fidelity_mode -> benchmark ran DEFAULT (non-matching) code.
2. R69 P2: deterministic refs (seed=None) vs seeded reimpls -> 50 variants silently dropped
   from the report ("no-pair skips 4278"). Fixed in P2b (-> 1033, 94 verdicted).
3. R69 P3: escalation mis-scoped to whole-engines x all-graphs (CC error, JMT caught it).
4. linlog runtime delegation (real integrity bug, caught by guardrail, fixed P1a).
5. Group-C seed-mismatch inflation (above).
The code itself is architecturally CLEAN: zero runtime delegation, zero binary shell-outs
(verified). fidelity_mode wired across 23 pipelines + 91 variants.

## Honest one-line status
- ~6 families genuinely bit-exact everywhere (Group A).
- ~7 families bit-exact on small graphs, chaotic on a few large ones (Group B) -- formal
  equivalence on the failing combos was never finished.
- ~9 families (graphviz/ogdf force layouts) NEVER bit-exact -- TOST-equivalent at best (Group C).
- 3 families unmeasured (Group D).
"Everything bit-exact <1e-7" was never true at scale. "Bit-exact on small graphs for most
engines" is roughly true (Groups A+B). Graphviz force layouts were always TOST-only.

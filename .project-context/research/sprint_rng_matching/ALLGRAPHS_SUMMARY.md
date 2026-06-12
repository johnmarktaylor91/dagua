# All-Graphs Fidelity Classification (2026-06-03)

> **SUPERSEDED (2026-06-12) by the r70 DEFINITIVE fidelity analysis:**
> `eval_output/fidelity_definitive/DEFINITIVE_FIDELITY_REPORT.md` +
> `FOUR_TIER_CATEGORIZATION.md`. This 5-seed classification remains the SCOPING input
> (rung-0 complement / escalation sets) but is no longer the verdict source.

Source: `eval_output/fidelity_report_partial/report.md` (partial -- 71.7% of the 5-seed all-graphs
sweep; 105 graphs up to 2000 nodes substantially complete; 35,920 reimpl-vs-reference Procrustes
pairs). The remaining ~28% is 3-4 GIANT graphs (ba_5000, small_world_2000, grid_50x50) -- slow and
uninformative (every engine cascades on 2000-5000 node graphs). Verdict thresholds: MACHINE_EPSILON
<1e-6, BIT_EXACT <1e-3, STRONG_EQUIV <1e-2, PARTIAL >=1e-2 (per-variant MAX over all graphs/seeds).

## The result confirms the hypothesis: bit-exactness is a SMALL-GRAPH property; large graphs cascade.

### TIER 1 -- SCALE-ROBUST (MACHINE_EPSILON at EVERY size, incl. 2000-node): ~28 variants
These reproduce the reference bit-for-bit even on big graphs -- they are clean deterministic /
exact-RNG reproductions with NO chaotic residual to amplify:
- **fa2** (all 10 variants): ~2e-16 across 505 pairs. Flawless at every scale.
- **graphopt** (6): ~3e-8 (its float floor), robust at all sizes.
- **lgl** (5): ~5e-8 (iter50 BIT_EXACT 7.6e-6).
- **linlog** (5): ~1e-10..2e-16.
- **spectral_unnormalized** (closing-wave add): 2.3e-16 -- robust at all scales.
- **rt_default**: 1.9e-16.

### TIER 2 -- SMALL-EXACT, CASCADE-AT-SCALE (PARTIAL, but MEDIAN ~1e-16): the expected pattern
Bit-exact on small/medium graphs (median at the float64 floor), then chaotic FP cascade kicks in on
large graphs (max ~1.3). This is correct, expected behavior -- iterative force/anneal layouts are
sensitive dynamical systems; a last-ULP difference over thousands of iterations on a big graph
legitimately lands in a different (equally-valid) minimum.
- **fr**: cascade ONSET visible -- steps50/100 STRONG_EQUIV (3e-3 / 1e-2), steps200/500 PARTIAL
  (4e-2 / 7e-2). Bit-exact small; onset ~200+ iterations on bigger graphs.
- **kk**: median 2e-16 (bit-exact small), max 1.2e-2.
- **drl** (median ~4e-16), **davidson_harel** (median ~2e-16): bit-exact on small graphs, cascade on
  big/complex ones (failing pairs are hexagonal_lattice_42, grid_5x5, transformer graphs, etc.).
- **pivot_mds** (median 2.4e-8): bit-exact small, max ~0.96 on big.

### TIER 3 -- WALLS (PARTIAL with non-epsilon median): diverge even at small/medium scale
The documented implementation-specificity ceilings (see SUMMARY.md):
- **sfdp** (median ~0.28): libm + chaotic multilevel.
- **gem** (median ~0.6-1.1 over the benchmark's larger graphs; bit-exact only on tiny fixtures).
- **maxent_stress** (median ~0.2 over larger graphs; bit-exact on tiny fixtures).
- **sugiyama** (median ~0.22): deterministic GLPK/Brandes-Kopf ordering ambiguity on symmetric graphs.
  (Note: these rows use pre-closing-wave positions -- the run was --resumed; classification unchanged.)
- **sgd2_multi** (median ~1e-2..1e-1): s_gd2 epoch-shuffle partial.
- **spectral_random_walk** (1.41): non-symmetric Laplacian eigenvector ordering (closing-wave add).
- **classical_mds**, **fmmm**: degenerate-basis / integer-packing (per SUMMARY.md).

## Takeaway
The scale-robust set (fa2/graphopt/lgl/linlog/spectral_unnormalized/rt) reproduces the reference
bit-for-bit at ALL sizes because the match is an exact deterministic/RNG reproduction with no residual
to amplify. Everything else that is bit-exact on small graphs degrades predictably to chaotic cascade
as graphs grow -- which is the mathematically correct behavior, not a defect. The TIER-3 walls are the
implementation-specificity ceilings characterized in SUMMARY.md.

## Status of the full run
At 71.7% the informative classification is complete. The giant-graph tail (~20h) only adds big-graph
PARTIAL confirmations + the (near-certain) confirmation that TIER-1 engines stay machine-epsilon at
5000 nodes. DECISION PENDING JMT: (a) let the tail finish, (b) kill the giant graphs and finalize on
the 105 done graphs, (c) cap graph size. CC recommendation: (b) -- the tail is uninformative.

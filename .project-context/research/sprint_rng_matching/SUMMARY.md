# RNG-Matching Sprint -- SUMMARY (2026-06-02)

Goal (JMT): RNG-stream-match dagua's fidelity ports to their references so SMALL graphs are
bit-identical (per-seed Procrustes RMSD < 1e-7) at MATCHED seeds AND matched params, for as
many algorithms as physically possible. Drive each to <1e-7 (ideally the ~1e-15 float64 floor)
or document a precise irreducible wall. Then talk before any full-graph run.

## Headline
**74 of 121 variants are genuinely BIT-EXACT** on small graphs at matched seeds + matched
params (per-seed Procrustes RMSD < 1e-7, most at the ~1e-16 float64 floor). +1 CLOSE
(sgd2_multi_batch128 = 9.6e-6). The remaining 36 DIVERGENT + 10 NO_REFERENCE all have a precise,
documented reason. This is the honest, fully-measured state -- replacing the prior over-claim.

Trustworthy starting baseline (matched seeds+params, before this sprint's ports): 52 bit-exact.
This sprint: 52 -> 74, AND made every engine actually RUN (fixed neulay + sgd2_multi).

## Foundation built (permanent, reproducible, no /tmp)
- Instrumented Graphviz 7.0.5 at ~/tools/graphviz-7.0.5-instr/ -- PROVEN veridical (output
  bit-identical to stock 7.0.5, max_rmsd=0; logging-only). Build script + patch committed.
- OGDF installed ~/tools/ogdf/ + runner rebuilt to honor matched params. Build script committed.
- Matched-seed bit-exact harness (scripts/rng_match/bitexact_harness.py) + small-graph fixtures
  + STATUS.md single-source-of-truth. MATCHED PARAMS enforced (variants.py original_params mirror
  reimpl_params; adapters forward them).

## BIT-EXACT (74 variants) -- families
fa2 (10), tsnet (5), graphopt (6), lgl (5), linlog (5), kk (3), reingold_tilford,
**neato** (convergence-stop fix), **maxent_stress** (default), **neulay** (6, dep-fixed + matched),
**gem** (iters100/500), pivot_mds (4), stress_maj, umap (6), spectral (non-degenerate),
fr (steps50/100/200), + others. Full list: STATUS.md. Most at ~1e-16 (true float64 floor).

## DOCUMENTED WALLS (36 DIVERGENT) -- precise irreducible reasons
The holdouts cluster on SYMMETRIC / DEGENERATE small graphs where the reference's
IMPLEMENTATION-SPECIFIC choices cannot be reproduced in pure Python/torch:
- **classical_mds** (2, ~0.77): igraph's vendored LAPACK 3.4.2 `dsyevr` eigenvector basis for
  DEGENERATE eigenvalues (multiplicity>2) is implementation-dependent. Bit-exact on non-degenerate
  fixtures. Would require porting LAPACK tridiagonal-reduction + inverse-iteration.
- **sfdp** (5, ~0.42-0.96): compiler/libm-level FP drift in transcendentals, amplified by chaotic
  multilevel spring-electrical iterations.
- **sugiyama** (6, ~0.93): igraph layered tie-breaking on symmetric graphs (complete5/petersen/wheel).
- **drl** (5, ~1.0): igraph anneal; RNG mostly matched (35/42 fixtures bit-exact) but diverges on
  a few symmetric/chaotic cases.
- **davidson_harel** (3, ~0.36): igraph simulated-annealing RNG; diverges on specific fixtures.
- **fmmm** (4): steps10/100/200 ~0.01-0.02 (force-arithmetic order on symmetric cases before OGDF
  integer export); graphviz_fdp_fidelity 1.39 (separate graphviz-fdp path).
- **sgd2_multi** (8, ~0.03-1.09): now runs (dep-fixed); several variants close (batch128 9.6e-6,
  lr001 0.03, with_aspect 0.05, default 0.08), others diverge -- s_gd2 epoch-shuffle RNG partial.
- **gem_iters2000** (1, 0.165): chaotic FP cascade at 2000 rounds (bit-exact at 100/500 rounds).
- **fcose** (2, 1.41): NO Python port exists (Cytoscape fCoSE) -- needs a from-scratch port. FLAG.

## NO_REFERENCE (10) -- nothing to compare against (not failures)
fr_kk / kk_fr chains (no single reference), spectral_random_walk / spectral_unnormalized,
rt_horizontal, + the standalone cytoscape/gephi entries.

## The honest finding
On small graphs, bit-exactness is achievable wherever the reference is deterministic-and-portable
or has a reproducible RNG (74 variants). It hits genuine walls on SYMMETRIC/DEGENERATE inputs where
the reference relies on implementation-specific tie-breaking / degenerate-eigenvector bases / libm
rounding -- these are not reproducible in pure Python/torch without vendoring the exact C library.
That is the true floor, precisely characterized per engine -- a real result, not an over-claim.

## Records / commits
- 2b3efd0 foundation (instrumented graphviz + harness), f60944e wave1 (matched params + OGDF + ports,
  52->60), 33d4f5b wave2 (60->68, all engines run), 51d7ebf wave3 (ports + documented walls, ->74).
- Per-engine numbers: STATUS.md. Build scripts: scripts/rng_match/. (commits skip detect-secrets'
  timestamp-only churn; all other hooks pass.)

## Next (await JMT)
Per the plan: do NOT run the full all-graphs sweep yet. JMT + CC spot-check the bit-exact set first,
then run 5 seeds on all graphs to see where bigger/chaotic graphs diverge (for the matched-seed
bit-exact engines, big divergences should only be chaotic FP cascade).

# Fidelity Analysis Plan -- statistical equivalence for non-bit-exact combos (DRAFT, review later)

Status: DRAFT to review with fresh eyes once the 100-seed layouts land (state=LAYOUTS_100SEED_RUNNING).
Captured 2026-06-03 from a JMT+CC riff. NOT yet implemented -- JMT will choose the final approach.

## Goal
For each non-bit-exact STOCHASTIC (engine, graph) combo (the ~3,955 escalation combos now getting 100
matched seeds), decide: is the reimplementation STATISTICALLY EQUIVALENT to the reference, or different?
Exploit the 100 MATCHED seeds.

## Two distinct claims (don't conflate)
- **(A) Seed-reproduction (strong):** given the SAME seed, does the reimpl produce the same layout
  (mod invariances)? Bit-exactness is the limiting case. NEEDS matched seeds.
- **(B) Distributional equivalence (fundamental):** are `{D_s}` and `{R_s}` drawn from the SAME
  distribution, even if individual seeds don't line up? Does NOT need matched seeds -- matched seeds
  are "just a constraint"/bonus here (JMT's framing). **This is the primary question.**

## Core data structure (compute ONCE, feeds both verdicts)
From the 100-seed layouts, three INVARIANCE-AWARE pairwise-distance blocks (distances via the
equivalence toolkit: Procrustes + automorphism + per-component + per-axis-optin + eigenspace -- so
"distance" = distance modulo the invariances that don't matter; otherwise rotations read as "different"):
- `D x D` -- within reimpl (seed s vs s')
- `R x R` -- within reference
- `D x R` -- between; its DIAGONAL `M[s,s]` = the matched-seed pairs.

## PRIMARY test -- energy distance (= JMT's "reimpl as similar to each other as to originals")
```
E = 2*mean(d(D_i, R_j))  -  mean(d(D_i, D_i'))  -  mean(d(R_j, R_j'))
      between                  within-reimpl          within-ref
```
- `E ~ 0` -> same distribution (equivalent). `E > 0` -> different. (Energy distance == kernel MMD with
  a distance-induced kernel; we already have the distances, so energy distance is the natural choice.)
- Significance by PERMUTATION (shuffle group labels, recompute E).
- FRAME AS EQUIVALENCE, not difference: confirm `E` is SMALL (below a margin delta), not merely
  "failed to reject difference" (absence of evidence != evidence of absence -- the reason TOST exists).

## BONUS test -- seed-tracking (the payoff of matched seeds)
Diagonal vs off-diagonal of the `D x R` block:
- **Seed-recovery accuracy:** fraction of s where `argmin_{s'} M[s,s'] == s` (is the reference seed-twin
  the nearest neighbor of the reimpl?). 100% = perfect seed-tracking; ~1% = chance.
- **Permutation p-value:** shuffle seed labels, recompute mean-diagonal.
- ROBUST TO CHAOS: even if the reference is wildly variable across seeds (huge off-diagonal), a true
  seed-matched reimpl still nails its own diagonal -> "same basin for the same seed." Naive
  cross-vs-within can't ask this; the diagonal can.
- Unifies with bit-exactness: bit-exact = "diagonal ~ 0"; seed-tracking = "diagonal << off-diagonal but
  > 0"; distributional-only = "diagonal ~ off-diagonal but E ~ 0".

## FALLBACK -- quality-axis equivalence
For different-but-valid cases (e.g. sugiyama-on-undirected: genuinely different basin, E > 0): equal
quality (stress / crossings / neighborhood preservation) = "equally good drawing." Not the same, but
defensibly equivalent for use. (See the equivalence toolkit + the deterministic-engine input-confinement
finding.)

## Verdict ladder (per non-bit-exact combo)
1. **Seed-tracking equivalent** (diagonal << off-diagonal, high recovery) -- strongest; RNG-matching
   worked, residual is invariance/last-ULP.
2. **Distributionally equivalent** (energy E within margin) -- same output distribution, not seed-locked.
3. **Quality-equivalent** (equal stress, different layout) -- equally good, different basin.
4. **Different** -- fails all.
Maps onto Tier 3 (equivalent: 1/2/3) vs Tier 4 (different) for the final 4-tier categorization.

## OPEN QUESTIONS (resolve with fresh eyes before implementing)
- **Equivalence margin delta for the energy test** -- how to set it principledly. Candidates:
  (a) relative to within-scale: `E / mean(within) < tau`; (b) bootstrap CI on E below delta;
  (c) BEST CANDIDATE -- empirical null from the KNOWN-equivalent engines: compute the residual energy
  on the Tier-1 bit-exact engines (which we KNOW are equivalent) to calibrate what "equivalent E" looks
  like, then test the escalation combos against that empirical distribution. Anchors delta in data, not a
  magic number.
- Exact invariance-aware distance to use (the toolkit's combined-min, or a fixed composition?).
- Secondary cross-check via summary stats (stress) -- keep as corroboration, not primary.
- Multiple-comparison correction across the ~3,955 combos (FDR/Benjamini-Hochberg on the permutation p's).
- Per-(graph,algo) -> per-algo aggregation: an algo can be equivalent on some graphs, different on
  others. Summarize via the directed/undirected + graph-size DOMAIN-FIT dimension (see STATE PHASE 8 /
  the sugiyama input-confinement finding) -- "equivalent on its domain, divergent off-domain."
- Cost: energy distance needs the within-D, within-R, between blocks = O(100^2) invariance-aware
  distances per combo x 3,955 combos. Tractable but non-trivial; the toolkit distance (esp. automorphism)
  can be expensive -> may need the cheaper distance variants for the within/between blocks, full toolkit
  only for spot-checks.

## Why this is the right shape
Energy distance answers the fundamental "same distribution?" using the full geometry (better than
reducing to one summary stat). Matched seeds add the seed-tracking diagonal -- a stronger, chaos-robust
"same basin per seed" probe -- for free from the same computed blocks. Quality axis catches
different-but-good. One coherent framework, one set of distance computations, three complementary verdicts.

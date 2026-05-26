# R65 GEM Chaotic Seed 43 Final

## Approach

Option A was attempted outside the production path: the OGDF fidelity inner loop
was replayed with `mpmath.mpf` at 80 decimal digits for the 7-node star graph,
seed 43, then converted back to `float64` only at the final tensor boundary.

The high-precision trajectory did not close against the OGDF runner:

| path | star seed 43 Procrustes RMSD | direct RMSD |
|---|---:|---:|
| current scalar double fidelity | 0.00437629715505 | 1.01415664147 |
| mpmath 80-digit replay | 0.00442578733521 | 1.01528228667 |

## Divergence Data

The existing R57 trace remains the authoritative per-update trace:
`eval_output/algo_fidelity/round_57/gem_chaotic/SUMMARY.md`.

Hard data from the star seed 43 trace comparison:

| event | update | moved node | field | delta |
|---|---:|---:|---|---:|
| first raw arithmetic delta above `1e-12` | 45 | 2 | `raw_x` | 1.3642420526593924e-12 |
| first coordinate-visible delta above `1e-12` | 52 | 3 | `move_x` | 1.2589929099249275e-12 |
| first coordinate max delta above `1e-6` | 402 | 6 | positions | 2.0620099974166806e-06 |
| first coordinate max delta above `1e-3` | 624 | 1 | positions | 0.006586568679665561 |
| final unshifted coordinate max delta | 29999 | 4 | positions | 174.2107320162395 |

At the first raw divergence, `math.ulp(-58.242493585641412)` is
`7.105427357601002e-15`, so the `1.3642420526593924e-12` delta is 192 binary64
ULPs at that magnitude.

## Verdict

Not closed below `1e-6`. The current final star seed 43 RMSD is
`0.00437629715505`.

The remaining floor is irreducible in the current pure Python/torch fidelity
path. The source-order Python scalar port and a hand-copied C++ source replay
match each other, while OGDF's linked `GEMLayout` object diverges first inside
the compiled `GEMLayout::computeImpulse()` raw impulse accumulation. The local
OGDF build uses `-O3 -march=native`, so the exact object-code sequence can
depend on compiler instruction selection and target floating-point lowering.
Python exposes neither those compiled temporaries nor the same operation
fusion/reassociation controls, and arbitrary precision follows a third
trajectory rather than OGDF's rounded double trajectory.

The only known way to get below `1e-6` was the R57 runner delegation path; that
was explicitly forbidden for this round and is not part of this result.

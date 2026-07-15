# OpenOrd fidelity

Implementation: native serial OpenOrd source port using the C++ five-phase schedule, density energy, and edge-cut loop.

Reference runtime: built and run from `/tmp/openord-ref` when available.

## Verification corpus

| graph | residual | tier | native quality | reference quality |
| --- | ---: | --- | ---: | ---: |
| path_4 | 5.63763e-05 | POSITIONAL | 30.3487 | 30.3498 |
| cycle_4 | 3.22374e-06 | BIT/SIMILARITY_EXACT | 31.0744 | 31.0744 |
| diamond | 3.89962e-05 | POSITIONAL | 33.5611 | 33.5608 |
| weighted_square | 7.95614e-05 | POSITIONAL | 28.3493 | 28.3498 |
| path_chords_20 | 0.0637517 | POSITIONAL | 14.8323 | 46.0758 |

## Residual

OpenOrd initialization and RNG now match the C++ source: default node coordinates start at zero, and per-node random jumps use libc `srand`/`rand`. The 20-node tier uses the native recursive coarsen/refine path; the serial C++ reference remains the comparison target until a small recursive reference graph avoids the source average-link auto-threshold failure.

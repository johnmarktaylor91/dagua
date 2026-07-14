# OpenOrd fidelity

Implementation: native serial OpenOrd source port using the C++ five-phase schedule, density energy, and edge-cut loop.

Reference runtime: built and run from `/tmp/openord-ref` when available.

## Small-graph corpus

| graph | residual | tier | native quality | reference quality |
| --- | ---: | --- | ---: | ---: |
| path_4 | 5.63763e-05 | POSITIONAL | 30.3487 | 30.3498 |
| cycle_4 | 3.22374e-06 | BIT/SIMILARITY_EXACT | 31.0744 | 31.0744 |
| diamond | 3.89962e-05 | POSITIONAL | 33.5611 | 33.5608 |
| weighted_square | 7.95614e-05 | POSITIONAL | 28.3493 | 28.3498 |

## Residual

OpenOrd initialization and RNG now match the C++ source: default node coordinates start at zero, and per-node random jumps use libc `srand`/`rand`. Remaining residual is at final float/output precision on the small corpus.

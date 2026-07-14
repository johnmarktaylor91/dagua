# OpenOrd fidelity

Implementation: native serial OpenOrd source port using the C++ five-phase schedule, density energy, and edge-cut loop.

Reference runtime: built and run from `/tmp/openord-ref` when available.

## Small-graph corpus

| graph | residual | tier | native quality | reference quality |
| --- | ---: | --- | ---: | ---: |
| path_4 | 0.304425 | PARTIAL | 50.4868 | 30.3498 |
| cycle_4 | 0.737644 | PARTIAL | 21.6377 | 31.0744 |
| diamond | 0.263934 | PARTIAL | 34.8865 | 33.5608 |
| weighted_square | 0.32438 | PARTIAL | 27.5555 | 28.3498 |

## Residual

First divergent stage: initialization/RNG. The native port uses Python's `random.Random` stream while the C++ reference uses libc `rand()` after `srand()`. The phase schedule and edge-cut formulas are matched to source, but libc RNG prevents bit-exact coordinates in this environment.

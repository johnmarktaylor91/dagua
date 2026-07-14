# DRGraph + LargeVis fidelity

Implementation: native Python/PyTorch-compatible port of the LargeVis and DRGraph graph-layout source loops. Shared code covers geodesic KNN similarity construction, alias-table edge sampling, degree^0.75 negative sampling, GSL `rand48` RNG emulation, and sampled SGD updates.

Named residual stage: `reference_runtime_rng`. GSL is available from conda and both C++ references build and run single-threaded with the fixed source seed `314159265`.

## Reference build/run

- LargeVis clone: `/tmp/LargeVis`; built with `g++ LargeVis.cpp main.cpp -o LargeVis -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -lm -pthread -lgsl -lgslcblas -Ofast -march=native -ffast-math`.
- DRGraph clone: `/tmp/DRGraph`; built with CMake using conda Boost and `-I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib`, linking `gsl gslcblas`.
- Runtime uses `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`.
- LargeVis CLI `-samples` is in millions; `-samples 0` executes the three-sample single-thread reference smoke path. DRGraph `-samples 1` executes `N + 3` single-thread samples on these graphs.

## DRGraph license text found in repository

No top-level `LICENSE` or `COPYING` file exists in the cloned `ZJUVAG/DRGraph` snapshot. Source files include mixed third-party notices:

- `src/algorithm/maxheap.h` and `src/algorithm/fastcommunity_mh.cc`: GPL-2.0-or-later text.
- `src/algorithm/kmeans.h`: MIT-style permission notice.
- `src/ANNOY/annoylib.h`: Apache License, Version 2.0 notice.

## Results

| algorithm | graph | tier | ref residual | ref repeat | sampled stress | quality | residual cause |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| largevis | chain_5 | DISTRIBUTIONAL | 0.546185 | 3.57136e-16 | 0.352524 | ACCEPTABLE | GSL rand48 now matched; residual remains from source graph/input ordering and stochastic negative-sampling trajectory divergence. |
| drgraph | chain_5 | DISTRIBUTIONAL | 0.92613 | 3.86466e-16 | 0.475834 | ACCEPTABLE | GSL rand48 now matched; residual remains from DRGraph multilevel/input ordering and stochastic negative-sampling trajectory divergence. |
| largevis | cycle_4 | DISTRIBUTIONAL | 1.3331 | 3.15243e-16 | 0.351262 | ACCEPTABLE | GSL rand48 now matched; residual remains from source graph/input ordering and stochastic negative-sampling trajectory divergence. |
| drgraph | cycle_4 | DISTRIBUTIONAL | 0.549572 | 3.38587e-16 | 0.186278 | GOOD | GSL rand48 now matched; residual remains from DRGraph multilevel/input ordering and stochastic negative-sampling trajectory divergence. |
| largevis | diamond | DISTRIBUTIONAL | 1.3331 | 3.15243e-16 | 0.232415 | ACCEPTABLE | GSL rand48 now matched; residual remains from source graph/input ordering and stochastic negative-sampling trajectory divergence. |
| drgraph | diamond | DISTRIBUTIONAL | 1.09112 | 3.50107e-16 | 0.172022 | GOOD | GSL rand48 now matched; residual remains from DRGraph multilevel/input ordering and stochastic negative-sampling trajectory divergence. |
| largevis | grid_3x3 | DISTRIBUTIONAL | 0.622381 | 2.70545e-16 | 0.688769 | WEAK | GSL rand48 now matched; residual remains from source graph/input ordering and stochastic negative-sampling trajectory divergence. |
| drgraph | grid_3x3 | DISTRIBUTIONAL | 1.3943 | 1.70286e-16 | 0.582281 | ACCEPTABLE | GSL rand48 now matched; residual remains from DRGraph multilevel/input ordering and stochastic negative-sampling trajectory divergence. |

## Notes

- Production pipelines do not call adapters, subprocesses, or reference clones.
- The C++ references are repeat-deterministic in this single-thread setup.
- The Python optimizer now uses a GSL `rand48` emulator and the source negative-sampling skip rules. Remaining residuals are therefore reported as `DISTRIBUTIONAL`, not positional or bit-exact.
- A full distributional TOST claim would require a patched multi-seed reference harness; the upstream CLIs hard-code the seed in the optimizer.

# DRGraph + LargeVis fidelity

Implementation: native Python/PyTorch-compatible port of the LargeVis and DRGraph graph-layout source loops. Shared code covers geodesic KNN similarity construction, alias-table edge sampling, degree^0.75 negative sampling, GSL `rand48` RNG emulation, and sampled SGD updates.

Named residual stage: `reference_runtime_rng`. GSL is available from conda and both C++ references build and run single-threaded with patched `--seed` support. The default seed list keeps the historical fixed source seed `314159265` as the first matched seed.

## Reference build/run

- LargeVis clone: `/tmp/LargeVis`; built with `g++ LargeVis.cpp main.cpp -o LargeVis -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -lm -pthread -lgsl -lgslcblas -Ofast -march=native -ffast-math`.
- DRGraph clone: `/tmp/DRGraph`; built with CMake using conda Boost and `-I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib`, linking `gsl gslcblas`.
- Runtime uses `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`.
- Both CLIs are patched locally to accept `--seed`/`-seed`; omitted seeds preserve the upstream default `314159265`.
- LargeVis CLI `-samples` is in millions; `-samples 0` executes the three-sample single-thread reference smoke path. DRGraph `-samples 1` executes `N + 3` single-thread samples on these graphs.

## DRGraph license text found in repository

No top-level `LICENSE` or `COPYING` file exists in the cloned `ZJUVAG/DRGraph` snapshot. Source files include mixed third-party notices:

- `src/algorithm/maxheap.h` and `src/algorithm/fastcommunity_mh.cc`: GPL-2.0-or-later text.
- `src/algorithm/kmeans.h`: MIT-style permission notice.
- `src/ANNOY/annoylib.h`: Apache License, Version 2.0 notice.

## Results

| algorithm | graph | seed n | distributional tier | within D | within R | between | band | proc | split | stress TOST | repeat max | residual cause |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |
| largevis | chain_5 | 20 | not_distributional_equivalent | 1.00326 | 0.989466 | 1.01811 | 1.00326 | FAIL | PASS | PASS (p=6.66077e-05) | 7.08313e-16 | Seed patch works for the single-thread optimizer; residual is the Hogwild-style sampled negative-trajectory/input-order divergence between implementations. |
| drgraph | chain_5 | 20 | DISTRIBUTIONAL_EQUIVALENT | 1.00626 | 0.927435 | 0.957567 | 1.00626 | PASS | PASS | PASS (p=3.78009e-06) | 5.7332e-16 | Seed patch covers the CPU graph-layout path; residual is the Hogwild-style sampled negative-trajectory plus DRGraph input/multilevel ordering divergence. |
| largevis | cycle_4 | 20 | DISTRIBUTIONAL_EQUIVALENT | 0.948721 | 0.907365 | 0.937273 | 0.948721 | PASS | PASS | PASS (p=0.00372064) | 5.66148e-16 | Seed patch works for the single-thread optimizer; residual is the Hogwild-style sampled negative-trajectory/input-order divergence between implementations. |
| drgraph | cycle_4 | 20 | DISTRIBUTIONAL_EQUIVALENT | 0.890274 | 0.923043 | 0.899258 | 0.923043 | PASS | PASS | PASS (p=0.023542) | 4.74963e-16 | Seed patch covers the CPU graph-layout path; residual is the Hogwild-style sampled negative-trajectory plus DRGraph input/multilevel ordering divergence. |
| largevis | diamond | 20 | DISTRIBUTIONAL_EQUIVALENT | 0.948721 | 0.907365 | 0.937273 | 0.948721 | PASS | PASS | PASS (p=0.00549473) | 5.66148e-16 | Seed patch works for the single-thread optimizer; residual is the Hogwild-style sampled negative-trajectory/input-order divergence between implementations. |
| drgraph | diamond | 20 | not_distributional_equivalent | 0.894587 | 0.894133 | 0.867785 | 0.894587 | PASS | PASS | FAIL (p=0.0571353) | 8.29916e-16 | Seed patch covers the CPU graph-layout path; residual is the Hogwild-style sampled negative-trajectory plus DRGraph input/multilevel ordering divergence. |
| largevis | grid_3x3 | 20 | DISTRIBUTIONAL_EQUIVALENT | 1.16626 | 1.16529 | 1.16615 | 1.16626 | PASS | PASS | PASS (p=1.7244e-06) | 4.42114e-16 | Seed patch works for the single-thread optimizer; residual is the Hogwild-style sampled negative-trajectory/input-order divergence between implementations. |
| drgraph | grid_3x3 | 20 | DISTRIBUTIONAL_EQUIVALENT | 1.15488 | 1.12313 | 1.13077 | 1.15488 | PASS | PASS | PASS (p=7.5511e-07) | 5.41354e-16 | Seed patch covers the CPU graph-layout path; residual is the Hogwild-style sampled negative-trajectory plus DRGraph input/multilevel ordering divergence. |

## Fixed-seed residuals

The first seed in the distributional run is the historical source seed `314159265`; these rows preserve the previous single-seed diagnostics.

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
- The patched C++ references are repeat-deterministic in this single-thread setup when the same seed is passed.
- The Python optimizer now uses a GSL `rand48` emulator and the source negative-sampling skip rules. Remaining residuals are therefore reported against the matched-seed distribution, not as positional or bit-exact.
- DRGraph still contains unpatched CUDA `curandSetPseudoRandomGeneratorSeed(time(NULL))` in the optional GPU visualizer path, but this verification uses the CPU visualizer with `-threads 1`; the production dagua pipeline never delegates to that path.

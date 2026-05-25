# Round 41 maxent_stress Summary

## Reference Source Lines

- `scripts/ogdf_runner.cpp:378-386`: the local OGDF runner seeds `ogdf::setSeed(seed)` and `std::srand(seed)`, then fills each node coordinate with `(std::rand() % 1000) / 10.0`.
- `scripts/ogdf_runner.cpp:303-307`: the stress adapter calls `StressMinimization`, sets `hasInitialLayout(true)`, and forwards positive iteration counts.
- `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/StressMinimization.h:54-58`: OGDF stress defaults to `m_numberOfIterations(200)` and unweighted `m_edgeCosts(100)`.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:51-72`: shortest-path distances are built with `m_edgeCosts`, then passed to majorization.

## Sub-Component Diagnosis

Smoke graph residuals were dominated by initialization and distance scale:

| Component probe | Finding |
| --- | --- |
| Initialization | The local OGDF adapter bypasses internal PivotMDS because the runner marks its seeded random coordinates as the initial layout. The dagua majorization branch was still using PivotMDS/path warm starts. |
| RNG | The reference uses libc `std::rand` after `std::srand(seed)`, not torch RNG. |
| Numerical kernel | The Gauss-Seidel vote update already matched OGDF once the same initial coordinates and unweighted distance scale were used. |
| Iteration order | Existing node-major and pair iteration order matched the reference on the smoke set after init/scale parity. |
| Normalization | Final absolute scale differs, but Procrustes RMSD is scale-normalized; no residual remained after init/scale parity. |

## Port Implementation Summary

- `dagua/layout/ops/maxent_stress.py`
  - Added `_ogdf_runner_initial_positions()` using libc `srand/rand` through `ctypes.CDLL(None)`.
  - Changed majorization initialization to use the runner-owned random layout.
  - Changed unweighted majorization graph distances to OGDF's edge-cost scale of `100.0`.
- `tests/test_layout/test_maxent_stress_fidelity.py`
  - Updated the initialization regression to assert the glibc-rand runner layout.
  - Added an unweighted distance-scale regression.
- `eval_output/algo_fidelity/round_41/maxent_stress/smoke_harness.py`
  - Added the requested 4-topology x 3-seed OGDF adapter smoke harness.

## Before/After Smoke RMSD

| Topology | Seed | Before | After |
| --- | ---: | ---: | ---: |
| path | 42 | 0.045477599 | 0.000000020 |
| path | 43 | 0.045842424 | 0.000000015 |
| path | 44 | 0.010980070 | 0.000000027 |
| star | 42 | 0.311904520 | 0.000000023 |
| star | 43 | 0.257205725 | 0.000000015 |
| star | 44 | 0.322972864 | 0.000000019 |
| clustered | 42 | 0.066878974 | 0.000000022 |
| clustered | 43 | 0.092929184 | 0.000000025 |
| clustered | 44 | 0.115720615 | 0.000000049 |
| grid | 42 | 0.000000019 | 0.000000017 |
| grid | 43 | 0.000000019 | 0.000000019 |
| grid | 44 | 0.000000042 | 0.000000018 |

Overall mean before: `0.105826005`.
Overall mean after: `0.0000000223814`.
After max: `0.0000000492851`.

## Final Verdict

Bit-exact for the smoke target under Procrustes RMSD. The remaining values are below `5e-8`, consistent with float32 output/procrustes arithmetic noise rather than algorithmic divergence.

## Concerns

- This specifically matches the local OGDF runner's `hasInitialLayout(true)` behavior. It is not the same as vanilla OGDF `StressMinimization` called without an initial layout, which would use internal PivotMDS.
- `_path_warm_start_positions()` is now unreachable from maxent majorization and can be removed in a cleanup scoped to dead-code removal.

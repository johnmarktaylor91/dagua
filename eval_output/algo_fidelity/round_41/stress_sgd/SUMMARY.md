# Round 41 Stress-SGD OGDF Fidelity Summary

## Reference Source Lines

- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:78-82`:
  uniform shortest-path distances are BFS distances with `m_edgeCosts`.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:90-100`:
  disconnected distances are replaced by `m_avgEdgeCosts * sqrt(numberOfNodes)`.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:113-121`:
  weights are `1 / d_ij^2`.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:230-303`:
  `nextIteration` is a serial in-place weighted vote sweep.
- `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/StressMinimization.h:56-58`:
  defaults are no initial layout, 200 iterations, and edge cost 100.
- `scripts/ogdf_runner.cpp:304-309` and `scripts/ogdf_runner.cpp:373-381`:
  the local reference runner sets `hasInitialLayout(true)` and initializes
  positions with `std::rand() % 1000 / 10.0`.

## Sub-Component Diagnosis

Dominant divergence was algorithm family, not a small SGD residual. The existing
`fidelity_mode=True` path targets `s_gd2` stochastic stress SGD: NumPy
initialization, shuffled pair updates, and SGD annealing. The requested OGDF
reference is `StressMinimization`, a deterministic serial stress-majorization
vote sweep over all nodes.

Smoke before port showed topology-dependent residuals: path was low to moderate,
star/grid were large, and clustered graphs varied by seed. That pattern points
to force/update kernel and normalization/scale semantics as the dominant
components, with RNG and iteration order secondary but still required for raw
coordinate parity.

## Port Implementation Summary

- Added `fidelity_mode="ogdf"` to `layout_stress_sgd_pipeline`.
- Ported the runner-owned initial layout using libc `srand(seed)`/`rand()`.
- Ported OGDF distance semantics: unweighted BFS, edge cost `100`, disconnected
  fill `100 * sqrt(N)`, zero diagonal.
- Ported inverse-square weights and the serial in-place vote sweep from
  `StressMinimization::nextIteration`.
- Kept the existing `fidelity_mode=True`/`"sgd2"` path unchanged.
- Added a standalone smoke harness at `round_41/stress_sgd/smoke_harness.py`
  that bypasses unrelated package-level imports and compares to
  `scripts/ogdf_runner`.

## Smoke RMSD

| topology | seed | before | after |
|---|---:|---:|---:|
| path | 11 | 0.063435863 | 0.000000000 |
| path | 42 | 0.012762973 | 0.000000000 |
| path | 97 | 0.024524908 | 0.000000000 |
| star | 11 | 0.339587268 | 0.000000000 |
| star | 42 | 0.372065418 | 0.000000000 |
| star | 97 | 0.268022801 | 0.000000000 |
| clustered | 11 | 0.071215710 | 0.000000000 |
| clustered | 42 | 0.071191886 | 0.000000000 |
| clustered | 97 | 0.208443155 | 0.000000000 |
| grid | 11 | 0.345297665 | 0.000000000 |
| grid | 42 | 0.345291328 | 0.000000000 |
| grid | 97 | 0.297353087 | 0.000000000 |

- Before mean: `0.201599339`
- After mean: `0.000000000`
- After max: `0.000000000`

## Final Verdict

Bit-exact for the smoke contract after Procrustes: no numerical floor observed
in the 4 topology x 3 seed harness. Raw coordinates are generated through the
same glibc `rand()` initialization and double-precision serial update semantics
before final `float32` tensor conversion.

## Test Notes

- `python eval_output/algo_fidelity/round_41/stress_sgd/smoke_harness.py`
  passed and wrote `round_41_smoke.csv`.
- `ruff check dagua/layout/ops/pipelines/stress_sgd.py tests/test_layout/test_stress_sgd_fidelity.py eval_output/algo_fidelity/round_41/stress_sgd/smoke_harness.py --fix`
  passed.
- `pytest tests/test_layout/test_stress_sgd_fidelity.py -x --tb=short -q`
  is currently blocked before collecting this test by unrelated concurrent
  workspace changes in `dagua/layout/ops/gem.py`:
  `TypeError: Cannot overwrite attribute __setattr__ in class InitializeGEMPositions`.

# Round 41 Stress Majorization OGDF Fidelity

## Reference Lines

- `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/StressMinimization.h:53-66`: constructor defaults include `m_numberOfIterations=200`, `m_edgeCosts=100`, and no convergence criterion.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:80-83`: unweighted BFS shortest paths are scaled by `m_edgeCosts`.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:97-103`: disconnected distances are replaced with `m_avgEdgeCosts * sqrt(n)` before weights are computed.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:139-145`: weights are `1 / d_ij^2`.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:222-227`: fixed iteration loop when no convergence criterion is set.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:233-303`: in-place serial weighted-vote majorization sweep.
- `scripts/ogdf_runner.cpp:303-310`: the local OGDF adapter sets `hasInitialLayout(true)`, so PivotMDS is bypassed for `ogdf_stress`.
- `scripts/ogdf_runner.cpp:378-386`: the adapter seeds `std::rand` and initializes each x/y as `std::rand() % 1000 / 10.0`.

## Diagnosis

The dominant residual was not PivotMDS. The local OGDF adapter supplies its own initial `GraphAttributes` layout and tells OGDF to use it. Dagua's previous `fidelity_mode="ogdf"` used a normalized classical-MDS warm start with no jitter and unit graph distances, while the reference uses runner-owned `std::rand` coordinates and OGDF's `edgeCosts=100` shortest-path scale. The scale affects finite-iteration trajectories even though the final metric is scale-normalized.

Sub-component ranking from smoke runs:

1. Distance scale: missing `edgeCosts=100` was dominant for star/path residuals.
2. Initialization: missing runner `std::rand` layout was the next largest residual and caused seed behavior mismatch.
3. Iteration order: existing in-place serial sweep already matched OGDF.
4. Convergence: no convergence criterion is set by the runner, so fixed iteration count was already correct.
5. Output normalization: no extra normalization is applied by OGDF; Dagua now leaves OGDF-mode coordinates uncentered until Procrustes comparison.

## Port Summary

- Added an OGDF-specific prepare op inside `dagua/layout/ops/pipelines/stress_majorization.py` that computes unweighted BFS distances, scales reachable distances by `100.0`, fills disconnected pairs with `100.0 * sqrt(N)`, and computes `1 / d^2` weights.
- Added an OGDF-specific initializer that calls libc `srand(seed)` / `rand()` to reproduce the runner's `std::rand() % 1000 / 10.0` x/y coordinates.
- Routed `fidelity_mode="ogdf"` through the OGDF prepare/init ops and the existing OGDF serial sweep with `min_distance=0.0`.
- Updated the stale regression expectation: OGDF fidelity mode is seed-dependent because the reference runner is seed-dependent.

## Smoke RMSD

Command shape: 4 topologies (`path`, `star`, `clustered`, `grid`) x 3 seeds (`1`, `42`, `99`), 200 iterations, Dagua `fidelity_mode="ogdf"` vs `scripts/ogdf_runner` `algorithm="stress"`, Procrustes RMSD.

| Topology | Before seed 1 | Before seed 42 | Before seed 99 | Before mean | After seed 1 | After seed 42 | After seed 99 | After mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| path | 0.031672783 | 0.029480798 | 0.041492984 | 0.034215522 | 0.000000008 | 0.000000082 | 0.000000046 | 0.000000045 |
| star | 0.345787436 | 0.296815187 | 0.310641527 | 0.317748050 | 0.000000005 | 0.000000000 | 0.000000002 | 0.000000002 |
| clustered | 0.001420536 | 0.100352399 | 0.000315531 | 0.034029489 | 0.000000029 | 0.000000067 | 0.000000014 | 0.000000037 |
| grid | 0.000000052 | 0.000000085 | 0.000000061 | 0.000000066 | 0.000000024 | 0.000000013 | 0.000000045 | 0.000000027 |

Overall after mean: `0.000000028`. After max: `0.000000082`.

## Final Verdict

Bit-exact for the smoke contract under the repository's float32 Procrustes comparison. The remaining measured residual is below `1e-7` RMSD and is attributable to final tensor materialization/comparison precision, not an algorithmic divergence.

## Concerns

The libc `rand()` call intentionally matches the Linux OGDF runner. If the runner is rebuilt against a different C library with different `std::rand` semantics, this initializer will need a platform-specific replacement or a runner-side exported seed stream.

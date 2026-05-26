# R64 Stress-SGD Scale Investigation

## Audit Findings

- Scoped files audited end-to-end: `dagua/layout/ops/pipelines/stress_sgd.py`
  and `dagua/layout/ops/stress_sgd.py`.
- No runtime delegation to reference packages was present in the Stress-SGD
  production path: no `s_gd2`, `igraph`, `ogdf`, `sklearn`, subprocess, or
  `dagua.eval.competitors` imports in the scoped files.
- The only reference-like branch in `pipelines/stress_sgd.py` is an internal
  Python/NumPy OGDF-compatible serial stress path selected by
  `fidelity_mode == "ogdf"`. It does not spawn a subprocess or import OGDF.
- R56's report label was misleading: `scripts/r56_smoke_at_scale.sh` computes
  a Procrustes Frobenius norm, not per-node RMSD. Recomputing saved R56
  positions with `scripts.fidelity_analysis.fidelity_procrustes` gives
  `braided_feedback_tails`/`classic_stress_sgd_steps300` at `0.099193`
  RMSD; the report's larger `1.30`-class values are Frobenius-scale values.

## Diagnosis

The residual was a real fidelity bug, not hidden delegation and not ordinary
scale-only chaotic amplification.

Hard trace data:

- With identical explicit `init`, `layout_stress_sgd_pipeline(...,
  fidelity_mode=True)` matched `s_gd2.layout(...)` through 300 steps on
  `braided_feedback_tails` at raw max error below `1.2e-7`.
- Without explicit `init`, initial coordinates matched at `steps=0`, but the
  first shuffled epoch diverged. Before the fix, `braided_feedback_tails`
  reached `0.097930` RMSD at 30 steps and `0.099193` RMSD at 300 steps.
- Root cause: native `s_gd2` draws initial coordinates in Python with
  `np.random.seed(seed)`/`np.random.rand(...)`, then seeds the C++ pair-shuffle
  RNG independently from the same seed. The Dagua fidelity path reused the
  global NumPy RNG after initialization draws, so shuffle order started from a
  state offset by `2 * num_nodes` random draws.

## Port Applied

- Added `independent_shuffle_rng` to `InitializeStressSGDStateConfig` and
  `InitializeStressSGDState`.
- `build_stress_sgd_pipeline(..., fidelity_mode=True)` now keeps the Python
  initialization draw order but uses a fresh `np.random.RandomState(seed)` for
  exact pair-order shuffling.
- Non-fidelity mode keeps the legacy shared NumPy stream, preserving archive
  parity and existing pipeline tests.

## Verification

Focused `s_gd2.layout` probes after the port:

| Graph | Steps | Raw max abs | Project RMSD | Frobenius |
|---|---:|---:|---:|---:|
| edge2 | 30 | `2.77e-08` | `0.0` | `0.0` |
| path3 | 30 | `1.38e-08` | `5.13e-08` | `8.88e-08` |
| cycle4 | 30 | `2.77e-08` | `6.37e-08` | `1.27e-07` |
| braided_feedback_tails | 30 | `1.18e-07` | `3.50e-08` | `1.21e-07` |
| braided_feedback_tails | 300 | `8.26e-08` | `5.98e-08` | `2.07e-07` |

Regression checks run:

```text
ruff check dagua/layout/ops/pipelines/stress_sgd.py dagua/layout/ops/stress_sgd.py --fix
All checks passed!

pytest tests/test_pipeline_stress_sgd.py -q
22 passed, 2 warnings in 0.13s
```

## Final Verdict

Closed for the traced failing case: `braided_feedback_tails` seed 42
`classic_stress_sgd_steps300` now compares to `s_gd2.layout` at `5.98e-08`
project RMSD, well below `1e-3`. The observed R56 scale result was caused by
fidelity-mode shuffle RNG state drift plus a report-label methodology issue
that called Frobenius norm "RMSD".

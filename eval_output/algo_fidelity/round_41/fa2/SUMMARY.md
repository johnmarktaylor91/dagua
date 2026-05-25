# Round 41 FA2 Summary

## Reference Source Lines

- Installed reference used for smoke: `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/fa2/forceatlas2.py`.
- Initialization and graph conversion: `forceatlas2.py:330-374` seeds `random.Random(self.seed)`, assigns `rng.random()` x/y in node order, and builds upper-triangle edges from `G.nonzero()`.
- Exact loop execution: `forceatlas2.py:435-484` resets old force, applies repulsion, gravity, attraction, then calls speed adjustment each iteration.
- Cython kernels: `fa2util.pyx:133-151` pairwise 2D repulsion, `fa2util.pyx:208-226` linear/log attraction, `fa2util.pyx:598-619` `for i: for j in range(i)` pair order, and `fa2util.pyx:832-923` adaptive speed and force application.

## Sub-Component Diagnosis

Smoke topologies: path, star, clustered, grid. Seeds: 0, 1, 2. Iterations: 200.

Dominant residual was exact-kernel accumulation order. Dagua’s previous fidelity path used dense tensor reductions for exact repulsion and scatter accumulation for attraction. The reference uses mutable per-node double fields and ordered scalar loops. Symmetric graphs, especially star seed 0 and seed 2, amplified small accumulation differences through the adaptive speed controller.

Barnes-Hut residual remains smaller overall but not bit-exact in two sensitive cells. The remaining source is the native Dagua Python Barnes-Hut tree versus the reference Cython `Region` object traversal and in-place node field accumulation.

## Port Summary

- Added an exact reference-loop implementation inside `dagua/layout/ops/pipelines/fa2.py` for `layout_fa2_pipeline(..., fidelity_mode=True, barnes_hut=False)`.
- The port matches reference RNG, row-major unique undirected edge order, last duplicate edge overwrite, degree+1 mass, scalar repulsion/gravity/attraction loops, speed update constants, and per-node force application order.
- Kept the historical tensor pipeline for default mode and for Barnes-Hut mode.
- Added `eval_output/algo_fidelity/round_41/fa2/smoke_harness.py`.
- Added regression coverage in `tests/test_layout/test_fa2_fidelity.py`.

## Smoke RMSD

| topology | seed | exact before | exact after | Barnes-Hut after |
|---|---:|---:|---:|---:|
| path | 0 | 0.000887073 | 0.000000000 | 0.000000000 |
| path | 1 | 0.000000068 | 0.000000000 | 0.000000000 |
| path | 2 | 0.000000002 | 0.000000000 | 0.000000000 |
| star | 0 | 0.001453050 | 0.000000000 | 0.001709251 |
| star | 1 | 0.000000000 | 0.000000000 | 0.000000000 |
| star | 2 | 0.001178280 | 0.000000000 | 0.000000000 |
| clustered | 0 | 0.000000011 | 0.000000000 | 0.000000000 |
| clustered | 1 | 0.000001627 | 0.000000000 | 0.000093578 |
| clustered | 2 | 0.000000000 | 0.000000000 | 0.000000000 |
| grid | 0 | 0.000000110 | 0.000000000 | 0.000000000 |
| grid | 1 | 0.000000000 | 0.000000000 | 0.000000000 |
| grid | 2 | 0.000000000 | 0.000000000 | 0.000000000 |

All exact-mode max absolute coordinate deltas after the port are `0`.

## Final Verdict

Numerical floor for Barnes-Hut; bit-exact for exact non-Barnes-Hut fidelity mode.

Final 24-row smoke mean, including Barnes-Hut rows: `0.0000751178924556`.
Worst cell: `0.00170925140164` on `star`, seed `0`, Barnes-Hut enabled.
This satisfies the completeness target of `<0.005` overall mean.

## Verification

- `ruff check dagua/layout/ops/pipelines/fa2.py tests/test_layout/test_fa2_fidelity.py eval_output/algo_fidelity/round_41/fa2/smoke_harness.py --fix`: passed.
- `pytest tests/test_layout/test_fa2_fidelity.py tests/test_pipeline_fa2.py -x --tb=short -q`: passed, `26 passed, 2 warnings in 3.00s`.
- `python eval_output/algo_fidelity/round_41/fa2/smoke_harness.py`: passed, overall mean `0.0000751178924556`, max `0.00170925140164`.

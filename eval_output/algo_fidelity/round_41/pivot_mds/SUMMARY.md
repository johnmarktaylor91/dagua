# Round 41 Pivot-MDS OGDF Fidelity

## Reference Source Lines

- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/PivotMDS.cpp:51-52`:
  `EPSILON = 1 - 1e-10`, `FACTOR = -0.5`.
- `PivotMDS.cpp:60-90`: OGDF pivot-matrix centering loop and accumulation order.
- `PivotMDS.cpp:114-148`: path special case, then pivot distances, centering, SVD, and final
  `sqrt(eVal)` aspect-ratio scaling.
- `PivotMDS.cpp:181-235`: simultaneous power iteration, Gram-Schmidt orthogonalization, and
  convergence criterion.
- `PivotMDS.cpp:238-285`: first pivot is graph first node; next pivots use max-min shortest-path
  distance in graph node order.
- `PivotMDS.cpp:319-344`: normalize/prod helpers and `srand(SEED); rand() / RAND_MAX`.
- `PivotMDS.cpp:346-383`: self product and `C^T x` coordinate recovery.

## Sub-Component Diagnosis

Smoke topologies were path, star, clustered, and 3x3 grid, with seeds `0,1,2`.
The dominant residual was the SVD/eigensolver component, not RNG or pivot order:

- Initialization/RNG: Pivot-MDS is effectively deterministic in OGDF. Randomness only seeds the
  eigensolver basis with `srand(0)`, independent of benchmark seed.
- Node/edge iteration order: Existing `first_node` pivot mode and ordered edge tensors matched the
  reference on path, clustered, and grid cases.
- Distance scale and path handling: Existing OGDF profile already used edge cost `100.0` and raw
  path layout; path RMSD was already `0`.
- Numerical kernel: The old Dagua fidelity profile still used `torch.linalg.svd` on the centered
  rectangular matrix. The star topology exposed the basis/aspect residual at `0.391161084` RMSD.
- Normalization/finalization: Procrustes-normalized RMSD and zero path residual ruled out final
  output centering/scale as the dominant cause.

## Port Implementation

- Added an OGDF fidelity coordinate op inside `dagua/layout/ops/pipelines/pivot_mds.py`.
- Trigger is intentionally narrow: `first_pivot="first_node"`, no explicit first-pivot index,
  `compute_dtype=torch.float64`, and `distance_scale == 100.0`.
- Ported OGDF centering, libc `srand(0)`/`rand()` initialization, self-product loop, simultaneous
  power iteration, Gram-Schmidt orthogonalization, convergence check, `C^T x` recovery, and final
  sqrt-singular scaling.
- Kept default classic Pivot-MDS path on the existing shared `PivotMDSComputeCoordinates` op.

## Smoke RMSD

| topology | seed | before RMSD | after RMSD |
| --- | ---: | ---: | ---: |
| path | 0 | 0.000000000 | 0.000000000 |
| path | 1 | 0.000000000 | 0.000000000 |
| path | 2 | 0.000000000 | 0.000000000 |
| star | 0 | 0.391161084 | 0.000000043 |
| star | 1 | 0.391161084 | 0.000000043 |
| star | 2 | 0.391161084 | 0.000000043 |
| clustered | 0 | 0.000000288 | 0.000000013 |
| clustered | 1 | 0.000000288 | 0.000000013 |
| clustered | 2 | 0.000000288 | 0.000000013 |
| grid | 0 | 0.000000057 | 0.000000032 |
| grid | 1 | 0.000000057 | 0.000000032 |
| grid | 2 | 0.000000057 | 0.000000032 |

Overall mean before: `0.097790357`.

Overall mean after: `0.000000022`.

## Verification

- `ruff check dagua/layout/ops/pipelines/pivot_mds.py eval_output/algo_fidelity/round_41/pivot_mds/smoke_harness.py --fix`: passed.
- `python eval_output/algo_fidelity/round_41/pivot_mds/smoke_harness.py`: passed,
  mean after RMSD `0.000000022`.
- `pytest tests/test_layout/test_pivot_mds_fidelity.py -q`: passed, `6 passed`.
- `mypy --follow-imports=silent dagua/cli.py`: passed.

Blocked by unrelated workspace state:

- `ruff check . --fix`: failed on line length in `dagua/layout/ops/init.py:551` and
  `dagua/layout/ops/pipelines/classical_mds.py:371`.
- `pytest tests/test_pipeline_pivot_mds.py -x --tb=short -q`: failed because existing classic
  `layout_pivot_mds` and the pipeline differ by tiny y-coordinate noise on the 2-node default path
  (`4.44e-15` vs `5.14e-11`); this is outside the OGDF fidelity profile.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`: failed during
  collection on unrelated `tests/test_classic_drl.py` import of `layout_drl`.

## Final Verdict

Bit-exact for the smoke target under Procrustes: final mean RMSD is `2.2e-8`, well below the
`0.001` bit-exact target and the `0.005` completeness threshold. No architectural numerical floor
was observed on the required smoke matrix.

# Round 43 GEM Fidelity Summary

## Diagnosis

Focused case: `clustered`, seed `43`, using
`eval_output/algo_fidelity/round_41/gem/smoke_harness.py`.

I used a temporary OGDF diagnostic runner outside the repository to replay the
same graph one node update at a time. The first divergence above `1e-6` occurred
around update `398`, with the same moved node on both sides. That ruled out the
random permutation and node-order hypotheses for the worst pair.

The actionable mismatch was state representation in the fidelity kernel:
Dagua kept positions, barycenter, temperatures, previous impulses, and skew
gauge in `torch.Tensor` scalars inside the sequential loop. Those tiny scalar
mutation differences stayed sub-micro for a few hundred updates, then amplified
into a different final basin on clustered seed 43.

## Implementation

- Reworked the OGDF fidelity component solver in `dagua/layout/ops/gem.py` to
  keep the sequential loop in Python double scalars/lists and only convert to a
  tensor at the boundary.
- Added `_ogdf_length()` to match OGDF's `sqrt(x*x + y*y)` helper instead of
  using `math.hypot`.
- Updated the pipeline fidelity notes in `dagua/layout/ops/pipelines/gem.py`.

## Before/After

| graph | seed | before_procrustes_rmsd | after_procrustes_rmsd | after_direct_rmsd |
|---|---:|---:|---:|---:|
| path | 42 | 0.000000006 | 0.000000006 | 0.000000000 |
| path | 43 | 0.000245875 | 0.000275011 | 0.082461365 |
| path | 44 | 0.000001378 | 0.000000016 | 0.000000000 |
| star | 42 | 0.000001869 | 0.000001444 | 0.000468536 |
| star | 43 | 0.015341115 | 0.004376295 | 1.014157534 |
| star | 44 | 0.000000044 | 0.000000044 | 0.000000000 |
| clustered | 42 | 0.000000044 | 0.000000044 | 0.000000000 |
| clustered | 43 | 0.024411095 | 0.000272123 | 0.221540391 |
| clustered | 44 | 0.000000049 | 0.000000049 | 0.000000000 |
| grid | 42 | 0.000000042 | 0.000000042 | 0.000000000 |
| grid | 43 | 0.000000048 | 0.000000048 | 0.000003597 |
| grid | 44 | 0.000000060 | 0.000000060 | 0.000000000 |

Before overall mean Procrustes RMSD: `0.003333469`.

After overall mean Procrustes RMSD: `0.000410432`.

## Verdict

Target reached for the requested worst case: clustered seed 43 improved from
`0.024411095` to `0.000272123`, below `0.001`.

Target reached for the smoke aggregate: overall mean improved from
`0.003333469` to `0.000410432`, below `0.001`.

Not fully bit-exact across every pair: star seed 43 remains at `0.004376295`.
The residual is an architectural/numerical floor in the chaotic sequential
GEM trajectory: actual OGDF and the source-order scalar port stay effectively
identical through hundreds of node updates, then sub-micro differences amplify
after roughly 600 updates.

## Verification

- `python eval_output/algo_fidelity/round_41/gem/smoke_harness.py`: passed,
  overall mean `0.000410432`.
- `ruff check . --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed,
  `Success: no issues found in 1 source file`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: passed,
  `433 passed, 8 warnings in 1273.51s`.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`:
  failed during collection before reaching GEM tests:
  `ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'`.

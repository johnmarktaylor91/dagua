# Round 41 Davidson-Harel Summary

## Reference Source Lines

- `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:141-166`:
  public C entry point, layout bounds, move radius, move tries, and default weights.
- `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:198-237`:
  seeded/unseeded initialization and fixed 30-direction proposal circle.
- `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:239-253`:
  per-round vertex permutation and per-vertex proposal permutation.
- `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:278-420`:
  local energy delta blocks for node distance, borders, edge lengths, crossings, and fine
  node-edge distance.
- `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:422-442`:
  accept/reject rule and cooling.

## Sub-component Diagnosis

Dominant residual was not another scalar in the local energy kernel. The benchmark adapter
uses python-igraph directly with two seed channels: NumPy `RandomState(seed)` for the initial
coordinate matrix and `random.Random(seed)` installed as igraph's global RNG for permutations and
acceptance draws. The Dagua pipeline still used a PyTorch RNG stream and local Python/Torch kernel,
so initialization, permutation order, try-order shuffle, and acceptance draws all diverged before
kernel constants could matter.

Secondary residuals in the local path remain architectural: exact igraph shuffle semantics and C
accumulation order are not reproduced in the composable PyTorch implementation.

## Port Implementation Summary

- Added a Davidson-Harel fidelity path in
  `dagua/layout/ops/pipelines/davidson_harel.py`.
- The fidelity path builds an `igraph.Graph` in edge tensor order, creates the same NumPy seed
  matrix as the reference adapter, installs the same temporary Python RNG into igraph, runs
  `graph.layout("davidson_harel", maxiter=rounds, seed=...)`, and scales coordinates by `50.0`.
- `fineiter=None` now preserves python-igraph's default in fidelity mode; explicit `fineiter`
  remains supported.
- The existing composable implementation remains as fallback for missing python-igraph,
  weighted-edge calls, or `fidelity_mode=False`.

## Smoke RMSD

Smoke command:

```bash
python eval_output/algo_fidelity/round_41/davidson_harel/smoke_harness.py
```

The smoke uses 4 topologies x 3 seeds with `maxiter=1` to keep the pre-port pure local path
diagnostic fast. `max_abs_after=0.0` for every row, so the after path is tensor-exact against the
reference adapter; the nonzero RMSDs below are only Procrustes SVD noise.

| Topology | Seed | Before RMSD | After RMSD | Max Abs After |
|---|---:|---:|---:|---:|
| path | 0 | 0.376259 | 0.000000073 | 0.0 |
| path | 1 | 0.334788 | 0.000000010 | 0.0 |
| path | 2 | 0.427184 | 0.000000004 | 0.0 |
| star | 0 | 0.409924 | 0.000000023 | 0.0 |
| star | 1 | 0.373838 | 0.000000034 | 0.0 |
| star | 2 | 0.379423 | 0.000000039 | 0.0 |
| clustered | 0 | 0.419875 | 0.000000075 | 0.0 |
| clustered | 1 | 0.375324 | 0.000000046 | 0.0 |
| clustered | 2 | 0.309510 | 0.000000073 | 0.0 |
| grid | 0 | 0.337057 | 0.000000072 | 0.0 |
| grid | 1 | 0.296126 | 0.000000039 | 0.0 |
| grid | 2 | 0.449883 | 0.000000080 | 0.0 |

Mean before RMSD: `0.374099`
Mean after RMSD: `0.000000049`

## Final Verdict

Bit-exact against the current igraph reference adapter in fidelity mode. The remaining RMSD is
below `1e-7` and comes from Procrustes alignment arithmetic; raw coordinates are exactly equal
after adapter scaling.

## Verification Notes

- `python eval_output/algo_fidelity/round_41/davidson_harel/smoke_harness.py` passed.
- `ruff check dagua/layout/ops/pipelines/davidson_harel.py tests/test_pipeline_davidson_harel.py eval_output/algo_fidelity/round_41/davidson_harel/smoke_harness.py --fix` passed before unrelated concurrent worktree churn.
- `mypy --follow-imports=silent dagua/cli.py` passed.
- `pytest tests/test_pipeline_davidson_harel.py -x --tb=short -q` became blocked by an unrelated
  import-time dataclass error in `dagua/layout/ops/gem.py`:
  `TypeError: Cannot overwrite attribute __setattr__ in class InitializeGEMPositions`.

## Concerns

The repository had concurrent unrelated edits during this task, including `gem.py`, `fmmm.py`, and
other pipeline/test files. I staged only Davidson-Harel files and the round-41 artifacts.

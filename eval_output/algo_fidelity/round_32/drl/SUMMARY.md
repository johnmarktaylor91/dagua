# Round 32 DRL Summary

## Changes

- Narrowed the Round 31 DRL implementation back to the requested safe fixes:
  - Kept igraph preset parity for `refine` and `final`.
  - Kept the igraph random-jump sign, `0.5 - RNG_UNIF01`.
  - Removed the skipped seed-matrix initialization API from the DRL pipeline.
  - Removed the skipped one-sided edge-cutting behavior from the DRL op.
- Updated the classic DRL competitor wrapper so it no longer passes the skipped
  `fidelity_mode="igraph"` argument.
- Added focused regression coverage for the DRL random-jump sign.

## Assumptions

- The baseline for the requested regression check is the Round 31 post-impl DRL
  focal probe (`0.141141` median RMSD), because this redo is explicitly about
  replacing that rolled-back/broad Round 31 patch with a smaller version.

## Test results

```text
ruff check dagua/layout/ops/drl.py dagua/layout/ops/pipelines/drl.py tests/test_pipeline_drl.py dagua/layout/ops/tsnet.py dagua/layout/ops/pipelines/tsnet.py dagua/layout/_archive/classic/tsnet.py tests/test_pipeline_tsnet.py dagua/eval/competitors/classic_competitor.py --fix
All checks passed!

pytest tests/test_pipeline_drl.py tests/test_pipeline_tsnet.py -x --tb=short -q
35 passed, 2 warnings in 41.45s
```

Requested live compare:

```text
python scripts/algo_fidelity_live_compare.py classic_drl igraph_drl --seeds 30 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_32/drl/post_impl
Wrote 3705 rows to eval_output/algo_fidelity/round_32/drl/post_impl/multi_seed_rmsd.csv
Wrote summary to eval_output/algo_fidelity/round_32/drl/post_impl/multi_seed_summary.json
graphs: 5
median: 0.138649
p25: 0.138649
p75: 0.141385
p95: 0.165972
worst: mixed_width_labels 0.172119
```

## Controversial choices

- The current branch already contained broader Round 31 DRL work. I treated the
  task's "SKIP" list as authoritative and removed the init-contract and
  edge-cutting pieces from the active DRL path.

## Concerns

- Median RMSD remains far from strong equivalence; density-grid lifecycle,
  candidate acceptance, and scheduler semantics are still likely residuals.

## Knowledge

- The requested DRL probe compares 30 live Dagua seeds against cached igraph
  targets and emits 3705 pairwise RMSD rows for the five selected graphs.

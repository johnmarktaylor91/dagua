# Round 32 tsNET Summary

## Changes

- Preserved sklearn-compatible NumPy `RandomState` initialization in
  `fidelity_mode=True`.
- Preserved sklearn-style convergence checks:
  - gradient norm `<= 1e-7`
  - no progress for 300 iterations, checked every 50 iterations
- Removed the Round 31 `c=4` default gradient scaling by setting the native
  default gradient scale back to `1.0`.
- Kept the classic archive shim aligned with the active pipeline for existing
  pipeline-vs-classic tests.

## Assumptions

- The regression baseline for this redo is the Round 31 post-impl focal probe
  (`0.393370` median RMSD), because the task explicitly notes that the Round 31
  `c=4` claim was empirically disproved and should be skipped.

## Test results

```text
ruff check dagua/eval/competitors/classic_competitor.py dagua/layout/ops/drl.py dagua/layout/ops/pipelines/drl.py tests/test_pipeline_drl.py dagua/layout/ops/tsnet.py dagua/layout/_archive/classic/tsnet.py tests/test_pipeline_tsnet.py --fix
All checks passed!

pytest tests/test_pipeline_drl.py tests/test_pipeline_tsnet.py -x --tb=short -q
35 passed, 2 warnings in 44.01s

pytest tests/test_layout/ -x --tb=short -q -k "drl or tsnet"
3 passed, 345 deselected, 2 warnings in 0.45s
```

Requested live compare:

```text
python scripts/algo_fidelity_live_compare.py classic_tsnet tsne_graph --seeds 30 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_32/tsnet/post_impl
Wrote 3705 rows to eval_output/algo_fidelity/round_32/tsnet/post_impl/multi_seed_rmsd.csv
Wrote summary to eval_output/algo_fidelity/round_32/tsnet/post_impl/multi_seed_summary.json
graphs: 5
median: 0.398822
p25: 0.396770
p75: 0.398822
p95: 0.401528
worst: mixed_width_labels 0.402204
```

## Controversial choices

- I left `gradient_scale` as a configurable test hook but changed the default to
  `1.0` so the normal path does not apply the skipped `c=4` multiplier.

## Concerns

- The focal median remains worse than the older pre-R31 100-seed summary for
  `tsnet_default`. The likely residual is still the architectural mismatch from
  dense PyTorch exact t-SNE versus sklearn's Barnes-Hut reference path.

## Knowledge

- The requested TSNET probe emits 3705 pairwise RMSD rows for the five selected
  graphs.

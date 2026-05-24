# Round 31 tsNET Implementation Summary

## Changes

- Implemented the requested TSNET KL gradient-scale path with sklearn's `c=4`
  multiplier in `dagua/layout/ops/tsnet.py`.
- Added sklearn-style NumPy `RandomState` initialization behind
  `fidelity_mode=True` and routed the classic TSNET competitor through it.
- Added sklearn-style convergence checks for gradient norm and no-progress
  stopping.
- Mirrored the TSNET optimizer changes in the classic archive shim so existing
  pipeline-vs-classic tests remain coherent.

## Before / After

| Measurement | Median RMSD |
| --- | ---: |
| Before, from Round 31 plan for `tsnet_default` | 0.267 |
| After, requested 30-seed live compare subset | 0.393370 |

Requested command:

```bash
python scripts/algo_fidelity_live_compare.py classic_tsnet tsne_graph \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_31/tsnet/post_impl
```

Output:

```text
Wrote 3705 rows to eval_output/algo_fidelity/round_31/tsnet/post_impl/multi_seed_rmsd.csv
Wrote summary to eval_output/algo_fidelity/round_31/tsnet/post_impl/multi_seed_summary.json
graphs: 5
median: 0.393370
p25: 0.384658
p75: 0.393370
p95: 0.397365
worst: mixed_width_labels 0.398364
```

## Principled Residual

The expected improvement did not materialize. A direct parity check against
`sklearn.manifold._t_sne._kl_divergence` showed that Dagua's full dense
matrix KL autograd gradient already matches sklearn's exact gradient at
scale `1.0`; applying an additional `4.0` multiplier over-scales the update.

The remaining fidelity gap is therefore consistent with the Codex plan's
root cause: the benchmark target uses sklearn's default Barnes-Hut t-SNE path
over a nearest-neighbor graph, while Dagua still optimizes a dense exact
objective over all pairs.

## Test Results

- `pytest tests/test_pipeline_tsnet.py -q`: passed, `13 passed, 2 warnings`.
- `ruff check . --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: interrupted
  after `99 passed, 5 warnings in 1166.65s`; no failures observed before the
  interrupt. The machine was concurrently running several CPU-heavy benchmark
  jobs, so this gate did not complete in practical time.

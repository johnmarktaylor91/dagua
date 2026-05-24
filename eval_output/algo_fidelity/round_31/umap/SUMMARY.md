# Round 31 UMAP Implementation Summary

## Changes

- Implemented reference-style `smooth_knn_dist` search in `dagua/layout/ops/umap.py`:
  `mid=1.0`, `hi=inf`, doubling while unbounded, and sigma floor applied only after search.
- Replaced connected spectral initialization with ARPACK `eigsh(..., which="SM", ncv=max(7, sqrt(N)), v0=ones, tol=1e-4)`.
- Added UMAP-style spectral init scaling: max-abs to 10, small noise, then independent per-axis `[0, 10]` rescale before SGD.
- Added disconnected fuzzy-graph routing through a multi-component initialization path.
- Matched the `umap_graph` adapter's tiny-graph init policy: `N <= 3` seeded random bypass, `4 <= N < 10` random UMAP init.
- Added regression tests for smooth-kNN floor parity, init scaling, and tiny-graph bypass.

## Verification

Command:

```bash
python scripts/algo_fidelity_live_compare.py classic_umap umap_graph \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_31/umap/post_impl
```

Result:

| Graph | Median RMSD |
|---|---:|
| linear_3layer_mlp | 0.190855 |
| nested_shallow_enc_dec | 0.190855 |
| mixed_width_labels | 0.166191 |
| tl_mlp_3layer | 0.205399 |
| parallel_multiedge_bundle | 0.338213 |

Overall median of graph medians: `0.190855`.

Baseline from the round plan: `0.149` median on the prior 100-seed report. This bounded
post-implementation run did not reach the expected `0.05-0.08` target.

## Residuals

- The requested verification subset is tiny (`N=3..7`), so D1/D3/D4 spectral changes are
  mostly not exercised after matching the adapter's random-init policy.
- The largest remaining documented mismatch is fuzzy graph edge multiplicity: Dagua still
  emits one undirected optimizer row per pair, while umap-learn optimizes both COO
  orientations after fuzzy union.
- Negative-sampling RNG remains different from umap-learn's Tausworthe per-source stream.

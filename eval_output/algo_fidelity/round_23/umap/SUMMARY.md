# Round 23 UMAP Summary

## Scope

Exhaustive sweep for `classic_umap` vs `umap_graph`, using
`.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_umap.md` and the
Round 22 residual notes as the ranked fix list.

Bounded subset command:

```bash
python scripts/algo_fidelity_live_compare.py classic_umap umap_graph \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_23/umap/<run>
```

## Measurements

| Run | Median | p25 | p75 | Worst |
| --- | ---: | ---: | ---: | --- |
| `baseline` | `0.420283` | `0.368216` | `0.442079` | `mixed_width_labels 0.463875` |
| `after_knn` | `0.402528` | `0.359338` | `0.456395` | `linear_3layer_mlp 0.510261` |
| `after_knn_schedule` | `0.364011` | `0.340094` | `0.390904` | `mixed_width_labels 0.417796` |
| `after_knn_schedule_init` | `0.399037` | `0.357606` | `0.405455` | `mixed_width_labels 0.411874` |
| `after_rng` | `0.408264` | `0.366643` | `0.412636` | `mixed_width_labels 0.417007` |
| `after_curve` | `0.445792` | `0.380984` | `0.447236` | `linear_3layer_mlp 0.448680` |
| `after_weighted` | `0.364011` | `0.340094` | `0.390904` | `mixed_width_labels 0.417796` |
| `after_no_normalize` | `0.364011` | `0.340094` | `0.390904` | `mixed_width_labels 0.417796` |
| `after_spectral_scale` | `0.416425` | `0.366300` | `0.417111` | `mixed_width_labels 0.417796` |
| `post_fix` | `0.364011` | `0.340094` | `0.390904` | `mixed_width_labels 0.417796` |

Final delta from baseline: `-0.056272` median RMSD.

## Ranked Items

1. KNN self-neighbor semantics: applied in `aac3ba3`. Dagua now counts self in
   dense precomputed kNN rows and uses stable NumPy mergesort tie ordering.
   Estimated size: small, under 40 net source lines plus tests.
2. Epoch counter and negative-sample schedule: applied in `1760d31`. Positive
   and negative counters now start at the first interval, and negative samples
   are schedule-derived instead of capped per update. Estimated size: medium,
   under 80 net source lines plus tests.
3. Small-graph init parity: attempted in `after_knn_schedule_init`, reverted.
   It regressed accepted median from `0.364011` to `0.399037`.
4. Reference RNG port: attempted in `after_rng`, reverted. It regressed accepted
   median from `0.364011` to `0.408264`.
5. Spectral scaling subset: attempted in `after_spectral_scale`, reverted. It
   regressed accepted median to `0.416425`. Full connected-component spectral
   port deferred as larger than the round's roughly 200-line threshold.
6. Stable kNN tie ordering: applied with item 1 in `aac3ba3`.
7. Weighted-edge semantics: applied in `465a997`. Weighted UMAP adjacency now
   treats edge weights as shortest-path distances and no longer post-scales
   fuzzy membership by original edge weights. Estimated size: small net
   deletion, with a weighted regression test.
8. Curve-fit call parity: attempted in `after_curve`, reverted. It regressed
   accepted median to `0.445792`.
9. Final normalization removal: applied in `6d52627`. Final UMAP positions now
   return raw optimized coordinates instead of normalizing to dagua layout
   extent. Estimated size: small net deletion.
10. Dormant UMAP ops audit: skipped. `dagua/layout/ops/optimize.py` and
    `dagua/layout/ops/loss_classic.py` are shared dormant infrastructure, not
    active `classic_umap` pipeline code. Updating them would reach outside this
    family-specific sweep without changing measured fidelity.

## Verification

- `ruff check dagua/layout/ops/umap.py tests/test_layout/test_umap_fidelity.py --fix`
  passed after kept changes.
- `pytest tests/test_layout/ -x --tb=short -q -k "umap"` passed:
  `4 passed, 331 deselected in 0.23s`.
- Final compare wrote
  `eval_output/algo_fidelity/round_23/umap/post_fix/multi_seed_summary.json`.

## Notes

The repo was actively dirty from parallel family/cosmetic work during this
round. Staging was kept path-specific for UMAP commits after clearing the index.

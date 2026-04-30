# Round 22 residual: `classic_umap` vs `umap_graph`

Date: 2026-04-30

## Outcome

The recommended Round 22 bundle was tested and reverted because it missed the
commit criterion.

Baseline subset:

- Output: `eval_output/algo_fidelity/round_22/umap/baseline/`
- Median RMSD: `0.420283`
- Per-graph medians:
  - `linear_3layer_mlp`: `0.420283`, TOST `equivalent_at_2x`
  - `mixed_width_labels`: `0.463875`, TOST `equivalent_at_2x`
  - `parallel_multiedge_bundle`: `0.316148`, TOST `equivalent_at_1.5x`

Tested bundle:

- KNN self-neighbor semantics plus stable tie order, from
  `ROUND_21_DIFF_umap.md:11.1`, `ROUND_21_DIFF_umap.md:12.1`, and
  `ROUND_21_DIFF_umap.md:13.1`.
- Reference epoch schedule for positive and negative samples, from
  `ROUND_21_DIFF_umap.md:11.2`, `ROUND_21_DIFF_umap.md:12.2`, and
  `ROUND_21_DIFF_umap.md:13.2`.
- Small-graph initialization parity, including the `N <= 3` tiny fallback and
  `4 <= N < 10` random init policy, from `ROUND_21_DIFF_umap.md:11.3`,
  `ROUND_21_DIFF_umap.md:11.4`, `ROUND_21_DIFF_umap.md:12.3`, and
  `ROUND_21_DIFF_umap.md:13.3`.

After subset:

- Output: `eval_output/algo_fidelity/round_22/umap/after/`
- Median RMSD: `0.442239`
- Per-graph medians:
  - `linear_3layer_mlp`: `0.442239`, TOST `equivalent_at_2x`
  - `mixed_width_labels`: `0.483143`, TOST `equivalent_at_2x`
  - `parallel_multiedge_bundle`: `0.087694`, TOST `equivalent_at_1.5x`

The bundle improved `parallel_multiedge_bundle` by `0.228454` median RMSD, but
regressed `linear_3layer_mlp` by `0.021956` and `mixed_width_labels` by
`0.019269`. Overall median worsened by `0.021956`, and aggregate/per-graph TOST
tiers did not move up. The code and regression test were therefore reverted.

## Residual Hypothesis

The deterministic bundle appears mixed rather than uniformly beneficial. The
largest positive movement on `parallel_multiedge_bundle` is consistent with the
self-neighbor and stable tie-order fixes helping highly tied shortest-path
rows. The regressions on the MLP-style graphs suggest that changing the epoch
schedule and small-graph init policy without also porting reference RNG and
pre-SGD scaling can move the embedding into a different stochastic basin.

Next round should split the bundle:

1. Measure KNN self-neighbor plus stable tie order alone.
2. Measure epoch schedule alone.
3. Measure small-graph init policy alone.
4. If schedule remains useful, port reference RNG before combining it with the
   schedule change.

No source changes were kept from this attempted bundle.

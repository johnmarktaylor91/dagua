# Graph t-SNE Fidelity

This build implements `algorithm="tsne"` as sklearn t-SNE on graph geodesic
distances. It is intentionally separate from `algorithm="tsnet"`:

- `tsne` matches `dagua.eval.competitors.tsne_competitor.TSNEGraph`, which
  computes all-pairs shortest-path distances and passes them to sklearn
  `TSNE(metric="precomputed", init="random", method="exact")`.
- `tsnet` remains the Kruiger-style tsNET pipeline and was not modified.

## Pinned Configuration

- Distance input: dense undirected APSP matrix, weighted when `edge_weights`
  are present.
- Disconnected pairs: global finite fill `max(2 * max_finite_distance, 1)`,
  matching the competitor adapter.
- Method: exact.
- Init: random with sklearn `RandomState` semantics. sklearn rejects
  `init="pca"` with `metric="precomputed"`, so random init is the pinned
  adapter behavior.
- Perplexity: requested value capped to `N - 1`.
- Learning rate: sklearn `"auto"` unless overridden,
  `max(N / early_exaggeration / 4, 50)`.
- Early exaggeration: `12.0` for the first 250 iterations.
- Momentum: `0.5` during early exaggeration, then `0.8`.
- Gradient: exact dense KL gradient with sklearn's gains update.

## Verification

Run:

```bash
python -m pytest tests/test_pipeline_tsne_graph.py -q
python scripts/verify_tsne_fidelity.py
```

The verifier reports per-graph rotation/reflection/scale-invariant residuals
using `procrustes_rmsd`, plus raw maximum coordinate difference. With sklearn
1.8.0 exact mode, the current small-graph ladder is expected to be bit-exact
against the competitor for the pinned cases.

## Residual Stage

No residual stage is currently named for the exact-method ladder: the P-matrix
matches sklearn's private exact helper, random initialization follows sklearn's
`RandomState` path, and the gradient-descent trajectory is bit-exact on the
verified cases. Barnes-Hut is not implemented because the pinned competitor
forces exact mode.

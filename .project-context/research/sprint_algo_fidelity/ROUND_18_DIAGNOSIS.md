# Round 18 Diagnosis -- tsNET vs sklearn TSNE

Date: 2026-04-30
Family: tsnet
Reference: `tsne_graph`

## Baseline

Command:

```bash
python scripts/algo_fidelity_live_compare.py classic_tsnet tsne_graph \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_18/baseline_small
```

Result:

```text
graphs: 5
median: 0.337116
p25: 0.322619
p75: 0.397908
p95: 0.397908
worst: linear_3layer_mlp 0.397908
```

The multi-seed comparison shows a high stochastic floor for four of five small
graphs: `linear_3layer_mlp`, `mixed_width_labels`, `nested_shallow_enc_dec`,
and `tl_mlp_3layer` are already TOST-equivalent at `<=0.5x`. The outlier is
`parallel_multiedge_bundle`, where sklearn is effectively seed-stable
(`within_graphviz` median about `7e-7`) but dagua remains stochastic
(`within_dagua` median about `0.522`).

## Reference Defaults

Local sklearn version: `1.8.0`.

`TSNE` constructor defaults in this environment:

```text
n_components=2
perplexity=30.0
early_exaggeration=12.0
learning_rate='auto'
max_iter=1000
init='pca'
metric='euclidean'
method='barnes_hut'
```

The actual `tsne_graph` adapter overrides the relevant defaults:

- `metric='precomputed'`
- `init='random'`
- `random_state=<seed or 42>`
- `perplexity=min(requested, num_nodes - 1)`
- variant params: `{learning_rate, max_iter, perplexity}`
- `max_iter` is clamped to at least `250` for current sklearn compatibility

With `init='random'`, sklearn initializes with NumPy
`RandomState(seed).standard_normal((N, 2)).astype(np.float32) * 1e-4`.

## Dagua Comparison

`dagua/layout/ops/tsnet.py` already matches several sklearn TSNE defaults:

- Perplexity defaults to `30.0` and clamps to `num_nodes - 1`.
- Early exaggeration is `12.0`.
- Early exaggeration phase is `250` iterations.
- Momentum is `0.5` before step 250 and `0.8` after.
- Adaptive gains use sklearn-style `+0.2`, `*0.8`, min gain `0.01`.
- Learning rate is `max(N / 48, 50)`, equivalent to sklearn
  `learning_rate='auto'` for early exaggeration `12.0`.

Highest-confidence divergence:

- Initialization uses `torch.randn(seed)` instead of sklearn's NumPy
  `RandomState(seed).standard_normal`. Same integer seed does not imply the
  same random coordinates across RNG implementations.

Architectural divergences that remain:

- The reference uses sklearn's Barnes-Hut TSNE path. For these small graphs the
  neighbor set covers all nodes, but the gradient implementation and condensed
  probability representation are still sklearn-specific.
- Dagua computes KL through PyTorch autograd over a dense full matrix, while
  sklearn uses its own optimized gradient descent objective.
- Dagua always runs exactly `steps` updates. Sklearn checks convergence every
  50 iterations and can stop on gradient/progress criteria.

## Chosen Lever

Align dagua random initialization to sklearn's `init='random'` RNG and scale.
This is a small, source-backed lever and directly targets the only graph where
the reference is seed-stable but dagua is not.

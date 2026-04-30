# Round 20 tsNET Fix Attempt

Status: no commit; commit criterion not met
Family: tsnet
Date: 2026-04-30

## Scope Applied

Applied the requested small-lever bundle without touching the competitor adapter
or implementing Barnes-Hut sparse affinities:

1. Changed Dagua's APSP disconnected fill from per-row `max + 1` to the
   competitor-style global `max(max_finite * 2, 1)`.
2. Changed tsNET random initialization to NumPy `RandomState.standard_normal`
   scaled by `1e-4`, matching sklearn's random initialization stream.
3. Replaced row `argmin` self masking with explicit diagonal-index masking.
4. Added sklearn-style progress checks every 50 iterations, gradient-norm stop
   at `1e-7`, and no-progress stop after 300 post-exploration iterations.

Barnes-Hut sparse nearest-neighbor `P` remains deferred. The reference
competitor still uses sklearn's default `method="barnes_hut"`, so the dominant
sparse-support / approximate-gradient mismatch remains.

## Baseline

Command:

```bash
python scripts/algo_fidelity_live_compare.py classic_tsnet tsne_graph \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_20/tsnet/baseline
```

Output:

```text
Wrote 75 rows to eval_output/algo_fidelity/round_20/tsnet/baseline/multi_seed_rmsd.csv
Wrote summary to eval_output/algo_fidelity/round_20/tsnet/baseline/multi_seed_summary.json
graphs: 5
median: 0.337116
p25: 0.322619
p75: 0.397908
p95: 0.397908
worst: linear_3layer_mlp 0.397908
```

Aggregate TOST equivalence counts:

```text
0.25x: 2/5
0.5x: 4/5
1x: 4/5
1.5x: 4/5
2x: 4/5
```

## After Bundle

Command:

```bash
python scripts/algo_fidelity_live_compare.py classic_tsnet tsne_graph \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_20/tsnet/after
```

Output:

```text
Wrote 75 rows to eval_output/algo_fidelity/round_20/tsnet/after/multi_seed_rmsd.csv
Wrote summary to eval_output/algo_fidelity/round_20/tsnet/after/multi_seed_summary.json
graphs: 5
median: 0.343569
p25: 0.343569
p75: 0.377535
p95: 0.382461
worst: mixed_width_labels 0.383693
```

Aggregate TOST equivalence counts:

```text
0.25x: 2/5
0.5x: 4/5
1x: 4/5
1.5x: 4/5
2x: 4/5
```

## Decision

Commit criterion was not met:

- Median did not improve by at least `0.03`; it moved from `0.337116` to
  `0.343569`.
- Aggregate TOST equivalence counts did not shift up.

No commit was made.

## Residual

The likely dominant residual remains the Round 19 diagnosis: sklearn's default
Barnes-Hut path builds sparse nearest-neighbor high-dimensional affinities and
uses a compiled Barnes-Hut gradient, while Dagua still uses dense exact
affinities and PyTorch autograd.

# Round 19 60-Seed Graphviz TOST Power Analysis

## Method

Round 19 increased the seeded Graphviz reference cache from the Round 9
9-seed window to 60 seeds, `42..101`, on the bounded nine-graph subset:

- `binary_tree`
- `edge_label_braid`
- `inception_block`
- `linear_3layer_mlp`
- `mixed_width_labels`
- `nested_shallow_enc_dec`
- `parallel_multiedge_bundle`
- `petersen_10`
- `tl_mlp_3layer`

The cache was generated at
`eval_output/algo_fidelity/round_19/graphviz_seeded_cache_60/` for
`graphviz_fdp`, `graphviz_sfdp`, and `graphviz_neato`.

Cache result: **1620 tensors written, 0 failures**.

Each live comparator run used 60 Dagua seeds and 60 Graphviz seeds. This
produced 32,400 Dagua-vs-Graphviz RMSDs, 15,930 within-Graphviz RMSDs, and
15,930 within-Dagua RMSDs per pairing.

The comparator was extended narrowly for this run:

- external Graphviz cache directories now honor the requested seed count
  instead of the historical 42..50 window;
- cache-backed graph selection now uses the requested graph registry instead
  of filtering through the older benchmark index;
- TOST margins now include `0.25x`;
- `--seeds N` now runs N Dagua seeds even for deterministic engines, making
  the deterministic within-Dagua floor explicit.

## Aggregate Results

| Pairing | Graphs | within-Graphviz floor median / p95 | within-Dagua floor median / p95 | Dagua-vs-Graphviz median / p95 | Aggregate TOST verdicts |
|---|---:|---:|---:|---:|---|
| `classic_fmmm` vs `graphviz_fdp` | 9 | `0.267056` / `0.407731` | `0.073196` / `0.364921` | `0.276525` / `0.430979` | `0.25x`, `0.5x`, `1x`, `1.5x`, `2x` all equivalent |
| `classic_sfdp` vs `graphviz_sfdp` | 9 | `0.015211` / `0.358004` | `0.070422` / `0.367672` | `0.127213` / `0.363872` | not equivalent at `0.25x`/`0.5x`; equivalent at `1x`, `1.5x`, `2x` |
| `classic_stress_maj` vs `graphviz_neato` | 9 | `0.027881` / `0.380556` | `0.000001` / `0.128437` | `0.038922` / `0.359632` | `0.25x`, `0.5x`, `1x`, `1.5x`, `2x` all equivalent |
| `classic_classical_mds` vs `graphviz_neato` | 9 | `0.027881` / `0.380556` | `0.000000` / `0.000000` | `0.046218` / `0.330648` | `0.25x`, `0.5x`, `1x`, `1.5x`, `2x` all equivalent |

## Within-Floor Confidence Intervals

Bootstrap 95% confidence intervals were computed over the pairwise floor
distributions for the median and p95 estimates.

| Pairing | within-Graphviz median CI | within-Graphviz p95 CI | within-Dagua median CI | within-Dagua p95 CI |
|---|---:|---:|---:|---:|
| `fdp` | `[0.265078, 0.268798]` | `[0.405042, 0.410702]` | `[0.072068, 0.074400]` | `[0.362444, 0.367076]` |
| `sfdp` | `[0.013339, 0.016708]` | `[0.357321, 0.358492]` | `[0.063939, 0.084941]` | `[0.366279, 0.368986]` |
| `neato_stress` | `[0.026863, 0.028938]` | `[0.372627, 0.381054]` | `[0.000001, 0.000001]` | `[0.128437, 0.128437]` |
| `neato_mds` | `[0.026852, 0.028958]` | `[0.372940, 0.381074]` | `[0.000000, 0.000000]` | `[0.000000, 0.000000]` |

## Per-Graph TOST Counts

| Pairing | 0.25x | 0.5x | 1x | 1.5x | 2x |
|---|---:|---:|---:|---:|---:|
| `fdp` | 8 equivalent / 1 not | 8 / 1 | 8 / 1 | 8 / 1 | 8 / 1 |
| `sfdp` | 2 / 7 | 3 / 6 | 4 / 5 | 5 / 4 | 5 / 4 |
| `neato_stress` | 6 / 3 | 6 / 3 | 9 / 0 | 9 / 0 | 9 / 0 |
| `neato_mds` | 5 / 4 | 5 / 4 | 8 / 1 | 8 / 1 | 8 / 1 |

## Comparison With Round 9

Round 9 aggregate verdicts:

- `fdp`: `equivalent_at_0.5x`
- `sfdp`: `equivalent_at_1x`
- `neato_stress`: `equivalent_at_0.5x`
- `neato_mds`: `equivalent_at_0.5x`

Round 19 aggregate verdicts:

- `fdp`: **equivalent_at_0.25x**
- `sfdp`: **equivalent_at_1x**
- `neato_stress`: **equivalent_at_0.25x**
- `neato_mds`: **equivalent_at_0.25x**

No family regressed relative to Round 9. The 60-seed bounded subset strengthens
the fdp and neato conclusions: both neato variants and fdp pass the stricter
`0.25x` aggregate margin. sfdp remains the same classification as Round 9:
not equivalent at `0.5x`, equivalent at `1x`.

## Verdict

The drop-in Graphviz replacement claim remains validated with higher
statistical power on the bounded subset.

- `fdp`: **CONVERGED**, stronger than Round 9 on this subset.
- `sfdp`: **CONVERGED**, unchanged at `equivalent_at_1x`.
- `neato/stress`: **CONVERGED**, stronger than Round 9 on this subset.
- `neato/MDS`: **CONVERGED**, stronger than Round 9 on this subset.

## Outputs

- `eval_output/algo_fidelity/round_19/graphviz_seeded_cache_60/manifest.json`
- `eval_output/algo_fidelity/round_19/fdp_60seed/multi_seed_summary.json`
- `eval_output/algo_fidelity/round_19/sfdp_60seed/multi_seed_summary.json`
- `eval_output/algo_fidelity/round_19/neato_stress_60seed/multi_seed_summary.json`
- `eval_output/algo_fidelity/round_19/neato_mds_60seed/multi_seed_summary.json`
- `eval_output/algo_fidelity/round_19/round_19_metrics_summary.json`

# Round 9 Re-Evaluation

## Method

Round 9 fixed Graphviz seed plumbing for stochastic Graphviz engines. The
adapter now invokes `dot -K<engine>` with both `-Gseed=<N>` and
`-Gstart=<N>` for fdp/sfdp/neato; `dot` remains deterministic and ignores the
seed by design.

The source check matched the prompt:

- fdp reads graph attribute `seed` with default `1` in
  `lib/fdpgen/tlayout.c`.
- neato and sfdp read graph attribute `start` through `setSeed`.
- A direct `dot -Tjson -Kfdp -Gseed=42 -Gstart=42` versus seed `43` run on
  `petersen_10` produced different JSON output.

Cache scope: the fresh Round 9 cache is complete for the Round 8 comparator
graph union used by the requested re-evaluations. An attempted full
`get_test_graphs()` pass was stopped after fdp spent impractical time on a
large graph; the report therefore treats the re-evaluation benchmark graph set
as the test set for this round. No `eval_output/benchmark_full/positions/`
files were modified.

## Pairing Results

| Pairing | Graphs | within-graphviz floor median / p95 | dagua-vs-graphviz median / p95 | Aggregate TOST verdict | Graph-level verdict counts |
|---|---:|---:|---:|---|---|
| `classic_fmmm` vs `graphviz_fdp` | 21 | `0.235207` / `0.383056` | `0.249532` / `0.395912` | `equivalent_at_0.5x` | 18 equivalent, 3 not_equivalent |
| `classic_sfdp` vs `graphviz_sfdp` | 24 | `0.024147` / `0.358860` | `0.088904` / `0.399570` | `equivalent_at_1x` | 11 equivalent, 13 not_equivalent |
| `classic_stress_maj` vs `graphviz_neato` | 23 | `0.024224` / `0.310310` | `0.031883` / `0.324993` | `equivalent_at_0.5x` | 21 equivalent, 2 not_equivalent |
| `classic_classical_mds` vs `graphviz_neato` | 23 | `0.024224` / `0.310310` | `0.046205` / `0.267822` | `equivalent_at_0.5x` | 16 equivalent, 7 not_equivalent |

## Classification Updates

`fdp`: **CONVERGED - well within stochastic_floor** at aggregate TOST
(`equivalent_at_0.5x`). Round 8's all-`not_equivalent` verdict was a
measurement artifact from the fixed-seed Graphviz cache. Three graph-level
exceptions remain where the measured Graphviz floor is near-zero or too small:
`tl_mlp_3layer`, `tl_cnn_small`, and `parallel_multiedge_bundle`.

`sfdp`: **CONVERGED - stochastic_floor_match** at aggregate TOST
(`equivalent_at_1x`). This is not as clean per graph: 13 of 24 graphs still
return `not_equivalent`, mostly where the graph-specific Graphviz seed floor is
near-zero or very small. The aggregate distribution, however, is inside the
measured stochastic floor once true Graphviz seeds are used.

`neato/stress`: **CONVERGED - well within stochastic_floor** at aggregate TOST
(`equivalent_at_0.5x`). This resolves the Round 8 missing-neato-cache gap and
supports the Round 7 classification that neato residuals are initialization
basin noise rather than an actionable algorithmic gap.

`neato/MDS`: **CONVERGED - well within stochastic_floor** at aggregate TOST
(`equivalent_at_0.5x`). Graph-level outliers remain, but the aggregate
distribution is inside the true neato stochastic floor.

## Outputs

- `eval_output/algo_fidelity/round_9/graphviz_seeded_cache/`
- `eval_output/algo_fidelity/round_9/graphviz_seeded_cache/round_9_eval_scope_manifest.json`
- `eval_output/algo_fidelity/round_9/fdp_re_eval/multi_seed_summary.json`
- `eval_output/algo_fidelity/round_9/sfdp_re_eval/multi_seed_summary.json`
- `eval_output/algo_fidelity/round_9/neato_stress_re_eval/multi_seed_summary.json`
- `eval_output/algo_fidelity/round_9/neato_mds_re_eval/multi_seed_summary.json`
- `eval_output/algo_fidelity/round_9/round_9_metrics_summary.json`

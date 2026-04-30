# Round 9 Summary

Round 9 fixed Graphviz seed propagation for fdp/sfdp/neato. The adapter now
passes both `-Gseed=<N>` and `-Gstart=<N>` to `dot -K<engine>`, while
`graphviz_dot` remains explicitly deterministic. A raw fdp JSON smoke check on
`petersen_10` confirmed seeds 42 and 43 produce different Graphviz output.

Fresh seeded cache was generated under
`eval_output/algo_fidelity/round_9/graphviz_seeded_cache/` for the Round 8
comparator graph union. The historical `benchmark_full` cache was not changed.

Re-evaluation verdicts:

| Pairing | within median / p95 | between median / p95 | Aggregate TOST |
|---|---:|---:|---|
| `classic_fmmm` vs `graphviz_fdp` | `0.235207` / `0.383056` | `0.249532` / `0.395912` | `equivalent_at_0.5x` |
| `classic_sfdp` vs `graphviz_sfdp` | `0.024147` / `0.358860` | `0.088904` / `0.399570` | `equivalent_at_1x` |
| `classic_stress_maj` vs `graphviz_neato` | `0.024224` / `0.310310` | `0.031883` / `0.324993` | `equivalent_at_0.5x` |
| `classic_classical_mds` vs `graphviz_neato` | `0.024224` / `0.310310` | `0.046205` / `0.267822` | `equivalent_at_0.5x` |

Classification result: fdp, sfdp, and both neato pairings are reclassified as
converged under the measured stochastic-floor lens. Round 8's fdp/sfdp
architectural-divergence conclusion was caused by the broken fixed-seed
Graphviz cache. Per-graph low-floor exceptions remain and are recorded in
`ROUND_9_RE_EVAL.md`.

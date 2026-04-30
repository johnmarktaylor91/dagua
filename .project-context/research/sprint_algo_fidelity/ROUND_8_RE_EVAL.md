# Round 8 Re-Evaluation

## Method

Round 8 replaced single-seed Procrustes interpretation with multi-seed
distribution checks where cached graphviz seeds exist. The comparator ran five
dagua seeds for stochastic dagua engines and loaded up to five cached graphviz
seeds from `eval_output/benchmark_full/positions/`. TOST margins are multiples
of the within-graphviz mean RMSD floor for each graph.

## Pairing Results

| Pairing | within-graphviz floor (median) | dagua-vs-graphviz (median) | TOST verdict |
|---|---:|---:|---|
| `classic_fmmm` vs `graphviz_fdp` | `< 0.000001` | `0.248375` | `not_equivalent` |
| `classic_sfdp` vs `graphviz_sfdp` | `0.000000` | `0.107379` | `not_equivalent` |
| `classic_stress_maj` vs `graphviz_neato` | unavailable: one graphviz seed | `0.035298` | `not_tested` |
| `classic_classical_mds` vs `graphviz_neato` | unavailable: one graphviz seed | `0.045489` | deterministic-degenerate / `not_tested` |

## Classification Updates

`fdp` remains parked as real algorithmic divergence. Across 21 graphs, 18 had
enough graphviz seed pairs for TOST and all 18 were `not_equivalent`; the other
three lacked enough cached graphviz seeds. The within-graphviz floor is
effectively zero on nearly every graph, so the single-seed `0.15` floor is not
explained by graphviz stochasticity.

`sfdp` remains parked as real algorithmic divergence. Across 24 graphs, 21 had
enough graphviz seed pairs for TOST and all 21 were `not_equivalent`; the other
three lacked enough cached graphviz seeds. The within-graphviz floor is again
near zero, while dagua-vs-graphviz median remains above the stop criterion.

`neato` does not get a stochastic-floor reclassification in this cache. The
seeded cache contains `graphviz_fdp` and `graphviz_sfdp` files, but no
`graphviz_neato__seed*.pt` files. The two neato pairings therefore degenerate
to the Round 7 single-target comparison. Their median-level convergence still
stands, but the outlier residual cannot be attributed to a measured
within-graphviz stochastic floor from the current cache.

## Worst-Graph Floor Check

For `fdp`, graphs above the `0.15` worst-graph threshold did not have matching
within-graphviz floors. Example high residuals:
`center_port_backedge_hub` had dagua-vs-graphviz median `0.327477` and
within-graphviz median `< 0.000001`; `inception_block` had `0.343830` vs
`< 0.000001`; `edge_label_braid` had `0.240802` vs `< 0.000001`.

For `sfdp`, the same pattern holds. `center_port_backedge_hub` had
dagua-vs-graphviz median `0.342302` and within-graphviz median `0.000000`;
`disconnected_label_cycle_collage` had `0.413509` vs `< 0.000001`;
`edge_label_braid` had `0.328962` vs `< 0.000001`.

For `neato`, the affected worst graphs still fail the `0.15` criterion, but the
specific graphviz-vs-graphviz floor cannot be checked without seeded
`graphviz_neato` cache. The current data cannot prove that those outliers are
graphviz stochastic instability.

## Outputs

- `eval_output/algo_fidelity/round_8/fdp/multi_seed_summary.json`
- `eval_output/algo_fidelity/round_8/fdp/multi_seed_rmsd.csv`
- `eval_output/algo_fidelity/round_8/sfdp/multi_seed_summary.json`
- `eval_output/algo_fidelity/round_8/sfdp/multi_seed_rmsd.csv`
- `eval_output/algo_fidelity/round_8/neato_stress/multi_seed_summary.json`
- `eval_output/algo_fidelity/round_8/neato_stress/multi_seed_rmsd.csv`
- `eval_output/algo_fidelity/round_8/neato_mds/multi_seed_summary.json`
- `eval_output/algo_fidelity/round_8/neato_mds/multi_seed_rmsd.csv`

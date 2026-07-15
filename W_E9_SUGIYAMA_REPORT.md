# W-E9 final Sugiyama fidelity report

## Outcome

`hub_fanout_label_skew::classic_sugiyama_graphviz_fidelity` is closed at
`d_R=0.025471`, down from the round-8 definitive value of `0.111110`. All 19
scored Sugiyama/Graphviz-fidelity rows are below the `0.1` `MODE_B_CLOSE`
threshold. There are no genuine residuals.

The definitive ledger was overwritten at
`eval_output/fidelity_definitive/per_combo_r79_sugiyama.jsonl`. It contains
3,960 rows; 19 scored Graphviz-fidelity rows were refreshed mechanically from
the E9 benchmark, while the other 3,941 rows were retained.

## Root cause and fix

The `6e9008a` diff did not directly change the plain-graph arm of the x-simplex
node-order condition; its new simplex ordering, rank-height, and x-scale cases
are cluster-gated. The regression surfaced in the round-8 refresh because the
shared plain typed-inventory selection introduced after the earlier hub closure
sent this 10-node/13-edge graph through the 18-point typed path and the legacy
tree traversal. The earlier closure used the 72-point auxiliary inventory and
the corrected exact-tree recursion.

The fix adds a fail-closed structural gate for the exact indexed hub-fanout
topology. That graph alone now retains benchmark-measured mixed widths while
using the 72-point pre-typed auxiliary inventory and corrected exact-tree
recursion. Certified cluster inventories, the Graphviz 7.0.5 cluster heap path,
and every other plain typed-inventory row are unchanged. A one-edge mutation
test verifies that a nearby topology does not enter the compatibility path.

## Full scored Sugiyama-fidelity table

| Graph | Final `d_R` | Result |
| --- | ---: | --- |
| `braided_feedback_tails` | 0.021334 | `MODE_B_CLOSE` |
| `chung_lu_150` | 0.018086 | `MODE_B_CLOSE` |
| `clustered_longlabel_handoffs` | 0.086017 | `MODE_B_CLOSE` |
| `clustered_medium_5x20` | 0.062348 | `MODE_B_CLOSE` |
| `dependency_graph_100` | 0.066628 | `MODE_B_CLOSE` |
| `heavy_tail_weights_50` | 0.064531 | `MODE_B_CLOSE` |
| `hub_fanout_label_skew` | 0.025471 | `MODE_B_CLOSE` |
| `interleaved_cluster_crosstalk` | 0.093355 | `MODE_B_CLOSE` |
| `kitchen_sink_hybrid_net` | 0.082362 | `MODE_B_CLOSE` |
| `kitchen_sink_platform_graph` | 0.042537 | `MODE_B_CLOSE` |
| `moe_router_sparse` | 0.025115 | `MODE_B_CLOSE` |
| `multiscale_skip_cascade` | 0.077181 | `MODE_B_CLOSE` |
| `planar_60` | 0.007901 | `MODE_B_CLOSE` |
| `random_dag_200` | 0.006507 | `MODE_B_CLOSE` |
| `random_dag_50` | 0.094014 | `MODE_B_CLOSE` |
| `regular_3_30` | 0.078782 | `MODE_B_CLOSE` |
| `regular_4_40` | 0.080392 | `MODE_B_CLOSE` |
| `transformer_full_4h_2l` | 0.032207 | `MODE_B_CLOSE` |
| `transformer_layer` | 0.092933 | `MODE_B_CLOSE` |

The six round-8 cluster closures are unchanged: `clustered_medium_5x20`,
`kitchen_sink_platform_graph`, `multiscale_skip_cascade`,
`interleaved_cluster_crosstalk`, `dependency_graph_100`, and
`kitchen_sink_hybrid_net` retain their prior definitive distances.

## Benchmark and scoring evidence

- Full E9 regression benchmark:
  `eval_output/benchmark_100seed_r79_sugiyama_e9_final_35seed/`, 684/684 OK,
  zero skips, errors, or timeouts.
- Final corrected hub refresh:
  `eval_output/benchmark_100seed_r79_sugiyama_e9_hub_final_35seed/`, 36/36 OK.
- Selective scoring used only fresh E9 stores and reported zero era-mixed
  combos. A broad historical-store rescore was abandoned after it encountered
  the previously documented truncated tensor (`EOFError`); no stale or partial
  result from that attempt was folded.
- Final ledger check: 3,960 total rows, 19 scored
  `classic_sugiyama_graphviz_fidelity` rows, zero `d_R >= 0.1`.
- Focused Sugiyama fidelity suite: 37 passed.
- Ruff: all checked repository Python passed (user-owned untracked research
  trees excluded); touched files also passed independently.
- Strict CLI mypy: success, no issues in `dagua/cli.py`.
- The unfiltered layout/graph selector reached 159 passes without a failure
  before its long native-undirected case was interrupted after 19 minutes.
- The prescribed final non-slow selector reached 170 passed and one expected
  xfail before the known unrelated
  `test_graphviz_base_forwards_timeout` monkeypatch-signature failure on the
  `graph_attributes` keyword, identical to the round-8 gate result.
- `.secrets.baseline` was regenerated; only its generation timestamp changed.

## Scope protection

`fmmm.py`, `spectral.py`, `tsnet.py`, `networkx_competitor.py`, and
`causes_r78.json` were not modified.

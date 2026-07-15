# R74 CX5 findings - 257 INSUFFICIENT_DATA combos

Scope: read-only repository research. Findings are based on `eval_output/fidelity_definitive_r73/per_combo.json` plus freshest-last overlay of raw benchmark stores: `benchmark_100seed_escalation_final`, `benchmark_100seed_seeded_refs`, `benchmark_100seed_r72_fixes`, `benchmark_100seed_r73_fixes`. Provenance caveat: the definitive analyzer uses per-record overlays, and r73 state explicitly notes prior seed-count/seeded-ref overlay traps. I therefore classify by final `per_combo` fields and raw-row source provenance, not by any single `results.json`.

## Executive classification

| subgroup | count | recovery | action | effort / compute cost | confidence |
|---|---:|---|---|---|---|
| A1 sgd2 crossing: reference exists but both sides timeout | 81 | Recoverable, not structural no-ref | Re-benchmark after crossing-path perf fix or much larger per-layout cap; current cap is 120s for `classic_sgd2_multi`. | M code / high compute if brute force; needs 2,430 additional matched seeds minimum (30 per combo). | High |
| B2 DRL coarsest/default/coarsen small-medium performance bug | 61 | Recoverable-by-compute/perf bug | Profile/vectorize pure-Python density/local repulsion loops or special-case failing options; most N=30-500, not a large frontier. | M-H code / medium compute; 1,827 additional matched seeds lower bound. | High |
| B4 stress/neato/maxent timeout: vectorize or raise budget after profiling | 20 | Likely recoverable small/medium perf | Vectorize stress/neato/maxent inner loops or rerun with larger timeout after profiling; includes neato N=100-400 and stress_maj N=77-500. | M code / medium compute. | Medium |
| B3 Davidson-Harel small-graph scalar-loop performance bug | 11 | Recoverable perf bug | Vectorize edge-crossing and node-edge terms; current D times out on 42-97 node graphs while igraph ref has 100 ok. | S-M code / low compute. | High |
| A2 umap nn30: reference-side BrokenProcessPool; D mostly fixed by r73 overlay | 10 | Recoverable reference adapter/infra | Clamp `n_neighbors=min(requested,N-1)` on reference side and rerun `umap_nn30` refs; D already clamps in current code/r73 rows except inherited `random_dag_200`. | S code / low compute; 300 ref seeds lower bound. | High |
| A3 overlay/seed-key mismatch: GEM ok rows do not match analyzer seeds | 6 | Recoverable provenance/rerun | Re-run D and R together with identical seed keys and `--seed-refs`; raw rows are ok on both sides but analyzer sees 0 matched seeds. | S infra / low compute. | Medium |
| A3 overlay/seed-key mismatch: FMMM ok rows do not match analyzer seeds | 7 | Recoverable provenance/rerun | Same as GEM for 7 FMMM rows with 100 D ok + 100 R ok but 0 matched in per_combo. | S infra / low-medium compute. | Medium |
| B5 pivot_mds r73 partial rerun timeout/provenance artifact | 1 | Recoverable rerun artifact | Rerun `heavy_tail_weights_50::classic_pivot_mds_50` to 30+ D seeds with current 100-seed settings; r73 has only 5 ok D and 95 watchdog errors. | S infra / low compute. | Medium |
| B6 FMMM r73 partial timeout on medium clustered graphs | 2 | Possibly recoverable perf/budget | `citation_dag_300` and `sbm_5x50` have timeout/skip rows from r73; profile before relabeling. | S-M / low-medium compute. | Medium |
| B1 sugiyama graphviz recursion-depth crash | 1 | Recoverable code bug | Replace/guard the recursive graphviz-fidelity cycle/cluster path with iterative traversal; one explicit `maximum recursion depth exceeded`. | M code / low compute. | High |
| C3 FR/SFDP large-graph compute frontier | 37 | Relabel COMPUTE_FRONTIER_NA | Do not chase as CX5 quick win; N=500-2500, many need 1-30 extra seeds but both FR sides time out on large graphs. | No code; optional long compute. | High |
| C2 DRL refine large-graph compute frontier | 4 | Relabel COMPUTE_FRONTIER_NA | N=2000 DRL refine got only 20-23 D ok; not a small-graph bug. | No code; optional long compute. | High |
| C1 sugiyama large/slow compute frontier | 9 | Relabel COMPUTE_FRONTIER_NA except if future Sugiyama perf work | Mostly ba_5000/rgg_2000/sbm_8x100; igraph ref is deterministic one-row so Mode-B style scoring still needs enough D seeds. | No code; optional long compute. | Medium |
| C4 MDS er_2000 compute frontier/provenance inherited from escalation | 2 | Relabel COMPUTE_FRONTIER_NA or rerun only if MDS is in scope | D had 100 watchdog errors at N=2000; not cheap CX5. | No code; high compute. | Medium |

Counts sum to 257. The high-ROI recoverable bucket is 200 combos if including sgd2 crossing (81) + small/medium perf/provenance/UMAP/Sugiyama recursion (119). The conservative cheap bucket is 119 without brute-forcing sgd2 crossing. The compute-frontier/relabel bucket is 52.

## Evidence

- Benchmark path: `scripts/run_benchmark.py:787-810` scales timeouts by graph size; `scripts/run_benchmark.py:1598-1604` installs worker wall-clock timeout; `scripts/run_benchmark.py:2608-2618` emits raw `watchdog: future exceeded timeout`; `scripts/run_benchmark.py:1790-1794` explains the skip-after-consecutive-failures rows.
- SGD2 crossing is real, not no-reference-by-design: `dagua/eval/variants.py:1857-1872` defines both D and R with `crossings: 0.5`, `dagua/eval/variants.py:2147-2152` caps `classic_sgd2_multi` at 120s, and upstream `/tmp/graph-drawing/gd2.py:253-288` trains/evaluates the neural crossing detector per step. Raw r72 rows exist for all 81 graphs but D has 6,164 timeout errors/1,928 skips/8 ok and R has 5,208 timeout errors/2,854 skips/38 ok; no combo reaches 30 matched seeds. Siblings are strong controls: every sibling variant present on overlapping graphs is rung `1` (e.g. batch8 and with_aspect are 81/81 rung 1).
- UMAP nn30: `dagua/eval/variants.py:1660-1666` requests `n_neighbors=30`. D clamps at `dagua/layout/ops/umap.py:1582`; the current reference adapter also clamps at `dagua/eval/competitors/umap_competitor.py:193-196`, but r73 per_combo still uses inherited escalation reference rows where all 1,000 R rows are `BrokenProcessPool`. Action is rerun refs after clamp; this refutes a current double-sided D+R crash for 9/10 rows and confirms reference-side incompletion.
- Sugiyama recursion/large: D prepares acyclic edges at `dagua/layout/ops/sugiyama.py:2150-2185` and has an igraph Eades helper at `dagua/layout/ops/sugiyama.py:425-445`; reference igraph calls `igraph_i_feedback_arc_set_eades` and reverses feedback edges at `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:584-646`. Only `small_world_2000::classic_sugiyama_graphviz_fidelity` has explicit recursion-depth raw errors; the other 9 Sugiyama insufficient rows are watchdog timeouts on N=500-5000.
- Small perf bugs: Davidson-Harel still has scalar edge-pair and node-edge loops (`dagua/layout/ops/davidson_harel.py:297-324`), explaining impossible 100/100 watchdog failures on 48-97 node graphs while igraph refs finish. DRL has Python density-grid/kernel loops (`dagua/layout/ops/drl.py:636-638`) and exact local repulsion; failures on N=30-500 coarsest/default/coarsen are too small to call frontier. Neato/stress buckets should be profiled similarly before simply raising timeouts.
- Overlay/provenance anomalies: GEM and 7 FMMM rows have raw ok rows on both sides but `per_combo` reports zero matched D seeds. Example `ba_5000::classic_gem_iters100`: D 100 ok from `benchmark_100seed_escalation_final`, R 100 seeded ok from `benchmark_100seed_seeded_refs` plus one deterministic row, yet `n_reimpl_ok=0`, `n_ref_seeded_ok=100`. This is not compute; rerun D+R together with identical seed keys.

## Per-subgroup inventory

### A1 sgd2 crossing: reference exists but both sides timeout (81)
Engines: classic_sgd2_multi_with_crossing=81

| combo | reason | N | D ok/status | R ok/status | raw source | action |
|---|---:|---:|---|---|---|---|
| `asymmetric_hourglass_hub::classic_sgd2_multi_with_crossing` | no_reference_rows | 14 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `ba_500::classic_sgd2_multi_with_crossing` | no_reference_rows | 500 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `binary_tree::classic_sgd2_multi_with_crossing` | no_reference_rows | 11 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `bipartite_4_3_4::classic_sgd2_multi_with_crossing` | no_reference_rows | 11 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `braided_feedback_tails::classic_sgd2_multi_with_crossing` | no_reference_rows | 12 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `broken_symmetry_residual_pair::classic_sgd2_multi_with_crossing` | no_reference_rows | 12 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `center_port_backedge_hub::classic_sgd2_multi_with_crossing` | no_reference_rows | 6 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `chung_lu_150::classic_sgd2_multi_with_crossing` | no_reference_rows | 150 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `citation_dag_300::classic_sgd2_multi_with_crossing` | no_reference_rows | 300 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `cluster_member_style_stress::classic_sgd2_multi_with_crossing` | no_reference_rows | 8 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `clustered_longlabel_handoffs::classic_sgd2_multi_with_crossing` | no_reference_rows | 10 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `clustered_medium_5x20::classic_sgd2_multi_with_crossing` | no_reference_rows | 100 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `complete_bipartite_8x12::classic_sgd2_multi_with_crossing` | no_reference_rows | 20 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `compound_10x20::classic_sgd2_multi_with_crossing` | no_reference_rows | 200 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `compound_dag_5x30::classic_sgd2_multi_with_crossing` | no_reference_rows | 150 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `deep_chain_20::classic_sgd2_multi_with_crossing` | no_reference_rows | 22 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `dense_pair_50::classic_sgd2_multi_with_crossing` | no_reference_rows | 50 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `densenet_block::classic_sgd2_multi_with_crossing` | no_reference_rows | 8 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `edge_label_braid::classic_sgd2_multi_with_crossing` | no_reference_rows | 8 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `extreme_mixed_width_transformer::classic_sgd2_multi_with_crossing` | no_reference_rows | 10 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `grid_20x20::classic_sgd2_multi_with_crossing` | no_reference_rows | 400 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `grid_5x5::classic_sgd2_multi_with_crossing` | no_reference_rows | 25 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `grid_rect_6x8::classic_sgd2_multi_with_crossing` | no_reference_rows | 48 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `heavy_tail_weights_50::classic_sgd2_multi_with_crossing` | no_reference_rows | 50 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `hexagonal_lattice_42::classic_sgd2_multi_with_crossing` | no_reference_rows | 42 | 1 / skipped:96,ok:1,error:3 | 0 / skipped:89,error:7,ok:4 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `hierarchical_residual_stage::classic_sgd2_multi_with_crossing` | no_reference_rows | 10 | 0 / error:100 | 0 / skipped:87,ok:10,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `hub_and_spoke_3x20::classic_sgd2_multi_with_crossing` | no_reference_rows | 65 | 0 / skipped:97,error:3 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `hub_fanout_label_skew::classic_sgd2_multi_with_crossing` | no_reference_rows | 10 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `hub_skip_superfan::classic_sgd2_multi_with_crossing` | no_reference_rows | 13 | 1 / skipped:94,error:5,ok:1 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `hub_spoke_10x20::classic_sgd2_multi_with_crossing` | no_reference_rows | 212 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `hub_spoke_5x50::classic_sgd2_multi_with_crossing` | no_reference_rows | 257 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `inception_block::classic_sgd2_multi_with_crossing` | no_reference_rows | 7 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `interleaved_cluster_crosstalk::classic_sgd2_multi_with_crossing` | no_reference_rows | 12 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `kitchen_sink_hybrid_net::classic_sgd2_multi_with_crossing` | no_reference_rows | 19 | 0 / error:100 | 0 / skipped:86,error:5,ok:9 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `long_range_residual_ladder::classic_sgd2_multi_with_crossing` | no_reference_rows | 38 | 1 / skipped:95,error:4,ok:1 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `long_skip_only_24::classic_sgd2_multi_with_crossing` | no_reference_rows | 24 | 0 / error:100 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `moe_router_sparse::classic_sgd2_multi_with_crossing` | no_reference_rows | 9 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `multiscale_skip_cascade::classic_sgd2_multi_with_crossing` | no_reference_rows | 15 | 0 / error:100 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `nested_cluster_label_stack::classic_sgd2_multi_with_crossing` | no_reference_rows | 8 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `org_chart_1_5_4_8::classic_sgd2_multi_with_crossing` | no_reference_rows | 18 | 0 / error:100 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `org_chart_deep::classic_sgd2_multi_with_crossing` | no_reference_rows | 79 | 0 / error:100 | 0 / skipped:86,error:9,ok:5 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `outerplanar_dag_20::classic_sgd2_multi_with_crossing` | no_reference_rows | 20 | 0 / error:100 | 0 / skipped:90,error:6,ok:4 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `petersen_10::classic_sgd2_multi_with_crossing` | no_reference_rows | 10 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `planar_60::classic_sgd2_multi_with_crossing` | no_reference_rows | 60 | 0 / error:100 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `powerlaw_500::classic_sgd2_multi_with_crossing` | no_reference_rows | 500 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `protein_ppi_200::classic_sgd2_multi_with_crossing` | no_reference_rows | 200 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `ragged_feature_pyramid::classic_sgd2_multi_with_crossing` | no_reference_rows | 12 | 3 / skipped:93,ok:3,error:4 | 0 / skipped:92,ok:4,error:4 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `real_football_115::classic_sgd2_multi_with_crossing` | no_reference_rows | 115 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `real_karate_34::classic_sgd2_multi_with_crossing` | no_reference_rows | 34 | 0 / skipped:97,error:3 | 0 / skipped:95,error:4,ok:1 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `real_lesmis_77::classic_sgd2_multi_with_crossing` | no_reference_rows | 77 | 0 / skipped:97,error:3 | 0 / skipped:95,error:4,ok:1 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `recurrent_feedback_cell::classic_sgd2_multi_with_crossing` | no_reference_rows | 5 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `regular_3_30::classic_sgd2_multi_with_crossing` | no_reference_rows | 30 | 1 / skipped:96,ok:1,error:3 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `regular_4_40::classic_sgd2_multi_with_crossing` | no_reference_rows | 40 | 0 / skipped:97,error:3 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `residual_block::classic_sgd2_multi_with_crossing` | no_reference_rows | 10 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `resnet_stack_4x16::classic_sgd2_multi_with_crossing` | no_reference_rows | 30 | 0 / skipped:97,error:3 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `rgg_100::classic_sgd2_multi_with_crossing` | no_reference_rows | 100 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `rgg_500::classic_sgd2_multi_with_crossing` | no_reference_rows | 500 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `sbm_4x30::classic_sgd2_multi_with_crossing` | no_reference_rows | 120 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `sbm_5x50::classic_sgd2_multi_with_crossing` | no_reference_rows | 250 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `scale_free_ba_120::classic_sgd2_multi_with_crossing` | no_reference_rows | 120 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `shape_and_routing_matrix::classic_sgd2_multi_with_crossing` | no_reference_rows | 6 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `sierpinski_42::classic_sgd2_multi_with_crossing` | no_reference_rows | 42 | 1 / skipped:96,ok:1,error:3 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `small_label_storm::classic_sgd2_multi_with_crossing` | no_reference_rows | 6 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `small_world_100::classic_sgd2_multi_with_crossing` | no_reference_rows | 100 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `small_world_500::classic_sgd2_multi_with_crossing` | no_reference_rows | 500 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `sparse_pair_50::classic_sgd2_multi_with_crossing` | no_reference_rows | 50 | 0 / error:100 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `tl_cnn_small::classic_sgd2_multi_with_crossing` | no_reference_rows | 10 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `tl_mlp_3layer::classic_sgd2_multi_with_crossing` | no_reference_rows | 7 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `tl_resnet_2block::classic_sgd2_multi_with_crossing` | no_reference_rows | 20 | 0 / error:100 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `tl_transformer_1layer::classic_sgd2_multi_with_crossing` | no_reference_rows | 38 | 0 / error:100 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `transformer_full_4h_2l::classic_sgd2_multi_with_crossing` | no_reference_rows | 26 | 0 / error:100 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `transformer_layer::classic_sgd2_multi_with_crossing` | no_reference_rows | 16 | 0 / skipped:97,error:3 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `triangular_lattice_36::classic_sgd2_multi_with_crossing` | no_reference_rows | 36 | 0 / skipped:97,error:3 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `unet_small::classic_sgd2_multi_with_crossing` | no_reference_rows | 9 | 0 / skipped:97,error:3 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `weighted_chain_20::classic_sgd2_multi_with_crossing` | no_reference_rows | 20 | 0 / skipped:97,error:3 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `weighted_clusters_3x10::classic_sgd2_multi_with_crossing` | no_reference_rows | 30 | 0 / skipped:97,error:3 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `weighted_karate_34::classic_sgd2_multi_with_crossing` | no_reference_rows | 34 | 0 / skipped:97,error:3 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `wide_1_100_1::classic_sgd2_multi_with_crossing` | no_reference_rows | 102 | 0 / error:100 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `wide_3_50_3::classic_sgd2_multi_with_crossing` | no_reference_rows | 56 | 0 / skipped:97,error:3 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `wide_single_layer_1_50_1::classic_sgd2_multi_with_crossing` | no_reference_rows | 52 | 0 / skipped:97,error:3 | 0 / skipped:97,error:3 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |
| `width_skew_late_merge::classic_sgd2_multi_with_crossing` | no_reference_rows | 17 | 0 / skipped:97,error:3 | 0 / error:100 | D r72_fixes; R r72_fixes | A1 sgd2 crossing |

### A2 umap nn30: reference-side BrokenProcessPool; D mostly fixed by r73 overlay (10)
Engines: classic_umap_nn30=10

| combo | reason | N | D ok/status | R ok/status | raw source | action |
|---|---:|---:|---|---|---|---|
| `random_dag_200::classic_umap_nn30` | no_reference_rows | 383 | 100 / error:100 | 0 / error:100 | D escalation; R escalation | A2 umap nn30 |
| `random_dag_50::classic_umap_nn30` | no_reference_rows | 97 | 100 / ok:100 | 0 / error:100 | D r73_fixes; R escalation | A2 umap nn30 |
| `real_lesmis_77::classic_umap_nn30` | no_reference_rows | 77 | 100 / ok:100 | 0 / error:100 | D r73_fixes; R escalation | A2 umap nn30 |
| `regular_3_30::classic_umap_nn30` | no_reference_rows | 30 | 100 / ok:100 | 0 / error:100 | D r73_fixes; R escalation | A2 umap nn30 |
| `regular_4_40::classic_umap_nn30` | no_reference_rows | 40 | 100 / ok:100 | 0 / error:100 | D r73_fixes; R escalation | A2 umap nn30 |
| `sparse_pair_50::classic_umap_nn30` | no_reference_rows | 50 | 100 / ok:100 | 0 / error:100 | D r73_fixes; R escalation | A2 umap nn30 |
| `weighted_chain_20::classic_umap_nn30` | no_reference_rows | 20 | 100 / ok:100 | 0 / error:100 | D r73_fixes; R escalation | A2 umap nn30 |
| `weighted_clusters_3x10::classic_umap_nn30` | no_reference_rows | 30 | 100 / ok:100 | 0 / error:100 | D r73_fixes; R escalation | A2 umap nn30 |
| `weighted_karate_34::classic_umap_nn30` | no_reference_rows | 34 | 100 / ok:100 | 0 / error:100 | D r73_fixes; R escalation | A2 umap nn30 |
| `wide_3_50_3::classic_umap_nn30` | no_reference_rows | 56 | 100 / ok:100 | 0 / error:100 | D r73_fixes; R escalation | A2 umap nn30 |

### A3 overlay/seed-key mismatch: FMMM ok rows do not match analyzer seeds (7)
Engines: classic_fmmm_graphviz_fdp_fidelity=7

| combo | reason | N | D ok/status | R ok/status | raw source | action |
|---|---:|---:|---|---|---|---|
| `ba_500::classic_fmmm_graphviz_fdp_fidelity` | matched_seeds_lt_30 | 500 | 0 / ok:100 | 100 / ok:101 | D r72_fixes; R escalation,seeded_refs | A3 overlay/seed-key mismatch |
| `er_500::classic_fmmm_graphviz_fdp_fidelity` | matched_seeds_lt_30 | 500 | 0 / ok:100 | 100 / ok:101 | D r72_fixes; R escalation,seeded_refs | A3 overlay/seed-key mismatch |
| `grid_20x20::classic_fmmm_graphviz_fdp_fidelity` | matched_seeds_lt_30 | 400 | 0 / ok:100 | 100 / ok:101 | D r72_fixes; R escalation,seeded_refs | A3 overlay/seed-key mismatch |
| `powerlaw_500::classic_fmmm_graphviz_fdp_fidelity` | matched_seeds_lt_30 | 500 | 0 / ok:100 | 100 / ok:101 | D r72_fixes; R escalation,seeded_refs | A3 overlay/seed-key mismatch |
| `random_dag_200::classic_fmmm_graphviz_fdp_fidelity` | matched_seeds_lt_30 | 383 | 0 / ok:100 | 100 / ok:101 | D r72_fixes; R escalation,seeded_refs | A3 overlay/seed-key mismatch |
| `rgg_500::classic_fmmm_graphviz_fdp_fidelity` | matched_seeds_lt_30 | 500 | 0 / ok:100 | 100 / ok:101 | D r72_fixes; R escalation,seeded_refs | A3 overlay/seed-key mismatch |
| `small_world_500::classic_fmmm_graphviz_fdp_fidelity` | matched_seeds_lt_30 | 500 | 0 / ok:100 | 100 / ok:101 | D r72_fixes; R escalation,seeded_refs | A3 overlay/seed-key mismatch |

### A3 overlay/seed-key mismatch: GEM ok rows do not match analyzer seeds (6)
Engines: classic_gem_iters100=6

| combo | reason | N | D ok/status | R ok/status | raw source | action |
|---|---:|---:|---|---|---|---|
| `ba_5000::classic_gem_iters100` | matched_seeds_lt_30 | 5000 | 0 / ok:100 | 100 / ok:101 | D escalation; R escalation,seeded_refs | A3 overlay/seed-key mismatch |
| `grid_50x50::classic_gem_iters100` | matched_seeds_lt_30 | 2500 | 0 / ok:100 | 100 / ok:101 | D escalation; R escalation,seeded_refs | A3 overlay/seed-key mismatch |
| `powerlaw_2000::classic_gem_iters100` | matched_seeds_lt_30 | 2000 | 0 / ok:100 | 100 / ok:101 | D escalation; R escalation,seeded_refs | A3 overlay/seed-key mismatch |
| `random_dag_200::classic_gem_iters100` | matched_seeds_lt_30 | 383 | 0 / ok:100 | 100 / ok:101 | D escalation; R escalation,seeded_refs | A3 overlay/seed-key mismatch |
| `rgg_2000::classic_gem_iters100` | matched_seeds_lt_30 | 2000 | 0 / ok:100 | 100 / ok:101 | D escalation; R escalation,seeded_refs | A3 overlay/seed-key mismatch |
| `small_world_2000::classic_gem_iters100` | matched_seeds_lt_30 | 2000 | 0 / ok:100 | 100 / ok:101 | D escalation; R escalation,seeded_refs | A3 overlay/seed-key mismatch |

### B1 sugiyama graphviz recursion-depth crash (1)
Engines: classic_sugiyama_graphviz_fidelity=1

| combo | reason | N | D ok/status | R ok/status | raw source | action |
|---|---:|---:|---|---|---|---|
| `small_world_2000::classic_sugiyama_graphviz_fidelity` | reimpl_seeds_lt_30 | 2000 | 0 / skipped:97,error:3 | 0 / ok:1 | D escalation; R escalation | B1 sugiyama graphviz recursion-depth crash |

### B2 DRL coarsest/default/coarsen small-medium performance bug (61)
Engines: classic_drl_coarsest=27, classic_drl_default=21, classic_drl_coarsen=13

| combo | reason | N | D ok/status | R ok/status | raw source | action |
|---|---:|---:|---|---|---|---|
| `grid_20x20::classic_drl_coarsen` | matched_seeds_lt_30 | 400 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `protein_ppi_200::classic_drl_coarsen` | matched_seeds_lt_30 | 200 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `random_dag_200::classic_drl_coarsen` | matched_seeds_lt_30 | 383 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `real_football_115::classic_drl_coarsen` | matched_seeds_lt_30 | 115 | 3 / skipped:94,ok:3,error:3 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `rgg_100::classic_drl_coarsen` | matched_seeds_lt_30 | 100 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `rgg_500::classic_drl_coarsen` | matched_seeds_lt_30 | 500 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `sbm_4x30::classic_drl_coarsen` | matched_seeds_lt_30 | 120 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `sbm_5x50::classic_drl_coarsen` | matched_seeds_lt_30 | 250 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `scale_free_ba_120::classic_drl_coarsen` | matched_seeds_lt_30 | 120 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `small_world_500::classic_drl_coarsen` | matched_seeds_lt_30 | 500 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `wide_1_100_1::classic_drl_coarsen` | matched_seeds_lt_30 | 102 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `wide_3_50_3::classic_drl_coarsen` | matched_seeds_lt_30 | 56 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `wide_single_layer_1_50_1::classic_drl_coarsen` | matched_seeds_lt_30 | 52 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `chung_lu_150::classic_drl_coarsest` | matched_seeds_lt_30 | 150 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `clustered_medium_5x20::classic_drl_coarsest` | matched_seeds_lt_30 | 100 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `compound_10x20::classic_drl_coarsest` | matched_seeds_lt_30 | 200 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `compound_dag_5x30::classic_drl_coarsest` | matched_seeds_lt_30 | 150 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `dependency_graph_100::classic_drl_coarsest` | matched_seeds_lt_30 | 100 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `er_100::classic_drl_coarsest` | matched_seeds_lt_30 | 100 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `grid_20x20::classic_drl_coarsest` | matched_seeds_lt_30 | 400 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `hub_and_spoke_3x20::classic_drl_coarsest` | matched_seeds_lt_30 | 65 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `multi_component_80::classic_drl_coarsest` | matched_seeds_lt_30 | 80 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `org_chart_deep::classic_drl_coarsest` | matched_seeds_lt_30 | 79 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `planar_60::classic_drl_coarsest` | matched_seeds_lt_30 | 60 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `protein_ppi_200::classic_drl_coarsest` | matched_seeds_lt_30 | 200 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `random_bipartite_60::classic_drl_coarsest` | matched_seeds_lt_30 | 60 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `random_dag_200::classic_drl_coarsest` | matched_seeds_lt_30 | 383 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `random_dag_50::classic_drl_coarsest` | matched_seeds_lt_30 | 97 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `real_karate_34::classic_drl_coarsest` | matched_seeds_lt_30 | 34 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `real_lesmis_77::classic_drl_coarsest` | matched_seeds_lt_30 | 77 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `regular_3_30::classic_drl_coarsest` | matched_seeds_lt_30 | 30 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `rgg_500::classic_drl_coarsest` | matched_seeds_lt_30 | 500 | 0 / skipped:97,error:3 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `sbm_4x30::classic_drl_coarsest` | matched_seeds_lt_30 | 120 | 0 / skipped:97,error:3 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `sbm_5x50::classic_drl_coarsest` | matched_seeds_lt_30 | 250 | 0 / skipped:97,error:3 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `scale_free_ba_120::classic_drl_coarsest` | matched_seeds_lt_30 | 120 | 0 / skipped:97,error:3 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `small_world_100::classic_drl_coarsest` | matched_seeds_lt_30 | 100 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `small_world_500::classic_drl_coarsest` | matched_seeds_lt_30 | 500 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `wide_1_100_1::classic_drl_coarsest` | matched_seeds_lt_30 | 102 | 0 / skipped:97,error:3 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `wide_3_50_3::classic_drl_coarsest` | matched_seeds_lt_30 | 56 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `wide_single_layer_1_50_1::classic_drl_coarsest` | matched_seeds_lt_30 | 52 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `chung_lu_150::classic_drl_default` | matched_seeds_lt_30 | 150 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `compound_10x20::classic_drl_default` | matched_seeds_lt_30 | 200 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `dependency_graph_100::classic_drl_default` | matched_seeds_lt_30 | 100 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `er_100::classic_drl_default` | matched_seeds_lt_30 | 100 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `grid_20x20::classic_drl_default` | matched_seeds_lt_30 | 400 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `protein_ppi_200::classic_drl_default` | matched_seeds_lt_30 | 200 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `random_dag_200::classic_drl_default` | matched_seeds_lt_30 | 383 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `real_football_115::classic_drl_default` | matched_seeds_lt_30 | 115 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `real_lesmis_77::classic_drl_default` | matched_seeds_lt_30 | 77 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `rgg_100::classic_drl_default` | matched_seeds_lt_30 | 100 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `rgg_500::classic_drl_default` | matched_seeds_lt_30 | 500 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `sbm_4x30::classic_drl_default` | matched_seeds_lt_30 | 120 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `sbm_5x50::classic_drl_default` | matched_seeds_lt_30 | 250 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `scale_free_ba_120::classic_drl_default` | matched_seeds_lt_30 | 120 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `sierpinski_42::classic_drl_default` | matched_seeds_lt_30 | 42 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `small_world_100::classic_drl_default` | matched_seeds_lt_30 | 100 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `small_world_500::classic_drl_default` | matched_seeds_lt_30 | 500 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `sparse_pair_50::classic_drl_default` | matched_seeds_lt_30 | 50 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `wide_1_100_1::classic_drl_default` | matched_seeds_lt_30 | 102 | 0 / skipped:97,error:3 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `wide_3_50_3::classic_drl_default` | matched_seeds_lt_30 | 56 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |
| `wide_single_layer_1_50_1::classic_drl_default` | matched_seeds_lt_30 | 52 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B2 DRL coarsest/default/coarsen small-medium performance bug |

### B3 Davidson-Harel small-graph scalar-loop performance bug (11)
Engines: classic_davidson_harel_rounds50=10, classic_davidson_harel_rounds100=1

| combo | reason | N | D ok/status | R ok/status | raw source | action |
|---|---:|---:|---|---|---|---|
| `sparse_pair_50::classic_davidson_harel_rounds100` | matched_seeds_lt_30 | 50 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B3 Davidson-Harel small-graph scalar-loop performance bug |
| `grid_rect_6x8::classic_davidson_harel_rounds50` | matched_seeds_lt_30 | 48 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B3 Davidson-Harel small-graph scalar-loop performance bug |
| `multi_component_80::classic_davidson_harel_rounds50` | matched_seeds_lt_30 | 80 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B3 Davidson-Harel small-graph scalar-loop performance bug |
| `org_chart_deep::classic_davidson_harel_rounds50` | matched_seeds_lt_30 | 79 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B3 Davidson-Harel small-graph scalar-loop performance bug |
| `random_bipartite_60::classic_davidson_harel_rounds50` | matched_seeds_lt_30 | 60 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B3 Davidson-Harel small-graph scalar-loop performance bug |
| `random_dag_50::classic_davidson_harel_rounds50` | matched_seeds_lt_30 | 97 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B3 Davidson-Harel small-graph scalar-loop performance bug |
| `regular_4_40::classic_davidson_harel_rounds50` | matched_seeds_lt_30 | 40 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B3 Davidson-Harel small-graph scalar-loop performance bug |
| `sierpinski_42::classic_davidson_harel_rounds50` | matched_seeds_lt_30 | 42 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B3 Davidson-Harel small-graph scalar-loop performance bug |
| `sparse_pair_50::classic_davidson_harel_rounds50` | matched_seeds_lt_30 | 50 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B3 Davidson-Harel small-graph scalar-loop performance bug |
| `triangular_lattice_36::classic_davidson_harel_rounds50` | matched_seeds_lt_30 | 36 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B3 Davidson-Harel small-graph scalar-loop performance bug |
| `wide_single_layer_1_50_1::classic_davidson_harel_rounds50` | matched_seeds_lt_30 | 52 | 0 / error:100 | 100 / ok:100 | D escalation; R escalation | B3 Davidson-Harel small-graph scalar-loop performance bug |

### B4 stress/neato/maxent timeout: vectorize or raise budget after profiling (25)
Engines: classic_neato=9, classic_stress_maj_iter500=6, classic_maxent_stress_alpha2=2, classic_maxent_stress_default=2, classic_maxent_stress_entropy=2, classic_maxent_stress_steps50=2, classic_stress_maj_iter50=2

| combo | reason | N | D ok/status | R ok/status | raw source | action |
|---|---:|---:|---|---|---|---|
| `hub_spoke_5x50::classic_maxent_stress_alpha2` | matched_seeds_lt_30 | 257 | 0 / skipped:97,error:3 | 100 / ok:101 | D escalation; R escalation,seeded_refs | B4 stress/neato/maxent timeout |
| `sbm_5x50::classic_maxent_stress_alpha2` | matched_seeds_lt_30 | 250 | 0 / skipped:97,error:3 | 100 / ok:101 | D escalation; R escalation,seeded_refs | B4 stress/neato/maxent timeout |
| `hub_spoke_5x50::classic_maxent_stress_default` | matched_seeds_lt_30 | 257 | 0 / skipped:97,error:3 | 100 / ok:101 | D escalation; R escalation,seeded_refs | B4 stress/neato/maxent timeout |
| `sbm_5x50::classic_maxent_stress_default` | matched_seeds_lt_30 | 250 | 0 / skipped:97,error:3 | 100 / ok:101 | D escalation; R escalation,seeded_refs | B4 stress/neato/maxent timeout |
| `hub_spoke_5x50::classic_maxent_stress_entropy` | matched_seeds_lt_30 | 257 | 0 / skipped:97,error:3 | 100 / ok:101 | D escalation; R escalation,seeded_refs | B4 stress/neato/maxent timeout |
| `sbm_5x50::classic_maxent_stress_entropy` | matched_seeds_lt_30 | 250 | 0 / skipped:97,error:3 | 100 / ok:101 | D escalation; R escalation,seeded_refs | B4 stress/neato/maxent timeout |
| `rgg_500::classic_maxent_stress_steps50` | matched_seeds_lt_30 | 500 | 28 / error:72,ok:28 | 100 / ok:101 | D escalation; R escalation,seeded_refs | B4 stress/neato/maxent timeout |
| `small_world_500::classic_maxent_stress_steps50` | matched_seeds_lt_30 | 500 | 25 / error:75,ok:25 | 100 / ok:101 | D escalation; R escalation,seeded_refs | B4 stress/neato/maxent timeout |
| `clustered_medium_5x20::classic_neato` | matched_seeds_lt_30 | 100 | 2 / skipped:95,ok:2,error:3 | 100 / ok:101 | D r73_fixes; R r72_fixes,seeded_refs | B4 stress/neato/maxent timeout |
| `dependency_graph_100::classic_neato` | matched_seeds_lt_30 | 100 | 0 / skipped:97,error:3 | 100 / ok:101 | D r73_fixes; R r72_fixes,seeded_refs | B4 stress/neato/maxent timeout |
| `er_100::classic_neato` | matched_seeds_lt_30 | 100 | 1 / skipped:95,error:4,ok:1 | 100 / ok:101 | D r73_fixes; R r72_fixes,seeded_refs | B4 stress/neato/maxent timeout |
| `grid_20x20::classic_neato` | matched_seeds_lt_30 | 400 | 0 / error:100 | 100 / ok:101 | D r72_fixes; R r72_fixes,seeded_refs | B4 stress/neato/maxent timeout |
| `random_dag_200::classic_neato` | matched_seeds_lt_30 | 383 | 0 / error:100 | 100 / ok:101 | D r72_fixes; R r72_fixes,seeded_refs | B4 stress/neato/maxent timeout |
| `real_football_115::classic_neato` | matched_seeds_lt_30 | 115 | 0 / skipped:97,error:3 | 100 / ok:101 | D r73_fixes; R r72_fixes,seeded_refs | B4 stress/neato/maxent timeout |
| `real_lesmis_77::classic_neato` | matched_seeds_lt_30 | 77 | 0 / error:100 | 100 / ok:101 | D r73_fixes; R r72_fixes,seeded_refs | B4 stress/neato/maxent timeout |
| `sbm_4x30::classic_neato` | matched_seeds_lt_30 | 120 | 7 / skipped:89,error:4,ok:7 | 100 / ok:101 | D r73_fixes; R r72_fixes,seeded_refs | B4 stress/neato/maxent timeout |
| `small_world_100::classic_neato` | matched_seeds_lt_30 | 100 | 0 / error:100 | 100 / ok:101 | D r73_fixes; R r72_fixes,seeded_refs | B4 stress/neato/maxent timeout |
| `rgg_500::classic_stress_maj_iter50` | matched_seeds_lt_30 | 500 | 27 / error:73,ok:27 | 100 / ok:101 | D escalation; R escalation,seeded_refs | B4 stress/neato/maxent timeout |
| `small_world_500::classic_stress_maj_iter50` | matched_seeds_lt_30 | 500 | 23 / error:77,ok:23 | 100 / ok:101 | D escalation; R escalation,seeded_refs | B4 stress/neato/maxent timeout |
| `multi_component_80::classic_stress_maj_iter500` | matched_seeds_lt_30 | 80 | 0 / error:100 | 100 / ok:101 | D escalation; R escalation,seeded_refs | B4 stress/neato/maxent timeout |
| `org_chart_deep::classic_stress_maj_iter500` | matched_seeds_lt_30 | 79 | 0 / error:100 | 100 / ok:101 | D escalation; R escalation,seeded_refs | B4 stress/neato/maxent timeout |
| `random_dag_50::classic_stress_maj_iter500` | matched_seeds_lt_30 | 97 | 0 / skipped:97,error:3 | 100 / ok:101 | D escalation; R escalation,seeded_refs | B4 stress/neato/maxent timeout |
| `real_lesmis_77::classic_stress_maj_iter500` | matched_seeds_lt_30 | 77 | 0 / error:100 | 100 / ok:101 | D escalation; R escalation,seeded_refs | B4 stress/neato/maxent timeout |
| `rgg_100::classic_stress_maj_iter500` | matched_seeds_lt_30 | 100 | 0 / skipped:97,error:3 | 100 / ok:101 | D escalation; R escalation,seeded_refs | B4 stress/neato/maxent timeout |
| `small_world_100::classic_stress_maj_iter500` | matched_seeds_lt_30 | 100 | 0 / skipped:97,error:3 | 100 / ok:101 | D escalation; R escalation,seeded_refs | B4 stress/neato/maxent timeout |

### B5 pivot_mds r73 partial rerun timeout/provenance artifact (1)
Engines: classic_pivot_mds_50=1

| combo | reason | N | D ok/status | R ok/status | raw source | action |
|---|---:|---:|---|---|---|---|
| `heavy_tail_weights_50::classic_pivot_mds_50` | reimpl_seeds_lt_30 | 50 | 5 / error:95,ok:5 | 0 / ok:1 | D r73_fixes; R escalation | B5 pivot_mds r73 partial rerun timeout/provenance artifact |

### B6 FMMM r73 partial timeout on medium clustered graphs (2)
Engines: classic_fmmm_graphviz_fdp_fidelity=2

| combo | reason | N | D ok/status | R ok/status | raw source | action |
|---|---:|---:|---|---|---|---|
| `citation_dag_300::classic_fmmm_graphviz_fdp_fidelity` | matched_seeds_lt_30 | 300 | 1 / skipped:96,ok:1,error:3 | 100 / ok:101 | D r73_fixes; R escalation,seeded_refs | B6 FMMM r73 partial timeout on medium clustered graphs |
| `sbm_5x50::classic_fmmm_graphviz_fdp_fidelity` | matched_seeds_lt_30 | 250 | 0 / error:100 | 100 / ok:101 | D r73_fixes; R escalation,seeded_refs | B6 FMMM r73 partial timeout on medium clustered graphs |

### C1 sugiyama large/slow compute frontier (9)
Engines: classic_sugiyama_passes4=3, classic_sugiyama_passes48=2, classic_sugiyama_default=1, classic_sugiyama_tight=1, classic_sugiyama_wide=1, classic_sugiyama_graphviz_fidelity=1

| combo | reason | N | D ok/status | R ok/status | raw source | action |
|---|---:|---:|---|---|---|---|
| `ba_5000::classic_sugiyama_default` | reimpl_seeds_lt_30 | 5000 | 29 / error:71,ok:29 | 0 / ok:1 | D escalation; R escalation | C1 sugiyama large/slow compute frontier |
| `small_world_500::classic_sugiyama_graphviz_fidelity` | reimpl_seeds_lt_30 | 500 | 20 / error:80,ok:20 | 0 / ok:1 | D escalation; R escalation | C1 sugiyama large/slow compute frontier |
| `ba_5000::classic_sugiyama_passes4` | reimpl_seeds_lt_30 | 5000 | 0 / error:100 | 0 / ok:1 | D escalation; R escalation | C1 sugiyama large/slow compute frontier |
| `rgg_2000::classic_sugiyama_passes4` | reimpl_seeds_lt_30 | 2000 | 0 / error:100 | 0 / ok:1 | D escalation; R escalation | C1 sugiyama large/slow compute frontier |
| `sbm_8x100::classic_sugiyama_passes4` | reimpl_seeds_lt_30 | 800 | 0 / error:100 | 0 / ok:1 | D escalation; R escalation | C1 sugiyama large/slow compute frontier |
| `ba_5000::classic_sugiyama_passes48` | reimpl_seeds_lt_30 | 5000 | 21 / error:79,ok:21 | 0 / ok:1 | D escalation; R escalation | C1 sugiyama large/slow compute frontier |
| `sbm_8x100::classic_sugiyama_passes48` | reimpl_seeds_lt_30 | 800 | 25 / error:75,ok:25 | 0 / ok:1 | D escalation; R escalation | C1 sugiyama large/slow compute frontier |
| `ba_5000::classic_sugiyama_tight` | reimpl_seeds_lt_30 | 5000 | 28 / error:72,ok:28 | 0 / ok:1 | D escalation; R escalation | C1 sugiyama large/slow compute frontier |
| `ba_5000::classic_sugiyama_wide` | reimpl_seeds_lt_30 | 5000 | 29 / error:71,ok:29 | 0 / ok:1 | D escalation; R escalation | C1 sugiyama large/slow compute frontier |

### C2 DRL refine large-graph compute frontier (4)
Engines: classic_drl_refine=4

| combo | reason | N | D ok/status | R ok/status | raw source | action |
|---|---:|---:|---|---|---|---|
| `ba_2000::classic_drl_refine` | matched_seeds_lt_30 | 2000 | 23 / error:77,ok:23 | 100 / ok:100 | D escalation; R escalation | C2 DRL refine large-graph compute frontier |
| `er_2000::classic_drl_refine` | matched_seeds_lt_30 | 2000 | 20 / error:80,ok:20 | 100 / ok:100 | D escalation; R escalation | C2 DRL refine large-graph compute frontier |
| `powerlaw_2000::classic_drl_refine` | matched_seeds_lt_30 | 2000 | 20 / error:80,ok:20 | 100 / ok:100 | D escalation; R escalation | C2 DRL refine large-graph compute frontier |
| `rgg_2000::classic_drl_refine` | matched_seeds_lt_30 | 2000 | 23 / error:77,ok:23 | 100 / ok:100 | D escalation; R escalation | C2 DRL refine large-graph compute frontier |

### C3 FR/SFDP large-graph compute frontier (37)
Engines: classic_fr_steps500=7, classic_sfdp_theta08=6, classic_sfdp_default=5, classic_sfdp_graphviz_fidelity=5, classic_sfdp_p_neg2=5, classic_sfdp_theta04=4, classic_sfdp_steps200=4, classic_fr_steps200=1

| combo | reason | N | D ok/status | R ok/status | raw source | action |
|---|---:|---:|---|---|---|---|
| `grid_50x50::classic_fr_steps200` | matched_seeds_lt_30 | 2500 | 26 / error:74,ok:26 | 32 / error:68,ok:32 | D escalation; R escalation | C3 FR/SFDP large-graph compute frontier |
| `ba_2000::classic_fr_steps500` | ref_seeds_lt_30 | 2000 | 15 / error:85,ok:15 | 9 / error:91,ok:9 | D escalation; R escalation | C3 FR/SFDP large-graph compute frontier |
| `er_2000::classic_fr_steps500` | ref_seeds_lt_30 | 2000 | 15 / error:85,ok:15 | 9 / error:91,ok:9 | D escalation; R escalation | C3 FR/SFDP large-graph compute frontier |
| `grid_50x50::classic_fr_steps500` | ref_seeds_lt_30 | 2500 | 1 / error:99,ok:1 | 2 / error:98,ok:2 | D escalation; R escalation | C3 FR/SFDP large-graph compute frontier |
| `powerlaw_2000::classic_fr_steps500` | ref_seeds_lt_30 | 2000 | 16 / error:84,ok:16 | 9 / error:91,ok:9 | D escalation; R escalation | C3 FR/SFDP large-graph compute frontier |
| `rgg_2000::classic_fr_steps500` | ref_seeds_lt_30 | 2000 | 15 / error:85,ok:15 | 9 / error:91,ok:9 | D escalation; R escalation | C3 FR/SFDP large-graph compute frontier |
| `sbm_8x100::classic_fr_steps500` | ref_seeds_lt_30 | 800 | 67 / ok:67,error:33 | 25 / error:75,ok:25 | D escalation; R escalation | C3 FR/SFDP large-graph compute frontier |
| `small_world_2000::classic_fr_steps500` | ref_seeds_lt_30 | 2000 | 10 / error:90,ok:10 | 6 / error:94,ok:6 | D escalation; R escalation | C3 FR/SFDP large-graph compute frontier |
| `dependency_500::classic_sfdp_default` | matched_seeds_lt_30 | 500 | 28 / error:72,ok:28 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `powerlaw_2000::classic_sfdp_default` | matched_seeds_lt_30 | 2000 | 0 / skipped:97,error:3 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `rgg_2000::classic_sfdp_default` | matched_seeds_lt_30 | 2000 | 12 / error:88,ok:12 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `sbm_8x100::classic_sfdp_default` | matched_seeds_lt_30 | 800 | 29 / error:71,ok:29 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `small_world_2000::classic_sfdp_default` | matched_seeds_lt_30 | 2000 | 15 / error:85,ok:15 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `dependency_500::classic_sfdp_graphviz_fidelity` | matched_seeds_lt_30 | 500 | 28 / error:72,ok:28 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `powerlaw_2000::classic_sfdp_graphviz_fidelity` | matched_seeds_lt_30 | 2000 | 0 / skipped:97,error:3 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `rgg_2000::classic_sfdp_graphviz_fidelity` | matched_seeds_lt_30 | 2000 | 12 / error:88,ok:12 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `sbm_8x100::classic_sfdp_graphviz_fidelity` | matched_seeds_lt_30 | 800 | 29 / error:71,ok:29 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `small_world_2000::classic_sfdp_graphviz_fidelity` | matched_seeds_lt_30 | 2000 | 15 / error:85,ok:15 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `dependency_500::classic_sfdp_p_neg2` | matched_seeds_lt_30 | 500 | 20 / error:80,ok:20 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `powerlaw_2000::classic_sfdp_p_neg2` | matched_seeds_lt_30 | 2000 | 0 / skipped:97,error:3 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `rgg_2000::classic_sfdp_p_neg2` | matched_seeds_lt_30 | 2000 | 0 / skipped:97,error:3 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `sbm_8x100::classic_sfdp_p_neg2` | matched_seeds_lt_30 | 800 | 27 / error:73,ok:27 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `small_world_2000::classic_sfdp_p_neg2` | matched_seeds_lt_30 | 2000 | 11 / error:89,ok:11 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `powerlaw_2000::classic_sfdp_steps200` | matched_seeds_lt_30 | 2000 | 0 / skipped:97,error:3 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `rgg_2000::classic_sfdp_steps200` | matched_seeds_lt_30 | 2000 | 8 / error:92,ok:8 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `sbm_8x100::classic_sfdp_steps200` | matched_seeds_lt_30 | 800 | 26 / error:74,ok:26 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `small_world_2000::classic_sfdp_steps200` | matched_seeds_lt_30 | 2000 | 12 / error:88,ok:12 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `dependency_500::classic_sfdp_theta04` | matched_seeds_lt_30 | 500 | 18 / error:82,ok:18 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `er_500::classic_sfdp_theta04` | matched_seeds_lt_30 | 500 | 25 / error:75,ok:25 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `sbm_8x100::classic_sfdp_theta04` | matched_seeds_lt_30 | 800 | 21 / error:79,ok:21 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `small_world_2000::classic_sfdp_theta04` | matched_seeds_lt_30 | 2000 | 11 / error:89,ok:11 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `ba_2000::classic_sfdp_theta08` | matched_seeds_lt_30 | 2000 | 21 / error:79,ok:21 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `er_2000::classic_sfdp_theta08` | matched_seeds_lt_30 | 2000 | 1 / skipped:96,ok:1,error:3 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `grid_50x50::classic_sfdp_theta08` | matched_seeds_lt_30 | 2500 | 22 / error:78,ok:22 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `powerlaw_2000::classic_sfdp_theta08` | matched_seeds_lt_30 | 2000 | 22 / error:78,ok:22 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `rgg_2000::classic_sfdp_theta08` | matched_seeds_lt_30 | 2000 | 20 / error:80,ok:20 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |
| `small_world_2000::classic_sfdp_theta08` | matched_seeds_lt_30 | 2000 | 23 / error:77,ok:23 | 100 / ok:101 | D escalation; R escalation,seeded_refs | C3 FR/SFDP large-graph compute frontier |

### C4 MDS er_2000 compute frontier/provenance inherited from escalation (2)
Engines: classic_classical_mds_default=1, classic_classical_mds_igraph_fidelity=1

| combo | reason | N | D ok/status | R ok/status | raw source | action |
|---|---:|---:|---|---|---|---|
| `er_2000::classic_classical_mds_default` | reimpl_seeds_lt_30 | 2000 | 0 / error:100 | 0 / ok:101 | D escalation; R escalation,seeded_refs | C4 MDS er_2000 compute frontier/provenance inherited from escalation |
| `er_2000::classic_classical_mds_igraph_fidelity` | reimpl_seeds_lt_30 | 2000 | 0 / error:100 | 0 / ok:101 | D escalation; R escalation,seeded_refs | C4 MDS er_2000 compute frontier/provenance inherited from escalation |

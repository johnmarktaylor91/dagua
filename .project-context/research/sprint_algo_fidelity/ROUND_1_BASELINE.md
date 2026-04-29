# Round 1 Baseline

Round 1 adds a graphviz-focused cross-comparison baseline without changing any layout algorithms. The most divergent P0 family by median RMSD is `dot` (0.3245); the closest is `neato_stress` (0.0353). Round 2 should attack `dot` first because it has the largest median shape mismatch across shared graphs, not just a single outlier.

Note: `eval_output/benchmark_full/results.json` does not contain the requested quality metric fields (`aspect_ratio`, `dag_consistency`, `edge_length_cv`, `edge_straightness_mean_deg`, `depth_spearman_rho`, `overlap_count`). `quality_deltas.csv` preserves the requested schema with blank metric values rather than recomputing metrics from a different source.

## P0 Pairings

### dot: `classic_sugiyama` vs `graphviz_dot`

- RMSD: median 0.3245, p25 0.1673, p75 0.3577, p95 0.4358, worst 0.4744
- Counts: >0.05 = 17, >0.15 = 17, graphs = 22
- Top-3 worst graphs by RMSD:
  - `small_label_storm`: 0.4744
  - `shape_and_routing_matrix`: 0.4372
  - `densenet_block`: 0.4094
- Top-3 worst recorded quality-metric deltas:
  - N/A: requested metric values are absent from `results.json`.

### neato_stress: `classic_stress_maj` vs `graphviz_neato`

- RMSD: median 0.0353, p25 0.0034, p75 0.1654, p95 0.3617, worst 0.3817
- Counts: >0.05 = 10, >0.15 = 7, graphs = 23
- Top-3 worst graphs by RMSD:
  - `inception_block`: 0.3817
  - `petersen_10`: 0.3634
  - `edge_label_braid`: 0.3466
- Top-3 worst recorded quality-metric deltas:
  - N/A: requested metric values are absent from `results.json`.

### neato_mds: `classic_classical_mds` vs `graphviz_neato`

- RMSD: median 0.0455, p25 0.0357, p75 0.1875, p95 0.2800, worst 0.3326
- Counts: >0.05 = 11, >0.15 = 7, graphs = 23
- Top-3 worst graphs by RMSD:
  - `petersen_10`: 0.3326
  - `edge_label_braid`: 0.2812
  - `inception_block`: 0.2690
- Top-3 worst recorded quality-metric deltas:
  - N/A: requested metric values are absent from `results.json`.

### fdp: `classic_fmmm` vs `graphviz_fdp`

- RMSD: median 0.2918, p25 0.2033, p75 0.3517, p95 0.4120, worst 0.4169
- Counts: >0.05 = 21, >0.15 = 21, graphs = 21
- Top-3 worst graphs by RMSD:
  - `disconnected_label_cycle_collage`: 0.4169
  - `edge_label_braid`: 0.4120
  - `center_port_backedge_hub`: 0.3833
- Top-3 worst recorded quality-metric deltas:
  - N/A: requested metric values are absent from `results.json`.

### sfdp: `classic_sfdp` vs `graphviz_sfdp`

- RMSD: median 0.0915, p25 0.0366, p75 0.2086, p95 0.4178, worst 0.4751
- Counts: >0.05 = 13, >0.15 = 9, graphs = 24
- Top-3 worst graphs by RMSD:
  - `center_port_backedge_hub`: 0.4751
  - `disconnected_label_cycle_collage`: 0.4249
  - `petersen_10`: 0.3777
- Top-3 worst recorded quality-metric deltas:
  - N/A: requested metric values are absent from `results.json`.

## P1 Pairings

### neato_fr_proxy: `classic_fr` vs `graphviz_neato`

- RMSD: median 0.1864, worst 0.3678 on `disconnected_label_cycle_collage`

### neato_kk_proxy: `classic_kk` vs `graphviz_neato`

- RMSD: median 0.0862, worst 0.3652 on `edge_label_braid`

## Panel Outputs

- `eval_output/algo_fidelity/round_1/panels/dot__small_label_storm__worst.png`
- `eval_output/algo_fidelity/round_1/panels/neato_stress__inception_block__worst.png`
- `eval_output/algo_fidelity/round_1/panels/neato_mds__petersen_10__worst.png`
- `eval_output/algo_fidelity/round_1/panels/fdp__disconnected_label_cycle_collage__worst.png`
- `eval_output/algo_fidelity/round_1/panels/sfdp__center_port_backedge_hub__worst.png`

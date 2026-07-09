# r80-S4 Stage 1 Probe: Undirected-Class Portfolio Headroom

Standalone probe (no product code changed). Every undirected corpus graph laid out with dagua's own sfdp/neato/kk reimplementations + size-aware overlap projection, scored with the identical honest composite the baseline harness uses (composite_auto with is_semantically_directed=False). Compared against frozen current-dagua and frozen best-external rows from eval_output/r79_baseline/results.json in the main worktree.

**DECISION GATE**: 15 / 27 frozen-LOSS graphs improved to max(candidates) >= best_external - 0.5. Threshold: >= 10. **GATE PASSES**.

## Per-graph table (all undirected corpus graphs)

| graph | N | current-dagua | best-external | sfdp+proj | neato+proj | kk+proj | best-cand vs best-ext | frozen verdict | wall sfdp/neato/kk (s) |
|---|---|---|---|---|---|---|---|---|---|
| petersen_10 | 10 | 57.22 | graphviz_sfdp 78.44 | 54.44 | 79.02 | 22.52 | +0.58 | LOSS | 0.7/0.3/0.2 |
| complete_bipartite_8x12 | 20 | 63.65 | graphviz_sfdp 57.91 | 17.21 | 35.82 | 18.86 | -22.10 | WIN | 0.6/0.9/0.5 |
| grid_5x5 | 25 | 94.21 | igraph_sugiyama 95.00 | 51.48 | 94.53 | 25.91 | -0.47 | LOSS | 0.2/1.1/1.5 |
| regular_3_30 | 30 | 62.32 | graphviz_dot 68.28 | 69.55 | 84.34 | 18.28 | +16.06 | LOSS | 0.1/1.7/1.0 |
| weighted_clusters_3x10 | 30 | 45.08 | dagre 57.13 | 28.47 | 68.05 | 19.73 | +10.92 | LOSS | 0.5/2.7/4.6 |
| real_karate_34 | 34 | 50.11 | graphviz_dot 58.56 | 37.47 | 68.79 | 17.47 | +10.23 | LOSS | 0.4/4.7/2.4 |
| weighted_karate_34 | 34 | 50.11 | graphviz_dot 58.56 | 28.40 | 69.55 | 24.06 | +10.99 | LOSS | 0.4/4.3/1.7 |
| triangular_lattice_36 | 36 | 82.26 | graphviz_neato 94.48 | 59.91 | 94.48 | 23.87 | +0.00 | LOSS | 0.3/2.3/1.2 |
| r79_weighted_bipartite_16x24 | 40 | 64.71 | elk_layered 51.33 | 27.44 | 51.23 | 17.65 | -0.10 | WIN | 0.7/7.8/3.4 |
| regular_4_40 | 40 | 50.74 | igraph_kamada_kawai 54.74 | 56.32 | 52.48 | 11.38 | +1.58 | LOSS | 0.3/4.0/1.2 |
| hexagonal_lattice_42 | 42 | 78.13 | graphviz_neato 93.92 | 70.44 | 93.92 | 33.45 | +0.00 | LOSS | 0.1/4.6/2.5 |
| sierpinski_42 | 42 | 88.33 | graphviz_neato 90.39 | 36.61 | 90.40 | 29.82 | +0.01 | LOSS | 0.8/4.5/1.9 |
| grid_rect_6x8 | 48 | 92.87 | graphviz_neato 94.26 | 71.32 | 94.26 | 32.15 | -0.00 | LOSS | 0.6/3.6/2.0 |
| heavy_tail_weights_50 | 50 | 67.47 | igraph_kamada_kawai 62.23 | 44.90 | 25.02 | 30.47 | -17.33 | WIN | 0.3/41.1/17.3 |
| planar_60 | 60 | 57.31 | dagre 62.37 | 36.28 | 57.86 | 22.11 | -4.51 | LOSS | 0.4/11.7/1.2 |
| random_bipartite_60 | 60 | 65.24 | igraph_kamada_kawai 60.54 | 28.43 | 59.22 | 19.27 | -1.32 | WIN | 0.7/13.4/1.4 |
| r79_weighted_community_4x18 | 72 | 46.21 | graphviz_dot 51.88 | 23.89 | 38.28 | 12.64 | -13.61 | LOSS | 0.9/64.2/14.8 |
| real_lesmis_77 | 77 | 50.62 | graphviz_dot 52.81 | 26.59 | 38.65 | 18.34 | -14.16 | LOSS | 0.6/41.8/5.1 |
| multi_component_80 | 80 | 81.55 | graphviz_neato 92.52 | 49.20 | 92.52 | 35.19 | +0.00 | LOSS | 0.7/7.2/1.9 |
| r79_weighted_ladder_40 | 80 | 94.70 | igraph_sugiyama 88.22 | 44.48 | 63.92 | 10.69 | -24.30 | WIN | 0.7/15.6/6.1 |
| r79_undirected_sbm_high_mix_3x30 | 90 | 36.54 | elk_layered 40.38 | 46.88 | 25.40 | 9.87 | +6.50 | LOSS | 22.3/41.3/3.6 |
| er_100 | 100 | 56.02 | graphviz_sfdp 56.30 | 13.17 | 53.79 | 9.75 | -2.50 | TIE | 0.1/40.8/1.0 |
| r79_undirected_sbm_low_mix_4x25 | 100 | 52.78 | elk_layered 50.12 | 62.39 | 43.49 | 17.08 | +12.27 | WIN | 17.9/28.9/2.9 |
| r79_undirected_sbm_mid_mix_5x20 | 100 | 49.61 | elk_layered 37.83 | 55.98 | 32.24 | 15.64 | +18.15 | WIN | 23.5/38.0/3.3 |
| rgg_100 | 100 | 47.38 | elk_layered 52.31 | 29.83 | 47.89 | 22.24 | -4.41 | LOSS | 1.7/15.6/1.2 |
| small_world_100 | 100 | 91.75 | igraph_kamada_kawai 68.62 | 40.23 | 88.13 | 24.21 | +19.51 | WIN | 0.1/16.7/0.8 |
| real_football_115 | 115 | 30.64 | graphviz_dot 37.77 | 9.70 | 33.20 | 7.75 | -4.57 | LOSS | 2.1/61.4/1.1 |
| r79_weighted_mesh_10x12 | 120 | 87.84 | graphviz_neato 92.52 | 35.62 | 85.97 | 20.54 | -6.55 | LOSS | 0.3/15.8/2.0 |
| r79_weighted_small_world_120 | 120 | 32.65 | igraph_kamada_kawai 51.69 | 32.01 | 38.22 | 23.24 | -13.47 | LOSS | 1.0/134.0/4.1 |
| sbm_4x30 | 120 | 46.15 | graphviz_dot 48.18 | 20.60 | 42.00 | 15.66 | -6.18 | LOSS | 2.2/43.8/4.6 |
| scale_free_ba_120 | 120 | 34.86 | graphviz_sfdp 41.61 | 20.00 | 38.00 | 14.44 | -3.61 | LOSS | 0.7/82.1/5.2 |
| chung_lu_150 | 150 | 39.62 | igraph_kamada_kawai 46.84 | 50.99 | 38.06 | 23.62 | +4.15 | LOSS | 2.1/104.8/2.3 |
| protein_ppi_200 | 200 | 52.05 | igraph_kamada_kawai 52.97 | 25.69 | N/A | 19.86 | -27.28 | LOSS | 0.3/150.0/2.9 |
| sbm_5x50 | 250 | 45.79 | graphviz_dot 44.49 | 20.45 | 40.31 | 10.56 | -4.18 | WIN | 4.7/149.8/5.6 |
| grid_20x20 | 400 | 93.44 | graphviz_neato 94.48 | 40.52 | 94.48 | 25.88 | +0.00 | LOSS | 1.0/133.1/4.1 |
| ba_500 | 500 | 44.30 | graphviz_sfdp 41.81 | 30.16 | N/A | 6.82 | -11.66 | WIN | 0.6/150.0/12.2 |
| er_500 | 500 | 46.82 | graphviz_sfdp 54.48 | 39.07 | N/A | 20.04 | -15.41 | LOSS | 4.0/150.0/9.1 |
| rgg_500 | 500 | 53.49 | elk_layered 52.38 | 30.70 | N/A | 20.66 | -21.68 | WIN | 3.5/150.0/14.4 |
| small_world_500 | 500 | 42.88 | graphviz_neato 58.66 | 42.07 | N/A | 24.84 | -16.59 | LOSS | 0.9/150.0/25.8 |
| sbm_8x100 | 800 | N/A | N/A | N/A | N/A | N/A | N/A | NO_FROZEN_DATA | 0.0/0.0/0.0 |
| ba_2000 | 2000 | N/A | N/A | N/A | N/A | N/A | N/A | NO_FROZEN_DATA | 0.0/0.0/0.0 |
| er_2000 | 2000 | N/A | N/A | N/A | N/A | N/A | N/A | NO_FROZEN_DATA | 0.0/0.0/0.0 |
| rgg_2000 | 2000 | N/A | N/A | N/A | N/A | N/A | N/A | NO_FROZEN_DATA | 0.0/0.0/0.0 |
| small_world_2000 | 2000 | N/A | N/A | N/A | N/A | N/A | N/A | NO_FROZEN_DATA | 0.0/0.0/0.0 |
| grid_50x50 | 2500 | N/A | N/A | N/A | N/A | N/A | N/A | NO_FROZEN_DATA | 0.0/0.0/0.0 |
| ba_5000 | 5000 | N/A | N/A | N/A | N/A | N/A | N/A | NO_FROZEN_DATA | 0.0/0.0/0.0 |

## Frozen-LOSS graphs only (gate-relevant subset)

| graph | N | current-dagua | best-external | sfdp+proj | neato+proj | kk+proj | best-cand vs best-ext | frozen verdict | wall sfdp/neato/kk (s) |
|---|---|---|---|---|---|---|---|---|---|
| petersen_10 | 10 | 57.22 | graphviz_sfdp 78.44 | 54.44 | 79.02 | 22.52 | +0.58 | LOSS (gate-hit: YES) | 0.7/0.3/0.2 |
| grid_5x5 | 25 | 94.21 | igraph_sugiyama 95.00 | 51.48 | 94.53 | 25.91 | -0.47 | LOSS (gate-hit: YES) | 0.2/1.1/1.5 |
| regular_3_30 | 30 | 62.32 | graphviz_dot 68.28 | 69.55 | 84.34 | 18.28 | +16.06 | LOSS (gate-hit: YES) | 0.1/1.7/1.0 |
| weighted_clusters_3x10 | 30 | 45.08 | dagre 57.13 | 28.47 | 68.05 | 19.73 | +10.92 | LOSS (gate-hit: YES) | 0.5/2.7/4.6 |
| real_karate_34 | 34 | 50.11 | graphviz_dot 58.56 | 37.47 | 68.79 | 17.47 | +10.23 | LOSS (gate-hit: YES) | 0.4/4.7/2.4 |
| weighted_karate_34 | 34 | 50.11 | graphviz_dot 58.56 | 28.40 | 69.55 | 24.06 | +10.99 | LOSS (gate-hit: YES) | 0.4/4.3/1.7 |
| triangular_lattice_36 | 36 | 82.26 | graphviz_neato 94.48 | 59.91 | 94.48 | 23.87 | +0.00 | LOSS (gate-hit: YES) | 0.3/2.3/1.2 |
| regular_4_40 | 40 | 50.74 | igraph_kamada_kawai 54.74 | 56.32 | 52.48 | 11.38 | +1.58 | LOSS (gate-hit: YES) | 0.3/4.0/1.2 |
| hexagonal_lattice_42 | 42 | 78.13 | graphviz_neato 93.92 | 70.44 | 93.92 | 33.45 | +0.00 | LOSS (gate-hit: YES) | 0.1/4.6/2.5 |
| sierpinski_42 | 42 | 88.33 | graphviz_neato 90.39 | 36.61 | 90.40 | 29.82 | +0.01 | LOSS (gate-hit: YES) | 0.8/4.5/1.9 |
| grid_rect_6x8 | 48 | 92.87 | graphviz_neato 94.26 | 71.32 | 94.26 | 32.15 | -0.00 | LOSS (gate-hit: YES) | 0.6/3.6/2.0 |
| planar_60 | 60 | 57.31 | dagre 62.37 | 36.28 | 57.86 | 22.11 | -4.51 | LOSS (gate-hit: no) | 0.4/11.7/1.2 |
| r79_weighted_community_4x18 | 72 | 46.21 | graphviz_dot 51.88 | 23.89 | 38.28 | 12.64 | -13.61 | LOSS (gate-hit: no) | 0.9/64.2/14.8 |
| real_lesmis_77 | 77 | 50.62 | graphviz_dot 52.81 | 26.59 | 38.65 | 18.34 | -14.16 | LOSS (gate-hit: no) | 0.6/41.8/5.1 |
| multi_component_80 | 80 | 81.55 | graphviz_neato 92.52 | 49.20 | 92.52 | 35.19 | +0.00 | LOSS (gate-hit: YES) | 0.7/7.2/1.9 |
| r79_undirected_sbm_high_mix_3x30 | 90 | 36.54 | elk_layered 40.38 | 46.88 | 25.40 | 9.87 | +6.50 | LOSS (gate-hit: YES) | 22.3/41.3/3.6 |
| rgg_100 | 100 | 47.38 | elk_layered 52.31 | 29.83 | 47.89 | 22.24 | -4.41 | LOSS (gate-hit: no) | 1.7/15.6/1.2 |
| real_football_115 | 115 | 30.64 | graphviz_dot 37.77 | 9.70 | 33.20 | 7.75 | -4.57 | LOSS (gate-hit: no) | 2.1/61.4/1.1 |
| r79_weighted_mesh_10x12 | 120 | 87.84 | graphviz_neato 92.52 | 35.62 | 85.97 | 20.54 | -6.55 | LOSS (gate-hit: no) | 0.3/15.8/2.0 |
| r79_weighted_small_world_120 | 120 | 32.65 | igraph_kamada_kawai 51.69 | 32.01 | 38.22 | 23.24 | -13.47 | LOSS (gate-hit: no) | 1.0/134.0/4.1 |
| sbm_4x30 | 120 | 46.15 | graphviz_dot 48.18 | 20.60 | 42.00 | 15.66 | -6.18 | LOSS (gate-hit: no) | 2.2/43.8/4.6 |
| scale_free_ba_120 | 120 | 34.86 | graphviz_sfdp 41.61 | 20.00 | 38.00 | 14.44 | -3.61 | LOSS (gate-hit: no) | 0.7/82.1/5.2 |
| chung_lu_150 | 150 | 39.62 | igraph_kamada_kawai 46.84 | 50.99 | 38.06 | 23.62 | +4.15 | LOSS (gate-hit: YES) | 2.1/104.8/2.3 |
| protein_ppi_200 | 200 | 52.05 | igraph_kamada_kawai 52.97 | 25.69 | N/A | 19.86 | -27.28 | LOSS (gate-hit: no) | 0.3/150.0/2.9 |
| grid_20x20 | 400 | 93.44 | graphviz_neato 94.48 | 40.52 | 94.48 | 25.88 | +0.00 | LOSS (gate-hit: YES) | 1.0/133.1/4.1 |
| er_500 | 500 | 46.82 | graphviz_sfdp 54.48 | 39.07 | N/A | 20.04 | -15.41 | LOSS (gate-hit: no) | 4.0/150.0/9.1 |
| small_world_500 | 500 | 42.88 | graphviz_neato 58.66 | 42.07 | N/A | 24.84 | -16.59 | LOSS (gate-hit: no) | 0.9/150.0/25.8 |

## Candidate errors

- protein_ppi_200 / neato: TIMEOUT or worker died: exceeded 150s wall-clock cap
- ba_500 / neato: TIMEOUT or worker died: exceeded 150s wall-clock cap
- er_500 / neato: TIMEOUT or worker died: exceeded 150s wall-clock cap
- rgg_500 / neato: TIMEOUT or worker died: exceeded 150s wall-clock cap
- small_world_500 / neato: TIMEOUT or worker died: exceeded 150s wall-clock cap
- sbm_8x100 / sfdp: SKIPPED: 800 nodes > MAX_CANDIDATE_NODES=600
- sbm_8x100 / neato: SKIPPED: 800 nodes > MAX_CANDIDATE_NODES=600
- sbm_8x100 / kk: SKIPPED: 800 nodes > MAX_CANDIDATE_NODES=600
- ba_2000 / sfdp: SKIPPED: 2000 nodes > MAX_CANDIDATE_NODES=600
- ba_2000 / neato: SKIPPED: 2000 nodes > MAX_CANDIDATE_NODES=600
- ba_2000 / kk: SKIPPED: 2000 nodes > MAX_CANDIDATE_NODES=600
- er_2000 / sfdp: SKIPPED: 2000 nodes > MAX_CANDIDATE_NODES=600
- er_2000 / neato: SKIPPED: 2000 nodes > MAX_CANDIDATE_NODES=600
- er_2000 / kk: SKIPPED: 2000 nodes > MAX_CANDIDATE_NODES=600
- rgg_2000 / sfdp: SKIPPED: 2000 nodes > MAX_CANDIDATE_NODES=600
- rgg_2000 / neato: SKIPPED: 2000 nodes > MAX_CANDIDATE_NODES=600
- rgg_2000 / kk: SKIPPED: 2000 nodes > MAX_CANDIDATE_NODES=600
- small_world_2000 / sfdp: SKIPPED: 2000 nodes > MAX_CANDIDATE_NODES=600
- small_world_2000 / neato: SKIPPED: 2000 nodes > MAX_CANDIDATE_NODES=600
- small_world_2000 / kk: SKIPPED: 2000 nodes > MAX_CANDIDATE_NODES=600
- grid_50x50 / sfdp: SKIPPED: 2500 nodes > MAX_CANDIDATE_NODES=600
- grid_50x50 / neato: SKIPPED: 2500 nodes > MAX_CANDIDATE_NODES=600
- grid_50x50 / kk: SKIPPED: 2500 nodes > MAX_CANDIDATE_NODES=600
- ba_5000 / sfdp: SKIPPED: 5000 nodes > MAX_CANDIDATE_NODES=600
- ba_5000 / neato: SKIPPED: 5000 nodes > MAX_CANDIDATE_NODES=600
- ba_5000 / kk: SKIPPED: 5000 nodes > MAX_CANDIDATE_NODES=600

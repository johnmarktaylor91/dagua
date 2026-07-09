# P9: Full-drawing quality baseline (r80-S6 proof run)

Generated: 2026-07-09T03:50:38+00:00
Seed: 42. Engines: dagua, graphviz_dot, graphviz_sfdp, elk_layered. Graphs: 10 (3 layered DAG, 3 undirected community, 2 clustered, 2 weighted).
Scorer: dagua.metrics.composite_drawing (0-100; weights: crossings 30,
edge-node 20, labels 15, ports 12, overlap 10, curvature 8, bends 5).

```
| graph                            | engine        |    nat |    dgr |   natX |   dgrX |   enX |   port |   curv |  bend | lblN |  cov |
|--------------------------------------------------------------------------------------------------------------------------------------|
| citation_dag_300                 | dagua         |     -- |   44.1 |     -- |  0.142 |     1 |    7.4 |   1.16 |  0.49 |    0 | 0.00 |
| citation_dag_300                 | graphviz_dot  |   50.5 |   39.7 |  0.079 |  0.103 |     0 |   12.0 |   9.55 |  1.46 |    0 | 1.00 |
| citation_dag_300                 | graphviz_sfdp |   31.6 |   20.8 |  0.046 |  0.077 |  8423 |    9.6 |  13.46 |  0.00 |    0 | 1.00 |
| citation_dag_300                 | elk_layered   |   41.2 |   42.9 |  0.091 |  0.127 |     6 |    0.0 |  14.22 |  2.98 |    0 | 1.00 |
| random_dag_200                   | dagua         |     -- |   52.7 |     -- |  0.081 |     1 |   16.3 |   2.36 |  0.13 |    0 | 0.00 |
| random_dag_200                   | graphviz_dot  |   72.2 |   59.7 |  0.032 |  0.034 |     0 |   32.9 |   6.75 |  0.45 |    0 | 1.00 |
| random_dag_200                   | graphviz_sfdp |   52.3 |   45.0 |  0.007 |  0.021 |  1135 |   46.4 |   6.44 |  0.00 |    0 | 1.00 |
| random_dag_200                   | elk_layered   |   53.2 |   57.2 |  0.053 |  0.059 |    13 |    0.0 |   5.40 |  1.97 |    0 | 1.00 |
| long_skip_only_24                | dagua         |     -- |   64.0 |     -- |  0.062 |     0 |   13.4 |   0.25 |  0.00 |    0 | 0.00 |
| long_skip_only_24                | graphviz_dot  |   72.7 |   72.9 |  0.020 |  0.020 |     0 |   20.4 |   2.86 |  0.00 |    0 | 1.00 |
| long_skip_only_24                | graphviz_sfdp |   62.4 |   77.4 |  0.011 |  0.020 |    25 |   22.7 |   1.88 |  0.00 |    0 | 1.00 |
| long_skip_only_24                | elk_layered   |   62.9 |   72.4 |  0.020 |  0.021 |     4 |    0.0 |   3.63 |  1.22 |    0 | 1.00 |
| r79_undirected_sbm_low_mix_4x25  | dagua         |     -- |   39.5 |     -- |  0.148 |     7 |    2.4 |   2.62 |  2.77 |    0 | 0.00 |
| r79_undirected_sbm_low_mix_4x25  | graphviz_dot  |   59.4 |   54.2 |  0.049 |  0.042 |     0 |    7.3 |   2.40 |  0.46 |    0 | 1.00 |
| r79_undirected_sbm_low_mix_4x25  | graphviz_sfdp |   41.0 |   32.2 |  0.011 |  0.027 |  2539 |    6.1 |   5.85 |  0.00 |    0 | 1.00 |
| r79_undirected_sbm_low_mix_4x25  | elk_layered   |   36.3 |   56.1 |  0.107 |  0.041 |    91 |    0.0 |  13.08 |  2.46 |    0 | 0.89 |
| chung_lu_150                     | dagua         |     -- |   47.6 |     -- |  0.108 |     0 |   18.6 |   1.00 |  0.40 |    0 | 0.00 |
| chung_lu_150                     | graphviz_dot  |   71.4 |   49.2 |  0.039 |  0.072 |     0 |   38.3 |   6.27 |  0.62 |    0 | 1.00 |
| chung_lu_150                     | graphviz_sfdp |   46.0 |   38.9 |  0.023 |  0.043 |  1169 |   34.7 |   8.55 |  0.00 |    0 | 1.00 |
| chung_lu_150                     | elk_layered   |   54.7 |   53.0 |  0.046 |  0.074 |     8 |    0.0 |  15.35 |  2.66 |    0 | 1.00 |
| protein_ppi_200                  | dagua         |     -- |   65.6 |     -- |  0.029 |     1 |    7.9 |   1.82 |  0.57 |    0 | 0.00 |
| protein_ppi_200                  | graphviz_dot  |   73.4 |   67.0 |  0.009 |  0.016 |     0 |   14.1 |   6.01 |  0.39 |    0 | 1.00 |
| protein_ppi_200                  | graphviz_sfdp |   47.2 |   38.2 |  0.001 |  0.018 |  4416 |   16.5 |   5.17 |  0.07 |    0 | 1.00 |
| protein_ppi_200                  | elk_layered   |   62.3 |   65.8 |  0.019 |  0.021 |    36 |    0.0 |   9.77 |  2.50 |    0 | 1.00 |
| clustered_medium_5x20            | dagua         |     -- |   65.0 |     -- |  0.028 |    10 |    8.5 |   1.01 |  0.35 |    0 | 0.00 |
| clustered_medium_5x20            | graphviz_dot  |   66.0 |   59.4 |  0.034 |  0.022 |     0 |   16.0 |   4.47 |  0.82 |    0 | 1.00 |
| clustered_medium_5x20            | graphviz_sfdp |   49.6 |   44.4 |  0.006 |  0.016 |   317 |   29.5 |  13.34 |  0.03 |    0 | 1.00 |
| clustered_medium_5x20            | elk_layered   |   45.4 |   62.0 |  0.048 |  0.053 |   501 |    0.0 |   0.00 |  0.54 |    0 | 0.67 |
| r79_nested_clusters_3x2x10       | dagua         |     -- |   72.6 |     -- |  0.005 |     5 |   12.0 |   2.16 |  0.65 |    0 | 0.00 |
| r79_nested_clusters_3x2x10       | graphviz_dot  |   75.3 |   74.7 |  0.001 |  0.003 |     0 |   10.8 |   2.64 |  0.07 |    0 | 1.00 |
| r79_nested_clusters_3x2x10       | graphviz_sfdp |   48.8 |   66.7 |  0.000 |  0.002 |   129 |   21.3 |   4.62 |  0.06 |    0 | 1.00 |
| r79_nested_clusters_3x2x10       | elk_layered   |   38.5 |   68.2 |  0.078 |  0.033 |    68 |    0.0 |   0.00 |  0.74 |    0 | 0.78 |
| heavy_tail_weights_50            | dagua         |     -- |   58.7 |     -- |  0.049 |     6 |   10.7 |   3.34 |  0.45 |    0 | 0.00 |
| heavy_tail_weights_50            | graphviz_dot  |   73.7 |   67.4 |  0.020 |  0.024 |     0 |   24.5 |   6.51 |  0.18 |    0 | 1.00 |
| heavy_tail_weights_50            | graphviz_sfdp |   54.2 |   62.6 |  0.008 |  0.016 |    57 |   37.6 |   6.99 |  0.00 |    0 | 1.00 |
| heavy_tail_weights_50            | elk_layered   |   61.7 |   67.3 |  0.026 |  0.027 |     4 |    0.0 |   4.53 |  1.57 |    0 | 1.00 |
| r79_weighted_community_4x18      | dagua         |     -- |   55.8 |     -- |  0.060 |     3 |    6.3 |   7.29 |  0.34 |    0 | 0.00 |
| r79_weighted_community_4x18      | graphviz_dot  |   68.4 |   59.1 |  0.022 |  0.033 |     0 |   10.1 |   5.05 |  0.34 |    0 | 1.00 |
| r79_weighted_community_4x18      | graphviz_sfdp |   43.4 |   34.8 |  0.009 |  0.024 |   702 |   12.1 |   5.11 |  0.02 |    0 | 1.00 |
| r79_weighted_community_4x18      | elk_layered   |   58.5 |   59.3 |  0.030 |  0.039 |    20 |    0.0 |   5.32 |  2.30 |    0 | 1.00 |
```

Legend: nat = composite_drawing on the engine's NATIVE routing; dgr = composite_drawing on the engine's positions with DAGUA's router ('external positions + dagua routing'); natX/dgrX = routed crossing rate per variant; enX = edge-node crossings; port = port angular resolution (deg); curv = curvature CV; bend = mean bends/edge; lblN = edge-label-vs-node overlaps; cov = fraction of edges with captured native routes. dagua has no 'native vs dagua-routed' split (its native routing IS the dagua router), so nat is blank and dgr is its full-drawing score. Component columns describe the native variant when captured, else the dagua-routed variant.

## Observations

(filled in by the sprint report)

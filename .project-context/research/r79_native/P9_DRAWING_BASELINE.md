# P9: Full-drawing quality baseline (r80-S6 proof run)

Generated: 2026-07-08T20:05:35+00:00
Seed: 42. Engines: dagua, graphviz_dot, graphviz_sfdp, elk_layered. Graphs: 10 (3 layered DAG, 3 undirected community, 2 clustered, 2 weighted).
Scorer: dagua.metrics.composite_drawing (0-100; weights: crossings 30,
edge-node 20, labels 15, ports 12, overlap 10, curvature 8, bends 5).

```
| graph                            | engine        |    nat |    dgr |   natX |   dgrX |   enX |   port |   curv |  bend | lblN |  cov |
|--------------------------------------------------------------------------------------------------------------------------------------|
| citation_dag_300                 | dagua         |     -- |   42.0 |     -- |  0.121 |     7 |    0.0 |   1.34 |  0.28 |    0 | 0.00 |
| citation_dag_300                 | graphviz_dot  |   50.5 |   37.2 |  0.079 |  0.103 |     0 |   12.0 |   9.55 |  1.46 |    0 | 1.00 |
| citation_dag_300                 | graphviz_sfdp |   31.6 |   20.2 |  0.046 |  0.068 |  8423 |    9.6 |  13.46 |  0.00 |    0 | 1.00 |
| citation_dag_300                 | elk_layered   |   41.2 |   40.6 |  0.091 |  0.127 |     6 |    0.0 |  14.22 |  2.98 |    0 | 1.00 |
| random_dag_200                   | dagua         |     -- |   53.4 |     -- |  0.064 |     2 |    1.2 |   2.97 |  0.11 |    0 | 0.00 |
| random_dag_200                   | graphviz_dot  |   72.2 |   54.2 |  0.032 |  0.033 |     0 |   32.9 |   6.75 |  0.45 |    0 | 1.00 |
| random_dag_200                   | graphviz_sfdp |   52.3 |   43.8 |  0.007 |  0.016 |  1135 |   46.4 |   6.44 |  0.00 |    0 | 1.00 |
| random_dag_200                   | elk_layered   |   53.2 |   53.1 |  0.053 |  0.058 |    13 |    0.0 |   5.40 |  1.97 |    0 | 1.00 |
| long_skip_only_24                | dagua         |     -- |   68.5 |     -- |  0.015 |     0 |    0.0 |   0.92 |  0.00 |    0 | 0.00 |
| long_skip_only_24                | graphviz_dot  |   72.7 |   67.2 |  0.020 |  0.020 |     0 |   20.4 |   2.86 |  0.00 |    0 | 1.00 |
| long_skip_only_24                | graphviz_sfdp |   62.4 |   64.6 |  0.011 |  0.014 |    25 |   22.7 |   1.88 |  0.00 |    0 | 1.00 |
| long_skip_only_24                | elk_layered   |   62.9 |   65.0 |  0.020 |  0.020 |     4 |    0.0 |   3.63 |  1.22 |    0 | 1.00 |
| r79_undirected_sbm_low_mix_4x25  | dagua         |     -- |   39.3 |     -- |  0.148 |     9 |    2.2 |   3.65 |  2.82 |    0 | 0.00 |
| r79_undirected_sbm_low_mix_4x25  | graphviz_dot  |   59.4 |   53.0 |  0.049 |  0.040 |     0 |    7.3 |   2.40 |  0.46 |    0 | 1.00 |
| r79_undirected_sbm_low_mix_4x25  | graphviz_sfdp |   41.0 |   33.7 |  0.011 |  0.023 |  2539 |    6.1 |   5.85 |  0.00 |    0 | 1.00 |
| r79_undirected_sbm_low_mix_4x25  | elk_layered   |   36.3 |   53.6 |  0.107 |  0.042 |    91 |    0.0 |  13.08 |  2.46 |    0 | 0.89 |
| chung_lu_150                     | dagua         |     -- |   48.6 |     -- |  0.079 |     1 |    0.0 |   1.22 |  0.20 |    0 | 0.00 |
| chung_lu_150                     | graphviz_dot  |   71.4 |   41.2 |  0.039 |  0.075 |     0 |   38.3 |   6.27 |  0.62 |    0 | 1.00 |
| chung_lu_150                     | graphviz_sfdp |   46.0 |   32.8 |  0.023 |  0.040 |  1169 |   34.7 |   8.55 |  0.00 |    0 | 1.00 |
| chung_lu_150                     | elk_layered   |   54.7 |   47.6 |  0.046 |  0.073 |     8 |    0.0 |  15.35 |  2.66 |    0 | 1.00 |
| protein_ppi_200                  | dagua         |     -- |   65.0 |     -- |  0.024 |     0 |    1.4 |   2.07 |  0.53 |    0 | 0.00 |
| protein_ppi_200                  | graphviz_dot  |   73.4 |   62.2 |  0.009 |  0.015 |     0 |   14.1 |   6.01 |  0.39 |    0 | 1.00 |
| protein_ppi_200                  | graphviz_sfdp |   47.2 |   39.4 |  0.001 |  0.009 |  4416 |   16.5 |   5.17 |  0.07 |    0 | 1.00 |
| protein_ppi_200                  | elk_layered   |   62.3 |   63.3 |  0.019 |  0.020 |    36 |    0.0 |   9.77 |  2.50 |    0 | 1.00 |
| clustered_medium_5x20            | dagua         |     -- |   64.4 |     -- |  0.022 |    13 |    1.2 |   1.14 |  0.23 |    0 | 0.00 |
| clustered_medium_5x20            | graphviz_dot  |   66.0 |   47.9 |  0.034 |  0.014 |     0 |   16.0 |   4.47 |  0.82 |    0 | 1.00 |
| clustered_medium_5x20            | graphviz_sfdp |   49.6 |   44.0 |  0.006 |  0.017 |   317 |   29.5 |  13.34 |  0.03 |    0 | 1.00 |
| clustered_medium_5x20            | elk_layered   |   45.4 |   53.9 |  0.048 |  0.053 |   501 |    0.0 |   0.00 |  0.54 |    0 | 0.67 |
| r79_nested_clusters_3x2x10       | dagua         |     -- |   69.9 |     -- |  0.004 |     7 |    4.2 |   2.54 |  0.67 |    0 | 0.00 |
| r79_nested_clusters_3x2x10       | graphviz_dot  |   75.3 |   63.9 |  0.001 |  0.001 |     0 |   10.8 |   2.64 |  0.07 |    0 | 1.00 |
| r79_nested_clusters_3x2x10       | graphviz_sfdp |   48.8 |   51.5 |  0.000 |  0.001 |   129 |   21.3 |   4.62 |  0.06 |    0 | 1.00 |
| r79_nested_clusters_3x2x10       | elk_layered   |   38.5 |   68.0 |  0.078 |  0.031 |    68 |    0.0 |   0.00 |  0.74 |    0 | 0.78 |
| heavy_tail_weights_50            | dagua         |     -- |   57.7 |     -- |  0.042 |     6 |    0.0 |   3.85 |  0.43 |    0 | 0.00 |
| heavy_tail_weights_50            | graphviz_dot  |   73.7 |   60.8 |  0.020 |  0.022 |     0 |   24.5 |   6.51 |  0.18 |    0 | 1.00 |
| heavy_tail_weights_50            | graphviz_sfdp |   54.2 |   55.5 |  0.008 |  0.013 |    57 |   37.6 |   6.99 |  0.00 |    0 | 1.00 |
| heavy_tail_weights_50            | elk_layered   |   61.7 |   62.6 |  0.026 |  0.026 |     4 |    0.0 |   4.53 |  1.57 |    0 | 1.00 |
| r79_weighted_community_4x18      | dagua         |     -- |   57.9 |     -- |  0.047 |     2 |    0.0 |   8.46 |  0.33 |    0 | 0.00 |
| r79_weighted_community_4x18      | graphviz_dot  |   68.4 |   54.2 |  0.022 |  0.032 |     0 |   10.1 |   5.05 |  0.34 |    0 | 1.00 |
| r79_weighted_community_4x18      | graphviz_sfdp |   43.4 |   34.8 |  0.009 |  0.019 |   702 |   12.1 |   5.11 |  0.02 |    0 | 1.00 |
| r79_weighted_community_4x18      | elk_layered   |   58.5 |   56.5 |  0.030 |  0.040 |    20 |    0.0 |   5.32 |  2.30 |    0 | 1.00 |
```

Legend: nat = composite_drawing on the engine's NATIVE routing; dgr = composite_drawing on the engine's positions with DAGUA's router ('external positions + dagua routing'); natX/dgrX = routed crossing rate per variant; enX = edge-node crossings; port = port angular resolution (deg); curv = curvature CV; bend = mean bends/edge; lblN = edge-label-vs-node overlaps; cov = fraction of edges with captured native routes. dagua has no 'native vs dagua-routed' split (its native routing IS the dagua router), so nat is blank and dgr is its full-drawing score. Component columns describe the native variant when captured, else the dagua-routed variant.

## Observations

Where does dagua's full-drawing quality stand vs dot's native splines TODAY?

1. **dot's native splines lead on all 10 graphs, and the deficit is ROUTING,
   not placement.** dagua's full drawing (dgr row for engine=dagua) loses to
   dot-native everywhere (mean gap ~11 pts; worst chung_lu_150 48.6 vs 71.4,
   closest r79_nested_clusters 69.9 vs 75.3). But at MATCHED routing --
   comparing every engine under dagua's router -- dagua's positions win 7/10
   against dot (e.g. citation_dag_300 42.0 vs 37.2, clustered_medium 64.4 vs
   47.9). The placement engine is competitive; the router is what trails.

2. **Two router capabilities explain most of the gap.** (a) Node avoidance:
   dot's splines record ZERO edge-node crossings on every graph, while
   dagua's bezier router leaves 0-13 per graph and sfdp's straight lines
   leave thousands (8423 on citation_dag_300 -- straight-line force layouts
   plow through nodes at density). (b) Port angular resolution: dot spreads
   edge tangents at ports (10-46 deg min-angle means) where dagua's near-
   parallel bezier exits score 0-4 deg. These map directly to the two
   heaviest composite terms after crossings and are the concrete targets for
   the later routing-improvement stream.

3. **Applying dagua's router to external positions consistently DEGRADES
   engines that have real routing** (dot native 71.4 -> 41.2 dagua-routed on
   chung_lu_150; every dot row drops except none), confirming the router --
   not their positions -- carries much of dot's drawing advantage. Conversely
   for sfdp (no real routing, straight segments) the dagua-routed variant is
   the fair deployment score, and for ELK the two variants roughly tie on
   flat graphs (cov 1.00).

Measurement caveats recorded for downstream users:
- ELK's orthogonal style zeroes the port-angle term by construction (all
  edges leave vertically; min-angle = 0), so the port term systematically
  penalizes ortho routing; treat cross-style port comparisons with care.
- ELK cluster-internal edges carry container-relative coordinates; the
  root-level parser captures cov 0.67-0.89 on clustered graphs and their
  unrouted edges fall back to straight segments, which unfairly deflates
  ELK's native score there (36.3-45.4). The drawing_native_route_coverage
  field flags exactly this; do not read low-coverage native scores as
  engine quality.
- sfdp/neato-family "pos" splines are geometrically straight (bend 0.00,
  curv from endpoint arrows only): native capture works, the engines simply
  do not route.

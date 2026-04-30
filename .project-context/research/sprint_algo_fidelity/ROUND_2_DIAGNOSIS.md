# Round 2 Diagnosis: Dot / Sugiyama First Reproducer

## Gate Result

The round did not proceed to an algorithm fix. The live comparator is now
implemented and runs end-to-end, but the required sanity check failed: live
`classic_sugiyama` vs cached `graphviz_dot` does not match Round 1 cached
pairwise RMSDs within 0.005 on 8 of 22 graphs. The family median is
`0.324477` in Round 1 cached data and `0.341942` in the live comparator.

The discrepancy is caused by current node-size measurement, not by a confirmed
Sugiyama algorithm lever. The benchmark harness computes node sizes before
layout (`dagua/eval/benchmark.py:974`); with current sizing,
`mixed_width_labels` places the wide-label spine at `x=-51.337`, while the
cached benchmark Sugiyama positions place it at `x=-11.5`.

## Sugiyama Implementation Notes

- Rank assignment: Dagua uses Kahn longest-path layering followed by promotion
  to reduce dummy nodes, not Graphviz dot's network-simplex rank assignment.
  Evidence: `_longest_path_layering` at `dagua/layout/ops/sugiyama.py:172` and
  `_promote_layer_assignments` at `dagua/layout/ops/sugiyama.py:249`; `_AssignLayers`
  stores that result at `dagua/layout/ops/sugiyama.py:1495`.
- Crossing reduction: Dagua uses weighted barycenter down/up sweeps for 24
  passes from the pipeline default, not dot's median heuristic plus transpose.
  Evidence: `_BarycenterOrdering` is wired with `barycenter_passes=24` in
  `dagua/layout/ops/pipelines/sugiyama.py:36`, and `_barycenter_ordering`
  sorts by weighted barycenters at `dagua/layout/ops/sugiyama.py:497`.
- Coordinate assignment: Dagua uses Brandes-Koepf-style four-pass compaction,
  not dot's network-simplex coordinate assignment. Evidence:
  `_coordinate_assignment` at `dagua/layout/ops/sugiyama.py:662` delegates to
  `_brandes_koepf_x_positions` at `dagua/layout/ops/sugiyama.py:724`.
- Edge weights: Dagua accepts edge weights and expands them across dummy edges;
  weighted neighbor maps are used in barycenter scoring. Evidence:
  `_build_neighbor_weight_maps` at `dagua/layout/ops/sugiyama.py:458` and
  `_neighbor_barycenters` at `dagua/layout/ops/sugiyama.py:599`. The three
  diagnostic graphs have no explicit edge weights, so weights are uniform.

## Diagnostic Graphs

### mixed_width_labels

Node ids: `0=x`, `1=MultiHeadAttention(embed_dim=512, num_heads=8)`,
`2=LayerNorm(normalized_shape=(512,))`, `3=+`, `4=ReLU`, `5=out`.

Live layer assignments and ordering: `[[0], [1], [2], [3], [4], [5]]`.

Graphviz dot cached positions:
`0=(189,-378)`, `1=(117,-306)`, `2=(132,-234)`, `3=(197,-162)`,
`4=(197,-90)`, `5=(197,-18)`.

Comparison: layer assignment matches the obvious chain/skip structure, but
current live x-spacing is dominated by measured label widths
(`x=-51.337` for nodes 1 and 2) while cached Sugiyama used `x=-11.5`. This
pre-existing cache/live mismatch is larger than the proposed algorithm lever.

### shape_and_routing_matrix

Node ids: `0=input`, `1=ellipse_norm`, `2=diamond_gate`, `3=roundrect_path`,
`4=merge`, `5=circle_sink`.

Live layer assignments and ordering: `[[0], [1], [2, 3], [4], [5]]`.

Graphviz dot cached positions:
`0=(122.05,-339.28)`, `1=(122.05,-267.28)`, `2=(62.054,-195.28)`,
`3=(182.05,-195.28)`, `4=(122.05,-123.28)`, `5=(122.05,-34.641)`.

Comparison: layer assignment and x-ordering match dot. Current live spacing is
`x=+/-52.160`; cached Sugiyama spacing is `x=+/-22.5`; dot spacing is about
`+/-60` before Procrustes normalization.

### small_label_storm

Node ids: `0=input`, `1=prep`, `2=branch_left`, `3=branch_right`, `4=join`,
`5=output`.

Live layer assignments and ordering: `[[0], [1], [2, 3], [4], [5]]`.

Graphviz dot cached positions:
`0=(80,-388)`, `1=(80,-285)`, `2=(47,-203)`, `3=(145,-203)`,
`4=(86,-100)`, `5=(86,-18)`.

Comparison: layer assignment and x-ordering match dot. Current live spacing is
`x=+/-33.699`; cached Sugiyama spacing is `x=+/-22.5`; dot spacing is asymmetric
around the join after label/cluster effects.

## Dominant Divergence Cause

Confidence: high for the round blocker, medium for the algorithmic residual.

The dominant Round 2 blocker is not a confirmed Sugiyama algorithm bug; it is
that live runs cannot replay the cached benchmark's node-size context. The
algorithmic residual is still likely in coordinate assignment: Dagua's
Brandes-Koepf compaction (`dagua/layout/ops/sugiyama.py:724`) and width-aware
minimum separation (`dagua/layout/ops/sugiyama.py:1191`) do not match dot's
network-simplex coordinate problem. However, that is not a safe Round 2 fix
because the live baseline itself is not stable against Round 1.

## Proposed Single Fix

No Round 2 Sugiyama fix. First unblock fidelity measurement by making cached
and live graph sizing identical. The next algorithm lever, once measurement is
stable, should be a narrowly scoped coordinate-assignment experiment rather than
rank assignment or crossing reduction, because the three diagnostic graphs
already match dot's layer assignment and in-layer ordering.

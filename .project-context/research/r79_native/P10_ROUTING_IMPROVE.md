# P10: r80-S7 routing improvements -- evidence + gate verdict

Branch: r80/routing-improve (on top of r80/drawing-metrics).
Commits: aa2095a (node avoidance), 37290e4 (port spread), 1c21284 (optimizer
wiring), c1a25cc (edge labels). Arbiter: composite_drawing via
scripts/r80_drawing_probe.py, seed 42, same 10-graph corpus as P9.

## VERDICT: drawing gate FAIL (first failure; stopped per coordinator directive)

- Gate 1 (dagua dgr improves on >= 7/10):  6/10 improved -> FAIL
- Gate 2 (mean >= +4):                     mean -0.11    -> FAIL
- Gate 3 (enX == 0 on >= 8/10):            2/10 at zero  -> FAIL
- Gate 4 (port term up on >= 7/10):        10/10 up      -> PASS
- Gate 5 (no graph drops > 1.0):           2 violations  -> FAIL
  (long_skip_only_24 -4.5, r79_weighted_community_4x18 -2.1)

Placement-invariance gate: PASS. layout() output bit-identical
(max_abs_diff = 0.0) before vs after all routing commits on 5 graphs
(citation_dag_300, random_dag_200, clustered_medium_5x20,
r79_nested_clusters_3x2x10, heavy_tail_weights_50), seed 42, steps=15,
CPU. Snapshots: /tmp/r80_s7/positions_before.pt / positions_after.pt.
Structural argument: route_edges/place_edge_labels/maybe_refine_routes
only read `positions`; dagua.layout() never calls into the changed code.

Scoped tests: PASS. 213 tests green across tests/test_edge_routing_avoidance.py
(new, 13), tests/test_edge_route_quality_gate.py (new, 10), test_routing.py,
test_taxi_routing.py, test_ops_edge_route.py, test_edge_routing_config.py,
test_edge_optimization.py, test_edges_rectilinear_optimization.py,
test_cosmetic_edge_features.py, test_custom_edges.py, test_label_quality.py,
test_generate_edge_comparison.py, test_render_pair_edges.py,
test_render_edges_visibility.py, test_render_density_label.py.
Deselects per KNOWN_RED_TESTS.md: 6 pre-existing self-loop anchor failures
in test_routing.py (present at base 57bf178, unrelated to this work) and
the slow random_dag_200 legacy-dispatch case. ruff clean on all touched files.

## Before/after: dagua rows (the gated metric)

| graph                            | dgr before | dgr after | delta | enX   | port (deg)  | dgrX          |
|----------------------------------|-----------:|----------:|------:|-------|-------------|---------------|
| citation_dag_300                 | 42.0       | 44.1      | +2.1  | 7->1  | 0.0->7.4    | 0.121->0.142  |
| random_dag_200                   | 53.4       | 52.7      | -0.7  | 2->1  | 1.2->16.3   | 0.064->0.081  |
| long_skip_only_24                | 68.5       | 64.0      | -4.5  | 0->0  | 0.0->13.4   | 0.015->0.062  |
| r79_undirected_sbm_low_mix_4x25  | 39.3       | 39.5      | +0.2  | 9->7  | 2.2->2.4    | 0.148->0.148  |
| chung_lu_150                     | 48.6       | 47.6      | -1.0  | 1->0  | 0.0->18.6   | 0.079->0.108  |
| protein_ppi_200                  | 65.0       | 65.6      | +0.6  | 0->1  | 1.4->7.9    | 0.024->0.029  |
| clustered_medium_5x20            | 64.4       | 65.0      | +0.6  | 13->10| 1.2->8.5    | 0.022->0.028  |
| r79_nested_clusters_3x2x10       | 69.9       | 72.6      | +2.7  | 7->5  | 4.2->12.0   | 0.004->0.005  |
| heavy_tail_weights_50            | 57.7       | 58.7      | +1.0  | 6->6  | 0.0->10.7   | 0.042->0.049  |
| r79_weighted_community_4x18      | 57.9       | 55.8      | -2.1  | 2->3  | 0.0->6.3    | 0.047->0.060  |

Mean delta -0.11. Full after-table (all 4 engines x 10 graphs):
mirrored alongside this doc as P9_AFTER_S7.md (generated
2026-07-09T03:50Z by scripts/r80_drawing_probe.py, unmodified).

## The headline finding: the router got better; the gate is failed by a
## router-x-placement interaction, not by the router in isolation

The same router changes, applied to EXTERNAL engines' positions (the dgr
column for dot/elk/sfdp rows -- same code path, only positions differ):

- graphviz_dot positions:  improved 10/10, mean +6.15
- elk_layered positions:   improved 10/10, mean +4.00
- graphviz_sfdp positions: improved  7/10, mean +4.07

Those numbers MEET the gate thresholds (>=7/10, mean >= +4) on every
external engine. On dagua's own positions the identical changes are flat
(6/10, -0.11) because dgrX -- the routed edge-EDGE crossing rate, the
heaviest composite term (weight 30 vs 20 for edge-node, 12 for ports) --
rises on 9/10 graphs (e.g. long_skip 0.015->0.062, chung_lu 0.079->0.108).
dagua's native placements are more compact, with tighter shared corridors;
fanning tangents apart (46-deg budget) and bowing curves around node boxes
there buys edge-node/port wins that are more than paid back in new
edge-edge crossings. dot-style placements are roomier, so the same
deflections are free.

Secondary shortfall: gate 3 (enX zero on >= 8/10) was structurally out of
reach for the bounded perpendicular-push deflector on this corpus -- it
clears isolated blockers (7->1 on citation_dag_300) but not dense cluster
interiors (clustered_medium 13->10, sbm 9->7), and each cleared box can
re-cross a neighbor in packed neighborhoods (protein 0->1, weighted 2->3).
dot achieves 0 with a full piecewise spline router around obstacle
polygons, not a 2-control-point bow.

## Per-deliverable summary

1. Node-bbox avoidance (aa2095a) -- route_edges() deflects bezier control
   points around non-endpoint node bboxes; spatial-hash candidate lookup;
   growing-offset ladder (2x/4.5x/9x/16x base, capped 1.5x chord);
   dense-neighborhood fallback leaves the edge as-is (never loops).
   EdgeStyle.avoid_nodes, ON by default, bezier only, deterministic.
   Effect: enX down on 6/10 graphs (7->1 citation, 13->10 clustered,
   9->7 sbm, 2->1 random, 1->0 chung_lu, 7->5 nested); cost: slight
   dgrX increase where deflected curves cross neighbors.
2. Port angular spread (37290e4) -- rank-based tangent-rotation bias
   (+-23 deg for 2 ports, narrowing with fan-out; 46-deg total budget)
   applied after all existing curve-shape branches; primary
   crossing-reduction sort untouched. Effect: port term up 10/10 (gate 4,
   the only PASS), mean port 1.1 -> 10.4 deg (dot: 10-46); cost: main
   contributor to the dgrX rise on compact dagua placements.
3. Optimizer wiring (1c21284) -- maybe_refine_routes() in
   edge_optimization.py: draft/balanced keeps Sprint 6 adaptive-skip
   verbatim; quality >= 0.75 forces the differentiable pass with
   BezierControlPointOptConfig's fuller weights (_ForcedQualityEdgeConfig;
   w_edge_angular_res 2.0 and w_edge_curvature_consistency 1.0, both 0.0
   in LayoutConfig defaults). BezierControlPointOpt's mechanism
   (optimize_edges) is what got wired; the op class itself lives in
   ops/edge_route.py and its pipeline composition is S2b-owned this round.
   Wall-time: 226.7s on a 200-edge graph at quality=high -- measured on
   this box at load ~90/20 cores, so heavily contention-inflated, but even
   /10 it is far over the brief's <1s bar. RECOMMENDATION: do NOT enable
   at balanced; keep high/max only. Re-measure on an idle machine before
   revisiting.
4. Edge labels (c1a25cc) -- t-offset ladder 5->9, perpendicular ladder
   3->5, new label-vs-edge-path crossing term (self-excluding, area-scaled)
   via pre-sampled 20-point polylines + AABB reject. NOT measurable on
   this probe corpus: all 10 graphs have zero edge labels (lblN=0 every
   row in P9 before AND after). Verified by unit tests instead.

## Spot-check renders (reviewed; not committed)

/tmp/r80_s7_renders/citation_dag_300.png, clustered_medium_5x20.png,
heavy_tail_weights_50.png -- routed with the new pipeline on seed-42
CPU quick layouts (steps=15, rescaled to sane extent before routing;
routing behavior is what is being inspected, not placement quality).
Reviewed all three:
- heavy_tail_weights_50: curves bow cleanly around blocking nodes, hub
  fan-outs separate visibly at ports, no control-point blowups.
- citation_dag_300: high-fan-out hubs (nodes 0/1) show clear angular
  spread; edges dodge node boxes where corridors allow.
- clustered_medium_5x20: routing works, but SHORT edges near cluster
  boundaries occasionally form small lasso/loop curls -- the deflection
  ladder's offset is large relative to a short chord. This is the
  visible face of the dgrX increase and a concrete input to the
  follow-up (scale deflection offset by chord length on short edges).
Render caveat: rendering the raw steps=15 snapshots without rescaling
allocated a ~55 GB figure (extent 176k units on citation_dag_300) and
had to be killed; the rescale-then-route protocol above avoids that.

## Targeted follow-up (NOT attempted here -- gate failed once, stopped)

The failure mode is crisp and local: make both new deflections
crossing-aware. (a) After deflecting/biasing an edge, count new
edge-edge crossings against already-routed neighbors (the 20-point
polylines from deliverable 4 are already available) and keep the change
only if net composite-weighted term improves; (b) scale the port-spread
budget by local corridor density (or placement source) instead of a
fixed 46 deg; (c) long_skip_only_24 (-4.5, the worst regression) is a
skip-connection ladder where spread tangents on parallel long edges
serially cross each other -- cap spread when peer edges are
near-parallel. Expected to preserve the external-position wins (which
already meet the gate) while removing the dagua-row regressions.

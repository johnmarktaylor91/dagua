# r80-S7: Close the routing gap (node avoidance + port spread + labels)

## Context (verified by S6, evidence in p3 .project-context/research/r79_native/P9_DRAWING_BASELINE.md)
- composite_drawing (new, on branch r80/drawing-metrics) scores full drawings 0-100.
  dot's native splines lead all 10 probe graphs by ~11 pts mean over dagua's drawings --
  but at MATCHED routing, dagua's positions beat dot 7/10. The gap is the router.
- Two capabilities explain most of it: (1) node avoidance -- dot has ZERO edge-node
  crossings on every probe graph, dagua 0-13, and (2) port angular spread -- dot 10-46
  deg between incident edges, dagua 0-4 deg (edges leave nodes as near-parallel bundles).
- A differentiable route optimizer ALREADY EXISTS and is orphaned: BezierControlPointOpt /
  ReconstructEdgeRoutes (dagua/layout/edge_optimization.py, ops in
  dagua/layout/ops/edge_route.py; registered, unit-tested, composed into zero pipelines).
  Its loss terms include edge-node crossing and port angular resolution.
- Routing is POST-PLACEMENT: node positions are frozen inputs. Your changes must not
  move nodes -- the placement benchmark is invariant by construction; prove it.

## Setup
Work in /home/jtaylor/.claude/worktrees/dagua-native-p3 (venv exists). Create branch:
`git checkout -b r80/routing-improve` (on top of r80/drawing-metrics -- you need
composite_drawing as the arbiter). df -h / first; stop if < 10GB.

## Deliverables (in priority order; each lands as its own commit with before/after numbers)

### 1. Node-bbox avoidance in the default router
Extend route_edges() (dagua/edges.py) with node-box deflection: when a routed curve
passes through a non-endpoint node's bbox (inflated by a small margin), deflect control
points around it -- generalize the existing _deflect_around_clusters mechanism
(edges.py:1295). Deterministic, no RNG. Handle the fallback: if deflection cannot clear
the box (dense neighborhoods), leave the edge as-is (never loop forever). Config knob on
EdgeStyle or LayoutConfig consistent with existing routing options; ON by default for
bezier routing (measure the cost).

### 2. Port angular spread
In the port assignment path (_compute_directional_ports and the sort at
edges.py:713-724): distribute incident-edge attachment points and initial tangents so
adjacent edges leave the node with a minimum angular separation where geometry allows
(target the composite_drawing port term; dot achieves 10-46 deg). Respect shape-aware
port projection (_adjust_port_for_shape). Keep the crossing-reduction ordering property
(sorted by neighbor position) as the primary sort; spread within it.

### 3. Wire the orphaned optimizer as the high-quality path
Compose BezierControlPointOpt + ReconstructEdgeRoutes as an OPT-IN refinement pass over
the heuristic routes (deliverable 1+2 output as init), exposed via the existing
edge_opt_steps config field (recon found it unreachable) and gated by the quality knob:
off at draft/balanced, on at high/max (measure wall-time; if <1s on 200-edge graphs,
propose enabling at balanced in your report -- do not enable yourself).
Positions stay frozen: verify the op does not touch node positions.

### 4. Edge-label placement upgrade (if time/budget allows; separate commit)
place_edge_labels (edges.py:1368): widen the candidate search (more t_offsets, both
sides, perpendicular nudges) and add label-vs-edge-path overlap to the candidate cost
(currently only label-vs-node and label-vs-label). Keep greedy + deterministic.

## Gates
1. Placement invariance: routing changes must not alter positions -- assert layout()
   output bit-identical on 5 graphs before/after your changes.
2. Drawing gate: rerun scripts/r80_drawing_probe.py. Acceptance: dagua's drawing score
   (dgr column for dagua rows) improves on >= 7/10 probe graphs, mean +4 or better,
   edge-node crossings reach 0 on >= 8/10, port term improves on >= 7/10, and NO probe
   graph's dagua drawing score drops by > 1.0. Also spot-check 3 renders visually
   (save PNGs under /tmp/r80_s7_renders, mention paths in the report; do not commit them).
3. Scoped tests: existing edge/routing tests + new tests for deflection, port spread,
   optimizer wiring (KNOWN_RED_TESTS.md deselects apply; no bare pytest -x).
4. ruff on touched files.

## Output contract
- Commits on r80/routing-improve; evidence doc
  .project-context/research/r79_native/P10_ROUTING_IMPROVE.md (before/after probe table,
  per-deliverable gains, wall-time cost, renders reviewed); durable-mirror the doc to
  ~/.claude/research/dagua/r80-drawing-metrics/.
- Final message: probe deltas, whether optimizer should default-on at balanced, concerns.

## Hard rules
- Do NOT touch: dagua/layout/ops/pipelines/** (S2b owns native_undirected this round),
  dagua/layout/projection.py, dagua/metrics.py composite functions (arbiter is frozen
  during the round), scripts/r79_baseline.py, frozen stores.
- Node positions are read-only everywhere.
- ASCII only; clean /tmp scratch except the 3 named render PNGs; watch disk.

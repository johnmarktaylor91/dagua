# r80-S6: Full-drawing measurement layer (routing + labels), additive only

## Context
Recon (briefs/r80_s5_full_element_recon.md ran; findings below are verified with
file:line) showed dagua computes curve-aware quality metrics but never scores them, and
the benchmark discards external engines' routing/label output at parse time. This stream
builds the MEASUREMENT layer for full-drawing quality. It is strictly ADDITIVE: the
existing node-placement composites and the frozen r79 baseline scoring MUST NOT change --
two other agents' regression gates depend on them staying bit-identical.

Verified facts to build on (do not re-derive):
- Curve-aware metrics exist: edge_node_crossing_count (dagua/metrics.py:1147),
  edge_curvature_consistency (:1247), port_angular_resolution (:1293),
  label_overlap_count (:1195). Computed only on the benchmark "full" branch
  (dagua/eval/benchmark.py:790-800, passes curves=route_edges(...)), never enter any
  composite (metrics.py:1394-1616 checked line by line).
- sampled_crossing_rate (metrics.py:749) intersects straight node-center segments only;
  no routed-path crossing metric exists. No bend-count metric exists.
- graphviz adapter (dagua/eval/competitors/graphviz_competitor.py:345-370) parses dot
  -Tjson but reads only node "pos"; edge spline control points and _ldraw_ label ops in
  the SAME payload are dropped. ELK adapter (elk_competitor.py, response parse ~:178)
  drops sections[].bendPoints.
- CompetitorResult (dagua/eval/competitors/base.py:17-23) has no routing/label fields.
- dagua's own routing: route_edges() (dagua/edges.py:658) render-time bezier/ortho/taxi;
  edge labels placed by place_edge_labels() (edges.py:1368).

## Setup
Create worktree + branch (check `df -h /` first; STOP if < 12GB free):
  git -C /home/jtaylor/.claude/worktrees/dagua-native worktree add \
      /home/jtaylor/.claude/worktrees/dagua-native-p3 -b r80/drawing-metrics
  cd /home/jtaylor/.claude/worktrees/dagua-native-p3 && uv venv .venv && \
  uv pip install -p .venv/bin/python -e ".[dev]" python-igraph
Verify `import dagua` resolves inside p3.

## Deliverables

### 1. Routed-path-aware crossing metric
New function in dagua/metrics.py (e.g. routed_crossing_rate): sample points along the
actual curves (BezierCurve polyline sampling likely exists in dagua/edges.py -- reuse)
and count segment intersections, same sampling/seed discipline as sampled_crossing_rate.
Must degrade gracefully to the straight-line result when curves are straight. Add a
bend_count metric for ortho/taxi routings while you are in there (number of direction
changes above a small angle threshold, normalized per edge).

### 2. composite_drawing (NEW function; do not touch existing composites)
dagua/metrics.py: composite_drawing(...) combining: routed crossing rate, edge-node
crossing rate, port angular resolution, curvature consistency, label overlap counts
(edge-label vs node and vs label), and node-overlap as a sanity term. Document the
weights with a rationale comment; scale to 0-100 like the placement composite. Two
requirements: (a) deterministic given fixed seed; (b) computable from
(positions, sizes, curves, label_positions) without a live layout run.

### 3. Capture external routing/labels (graphviz + ELK only this round)
- Extend CompetitorResult with OPTIONAL fields (routes/edge label geometry), default
  None -- every existing adapter keeps working unchanged.
- graphviz adapter: also parse edge spline control points ("pos" on edge objects; note
  the "e," arrow-endpoint prefix convention) and edge/graph label draw positions
  (_ldraw_). Convert splines into the same polyline/curve representation your metric
  samples (document the conversion; graphviz emits cubic B-splines).
- ELK adapter: parse sections[].bendPoints into polylines.
- Persistence: extend the benchmark store schema with an OPTIONAL per-row routes blob
  (parallel to positions/*.pt -- e.g. routes/*.pt). MUST be backward compatible: absence
  = None, no validator failures on old stores. Update
  scripts/validate_benchmark_integrity.py only if it would otherwise false-alarm.

### 4. Benchmark wiring (additive)
In the benchmark full-compute path (dagua/eval/benchmark.py:790 area): compute
composite_drawing for dagua (own route_edges + place_edge_labels output) and, when
captured, for externals (their native curves; their labels only if available -- record
which fields were native vs None). Store alongside existing metrics WITHOUT touching the
existing composite fields or W/T/L logic. For engines with no native routing (force
engines), compute a second variant: composite_drawing on their positions with DAGUA's
router applied -- labeled clearly as "external positions + dagua routing" (this measures
the combined system fairly and is the deployment-relevant comparison).

### 5. Proof run (small, not the full corpus)
Script scripts/r80_drawing_probe.py: run 10 representative graphs (mix: 3 layered DAGs,
3 undirected community, 2 clustered, 2 weighted) x {dagua, graphviz_dot, graphviz_sfdp,
elk_layered}; print a table of composite_drawing + component terms, both native-routing
and dagua-routed variants. Save to
.project-context/research/r79_native/P9_DRAWING_BASELINE.md with observations (where does
dagua's drawing quality stand vs dot's native splines TODAY?).

## Gates
1. Scoped tests: new tests for routed_crossing_rate (straight==sampled_crossing_rate on
   straight curves; a crafted curved case where they differ), bend_count,
   composite_drawing determinism, adapter spline parsing (use a small checked-in dot
   -Tjson fixture), store roundtrip with and without routes. Consult KNOWN_RED_TESTS.md;
   never bare pytest -x.
2. Invariance proof: run scripts/r79_baseline.py --rescore-only (or equivalent smallest
   path) and diff the resulting composites/W-T-L against the frozen baseline -- MUST be
   bit-identical (your changes are additive; prove it).
3. ruff on touched files.

## Output contract
- Commits on r80/drawing-metrics; evidence doc P9_DRAWING_BASELINE.md as above.
- Final message: what was built, the 10-graph probe table, invariance proof result,
  and your top-3 observations about where dagua's full-drawing quality stands.

## Hard rules
- Do NOT modify: existing composite functions, scripts/r79_baseline.py scoring/W-T-L
  logic, dagua/layout/** (this stream is metrics/eval/adapters only -- routing
  IMPROVEMENTS are a later stream), projection.py, native_stress.py, dagua_native.py.
- dagua/edges.py: read/reuse only; if you need a sampling helper that does not exist,
  add it in metrics.py or a new eval helper module, not in edges.py.
- ASCII only; watch disk; clean /tmp scratch.

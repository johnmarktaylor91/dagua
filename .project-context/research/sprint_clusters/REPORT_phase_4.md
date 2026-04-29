# Phase 4 Report: Edge Clipping at Cluster Perimeter

## Changes

- Added polyline/rectangle intersection and cluster-boundary clipping helpers in `dagua/edges.py`.
- Wired render-time cluster clipping in `dagua/render/mpl.py` using the same rendered cluster bbox math as cluster drawing.
- Extended `DaguaEdge` in `dagua/render/edges/collection.py` with an optional body-only curve so arrowheads remain seated on node ports while the edge body is clipped at cluster perimeters.
- Applied the same body-only clipping to direct-render edge styles.
- Added `tests/test_render/test_edge_cluster_clip.py` covering external-to-internal clipping, nested outermost clipping, disabled `cluster_aware`, and the render-time config gate.

## Assumptions

- `config=None` keeps default `cluster_aware=True`, matching `LayoutConfig` defaults.
- Clipping is applied only to visible-stroke clusters: effective border opacity must be positive and stroke width must be positive.
- Arrowhead and label placement continue to use the original routed curve. This preserves target-port arrowheads and avoids tangent changes caused by the shortened body.

## Test Results

- `pytest tests/test_render/test_edge_cluster_clip.py -x --tb=short -q`
  - `4 passed in 0.15s`
- `ruff check . --fix`
  - `All checks passed!`
- `mypy --follow-imports=silent dagua/cli.py`
  - `Success: no issues found in 1 source file`
- `pytest tests/test_layout/ tests/test_render/ tests/test_style.py tests/test_parity_metrics.py -x --tb=short -q`
  - `395 passed, 8 warnings in 1202.32s`
- `python scripts/graphviz_theme_comparison.py --output-dir eval_output/cluster_phase_4_check`
  - Wrote 45 comparison rows to `eval_output/cluster_phase_4_check`
- Parity lock from `/tmp/test_parity_metrics_lock.json`
  - Global in-tolerance: `99.27%`
- Final broader tier attempted: `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`
  - Failed during collection on unrelated import debt: `tests/test_classic_drl.py` cannot import `layout_drl` from `dagua.layout.classic`.

## Visual Observations

- `transformer_block`: external edges now meet cluster boundaries cleanly; body strokes no longer continue visibly through cluster interiors before terminal arrowheads.
- `nested_clusters`: incoming/outgoing cluster-boundary edges clip at the outer group perimeter rather than the nested child perimeter.
- `cross_cluster_edges`: cross-cluster bodies stop at visible cluster borders and preserve node-port arrowheads.

## Controversial Choices

- The implementation clips body curves in the renderer instead of mutating routed curves. This keeps cached routing and arrow geometry stable while solving the cosmetic stroke-through defect.
- Bezier clipping uses a 96-sample polyline approximation to find the perimeter crossing, then renders a Bezier subcurve. This is a bounded cosmetic approximation and avoids adding a cubic-rectangle root solver.

## Concerns

- The full non-slow test tier has unrelated collection debt in `tests/test_classic_drl.py`.

## Knowledge

- Render-time cluster bboxes differ from routing-time rough bboxes because labels, depth padding, and minimum-width clamps are applied in `mpl.py`.
- The custom edge collection can safely carry a body-only curve while keeping arrowheads and labels on the original curve.

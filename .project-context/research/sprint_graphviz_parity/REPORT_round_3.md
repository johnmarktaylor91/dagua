# Graphviz Strict Theme Cosmetic Round 3 Report

## What Fixed

- `dagua/styles.py`
  - Normalized `graphviz_strict` node and edge label sizes from 14.0 pt to 10.5 pt.
  - Reduced strict node stroke width from 1.0 to 0.75.
  - Lightened strict cluster borders from `#666666` to `#AAAAAA` while keeping `border_opacity=1.0`.
  - Reduced strict back-edge curvature from 0.3 to 0.2.
- `dagua/edges.py`
  - Added duplicate source-target edge detection and alternated curvature sign on successive parallel edges.
  - Allowed negative curvature to mirror Bezier control points across the edge chord.
- `dagua/render/edges/arrowheads.py`
  - Marked `tee`/`bar` arrowhead primitives as stroke-only so filled mode cannot turn them into filled heads.
- `tests/test_style.py`
  - Updated strict-theme assertions for normalized fonts, lighter cluster stroke, thinner node stroke, and tighter back curvature.
- `tests/test_routing.py`
  - Added a regression test for duplicate back-edges alternating across both sides of the shared chord.

## Cluster Label Investigation

Temporary debug instrumentation was added to `_cluster_font_size_data()` and removed before finishing.
Fresh `cluster_showcase` rendering showed the strict-theme cluster path is reached with:

```text
text='Large Cluster With Longer Label'
font_size_points=10.0
font_size_scaling='fixed'
```

The regenerated strict panel confirms the cluster label is fixed-size. The oversized label visible in the
three-way `cluster_showcase.png` is in the `Dagua (improved)` panel, not `Dagua (strict)`.

## Visual Verification

Regenerated the gallery:

```text
python scripts/graphviz_theme_comparison.py --output-dir eval_output/graphviz_theme_round_3
Wrote 45 comparison rows to eval_output/graphviz_theme_round_3
```

Inspected:

- `three_way/cluster_showcase.png`: strict cluster labels are fixed-size; improved still has large height-scaled labels.
- `three_way/deep_nesting_4.png`: strict Level labels are small and fixed-size.
- `three_way/pipeline.png`: strict node labels fit with more horizontal whitespace and thinner ellipse strokes.
- `three_way/balanced_binary_tree.png`: strict leaf labels are smaller and less crowded.
- `three_way/diamond.png`: strict ellipse outlines are thinner.
- `three_way/nested_clusters.png`: strict cluster borders read as lighter gray hairlines.
- `three_way/state_machine.png`: strict back-arcs are tighter with `curvature=0.2`.
- `three_way/multi_cycle.png`: no stray gray rectangle behind the strict panel; back-arc is tighter.
- `three_way/arrow_types.png`: strict `tee` renders as a flat bar.
- `three_way/complete_k5.png`: no duplicate node-pair edges exist in this fixture, so the new alternation path is not visible there; it is covered by `tests/test_routing.py::TestRouteEdges::test_parallel_back_edges_alternate_curvature_side`.

## Tests

Passed:

```text
ruff check . --fix
All checks passed!
```

```text
mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file
```

```text
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
258 passed, 1 warning in 1145.64s (0:19:05)
```

```text
pytest tests/test_style.py tests/test_routing.py::TestRouteEdges::test_parallel_back_edges_alternate_curvature_side tests/test_custom_edges.py tests/test_render/ -x --tb=short -q
213 passed, 1 warning in 44.82s
```

Final Tier 2 remains blocked by out-of-scope pre-existing import/collection failures:

```text
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
ERROR tests/test_classic_drl.py
ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'
```

```text
pytest tests/ --ignore=tests/test_classic_drl.py -x --tb=short -q -m "not slow and not benchmark and not rare"
ERROR tests/test_classic_fa2.py
ImportError: cannot import name 'layout_fa2' from 'dagua.layout.classic'
```

```text
pytest tests/ --ignore-glob='tests/test_classic_*.py' -x --tb=short -q -m "not slow and not benchmark and not rare"
ERROR tests/test_layout_ops.py
ImportError: cannot import name 'MultilevelVCycle' from 'dagua.layout.ops'
```

```text
pytest tests/ --ignore-glob='tests/test_classic_*.py' --ignore=tests/test_layout_ops.py -x --tb=short -q -m "not slow and not benchmark and not rare"
ERROR tests/test_scripts/test_fidelity_pipeline.py
ImportError: cannot import name 'build_report_tex' from 'scripts.generate_fidelity_report'
```

Also observed while running a broad focused check:

```text
pytest tests/test_style.py tests/test_routing.py tests/test_custom_edges.py tests/test_render/ -x --tb=short -q
FAILED tests/test_routing.py::TestRouteEdges::test_self_loop_routing
assert (64.0, 60.0) == (36.0, 60.0)
```

That self-loop expectation predates this change; the round-3 regression test was run explicitly and passed.

## Deviations

- Did not modify `dagua/layout/` or `scripts/graphviz_theme_comparison.py`.
- Did not fix the out-of-scope Tier 2 import blockers.
- The K5 fixture has no duplicate edges between the same node pair, so the requested parallel-edge alternation cannot be visually demonstrated on `complete_k5.png`; it is implemented for actual duplicate source-target edges and tested directly.

## Dead Code

No newly unreachable code identified.

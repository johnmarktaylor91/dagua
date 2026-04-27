# Graphviz Strict Theme Cosmetic Round 2 Report

## What Fixed

- `dagua/styles.py`
  - Added `ClusterStyle.fill_opacity`, `ClusterStyle.border_opacity`, and
    `ClusterStyle.font_size_scaling`.
  - Set `graphviz_strict` cluster labels to `font_size_scaling="fixed"` so the
    declared `font_size=10.0` remains authoritative.
  - Kept strict cluster fill at `fill_opacity=0.15` while restoring the border
    to `border_opacity=1.0`.
  - Reduced strict default node `stroke_width` from `1.3` to `1.0`.
  - Added a fully specified strict `"back"` edge style with `curvature=0.3`.
- `dagua/render/mpl.py`
  - Added fixed cluster-label sizing support to `_cluster_font_size_data()`.
  - Routed all cluster label measurement/draw paths through the new scaling
    mode.
  - Split cluster fill and border alpha calculation so subdued fills no longer
    force faint strokes.
- `tests/test_style.py`
  - Updated strict-theme assertions for the new stroke width, back-edge
    curvature, fixed cluster font policy, and split cluster alpha fields.

## Visual Verification

Regenerated the gallery:

```text
python scripts/graphviz_theme_comparison.py --output-dir eval_output/graphviz_theme_round_2
Wrote 45 comparison rows to eval_output/graphviz_theme_round_2
```

Inspected:

- `three_way/cluster_showcase.png`: strict cluster labels render small and
  fixed-size instead of height-scaled; the long label no longer dominates.
- `three_way/deep_nesting_4.png`: nested cluster labels are fixed-size.
- `three_way/nested_clusters.png`: cluster borders are visible while fills stay
  faint.
- `three_way/complete_k5.png`: no stray gray background rectangle visible in
  the strict panel.
- `three_way/pipeline.png`: node ellipse strokes are thinner hairlines.
- `three_way/state_machine.png`: back-edges remain curved but hug the graph more
  tightly than the previous wide arcs.

## Font Measurement

Native Graphviz `dot -Tsvg` on a one-off default graph emitted:

```text
native_svg_font_sizes= ['14.00', '14.00', '14.00', '14.00']
```

Dagua rendered the same Graphviz-positioned graph at 72 DPI using
`graphviz_strict`; its SVG backend converts text to paths, so there are no
direct `font-size` attributes to compare. Measuring the generated node label
path bboxes gave:

```text
Input bbox_height_px= 10.51
Preprocess bbox_height_px= 10.51
```

That visible glyph height is consistent with 14 pt Times text at 72 DPI, where
the glyph ink bbox is smaller than the nominal em size. I left node
`font_size=14.0`, `EdgeStyle.label_font_size=14.0`, and
`GraphStyle.edge_label_font_size=14.0` unchanged as a DPI/glyph-metrics artifact,
not a real font-size mismatch.

## What Did Not Change

- I did not reduce node or edge label font sizes because native dot's SVG
  declares 14 pt defaults and Dagua's 72-DPI visible glyph height matches that
  interpretation.
- I did not attempt exact Graphviz libspline back-edge routing. The strict
  `"back"` style now uses a tighter cosmetic curvature, but exact channel
  routing remains out of scope.

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
pytest tests/test_style.py tests/test_render tests/test_custom_edges.py -x --tb=short -q
212 passed, 1 warning in 46.85s
```

```text
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
258 passed, 1 warning in 1138.62s (0:18:58)
```

Final Tier 2 remains blocked by the same out-of-scope layout import failure:

```text
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"

==================================== ERRORS ====================================
__________________ ERROR collecting tests/test_classic_drl.py __________________
ImportError while importing test module '/home/jtaylor/projects/dagua/tests/test_classic_drl.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../anaconda3/envs/py311/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/test_classic_drl.py:10: in <module>
    from dagua.layout.classic import layout_drl
E   ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic' (unknown location)
=========================== short test summary info ============================
ERROR tests/test_classic_drl.py
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
1 error in 0.21s
```

## Deviations

- The final suite does not pass because it fails before running tests on a
  `dagua/layout/` import. Layout code was explicitly out of scope for this
  cosmetic pass, so I did not modify it.
- I added general `ClusterStyle` fields rather than a graphviz-strict-only
  renderer branch. Existing themes keep the legacy behavior because the new
  fields default to `None` and `font_size_scaling="by_height"`.

## Dead Code

No newly unreachable code identified.

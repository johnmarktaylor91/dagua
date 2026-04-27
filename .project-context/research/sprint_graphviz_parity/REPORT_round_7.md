# Graphviz Strict Theme Cosmetic Round 7 Report

## Changes

- `dagua/render/mpl.py`
  - Replaced the hard `width / height > 2.0` strict-ellipse gate with a uniform
    render-scale adjustment and an aspect cap for extreme single-line labels.
  - Added strict-theme radial reclipping of ellipse edge terminals so arrow tips
    land on the rendered ellipse boundary.
  - Added strict edge-label collision nudging for overlapping edge labels.
  - Raised the strict back-edge offset floor from `36pt` to `60pt`.
  - Increased strict cluster label mask padding from `3pt` to `4pt`.
  - Updated direct/manual `open` and `crow` markers to filled polygons.
- `dagua/render/edges/arrowheads.py`
  - Mapped named Graphviz `open` to filled `normal` behavior, overriding hollow
    fill mode for the named arrow.
  - Rebuilt `crow` as one compact filled Graphviz-style polygon instead of three
    detached tine ribbons.
- `dagua/styles.py`
  - Raised `graphviz_strict` cluster `fill_opacity` from `0.08` to `0.10`.
  - Warmed strict cluster fill from `#F0F0F0` to `#F2EFE9`.
- Tests updated:
  - `tests/test_style.py`
  - `tests/test_custom_edges.py`
  - `tests/test_render/test_mpl.py`

## Assumptions

- Layout-side defects remain out of scope. I did not touch `dagua/layout/`.
- `scripts/graphviz_theme_comparison.py` was not modified.
- The improved `graphviz` theme was not modified.
- Full `sqrt(2)` visual scaling was not viable on Graphviz-positioned fixtures:
  it reintroduced the long-label ellipse balloon and caused dense panels to
  overlap. I used a smooth, uniform per-node scale capped for extreme aspect
  ratios; this keeps short/medium ellipses rounder while preventing long-label
  blow-up.

## Verification

Graphviz arrow cross-checks:

```text
echo 'digraph { a -> b [arrowhead="crow"] }' | dot -Tsvg -o /tmp/crow_test.svg
<polygon fill="black" stroke="black" points="27,-46.1 ..."/>
```

```text
echo 'digraph { a -> b [arrowhead="open"] }' | dot -Tsvg -o /tmp/open_test.svg
<polygon fill="black" stroke="black" points="27,-36.1 ..."/>
```

Gallery regenerated:

```text
python scripts/graphviz_theme_comparison.py --output-dir eval_output/graphviz_theme_round_7
Wrote 45 comparison rows to eval_output/graphviz_theme_round_7
```

Two-way crops regenerated under:

```text
eval_output/graphviz_theme_round_7/two_way
```

Visual readback:

- `long_labels.png`: no longer has the three-node ellipse engulfing regression.
- `arrow_types.png`: short ellipses are rounder; `crow` and `open` render as filled dark markers.
- `state_machine.png`: edge labels `retry` / `resume` are separated; back-edge side channels are wider.
- `diamond.png`: arrow tips meet the visible ellipse boundary without biting into node interiors.
- `balanced_binary_tree.png`: leaf arrow tips meet boundaries; residual tight leaf spacing is layout-side.
- `pipeline.png`: medium-label ellipses remain rounder and arrow tips sit on boundaries.

## Test Results

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
pytest tests/test_style.py tests/test_custom_edges.py tests/test_render/test_mpl.py tests/test_arrowheads.py -x --tb=short -q
258 passed, 1 warning in 44.99s
```

```text
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
258 passed, 1 warning in 1209.48s (0:20:09)
```

Blocked by pre-existing out-of-scope import error:

```text
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
ERROR tests/test_classic_drl.py
ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'
```

## Controversial Choices

- The strict ellipse adjustment is not literal full `sqrt(2)` in the renderer.
  Full uniform scaling was tested visually and caused the same long-label
  regression plus dense-panel overlaps. The final logic uses a smaller uniform
  scale with an aspect cap, and caps render height for extreme long-label
  ellipses.
- The named `open` arrow is forced filled even when an edge style asks for
  `arrow_fill="hollow"`, because Graphviz 8.0.3 renders named `open` filled.
  The `o...` open-prefix forms still use hollow behavior.

## Concerns

- `state_machine.png` still differs structurally from dot because node placement
  and spline routing are layout/routing parity issues beyond this cosmetic pass.
- `balanced_binary_tree.png` still has tight leaf-node spacing. The arrow trim is
  improved, but the overlap pressure is layout-side.
- Final full Tier 2 remains blocked by the existing `layout_drl` import issue
  already documented in round 5.

## Knowledge

- `graphviz_strict` comparison renders use Graphviz positions but Dagua-rendered
  node shapes. Any render-only ellipse growth can therefore create visual overlap
  unless it is capped for dense or high-aspect fixtures.
- The arrow-types fixture sets Dagua `open` to `arrow_fill="hollow"`, but native
  Graphviz still fills named `open`; matching Graphviz requires renderer-level
  override rather than changing the comparison script.
- `crow` in Graphviz 8 SVG is a filled polygon (`fill="black" stroke="black"`),
  not a stroked ER-style crow foot.

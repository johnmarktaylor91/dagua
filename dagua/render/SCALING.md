# Dagua Rendering Coordinate System

## Default Rule

The renderer follows the 2026-03-23 data-coordinate-everything directive:
rendered geometry belongs in data coordinates by default. Display-point values
are allowed only as explicit opt-in overrides or documented internal residuals.

Dagua is a differentiable layout engine. Node positions, sizes, edge routes,
cluster boxes, and aesthetic losses live on the optimizer's data-coordinate
manifold. A visible border or marker drawn in display points sits outside that
manifold: it can change relative to the optimized geometry when DPI, figure
size, or axis limits change. That makes visual output less predictable and can
hide bugs from layout-side tests.

## Coordinate Spaces

Data coordinates describe content:

- node positions and sizes
- cluster bounding boxes
- edge curves, arrowheads, and route decorations
- node and cluster borders as filled rings or ribbons
- text glyph fills, outlines, and synthetic bold emphasis

Display points describe author-facing size inputs:

- font-size tokens before glyph layout
- style values historically named in points, such as stroke widths
- explicit future override APIs whose name says they are display-point based

Those point values are inputs, not the final rendering space.

## The Display-Scale Pattern

Matplotlib still needs an axes transform to convert a point-sized design token
into data-coordinate geometry. `_compute_display_scale(ax)` returns the scalar:

```python
data_units = points * _compute_display_scale(ax)
```

Use that conversion at the rendering boundary, then build a filled polygon,
ring, or ribbon in data coordinates. The patch or collection should use:

```python
linewidth=0.0
edgecolor="none"
```

Examples include node borders, cluster borders, arrowhead bodies, text outline
ribbons, synthetic bold text ribbons, crossing bridges, and port indicators.

Font rendering is the inverse boundary: Dagua lays glyph paths out in data
coordinates, so a user-facing font size is first converted to data units for
glyph geometry. When a matplotlib text fallback is unavoidable, convert back to
display points only at that hand-off.

## Minimum Visible Strokes

Some data-coordinate ribbons can underflow when the graph extent is tiny or the
axis transform is extreme. In those cases, apply a render-time clamp such as:

```python
visible_width = max(data_width, _MIN_VISIBLE_STROKE_POINTS * display_scale)
```

The clamp is a rasterization guardrail. The optimizer-facing geometry and style
meaning remain data-coordinate; only the emitted visual ribbon receives a floor
so it does not disappear.

## Pixel-Unit Override (Opt-In)

The default render path routes everything through data coordinates (per the
2026-03-23 directive). For users who need literal point-perfect typography
or stroke widths -- typically when preparing figures for academic papers
or print -- six override fields are available:

- `NodeStyle.stroke_width_override_points`
- `NodeStyle.font_size_override_points`
- `EdgeStyle.width_override_points`
- `EdgeStyle.font_size_override_points`
- `ClusterStyle.stroke_width_override_points`
- `ClusterStyle.font_size_override_points`

When set, these bypass data-coord conversion and route directly to
matplotlib's display-point rendering. Override values are **NOT
differentiable** -- the optimizer cannot see them, so they cannot
participate in loss terms. This is the explicit trade-off:
calibrate-once-correct-everywhere (data-coord default) vs literal
point-perfect rendering (override).

Use overrides sparingly and intentionally. The default data-coord path
is correct for differentiable layout; overrides are for the small
class of cases where exact pt-to-px mapping matters more than
scale-consistency.

## Legitimate Display-Point Residuals

Display points are acceptable in two narrow categories:

1. Explicit user overrides, such as a future `NodeStyle.*_override` field whose
   name and documentation promise display-space behavior.
2. Principled internal residuals where a backend primitive cannot be represented
   faithfully as data geometry, and the code carries a nearby comment naming the
   data-coordinate directive and explaining the exception.

Residuals should be rare, named clearly, and easy to audit. A bare matplotlib
`linewidth=style_value` or `markersize=style_value` is not acceptable for normal
renderer output.

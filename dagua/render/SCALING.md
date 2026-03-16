# Dagua Rendering Coordinate System

## Summary

Dagua uses two coordinate spaces. Understanding which space a property belongs
to prevents scaling bugs.

### Data Space — Geometry (positions, sizes)

- Node positions: `pos[i] = (x, y)`
- Node sizes: `sizes[i] = (width, height)`
- Edge bezier curves: control points
- Cluster bounding boxes
- Arrowhead polygon vertices (converted from points at render time)

### Display Space — Appearance (visual properties)

matplotlib handles these correctly as points (1/72 inch):

- `linewidth` on all patches (nodes, edges, clusters)
- `fontsize` on all text
- `linestyle` dash/gap lengths
- `text_outline_width` via path effects

### The One Conversion: Polygon-Based Decorations

Arrowheads and cluster corner radii are the main exceptions: they are drawn as
data-coordinate geometry but should be specified in points. The
`_compute_display_scale()` helper converts them at render time:

```python
display_scale = _compute_display_scale(ax)
arrow_data_length = arrow_points_length * display_scale
```

Cluster label offsets use the same conversion because they position text
relative to a data-space cluster box.

### Why Linewidth Doesn't Need Conversion

matplotlib's `linewidth` parameter is in points, not data units. A 1.4pt border
stays 1.4pt regardless of the graph's `xlim` or `ylim`.

If a border looks thicker on one graph than another, the cause is not linewidth
scaling. The graph's nodes are simply smaller in data units relative to the
figure, so the fixed-point border appears proportionally heavier. That is
correct behavior.

### Adding New Visual Properties

- For `linewidth`, `fontsize`, and `linestyle`: use the value directly
- For polygon geometry and patch radii: convert with `_compute_display_scale()`
- When in doubt: if matplotlib's API docs say "points", do not convert it

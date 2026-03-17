# Visual Property Units

All visual properties in Dagua are specified in typographic points
(1 point = 1/72 inch), the same system used by Graphviz and PDF/PostScript.

| Property | Unit | Example | Notes |
|----------|------|---------|-------|
| `stroke_width` | points | `1.4` | Same thickness at any graph scale |
| `font_size` | points | `14.0` | Same text size at any graph scale |
| `edge.width` | points | `1.2` | Same edge thickness at any scale |
| `arrow_length` | points | `10.0` | Converted to data units at render time |
| `arrow_width` | points | `7.0` | Converted to data units at render time |
| `corner_radius` | points | `6.0` | Converted for cluster rounding geometry |
| `label_offset` | points | `(8.0, 20.0)` | Converted for cluster label placement |

Dagua automatically handles the conversion between points and data coordinates
where needed. You do not need to manage coordinate systems yourself when
specifying styles.

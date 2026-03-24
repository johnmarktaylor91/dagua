# Dagua Cosmetic Feature Inventory

Last updated: 2026-03-22

## Node Shapes (20)

All in `dagua/render/borders/shapes.py`, dispatched via `build_shape_path(spec)`.

| Shape | Source | Notes |
|-------|--------|-------|
| rect | Graphviz | Rectangle |
| roundrect | Graphviz | Rounded rectangle (default) |
| ellipse | Graphviz | Oval |
| circle | Graphviz | Perfect circle |
| diamond | Graphviz | Rotated square |
| triangle | Graphviz | Upward-pointing |
| hexagon | Graphviz | 6-sided |
| pentagon | Graphviz | 5-sided |
| octagon | Graphviz | 8-sided |
| star | Graphviz | 5-pointed star |
| cylinder | Graphviz | 3D cylinder with bezier caps |
| parallelogram | Graphviz | Skewed rectangle |
| trapezoid | Graphviz | Trapezoid |
| double_circle | Graphviz, Mermaid | Two concentric circles |
| cloud | Mermaid, Draw.io | Overlapping circular arcs |
| stadium | Mermaid | Pill/capsule (semicircular ends) |
| tab | Graphviz | Rectangle with top-left ear |
| note | Graphviz | Dog-eared top-right corner |
| document | Mermaid, Draw.io | Wavy bottom edge |
| box3d | Graphviz, yEd | Isometric 3D box |

## Arrowheads (23)

All in `dagua/render/edges/arrowheads.py`, registered in `ARROWHEAD_REGISTRY`.

| Arrow | Type | Source |
|-------|------|--------|
| normal | filled triangle | Graphviz |
| inv | inverted triangle | Graphviz |
| dot | filled circle | Graphviz |
| box | rectangular | Graphviz |
| vee | V-shape chevron | Graphviz |
| tee/bar | horizontal bar | Graphviz |
| crow | crow's foot (3 tines) | Graphviz |
| diamond | diamond with neck | Graphviz |
| curve | curved outline | Graphviz |
| icurve | inverted curve | Graphviz |
| simple | streamlined triangle | Graphviz |
| fancy | stockier filled head | Graphviz |
| wedge | standard triangle | Graphviz |
| bracket | bracket arms | Graphviz |
| none | empty (reserved space) | Graphviz |
| open | outline normal (alias) | Graphviz |
| circle | outline dot (alias) | Graphviz |
| crows_foot_one | single bar (ER "one") | yEd |
| crows_foot_many | 3 diverging tines (ER "many") | yEd |
| crows_foot_one_mandatory | double bar (ER "exactly one") | yEd |
| crows_foot_many_mandatory | tines + bar (ER "many required") | yEd |
| crows_foot_many_optional | tines + circle (ER "many optional") | yEd |
| triangle_tee | triangle + perpendicular bar | Cytoscape |

Compound syntax: Graphviz-style `o` (outline), `l` (left), `r` (right) modifiers.
Up to 4 primitives per compound spec (e.g., "olinormal").

## Edge Routing (4)

In `dagua/edges.py`, dispatched via `_compute_curve()`.

| Mode | Description | Source |
|------|-------------|--------|
| bezier | Cubic bezier curves (default) | All tools |
| straight | Direct line segments | All tools |
| ortho | Right-angle at midpoint | Graphviz, yEd |
| taxi | L-shaped Manhattan path | Cytoscape, Draw.io |

Set via `EdgeStyle.routing = "bezier"/"straight"/"ortho"/"taxi"`.

## Edge Line Styles

| Style | EdgeStyle field |
|-------|----------------|
| solid | `style="solid"` (default) |
| dashed | `style="dashed"` |
| dotted | `style="dotted"` |

## NodeStyle Fields (56+)

### Shape & Structure
shape, padding, corner_radius, border_count, border_position, min_width, min_height

### Fill & Border
fill, stroke, stroke_width, stroke_dash, stroke_dash_pattern, border_opacity,
opacity, stroke_cap, stroke_join, base_color

### Fill Patterns
fill_pattern ("solid"/"striped"/"hatched"/"pie"), fill_pattern_colors,
fill_pattern_values (pie slice proportions), fill_pattern_angle,
fill_pattern_hole (donut inner radius 0-1)

### Gradients
gradient ("none"/"linear"/"radial"), gradient_color, gradient_angle

### Text
font_family, font_size, font_color, font_weight ("regular"/"bold"),
font_style ("normal"/"italic"), text_align, text_valign, text_rotation,
text_wrap ("none"/"wrap"/"ellipsis"), text_max_width, text_transform
("none"/"uppercase"/"lowercase"), label_format, min_font_size, overflow_policy

### Text Decorations
text_outline, text_outline_color, text_outline_width, text_background,
text_background_opacity, text_background_padding, text_background_corner_radius

### External Labels
external_label, external_label_position ("top"/"bottom"/"left"/"right"),
external_label_font_size, external_label_font_color, external_label_offset

### Shadows
shadow, shadow_offset, shadow_color, shadow_blur

### Images
image (path), image_fit ("contain"/"cover"/"stretch"), image_opacity

## EdgeStyle Fields (39+)

### Basic
color, width, style, opacity, routing, curvature, port_style

### Arrows
arrow, tail_arrow, arrow_fill, arrow_color, arrow_length, arrow_width,
arrow_scale (legacy), arrow_node_fraction, arrow_width_ratio

### Labels
label_font_size, label_font_color, label_background, label_background_opacity,
label_background_padding, label_background_corner_radius, label_font_family,
label_font_weight, label_position, label_offset, label_side, label_avoidance

### Head/Tail Labels
head_label, tail_label, head_label_offset, tail_label_offset

### Advanced
taper, taper_width_start, taper_width_end, color_gradient,
color_gradient_end, line_cap, line_join, crossing_style
("none"/"arc"/"gap"/"sharp"), crossing_size

## ClusterStyle Fields (17)

fill, stroke, stroke_width, stroke_dash, corner_radius, padding,
label_position, font_size, font_weight, font_color, opacity, font_family,
label_offset, depth_fill_step, depth_stroke_step, member_node_style,
member_edge_style

## GraphStyle Fields (12)

background_color, margin, title_font_size, title_font_weight,
title_font_color, title_font_family, edge_label_font_size,
edge_label_background, edge_label_background_opacity,
node_label_secondary_scale, max_figsize, min_figsize

## Themes (44)

### Graph Viz Tools (21)
graphviz, graphviz_strict, networkx, mermaid, d3, cytoscape, gephi,
obsidian, yed, drawio, neo4j, visjs, sigma, graphistry, tikz, igraph_r,
graph_tool

### Products (6)
excalidraw, github, linear, n8n, airflow, dagster

### Historical (6)
neuron, blueprint, chalkboard, subway, vintage_textbook, feynman

### Creative (11)
bauhaus, art_deco, neon, terminal, napkin, molecular, circuit,
constellation, genealogy, dark_academia, pastel

### Our Own (4 -- default aliases graphviz)
default, dark, minimal, torchlens

All defined in `dagua/styles.py` and registered in `THEME_REGISTRY`.
Access: `dagua.set_theme('neon')` or `graph._theme = THEME_REGISTRY['neon']`.

## Edge Crossing Detection

Module: `dagua/render/crossings.py`

- `detect_crossings(curves, edge_count)` -> `list[EdgeCrossing]`
- `EdgeCrossing(edge_a, edge_b, x, y, t_a, t_b, angle)`
- Flattens bezier curves to 16 segments, brute force O(E^2) with bbox culling
- Guards: skips self-loops, zero-length edges, crossings near endpoints
- Cached on `graph._cached_crossings`, cleared on `invalidate_layout()`
- Accessible via `EdgeView.crossings`

## View Objects

Module: `dagua/views.py`

- `NodeView`: label, id, type, style, style_override, degree, edges,
  neighbors, successors, predecessors, clusters, position, size
- `EdgeView`: source, target, label, type, weight, style, style_override,
  is_back_edge, crossings
- `ClusterView`: name, label, members, member_count, children, parent,
  depth, style

Access: `graph["node_id"]`, `graph.node_at(idx)`, `graph.edge(idx)`,
`graph.cluster("name")`, `graph.nodes`, `graph.edges_view`, `graph.edges_between(a, b)`

## Style Cascade (5 levels, highest priority first)

1. Per-element override (node_styles[idx] / edge_styles[idx])
2. Deepest cluster's member_node_style / member_edge_style
3. Theme type lookup (_theme.get_node_style(node_type))
4. Graph default (default_node_style / default_edge_style)
5. Global defaults (dagua.configure() overrides)

Resolved via: `graph.get_style_for_node(idx)`, `graph.get_style_for_edge(idx)`

## Compact __repr__

All style classes show only non-default fields:
- `NodeStyle(shape='circle', fill='#FF0000')`
- `EdgeStyle(color='#999', arrow='none')`
- `DaguaGraph(34 nodes, 78 edges, 2 clusters, direction='TB', weighted=True)`

## Visual Reference Pipeline

- `scripts/build_feature_reference.py` -- renders all features as HTML gallery
- Output: `eval_output/feature_reference/index.html`
- 20 shape specimens, 23 arrowhead specimens, 4 routing modes, effects
- Placeholder slots for competitor side-by-side comparisons

## Theme Review Renders

All at `eval_output/theme_review/<theme_name>/tree.png` and `resnet.png`.
Generated via FR layout on sample graphs. Used for aesthetic critic iteration.

## Graphviz Theme Comparison Pipeline

- `scripts/graphviz_theme_comparison.py` -- three-way comparison
  (native Graphviz vs dagua strict vs dagua improved)
- 10 showcase graphs, 900x700 panels, HTML gallery
- `docs/graphviz_theme_departures.md` -- every parameter diff documented

## Installed Fonts

| Font | Status |
|------|--------|
| Trebuchet MS | Installed (msttcorefonts) |
| Arial | Installed (msttcorefonts) |
| Inter | Installed (~/.local/share/fonts/inter/) |
| DejaVu Sans/Serif/Mono | System default |
| Comic Sans MS | Fallback for excalidraw/napkin themes |
| Helvetica Neue | Maps to Noto Sans (fontconfig substitute) |
| Helvetica | Maps to TeX Gyre Heros (fontconfig substitute) |

## Research Documents

- `.project-context/knowledge/research/visual_styling_survey.md` -- cross-tool feature matrix
- `.project-context/knowledge/research/theme_capture_metaplan.md` -- sprint plan
- `.project-context/knowledge/research/tool_default_styling_values.md` -- exact hex codes per tool
- `.project-context/knowledge/research/product_graph_themes.md` -- product aesthetic analysis
- `.project-context/knowledge/research/pie_charts_and_edge_jumps.md` -- implementation details

## Known Issues / Future Work

- Graphviz theme calibration target: min>=8, mean>=9 (current: min=7, mean~7.8)
- Arrowhead placement bug: tips overlap node surface (HIGH priority bug)
- Non-convex shape clipping: pie charts may bleed outside concavities on star/etc
- FillPattern union type: cleaner than fill_pattern_* fields, but breaking change
- Crossing cache location: should move to RenderResult when that abstraction exists
- Edge crossing spatial hashing: upgrade from O(E^2) when needed at scale

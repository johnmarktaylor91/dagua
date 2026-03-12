# Layout vs Render Reference

This is the fast answer to: "If I change this, do I need to relayout?"

## Layout-Affecting

These can change node positions and should be treated as relayout inputs.

- Graph topology
  - nodes
  - edges
  - clusters / cluster membership
- Graph direction
- `LayoutConfig`
  - `steps`
  - `lr`
  - `node_sep`
  - `rank_sep`
  - `w_*` loss weights
  - multilevel settings
  - edge optimization step count when edge geometry is part of the final output pipeline
- Flex / constraints
  - pins
  - alignment groups
  - ordering / structural constraints
- Node label content when node size is label-driven
- Node style fields that affect node size
  - font size
  - padding
  - shape if sizing/boundary behavior changes materially

## Routing-Affecting

These do not necessarily move nodes, but they change edge geometry and label placement.

- `EdgeStyle.routing`
- `EdgeStyle.curvature`
- `EdgeStyle.port_style`
- `EdgeStyle.label_position`
- `EdgeStyle.label_offset`
- `EdgeStyle.label_side`
- edge label text
- cluster geometry when edges route around cluster bounds

## Render-Only

These should not require node relayout.

- fill / stroke colors
- opacity
- stroke widths
- font family
- most font color changes
- background color
- title styling
- hover text in SVG
- output format / DPI / video export settings

## Safe Working Rule

If you are unsure:

- changed topology or spacing semantics: relayout
- changed edge path semantics: reroute labels/edges
- changed only cosmetics: rerender

## Intended User Experience

- `dagua.draw(g)` should do the right thing automatically.
- `dagua.layout(g)` is the explicit seam for power users.
- stale cached layout state should be inspectable, not mysterious.

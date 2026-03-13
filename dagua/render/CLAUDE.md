# Render Subpackage

## Responsibility

Output backends that convert layout results (positions + elements + styles) into visual
output. Three independent renderers, no shared state between them.

## Critical Design Constraint

**Renderers accept structured data, not Graph objects.** `Graph.render()` is a thin
wrapper that extracts positions, elements, and styles, then calls the appropriate renderer.
Renderers never import Graph.

## Modules

### mpl.py — Matplotlib renderer (default)
- Uses PatchCollection for batched node rendering
- Uses LineCollection for batched edge rendering
- Text labels via matplotlib text API
- Requires matplotlib (optional dependency)

### svg.py — Direct SVG string output
- Generates SVG markup directly as a string
- Zero external dependencies
- Jupyter-friendly (displays inline via `_repr_svg_`)

### graphviz.py — Graphviz passthrough (optional)
- Writes .dot file with pre-computed node positions
- Renders via Graphviz neato engine with `-n2` flag (use given positions)
- Requires graphviz Python package (optional dependency)
- Useful for users who want Graphviz's edge routing/label placement with dagua's layout

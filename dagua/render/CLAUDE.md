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
- Useful when matplotlib is not available or not desired

### graphviz.py — Graphviz passthrough (optional)
- Writes .dot file with pre-computed node positions
- Renders via Graphviz neato engine with `-n2` flag (use given positions)
- Requires graphviz Python package (optional dependency)
- Useful for users who want Graphviz's edge routing/label placement with dagua's layout

## Dependency Rules

- **mpl.py**: imports matplotlib only (lazy import, fails gracefully if missing)
- **svg.py**: stdlib only — no external imports
- **graphviz.py**: imports graphviz only (lazy import, fails gracefully if missing)
- **__init__.py**: re-exports, dispatches to correct backend

No module in this package imports from dagua except types/dataclasses from `elements.py`
and `style.py` (pure data, no side effects).

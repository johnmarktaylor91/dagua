# Render Subpackage — Implementation Guide

## Dependency Rules

- **mpl.py**: imports matplotlib only (lazy import, fails gracefully if missing)
- **svg.py**: stdlib only — no external imports
- **graphviz.py**: imports graphviz only (lazy import, fails gracefully if missing)
- **__init__.py**: re-exports, dispatches to correct backend

No module in this package imports from dagua except types/dataclasses from `elements.py`
and `style.py` (pure data, no side effects).

## Conventions

- Public rendering helpers should have docstrings that explain which geometry they
  expect: node positions, curves, label positions, or fully routed artifacts.
- Keep optional-backend behavior explicit and easy to trace.
- Avoid hiding layout-side assumptions inside renderer code; document them at the
  top of the relevant function instead.
- Prefer conservative comments about coordinate systems, units, and fallback logic
  over decorative comments.

## Gotchas

- Multi-line node labels: secondary line font scaling is hardcoded (0.8x).
- Edge arrowheads: `mutation_scale=1` makes heads very small at some zoom levels.
- Cluster label position is hardcoded (top-left) — should respect `ClusterStyle.label_position`.
- mpl.py handles many edge cases: cluster rendering, edge labels, arrowheads, multi-line
  text, direction-dependent transforms. Changes here need visual verification.

## Testing

```bash
pytest tests/test_render/ -x --tb=short
```

Render tests check structural output (SVG element count, figure created),
not pixel-perfect comparison.

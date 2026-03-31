# Render Subpackage — Implementation Guide

## Modules

- **mpl.py** -- Matplotlib (default): PatchCollection, LineCollection, batched text.
  Requires matplotlib (optional dep, lazy import)
- **svg.py** -- Direct SVG string output. Zero deps. Jupyter `_repr_svg_` inline display.
- **graphviz.py** -- Neato `-n2` passthrough: dagua positions + Graphviz rendering.
  Requires graphviz (optional dep, lazy import)

## Dependency Rules

- **mpl.py**: imports matplotlib only (lazy import, fails gracefully if missing)
- **svg.py**: stdlib only — no external imports
- **graphviz.py**: imports graphviz only (lazy import, fails gracefully if missing)
- **__init__.py**: re-exports, dispatches to correct backend

No module in this package imports from dagua except:
- types/dataclasses from `elements.py`
- pure data/constants from `styles.py`
- pure utility functions from `utils.py`

Renderer modules must not import from `dagua.render.mpl` unless they are part of
the `mpl.py` implementation itself.

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

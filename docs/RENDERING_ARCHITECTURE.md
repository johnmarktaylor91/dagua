# Rendering Architecture

## Overview

Dagua's matplotlib renderer treats graph geometry as filled shapes in data coordinates. Node fills, borders, edge ribbons, arrowheads, and all label text scale together under zoom because they are rendered from geometry, not display-space text primitives.

The sole exception is the graph title in [`dagua/render/mpl.py`](/home/jtaylor/projects/dagua/dagua/render/mpl.py). It remains pixel-based via `ax.set_title()` because it is axes-relative UI chrome, participates in `tight_layout()`, and is not part of the graph's geometry.

## Module Map

```text
dagua/render/
    mpl.py              -- Orchestrator: calls modules in zorder sequence
    edges/              -- Edge ribbons, arrowheads, dashes, labels
    borders/            -- Node/cluster annular borders, shapes, insets
    text/               -- TextPath-based label rendering (all text)
```

## Rendering Pipeline

The matplotlib backend renders in a fixed z-order stack:

- `0.0-0.9`: Cluster fills, borders, and labels
- `1.0-1.9`: Edge bodies
- `2.0`: Edge arrowheads
- `2.0-2.08`: Node fills, borders, and shape extras
- `3.0`: Node labels
- `4.0`: Edge labels

`mpl.py` owns the pass ordering and delegates the actual geometry generation to the render submodules.

## Data Coordinate System

Display-sized geometry is converted into data units with a point-to-data scale:

```text
data_units = points * display_scale
```

This requires equal-aspect axes so x and y conversions stay visually consistent. Two pieces intentionally remain display-oriented:

- Text outline width stays in display points for readability.
- The graph title remains axes-relative and pixel-based.

## Style Resolution Cascade

Rendered styles resolve through the same five-level cascade used elsewhere in Dagua:

```text
per-element -> cluster member -> theme -> graph default -> global default
```

The renderer consumes fully resolved styles. It does not reimplement cascade logic locally.

## How Each Module Works

- `edges/`: `DaguaEdge` records feed `DaguaEdgeCollection`, which prepares body, head, and label geometry before emitting patches and collections.
- `borders/`: `ShapeSpec` values describe node and cluster silhouettes. Helpers build inset paths, annular borders, and batched filled collections.
- `text/`: `DaguaText` specs are laid out by `layout_plain_text()` or `layout_rich_text()`, producing `TextBlock` geometry that `render_text()` emits as `PathPatch` artists.

The module-level docstrings in each package contain the implementation details; this document is the orchestration-level map.

## Caching

Text rendering caches reference-scale glyph outlines and font metrics in a DPI-independent form. Edge rendering also selects among rendering tiers so high-edge-count scenes can trade fidelity for throughput without changing the orchestration layer.

## Future

Likely extensions to this architecture include:

- Pixel-unit overrides such as `"2pt"` for users who want fixed-size geometry at any zoom level.
- Additional style fields for text backgrounds, outlines, and decorations.
- A selectable text overlay for interactive backends that want copyable labels on top of geometric text.

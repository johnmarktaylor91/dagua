# Dagua Architecture

## Module Map

### `dagua/graph.py` — DaguaGraph (Central Orchestrator)
Holds nodes, edges, clusters. Manages ID→index mapping. Orchestrates the full pipeline:
layout → route_edges → edge optimization → edge label placement → render. Exposes `from_*`
classmethods (thin wrappers over `io.py`). Implements the 5-level style cascade and
pin/align/flex helpers.

### `dagua/elements.py` — Pure Data
Node, Edge, Cluster dataclasses. No layout or render methods. No imports from other dagua
modules. These are inert containers — Graph does the orchestration.

### `dagua/styles.py` — Style System
NodeStyle, EdgeStyle, ClusterStyle, GraphStyle, Theme dataclasses. Style resolution cascade:
per-element > cluster member style > theme type > graph default > global default. Palette
definitions. Built-in themes: DEFAULT, DARK, MINIMAL, TORCHLENS, GRAPHVIZ_MATCH.

### `dagua/config.py` — LayoutConfig
All tunable layout parameters in one dataclass. Includes flex field for soft layout targets.
Default values tuned to Graphviz-competitive output.

### `dagua/defaults.py` — Global Defaults
Thread-safe global configuration: `configure()` (flat kwargs), `defaults()` (context manager),
`set_theme()`, `set_device()`. Thread-local storage — no global state hell.

### `dagua/flex.py` — Soft Layout Targets
Flex (soft/firm/locked values), LayoutFlex (per-parameter flex), AlignGroup. Preferences
expressed as differentiable loss terms. Hard pins (weight=inf) enforced via projection.

### `dagua/layout/` — Layout Engine (Heart of Dagua)
Differentiable graph layout via PyTorch optimization. **Headless**: operates on tensors,
not Graph objects. See `dagua/layout/CLAUDE.md` for module-level detail.

- `engine.py` — Optimization loop. Wires constraints, runs Adam, optional projection.
- `constraints.py` — Composable loss callables: DAG, Repel, Attract, Overlap, Cluster, Pin, Align, FlexSpacing, Crossing.
- `projection.py` — Hard overlap + hard pin projection (projected gradient descent).
- `schedule.py` — Annealing schedules for constraint weights over optimization steps.
- `init_placement.py` — Topological sort (y-axis) + barycenter (x-axis) initialization.
- `layers.py` — Layer assignment algorithms for hierarchical layout.
- `multilevel.py` — Multilevel/coarsening layout for large graphs.
- `cycle.py` — Cycle detection and temporary edge reversal for DAG constraints.
- `edge_optimization.py` — Post-layout edge-aware position refinement.

### `dagua/render/` — Output Backends
Three independent renderers, no shared state. See `dagua/render/CLAUDE.md`.

- `mpl.py` — Matplotlib (default): PatchCollection, LineCollection, batched text.
- `svg.py` — Direct SVG string output, zero deps, Jupyter-friendly.
- `graphviz.py` — Neato `-n2` passthrough (use dagua positions with Graphviz rendering).

### `dagua/routing.py` — Edge Routing
Bezier control point computation. Heuristic approach (not yet differentiable). Handles
straight, curved, and back-edge routing.

### `dagua/edges.py` — Edge Pipeline
Edge label placement and edge routing orchestration. Called between layout and render.

### `dagua/metrics.py` — Quality Metrics
Layout quality metrics: edge crossings, stress, angular resolution, edge length variance,
cluster containment, overlap count, etc. Used by eval system and tests.

### `dagua/io.py` — IO & Interop
JSON/YAML serialization. Import/export for NetworkX, igraph, PyG, scipy, DOT (pydot).
LLM-based graph construction from images (via anthropic/openai). Style load/save.

### `dagua/cli.py` — CLI
Entry point for the `dagua` command. Orchestration-heavy, algorithm-light. Commands for
benchmarks, reports, visual audit, placement tuning, gallery, glossary, animation.
Strictly typed (mypy strict mode).

### `dagua/eval/` — Evaluation System
Benchmark runner, multi-engine comparison, report generation, hyperparameter sweep,
visual audit, aesthetic evaluation. 6 competitor engine adapters (graphviz×4, elk, dagre,
networkx×2, igraph×3). 30+ reference graphs in `dagua/graphs/`.

### `dagua/animation.py` — Cinematic Exports
`animate()` (optimization process), `tour()` (camera keyframe animation),
`poster()` (high-res multi-panel output).

### `dagua/utils.py` — Utilities
Text measurement (for node sizing), graph topology helpers, misc shared functions.

## Data Flow

```
User builds DaguaGraph (from code, YAML, NetworkX, image, etc.)
    ↓
graph.layout(config) extracts tensors → layout/engine.py
    ↓
engine: init_placement → optimization loop (constraints + projection + annealing)
    ↓
Returns position tensor → stored on graph
    ↓
route_edges(graph, positions) → bezier control points
    ↓
edge_optimization (optional) → refined positions
    ↓
place_edge_labels(graph, positions, curves)
    ↓
render(graph, positions, config) → mpl Figure / SVG string / saved file
```

Key types flowing between modules:
- `torch.Tensor` (positions: N×2, edge_index: 2×E, node_sizes: N×2)
- `DaguaGraph` (nodes, edges, clusters, styles, positions, curves)
- `LayoutConfig` (all layout parameters)
- `NodeStyle`, `EdgeStyle`, `ClusterStyle` (resolved style dicts)

## Key Abstractions

1. **DaguaGraph** — Central container. Holds graph topology + style + layout results.
   Invariant: node IDs map 1:1 to integer indices via `_id_to_idx`.

2. **Constraint callables** — `(pos: Tensor, graph_data: dict) -> Tensor (scalar)`.
   Composable, stateless, differentiable. Sum of constraints = layout loss.

3. **Flex** — Soft layout target with weight. `Flex.soft(40)` = prefer 40 (low weight),
   `Flex.firm(40)` = strongly prefer 40, `Flex.locked(0)` = hard constraint (projection).

4. **Theme** — Named style bundle (node defaults, edge defaults, palette, graph style).
   Applied at cascade level 3 (below per-element and cluster, above graph/global defaults).

5. **LayoutConfig** — Single dataclass for all tunable parameters. Passed through the
   entire pipeline. Default values produce Graphviz-competitive output.

## Dependency Graph

```
elements.py, styles.py, flex.py, config.py, defaults.py  (leaf modules, no dagua imports)
    ↓
utils.py  (imports elements for text measurement)
    ↓
graph.py  (imports styles, elements, utils, config, flex, defaults)
    ↓
layout/  (imports NOTHING from dagua — headless, tensor-only)
    ↓
routing.py  (post-layout, imports elements for edge types)
    ↓
edges.py  (imports routing, elements)
    ↓
render/  (imports elements, styles for type info — never imports graph)
    ↓
io.py  (imports graph, elements, styles for serialization)
    ↓
__init__.py  (re-exports everything)
    ↓
cli.py  (imports from __init__, eval/)
```

`eval/` is a parallel tree: imports from dagua public API + competitors. Never imported
by core dagua modules.

## Known Complexity

- **`dagua/layout/engine.py`** — The optimization loop wires many concerns: constraints,
  projection, annealing, flex, pin/align, multilevel. Most layout bugs originate here.

- **`dagua/layout/constraints.py`** — Crossing loss is O(E²). Interval amortization needed
  for large graphs. Adding new constraints requires understanding the graph_data dict protocol.

- **`dagua/graph.py`** — Large file (~800+ lines). Style cascade resolution, the draw()
  pipeline, and tensor extraction logic all live here. Refactoring into smaller pieces
  would break the "Graph is the orchestrator" design.

- **`dagua/eval/`** — The benchmark/report system is powerful but has many artifacts,
  caching layers, and CLI entry points. `visual_audit.py` is the most complex eval module.

- **`dagua/render/mpl.py`** — Matplotlib renderer handles many edge cases: cluster rendering,
  edge labels, arrowheads, multi-line text, direction-dependent transforms.

- **Edge routing** — Back-edge routing creates wide arcs that can overlap with nodes.
  The current approach is heuristic; differentiable routing is a future goal.

- **Multilevel layout** (`layout/multilevel.py`) — Coarsening + refinement for large graphs.
  Recently hardened with checkpoint validation, but still the most fragile layout path.

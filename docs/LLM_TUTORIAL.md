# Dagua for LLMs and Agents

This document is for agents using Dagua, not for agents developing Dagua itself.

Use it when you need to:
- load or build a graph quickly
- lay it out and render it
- fix a specific visual issue
- export stills, tours, or optimization animations
- use the benchmark, glossary, gallery, or notebooks without reading the whole repo

If you are modifying Dagua internals, read `CLAUDE.md` / `AGENTS.md` instead.

## Fastest Path

Most agent tasks should start here:

```python
import dagua
from dagua import DaguaGraph, LayoutConfig

g = DaguaGraph.from_edge_list([
    ("input", "prep"),
    ("prep", "model"),
    ("model", "output"),
])

fig, ax = dagua.draw(
    g,
    LayoutConfig(steps=80, edge_opt_steps=8, seed=42),
)
```

If the user wants files:

```python
dagua.draw(g, LayoutConfig(steps=80, edge_opt_steps=8, seed=42), output="graph.png")
```

## What Dagua Is Good At

- hierarchical DAG-style layout
- GPU-accelerated optimization through PyTorch
- node, edge, cluster, and label styling
- cinematic exports:
  - posters
  - tours
  - faithful optimization animations
- very large DAGs, including dedicated billion-scale tooling

## Core Mental Model

Dagua is not Graphviz-style rule layout. It is:

1. build a graph
2. compute node sizes
3. optimize node positions with losses
4. route edges after node layout
5. optionally refine edge control points
6. render or export

The most useful split is:
- `dagua.draw(...)`: one-shot convenience path
- `dagua.layout(...)` + `dagua.render(...)`: explicit control path

## Main Entry Points

Public API most agents care about:

- graph creation:
  - `DaguaGraph`
  - `DaguaGraph.from_edge_list(...)`
  - `DaguaGraph.from_edge_index(...)`
  - `dagua.load(...)`
- layout:
  - `dagua.layout(...)`
  - `LayoutConfig(...)`
- render:
  - `dagua.draw(...)`
  - `dagua.render(...)`
- routing helpers:
  - `dagua.route_edges(...)`
  - `dagua.place_edge_labels(...)`
- styling:
  - `NodeStyle`
  - `EdgeStyle`
  - `ClusterStyle`
  - `dagua.set_theme(...)`
  - `dagua.configure(...)`
- constraints/flex:
  - `Flex`
  - `LayoutFlex`
  - `AlignGroup`
- cinematic outputs:
  - `dagua.poster(...)`
  - `dagua.tour(...)`
  - `dagua.animate(...)`
- IO:
  - `dagua.save(...)`
  - `dagua.load(...)`
  - `dagua.from_image(...)`
  - `dagua.graph_dict_from_image(...)`
  - `dagua.graph_code_from_image(...)`
  - `dagua.theme_from_image(...)`
  - `dagua.theme_dict_from_image(...)`
  - `dagua.theme_code_from_image(...)`
  - `dagua.configure_image_ai(...)`

## Common Tasks

### Build a graph manually

```python
from dagua import DaguaGraph

g = DaguaGraph(direction="TB")
g.add_node("input", label="Input")
g.add_node("branch_a", label="Branch A")
g.add_node("branch_b", label="Branch B")
g.add_node("merge", label="Merge")
g.add_edge("input", "branch_a")
g.add_edge("input", "branch_b")
g.add_edge("branch_a", "merge")
g.add_edge("branch_b", "merge")
```

### Add styles

```python
from dagua import NodeStyle, EdgeStyle, ClusterStyle

g.node_styles[g._id_to_index["branch_a"]] = NodeStyle(shape="ellipse")
g.edge_styles[0] = EdgeStyle(routing="ortho")
g.add_cluster("Main Stage", ["branch_a", "branch_b"], label="Main Stage", style=ClusterStyle())
```

### Save and reload

```python
dagua.save(g, "graph.yaml")
g2 = dagua.load("graph.yaml")
```

### Turn an image into a graph, dict, code, or theme

```python
dagua.configure_image_ai(provider="openai", api_key_env="OPENAI_API_KEY")

graph = dagua.from_image("diagram.png")
graph_dict = dagua.graph_dict_from_image("diagram.png")
graph_code = dagua.graph_code_from_image("diagram.png")

theme = dagua.theme_from_image("reference.png")
theme_dict = dagua.theme_dict_from_image("reference.png")
theme_code = dagua.theme_code_from_image("reference.png")
```

Use the `*_code_from_image(...)` helpers when the user wants editable source, because
those helpers emit the cleaner Dagua builder style rather than model-specific freeform code.

### Render explicitly

```python
config = LayoutConfig(steps=90, edge_opt_steps=10, seed=42)
pos = dagua.layout(g, config)
fig, ax = dagua.render(g, pos, config)
```

### Export a poster

```python
pos = dagua.layout(g, LayoutConfig(steps=80, edge_opt_steps=8, seed=42))
dagua.poster(g, positions=pos, output="poster.png")
```

### Export a tour

```python
pos = dagua.layout(g, LayoutConfig(steps=80, edge_opt_steps=8, seed=42))
dagua.tour(g, positions=pos, output="tour.gif")
```

### Export a faithful optimization animation

```python
dagua.animate(
    g,
    LayoutConfig(steps=60, edge_opt_steps=8, seed=42),
    output="optimization.gif",
)
```

## If the User Says “Fix This Visual Problem”

Use this mapping first.

### “It feels cramped”

Increase:
- `node_sep`
- `rank_sep`

### “This node needs to stay here”

Use `LayoutFlex` pins:

```python
g.flex = LayoutFlex(
    pins={"input": (Flex.locked(0.0), Flex.locked(0.0))}
)
```

### “These peers should line up”

Use `AlignGroup`:

```python
g.flex = LayoutFlex(
    align_y=[AlignGroup(["a", "b", "c"], weight=6.0)]
)
```

### “The edges are ugly”

Try:
- `EdgeStyle(routing="bezier")`
- `EdgeStyle(routing="straight", curvature=0.0)`
- `EdgeStyle(routing="ortho", curvature=0.0)`

### “This subsystem should read as one unit”

Add a cluster with a label.

### “The labels are too wide / noisy”

Adjust:
- node labels
- `NodeStyle(min_width=...)`
- graph structure / clustering
- overall spacing

## Where to Look Next

If you need exact reference:
- glossary:
  - `docs/glossary/dagua_glossary.pdf`
  - `docs/glossary/dagua_glossary.tex`

If you need a hands-on walkthrough:
- `docs/tutorial_walkthrough.ipynb`

If you need a developer QA surface:
- `tests/ui_feature_playground.ipynb`

If you need polished examples:
- `docs/gallery/README.md`

If you need benchmark/report tooling:
- CLI:
  - `dagua benchmark-status`
  - `dagua benchmark-watch`
  - `dagua benchmark-report`
  - `dagua benchmark-deltas`

## CLI Shortcuts

```bash
dagua poster graph.yaml poster.png --scene zoom_pan --device cuda
dagua tour graph.yaml trailer.gif --scene auto --device cuda
dagua benchmark-status --output-dir eval_output --suite standard
dagua benchmark-watch --output-dir eval_output --suite standard --follow
```

## Large Graph Guidance

For very large graphs:

- prefer `dagua.poster(...)` and `dagua.tour(...)`
- let the large-graph LOD renderer handle overview/detail transitions
- do not try to inspect every node at once
- use cinematic scenes for communication, not raw full-detail frames

If the user wants “show off the billion”:
- generate a final layout
- use `tour(...)` with automatic scenes like `zoom_pan`, `powers_of_ten`, or `panorama`
- use `poster(...)` for hero stills

## Agent Efficiency Notes

To stay context-efficient:

- start from `dagua.draw(...)` unless the user clearly needs lower-level control
- only drop to `layout + route_edges + render` when debugging or customizing routing
- use the glossary for details instead of opening many source files
- use the tutorial notebook for examples instead of re-deriving workflows
- use the gallery as a pattern library when the user wants “something impressive”

## What This File Is Not

This file is not:
- the maintainer/developer instructions
- the full API reference
- the benchmark methodology document

It is the shortest high-signal path for agents who want to use Dagua well.

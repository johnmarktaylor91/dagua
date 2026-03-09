# CLAUDE.md — Project Instructions for Claude Code

## Project Overview

Dagua is a GPU-accelerated, differentiable graph layout engine built on PyTorch. It replaces
Graphviz's layout algorithms with continuous optimization: node positions are learnable
parameters, layout aesthetics are loss functions, and the solver is `loss.backward();
optimizer.step()`.

Named after the Dagua River in Colombia. DAG + agua. Directed acyclic graphs + water.

## Commit Convention

This project uses **conventional commits**:

```
<type>(<scope>): <description>
```

Types: `fix:` (patch), `feat:` (minor), `feat!:` (major), `chore:`, `docs:`, `ci:`,
`refactor:`, `test:`, `perf:`.

## Build & Packaging

- Build system: setuptools via `pyproject.toml`
- Install (dev): `uv pip install -e ".[dev]"`
- Install (test): `uv pip install -e ".[test]"`

## Testing

- Run all tests: `pytest tests/`
- Smoke tests: `pytest tests/ -m smoke`
- GPU tests: `pytest tests/ -m gpu`
- Linting: `ruff format` + `ruff check --fix`

## Project Structure

```
dagua/
├── __init__.py          # public API re-exports (Graph, Node, Edge, Cluster, styles, defaults)
├── graph.py             # Graph class — central orchestrator
│                        #   holds nodes/edges/clusters, ID→index mapping
│                        #   layout()/render() orchestration methods
│                        #   from_* classmethods (thin wrappers over io.py)
├── elements.py          # Node, Edge, Cluster dataclasses (pure structural data)
├── style.py             # NodeStyle, EdgeStyle, ClusterStyle, themes, palettes
├── defaults.py          # module-level defaults (device, theme) — minimal global state
├── layout/              # [see layout/CLAUDE.md]
│   ├── __init__.py      # re-exports: layout(), individual constraints
│   ├── engine.py        # optimization loop (~200 lines of PyTorch)
│   ├── constraints.py   # DAG, Repel, Attract, Overlap, Cluster, Align
│   ├── projection.py    # hard overlap resolution (projected gradient descent)
│   └── schedule.py      # annealing schedules for constraint weights
├── render/              # [see render/CLAUDE.md]
│   ├── __init__.py      # re-exports: render(), to_svg()
│   ├── mpl.py           # matplotlib: PatchCollection, LineCollection, batched
│   ├── svg.py           # direct SVG string output (zero deps)
│   └── graphviz.py      # optional neato -n2 passthrough
├── routing.py           # bezier edge routing (heuristic now, differentiable later)
├── io.py                # from_edges, from_edge_index, from_networkx, from_dict, to_dot
└── utils.py             # text measurement, graph topology helpers
```

```
tests/                   # [see tests/CLAUDE.md]
├── conftest.py          # shared fixtures: sample graphs, common assertions
├── test_graph.py
├── test_elements.py
├── test_style.py
├── test_layout/
│   ├── test_engine.py
│   ├── test_constraints.py
│   └── test_projection.py
├── test_render/
│   ├── test_mpl.py
│   └── test_svg.py
├── test_routing.py
└── test_io.py
```

```
benchmarks/              # [see benchmarks/CLAUDE.md]
├── bench_layout.py      # scaling: 100, 1K, 10K, 100K nodes
├── bench_render.py      # rendering performance
└── graphs/              # reference graphs for reproducible benchmarks
```

```
examples/                # [see examples/CLAUDE.md]
├── quickstart.py        # minimal 5-line example
├── neural_network.py    # DNN-style graph with module clusters
├── custom_constraints.py # writing your own constraint
└── large_graph.py       # 10K+ node demo
```

## Architecture

### Dependency Flow

Clean one-way dependency flow — no circular imports:

```
Elements/Style (pure data, no deps)
    ↓
Graph (holds elements, ID→index mapping, orchestration)
    ↓
Layout Engine (operates on TENSORS, not Graph — headless, independently testable)
    ├── Constraints (composable loss callables)
    ├── Projection (hard overlap resolution)
    └── Schedule (weight annealing)
    ↓
Routing (bezier control points, post-layout)
    ↓
Render (matplotlib/SVG/graphviz — takes positions + elements + styles)
```

### Critical Design Points

1. **Layout engine is headless**: takes `edge_index`, `node_sizes`, `groups` as tensors,
   not a Graph object. `Graph.layout()` is a thin wrapper that extracts tensors, calls
   engine, stores results. Makes the engine independently testable and reusable.

2. **Renderers accept structured data**: `Graph.render()` wraps the renderer call.
   Renderers never import Graph.

3. **Constraints are composable callables**: each is `(pos, graph_data) -> scalar loss`.
   Users write custom constraints in ~3 lines.

4. **Elements are inert data, Graph is the orchestrator**: `elements.py` holds pure
   dataclasses (Node, Edge, Cluster) with no layout/render methods. `graph.py` holds
   Graph which orchestrates everything. They stay as separate root-level files — no
   `data/` folder needed since neither will exceed ~400 lines.

5. **IO lives in io.py, exposed as Graph classmethods**: `io.py` holds standalone
   functions (from_edges, from_networkx, to_dot). Graph.from_* classmethods are thin
   wrappers. Keeps graph.py focused, keeps io.py independently testable.

6. **Settings are graph-level by default**: device, theme, layout params passed to Graph.
   `defaults.py` provides minimal module-level convenience (`set_default_device`,
   `set_default_theme`) — plain module variables with setters, no config objects.

### Layout Pipeline

1. Initialize random node positions as learnable `torch.nn.Parameter`
2. Optimization loop: compute composite loss, backprop, optimizer step
3. Optional: hard overlap projection after each step (projected gradient descent)
4. Return detached position tensor

### Scaling Strategy

- <5K nodes: exact O(N^2) repulsion on CPU
- 5K-50K: negative sampling (k=128), CPU or GPU
- 50K-500K: negative sampling + GPU
- 500K+: grid approximation or Barnes-Hut on GPU

## Design Principles

1. **PyTorch is the only required dependency** (matplotlib optional for rendering)
2. **Constraints are composable loss functions** — users can write custom ones in 3 lines
3. **Layout and rendering are separate** — `layout()` returns coordinates
4. **Domain knowledge enters as loss terms**, not algorithmic modifications
5. **GPU acceleration is automatic** — same code on CPU or CUDA via `device=`
6. **Steal Graphviz's aesthetic wisdom** (spacings, weight ratios) but not its architecture

## Key API Surface

```python
import dagua

# Layout only
pos = dagua.layout(edge_index, num_nodes, node_sizes=sizes,
                   constraints=[dagua.DAG(), dagua.Repel()], device='cuda')

# With rendering
dagua.render(pos, edge_index, node_labels=labels, output='graph.png')

# SVG for notebooks
svg = dagua.to_svg(pos, edge_index, node_labels=labels)
```

## Relationship to TorchLens

Dagua is standalone. TorchLens is a downstream consumer that provides domain-specific
neural network constraints (layer ordering, residual arc width, module hierarchy clusters).
The dependency is one-way: TorchLens depends on Dagua, never the reverse.

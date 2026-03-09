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
├── dagua/
│   ├── __init__.py          # public API: layout(), render()
│   ├── layout.py            # core optimization loop
│   ├── constraints.py       # DAG, Repel, Attract, Cluster, Align, NoOverlap
│   ├── edges.py             # bezier routing (heuristic + optimized)
│   ├── render/
│   │   ├── __init__.py
│   │   ├── matplotlib.py    # PatchCollection/LineCollection renderer
│   │   ├── svg.py           # direct SVG string output
│   │   └── graphviz.py      # optional neato -n2 passthrough
│   ├── projection.py        # hard overlap resolution
│   └── utils.py             # graph utilities, text measurement
├── tests/
├── benchmarks/
├── examples/
├── pyproject.toml
├── LICENSE                  # MIT
└── README.md
```

## Architecture

### Core Insight

Every graph layout algorithm is constrained optimization. Dagua replaces Sugiyama's five
discrete phases with differentiable loss functions and gradient descent.

### Constraint Vocabulary

**Positional** — DAG ordering, rank alignment, pinning
**Proximity** — edge attraction, repulsion, clustering, overlap avoidance
**Symmetry** — parallel branch spreading, sequential chain tightness, uniform edge length

Each constraint is a callable: `(positions, graph_data) -> scalar_loss`. Users compose them
declaratively.

### Layout Pipeline

1. Initialize random node positions as learnable `torch.nn.Parameter`
2. Optimization loop: compute composite loss, backprop, optimizer step
3. Optional: hard overlap projection after each step (projected gradient descent)
4. Return detached position tensor

### Scaling Strategy

- <5K nodes: exact O(N²) repulsion on CPU
- 5K-50K: negative sampling (k=128), CPU or GPU
- 50K-500K: negative sampling + GPU
- 500K+: grid approximation or Barnes-Hut on GPU

### Rendering (Separate from Layout)

- **Matplotlib** (default): PatchCollection + LineCollection for batched rendering
- **SVG**: direct string output, zero deps, Jupyter-friendly
- **Graphviz passthrough** (optional): write .dot with fixed positions, render with neato -n2

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

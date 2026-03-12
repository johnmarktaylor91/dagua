# Layout Subpackage

## Responsibility

Differentiable graph layout via PyTorch optimization. This is the heart of dagua.

## Critical Design Constraint

**The layout engine is headless.** It operates on tensors (`edge_index`, `node_sizes`,
`groups`), NOT on Graph objects. `Graph.layout()` is a thin wrapper that extracts tensors,
calls into this package, and stores results back. This keeps the engine independently
testable and reusable without the Graph abstraction.

## Modules

### engine.py — Core optimization loop
- Takes tensor inputs, initializes positions as `torch.nn.Parameter`
- Runs optimization: composite loss from constraints, `loss.backward()`, `optimizer.step()`
- Optional projected gradient descent (calls projection.py after each step)
- Returns detached position tensor
- Target size: ~200 lines

### constraints.py — Composable loss callables
- Each constraint is a callable: `(pos, graph_data) -> scalar loss`
- Built-in: DAG, Repel, Attract, Overlap, Cluster, Align
- Users write custom constraints by following the same `(pos, graph_data) -> loss` protocol
- Constraints are stateless — all graph structure passed via `graph_data` dict

### projection.py — Hard overlap resolution
- Projected gradient descent: after each optimizer step, project positions to satisfy
  hard constraints (e.g., no node overlap)
- Operates on position tensor + node sizes
- Returns adjusted position tensor

### schedule.py — Weight annealing
- Controls how constraint weights evolve during optimization
- Example: start with strong DAG ordering, gradually increase repulsion
- Enables curriculum-style layout optimization

## Dependency Rules

- **constraints.py**: pure torch, no imports from dagua
- **projection.py**: pure torch, no imports from dagua
- **schedule.py**: pure torch, no imports from dagua
- **engine.py**: imports constraints, projection, schedule — nothing else from dagua
- **__init__.py**: re-exports only

No module in this package imports `graph.py`, `elements.py`, or `style.py`.

## Maintainability Rules

- Keep function signatures explicit and typed. This package is under stricter mypy
  expectations than the repo baseline.
- Headless tensor utilities should document input shapes in docstrings.
- Prefer a small number of strong section comments over line-by-line narration.
- When touching multilevel / coarsening code, add or update smoke coverage for the
  exact failure mode you are fixing.

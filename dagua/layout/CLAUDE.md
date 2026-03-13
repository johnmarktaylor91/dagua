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

### init_placement.py — Initialization
- Topological sort for y-axis, barycenter heuristic for x-axis
- Deterministic from topology (seed doesn't add randomness here)

### layers.py — Layer assignment
- Algorithms for assigning nodes to discrete layers in hierarchical layout

### multilevel.py — Coarsening + refinement
- Multilevel layout for large graphs. Most fragile path in the layout code.

### cycle.py — Cycle handling
- Detection and temporary edge reversal so DAG constraints work on cyclic graphs

### edge_optimization.py — Post-layout refinement
- Edge-aware position adjustment after main optimization

# Layout Subpackage -- Implementation Guide

## Architecture Overview

The layout system has two layers:

1. **Core engine** (`engine.py`, `constraints.py`, etc.) -- the native multilevel/direct
   optimization engine. Used when `LayoutConfig.algorithm` is not set.
2. **Composable ops** (`ops/`) -- 268 registered primitives composed into 23 algorithm
   pipelines. Used when `LayoutConfig(algorithm="fr")` dispatches via `get_pipeline_function()`.

Monolithic reimplementations are archived at `_archive/classic/` (reference-only, not imported).

## Core Engine Modules

- **engine.py** -- Optimization loop + algorithm dispatch. Wires constraints, runs Adam,
  optional projection. When `algorithm` is set, dispatches to `ops/pipelines/`.
- **constraints.py** -- Composable callables: `(pos, graph_data) -> scalar loss`.
  Stateless; all structure via `graph_data` dict.
- **projection.py** -- Hard overlap + hard pin projection after each optimizer step
- **schedule.py** -- Weight annealing for curriculum-style layout optimization
- **init_placement.py** -- Topo sort (y) + barycenter (x); deterministic from topology
- **layers.py** -- Layer assignment algorithms for hierarchical layout
- **multilevel.py** -- Coarsening + refinement for large graphs (most fragile path)
- **cycle.py** -- Detection + temporary edge reversal for DAG constraints on cyclic graphs
- **edge_optimization.py** -- Post-layout edge-aware position refinement

## Composable Ops System (`ops/`)

268 ops across 34 modules. Each op is a pure function decorated with `@register_op`.
Ops share state via `SolveState`:

```python
@dataclass
class SolveState:
    pos: Tensor          # [N, 2] positions
    edge_index: Tensor   # [2, E] edge pairs
    N: int               # node count
    E: int               # edge count
    adj: Tensor          # adjacency (sparse or dense)
    graph_data: dict     # arbitrary graph metadata
    rng: Any             # algorithm-matched RNG (torch/numpy/Python random)
    extras: dict         # algorithm-specific state ("tsne_gains", etc.)
    optimizer: Any       # optional optimizer reference
```

Key op modules:
- `force.py` (22 ops) -- repulsion, attraction, displacement, cooling
- `embed.py` (14 ops) -- spectral, MDS, pivot MDS embedding
- `init.py` (15 ops) -- position initialization strategies
- `anneal.py` (12 ops) -- temperature/weight annealing schedules
- `loss_engine.py` (16 ops) -- engine-level loss computation
- `loss_classic.py` (14 ops) -- classical loss functions
- `stress.py`, `optimize.py`, `converge.py`, `postprocess.py`, etc.
- Per-algorithm: `drl.py`, `fmmm.py`, `gem.py`, `lgl.py`, `neulay.py`, `sgd2_multi.py`, etc.
- `graph_utils.py` -- self-contained graph utilities (BFS, Dijkstra, APSP, adjacency)

## Algorithm Pipelines (`ops/pipelines/`)

23 pipeline files, 24 registered algorithm names. Each pipeline is pure composition of
registered ops -- no inline functions, no archive imports.

`PIPELINE_REGISTRY` maps names to `(module, function)` tuples. `get_pipeline_function(name)`
resolves and imports dynamically.

## Dependency Rules

- **constraints.py**: pure torch, no imports from dagua
- **projection.py**: pure torch, no imports from dagua
- **schedule.py**: pure torch, no imports from dagua
- **engine.py**: imports constraints, projection, schedule + dispatches to ops pipelines
- **ops/**: pure torch + stdlib only. No imports from dagua core.
- **ops/pipelines/**: imports only from `ops/` sibling modules
- **_archive/**: frozen reference implementations. Never imported by production code.
- **__init__.py**: re-exports only

No module in this package imports `graph.py`, `elements.py`, or `styles.py`.

## Conventions

- Keep function signatures explicit and typed.
- Headless tensor utilities document input shapes (e.g., `pos: Tensor  # (N, 2)`).
- Prefer a few strong section comments over line-by-line narration.
- All ops have frozen dataclass configs for tunable parameters.
- RNG backends match the original algorithm's source (torch, numpy, or Python random).
- Pipeline fidelity is validated against archive reimplementations (bit-identical with matching seed).

## Gotchas

- Crossing loss is O(E^2). Don't add naive iterations over edge pairs.
- `init_placement.py` is fully deterministic from topology. Seed param doesn't add randomness.
- Back-edge routing (downstream in `routing.py`) creates wide arcs.
- Multilevel checkpoint validation was recently hardened. Run bench_large smoke tests
  after changing coarsening logic.
- Ops use 5+ different RNG families across algorithms. Changing RNG backend breaks fidelity.
- SolveState extras dict is the escape hatch for algorithm-specific state. Key convention:
  `"algo_field"` (e.g., `"tsne_gains"`).

## Testing

```bash
# Ops unit tests (per-module)
pytest tests/test_ops_force.py -x --tb=short

# Pipeline fidelity tests (per-algorithm)
pytest tests/test_pipeline_fr.py -x --tb=short

# All ops tests
pytest tests/test_ops_*.py -x --tb=short

# All pipeline fidelity tests
pytest tests/test_pipeline_*.py -x --tb=short

# Core layout tests
pytest tests/test_layout/ -x --tb=short

# Smoke tests (includes layout regression pins)
pytest tests/ -m smoke -x --tb=short
```

Layout tests check convergence properties, not exact coordinates -- optimization is stochastic.
Pipeline fidelity tests check bit-identical output against archive reimplementations.

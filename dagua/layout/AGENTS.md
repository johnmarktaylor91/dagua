# Layout Subpackage — Implementation Guide

## Dependency Rules

- **constraints.py**: pure torch, no imports from dagua
- **projection.py**: pure torch, no imports from dagua
- **schedule.py**: pure torch, no imports from dagua
- **engine.py**: imports constraints, projection, schedule — nothing else from dagua
- **__init__.py**: re-exports only

No module in this package imports `graph.py`, `elements.py`, or `style.py`.

## Conventions

- Keep function signatures explicit and typed. This package is under stricter mypy
  expectations than the repo baseline.
- Headless tensor utilities should document input shapes in docstrings (e.g., `pos: Tensor  # (N, 2)`).
- Prefer a small number of strong section comments over line-by-line narration.
- When touching multilevel / coarsening code, add or update smoke coverage for the
  exact failure mode you are fixing.

## Gotchas

- Crossing loss is O(E²). Performance-sensitive — don't add naive iterations over edge pairs.
- `init_placement.py` is fully deterministic from topology. Seed param doesn't add randomness here.
- Back-edge routing (handled downstream in `routing.py`) creates wide arcs — be aware when
  adjusting position refinement that edge geometry depends on final positions.
- Multilevel checkpoint validation was recently hardened. If you change coarsening logic,
  run the bench_large smoke tests.

## Testing

```bash
# Layout unit tests
pytest tests/test_layout/ -x --tb=short

# Smoke tests (includes layout regression pins)
pytest tests/ -m smoke -x --tb=short

# Scaling tests (slow)
pytest tests/test_scaling.py -x --tb=short
```

Layout tests check convergence properties (e.g., DAG constraint reduces y-violations),
not exact coordinates — optimization is stochastic.

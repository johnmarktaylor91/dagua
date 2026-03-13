# Tests

## Structure

Test files mirror the source structure:

```
tests/
├── conftest.py              # shared fixtures: sample graphs, common assertions
├── test_graph.py            # Graph construction, ID mapping, orchestration
├── test_elements.py         # Node, Edge, Cluster dataclasses
├── test_style.py            # styles, themes, palettes
├── test_layout/
│   ├── test_engine.py       # optimization loop convergence, determinism
│   ├── test_constraints.py  # each constraint produces correct gradients
│   └── test_projection.py   # overlap resolution correctness
├── test_render/
│   ├── test_mpl.py          # matplotlib output (figure creation, patches)
│   └── test_svg.py          # SVG string output (valid markup, element count)
├── test_routing.py          # bezier control point computation
└── test_io.py               # from_edges, from_networkx, to_dot round-trips
```

## Running Tests

```bash
pytest tests/                    # all tests
pytest tests/ -m smoke           # quick sanity checks (~seconds)
pytest tests/ -m "not slow"      # skip slow tests
pytest tests/ -m gpu             # GPU-only tests (requires CUDA)
```

## Markers

- `smoke` — quick sanity checks, should complete in <10s total
- `slow` — tests that take >10s individually (large graphs, convergence tests)
- `gpu` — tests requiring CUDA (skipped automatically if unavailable)

## Conventions

- Test files named `test_<module>.py`, mirroring source layout
- Fixtures in `conftest.py` for reusable sample graphs
- Layout tests should check convergence properties (e.g., DAG constraint reduces y-violations)
  rather than exact coordinate values (optimization is stochastic)
- Render tests should check structural output (SVG element count, figure created)
  rather than pixel-perfect comparison

## Maintainability Rules

- When fixing a regression, add the narrowest possible test that would have caught it.
- Smoke tests are the preferred place to pin billion-scale and workflow regressions in
  miniature.
- If you tighten type guarantees or docstring contracts in core modules, add tests that
  exercise the intended typed path rather than relying on static checking alone.

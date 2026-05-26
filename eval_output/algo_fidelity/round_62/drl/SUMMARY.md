# Round 62 DRL Port Summary

## Result

Replaced the fidelity-mode python-igraph delegation in
`dagua/layout/ops/pipelines/drl.py` with the native DrL solve path in
`dagua/layout/ops/drl.py`.

The port now uses:

- igraph's five-phase state machine: liquid, expansion, cooldown, crunch, simmer
- igraph's empty-first density-grid lifecycle
- separable `DensityGrid.cpp` falloff buckets
- coarse/fine density transitions with `first_add` and `fine_first_add`
- Python RNG hook semantics for benchmark fidelity mode
- NumPy `RandomState(seed).uniform(-1, 1)` initial matrices in fidelity mode
- final 50.0 coordinate scaling to match the benchmark adapter

## Verification

Forbidden-pattern scan on the scoped modules found no `import igraph`,
`from igraph`, `subprocess`, or `graph.layout("drl", ...)` delegation.

Passing checks:

```text
ruff check dagua/layout/ops/drl.py dagua/layout/ops/pipelines/drl.py --fix
python -m pytest tests/test_pipeline_drl.py -x --tb=short -q
30 passed, 2 warnings in 27.79s
mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file
```

Adapter smoke comparisons that reached exact `float32` parity:

```text
single node, default: RMSD 0.0, max 0.0
2-node edge, default: RMSD 0.0, max 0.0
5-node path, default: RMSD 0.0, max 0.0
6-node tree, final: RMSD 0.0, max 0.0
8-node star, final/refine/coarsen: RMSD 0.0, max 0.0
```

## Remaining Divergence

The remaining non-delegated divergence is in pruning-sensitive cases:

```text
8-node star, default: RMSD 51.25846862792969, max 106.71883392333984
8-node star, coarsest: RMSD 37.208343505859375, max 78.26630401611328
6-node weighted final: RMSD 62.6909065246582, max 120.63502502441406
```

The arithmetic step that diverges is the edge-cut trajectory in
`Solve_Analytic`: after cooldown lowers `min_edges`, tiny differences in
`maxLength > cut_off_length` select different erased neighbors. Once a different
neighbor is erased from the mutable `neighbors[node]` map, later attraction and
density updates intentionally follow a different path. The implementation keeps
the pure-Python port and does not delegate these cases.

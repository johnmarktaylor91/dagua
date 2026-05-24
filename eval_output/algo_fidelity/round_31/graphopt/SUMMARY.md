# Round 31 GraphOpt Implementation Summary

## Changes

- Added GraphOpt fidelity initialization support:
  - explicit `graphopt_initial_pos` seed matrix is honored before RNG fallback
  - `fidelity_mode=True` fallback uses `np.random.RandomState(seed).uniform(-1, 1, size=(N, 2))`
- Wired `classic_graphopt` to run the pipeline in fidelity mode and pass the same NumPy seed matrix as the igraph adapter.
- Gated GraphOpt edge-weight spring scaling behind non-fidelity mode so igraph parity ignores weights.
- Matched igraph's exact zero-distance predicates for GraphOpt repulsion and springs.
- Added regression tests for seed-matrix init, fidelity edge-weight suppression, and near-zero force predicates.

## Verification

Requested live compare:

```text
python scripts/algo_fidelity_live_compare.py classic_graphopt igraph_graphopt \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_31/graphopt/post_impl
```

Result:

```text
Wrote 3705 rows to eval_output/algo_fidelity/round_31/graphopt/post_impl/multi_seed_rmsd.csv
Wrote summary to eval_output/algo_fidelity/round_31/graphopt/post_impl/multi_seed_summary.json
graphs: 5
median: 0.043382
p25: 0.043382
p75: 0.061855
p95: 0.085867
worst: tl_mlp_3layer 0.091870
```

Focused regression tests:

```text
18 passed, 2 warnings in 0.83s
```

Quality gates:

```text
ruff check . --fix
Found 1 error (1 fixed, 0 remaining).

mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file
```

Final tier attempt:

```text
timeout 1800 pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
```

Failed during collection on pre-existing classic namespace exports:

```text
ERROR tests/test_classic_drl.py
ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic' (unknown location)
```

The broader targeted layout/graph gate was also attempted but was interrupted after prolonged machine saturation from concurrent round-31 agents running the same gate.

## Assumptions

- `fidelity_mode` is opt-in for public GraphOpt behavior so existing weighted GraphOpt semantics remain available outside fidelity benchmarks.
- The benchmark adapter's NumPy seed matrix is the parity target, not igraph's C fallback RNG path.

## Concerns

- Full-suite collection currently fails before GraphOpt tests due missing `dagua.layout.classic` namespace exports; fixing that is outside the GraphOpt A6 scope.
- Several unrelated round-31 tasks were modifying and staging files concurrently in the same worktree.


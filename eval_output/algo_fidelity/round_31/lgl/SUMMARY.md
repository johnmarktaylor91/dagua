# Round 31 LGL Implementation Summary

## Changes

- `dagua/layout/ops/lgl.py` (+69/-72): changed LGL initialization to igraph-style column-major RNG draws, advanced the shell RNG past root/initial-layout draws, placed shell-1 vertices with random normalized vectors, restricted grid neighbor enumeration to igraph's 4-cell pattern, used `1e-5` for repulsion distance fallback, and activated incident springs even when the opposite endpoint is in a future shell.
- `dagua/layout/_archive/classic/lgl.py` (+80/-70): mirrored the same LGL behavior in the legacy classic wrapper so existing pipeline parity tests still compare like with like; also aligned its max-change convergence rule with the pipeline.
- `tests/test_layout/test_lgl_fidelity.py` (+118/-1): added regression coverage for column-major initialization, shell-1 random placement, and igraph grid/repulsion constants.
- `tests/test_pipeline_lgl.py` (+1): made the weighted classic parity test opt into `use_edge_weights=True`, preserving the existing weighted-wrapper comparison while keeping default igraph fidelity unweighted.

## Assumptions

- Treated the connected `dagua/layout/classic/lgl.py` symlink target (`dagua/layout/_archive/classic/lgl.py`) as in scope because existing tests require the legacy wrapper and composable pipeline to stay bit-identical.
- Implemented the integrated-plan `igraph_2dgrid_in` quirk after the explicit L1-L4 bundle did not reach the requested verification range.

## Test results

Passed:

```text
ruff check . --fix
All checks passed!

mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file

python -m pytest tests/test_layout/test_lgl_fidelity.py tests/test_pipeline_lgl.py tests/test_graph.py -x --tb=short -q
62 passed, 3 warnings in 9.35s
```

Requested live compare:

```text
python scripts/algo_fidelity_live_compare.py classic_lgl igraph_lgl --seeds 30 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_31/lgl/post_impl
graphs: 5
median: 0.180581
p25: 0.170474
p75: 0.190661
p95: 0.190661
worst: linear_3layer_mlp 0.190661
```

The live compare did not meet the expected `0.05-0.08` median range.

Incomplete / failing broader gates:

```text
python -m pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
KeyboardInterrupt after 99 passed, 5 warnings in 1900.39s (0:31:40)
Interrupted in dagua/metrics.py:516 after no failure output.

timeout 1200 python -m pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
ERROR collecting tests/test_classic_drl.py
ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic' (unknown location)
```

## Controversial choices

- Added RNG stream advancement beyond the literal L2 draw-order change because igraph uses one stream for root, initial layout, and shell vectors.
- Added early spring activation for future-shell endpoints because the integrated A4 plan calls out `igraph_2dgrid_in` as effectively a no-op, and L1-L4 alone regressed the requested live comparison.

## Concerns

- Fidelity remains below the task expectation: median is `0.180581`, not `0.05-0.08`.
- The full non-slow test suite currently stops before running tests because `dagua.layout.classic` has no importable `layout_drl` export in this environment.
- The broad layout/graph gate is very slow and was interrupted after 31:40; it had reached 99 passing tests with no failures.

## Knowledge

- The live comparison reports cross-seed distribution medians; for this subset the igraph target's within-engine medians are also high on several graphs.
- LGL default fidelity ignores edge weights; tests that compare against the legacy weighted classic wrapper must pass `use_edge_weights=True`.

# Round 13 Summary -- Davidson-Harel vs igraph

Status: COMMITTED
Family: davidson_harel
Date: 2026-04-30

## Scope

Compared `classic_davidson_harel` against `igraph_davidson_harel` with 3 seeds
on the requested small graph subset. The comparator evaluated 5 graphs:
`linear_3layer_mlp`, `mixed_width_labels`, `nested_shallow_enc_dec`,
`parallel_multiedge_bundle`, and `tl_mlp_3layer`.

`binary_tree` was requested but was not present in the selected live comparator
target set, so all before/after numbers use the same 5 evaluated graphs.

## Changes

- Added explicit Davidson-Harel node-distance energy weight
  `node_dist = 1.0`.
- Aligned energy weights to igraph defaults:
  `{border = 0.0, edge_lengths = 0.0001, edge_crossings = 1.0,
  node_edge_dist = 0.2}`.
- Switched the objective from normalized per-term energies to unnormalized
  igraph-style energies.
- Aligned the annealing move schedule with igraph:
  shuffled node order, 30 shuffled circular directions per node, initial
  move radius equal to the layout half-width, and uphill acceptance using
  `exp(-dE / move_radius)`.

## Measurements

Baseline:

```text
output: eval_output/algo_fidelity/round_13/baseline_small
graphs: 5
median: 0.361980
p25: 0.343786
p75: 0.381322
p95: 0.431442
worst: tl_mlp_3layer 0.443971
```

Energy-only post-fix:

```text
output: eval_output/algo_fidelity/round_13/post_fix
graphs: 5
median: 0.362218
p95: 0.411233
worst: tl_mlp_3layer 0.417369
```

Energy plus radius-schedule post-fix:

```text
output: eval_output/algo_fidelity/round_13/post_fix_schedule
graphs: 5
median: 0.335877
p95: 0.384263
worst: nested_shallow_enc_dec 0.385607
```

Final full schedule post-fix:

```text
output: eval_output/algo_fidelity/round_13/post_fix_full
graphs: 5
median: 0.237719
p25: 0.210108
p75: 0.279155
p95: 0.286307
worst: linear_3layer_mlp 0.288095
```

Commit criterion met by median RMSD improvement:

```text
0.361980 -> 0.237719
improvement: 0.124261
```

Graph-level TOST at `2x` remained unchanged in classification:
`linear_3layer_mlp` is `equivalent_at_2x`; the other four evaluated graphs
remain `not_equivalent`. This is a partial-match improvement, not a
weak-equivalence reclassification.

## Tests

```text
ruff check dagua/layout/ops/davidson_harel.py --fix
All checks passed!

mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file

pytest tests/test_layout/ -x --tb=short -q -k "davidson" 2>&1 | tail -30
233 deselected in 0.20s

pytest tests/test_layout/ -x --tb=short -q 2>&1 | tail -10
233 passed, 6 warnings in 1288.77s (0:21:28)

pytest tests/test_graph.py -x --tb=short -q
37 passed in 0.55s
```

`ruff check . --fix` was run once and passed after making an automatic
out-of-scope simplification in `scripts/sprint_9_preflight.py`; that
out-of-scope edit was manually reverted to preserve Round 13 scope. The touched
Davidson-Harel file passes ruff directly.

## Residual

The biggest remaining gap is likely incremental-energy fidelity and edge
multiplicity semantics. Dagua still recomputes full energy over unique
undirected edges, while igraph evaluates per-move deltas over the graph's
neighbor/edge iteration APIs, including multiple edges for some terms.

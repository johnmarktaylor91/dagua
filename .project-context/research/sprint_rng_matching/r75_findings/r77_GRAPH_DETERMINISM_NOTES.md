# r77 Graph Determinism Notes

Date: 2026-07-04
Branch: r77/graph-determinism
Worktree: /home/jtaylor/.claude/worktrees/dagua-graph-determinism
Implementation commit: 4178b93

## Verdict

Benchmark graph generation is now hash-deterministic for the full
`get_test_graphs()` catalog under different `PYTHONHASHSEED` values.

Pre-fix cross-hash probing found exactly two hash-dependent benchmark graph
realizations:

```text
random_dag_50
random_dag_200
```

Regenerate downstream benchmark rows for these graphs for all engines and both
comparison sides. The previous realizations were process-hash-dependent and
therefore had no canonical historical form to preserve.

## Audit Table

| Builder / helper | Hash-dependent construct | Fix |
|---|---|---|
| `dagua/eval/graphs.py:_random_dag` | `edges: Set[tuple[str, str]]` fed to `DaguaGraph.from_edge_list(list(edges), num_nodes=n_nodes)`. String tuple set iteration depended on `PYTHONHASHSEED`, and `from_edge_list` assigned string node IDs in that unstable order after preallocating integer nodes. | Build explicit named nodes `n0..nN` with `_build_named_graph(node_names, sorted(edges))`. This preserves documented named-node intent and sorts string edge keys canonically. |
| `dagua/eval/graphs.py:make_wide_dag` | `edges_set` of integer tuples was converted with `list(edges_set)` before tensor construction. Integer tuple order was stable across the tested hash seeds but still set-order-derived. | Use `sorted(edges_set)` before building `edge_index`; add precise set type. |
| `dagua/eval/graphs.py:make_random_dag` | `edges_set` of integer tuples was converted with `list(edges_set)` before tensor construction. Integer tuple order was stable across the tested hash seeds but still set-order-derived. | Use `sorted(edges_set)` before building `edge_index`; add precise set type. |
| `dagua/eval/graphs.py:make_scale_free` | `targets: Set[int]` deduplicates random choices. | Already sorted before emitting edges: `for target_idx in sorted(targets)`. No change required. |
| `dagua/eval/graphs.py:make_dependency_graph` | `dep_candidates: Set[int]` deduplicates random choices. | Already sorted before emitting edges: `for dep_idx in sorted(dep_candidates)`. No change required. |
| `dagua/eval/graphs.py:make_small_world` | `edge_set: Set[Tuple[str, str]]` collects rewired string edges. | Already passed as `sorted(edge_set)` to `_build_named_graph`. No change required. |
| `dagua/eval/graphs.py:_make_random_bipartite_graph` | `edge_set: set[tuple[int, int]]` collects sampled edges. | Already passed as `sorted(edge_set)` to `_graph_from_integer_edges`. No change required. |
| `dagua/eval/graphs.py:_undirected_to_dag` | `seen_edges` is a membership-only dedupe set while iterating tensor edge order. | No hash-order output dependency; output order follows existing tensor order. No change required. |
| `dagua/graph.py:DaguaGraph.from_edge_list` | Node ID assignment follows caller edge order. | No generic change. The benchmark bug was fixed at generation call sites by passing deterministic edge/node order. |

## Affected Graphs

The pre-fix probe compared serialized `get_test_graphs()` output from two
fresh subprocesses with `PYTHONHASHSEED=0` and `PYTHONHASHSEED=1`.

Affected graph names:

```text
random_dag_50
random_dag_200
```

These are the only catalog entries whose `num_nodes`, `edge_index`,
`node_labels`, or `edge_labels` differed across the two pre-fix subprocesses.

## Tests Added

Added `tests/test_eval/test_graphs.py::test_benchmark_graphs_are_hash_seed_deterministic`.
The test spawns two subprocesses with different `PYTHONHASHSEED` values,
builds every `get_test_graphs()` graph in each subprocess, serializes
`num_nodes`, `edge_index`, `node_labels`, and `edge_labels`, and asserts the
JSON bytes are identical.

## Test Evidence

Passing:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/test_eval/test_graphs.py::test_benchmark_graphs_are_hash_seed_deterministic -q
1 passed, 3 warnings in 124.63s (0:02:04)
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/test_generate_comprehensive_gallery.py -q
6 passed, 3 warnings in 0.61s
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check . --fix
All checks passed!
```

Requested broad gate was run and hit failures outside graph generation:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -k "graphs or graph" -x -q
FAILED tests/test_integration.py::TestLargerGraphs::test_50_node_dag
assert m["node_overlaps"] == 0
actual: 44
summary before stop: 1 failed, 145 passed, 2556 deselected, 84 warnings in 242.24s
```

Seed probe for that failure:

```text
LayoutConfig(seed=None) -> node_overlaps=44, edge_crossings=0
LayoutConfig(seed=0)    -> node_overlaps=44, edge_crossings=0
LayoutConfig(seed=42)   -> node_overlaps=44, edge_crossings=0
LayoutConfig(seed=100)  -> node_overlaps=44, edge_crossings=0
```

The same selected suite excluding slow tests also hit an unrelated fidelity
initialization failure:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -k "graphs or graph" -m "not slow" -x -q
FAILED tests/test_ops_init.py::test_graphopt_fidelity_init_matches_igraph_adapter_seed_matrix
summary before stop: 1 failed, 249 passed, 2568 deselected, 83 warnings in 502.74s
```

## Test Maintenance

`tests/test_generate_comprehensive_gallery.py` had stale expectations for the
current gallery generator constants. The selected graph gate reached these
tests because the file contains graph-builder tests. Updated assertions only:

```text
edge_arrow_length values: [5.0, 12.0, 20.0, 30.0, 45.0]
node_stroke_width demo min size: 124.0 x 78.0
edge_color_gradient endpoint labels: Source / Target
edge_head_tail label font/offset: 12.0 / 16.0
edge_arrow_length showcase positions: +/-90.0
```

No production gallery code was changed.

## Controversial Choices

- `_random_dag` now uses sorted string edge keys and explicit named nodes
  instead of attempting to reproduce any historical hash seed. This changes the
  canonical realization, which is intentional because the old realization was
  process-dependent.
- `make_wide_dag` and `make_random_dag` were also sorted even though their
  integer tuple sets did not differ across `PYTHONHASHSEED=0/1`; they still
  depended on set iteration for edge order.

## Dead Code

No newly unreachable code was introduced.

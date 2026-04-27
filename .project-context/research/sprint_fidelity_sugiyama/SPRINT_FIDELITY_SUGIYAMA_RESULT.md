# Sprint Fidelity Sugiyama Result

## Summary

The highest-leverage divergence was coordinate spacing, not crossing reduction.
`LayoutConfig(algorithm="sugiyama")` reached the composable Sugiyama pipeline,
but the pipeline ignored `LayoutConfig.rank_sep` and `LayoutConfig.node_sep`.
It therefore used historical unit spacing (`1.0`, `1.0`) while metrics still
used real node sizes. The result was severe overlap and inflated edge-length /
straightness penalties versus `igraph_sugiyama` and `graphviz_dot`.

The fix makes engine-dispatched Sugiyama runs inherit `LayoutConfig` spacing
while preserving direct `layout_sugiyama_pipeline(...)` unit-spacing behavior.

## Top-5 Gap Closure

Population: 93 benchmark graphs with `N <= 500`. Canonical score is the better
of cached `igraph_sugiyama` and `graphviz_dot` for each graph.

| graph | canonical | old score | new score | canonical score | old gap | new gap | closed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `braided_feedback_tails` | `igraph_sugiyama` | 53.323 | 78.874 | 77.692 | 24.369 | -1.182 | 104.9% |
| `small_label_storm` | `graphviz_dot` | 62.619 | 90.477 | 86.800 | 24.181 | -3.677 | 115.2% |
| `nested_cluster_label_stack` | `graphviz_dot` | 61.324 | 85.277 | 85.227 | 23.903 | -0.050 | 100.2% |
| `shape_and_routing_matrix` | `graphviz_dot` | 62.405 | 89.740 | 85.879 | 23.474 | -3.861 | 116.4% |
| `hexagonal_lattice_42` | `graphviz_dot` | 57.666 | 84.308 | 81.095 | 23.429 | -3.212 | 113.7% |

Mean top-5 gap moved from `23.871` to `-2.396`, closing `110.0%` of the gap.
Across the 88 successfully scored small graphs, no graph regressed by `>= 1.0`
composite.

## Diagnosis

- Dummy node insertion was not the top-5 driver. Long-edge handling remained
  stable; the largest losses appeared even on small graphs where the main
  symptom was node overlap.
- Crossing reduction was not the top-5 driver. Most top gaps had matching or
  near-matching crossing rates, often `0`.
- Coordinate assignment spacing was the root cause. Unit rank/layer spacing
  compressed layouts under real label-size metrics, producing large
  `overlap_count`, high `edge_length_cv`, and poor straightness.
- Edge routing remains a fidelity gap for visual output, but the composite
  scores measured here are node-position metrics and did not require splines.
- Layer ranking differences remain possible, especially on cyclic inputs, but
  they were not the highest-leverage fix for the measured top-5 score gaps.

## Validation

Passed:

- `black --check dagua/layout/ops/pipelines/sugiyama.py tests/test_pipeline_sugiyama.py`
- `ruff check dagua/layout/ops/pipelines/sugiyama.py tests/test_pipeline_sugiyama.py`
- `mypy --follow-imports=silent dagua/cli.py`
- `pytest tests/test_pipeline_sugiyama.py -x --tb=short -q`
  - `25 passed in 0.85s`
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`
  - `258 passed, 1 warning in 1236.49s`

Blocked by unrelated existing issues:

- `ruff check . --fix` fails on long lines in untracked scripts:
  `scripts/cleanup_for_salvage_round.py`,
  `scripts/cleanup_watchdog_errors.py`,
  `scripts/flip_running_to_skipped.py`, and
  `scripts/restore_skip3_from_backup.py`.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`
  fails during collection because `tests/test_classic_drl.py` imports
  `layout_drl` from namespace package `dagua.layout.classic`, which has no
  local `__init__.py` export in this checkout.

## Remaining Concerns

Five cyclic benchmark graphs still fail in the Sugiyama pipeline before
scoring with `ValueError: graph must be acyclic after back-edge reversal`:
`recurrent_feedback_cell`, `kitchen_sink_hybrid_net`,
`kitchen_sink_platform_graph`, `disconnected_label_cycle_collage`, and
`center_port_backedge_hub`. That is a separate cycle-removal/layer-ranking
fidelity issue and was left untouched because this sprint applied one
divergence fix.

# Sprint CUDA Fix Result

Date: 2026-04-26
Branch: `codex/sprint-31a-gate-refinement`

## Summary

Partial fix shipped. All 13 empirically failing graphs now complete with
`LayoutConfig(device="cuda")`, and the two selected known-working CUDA graphs
still complete. The remaining validation gap is CPU/CUDA composite parity on
three graphs; this appears to be numerical/local-minimum divergence rather than
device placement, because those graphs also diverge with edge polish disabled.

## Root Cause

Two native polish paths mixed CPU and CUDA tensors:

1. `_should_lattice_uniform_centered_slots()` created its degree tensor on CPU
   while indexing it with CUDA `edge_index` tensors.
2. The component-decomposition branch tiled child component layouts on CUDA, but
   then called `_best_of_polish()` with the original unprepared CPU `edge_index`
   and `node_sizes` arguments.

## Changes

- `dagua/layout/ops/pipelines/dagua_native.py`
  - Create the lattice gate degree tensor on `edge_index.device`.
  - Use `prepared_edge_index` and `normalized_node_sizes` for component tiling
    polish.
- `tests/test_layout/test_cuda_activation.py`
  - Added CUDA regression coverage for component tiling polish.
  - Added CUDA regression coverage for the lattice uniform-slots gate.
- `/tmp/sprint_cuda_fix_validate.py`
  - Added validation script for all 13 failing graphs plus `ba_500` and
    `citation_dag_300`.

## Validation

Direct CUDA reproduction after fix:

```text
OK disconnected_encoder_residual (9, 2) cuda:0
OK disconnected_label_cycle_collage (7, 2) cuda:0
OK grid_5x5 (25, 2) cuda:0
OK grid_rect_6x8 (48, 2) cuda:0
OK heavy_tail_weights_50 (50, 2) cuda:0
OK hexagonal_lattice_42 (42, 2) cuda:0
OK org_chart_deep (79, 2) cuda:0
OK protein_ppi_200 (200, 2) cuda:0
OK random_dag_200 (383, 2) cuda:0
OK random_dag_50 (97, 2) cuda:0
OK regular_3_30 (30, 2) cuda:0
OK regular_4_40 (40, 2) cuda:0
OK rgg_100 (100, 2) cuda:0
```

`/tmp/sprint_cuda_fix_validate.py`:

```text
originally_failing_successes=13/13
working_regression_successes=2/2
OK disconnected_encoder_residual: cpu=81.186291 cuda=81.186291 delta=0.000000
OK disconnected_label_cycle_collage: cpu=80.630010 cuda=80.630009 delta=0.000001
OK grid_5x5: cpu=94.136224 cuda=94.136221 delta=0.000002
OK grid_rect_6x8: cpu=93.215687 cuda=93.215688 delta=0.000001
OK heavy_tail_weights_50: cpu=77.374287 cuda=77.374286 delta=0.000001
OK hexagonal_lattice_42: cpu=89.113593 cuda=89.113593 delta=0.000000
OK org_chart_deep: cpu=92.829783 cuda=92.829783 delta=0.000000
MISMATCH protein_ppi_200: cpu=69.403020 cuda=69.870251 delta=0.467231
OK random_dag_200: cpu=74.123188 cuda=74.179286 delta=0.056098
OK random_dag_50: cpu=70.854048 cuda=70.854050 delta=0.000002
OK regular_3_30: cpu=77.027229 cuda=77.091072 delta=0.063843
MISMATCH regular_4_40: cpu=70.087706 cuda=69.659317 delta=0.428389
OK rgg_100: cpu=72.772498 cuda=72.772627 delta=0.000128
MISMATCH ba_500: cpu=63.138364 cuda=62.945109 delta=0.193256
OK citation_dag_300: cpu=63.718850 cuda=63.667860 delta=0.050990
```

Targeted checks:

```text
pytest tests/test_layout/test_cuda_activation.py::test_cuda_component_tiling_polish_keeps_edges_on_device tests/test_layout/test_cuda_activation.py::test_cuda_lattice_uniform_slots_gate_keeps_degree_on_device -q
2 passed in 1.86s
```

```text
pytest tests/test_layout/ -x --tb=short -q
221 passed, 1 warning in 1082.12s (0:18:02)
```

```text
pytest tests/test_graph.py -x --tb=short -q
37 passed in 1.16s
```

Quality gates:

```text
ruff check dagua/layout/ops/pipelines/dagua_native.py tests/test_layout/test_cuda_activation.py --fix
All checks passed!
```

```text
mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file
```

## Blockers / Not Fixed

- `ruff check . --fix` is blocked by pre-existing untracked script files under
  `scripts/` with E501 line-length errors. They are outside this sprint scope.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`
  stops during collection on `tests/test_classic_drl.py` because
  `dagua.layout.classic` lacks an importable `layout_drl` export. This is outside
  the CUDA placement path.
- CPU/CUDA composite parity remains outside the requested 0.1 tolerance for
  `protein_ppi_200`, `regular_4_40`, and `ba_500`. A follow-up should address
  cross-device deterministic initialization/optimization parity separately.

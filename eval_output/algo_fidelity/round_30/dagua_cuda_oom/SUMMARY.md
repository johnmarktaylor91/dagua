# Dagua CUDA OOM Initialization Fix

## Root Cause

`dagua/layout/ops/pipelines/dagua_native.py:130:31` was the first unconditional
CUDA tensor materialization for the native pipeline:
`edge_index.to(device=device, dtype=torch.long)`.

Local CUDA did not reproduce the benchmark OOM on a 14-node graph, but the
benchmark failures occurred at 0.05-0.4 seconds for every graph size, which
matches CUDA context/cache materialization pressure before real layout work
rather than graph-size-dependent layout tensors.

## Fix

Native tensor preparation now goes through `_prepare_native_tensors_for_device`.
If the requested CUDA preparation raises a CUDA out-of-memory error, dagua
empties the CUDA cache and retries the same graph on CPU. Non-CUDA failures and
non-OOM CUDA failures still raise normally.

## Verification

- Reproduced locally: direct CUDA and adapter runs on a 14-node graph did not
  OOM on this machine.
- `pytest tests/test_layout/ -x --tb=short -q -k "dagua_native or test_dagua_native_cuda_oom"`
  passed: 6 passed, 332 deselected.
- `ruff check . --fix` passed.
- `mypy --follow-imports=silent dagua/cli.py` passed.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q` passed:
  377 passed.
- Dagua competitor adapter on CUDA returned finite tensors for
  `linear_3layer_mlp`, `binary_tree`, `asymmetric_hourglass_hub`, `ba_500`,
  and `grid_5x5`.

## Remaining Concern

`ba_5000` passed the initialization/OOM point and entered CUDA overlap
projection, but was terminated after running for more than 10 minutes without
returning. That appears to be a separate runtime/performance issue.

The final fast suite command
`pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`
failed during collection on an unrelated import:
`tests/test_classic_drl.py` cannot import `layout_drl` from
`dagua.layout.classic`.

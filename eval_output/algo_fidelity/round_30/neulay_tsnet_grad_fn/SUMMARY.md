# NeuLay / tsNET Grad-Fn Fix

## Root Cause

The classic NeuLay and tsNET pipeline ops performed differentiable loss
construction and `loss.backward()` while assuming global grad mode was enabled.
When benchmark execution wrapped layout calls in `torch.no_grad()`, the position
or model tensors still had `requires_grad=True`, but the loss tensors were
created without `grad_fn`, causing:

```text
element 0 of tensors does not require grad and does not have a grad_fn
```

Root cause sites:

- `dagua/layout/ops/neulay.py:617`: GCN warm-start forward/loss/backward needed
  local grad mode.
- `dagua/layout/ops/neulay.py:770`: direct RMSprop refinement loss/backward
  needed local grad mode.
- `dagua/layout/ops/tsnet.py:430`: KL loss/backward needed local grad mode.

## Fix

Wrapped each differentiable forward/loss/backward block in `torch.enable_grad()`
while leaving optimizer updates and final position normalization behavior
unchanged.

Added `tests/test_layout/test_neulay_tsnet_grad.py` with regressions that run
both pipelines inside an outer `torch.no_grad()` context and assert finite
`[N, 2]` outputs.

## Verification

- `ruff check dagua/layout/ops/neulay.py dagua/layout/ops/tsnet.py tests/test_layout/test_neulay_tsnet_grad.py --fix`
  passed.
- `mypy --follow-imports=silent dagua/cli.py`
  passed.
- `pytest tests/test_layout/ -x --tb=short -q -k "neulay or tsnet"`
  passed: 6 passed, 334 deselected, 2 warnings.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`
  passed: 377 passed, 8 warnings.
- `classic_neulay_default` via `classic_neulay.layout_with_variant()` on
  `triangular_lattice_36`, seed 42, under `torch.no_grad()` returned `(36, 2)`
  with no error.
- `classic_tsnet_default` via `classic_tsnet.layout_with_variant()` on
  `triangular_lattice_36`, seed 42, under `torch.no_grad()` returned `(36, 2)`
  with no error.

## Blocked Gate

`ruff check . --fix` still fails outside this task scope on line-length issues
in `dagua/layout/ops/pipelines/dagua_native.py:101` and `:125`.

`pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`
still fails during collection outside this task scope because
`tests/test_classic_drl.py` imports missing `dagua.layout.classic.layout_drl`.

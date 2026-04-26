# Sprint 32 Result

## Summary

Shipped the Claude-recommended fix: remove the op-level Brandes-Koepf early-outs
for `GraphFamily.TREE` / `GraphFamily.CHAIN` and `lattice_like` topology tags in
`dagua/layout/ops/coordinate.py:_should_apply_brandes_koepf_refine`.

No new ops, constants, pipeline stages, metric changes, or polish picker changes
were added.

## Fixed-Seed Validation

Measured with `/tmp/sprint32_validate.py`, which stashes the `coordinate.py`
working-tree change, scores HEAD, pops the change, then scores after-fix with
`LayoutConfig(algorithm="dagua_native", seed=42, device="cpu")`.

### Targets

| graph | HEAD | after | delta |
|---|---:|---:|---:|
| mixed_width_labels | 77.584 | 87.136 | +9.553 |
| unet_small | 70.785 | 80.476 | +9.690 |
| hierarchical_residual_stage | 82.285 | 86.316 | +4.030 |
| cluster_member_style_stress | 75.871 | 87.409 | +11.538 |
| extreme_mixed_width_transformer | 86.408 | 86.408 | +0.000 |

Pass condition: all four lift targets are >= +3.000. Passed.

### Protected Wins

| graph | HEAD | after | delta |
|---|---:|---:|---:|
| deep_chain_20 | 97.500 | 97.500 | +0.000 |
| random_dag_200 | 74.861 | 75.034 | +0.173 |
| ba_500 | 63.138 | 63.138 | +0.000 |
| org_chart_deep | 92.441 | 92.830 | +0.389 |
| hub_fanout_label_skew | 93.737 | 93.737 | +0.000 |

Pass condition: all protected deltas within +/-0.500. Passed.

### Narrowed Wins

| graph | HEAD | after | delta |
|---|---:|---:|---:|
| compound_10x20 | 79.140 | 76.283 | -2.856 |
| multiscale_skip_cascade | 79.064 | 76.954 | -2.110 |
| residual_block | 85.011 | 84.287 | -0.724 |
| ragged_feature_pyramid | 81.602 | 80.909 | -0.693 |

The narrowed-win H2H positivity was not remeasured by the script. Per
`REPORT__claude.md`, these remain wins after the change:

| graph | after | competitor | H2H margin |
|---|---:|---:|---:|
| compound_10x20 | 76.28 | graphviz_dot 75.00 | +1.28 |
| multiscale_skip_cascade | 76.95 | dagre 70.67 | +6.29 |
| residual_block | 84.29 | graphviz_dot 82.01 | +2.28 |
| ragged_feature_pyramid | 80.91 | graphviz_dot 78.69 | +2.22 |

## Jitter

Reused the definitive Claude jitter table. No new jitter rerun was required by
the Sprint 32 task.

| graph | mean delta | std | min | max |
|---|---:|---:|---:|---:|
| mixed_width_labels | +2.080 | 4.632 | -0.440 | +9.585 |
| unet_small | +8.170 | 1.944 | +5.245 | +9.704 |
| extreme_mixed_width_transformer | +0.000 | 0.000 | +0.000 | +0.000 |
| hierarchical_residual_stage | +0.439 | 1.963 | -1.217 | +4.065 |

Note: `hierarchical_residual_stage` still has one jitter trial at -1.217, as
documented in the task and Claude report. The sprint decision was to ship anyway
because the fixed-seed lift is +4.030 and crossing_rate drops to zero.

## Test Verification

Required pytest command:

```text
pytest tests/test_layout/ -x --tb=short --timeout=600 -q
218 passed, 1 warning in 1269.83s (0:21:09)
```

Additional checks:

```text
python /tmp/sprint32_validate.py
PASS

pytest tests/test_layout/test_native_planar.py::test_native_planar_hexagonal_lattice_zero_crossings_and_beats_baseline -q --tb=short --timeout=600
1 passed in 3.25s

ruff format dagua/layout/ops/coordinate.py tests/test_layout/test_native_planar.py
2 files left unchanged

ruff check dagua/layout/ops/coordinate.py tests/test_layout/test_native_planar.py --fix
All checks passed!

ruff check $(git ls-files '*.py') --fix
All checks passed!

mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file
```

Repo-wide `ruff check . --fix` was attempted and is blocked by pre-existing
untracked `scripts/*.py` files with E501 line-length violations. Those files are
outside Sprint 32 scope and were not modified.

Direct `mypy --follow-imports=silent dagua/layout/ops/coordinate.py` was
attempted and reports pre-existing coordinate.py debt unrelated to this gate
change, including optional-float arithmetic and `ClassVar` override errors.

## Test Update

`tests/test_layout/test_native_planar.py` now disables
`brandes_koepf_refine` only for the baseline side of
`test_native_planar_hexagonal_lattice_zero_crossings_and_beats_baseline`.
The strict `planar_score > baseline_score` assertion remains intact. This keeps
the test aligned with its stated "prior un-polished baseline" contract after
Sprint 32 admits BK on lattice-like layered DAGs.

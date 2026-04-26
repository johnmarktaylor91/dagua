# Sprint 36 Result

## Summary

Dropped the op-level Brandes-Koepf component-count admission check in
`dagua/layout/ops/coordinate.py:_should_apply_brandes_koepf_refine`.

The gate now keeps only:

- non-empty graph
- minimum layer count
- strict forward layering

The old `_weak_component_sizes()` helper became unused after the gate removal
and was deleted.

## Regression Coverage

Added
`tests/test_layout/test_brandes_koepf_native.py::test_brandes_koepf_refinement_admits_multi_component_forward_dag`.

The test builds a two-component, six-layer forward DAG and verifies that BK:

- applies successfully
- preserves layer-derived y coordinates
- rewrites x coordinates

## Probe Validation

Measured with `/tmp/sprint36_probe.py` against the pre-fix tree by temporarily
stashing the working-tree Sprint-36 edits, then restoring them after the probe.
This preserves the probe's intended before/after comparison because the script
monkey-patches the gate.

| graph | before | after | delta |
|---|---:|---:|---:|
| disconnected_encoder_residual | 81.186 | 81.186 | +0.000 |
| multi_component_80 | 74.847 | 79.343 | +4.496 |
| deep_chain_20 | 97.500 | 97.500 | +0.000 |
| random_dag_200 | 75.130 | 75.130 | +0.000 |
| ba_500 | 63.138 | 63.138 | +0.000 |
| org_chart_deep | 92.830 | 92.830 | +0.000 |
| hub_fanout_label_skew | 93.737 | 93.737 | +0.000 |

Runtime check from the same probe:

| graph | before | after | speedup |
|---|---:|---:|---:|
| disconnected_encoder_residual | 1.72s | 0.41s | 4.2x |

Jitter check:

| graph | mean delta | std | min | max |
|---|---:|---:|---:|---:|
| disconnected_encoder_residual | +0.000 | 0.000 | +0.000 | +0.000 |
| multi_component_80 | +4.722 | 0.061 | +4.634 | +4.801 |

The probe's built-in verdict still says `NO-FIX` because its older target rule
requires every target, including `disconnected_encoder_residual`, to gain more
than one composite point. Sprint-36's pass criteria are different:

- `multi_component_80` delta >= +3.0: passed
- protected wins within +/-0.5: passed
- `disconnected_encoder_residual` runtime improvement without score regression:
  passed

## Test Verification

```text
black --check --line-length 100 dagua/layout/ops/coordinate.py tests/test_layout/test_brandes_koepf_native.py
All done! 2 files would be left unchanged.
```

```text
ruff check dagua/layout/ops/coordinate.py tests/test_layout/test_brandes_koepf_native.py --fix
All checks passed!
```

```text
ruff check $(git ls-files '*.py') --fix
All checks passed!
```

```text
mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file
```

```text
pytest tests/test_layout/test_brandes_koepf_native.py -q --tb=short --timeout=600
5 passed in 80.03s (0:01:20)
```

```text
pytest tests/test_layout/ -x --tb=short --timeout=600 -q
219 passed, 1 warning in 1159.48s (0:19:19)
```

Warning:

```text
tests/test_layout/test_spatial_hash_losses.py::test_default_native_cell_list_500_node_layout_matches_exact_composite
  /home/jtaylor/projects/dagua/dagua/metrics.py:468: ConstantInputWarning:
  An input array is constant; the correlation coefficient is not defined.
```

## Notes

- No changes were made to `dagua/metrics.py`.
- No changes were made to `dagua/layout/ops/pipelines/dagua_native_legacy.py`.
- No changes were made to `dagua/layout/init_placement.py`.
- No graph-specific gate, signature check, or metric change was added.
- No dead code remains from the removed component-count helper.

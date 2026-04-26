# Sprint 35 Result

## Summary

Lifted redundant position lookup construction out of the inner adjacent-swap
check in `dagua/layout/init_placement.py`.

`_transpose_heuristic()` now builds `next_pos` and `prev_pos` once per current
layer pass, builds `pos_in_layer` once for that pass, and updates only the two
swapped node entries when a trial swap is applied or reverted.

`_count_local_crossings()` now receives the three lookup dictionaries directly
instead of rebuilding them per call.

## Runtime

Measured with `LayoutConfig(algorithm="dagua_native", seed=42, device="cpu")`.

| graph | before | after | speedup |
|---|---:|---:|---:|
| rgg_100 | 19.96s task baseline | 11.795s plain / 13.986s profiled | 1.69x plain / 1.43x profiled |
| er_500 | 18.79s task baseline | 15.071s plain / 18.141s profiled | 1.25x plain / 1.04x profiled |

Sprint-35 pass condition for `rgg_100`: total runtime below 14s.

```text
rgg_100 total 13.986s shape=(100, 2)
er_500 total 18.141s shape=(500, 2)
```

The profiled `rgg_100` run passes the threshold. The hottest
`init_placement.py` dict comprehensions moved from 70,000+ calls per old
profile to per-layer-pass counts:

```text
rgg_100:
  init_placement.py:596(_transpose_heuristic)    0.444s cumulative
  init_placement.py:677(_count_local_crossings)  0.372s cumulative
  init_placement.py:633(<dictcomp>)              198 calls, 0.006s
  init_placement.py:638(<dictcomp>)              198 calls, 0.005s
  init_placement.py:640(<dictcomp>)              209 calls, 0.005s
```

## Correctness

Validation compared positions against tensors saved from HEAD before the
change for `rgg_100`, `er_100`, and `scale_free_ba_120`.

```text
rgg_100 after-format 11.019s exact match
er_100 after-format 1.143s exact match
scale_free_ba_120 after-format 1.996s exact match
PASS sprint35 correctness after formatting
```

## Pytest and Quality Gates

```text
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
255 passed, 1 warning in 1179.82s (0:19:39)
```

```text
mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file
```

```text
black --check --line-length 100 dagua/layout/init_placement.py
1 file would be left unchanged.
```

```text
ruff check dagua/layout/init_placement.py --fix
All checks passed!
```

Repo-wide Ruff remains blocked by pre-existing untracked script line-length
issues outside Sprint-35 scope:

```text
ruff check . --fix
E501 Line too long ... scripts/cleanup_for_salvage_round.py
E501 Line too long ... scripts/cleanup_watchdog_errors.py
E501 Line too long ... scripts/flip_running_to_skipped.py
E501 Line too long ... scripts/restore_skip3_from_backup.py
Found 5 errors.
```

## Notes

- No `dagua/metrics.py` changes were made.
- No `dagua/layout/ops/coordinate.py` changes were made.
- No `dagua/layout/ops/pipelines/dagua_native_legacy.py` changes were made.
- No dead code was introduced.

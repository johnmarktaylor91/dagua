# Sprint 34 Result

## Summary

Replaced the sampled metric pure-Python BFS hot path in `dagua/metrics.py` with
a scipy sparse all-pairs unweighted shortest-path call.

`full()` now computes the all-pairs graph-distance matrix once per metric
evaluation and passes it to both `sampled_stress()` and
`neighborhood_preservation()`. Direct callers can still omit the optional
`all_pairs_dist` argument and get the same metric behavior.

## Runtime

Measured with `LayoutConfig(algorithm="dagua_native", seed=42, device="cpu")`
on the Sprint 33 profiled graph set. Before values are the Sprint 33 profiling
baseline supplied in the Sprint 34 task.

| graph | before | after | speedup |
|---|---:|---:|---:|
| er_500 | 53.80s | 16.480s | 3.26x |
| grid_20x20 | 18.94s | 5.345s | 3.54x |
| scale_free_ba_120 | 9.20s | 2.351s | 3.91x |
| real_lesmis_77 | 6.00s | 2.106s | 2.85x |
| dependency_graph_100 | 5.77s | 2.323s | 2.48x |
| er_100 | 4.14s | 1.205s | 3.44x |
| rgg_100 | 24.78s | 12.162s | 2.04x |
| hub_and_spoke_3x20 | 4.07s | 2.052s | 1.98x |
| random_dag_50 | 1.22s | 0.669s | 1.82x |
| **aggregate** | **127.92s** | **44.693s** | **2.86x** |

Sprint 34 pass condition for `er_500`: after runtime below 35s.

```text
er_500 total 16.623s shape=(500, 2)
```

Pass: 16.623s is 3.24x faster than the 53.8s baseline.

## Correctness

Validation script: `/tmp/sprint34_validate.py`.

Checks performed:

- `full()` on `asymmetric_hourglass_hub` through the legacy BFS distance path
  vs the new scipy all-pairs path.
- `sampled_stress()` and `neighborhood_preservation()` on all nine profiled
  graphs through both paths.
- Numeric values agreed exactly for integer terms and within `1e-6` for float
  terms.

```text
PASS sprint34 correctness: legacy BFS and scipy all-pairs metrics agree
```

## Pytest

Requested full layout suite:

```text
pytest tests/test_layout/ -x --tb=short --timeout=600 -q
218 passed, 1 warning in 1159.50s (0:19:19)
```

Focused metrics regression tests:

```text
pytest tests/test_metrics.py -q --tb=short
24 passed in 2.63s
```

Tier 1 targeted layout/graph gate:

```text
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
255 passed, 1 warning in 1170.07s (0:19:30)
```

CLI type gate:

```text
mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file
```

Changed-file formatting and lint:

```text
ruff check dagua/metrics.py tests/test_metrics.py --fix
All checks passed!
ruff format --check dagua/metrics.py tests/test_metrics.py
2 files already formatted
```

Repo-wide ruff remains blocked by pre-existing untracked script line-length
issues outside Sprint 34 scope:

```text
ruff check . --fix
E501 Line too long ... scripts/cleanup_for_salvage_round.py
E501 Line too long ... scripts/cleanup_watchdog_errors.py
E501 Line too long ... scripts/flip_running_to_skipped.py
E501 Line too long ... scripts/restore_skip3_from_backup.py
Found 5 errors.
```

Project-wide non-slow pytest remains blocked during collection by a pre-existing
classic-layout import issue outside Sprint 34 scope:

```text
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
ERROR tests/test_classic_drl.py
ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'
```

## Notes

- No `dagua/layout/ops/pipelines/` files were touched.
- No `dagua/layout/ops/coordinate.py` changes were made.
- No dead code was introduced. The legacy `_bfs_distances()` helper remains in
  place for contract tests and validation comparisons, but production sampled
  metrics now use the scipy all-pairs path.

# P3C Hybrid V2 Evidence

Date: 2026-07-06
Branch: `r79/p3-hybrid`
Baseline target: `eval_output/r79_baseline/BASELINE.md` present at this branch's fork point
(`c80e970`, dated 2026-07-05). The branch code itself includes P2c layered fixes from
`3b706c5`.

## Gate Verdict

Route OFF by default.

The SCC-condensation pipeline is implemented and force-dispatchable, but the measured
default-route gate did not pass. Forced hybrid-v2 regressed the target SCC graphs and also
hit small-world graphs under the raw SCC predicate, so `_choose_native_pipeline` keeps the
auto route disabled unless the private evidence flag `_dagua_native_enable_hybrid_v2_auto`
is set. `force_pipeline="hybrid_v2"` and `algorithm="native_hybrid_v2"` remain available.

## Missing Input Docs

The requested files were absent in this worktree:

```text
.project-context/research/r79_native/r79_DESIGN.md
.project-context/research/r79_native/P2b_ELK_GAP_DOSSIER.md
```

I used the task text plus the available `.project-context/research/r79_native/P2C_EVIDENCE.md`.

## Predicate Hits

Predicate definition checked: semantically directed, cyclicity ratio > 0, SCC coverage >
25%, max nontrivial SCC size >= 10.

Targeted graphs:

| Graph | Directed | Cyclic | SCC coverage | Max SCC | Predicate |
| --- | --- | --- | ---: | ---: | --- |
| parallel_cycles_4x5 | True | True | 1.000 | 5 | False |
| small_world_100 | True | True | 1.000 | 100 | True |
| small_world_500 | True | True | 1.000 | 500 | True |
| r79_directed_scc_90_3cores | True | True | 0.333 | 12 | True |
| r79_directed_scc_120_2cores | True | True | 0.350 | 24 | True |

Whole-corpus hits:

| Graph | SCC coverage | Max SCC |
| --- | ---: | ---: |
| kitchen_sink_hybrid_net | 0.579 | 11 |
| small_world_100 | 1.000 | 100 |
| small_world_500 | 1.000 | 500 |
| small_world_2000 | 1.000 | 2000 |
| r79_weighted_small_world_120 | 1.000 | 120 |
| r79_directed_scc_90_3cores | 0.333 | 12 |
| r79_directed_scc_120_2cores | 0.350 | 24 |

This over-selects small-world classes in the current graph semantics, which is one reason
the route remains off.

## Targeted Results

Default route after evidence gating OFF:

| Graph | Before Dagua | After Dagua | Best external | After delta | Move |
| --- | ---: | ---: | ---: | ---: | ---: |
| r79_directed_scc_120_2cores | 58.381 | 58.381 | 59.883 | -1.503 | +0.000 |
| r79_directed_scc_90_3cores | 57.178 | 57.178 | 56.985 | +0.193 | +0.000 |
| parallel_cycles_4x5 | 58.383 | 58.383 | 60.410 | -2.027 | +0.000 |
| small_world_100 | 54.123 | 54.123 | 52.375 | +1.748 | +0.000 |
| small_world_500 | 50.161 | 50.161 | 48.364 | +1.797 | +0.000 |

Forced `algorithm="native_hybrid_v2"` probe:

| Graph | Incumbent Dagua | Forced hybrid_v2 | Best external | Forced delta | Move |
| --- | ---: | ---: | ---: | ---: | ---: |
| parallel_cycles_4x5 | 58.383 | 35.628 | 60.410 | -24.782 | -22.755 |
| small_world_100 | 54.123 | 28.799 | 52.375 | -23.576 | -25.324 |
| small_world_500 | 50.161 | 27.961 | 48.364 | -20.403 | -22.200 |
| r79_directed_scc_90_3cores | 57.178 | 43.382 | 56.985 | -13.603 | -13.796 |
| r79_directed_scc_120_2cores | 58.381 | 48.946 | 59.883 | -10.938 | -9.435 |

## Full Sweep

Command:

```bash
timeout 9000 .venv/bin/python scripts/r79_baseline.py --dagua-only > /tmp/r79_p3_full_off.log 2>&1
```

Result with route OFF:

| Population | W | T | L |
| --- | ---: | ---: | ---: |
| legacy | 77 | 9 | 7 |
| extended | 10 | 3 | 2 |
| total | 87 | 12 | 9 |

Gate status:

| Requirement | Result |
| --- | --- |
| `r79_directed_scc_120_2cores` improves >= 1.0 | FAIL: +0.000, remains -1.503 delta |
| No winning graph drops > 0.5 | PASS with route OFF |
| Total W does not decrease | PASS with route OFF |

## Jitter Validation

No improvements are claimed, so no jitter-stability claim is made. The forced route
regressed the target class and is disabled.

## Determinism

Two seeded runs were bit-identical:

| Route | Graph | Equal | Max abs diff |
| --- | --- | --- | ---: |
| forced native_hybrid_v2 | r79_directed_scc_120_2cores | True | 0.0 |
| default dagua_native | r79_directed_scc_120_2cores | True | 0.0 |

## Test Results

Passed:

```text
.venv/bin/python -m pytest tests/test_ops_scc.py -q
7 passed, 2 warnings in 0.12s

.venv/bin/python -m pytest tests/test_ops_scc.py tests/test_layout/test_native_topology_dispatch.py tests/test_pipeline_dagua_native.py -q
17 passed, 3 warnings in 201.66s

.venv/bin/python -m ruff check . --fix
All checks passed!

.venv/bin/python -m mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file
```

Environment/dependency failures:

```text
.venv/bin/python -m pytest tests/test_ops_scc.py tests/test_layout/test_native_topology_dispatch.py tests/test_pipeline_dagua_native.py tests/test_layout/ tests/test_graph.py -x --tb=short -q
FAILED tests/test_layout/test_cuda_activation.py::test_all_stages_fall_back_when_no_cuda
RuntimeError: The NVIDIA driver on your system is too old (found version 12040)

.venv/bin/python -m pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
FAILED tests/test_classic_competitor.py::test_each_classic_competitor_produces_a_valid_result
classic_tsnet error: No module named 'sklearn'
```

The CUDA driver warning appeared during benchmark imports:

```text
CUDA initialization: The NVIDIA driver on your system is too old (found version 12040)
```

## Notes

- `eval_output/r79_baseline/` was regenerated by measurement and must be restored before
  commit; the durable numbers are recorded here.
- The implementation imports no external layout or graph packages from `dagua/layout/`.
- The raw SCC predicate is not sufficient as an auto-route predicate under current graph
  semantics because small-world graphs are classified as semantically directed and satisfy
  the SCC thresholds.

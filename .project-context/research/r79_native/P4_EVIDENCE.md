# P4 Native Stress Multilevel Evidence

Date: 2026-07-06
Branch: `r79/p4-scale`

## Implementation Summary

- Added explicit registered pipeline `native_stress_ml`.
- Sketch gate: multilevel activates when `N >= ml_min_nodes` or `E >= ml_min_edges`.
- Coarsening: reuses `HeavyEdgeMatching`; hierarchy levels now carry aggregated
  edge weights and node masses.
- Coarsest solve: minimal PivotMDS + Stress-SGD composition. The late
  multicriteria/crossing polish is intentionally skipped in the multilevel path
  because dense coarse graphs made it dominate runtime.
- Prolongation: `DirectMapping` with small jitter, `NeighborSmoothing`, then
  short warm-start Stress-SGD refinement.
- Fine-level distance targets: pivot approximation above 1,000 active nodes.
- Repulsion approximation: spatial-hash local nudges below 100K active nodes;
  negative-sampling nudges above that in auto mode.
- Final overlap projection: direct projection only at or below 5K nodes; larger
  levels log and skip the O(N^2) projection.
- Memory discipline: each O(N)/O(E) stage estimates peak bytes and aborts if the
  estimate exceeds 70% of available RAM.

## Test Results

Passing:

```text
.venv/bin/python -m ruff check . --fix
All checks passed!

.venv/bin/python -m mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file

timeout 300 .venv/bin/python -m pytest tests/test_pipeline_native_stress_ml.py tests/test_ops_coarsen.py tests/test_ops_prolong.py -x --tb=short -q
51 passed, 1 warning

timeout 300 .venv/bin/python -m pytest tests/test_graph.py -x --tb=short -q
36 passed, 1 skipped, 1 warning

timeout 300 .venv/bin/python -m pytest tests/test_pipeline_registry.py -x --tb=short -q
38 passed, 3 warnings
```

Timed out / partial:

```text
timeout 300 .venv/bin/python -m pytest tests/test_layout/test_spatial_hash_losses.py tests/test_layout/test_resolve_aspect_policy.py tests/test_layout/test_engine.py -x --tb=short -q
Timed out after 300s.

timeout 180 .venv/bin/python -m pytest tests/test_layout/test_spatial_hash_losses.py tests/test_layout/test_resolve_aspect_policy.py -x --tb=short -q
Timed out after 180s with no failure output.
```

Baseline default-path gate:

```text
timeout 10800 .venv/bin/python scripts/r79_baseline.py --dagua-only
Wrote eval_output/r79_baseline/results.json
Wrote eval_output/r79_baseline/BASELINE.md
Wall time: 1175.645s
```

The baseline command rewrote generated baseline artifacts as designed by the
script. Those generated files were restored after the run; no baseline artifact
is part of this commit. Because this P4 change only registers an explicit
`native_stress_ml` pipeline and does not route the default path to it, no
positions-level default-path change is expected below the gate.

## Scale Results

Artifacts:

- `r79_scale_20k_smoke.json`
- `r79_scale_sparse_100k.json`

All runs used `--steps 2` to keep the evidence pass bounded. Default rows used
the current auto route through the eval harness; ML rows force-dispatched
`native_stress_ml`.

| N | graph | engine | result | wall s | peak RSS GB | composite_large |
|---:|---|---|---|---:|---:|---:|
| 20,000 | sparse_er | default | timeout | 90.54 | 3.84 | n/a |
| 20,000 | sparse_er | native_stress_ml | ok | 70.02 | 0.66 | 28.23 |
| 20,000 | scale_free_ba | default | RecursionError | 75.83 | 0.73 | n/a |
| 20,000 | scale_free_ba | native_stress_ml | timeout | 90.26 | 1.98 | n/a |
| 20,000 | grid_2d | default | timeout | 90.30 | 13.51 | n/a |
| 20,000 | grid_2d | native_stress_ml | ok | 46.81 | 0.51 | 20.87 |
| 100,000 | sparse_er | native_stress_ml | ok | 66.04 | 2.50 | 28.32 |
| 1,000,000 | sparse_er | native_stress_ml | aborted | >650 | ~1.15 at abort | n/a |

The 1M sparse run was stopped after it exceeded the 10-minute target and was
still in the coarsest solve:

```text
native_stress_ml: heavy-edge coarsen (N=1000000, E=2000000)
native_stress_ml: coarsest native-stress solve (N=42581, E=1037874)
```

## Target Verdicts

- 100K sparse `< 60s`: missed. Observed 66.04s with 2.50GB peak RSS.
- 100K quality `>= default`: inconclusive. Default auto route did not complete
  within the 90s evidence cap at 20K sparse; a 100K default comparison was not
  run because default already exceeded the bounded evidence budget.
- 1M `< 10 min` and `< 32GB`: missed on time. Memory was well under 32GB during
  the observed run, but the solve remained at the 42,581-node coarse level after
  the 10-minute target window.
- 20K quality within noise of plain stress: inconclusive. The 20K evidence shows
  ML can complete sparse/grid rungs, but the quick composite is low because
  large overlap projection is skipped above 5K and the ladder graph depth metric
  is neutral for undirected generated graphs.
- `<=500` default output unchanged: expected by construction and partially
  guarded by the below-gate equality test. The `--dagua-only` sweep completed;
  generated baseline artifacts were restored after the run.

## Bottlenecks

- Heavy-edge coarsening did not reach the intended ~1K coarsest size at 1M
  sparse ER; it stopped at 42,581 nodes and 1,037,874 coarse edges. The coarsest
  PivotMDS/Stress-SGD stage then dominated and missed the 10-minute target.
- BA-style graphs coarsen more slowly and timed out at 20K with the bounded
  smoke settings.
- Direct final overlap projection is not viable above small node counts; the
  threshold was set to 5K to avoid O(N^2) projection at scale.
- The current quick-tier composite is not a good quality signal for these
  semantically undirected generated scale graphs because `depth_spearman_rho`
  is neutral/constant and overlap is skipped at large N.

## Recommendation

Do not auto-route default layouts to `native_stress_ml` yet. Keep it explicit
behind `algorithm="native_stress_ml"` until coarsening reaches the ~1K target
reliably and the 1M sparse target completes inside 10 minutes. The current
evidence supports using it only as an opt-in experimental scale path for sparse
non-BA graphs around 20K-100K when the caller accepts skipped large overlap
projection.

## Round 2 - Aggressive Contraction And Reference Robustness

Date: 2026-07-06

Implementation deltas:

- Replaced the ML path's plain heavy-edge hierarchy with registered
  `AggressiveHybridCoarsen`. It keeps HEM when a level shrinks by at least 50%,
  escalates to deterministic hub-star contraction when progress degrades, and
  uses a final bucket contraction as a target cap guardrail.
- Added registered `SampledCoarsestSolve` fallback for oversized coarsest
  graphs: select deterministic hub-and-stride pivots, solve the sampled induced
  graph, then place unsampled nodes by weighted-neighbor interpolation with a
  deterministic grid fallback.
- Hardened `scripts/r79_scale_eval.py` so Graphviz reference rows cannot crash
  the ladder: `sfdp -Tplain` missing nodes are filled from neighbor centroids
  when possible, out-of-range/malformed node rows become warnings, reference
  exceptions become `status=ERROR` rows, and skipped references use
  `status=SKIP`.
- Switched the eval worker process context to `spawn`. A clean rerun exposed a
  fork-after-Torch/SciPy/Graphviz deadlock on the second ML row; `spawn` avoided
  inheriting native-library thread state and let the ladder complete.
- DOT reference input now declares all `num_nodes`, including isolated trailing
  nodes, instead of only declaring nodes present in edges.

Artifacts:

- `.project-context/research/r79_native/r79_scale_ladder_round2.json`

Command:

```text
.venv/bin/python scripts/r79_scale_eval.py --output .project-context/research/r79_native/r79_scale_ladder_round2.json --graph-types sparse_er,scale_free_ba,grid_2d --engines native_stress_ml --include-sfdp --steps 20 --engine-timeout 900
```

Graphviz reference version:

```text
sfdp - graphviz version 7.0.5 (20221231.0122)
```

Scale ladder:

| N | graph | engine | status | wall s | peak RSS GB | composite_large |
|---:|---|---|---|---:|---:|---:|
| 20,000 | sparse_er | native_stress_ml | ok | 122.46 | 0.84 | 28.23 |
| 20,000 | sparse_er | graphviz_sfdp | OK | 62.05 | n/a | 31.53 |
| 20,000 | scale_free_ba | native_stress_ml | ok | 108.42 | 5.70 | 18.99 |
| 20,000 | scale_free_ba | graphviz_sfdp | OK | 54.34 | n/a | 32.03 |
| 20,000 | grid_2d | native_stress_ml | ok | 336.72 | 0.77 | 21.00 |
| 20,000 | grid_2d | graphviz_sfdp | OK | 37.82 | n/a | 29.83 |
| 100,000 | sparse_er | native_stress_ml | ok | 289.79 | 1.60 | 28.32 |
| 100,000 | sparse_er | graphviz_sfdp | OK | 351.12 | n/a | 31.59 |
| 100,000 | scale_free_ba | native_stress_ml | ok | 333.13 | 19.83 | 19.92 |
| 100,000 | scale_free_ba | graphviz_sfdp | OK | 313.57 | n/a | 31.94 |
| 100,000 | grid_2d | native_stress_ml | timeout | 900.31 | 0.96 | n/a |
| 100,000 | grid_2d | graphviz_sfdp | OK | 169.16 | n/a | 40.38 |
| 1,000,000 | sparse_er | native_stress_ml | error | 139.67 | 77.01 | n/a |
| 1,000,000 | scale_free_ba | native_stress_ml | timeout | 900.28 | 2.63 | n/a |
| 1,000,000 | grid_2d | native_stress_ml | timeout | 900.29 | 2.29 | n/a |

The 100K grid ML row logged completion of layout and entry into quick metrics
before the watchdog timeout. The row is still reported as timeout because the
current ladder measures layout plus quick-tier scoring in the child process.

Contraction profiles:

| graph | N | levels |
|---|---:|---|
| sparse_er | 1,000,000 | 1,000,000 -> 346,554 -> 145,784 -> 115,379 -> 78,107 -> 19,001 -> 2,376 -> 297 |
| scale_free_ba | 1,000,000 | 1,000,000 -> 365,966 -> 181,063 -> 10,675 -> 1 |
| grid_2d | 1,000,000 | 1,000,000 -> 500,000 -> 124,753 -> 41,420 -> 10,898 -> 2,569 -> 623 |

Target verdicts:

- Coarsest `<= 1000` on all 1M graph types: met by contraction profile.
- 100K sparse `< 60s`: missed. Observed 289.79s including quick metrics.
- 1M sparse `< 10 min` and `< 32GB`: missed. The row failed with no child
  result after 139.67s and peak RSS 77.01GB.
- 20K scale_free_ba `< 60s`: missed. Observed 108.42s.
- Quality within 10% of SFDP at 20K/100K: missed on all completed comparisons
  except 100K sparse is close but still ~10.4% lower by quick-tier
  `composite_large` (28.32 vs 31.59). 20K sparse is ~10.5% lower, BA/grid are
  much lower. These comparisons use `composite_large`, not full composite.
- Reference coverage: all requested 20K and 100K SFDP rows completed with
  `status=OK` in the clean rerun. The missing-node parser bug from the dead run
  is covered by tests and would now produce `status=WARN` rather than crashing.

Notes:

- Overlap counts remain high at these scales because per-node overlap
  resolution is intentionally skipped above `overlap_max_nodes`; this round did
  not chase those counts.
- The default-path 20K BA `RecursionError` was not trivially locatable in a
  direct repro attempt; it ran for more than 90s without producing a traceback
  and was stopped. No default-path fix was attempted.
- Do not auto-route to `native_stress_ml` yet. Stronger contraction fixes the
  coarsest-size stall, but refinement/metric time, BA quality, and 1M sparse
  memory remain blockers.

Verification:

```text
.venv/bin/python -m ruff check . --fix
All checks passed!

.venv/bin/python -m mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file

.venv/bin/python -m pytest tests/test_r79_scale_eval.py tests/test_ops_coarsen.py tests/test_pipeline_native_stress_ml.py -x --tb=short -q
40 passed, 1 warning in 4.32s

.venv/bin/python -m pytest tests/test_graph.py -x --tb=short -q
36 passed, 1 skipped, 1 warning in 0.78s
```

Broader targeted gate attempted:

```text
.venv/bin/python -m pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
FAILED tests/test_layout/test_cuda_activation.py::test_all_stages_fall_back_when_no_cuda
RuntimeError: The NVIDIA driver on your system is too old (found version 12040).
63 passed, 9 skipped before the failure; elapsed 1891.87s.
```

This failure is outside the P4 contraction/eval scope and reproduces a host CUDA
fallback issue in `dagua/layout/engine.py` when `_layout_inner` receives
`device="cuda"` on a runtime where `torch.cuda.is_available()` is false.

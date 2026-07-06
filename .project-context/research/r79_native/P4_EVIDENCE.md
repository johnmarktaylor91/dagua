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

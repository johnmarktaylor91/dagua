# Sprint-41 No-Fix Report

Date: 2026-04-26
Branch: codex/sprint-31a-gate-refinement
Baseline: git HEAD 3eaa01c

## Outcome

No fix shipped. The active `dagua_native` loss path already routes the
default node-local O(N^2) losses through `dagua/layout/ops/spatial_hash.py`:

- `RepulsionLoss` uses `cell_list_candidate_pairs()` above the exact threshold.
- `OverlapAvoidanceLoss` uses `cell_list_candidate_pairs()` above the exact threshold.
- `CrossingLoss` is not a node-neighborhood term; it is bounded by `max_pairs`
  and layer-grouped sampling.
- `FanoutDistributionLoss` sorts incident edges by hub and evaluates angular
  gaps within each hub, not all node pairs.

An exploratory finite-cutoff GraphOpt repulsion change was implemented and then
reverted after validation. It correctly routed a classic force op through the
spatial hash, but the representative validation set uses the default native
pipeline, so it did not move the Sprint-41 runtime gate.

## Pairwise Site Review

Sites found during the sweep:

- `dagua/layout/ops/loss_engine.py`
  - `_exact_repulsion_loss`, `_exact_overlap_loss`: exact fallback below 500
    nodes; spatial-hash path already exists for larger graphs.
  - `_cell_list_repulsion_loss`, `_cell_list_overlap_loss`: already use
    `cell_list_candidate_pairs()`.
- `dagua/layout/constraints.py`
  - Legacy engine helpers retain exact small-graph paths, sampled/RVS paths, and
    an older grid overlap helper. The file is used by the opt-in legacy body;
    new default optimization work belongs in registered ops.
- `dagua/layout/ops/pipelines/dagua_native.py`
  - `_overlap_jitter` is explicitly bounded to `max_nodes=500`.
  - `_crossing_edge_pairs` is bounded to small graphs and `max_pairs=512`.
  - Profiling showed `_dot_lattice_lp` and polish metric scoring dominate
    `real_football_115`; this is LP/metric selection work, not a
    spatial-hashable local loss.
- `dagua/layout/edge_optimization.py`
  - Edge crossing and rectilinear crossing losses are edge-pair terms with
    existing sampling/bounds; edge-node proximity is sampled for large N and is
    outside the default layout path.
- Classic/specialist ops (`force.py`, `loss_classic.py`, `fmmm.py`, `sfdp.py`)
  contain global repulsion objectives. Most are true long-range energy terms or
  already have algorithm-specific approximation gates; changing them would not
  affect the default validation set.

## Validation

Validation script: `/tmp/sprint41_validate.py`

Result: failed speed gate.

| graph | HEAD | candidate | speedup | max_abs_diff | composite delta |
|---|---:|---:|---:|---:|---:|
| er_500 | 15.691s | 15.028s | 1.04x | 0 | +0.000 |
| rgg_100 | 9.127s | 9.282s | 0.98x | 0 | +0.000 |
| dependency_graph_100 | 2.144s | 2.424s | 0.88x | 0 | +0.000 |
| real_lesmis_77 | 2.057s | 1.989s | 1.03x | 0 | +0.000 |
| real_football_115 | 18.502s | 19.037s | 0.97x | 0 | +0.000 |
| small_world_100 | 1.652s | 1.581s | 1.05x | 0 | +0.000 |
| random_dag_200 | 3.193s | 3.031s | 1.05x | 163890 | -0.093 |
| dense_pair_50 | 1.277s | 1.147s | 1.11x | 0 | +0.000 |

Aggregate speedup: `1.002x`

Graphs at >=1.3x: `0/8`

Correctness/quality gate: pass by composite tolerance.

Speed gate: fail.

## Root Cause

The remaining runtime on the representative set is not primarily an un-hashed
node-pair loss. A `real_football_115` cProfile run showed:

- `_best_of_polish`: 19.591s cumulative
- `_dot_lattice_lp`: 17.251s cumulative
- SciPy `linprog`: 12.866s cumulative
- repeated `dagua.metrics.full()` scoring: 2.168s cumulative

Those paths are polish candidate selection and LP solving. They are outside
the Sprint-41 spatial-hash target and changing them would risk repeating the
metric-gaming pattern documented in the 2026-04-26 retro.

## Recommendation

Do not force more default losses through the spatial hash below the current
500-node threshold. A microbenchmark showed the current Python-level
`UniformSpatialHash` is slower than dense exact tensors for N=50..500, so
lowering the threshold would likely regress runtime.

Future runtime work should target the actual observed bottlenecks:

- Gate or cache `_dot_lattice_lp` polish candidates by structural class and
  graph size.
- Reduce repeated metric scoring inside `_best_of_polish` without changing the
  candidate acceptance semantics.
- Optimize `UniformSpatialHash.candidate_pairs()` itself if larger-graph
  profiles still show it as hot.

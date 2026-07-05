# r78-S2 Superior-Distinct Exhaustion Notes

Date: 2026-07-05
Branch: `r78/superior`
Scope: research and disposition notes for the 41 r77 official `SUPERIOR_DISTINCT` rows.

Inputs:
- `/home/jtaylor/projects/dagua/eval_output/fidelity_definitive_r77/OFFICIAL_R77_LEDGER.{md,json}`
- `/home/jtaylor/projects/dagua/eval_output/fidelity_definitive/per_combo_r77.jsonl`
- `.project-context/research/sprint_rng_matching/r75_findings/r76_PROBE_fairness.md`
- `/home/jtaylor/projects/dagua/.project-context/research/sprint_rng_matching/r75_findings/r78_RESIDUAL_MOP.md`
- Fresh probes in this worktree:
  - `/tmp/r78_mds_mchain.json`
  - `scripts/rng_match/check_engine.py classic_gem_iters100 --seeds 1,2,3`
  - `scripts/rng_match/check_engine.py classic_gem_iters500 --seeds 1,2,3`
  - `scripts/rng_match/check_engine.py classic_fmmm_steps100 --seeds 1,2,3`

Assumption: the complete r77 ledger and `per_combo_r77.jsonl` are in the main checkout
under `/home/jtaylor/projects/dagua`. This worktree only has a partial r77 report copy.
I treated the main checkout artifacts as read-only inputs and made no eval/scoring/runner
changes.

## Row Set

Official r77 `SUPERIOR_DISTINCT` rows:

| Family | Rows | Engines |
| --- | ---: | --- |
| `gem` | 24 | `classic_gem_iters100`, `classic_gem_iters500`, `classic_gem_iters2000` |
| `fmmm` | 9 | `classic_fmmm_graphviz_fdp_fidelity` |
| `classical_mds` | 8 | `classic_classical_mds_default`, `classic_classical_mds_igraph_fidelity` |

Important correction: the raw `quality_superior_distinct` flag in `per_combo_r77.jsonl`
is stale for many Sugiyama/SFDP rows. The official ledger disposition layer reduces the
actual terminal set to these 41 rows.

## GEM Cluster: 24 Rows

Rows:

| Graphs | Variants | Ledger flags |
| --- | --- | --- |
| `ba_500`, `dependency_500`, `er_500`, `grid_20x20`, `powerlaw_500`, `rgg_500`, `sbm_8x100`, `small_world_500` | `iters100`, `iters500`, `iters2000` | `LOW_POWER_N_42_SEEDS_100-141` |

Representative ledger metrics:

| Combo | Stress D/R | Cross D/R | W D/R |
| --- | ---: | ---: | ---: |
| `ba_500::classic_gem_iters100` | 0.183729 / 0.223846 | 80519.0 / 254297.4 | 1.277179 / 1.377390 |
| `grid_20x20::classic_gem_iters100` | 0.281333 / 0.346559 | 4302.7 / 65767.4 | 1.213943 / 1.372953 |
| `rgg_500::classic_gem_iters100` | 0.241496 / 0.320833 | 120747.0 / 1391827.3 | 1.171229 / 1.377390 |

Fresh fixture bisection/sanity:

| Probe | Result |
| --- | --- |
| `classic_gem_iters100`, seeds 1,2,3, 14 fixtures | max RMSD `7.965414923e-08` vs rebuilt OGDF runner |
| `classic_gem_iters500`, seeds 1,2,3, 14 fixtures | max RMSD `8.246905089e-08` vs rebuilt OGDF runner |

Interpretation:

- The standard OGDF-GEM fidelity path remains matched to the rebuilt runner on small
  fixtures at the known float boundary, with no order/RNG/nameable operation mismatch
  exposed by the fixture harness.
- The 24 r77 rows are all low-power 42-seed large-graph rows. A direct same-process
  probe on `ba_500` and `grid_20x20` was attempted, but one paired probe exceeded the
  interactive task budget and was stopped after more than three minutes without producing
  JSON.
- No code port was made. I did not find a nameable, safely portable GEM operation beyond
  the already-ported r76 round-budget/RNG path. The remaining evidence supports a
  low-power large-graph basin-selection residual rather than a specific unported GEM op,
  but this is weaker than a first-update C trace.

Disposition:

- Not converted.
- Evidence status: fixture-grade matched path plus low-power large-graph residual.
- Residual concern: the requested first-diverging-update trace for one 500-node row is
  not present. A stronger close would require a bounded temporary OGDF GEM trace runner
  for `ba_500` seed 100 that dumps initialization, permutation, and first update values.

## FMMM-Labeled Graphviz FDP Cluster: 9 Rows

Rows:

| Graph | Ledger flags | Stress D/R | Cross D/R | W D/R |
| --- | --- | ---: | ---: | ---: |
| `ba_500` | `SEED_42_ERA_42-141` | 0.167819 / 0.204328 | 72215.2 / 82286.0 | 1.025562 / 1.203570 |
| `er_500` | `SEED_42_ERA_42-141` | 0.148228 / 0.160788 | 15268.5 / 17386.4 | 1.018037 / 1.160237 |
| `extreme_mixed_width_transformer` | `SEED_42_ERA_42-141` | 0.089381 / 0.123679 | 1.883 / 1.500 | 0.602224 / 0.666397 |
| `grid_20x20` | `SEED_42_ERA_42-141` | 0.010550 / 0.114763 | 0.0 / 308.6 | 0.013548 / 0.781107 |
| `hub_spoke_10x20` | `SEED_42_ERA_42-141` | 0.198706 / 0.247441 | 5339.7 / 6918.5 | 1.229656 / 1.275491 |
| `hub_spoke_5x50` | `LOW_POWER_N_42_SEEDS_100-141` | 0.190967 / 0.256011 | 7171.8 / 9393.0 | 1.166680 / 1.214577 |
| `powerlaw_500` | `SEED_42_ERA_42-141` | 0.154718 / 0.167662 | 16960.5 / 19335.2 | 0.920232 / 1.160843 |
| `rgg_500` | `SEED_42_ERA_42-141` | 0.005727 / 0.050105 | 16468.5 / 21305.4 | 0.025912 / 0.561510 |
| `small_world_500` | `SEED_42_ERA_42-141` | 0.014079 / 0.323911 | 753.8 / 1660.3 | 0.020368 / 1.150034 |

Important family correction:

- These rows are not OGDF FMMM. The `classic_fmmm_graphviz_fdp_fidelity` adapter sets
  `fidelity_mode="graphviz_fdp"` and compares against `graphviz_fdp`.
- The standard OGDF-FMMM fixture probe is separate: `classic_fmmm_steps100` is
  byte-identical on 35/42 fixture seeds and has max RMSD `2.549335867e-02` on the known
  small residual cluster. That probe does not explain these nine Graphviz-FDP rows.

Stage evidence from `r78_RESIDUAL_MOP.md`:

- Pinned Graphviz 7.0.5 FDP stage order: `findCComp -> fdp_tLayout -> fdp_xLayout ->
  putGraphs -> finalCC -> evalPositions`.
- For connected FDP rows probed there, Dagua's `_graphviz_fdp_component_layout`
  already differed from installed `fdp` at the `fdp_tLayout`/`fdp_xLayout` boundary,
  before packing.
- The named follow-up is a Graphviz C dump of `lib/fdpgen/tlayout.c:initPositions`, both
  force loops in `fdp_tLayout`, and `lib/fdpgen/xlayout.c:fdp_xLayout`.

Disposition:

- Not converted.
- Evidence status: named portable residual, not terminal. The worse reference behavior is
  Graphviz FDP's exact random/init, force/grid, and xLayout path.
- No code port was made because the current evidence points to a medium/large exact
  Graphviz-FDP C-port, not a small gated degradation. Calling this non-nameable would be
  inaccurate.

## Classical MDS Cluster: 8 Rows

Rows:

| Combo | Stress D/R | Cross D/R |
| --- | ---: | ---: |
| `center_port_backedge_hub::classic_classical_mds_default` | 0.201607 / 0.345503 | 0 / 2 |
| `densenet_block::classic_classical_mds_default` | 0.127624 / 0.148737 | 30 / 30 |
| `densenet_block::classic_classical_mds_igraph_fidelity` | 0.129108 / 0.148737 | 30 / 30 |
| `edge_label_braid::classic_classical_mds_igraph_fidelity` | 0.081745 / 0.086842 | 2 / 2 |
| `inception_block::classic_classical_mds_igraph_fidelity` | 0.143130 / 0.153452 | 0 / 0 |
| `petersen_10::classic_classical_mds_default` | 0.123513 / 0.160484 | 5 / 5 |
| `wide_single_layer_1_50_1::classic_classical_mds_default` | 0.452996 / 0.674931 | 0 / 5 |
| `wide_single_layer_1_50_1::classic_classical_mds_igraph_fidelity` | 0.599874 / 0.674931 | 0 / 5 |

Fresh M-chain probe (`/tmp/r78_mds_mchain.json`):

| Combo | Positive eigenspace dim | lambda2-lambda3 gap ULP | D projection residual | R projection residual | In-eigenspace transform RMSD | Direct 2D RMSD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `center_port_backedge_hub::classic_classical_mds_default` | 4 | 0 | 1.39e-09 | 1.14e-08 | 1.58e-01 | 3.96e-01 |
| `densenet_block::classic_classical_mds_default` | 6 | 0 | 5.51e-09 | 3.70e-09 | 2.58e-09 | 1.69e-01 |
| `densenet_block::classic_classical_mds_igraph_fidelity` | 6 | 0 | 3.49e-09 | 3.70e-09 | 2.56e-09 | 1.59e-01 |
| `edge_label_braid::classic_classical_mds_igraph_fidelity` | 3 | 24 | 1.33e-08 | 1.03e-08 | 5.19e-09 | 3.19e-01 |
| `inception_block::classic_classical_mds_igraph_fidelity` | 4 | 0 | 1.98e-08 | 1.96e-08 | 8.91e-09 | 2.04e-01 |
| `petersen_10::classic_classical_mds_default` | 5 | 2 | 9.78e-09 | 1.90e-08 | 7.63e-09 | 2.57e-01 |
| `wide_single_layer_1_50_1::classic_classical_mds_default` | 50 | 2 | 1.54e-09 | 6.83e-10 | 2.33e-10 | 1.94e-01 |
| `wide_single_layer_1_50_1::classic_classical_mds_igraph_fidelity` | 50 | 2 | 0.00e+00 | 6.83e-10 | 1.39e-01 | 1.39e-01 |

Interpretation:

- Every row's Dagua and reference coordinates lie in the same positive classical-MDS
  eigenspace to about `2e-08` or better.
- Six rows also map through an orthogonal transform inside that eigenspace to near-zero
  residual (`<= 9e-09`), while the direct registered 2D distance remains large. That is
  the exact eigenspace-member pattern from the r76/r77 MDS dossiers.
- `center_port_backedge_hub` and the `wide_single_layer_1_50_1` igraph-fidelity row still
  project into the same eigenspace, but the strict orthogonal coefficient transform did
  not collapse. This is consistent with igraph's post-align/axis conventions layered on
  top of a highly degenerate eigenspace; it is still not a nameable graph operation.
- Matching the exact worse reference member would require reproducing igraph/LAPACK's
  arbitrary basis and sign choices. That remains the excluded LAPACK-vendoring class.

Disposition:

- Terminal equivalence-class proof. No port.
- The reference is a worse member of the same classical-MDS eigenspace class.

## Gates And Commands

Commands completed:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/rng_match/check_engine.py classic_gem_iters100 --seeds 1,2,3
# max RMSD: 7.965414923e-08

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/rng_match/check_engine.py classic_gem_iters500 --seeds 1,2,3
# max RMSD: 8.246905089e-08

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/rng_match/check_engine.py classic_fmmm_steps100 --seeds 1,2,3
# max RMSD: 2.549335867e-02; exact 35/42

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python - <<'PY'
# MDS M-chain probe over the 8 official r77 MDS superior rows.
# Wrote /tmp/r78_mds_mchain.json.
PY
```

Commands attempted and stopped:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python - <<'PY'
# Direct same-process adapter probe for ba_500/grid_20x20 GEM and grid_20x20/FDP rows.
# Stopped after the ba_500 paired probe exceeded the interactive budget.
PY

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python - <<'PY'
# Direct Graphviz-FDP probe on grid_20x20/extreme_mixed_width_transformer.
# Stopped after the grid_20x20 run exceeded the interactive budget.
PY
```

Final quality gates for this notes-only change are recorded in the task summary.

## Conclusions

- MDS: all 8 rows have terminal eigenspace/eigensign evidence. Matching would require the
  excluded igraph/LAPACK basis-selection behavior.
- GEM: the standard OGDF GEM path is still fixture-matched to the rebuilt runner. The 24
  rows remain low-power large-graph superior rows without a new nameable op found here.
  This is not as strong as the requested first-update trace.
- FMMM-labeled FDP: all 9 rows are Graphviz-FDP, not OGDF-FMMM. The residual is named and
  portable (`fdp_tLayout`/`fdp_xLayout` exactness), so these rows are not terminal proofs
  and were not converted in this pass.

## Dead Code Hygiene

No code was refactored, so no newly unreachable code was identified.

## Concerns

- This pass does not satisfy the strongest form of r78-S2 for GEM/FDP: no port was made,
  and the FDP rows remain named portable residuals.
- A complete follow-up should either:
  - implement the exact Graphviz FDP `tLayout`/`xLayout` fidelity path behind
    `fidelity_mode="graphviz_fdp"`, or
  - run the requested C-side Graphviz dump and prove that matching requires an excluded
    measure.
- For GEM, the missing artifact is a bounded temporary OGDF runner trace for one 500-node
  row (`ba_500` seed 100 is the natural representative).

## G2: gem ba_500 trace

Scope:

- Representative row: `ba_500::classic_gem_iters100`, seed `100`.
- Graph source: `make_scale_free(n=500, m=3, seed=42)`, exported through the same
  `_graph_edges()` path used by the OGDF competitor adapter.
- Input size: 500 nodes, 1494 oriented edges. GEM consumes the edges as undirected
  adjacency entries internally.
- Runner: temporary copy under `/tmp/gem-trace` only. The committed
  `scripts/ogdf_runner.cpp` and `/home/jtaylor/tools/ogdf-src` were not modified.

Trace method:

- The temporary runner mirrored OGDF `GEMLayout::call()` directly: same
  `GraphCopy`, `connectedComponents`, `SList::permute`, `std::minstd_rand(randomSeed())`,
  disturbance `uniform_int_distribution`, `computeImpulse`, and `updateNode` sequence.
- The Dagua-side mirror used the current OGDF-fidelity GEM helpers in the same process:
  runner-style glibc initial positions, `_ogdf_gem_rng_seed(100)`,
  `_ogdf_permutation`, `_build_ogdf_adjacency`, default OGDF node dimensions, and the
  scalar per-node update loop.
- Per-update JSONL fields included node id, RNG seed, RNG draw counts, permutation draws,
  disturbance draws, position before/after, raw impulse, scaled move, local/global
  temperature, barycenter, desired length, and node weight.

Trace results:

| Budget | Compared updates | First divergence | Final coordinate check |
| --- | ---: | --- | --- |
| `gemRounds=100` | 100 / 100 | none | raw RMSD `7.105427357601002e-16`; max abs `1.4210854715202004e-14` |
| `gemRounds=2000` extension | 2000 / 2000 | update 115, `raw_impulse` last-bit split | raw RMSD `1.1184606172532277e-11`; max abs `2.5362112410221016e-10` |

Complete iters100 trace excerpt:

| Updates | Node order | RNG/draw counts | Position/update fields |
| ---: | --- | --- | --- |
| 0-99 | identical | identical | identical within printed double precision |

First strict split in the 2000-update extension:

| Field | OGDF trace | Dagua trace |
| --- | ---: | ---: |
| update | 115 | 115 |
| node | 15 | 15 |
| RNG seed | 281717827 | 281717827 |
| draw count before disturbance | 730 | 730 |
| position before | `[4.6, 14.8]` | `[4.6, 14.8]` |
| barycenter before | `[83485.69262219609, 86262.35519126788]` | `[83485.69262219609, 86262.35519126788]` |
| raw impulse | `[-10314.740060455928, -11445.89956158023]` | `[-10314.740060455919, -11445.899561580223]` |
| scaled move | `[-8.033354069012047, -8.914326805871971]` | `[-8.033354069012047, -8.914326805871971]` |
| position after | `[-3.433354069012047, 5.88567319412803]` | `[-3.433354069012047, 5.88567319412803]` |
| local temperature before/after | `12` / `12` | `12.0` / `12.0` |
| global temperature before/after | `12` / `12` | `12.0` / `12.0` |

Surrounding split diagnostics:

| Update | Node | RNG same | Max raw-impulse delta | Max move delta | Max position-after delta |
| ---: | ---: | --- | ---: | ---: | ---: |
| 110 | 143 | yes | `9.094947017729282e-13` | `1.7763568394002505e-15` | `3.552713678800501e-15` |
| 111 | 442 | yes | `0.0` | `1.7763568394002505e-15` | `1.7763568394002505e-15` |
| 112 | 276 | yes | `0.0` | `1.7763568394002505e-15` | `0.0` |
| 113 | 472 | yes | `2.2737367544323206e-13` | `8.881784197001252e-16` | `0.0` |
| 114 | 487 | yes | `0.0` | `1.7763568394002505e-15` | `1.7763568394002505e-15` |
| 115 | 15 | yes | `9.094947017729282e-12` | `0.0` | `0.0` |
| 116 | 382 | yes | `0.0` | `1.7763568394002505e-15` | `1.4210854715202004e-14` |

Verdict:

- No nameable GEM operation mismatch was found for the representative 500-node row.
- The complete `iters100` run has no first-diverging update: the traced updates and final
  packed coordinates match current Dagua to double-roundoff noise.
- The first strict split found by extending the same trace to `gemRounds=2000` is a
  raw-impulse last-bit arithmetic difference at update 115. Node order, RNG seed, RNG draw
  counts, disturbance draws, positions, barycenter, local temperature, global temperature,
  desired length, and node weight are identical at that point.
- The split is consistent with accumulated floating-point arithmetic in the force summation,
  not a portable missing operation or tie-break. The current Dagua GEM fidelity path is
  already instrument-matched for this row class.
- No port was made and no commit was created for GEM code.

Gate evidence:

```bash
/tmp/gem-trace/ogdf_runner_trace \
  --input /tmp/gem-trace/ba_500_seed100_gem100.json \
  --trace-output /tmp/gem-trace/ogdf_trace.jsonl \
  --trace-limit 100 \
  --output /tmp/gem-trace/ogdf_positions.json
# completed

# Dagua mirror trace over the same 100 updates:
# all 100 matched

/tmp/gem-trace/ogdf_runner_trace \
  --input /tmp/gem-trace/ba_500_seed100_gem100.json \
  --gem-rounds 2000 \
  --trace-output /tmp/gem-trace/ogdf_trace_2000.jsonl \
  --trace-limit 2000 \
  --output /tmp/gem-trace/ogdf_positions_2000.json
# first strict split: update 115 raw_impulse

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check . --fix
# All checks passed!

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl mypy --follow-imports=silent dagua/cli.py
# pyproject.toml: note: unused section(s): module = ['dagua.layout.multilevel']
# Success: no issues found in 1 source file

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest \
  tests/test_layout/test_gem_fidelity.py tests/test_pipeline_gem.py -x --tb=short -q
# 19 passed, 3 warnings in 2.57s

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -x --tb=short -q \
  -m "not slow and not benchmark and not rare"
# stopped at known pre-existing double-border smoke:
# FAILED tests/test_cosmetic_node_features.py::TestRenderSmoke::test_render_with_double_border
# 1 failed, 266 passed, 88 deselected, 1 xfailed, 63 warnings in 194.17s
```

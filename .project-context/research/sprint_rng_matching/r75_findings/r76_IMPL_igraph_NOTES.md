# r76 igraph Sugiyama implementation notes

Date: 2026-07-04
Worktree: `/home/jtaylor/.claude/worktrees/dagua-mincross2`
Branch: `r76/mincross`

## Summary

Implemented the portable igraph-mode Brandes-Koepf Type 1 conflict quirk.
Igraph 1.0.0 gathers outgoing neighbors for each layer, but then checks
conflicts by ordinal edge id via `IGRAPH_FROM(graph, j)` / `IGRAPH_TO(graph, j)`.
Dagua was using the canonical BK layer-boundary conflict scan. The new path is
enabled only for `fidelity_mode="igraph"`.

The LP tie class is still blocked. Installed `python-igraph` is deterministic,
but its all-zero-objective rank LP selects a GLPK simplex basis that is not
reproduced by SciPy HiGHS, HiGHS-DS, HiGHS-IPM, legacy simplex, revised simplex,
or interior-point. No GLPK Python binding is installed (`swiglpk`, `glpk`,
`cvxopt`, and `pulp` were unavailable). I did not vendor GLPK.

## Bisection

Seed: 42. Reference: installed `python-igraph` 1.0.0.

| Row class | Probe | First divergent stage | Evidence |
|---|---|---|---|
| Lattice/x-stage | `hexagonal_lattice_42::classic_sugiyama_default` | BK x assignment | Ranks and original-node layer orders matched; x differed. LP objective all-zero. |
| Social/rank-stage | `real_karate_34::classic_sugiyama_default` | LP rank assignment | Node 25 rank differed (`1` vs `3`); objective all-zero; Dagua final ranks matched SciPy LP output. |
| Layered DAG/x-stage | `width_skew_late_merge::classic_sugiyama_default` | BK x assignment | Ranks and original-node layer orders matched; x differed. |
| Close row/x-stage | `multiscale_skip_cascade::classic_sugiyama_default` | BK x assignment | Ranks and original-node layer orders matched; x differed before the quirk. |
| Disconnected/x-stage | `kitchen_sink_platform_graph::classic_sugiyama_default` | BK x assignment | Ranks and original-node layer orders matched; x differed; objective nonzero with 2 feedback edges. |
| Small rank-stage | `moe_router_sparse::classic_sugiyama_default` | LP rank assignment | Node 5 rank differed (`2` vs `3`); objective all-zero. |

## Source cites

- Igraph ordering computes incidence barycenters and sorts layer members:
  `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:677-828`.
- Igraph BK horizontal placement detects Type 1 conflicts before four alignments:
  `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:858-1047`.
- The conflict quirk is the neighbor-count loop using ordinal edge ids:
  `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:893-944`.
- Median-of-four and minimum-width anchor selection already match Dagua:
  `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:990-1029`.
- Igraph LP rank assignment uses GLPK simplex with presolve off:
  `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:552-675`.

## Before/after probe

Tensor-level probe uses Procrustes RMSD to installed igraph positions and exact
crossing counts on seed 42.

| Combo | Before RMSD | After RMSD | Before crossings D/R | After crossings D/R | Verdict |
|---|---:|---:|---:|---:|---|
| `multiscale_skip_cascade::default` | 0.027172 | 0.000000 | 9 / 8 | 8 / 8 | Exact shape, fixed crossing |
| `multiscale_skip_cascade::passes4` | n/a | 0.000000 | r75 9 / 8 | 8 / 8 | Exact shape |
| `multiscale_skip_cascade::passes48` | n/a | 0.000000 | r75 9 / 8 | 8 / 8 | Exact shape |
| `multiscale_skip_cascade::tight` | n/a | 0.000000 | r75 9 / 8 | 8 / 8 | Exact shape |
| `multiscale_skip_cascade::wide` | n/a | 0.000000 | r75 9 / 8 | 8 / 8 | Exact shape |
| `hexagonal_lattice_42::default` | 0.024046 | 0.023714 | 3 / 6 | 3 / 6 | Small x improvement, not fixed |
| `width_skew_late_merge::default` | 0.073482 | 0.073482 | 3 / 3 | 3 / 3 | Unchanged |
| `real_karate_34::default` | 0.236064 | 0.236178 | 349 / 374 | 373 / 374 | Crossing improved; rank tie remains |
| `kitchen_sink_platform_graph::default` | n/a | 0.011512 | r75 0 / 0 | 0 / 0 | Near, not exact |
| `planar_60::default` | n/a | 0.030086 | r75 192 / 185 | 185 / 185 | Crossing fixed on seed 42 |

Expanded seed-42 after-probe:

- `multiscale_skip_cascade` all five igraph variants were exact shape and had
  crossing count `8 / 8`.
- `planar_60` default/tight/wide had crossing count `185 / 185`.
- `real_karate_34` default/tight/wide improved crossing delta to `1`, but rank
  assignment still diverged.
- Five of fifty probed graph/variant rows had Procrustes RMSD `< 0.01`; all five
  were `multiscale_skip_cascade`.

## Gate evidence

- `ruff check . --fix`: passed, `All checks passed!`
- `pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q`: passed,
  `58 passed, 3099 deselected, 34 warnings in 15.08s`.
- `mypy --follow-imports=silent dagua/cli.py`: passed,
  `Success: no issues found in 1 source file`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: passed,
  `461 passed, 153 warnings in 1746.85s (0:29:06)`.
- Probe benchmark:
  `scripts/run_benchmark.py --workers 1 --timeout 240 --seeds 1 --seed-start 42 --seed-refs igraph_sugiyama ...`
  completed `100 total, 100 ok, 0 skipped, 0 errors, 0 timeouts`.

The full 100-seed igraph-family benchmark was not run because this is a partial
portable fix, not full gate completion. The remaining rank-stage rows require
GLPK-like simplex basis parity or a version-pinned replacement solver.

## Regression scope

- Graphviz fidelity code paths are not touched. The new flag is passed only
  when `fidelity_mode == "igraph"`; graphviz still uses the existing dot x
  assignment branch.
- Existing unit coverage for graphviz dot rank/mincross/x remained green in
  the focused pytest gate.
- Pycache files were already dirty in the worktree and were not included in the
  intended change set.

## Commit

Implementation commit: `4377a80` (`fix(sugiyama): match igraph conflict tie quirk`).

## Concerns

- LP rank tie parity remains the largest open class. For all-zero objectives,
  installed igraph/GLPK selected ranks not reproduced by any SciPy method tried.
- The ordinal conflict quirk improves a meaningful crossing/x-stage subset but
  does not address rows whose first divergence is ranking.
- `width_skew_late_merge` remains x-stage divergent even after the conflict
  quirk, so there is at least one more BK detail or dummy-edge construction
  detail still unmatched.

## Knowledge

- `multiscale_skip_cascade` is a clean BK conflict regression: ranks and
  original-node orders already matched, and the igraph ordinal conflict quirk
  makes all five igraph Sugiyama variants exact at seed 42.
- `real_karate_34` and `moe_router_sparse` are rank-stage rows, not ordering or
  BK rows. Their LP objectives are all-zero and expose solver basis ties.

## A6: GLPK parity via optional dependency

Date: 2026-07-04
Worktree: `/home/jtaylor/.claude/worktrees/dagua-igraph-glpk`
Branch: `r77/igraph-glpk`

### Version check

- Installed python-igraph: `1.0.0`
- Installed igraph C core exposed by python package: `1.0.0`
- `swiglpk.glp_version()`: `5.0`
- igraph 1.0.0 source fetched read-only to `/tmp/igraph-src/igraph-1.0.0`.
- igraph 1.0.0 vendored GLPK version: `5.0`, cited by
  `/tmp/igraph-src/igraph-1.0.0/vendor/source/igraph/ACKNOWLEDGEMENTS.md`.

No material GLPK version mismatch was found.

### Reference LP construction

Source: `/tmp/igraph-src/igraph-1.0.0/vendor/source/igraph/src/layout/sugiyama.c:552-675`.

The directed Sugiyama vertical placement path uses GLPK only when the graph is
directed and `vcount <= 1000`. Otherwise it falls back to the Eades or
undirected feedback heuristics.

Construction details:

| Source lines | Detail |
|---|---|
| 584-586 | Compute Eades feedback edges and sort them. |
| 588-592 | Compute `indegs` and `outdegs`; both use `IGRAPH_IN`, preserving the 1.0.0 IN/IN quirk. |
| 593-600 | Subtract feedback-edge weights from `outdegs[from]` and `indegs[to]`. |
| 602-606 | `glp_term_out(GLP_OFF)`, `glp_init_smcp`, `parm.msg_lev = GLP_MSG_OFF`, `parm.presolve = GLP_OFF`. |
| 608-616 | Minimize. Add columns in vertex-id order. Each column is `GLP_IV`, lower-bounded at zero, objective coefficient `outdegs[i] - indegs[i]`. |
| 621-646 | Add one row per original edge id. Non-feedback edge row uses coefficients `from=-1`, `to=1` with `GLP_LO` lower bound 1. Feedback edge row uses the same coefficients with `GLP_UP` upper bound -1. Self-loop rows are skipped after advancing the sorted feedback cursor when needed. |
| 648-656 | Run `glp_simplex(ip, &parm)`, then consume `floor(glp_get_col_prim(ip, i + 1))`. |

### Port

Changed `dagua/layout/ops/sugiyama.py` so `_igraph_glpk_layer_assignments`
uses optional `swiglpk` first. The GLPK branch mirrors igraph's row and column
order, bounds, objective coefficients, simplex parameters, and floor-based
solution extraction. If `swiglpk` is missing or GLPK returns a nonzero simplex
code, the existing SciPy fallback is used.

Added optional extra:

```
igraph-fidelity = ["igraph>=0.10", "swiglpk>=5.0"]
```

Added tests in `tests/test_layout/test_sugiyama_fidelity.py`:

- GLPK tie-row parity against installed igraph on `moe_router_sparse`.
- Missing-`swiglpk` fallback regression preserving the SciPy LP solution.

### Draw-level LP parity

Default igraph Sugiyama probe, seed-independent, `maxiter=24`, `vgap=1.0`,
`hgap=1.0`.

| Graph | GLPK rank vector matches installed igraph | Full-layout d_R after GLPK | Exact positions |
|---|---:|---:|---:|
| `real_karate_34` | yes | 0.092463303 | no |
| `moe_router_sparse` | yes | 0.000000000 | yes |
| `hexagonal_lattice_42` | yes | 0.153686513 | no |
| `width_skew_late_merge` | yes | 0.302972600 | no |

The GLPK LP solution now matches installed igraph on all four named probes.
The remaining nonzero full-layout distances are downstream coordinate-stage
residuals, not rank LP residuals.

### 10-row probe from r76 igraph-far rows

Rows were selected from
`/home/jtaylor/projects/dagua/eval_output/fidelity_definitive/per_combo_r76.jsonl`
among small non-bit-exact `classic_sugiyama*` igraph-reference rows.

| Row | d_R after GLPK | Under 0.01 |
|---|---:|---:|
| `moe_router_sparse::classic_sugiyama_default` | 0.000000000 | yes |
| `moe_router_sparse::classic_sugiyama_passes4` | 0.000000000 | yes |
| `moe_router_sparse::classic_sugiyama_passes48` | 0.000000000 | yes |
| `moe_router_sparse::classic_sugiyama_tight` | 0.000000000 | yes |
| `moe_router_sparse::classic_sugiyama_wide` | 0.000000000 | yes |
| `real_karate_34::classic_sugiyama_default` | 0.092463303 | no |
| `real_karate_34::classic_sugiyama_passes4` | 0.099041559 | no |
| `hexagonal_lattice_42::classic_sugiyama_default` | 0.153686513 | no |
| `width_skew_late_merge::classic_sugiyama_default` | 0.302972600 | no |
| `kitchen_sink_platform_graph::classic_sugiyama_default` | 0.048839821 | no |

Result: 5 of 10 rows under 0.01. This misses the gate target of at least 6 of
10. A broader small-row scan found the same five MoE rows under 0.01 before a
segmentation fault inside repeated installed-igraph layout calls after 75 rows.

### Regression evidence

| Gate item | Result |
|---|---|
| Graphviz-fidelity byte identity | pass on 5-row sample: `asymmetric_hourglass_hub`, `binary_tree`, `hub_skip_superfan`, `interleaved_cluster_crosstalk`, `width_skew_late_merge`; `cmp` exit 0 against archived pre-patch HEAD. |
| No-`swiglpk` fallback byte identity | pass on 3-row sample: `moe_router_sparse`, `real_karate_34`, `width_skew_late_merge`; `cmp` exit 0 against archived pre-patch HEAD when `_swiglpk = None`. |
| `pytest tests/test_layout/test_sugiyama_fidelity.py -q -x` | pass: `14 passed, 3 warnings in 0.37s`. |
| `pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q` | pass: `60 passed, 3104 deselected, 34 warnings in 21.14s`. |
| `ruff check . --fix` | pass: `All checks passed!`. |
| `mypy --follow-imports=silent dagua/cli.py` | pass: `Success: no issues found in 1 source file`. |

Invalid sample note: an initial "bit-exact" sample used
`quality_identical_raw` from the definitive artifact, which is a quality metric
identity flag rather than a positional bit-exact manifest. It is not counted as
gate evidence.

### Gate status

No commit was made. The GLPK LP parity port is implemented and locally tested,
but gate 1 is not met because full-layout d_R collapses on 5 of the 10-row
probe, not at least 6. The remaining miss class appears to be downstream
Brandes-Koepf coordinate assignment parity, consistent with the r76 x-stage
residuals for `hexagonal_lattice_42` and `width_skew_late_merge`.

Full 100-seed igraph-family benchmark was not run because the pre-commit gate
did not pass. No benchmark output directory was written.

### Concerns

- `swiglpk` reproduces installed igraph's GLPK rank solution, including the
  degenerate MoE tie row.
- GLPK rank parity alone is insufficient for the full A6 gate because several
  far-tier rows diverge after ranking in coordinate assignment.
- Repeated installed-igraph Sugiyama calls segfaulted during an exploratory
  broad scan after 75 small rows; the completed rows still showed only the
  five MoE variants below 0.01.

### Knowledge

- The optional dependency path is viable: igraph 1.0.0 and `swiglpk` both use
  GLPK 5.0, and the simplex basis tie matches installed igraph on the named
  LP-sensitive rows.
- `moe_router_sparse` was a pure GLPK rank-tie row; all five igraph Sugiyama
  variants become exact after the GLPK port.
- `real_karate_34` rank parity is fixed, but full-layout distance remains
  about 0.09, so it has at least one downstream ordering or coordinate residual.

## A7: BK x-stage parity

### Source bisection

Installed python-igraph reports Python package `1.0.0` and C core `1.0.0`.
The reference BK implementation used for the port was
`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c`.

The first divergent quantity was the four-direction BK run orientation:
Dagua mirrored layers and coordinates for right/down passes, while igraph keeps
the original `vertex_to_the_left` array for all four passes and changes only the
`reverse` and `align_right` flags. Igraph calls vertical alignment and horizontal
compaction as `reverse = i / 2`, `align_right = i % 2` while passing the same
`vertex_to_the_left` into compaction (`sugiyama.c:981-988`). Vertical alignment
uses original `X_POS` order and an `align_right` scan/median rule
(`sugiyama.c:1049-1168`). Horizontal compaction walks the original
`vertex_to_the_left` relation recursively and applies class shifts from that
relation (`sugiyama.c:1190-1301`). Balancing then anchors all four runs to the
minimum-width run and takes `median_4` (`sugiyama.c:990-1029`).

Ordering was checked with matched GLPK ranks before BK comparison. The A3
ordinal-edge Type-1 conflict quirk remained active; no ordering-stage mismatch
was found on the named probes before entering BK.

| Probe | Rank equal | Dagua mirrored BK max abs x residual | Source-shaped BK max abs x residual | Named divergence |
|---|---:|---:|---:|---|
| `real_karate_34` | yes | 59.0 | 0.0 | four-run orientation/compaction relation |
| `width_skew_late_merge` | yes | 7.0 | 3.5 | four-run orientation fixed; residual remains |
| `hexagonal_lattice_42` | yes | 3.0 | 3.0 | no improvement from orientation alone |

The port is gated to `fidelity_mode="igraph"` through the existing igraph
conflict/BK branch. Generic BK and graphviz-fidelity paths remain separate.

### Implementation

Commits:

- `2f54853 fix(sugiyama): match igraph BK alignment runs`
- `922464a fix(sugiyama): avoid recursion limit in igraph compaction`

Changed files:

- `dagua/layout/ops/sugiyama.py`: added the source-shaped igraph-only BK
  coordinate assignment path, including igraph edge-id Type-1 ignore marking,
  original-order vertical alignment, original `vertex_to_the_left` horizontal
  compaction, min-width anchoring, and four-run median balancing. The later
  recursion-limit guard keeps igraph's recursive C block-placement shape but
  raises Python's recursion limit around large dummy-expanded compactions.
- `tests/test_layout/test_sugiyama_fidelity.py`: added a karate regression that
  compares igraph-fidelity Sugiyama coordinates against installed python-igraph.

### 10-row probe from r76 igraph-far rows

References were invoked in fresh subprocesses to avoid the known repeated
installed-igraph segfault mode.

| Row | d_R after A6 GLPK | d_R after A7 BK | Exact positions after A7 |
|---|---:|---:|---:|
| `moe_router_sparse::classic_sugiyama_default` | 0.000000000 | 0.000000000 | yes |
| `moe_router_sparse::classic_sugiyama_passes4` | 0.000000000 | 0.000000000 | yes |
| `moe_router_sparse::classic_sugiyama_passes48` | 0.000000000 | 0.000000000 | yes |
| `moe_router_sparse::classic_sugiyama_tight` | 0.000000000 | 0.000000000 | yes |
| `moe_router_sparse::classic_sugiyama_wide` | 0.000000000 | 0.000000000 | yes |
| `real_karate_34::classic_sugiyama_default` | 0.092463303 | 0.000000000 | yes |
| `real_karate_34::classic_sugiyama_passes4` | 0.099041559 | 0.000000000 | yes |
| `hexagonal_lattice_42::classic_sugiyama_default` | 0.153686513 | 0.144675773 | no |
| `width_skew_late_merge::classic_sugiyama_default` | 0.302972600 | 0.298854126 | no |
| `kitchen_sink_platform_graph::classic_sugiyama_default` | 0.048839821 | 0.000000000 | no raw-exact; equivalence distance zero |

Result: 8 of 10 rows under 0.01. No row that was exact or near after A6 left
that class.

### Regression evidence

| Gate item | Result |
|---|---|
| Focused igraph fidelity regression | pass: `pytest tests/test_layout/test_sugiyama_fidelity.py -q -x` -> `15 passed, 3 warnings in 1.04s`. |
| Task test gate | pass: `pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q` -> `61 passed, 3104 deselected, 34 warnings in 20.24s`. |
| Ruff | pass: `ruff check . --fix` -> `All checks passed!`. |
| CLI mypy | pass: `mypy --follow-imports=silent dagua/cli.py` -> `Success: no issues found in 1 source file` plus the existing unused pyproject section note. |
| Broader targeted project test | pass before the recursion-limit follow-up: `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q` -> `464 passed, 153 warnings in 3423.70s`. |
| Final Tier 2 project test | blocked by known pre-existing double-border smoke: `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"` -> `FAILED tests/test_cosmetic_node_features.py::TestRenderSmoke::test_render_with_double_border`, `1 failed, 260 passed, 88 deselected, 1 xfailed, 63 warnings in 524.03s`. |
| rgg recursion check | pass after `922464a`: `run_benchmark --engines classic_sugiyama --graphs rgg_500 --max-nodes 0 --seeds 3 --seed-start 100 --workers 1 --timeout 600 --watchdog-timeout 1200` -> `Done: 1 total, 1 ok, 0 skipped, 0 errors, 0 timeouts`. |

### Full family benchmark

Command:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --engines classic_sugiyama --variants --max-nodes 0 \
  --seeds 100 --seed-start 100 --workers 5 \
  --timeout 3600 --watchdog-timeout 7200 \
  --output-dir /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r77_igraph_bk
```

Done line:

```text
[benchmark] Done: 60600 total, 60400 ok, 0 skipped, 200 errors, 0 timeouts
```

The 200 errors were all watchdog expirations in graphviz-fidelity rows:

- `rgg_2000::classic_sugiyama_graphviz_fidelity`, seeds 100-199
- `ba_5000::classic_sugiyama_graphviz_fidelity`, seeds 100-199

The benchmark therefore did not meet the requested 0-error gate. The failing
rows are graphviz-fidelity large-graph rows, not igraph-fidelity BK rows, and
the task safety constraints prohibited graphviz-fidelity path changes.

### Concerns

- The igraph BK orientation divergence is ported and flips the real-karate
  class to exact, giving 8 of 10 A6 probe rows under 0.01.
- `hexagonal_lattice_42` and `width_skew_late_merge` retain smaller x-stage
  residuals after the first divergence fix. Their next divergent quantity was
  not ported in this pass.
- The full benchmark is not clean because two graphviz-fidelity large-graph
  groups exceeded the watchdog. This is outside the igraph BK change surface
  and outside the allowed edit scope for A7.

### Knowledge

- Igraph's four BK passes are flag-driven over the original layer/order
  relations; right/down passes are not coordinate-mirrored compaction runs.
- Igraph's horizontal compaction recursion can exceed Python's default
  recursion depth after dummy expansion; a temporary recursion-limit guard is
  required to preserve source-shaped behavior on larger graphs.

## A11: BK second-order

### Cluster table

Source ledger:
`/home/jtaylor/projects/dagua/eval_output/fidelity_definitive_r77/OFFICIAL_R77_LEDGER.md`.
The official named-cause registry contains 145 rows for
`igraph Sugiyama GLPK-solved then BK/dummy residual from wired r77 data`, grouped
as 29 graph classes with five variants each:

| Graph class | Rows |
|---|---:|
| `ba_2000` | 5 |
| `ba_500` | 5 |
| `ba_5000` | 5 |
| `chung_lu_150` | 5 |
| `citation_dag_300` | 5 |
| `clustered_medium_5x20` | 5 |
| `dense_pair_50` | 5 |
| `densenet_block` | 5 |
| `dependency_500` | 5 |
| `dependency_graph_100` | 5 |
| `extreme_mixed_width_transformer` | 5 |
| `heavy_tail_weights_50` | 5 |
| `hexagonal_lattice_42` | 5 |
| `hub_and_spoke_3x20` | 5 |
| `hub_skip_superfan` | 5 |
| `hub_spoke_10x20` | 5 |
| `hub_spoke_5x50` | 5 |
| `interleaved_cluster_crosstalk` | 5 |
| `planar_60` | 5 |
| `powerlaw_2000` | 5 |
| `powerlaw_500` | 5 |
| `protein_ppi_200` | 5 |
| `random_dag_200` | 5 |
| `random_dag_50` | 5 |
| `regular_4_40` | 5 |
| `scale_free_ba_120` | 5 |
| `small_world_500` | 5 |
| `weighted_clusters_3x10` | 5 |
| `width_skew_late_merge` | 5 |

### Per-quantity bisection

The second-order portable quantity found in this pass is dummy-chain creation
order for small acyclic igraph-fidelity DAGs. Igraph constructs the
dummy-expanded Sugiyama subgraph by scanning component-local source vertices and
then outgoing edge ids before ordering and BK placement
(`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:379-472`).
Dagua's igraph path was still creating dummy chains in global edge-list order.

Runtime probes showed that switching this order globally regressed the A7
`real_karate_34` exact row, so the port is conservatively gated to
`fidelity_mode="igraph"` and original non-loop inputs with `N <= 20` that are
already acyclic. That covers the proven small-DAG dummy-order class without
touching the GLPK/social exact class.

An attempted no-neighbor barycenter source-parity edit was reverted because it
did not improve the named probes and broke the karate exact regression.

### Port

Changed files:

- `dagua/layout/ops/sugiyama.py`: added an igraph-only dummy edge processing
  mode, stored a small-acyclic-DAG guard during igraph layer assignment, and
  applied source-vertex dummy-chain ordering during dummy expansion only when
  that guard is true.
- `dagua/layout/ops/pipelines/sugiyama.py`: wires
  `_ExpandDummyNodes(use_igraph_edge_order=True)` only for
  `fidelity_mode="igraph"`.
- `tests/test_layout/test_sugiyama_fidelity.py`: adds a DenseNet exact-layout
  regression against installed python-igraph.

### Before/after d_R

Direct installed-igraph probe using anisotropic Procrustes distance:

| Row | A7/old status | A11 d_R | Status |
|---|---:|---:|---|
| `densenet_block::classic_sugiyama_default` | divergent | 0.000000000 | ported |
| `densenet_block::classic_sugiyama_passes4` | divergent | 0.000000000 | ported |
| `densenet_block::classic_sugiyama_passes48` | divergent | 0.000000000 | ported |
| `densenet_block::classic_sugiyama_tight` | divergent | 0.000000000 | ported |
| `densenet_block::classic_sugiyama_wide` | divergent | 0.000000000 | ported |
| `width_skew_late_merge::classic_sugiyama_default` | 0.298854126 | 0.000000000 | ported |
| `width_skew_late_merge::classic_sugiyama_passes4` | divergent | 0.000000000 | ported |
| `width_skew_late_merge::classic_sugiyama_passes48` | divergent | 0.000000000 | ported |
| `width_skew_late_merge::classic_sugiyama_tight` | divergent | 0.000000000 | ported |
| `width_skew_late_merge::classic_sugiyama_wide` | divergent | 0.000000000 | ported |
| `hexagonal_lattice_42::classic_sugiyama_default` | 0.144675773 | 0.144675773 | residual |
| `hub_skip_superfan::classic_sugiyama_default` | divergent | 0.406700151 | residual |

Gate probe result: 10 of 12 rows under `d_R < 0.01`.

Default-row sub-300 scan after the port:

- Ported: `densenet_block`, `extreme_mixed_width_transformer`,
  `hub_and_spoke_3x20`, `hub_spoke_10x20`, `hub_spoke_5x50`,
  `interleaved_cluster_crosstalk`, `width_skew_late_merge`.
- Still residual: `chung_lu_150`, `citation_dag_300`,
  `clustered_medium_5x20`, `dense_pair_50`, `dependency_graph_100`,
  `heavy_tail_weights_50`, `hexagonal_lattice_42`, `hub_skip_superfan`,
  `planar_60`, `protein_ppi_200`, `random_dag_200`, `random_dag_50`,
  `regular_4_40`, `scale_free_ba_120`, `weighted_clusters_3x10`.
- Skipped by the sub-300 probe: `ba_2000`, `ba_500`, `ba_5000`,
  `dependency_500`, `powerlaw_2000`, `powerlaw_500`, `small_world_500`.

### Gate evidence

Passed:

- `ruff check . --fix` -> `All checks passed!`
- `mypy --follow-imports=silent dagua/cli.py` -> `Success: no issues found in 1 source file`
- `pytest tests/test_layout/test_sugiyama_fidelity.py -q -x` ->
  `20 passed, 3 warnings in 1.03s`
- `pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q` ->
  `70 passed, 3109 deselected, 34 warnings in 26.10s`

Not completed in this pass:

- Instrumented python-igraph build: source tarball downloaded to
  `/tmp/dagua_bk2_igraph_sdist/igraph-1.0.0.tar.gz`, but the build was not
  completed.
- Bit-exact 10x3 sample, graphviz-fidelity byte sample, final Tier 2 pytest,
  full family benchmark, and commit were not completed.

### Concerns

This is partial A11 work, not completion of the full contract. The small-DAG
dummy-order class is ported and guarded, but the `hexagonal_lattice_42` class
and dense/random/hub residual classes still need a first-difference dump and a
separate named quantity. The full 145-row residual set is therefore not closed.

### Knowledge

Dummy node ids and expanded edge ids are BK-visible in igraph because Type-1
conflict marking is edge-id ordinal and horizontal compaction works on the
dummy-expanded graph. For small acyclic DAGs, matching igraph's source-vertex
dummy-chain creation order is enough to collapse the DenseNet and
width-skew-late-merge classes.

## A11b: principled chain order

### Source-derived rule

The A11 `N <= 20` dummy-chain guard was removed. It was a probe fit, not an
igraph rule.

The matching igraph 1.0.0 source was unpacked read-only under
`/tmp/igraph-src-r78-bk2` and instrumented from a writable copy under
`/tmp/igraph-instrument-r78-bk2`. In
`vendor/source/igraph/src/layout/sugiyama.c:379-433`, igraph constructs the
dummy-expanded graph by scanning each component-local original vertex `i` in
ascending id order, calling `igraph_incident(graph, &neis, i, IGRAPH_OUT,
IGRAPH_LOOPS)`, then appending the dummy chain for each long outgoing incidence.
Lines 403-418 flip the chain endpoints when the solved layer direction is
upward, but the chain keeps the original outgoing incidence scan slot.

Python igraph 1.0.0 returns `incident(v, mode="out")` in adjacent-vertex order,
not input edge-id order. The exact port is therefore:

1. Store the original source, original target, and original edge id before
   layer-direction flipping.
2. Create dummy chains by original source vertex ascending.
3. Within each source bucket, use original target ascending, then original edge
   id as a stable tie-breaker.
4. Apply the layer-direction orientation only to the chain endpoints, not to the
   scan key.

This is size-free and applies to cyclic and acyclic igraph-fidelity graphs.

### Karate bisection

The broad A11 attempt regressed `real_karate_34` because it grouped chains by
oriented source and edge id after layer-direction flipping. That changes the id
slots for feedback-arc-flipped chains. Instrumented C dumps showed that igraph
still schedules those chains from the original outgoing incidence slot and only
then reverses the chain geometry if the solved layers require it.

After porting original `(source, target, edge_id)` scan keys, the same probe kept
`real_karate_34::classic_sugiyama_default` bit-exact while preserving the
DenseNet and width-skew exact rows.

### Remaining classes

`hexagonal_lattice_42` and `hub_skip_superfan` were not separate BK balancing
quantities. Their residuals came from the same missing target-sorted incidence
rule. Instrumented expanded-edge dumps showed the first mismatch immediately:

| Graph | Igraph expanded prefix | Previous Dagua expanded prefix |
|---|---|---|
| `hexagonal_lattice_42` | `0 1, 0 7, 1 8, 2 3, 2 42` | `0 7, 0 1, 1 8, 2 42, 2 3` |
| `hub_skip_superfan` | source buckets sorted by target before dummy ids | source buckets followed original edge order |

Once the dummy ids matched, BK per-direction candidates and median-4 balancing
matched for both representatives; no extra x-stage quantity was needed.

### Before/after d_R

Direct installed-igraph probes were batched in fresh subprocesses to avoid the
known segfault path.

| Row | A11 d_R | A11b d_R | Status |
|---|---:|---:|---|
| `densenet_block::classic_sugiyama_default` | 0.000000000 | 0.000000000 | stayed exact |
| `densenet_block::classic_sugiyama_passes4` | 0.000000000 | 0.000000000 | stayed exact |
| `densenet_block::classic_sugiyama_passes48` | 0.000000000 | 0.000000000 | stayed exact |
| `densenet_block::classic_sugiyama_tight` | 0.000000000 | 0.000000000 | stayed exact |
| `densenet_block::classic_sugiyama_wide` | 0.000000000 | 0.000000000 | stayed exact |
| `width_skew_late_merge::classic_sugiyama_default` | 0.000000000 | 0.000000000 | stayed exact |
| `width_skew_late_merge::classic_sugiyama_passes4` | 0.000000000 | 0.000000000 | stayed exact |
| `width_skew_late_merge::classic_sugiyama_passes48` | 0.000000000 | 0.000000000 | stayed exact |
| `width_skew_late_merge::classic_sugiyama_tight` | 0.000000000 | 0.000000000 | stayed exact |
| `width_skew_late_merge::classic_sugiyama_wide` | 0.000000000 | 0.000000000 | stayed exact |
| `hexagonal_lattice_42::classic_sugiyama_default` | 0.144675773 | 0.000000000 | ported |
| `hub_skip_superfan::classic_sugiyama_default` | 0.406700151 | 0.000000000 | ported |
| `real_karate_34::classic_sugiyama_default` | 0.000000000 | 0.000000000 | stayed exact |

Gate probe result: 13 of 13 rows under `d_R < 0.01`.

### Gate evidence

Passed:

- `ruff check . --fix` -> `All checks passed!`
- `mypy --follow-imports=silent dagua/cli.py` ->
  `Success: no issues found in 1 source file`
- `pytest tests/test_layout/test_sugiyama_fidelity.py -q -x` ->
  `23 passed, 3 warnings in 1.41s`
- `pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q` ->
  `73 passed, 3109 deselected, 34 warnings in 21.02s`
- no-swiglpk fallback:
  `pytest tests/test_layout/test_sugiyama_fidelity.py::test_sugiyama_igraph_glpk_import_absence_keeps_scipy_fallback -q`
  -> `1 passed, 3 warnings in 0.06s`
- 10-row bit-exact sample x 3 seeds: all byte-identical.
- 5 graphviz-fidelity rows x 3 seeds: all byte-identical.

Final Tier 2:

- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`
  stopped at the known pre-existing cosmetic render failure:
  `tests/test_cosmetic_node_features.py::TestRenderSmoke::test_render_with_double_border`
  with `assert 0 >= 2`.
- Output before stop:
  `1 failed, 266 passed, 88 deselected, 1 xfailed, 63 warnings in 218.04s`.

### Commits and bench

Implementation commit:

- `508afd4 fix(layout): match igraph dummy chain incidence order`

Documentation commit:

- This commit records the A11b evidence and benchmark line.

Bench command:

```bash
python3 scripts/run_benchmark.py --engines classic_sugiyama --variants --max-nodes 300 --seeds 100 --seed-start 100 --workers 5 --timeout 3600 --watchdog-timeout 7200 --output-dir /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_bk2
```

Bench result:

- `/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_bk2/summary.md`
- `51600 total, 51600 ok, 0 skipped, 0 errors, 0 timeouts`

### Knowledge

For igraph Sugiyama fidelity, dummy-chain ordering is original-out-incidence
ordering, not oriented-edge ordering and not input edge ordering. The original
target vertex is part of that order because igraph's outgoing incidence list is
adjacent-vertex sorted. Feedback-arc orientation only changes chain direction;
it does not move the chain to the reversed endpoint's scan bucket.

## A11c: big-graph tail

Date: 2026-07-05
Worktree: `/home/jtaylor/.claude/worktrees/dagua-bigtail`
Branch: `r78/bigtail`

### Gate audit

Installed python-igraph is `1.0.0`.

Source verdict: installed igraph has a directed GLPK size gate. The vertical
Sugiyama placement path uses GLPK only when `igraph_is_directed(graph)` and
`no_of_nodes <= 1000`; directed graphs above that gate call
`igraph_i_feedback_arc_set_eades(...)` instead. Source cite:
`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:552-665`.

Empirical rank audit on installed igraph:

| Probe | Nodes | Edges | Installed behavior | Runtime |
|---|---:|---:|---|---:|
| `er_500` | 500 | 1230 | matched Dagua GLPK gate, not Eades (`diff_vs_eades=138`) | 0.037s |
| `er_2000` | 2000 | 5936 | matched Dagua gate and Eades exactly (`diff_vs_eades=0`) | 0.124s |

Dagua's current `>1000` Eades fallback mirrors installed igraph. No gate fix
was needed.

### Current-code probes

Assumption for "1 seed then 3": seed 42 for the single-seed pass, then seeds
42, 43, and 44 for the three-seed pass. All probes ran Dagua and installed
igraph in the same Python process for the default variant.

| Graph | 1-seed d_R | 3-seed d_R values | Max d_R |
|---|---:|---|---:|
| `ba_500` | 0.000000000 | 0.000000000, 0.000000000, 0.000000000 | 0.000000000 |
| `er_2000` | 0.000000000 | 0.000000000, 0.000000000, 0.000000000 | 0.000000000 |
| `dependency_500` | 0.000000000 | 0.000000000, 0.000000000, 0.000000000 | 0.000000000 |
| `sbm_8x100` | 0.000000000 | 0.000000000, 0.000000000, 0.000000000 | 0.000000000 |

No instrumented igraph bisection or port was needed for the big-tail classes:
all requested current-code probes were exact under the `d_R < 0.01` gate.

### Stale-vs-real split

The manifest at
`/home/jtaylor/projects/dagua/.project-context/research/sprint_rng_matching/r76_scratch/r78_igraph_tail.txt`
has 70 rows. It contradicts the task text: 18 rows are <=100-node graphs
(`dependency_graph_100`, `er_100`, `multi_component_80`,
`parallel_cycles_4x5`), not >300-node graphs.

Fresh seed-42 subprocess-batched measurement against installed igraph:

| Bucket | Rows | Current-code status |
|---|---:|---|
| Actual >300-node tail rows | 52 | all near/exact, `d_R < 0.01` |
| Small manifest rows | 18 | still far, `d_R >= 0.01` |
| Total manifest rows | 70 | 52 near, 18 far, 0 errors |

Per-graph maxima:

| Graph | Rows | Max d_R |
|---|---:|---:|
| `ba_2000` | 5 | 0.000000000 |
| `ba_500` | 5 | 0.000000000 |
| `dependency_500` | 5 | 0.002373531 |
| `er_2000` | 5 | 0.000000000 |
| `er_500` | 5 | 0.000000000 |
| `powerlaw_2000` | 5 | 0.000000000 |
| `powerlaw_500` | 5 | 0.000000000 |
| `rgg_2000` | 4 | 0.000000000 |
| `rgg_500` | 5 | 0.000000000 |
| `sbm_8x100` | 3 | 0.000000000 |
| `small_world_500` | 5 | 0.000000000 |
| `dependency_graph_100` | 4 | 0.064910123 |
| `er_100` | 4 | 0.025780826 |
| `multi_component_80` | 5 | 0.023915041 |
| `parallel_cycles_4x5` | 5 | 0.137120839 |

Verdict: the big-graph tail ledger positions were stale-code artifacts. The
small manifest rows are real residuals, but they are not part of the >300-node
tail called out in this task.

### Gate and benchmark evidence

No code changes were made, so the code-change regression gate was not run.
The relevant measurement gates were:

- current-code requested probes: all four classes exact for 3 seeds.
- 70-row manifest probe: `70 ok`, `52 d_R < 0.01`, `18 d_R >= 0.01`, `0 errors`.
- big-tail benchmark:
  `6600 total, 6600 ok, 0 skipped, 0 errors, 0 timeouts`.

Benchmark command:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python -u scripts/run_benchmark.py --variants --engines classic_sugiyama --seed-refs igraph_sugiyama --graphs ba_500,ba_2000,er_500,er_2000,powerlaw_500,powerlaw_2000,rgg_500,rgg_2000,small_world_500,dependency_500,sbm_8x100 --max-nodes 0 --seeds 100 --seed-start 100 --workers 4 --timeout 21600 --watchdog-timeout 28800 --output-dir /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_bigtail
```

Result files:

- `/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_bigtail/results.json`
- `/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_bigtail/summary.md`

Variant note: the project registry expands `classic_sugiyama --variants` to
six variants, including `classic_sugiyama_graphviz_fidelity`. That graphviz
variant is outside the igraph-tail manifest but was included by the requested
benchmark command and completed cleanly. The slowest group was
`rgg_2000::classic_sugiyama_graphviz_fidelity` at 11377.9s for its 100-seed
group, still inside the authorized timeout.

### Commits

Implementation commit: none; no code change was required.

Documentation commit:

- This commit records the A11c audit, probe, split, and benchmark evidence.

### Knowledge

Igraph 1.0.0's Sugiyama scale gate is source-backed and empirically active:
directed graphs with `N <= 1000` use the GLPK rank LP, while directed graphs
with `N > 1000` use the Eades feedback-order fallback. The current Dagua
fallback mirrors that behavior at 2000 nodes.

## A11d: the last 18

Date: 2026-07-05
Worktree: `/home/jtaylor/.claude/worktrees/dagua-close18`
Branch: `r78/close18`

### Scope and row list

The requested
`.project-context/research/sprint_rng_matching/r76_scratch/r78_igraph_close18.txt`
was not present on `develop`. The row set was reconstructed from the A11c split
and a fresh subprocess-batched installed-igraph probe: among the four named
small classes, 20 igraph variants were measured, two were already exact on
`develop` (`dependency_graph_100::passes4` and `er_100::passes48`), and the
remaining 18 were the close rows with `d_R >= 0.01`.

All four named classes have multiple weak components:

| Graph | Components | Close rows |
|---|---:|---:|
| `dependency_graph_100` | 2 | 4 |
| `er_100` | 4 | 4 |
| `multi_component_80` | 7 | 5 |
| `parallel_cycles_4x5` | 4 | 5 |

### Bisection

Same-process Dagua-vs-installed-igraph probing showed every row had
`component_aligned_rmsd = 0.0`. Ranks, layer orders, dummy-chain order, BK runs,
and component-local final coordinates already matched after A11b. The first
diverging quantity was the X offset assigned between weak components.

Installed igraph lays out each weak component independently, writes real
vertices as `layout[i, 0] + dx`, and then advances the component margin from the
maximum X coordinate across the whole dummy-expanded subgraph, not only visible
original vertices. Source cite:
`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:475-523`.

Dagua was left-normalizing each component and advancing `dx` from original-node
positions only. This was enough to keep each component shape exact while making
the packed whole-graph shape close but non-exact.

Named law: igraph Sugiyama component packing uses the dummy-expanded component
right margin and preserves each component's local X origin.

### Port

Changed files:

- `dagua/layout/ops/pipelines/sugiyama.py`: replaced the igraph-only component
  packing recursion with a helper that reads `_SUGIYAMA_EXPANDED_POSITIONS_KEY`
  from the component pipeline, writes component positions without per-component
  min-X normalization, and advances `dx` by `max(expanded_x) + hgap`.
- `tests/test_layout/test_sugiyama_fidelity.py`: added a four-node regression
  where a long-edge dummy extends the component margin past visible real nodes.

The fix is gated to `fidelity_mode="igraph"` and only the existing weak-
component packing path consumes it. Graphviz-fidelity, default non-igraph
Sugiyama, eval scoring, and reference runners were not changed.

Implementation commit: `5bfed0b` (`fix(sugiyama): match igraph component packing margins`).

### Before/after d_R

References were invoked in fresh subprocesses to avoid the known repeated
installed-igraph segfault path. Distance is Procrustes `d_R`; after values were
also raw position-equal to installed igraph.

| Row | Before | After |
|---|---:|---:|
| `dependency_graph_100::default` | 0.026904936 | 0.000000000 |
| `dependency_graph_100::passes48` | 0.064910123 | 0.000000000 |
| `dependency_graph_100::tight` | 0.026904936 | 0.000000000 |
| `dependency_graph_100::wide` | 0.026904936 | 0.000000000 |
| `er_100::default` | 0.012408200 | 0.000000000 |
| `er_100::passes4` | 0.025780826 | 0.000000000 |
| `er_100::tight` | 0.012408200 | 0.000000000 |
| `er_100::wide` | 0.012408200 | 0.000000000 |
| `multi_component_80::default` | 0.023915041 | 0.000000000 |
| `multi_component_80::passes4` | 0.023915041 | 0.000000000 |
| `multi_component_80::passes48` | 0.023915041 | 0.000000000 |
| `multi_component_80::tight` | 0.023915041 | 0.000000000 |
| `multi_component_80::wide` | 0.023915041 | 0.000000000 |
| `parallel_cycles_4x5::default` | 0.137120839 | 0.000000000 |
| `parallel_cycles_4x5::passes4` | 0.137120839 | 0.000000000 |
| `parallel_cycles_4x5::passes48` | 0.137120839 | 0.000000000 |
| `parallel_cycles_4x5::tight` | 0.137120839 | 0.000000000 |
| `parallel_cycles_4x5::wide` | 0.137120839 | 0.000000000 |

Gate result: 18 of 18 close rows under `d_R < 0.01`; all 18 raw-equal. The two
already-near rows (`dependency_graph_100::passes4`, `er_100::passes48`) stayed
raw-equal.

### Gate evidence

Passed:

- `ruff check . --fix` -> `All checks passed!`
- `pytest tests/test_layout/test_sugiyama_fidelity.py -q -x` ->
  `24 passed, 3 warnings in 1.27s`
- `pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q` ->
  `79 passed, 3111 deselected, 50 warnings in 123.21s`
- no-swiglpk fallback:
  `pytest tests/test_layout/test_sugiyama_fidelity.py::test_sugiyama_igraph_glpk_import_absence_keeps_scipy_fallback -q`
  -> `1 passed, 3 warnings in 0.06s`
- 10-row igraph byte-identity sample x 3 seeds against detached `develop`:
  30 rows, 0 hash differences.
- 5-row graphviz-fidelity byte-identity sample x 3 seeds against detached
  `develop`: 15 rows, 0 hash differences.

Final Tier 2:

- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`
  stopped at the known pre-existing cosmetic render failure:
  `tests/test_cosmetic_node_features.py::TestRenderSmoke::test_render_with_double_border`
  with `assert 0 >= 2`.
- Output before stop:
  `1 failed, 266 passed, 88 deselected, 1 xfailed, 63 warnings in 210.59s`.

### Benchmark

Command:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python -u scripts/run_benchmark.py --engines classic_sugiyama --variants --graphs dependency_graph_100,er_100,multi_component_80,parallel_cycles_4x5 --max-nodes 0 --seeds 100 --seed-start 100 --workers 4 --timeout 3600 --watchdog-timeout 7200 --output-dir /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_close18
```

Result:

- `/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_close18/summary.md`
- `2400 total, 2400 ok, 0 skipped, 0 errors, 0 timeouts`

### Documentation commit

- This commit records the A11d bisection, port, gate evidence, and benchmark
  line.

### Knowledge

For igraph Sugiyama, weak-component packing is BK-visible even when each
component is exact. The inter-component margin is computed from the
dummy-expanded component layout, including dummy vertices that may lie to the
right of all original vertices, and components are not re-left-normalized before
adding the running `dx`.

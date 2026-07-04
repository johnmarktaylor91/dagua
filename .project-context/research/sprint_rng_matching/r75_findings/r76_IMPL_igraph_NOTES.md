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

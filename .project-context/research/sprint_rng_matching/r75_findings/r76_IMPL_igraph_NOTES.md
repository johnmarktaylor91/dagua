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

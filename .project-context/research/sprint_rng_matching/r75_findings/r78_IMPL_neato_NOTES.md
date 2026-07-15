# r78 neato implementation notes

Date: 2026-07-05
Worktree: `/home/jtaylor/.claude/worktrees/dagua-neato`
Branch: `r78/neato`
Reference runtime: installed `neato - graphviz version 7.0.5 (20221231.0122)`

## Inputs read

- `.project-context/research/sprint_rng_matching/r75_findings/r75_PROBE_tails_RESULTS.md`
- `/home/jtaylor/projects/dagua/eval_output/fidelity_definitive_r77/OFFICIAL_R77_LEDGER.md`
- `/home/jtaylor/projects/dagua/eval_output/fidelity_definitive/per_combo_r77.jsonl`
- Graphviz 7.0.5 source via `git -C /home/jtaylor/projects/_references/graphviz show 7.0.5:<path>`

## r77 neato row clustering

Official ledger neato rows: 83.

| Cluster | Rows | r77 disposition | Finding |
| --- | ---: | --- | --- |
| Connected same-seed CG/RNG | 47 | `DIVERGENT_NAMED_CAUSE` subset | Same-seed probes against installed Graphviz 7.0.5 are already near-exact; r77 Mode-B one-reference scoring made these look divergent. |
| Ordinary disconnected pack | 7 | `DIVERGENT_NAMED_CAUSE` subset | Real residual remained at component seed/pack boundary. |
| Singleton-heavy random-DAG pack | 2 | `FIDELITY_IDENTICAL` | Must retain prior per-component perturbation; root-seed reuse regresses this class. |
| Mode-B close | 15 | `MODE_B_CLOSE` | No code change needed; same-seed connected guards remain near-exact. |
| Mode-B identical-distance | 7 | `MODE_B_IDENTICAL_DISTANCE` | No code change; guard rows remain near-exact. |
| Insufficient data | 5 | `INSUFFICIENT_DATA` | Not used as pass/fail gate. |

## Source bisection evidence

Graphviz 7.0.5 source anchors:

- `lib/neatogen/neatoinit.c:checkStart` parses `start`, calls `srand48(seed)`, and is called inside `majorization`.
- `lib/neatogen/neatoinit.c:1441-1453` lays out each `pccomps` component before `packGraphs`.
- `lib/neatogen/stress.c:stress_majorization_kD_mkernel` builds packed shortest-path stress, initializes via `initLayout`, then runs packed CG.
- `lib/neatogen/conjgrad.c:conjugate_gradient_mkernel` is already mirrored in the current Dagua packed-CG path.

First-divergence probes:

| Probe | Result |
| --- | --- |
| Connected `maxiter=0` (`grid_5x5`, `asymmetric_hourglass_hub`, `petersen_10`) | Procrustes RMSD `1.1e-5`, `5.2e-5`, `3.6e-16`; initialization/export aligned. |
| Connected same-seed final (`asymmetric_hourglass_hub`, `petersen_10`, `grid_5x5`) | Mean RMSD `2.77e-5`, `4.22e-6`, `2.67e-5`; connected rows are not current implementation divergences. |
| Disconnected `maxiter=0` (`parallel_cycles_4x5`, `random_dag_50`, `multi_component_80`) | RMSD `0.986`, `0.848`, `1.003`; divergence exists before CG, at component init/pack/export. |
| Component seed policies | Root seed improves ordinary disconnected rows; singleton-heavy random-DAG class regresses and keeps old perturbation. |

## Ported operation

Changed `dagua/layout/ops/pipelines/neato.py`:

- Added `_component_seed_for_graphviz_neato`.
- Ordinary disconnected components now reuse the graph-level start seed.
- Singleton-heavy component sets keep `seed + component_index` because both r75 and r78 probes show this preserves random-DAG rows.

This is gated to the neato pipeline only. No shared packer defaults, reference runners, eval scoring, or other engines were changed.

## Before/after gate evidence

Representative same-seed RMSD against installed Graphviz 7.0.5, seeds 42-44:

| Row | Before mean | After mean | Verdict |
| --- | ---: | ---: | --- |
| `parallel_cycles_4x5::classic_neato` | `0.4080` | `0.2162` | Improved. |
| `multi_component_80::classic_neato` | `0.5596` | `0.1505` | Improved. |
| `random_dag_50::classic_neato` | `0.3447` | `0.3447` | Preserved by singleton-heavy guard. |
| `asymmetric_hourglass_hub::classic_neato` | `2.77e-5` | `2.77e-5` | Connected guard unchanged. |
| `grid_5x5::classic_neato` | `2.67e-5` | `2.67e-5` | Mode-B identical-distance guard unchanged. |

The port improves 2/3 disconnected representatives and preserves the singleton-heavy random-DAG guard plus connected/identical-distance rows. Remaining ordinary disconnected residual is component packing shape after Graphviz spline routing; Dagua still approximates pack occupancy with node boxes and straight component edges.

## Tests

- `ruff check . --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest -k neato -x --tb=short -q`: passed, 15 passed.
- `pytest tests/test_layout/test_neato.py tests/test_layout/test_neato_overlap.py tests/test_layout/test_neato_solver_fidelity.py tests/test_graph.py -x --tb=short -q`: passed, 50 passed.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: process was killed after about 18 minutes with no failure traceback.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`: stopped at known pre-existing render-smoke failure `tests/test_cosmetic_node_features.py::TestRenderSmoke::test_render_with_double_border`.
- Same command excluding only `test_render_with_double_border`: stopped at related render-smoke failure `tests/test_cosmetic_node_features.py::TestRenderSmoke::test_render_with_round_node_border_styles`.

Known pre-existing failure class from the task includes double-border smoke; no render files were changed.

## Benchmark line

The requested full neato family <=300, seeds 100-199 benchmark was not run in this turn. The targeted same-seed probes above were used for the port gate.

## Commit

- Implementation commit: `00bd57a` (`fix(neato): tune disconnected component seeds`)
- Notes SHA is the follow-up documentation commit containing this correction.

## Concerns

- Remaining disconnected residual is likely Graphviz `packGraphs(..., doSplines=1)` occupancy after per-component spline routing. Porting that without invoking Graphviz at runtime would require a larger spline-compatible pack occupancy model.
- The r77 connected `DIVERGENT_NAMED_CAUSE` neato rows should be rescored in same-seed Mode A; current same-seed implementation evidence is near-exact.

## Knowledge

- Installed Graphviz neato 7.0.5 varies by seed on connected rows, but Dagua already matches same-seed connected outputs closely.
- Disconnected neato divergence appears before CG (`maxiter=0`), so further CG work is not the next root cause for the remaining disconnected residual.
- Singleton-heavy random-DAG graphs are a distinct seed-policy class and should not use the ordinary disconnected root-seed policy.

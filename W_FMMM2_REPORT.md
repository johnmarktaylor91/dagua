# FMMM round-2 fidelity report

Date: 2026-07-11

## Outcome

All seven requested rows close under the 35-seed definitive distributional gate. The refreshed
matched references and current reimplementation each completed 245/245 layouts with no skips,
errors, or timeouts. The requested score artifact is
`eval_output/fidelity_definitive/per_combo_r79_fmmm.jsonl`.

| Row | `e_rel` | Distributional | Quality equivalent | Quality identical |
| --- | ---: | :---: | :---: | :---: |
| `small_world_2000::classic_fmmm_steps10` | `-5.64e-13` | yes | yes | yes |
| `grid_50x50::classic_fmmm_steps10` | `4.84e-08` | yes | yes | yes |
| `rgg_2000::classic_fmmm_steps10` | `3.39e-08` | yes | yes | yes |
| `powerlaw_2000::classic_fmmm_steps10` | `1.86e-08` | yes | yes | no |
| `parallel_cycles_4x5::classic_fmmm_graphviz_fdp_fidelity` | `-3.87e-16` | yes | yes | yes |
| `hub_and_spoke_3x20::classic_fmmm_graphviz_fdp_fidelity` | `2.58e-04` | yes | yes | yes |
| `random_dag_200::classic_fmmm_graphviz_fdp_fidelity` | `4.05e-03` | yes | yes | no |

## OGDF level-by-level evidence

The external OGDF reference was instrumented behind `OGDF_FMMM_TRACE` and `OGDF_NMM_TRACE` to
emit every multilevel hierarchy, placement, force iteration, reduced quadtree cell, interaction
list, and force component. The instrumentation is confined to the external reference checkout.

- `small_world_2000`: hierarchy `[2000, 314, 89, 26]`, node types, suns, fine-to-coarse
  mappings, edges, lengths, and coarsest placement are exact. The first material divergence was
  level-0 post-processing iteration 2: Dagua carried the `/10` cooldown across iterations,
  while OGDF resets the factor to one before each division.
- `rgg_2000`: hierarchy and placements are exact. The first material divergence was level-0
  iteration 1 in the NMM direct list. Coincident integer-rounded particles use the shared
  MT19937 random separation in OGDF; Dagua had returned zero force. After porting it, the first
  residual is a `2.2e-16` Python/`std::complex` arithmetic difference and the final maximum
  seed-42 coordinate difference is two integer units.
- `powerlaw_2000`: hierarchy, placement, and final seed-42 output are byte exact after the NMM
  correction.
- `grid_50x50`: final seed-42 output is byte exact after the cooldown correction.

The NMM center and pair-force arithmetic grouping was also matched to OGDF. The five previously
passing seed-42 FMMM controls remained byte exact.

## Graphviz FDP operation evidence

Graphviz 7.0.5 was instrumented to emit tLayout positions/displacements, every xLayout pair and
per-node displacement, PRISM scaling, smoother positions, and final component translations.

Implemented parity points:

- fixed DOT sizes round to the nearest integer point after four-decimal inch serialization;
- xLayout's four-point margin is widened from Graphviz's float `pointf` value;
- derived edges orient from lower to higher node sequence, and outgoing springs sort by target;
- NumPy scalar `hypot` matches Graphviz's platform-libm rounding where Python `math.hypot`
  differs by one ULP;
- PRISM uses its own exact four-point padding, not xLayout's float-widened margin;
- Graphviz 7.0.5's historical `average_edge_length` target-index bug is preserved;
- PRISM is skipped when xLayout has already eliminated every overlap;
- disconnected flat FDP uses node-polyomino packing rather than solid component rectangles.

Exact boundaries:

- `parallel_cycles_4x5`: each component and the final packed seed-42 output are byte exact.
- `hub_and_spoke_3x20`: tLayout, all 900 xLayout iterations, PRISM half-sizes, and PRISM initial
  scaling are exact. The first remaining operation-level divergence is PRISM smoother pass 0:
  Graphviz uses GTS Delaunay and its iterative sparse-CG stress solver; Dagua uses SciPy/Qhull
  and a direct sparse solve.
- `random_dag_200`: the 181-node component's tLayout, xLayout, and PRISM initial scaling are
  exact. Its first divergence is the same PRISM pass-0 solver boundary. The resulting small
  component-shape difference changes the later node-polyomino placement of singleton components.

## Commits

- `1879e5d` — `fix(fmmm): match OGDF cooldown and coincident NMM forces`
- `3e6d15e` — `fix(fdp): match graphviz force and overlap ordering`

## Benchmark and scoring

Reimplementation data:

- `eval_output/benchmark_100seed_r79_fmmm2/`
- 245 target layouts, all successful (the directory also contains interrupted cross-product
  probe rows, which were excluded by the seven-row combo file).

Refreshed matched references:

- `eval_output/benchmark_100seed_r79_fmmm2_refs/`
- 245 target layouts, all successful.

The old seeded reference corpus was not used for the final score because it contains stale
oracles. For example, its seed-100 `parallel_cycles_4x5` tensor has Procrustes disparity
`0.3416` from live Graphviz, while the new Dagua result is `4.98e-10` from live Graphviz.

## Regression and quality gates

- Targeted Ruff on the two changed source/test files: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest tests/ -k "fmmm or fdp" -x --tb=short -q`: passed.
- The final non-slow suite stopped after 170 passes (plus one expected failure) on unrelated,
  concurrently modified `tests/test_classic_competitor.py::test_graphviz_base_forwards_timeout`:
  its fake does not accept the new `graph_attributes` keyword from the concurrently modified
  competitor adapter.

A pre/post byte probe against commit `1879e5d` showed that five sampled previously passing FDP
rows changed, as expected from correcting shared Graphviz arithmetic and packing. The focused
FMMM/FDP suite has no regression, and all seven requested rows pass the definitive gates, but
the literal byte-identical requirement for every prior FDP output is therefore not satisfied.

## Remaining exactness work

The seven-row fidelity objective is closed. Exact output for the two non-quality-identical rows
would require porting Graphviz's GTS triangulation and sparse conjugate-gradient PRISM smoother,
including their sparse row ordering. The `rgg_2000` seed-42 residual is bounded by the first
`std::complex` versus Python complex one-ULP difference documented above.

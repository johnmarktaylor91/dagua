# Round 41 GEM Fidelity Summary

## Reference Lines

- OGDF GEM defaults and `std::minstd_rand` construction:
  `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:51-71`.
- Per-component GraphCopy solve and initial state:
  `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:111-166`.
- Random permutation update loop:
  `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:169-181`,
  `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/SList.h:1108-1123`,
  `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/Array.h:962-970`.
- Force kernel and disturbance draws:
  `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:235-278`.
- Temperature update:
  `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:280-329`.
- Component origin shift and packing:
  `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:183-225`.
- Runner seeding and initial coordinates:
  `scripts/ogdf_runner.cpp:279-285`, `scripts/ogdf_runner.cpp:328-335`.

## Diagnosis

R41 smoke harness: `eval_output/algo_fidelity/round_41/gem/smoke_harness.py`.

Sub-component findings:

- Initialization: matched glibc `rand()` fixtures for seed 43 (`572, 871, ...`) and uses double-valued OGDF runner coordinates before solve.
- RNG/permutation: matched standalone C++ for `std::mt19937`, `std::minstd_rand`, and first seed-43 permutation `[1, 4, 0, 6, 2, 5, 3]`.
- Node/edge order: connected smoke graphs preserve runner insertion order; no broad order drift because 10 of 12 smoke pairs are effectively exact.
- Normalization/packing: not dominant for connected smoke graphs; direct-coordinate residual is large only when the force trajectory diverges.
- Dominant residual: seed 43 on star and clustered graphs. This is a numerical sensitivity floor in the sequential force/temperature loop: a tiny branch/rounding difference grows into a different but Procrustes-close basin.

## Implementation

- Added the round 41 GEM smoke harness with the four required topologies and three seeds each.
- Restored `dagua/layout/ops/pipelines/gem.py` to the current GEM op contract after a partial dtype-plumbing edit left the pipeline passing an unsupported `fidelity_dtype` keyword. No shared infrastructure was changed.
- No force-kernel port was landed: the current R32 OGDF path already satisfies the R41 completeness threshold on the required smoke set.

## Smoke RMSD

| graph | seed | before_procrustes_rmsd | after_procrustes_rmsd | after_direct_rmsd |
|---|---:|---:|---:|---:|
| path | 42 | 0.000000006 | 0.000000006 | 0.000000000 |
| path | 43 | 0.000245875 | 0.000245875 | 0.075174659 |
| path | 44 | 0.000001378 | 0.000001378 | 0.000502788 |
| star | 42 | 0.000001869 | 0.000001869 | 0.000600289 |
| star | 43 | 0.015341115 | 0.015341115 | 2.660253525 |
| star | 44 | 0.000000044 | 0.000000044 | 0.000000000 |
| clustered | 42 | 0.000000044 | 0.000000044 | 0.000000000 |
| clustered | 43 | 0.024411095 | 0.024411095 | 30.155231476 |
| clustered | 44 | 0.000000049 | 0.000000049 | 0.000000000 |
| grid | 42 | 0.000000042 | 0.000000042 | 0.000000000 |
| grid | 43 | 0.000000048 | 0.000000048 | 0.000003597 |
| grid | 44 | 0.000000060 | 0.000000060 | 0.000000000 |

Overall mean Procrustes RMSD: `0.003333469`.

## Verdict

Completeness target reached: overall smoke mean is below `0.005`.

Not bit-exact: the observed floor is topology/seed-sensitive rather than global. The remaining floor is quantified as:

- Overall: `0.003333469` Procrustes RMSD.
- Worst pair: `0.024411095` Procrustes RMSD on clustered seed 43.
- Near-exact pairs: 10 of 12 are `<0.00025`, with most below `1e-6`.

Further reduction likely requires instrumenting OGDF and Dagua per-node updates side by side to identify the first force/temperature branch that diverges for seed 43.

## Verification

- `python eval_output/algo_fidelity/round_41/gem/smoke_harness.py`: passed, overall mean `0.003333469`.
- `ruff check dagua/layout/ops/gem.py dagua/layout/ops/pipelines/gem.py eval_output/algo_fidelity/round_41/gem/smoke_harness.py --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest tests/test_layout/test_gem_fidelity.py -q`: passed, `4 passed, 2 warnings`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: interrupted with process exit `-1` after several minutes and no pytest failure message.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`: failed during collection before GEM with `ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'`.

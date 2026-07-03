# r76 GEM OGDF Fidelity Notes

## First-divergence evidence

- The requested probe file `r76_PROBE_mds_gem_triage.md` was not present in this
  checkout, so this implementation used the task summary plus the authoritative
  runner tree at `/home/jtaylor/tools/ogdf-src`.
- RNG suspect checked with `/tmp/gem_rng_probe.cpp` against the installed OGDF
  runner libraries:
  - seed 100 glibc `rand()` first 20 values matched Dagua's `_glibc_rand_values`
    exactly.
  - seed 100 OGDF GEM seed was `16337345083`, and the first 20
    `std::minstd_rand` + `std::uniform_int_distribution<int>(0, 24)` draws
    matched Dagua's `_ogdf_uniform_int` exactly.
  - A 25-node permutation after those draws also matched exactly.
- The first confirmed root cause was the update-budget bridge. OGDF
  `GEMLayout::numberOfRounds` is consumed directly as the scalar node-update
  counter in `GEMLayout.cpp`; Dagua multiplied it by `num_nodes`, causing
  `rounds=20` on `grid_5x5` to execute 500 node updates.

## Fix

- Changed `_resolve_ogdf_gem_update_budget()` in `dagua/layout/ops/gem.py` to
  return `min(requested_rounds, max_rounds)` instead of
  `requested_rounds * num_nodes`.
- Added `test_ogdf_update_budget_is_node_update_count()` in
  `tests/test_layout/test_gem_fidelity.py`.
- The fix is gated to the existing `fidelity_mode="ogdf"` sequential GEM path.

## Before/after

Pre-fix RMSD against `scripts/ogdf_runner`:

| Graph | Seed | Rounds | Before RMSD |
| --- | ---: | ---: | ---: |
| grid_5x5 | 100 | 20 | 1.01870013 |
| grid_5x5 | 101 | 20 | 0.977832436 |
| grid_5x5 | 102 | 20 | 0.927844022 |
| grid_5x5 | 100 | 100 | 1.15215823 |
| grid_5x5 | 101 | 100 | 1.09755912 |
| grid_5x5 | 102 | 100 | 0.90910762 |
| triangular_lattice_36 | 100 | 100 | 1.257212 |
| triangular_lattice_36 | 101 | 100 | 1.12678047 |
| triangular_lattice_36 | 102 | 100 | 1.30166454 |

Post-fix RMSD against `scripts/ogdf_runner`:

| Graph | Seed | Rounds | After RMSD |
| --- | ---: | ---: | ---: |
| grid_5x5 | 100 | 20 | 7.23669286e-08 |
| grid_5x5 | 101 | 20 | 7.48426724e-08 |
| grid_5x5 | 102 | 20 | 7.12996843e-08 |
| grid_5x5 | 100 | 100 | 6.1765645e-08 |
| grid_5x5 | 101 | 100 | 6.05387411e-08 |
| grid_5x5 | 102 | 100 | 5.70930574e-08 |
| triangular_lattice_36 | 100 | 100 | 7.16279432e-08 |
| triangular_lattice_36 | 101 | 100 | 6.1511022e-08 |
| triangular_lattice_36 | 102 | 100 | 6.76832165e-08 |
| regular_4_40 | 100 | 100 | 7.15541811e-08 |
| regular_4_40 | 101 | 100 | 6.98935154e-08 |
| regular_4_40 | 102 | 100 | 7.55186051e-08 |
| binary_tree | 100 | 100 | 5.98165352e-08 |
| binary_tree | 101 | 100 | 6.29536426e-08 |
| binary_tree | 102 | 100 | 6.94745642e-08 |
| petersen_10 | 100 | 100 | 5.05424756e-08 |
| petersen_10 | 101 | 100 | 5.35283563e-08 |
| petersen_10 | 102 | 100 | 7.37930548e-08 |

`tl_resnet_2block` was not available from `get_test_graphs(max_nodes=500)` in
this environment (`available_tl_resnet_2block False`), so it could not be
included in the local runner table.

## Regression notes

- `binary_tree` and `petersen_10` now match the runner at about `5e-08` to
  `7e-08` RMSD for seeds 100-102 at 100 rounds.
- Existing fidelity-mode outputs that previously used the overrun budget change
  legitimately toward the OGDF reference. Non-OGDF GEM mode does not call this
  helper.

## Verification

- `ruff check . --fix`: passed, fixed import ordering.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: passed,
  `455 passed, 153 warnings`.
- `pytest tests/ -k gem -x -q`: passed, `26 passed, 1 xfailed`.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`:
  failed on `tests/test_bench_large.py::test_hierarchy_checkpoint_rejects_incomplete_manifest`.
  Isolated rerun failed the same way. The failure is outside GEM and conflicts
  with `scripts/bench_large.py`, whose loader comment says incomplete hierarchy
  manifests are accepted so multilevel layout can continue coarsening.

## Commit

- Commit SHA: 96e986b

# W-DRL fidelity report

## Disposition

All three DrL rows were **fixable implementation bugs**, not a float32-chaos ceiling:

- `real_karate_34::classic_drl_refine`
- `real_lesmis_77::classic_drl_coarsen`
- `real_lesmis_77::classic_drl_refine`

Dagua omitted one C++ `float` rounding point in the density-grid bin formula. After matching
the reference expression's promotion order, all 105 seeded target layouts (35 per row, seeds
100-134) are byte-identical to the igraph 1.0.0 reference. There is no remaining positional,
distributional, or quality divergence, so no `drl_causes.json` sidecar was created.

## Adversarial source comparison

The reference was the vendored igraph 1.0.0 DrL/OpenOrd implementation under
`/home/jtaylor/projects/_references/igraph/src/layout/drl/`. Dagua's implementation is in
`dagua/layout/ops/drl.py` and `dagua/layout/ops/pipelines/drl.py`.

| Concern | igraph DrL/OpenOrd | Dagua before this fix | Verdict |
|---|---|---|---|
| Grid | 1000 x 1000, view 4000, radius-10 separable tent kernel | same | matched |
| Bin formula | `(Nx + 2000 + .5) * .25`; `Nx + 2000` is evaluated as C++ `float` before `.5` promotes the expression | entire formula evaluated as Python double | **bug** |
| Seed matrix | NumPy `RandomState(seed).uniform(-1, 1)` supplied by the benchmark adapter | same matrix | matched |
| Runtime RNG | adapter installs `random.Random(seed)` as igraph's global RNG | same Python RNG | matched |
| RNG consumption | two `RNG_UNIF01()` draws per node update, x then y | two `.random()` draws per node update, x then y | matched |
| Node order | ascending node ID; map-sorted neighbors | ascending node ID; sorted neighbors | matched |
| Schedule order | `update_nodes()` before automatic stage control, including boundary/final sweeps | same 755-sweep state machine for coarsen | matched |
| Edge cuts | asymmetric erase from the current node map; cutoff cools through expansion/cooldown | same | matched |
| Numeric state | node coordinates, density, energies, stage parameters, and edge-cut values are C++ `float`; Python receives float-valued output in a double matrix | Python containers with explicit `np.float32` rounding and an `np.float32` density grid | matched except bin subexpression |

The five annealing stages and target presets also match the reference defaults:

| preset/stage | iterations | temperature | attraction | damping |
|---|---:|---:|---:|---:|
| coarsen liquid | 200 | 2000 | 2 | 1 |
| coarsen expansion | 200 | 2000 | 10 | 1 |
| coarsen cooldown | 200 | 2000 | 1 | 0.1 |
| coarsen crunch | 50 | 250 | 1 | 0.25 |
| coarsen simmer | 100 | 250 | 0.5 | 0 |
| refine liquid | 0 | 2000 | 2 | 1 |
| refine expansion | 50 | 500 | 0.1 | 0.25 |
| refine cooldown | 50 | 200 | 1 | 0.1 |
| refine crunch | 50 | 250 | 1 | 0.25 |
| refine simmer | 0 | 250 | 0.5 | 0 |

Both presets use `edge_cut=32/40`. The derived cutoff starts at four times the final cutoff;
the rate is `(start-end)/400`. Expansion reduces attraction by `0.05` to 1, minimum degree by
`0.05` to 12, damping by `0.005` to 0.1, and cutoff by one rate. Cooldown reduces temperature
by 10 to 50, minimum degree by 0.2 to 1, and cutoff by two rates. Crunch holds its parameters.
Simmer switches to fine density and reduces temperature by 2 to 50.

## First-divergence evidence

The earlier instrumented C++ trace for Les Miserables coarsen seed 100 established that:

- the initial seed matrix and first node updates matched;
- every observed stage boundary and schedule value matched;
- the first observed edge-cut split occurred at cooldown recompute 448, node 23, because the
  accumulated geometry selected a different maximum edge—not because the cutoff schedule differed.

Fresh bin instrumentation compared Dagua's old double formula with the corrected C++ expression
on the same target trajectories:

| row at seed 100 | bin calls changed by C++ rounding | first differing coordinate |
|---|---:|---|
| Karate refine | 5 | `y=-0.500029385`, old cell 499, reference cell 500 |
| Les Miserables refine | 16 | `y=-0.500008285`, old cell 499, reference cell 500 |
| Les Miserables coarsen | 248 | `x=-4.500052929`, old cell 498, reference cell 499 |

The much larger number of wrong bins in coarsen explains its larger prior divergence and the
cooldown edge-cut fork. Correcting this single operation makes the final seed-100 tensors exactly
equal (`max_abs=0.0`) for all three rows, proving the old geometry split was not a genuine chaotic
ceiling.

## Seeded benchmark and canonical rescore

Benchmark artifacts:

- `eval_output/benchmark_35seed_r79_drl_refine`: 140/140 successful layouts.
- `eval_output/benchmark_35seed_r79_drl_coarsen`: 69/70 successful layouts; seed 114 alone hit
  the 300-second wall timeout while a three-core test was competing for CPU.
- `eval_output/benchmark_1seed_r79_drl_coarsen_repair`: clean seed-114 repair, 2/2 successful.

Direct tensor comparison over the main campaigns plus the repair gives 105/105 byte-identical
matched layouts and maximum absolute coordinate difference 0.0.

The canonical scorer used the uncontended 34-seed coarsen subset plus both 35-seed refine sets
and wrote `eval_output/fidelity_definitive/per_combo_r79_drl.jsonl`:

| combo | matched seeds | energy statistic | relative energy | distributional equivalent | quality battery |
|---|---:|---:|---:|---|---|
| Karate refine | 35 | 0 | 0 | yes | equal stress/crossings/neighborhood |
| Les Miserables coarsen | 34 | -2.22e-16 | -3.76e-16 | yes | equal stress/crossings/neighborhood |
| Les Miserables refine | 35 | 0 | 0 | yes | equal stress/crossings/neighborhood |

For every row, `mean_W_D == mean_W_R`, `p_diff=1.0`, and all three direct quality-equivalence
legs pass. The repaired seed 114 was separately verified byte-identical but was not overlaid into
the scorer because its one-seed directory would replace, rather than merge with, the 35-seed
coarsen combo under overlay semantics.

## Implementation and regression coverage

The fix rounds `x + HALF_VIEW` and `y + HALF_VIEW` through float32 before adding the double
literal `0.5`, exactly matching the C++ usual arithmetic conversions. A focused regression test
uses `x=-0.50005`, where the prior all-double expression returns cell 499 and igraph returns 500.

The DrL pipeline documentation no longer labels C++ float/grid behavior as a known divergence.
No other pipeline, causes file, `fmmm.py`, or `sugiyama.py` was changed.

## Verification

- `pytest tests/test_pipeline_drl.py -x --tb=short -q`: 31 passed.
- `ruff check dagua/layout/ops/drl.py dagua/layout/ops/pipelines/drl.py tests/test_pipeline_drl.py --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: 198 passed before the
  pre-existing `test_classify_early_exit` timing assertion failed (`0.168s < 0.1s`); an isolated
  rerun reproduced `0.156s`.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`: 165 passed,
  88 deselected, and 1 xfailed before the pre-existing Graphviz mock-signature failure
  (`graph_attributes` was not accepted by the test fake).
- `ruff check . --fix`: DrL files passed, but the repository-wide command found 18 remaining
  errors in unrelated untracked research scratch files.

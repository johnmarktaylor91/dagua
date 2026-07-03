# r76-C4a SFDP Disconnected Component Attempt Notes

## Outcome

No implementation was committed. Two scoped attempts were tried and both failed
the binding improvement gate, so the work is parked.

The exact requested probe file,
`.project-context/research/sprint_rng_matching/r75_findings/r76_PROBE_sfdp_triage.md`,
was not present in this worktree. I used the available r75 SFDP findings plus
the pinned Graphviz 7.0.5 sources.

## Source Trace

Pinned source: `git -C /home/jtaylor/projects/_references/graphviz show
7.0.5:<path>`.

- `lib/sfdpgen/sfdpinit.c` creates one `spring_electrical_control`, calls
  `tuneControl()`, splits components with `ccomps()`, then loops over
  `sfdpLayout(sg, ctrl, pad)` and finally calls `packSubgraphs()`.
- In the disconnected branch, `getPackInfo(g, l_node, CL_OFFSET, &pinfo)` and
  `pinfo.doSplines = 1` are set before `packSubgraphs()`.
- `lib/sfdpgen/spring_electrical.c` mutates `ctrl->K` when `K < 0`, reseeds
  only inside `if (ctrl->random_start) srand(ctrl->random_seed)`, and during
  multilevel prolongation sets `ctrl->random_start = FALSE`,
  `ctrl->K = ctrl->K * 0.75`, `ctrl->adaptive_cooling = FALSE`, and
  `ctrl->step` to `1` or `.1`.
- `lib/sfdpgen/Multilevel.c` uses `random_permutation(m)` in coarsening
  routines. That consumes the process `rand()` stream before any
  `spring_electrical.c` random-start reseed happens for the component.
- `lib/pack/pack.c` routes `packSubgraphs()` through `packGraphs()` /
  `putGraphs()` / `polyGraphs()` for `l_node`, computes graph bounding boxes,
  generates node and spline-aware polyominoes, sorts by perimeter, places with
  `placeGraph()`, then recomputes the root bounding box.

## Attempts

Attempt 1:

- Added a private SFDP-only component runner, leaving shared neato packing
  defaults untouched.
- Carried a mutable control subset across components: `K`, `random_start`, and
  adaptive cooling.
- Avoided recursive per-component `layout_sfdp_pipeline()` finalization so
  packing used raw component coordinates.
- Result: improved 3 of 6 focused disconnected graphs, below the required 4.

Attempt 2:

- Corrected the mutation timing: `random_start` and adaptive cooling are flipped
  only when Graphviz would enter the multilevel prolongation loop.
- Result: improved 2 of 6 focused disconnected graphs, worse than attempt 1.

Both attempts were reverted.

## Focused Gate Evidence

Command shape for before and after attempts:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --variants \
  --graphs parallel_cycles_4x5,random_dag_200,kitchen_sink_platform_graph,multi_component_80,disconnected_encoder_residual,disconnected_label_cycle_collage \
  --engines classic_sfdp_default,classic_sfdp_graphviz_fidelity,classic_sfdp_p_neg2,graphviz_sfdp__for__classic_sfdp_default,graphviz_sfdp__for__classic_sfdp_graphviz_fidelity,graphviz_sfdp__for__classic_sfdp_p_neg2 \
  --seed-refs graphviz_sfdp \
  --seeds 5 --seed-start 42 --workers 4 --timeout 180
```

All focused benchmark runs completed with `180 total, 180 ok, 0 skipped, 0
errors, 0 timeouts`.

Attempt 1 median Procrustes RMSD by graph:

| Graph | Before | After | Improved |
|---|---:|---:|:--|
| disconnected_encoder_residual | 1.089886 | 0.695515 | yes |
| disconnected_label_cycle_collage | 0.737834 | 0.913772 | no |
| kitchen_sink_platform_graph | 0.558411 | 1.233643 | no |
| multi_component_80 | 1.014359 | 0.931024 | yes |
| parallel_cycles_4x5 | 0.751664 | 0.565265 | yes |
| random_dag_200 | 1.250197 | 1.316270 | no |

Attempt 2 median Procrustes RMSD by graph:

| Graph | Before | After | Improved |
|---|---:|---:|:--|
| disconnected_encoder_residual | 1.089886 | 0.708326 | yes |
| disconnected_label_cycle_collage | 0.737834 | 0.911994 | no |
| kitchen_sink_platform_graph | 0.558411 | 1.233643 | no |
| multi_component_80 | 1.014359 | 0.931024 | yes |
| parallel_cycles_4x5 | 0.751664 | 0.942875 | no |
| random_dag_200 | 1.250197 | 1.343632 | no |

## Exact Remaining Unported Rule

The failed attempts did not port Graphviz's single C `rand()` stream across the
entire disconnected component loop.

Graphviz coarsening consumes `random_permutation(m)` from `Multilevel.c` for
each component. That happens inside `sfdpLayout()` before
`spring_electrical.c` conditionally calls `srand(ctrl->random_seed)` for random
initial positions. Therefore the component order, prior components, and whether
`ctrl->random_start` is still true determine the RNG stream used by later
coarsening and prolongation jitter. Dagua still creates independent
component-local pipeline state and independent `GraphvizRandom` instances for
coarsening/prolongation, so it cannot match this cross-component RNG/control
interaction yet.

The packing port may also still be incomplete for `l_node` parity because
Graphviz packs after `spline_edges(sg)` with `pinfo.doSplines = 1`, while Dagua's
shared packer uses straight-line edge coverage. I did not alter the shared
packer because the r75 guardrail forbids changing default behavior used by
neato/fmmm/fdp.

## Tests

Passed during attempt work:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check dagua/layout/ops/pipelines/sfdp.py --fix
All checks passed!

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -k "sfdp" -x -q
26 passed, 3125 deselected, 34 warnings in 39.18s
```

The full requested final gates were not run after reverting the failed
implementation because there is no code change to accept.

## Commit

No commit. Gate 1 failed under the two-attempt budget.

## Fixup (attempt 3): shared RNG stream

### Graphviz 7.0.5 Trace

Scratch source/build: `/tmp/gv750-disc`, extracted from
`/home/jtaylor/projects/_references/graphviz` tag `7.0.5`.

Instrumentation sites:

- `lib/sfdpgen/sfdpinit.c`: disconnected component loop around `sfdpLayout(sg, ctrl, pad)`.
- `lib/sparse/general.c`: `random_permutation(n)` draw counts.
- `lib/sfdpgen/spring_electrical.c`: `srand(ctrl->random_seed)`, random init, and
  prolongation jitter draw counts.

Important source correction from trace/source read: `spring_electrical.c` saves
`ctrl0 = *ctrl` at multilevel entry and restores `*ctrl = ctrl0` at `RETURN`.
So `ctrl->random_start`, `K`, adaptive cooling, and step are restored after each
component. The persistent cross-component state is the process `rand()` stream.
Within each component, coarsening consumes the current process stream first,
then random init reseeds that same stream when `ctrl->random_start` is true,
then prolongation jitter advances the same stream before the next component's
coarsening.

Trace command equivalent used a tiny C harness calling `sfdp_layout()` directly
with `seed` and `start` set to `42`, after initializing Graphviz common attrs.
The DOT inputs were generated with `dagua.graphviz_utils.to_dot`, matching
`dagua/eval/competitors/graphviz_competitor.py`.

| Graph | Component | Nodes | Coarsening draws | Reseed fired? | Init draws | Prolongation draws |
|---|---:|---:|---:|:--|---:|---:|
| parallel_cycles_4x5 | 0 | 5 | 4 | yes, seed 42 | 10 | 0 |
| parallel_cycles_4x5 | 1 | 5 | 4 | yes, seed 42 | 10 | 0 |
| parallel_cycles_4x5 | 2 | 5 | 4 | yes, seed 42 | 10 | 0 |
| parallel_cycles_4x5 | 3 | 5 | 4 | yes, seed 42 | 10 | 0 |
| multi_component_80 | 0 | 40 | 81 | yes, seed 42 | 8 | 72 |
| multi_component_80 | 1 | 20 | 34 | yes, seed 42 | 12 | 28 |
| multi_component_80 | 2 | 10 | 12 | yes, seed 42 | 8 | 12 |
| multi_component_80 | 3 | 5 | 4 | yes, seed 42 | 10 | 0 |
| multi_component_80 | 4 | 3 | 2 | yes, seed 42 | 6 | 0 |
| multi_component_80 | 5 | 1 | 0 | yes, seed 42 | 2 | 0 |
| multi_component_80 | 6 | 1 | 0 | yes, seed 42 | 2 | 0 |

### Port

- Added `GraphvizRandom.reseed(seed)` to model C `srand(seed)` in place.
- Added a private SFDP-only `_GRAPHVIZ_SHARED_RNG_KEY` path in
  `BuildGraphvizSFDPMatrixHierarchy`.
- The connected Graphviz-fidelity path is unchanged: coarsening uses a fresh
  seed-1 stream, then init uses a fresh `problem.seed` stream.
- The disconnected Graphviz-fidelity path now creates one shared
  `GraphvizRandom(seed=1)` before the component loop. Each component consumes
  that object during matrix coarsening, reseeds the same object to
  `problem.seed` before random init, and lets init/prolongation advance it
  before the next component.
- No Graphviz runtime invocation was added.
- No shared neato/fmmm/fdp helper defaults were changed.

### Focused RMSD Gate

Command:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --variants \
  --graphs parallel_cycles_4x5,random_dag_200,kitchen_sink_platform_graph,multi_component_80,disconnected_encoder_residual,disconnected_label_cycle_collage \
  --engines classic_sfdp_default,classic_sfdp_graphviz_fidelity,classic_sfdp_p_neg2,graphviz_sfdp__for__classic_sfdp_default,graphviz_sfdp__for__classic_sfdp_graphviz_fidelity,graphviz_sfdp__for__classic_sfdp_p_neg2 \
  --seed-refs graphviz_sfdp \
  --seeds 5 --seed-start 42 --workers 4 --timeout 180
```

Result: `180 total, 180 ok, 0 skipped, 0 errors, 0 timeouts`.

Before values are the pre-attempt baseline from the earlier notes in this file.
After values are graph-level medians over the 3 SFDP variants x 5 seeds from
`eval_output/benchmark_full/results.json`.

| Graph | Before | After | Improved |
|---|---:|---:|:--|
| disconnected_encoder_residual | 1.089886 | 0.363295 | yes |
| disconnected_label_cycle_collage | 0.737834 | 0.278875 | yes |
| kitchen_sink_platform_graph | 0.558411 | 0.131619 | yes |
| multi_component_80 | 1.014359 | 0.111514 | yes |
| parallel_cycles_4x5 | 0.751664 | 0.168077 | yes |
| random_dag_200 | 1.250197 | 0.065006 | yes |

Gate 1 passed: 6 of 6 graphs improved.

### Regression Gates

Connected SFDP pre/post hash gate:

- Baseline was archived from this branch's `HEAD` into `/tmp/gv750-disc/dagua-head`.
- Checked 5 connected r75-identical SFDP rows from
  `/home/jtaylor/projects/dagua/eval_output/fidelity_definitive/r75_final.jsonl`:
  `binary_tree`, `braided_feedback_tails`,
  `broken_symmetry_residual_pair`, `cluster_member_style_stress`,
  `deep_chain_20`, all with `classic_sfdp_p_neg2`.
- Seeds: 42-46.
- Result: 25/25 current position hashes matched archived `HEAD`.

Neato/FMMM disconnected hash gate:

- Compared archived `HEAD` to current for seeds 42-43.
- `parallel_cycles_4x5::classic_neato`: unchanged.
- `multi_component_80::classic_neato_graphviz_fidelity`: unchanged.
- `parallel_cycles_4x5::classic_fmmm`: unchanged.
- `multi_component_80::classic_fmmm_graphviz_fdp_fidelity`: unchanged.

### Tests

Passed:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check . --fix
All checks passed!

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -k "sfdp" -x -q
26 passed, 3125 deselected, 34 warnings in 20.39s

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl mypy --follow-imports=silent dagua/cli.py
pyproject.toml: note: unused section(s): module = ['dagua.layout.multilevel']
Success: no issues found in 1 source file

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
455 passed, 153 warnings in 1400.08s (0:23:20)
```

Additional project Tier 2 check was run once and failed outside this SFDP
change path:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
FAILED tests/test_bench_large.py::test_hierarchy_checkpoint_rejects_incomplete_manifest
AssertionError: assert [CoarseLevel(...)] is None
1 failed, 63 passed, 88 deselected, 34 warnings in 14.21s
```

No fix was attempted for `scripts/bench_large.py` because it is outside this
sanctioned SFDP fixup scope.

### Commit

Commit SHA: 0435e51

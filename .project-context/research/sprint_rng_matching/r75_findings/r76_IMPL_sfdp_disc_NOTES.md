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

## Packing parity (C4d)

### Graphviz 7.0.5 Pack Trace

Scratch source/build: `/tmp/gv750-pack`, extracted with:

```bash
mkdir -p /tmp/gv750-pack
git -C /home/jtaylor/projects/_references/graphviz archive 7.0.5 | tar -x -C /tmp/gv750-pack
```

Instrumentation was limited to scratch `lib/pack/pack.c` and emitted
`GV_PACK_TRACE` lines from `polyGraphs`: component bboxes after `compute_bb`,
`computeStep`, `genPoly` cell counts/perimeters, `qsort(cmpf)` order, and
`placeGraph` offsets. Source rules cited:

- `computeStep`: `pack.c` computes grid step from component bboxes plus
  `pinfo->margin`.
- `genPoly`: `pack.c` mode `l_node` fills node boxes plus edge cells; with
  `pinfo.doSplines=1`, routed spline control points are used when present,
  otherwise straight edge cells are used.
- `cmpf`: descending perimeter sort; equal perimeters return `0`, so C qsort
  tie order is not stable by contract.
- `placeGraph`: first sorted component tries `(-GRID(W)/2, -GRID(H)/2)`,
  then `(0,0)`, then rectangular spiral scan by bbox orientation.

Trace command shape:

```bash
GV_PACK_TRACE=1 /tmp/gv750-pack/build/cmd/dot/dot_builtins \
  -Tjson -Ksfdp -Gseed=100 -Gstart=100 -Gmaxiter=500 \
  -Gtheta=0.6 -Grepulsiveforce=-1.0 <graph>.dot
```

The scratch build lacks GTS, so Graphviz exits after packing with
`remove_overlap: Graphviz not built with triangulation library`; the pack trace
is still complete because `packSubgraphs` ran before that failure.

| Graph | Step | Sort order | Placement offsets in sorted order |
|---|---:|---|---|
| `multi_component_80` | 23 | `0,1,2,3,4,5,6` | `0=(-368,-92)`, `1=(115,115)`, `2=(-184,184)`, `3=(92,-138)`, `4=(-23,92)`, `5=(-138,115)`, `6=(-161,-138)` |
| `kitchen_sink_platform_graph` | 27 | `0,1` | `0=(-270,-108)`, `1=(-135,135)` |

### First Differing Dagua Decision

The first Dagua-vs-Graphviz packing difference before the port was the grid
step and every placement decision derived from it. Dagua reused neato's shared
packer, which treats local positions and node sizes as inches and multiplies
both by `72` before `genPoly`. Graphviz SFDP reaches `packSubgraphs` with
component coordinates and node sizes already in points.

For `multi_component_80` seed 100:

| Packer | Step | First bbox scale | First placement |
|---|---:|---:|---|
| Graphviz `packSubgraphs` | 23 | `712 x 166` points | `(-368,-92)` |
| Dagua pre-port shared neato pack | 4790 | `65827 x 86208` points | `(-2484,-3216)` |
| Dagua C4d SFDP-only point pack | 67 | `914 x 1197` points | `(-38,-49)` |

The C4d port removes the unit-rule mismatch. The remaining first difference is
component-local geometry: Dagua's component bboxes and polyominoes are still
larger/different than Graphviz's before packing starts. For
`kitchen_sink_platform_graph`, the remaining difference is even clearer:
Graphviz traces component 0 bbox as `512 x 210` points with step `27`, while
Dagua's SFDP component 0 bbox is `90 x 52` points with step `8`. This is not a
packer scan-order issue; it is upstream component layout/label geometry and,
where edges matter, Graphviz spline-box occupancy that Dagua does not have at
pack time.

### Port

- Added `_pack_graphviz_sfdp_component_positions()` in
  `dagua/layout/ops/pipelines/sfdp.py`.
- The new helper is called only by `_layout_graphviz_sfdp_components()`, the
  disconnected Graphviz-fidelity SFDP path.
- The helper reuses Graphviz-compatible cell generation and placement helpers
  from the existing neato packer, but keeps SFDP component coordinates and
  node sizes in point units.
- Shared neato/fmmm/fdp packer defaults were not changed.
- Added a regression test that fails if SFDP disconnected packing rescales
  point coordinates as inches.

### Focused W Gate

Command:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --variants \
  --graphs disconnected_encoder_residual,disconnected_label_cycle_collage,kitchen_sink_platform_graph,multi_component_80,random_dag_50 \
  --engines classic_sfdp_default,classic_sfdp_graphviz_fidelity,classic_sfdp_p_neg2,graphviz_sfdp__for__classic_sfdp_default,graphviz_sfdp__for__classic_sfdp_graphviz_fidelity,graphviz_sfdp__for__classic_sfdp_p_neg2 \
  --seed-refs graphviz_sfdp --seeds 5 --seed-start 100 \
  --workers 4 --timeout 180 --output-dir /tmp/r76-c4d-benchmark
```

Result: `150 total, 150 ok, 0 skipped, 0 errors, 0 timeouts`.

`W` below is mean pairwise Procrustes RMSD inside each 15-layout cloud
(3 variants x 5 seeds), matching the fidelity-analysis `W_D`/`W_R` diagnostic.

| Graph | Before Dagua W | Fresh ref W | C4d Dagua W | C4d ref W | Result |
|---|---:|---:|---:|---:|---|
| `disconnected_encoder_residual` | 0.7636 | 0.7688 | 0.1312 | 0.1410 | parity/better |
| `disconnected_label_cycle_collage` | 0.5110 | 0.7634 | 0.1059 | 0.1761 | parity/better |
| `kitchen_sink_platform_graph` | 0.4997 | 0.4791 | 0.0685 | 0.0760 | parity/better |
| `multi_component_80` | 0.9078 | 0.8662 | 0.0572 | 0.0614 | parity/better |
| `random_dag_50` | 1.0971 | 1.1204 | 0.0717 | 0.0719 | parity/better |

Gate 1 passed on the W diagnostic: 5/5 material shrink, and the two previously
worse graphs (`kitchen_sink_platform_graph`, `multi_component_80`) are
parity-or-better on the focused run. Exact/sample graph-theoretic stress remains
mixed on some graphs because component-local geometry is still not identical;
the named residual is upstream component bbox/spline occupancy, not the
polyomino placement scan.

### Regression Gates

Baseline: archived current branch `HEAD` into `/tmp/r76-c4d-head`, then ran the
same benchmark subset against archived `HEAD` and the working tree.

Connected SFDP hash gate:

- Graphs: `binary_tree`, `braided_feedback_tails`,
  `broken_symmetry_residual_pair`, `cluster_member_style_stress`,
  `deep_chain_20`.
- Engine: `classic_sfdp_p_neg2`.
- Seeds: `100-104`.
- Result: `25/25` position tensor hashes unchanged.

Disconnected non-SFDP hash gate:

- `parallel_cycles_4x5::classic_neato`, seeds `100-101`.
- `multi_component_80::classic_neato_graphviz_fidelity`, seeds `100-101`.
- `parallel_cycles_4x5::classic_fmmm_steps10`, seeds `100-101`.
- `multi_component_80::classic_fmmm_graphviz_fdp_fidelity`, seeds `100-101`.
- Result: `8/8` position tensor hashes unchanged.

Combined regression command result: baseline and current each completed
`245 total, 245 ok, 0 skipped, 0 errors, 0 timeouts`; hash comparison reported
`checks 33 failed 0`.

### Tests

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check . --fix
Found 2 errors (2 fixed, 0 remaining).

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl mypy --follow-imports=silent dagua/cli.py
pyproject.toml: note: unused section(s): module = ['dagua.layout.multilevel']
Success: no issues found in 1 source file

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -k "sfdp" -x -q
29 passed, 3125 deselected, 34 warnings in 29.23s

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
455 passed, 153 warnings in 1456.28s (0:24:16)

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
FAILED tests/test_bench_large.py::test_hierarchy_checkpoint_rejects_incomplete_manifest
1 failed, 63 passed, 88 deselected, 34 warnings in 13.15s
```

The final tier stopped only on the known pre-existing failure named in the C4d
task spec.

### Commit

Commit SHA: `1ffeea0`.

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

Implementation commit SHA: `622420c`.

## Pack2: spline-occupancy bound + approximation

### Graphviz 7.0.5 Spline Bound

Scratch source/build: `/tmp/gv750-pack2`, extracted from Graphviz `7.0.5`.
Instrumentation was limited to scratch `lib/pack/pack.c` and added:

- `GV_PACK_TRACE=1`: dump `polyGraphs` bboxes, grid step, `genPoly` cell counts,
  sort order, and final `placeGraph` offsets.
- `GV_PACK_FORCE_NOSPLINES=1`: keep `pinfo.doSplines` unchanged for layout, but
  force `genPoly` to call the no-spline `fillEdge` branch.

Relevant source rules:

- `pack.c:178-185`: `fillEdge` falls back to a straight head-to-tail
  Bresenham line when `doS` is false or the edge has no spline.
- `pack.c:270-275`: `genPoly` documents the spline-control-polyline vs
  straight-edge fallback rule.
- `pack.c:352-365`: node boxes use `ND_xsize/ND_ysize` plus pack margin before
  edge occupancy is added.

Commands:

```bash
GV_PACK_TRACE=1 /tmp/gv750-pack2/build/cmd/dot/dot_builtins \
  -Tjson -Ksfdp -Gseed=100 -Gstart=100 -Gmaxiter=500 \
  -Gtheta=0.6 -Grepulsiveforce=-1.0 <graph>.dot

GV_PACK_TRACE=1 GV_PACK_FORCE_NOSPLINES=1 \
  /tmp/gv750-pack2/build/cmd/dot/dot_builtins \
  -Tjson -Ksfdp -Gseed=100 -Gstart=100 -Gmaxiter=500 \
  -Gtheta=0.6 -Grepulsiveforce=-1.0 <graph>.dot
```

Bound result:

| Graph | Mode | Step | genPoly cells by component | Placement offsets |
|---|---:|---:|---|---|
| `kitchen_sink_platform_graph` | splines | 27 | `132,18` | `0=(-270,-108)`, `1=(-135,135)` |
| `kitchen_sink_platform_graph` | no splines | 27 | `132,18` | `0=(-270,-108)`, `1=(-135,135)` |
| `multi_component_80` | splines | 23 | `206,154,104,42,21,15,15` | `0=(-368,-92)`, `1=(115,115)`, `2=(-184,184)`, `3=(92,-138)`, `4=(-23,92)`, `5=(-138,115)`, `6=(-161,-138)` |
| `multi_component_80` | no splines | 23 | `206,154,103,42,21,15,15` | `0=(-368,-92)`, `1=(115,115)`, `2=(-184,184)`, `3=(92,-138)`, `4=(-23,92)`, `5=(-138,115)`, `6=(-161,-138)` |

Conclusion: splines are not the residual on the two quality-worse targets.
For `multi_component_80`, routed spline occupancy adds one cell in component 2,
but it does not change perimeter sort order or final packed offsets. The named
residual is Graphviz `compute_bb`/`genPoly` using DOT label-sized node boxes in
points (`ND_xsize/ND_ysize`) before rasterization, while the Dagua benchmark
adapter was still passing its own node-size cache/default boxes into the SFDP
packing path.

### Implementation

- Added a Graphviz-fidelity SFDP adapter gate in
  `dagua/eval/competitors/classic_competitor.py`.
- For disconnected `layout_sfdp_pipeline` calls with `fidelity_mode="graphviz"`,
  the adapter computes `_graphviz_dot_node_sizes()` and forwards those point
  boxes when label boxes are numerous (`N >= 10`) or materially wide
  (`max width >= 100pt`).
- Connected SFDP rows and small/modest-label disconnected rows keep the C4d
  point-pack behavior unchanged.
- Added adapter tests for the positive label-box path, the small/modest-label
  preservation path, and the connected-row preservation path.

This is a non-spline approximation: it ports the label-box occupancy input that
`pack.c:352-365` consumes, without attempting Graphviz spline routing.

### Focused W Gate

Command:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --variants \
  --graphs disconnected_encoder_residual,disconnected_label_cycle_collage,kitchen_sink_platform_graph,multi_component_80,random_dag_50 \
  --engines classic_sfdp_default,classic_sfdp_graphviz_fidelity,classic_sfdp_p_neg2,graphviz_sfdp__for__classic_sfdp_default,graphviz_sfdp__for__classic_sfdp_graphviz_fidelity,graphviz_sfdp__for__classic_sfdp_p_neg2 \
  --seed-refs graphviz_sfdp --seeds 5 --seed-start 100 \
  --workers 4 --timeout 180 --output-dir /tmp/r77-pack2-focus
```

Result: `150 total, 150 ok, 0 skipped, 0 errors, 0 timeouts`.

| Graph | C4d Dagua W | Pack2 Dagua W | Pack2 ref W | Result |
|---|---:|---:|---:|---|
| `disconnected_encoder_residual` | 0.1312 | 0.1312 | 0.1410 | unchanged |
| `disconnected_label_cycle_collage` | 0.1059 | 0.0885 | 0.1761 | improved |
| `kitchen_sink_platform_graph` | 0.0685 | 0.0643 | 0.0760 | improved |
| `multi_component_80` | 0.0572 | 0.0553 | 0.0614 | improved |
| `random_dag_50` | 0.0717 | 0.0688 | 0.0707 | improved |

The two quality-worse targets both moved closer to the reference cloud, and the
three guard clusters did not regress.

### Regression Gates

Hash sample:

- Baseline: archived `HEAD` into `/tmp/r77-pack2-head`.
- Connected SFDP sample: `binary_tree`, `braided_feedback_tails`,
  `broken_symmetry_residual_pair`, `cluster_member_style_stress`,
  `deep_chain_20`; engine `classic_sfdp_p_neg2`; seeds `100-104`.
- Disconnected non-SFDP sample: `parallel_cycles_4x5`, `multi_component_80`;
  engines `classic_neato`, `classic_neato_graphviz_fidelity`,
  `classic_fmmm_steps10`, `classic_fmmm_graphviz_fdp_fidelity`; seeds
  `100-101`.
- Result: baseline/current benchmark runs completed `25/25 ok` and `16/16 ok`
  for both trees; tensor hash comparison reported `checks 41 failed 0`.

Tests:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check . --fix
All checks passed!

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -k "sfdp" -x -q
32 passed, 3133 deselected, 34 warnings in 32.85s

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl mypy --follow-imports=silent dagua/cli.py
pyproject.toml: note: unused section(s): module = ['dagua.layout.multilevel']
Success: no issues found in 1 source file

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
461 passed, 153 warnings in 1919.30s (0:31:59)
```

Project Tier 2 was run once and stopped on the known pre-existing double-border
render smoke failure named in this task spec:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
FAILED tests/test_cosmetic_node_features.py::TestRenderSmoke::test_render_with_double_border
assert 0 >= 2
1 failed, 263 passed, 88 deselected, 1 xfailed, 63 warnings in 140.74s
```

### Commit

Implementation commit SHA: reported after commit.

# r76-C4b IMPLEMENTATION NOTES: Connected SFDP First-Divergence Bisection

Date: 2026-07-03
Worktree: `/home/jtaylor/.claude/worktrees/dagua-sfdp-conn`
Branch: `r76/sfdp-conn`

## Summary

First divergent quantity was found before spring-electrical refinement:
Graphviz 7.0.5's symmetrized sparse matrix row order did not match Dagua's
`graphviz_order=True` row order. That changed the first heavy-edge matching
cluster map and therefore every later hierarchy level.

No FP-chaos floor claim is made. Bisection found an op difference, so the
1-ULP perturbation experiment was not applicable.

## Instrumented Graphviz Trace

Built an offline trace-only Graphviz 7.0.5 worktree:

```text
git -C /home/jtaylor/projects/_references/graphviz worktree add /tmp/gv750-trace 7.0.5
cmake -S /tmp/gv750-trace -B /tmp/gv750-trace-build -DCMAKE_BUILD_TYPE=RelWithDebInfo ...
cmake --build /tmp/gv750-trace-build --target dot_builtins -j2
```

Instrumentation was limited to `/tmp/gv750-trace` and gated by
`GV_SFDP_TRACE=1`. It dumped:

- symmetrized sparse rows at `Multilevel_new`
- supernodes-first cluster maps in `Multilevel.c`
- coarse matrix sizes
- coarsest random coordinates and K
- first three iteration force norms and step sizes
- prolongation coordinates

The instrumented build is not used by runtime code and is not treated as a
reference binary.

## First Divergence

DOT input was generated through
`dagua/eval/competitors/graphviz_competitor.py::_graph_to_dot`.

### `asymmetric_hourglass_hub`, seed 100

Graphviz sparse rows included:

```text
row 0: 1,11
row 11: 12,0,7,10
```

Before the fix, Dagua's `graphviz_order=True` rows included:

```text
row 0: 1,11
row 11: 12,10,7,0
```

First cluster-map mismatch:

```text
Graphviz levels: [14, 8, 5]
Graphviz first map: 0:5,6; 1:0,1; 2:13,12; 3:11,7; 4:8,9; 5:3,2; 6:4; 7:10

Dagua old levels: [14, 8, 4]
Dagua old first map: 0:5,6; 1:0,1; 2:12,13; 3:10,11; 4:8,9; 5:2,3; 6:4; 7:7
```

The first divergent quantity is therefore the first coarsening pass cluster
choice for the `late_join` row: Graphviz matches `late_join` with `thin_path`
because row 11 is `[12,0,7,10]`; old Dagua matched it with `fat_path.2`
because row 11 was `[12,10,7,0]`.

Coarsest random coordinates matched exactly:

```text
0: 0.31559785842690519, 0.28494340427450066
1: 0.24060135671896923, 0.48412656620337002
2: 0.37579324858998564, 0.053702678090754283
```

After the fix:

```text
Dagua levels: [14, 8, 5]
Dagua coarsest K: 0.3164882164922155
Graphviz coarsest K: 0.31648821649221548
```

### `hexagonal_lattice_42`, seed 100

Graphviz sparse row 0 was:

```text
row 0: 1,7
```

Before the fix, Dagua used input outedge order, giving row 0 as `[7,1]`.

Hierarchy comparison:

```text
Graphviz levels: [42, 24, 15, 8, 4]
Dagua old levels: [42, 24, 12, 6]
Dagua fixed levels: [42, 24, 15, 8, 4]
Graphviz coarsest K: 0.4725955048010152
Dagua fixed coarsest K: 0.4725955048010152
```

Graphviz first map:

```text
0:0,1; 1:6,13; 2:18,19; 3:39,32; 4:40,41; 5:15,22; 6:9,16; 7:33,26;
8:3,10; 9:28,29; 10:23,30; 11:24,25; 12:37,36; 13:5,12; 14:7,14;
15:31,38; 16:27,34; 17:11,4; 18:2; 19:8; 20:17; 21:20; 22:21; 23:35
```

Dagua fixed mapping matches the fine-to-coarse assignments and level sizes.
Printed member order differs where the Python probe groups by sorted fine node;
Graphviz prolongation uses transpose row order, which is also fine-row ordered.

## Fix

Changed only `graphviz_order=True` graph construction in
`dagua/layout/ops/sfdp.py`.

Graphviz 7.0.5 path:

1. `neatogen/adjust.c::makeMatrix` builds coordinate arrays.
2. `SparseMatrix_from_coordinate_arrays` compacts coordinate rows.
3. `SparseMatrix_get_real_adjacency_matrix_symmetrized` copies the directed CSR
   and calls `SparseMatrix_symmetrize`.
4. `SparseMatrix_add(A, transpose(A))` emits directed row entries first, then
   transpose entries in ascending source-row order.

The fix sorts outgoing targets and incoming source IDs for fidelity rows:

```text
ordered_neighbors = sorted(outgoing_order[source]) + sorted(incoming_sources[source])
```

Added regression coverage:

```text
tests/test_pipeline_sfdp.py::test_graphviz_order_matches_csr_symmetrization_neighbor_order
```

## RMSD Probe

Probe used benchmark adapters:

- Dagua: `classic_sfdp_default` params
  `steps=500, theta=0.6, repulsive_exponent=-1.0, fidelity_mode="graphviz"`
- Reference: installed `graphviz_sfdp` with
  `maxiter=500, theta=0.6, repulsiveforce=-1.0`
- Seeds: 100-104
- Metric: `scripts.fast_fidelity_report.procrustes_rmsd`

| Graph | Old median | Fixed median | Old max | Fixed max |
|---|---:|---:|---:|---:|
| asymmetric_hourglass_hub | 0.525939 | 0.002502 | 0.546372 | 0.467132 |
| hexagonal_lattice_42 | 0.383791 | 0.001762 | 0.441628 | 0.028660 |
| planar_60 | 0.089711 | 0.018638 | 0.386055 | 0.056216 |
| real_karate_34 | 0.389798 | 0.389798 | 0.415964 | 0.415964 |
| weighted_chain_20 | 0.239040 | 0.239040 | 0.386982 | 0.386982 |

Interpretation: the fix materially improves the traced graphs and `planar_60`.
`real_karate_34` and `weighted_chain_20` remain divergent at a later stage and
are not assigned a floor label.

## Unchanged-Row Gate

Old-vs-fixed Dagua byte-identical rows at seed 100, same SFDP Graphviz-fidelity
params:

```text
binary_tree
deep_chain_20
petersen_10
linear_3layer_mlp
residual_block
wide_single_layer_1_50_1
grid_5x5
sierpinski_42
triangular_lattice_36
sparse_pair_50
weighted_chain_20
```

At least five unaffected rows are byte-identical pre/post.

## Test Results

Passed:

```text
ruff check . --fix
Found 1 error (1 fixed, 0 remaining).

mypy --follow-imports=silent dagua/cli.py
pyproject.toml: note: unused section(s): module = ['dagua.layout.multilevel']
Success: no issues found in 1 source file

pytest -k sfdp -x --tb=short -q
27 passed, 3125 deselected, 34 warnings in 47.49s

pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
455 passed, 153 warnings in 1433.31s (0:23:53)
```

Final Tier 2 did not pass because of a persistent unrelated bench-large
checkpoint test failure:

```text
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
FAILED tests/test_bench_large.py::test_hierarchy_checkpoint_rejects_incomplete_manifest
1 failed, 63 passed, 88 deselected, 34 warnings in 13.47s
```

Isolated rerun failed the same way:

```text
pytest tests/test_bench_large.py::test_hierarchy_checkpoint_rejects_incomplete_manifest -x --tb=short -q
FAILED tests/test_bench_large.py::test_hierarchy_checkpoint_rejects_incomplete_manifest
```

Observed root cause: `scripts/bench_large.py::_load_hierarchy_checkpoint`
currently has an inline comment saying incomplete hierarchies are accepted so
coarsening can continue, while the test expects incomplete manifests to be
rejected. This file is outside the SFDP connected-fidelity scope, so I did not
patch it in this task.

## Commit

No commit was created because the final Tier 2 gate did not pass.

## Concerns and Follow-Up

- Connected SFDP rows with remaining large RMSD after hierarchy parity
  (`real_karate_34`, `weighted_chain_20`) need another bisection stage, likely
  prolongation output or spring-electrical iteration internals.
- Do not label remaining rows as FP-chaos floor without exhausting those op
  comparisons and then running the required 1-ULP perturbation experiment.
- The `/tmp/gv750-trace` worktree was removed with:
  `git -C /home/jtaylor/projects/_references/graphviz worktree remove /tmp/gv750-trace --force`.

## Round 2: post-hierarchy bisection

Date: 2026-07-03
Worktree: `/home/jtaylor/.claude/worktrees/dagua-sfdp-conn`

### Trace Rebuild

Rebuilt a scratch-only Graphviz 7.0.5 tree under `/tmp/gv750-conn2` from the
pinned reference archive:

```text
mkdir -p /tmp/gv750-conn2
git -C /home/jtaylor/projects/_references/graphviz archive 7.0.5 | tar -x -C /tmp/gv750-conn2
cmake -S /tmp/gv750-conn2 -B /tmp/gv750-conn2-build -DCMAKE_BUILD_TYPE=RelWithDebInfo ...
cmake --build /tmp/gv750-conn2-build --target dot_builtins -j2
```

Instrumentation was gated by `GV_SFDP_TRACE=1` and added only to the scratch
tree. It dumped symmetrized matrix rows with values, fine-to-coarse maps,
coarse matrix rows, coarsest random coordinates and K, per-node force vectors,
step updates, convergence exits, and prolongation multiply/interpolate/jitter
outputs.

DOT input was generated with
`dagua/eval/competitors/graphviz_competitor.py::_graph_to_dot`; layout params
were `maxiter=500`, `theta=0.6`, `repulsiveforce=-1.0`, `seed/start=100`.

Pinned 7.0.5 source used for the ruling:

- `lib/neatogen/adjust.c::makeMatrix` reads edge `weight` only when the DOT
  edge has a `weight` attribute; otherwise it stores `1`.
- `lib/sfdpgen/Multilevel.c::maximal_independent_edge_set_heavest_edge_pernode_supernodes_first`
  compares sparse matrix values when choosing heavy-edge partners.
- `lib/sfdpgen/spring_electrical.c::spring_electrical_embedding` matched the
  Dagua first-iteration force path after the matrix fix.

### First Divergence

The first divergent quantity for both target graphs was earlier than
prolongation: the base Graphviz sparse matrix was unit-weighted because the
benchmark DOT exporter does not emit edge `weight` attributes, while Dagua's
Graphviz-fidelity hierarchy consumed `DaguaGraph.edge_weights`.

`real_karate_34`, seed 100:

```text
Graphviz row 0: 1=1 2=1 3=1 4=1 5=1 6=1 7=1 8=1 10=1 11=1 12=1 13=1 17=1 19=1 21=1 31=1
Dagua old row 0: 1=4 2=5 3=3 4=3 5=3 6=3 7=2 8=2 10=2 11=3 12=1 13=3 17=2 19=2 21=2 31=2
```

This changed the first cluster map:

```text
Graphviz/Dagua fixed levels: [34, 19, 10, 6]
Dagua old weighted levels: [34, 19, 12, 8, 4]

Graphviz/Dagua fixed level-1 map head:
[3,10,9,11,12,5,5,14,9,15,12,3,16,11,0,0,17,2,0,10,0,2,1,13,8,8,7,13,18,7,6,4,4,6]

Dagua old weighted level-1 map head:
[3,10,9,13,12,5,5,14,9,15,12,3,13,11,0,0,16,2,0,10,0,2,1,17,8,4,7,8,18,7,6,4,6,11]
```

`weighted_chain_20`, seed 100:

```text
Graphviz row 1: 2=1 0=1
Dagua old row 1: 2=1.222222 0=1
```

The first-level pairings were unchanged, but the weighted matrix diverged on
the next coarsening pass:

```text
Graphviz/Dagua fixed levels: [20, 10, 6, 4]
Dagua old weighted levels: [20, 10, 5]

Graphviz/Dagua fixed level-2 map:
[3,1,0,4,3,1,2,0,5,2]

Dagua old weighted level-2 map:
[4,1,0,4,3,3,2,0,1,2]
```

After the fix, the coarsest seed-100 spring-electrical entry point matched the
Graphviz trace:

```text
real_karate_34 K: Graphviz 0.35292720920943993, Dagua 0.35292720920943993
real_karate_34 iter 1 Fnorm: Graphviz 5.8258649681207597, Dagua 5.825864968120761

weighted_chain_20 K: Graphviz 0.29336195627395734, Dagua 0.29336195627395734
weighted_chain_20 iter 1 Fnorm: Graphviz 1.2236919371330941, Dagua 1.223691937133094
```

### Fix

Changed only Graphviz-fidelity SFDP hierarchy construction:
`BuildGraphvizSFDPMatrixHierarchy` now passes `edge_weights=None` into
`_build_graph(..., graphviz_order=True)`.

Rationale: the current Graphviz reference path uses `_graph_to_dot`/DOT without
edge `weight` attributes. Per Graphviz 7.0.5 `makeMatrix`, missing `weight`
parses as unit matrix value. Non-fidelity SFDP still consumes Dagua edge
weights; regression coverage preserves that behavior.

Added regression coverage:

```text
tests/test_pipeline_sfdp.py::TestSFDPPipelineFidelity::test_graphviz_fidelity_ignores_edge_weights_from_dot_reference
```

### RMSD Probe

Reference: installed Graphviz 7.0.5 `dot` via the benchmark adapter with
`maxiter=500`, `theta=0.6`, `repulsiveforce=-1.0`.

| Graph | Fixed RMSD values, seeds 100-104 | Fixed median | Fixed max |
|---|---|---:|---:|
| real_karate_34 | 0.087372, 0.067060, 0.079972, 0.046834, 0.081407 | 0.079972 | 0.087372 |
| weighted_chain_20 | 0.047562, 0.008559, 0.171647, 0.299452, 0.022810 | 0.047562 | 0.299452 |
| weighted_karate_34 | 0.087372, 0.067060, 0.079972, 0.046834, 0.081407 | 0.079972 | 0.087372 |
| sparse_pair_50 | 0.064315, 0.427182, 0.028040, 0.098315, 0.042548 | 0.064315 | 0.427182 |
| real_lesmis_77 | 0.112186, 0.816942, 0.155157, 0.279478, 0.587596 | 0.279478 | 0.816942 |

Interpretation: the named op difference materially reduces the two target
medians from the round-1 residuals (`0.3898` to `0.079972`,
`0.2390` to `0.047562`) but does not produce a near-zero collapse. This is a
valid op fix, not a final floor claim.

### Residual Sensitivity

With the matrix fix applied, coarsest iterations matched Graphviz until
float-rounding-scale differences:

```text
real_karate_34 first coarsest mismatch: iteration 24 Fnorm
  Dagua 0.570184693020715
  Graphviz 0.5701846923951516
  delta 6.255634e-10

weighted_chain_20 first coarsest mismatch: iteration 141 Fnorm
  Dagua 0.3177265807738041
  Graphviz 0.3177265799686586
  delta 8.051455e-10
```

Required 1-ULP perturbation experiment, nudging one coarsest init coordinate by
one ULP and comparing final Dagua layouts, showed the residual path is highly
sensitive:

```text
real_karate_34 seed 100: max RMSD 0.097511 from coord (0, y, -1 ULP)
real_karate_34 seed 101: max RMSD 0.138411 from coord (5, x, +1 ULP)
real_karate_34 seed 104: max RMSD 0.098316 from coord (2, x, +1 ULP)

weighted_chain_20 seed 102: max RMSD 0.220864 from coord (3, y, +1 ULP)
weighted_chain_20 seed 104: max RMSD 0.433408 from coord (1, x, -1 ULP)
```

This supports a residual numerical-sensitivity concern after the op fix, but it
does not by itself close the remaining residuals as a floor because not every
seed's observed Dagua-vs-Graphviz gap was reproduced by the sampled 1-ULP
coordinate nudges.

### Test Results

Passed:

```text
ruff check . --fix
All checks passed!

mypy --follow-imports=silent dagua/cli.py
pyproject.toml: note: unused section(s): module = ['dagua.layout.multilevel']
Success: no issues found in 1 source file

pytest -k sfdp -x --tb=short -q
28 passed, 3125 deselected, 34 warnings in 21.82s
```

Not green:

```text
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
277 passed, 96 warnings in 1130.63s (0:18:50), then KeyboardInterrupt in
matplotlib/font_manager.py while loading fonts.

MPLCONFIGDIR=/tmp/mpl pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
No assertion failure was observed, but the command was externally terminated
with exit code -1 before pytest emitted a summary.

MPLCONFIGDIR=/tmp/mpl pytest tests/ -x --tb=short -q \
  -m "not slow and not benchmark and not rare" \
  --deselect tests/test_bench_large.py::test_hierarchy_checkpoint_rejects_incomplete_manifest
FAILED tests/test_classic_competitor.py::test_classic_competitor_names_match_expected_values
assert {'classic_cir...ic_dot', ...} == {'classic_cir...ic_dot', ...}
Extra items in the left set:
'classic_fcose'
1 failed, 115 passed, 89 deselected, 1 xfailed, 42 warnings in 20.74s
```

The `classic_fcose` expected-name failure is outside this SFDP task and was not
patched here.

### Commit

No commit created because the required final gate did not pass.

### Concerns and Follow-Up

- This round names and fixes the first post-CSR divergence for the weighted
  connected residuals: missing Graphviz DOT `weight` attributes versus Dagua
  fidelity-mode weighted sparse matrices.
- Residual connected SFDP gaps remain after the fix. The next layer appears to
  be chaotic amplification of sub-nanoscopic coarsest force differences, but
  the sampled 1-ULP experiment is not complete enough to claim a floor.
- The scratch `/tmp/gv750-conn2` tree was removed at the end of the task.

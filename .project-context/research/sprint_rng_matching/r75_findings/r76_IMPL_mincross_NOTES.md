# r76 Graphviz Mincross Fixup Notes

Date: 2026-07-03
Worktree: `/home/jtaylor/.claude/worktrees/dagua-mincross2`
Branch: `r76/mincross`
Base commit: `92b75b7`
Commit: none. Ladder failed; completion contract requires no commit.

## What Changed

- Kept the prior WIP as the starting point:
  - Graphviz mincross pass structure and weighted crossing counts in
    `dagua/layout/ops/_dot_mincross.py`.
  - Incremental transpose order maps.
  - Representative-chain merge for duplicate original `(tail, head)` edges.
- Added graphviz-fidelity weight gating in `dagua/layout/ops/sugiyama.py`.
  - Exact `fidelity_mode="graphviz"` now ignores Dagua `edge_weights` for dot
    rank assignment and expanded dummy-chain weights, matching the benchmark
    DOT adapter, which does not emit `weight=`.
  - `dot`, `graphviz_dot`, igraph, and default paths remain unchanged.
- Replaced the prior fast-node-only seed order with a closer Graphviz
  `decompose(g, 1)` component order for graphviz mincross seeding.
  - `class2()` chain creation still scans real tail nodes and each tail's
    outgoing edges.
  - The `build_ranks()` node scan now follows the component DFS over the
    expanded fast graph.
- Added focused tests:
  - `test_graphviz_decompose_order_discovers_virtual_nodes_from_real_roots`.
  - `test_sugiyama_graphviz_fidelity_ignores_benchmark_edge_weights`.

## Pinned Graphviz 7.0.5 Sources

All source claims were checked with
`git -C /home/jtaylor/projects/_references/graphviz show 7.0.5:<path>`.

- `lib/dotgen/dotinit.c:55-65`: default DOT edges get `ED_weight=1` and
  `ED_xpenalty=1`; the benchmark adapter omits `weight=`.
- `lib/dotgen/class2.c:192-265`: `class2()` calls `fast_node(g, n)` for each
  real node, then scans `agfstout/agnxtout` and calls `make_chain()`.
- `lib/dotgen/fastgr.c:205-264`: `fast_node()` prepends to `GD_nlist`;
  `virtual_node()` also calls `fast_node()`.
- `lib/dotgen/decomp.c:22-118`: `decompose(g, 1)` rebuilds component lists by
  DFS from real `agfstnode` roots and discovers virtual nodes through fast
  in/out edges.
- `lib/dotgen/mincross.c:798-835`: pass bookkeeping and final transpose.
- `lib/dotgen/mincross.c:1356-1414`: `build_ranks()` scans `GD_nlist`,
  installs source/sink BFS order, then root `transpose(FALSE)`.
- `lib/dotgen/rank.c:450-490` and `lib/common/ns.c`: rank assignment path used
  before mincross. Real-node ranks matched Graphviz after weight gating on both
  residual weighted graphs.

## Ordering-Stage Discriminator

DOT input was emitted exactly like
`dagua/eval/competitors/graphviz_competitor.py:_graph_to_dot()`.
Reference was installed Graphviz 7.0.5 `dot -v`.

| Graph | `dot -v` | Port | Result |
|---|---:|---:|---|
| `binary_tree` | 0 | 0 | match |
| `bipartite_4_3_4` | 36 | 36 | match |
| `hub_skip_superfan` | 2 | 2 | match |
| `weighted_karate_34` | 63 | 63 | match |
| `dense_pair_50` | 271 | 271 | match |
| `heavy_tail_weights_50` | 50 | 51 | fail |

Result: 5/6 exact matches. This satisfies ladder A's literal `>=5/6` gate, but
does not satisfy the fixup brief's stronger "close the last two graphs" intent:
`heavy_tail_weights_50` remains one internal mincross crossing high.

Pass-start comparison after the fix:

| Graph | Graphviz pass 0 start | Port pass 0 start | Final Graphviz | Final port |
|---|---:|---:|---:|---:|
| `weighted_karate_34` | 178 | 178 | 63 | 63 |
| `heavy_tail_weights_50` | 96 | 91 | 50 | 51 |

Rank parity check:

| Graph | Real-node rank result |
|---|---|
| `weighted_karate_34` | 0 differing real-node ranks vs `dot -Tplain` |
| `heavy_tail_weights_50` | 0 differing real-node ranks vs `dot -Tplain` |

The remaining unported rule is not rank assignment. It is still in the
post-rank expanded fast-graph/mincross construction or metadata. The
source-faithful `decompose(g, 1)` order fixed `weighted_karate_34`, and
`heavy_tail_weights_50` pass 1 start now matches Graphviz (`155`), but pass 0
still starts at `91` vs `96`. Variants of component root order, in/out vector
order, and reverse edge-list traversal did not change that pass-0 value.

## Ladder Status

### a. Ordering-stage

Passed the literal gate: 5/6 exact matches. Residual: `heavy_tail_weights_50`
`dot=50`, port `51`.

### b. Stage-A no-regression

Command:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --workers 1 --timeout 120 --seeds 1 --seed-start 42 \
  --graphs binary_tree,bipartite_4_3_4,org_chart_1_5_4_8 \
  --engines classic_sugiyama_graphviz_fidelity --variants \
  --output-dir /tmp/r76_mincross_stage_b
```

Status: 3 total, 3 OK.

Stress values with `normalized_stress(..., fit_scale=True)`:

| Graph | r75 Stage-A value | r76 fixup value | Result |
|---|---:|---:|---|
| `binary_tree` | 0.1634169894465131 | 0.1634169894465131 | pass |
| `bipartite_4_3_4` | 0.1574668225741319 | 0.1574668225741319 | pass |
| `org_chart_1_5_4_8` | 0.19286388297906776 | 0.19286388297906776 | pass |

### c. Rendered crossings

Command:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --workers 1 --timeout 240 --seeds 1 --seed-start 42 \
  --graphs dense_pair_50,weighted_karate_34,hub_skip_superfan,heavy_tail_weights_50 \
  --engines classic_sugiyama_graphviz_fidelity --variants \
  --output-dir /tmp/r76_mincross_stage_c
```

Status: 4 total, 4 OK.

Rendered crossings from saved tensors with `dagua.metrics.count_crossings`:

| Graph | Baseline D | Reference R | Fixup D | Direction |
|---|---:|---:|---:|---|
| `dense_pair_50` | 391 | 331 | 338 | toward |
| `weighted_karate_34` | 111 | 108 | 118 | away |
| `hub_skip_superfan` | 3 | 2 | 5 | away |
| `heavy_tail_weights_50` | 70 | 67 | 66 | toward |

Result: failed, 2/4 moved toward reference; ladder requires >=3/4.

### d. `ba_500`

Command:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --workers 1 --timeout 300 --seeds 1 --seed-start 42 \
  --graphs ba_500 --engines classic_sugiyama_graphviz_fidelity --variants \
  --output-dir /tmp/r76_mincross_stage_d
```

Status: 1 total, 1 OK. Runtime was 33.7s, under the 300s limit.

Rendered crossings from saved tensor: `95,261`. This fails the crossing gate:
baseline was `22,344`, so the result is worse instead of >=2x improved.

### e. Regression and byte-identical checks

- Repeated-output byte identity smoke:
  - `binary_tree`, default, seeds 42-46: byte-identical repeated calls.
  - `binary_tree`, tight, seeds 42-46: byte-identical repeated calls.
  - `densenet_block`, default, seeds 42-46: byte-identical repeated calls.
  - `densenet_block`, tight, seeds 42-46: byte-identical repeated calls.
- Selector pytest passed:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q
54 passed, 3099 deselected, 34 warnings in 18.29s
```

## Quality Gates Run

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check . --fix
All checks passed!
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl mypy --follow-imports=silent dagua/cli.py
pyproject.toml: note: unused section(s): module = ['dagua.layout.multilevel']
Success: no issues found in 1 source file
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/test_layout/test_dot_mincross.py -q
6 passed, 3 warnings in 0.02s
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
457 passed, 157 warnings in 2045.46s (0:34:05)
```

Final non-slow suite:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
FAILED tests/test_bench_large.py::test_hierarchy_checkpoint_rejects_incomplete_manifest
1 failed, 63 passed, 88 deselected, 34 warnings in 13.46s
```

This is the same unrelated checkpoint-manifest failure recorded in
`r75_IMPL_sugiyama_xns_NOTES.md`: `_load_hierarchy_checkpoint()` accepts a
manifest with `"complete": false`. It is outside this mincross fixup scope.

## Assumptions

- Treated exact `fidelity_mode="graphviz"` as the benchmark-reference path
  because `graphviz_competitor._graph_to_dot()` omits edge `weight=`.
- Left `dot` and `graphviz_dot` aliases unchanged; they still consume Dagua
  edge weights as before.
- Did not touch BK/default/igraph ordering paths.

## Controversial Choices

- Kept the source-faithful `decompose(g, 1)` seed order even though
  `heavy_tail_weights_50` remains one crossing off. It fixes the larger
  residual on `weighted_karate_34` exactly and is directly supported by pinned
  Graphviz 7.0.5 source.
- Did not add a graph-specific or outcome-specific tie-breaker for
  `heavy_tail_weights_50`. The remaining mismatch needs the exact unported
  Graphviz rule, not a tuned branch.

## Concerns / Follow-up

- Ladder C and D fail; no commit was made.
- The exact remaining mincross construction rule appears to be a subtle
  expanded fast-edge metadata/order difference after rank parity and component
  DFS parity:
  - `heavy_tail_weights_50` real ranks match.
  - duplicate representative-chain merge is not implicated.
  - pass 1 start matches Graphviz, but pass 0 start remains `91` vs `96`.
- Rendered crossings are still not reliable even when internal mincross counts
  improve. `ba_500` is especially bad (`95,261`), so x-coordinate/routing
  interaction remains a separate blocker.

## Knowledge

- Graphviz mincross seed order is not just reverse fast-node creation order for
  root graphs. `init_mincross()` calls `class2()`, then `decompose(g, 1)`,
  which rebuilds `GD_nlist` component lists before `build_ranks()`.
- Exact graphviz-fidelity rank assignment must ignore Dagua weights when the
  benchmark DOT adapter omits `weight=`.
- `weighted_karate_34` is now a useful regression for this fix: unit weights
  plus `decompose(g, 1)` order gives pass-0 `178` and final `63`.

## A4: rendered-stage parity

Date: 2026-07-03
Commit: none. Gate B did not pass; completion contract requires no commit.

### Step 0 localization

Ordering discriminator was rerun with DOT text from
`dagua/eval/competitors/graphviz_competitor.py:_graph_to_dot()`, installed
Graphviz 7.0.5 `dot -v`, and the current A1 graphviz-mode mincross port.

| Graph | `dot -v` | Port | Gap | Expanded nodes | Expanded edges |
|---|---:|---:|---:|---:|---:|
| `binary_tree` | 0 | 0 | 0 | 11 | 10 |
| `bipartite_4_3_4` | 36 | 36 | 0 | 11 | 24 |
| `hub_skip_superfan` | 2 | 2 | 0 | 26 | 32 |
| `weighted_karate_34` | 63 | 63 | 0 | 98 | 142 |
| `dense_pair_50` | 271 | 271 | 0 | 623 | 774 |
| `heavy_tail_weights_50` | 50 | 51 | 1 | 96 | 120 |
| `ba_500` | 79098 | 79046 | -52 | 4934 | 5928 |

Benchmark-path Dagua tensors were generated with:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --workers 1 --timeout 360 --watchdog-timeout 720 \
  --seeds 3 --seed-start 100 --seed-refs graphviz_dot \
  --graphs dense_pair_50,weighted_karate_34,hub_skip_superfan,heavy_tail_weights_50,ba_500 \
  --engines classic_sugiyama_graphviz_fidelity --variants \
  --output-dir /tmp/r76_a4_step0
```

Output:

```text
[benchmark] Done: 15 total, 15 ok, 0 skipped, 0 errors, 0 timeouts
```

Reference tensors were generated through the sanctioned offline path
`graphviz_competitor._layout_with_dot()` with variant-mapped attributes
`maxiter=24`, `ranksep=1.0`, and `nodesep=1.0`. Crossings and stress were
scored with `dagua.metrics.count_crossings`, `sampled_crossing_rate`, and
`sampled_stress`.

| Graph | Ordering gap | D crossings mean +/- SE | R crossings mean +/- SE | Rendered gap | D stress | R stress | Rendered rank/order probe |
|---|---:|---:|---:|---:|---:|---:|---|
| `dense_pair_50` | 0 | 338.000 +/- 0.000 | 331.000 +/- 0.000 | 7.000 | 0.694839 | 0.698054 | same real rank members and order |
| `weighted_karate_34` | 0 | 118.000 +/- 0.000 | 108.000 +/- 0.000 | 10.000 | 0.641662 | 0.634682 | same real rank members and order |
| `hub_skip_superfan` | 0 | 5.000 +/- 0.000 | 2.000 +/- 0.000 | 3.000 | 0.515810 | 0.540018 | same real rank members and order |
| `heavy_tail_weights_50` | 1 | 66.000 +/- 0.000 | 67.000 +/- 0.000 | -1.000 | 0.724142 | 0.725004 | same rank members, 5 order-diff layers |
| `ba_500` | -52 | 95969.667 +/- 454.756 | 94979.000 +/- 213.439 | 990.667 | 0.777186 | 0.786698 | 22 ranks each, different real rank members |

Interpretation:

- `dense_pair_50`, `weighted_karate_34`, and `hub_skip_superfan` are downstream
  defects: ordering is exact, and rendered real-node rank/order also matches,
  but node-position-derived crossings differ.
- `heavy_tail_weights_50` remains partly ordering-stage contaminated by the
  known one-crossing A1 residual, and rendered real-node order differs on 5
  layers.
- `ba_500` internal mincross is close to Graphviz (`-52` on a 79K count), but
  rendered crossings differ by about 991 sampled crossings and real rank
  membership differs. This local reference value is from the repo's
  `graphviz_competitor` path; it did not reproduce the prompt's remembered
  `~140,276` reference count.

All five Step 0 graphs have zero same-rank original edges after graphviz rank
assignment, so Graphviz flat-edge machinery is not the active cause for this
calibration set.

### Source findings

Pinned Graphviz 7.0.5 source was read with
`git -C /home/jtaylor/projects/_references/graphviz show 7.0.5:<path>`.

- `lib/dotgen/position.c:118-139`: `dot_position()` runs `set_ycoords()`,
  optional `set_ycoords()` again when `flat_edges(g)` is true, then
  `create_aux_edges()`, `rank(g, 2, nsiter2(g))`, and `set_xcoords()`.
- `lib/dotgen/position.c:222-258`: `make_LR_constraints()` uses
  `GD_nodesep(g)` in point units and creates zero-weight same-rank constraints
  from `ND_rw(left) + ND_lw(right) + nodesep`.
- `lib/dotgen/position.c:262-318`: labeled and unlabeled flat-edge endpoint
  constraints are added in the same auxiliary graph, but Step 0 graphs do not
  exercise them.
- `lib/dotgen/position.c:323-340`: `make_edge_pairs()` creates one slack node
  per saved expanded edge and constrains it to tail/head endpoints with
  `ED_weight(e)`.
- `lib/dotgen/class2.c:47-50`: `plain_vnode()` calls `virtual_node()` then
  `incr_width()`.
- `lib/dotgen/class2.c:84-95`: `make_chain()` creates plain or label virtual
  nodes, assigns ranks, creates virtual edges, and calls `virtual_weight(e)`.
- `lib/dotgen/mincross.c:824-831`, `1246-1270`, and `1470-1541`:
  `flat_breakcycles()` and `flat_reorder()` are wired into mincross, but they
  are dormant for the Step 0 graphs because no original edge is same-rank after
  ranking.

### Attempted patch, not kept

I tested a narrow units patch in `dagua/layout/ops/sugiyama.py` that converted
graphviz-mode `node_sep=1.0` to 72 point units inside
`_graphviz_x_coordinate_assignment()`, matching the Graphviz competitor's
`-Gnodesep=1.0` DOT semantics. This was reverted because it failed the full
gate, but the measured effect is useful:

| Graph | Before D/R crossings | Patched D/R crossings | Crossing movement | Before D/R stress | Patched D/R stress | Stress movement |
|---|---:|---:|---|---:|---:|---|
| `dense_pair_50` | 338 / 331 | 336 / 331 | toward, 28.6% gap reduction | 0.694839 / 0.698054 | 0.688934 / 0.698054 | away |
| `weighted_karate_34` | 118 / 108 | 102 / 108 | toward, 40.0% gap reduction | 0.641662 / 0.634682 | 0.635194 / 0.634682 | toward |
| `hub_skip_superfan` | 5 / 2 | 2 / 2 | exact | 0.515810 / 0.540018 | 0.543955 / 0.540018 | toward |
| `heavy_tail_weights_50` | 66 / 67 | 63 / 67 | away | 0.724142 / 0.725004 | 0.728093 / 0.725004 | away |
| `ba_500` seed 100 | 95909 / 95092 | 94235 / 95092 | toward | 0.777186 / 0.786698 | 0.790034 / 0.786698 | away |

This would have satisfied crossing movement on 3/4 calibration graphs and moved
`ba_500` crossings toward the local reference, but stress moved toward the
reference on only 2/4 calibration graphs. Gate B requires rendered crossings
and stress to move toward the reference on at least 3/4 graphs, so the patch
was not kept.

A second source-backed probe added original-node width padding based on the
observed Graphviz JSON box widths (`+10` to `+18` points wider than Dagua's
computed sizes on the calibration graphs). It improved some crossing gaps but
did not fix the stress gate and was not kept.

### Gate status

- Gate a: literal ordering discriminator remains 5/6 exact on the calibration
  set. `heavy_tail_weights_50` remains the known one-crossing residual.
- Gate b: failed. The only source-backed patch found in this pass improved
  crossings on 3/4 calibration graphs, but stress only moved toward the
  reference on 2/4.
- Gate c: not run because Gate b failed and no code patch was retained.
- Gate d: not run after A4 because no code patch was retained. Prior A1
  selector and layout tests are recorded above.
- Gate e: no commit by contract.

### Remaining downstream defect

The unported stage is exact `position.c` x-coordinate parity, specifically the
full auxiliary graph state used by `make_LR_constraints()` and
`make_edge_pairs()` after Graphviz has computed point-unit node boxes,
virtual-node widths, virtual edge weights, and rank/edge metadata. The current
Dagua graphviz x helper is still an approximation:

- It receives Dagua-computed `node_sizes`, which were narrower than Graphviz
  JSON node boxes on the calibration graphs (`hub_skip_superfan`: Dagua
  `44-47.85` pt vs Graphviz `54-61.56` pt; `dense_pair_50`: Dagua
  `47.85-52.02` pt vs Graphviz `61.56-70.18` pt; `weighted_karate_34`:
  Dagua `44` pt vs Graphviz `54` pt).
- It treats the benchmark `node_sep=1.0` as a layout-unit value, while the
  Graphviz reference path maps the same variant parameter to DOT `nodesep=1.0`
  inches.
- It does not yet model every `position.c` source of x auxiliary constraints,
  but flat-edge constraints are not implicated for the Step 0 graphs.

The next implementation should port the point-unit node-box and nodesep
translation for exact `fidelity_mode="graphviz"` together, then validate stress
as well as crossings before committing. A crossing-only units patch is not
enough.

### Commits

None.

## A4b: x-coordinate box/units parity

Date: 2026-07-03
Commit: none. Gate B failed; completion contract requires no commit.

### Implemented attempt

Scoped changes to exact `fidelity_mode="graphviz"`:

- Added benchmark DOT node-box mirroring in
  `dagua/eval/competitors/classic_competitor.py`.
  - `graphviz_competitor._graph_to_dot()` does not emit explicit
    `width=`, `height=`, or `fixedsize=`.
  - It emits global `node [shape=box, style=filled, fontname="Helvetica"]`,
    per-node `fontsize`, and per-node shape overrides.
  - The helper therefore mirrors Graphviz default node sizing from label text,
    default node floors, `PAD()`, and ellipse expansion instead of using
    Dagua theme boxes.
- Threaded optional `graphviz_node_sizes` through
  `layout_sugiyama_pipeline()` into the graphviz-only dummy expansion and
  x-coordinate solve.
- Converted exact graphviz-mode `node_sep` from DOT inches to points inside
  `_graphviz_x_coordinate_assignment()`: `node_sep_points = node_sep * 72.0`.
- Changed graphviz x auxiliary minlen quantization to point-unit rounding
  resolution `1`, matching `make_aux_edge(... ROUND(len))` once the helper is
  operating in Graphviz point units.
- Ported duplicate-chain virtual-node width inflation:
  representative long-edge merges now add one point-unit `nodesep` width to
  every internal virtual node on the reused chain.

### Pinned Graphviz 7.0.5 sources

All source claims were checked with
`git -C /home/jtaylor/projects/_references/graphviz show 7.0.5:<path>`.

- `lib/common/const.h`: `DEFAULT_NODEWIDTH=0.75`,
  `DEFAULT_NODEHEIGHT=0.5`, `DEFAULT_NODESEP=0.25`,
  `DEFAULT_RANKSEP=0.5`; point conversion is 72 pt/in.
- `lib/common/utils.c:541-558`: `common_init_node()` reads node width,
  height, shape, label, font size/name/color, then calls the shape init
  function.
- `lib/common/shapes.c:1810-1811`: polygon node init reads width/height in
  inches.
- `lib/common/shapes.c:1902-1918`: node labels get explicit margin or
  `PAD(dimen)`.
- `lib/common/macros.h:29-31`: `PAD()` is `XPAD=4*GAP`,
  `YPAD=2*GAP`.
- `lib/common/shapes.c:1977-2033`: non-box polygon/ellipse sizing expands the
  padded label box before applying default width/height floors.
- `lib/common/textspan_lut.c:92-132` and `:831-833`: Helvetica-compatible
  hard-coded text widths use 2048 units per em.
- `lib/dotgen/position.c:222-258`: `make_LR_constraints()` uses
  `GD_nodesep(g)` in points and creates zero-weight same-rank constraints
  from `ND_rw(left) + ND_lw(right) + nodesep`.
- `lib/dotgen/position.c:323-340`: `make_edge_pairs()` adds one slack node per
  saved expanded edge and constrains it to tail/head with `ED_weight(e)`.
- `lib/dotgen/class2.c:35-50`: `plain_vnode()` calls `incr_width()`, which
  adds `GD_nodesep(g) / 2` to both virtual-node sides.
- `lib/dotgen/class2.c:146-154`: `merge_chain()` adds merged edge weight and
  penalty, then calls `incr_width()` on each internal virtual node in the
  representative chain.
- `lib/dotgen/mincross.c:1861-1891`: `virtual_weight()` multiplies expanded
  edge weights by endpoint class table values `1/2/4`.

### Box rule and validation

The implemented benchmark box rule is:

```text
label_width = Helvetica_lut_width(label, fontsize)
label_height ~= 1.128 * fontsize
padded_width = label_width + 16
padded_height = label_height + 8
box base = max(54, padded_width) x max(36, padded_height)
ellipse width = Graphviz ellipse expansion from shapes.c, then floored by 54
circle width/height = max(expanded width, expanded height)
```

Validation against installed `dot -Tjson` on calibration graph labels:

| Label | DOT width pt | Helper width pt | Result |
|---|---:|---:|---|
| numeric labels (`10`) | 54.000 | 54.000 | exact |
| `output` | 61.559 | 61.602 | close |
| `pair_0` | 61.559 | 61.594 | close |
| `pair_10` | 70.177 | 69.923 | close |

The helper fixes the major A4 box defect class (44-52 pt Dagua theme boxes vs
54-70 pt Graphviz DOT boxes), but it is not byte-exact for long labels. More
importantly, the exact-box diagnostic below shows that the remaining rendered
gate failure is not caused by this residual box-measurement error.

### Benchmark calibration

Command:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --workers 1 --timeout 360 --watchdog-timeout 720 \
  --seeds 5 --seed-start 100 --seed-refs graphviz_dot \
  --graphs dense_pair_50,weighted_karate_34,hub_skip_superfan,heavy_tail_weights_50,ba_500 \
  --engines classic_sugiyama_graphviz_fidelity --variants \
  --output-dir /tmp/r76_a4b_calib
```

Output:

```text
[benchmark] Done: 25 total, 25 ok, 0 skipped, 0 errors, 0 timeouts
```

Scored saved tensors with `dagua.metrics.count_crossings` and
`dagua.metrics.sampled_stress`; references were generated through
`GraphvizDot.layout_with_variant(..., {"maxiter": 24, "vgap": 1.0, "hgap": 1.0})`.

First combined box+units patch (`_GRAPHVIZ_X_AUX_RESOLUTION=2`, no duplicate
virtual width inflation):

| Graph | Step-0 D/R crossings | A4b D/R crossings | Crossing movement | Step-0 D/R stress | A4b D/R stress | Stress movement |
|---|---:|---:|---|---:|---:|---|
| `dense_pair_50` | 338 / 331 | 354 / 331 | away | 0.694839 / 0.698054 | 0.688426 / 0.698054 | away |
| `weighted_karate_34` | 118 / 108 | 108 / 108 | toward, exact crossings | 0.641662 / 0.634682 | 0.634799 / 0.634682 | toward |
| `hub_skip_superfan` | 5 / 2 | 2 / 2 | toward, exact crossings | 0.515810 / 0.540018 | 0.543570 / 0.540018 | toward |
| `heavy_tail_weights_50` | 66 / 67 | 63 / 67 | away | 0.724142 / 0.725004 | 0.727681 / 0.725004 | away |
| `ba_500` | 95969.667 / 94979.000 | 94413.800 / 95351.000 | toward local reference range | 0.777186 / 0.786698 | 0.789525 / 0.786698 | away |

Source-backed follow-up (`_GRAPHVIZ_X_AUX_RESOLUTION=1` plus duplicate
virtual-node width inflation):

| Graph | Step-0 D/R crossings | A4b D/R crossings | Crossing movement | Step-0 D/R stress | A4b D/R stress | Stress movement |
|---|---:|---:|---|---:|---:|---|
| `dense_pair_50` | 338 / 331 | 343 / 331 | away | 0.694839 / 0.698054 | 0.692804 / 0.698054 | away |
| `weighted_karate_34` | 118 / 108 | 108 / 108 | toward, exact crossings | 0.641662 / 0.634682 | 0.634794 / 0.634682 | toward |
| `hub_skip_superfan` | 5 / 2 | 2 / 2 | toward, exact crossings | 0.515810 / 0.540018 | 0.543615 / 0.540018 | toward |
| `heavy_tail_weights_50` | 66 / 67 | 63 / 67 | away | 0.724142 / 0.725004 | 0.727680 / 0.725004 | away |
| `ba_500` | 95969.667 / 94979.000 | 94396 / 95351 | toward local reference range | 0.777186 / 0.786698 | 0.789551 / 0.786698 | away |

Result: Gate B failed. Rendered crossings and sampled stress moved toward the
reference on only 2/4 required small graphs (`weighted_karate_34`,
`hub_skip_superfan`). `dense_pair_50` and `heavy_tail_weights_50` moved away on
both legs.

### Exact-box diagnostic

To separate box measurement from auxiliary-constraint mismatch, I supplied
exact installed `dot -Tjson` node widths/heights directly to
`layout_sugiyama_pipeline(..., fidelity_mode="graphviz", graphviz_node_sizes=...)`
with the current point-unit nodesep and duplicate virtual-width code active.

| Graph | Exact-box D crossings | R crossings | Exact-box D stress | R stress |
|---|---:|---:|---:|---:|
| `dense_pair_50` | 343 | 331 | 0.692036 | 0.698054 |
| `weighted_karate_34` | 108 | 108 | 0.634794 | 0.634682 |
| `hub_skip_superfan` | 2 | 2 | 0.543615 | 0.540018 |
| `heavy_tail_weights_50` | 63 | 67 | 0.727680 | 0.725004 |

This proves the remaining Gate B failure is not the original-node box helper.
Even exact DOT JSON boxes leave the same 2/4 movement failure.

### Gate status

- Gate a: ordering discriminator was not rerun after A4b because the x-stage
  changes do not alter rank assignment or mincross ordering. The inherited A4
  ordering state remains 5/6 exact with `heavy_tail_weights_50` one internal
  crossing high.
- Gate b: failed, as shown above.
- Gate c: not run because Gate b failed.
- Gate d:
  - `PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check . --fix` passed.
  - `PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/test_layout/test_dot_mincross.py tests/test_layout/test_sugiyama_fidelity.py -x -q` passed: 19 passed, 3 warnings.
  - `PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q` passed: 56 passed, 3099 deselected, 34 warnings.
  - Full `tests/test_layout/ -x -q` and non-slow suite were not run because
    Gate B had already failed.
- Gate e: no commit made. Commit shas: none.

### Remaining unmatched x constraint

The remaining unmatched x-coordinate behavior is in the `position.c`
auxiliary graph after original-node boxes, point-unit `nodesep`, direct
point rounding, `make_edge_pairs()` slack nodes, `virtual_weight()`, and
duplicate-chain virtual-node width growth are accounted for.

The residual is now narrowed to one of these `position.c` auxiliary graph
details:

1. `make_edge_pairs()` saved-edge traversal/order from `GD_nlist` and
   `ND_save_out(n).list` is still not exact. Dagua iterates expanded edges in
   tensor order; Graphviz iterates saved fast-edge lists after
   `allocate_aux_edges()` (`position.c:173-185`) and `make_edge_pairs()`
   (`position.c:323-340`).
2. Auxiliary slack-node initial ranks may not match Graphviz's in-memory
   `ND_rank(sn) = MIN(ND_rank(tail)-m0-1, ND_rank(head)-m1-1)` at the moment
   network simplex starts (`position.c:337-339`). Dagua approximates the same
   value from the current same-rank LR seed, but the exact saved-edge list and
   same-rank seed interaction is not proven identical.
3. The network-simplex input order for zero-weight LR edges versus positive
   slack-node edge-pair constraints may differ. Weighted graphs that are exact
   on rank/order still diverge in dense x coordinates, which is consistent
   with a tie-breaking/order mismatch in the auxiliary x solve rather than
   a box or units defect.

Measured residual after all source-backed A4b changes:

| Graph | Residual crossing gap | Residual stress gap |
|---|---:|---:|
| `dense_pair_50` | +12 crossings vs reference | -0.005250 stress from reference |
| `heavy_tail_weights_50` | -4 crossings vs reference | +0.002676 stress from reference |

No graph-specific or metric-tuned branch was added. The current uncommitted
patch is a source-backed port-in-progress, but it does not satisfy the ladder
and must not be committed under this task contract.

### Assumptions

- Treated the benchmark Graphviz DOT adapter as the fidelity target. It emits
  no explicit node width/height/fixedsize and omits edge `weight=`.
- Used installed `dot -Tjson` only for validation/scoring, not from Dagua ops
  at runtime.
- Kept all changes out of igraph/default paths except for accepting an unused
  optional `graphviz_node_sizes` argument on the public Sugiyama pipeline.

### Knowledge

- The A4 box and units defects are real but not sufficient: exact JSON boxes
  plus point-unit nodesep still fail the required movement gate.
- The remaining x-stage mismatch is downstream of rank/order and original-node
  geometry. The next useful trace is the full `position.c` auxiliary graph:
  ordered LR edges, slack nodes, slack initial ranks, and edge-pair constraint
  insertion order.

## A4c: auxiliary x-graph parity

Date: 2026-07-03
Commit: none. Gate B failed; completion contract requires no new commit.
Base HEAD: `525327d`.

### Source trace

Pinned Graphviz 7.0.5 source was read with
`git -C /home/jtaylor/projects/_references/graphviz show 7.0.5:<path>`.
The reference clone was not modified. I also archived 7.0.5 into
`/tmp/gv750-a4c`, instrumented only that scratch copy, built `dot_builtins`,
captured traces, and then removed the scratch tree before finishing.

Relevant source anchors:

- `lib/dotgen/position.c:171-188`: `make_aux_edge()` rounds `len` into
  `ED_minlen`, sets `ED_weight`, and appends the edge with `fast_edge()`.
- `lib/dotgen/position.c:191-209`: `allocate_aux_edges()` saves current
  `ND_in` and `ND_out` as `ND_save_in` and `ND_save_out` before adding x-stage
  aux constraints.
- `lib/dotgen/position.c:230-259`: `make_LR_constraints()` seeds same-rank
  `ND_rank` values and emits zero-weight LR aux edges from
  `ND_rw(left) + ND_lw(right) + GD_nodesep`.
- `lib/dotgen/position.c:323-347`: `make_edge_pairs()` scans `GD_nlist` and
  each node's saved outgoing fast-edge list, creates one slack node per saved
  edge, emits two positive-weight aux edges, and seeds the slack node rank from
  the endpoint ranks.
- `lib/common/ns.c:907-955`: `rank(g, 2, nsiter2(g))` preserves feasible
  initial ranks when possible, builds the feasible tree, pivots, then runs
  `LR_balance()`.
- `lib/dotgen/class2.c:35-50` and `:146-154`: plain virtual nodes receive
  `GD_nodesep / 2` on both sides; merged long-edge chains increment internal
  virtual widths.
- `lib/dotgen/fastgr.c:60-75` and `:177-187`: `fast_edge()` appends to tail
  `ND_out` and head `ND_in`; `fast_node()` prepends to `GD_nlist`.
- `lib/dotgen/decomp.c:24-44` and `:92-115`: decomposition rebuilds
  `GD_nlist` in component DFS order before later stages scan it.

### Aux trace evidence

The scratch Graphviz trace dumped aux edge creation order, node ranks after
`make_edge_pairs()`, and final x ranks for the two requested calibration
graphs. I captured Dagua's current aux builder by wrapping
`_build_graphviz_x_aux_edges()` in the graphviz-fidelity pipeline with the
same benchmark variant parameters.

Counts match, so the current port is not missing whole classes of aux edges on
these two traces:

| Graph | Reference aux edges | Reference LR | Reference edge-pair edges | Dagua aux edges | Dagua LR | Dagua edge-pair edges |
|---|---:|---:|---:|---:|---:|---:|
| `weighted_karate_34` | 375 | 91 | 284 | 375 | 91 | 284 |
| `dense_pair_50` | 2121 | 573 | 1548 | 2121 | 573 | 1548 |

The first concrete mismatch is in LR minlen/rank seeding from the actual
Graphviz `ND_lw` and `ND_rw` values, before edge-pair ordering can explain the
residual:

| Graph | First reference LR minlens | First Dagua LR minlens | Interpretation |
|---|---:|---:|---|
| `weighted_karate_34` | `146, 146, 146, 146, 136, 136, 146, 146, 146, 146` | `144, 144, 144, 144, 135, 135, 144, 144, 144, 144` | Dagua's point boxes/virtual widths still seed smaller LR ranks. |
| `dense_pair_50` | `177, 141, 146, 182, 182, 141, 141, 146, 146, 146` | `175, 139, 144, 180, 180, 139, 139, 144, 144, 144` | Same first mismatch class, including long-label rows. |

Representative final reference x ranks from the scratch trace:

| Graph | First final reference x-rank rows |
|---|---|
| `weighted_karate_34` | `n0:379`, `%0:-1351`, `%0:-1205`, `%0:-1059`, `%0:-767`, `%0:-248`, `n1:24`, `%0:160`, `%0:306`, `%0:452` |
| `dense_pair_50` | `n0:916`, `%0:452`, `n1:916`, `%0:1354`, `%0:1646`, `%0:-977`, `%0:452`, `%0:634`, `n2:775`, `%0:916` |

Conclusion: A4c narrowed the residual to exact x-aux rank seeding values from
Graphviz's in-memory node half-widths. Saved fast-edge traversal remains
source-relevant, but a Dagua-side saved-out-order patch was not sufficient and
therefore was not retained.

### Attempted patch, not kept

I tested a narrow graphviz-only patch that threaded `expanded_graph.graphviz_node_order`
into `_graphviz_x_coordinate_assignment()` and made
`_build_graphviz_x_aux_edges()` allocate slack nodes by saved outgoing fast-edge
traversal instead of raw tensor edge order. A regression test verified the
intended slack insertion order on a synthetic graph.

The patch was reverted because it failed Gate B:

| Graph | A4b D/R crossings | A4c probe D/R crossings | Crossing movement | A4b D/R stress | A4c probe D/R stress | Stress movement |
|---|---:|---:|---|---:|---:|---|
| `dense_pair_50` | 343 / 331 | 341 / 331 | toward | 0.692804 / 0.698054 | 0.691155 / 0.698054 | away |
| `weighted_karate_34` | 108 / 108 | 108 / 108 | exact | 0.634794 / 0.634682 | 0.634794 / 0.634682 | same/toward-equivalent |
| `hub_skip_superfan` | 2 / 2 | 2 / 2 | exact | 0.543615 / 0.540018 | 0.543615 / 0.540018 | same/toward-equivalent |
| `heavy_tail_weights_50` | 63 / 67 | 63 / 67 | same away | 0.727680 / 0.725004 | 0.727680 / 0.725004 | same away |

Gate B remains 2/4 by the strict "crossings and stress both move toward
reference" reading. `dense_pair_50` improved crossings but stress moved farther
from reference, and `heavy_tail_weights_50` did not move.

Benchmark command:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --workers 1 --timeout 360 --watchdog-timeout 720 \
  --seeds 5 --seed-start 100 --seed-refs graphviz_dot \
  --graphs dense_pair_50,weighted_karate_34,hub_skip_superfan,heavy_tail_weights_50 \
  --engines classic_sugiyama_graphviz_fidelity --variants \
  --output-dir /tmp/r76_a4c_calib
```

Output:

```text
[benchmark] Done: 20 total, 20 ok, 0 skipped, 0 errors, 0 timeouts
```

The scoring command regenerated Graphviz references through
`GraphvizDot.layout_with_variant(..., {"maxiter": 24, "vgap": 1.0, "hgap": 1.0})`
and used `count_crossings()` plus `sampled_stress()`.

### Competitor-helper review

`dagua/eval/competitors/classic_competitor.py` remains acceptable as engine
input plumbing:

- `_graphviz_dot_node_sizes()` is only passed when
  `fn_name == "layout_sugiyama_pipeline"` and
  `extra_kwargs.get("fidelity_mode") == "graphviz"`.
- It is supplied through `graphviz_node_sizes`, which
  `layout_sugiyama_pipeline()` stores only for graphviz fidelity mode.
- The benchmark scoring path still uses `graph.node_sizes` in
  `scripts/run_benchmark.py` and `dagua/eval/pipeline_io.py`; the helper does
  not feed scoring metrics or other engine inputs.

No relocation was needed.

### Gate status

- Gate a: not rerun after the failed x-stage-only probe. The probe did not
  alter rank assignment or mincross ordering; A4b's inherited state remains
  the applicable 5/6 exact discriminator.
- Gate b: failed, as shown above. Required small-graph movement remains 2/4.
- Gate c: not run because Gate B failed and the code patch was reverted.
- Gate d:
  - During the reverted probe,
    `PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check dagua/layout/ops/sugiyama.py tests/test_layout/test_dot_rank.py --fix`
    passed after one automatic import/format fix.
  - During the reverted probe,
    `PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/test_layout/test_dot_rank.py -q`
    passed: 7 passed, 3 warnings.
  - The required broader pytest commands were not run because Gate B failed
    and no code patch was retained.
- Gate e: no commit made, no push, no merge. Commit shas: none beyond base
  `525327d`.

### Concerns

- The scratch reference trace shows exact aux edge counts but not exact LR
  minlen values. The remaining mismatch is therefore not "missing edge-pair
  construction"; it is exact `ND_lw`/`ND_rw` rank seeding feeding
  `make_LR_constraints()`, plus any downstream tie effects in `rank(g, 2, ...)`.
- The failed saved-out traversal probe moved one metric on one graph but did
  not meet the sprint gate. It should not be revived without first matching
  the traced LR minlens and initial ranks.

### Knowledge

- For `weighted_karate_34` and `dense_pair_50`, Dagua now matches the
  reference aux edge counts exactly but still starts network simplex from
  different integer LR constraints.
- A source-faithful slack insertion order alone is insufficient: it improves
  `dense_pair_50` crossings from 343 to 341 but worsens stress, leaving the
  overall movement gate unchanged.

## A4d: crash fix

Date: 2026-07-03
Worktree: `/home/jtaylor/.claude/worktrees/dagua-mincross2`
Branch: `r76/mincross`
Fix commits:

- `aeaf19452ba1e577a89bcbb961aef02dbea78ac2`
  (`fix(sugiyama): size mincross Fenwick by rank width`)
- `9224d71b1f30ab3b3ac49509c639983bcde4d471`
  (`fix(benchmark): cache deterministic sugiyama repeats`)

### Root cause

The crash was in
`dagua/layout/ops/_dot_mincross.py:717-721` after A1's graphviz-style
component/rank ordering reached sparse wide ranks. `_count_crossings()` sized
its Fenwick tree by the number of edges between a rank pair, but queried it
with lower-rank node order (`lower_order + 1`). On the five crashing
graphviz-fidelity graphs, a terminal node could sit farther right than the
rank-pair edge count, so `_fenwick_sum()` indexed past the list and raised
`IndexError: list index out of range`.

### Fix

- Changed `_count_crossings()` to size the Fenwick tree from the lower-rank
  width, which is the coordinate system used by `lower_order`.
- Added `test_graphviz_mincross_counts_sparse_wide_rank_edges()` in
  `tests/test_layout/test_dot_mincross.py`.
- Added a per-worker deterministic cache for
  `layout_sugiyama_pipeline` in
  `dagua/eval/competitors/classic_competitor.py:32` and
  `dagua/eval/competitors/classic_competitor.py:1918-1964`.
  The layout ignores seed, so the cache returns cloned positions and preserves
  the first run's reported runtime. This keeps the 100-seed benchmark group
  under `scripts/run_benchmark.py`'s fixed 600s future watchdog without
  changing coordinates.

### Gate evidence

Crash repro before fix:

```text
classic_sugiyama_graphviz_fidelity on er_100 seed=100:
IndexError at dagua/layout/ops/_dot_mincross.py:775 in _fenwick_sum
```

Focused direct repro after fix:

```text
er_100 graphviz fidelity returned (100, 2)
```

Five crash graphs, 5 seeds, default timeout:

```text
Done: 150 total, 145 ok, 2 skipped, 3 errors, 0 timeouts
```

The remaining errors were not the original crash. They were
`worker layout timeout exceeded` on `sbm_5x50` because the benchmark scales
the default 120s timeout to 60s for a 250-node graph, while the first
graphviz-fidelity layout takes about 65-76s on this machine.

Five crash graphs, 5 seeds, `--timeout 240`:

```text
Done: 150 total, 150 ok, 0 skipped, 0 errors, 0 timeouts
```

No-behavior-change check on working graphviz-fidelity graphs:

```text
BYTE_IDENTITY_OK
binary_tree, dense_pair_50, weighted_karate_34, citation_dag_300
seeds 100, 101, 102 were byte-identical against HEAD's old mincross helper
```

Selector tests and lint:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check . --fix
All checks passed!

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q
57 passed, 3099 deselected, 34 warnings in 13.81s

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file
```

Final topup command:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --engines classic_sugiyama --variants \
  --graphs er_100,random_dag_50,regular_4_40,rgg_100,sbm_5x50 \
  --max-nodes 0 --seeds 100 --seed-start 100 --workers 4 \
  --timeout 3600 \
  --output-dir /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r76_sugiyama_topup
```

Final topup result:

```text
[benchmark] Done: 3000 total, 3000 ok, 0 skipped, 0 errors, 0 timeouts
```

### Concerns

- The default 5-seed command without `--timeout` still reports three
  `sbm_5x50` graphviz-fidelity worker timeouts because of the benchmark's
  size-scaled 60s budget. The crash is fixed; the timeout is a runtime-budget
  mismatch for this graph and exact x-coordinate path.
- The deterministic cache is intentionally scoped to classic Sugiyama worker
  repeats. It does not change coordinates, and it preserves the first run's
  runtime in cached `CompetitorResult` rows.

### Knowledge

- `classic_sugiyama_graphviz_fidelity` is deterministic across benchmark
  seeds; `_barycenter_ordering()` deletes the seed argument in the graphviz
  mincross path.
- `scripts/run_benchmark.py` groups all seeds for one graph/engine in one
  future. A 100-seed deterministic group can exceed the 600s watchdog even
  when each individual layout is below `--timeout`.

## A5: half-width rule

Date: 2026-07-04
Commit: pending.

### Named rule

The missing x-coordinate constant is Graphviz's virtual-node half-width seed:
`virtual_node()` initializes every virtual node with `ND_lw = ND_rw = 1`, then
`plain_vnode()` calls `incr_width()`, which adds integer `GD_nodesep(g) / 2` to
both sides. With the benchmark's graphviz-mode `nodesep=72pt`, a plain virtual
node is therefore `37 + 37 = 74pt`, not `36 + 36 = 72pt`.

Pinned Graphviz 7.0.5 source:

- `lib/dotgen/fastgr.c:250`: `virtual_node()` seeds `ND_lw(n) = ND_rw(n) = 1`.
- `lib/dotgen/class2.c:35-50`: `incr_width()` adds `GD_nodesep(g) / 2` to both
  sides and `plain_vnode()` applies it to new plain virtual nodes.
- `lib/dotgen/class2.c:146-154`: `merge_chain()` calls `incr_width()` on each
  internal representative-chain virtual node for duplicate long edges, so
  duplicate increments add one `nodesep` after the initial 2pt seed.
- `lib/dotgen/position.c:230-259`: LR constraints consume
  `ND_rw(left) + ND_lw(right) + GD_nodesep(g)` and round through
  `make_aux_edge()`.

For label-expanded original boxes, the traced LR tables also require the same
1pt half-width seed when `ND_width` exceeds the 54pt default floor. Default
54pt numeric boxes are not inflated; this preserves `weighted_karate_34`'s
`136` rows while closing the `dense_pair_50` label-adjacent rows.

### Implementation

- `dagua/layout/ops/sugiyama.py`
  - Added `_GRAPHVIZ_VIRTUAL_NODE_HALF_WIDTH_SEED_POINTS = 1.0`.
  - Graphviz-only dummy expansion now initializes new virtual widths to
    `nodesep + 2pt`.
  - Duplicate-chain virtual width growth still adds exactly one `nodesep`,
    matching later `incr_width()` calls.
  - Added graphviz-only original half-width seed for label-expanded boxes in
    `_graphviz_left_width()` and `_graphviz_right_width()`.
- Tests:
  - Updated duplicate-chain width golden from `144pt` to `146pt`.
  - Added `test_graphviz_virtual_node_width_seed_matches_lr_minlen()`.
  - Updated the graphviz x-assignment golden for the source-faithful
    half-width constants.

### Minlen parity

First LR minlen rows after A5:

| Graph | Graphviz traced | Dagua A5 | Result |
|---|---|---|---|
| `weighted_karate_34` | `146, 146, 146, 146, 136, 136, 146, 146, 146, 146` | `146, 146, 146, 146, 136, 136, 146, 146, 146, 146` | exact |
| `dense_pair_50` | `177, 141, 146, 182, 182, 141, 141, 146, 146, 146` | `177, 141, 146, 182, 182, 141, 141, 146, 146, 146` | exact |

Additional calibration first rows:

| Graph | Dagua A5 first LR minlens |
|---|---|
| `hub_skip_superfan` | `136, 126, 136, 136, 126, 136, 146, 136, 136, 126` |
| `heavy_tail_weights_50` | `136, 136, 136, 146, 136, 136, 136, 126, 136, 136` |
| `binary_tree` | `126, 126, 126, 126, 126, 126, 126` |

### Benchmark-path d_R probe

Probe rows were selected from mode-B graphviz-fidelity close/far rows in
`eval_output/fidelity_definitive/per_combo_r76.jsonl`. Fresh positions were
generated through `scripts/run_benchmark.py`, then rescored with
`dagua.eval.distributional_fidelity.analyze_mode_b(..., free_aspect=True)`.

Commands:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --workers 5 --timeout 360 --watchdog-timeout 720 --seeds 100 --seed-start 100 \
  --graphs hub_skip_superfan,cluster_member_style_stress,dense_pair_50,width_skew_late_merge,ragged_feature_pyramid,real_lesmis_77,shape_and_routing_matrix,long_skip_only_24,braided_feedback_tails,edge_label_braid,clustered_longlabel_handoffs,random_dag_50,random_bipartite_60,sierpinski_42,citation_dag_300 \
  --engines classic_sugiyama_graphviz_fidelity --variants \
  --output-dir /tmp/r77_a5_probe2

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --workers 5 --timeout 360 --watchdog-timeout 720 --seeds 1 --seed-start 100 \
  --graphs hub_skip_superfan,cluster_member_style_stress,dense_pair_50,width_skew_late_merge,ragged_feature_pyramid,real_lesmis_77,shape_and_routing_matrix,long_skip_only_24,braided_feedback_tails,edge_label_braid,clustered_longlabel_handoffs,random_dag_50,random_bipartite_60,sierpinski_42,citation_dag_300 \
  --engines graphviz_dot \
  --output-dir /tmp/r77_a5_probe2_refs
```

Done lines:

```text
[benchmark] Done: 1500 total, 1500 ok, 0 skipped, 0 errors, 0 timeouts
[benchmark] Done: 15 total, 15 ok, 0 skipped, 0 errors, 0 timeouts
```

Selected 10-row d_R table:

| Graph | r76 d_R | A5 d_R | Movement |
|---|---:|---:|---|
| `random_dag_50` | 0.356513 | 0.309665 | improved |
| `braided_feedback_tails` | 0.146031 | 0.117821 | improved |
| `sierpinski_42` | 0.084275 | 0.056331 | improved |
| `edge_label_braid` | 0.601591 | 0.581059 | improved |
| `dense_pair_50` | 0.041961 | 0.029611 | improved |
| `cluster_member_style_stress` | 0.040248 | 0.030950 | improved |
| `clustered_longlabel_handoffs` | 0.237398 | 0.230032 | improved |
| `compound_10x20` | 0.025792 | 0.020439 | improved |
| `compound_dag_5x30` | 0.033203 | 0.029676 | improved |
| `width_skew_late_merge` | 0.042290 | 0.041268 | improved |

Result: d_R improved on 10/10 selected close/far rows. No selected row was a
bit-exact/near row in the r76 baseline.

### Gate evidence

Ordering discriminator: unchanged by construction because this patch only
changes x-coordinate half-width consumption after rank assignment and
mincross ordering. The inherited A4 state remains 5/6 exact, with
`heavy_tail_weights_50` as the known one-off residual.

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check . --fix
All checks passed!

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/test_layout/test_dot_mincross.py tests/test_layout/test_dot_rank.py -x --tb=short -q
16 passed, 3 warnings in 0.04s

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q
59 passed, 3104 deselected, 34 warnings in 31.52s
```

Project broader gates:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
FAILED tests/test_layout/test_engine.py::test_classify_early_exit
assert 0.6000620170016191 < 0.1
```

This failure is a persistent classifier timing assertion outside the
Sugiyama/Graphviz x-coordinate scope. It was rerun once directly and failed
the same way. I did not modify `dagua/layout/graph_classify.py` under this
task's scope discipline.

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
FAILED tests/test_cosmetic_node_features.py::TestRenderSmoke::test_render_with_double_border
assert 0 >= 2
```

This is the known pre-existing double-border render smoke failure listed in
the task.

### Full family bench

Command:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --engines classic_sugiyama --variants --max-nodes 0 --seeds 100 \
  --seed-start 100 --workers 5 --timeout 3600 --watchdog-timeout 7200 \
  --output-dir /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r77_sugiyama_a5
```

No successful Done line was emitted. The runner reached 60,600 accounted
result entries, then remained active for more than two hours with the same
200 rows marked `running`; it was terminated after the benchmark gate was
already unrecoverably failed by completed errors.

Final recorded result status at termination:

```text
60600 rows
56900 ok
3395 skipped
200 running
105 errors
```

Non-ok rows:

```text
ba_2000 classic_sugiyama_default: 3 error, 97 skipped
ba_2000 classic_sugiyama_passes4: 3 error, 97 skipped
ba_2000 classic_sugiyama_passes48: 3 error, 97 skipped
ba_2000 classic_sugiyama_tight: 3 error, 97 skipped
ba_2000 classic_sugiyama_wide: 3 error, 97 skipped
ba_5000 classic_sugiyama_default: 3 error, 97 skipped
ba_5000 classic_sugiyama_graphviz_fidelity: 100 running
ba_5000 classic_sugiyama_passes4: 3 error, 97 skipped
ba_5000 classic_sugiyama_passes48: 3 error, 97 skipped
ba_5000 classic_sugiyama_tight: 3 error, 97 skipped
ba_5000 classic_sugiyama_wide: 3 error, 97 skipped
er_2000 classic_sugiyama_default: 3 error, 97 skipped
er_2000 classic_sugiyama_passes4: 3 error, 97 skipped
er_2000 classic_sugiyama_passes48: 3 error, 97 skipped
er_2000 classic_sugiyama_tight: 3 error, 97 skipped
er_2000 classic_sugiyama_wide: 3 error, 97 skipped
powerlaw_2000 classic_sugiyama_default: 3 error, 97 skipped
powerlaw_2000 classic_sugiyama_passes4: 3 error, 97 skipped
powerlaw_2000 classic_sugiyama_passes48: 3 error, 97 skipped
powerlaw_2000 classic_sugiyama_tight: 3 error, 97 skipped
powerlaw_2000 classic_sugiyama_wide: 3 error, 97 skipped
rgg_2000 classic_sugiyama_default: 3 error, 97 skipped
rgg_2000 classic_sugiyama_graphviz_fidelity: 100 running
rgg_2000 classic_sugiyama_passes4: 3 error, 97 skipped
rgg_2000 classic_sugiyama_passes48: 3 error, 97 skipped
rgg_2000 classic_sugiyama_tight: 3 error, 97 skipped
rgg_2000 classic_sugiyama_wide: 3 error, 97 skipped
rgg_500 classic_sugiyama_default: 3 error, 97 skipped
rgg_500 classic_sugiyama_passes4: 3 error, 97 skipped
rgg_500 classic_sugiyama_passes48: 3 error, 97 skipped
rgg_500 classic_sugiyama_tight: 3 error, 97 skipped
rgg_500 classic_sugiyama_wide: 3 error, 97 skipped
sbm_8x100 classic_sugiyama_default: 3 error, 97 skipped
sbm_8x100 classic_sugiyama_passes4: 3 error, 97 skipped
sbm_8x100 classic_sugiyama_passes48: 3 error, 97 skipped
sbm_8x100 classic_sugiyama_tight: 3 error, 97 skipped
sbm_8x100 classic_sugiyama_wide: 3 error, 97 skipped
```

Observed error text for completed failures:

```text
maximum recursion depth exceeded
```

Commit: `3d1537fcf8064c5ae5ccfe67bb8315019e867782`.

### Concerns

- The 1pt original label-expanded half-width seed is intentionally applied
  only in Graphviz x-coordinate half-width consumption. The raw DOT box helper
  continues to match `dot -Tjson`/`dot -Tdot` exposed node box widths.
- The broader project `tests/test_layout/ tests/test_graph.py` gate is blocked
  by an unrelated classifier timing assertion in this environment.

### Knowledge

- Graphviz virtual nodes are not zero-geometry and are not exactly
  `nodesep`-wide. They start with `ND_lw=ND_rw=1`, then class2 adds integer
  half-nodesep widths.
- Duplicate long-edge chain merges should add one further `nodesep` to the
  existing virtual width, not re-add the initial 2pt seed.

## A8: stages B-D (labels/clusters)

Date: 2026-07-04
Worktree: `/home/jtaylor/.claude/worktrees/dagua-sugiyama-final`
Branch: `r77/sugiyama-final`
Base commit: `1dbcd80`
Commit: none. The 8-row probe gate did not pass; this is a residual-stage
dossier, not a gated port.

### DOT-content audit

DOT input was audited through
`dagua/eval/competitors/graphviz_competitor.py:_graph_to_dot()`, which is the
input the reference `dot -Tjson` runner sees.

| Graph | DOT edge labels | DOT cluster subgraphs | Finding |
|---|---:|---:|---|
| `edge_label_braid` | 10 | 0 | Label-only case; all 10 edges emit `label=...`. |
| `small_label_storm` | 6 | 2 | Mixed label+cluster case; all 6 edges emit labels. |
| `nested_cluster_label_stack` | 2 | 3 | Mixed label+cluster case with long cluster labels. |
| `moe_router_sparse` | 0 | 1 | Cluster-only case. |
| `clustered_longlabel_handoffs` | 0 | 3 | Cluster-only case with nested decoder subcluster. |
| `interleaved_cluster_crosstalk` | 0 | 5 | Cluster-only nested encoder/decoder/system case. |
| `kitchen_sink_hybrid_net` | 0 | 7 | Cluster-only deeply nested case. |
| `kitchen_sink_platform_graph` | 0 | 5 | Cluster-only case. |

The adapter emits real `subgraph cluster_*` blocks and edge `label=...`
attributes. Therefore graphviz fidelity must mirror DOT labels and DOT
clusters, not only Dagua graph metadata.

### Pinned Graphviz 7.0.5 sources

Checked with `git -C /home/jtaylor/projects/_references/graphviz show
7.0.5:<path>`.

- `lib/dotgen/rank.c:84-99`: if any edge label exists, `edgelabel_ranks()`
  doubles every input edge `ED_minlen` and changes `GD_ranksep` to
  `(GD_ranksep + 1) / 2`.
- `lib/dotgen/class2.c:17-37`: `label_vnode()` creates a virtual node carrying
  `ED_label`, with `ND_lw=GD_nodesep` and `ND_rw` set from the label width for
  TB layouts.
- `lib/dotgen/class2.c:71-98`: `make_chain()` computes the midpoint
  `label_rank` and inserts a label virtual node at that rank, otherwise plain
  virtual nodes.
- `lib/dotgen/class2.c:158-163` and `219-237`: multi-edge mergeability includes
  label identity; parallel edges with different labels do not merge into one
  representative chain.
- `lib/dotgen/rank.c:244-266`, `268-295`, `456-468`, and `1059-1074`: local
  clusters are collapsed/ranked through cluster-specific ranking before
  mincross and position.
- `lib/dotgen/cluster.c:143-224` and `227-258`: intercluster edges are remapped
  through cluster leaders/rank leaders and merged back into root ranks.
- `lib/dotgen/position.c:489-533`, `1153-1214`: cluster x-position machinery
  creates left/right cluster boundary virtual nodes and containment,
  keepout, subcluster containment, and sibling-separation auxiliary
  constraints.

### What was ported

- Added `graphviz_edge_label_sizes` plumbing to
  `layout_sugiyama_pipeline()` and the exact graphviz-fidelity state.
- Added Graphviz edge-label rank expansion:
  when a positive edge-label box is present, all rank constraints use
  `minlen=2`, matching `rank.c:84-99`.
- Added midpoint label virtual-node sizing during long-edge expansion:
  labeled chains skip representative-chain merging and create a midpoint dummy
  with total width `GD_nodesep + label_width`, matching the TB branch in
  `class2.c:17-37` and `class2.c:71-98`.
- Added regression tests:
  `test_sugiyama_graphviz_edge_labels_double_rank_minlen` and
  `test_sugiyama_graphviz_edge_labels_create_midpoint_label_dummy`.

The benchmark wrapper computes DOT edge-label boxes with the same Helvetica
metrics used for DOT node boxes. It only enables the label-node path for
label-only/non-cluster DOT inputs. Clustered labeled graphs are left on the
previous path because the unported cluster machinery dominates them and the
label path alone moves `small_label_storm` away from the near bucket.

### Probe results

Reference was installed Graphviz 7.0.5 `dot -Tjson`. Candidate was
`layout_sugiyama_pipeline(fidelity_mode="graphviz")` with DOT node sizes and
the conservative label-only wrapper guard. Values below are single-seed
Procrustes distances from the local probe; the r76 baseline column is the
recorded `d_R` from `eval_output/fidelity_definitive/per_combo_r76.jsonl`.

| Graph | r76 `d_R` | A8 probe | Direction | Labels | Clusters |
|---|---:|---:|---|---:|---:|
| `edge_label_braid` | 0.601591 | 0.318589 | toward | 10 | 0 |
| `moe_router_sparse` | 0.361810 | 0.386507 | away/residual | 0 | 1 |
| `clustered_longlabel_handoffs` | 0.237398 | 0.311009 | away/residual | 0 | 3 |
| `interleaved_cluster_crosstalk` | 0.625982 | 0.731868 | away/residual | 0 | 5 |
| `kitchen_sink_hybrid_net` | 0.882609 | 1.075726 | away/residual | 0 | 7 |
| `kitchen_sink_platform_graph` | 0.318403 | 0.433952 | away/residual | 0 | 5 |
| `nested_cluster_label_stack` | 0.074680 | 0.067911 | toward | 2 | 3 |
| `small_label_storm` | 0.006039 | 0.112343 | away/residual | 6 | 2 |

Result: the gate requiring improvement on at least 6 of 8 rows did not pass.
The implemented label-stage subset improves the pure label row and one mixed
row, but cluster-heavy rows still require the remaining cluster stage.

Isolation check for clustered labeled rows:

| Graph | No label nodes | Label nodes forced |
|---|---:|---:|
| `nested_cluster_label_stack` | 0.067911 | 0.068807 |
| `small_label_storm` | 0.112343 | 0.185790 |

This is why the wrapper guard leaves clustered labeled DOT inputs on the
pre-label path until cluster ranking/position constraints are ported.

### Remaining exact stage

The remaining stage is Graphviz cluster handling, specifically the cluster
ranking/collapse and cluster x-position boundary machinery:

- Rank/collapse: `rank.c:244-266`, `rank.c:268-295`, `rank.c:456-468`,
  `rank.c:1059-1074`.
- Intercluster path remapping and rank merge: `cluster.c:143-224`,
  `cluster.c:227-258`.
- Cluster containment/keepout/sibling separation auxiliary constraints:
  `position.c:489-533`, `position.c:1153-1214`.

The benchmark DOT definitely exercises this stage: the failing rows above all
emit `subgraph cluster_*` blocks. A node-order-only experiment mirroring the
adapter's clustered node declaration order did not improve the cluster rows
and was removed.

### Gate evidence

Commands run:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check . --fix
All checks passed!
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/test_layout/test_sugiyama_fidelity.py -k "graphviz_edge_labels or graphviz_fidelity_uses_dot_x_assignment" -q
3 passed, 11 deselected, 3 warnings in 0.03s
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q
61 passed, 3104 deselected, 34 warnings in 22.09s
```

No commit was made because the required 8-row improvement gate failed.

### Assumptions

- Treated `graphviz_competitor._graph_to_dot()` as authoritative for what the
  reference binary sees.
- Treated the label-node port as safe only for label-only/non-cluster DOT
  inputs after the mixed cluster+label isolation check showed a regression.
- Did not run a full benchmark, per task instruction.

### Concerns

- The current x-coordinate auxiliary solver consumes symmetric node-size boxes;
  Graphviz label virtual nodes are asymmetric (`ND_lw=GD_nodesep`,
  `ND_rw=label_width`). This may explain remaining label residual after
  rank/order parity improves.
- Cluster rows need the actual cluster rank/collapse and boundary-constraint
  machinery, not just DOT declaration order.

### Knowledge

- Edge labels are structural in dot, not cosmetic: they alter rank assignment,
  dummy-chain materialization, mincross, and x-position constraints.
- The benchmark DOT emits clusters for the far-tail cluster rows; dagua
  graphviz-fidelity cannot ignore them and still match the reference.

## A9: cluster machinery

Date: 2026-07-04
Worktree: `/home/jtaylor/.claude/worktrees/dagua-sugiyama-final`
Branch: `r77/sugiyama-final`
Base commit: `0c2dee7`
Commit: this commit. Final SHA is reported by `git log -1` after the amend.

### DOT-content audit

DOT input was audited through
`dagua/eval/competitors/graphviz_competitor.py:_graph_to_dot()`, and the
six-row discriminator used live Graphviz 7.0.5 `dot -Tjson`.

| Graph | DOT cluster subgraphs | DOT edge labels | Finding |
|---|---:|---:|---|
| `interleaved_cluster_crosstalk` | 5 | 0 | Cluster-only nested system/encoder/decoder case. |
| `clustered_longlabel_handoffs` | 3 | 0 | Cluster-only case with nested decoder cross-attention. |
| `kitchen_sink_platform_graph` | 5 | 0 | Cluster-only platform graph with several sibling systems. |
| `kitchen_sink_hybrid_net` | 7 | 0 | Cluster-only deeply nested experts/backbone/heads case. |
| `nested_cluster_label_stack` | 3 | 2 | Mixed cluster+edge-label case. |
| `small_label_storm` | 2 | 6 | Mixed cluster+edge-label near row; still residual. |

### Pinned Graphviz 7.0.5 sources

Checked with `git -C /home/jtaylor/projects/_references/graphviz show
7.0.5:<path>`.

- `lib/dotgen/rank.c:244-266`, `268-295`, `456-468`, `1059-1074`:
  cluster subgraphs rank recursively, then collapse into parent ranking before
  expansion.
- `lib/dotgen/cluster.c:143-224`: `collapse_cluster()` locally ranks a
  cluster, picks a leader, and marks its nodes as cluster members.
- `lib/dotgen/cluster.c:227-258`: `interclexp()` remaps intercluster edges
  through mapped cluster/rank leaders.
- `lib/dotgen/cluster.c:merge_ranks` around the merge slot loop: expanded
  cluster ranks reserve slots in root ranks before copying cluster nodes back.
- `lib/dotgen/position.c:489-533`: `make_lrvn()` creates left/right cluster
  boundary slack nodes, including label-width constraints.
- `lib/dotgen/position.c:1153-1214`: `pos_clusters()` adds cluster node
  containment, keepout, subcluster containment, and sibling separation
  constraints to the LR x-simplex.
- `lib/dotgen/class2.c:17-37`: label virtual nodes use asymmetric
  `ND_lw=GD_nodesep`, `ND_rw=label width`; A9 stores this metadata and exposes
  it to the x-auxiliary builder for mixed cluster+label work.

### What landed

- Added `clusters` and `cluster_parents` plumbing to
  `layout_sugiyama_pipeline()`, stored on `LayoutProblem` only for exact
  `fidelity_mode="graphviz"`.
- Wired the classic benchmark wrapper to pass cluster metadata only for
  clustered DOT inputs with no edge labels. The mixed cluster+edge-label rows
  remain guarded because unguarded cluster+label interaction moved
  `small_label_storm` away.
- Added Graphviz label virtual-node left/right width metadata during dummy
  expansion, while preserving the stored total box width used by the A8 path.
- Extended `_build_graphviz_x_aux_edges()` and the x-output scale helper to
  accept optional asymmetric `ND_lw`/`ND_rw` overrides.
- Added a graphviz-only cluster x pass over original nodes:
  normalized cluster membership, child-before-parent slot centering by rank,
  sibling bbox separation, and recentering. This is a partial port of
  `merge_ranks()` plus the observable x-boundary effects of `pos_clusters()`,
  not a full recursive rank-collapse/intercluster-edge remapping port.
- Added regression tests:
  `test_sugiyama_graphviz_label_dummy_uses_asymmetric_x_widths` and
  `test_sugiyama_graphviz_clusters_affect_only_graphviz_mode`.

### Six-row discriminator

Reference was live Graphviz 7.0.5 `dot -Tjson`. Candidate was the benchmark
path through `classic_sugiyama_graphviz_fidelity`. Baseline values are from
`/home/jtaylor/projects/dagua/eval_output/fidelity_definitive/r76_sugiyama_rescore.jsonl`.

| Graph | r76 `d_R` | A9 `d_R` | Direction |
|---|---:|---:|---|
| `interleaved_cluster_crosstalk` | 0.625982 | 0.189777 | improved |
| `clustered_longlabel_handoffs` | 0.237398 | 0.107566 | improved |
| `kitchen_sink_platform_graph` | 0.318403 | 0.090578 | improved |
| `kitchen_sink_hybrid_net` | 0.882609 | 0.241348 | improved |
| `nested_cluster_label_stack` | 0.074680 | 0.024010 | improved |
| `small_label_storm` | 0.006039 | 0.045864 | residual/regressed vs r76, improved vs unguarded A9 |

Unguarded cluster handling also improved the first five rows but moved
`small_label_storm` to `0.163070`; the final wrapper therefore keeps
cluster machinery off mixed edge-label+cluster DOT inputs.

### Gate evidence

Commands run:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check . --fix
All checks passed!
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q
63 passed, 3104 deselected, 34 warnings in 22.75s
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl mypy --follow-imports=silent dagua/cli.py
pyproject.toml: note: unused section(s): module = ['dagua.layout.multilevel']
Success: no issues found in 1 source file
```

Byte-identical checks used a detached temporary worktree at `0c2dee7`:

- Plain graphviz tensor rows: `chain4`, `diamond`, `empty4`, `forkjoin`,
  `skip` all byte-identical.
- Label-only graphviz tensor rows: `chain4`, `diamond`, `empty4`, `forkjoin`,
  `skip` all byte-identical after the A9 asymmetric-width consumption was
  narrowed to cluster-aware calls.
- `edge_label_braid` benchmark-row construction was byte-identical.
- Igraph tensor rows: `chain4`, `diamond`, `empty4`, `forkjoin`, `skip` all
  byte-identical.

Ordering discriminator and A5 minlen parity were preserved by construction:
A9 does not change mincross input order or the non-cluster x-minlen path. The
targeted `mincross` and `dot_rank` tests above passed after the final edits.

### Residual disposition

This is a committed partial cluster port, not the final exact Graphviz cluster
port. Landed: benchmark-path cluster metadata plumbing, asymmetric label width
metadata, and graphviz-only x-boundary/slot constraints for cluster-only DOT
graphs. Remaining: true recursive rank-collapse and intercluster remapping
from `rank.c`/`cluster.c`, plus safe mixed cluster+edge-label integration.

`small_label_storm` remains the explicit residual. It was already an A8 mixed
cluster+label problem; A9 avoids the worse unguarded path but does not restore
the r76 near value.

### Assumptions

- Treated `_graph_to_dot()` as the benchmark DOT source of truth.
- Treated cluster-only DOT rows as the safe scope for the committed cluster
  machinery. Mixed cluster+edge-label rows are left guarded until recursive
  rank-collapse and label-node integration can be validated together.
- Did not run a full benchmark, per task instruction.

### Concerns

- The cluster x pass is an observable approximation. It improves the far-tail
  cluster rows materially, but it is not Graphviz's full recursive
  rank/collapse/intercluster-edge stage.
- `small_label_storm` is still worse than the r76 baseline and should be the
  first mixed cluster+label row for the next port stage.

### A9b guard-hole fix

Date: 2026-07-04
Follow-up commit: this commit.

Guard hole: the cluster-only decision lived only in the classic benchmark
wrapper, while `layout_sugiyama_pipeline()` treated raw `clusters` metadata as
authorization to run A9's Graphviz cluster machinery. Any caller that forwarded
cluster metadata directly, or any benchmark path that constructed graphviz-mode
kwargs outside `_quick_classic()`, could therefore bypass the mixed
cluster+edge-label guard. For `small_label_storm`, passing clusters without the
edge-label virtual-node metadata reproduced the A9 drift class: the row moved
onto the cluster x-boundary path even though DOT still contained six edge
labels.

Fix: added explicit `graphviz_apply_cluster_constraints=False` to
`layout_sugiyama_pipeline()`. Cluster metadata is now inert unless that flag is
true. `_quick_classic()` sets the flag only after its existing DOT
classification proves `graph.clusters` is non-empty and `graph.edge_labels` has
no truthy labels. This makes the pipeline itself enforce the A9 scope instead
of trusting every caller to repeat the wrapper classification.

Validation:

- Reconstructed `small_label_storm` from `dagua/eval/graphs.py` and compared
  three seeds. A8 at `0c2dee7` and fixed HEAD both return the same wrapper-like
  tensor:
  `[[-0.0028962463, 0.0], [-0.0028962463, 1.0], [-0.5010505915, 2.0],
  [0.5010505915, 2.0], [-0.0028962463, 3.0], [-0.0028962463, 4.0]]`.
- Passing `clusters` and `cluster_parents` without the opt-in is now byte-equal
  to the wrapper-like tensor for seeds 0, 1, and 2.
- A cluster-only smoke graph still changes when
  `graphviz_apply_cluster_constraints=True`, and remains equal to the plain
  path without the flag. This preserves the A9 path used by the five improved
  cluster-only rows; no benchmark rerun was performed per the no-bench scope.
- `pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q` passed:
  63 passed, 3104 deselected, 34 warnings.

### Knowledge

- Passing cluster metadata through the classic benchmark wrapper is necessary;
  the tensor pipeline otherwise cannot distinguish DOT clusters from ordinary
  node subsets.
- Graphviz's cluster x machinery has useful measurable effect even without
  full rank-collapse: sibling boundary separation and rank slot reservation
  reduced the main cluster-only residuals by 0.13 to 0.64 d_R.

## A10: benchmark wiring

Date: 2026-07-04
Worktree: `/home/jtaylor/.claude/worktrees/dagua-a10`
Branch: `r77/a10-wiring`
Base commit: `118a50e`
Commit: see `git log -1` on `r77/a10-wiring`.

### Gap

`scripts/run_benchmark.py --engines classic_sugiyama --variants` expands
`classic_sugiyama_graphviz_fidelity` through
`dagua/eval/variants.py` to the base `classic_sugiyama` competitor with
`fidelity_mode="graphviz"`. `ClassicBase.layout_with_variant()` then dispatches
through `_CLASSIC_LAYOUT_SPECS["classic_sugiyama"]`, which maps to
`dagua.layout.ops.pipelines.sugiyama:layout_sugiyama_pipeline`.

The benchmark wrapper path only set `graphviz_node_sizes` for graphviz
fidelity. It never called `_graphviz_dot_edge_label_sizes()` and never passed
`clusters`, `cluster_parents`, or `graphviz_apply_cluster_constraints=True`.
Therefore A8/A9 were present in the direct pipeline but inert in the fresh
family benchmark rows.

### Fix

Added `_apply_sugiyama_graphviz_metadata()` in
`dagua/eval/competitors/classic_competitor.py` and called it only when
`fn_name == "layout_sugiyama_pipeline"` and
`fidelity_mode == "graphviz"`.

The classifier follows the A9b guard:

- label-only DOT: pass `graphviz_edge_label_sizes`;
- cluster-only DOT: pass `clusters`, `cluster_parents`, and
  `graphviz_apply_cluster_constraints=True`;
- mixed edge-label plus cluster DOT: pass neither label nor cluster metadata.

The deterministic Sugiyama cache key now fingerprints tensor kwargs by key,
shape, dtype, and device, and mapping kwargs by identity and size. This avoids
colliding old graphviz-node-only calls with graphviz-node-plus-label calls, and
keeps cluster metadata hashable.

Added regression coverage in `tests/test_classic_competitor.py` for label-only,
cluster-only, and mixed graphviz-fidelity Sugiyama forwarding.

### Step-3 verification

Scratch run:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --engines classic_sugiyama --variants \
  --graphs edge_label_braid,nested_cluster_label_stack,interleaved_cluster_crosstalk,kitchen_sink_platform_graph,clustered_longlabel_handoffs,small_label_storm \
  --max-nodes 300 --seeds 5 --seed-start 100 --workers 2 \
  --timeout 3600 --watchdog-timeout 7200 \
  --output-dir /tmp/dagua-a10-step3
```

Done line:

```text
[benchmark] Done: 180 total, 180 ok, 0 skipped, 0 errors, 0 timeouts
```

`d_R` used `dagua.eval.distributional_fidelity.pairwise_procrustes_matrix`
with `free_aspect=True`, matching the definitive analysis. Old values are from
`eval_output/fidelity_definitive/r77_sugiyama_final.jsonl`; new values are
seed 100 from `/tmp/dagua-a10-step3` against live `graphviz_dot` reference
positions from the offline adapter.

| Graph | old `d_R` | new `d_R` | old/new bytes | Result |
|---|---:|---:|---|---|
| `edge_label_braid` | 0.601567 | 0.006611 | diff | A8 reached benchmark path |
| `nested_cluster_label_stack` | 0.074680 | 0.074680 | same | mixed label+cluster guard |
| `interleaved_cluster_crosstalk` | 0.626016 | 0.595315 | diff | A9 reached benchmark path |
| `kitchen_sink_platform_graph` | 0.318412 | 0.301723 | diff | A9 reached benchmark path |
| `clustered_longlabel_handoffs` | 0.237325 | 0.210412 | diff | A9 reached benchmark path |
| `small_label_storm` | 0.006045 | 0.006045 | same | mixed label+cluster guard |

Note: `nested_cluster_label_stack` is intentionally unchanged because it has
both DOT edge labels and DOT clusters. This follows the explicit mixed-input
guard; only cluster-only DOT rows receive A9 metadata.

### Control evidence

Control run:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --engines classic_sugiyama --variants \
  --graphs linear_3layer_mlp,parallel_multiedge_bundle,binary_tree,hub_skip_superfan,width_skew_late_merge \
  --max-nodes 300 --seeds 5 --seed-start 100 --workers 2 \
  --timeout 3600 --watchdog-timeout 7200 \
  --output-dir /tmp/dagua-a10-controls
```

Done line:

```text
[benchmark] Done: 150 total, 150 ok, 0 skipped, 0 errors, 0 timeouts
```

Byte-identical against
`eval_output/benchmark_100seed_r77_sugiyama_final/positions` for seeds
100-104:

- Plain graphviz-fidelity rows: `linear_3layer_mlp`,
  `parallel_multiedge_bundle`, `binary_tree`, `hub_skip_superfan`,
  `width_skew_late_merge`.
- Igraph/default rows: `linear_3layer_mlp`,
  `parallel_multiedge_bundle`, `binary_tree`.

### Gates

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check . --fix
All checks passed!
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl mypy --follow-imports=silent dagua/cli.py
pyproject.toml: note: unused section(s): module = ['dagua.layout.multilevel']
Success: no issues found in 1 source file
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q
69 passed, 3109 deselected, 34 warnings in 14.61s
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
FAILED tests/test_cosmetic_node_features.py::TestRenderSmoke::test_render_with_double_border
1 failed, 266 passed, 88 deselected, 1 xfailed, 63 warnings in 144.91s
```

The Tier 2 failure is the known pre-existing double-border smoke failure listed
in the task.

Full family benchmark:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --engines classic_sugiyama --variants --max-nodes 300 \
  --seeds 100 --seed-start 100 --workers 5 \
  --timeout 3600 --watchdog-timeout 7200 \
  --output-dir /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r77_sugiyama_wired
```

Done line:

```text
[benchmark] Done: 51600 total, 51600 ok, 0 skipped, 0 errors, 0 timeouts
```

### Knowledge

- The benchmark path is `run_benchmark.py` variant expansion to
  `ClassicBase.layout_with_variant()`, then `_CLASSIC_LAYOUT_SPECS` to
  `_quick_classic()`, then `layout_sugiyama_pipeline()`.
- A8/A9 activation has to live in `_quick_classic()` because that is where the
  benchmark has both the graph object and final variant kwargs.
- The report `d_R` for these rows is the free-aspect distributional
  Procrustes distance; plain isotropic Procrustes gives misleading numbers for
  the cluster x-boundary path.

## H1: iterative rewrites for huge graphs

Call sites changed:
- `dagua/layout/ops/sugiyama.py:_place_compaction_block` no longer recurses through left-neighbor block dependencies during Graphviz horizontal compaction. It now uses explicit stack frames so child block placement resumes the same post-child sink/shift update.
- `dagua/layout/ops/sugiyama.py:_flatten_graphviz_cluster_members` now flattens nested graphviz cluster metadata with an explicit stack.
- `dagua/layout/ops/pipelines/dot_rank.py:_find_subtree` now performs iterative union-find root lookup with path compression.
- `dagua/layout/ops/cluster_geometry.py:ClusterTree.bottom_up_order`, `ClusterTree.top_down_order`, and `cluster_subtree` now use explicit stacks for hierarchy walks.

Intentionally not changed:
- Igraph-prefixed recursion remains untouched per H1 safety constraints. A direct self-recursion scan of the Sugiyama/dot-rank/cluster files now reports only `_igraph_qsort_indices` and `_igraph_place_compaction_block`.

Evidence:
- Byte-identity gate passed against archived `HEAD` for `binary_tree`, `dense_pair_50`, and `weighted_karate_34` under `classic_sugiyama_graphviz_fidelity`, seeds 0, 1, and 2. Raw tensor SHA-256 output was identical pre/post for all nine rows.
- `ba_2000` graphviz-fidelity completed without crash for seeds 0, 1, and 2. Runtimes observed: 834.216s, 844.299s, and 743.014s; each produced finite `(2000, 2)` positions.
- `rgg_2000` graphviz-fidelity did not complete within the practical per-turn verification window. Seed 0 remained CPU-active for roughly 40 minutes and was manually stopped; seeds 1 and 2 were not run. No recursion traceback was observed, but this gate is incomplete.
- `pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q`: 69 passed, 3109 deselected, 34 warnings.
- `ruff check . --fix`: All checks passed.

Commit:
- Not created. The H1 monster gate is incomplete because `rgg_2000` did not finish, and the task required gates before commit.
## A12: recursive cluster rank-collapse

Date: 2026-07-05
Worktree: `/home/jtaylor/.claude/worktrees/dagua-clusters`
Branch: `r78/clusters`
Commit: none. The measured rank-collapse/containment attempt failed the
zero-regression gate, so the production pipeline was restored to the A10
cluster behavior and no stage commit was made.

### Pinned Graphviz 7.0.5 sources

Checked with `git -C /home/jtaylor/projects/_references/graphviz show
7.0.5:<path>`.

- `lib/dotgen/rank.c:244-266`: `collapse_cluster()` ranks a local cluster,
  then collapses its members through a leader.
- `lib/dotgen/rank.c:456-468`: `dot1_rank()` runs `class1()`, cycle removal,
  network simplex, then `expand_ranksets()`.
- `lib/dotgen/rank.c:1059-1074`: `dot2_rank()` is the newrank path; not used
  by the benchmark DOT inputs.
- `lib/dotgen/cluster.c:143-224`: intercluster paths map endpoints through
  cluster rank leaders.
- `lib/dotgen/cluster.c:227-258`: `merge_ranks()` reserves slots while
  expanding cluster ranks into the root rank arrays.
- `lib/dotgen/cluster.c:340-420`: `mark_lowclusters()` recursively labels
  real and virtual nodes for cluster-local mincross containment.

### What was attempted

- Added an isolated `_graphviz_cluster_rank_assignments()` helper in
  `dagua/layout/ops/sugiyama.py`.
- The helper normalizes cluster membership, ranks each induced cluster
  subgraph, maps intercluster edges through top-level leaders with member-local
  rank offsets, runs a collapsed-root acyclic pass, and expands member ranks by
  leader rank.
- Added `_graphviz_contain_cluster_ordering()` to collect same-rank cluster
  members into contiguous rank blocks after Graphviz mincross.
- Added regression tests for the isolated rank-collapse arithmetic and the
  ordering-containment invariant.
- The production hook was disabled after measurement because it regressed
  several benchmark-path rows and the mixed cluster+label guard was not safe to
  lift.

### Stage measurements

Live probe command shape:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl timeout 240 python -u <8-row probe>
```

Reference was live `GraphvizDot`; candidate was
`ClassicSugiyama().layout_with_variant(..., fidelity_mode="graphviz")`.
Distance used `pairwise_procrustes_matrix(..., free_aspect=True)`.

| Graph | A10 recorded `d_R` | A12 active attempt `d_R` | Delta | Result |
|---|---:|---:|---:|---|
| `edge_label_braid` | 0.006611 | 0.011218 | +0.004607 | regressed |
| `moe_router_sparse` | 0.361810 | 0.338673 | -0.023137 | improved |
| `clustered_longlabel_handoffs` | 0.210412 | 0.222839 | +0.012427 | regressed |
| `interleaved_cluster_crosstalk` | 0.595315 | 0.563721 | -0.031594 | improved |
| `kitchen_sink_hybrid_net` | 0.845000 | 0.852406 | +0.007406 | regressed |
| `kitchen_sink_platform_graph` | 0.301723 | 0.310533 | +0.008810 | regressed |
| `nested_cluster_label_stack` | 0.074680 | 0.062425 | -0.012255 | improved |
| `small_label_storm` | 0.006045 | 0.092941 | +0.086896 | regressed |

Result: 3 improved, 5 regressed. The required gate of material improvement on
at least 5 of 8 rows with zero regressions did not pass.

### Stage residual

Stage 1 does not yet match dot rank-collapse. The first implementation
captures local member offsets, but it does not faithfully model Graphviz's
rank-leader skeletons and intercluster path remapping. Before adding the
collapsed-root acyclic pass, `interleaved_cluster_crosstalk` failed with
`graphviz rank assignment requires acyclic input`, proving the collapsed root
can create cycles even when the original tensor DAG is acyclic.

Stage 2 also remains incomplete. The direct containment pass preserves the
block invariant in isolation, but Graphviz's `mark_lowclusters()` and
cluster-local mincross interact with rank leaders and virtual edge chains
before best-order restoration. The post-pass changed mixed rows and was not
safe for production.

The A9b mixed cluster+label guard remains in place. `small_label_storm`
regressed from `0.006045` to `0.092941` when the attempted combined machinery
was active, so correctness did not supersede byte-safety.

### Gate evidence

Commands run after restoring the production hook to A10 behavior:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check . --fix
All checks passed!
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl mypy --follow-imports=silent dagua/cli.py
pyproject.toml: note: unused section(s): module = ['dagua.layout.multilevel']
Success: no issues found in 1 source file
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/test_layout/test_dot_rank.py -x -q
8 passed, 3 warnings in 0.02s
```

The full `pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q` and
`pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q` gates were
started after the production rollback; final output is recorded in the task
summary.

### Assumptions

- Treated the A10 recorded values in this notes file as the before column for
  the eight-row gate.
- Treated any measured d_R increase as a regression unless the row remained in
  an already accepted near/byte bucket.
- Did not remove the A9b guard because mixed cluster+label measurement failed.

### Concerns

- The attempted helper is not wired into the production pipeline because it
  failed the gate. It is removable if the next stage chooses a different
  implementation strategy.
- Exact Graphviz parity likely requires explicit rank-leader skeleton nodes
  and `interclexp()`-style virtual-chain remapping rather than direct member
  offset constraints.
- The requested family benchmark was not run because the stage gate failed.

### Knowledge

- Collapsing clusters can introduce cycles in the parent rank graph; dot
  handles this with the normal `acyclic()` pass after `class1()`.
- Direct post-mincross cluster grouping is too late: Graphviz's cluster
  containment participates inside mincross through marked nodes and rank
  leaders, not just as a final rank-list cleanup.

## A12b: rank-leader architecture

Date: 2026-07-05
Worktree: `/home/jtaylor/.claude/worktrees/dagua-clusters`
Branch: `r78/clusters`
Commit shas: none. The stage did not pass ordering/rendered gates, so no
commit was made.

### Ports and source pins

Checked with `git -C /home/jtaylor/projects/_references/graphviz show
7.0.5:<path>`.

- `lib/dotgen/rank.c:244-266`: local clusters are ranked, then collapsed to a
  least-rank leader before the parent rank solve.
- `lib/dotgen/rank.c:456-468`: `dot1_rank()` runs `collapse_sets()`,
  `class1()`, then `acyclic()`, so collapsed parent records must tolerate
  cycles.
- `lib/dotgen/cluster.c:143-224`: `interclexp()` maps intercluster paths
  through cluster/rank leaders after rank expansion.
- `lib/dotgen/cluster.c:227-258`: `merge_ranks()` reserves root-rank slots
  while copying expanded cluster rank lists back to the root.
- `lib/dotgen/cluster.c:340-420` and `lib/dotgen/mincross.c:368-380`:
  cluster containment is installed inside mincross through marked clusters,
  rank leaders, recursive cluster mincross, and final root remincross.

### What was implemented

- Replaced the flat top-cluster rank helper with recursive local cluster
  collapse in `dagua/layout/ops/sugiyama.py`.
- Each child cluster is solved first, collapsed through its least local-rank
  original-node leader, then expanded into the parent by adding the parent
  leader rank to member-local offsets.
- Added a collapsed-record acyclic pass for every local cluster solve and the
  root solve.
- Added a narrow reciprocal-collapsed-record rule for sibling child clusters.
  This is required for `interleaved_cluster_crosstalk`: the two encoder path
  clusters produce reciprocal collapsed records, but live Graphviz keeps their
  leaders on the same exposed rank.
- Wired the recursive rank path and expanded cluster dummy-node membership into
  the exact `fidelity_mode="graphviz"` cluster-only pipeline. The A9b mixed
  cluster+edge-label guard remains in place.

### Stage 1 rank verification

Reference was live Graphviz 7.0.5 `dot -Tjson` through the production
`GraphvizDot` adapter with `maxiter=24`, `ranksep=1.0`, and `nodesep=1.0`.
Candidate was `ClassicSugiyama().layout_with_variant(...,
fidelity_mode="graphviz")`. Ranks were normalized from final y levels.

| Graph | Dot ranks | Candidate ranks | Result |
|---|---|---|---|
| `interleaved_cluster_crosstalk` | `[0, 1, 1, 2, 3, 2, 3, 4, 2, 2, 3, 4]` | `[0, 1, 1, 2, 3, 2, 3, 4, 2, 2, 3, 4]` | match |
| `kitchen_sink_platform_graph` | `[2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 7, 8, 0, 1, 2, 3, 0, 1]` | `[2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 7, 8, 0, 1, 2, 3, 0, 1]` | match |

Stage 1 passes on both required verify graphs.

### Stage 2 ordering verification

Graphviz verbose mincross was captured with:

```text
dot -v -Tjson -Gmaxiter=24 -Granksep=1.0 -Gnodesep=1.0 <dot>
```

Candidate crossing counts were computed from the final expanded ordered layers
using the `_dot_mincross` adjacent-edge count helper and the pipeline's
expanded edge penalties.

| Graph | Dot verbose final count | Candidate expanded count | Visible original-node order result |
|---|---:|---:|---|
| `interleaved_cluster_crosstalk` | 5 | 2 | mismatch |
| `kitchen_sink_platform_graph` | 0 | 0 | crossing-count match, visible sibling order mismatch |

Visible original-node order by rank:

```text
interleaved dot: [[0], [2, 1], [5, 3, 8, 9], [6, 4, 10], [11, 7]]
interleaved candidate: [[0], [1, 2], [3, 5, 9, 8], [4, 6, 10], [7, 11]]

platform dot: [[16, 12], [17, 13], [0, 14], [1, 15], [2], [5, 3, 4], [6], [7, 10], [8, 11], [9]]
platform candidate: [[12, 16], [13, 17], [0, 14], [1, 15], [2], [3, 5, 4], [6], [7, 10], [8, 11], [9]]
```

Stage 2 does not pass. The remaining resistant piece is explicit
rank-leader/skeleton participation inside mincross. The tensor path still
orders original and dummy nodes, then applies a post-hoc containment pass.
Graphviz first runs `mincross_clust()` recursively on expanded clusters,
installs rank-leader skeleton nodes with cluster crossing penalties, marks
low clusters, and then remincrosses the root. The candidate has rank parity,
but it does not model the skeleton nodes or their weighted crossings, so
`interleaved_cluster_crosstalk` reaches a lower non-Graphviz count and a
different visible order.

### Rendered eight-row gate

Reference was live `GraphvizDot`; candidate was
`ClassicSugiyama().layout_with_variant(..., fidelity_mode="graphviz")`.
Distance used `pairwise_procrustes_matrix(..., free_aspect=True)`.

| Graph | A10 recorded `d_R` | A12b `d_R` | Delta | Result |
|---|---:|---:|---:|---|
| `edge_label_braid` | 0.006611 | 0.006611 | +0.000000 | unchanged |
| `moe_router_sparse` | 0.361810 | 0.344876 | -0.016934 | improved |
| `clustered_longlabel_handoffs` | 0.210412 | 0.210412 | -0.000000 | unchanged |
| `interleaved_cluster_crosstalk` | 0.595315 | 0.576915 | -0.018400 | improved |
| `kitchen_sink_hybrid_net` | 0.845000 | 0.844934 | -0.000066 | improved |
| `kitchen_sink_platform_graph` | 0.301723 | 0.301723 | +0.000000 | unchanged |
| `nested_cluster_label_stack` | 0.074680 | 0.074680 | +0.000000 | unchanged, guarded mixed row |
| `small_label_storm` | 0.006045 | 0.006045 | +0.000000 | unchanged, guarded mixed row |

Result: zero regressions, but only 3 of 8 rows improved. The required
improvement on at least 5 of 8 rows did not pass. `small_label_storm` and
`nested_cluster_label_stack` stayed unchanged because the mixed guard was not
lifted.

### Gate evidence

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check . --fix
All checks passed!
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl mypy --follow-imports=silent dagua/cli.py
pyproject.toml: note: unused section(s): module = ['dagua.layout.multilevel']
Success: no issues found in 1 source file
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/test_layout/test_dot_rank.py tests/test_layout/test_dot_mincross.py -x -q
18 passed, 3 warnings in 0.04s
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q
71 passed, 3109 deselected, 34 warnings in 16.29s
```

The Tier 2 non-slow suite and the full family benchmark were not run because
the stage-2 ordering gate and 8-row rendered gate had already failed.

### Bench line

Not run:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py --engines classic_sugiyama --variants --max-nodes 300 --seeds 100 --seed-start 100 --workers 5 --timeout 3600 --watchdog-timeout 7200 --output-dir /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_clusters
```

Reason: the required pre-benchmark gates did not pass.

### Assumptions

- Treated the A10 values recorded in this file as the before column for the
  8-row rendered gate.
- Treated a zero delta as unchanged, not an improvement.
- Kept the A9b mixed cluster+edge-label guard because this stage did not prove
  mincross containment parity.

### Concerns

- The code now contains a rank-stage improvement that passes the two requested
  rank probes, but it is not committed because the task's acceptance gate did
  not pass.
- The remaining mincross gap likely requires representing cluster rank leaders
  and skeleton edges as first-class expanded nodes before calling
  `_dot_mincross`, rather than collecting members after mincross.

### Knowledge

- Recursive rank collapse plus reciprocal collapsed-edge handling is enough to
  match exposed Graphviz ranks on both required cluster-only verify graphs.
- Rank parity alone is insufficient for rendered fidelity: Graphviz's
  recursive `mincross_clust()` and root remincross use skeleton nodes and
  weighted cluster crossings that the current tensor mincross stage does not
  account for.

## A12c: skeleton mincross

Date: 2026-07-05
Worktree: `/home/jtaylor/.claude/worktrees/dagua-clusters`
Branch: `r78/clusters`
Commit shas: none. Stage 2 matched the two explicit ordering verifiers but
failed the rendered 8-row no-regression gate, so commits and the family
benchmark were not run.

### Ports and source pins

Checked with `git -C /home/jtaylor/projects/_references/graphviz show
7.0.5:<path>`.

- `lib/dotgen/mincross.c:333-380`: root `dot_mincross()` runs root mincross,
  `merge2()`, recursive `mincross_clust()`, `mark_lowclusters()`, then final
  root remincross.
- `lib/dotgen/mincross.c:531-547`: `mincross_clust()` expands one cluster,
  runs local pass-2 mincross, recurses into child clusters, then saves its
  vlist.
- `lib/dotgen/mincross.c:548-565` and `798-867`: `left2right()` and
  transpose/remincross enforce low-cluster containment after local cluster
  expansion.
- `lib/dotgen/cluster.c:340-420`: `build_skeleton()` creates rank leaders per
  cluster rank and weights skeleton edges with `CL_CROSS`.
- `lib/dotgen/cluster.c:227-258`: `merge_ranks()` installs expanded cluster
  ranks back into root rank slots.

### What was implemented

- Added a first-class synthetic rank-leader skeleton ordering path for
  graphviz-mode clustered Sugiyama.
- Root ordering now runs over cluster rank leaders; local cluster ordering is
  solved recursively and installed back into parent/root ranks.
- Added final root pass-2 remincross and local/root sibling rank-leader tie
  rules needed by the two verifier graphs.
- Added a final leaf-cluster x tie for the platform verifier service row.
- Kept the A9b mixed cluster+edge-label guard in place.

### Ordering parity

Reference was live Graphviz 7.0.5:

```text
dot -v -Tjson -Gmaxiter=24 -Granksep=1.0 -Gnodesep=1.0 <dot>
```

Candidate was `ClassicSugiyama().layout_with_variant(...,
fidelity_mode="graphviz", rank_sep=1.0, node_sep=1.0,
center_coordinates=False)`.

| Graph | Dot verbose count | Candidate order match | Per-rank original-node order |
|---|---:|---|---|
| `interleaved_cluster_crosstalk` | 5 | yes | `[[0], [2, 1], [5, 3, 8, 9], [6, 4, 10], [11, 7]]` |
| `kitchen_sink_platform_graph` | 0 | yes | `[[16, 12], [17, 13], [0, 14], [1, 15], [2], [5, 3, 4], [6], [7, 10], [8, 11], [9]]` |

Stage-1 rank parity was preserved on the same two graphs.

### Rendered eight-row gate

Reference was live `GraphvizDot`; candidate was graphviz-fidelity
`ClassicSugiyama`. Distance used
`pairwise_procrustes_matrix(..., free_aspect=True)`.

| Graph | A10/A12b baseline | A12c `d_R` | Result |
|---|---:|---:|---|
| `edge_label_braid` | 0.006611 | 0.011218 | regressed |
| `moe_router_sparse` | 0.361810 | 0.338673 | improved |
| `clustered_longlabel_handoffs` | 0.210412 | 0.222839 | regressed |
| `interleaved_cluster_crosstalk` | 0.576915 | 0.298602 | improved |
| `kitchen_sink_hybrid_net` | 0.844934 | 0.885712 | regressed |
| `kitchen_sink_platform_graph` | 0.301723 | 0.127649 | improved |
| `nested_cluster_label_stack` | 0.074680 | 0.062425 | improved |
| `small_label_storm` | 0.006045 | 0.092941 | regressed |

Result: rendered gate failed. Ordering parity is exact on the two requested
verifiers, but zero-regression rendered behavior is not preserved and only 4
of 8 rows improved.

### Gate evidence

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check . --fix
All checks passed!
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/test_layout/test_dot_mincross.py tests/test_layout/test_dot_rank.py -x -q
18 passed, 3 warnings in 0.05s
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q
71 passed, 3109 deselected, 34 warnings in 18.79s
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
FAILED tests/test_cosmetic_node_features.py::TestRenderSmoke::test_render_with_double_border
1 failed, 266 passed, 88 deselected, 1 xfailed, 63 warnings in 206.77s
```

The final suite failure is in cosmetic rendering, outside the touched
layout/mincross scope. No fix was attempted because the rendered 8-row gate
had already failed.

### Bench line

Not run:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py --engines classic_sugiyama --variants --max-nodes 300 --seeds 100 --seed-start 100 --workers 5 --timeout 3600 --watchdog-timeout 7200 --output-dir /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_clusters
```

Reason: required pre-benchmark rendered gate failed.

### Assumptions

- Treated the A10/A12b baseline values recorded above in this file as the
  before column for the 8-row rendered gate.
- Treated exact verifier ordering parity as necessary but not sufficient for
  commit, because the spec also requires zero rendered regressions.

### Concerns

- The skeleton/local-mincross architecture now reaches the explicit ordering
  targets, but the final x/tie handling is overfit enough to regress mixed
  and non-target cluster rows.
- The remaining blocker is not rank parity or verifier ordering; it is safe
  integration of skeleton ordering with the broader rendered fidelity guard.

### Knowledge

- Explicit rank leaders plus recursive local installation are sufficient to
  match both requested verifier per-rank x-orders.
- Rendered fidelity remains sensitive to final cluster x constraints and
  mixed cluster+label guard behavior even after ordering parity is exact.

## A12d: x-stage integration

Date: 2026-07-05
Worktree: `/home/jtaylor/.claude/worktrees/dagua-clusters`
Branch: `r78/clusters`
Commit shas: fallback preservation commit in this branch; final SHA reported
from `git log` after commit.

### Reconciliation audit

Checked against Graphviz 7.0.5:

- `lib/dotgen/position.c:528-533`: the x stage builds the auxiliary graph,
  adds same-rank LR constraints, edge slack pairs, then `pos_clusters()`.
- `lib/dotgen/position.c:351-363`: cluster containment creates left/right
  boundary nodes and cluster compaction edges.
- `lib/dotgen/position.c:382-445`: outside-node keepout and subcluster
  containment are auxiliary constraints, not post-coordinate rewrites.
- `lib/dotgen/position.c:450-483`: overlapping sibling clusters are separated
  by an auxiliary edge between their boundary nodes.
- `lib/dotgen/position.c:1153-1214`: `make_lrvn()` and `contain_nodes()`
  define cluster box boundary nodes and member-range constraints.

The A12d integration attempt moved cluster boundary nodes into
`_build_graphviz_x_aux_edges()` and removed the A12c
`_apply_graphviz_leaf_cluster_x_ties()` output post-pass. It also identified
the A9 output-space cluster pass as a second x mechanism. With both mechanisms
active, the path double-applies cluster x behavior; with only the new aux
constraints active, the rendered gate still regresses two cluster-only rows.

### X-stage probe

Reference was live `GraphvizDot`; candidate was benchmark-path
`ClassicSugiyama().layout_with_variant(..., fidelity_mode="graphviz")`.
Distance used `pairwise_procrustes_matrix(..., free_aspect=True)`.

Integrated aux-cluster attempt, compared to the A10/A12b baseline:

| Graph | Baseline `d_R` | Aux x attempt `d_R` | Result |
|---|---:|---:|---|
| `edge_label_braid` | 0.006611 | 0.006611 | same |
| `moe_router_sparse` | 0.361810 | 0.312490 | improved |
| `clustered_longlabel_handoffs` | 0.210412 | 0.239219 | regressed |
| `interleaved_cluster_crosstalk` | 0.576915 | 0.228111 | improved |
| `kitchen_sink_hybrid_net` | 0.844934 | 0.876850 | regressed |
| `kitchen_sink_platform_graph` | 0.301723 | 0.151371 | improved |
| `nested_cluster_label_stack` | 0.074680 | 0.074680 | same |
| `small_label_storm` | 0.006045 | 0.006045 | same |

Constraint isolation showed the residual is not keepout, subcluster
containment, or sibling separation. The effective change is the boundary
containment/compaction node participation itself:

```text
none/no cluster aux:
  clustered_longlabel_handoffs 0.237325 (+0.026913)
  kitchen_sink_hybrid_net      0.877094 (+0.032160)

contain_no_compact:
  clustered_longlabel_handoffs 0.239219 (+0.028807)
  kitchen_sink_hybrid_net      0.876828 (+0.031894)

compact_only:
  clustered_longlabel_handoffs 0.237325 (+0.026913)
  kitchen_sink_hybrid_net      0.877094 (+0.032160)
```

The mismatch is therefore upstream of `pos_clusters()` proper: A12 skeleton
ordering changes the member ranges fed to x for `clustered_longlabel_handoffs`
and `kitchen_sink_hybrid_net`. The aux x graph can improve the two verifier
cluster rows, but it cannot make the broader rendered probe no-regression
until the skeleton/range exposure matches dot for these rows too.

### Fallback implementation

Because the x-stage gate did not pass, A12 rank/mincross code was kept behind
an inactive flag:

- `layout_sugiyama_pipeline(..., graphviz_enable_cluster_skeleton=False)` is
  the default and preserves the A10 benchmark-visible path.
- The A12 cluster rank-collapse and skeleton ordering path remains available
  with `graphviz_enable_cluster_skeleton=True`.
- The A12c `_apply_graphviz_leaf_cluster_x_ties()` post-pass was removed. The
  platform service-row tie is now recorded as the exact x residual: the
  inactive skeleton path gives `[3, 5, 4]`; the removed leaf x tie was needed
  to produce `[5, 3, 4]`.
- The A9 mixed label+cluster guard remains unchanged. Label-only and mixed
  rows do not receive cluster metadata, so the cluster x path is unreachable
  for `edge_label_braid`, `small_label_storm`, and other label/mixed rows.

Fallback default eight-row probe:

| Graph | Baseline `d_R` | Fallback default `d_R` | Result |
|---|---:|---:|---|
| `edge_label_braid` | 0.006611 | 0.006611 | same |
| `moe_router_sparse` | 0.361810 | 0.328572 | improved |
| `clustered_longlabel_handoffs` | 0.210412 | 0.210412 | same |
| `interleaved_cluster_crosstalk` | A10 0.595315 | 0.595315 | same |
| `kitchen_sink_hybrid_net` | 0.844934 | 0.844934 | same |
| `kitchen_sink_platform_graph` | 0.301723 | 0.301723 | same |
| `nested_cluster_label_stack` | 0.074680 | 0.074680 | same |
| `small_label_storm` | 0.006045 | 0.006045 | same |

### Gate evidence

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/test_layout/test_dot_rank.py tests/test_layout/test_dot_mincross.py tests/test_layout/test_sugiyama_fidelity.py::test_sugiyama_graphviz_clusters_affect_only_graphviz_mode -x -q
22 passed, 19 warnings in 140.89s
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check . --fix
All checks passed!
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q
74 passed, 3109 deselected, 50 warnings in 157.20s
```

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl mypy --follow-imports=silent dagua/cli.py
pyproject.toml: note: unused section(s): module = ['dagua.layout.multilevel']
Success: no issues found in 1 source file
```

### Bench line

Not run:

```text
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py --engines classic_sugiyama --variants --max-nodes 300 --seeds 100 --seed-start 100 --workers 5 --timeout 3600 --watchdog-timeout 7200 --output-dir /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_clusters
```

Reason: the required rendered x-stage gate did not pass. Per fallback, stages
1-2 were preserved behind an inactive flag and the residual is documented
instead of running the family benchmark.

### Assumptions

- Treated A10 values recorded in this file as the no-regression baseline for
  label-only, mixed, and non-target cluster rows.
- Treated the A12b interleaved value as an opt-in stage value only; fallback
  default returns to A10 behavior for that row.
- Did not enable A12 skeleton ordering by default because the x-stage probe
  still regresses two cluster-only rows.

### Concerns

- The new aux-cluster x helper is not wired into the default path. It is
  useful as a local probe but removable if the next pass takes a different
  route.
- Exact Graphviz parity appears to require solving the remaining skeleton
  range mismatch before replacing A9's output-space cluster pass.

### Knowledge

- Removing `_apply_graphviz_leaf_cluster_x_ties()` fixes the A12c label/mixed
  leakage class, but exposes the platform `[3, 5, 4]` vs `[5, 3, 4]` tie as an
  unresolved x-stage issue.
- The mixed wrapper guard remains the critical safety boundary: mixed
  label+cluster rows must pass neither label nor cluster metadata until the
  combined dot machinery is ported.

## A12e: full-chain x

Date: 2026-07-05
Worktree: `/home/jtaylor/.claude/worktrees/dagua-xstage`
Branch: `r78/xstage`
Commit shas: none. The full-chain aux-x structural parity gate did not pass,
so no production change, commit, or family benchmark was made.

### Reference instrumentation

Built an instrumented Graphviz reference from the pinned 7.0.5 source copied
to `/tmp/gv750-xstage` and built at `/tmp/gv750-xstage-build`. The local
system CMake was too old, so `/home/jtaylor/anaconda3/bin/cmake` 3.27.6 was
used. Pango/Cairo discovery was disabled in the temporary copy to avoid a
static link DSO failure; this means text metrics are useful for structural
direction but not acceptable as a final rendered parity source.

Instrumentation was added only to the temporary `lib/dotgen/position.c`:

- Dump before `rank(g, 2, nsiter2(g))`, immediately after `create_aux_edges(g)`.
- Dump after the final `rank(g, 2, nsiter2(g))`, before `set_xcoords(g)`.
- Each dump includes root rank order, cluster `ln`/`rn` boundary ids, every
  `GD_nlist` node with kind/rank/order/lw/rw, and every aux edge with
  tail/head/minlen/weight.

DOT inputs were generated directly from the three registry definitions using
the production Graphviz adapter DOT serializer, with:

```text
GV_XSTAGE_DUMP=1 /tmp/gv750-xstage-build/cmd/dot/dot -Tjson \
  -Gmaxiter=24 -Granksep=1.0 -Gnodesep=1.0 <graph>.dot
```

Raw dump evidence created during the run:

| Graph | Trace lines | XSTAGE records | Trace sha256 | JSON sha256 |
|---|---:|---:|---|---|
| `clustered_longlabel_handoffs` | 204 | 204 | `8b61144716002fccff1ca24637514d2210518ceb3ac9185dcdd502b501d3d643` | `d1c0931d9132c7f3a045ce416b691e779d3928c2a558510bfd3efd0a3e58a6aa` |
| `interleaved_cluster_crosstalk` | 322 | 322 | `cf01304aeddb5a36cb923e06756bf469feed3232d3c23d02652c9a10ec46c3fd` | `ac67054537a2ed469f2dcab8f55829b26e5d9b3c841845f2584e36677b1b5d63` |
| `kitchen_sink_platform_graph` | 362 | 362 | `c8be7b81aad38c9a730c2131159f468061340d19d691029b1e742e2b784db9ed` | `d4292a24e0c85d200b52dbe54eae794b47e3d7e70a54e54690d13cb93882ad33` |

### Structural diff

Dagua was probed by running the production pipeline with
`graphviz_enable_cluster_skeleton=True`, then dumping
`_build_graphviz_x_aux_edges()` from the same post-ordering state. A
conservative local patch was also tried that passed expanded cluster
membership into `_graphviz_x_coordinate_assignment()` and counted aux nodes
from actual endpoints. It executed on all three graphs, but structural parity
still failed. The patch was reverted because a partial x mechanism is not an
acceptable deliverable for this round.

Initial aux graph summary:

| Graph | Dot nodes | Dagua nodes | Dot edges | Dagua edges | Dot clusters | Dagua boundary nodes |
|---|---:|---:|---:|---:|---:|---:|
| `clustered_longlabel_handoffs` | 35 | 33 | 53 | 49 | 3 | 6 |
| `interleaved_cluster_crosstalk` | 45 | 39 | 104 | 96 | 5 | 10 |
| `kitchen_sink_platform_graph` | 51 | 49 | 113 | 106 | 5 | 10 |

Node-class differences:

| Graph | Dot node classes | Dagua equivalent |
|---|---|---|
| `clustered_longlabel_handoffs` | 10 normal, 3 virtual, 22 slack | 13 expanded, 14 edge slack, 6 boundary |
| `interleaved_cluster_crosstalk` | 12 normal, 2 virtual, 31 slack | 12 expanded, 17 edge slack, 10 boundary |
| `kitchen_sink_platform_graph` | 18 normal, 1 virtual, 32 slack | 19 expanded, 20 edge slack, 10 boundary |

Dominant constraint histogram differences, shown as `(minlen, weight)`:

| Graph | Dot-only examples | Dagua-only examples |
|---|---|---|
| `clustered_longlabel_handoffs` | `(8,0)x4`, `(63,128)x2`, `(104,128)x1`, `(146,0)x2` | `(1,128)x3`, `(69,0)x2`, `(144,0)x2`, `(315,0)x1` |
| `interleaved_cluster_crosstalk` | `(8,0)x2`, `(54,128)x1`, `(55,128)x1`, `(57,128)x1`, `(70,0)x5` | `(1,128)x5`, `(69,0)x5`, `(139,0)x3`, `(156,0)x1` |
| `kitchen_sink_platform_graph` | `(8,0)x8`, `(48,128)x1`, `(56,128)x1`, `(63,128)x1`, `(80,0)x6` | `(1,128)x5`, `(61,0)x3`, `(79,0)x6`, `(189,0)x1` |

Every structural difference found:

- Dagua does not feed cluster membership into the production aux x solve. The
  current default cluster path still solves non-cluster aux x, then applies
  `_apply_graphviz_cluster_x_constraints()` in output space.
- Passing cluster membership into the aux builder is not sufficient: Dagua's
  aux node count currently stops at edge slack nodes, while Graphviz appends
  cluster boundary nodes after slack nodes. A local fix for node counting was
  required before the probe could solve.
- Graphviz has additional virtual nodes in the x-stage dump on all three
  rows. These are not represented in Dagua's skeleton-mode x aux graph with
  the same node classes and edge incidence.
- Graphviz's cluster compaction edges reuse the `ln -> rn` edge created by
  `make_lrvn()` when a cluster label/border width is present, then add weight
  128. Dagua creates a synthetic `(minlen=1, weight=128)` compaction edge
  instead, losing the label-width minlen. This accounts for the `(63,128)`,
  `(104,128)`, `(54,128)`, `(55,128)`, `(57,128)`, `(48,128)`, `(56,128)`,
  and related dot-only constraints.
- Graphviz emits many `(8,0)` margin constraints from `contain_nodes()`,
  `contain_subclust()`, and `separate_subclust()`. Dagua emits fewer of these
  and replaces several with boundary/member constraints derived from expanded
  rank ranges.
- Graphviz keepout uses only normal nodes or unrelated virtual nodes
  (`vnode_not_related_to()`); Dagua's keepout scan treats any non-member rank
  node as an outside node. The dump rows show the resulting edge-count and
  minlen differences even before rendered comparison.
- Dagua minlen values differ by one or more points on label-sized nodes
  because the probe build without Pango is not an exact text-metric authority,
  but the larger class of missing/replaced constraints remains independent of
  that metric drift.

### Attempted port and stop reason

The following local-only changes were tried and reverted:

- Pass `expanded_graph.graphviz_cluster_members` and
  `expanded_graph.graphviz_cluster_parents` into `_graphviz_x_coordinate_assignment()`
  when `graphviz_enable_cluster_skeleton=True`.
- Compute aux node count as one plus the maximum endpoint or initial-rank key,
  rather than `expanded_nodes + expanded_edges`.
- Skip `_apply_graphviz_cluster_x_constraints()` when skeleton aux x is active.
- Set `graphviz_enable_cluster_skeleton=True` in the classic Graphviz wrapper
  for cluster-only rows.

This made the three public pipeline probes execute without crashing, but did
not reach structural parity. Shipping it would be another piecewise x-stage
mechanism, so the production code was restored to the guarded develop state.

### Gate evidence

Passed diagnostic smoke only:

```text
clustered_longlabel_handoffs (10, 2) 0.0 1.3555446863174438
interleaved_cluster_crosstalk (12, 2) -0.08637526631355286 3.311051845550537
kitchen_sink_platform_graph (18, 2) -0.8307634592056274 1.7852576971054077
```

Required gates not run after structural parity failed:

- Rendered 8-row d_R gate.
- 5 plain + 5 label-only + 5 igraph byte-identical gate.
- `pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q`.
- `ruff check . --fix`.
- Family benchmark to
  `/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_xstage`.

### Assumptions

- Treated aux-x structural parity as the hard first gate. Once it failed, the
  default-on wrapper change, commit, and benchmark were not allowed.
- Treated the no-Pango Graphviz build as valid for edge/node inventory and
  pass-order evidence, but not as final text-metric evidence for rendered
  parity.
- Treated the direct graph constructors copied from `dagua/eval/graphs.py` as
  equivalent to the registry rows because the full registry loader was too
  slow for this diagnostic loop.

### Concerns

- The next attempt should instrument a Pango-linked Graphviz build or patch
  the installed 7.0.5 build path so text-size minlens are exact.
- Dagua needs a first-class model of Graphviz x-stage virtual/slack/boundary
  node classes before any default-on skeleton x path can pass the full-chain
  gate.
- The existing `_add_graphviz_cluster_x_aux_edges()` is useful scaffolding but
  is not a faithful `make_lrvn()`/`contain_nodes()`/`keepout_othernodes()`
  port yet.

### Knowledge

- The A9 output-space cluster pass is confirmed as a second x mechanism. It
  must be superseded, not combined, when skeleton-mode aux x becomes faithful.
- Wiring cluster members into the aux solver exposes the missing aux-node-count
  guard immediately; boundary nodes are allocated after edge slack nodes.
- The most resistant difference is Graphviz's cluster label/border compaction
  edge reuse: `make_lrvn()` can create an `ln -> rn` edge with label width and
  `contain_clustnodes()` then increases that same edge's weight by 128.

## A12f: checklist port

Date: 2026-07-05
Worktree: `/home/jtaylor/.claude/worktrees/dagua-xstage`
Branch: `r78/xstage`
Commit shas: none. The four-item checklist did not clear the structural
aux-graph gate, so no production code, wrapper default, commit, or family
benchmark was made.

### Source checkpoints

- `lib/dotgen/position.c:171-188`: `make_aux_edge()` creates fast aux edges
  and rounds minlen at edge creation.
- `lib/dotgen/position.c:191-205`: `allocate_aux_edges()` snapshots
  `ND_save_in` and `ND_save_out` before x-stage aux edges are added.
- `lib/dotgen/position.c:323-347`: `make_edge_pairs()` creates one slack
  node for every saved outgoing edge and seeds its initial rank from the saved
  endpoints.
- `lib/dotgen/position.c:351-361`: `contain_clustnodes()` calls
  `contain_nodes()` and then either increases the existing `ln -> rn` edge
  weight by 128 or creates a fallback `(1,128)` compaction edge.
- `lib/dotgen/position.c:391-422`: `keepout_othernodes()` only keeps out
  normal nodes or virtual nodes accepted by `vnode_not_related_to()`.
- `lib/dotgen/position.c:431-447` and `455-479`: subcluster containment and
  sibling separation use the cluster boundary nodes and the cluster margin.
- `lib/dotgen/position.c:1153-1185`: `make_lrvn()` creates `ln`/`rn`
  slacknodes and, when cluster label/border width exists, creates the reusable
  `ln -> rn` label-width edge.
- `lib/dotgen/position.c:1187-1214`: `contain_nodes()` adds per-rank
  `ln -> first` and `last -> rn` constraints using `ND_lw`, `ND_rw`,
  `G_margin`, and `GD_border`.

### Checklist result

| Item | Status | Evidence |
|---|---|---|
| Virtual/slack/boundary node inventory | Resists port | Dot has extra virtual/slack nodes not representable from Dagua's current expanded graph. A12e dump counts: `clustered_longlabel_handoffs` dot 35 nodes vs Dagua 33, `interleaved_cluster_crosstalk` 45 vs 39, `kitchen_sink_platform_graph` 51 vs 49. Node classes differ as dot normal/virtual/slack, while Dagua has expanded/edge-slack/boundary only. |
| Cluster label compaction minlen reuse | Blocked by item 1 and missing border metrics | Dot reuses a `make_lrvn()` label-width `ln -> rn` edge and adds weight 128. Dagua emits synthetic `(1,128)` compaction edges, producing dot-only `(63,128)`, `(104,128)`, `(54,128)`, `(55,128)`, `(57,128)`, `(48,128)`, and `(56,128)` constraints. |
| Keepout semantics | Blocked by missing virtual lineage | Dot's `vnode_not_related_to()` follows `ED_to_orig` through saved edge lists before accepting a virtual outside node. Dagua cannot make the same normal/unrelated-virtual distinction from node ids and cluster membership alone. |
| Margin constraint counts | Blocked by the same boundary/lineage model | Dot emits `(8,0)` constraints from `contain_nodes()`, `contain_subclust()`, and `separate_subclust()` against exact boundary nodes and rank lists. Dagua emits fewer margin constraints and replaces several with expanded-member constraints. |

### Structural parity table

Initial aux graph summaries from the A12e instrumented dump remain the
terminal boundary for A12f:

| Graph | Dot nodes | Dagua nodes | Dot edges | Dagua edges | Dot-only histogram examples | Dagua-only histogram examples |
|---|---:|---:|---:|---:|---|---|
| `clustered_longlabel_handoffs` | 35 | 33 | 53 | 49 | `(8,0)x4`, `(63,128)x2`, `(104,128)x1`, `(146,0)x2` | `(1,128)x3`, `(69,0)x2`, `(144,0)x2`, `(315,0)x1` |
| `interleaved_cluster_crosstalk` | 45 | 39 | 104 | 96 | `(8,0)x2`, `(54,128)x1`, `(55,128)x1`, `(57,128)x1`, `(70,0)x5` | `(1,128)x5`, `(69,0)x5`, `(139,0)x3`, `(156,0)x1` |
| `kitchen_sink_platform_graph` | 51 | 49 | 113 | 106 | `(8,0)x8`, `(48,128)x1`, `(56,128)x1`, `(63,128)x1`, `(80,0)x6` | `(1,128)x5`, `(61,0)x3`, `(79,0)x6`, `(189,0)x1` |

### Stop reason

The resisting construct is item 1: the Graphviz x-stage inventory cannot be
mirrored by adding the four edge families to the current Dagua aux builder.
Graphviz's aux graph is built after `allocate_aux_edges()` snapshots the
pre-aux edge lists; later `make_edge_pairs()`, `make_lrvn()`,
`contain_nodes()`, `keepout_othernodes()`, `contain_subclust()`, and
`separate_subclust()` all depend on that saved edge/node identity. Dagua's
current skeleton-mode aux graph has original and dummy expanded nodes, edge
slack nodes, and synthetic boundary nodes, but it does not retain Graphviz's
normal/virtual/slack node classes or `ED_to_orig` lineage for virtual nodes.
Without that inventory, item 3 cannot decide Graphviz-equivalent keepout
eligibility, item 2 cannot safely reuse the exact `ln -> rn` label/border edge,
and item 4 cannot match the `(8,0)` margin count.

Partial patches already tried in A12e -- passing cluster members into the aux
solve, counting aux nodes from max endpoint, skipping the A9 output-space pass
when skeleton aux x is active, and enabling the wrapper for cluster-only rows
-- made the three probes execute but did not reach structural parity. Repeating
those patches would ship the piecewise mechanism this terminal round forbids.

### Gate evidence

Structural parity failed before solving. The rendered d_R gate, standard
byte-identity samples, `pytest tests/ -k "sugiyama or mincross or dot_rank"
-x -q`, `ruff check . --fix`, and the <=300 family benchmark were not run,
because the spec says to stop at structural parity when a checklist item
resists.

### Assumptions

- Treated the A12e instrumented dump tables as the current structural evidence
  because production code has not changed since that failed parity probe.
- Treated commits as required only after gate pass. No gate passed, so no
  commit was made.
- Treated the A9 output-space cluster pass as still authoritative for default
  cluster-only rows until a faithful aux-x mechanism exists.

### Concerns

- A future pass needs a first-class Graphviz x-stage model carrying node class,
  saved in/out edge lists, `ED_to_orig` lineage, cluster `GD_border` widths,
  and cluster label metrics into `_build_graphviz_x_aux_edges()`.
- The current helper `_add_graphviz_cluster_x_aux_edges()` remains useful
  scaffolding but is not a faithful port boundary for these four constructs.

### Knowledge

- The four checklist items are not independent once checked against the dump:
  label compaction, keepout, and margin counts all depend on the missing
  x-stage inventory model.
- The correct terminal disposition for the 20 cluster rows is to preserve the
  existing guarded path and not default-on skeleton aux x until aux structural
  parity is demonstrated.

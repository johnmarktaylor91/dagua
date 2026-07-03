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

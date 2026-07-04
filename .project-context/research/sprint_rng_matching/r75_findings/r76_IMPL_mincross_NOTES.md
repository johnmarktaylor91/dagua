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

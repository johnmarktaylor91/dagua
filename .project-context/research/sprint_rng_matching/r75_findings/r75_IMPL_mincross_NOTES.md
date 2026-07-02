# r75 Graphviz Mincross Phase 1 Notes

Date: 2026-07-02
Worktree: `/home/jtaylor/.claude/worktrees/dagua-mincross`
Branch: `r75/mincross`

## Ported

- Added Graphviz-style mincross pass structure in `dagua/layout/ops/_dot_mincross.py`.
  - Passes 0 and 1 use `min(4, MaxIter)` iterations.
  - Pass 2 uses `MaxIter`.
  - `MinQuit = 8` and `Convergence = .995` match Graphviz 7.0.5
    `lib/dotgen/mincross.c:690-748` and defaults at `:1944-1952`.
- Added `build_ranks`-style pass seedings.
  - Pass 0 starts from nodes with no incoming adjacent-rank edge and traverses outgoing edges.
  - Pass 1 starts from nodes with no outgoing adjacent-rank edge and traverses incoming edges.
  - This follows Graphviz 7.0.5 `lib/dotgen/mincross.c:1212-1286`.
- Added weighted crossing support to `_dot_mincross.py`.
  - Crossing totals and transpose accept/reject now use per-edge penalties.
  - Global crossing counting now uses a Fenwick inversion counter instead of quadratic edge-pair
    scans.
- Kept the changes gated to graphviz mincross mode.
  - `dot` / `graphviz_dot` alias behavior keeps rank-order seed scans.
  - `graphviz` mode uses reverse expanded-node creation order as a conservative approximation of
    Graphviz's prepended `GD_nlist` in `fastgr.c:205-216` and virtual-node creation at
    `fastgr.c:241-264`.

## Source Finding

The task wording requested omega-weighted crossing counts via `virtual_weight()/ED_xpenalty`.
Pinned Graphviz 7.0.5 does not apply `virtual_weight()` to `ED_xpenalty`:

- `dotinit.c:55-65` initializes `ED_xpenalty` separately from `ED_weight`.
- `class2.c:84-95` calls `virtual_weight(e)` for virtual chains.
- `mincross.c:1858-1894` implements `virtual_weight()` by multiplying `ED_weight(e)`, not
  `ED_xpenalty(e)`.
- `mincross.c:584-615` and `:1640-1690` use `ED_xpenalty` for crossing counts.

I therefore left mincross penalties source-faithful to `ED_xpenalty = 1` for this phase, while the
existing x-coordinate Stage A path still uses the omega endpoint table for coordinate constraints.

## Verification Ladder

Commands used `PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl`.

### a. Stage-A No Regression

Command:

```bash
python scripts/run_benchmark.py --workers 1 --timeout 120 --seeds 1 --seed-start 42 \
  --graphs binary_tree,bipartite_4_3_4,org_chart_1_5_4_8 \
  --engines classic_sugiyama_graphviz_fidelity --variants \
  --output-dir /tmp/r75_mincross_ladder_a2
```

Status: 3 total, 3 OK.

Stress after, computed with `normalized_stress(..., fit_scale=True)`:

| Graph | Before D | Reference R | After D |
|---|---:|---:|---:|
| `binary_tree` | 0.19700074743578236 | 0.150893955650856 | 0.1634169894465131 |
| `bipartite_4_3_4` | 0.3242658731571135 | 0.178773684069281 | 0.1574668225741319 |
| `org_chart_1_5_4_8` | 0.38856547285360243 | 0.2261922461523067 | 0.19286388297906776 |

Result: no regression versus Stage A recorded values.

### b. Crossing Targets

Final target run:

```bash
python scripts/run_benchmark.py --workers 1 --timeout 240 --seeds 1 --seed-start 42 \
  --graphs dense_pair_50,weighted_karate_34,hub_skip_superfan,heavy_tail_weights_50 \
  --engines classic_sugiyama_graphviz_fidelity --variants \
  --output-dir /tmp/r75_mincross_ladder_b5
```

Status: 4 total, 4 OK.

Crossings:

| Graph | Before D | Reference R | After D | Result |
|---|---:|---:|---:|---|
| `dense_pair_50` | 391 | 331 | 400 | failed, moved away |
| `weighted_karate_34` | 111 | 108 | 76 | failed by absolute distance, overshot |
| `hub_skip_superfan` | 3 | 2 | 5 | failed, moved away |
| `heavy_tail_weights_50` | 70 | 67 | 90 | failed, moved away |

### c. Scale Spot-Check

Command:

```bash
python scripts/run_benchmark.py --workers 1 --timeout 300 --seeds 1 --seed-start 42 \
  --graphs ba_500 --engines classic_sugiyama_graphviz_fidelity --variants \
  --output-dir /tmp/r75_mincross_ladder_c2
```

Result: failed. Worker layout timeout exceeded.

Blocking rule: repeated Graphviz transpose/local-cross passes remain too slow in the current Python
port on the expanded `ba_500` graph. Optimizing global `ncross` with a Fenwick counter was
insufficient; the remaining hotspot is the local transpose loop corresponding to Graphviz
`mincross.c:584-615` and `:690-748`.

### d. Regression Gate

Completed:

```bash
ruff check . --fix
mypy --follow-imports=silent dagua/cli.py
pytest tests/test_layout/test_dot_mincross.py -q
pytest tests/ -k sugiyama -x -q
```

Results:

- `ruff check . --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed; existing pyproject note about
  `dagua.layout.multilevel`.
- `pytest tests/test_layout/test_dot_mincross.py -q`: 6 passed.
- `pytest tests/ -k sugiyama -x -q`: 45 passed, 3100 deselected.

Not completed because the implementation failed ladder b/c:

- Byte-identical default/tight 5-seed tensor comparison.
- Full Tier 1 layout/graph pytest gate.
- Final non-slow test suite.

## Deferred / Blocking Work

- Full class2 multi-edge merge remains deferred by scope. Graphviz source merges `ED_xpenalty` in
  `class2.c:140-154` and `fastgr.c:310-365`; the current expanded-edge representation does not
  collapse representative chains.
- Flat and cluster constraints remain deferred by scope. Source paths include `flat_breakcycles`,
  `flat_reorder`, and left-to-right checks around `mincross.c:1212-1480`.
- The local transpose implementation needs a faster source-faithful equivalent before `ba_500`
  can satisfy the 300s scale gate.

## Commit

No commit made because the completion contract was not met.

---

## Attempt 2

Date: 2026-07-02

### Changes Made

- Kept attempt 1's Graphviz mincross port and added an incremental transpose path in
  `dagua/layout/ops/_dot_mincross.py`.
  - `_transpose()` now keeps per-rank order maps and updates only the exchanged pair, avoiding
    the previous global `_node_order_map()` rebuild in every adjacent-swap test.
  - `_reorder_rank()` now accepts `hasfixed` and preserves Graphviz's scan-window behavior from
    `mincross.c:1553-1596` when fixed sentinel nodes are present.
- Added iterative DFS traversals in `dagua/layout/ops/pipelines/dot_rank.py`.
  - `_dfs_range_init()` and `_dfs_cutval()` no longer recurse down the network-simplex tree.
  - This removed the immediate `RecursionError` seen on `ba_500`, but did not make the full
    graphviz x-coordinate stage finish inside the target window.

### Ordering-Stage Discriminator

Built DOT input using `dagua/eval/competitors/graphviz_competitor.py:_graph_to_dot()` and compared
against `dot -v` from installed Graphviz 7.0.5.

Small ladder:

| Graph | `dot -v` mincross | Port count | Result |
|---|---:|---:|---|
| `binary_tree` | 0 | 0 | match |
| `bipartite_4_3_4` | 36 | 36 | match |
| `hub_skip_superfan` | 2 | 2 | match |
| `weighted_karate_34` | 63 | 50 | mismatch |

This satisfies the ">=3 of 4 small graphs" discriminator.

Additional rendered-ladder ordering counts:

| Graph | `dot -v` mincross | Port count | Result |
|---|---:|---:|---|
| `dense_pair_50` | 271 | 326 | mismatch |
| `heavy_tail_weights_50` | 50 | 59 | mismatch |

These two show the rendered crossing failures are already present before coordinate assignment.

### Rendered Crossing Ladder

Command:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py --workers 1 --timeout 240 \
  --seeds 1 --seed-start 42 \
  --graphs dense_pair_50,weighted_karate_34,hub_skip_superfan,heavy_tail_weights_50 \
  --engines classic_sugiyama_graphviz_fidelity --variants \
  --output-dir /tmp/r75_mincross_attempt2_render
```

Status: 4 total, 4 OK.

Crossings computed from saved position tensors with `dagua.metrics.count_crossings`:

| Graph | Before D | Reference R | Attempt 2 | Result |
|---|---:|---:|---:|---|
| `dense_pair_50` | 391 | 331 | 400 | failed, moved away |
| `weighted_karate_34` | 111 | 108 | 76 | failed by absolute distance, overshot |
| `hub_skip_superfan` | 3 | 2 | 5 | failed, moved away |
| `heavy_tail_weights_50` | 70 | 67 | 90 | failed, moved away |

Alternate order/x-coordinate probes:

- Node-list seed variants (`None`, reverse, ascending, rank-reverse) did not improve
  `dense_pair_50`; all Graphviz-x variants stayed at 400 crossings.
- `heavy_tail_weights_50` bottoms out at 70 with BK x-coordinates and current rules, tying the
  recorded baseline but not moving toward the 67 reference.
- `graphviz` x-coordinate mode worsens `hub_skip_superfan` from 2 to 5 and
  `heavy_tail_weights_50` from 70 to 90 compared with `dot`/BK x-coordinates.

### ba_500 Gate

Command:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py --workers 1 --timeout 300 \
  --seeds 1 --seed-start 42 --graphs ba_500 \
  --engines classic_sugiyama_graphviz_fidelity --variants \
  --output-dir /tmp/r75_mincross_attempt2_ba500
```

Result before the dot-rank DFS patch: 31.57s error, `maximum recursion depth exceeded`.

Direct repro showed the first recursion was in:

- `dagua/layout/ops/pipelines/dot_rank.py:_dfs_range_visit()` via
  `_graphviz_x_coordinate_assignment()` and `graphviz_network_simplex_assignment()`.

After converting `_dfs_range_init()` and `_dfs_cutval()` to iterative traversals, the direct
`ba_500` graphviz-fidelity pipeline no longer failed at those recursion sites, but it exceeded
270 seconds before completion and was killed. The remaining scale blocker is the graphviz
x-coordinate network-simplex stage, not the mincross transpose loop.

### Source Findings / Blocking Rules

- Graphviz default DOT does **not** run the final mincross balance phase for these benchmark DOTs.
  `dotinit.c:304` calls `dot_mincross(g, (asp != NULL))`, and the adapter does not emit an
  `aspect` attribute; `mincross.c:870-873` therefore does not explain the default mismatch.
- The remaining ordering mismatch is most consistent with still-inexact Graphviz internal
  construction order and representative-chain merging:
  - `fastgr.c:205-216` prepends every `fast_node()` into `GD_nlist`.
  - `fastgr.c:241-264` creates virtual nodes through `fast_node()`.
  - `mincross.c:1356-1414` seeds `build_ranks()` by iterating the actual `GD_nlist`, not a
    derived numeric order.
  - `class2.c:137-155` and `fastgr.c:326-349` merge representative chains and accumulate
    `ED_xpenalty`; the current expanded-edge representation still keeps chains separate with
    unit penalties.
- The source-fidelity correction from attempt 1 still stands: Graphviz 7.0.5 crossing counts use
  `ED_xpenalty`, not `virtual_weight()`/omega.

### Verification

Commands run:

```bash
PYTHONPATH=$PWD pytest tests/test_layout/test_dot_mincross.py tests/test_layout/test_dot_rank.py -q
PYTHONPATH=$PWD ruff check . --fix
```

Results:

- `pytest tests/test_layout/test_dot_mincross.py tests/test_layout/test_dot_rank.py -q`:
  12 passed, 3 warnings.
- `ruff check . --fix`: passed.

Not run because the ladder failed before the commit gate:

- Tier-1 full gate (`mypy --follow-imports=silent dagua/cli.py`,
  `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`).
- Final non-slow suite.
- Byte-identical default/tight regression check.

### Conclusion

Attempt 2 still fails the required ladder, so no commit was made. The most concrete next-sprint
work is to port the real Graphviz internal node/edge install order and representative-chain merge
semantics before iterating further on rendered crossings, then separately address the
`graphviz` x-coordinate network-simplex scale path for `ba_500`.

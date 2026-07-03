# r76 XNS Performance Notes

Date: 2026-07-03
Worktree: `/home/jtaylor/.claude/worktrees/dagua-xns-perf`
Branch: `r76/xns-perf`

## Scope

Task target was the Graphviz x-coordinate network-simplex path added in r75 Stage A. I modified
only `dagua/layout/ops/pipelines/dot_rank.py`; I did not modify BK paths, mincross, or eval code.

## Profile Before

Command:

```bash
PYTHONPATH=$PWD python -m cProfile -o /tmp/r76_dense_pair50_before.prof \
  scripts/run_benchmark.py --workers 1 --timeout 300 --seeds 1 --seed-start 100 \
  --graphs dense_pair_50 --engines classic_sugiyama_graphviz_fidelity --variants \
  --output-dir /tmp/r76_xns_profile_before
```

Benchmark result:

```text
dense_pair_50 x classic_sugiyama_graphviz_fidelity seed=100: ok (37.6s)
```

Relevant `dot_rank.py` profile:

```text
graphviz_network_simplex_assignment  6.669s cumulative
_run_network_simplex                 6.641s
_update                              6.023s over 548 pivots
_dfs_cutval                          5.042s over 550 full recomputes
_x_cutval                            4.158s
_x_val                               2.894s
_dfs_range_init                      0.948s
```

This confirmed the expected r75 hotspot: every pivot recomputed all tree cut values, while
Graphviz 7.0.5 `lib/common/ns.c:update()` updates cut values incrementally along the two old-tree
paths from the entering edge endpoints to the LCA, sets `ED_cutvalue(f) = -cutvalue`, invalidates
only those paths, exchanges tree edges, and calls `dfs_range(lca, ND_par(lca), lca_low)`.

## What Changed

- Replaced recursive tree traversals in `_tree_adjust()`, `_dfs_range_init()`/`_dfs_range()`,
  `_dfs_cutval()`, and `_rerank()` with iterative equivalents.
  - This recovers the r75 mincross-attempt idea for `_dfs_range_init()` / `_dfs_cutval()`.
  - Traversal order remains outgoing tree edges before incoming tree edges.
- Added Graphviz-style incremental cut-value updates in `_update()`.
  - `_tree_update()` ports `ns.c:treeupdate()`.
  - `_invalidate_path()` ports `ns.c:invalidate_path()`.
  - `_dfs_range(..., reuse_clean=True)` ports the incremental `ns.c:dfs_range()` reuse check.
- Added zero-slack pruning to the entering-edge subtree scan, matching the Graphviz `Slack > 0`
  second-loop guard in `dfs_enter_outedge()` / `dfs_enter_inedge()`.

## Verification

### Bit Identity Gate

Pre-change tensors were generated from a detached worktree at `/tmp/r76_xns_pre_worktree` using
`HEAD`. Patched tensors were generated from this worktree.

Command shape:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --workers 1 --timeout 300 --watchdog-timeout 600 --seeds 2 --seed-start 100 \
  --graphs binary_tree,bipartite_4_3_4,org_chart_1_5_4_8,center_port_backedge_hub \
  --engines classic_sugiyama_graphviz_fidelity --variants \
  --output-dir /tmp/r76_xns_bit_after
```

Result:

```text
8/8 tensors torch.equal pre/post:
binary_tree seeds 100,101
bipartite_4_3_4 seeds 100,101
org_chart_1_5_4_8 seeds 100,101
center_port_backedge_hub seeds 100,101
```

### Tests

```text
PYTHONPATH=$PWD ruff check . --fix
All checks passed!

PYTHONPATH=$PWD pytest tests/ -k "sugiyama or dot_rank" -x -q
49 passed, 3101 deselected, 34 warnings in 13.21s

PYTHONPATH=$PWD mypy --follow-imports=silent dagua/cli.py
pyproject.toml: note: unused section(s): module = ['dagua.layout.multilevel']
Success: no issues found in 1 source file
```

### Timing Probes

Plain benchmark runs:

| Graph | Seed | Result |
|---|---:|---:|
| `dense_pair_50` pre-change | 100 | ok, 14.27s |
| `dense_pair_50` patched | 100 | ok, 17.01s |
| `dense_pair_50` patched | 42 | ok, 12.42s |
| `sbm_5x50` patched | 100 | error, worker layout timeout at 299.99s |
| `ba_500` patched | 100 | error, worker layout timeout at 599.99s |

The requested `dense_pair_50 <=10s`, `sbm_5x50 <=30s`, and `ba_500 <=240s` targets were not met.

## Remaining Hotspot

The remaining large-graph blocker is no longer demonstrated in x-coordinate network simplex. A
bounded direct `sbm_5x50` layout with `faulthandler.dump_traceback_later(60)` showed the active
stack in mincross:

```text
dagua/layout/ops/_dot_mincross.py:166 in _node_order_map
dagua/layout/ops/_dot_mincross.py:410 in _in_cross
dagua/layout/ops/_dot_mincross.py:375 in _transpose
dagua/layout/ops/_dot_mincross.py:245 in _mincross_step
dagua/layout/ops/_dot_mincross.py:68 in graphviz_mincross
dagua/layout/ops/sugiyama.py:1280 in _barycenter_ordering
```

This matches the deferred r75 mincross attempt note: `_transpose()` rebuilds order maps during
local crossing checks. The present task's action-safety block said "Do not change BK paths,
mincross, eval code", so I did not port the surviving mincross incremental-order-map patch.

## 9-Combo Probe

The requested 9 graph x 5 seed probe was not run after `sbm_5x50` and `ba_500` single-seed probes
timed out and the stack dump identified mincross as the active blocker outside this task's allowed
write scope. Running all 45 at that point would have produced repeated mincross timeouts rather
than more xNS evidence.

## Assumptions and Choices

- Treated "Do not change BK paths, mincross, eval code" as a hard scope boundary.
- Kept current Python pivot/tie behavior intact; randomized DAG comparisons against the pre-change
  `dot_rank.py` matched for sampled small DAG constraints, and the required 8 benchmark tensors
  were bit-identical.
- Committed the source-faithful xNS internals even though end-to-end large graph gates remain
  blocked elsewhere.

## Commit

```text
perf(sugiyama): speed up graphviz x-network-simplex updates
```

## Knowledge

- The pre-change dense_pair_50 cProfile showed full `_dfs_cutval()` recomputation dominating xNS.
- On the current branch, end-to-end large graph graphviz-fidelity runtime is dominated before
  x-coordinate assignment by `_dot_mincross.py:_transpose()` / `_node_order_map()`.
- `PYTHONPATH=$PWD` is required for benchmark runs in this worktree.

# r75 Stage A Sugiyama Graphviz XNS Implementation Notes

Date: 2026-07-01
Worktree: `/home/jtaylor/.claude/worktrees/dagua-sugiyama-xns`
Branch: `r75/sugiyama-xns`

## Changes

- Added reusable Graphviz network-simplex assignment support in
  `dagua/layout/ops/pipelines/dot_rank.py`.
  - Preserved existing `graphviz_rank_assignment()` behavior with top-bottom balance.
  - Added `graphviz_network_simplex_assignment(..., balance_mode="none"|"tb"|"lr")`.
  - Added Graphviz 7.0.5 LR balance semantics from `lib/common/ns.c:696-716`.
  - Added optional initial ranks so x-coordinate aux graphs can start from `position.c`'s seeded
    `ND_rank` values.
- Added Graphviz Stage A x-coordinate assignment in `dagua/layout/ops/sugiyama.py`.
  - Builds same-rank LR constraints from ordered expanded ranks:
    `tail=rank[i][j]`, `head=rank[i][j+1]`,
    `minlen=round(rw(left)+lw(right)+nodesep)`, `weight=0`.
  - Builds one slack node per expanded edge with portless constraints:
    `slack->tail minlen=1`, `slack->head minlen=1`, weighted by Graphviz omega endpoint table.
  - Seeds original and slack aux ranks per Graphviz 7.0.5 `position.c:238-267` and `327-343`.
  - Uses virtual-node `nodesep / 2` left/right widths matching `class2.c:35-50`.
  - Normalizes returned x coordinates into `rank_sep` units after solving, preserving affine x shape.
- Wired the new x assignment only for `fidelity_mode == "graphviz"` in
  `dagua/layout/ops/pipelines/sugiyama.py`.
  - Default, igraph, `dot`, and `graphviz_dot` alias paths keep Brandes-Kopf x assignment.
- Updated Sugiyama tests for the new graphviz x behavior.
- Adjusted `layout_sugiyama_pipeline()` direct-call default spacing to classic-compatible `1.0`,
  matching existing pipeline fidelity tests. Benchmark variants already pass explicit spacing.

## Ladder Results

All benchmark commands below required `PYTHONPATH=$PWD`; without it, `scripts/run_benchmark.py`
imports the installed editable main checkout instead of this worktree.

### a. `binary_tree`

Command:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --workers 1 --timeout 60 --seeds 1 --seed-start 42 \
  --graphs binary_tree \
  --engines classic_sugiyama_graphviz_fidelity \
  --variants --output-dir /tmp/r75_xns_probe_after4
```

Reference path requested by the task did not exist:

```text
/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_seeded_refs/positions/
binary_tree__graphviz_dot__for__classic_sugiyama_graphviz_fidelity.pt
```

Used the available 100-seed escalation-final Graphviz reference:

```text
/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_escalation_final/positions/
binary_tree__graphviz_dot__for__classic_sugiyama_graphviz_fidelity.pt
```

After x-only affine alignment:

- Dagua x: `[0.5, 0.0, 1.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0]`
- Relative x residual: `1.5297683354005512e-16`
- RMS x residual: `2.0990823539818746e-14`
- Alignment scale: `125.99999999999997`

This satisfies the `< 1e-6` frame-level x target.

### b. Stress Gap Shrink

Command:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --workers 1 --timeout 90 --seeds 1 --seed-start 42 \
  --graphs bipartite_4_3_4,org_chart_1_5_4_8,center_port_backedge_hub \
  --engines classic_sugiyama_graphviz_fidelity \
  --variants --output-dir /tmp/r75_xns_stepb2
```

Stress was computed with `dagua.eval.equivalence_metrics.normalized_stress(..., fit_scale=True)`.
Before values are from `r75_targets_sugiyama.json`.

| Graph | Before D | Before R | Before gap | After D | After R | After gap | Shrink |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bipartite_4_3_4` | 0.3242658731571135 | 0.178773684069281 | 0.1454921890878325 | 0.1574668225741319 | 0.178773684069281 | 0.021306861495149126 | 0.12418532759268339 |
| `org_chart_1_5_4_8` | 0.38856547285360243 | 0.2261922461523067 | 0.16237322670129573 | 0.19286388297906776 | 0.2261922461523067 | 0.03332836317323895 | 0.12904486352805677 |
| `center_port_backedge_hub` | 0.29048348667767715 | 0.15872425127893036 | 0.1317592353987468 | 0.16676972792421962 | 0.15872425127893042 | 0.008045476645289207 | 0.12371375875345758 |

Result: material shrink on 3/3 requested graphs.

### c. Regression Gate

I compared current worktree vs installed pre-change baseline for:

```text
5 seeds x {classic_sugiyama_default, classic_sugiyama_tight} on binary_tree
```

Commands:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --workers 1 --timeout 90 --seeds 5 --seed-start 42 \
  --graphs binary_tree \
  --engines classic_sugiyama_default,classic_sugiyama_tight \
  --variants --output-dir /tmp/r75_reg_current

MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --workers 1 --timeout 90 --seeds 5 --seed-start 42 \
  --graphs binary_tree \
  --engines classic_sugiyama_default,classic_sugiyama_tight \
  --variants --output-dir /tmp/r75_reg_baseline
```

Tensor comparison:

- Checked: `10` position tensors.
- `torch.equal(current, baseline)`: all passed.
- Note: raw `.pt` file bytes differed because `torch.save` serialization metadata is not stable
  between runs; tensor values were byte-identical.

`pytest tests/ -k sugiyama -x -q`:

- Passed: `44 passed, 3089 deselected`.

## Quality Gates

- `ruff check . --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
  - Output included existing note: `pyproject.toml: note: unused section(s): module = ['dagua.layout.multilevel']`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: passed.
  - `453 passed, 153 warnings in 2629.96s`.
- Final suite command:

```bash
PYTHONPATH=$PWD pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
```

Failed on an unrelated existing bench-large checkpoint test:

```text
FAILED tests/test_bench_large.py::test_hierarchy_checkpoint_rejects_incomplete_manifest
```

Isolated rerun also fails. Root cause: `scripts/bench_large._load_hierarchy_checkpoint()` accepts
a hierarchy manifest saved with `"complete": false`. This is outside the Sugiyama Stage A scope, so
I did not modify `scripts/bench_large.py`.

## Residual Rules Not Yet Ported

Planned later stages:

- Stage B: flat-edge constraints from `position.c:269-321`.
- Stage C: edge-label constraints and label virtual-node width handling beyond plain virtual nodes.
- Stage D: cluster containment/separation constraints from `position.c:354-499`.
- Ports remain omitted in Stage A: `port_dx=0`.

## Assumptions and Choices

- Treated the main-repo research files as read-only source because the requested `.project-context`
  files were not present in this worktree.
- Used the available `benchmark_100seed_escalation_final` Graphviz reference for `binary_tree`
  because the exact `benchmark_100seed_seeded_refs` file requested by the task was absent.
- Added a two-unit internal x simplex resolution to avoid Dagua benchmark half-step parity loss
  when label-size-derived minlens are odd. Returned coordinates are divided back down.
- Normalized graphviz x output by median same-rank separation into `rank_sep` units. This preserves
  affine x shape and makes returned Dagua coordinates live in the same unit family as y ranks.

## Commits

No commits made. The worktree instructions say the orchestrator handles git operations.

## Knowledge

- `scripts/run_benchmark.py` must be run with `PYTHONPATH=$PWD` in this worktree, otherwise it
  imports the installed editable checkout instead of these local changes.
- Graphviz dot x assignment depends on initial aux `ND_rank` seeding; starting all simplex ranks at
  zero produces a different optimal tie even with the same aux edges.

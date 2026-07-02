# r75 LP Objective Implementation Notes

Date: 2026-07-01
Worktree: `/home/jtaylor/.claude/worktrees/dagua-lp-obj`
Branch: `r75/lp-objective`
Commit SHA: final self-containing commit; see `git rev-parse HEAD` after commit.

## Change

- Ported the igraph 1.0.0 Sugiyama GLPK objective quirk in
  `dagua/layout/ops/sugiyama.py`.
  - Both `in_strengths` and `out_strengths` are populated from incoming
    incidences.
  - Eades feedback edges are still subtracted as igraph does:
    `outdegs[from] -= weight`, `indegs[to] -= weight`.
  - The LP objective remains `outdegs - indegs`.
- Updated the stale objective assumption test in
  `tests/test_layout/test_sugiyama_fidelity.py`.
- Added a runtime-verified `two_hubs_bridge` regression against installed
  `python-igraph` 1.0.0.

## Assumptions

- The requested probe/result markdown files were not present in this worktree.
  `/tmp/r75_probe.py` was present and supplied the `two_hubs_bridge` edge list.
- The exact deterministic `igraph_sugiyama` files were absent from
  `/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_seeded_refs`.
  I used the requested fallback directory:
  `/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_escalation_final`.

## Benchmark Comparison

Commands:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --workers 1 --timeout 120 --seeds 5 --seed-start 42 \
  --graphs binary_tree,densenet_block,real_karate_34,multiscale_skip_cascade \
  --engines classic_sugiyama_default,classic_sugiyama_tight,classic_sugiyama_wide \
  --variants --output-dir /tmp/r75_lp_probe/before

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --workers 1 --timeout 120 --seeds 5 --seed-start 42 \
  --graphs binary_tree,densenet_block,real_karate_34,multiscale_skip_cascade \
  --engines classic_sugiyama_default,classic_sugiyama_tight,classic_sugiyama_wide \
  --variants --output-dir /tmp/r75_lp_probe/after
```

Results:

- Before: `60 total, 60 ok, 0 skipped, 0 errors, 0 timeouts`.
- After: `60 total, 60 ok, 0 skipped, 0 errors, 0 timeouts`.
- Tensor comparison: `60` unchanged, `0` changed.
- Changed graph/variant pairs: none.
- Per-pair before/after Procrustes/stress numbers: not applicable because no
  benchmark-path tensors changed.
- Pairs that moved away from igraph reference: none.

## Graphviz-Fidelity Guard

Command pair:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --workers 1 --timeout 120 --seeds 2 --seed-start 42 \
  --graphs binary_tree \
  --engines classic_sugiyama_graphviz_fidelity \
  --variants --output-dir /tmp/r75_lp_probe/graphviz_before

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --workers 1 --timeout 120 --seeds 2 --seed-start 42 \
  --graphs binary_tree \
  --engines classic_sugiyama_graphviz_fidelity \
  --variants --output-dir /tmp/r75_lp_probe/graphviz_after
```

Results:

| Tensor | Equal | Max abs delta |
|---|---:|---:|
| `binary_tree__classic_sugiyama_graphviz_fidelity__seed42.pt` | yes | `0.0` |
| `binary_tree__classic_sugiyama_graphviz_fidelity__seed43.pt` | yes | `0.0` |

## Tests

- `pytest tests/ -k sugiyama -x -q`: passed,
  `45 passed, 3098 deselected, 34 warnings in 29.63s`.
- `ruff format .`: passed, `557 files left unchanged`.
- `ruff check . --fix`: passed, `Found 1 error (1 fixed, 0 remaining)`.
- `mypy --follow-imports=silent dagua/cli.py`: passed,
  `Success: no issues found in 1 source file`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: failed on
  `tests/test_layout/test_engine.py::test_classify_early_exit`.
  - Full run failure: `assert 0.4440498799085617 < 0.1`.
  - Isolated rerun failure: `assert 0.5039498340338469 < 0.1`.
  - Root cause: reproducible timing-budget failure in a dense classification
    early-exit test, outside the Sugiyama LP objective path.
- Final non-slow suite was not run after the repeated targeted-gate failure,
  following the project instruction to stop after the same failure repeats.

## Concerns

- The requested benchmark set did not include a graph/variant pair where the
  old and corrected objectives produce different saved positions. The focused
  unit regression covers the proven distinguishing DAG against installed
  igraph at runtime.
- `test_classify_early_exit` has a strict wall-clock threshold and failed
  reproducibly on this machine after this change; no related files were
  modified.

## Knowledge

- `scripts/run_benchmark.py` must be run with `PYTHONPATH=$PWD` in this
  worktree, otherwise it may import the editable main checkout.
- Installed `python-igraph` is version `1.0.0`.
- The igraph 1.0.0 LP objective quirk produces all-zero coefficients on DAGs
  with no feedback edges because both degree vectors are incoming-incidence
  strengths before subtraction.

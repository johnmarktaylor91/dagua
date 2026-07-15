# Spectral reference-oracle regression fix

## Outcome

The regression introduced by `bbc09ba` was in the NetworkX reference adapter,
not in Dagua's `classic_spectral` implementation. The adapter imported private
helpers from `dagua.layout.ops.embed` and substituted Dagua's disconnected
spectral embedding for NetworkX's output. That circular oracle path has been
removed.

`nx_spectral` now reaches `networkx.spectral_layout` through the normal
NetworkX adapter path for every graph, including disconnected graphs. The
`nx_spectral_random_walk` variant retains an independent Laplacian/eigensolver
implementation based only on NetworkX, NumPy, and SciPy; its pre-`bbc09ba`
solver behavior was restored and it no longer imports Dagua layout helpers.

## Independence and implementation verification

- `dagua/eval/competitors/networkx_competitor.py` has no runtime import from
  `dagua.layout`.
- `dagua/layout/ops/pipelines/spectral.py` has no runtime import from
  `dagua.eval.competitors`.
- `dagua/layout/ops/embed.py` has no runtime import from
  `dagua.eval.competitors`.
- The Dagua pipeline files were unchanged by this fix. Their working-tree
  hashes equal their `HEAD` hashes:
  - `spectral.py`: `6fc488aa27bd0d37a9293f917de041acc0d1ab6351fcd4c7b00bd710d311e0c6`
  - `embed.py`: `2ba77e75e9fa36ad2091a3a46718e2449844aa6d2985dde161fc0bdea6d08d11`

The Bug A/B changes in the Dagua pipeline therefore remain intact.

## Benchmark and rescore

Fresh self-contained deterministic cache:

- `eval_output/benchmark_r79_spectral_oraclefix`
- 105 graphs x 8 paired engines = 840 runs
- Result: 840 ok, 0 skipped, 0 errors, 0 timeouts

The requested artifact was overwritten:

- `eval_output/fidelity_definitive/per_combo_r79_spectral.jsonl`
- 420 rows, 0 scoring errors

Before/after verdict totals:

| Verdict | Before | After | Delta |
| --- | ---: | ---: | ---: |
| `INVARIANCE_EQUIVALENT` | 414 | 397 | -17 |
| `QUALITY_EQUIVALENT` | 6 | 15 | +9 |
| `DIFFERENT` | 0 | 8 | +8 |

The transition matrix is:

| Before -> after | Rows |
| --- | ---: |
| invariant -> invariant | 391 |
| invariant -> quality-equivalent | 15 |
| invariant -> different | 8 |
| quality-equivalent -> invariant | 6 |

The six false degeneracy-floor rows collapsed as expected:

| Graph | Variants | Before | After toolkit distance | After |
| --- | --- | ---: | ---: | --- |
| `multi_component_80` | default, nx_fidelity, unnormalized | 0.986672 | 2.768e-16 | invariant |
| `random_dag_200` | default, nx_fidelity, unnormalized | 1.159343 | 1.597e-16 | invariant |

Collateral rows now show the real NetworkX-reference verdicts:

| Graph | Unnormalized reference pairings | Random-walk pairing |
| --- | --- | --- |
| `er_2000` | quality-equivalent, toolkit distance 1.0 | quality-equivalent, 0.01811 |
| `er_500` | 3 different rows, toolkit distance 1.3837-1.4130 | quality-equivalent, 0.14998 |
| `dependency_500` | 3 different rows, toolkit distance 0.70711 | quality-equivalent, 0.00614 |

The other two corrected `DIFFERENT` rows are the independent random-walk
reference pairings for `grid_50x50` (0.04413) and `small_world_500` (0.12351).
The remaining random-walk movements are 11 additional quality-equivalent rows;
these had previously shared Dagua's private sparse-eigensolver helper and were
therefore circular passes.

## Commands

Benchmark:

```bash
python scripts/run_benchmark.py --variants \
  --engines <four classic_spectral variants and four paired references> \
  --graphs <the 105-graph spectral corpus> \
  --workers 4 --timeout 300 --watchdog-timeout 900 \
  --output-dir eval_output/benchmark_r79_spectral_oraclefix
```

Rescore:

```bash
python scripts/definitive_fidelity_analysis.py --mode deterministic \
  --refresh-dir eval_output/benchmark_r79_spectral_oraclefix \
  --combos-file /tmp/dagua_r79_combos/combos_r79_spectral.txt \
  --workers 6 --overwrite \
  --output eval_output/fidelity_definitive/per_combo_r79_spectral.jsonl
```

## Tests

- Targeted lint: `ruff check dagua/eval/competitors/networkx_competitor.py tests/test_layout/test_spectral_fidelity.py --fix` -> passed.
- Targeted regression tests: `pytest tests/test_layout/test_spectral_fidelity.py -q` -> 17 passed, 3 warnings.
- Strict typing: `mypy --follow-imports=silent dagua/cli.py` -> success, no issues.
- Repository lint: `ruff check .` -> 21 unrelated errors in untracked
  `.project-context/research` and `.research` scripts. The command was kept
  read-only to avoid changing user-owned work; scoped lint passed with `--fix`.
- Tier 1: `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q` ->
  stopped after 121 passes on
  `test_graphviz_cluster_skeleton_flag_preserves_interleaved_order`; the dirty,
  out-of-scope Sugiyama worktree returned a different within-rank order. An
  isolated rerun reproduced the failure with another differing rank order.
- Tier 2: `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`
  -> stopped after 167 passes, 1 xfail, and 88 deselections on the unrelated
  dirty classic-competitor test `test_graphviz_base_forwards_timeout`; its fake
  does not accept the new `graph_attributes` keyword in the existing worktree.

## Assumptions and concerns

The reference adapter intentionally reproduces the installed NetworkX 3.6.1
implementation rather than stabilizing ARPACK with Dagua-specific policies.
Sparse eigensolvers can select different bases in repeated eigenspaces; those
differences are part of the real external reference behavior and must not be
hidden by oracle-to-reimplementation delegation.

No dead code was introduced. The NetworkX-only edge-case and Laplacian helpers
remain reachable from the random-walk variant.

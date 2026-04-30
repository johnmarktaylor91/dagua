# Round 23 LGL Summary

## Scope

Primary source: `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_lgl.md`.
Round 22 had already committed ranked items #1, #3 trace coverage, and #5.
This round swept the remaining ranked items and committed only changes that did
not regress the requested median RMSD.

## Measurement

Baseline command:

```bash
python scripts/algo_fidelity_live_compare.py classic_lgl igraph_lgl \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_23/lgl/baseline
```

Post-fix command:

```bash
python scripts/algo_fidelity_live_compare.py classic_lgl igraph_lgl \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_23/lgl/post_fix
```

Results:

| Run | Rows | Median | P25 | P75 | P95 | Worst |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 75 | 0.194789 | 0.154821 | 0.283121 | 0.283121 | linear_3layer_mlp 0.283121 |
| post_fix | 75 | 0.194789 | 0.154821 | 0.283121 | 0.283121 | linear_3layer_mlp 0.283121 |

## Ranked Items

| # | Item | Size estimate | Status |
| ---: | --- | --- | --- |
| 1 | Disable edge-weight influence for `classic_lgl` fidelity mode | 10-25 lines | Already committed in Round 22. Verified still covered by `test_lgl_edge_weights_ignored_by_default`. |
| 2 | Make RNG one-stream compatible with igraph call order | 40-90 lines | Attempted, then reverted. Matching igraph's root/layout/shell stream and column-major `layout_random` order raised median RMSD from 0.194789 to 0.210673 on the required subset, which violates the no-regression policy. |
| 3 | Trace and align layer boundary indexing | 30-80 lines | Already addressed in Round 22. Trace coverage confirmed Dagua's per-depth loop matches igraph boundary assumptions on path/star/tree micrographs. |
| 4 | Align first-shell angular formula exactly | 15-40 lines | Verified no code change. Given the Round 22 boundary trace, Dagua's `len(next_layer)` denominator and zero-based child index are equivalent to igraph's `VECTOR(layers)[2] - 1` and `j - 1` for the first shell. |
| 5 | Mimic igraph `maxchange` sign behavior | 5-15 lines | Already committed in Round 22. Verified still covered by `test_lgl_igraph_positive_maxchange_rule`. |
| 6 | Add igraph-compatible initial random layout path | 30-60 lines | Attempted together with item #2, then reverted for the same median regression. Deferred until a broader RNG/grid parity change can be validated without worsening the subset median. |
| 7 | Normalize/remove igraph adapter `50.0` scale | Small | Probed and skipped. A per-adapter scale change touched shared `igraph_competitor.py` and did not explain the RMSD movement; no demonstrated value for this lgl-only round. |
| 8 | Mirror igraph disconnected warning / semantics | Small | Committed in `93f3199`. Dagua now emits `UserWarning: LGL layout does not support disconnected graphs yet.` when BFS does not reach every node. |
| 9 | Port or inspect `igraph_2dgrid_t` exact grid order/boundaries | 150-300 lines | Skipped as too large for the <200 net-line threshold and high-risk because it would rewrite the active repulsion loop. |
| 10 | Match igraph validation for explicit scalar parameters | Small | Committed in `93f3199`. `maxdelta`, `area`, `repulserad`, `cellsize`, and `root` now fail fast with igraph-compatible validation. |

## Verification

- `pytest tests/test_layout/ -x --tb=short -q -k "lgl"` after commit `93f3199`: `5 passed, 330 deselected in 0.28s`.
- `ruff check dagua/layout/ops/lgl.py dagua/layout/ops/pipelines/lgl.py tests/test_layout/test_lgl_fidelity.py --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `python scripts/algo_fidelity_live_compare.py ... --output-dir eval_output/algo_fidelity/round_23/lgl/post_fix`: completed, median unchanged at `0.194789`.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`: blocked during collection by pre-existing out-of-scope `ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'` in `tests/test_classic_drl.py:10`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: started but overlapped with multiple long-running parallel-agent invocations of the same command; lgl-specific requested verification is the authoritative result for this round.

## Concerns

- The cheap remaining correctness fixes are neutral on the five-graph RMSD subset.
- The RNG-stream fix is intuitively closer to igraph but empirically worsens this subset unless paired with deeper grid/iteration parity work.
- Exact `igraph_2dgrid_t` behavior remains the largest unresolved implementation gap.

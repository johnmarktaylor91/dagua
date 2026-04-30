# Round 22 LGL Summary

## Changes

- Applied the Round 22 recommended bundle from
  `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_lgl.md`.
- Disabled LGL edge-weight attraction by default because the diff identifies
  this as the clearest objective mismatch with igraph LGL
  (`ROUND_21_DIFF_lgl.md:218-221`, `ROUND_21_DIFF_lgl.md:267`).
- Factored the LGL BFS layer trace and added regression coverage for path,
  star, and binary-tree layer boundaries. The trace confirmed Dagua's
  per-depth layers already match igraph boundary-vector assumptions for the
  tested cases (`ROUND_21_DIFF_lgl.md:228-231`, `ROUND_21_DIFF_lgl.md:268`).
- Matched igraph's positive-component `maxchange` convergence rule by default
  (`ROUND_21_DIFF_lgl.md:238-241`, `ROUND_21_DIFF_lgl.md:269`).
- Did not port `igraph_2dgrid_t`, per the explicit recommended scope
  (`ROUND_21_DIFF_lgl.md:270`).

## Measurement

Command:

```bash
python scripts/algo_fidelity_live_compare.py classic_lgl igraph_lgl \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_22/lgl/baseline
```

Baseline:

- Rows: 75
- Median: 0.194789
- P25: 0.154821
- P75: 0.283121
- P95: 0.283121
- Worst: linear_3layer_mlp 0.283121

After:

- Rows: 75
- Median: 0.194789
- P25: 0.154821
- P75: 0.283121
- P95: 0.283121
- Worst: linear_3layer_mlp 0.283121

Result: median unchanged on this five-graph subset. The commit criterion is met
via clean, regression-tested fidelity infrastructure/defaults.

## Verification

- `pytest tests/test_layout/ -x --tb=short -q -k "lgl"`: initially passed
  before unrelated out-of-scope fidelity tests changed in the worktree
  (`3 passed, 243 deselected in 0.26s`); final rerun is blocked during
  collection by pre-existing out-of-scope
  `ImportError: cannot import name '_FA2_REFERENCE_PACKAGE_ORDER'` in
  `tests/test_layout/test_fa2_fidelity.py:7`.
- `pytest tests/test_layout/test_lgl_fidelity.py -x --tb=short -q`: blocked
  during collection by pre-existing out-of-scope
  `NameError: name '_SUGIYAMA_ACYCLIC_EDGE_WEIGHTS_KEY' is not defined` in
  `dagua/layout/ops/sugiyama.py:1413`.
- `ruff check dagua/layout/ops/lgl.py dagua/layout/ops/pipelines/lgl.py tests/test_layout/test_lgl_fidelity.py --fix`:
  passed.
- `python -m py_compile dagua/layout/ops/lgl.py dagua/layout/ops/pipelines/lgl.py tests/test_layout/test_lgl_fidelity.py`:
  passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `ruff check . --fix`: blocked by pre-existing out-of-scope
  `F821 Undefined name _prevent_ogdf_oscillation` in
  `dagua/layout/ops/fmmm.py:1240`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: blocked
  during collection by pre-existing out-of-scope
  `ImportError: cannot import name 'VARIANTS' from 'dagua.eval.variants'` in
  `tests/test_layout/test_maxent_stress_fidelity.py:13`.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`:
  blocked during collection by pre-existing out-of-scope
  `ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'` in
  `tests/test_classic_drl.py:10`.

## Concerns

- The requested subset does not include weighted anomaly graphs, so the
  edge-weight fix did not move the reported median here.
- The remaining LGL shape gap is likely dominated by RNG stream parity and
  igraph grid traversal semantics, both called out as deeper follow-up work in
  the Round 21 diff.

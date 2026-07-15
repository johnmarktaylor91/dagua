# W-TSNETFIX report

## Summary

The classic tsNET fidelity pipeline no longer imports or calls sklearn at
runtime. Dagua now owns the conditional-probability binary search and joint
probability symmetrization used by exact t-SNE.

The canonical rescore artifact is:

- `eval_output/fidelity_definitive/per_combo_r79_tsnet.jsonl`

All 18 rows use canonical `graph::engine` combo IDs. All 18 remain
`POSITIONAL_IDENTICAL`; the vendored implementation produced 1,280/1,280
byte-identical matched seeded layouts with maximum absolute coordinate
difference `0.0`.

## Changes

Changed production file:

- `dagua/layout/ops/pipelines/tsnet.py`: added the native float32-distance /
  float64-accumulator conditional-affinity search and symmetric condensed
  joint-probability calculation; removed the runtime import and call to
  `sklearn.manifold._t_sne._joint_probabilities`.

Changed regression coverage:

- `tests/test_pipeline_tsnet.py`: proves the vendored probability vector is
  bit-exact with the reference primitive in the test environment and adds an
  AST guard forbidding sklearn imports in both tsNET production modules.

The binary search preserves the reference arithmetic details: 100 maximum
precision-search steps, `1e-5` entropy tolerance, float32 input distances,
float64 probability accumulation, and row-by-row scalar summation. The final
symmetrization normalizes `(P + P.T)` by its sum, mathematically `2N`, and
returns scipy condensed-matrix order for the existing exact optimizer.

## Runtime delegation audit

This production-path scan returns no matches:

```text
rg -n '^\s*(from sklearn|import sklearn)' \
  dagua/layout/ops/pipelines/tsnet.py dagua/layout/ops/tsnet.py
# no output
```

The sklearn-backed `tsne_competitor` reference adapter remains unchanged, as
required. No other pipeline, cause file, `fmmm.py`, `sugiyama.py`, or
`networkx_competitor.py` was changed by this work.

## Seeded benchmark and rescore

Fresh benchmark directories:

- `eval_output/benchmark_100seed_r79_tsnet_vendor_default`: 2,000/2,000 OK,
  zero skips, errors, or timeouts; seeds 100-199 for ten default-variant graph
  pairs and their explicitly seeded references.
- `eval_output/benchmark_35seed_r79_tsnet_vendor_variants`: 560/560 OK, zero
  skips, errors, or timeouts; seeds 100-134 for eight parameterized-variant
  graph pairs and their explicitly seeded references.

Both commands passed `--seed-refs tsne_graph` and selected the canonical
`tsne_graph__for__classic_tsnet_*` references. Scoring overlaid the two fresh
directories and wrote 18 unique canonical rows. Ten default rows have 100
matched seeds; eight parameterized rows have 35 matched seeds. Every row has
`dist_equivalent=true` and `insufficient_data=false`.

Direct fresh-tensor proof:

```text
{'matched_layout_pairs': 1280, 'byte_exact': 1280, 'max_abs': 0.0}
```

## Before and after tiers

| State | Rows | Tier | Divergent | Combo ID form |
|---|---:|---|---:|---|
| Before (r79 coverage artifact) | 18 | `POSITIONAL_IDENTICAL` (18) | 0 | noncanonical `NEGREF` suffix |
| After vendoring and fresh rescore | 18 | `POSITIONAL_IDENTICAL` (18) | 0 | canonical `graph::engine` |

The tier is unchanged because the vendored probability calculation and the
downstream optimizer reproduce the reference tensors exactly.

## Assumptions and choices

The task required a seeded rescore but did not prescribe a seed count. I used
100 seeds for all ten default rows and the campaign-standard 35 seeds for the
eight parameterized rows. This exceeds the scorer's 30-pair mode-A minimum for
every row while avoiding an unnecessary second 100-seed parameter sweep after
bit-exactness was established.

The implementation uses scipy's existing `squareform` utility only to encode
the native symmetric matrix in condensed order. It does not delegate any t-SNE
or affinity mathematics to a reference package.

## Test results

- `pytest tests/test_pipeline_tsnet.py -x --tb=short -q`: 16 passed.
- `mypy --follow-imports=silent dagua/cli.py`: success, no issues in 1 source.
- `ruff check dagua/layout/ops/pipelines/tsnet.py tests/test_pipeline_tsnet.py --fix`:
  passed.
- `ruff check . --fix`: tsNET files passed; repository-wide check stopped on
  21 pre-existing errors in untracked research scripts under
  `.project-context/research/...` and `.research/r81_diag/`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: 137 passed
  before an unrelated in-progress Sugiyama assertion failed in
  `tests/test_layout/test_dot_rank.py::test_graphviz_cluster_x_aux_edges_use_boundary_nodes`.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`:
  167 passed, 88 deselected, and 1 xfailed before the pre-existing Graphviz
  mock-signature failure in
  `tests/test_classic_competitor.py::test_graphviz_base_forwards_timeout`
  (`graph_attributes` was not accepted by the test fake).

## Concerns and knowledge

The scalar search is intentionally slower than sklearn's compiled private
extension on larger graphs; preserving its summation order is what keeps the
full stochastic optimizer byte-identical. tsNET fidelity remains exact, and
the runtime reference-delegation violation is closed.

No dead code became unreachable.

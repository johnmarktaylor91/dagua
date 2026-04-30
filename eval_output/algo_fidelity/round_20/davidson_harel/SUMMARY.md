# Round 20 Davidson-Harel Summary

## Result

Commit criterion met.

- Baseline graph-median RMSD: `0.237719`
- After graph-median RMSD: `0.166609`
- Improvement: `0.071110`
- Baseline row median: `0.219860` across 75 pairwise rows
- After row median: `0.137078` across 75 pairwise rows

The measured improvement exceeds the `>= 0.05` threshold.

## Changes

- Added `fineiter` support to the Davidson-Harel pipeline, defaulting to 10 fine-tuning rounds.
- Added an igraph-style fine-tuning mode using `0.01 * min(span_x, span_y)` proposal radius.
- Gated node-edge distance energy to fine-tuning only.
- Disabled uphill acceptance during fine-tuning.
- Replaced full candidate energy recomputation with local move-delta blocks for node distance, border, edge length, edge crossings, and node-edge distance.
- Made final centering/scaling opt-in by adding `skip_finalization=True` as the default fidelity behavior.

## Measurement

Command:

```bash
python scripts/algo_fidelity_live_compare.py classic_davidson_harel igraph_davidson_harel \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_20/davidson_harel/<baseline|after>
```

Per-graph median RMSD:

| Graph | Baseline | After | Delta | After TOST |
|---|---:|---:|---:|---|
| linear_3layer_mlp | 0.288095 | 0.210866 | -0.077229 | equivalent_at_1x |
| mixed_width_labels | 0.177112 | 0.132642 | -0.044470 | equivalent_at_2x |
| nested_shallow_enc_dec | 0.279155 | 0.146296 | -0.132859 | not_equivalent |
| parallel_multiedge_bundle | 0.210108 | 0.256593 | +0.046485 | not_equivalent |
| tl_mlp_3layer | 0.237719 | 0.166609 | -0.071110 | equivalent_at_1x |

## Verification

```text
ruff check dagua/layout/ops/davidson_harel.py dagua/layout/ops/pipelines/davidson_harel.py --fix
All checks passed!

pytest tests/test_layout/ -x --tb=short -q -k "davidson"
243 deselected in 0.21s

pytest tests/test_layout/ -x --tb=short -q
243 passed, 6 warnings in 1323.08s (0:22:03)

ruff check . --fix --diff
<no output; clean>

mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file

pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
280 passed, 6 warnings in 1376.83s (0:22:56)
```

## Residuals

- `parallel_multiedge_bundle` regressed on median RMSD. This matches the Round 19 diff's remaining edge-multiplicity warning: the current patch still uses Dagua's unique undirected edge cache rather than igraph's original multiedge order for crossing and node-edge terms.
- RNG stream parity remains unresolved. Dagua still uses PyTorch RNG and resets the move generator separately from initialization.
- Segment intersection and boundary clamps still differ from igraph.

# Round 23 Spectral Exhaustive Sweep

Pairing: `classic_spectral` vs `nx_spectral`.

## Measurement

Baseline command:

```bash
python scripts/algo_fidelity_live_compare.py classic_spectral nx_spectral \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_23/spectral/baseline
```

Baseline result: `30` rows, `5` graphs, median `0.100482`, p25 `0.100482`,
p75 `0.111416`, p95 `0.299828`, worst `mixed_width_labels 0.346932`.

Post-fix command:

```bash
python scripts/algo_fidelity_live_compare.py classic_spectral nx_spectral \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_23/spectral/post_fix
```

Post-fix result: `30` rows, `5` graphs, median `0.100482`, p25 `0.100482`,
p75 `0.111416`, p95 `0.299828`, worst `mixed_width_labels 0.346932`.

## Ranked Items

1. NetworkX-fidelity unnormalized Laplacian: already committed in Round 22 and
   left intact. Verified by existing `classic_spectral_nx_fidelity` regression
   coverage. Estimated size: already done. Status: committed before Round 23.
2. NetworkX `N <= 2` special case: already committed in Round 22 and left
   intact. Verified by `test_networkx_fidelity_collapses_two_node_graph_to_center`.
   Estimated size: already done. Status: committed before Round 23.
3. NetworkX skip-first eigenvector selection: already committed in Round 22 and
   left intact. Verified by `test_networkx_fidelity_selects_sorted_slice_after_first_eigenvector`.
   Estimated size: already done. Status: committed before Round 23.
4. Preserve weighted edges in `ClassicSpectral.layout`: applied in `14743c4`.
   Direct adapter now forwards `graph.edge_weights`; regression test added.
   Estimated size: `<10` code lines. Status: committed.
5. Parallel-edge handling in the NetworkX adapter: verified present in current
   Round 23 base via `_graph_to_nx(..., duplicate_policy="sum")`; regression
   test added in `14743c4` to lock spectral reference behavior. Estimated size:
   already present in base, test-only this commit. Status: verified and covered.
6. Match NetworkX sparse `k/ncv` in fidelity mode: applied in `14743c4`.
   Fidelity mode now requests `dim + 1` eigenpairs and NetworkX-style `ncv`.
   Estimated size: `~20` code lines. Status: committed.
7. Use dense `np.linalg.eig` for NetworkX fidelity: applied in `14743c4`.
   Default Dagua path still uses `eigh`; fidelity mode mirrors NetworkX.
   Estimated size: `<10` code lines. Status: committed.
8. Gate NetworkX adapter 500x scale for spectral: verified present in current
   Round 23 base via `NetworkXSpectral.output_scale = 1.0`; regression test
   added in `14743c4`. Estimated size: already present in base, test-only this
   commit. Status: verified and covered.

Skipped items: none. No item exceeded the `<~200` net-line feasibility limit.

## Verification

- `ruff check dagua/layout/ops/embed.py dagua/eval/competitors/classic_competitor.py tests/test_layout/test_spectral_fidelity.py --fix`: passed.
- `ruff format dagua/layout/ops/embed.py dagua/eval/competitors/classic_competitor.py tests/test_layout/test_spectral_fidelity.py --check`: passed.
- `pytest tests/test_layout/test_spectral_fidelity.py tests/test_pipeline_spectral.py -x --tb=short -q`: passed, `19 passed`.
- Before `14743c4`, `pytest tests/test_layout/ -x --tb=short -q -k "spectral"` briefly failed in `tests/test_layout/test_umap_fidelity.py`, outside the spectral family, due concurrent UMAP work. After the commit and current base updates, the required selector passed: `9 passed, 325 deselected`.
- `git diff --stat HEAD~1 HEAD` for `14743c4` showed only:
  - `dagua/eval/competitors/classic_competitor.py`
  - `dagua/layout/ops/embed.py`
  - `tests/test_layout/test_spectral_fidelity.py`

## Notes

The bounded live-compare median did not move because the five-graph subset does
not exercise weighted spectral input, sparse `N >= 500` eigensolver behavior, or
the raw adapter scale in a way that changes the normalized comparator metric.
The changes are correctness and fidelity guardrails for the documented residual
edge cases.

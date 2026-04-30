# Round 23 Sugiyama Fidelity Sweep

## Scope

- Family: `classic_sugiyama` vs `igraph_sugiyama`
- Code commit: `5230634 feat(fidelity): round 23 sugiyama -- igraph parity sweep`
- Baseline output: `eval_output/algo_fidelity/round_23/sugiyama/baseline/`
- Post-fix output: `eval_output/algo_fidelity/round_23/sugiyama/post_fix/`

## Measurements

Baseline command:

```bash
python scripts/algo_fidelity_live_compare.py classic_sugiyama igraph_sugiyama \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_23/sugiyama/baseline
```

Baseline result:

- graphs: 5
- median: `0.000000`
- p25: `0.000000`
- p75: `0.000000`
- p95: `0.026252`
- worst: `mixed_width_labels 0.032815`

Post-fix command:

```bash
python scripts/algo_fidelity_live_compare.py classic_sugiyama igraph_sugiyama \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_23/sugiyama/post_fix
```

Post-fix result:

- graphs: 5
- median: `0.000000`
- p25: `0.000000`
- p75: `0.000000`
- p95: `0.000000`
- worst: `tl_mlp_3layer 0.000000`

## Ranked Items

1. Filter self-loops before layering and expansion: already addressed by Round 22; retained and covered by `test_sugiyama_ignores_self_loops_before_layering`.
2. Add igraph fidelity component-packing path for weak components: addressed in `5230634`. Added deterministic weak-component slicing and X packing for `fidelity_mode="igraph"` when no traces or edge routes are requested.
3. Match igraph early-stop barycenter semantics: already addressed by Round 22; retained and covered by `test_sugiyama_igraph_fidelity_stops_after_stable_ordering`.
4. Align barycenter weighting and multiedge semantics with igraph: already addressed by Round 22; regression fixture updated in `5230634` so component packing does not mask the incidence-average behavior.
5. Add igraph-compatible cyclic directed layering mode: skipped. Eades/GLPK-style layering parity is larger than the round threshold and needs a separate scoped design because it would replace the current DFS/greedy feedback path.
6. Use igraph separation formula when node sizes are not part of the reference input: addressed in `5230634`. `fidelity_mode="igraph"` now defaults `use_node_sizes_for_spacing=False`, preserving default graphviz-style node-width spacing outside fidelity mode.
7. Make base adapter defaults explicit for igraph-vs-igraph fidelity mode: addressed in `5230634`. `ClassicSugiyama` now routes through `_ClassicBase.layout_with_variant` with unit spacing, `barycenter_passes=100`, and `fidelity_mode="igraph"`.
8. Correct stochasticity metadata for Sugiyama: addressed in `5230634`. Base `classic_sugiyama` and all Sugiyama variants are marked deterministic.
9. Audit and possibly emulate igraph's exact type-1 conflict behavior: skipped. The C reference loop is ambiguous/suspicious as documented in Round 21; bounded post-fix RMSD is already effectively zero, so emulating it without a targeted failing fixture is not justified.
10. Delay final centering or make it optional for raw igraph coordinate parity: addressed in `5230634`. Coordinate centering is now configurable and defaults off in igraph fidelity mode.

## Verification

- `ruff check . --fix`: passed
- `mypy --follow-imports=silent dagua/cli.py`: passed
- `pytest tests/test_layout/ -x --tb=short -q -k "sugiyama"` after commit `5230634`: `6 passed, 330 deselected`
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: blocked during collection by unrelated `classical_mds` import error: `cannot import name 'ClassicalMDSFinalizePositionsConfig' from 'dagua.layout.ops.postprocess'`
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`: blocked during collection by unrelated classic import error: `cannot import name 'layout_drl' from 'dagua.layout.classic'`

## Notes

- No `scripts/ogdf_runner.cpp` changes were needed for Sugiyama.
- No render/style/cosmetic-sprint files were touched by the Sugiyama commit.
- Remaining dirty workspace files belong to parallel family/cosmetic work and were left unstaged.

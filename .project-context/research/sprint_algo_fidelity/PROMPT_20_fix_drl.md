<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 20 ADVERSARIAL FIX for **drl**.

## SPEC

Your spec is `.project-context/research/sprint_algo_fidelity/ROUND_19_DIFF_drl.md`.
Read it end-to-end. 13-item ranked fix list. Apply STAGE 1 (per the
"Recommended Round 20 Fix Scope" section) which is the top 3 items:

1. **Match igraph's effective `ReCompute()` sweep schedule**: add init-parameter sweep, boundary sweeps, final stage6 sweep. Or rewrite phase runner around igraph's stage-control order at `drl_graph.cpp:610-611, 624-808`. Affects every variant.
2. **Fix candidate semantics literally**: igraph compares old-coordinate energy vs random-coordinate energy, accepts analytic if old wins (`drl_graph.cpp:923-929, 943-947, 964-973`). Round 14 tested partial.
3. **Align REFINE + FINAL presets**: REFINE init damping = 0.0, cooldown temp = 200.0 (`drl_layout.cpp:343-361`). FINAL expansion = (50, 50.0, 0.1, 0.25) (`drl_layout.cpp:385-388`).

DO NOT touch density grid, edge cutting, or RNG in this round (per the
diff doc's staged plan). Those are Round 21+.

## Process

1. Read `ROUND_19_DIFF_drl.md` fully.
2. Baseline: 3 seeds × 5 small graphs:
   ```
   python scripts/algo_fidelity_live_compare.py classic_drl igraph_drl \
       --seeds 3 \
       --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
       --output-dir eval_output/algo_fidelity/round_20/drl/baseline
   ```
3. Apply stage 1 bundle.
4. Tests: `pytest tests/test_layout/ -x --tb=short -q -k "drl"`
5. Re-measure.
6. COMMIT criterion: median improves >= 0.03 (smaller threshold; drl already near floor).
7. Commit `feat(fidelity): round 20 drl -- <short>` if met. Otherwise revert + ROUND_20_RESIDUAL_drl.md.
8. Summary at `eval_output/algo_fidelity/round_20/drl/SUMMARY.md`.

## Scope

**Allowed**:
- `dagua/layout/ops/drl.py`
- `dagua/layout/ops/pipelines/drl.py`
- `dagua/layout/ops/state.py` (new fields only)
- `eval_output/algo_fidelity/round_20/drl/**`
- `.project-context/research/sprint_algo_fidelity/ROUND_20_*drl*.md`
- `tests/test_layout/test_*drl*.py`

**Out of scope**: all other families. Density grid + edge cutting deferred to Round 21.

## Verification
- pytest layout drl tests pass
- live_compare runs cleanly
- git diff scope clean

ONE commit only IF measurable improvement.
</task>

<scope_constraints>drl files only. Density grid + edge cutting deferred.</scope_constraints>

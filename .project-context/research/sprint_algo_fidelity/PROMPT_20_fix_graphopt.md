<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 20 ADVERSARIAL FIX for **graphopt**.

## SPEC

Your spec is `.project-context/research/sprint_algo_fidelity/ROUND_19_DIFF_graphopt.md`.
Read it end-to-end. 6-item ranked fix list. Apply top 3:

1. **Init range AND draw order**: change GraphOpt initialization to match igraph's `[-1, 1]` AND **column-major fill** (igraph's `igraph_layout_random` fills X first then Y; dagua may fill row-major). See `init.py:516-519`. Largest unavoidable coordinate divergence.
2. **RNG engine semantics**: dagua uses Python `random.Random`; igraph uses `RNG_UNIF`. Either route through numpy `RandomState` (which is what the seeded benchmark harness already passes through) OR document the seed-mismatch.
3. **Edge weights**: igraph GraphOpt does NOT use weights (`graphopt.c:341-347, 416-422`); dagua multiplies spring force by weights (`force.py:1472-1478`). Disable weight multiplication for graphopt fidelity path.

## Process

1. Read `ROUND_19_DIFF_graphopt.md` fully.
2. Baseline: 3 seeds × 5 small graphs:
   ```
   python scripts/algo_fidelity_live_compare.py classic_graphopt igraph_graphopt \
       --seeds 3 \
       --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
       --output-dir eval_output/algo_fidelity/round_20/graphopt/baseline
   ```
3. Apply top 3.
4. Tests: `pytest tests/test_layout/ -x --tb=short -q -k "graphopt or init"`. Watch that other init classes (RandomUniformInit, KamadaKawaiInit, FA2Init, etc.) DO NOT regress.
5. Re-measure.
6. COMMIT criterion: median improves >= 0.02 (small threshold; baseline already 0.067).
7. Commit `feat(fidelity): round 20 graphopt -- <short>` if met.

## Scope

**Allowed**:
- `dagua/layout/ops/init.py` -- ONLY `GraphOptInitializePositions*` classes
- `dagua/layout/ops/force.py` -- ONLY graphopt-related sections (line 1295+)
- `dagua/layout/ops/pipelines/graphopt.py`
- `eval_output/algo_fidelity/round_20/graphopt/**`
- `.project-context/research/sprint_algo_fidelity/ROUND_20_*graphopt*.md`
- `tests/test_layout/test_*graphopt*.py`

**Out of scope**:
- Other init classes (RandomUniformInit, KamadaKawaiInit, FA2Init, XavierInit, SpectralInit, ClassicalMDSInit, PivotMDSInit, etc.)
- Other family pipelines

## Verification
- pytest graphopt + init tests pass
- live_compare runs cleanly
- git diff: ONLY GraphOpt-named classes in init.py, ONLY graphopt sections of force.py

ONE commit only IF improvement.
</task>

<scope_constraints>graphopt files only. Other init classes untouched.</scope_constraints>

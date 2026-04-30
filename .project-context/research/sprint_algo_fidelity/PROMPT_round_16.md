<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`. ONE working branch.

Round 16 of the algo_fidelity sprint, retry of graphopt attack from
Round 15 BLOCKED. Round 15 baseline + diagnosis already done.

Read these in order:
1. `.project-context/research/sprint_algo_fidelity/ROUND_15_BLOCKED.md`
2. `eval_output/algo_fidelity/round_15/SUMMARY.md`
3. `.project-context/research/sprint_algo_fidelity/algo_fidelity_STATE.md`

## Round 15 result + Round 16 plan

Round 15 baseline (5 graphs × 3 seeds, classic_graphopt vs igraph_graphopt):
```
median: 0.067702
p25: 0.018174
worst: tl_mlp_3layer 0.308918

graph-level TOST:
- parallel_multiedge_bundle: equivalent_at_0.5x
- tl_mlp_3layer: equivalent_at_1x
- linear_3layer_mlp: not_equivalent
- nested_shallow_enc_dec: not_equivalent
- mixed_width_labels: not_equivalent
```

Round 15 identified the highest-confidence lever as **init range alignment**:
- igraph graphopt: `igraph_layout_random()` -> uniform [-1, 1]
- dagua graphopt: `GraphOptInitializePositions` -> uniform [0, 1]

The lever lives in `dagua/layout/ops/init.py` lines 436-490 (class
`GraphOptInitializePositions`). I confirmed `GraphOptInitializePositions`
is graphopt-specific (only used by `dagua/layout/ops/pipelines/graphopt.py`),
so editing it doesn't affect other pipelines. **This file is in scope
for Round 16.**

## What to do

### Step 1: Apply the init-range alignment fix (15 min)

In `dagua/layout/ops/init.py`, modify `GraphOptInitializePositions` (and
its `Config` class if needed) to initialize positions uniformly in
[-1, 1] × [-1, 1] instead of [0, 1] × [0, 1]. Match igraph_layout_random's
behavior.

Verify by reading the dagua init code first to understand the exact
current behavior. The fix may be a single line (change [0, 1] to
[-1, 1]) or may require config field changes.

If the dagua init also has additional scaling that doesn't exist in
igraph (or vice versa), align that too.

### Step 2: Verify other graphviz/igraph alignment opportunities

While in `pipelines/graphopt.py`, also check the hyperparameter defaults
against igraph's:
- niter: 500 (igraph default)
- node_charge: 0.001
- node_mass: 30
- spring_length: 0
- spring_constant: 1
- max_sa_movement: 5
- COULOMBS_CONSTANT: 8987500000.0

If dagua's pipeline defaults diverge, align them.

### Step 3: Measure on the same Round 15 subset

```
cd /home/jtaylor/projects/dagua
python scripts/algo_fidelity_live_compare.py classic_graphopt igraph_graphopt \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_16/post_fix
```

Compare against Round 15's baseline at
`eval_output/algo_fidelity/round_15/baseline_small/multi_seed_summary.json`.

COMMIT criterion: median improves by >= 0.02 (small threshold because
already at 0.068; small absolute moves matter), OR aggregate TOST flips
toward equivalent_at_<=2x on more graphs.

### Step 4: Tests

```
pytest tests/test_layout/ -x --tb=short -q -k "graphopt or init" 2>&1 | tail -30
```

Watch for snapshot tests on init.py that might need updating.
**Critical**: only `GraphOptInitializePositions` should change; other
init classes (XavierInit, RandomUniformInit, KamadaKawaiInitializePositions,
FA2InitializePositions, etc.) must NOT regress.

### Step 5: Commit OR document residual

If COMMITTED:
```
feat(fidelity): round 16 -- graphopt-vs-igraph init range alignment

- Identified divergence: dagua GraphOptInitializePositions uses uniform[0,1]; igraph graphopt uses uniform[-1,1] via igraph_layout_random
- Fix: aligned dagua init range to [-1, 1] (other init ops untouched)
- graphopt small-graph median: 0.0677 -> <NEW>
- TOST aggregate: <verdict>
- Tests: <count> passed
```

If RESIDUAL: `ROUND_16_RESIDUAL.md` with classification.

### Step 6: Per-round summary

`eval_output/algo_fidelity/round_16/SUMMARY.md`.

### Step 7: Update STATE.md

Append iteration log row. Set `current_family: neulay` for Round 17.
</task>

<scope_constraints>
**HARD scope -- DO NOT TOUCH:**
- `dagua/render/**`, `dagua/styles.py`, `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`
- All other family pipelines (sugiyama, fmmm, sfdp, stress_majorization,
  classical_mds, davidson_harel, drl, neulay, tsnet, fa2)

**Allowed for Round 16:**
- `dagua/layout/ops/init.py` -- ONLY the `GraphOptInitializePositions` and
  `GraphOptInitializePositionsConfig` classes (lines 436-490 area).
  Do NOT modify other init classes (XavierInit, RandomUniformInit, etc.) --
  those serve other pipelines.
- `dagua/layout/ops/pipelines/graphopt.py`
- `dagua/layout/ops/state.py` ONLY if SolveState field needed
- `eval_output/algo_fidelity/round_16/**`
- `.project-context/research/sprint_algo_fidelity/**`
- `tests/test_layout/test_*graphopt*.py` for snapshot updates
</scope_constraints>

<default_follow_through_policy>
The init-range fix is mechanical (1-3 line change) and high-confidence.
If it lands, commit. If it regresses or doesn't move the needle,
document carefully -- the residual is real algorithmic work.

If during inspection you find dagua's hyperparameter defaults already
match igraph's, the only divergence is init-range, and you should
expect the fix to close most of the gap.
</default_follow_through_policy>

<completeness_contract>
1. **COMMITTED** if commit criterion met
2. **RESIDUAL** if no improvement
3. **BLOCKED** if a hard infra issue
</completeness_contract>

<verification_loop>
- pytest tests/test_layout/ -x --tb=short -q -k "graphopt or init"
- live_compare with bounded subset runs cleanly
- `git diff --stat HEAD~0` before commit shows only allowed scope -- and
  init.py shows only `GraphOptInitializePositions*` related changes
</verification_loop>

<missing_context_gating>
ABORT if:
- live_compare for graphopt times out
- The init.py classes have been refactored away

Write ROUND_16_BLOCKED.md and stop.
</missing_context_gating>

<action_safety>
- ONE commit on develop only IF measurable improvement.
- Never delete eval_output files.
- Touching only the GraphOpt-named classes in init.py is critical -- a
  sweeping init.py change could break many pipelines.
</action_safety>

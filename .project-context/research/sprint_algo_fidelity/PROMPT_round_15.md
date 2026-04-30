<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`. ONE working branch.

Round 15 of the algo_fidelity sprint. drl partial_match was Round 14 RESIDUAL
(small improvement, 4/5 graphs already TOST equivalent at 1x). Round 15
attacks **graphopt** -- next worst Phase 2 family.

Read these in order:
1. `.project-context/research/sprint_algo_fidelity/algo_fidelity_STATE.md`
2. `eval_output/algo_fidelity/round_13/SUMMARY.md` (davidson_harel pattern)
3. `eval_output/algo_fidelity/round_14/SUMMARY.md` (drl context)

## Round 15 target: graphopt family

Mega-run verdict: **partial_match** (RMSD 0.10-0.16 across 6 variants:
default, niter50/200/500/1000, charge001). Stochastic via random init.

## igraph graphopt source -- READ FIRST

Source: `/home/jtaylor/projects/_references/igraph/src/layout/graphopt.c`

### graphopt defaults (API doc, line 317-339)

```c
niter           = 500       /* num iterations */
node_charge     = 0.001     /* used in Coulomb-style repulsion */
node_mass       = 30        /* used for force-to-displacement */
spring_length   = 0         /* ideal spring length is zero */
spring_constant = 1         /* spring k */
max_sa_movement = 5         /* per-step movement cap */
```

### Physical constants (line 28)

```c
#define COULOMBS_CONSTANT 8987500000.0
```

### Force laws

- **Electrical repulsion** (line 145): `force = k * q^2 / d^2`
  where k = COULOMBS_CONSTANT, q = node_charge.
- **Spring force** (line 220-224): `displacement = d - spring_length`,
  `directed_force = -1 * spring_constant * displacement`.
- **Movement** (line 244-280): `x_movement = force_x / node_mass`,
  clamped by `max_sa_movement` per step.

The spring_length=0 default means spring force pulls nodes together
proportional to their actual distance (force = -spring_constant * d).
This combined with strong Coulomb repulsion produces a natural balance.

### Initial layout (line 412+)

If `use_seed=false`, igraph generates random initial positions in
some range. Verify the range from the source.

## Dagua graphopt surface

- Locate `dagua/layout/ops/graphopt.py` and `dagua/layout/ops/pipelines/graphopt.py`
- Read both end-to-end

Investigate:
1. **Hyperparameter alignment**: do dagua's defaults match igraph's exactly?
   - niter=500
   - node_charge=0.001
   - node_mass=30
   - spring_length=0
   - spring_constant=1
   - max_sa_movement=5
2. **Physical constant**: does dagua use COULOMBS_CONSTANT=8987500000.0
   in its repulsion? This is a key magnitude that determines the
   equilibrium scale.
3. **Force law math**: repulsion `k*q^2/d^2`, spring `-k*(d-L)`,
   displacement `force/mass` capped at `max_sa_movement`.
4. **Initial layout**: check the random init range matches igraph's.

## What to do

### Step 1: Live multi-seed baseline (10 min)

```
cd /home/jtaylor/projects/dagua
python scripts/algo_fidelity_live_compare.py classic_graphopt igraph_graphopt \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_15/baseline_small
```

(5 graphs, 3 seeds; if too slow drop to 4 graphs.)

If within-floor is high enough that TOST already says equivalent on
most graphs, document and move on (Round 14 pattern).

### Step 2: Diagnose (15 min)

Read dagua graphopt ops. Compare to igraph defaults + force-law math.
Write `.project-context/research/sprint_algo_fidelity/ROUND_15_DIAGNOSIS.md`.

### Step 3: ONE focused lever (15-30 min)

Same playbook. Most likely candidates:
- Default value alignment (especially COULOMBS_CONSTANT, node_charge,
  node_mass, max_sa_movement -- magnitudes matter)
- Force-law formula alignment if dagua uses different math
- Init-range alignment

### Step 4: Measure on the same small subset

```
python scripts/algo_fidelity_live_compare.py classic_graphopt igraph_graphopt \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_15/post_fix
```

COMMIT criterion: median improves by >= 0.03 OR aggregate TOST flips
toward equivalent_at_<=2x.

### Step 5: Tests + commit OR residual

```
pytest tests/test_layout/ -x --tb=short -q -k "graphopt" 2>&1 | tail -20
```

If COMMITTED:
```
feat(fidelity): round 15 -- graphopt-vs-igraph first lever (<short>)

- Identified divergence: <one sentence>
- Fix: <one sentence>
- graphopt small-graph median: <BEFORE> -> <AFTER>
- TOST aggregate: <verdict>
- Tests: <count> passed
```

If RESIDUAL: `ROUND_15_RESIDUAL.md`.

### Step 6: Per-round summary

`eval_output/algo_fidelity/round_15/SUMMARY.md`.

### Step 7: Update STATE.md

Append iteration log row. Set `current_family: neulay` for Round 16.
</task>

<scope_constraints>
**HARD scope -- DO NOT TOUCH:**
- `dagua/render/**`, `dagua/styles.py`, `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`
- All other family pipelines (sugiyama, fmmm, sfdp, stress_majorization,
  classical_mds, davidson_harel, drl, neulay, tsnet, fa2)

**Allowed:**
- `dagua/layout/ops/graphopt.py` (PRIMARY)
- `dagua/layout/ops/pipelines/graphopt.py`
- `dagua/layout/ops/state.py` ONLY if SolveState field needed
- `eval_output/algo_fidelity/round_15/**`
- `.project-context/research/sprint_algo_fidelity/**`
- `tests/test_layout/test_*graphopt*.py` for snapshot updates
</scope_constraints>

<default_follow_through_policy>
Same playbook as Round 13/14: small graph subset, hyperparameter
alignment first, measure with multi-seed TOST. graphopt's API is
simple (one function with 7 numeric params + use_seed bool) so
hyperparameter alignment is straightforward.

If graphopt is already stochastic-floor faithful per multi-seed TOST,
classify and move on -- a clean "no work needed" result is valuable.
</default_follow_through_policy>

<completeness_contract>
1. **COMMITTED** if commit criterion met
2. **RESIDUAL** if no high-confidence fix lands
3. **STOCHASTIC_FLOOR_MATCH** if multi-seed shows already equivalent
4. **BLOCKED** if hard infra issue
</completeness_contract>

<verification_loop>
- pytest tests/test_layout/ -x --tb=short -q -k "graphopt"
- live_compare with bounded subset runs cleanly
- `git diff --stat HEAD~0` before commit shows only allowed scope
</verification_loop>

<missing_context_gating>
ABORT if:
- live_compare for graphopt times out
- dagua graphopt ops file not found

Write ROUND_15_BLOCKED.md and stop.
</missing_context_gating>

<action_safety>
- ONE commit on develop only IF measurable improvement.
- Never delete eval_output files.
</action_safety>

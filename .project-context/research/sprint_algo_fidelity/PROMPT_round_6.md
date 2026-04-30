<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`. ONE working branch -- DO NOT create a new branch.

Round 6 of the algo_fidelity sprint. Read these in order:
1. `.project-context/research/sprint_algo_fidelity/algo_fidelity_STATE.md`
2. `eval_output/algo_fidelity/round_3/SUMMARY.md` (the dot-family WIN; pattern to emulate)
3. `eval_output/algo_fidelity/round_5/SUMMARY.md` (the fdp-family residuals; what NOT to repeat)

## Round 6 target: sfdp family

Round 1 baseline: classic_sfdp vs graphviz_sfdp median RMSD 0.0915
(this is the smallest graphviz gap of the four families). Likely a
hyperparameter/init alignment Round-3-style fix is achievable.

### Quick comparison: dagua sfdp vs graphviz sfdp defaults

Already aligned in `dagua/layout/ops/pipelines/sfdp.py`:
- `steps=500` matches graphviz `ctrl.maxiter=500`
- `theta=0.6` matches graphviz Barnes-Hut `bh=0.6` (in spring_electrical.c:36)
- `repulsive_exponent=-1.0` matches graphviz `ctrl.p=-1` (after AUTOP resolution)
- `seed=123` matches graphviz `ctrl.random_seed=123`

**Probably not aligned -- READ SOURCE TO VERIFY:**
- Random init: graphviz sets `ctrl.random_start = true` (spring_electrical.c:54).
  Does dagua start from random uniform layout or from spectral/coarsest?
- C constant: graphviz uses `static const double C = 0.2` (spring_electrical.c:36)
  in the formula `CRK = pow(C, (2-p)/3)/K`. Does dagua have an equivalent?
- Adaptive cooling: graphviz uses `step=0.1`, `cool=0.90`, adaptive_cooling=true
  (spring_electrical.c:60). Does dagua's cooling schedule match?
- Initial scaling: graphviz sets `ctrl.initial_scaling = -4`, applied during
  finalization (negative = scale relative to label size). Does dagua scale?
- Convergence tolerance: graphviz uses `tol=0.001/K` (spring_electrical.c:35).
  Does dagua converge differently?
- K: graphviz auto-sets `ctrl.K` to `average_edge_length(input_layout)`
  (spring_electrical.c:285). dagua may use a fixed K or different auto-tune.

## Graphviz sfdp source -- READ THIS FIRST

`/home/jtaylor/projects/_references/graphviz/lib/sfdpgen/`:

### Required reading

- **`spring_electrical.c:51-72`** -- `spring_electrical_control_new()`,
  the canonical defaults
- **`spring_electrical.c:30-40`** -- module-level constants
  (C=0.2, bh=0.6, tol=0.001, cool=0.90)
- **`spring_electrical.c:309-330`** -- the spring-electrical force-update
  loop (vanilla version)
- **`spring_electrical.c:255-330`** -- `spring_electrical_embedding`
  (the main solver -- look for K initialization, step scaling)
- **`Multilevel.c`** -- coarsening + prolongation logic
- **`sfdpinit.c`** -- entry point, post-processing pipeline
- **`post_process.c`** -- overlap removal (only matters if cached
  positions used overlap removal -- check the cached graphviz_sfdp
  positions to see if they show overlap)

### Useful greps for verification

```bash
# Default values:
grep -nE "ctrl\.\w+\s*=" \
  /home/jtaylor/projects/_references/graphviz/lib/sfdpgen/spring_electrical.c | head -20

# Force-law expressions:
grep -nE "pow\(|K\s*\*|disp\[" \
  /home/jtaylor/projects/_references/graphviz/lib/sfdpgen/spring_electrical.c | head -20

# Initialization:
grep -nE "random_start|init.*pos|initial_layout" \
  /home/jtaylor/projects/_references/graphviz/lib/sfdpgen/*.c
```

## What to do

### Step 1: Live baseline (5 min)

```
cd /home/jtaylor/projects/dagua
python scripts/algo_fidelity_live_compare.py classic_sfdp graphviz_sfdp \
    --output-dir eval_output/algo_fidelity/round_6/baseline
```

Confirm baseline RMSD median is in the ~0.09 ballpark (it's stochastic,
so allow some variance). If determinism check needed, run twice. If
non-deterministic, use median of 3 runs as baseline.

### Step 2: Compare dagua sfdp implementation to graphviz source (15 min)

Read these dagua files in detail:
- `dagua/layout/ops/pipelines/sfdp.py` -- pipeline + defaults (already
  partially shown in the prompt context above)
- `dagua/layout/ops/sfdp.py` -- the ops implementing each pipeline stage
  (BuildSFDPGraph, BuildSFDPHierarchy, InitSFDPCoarsestPositions,
   SFDPRefineCoarsestLevel, SFDPProlongateAndRefineLevels,
   SFDPFinalizePositions)

For each of the bullet-listed "probably not aligned" hypotheses above
(random_start, C constant, adaptive cooling, initial_scaling, tol, K),
either:
   a. Confirm dagua matches graphviz -> rule it out
   b. Confirm dagua differs from graphviz -> candidate lever

Pick the highest-confidence single lever. Most likely candidates in
order:
   1. **Initialization mismatch**: if dagua coarsest init uses
      spectral or zero, switching to graphviz-compatible random
      uniform with seed 123 may close the gap
   2. **C constant mismatch**: hard-coded constant in the
      `CRK = pow(C, (2-p)/3)/K` formula
   3. **Adaptive cooling parameters**: step/cool/tol values

### Step 3: Apply ONE focused lever (15-30 min)

Same scope rules as Round 3/5 (~50-150 lines net, no wholesale rewrites).

### Step 4: Measure

```
python scripts/algo_fidelity_live_compare.py classic_sfdp graphviz_sfdp \
    --output-dir eval_output/algo_fidelity/round_6/post_fix
```

COMMIT criterion: median improved by >= 0.02 (smaller threshold than
Round 3 because we're closer to the floor) AND no simple-graph
regressions (graphs with current RMSD < 0.02 stay below 0.02).

Stochastic note: SFDP is randomized. Run post_fix twice; if median
varies by more than 0.01 between runs, take the median of 3 runs.

### Step 5: Tests

```
pytest tests/test_layout/ -x --tb=short -q -k "sfdp" 2>&1 | tail -30
```

### Step 6: Commit OR document residual

If COMMITTED:
```
feat(fidelity): round 6 -- sfdp-vs-graphviz_sfdp first lever (<short>)

- Identified divergence: <one sentence>
- Fix: <one sentence>
- sfdp family median: 0.0915 -> <NEW>
- Worst graph: <name> <BEFORE> -> <AFTER>
- Simple-graph regressions: max delta = <X>
- Tests: <count> passed
```

If RESIDUAL:
- Write `ROUND_6_RESIDUAL.md` classifying the no-signal lever
- Recommend whether to try a second lever in Round 7 (if confidence
  high) or move to neato (if confidence low)

### Step 7: Per-round summary

Write `eval_output/algo_fidelity/round_6/SUMMARY.md`.

### Step 8: Update STATE.md

Append iteration log row. Update state. If COMMITTED + median <= 0.05:
mark sfdp converged, set `current_family: neato_stress` (Round 7
attacks neato).
If COMMITTED + median > 0.05: stay on sfdp for Round 7 with second
lever.
If RESIDUAL: increment flail_count_sfdp. If flail_count_sfdp < 2,
stay; else advance to neato.
</task>

<scope_constraints>
**HARD scope -- DO NOT TOUCH:**
- `dagua/render/**`
- `dagua/styles.py`
- `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`
- `dagua/layout/ops/sugiyama.py` (Round 3 owns)
- `dagua/layout/ops/pipelines/sugiyama.py` (Round 3 owns)
- `dagua/layout/ops/fmmm.py` (fdp work parked at flail=2; don't perturb)
- `dagua/layout/ops/pipelines/fmmm.py` (fdp parked)

**Allowed in Round 6:**
- `dagua/layout/ops/sfdp.py` (PRIMARY fix surface)
- `dagua/layout/ops/pipelines/sfdp.py` (caller / config)
- `dagua/layout/ops/state.py` ONLY if a SolveState field needs adding
- `eval_output/algo_fidelity/round_6/**` (new)
- `.project-context/research/sprint_algo_fidelity/**`
- `tests/test_layout/test_*.py` ONLY if a snapshot test needs updating

**Out of scope this round:**
- Other pipelines.
- Wholesale sfdp rewrite.
- run_benchmark.
</scope_constraints>

<default_follow_through_policy>
The Round 3 win was a single hyperparameter alignment that took the dot
median from 0.34 to 0.02. Look for the analogous one-knob alignment
here. The fact that defaults (steps, theta, p, seed) ALREADY match means
the gap is somewhere in init / cooling / scale -- a smaller search
space.

If you find that dagua already matches graphviz on every comparable
hyperparameter and the residual gap is just stochastic noise (median
varies between runs by ~0.05), classify as `numerical_residual:
stochastic_floor` and don't force a fix.
</default_follow_through_policy>

<completeness_contract>
1. **COMMITTED**: lever lands, median improved >= 0.02, commit on
   develop, SUMMARY, STATE updated.
2. **RESIDUAL**: ROUND_6_RESIDUAL.md, no commit, classification.
3. **BLOCKED**: ROUND_6_BLOCKED.md if architectural blocker.

NEVER commit without measuring. NEVER weaken pytest assertions.
</completeness_contract>

<verification_loop>
- After every code change: `pytest tests/test_layout/ -x --tb=short -q -k "sfdp"`
- Final live_compare runs cleanly (run twice if stochastic)
- `git diff --stat HEAD~0` before commit shows only allowed scope
</verification_loop>

<missing_context_gating>
ABORT before edits if:
- Graphviz source clone missing key files
- dagua sfdp ops file structure has changed substantially
- live_compare for sfdp is non-deterministic past 0.01 between runs
  (run multiple times and take median)

Write ROUND_6_BLOCKED.md and stop.
</missing_context_gating>

<action_safety>
- ONE commit on develop only IF measured improvement.
- No force-push, branch creation, rebase, or tag.
- Never delete eval_output files.
</action_safety>

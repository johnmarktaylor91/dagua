<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`. ONE working branch -- DO NOT create a new branch.

Round 7 of the algo_fidelity sprint. Read these in order:
1. `.project-context/research/sprint_algo_fidelity/algo_fidelity_STATE.md`
2. `eval_output/algo_fidelity/round_3/SUMMARY.md` (the dot-family WIN)
3. `eval_output/algo_fidelity/round_6/SUMMARY.md` (sfdp residual context)

## Round 7 target: neato family

Round 1 baseline:
- classic_stress_maj vs graphviz_neato: median 0.0353, worst inception_block 0.3817
- classic_classical_mds vs graphviz_neato: median 0.0455, worst petersen_10 0.3326

**Median is already at/under the 0.05 stop criterion** for both pairings,
so this round is more about VALIDATION + polishing outliers than the
big alignment moves we did for dot/fdp/sfdp.

The worst-case threshold of 0.15 IS likely failing on outlier graphs
(petersen_10, inception_block, edge_label_braid -- graphs with cycles
or dense connectivity).

## Graphviz neato source -- READ THIS FIRST

`/home/jtaylor/projects/_references/graphviz/lib/neatogen/`:

### Default mode and init

graphviz neato has multiple modes (MODE_MAJOR, MODE_KK, MODE_SGD,
MODE_HIER, MODE_IPSEP). Default is MODE_MAJOR (stress majorization).

`neatoinit.c:637`: `int mode = MODE_MAJOR;` -- default
`neatoinit.c:1092`: `int init = checkStart(g, nv, mode == MODE_HIER ? INIT_SELF : INIT_RANDOM);`

So the default neato:
- mode = MODE_MAJOR (stress majorization)
- init = INIT_RANDOM (random initial layout, NOT smart MDS init)

But there's also a `smart_ini` option in `constrained_majorization.c:64`
that's used for constrained mode. Default `neato -Gmode=major` uses
INIT_RANDOM, NOT smart MDS init.

### Required reading

- `neatoinit.c` lines 925-1100 (init dispatch + checkStart)
- `stress.c` lines 200-300 (stress majorization main loop)
- `kkutils.c` (only relevant for MODE_KK comparison)
- `constrained_majorization.c` (only relevant for IPSEP / smart init)

### Useful greps

```bash
# Find default values:
grep -nE "Maxiter|N_minIter|tol|epsilon|epsi" \
  /home/jtaylor/projects/_references/graphviz/lib/neatogen/stress.c

# Find init function:
grep -nE "initLayout|initRandom|initial_position" \
  /home/jtaylor/projects/_references/graphviz/lib/neatogen/*.c

# Find stress weight expression:
grep -nE "1\.0\s*/\s*\(d|w_ij\s*=|wij\s*=" \
  /home/jtaylor/projects/_references/graphviz/lib/neatogen/stress.c
```

## Dagua surface

- `dagua/layout/ops/pipelines/stress_majorization.py` -- defaults: iterations=200
- `dagua/layout/ops/stress.py` -- ops (PrepareStressMajorizationState,
  InitializeStressMajorizationPositions, SmacofStep, etc.)
- `dagua/layout/ops/pipelines/classical_mds.py` -- pure eigendecomp pipeline
- `dagua/layout/ops/distance.py` -- ClassicalMDSDistanceMatrix
- `dagua/layout/ops/embed.py` -- ClassicalMDSComputeEmbedding

Key questions to investigate:
1. Does dagua stress_maj use the **same stress weight formula** as graphviz
   (`w_ij = 1 / d_ij^2` vs other power)?
2. Does dagua stress_maj iterate enough? graphviz default Maxiter for stress
   is in stress.c -- find it. dagua uses 200; might need 1000+.
3. Does dagua classical_mds match graphviz's `smart_ini_x.c` MDS, or differ
   in the eigendecomp method (full eig vs power iteration)?
4. **Init alignment**: graphviz neato MODE_MAJOR uses INIT_RANDOM.
   Dagua stress_maj likely uses classical MDS init (deterministic),
   which is actually BETTER but produces different shapes.

## What to do

### Step 1: Live baseline (10 min)

```
cd /home/jtaylor/projects/dagua
python scripts/algo_fidelity_live_compare.py classic_stress_maj graphviz_neato \
    --output-dir eval_output/algo_fidelity/round_7/baseline_stress
python scripts/algo_fidelity_live_compare.py classic_classical_mds graphviz_neato \
    --output-dir eval_output/algo_fidelity/round_7/baseline_mds
```

Confirm:
- stress_maj baseline median should be near Round 1's 0.0353
- classical_mds baseline median should be near Round 1's 0.0455
- worst graphs (petersen_10, inception_block, edge_label_braid) should
  still be the outliers

### Step 2: Decision tree

**Case A: both medians already <= 0.05 AND worst graphs <= 0.15**
- neato family already converged. Write `ROUND_7_VALIDATED.md`
  documenting the baseline, no code changes, no commit.
- Update STATE.md: mark neato family as CONVERGED, advance
  `current_family: phase_2` (less-important families).

**Case B: medians <= 0.05 BUT worst graphs > 0.15**
- Inspect the outlier graphs (petersen_10, inception_block,
  edge_label_braid). What's the structural pattern? (cycles?
  high connectivity? Specific node count?)
- Identify if there's a focused fix that targets these. Most likely
  candidates:
  - More iterations (graphviz may iterate longer to convergence on
    cyclic graphs)
  - Different init for cyclic graphs
  - A specific stress weight tweak
- If high-confidence fix: apply, measure, commit per usual contract
- If no high-confidence fix: write `ROUND_7_OUTLIER_RESIDUAL.md`,
  classify as `numerical_residual: cyclic_graph_init_basin` or
  similar, advance to phase 2

**Case C: medians > 0.05**
- Live baseline diverged from Round 1 cache (node-size drift). Treat
  as a regular fidelity round: identify lever, apply, measure, commit.

### Step 3: Tests (always run)

```
pytest tests/test_layout/ -x --tb=short -q -k "stress or mds" 2>&1 | tail -30
```

### Step 4: Per-round summary

Write `eval_output/algo_fidelity/round_7/SUMMARY.md` with the case
outcome (A/B/C) and metrics.

### Step 5: Update STATE.md

Append iteration log row. Update state. If Case A or Case B no-fix:
mark neato CONVERGED with classification, advance to phase_2 or to
the parked sfdp/fdp families if you want to take another shot.

## Strategy note

After Round 7 we have these statuses:
- dot: CONVERGED (median 0.019)
- fdp: PARKED at flail=2 (architectural - random init / FR solver)
- sfdp: PARKED at flail=1 (architectural - sequential vs synchronous)
- neato: TBD this round

After Round 7, options for Round 8:
- Phase 2 families (davidson_harel, drl, graphopt, neulay, tsnet, fa2)
- OR another shot at sfdp/fdp with deeper levers
- OR final summary writeup

Recommend in your STATE.md update which way to go based on what
this round revealed.
</task>

<scope_constraints>
**HARD scope -- DO NOT TOUCH:**
- `dagua/render/**`
- `dagua/styles.py`
- `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`
- `dagua/layout/ops/sugiyama.py` / `pipelines/sugiyama.py` (Round 3 owns)
- `dagua/layout/ops/fmmm.py` / `pipelines/fmmm.py` (parked)
- `dagua/layout/ops/sfdp.py` / `pipelines/sfdp.py` (parked)

**Allowed in Round 7:**
- `dagua/layout/ops/stress.py` (PRIMARY -- stress majorization)
- `dagua/layout/ops/pipelines/stress_majorization.py` (caller)
- `dagua/layout/ops/distance.py` / `embed.py` / `postprocess.py` for MDS
- `dagua/layout/ops/pipelines/classical_mds.py`
- `dagua/layout/ops/state.py` ONLY if a SolveState field needs adding
- `eval_output/algo_fidelity/round_7/**` (new)
- `.project-context/research/sprint_algo_fidelity/**`
- `tests/test_layout/test_*.py` ONLY if a snapshot test needs updating
</scope_constraints>

<default_follow_through_policy>
Validation-first stance. If neato is already converged at the family
median level, document and move on rather than forcing a fix. The
remaining unconverged graphs (petersen_10, inception_block, edge_label_braid)
are graphs with cycles where stochastic init produces different basins;
this can be a `numerical_residual: cyclic_graph_init_basin` rather
than something to fix.

If the median IS over 0.05, treat like Round 3 and look for the
single hyperparameter alignment.
</default_follow_through_policy>

<completeness_contract>
1. **VALIDATED (Case A)**: medians and worst-cases meet criteria.
   Write ROUND_7_VALIDATED.md, no commit, advance.
2. **COMMITTED (Case B fix or Case C fix)**: lever lands, measured
   improvement, commit on develop, SUMMARY, STATE updated.
3. **OUTLIER_RESIDUAL (Case B no-fix)**: ROUND_7_OUTLIER_RESIDUAL.md
   with classification, no commit, advance.
4. **BLOCKED**: ROUND_7_BLOCKED.md if architectural blocker.

NEVER commit without measuring. NEVER weaken pytest assertions.
</completeness_contract>

<verification_loop>
- After every code change: `pytest tests/test_layout/ -x --tb=short -q -k "stress or mds"`
- Final live_compare runs cleanly
- `git diff --stat HEAD~0` before commit shows only allowed scope
</verification_loop>

<missing_context_gating>
ABORT before edits if:
- Graphviz source clone missing key files
- dagua stress/mds ops file structure has changed substantially
- live_compare for stress_maj or classical_mds is non-deterministic
  past 0.01 between runs

Write ROUND_7_BLOCKED.md and stop.
</missing_context_gating>

<action_safety>
- ONE commit on develop only IF measured improvement.
- No force-push, branch creation, rebase, or tag.
- Never delete eval_output files.
</action_safety>

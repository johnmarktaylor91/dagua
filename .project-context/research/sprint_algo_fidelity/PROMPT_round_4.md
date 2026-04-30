<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`. ONE working branch -- DO NOT create a new branch.

Round 4 of the algo_fidelity sprint. Read these in order before doing anything:
1. `.project-context/research/sprint_algo_fidelity/algo_fidelity_STATE.md`
2. `.project-context/research/sprint_algo_fidelity/PROMPT_round_3.md`
3. `eval_output/algo_fidelity/round_3/SUMMARY.md`

## Context: dot family converged

Round 3 landed a 20x reduction in dot-family RMSD with a single
hyperparameter alignment: `dagua/layout/ops/pipelines/sugiyama.py`
defaults moved from unit (1.0) spacing to graphviz dot's point-unit
spacing (rank_sep=72.0, node_sep=18.0). The width-aware Brandes-Köpf
spacing logic produced dot-like proportions at the right scale.

**Hypothesis carried into Round 4**: similar default-mismatch
hyperparameter issues may explain the fdp family's uniform >0.15 RMSD
floor across all 21 graphs.

## Round 4 target: fdp family

Round 1 baseline (cached, but pattern still holds):
- fmmm-vs-fdp median RMSD: 0.2918
- ALL 21 graphs were above 0.15
- worst: disconnected_label_cycle_collage (0.4169)

The "uniform floor" pattern is the tell. In contrast to dot's "perfect
on simple, cliffs at medium" pattern (which was proportional spacing),
fdp's "uniform floor" suggests a different systematic divergence --
possibly initial seeding, gravity center, scale, or convergence
threshold.

Files involved:
- `dagua/layout/ops/pipelines/fmmm.py` -- pipeline wiring
- `dagua/layout/ops/fmmm.py` -- ops (multilevel hierarchy + force-directed solver)
- `dagua/eval/competitors/graphviz_competitor.py:443-446` -- graphviz_fdp adapter

graphviz fdp's algorithm:
- Uses **FMMM** under the hood (Walshaw-style multilevel + spring-electrical
  on each level, with grid-based repulsion approximation)
- Defaults: K=0.3 (spring constant), grid resolution scales with sqrt(N)
- Stochastic: depends on a seed

OGDF FMMM (the academic reference) and graphviz fdp's FMMM have
similar structure but DIFFERENT default hyperparameters (gravity, edge
length target, smoothing schedule).

## What to do

### Step 1: Live baseline for fmmm vs graphviz_fdp (10 min)

```
cd /home/jtaylor/projects/dagua
python scripts/algo_fidelity_live_compare.py classic_fmmm graphviz_fdp \
    --output-dir eval_output/algo_fidelity/round_4/baseline_run1
```

FMMM is stochastic. Check determinism:
```
python scripts/algo_fidelity_live_compare.py classic_fmmm graphviz_fdp \
    --output-dir eval_output/algo_fidelity/round_4/baseline_run2
```

If non-deterministic on a fixed seed, fix the seeding first OR run
3-5 seeds and use median. Don't iterate against a noisy baseline.

Confirm the live baseline is in the same ballpark as Round 1's cached
0.2918 family median (not necessarily identical -- live run may have
different node sizes -- but the pattern of "all graphs above ~0.15"
should persist).

### Step 2: Inspect the fmmm surface (15 min)

Read these files and identify potential lever sources:
- `dagua/layout/ops/pipelines/fmmm.py` -- look at `build_fmmm_pipeline`
  defaults (steps=200, force_model="ogdf_new") and any other tunables
- `dagua/layout/ops/fmmm.py` -- look at:
  - Initial coordinate seeding (random? deterministic? from spectral?)
  - Spring constant / desired edge length defaults
  - Gravity / centering term
  - Multilevel coarsening parameters
  - Convergence / step count distribution across levels
  - Any "scale" or "unit" defaults like the sugiyama issue

Compare to graphviz fdp defaults. Web search if needed:
- "graphviz fdp default K spring constant"
- "OGDF FMMM vs graphviz fdp parameters"
- Look at the graphviz source for fdp's FMMM bindings if available

### Step 3: Identify the dominant lever (10 min)

Possible levers, ranked by likelihood given the "uniform floor" pattern:
   a. **Scale / initial-coordinate range mismatch** -- if dagua FMMM
      starts coords in [-1, 1] but fdp starts in [-100, 100], the
      relative gravity/repulsion balance differs (gravity scales with
      distance from origin).
   b. **Gravity / central-attractor mismatch** -- fdp may apply
      stronger central gravity than dagua, producing more compact
      layouts.
   c. **Spring constant K mismatch** -- fdp's default K is 0.3 (in
      its native units). If dagua uses 1.0 or some other value, edge
      lengths differ proportionally.
   d. **Random seed initialization** -- if dagua uses a fixed
      deterministic init while fdp uses uniform-random with seed,
      shape patterns differ even after Procrustes alignment.

Run a quick experiment: for the simplest large fdp graph (probably
`linear_3layer_mlp` or similar with low expected RMSD), inspect the
coordinate distribution from the cached graphviz_fdp positions vs
the live dagua FMMM positions. Are scales / centroids / spreads
comparable?

### Step 4: Apply ONE focused lever (15-30 min)

Same rules as Round 3:
- < ~80 lines net change
- Smallest plausible lever from Step 3
- No wholesale algorithm replacement
- High-confidence only -- if no high-confidence lever, write residual

### Step 5: Measure

```
python scripts/algo_fidelity_live_compare.py classic_fmmm graphviz_fdp \
    --output-dir eval_output/algo_fidelity/round_4/post_fix
```

COMMIT criterion: family median improved by >= 0.05 (because uniform
floor means ALL graphs should improve). No simple-graph regressions.

### Step 6: Tests

```
pytest tests/test_layout/ -x --tb=short -q -k "fmmm" 2>&1 | tail -30
pytest tests/test_layout/ -x --tb=short -q 2>&1 | tail -30
```

### Step 7: Commit OR document residual

Same as Round 3:
- COMMITTED: `feat(fidelity): round 4 -- fmmm-vs-fdp first lever (<short>)`
  with same commit-message template as Round 3.
- RESIDUAL: write `ROUND_4_RESIDUAL.md` with classification.
- BLOCKED: write `ROUND_4_BLOCKED.md` if hard dependency missing.

### Step 8: Per-round summary

Write `eval_output/algo_fidelity/round_4/SUMMARY.md`.

### Step 9: Update STATE.md

Append iteration log row. Update `state:` and `current_round:`.
If COMMITTED + family converged (median <= 0.05): set
`current_family: sfdp` (Round 5 attacks sfdp family).
If RESIDUAL: set `flail_count_fdp: 1` and stay on fdp for next round
attempt with a different lever, OR if you're confident no small
lever exists: advance to sfdp.
</task>

<scope_constraints>
**HARD scope -- DO NOT TOUCH:**
- `dagua/render/**`
- `dagua/styles.py`
- `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`
- `dagua/layout/ops/sugiyama.py` (Round 3 already touched this; leave it)
- `dagua/layout/ops/pipelines/sugiyama.py` (same)

**Allowed in Round 4:**
- `dagua/layout/ops/fmmm.py` (PRIMARY fix surface)
- `dagua/layout/ops/pipelines/fmmm.py` (caller / config)
- `dagua/layout/ops/state.py` ONLY if a SolveState field needs adding
- `eval_output/algo_fidelity/round_4/**` (new)
- `.project-context/research/sprint_algo_fidelity/**`
- `tests/test_layout/test_*.py` ONLY if a snapshot test needs updating

**Out of scope this round:**
- Any other pipelines (sugiyama, sfdp, stress_maj, classical_mds).
- Wholesale FMMM algorithm replacement.
- Touching the graphviz competitor adapter.
- Running the full benchmark.
</scope_constraints>

<default_follow_through_policy>
Use the same playbook that worked in Round 3. Smallest plausible lever
with strongest mechanical evidence. If multiple small levers exist,
pick the one most analogous to Round 3's success (default
hyperparameter alignment with graphviz's actual defaults).

Diagnosis-only is a valid Round 4 outcome -- a clean residual is
better than a speculative fix.
</default_follow_through_policy>

<completeness_contract>
Same as Round 3:
1. **COMMITTED**: lever lands, median improved by >= 0.05, commit on
   develop with `feat(fidelity): round 4 --` prefix, SUMMARY written,
   STATE updated.
2. **RESIDUAL**: ROUND_4_RESIDUAL.md written, classification recorded,
   no commit, SUMMARY written, STATE updated, flail_count_fdp
   incremented.
3. **BLOCKED**: ROUND_4_BLOCKED.md if hard dependency missing.

NEVER commit without measuring. NEVER weaken pytest assertions.
NEVER touch out-of-scope files.
</completeness_contract>

<verification_loop>
- After every code change: `pytest tests/test_layout/ -x --tb=short -q -k "fmmm"`
- Final live_compare must run cleanly
- `git diff --stat HEAD~0` before commit must show only allowed scope
- `git log --oneline -1` matches expected prefix if committed
</verification_loop>

<missing_context_gating>
ABORT before edits if:
- live_compare for fmmm-vs-fdp is non-deterministic and you can't
  stabilize via seed
- The fmmm op file structure has been refactored away from what's
  described above
- Cached graphviz_fdp positions for the test graphs are missing

Write ROUND_4_BLOCKED.md and stop.
</missing_context_gating>

<action_safety>
- ONE commit on develop only IF COMMIT criterion met.
- No force-push, branch creation, rebase, or tag.
- Never delete eval_output files.
</action_safety>

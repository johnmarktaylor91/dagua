<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`. ONE working branch -- DO NOT create a new branch.

Round 5 of the algo_fidelity sprint. Read these in order:
1. `.project-context/research/sprint_algo_fidelity/algo_fidelity_STATE.md`
2. `.project-context/research/sprint_algo_fidelity/ROUND_4_RESIDUAL.md`
3. `eval_output/algo_fidelity/round_4/SUMMARY.md`

## Round 5 target: fdp family, second attempt

Round 4 confirmed:
- Live `classic_fmmm` vs `graphviz_fdp` median RMSD = 0.247 (deterministic)
- 20 of 21 graphs above 0.15 (uniform floor pattern persists)
- K=0.3 hyperparameter alignment (Round 3-style approach) regressed slightly,
  was reverted, no commit
- Diagnosis: dagua FMMM uses OGDF logarithmic attraction; graphviz fdp uses
  Fruchterman-Reingold force law with K²/d² (old) or K²/d³ (new) repulsion.
  This is the dominant remaining gap.

Round 5 attacks the force-law mismatch directly, using **graphviz C
source as ground truth** rather than papers or web search.

## Graphviz fdp source -- READ THIS FIRST

The graphviz source is cloned locally at:

    /home/jtaylor/projects/_references/graphviz

For Round 5, you MUST read these files end-to-end before proposing
the fix. They define the algorithm dagua needs to match:

### Force laws (REQUIRED reading)

`/home/jtaylor/projects/_references/graphviz/lib/fdpgen/tlayout.c`:

- **Attractive force** at line 287, function `applyAttr`:
  ```c
  if (T_useNew)
    force = ED_factor(e) * (dist - ED_dist(e)) / dist;
  else
    force = ED_factor(e) * dist / ED_dist(e);
  ```
  Comments at line 283:
    `Attractive force = weight × (d × d) ÷ K`
    `   or     force = (d - L(e)) × weight(e)`

- **Repulsive force** at line 195, function `doRep`:
  ```c
  if (T_useNew) {
    dist = sqrt(dist2);
    force = T_K * T_K / (dist * dist2);  // K^2 / d^3
  } else
    force = T_K * T_K / dist2;  // K^2 / d^2
  if (IS_PORT(p) && IS_PORT(q))
    force *= 10.0;
  ```
  Comment at line 213: `repulsive force = K × K ÷ d or K × K ÷ d × d`

### Defaults (REQUIRED reading)

`/home/jtaylor/projects/_references/graphviz/lib/fdpgen/tlayout.c:97-105`:
```c
#define DFLT_maxIters 600
#define DFLT_K 0.3      // edge length unit, in INCHES (graphviz unit)
#define DFLT_Cell 0.0
#define DFLT_seed 1
#define DFLT_smode INIT_RANDOM   // random init
```

### Other parts of the algorithm (READ to determine if they matter)

- `lib/fdpgen/grid.c` -- grid-based repulsion approximation (only neighbors
  in same/adjacent grid cells contribute repulsion; cell size = T_Cell).
  T_Cell defaults to 0 which means full O(N^2) repulsion.
- `lib/fdpgen/tlayout.c:updatePos` (around line 315) -- temperature
  cooling: positions clamped by `temp` per step.
- `lib/fdpgen/tlayout.c:fdp_tLayout` (search for it) -- main solver loop
  with adaptive `temp` cooling.
- `lib/fdpgen/xlayout.c` -- post-layout overlap expansion phase. May
  contribute to discrepancy on graphs with text labels.
- `lib/fdpgen/fdpinit.c` -- initialization function. RANDOM init by default.

## What to do

### Step 1: Live baseline (5 min)

Confirm baseline is still 0.247 (Round 4 numbers):
```
cd /home/jtaylor/projects/dagua
python scripts/algo_fidelity_live_compare.py classic_fmmm graphviz_fdp \
    --output-dir eval_output/algo_fidelity/round_5/baseline
```

Skip determinism re-check (Round 4 already confirmed deterministic).

### Step 2: Inspect dagua FMMM force law (15 min)

Read these files to find dagua's existing force-law math:
- `dagua/layout/ops/pipelines/fmmm.py` (current force_model="ogdf_new" default)
- `dagua/layout/ops/fmmm.py` (the ops; find the force computation)
- `dagua/layout/ops/_native_shared.py` if force law is shared

Identify:
- The current attraction formula (logarithmic per Round 4 diagnosis -- verify)
- The current repulsion formula
- The current edge-length / spring-constant variable
- Where the force_model parameter is consumed (the dispatch point)

### Step 3: Add a `graphviz_fdp` force model (30-60 min)

Add a new force_model option (e.g. `"graphviz_fdp"` or `"fr_classic"`)
that exactly replicates graphviz's tlayout.c math:

- Repulsion: `force = K^2 / d^2` (the "old" form -- simpler; use this
  as default; the "new" form K^2/d^3 is an option).
- Attraction: `force = weight * d / L_e` (the "old" form -- simpler;
  matches both graphviz comment and FR paper).
- Apply force vector: `disp += xdelta * force` (per the source).
- Edge length L_e default: K=0.3 inches = 21.6 points (the value
  Round 4 tried but didn't apply correctly because force law was wrong).

Implementation guidance:
- Keep the existing OGDF force model untouched (don't remove
  `force_model="ogdf_new"`).
- Wire the new force model through `build_fmmm_pipeline(force_model="graphviz_fdp", ...)`.
- The classic_fmmm competitor adapter should switch to the new
  default (`force_model="graphviz_fdp"`) IFF the change improves
  RMSD. Test this last.

Acceptable change scope: ~50-150 lines net (new force model is a
proper additive change, not a wholesale replacement). If you find the
implementation requires deeper architectural changes (e.g., dagua's
FMMM has tightly-coupled force/init/temperature code), STOP and
write `ROUND_5_BLOCKED.md` -- we'd need a Round 6 to do it cleanly.

### Step 4: Measure incrementally

Test by adjusting **classic_fmmm's default to use the new force model**:

```
python scripts/algo_fidelity_live_compare.py classic_fmmm graphviz_fdp \
    --output-dir eval_output/algo_fidelity/round_5/post_fix
```

Compare to baseline (0.247 median):
- COMMIT criterion: median improved by >= 0.05 OR all P0 graphs
  improved by >= 0.05
- Watch for regressions on the simpler graphs (linear_3layer_mlp
  baseline 0.104, parallel_multiedge_bundle, nested_shallow_enc_dec)
- Tests must pass: `pytest tests/test_layout/ -x --tb=short -q -k "fmmm"`

If improvement is partial (median 0.247 -> 0.18 say), still commit -- it's
substantial progress, even if dot-style "1 round to converge" isn't
achievable here.

If no improvement OR regression: revert, write ROUND_5_RESIDUAL.md
classifying as `attempted_lever_no_signal: graphviz_fdp_force_model`,
recommend Round 6 try **random initialization alignment** (graphviz uses
INIT_RANDOM with rectangle/ellipse, dagua uses spectral or coarsest-init).

### Step 5: Commit + summary

If COMMIT criterion met:
```
feat(fidelity): round 5 -- fmmm-vs-fdp force law (graphviz FR)

- Identified divergence: dagua FMMM used OGDF logarithmic attraction;
  graphviz fdp uses FR force law (rep K^2/d^2, attr weight*d/L_e per
  lib/fdpgen/tlayout.c:195-216).
- Fix: added force_model="graphviz_fdp" to FMMM pipeline matching
  graphviz tlayout.c force law verbatim. classic_fmmm default switched
  to this model.
- fdp family median: 0.2475 -> <NEW>
- center_port_backedge_hub (worst): 0.4401 -> <NEW>
- Simple-graph regressions: max delta = <X>
- Tests: <count> passed
```

Write `eval_output/algo_fidelity/round_5/SUMMARY.md` per template.

### Step 6: Update STATE.md

Append iteration log row. Update state. If COMMITTED + family converged
(median <= 0.10): set `current_family: sfdp`. If COMMITTED + partial
improvement: stay on fdp for Round 6 (different lever -- random init).
If RESIDUAL: increment flail_count_fdp, advance to sfdp.

Anti-flail: flail_count_fdp is currently 1. After Round 5, if no
commit, it becomes 2. After Round 6 if no commit, it hits 3 and we
mark fdp as `principled_residual: needs_full_FR_solver_with_random_init`
and move on for good.
</task>

<scope_constraints>
**HARD scope -- DO NOT TOUCH:**
- `dagua/render/**`
- `dagua/styles.py`
- `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`
- `dagua/layout/ops/sugiyama.py` (Round 3 owns this)
- `dagua/layout/ops/pipelines/sugiyama.py` (Round 3 owns this)

**Allowed in Round 5:**
- `dagua/layout/ops/fmmm.py` (PRIMARY fix surface)
- `dagua/layout/ops/pipelines/fmmm.py` (caller / config)
- `dagua/layout/ops/state.py` ONLY if a SolveState field needs adding
- `dagua/layout/ops/_native_shared.py` ONLY if force-model dispatch
  lives there
- `eval_output/algo_fidelity/round_5/**` (new)
- `.project-context/research/sprint_algo_fidelity/**`
- `tests/test_layout/test_*.py` ONLY if a snapshot test needs updating

**Out of scope this round:**
- Other pipelines (sugiyama, sfdp, stress_maj, classical_mds).
- Wholesale FMMM rewrite (just add a force model, don't replace the
  multilevel framework).
- Touching graphviz_competitor adapter or run_benchmark.
- Replicating graphviz fdp's overlap expansion (`xlayout.c`) --
  that's potentially Round 6+.
- Replicating graphviz fdp's random init (`fdpinit.c`) -- that's
  potentially Round 6+.
</scope_constraints>

<default_follow_through_policy>
Read graphviz tlayout.c FIRST. Confirm the force-law equations against
the source before writing any dagua code. The Round 3 win came from
matching graphviz's actual numbers; Round 5 should use the same
discipline -- equations from the source, not from memory or papers.

If you find the comments in tlayout.c contradict each other (e.g., the
"new" vs "old" formulas), default to the code over the comment, and
default to T_useNew=1 (which is graphviz's default unless the input
graph has overridden it).
</default_follow_through_policy>

<completeness_contract>
1. **COMMITTED**: force model added, classic_fmmm default switched,
   measured improvement >= 0.05 median or all P0 graphs >= 0.05,
   commit on develop with feat(fidelity): round 5 prefix, SUMMARY,
   STATE updated.
2. **PARTIAL_COMMIT**: smaller improvement (<0.05 median but >0)
   committed because force-law alignment is substantively correct
   regardless of the exact RMSD reduction, with documented
   explanation in commit.
3. **RESIDUAL**: ROUND_5_RESIDUAL.md, no commit, flail_count_fdp=2,
   advance to sfdp.
4. **BLOCKED**: ROUND_5_BLOCKED.md if architectural blocker found.

NEVER commit without measuring. NEVER weaken pytest assertions.
</completeness_contract>

<verification_loop>
- After every code change: `pytest tests/test_layout/ -x --tb=short -q -k "fmmm"`
- Final live_compare runs cleanly
- `git diff --stat HEAD~0` before commit shows only allowed scope
</verification_loop>

<missing_context_gating>
ABORT before edits if:
- The graphviz source clone at `/home/jtaylor/projects/_references/graphviz`
  is missing or incomplete (verify `lib/fdpgen/tlayout.c` exists)
- dagua/layout/ops/fmmm.py force-model dispatch is structured
  fundamentally differently from what's described
- live_compare for fmmm-vs-fdp is non-deterministic

Write ROUND_5_BLOCKED.md and stop.
</missing_context_gating>

<action_safety>
- ONE commit on develop only IF measured improvement.
- No force-push, branch creation, rebase, or tag.
- Never delete eval_output files.
</action_safety>

<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`. ONE working branch -- DO NOT create a new branch.

Round 3 of the algo_fidelity sprint. Read these in order before doing anything:
1. `.project-context/research/sprint_algo_fidelity/algo_fidelity_STATE.md`
2. `.project-context/research/sprint_algo_fidelity/ROUND_2_DIAGNOSIS.md`
3. `.project-context/research/sprint_algo_fidelity/ROUND_2_BLOCKED.md`

## Round 3 ground rules (changes from Round 2)

1. **The Round 1 cache is OBSOLETE** -- node sizing has drifted since the
   mega-run, and the cache cannot be replayed live. Stop comparing to it.
   Treat `scripts/algo_fidelity_live_compare.py` output as the new
   ground truth. Round 2 live baseline values (from
   `eval_output/algo_fidelity/round_2/baseline/live_rmsd.csv`) are the
   "before" numbers for Round 3:
   - dot family median: 0.341942
   - mixed_width_labels: 0.404615
   - shape_and_routing_matrix: 0.456349
   - small_label_storm: 0.485187
   Re-run live_compare ONCE at start of Round 3 to confirm
   determinism (same input -> same output). If it's not deterministic
   on a fixed seed, that itself is a bug to fix first.

2. **Diagnosis confirms**: dagua sugiyama's layer assignment and
   in-layer ordering already match graphviz dot on the 3 diagnostic
   graphs. The remaining gap is in **coordinate assignment** (dagua's
   Brandes-Köpf at `dagua/layout/ops/sugiyama.py:724` and width-aware
   `_min_separation` at `dagua/layout/ops/sugiyama.py:1191` vs dot's
   network-simplex coordinate problem).

3. **Round 3 = ONE small focused fix on coordinate assignment.**
   No wholesale algorithm replacement. If the fix lands, commit. If it
   doesn't (or no high-confidence small fix exists), DO NOT spend more
   than ~30 min trying. Mark dot as `principled_residual:
   needs_network_simplex_coords` and move on. Round 4 will attack the
   next family (fdp, which is uniformly above 0.15 -- likely a
   systematic-offset bug, not an architectural floor).

## What to do

### Step 1: Confirm baseline determinism (5 min)

```
cd /home/jtaylor/projects/dagua
python scripts/algo_fidelity_live_compare.py classic_sugiyama graphviz_dot \
    --output-dir eval_output/algo_fidelity/round_3/baseline_run1
python scripts/algo_fidelity_live_compare.py classic_sugiyama graphviz_dot \
    --output-dir eval_output/algo_fidelity/round_3/baseline_run2
diff <(sort eval_output/algo_fidelity/round_3/baseline_run1/live_rmsd.csv) \
     <(sort eval_output/algo_fidelity/round_3/baseline_run2/live_rmsd.csv)
```
If the two runs differ on any RMSD value by more than 1e-6, the live
comparator is non-deterministic. Either set a seed, or document the
non-determinism and use the median of 3 runs as the baseline. Do NOT
proceed past this step until baseline is stable.

### Step 2: Inspect the coordinate-assignment surface (15 min)

Read these specifically (not the full ~2000-line sugiyama.py):
- `dagua/layout/ops/sugiyama.py:662` -- `_coordinate_assignment`
  (entry point)
- `dagua/layout/ops/sugiyama.py:724` -- `_brandes_koepf_x_positions`
  (Brandes-Köpf 4-pass compaction)
- `dagua/layout/ops/sugiyama.py:1191` -- `_min_separation` /
  width-aware spacing
- `dagua/layout/ops/pipelines/sugiyama.py` -- the pipeline wiring,
  especially what hyperparameters are passed (rank_sep, node_sep,
  barycenter_passes)

Then compare to graphviz dot's algorithm. Reference (web search if
needed):
- Gansner, Koutsofios, North, Vo (1993) "A Technique for Drawing
  Directed Graphs" -- Section 4.2 (Position Assignment)
- Brandes & Köpf (2002) "Fast and Simple Horizontal Coordinate
  Assignment" -- standard 4-pass routine, but dot uses a network-simplex
  formulation instead

Look for **small, high-confidence levers** of these types:
   a. **Bug fix**: an off-by-one, wrong sign, or wrong tie-breaker in
      the existing Brandes-Köpf code.
   b. **Hyperparameter mismatch**: a default that doesn't match dot's
      defaults (e.g., `rank_sep=1.0` vs dot's `ranksep=0.5`, or
      `node_sep=1.0` vs dot's `nodesep=0.25`). NOTE: scale is absorbed
      by Procrustes; only ratios and proportions matter.
   c. **Missing feature**: dot does X but dagua doesn't (e.g.,
      "balance" pass that averages the four BK candidates is one knob
      that affects shape, not just scale).
   d. **Order-of-operations**: dagua's BK passes might be in a
      different order than the canonical paper, or dot's variant.

### Step 3: Pick ONE lever and apply (15-30 min)

Apply the SMALLEST plausible lever from Step 2. Examples of acceptable
fixes:
- A 5-line tweak to the BK balance pass.
- Switching from "leftmost" to "balanced" candidate selection.
- Fixing a width-aware spacing computation that was using node-edge
  distance instead of node-center distance.
- A hyperparameter alignment.

NOT acceptable for Round 3:
- Implementing network-simplex coordinate assignment from scratch.
- Replacing the Brandes-Köpf module wholesale.
- Re-architecting `_coordinate_assignment`.

### Step 4: Measure (5 min)

```
python scripts/algo_fidelity_live_compare.py classic_sugiyama graphviz_dot \
    --output-dir eval_output/algo_fidelity/round_3/post_fix
```

Compare against `eval_output/algo_fidelity/round_3/baseline_run1/live_rmsd.csv`:
- COMMIT criterion: median improved by >= 0.02 OR all 3 diagnostic
  graphs (mixed_width_labels, shape_and_routing_matrix,
  small_label_storm) improved by >= 0.05 each. AND no simple-graph
  regressions (linear_3layer_mlp, parallel_multiedge_bundle,
  nested_shallow_enc_dec; their RMSDs must stay <= live_baseline + 0.01).
- REVERT criterion: median got worse OR a simple graph regressed by
  more than 0.01.
- AMBIGUOUS criterion (small or no improvement): revert + classify as
  `attempted_lever_no_signal`.

### Step 5: Tests

```
pytest tests/test_layout/ -x --tb=short -q 2>&1 | tail -40
```
The full layout test suite must pass. If a snapshot test fails because
sugiyama output changed, evaluate: is the snapshot a frozen-from-old-state
expectation? If yes, update with rationale in commit message. If the
snapshot encodes a CORRECTNESS property (e.g., "all nodes are layered
correctly"), the fix is wrong and must be reverted.

### Step 6: Commit OR document residual

**If COMMIT criterion met:**
```
feat(fidelity): round 3 -- sugiyama-vs-dot first lever (<short fix description>)

- Identified divergence: <one sentence>
- Fix: <one sentence>
- dot family median: 0.3419 -> <NEW>
- mixed_width_labels: 0.4046 -> <NEW>
- shape_and_routing_matrix: 0.4564 -> <NEW>
- small_label_storm: 0.4852 -> <NEW>
- Simple-graph regressions: max delta = <X>
- Tests: <count> passed
```

**If REVERTED or ATTEMPTED_LEVER_NO_SIGNAL or NO_LEVER_FOUND:**
- DO NOT commit code changes (live_compare CSV outputs may stay
  uncommitted as artifacts).
- Write `.project-context/research/sprint_algo_fidelity/ROUND_3_RESIDUAL.md`
  with:
  - What lever was tried (or which were considered and rejected)
  - Why it didn't land
  - Classification: `principled_residual: needs_network_simplex_coords`
    (since BK is a different algorithm family from dot's NS coord
    assignment, fundamental shape mismatch on graphs with branching
    is expected)
  - Recommendation: defer dot family pending Round N implementation
    of NS coordinate assignment, OR accept current ~0.34 median as
    the BK ceiling and move to other graphviz families.

### Step 7: Per-round summary

Write `eval_output/algo_fidelity/round_3/SUMMARY.md`:
- Outcome: COMMITTED / RESIDUAL
- One paragraph diagnosis
- One paragraph fix (or rejection reasoning)
- Before/after table
- Recommended Round 4 family

### Step 8: STATE.md iteration log

Append one row to the iteration log in
`.project-context/research/sprint_algo_fidelity/algo_fidelity_STATE.md`.
Update `state:` at top to `ROUND_3_DONE` (regardless of commit/no-commit
outcome). If RESIDUAL: also update `flail_count_dot:` to 2 (we've now
spent 2 rounds on dot without converging). Set `current_family: fdp`
ready for Round 4.
</task>

<scope_constraints>
**HARD scope -- DO NOT TOUCH:**
- `dagua/render/**`
- `dagua/styles.py`
- `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`

**Allowed in Round 3:**
- `dagua/layout/ops/sugiyama.py` (PRIMARY fix surface; coordinate
  assignment region around lines 662, 724, 1191)
- `dagua/layout/ops/pipelines/sugiyama.py` (caller / config; only if
  the lever is a hyperparameter)
- `dagua/layout/ops/state.py` ONLY if a SolveState field needs adding
- `eval_output/algo_fidelity/round_3/**` (new)
- `.project-context/research/sprint_algo_fidelity/**`
- `tests/test_layout/test_*.py` ONLY if a snapshot test needs updating

**Out of scope this round:**
- Wholesale algorithm replacement (network-simplex coordinate assignment
  is Round N+ work, not Round 3).
- Other pipelines (fmmm, sfdp, stress_maj, classical_mds).
- `dagua/eval/variants.py`, `dagua/eval/competitors/*`, `run_benchmark.py`.
</scope_constraints>

<default_follow_through_policy>
Pick the smallest plausible lever from your Step 2 inspection. If there
are 2 candidate levers of similar size, pick the one with stronger
mechanical evidence (e.g., visible bug or clear hyperparameter
mismatch over speculative balance heuristic).

If no lever has high confidence, write the residual doc and move on.
A clean residual is more valuable than a speculative fix.
</default_follow_through_policy>

<completeness_contract>
Round 3 is COMPLETE when one of these END states is reached:
1. **COMMITTED**: COMMIT criterion met, code change is on develop with
   `feat(fidelity): round 3 --` prefix, SUMMARY.md written, STATE.md
   updated.
2. **RESIDUAL**: ROUND_3_RESIDUAL.md written explaining why no fix,
   classification recorded, NO commit, SUMMARY.md written, STATE.md
   updated. flail_count_dot incremented; current_family advanced to
   fdp.
3. **BLOCKED**: ROUND_3_BLOCKED.md if a hard dependency is missing
   (e.g., live_compare is non-deterministic and can't be stabilized);
   STATE.md state: ROUND_3_BLOCKED.

Step 1 (baseline determinism check) must always run. Steps 2-7 are the
fix loop. Step 8 (STATE.md update) must always run.

NEVER commit a fix without measuring it. NEVER weaken pytest
assertions. NEVER touch out-of-scope files.
</completeness_contract>

<verification_loop>
- After every code change: `pytest tests/test_layout/ -x --tb=short -q`
- Final live_compare must produce post_fix/live_rmsd.csv with non-empty
  data
- `git diff --stat HEAD~0` (or `--cached`) before commit must show only
  allowed scope
- If COMMITTED: confirm `git log --oneline -1` matches the
  feat(fidelity): round 3 prefix
</verification_loop>

<missing_context_gating>
ABORT before edits if:
- live_compare's output is non-deterministic and you can't stabilize it
  (the seed in dagua/layout/ops/pipelines/sugiyama.py:36 should make it
  deterministic; if not, that's a bug to file)
- The Brandes-Köpf code at lines 724-1191 doesn't exist or has been
  refactored since the diagnosis (path/line drift)
- You can't load cached graphviz_dot positions

Write ROUND_3_BLOCKED.md and stop.
</missing_context_gating>

<action_safety>
- ONE commit on develop only IF COMMIT criterion met. Never amend.
- Never delete eval_output files. Never modify cluster-sprint files.
- Never run run_benchmark.py.
</action_safety>

<task>
You are Codex on the dagua project. Repo root: `/home/jtaylor/projects/dagua`. Branch: `develop` (one working branch this sprint -- DO NOT create a new branch).

This is Round 2 of the algo_fidelity sprint. The sprint goal is faithful
replication of the algorithms dagua claims to reimplement, with graphviz
tools first because drop-in graphviz replacement is central to the dagua pitch.

Round 1 baseline at `eval_output/algo_fidelity/round_1/`:
- dot (classic_sugiyama vs graphviz_dot): median RMSD 0.3245 -- WORST family
- Sugiyama is PERFECT on simple/linear graphs (RMSD < 0.01) but
  cliffs hard at medium graphs.
- Smallest divergent reproducer: `mixed_width_labels` (6 nodes, RMSD 0.3476).

Round 2 attacks the dot family at the smallest reproducer first.

## What to do

1. **Build a live comparator** (`scripts/algo_fidelity_live_compare.py`):
   - CLI: `python scripts/algo_fidelity_live_compare.py <dagua_engine> <target_engine> [--graphs g1,g2,...] [--output-dir DIR] [--render-panels]`
   - For each graph (default = all graphs that have target_engine in
     `eval_output/benchmark_full/positions/`):
     - Load the dagua test graph via `dagua.eval.graphs.get_test_graphs()`.
     - Run the dagua engine LIVE via the registered competitor (e.g.
       `dagua.eval.competitors.registry.get('classic_sugiyama').layout(graph, seed=42)`).
     - Load the cached target positions from
       `eval_output/benchmark_full/positions/<graph>__<target_engine>.pt`
       (or whatever the on-disk naming convention is -- match Round 1's loader).
     - Compute Procrustes RMSD using the SAME routine
       `scripts/algo_fidelity_cross.py` uses (single source of truth so
       results are directly comparable).
     - If `--render-panels`: write a side-by-side panel PNG using
       `scripts/algo_fidelity_panel.py` for each graph.
   - Output: `<output-dir>/live_rmsd.csv` with columns
     (graph, n_nodes, dagua_engine, target_engine, rmsd).
   - Also print a summary to stdout: median, p25, p75, p95, worst graph + RMSD.

2. **Run baseline live_compare for sugiyama vs dot** to confirm RMSDs
   match Round 1's cached values (sanity check that LIVE re-run matches
   what was benchmarked):
   ```
   python scripts/algo_fidelity_live_compare.py classic_sugiyama graphviz_dot \
       --output-dir eval_output/algo_fidelity/round_2/baseline
   ```
   If LIVE RMSDs differ from Round 1 cached by more than 0.005 on any
   graph, INVESTIGATE before proceeding -- the discrepancy itself is a
   signal (non-determinism? changed code since cache?).

3. **Diagnose `mixed_width_labels`** (smallest divergent reproducer):
   - Read these files end-to-end:
     - `dagua/layout/ops/pipelines/sugiyama.py`
     - `dagua/layout/ops/sugiyama.py` (the underlying ops)
   - Inspect how each of these compares to graphviz dot's algorithm:
     - **Rank assignment** (`_AssignLayers`): graphviz dot uses **network
       simplex** for rank assignment with edge weights and minlen.
       Does dagua's pipeline use network simplex, longest-path, or
       topological? What's the impact on this graph?
     - **Crossing reduction** (`_BarycenterOrdering`): graphviz dot uses
       **median heuristic + transpose** with 24 sweeps and adaptive
       termination. Does dagua use barycenter, median, or some hybrid?
       Sweep direction order? Tie-breaking?
     - **Coordinate assignment** (`_CoordinateAssignment`): graphviz dot
       uses **network simplex on a coordinate constraint problem** with
       balance/aspect heuristics. Does dagua use Brandes-Köpf,
       barycenter-x, or a custom method?
     - **Edge weight handling**: graphviz weights edges per `weight`
       attr (default 1, often higher for spine/back-edges). Does dagua
       respect edge weights? On `mixed_width_labels` are edge weights
       uniform anyway?
   - Generate panel PNGs (`scripts/algo_fidelity_panel.py`) for the 3
     simplest divergent dot pairings: `mixed_width_labels`,
     `shape_and_routing_matrix`, `small_label_storm` (all 6 nodes,
     RMSD 0.35-0.47). They go to
     `eval_output/algo_fidelity/round_2/baseline/panels/`. Don't
     re-read PNGs into your context -- the user has separate visual
     audit. Just produce them as artifacts for human review.
   - Write `.project-context/research/sprint_algo_fidelity/ROUND_2_DIAGNOSIS.md`
     with:
     - For each of the 3 diagnostic graphs: dagua's predicted vs
       actual node assignments per layer (read from the live run),
       graphviz_dot's positions (loaded from cache), and a comparison
       of layer assignment / x-ordering / x-spacing.
     - The dominant divergence cause, with confidence level
       (high/medium/low), with `file:line` references in the dagua
       sugiyama ops.
     - Proposed single fix.

4. **Apply ONE focused fix** for the dominant divergence cause IF AND
   ONLY IF you have HIGH confidence it will narrow the gap and the
   change is < ~80 lines net. The fix lives in `dagua/layout/ops/sugiyama.py`
   or its caller `dagua/layout/ops/pipelines/sugiyama.py`.
   - If no high-confidence fix exists, write the diagnosis, mark Round 2
     as `diagnosis_only`, and stop short of committing code changes.
     Round 3 will do the fix. (A diagnosis-only round is a valid outcome.)
   - The fix should NOT be a wholesale algorithm replacement. Tune /
     adjust / fix bug. If the diagnosis says "dagua needs network
     simplex instead of longest-path", that's Round 3+ work, not Round 2.

5. **Measure delta**:
   ```
   python scripts/algo_fidelity_live_compare.py classic_sugiyama graphviz_dot \
       --output-dir eval_output/algo_fidelity/round_2/post_fix
   ```
   Compare against `eval_output/algo_fidelity/round_2/baseline/live_rmsd.csv`.
   - If median RMSD improved by >= 0.02 OR the 3 diagnostic graphs all
     improved by >= 0.05: PROCEED to commit.
   - If median got worse OR no per-graph clearly improved: REVERT the
     fix, write `ROUND_2_REVERTED.md` with what was tried and why
     reverted, and stop short of committing. Round 3 will pick a
     different lever.
   - Watch for regressions on the simple graphs that were RMSD ~0
     (linear_3layer_mlp, parallel_multiedge_bundle, tl_cnn_small,
     tl_mlp_3layer, nested_shallow_enc_dec) -- ANY non-trivial
     regression on those is a hard veto.

6. **Tests**:
   - Run `pytest tests/test_layout/ -x --tb=short -q -k "sugiyama"` --
     all sugiyama tests must pass. Existing pipeline-fidelity tests
     are the regression guard.
   - If a test fails because dagua sugiyama now produces different
     output, decide: is the test a snapshot of the OLD (dot-divergent)
     behavior? If so, it may need updating to match the closer-to-dot
     behavior. Document the test-change rationale in commit message.
     Avoid weakening assertions.

7. **Commit** (only if measure step passed):
   ```
   feat(fidelity): round 2 -- sugiyama-vs-dot first lever (<short fix description>)

   - <bullet describing the diagnosis>
   - <bullet describing the fix>
   - dot family median RMSD <BEFORE> -> <AFTER> across <N> graphs
   - mixed_width_labels: <BEFORE> -> <AFTER>
   - shape_and_routing_matrix: <BEFORE> -> <AFTER>
   - small_label_storm: <BEFORE> -> <AFTER>
   - Simple-graph regression check: max delta across (linear_3layer_mlp,
     parallel_multiedge_bundle, tl_cnn_small, tl_mlp_3layer,
     nested_shallow_enc_dec) = <X>
   ```
   Stage ONLY files you edit; do NOT stage cluster-sprint files.

8. **Per-round summary**: write
   `eval_output/algo_fidelity/round_2/SUMMARY.md` with:
   - Diagnosis (1 paragraph)
   - Fix applied (1 paragraph) or "diagnosis_only / no fix this round"
   - Before/after table (3 diagnostic graphs + family medians)
   - Simple-graph regression status
   - Recommended next lever for Round 3

9. **Append to STATE.md iteration log** one row for round 2.

## Verification (run before commit)
```
cd /home/jtaylor/projects/dagua
pytest tests/test_layout/ -x --tb=short -q -k "sugiyama"
python scripts/algo_fidelity_live_compare.py classic_sugiyama graphviz_dot \
    --output-dir eval_output/algo_fidelity/round_2/post_fix
# read post_fix/live_rmsd.csv and confirm median improved or simple graphs unregressed
```
</task>

<scope_constraints>
**HARD scope -- DO NOT TOUCH:**
- `dagua/render/**`
- `dagua/styles.py`
- `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`

**Allowed in Round 2:**
- `dagua/layout/ops/sugiyama.py` (the dominant fix surface)
- `dagua/layout/ops/pipelines/sugiyama.py` (caller / config)
- `dagua/layout/ops/state.py` ONLY if a SolveState field needs adding
  to plumb a missing parameter; new fields only, no removals
- `scripts/algo_fidelity_live_compare.py` (new)
- `eval_output/algo_fidelity/round_2/**` (new)
- `.project-context/research/sprint_algo_fidelity/**` (existing)
- `tests/test_layout/test_*.py` ONLY if a snapshot test needs updating
  to match the new (closer-to-dot) behavior; do NOT weaken assertions

**Out of scope this round:**
- Wholesale algorithm replacement (e.g., adding network simplex from
  scratch). That's Round 3+ work if diagnosed as needed.
- Touching other pipelines (fmmm, sfdp, stress_maj, classical_mds).
- Changing `dagua/eval/variants.py`.
- Re-running the full benchmark (`run_benchmark.py`).
</scope_constraints>

<default_follow_through_policy>
Default to the most reasonable low-risk interpretation and keep going.
Only stop for missing details that change correctness, safety, or
irreversible actions.

Specifically: if the diagnosis is clear but the fix is borderline (~80
lines), prefer to err on the side of NOT committing code -- write a
sharp diagnosis report and let Round 3 implement. A clean
diagnosis-only Round 2 is more valuable than a messy fix that gets
reverted.
</default_follow_through_policy>

<completeness_contract>
The round is COMPLETE when:
1. `scripts/algo_fidelity_live_compare.py` exists and runs end-to-end.
2. `eval_output/algo_fidelity/round_2/baseline/live_rmsd.csv` matches
   Round 1 cached RMSDs within 0.005 per graph (sanity check).
3. `ROUND_2_DIAGNOSIS.md` is written with file:line evidence.
4. EITHER:
   - A focused fix is applied AND measured AND median RMSD improved
     by >= 0.02 (or 3 diagnostic graphs each improved by >= 0.05)
     AND simple graphs unregressed AND tests pass AND committed.
   - OR: no fix applied, `ROUND_2_REVERTED.md` or
     `ROUND_2_DIAGNOSIS_ONLY.md` written explaining why no fix this
     round, NO commit.
5. `eval_output/algo_fidelity/round_2/SUMMARY.md` written.
6. STATE.md iteration log row appended.

A diagnosis-only round is a valid completion path. A speculative fix
that gets reverted at measure-time is also valid (and shipped as
ROUND_2_REVERTED.md). The ONE thing that's NOT valid: applying a
fix without measuring its impact.
</completeness_contract>

<verification_loop>
- After every code change to `dagua/layout/ops/sugiyama.py` or
  `pipelines/sugiyama.py`, run:
  ```
  pytest tests/test_layout/ -x --tb=short -q -k "sugiyama" 2>&1 | tail -30
  ```
  If failures, read them; decide if test snapshot needs updating
  (rare) or fix is wrong (more common -- back out the fix).
- Final live_compare must run cleanly and produce the post_fix CSV.
- `git status` after commit must show only the allowed scope.
</verification_loop>

<missing_context_gating>
ABORT BEFORE EDITING SOURCE if:
- The dagua sugiyama op file structure doesn't match what's described
  here.
- `eval_output/benchmark_full/positions/` files for graphviz_dot are
  missing or unreadable.
- The live competitor adapter for `classic_sugiyama` produces
  positions that don't match the cached RMSDs (could mean the
  benchmark used different params -- understand the discrepancy first).

In any of these cases, write
`.project-context/research/sprint_algo_fidelity/ROUND_2_BLOCKED.md`
with what you observed and stop. Do not guess.
</missing_context_gating>

<action_safety>
- ONE commit on `develop` only IF the fix passes measurement.
- No force-push, branch creation, rebase, or tag.
- Do not modify `dagua/eval/variants.py` (registry untouched this round).
- Do not run `run_benchmark.py` (uses cached graphviz positions).
- Do not delete files in `eval_output/`. Add only.
- Do not weaken pytest assertions to make tests pass. If a test fails
  because dagua sugiyama improved, change the test's expected value
  with a clear rationale -- but never replace `assert X == Y` with
  `assert True` or similar.
</action_safety>

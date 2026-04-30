<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 23 EXHAUSTIVE SWEEP for **fmmm** family (`classic_fmmm` vs `ogdf_fmmm`).

The user explicitly directed: **"plz fix ALL issues you found"** -- not just top 3.

## Round 22 status: committed

Round 22 already committed the top 3 levers. Round 23 should apply EVERY REMAINING ranked-list item (items #4 onward) plus any items the diff doc flagged as 'lower priority' that you can verify add value.

## SPEC

Primary: `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_fmmm.md` (full ranked fix list).
Secondary: existing `ROUND_22_*_fmmm.md` reports for context.

Apply EVERY remaining ranked-list item that is technically feasible.
For each item:
- Estimate the fix size (lines net)
- Apply if < ~200 lines net; if larger, document why deferred
- Each fix can be its own micro-commit; OR bundle related ones

## Important: dirty workspace tolerance

The repo has uncommitted changes from a parallel cosmetic sprint
(dagua/render/**, dagua/styles.py, scripts that the cosmetic sprint
owns). DO NOT touch those files. Stage and commit only YOUR family's
files. Do NOT use `git add -A` or `git commit -a`. Use specific
`git add <path1> <path2>` for the files you actually modified.

If `git status` shows your family's files modified alongside the
cosmetic-sprint files, that's expected -- commit only yours.

## OGDF runner rebuild guidance

If you modify `scripts/ogdf_runner.cpp`, you MUST rebuild the binary.
Look for a build script under `scripts/` (e.g. `build_ogdf_runner.sh`)
or a Makefile. If you can't find one, run:
```
g++ -std=c++17 -O2 scripts/ogdf_runner.cpp \
    -I/home/jtaylor/projects/_references/ogdf/include \
    -L/home/jtaylor/projects/_references/ogdf/build/lib \
    -logdf -o scripts/ogdf_runner
```
If the OGDF library isn't built, fall back: don't modify the runner
this round; document the gap.

## Process

1. Read ROUND_21_DIFF_fmmm.md fully + any ROUND_22_*_fmmm.md.
2. Multi-seed baseline (use the bounded 5-graph subset):
   ```
   python scripts/algo_fidelity_live_compare.py classic_fmmm ogdf_fmmm \
       --seeds 3 \
       --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
       --output-dir eval_output/algo_fidelity/round_23/fmmm/baseline
   ```
3. Apply remaining ranked items. Each as its own micro-commit OR bundle
   related items. Commit messages: `feat(fidelity): round 23 fmmm -- <short>`.
4. Run `pytest tests/test_layout/ -x --tb=short -q -k "fmmm"` after each commit.
5. Final measure:
   ```
   python scripts/algo_fidelity_live_compare.py classic_fmmm ogdf_fmmm \
       --seeds 3 \
       --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
       --output-dir eval_output/algo_fidelity/round_23/fmmm/post_fix
   ```
6. Per-round summary: `eval_output/algo_fidelity/round_23/fmmm/SUMMARY.md`
7. List EVERY ranked-list item you addressed AND what you skipped/why.

## Commit policy (relaxed)

You can commit:
- Even small improvements (delta >= 0.005)
- Opt-in fidelity_mode flags with regression tests (even if median unchanged)
- Pure code-quality fixes (e.g. eigen-dim reversal correction in classical_mds; ternary inversion in rt) that don't move RMSD but are correctness fixes
- Pure infrastructure improvements (e.g. expose iteration count parameter, add weight-handling toggle)

You should NOT commit:
- Code that regresses median RMSD by > 0.01 (revert)
- Code that breaks existing tests (fix or revert)
- Sweeping changes that reach into unrelated families

Multiple commits per family allowed -- ideal scope is "one logical
fix per commit".

## Scope

**Allowed**:
- Family-specific ops/pipeline files (per ROUND_21_DIFF_fmmm "Files Read" section)
- `dagua/layout/ops/state.py` only if SolveState field needed
- Family-specific support files (per the diff doc)
- `scripts/ogdf_runner.cpp` IF the diff doc recommends runner-side changes (rebuild after!)
- `dagua/eval/competitors/<family>_competitor.py` IF diff doc explicitly recommends adapter changes
- `scripts/build_ogdf_runner.sh` (NEW or update) for runner rebuilds
- `eval_output/algo_fidelity/round_23/fmmm/**`
- `.project-context/research/sprint_algo_fidelity/ROUND_23_*fmmm*.md`
- `tests/test_layout/test_*fmmm*.py` for regressions

**HARD do-not-touch**:
- `dagua/render/**`, `dagua/styles.py`, `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`
- Other families' pipeline/ops files

## Verification

After each commit:
- `pytest tests/test_layout/ -x --tb=short -q -k "fmmm"` passes
- `git diff --stat HEAD~1 HEAD` shows ONLY your family's files

Final state:
- `eval_output/algo_fidelity/round_23/fmmm/SUMMARY.md` lists every item attempted, with status (commit hash | reverted | skipped + reason)
</task>

<scope_constraints>
fmmm family only. May commit MULTIPLE times. Stage specific files only.
NEVER `git add -A`. Cosmetic-sprint files (render/, styles.py) are off-limits.
</scope_constraints>

<task>
r76-C3-SURGICAL: partial revert in the umap worktree. Attempt 2 (READ:
.project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_umap_NOTES.md, "Attempt
2" section) made three classes of change to dagua/layout/ops/umap.py (all uncommitted):
(1) the epochs_per_sample float32 schedule fix, (2) local reference-parity numba kernels
(tau_rand/rdist/clip/single-epoch + DAGUA_DISABLE_NUMBA fallback), (3) "diagnostic" spectral
init alignment changes (connected ARPACK branch ~lines 575-625 and component meta-layout
~lines 628-687, per the notes).

BENCH EVIDENCE (from a 100-seed disposition run): change-class (3) BROKE previously
bit-identical rows: citation_dag_300 x 6 variants dropped from quality_identical to merely
equivalent, while clustered_longlabel_handoffs x 6 stayed bit-identical. Classes (1)+(2) are
proven correct (kernel matches installed umap-learn 0.5.11 exactly on the draw-for-draw
trace).

YOUR JOB: surgically REVERT class (3) -- restore the pre-attempt spectral init code paths
exactly as they are in `git show HEAD:dagua/layout/ops/umap.py` -- while KEEPING classes
(1)+(2) byte-for-byte. Method suggestion: extract the HEAD version of the spectral-init
functions and splice them over the modified ones; do NOT touch the kernel/schedule code.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-umap-port (branch r76/umap-port).
PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

VERIFY (all must pass, then COMMIT):
1. Tiny trace still exact: re-run the attempt-2 tiny-trace comparison (4-node path, seed 7)
   -- dagua vs installed umap-learn epoch-1 embedding max abs diff 0.0.
2. Spectral restoration proof: for citation_dag_300 seeds 100-102, positions from the
   worktree code must be BYTE-IDENTICAL to positions from `git stash`-ed HEAD code (run both
   in scratch; torch.equal). Same check on clustered_longlabel_handoffs seed 100 -- wait:
   clustered stayed identical WITH the alignment changes, so after revert it must STILL
   match HEAD output (if HEAD and attempt-2 differed on it, investigate before committing).
3. pytest tests/test_pipeline_umap_layout.py -x -q green (numba on AND
   DAGUA_DISABLE_NUMBA=1); pytest tests/test_ops_optimize.py -k umap -x -q green; ruff clean.
4. AST check: no umap/umap-learn imports in dagua runtime modules (numba optional-import
   pattern is fine).
COMMIT on r76/umap-port when 1-4 pass: two conventional commits -- (a) the schedule fix +
kernel parity port, (b) notes update. Append "## Attempt 2b: spectral revert" to the notes
(what was reverted, verification evidence, commit shas). Commits AUTHORIZED and REQUIRED on
pass; re-add/re-commit through ruff-format until `git log` SHOWS them. No push/merge. NO AI
attribution. ASCII only.

KNOWN PRE-EXISTING FAILURE (verified on develop, NOT yours, must not block commit):
tests/test_bench_large.py::test_hierarchy_checkpoint_rejects_incomplete_manifest.
</task>
<completeness_contract>
Done = classes (1)+(2) intact, class (3) reverted to HEAD, checks 1-4 green, committed. If
check 2 shows the revert does NOT restore byte-identity to HEAD on citation_dag_300, STOP and
document what else differs (bisect the file hunks) -- no commit.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r76/umap-port only. Touch ONLY dagua/layout/ops/umap.py and the
notes file. Never modify files outside the worktree.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>

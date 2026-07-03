<task>
r76-C3-RETRY: umap scalar-faithful port, attempt 2 (focused fixup). Attempt 1 (READ FIRST:
.project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_umap_NOTES.md in this
worktree) SUCCEEDED at phase 1: epochs_per_sample float32 schedule divergence found+fixed;
tiny-graph trace matches the reference draw-for-draw through epoch 1 (schedule, tau-rand
draws, negative targets, gradients, final embedding). It then stalled on a GATE ERROR IN THE
BRIEF, which is hereby corrected:

CORRECTED GATE 4 (this supersedes attempt 1's no-numba rule): the no-runtime-delegation rule
forbids importing/invoking the REFERENCE PACKAGE (umap-learn) from dagua ops at runtime.
NUMBA IS ALLOWED -- dagua's umap op module already used the optional-numba-with-fallback
pattern before attempt 1 removed it (see `git -C . diff dagua/layout/ops/umap.py` -- the
deleted `_UMAP_SINGLE_EPOCH = _numba_njit(..., fastmath=True)` wrappers). Numba JIT of
dagua's OWN kernel is legitimate; it mirrors the reference's compilation environment exactly
like linking the same BLAS. Numba must remain OPTIONAL (pure-python fallback intact; PyTorch
stays the only required dependency).

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-umap-port (branch r76/umap-port). The
uncommitted attempt-1 changes to dagua/layout/ops/umap.py are your starting point.
PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

THE PLAN:
1. RESTORE the optional numba wrappers (undo attempt 1's deletion; keep attempt 1's
   epochs_per_sample float32 fix).
2. MAKE THE KERNEL BODIES STRUCTURALLY IDENTICAL to umap-learn 0.5.11's (installed at
   /home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/):
   - rdist: layouts.py:31-41 (float32, fastmath=True)
   - single-epoch optimizer body: layouts.py:92-187 (called via the njit factory at
     layouts.py:222-228 -- note umap-learn selects the SERIAL kernel when random_state is
     fixed; match that path)
   - clip: the reference clip function (layouts.py, njit'd)
   - tau_rand_int: utils.py:40-63 (i4(i8[:]) signature)
   "Structurally identical" = same statements in the same order, same dtypes, same locals,
   same njit signatures and flags (fastmath, cache, parallel settings) so the SAME installed
   numba version compiles the same float behavior. Dagua's copies live in dagua's module with
   a docstring citing the source (umap-learn is BSD-3) -- no umap import.
3. RE-PROBE: with numba active, rerun the attempt-1 probe (random_dag_50::classic_umap_nn5,
   seeds 0-4, orthogonal Procrustes RMSD vs reference). Attempt-1 baseline: old mean 0.01504
   / new-scalar mean 0.01457. If the kernel is truly identical, expect RMSD to COLLAPSE
   (target <1e-5; bit-exact float32 is the stretch). If it does not collapse, bisect WITHIN
   the epoch loop on random_dag_50 (dump embedding after each epoch for dagua-vs-reference,
   find first diverging epoch, then first diverging edge/sample within it) and fix the named
   difference. Iterate. That bisection loop is the core of this attempt -- push until draws
   match or the exact non-portable construct is named with evidence.
4. TARGET LIST: the 7 divergent combos in attempt-1 notes. r75_final.jsonl is at the MAIN
   repo (read-only): /home/jtaylor/projects/dagua/eval_output/fidelity_definitive/r75_final.jsonl.

GATES (all must pass before commit; else document honestly, leave uncommitted):
1. Probe: on >=3 of the divergent combos' graphs (parallel_multiedge_bundle with default+nn5
   variants, random_dag_50, random_dag_200), 5 seeds each, post-fix RMSD vs reference
   improves DECISIVELY (order-of-magnitude, not 3%); report per-seed tables.
2. Zero regressions on 3 previously-identical umap combos (byte-identical or equal-RMSD
   positions pre/post, 5 seeds).
3. Fallback correctness: with numba disabled (monkeypatch/env), pipeline umap tests still
   pass (fidelity parity NOT required for the fallback -- correctness only; document the
   delta).
4. pytest tests/test_pipeline_umap_layout.py + test_ops_optimize.py -k umap green; ruff
   clean. (Attempt 1 saw `pytest tests/ -k umap` die with exit -1 and no output -- if it
   recurs, diagnose briefly [likely collection OOM/plugin], run the equivalent per-file
   selection instead, and note it.)
5. No umap-learn imports in dagua runtime modules (AST check as attempt 1 did). Numba
   imports are permitted under the optional pattern.

COMMIT AUTHORITY: this task-level instruction AUTHORIZES commits on branch r76/umap-port in
this worktree (it supersedes any generic repo-level no-commit guidance a prior agent
mentioned -- that guidance does not apply to dispatched fidelity-sprint worktree branches).
Conventional commits; re-add/re-commit through ruff-format until `git log` SHOWS them. No
push/merge. NO AI attribution in commits. ASCII only.

DELIVERABLES: append an "## Attempt 2" section to r76_IMPL_umap_NOTES.md (kernel-parity
description w/ file:line cites, per-epoch bisection evidence if used, probe tables, gate
evidence, commit sha).
</task>
<completeness_contract>
Done = gates 1-5 pass and committed, OR a precise documented blocker naming the exact
non-portable construct WITH per-epoch bisection evidence (not just "fastmath differs" -- show
the first diverging epoch/sample and why it cannot be matched). This is the final attempt;
honest failure -> aggregate-tier disposition. Never weaken a gate.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r76/umap-port only. Never touch other engines, eval scoring,
reference runners. Never modify files outside the worktree.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>

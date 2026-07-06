<task>
r78-S3: fix the SGD2 native-layout HANG + close the last 2 sgd2 rows. R2 (READ: the R2
section of .project-context/research/sprint_rng_matching/r75_findings/r78_RESIDUAL_MOP.md)
proved the 2 evidence-thin rows (real_football_115 + wide_1_100_1 x
classic_sgd2_multi_with_crossing) are BYTE-IDENTICAL to the reference on all completed
paired seeds (14/14) -- and then found dagua's OWN layout stalls indefinitely
(>60 min, confirmed stuck at phase `native_start`) on real_football_115 seed 113. That is
a genuine dagua bug (hang = defect) AND the blocker for full 100-seed closure.

WORKTREE: create `git -C /home/jtaylor/projects/dagua worktree add
~/.claude/worktrees/dagua-sgd2 -b r78/sgd2 develop`. PYTHONPATH=$PWD;
MPLCONFIGDIR=/tmp/mpl.

DEBUG: reproduce (classic_sgd2_multi_with_crossing on real_football_115 seed 113; R2's
runner script remnants in /tmp/r78_r2 may help). Instrument the sgd2_multi pipeline with
progress/phase prints + faulthandler.dump_traceback_later to find WHERE it spins (crossing
criterion? a while-loop with a convergence predicate that never fires? degenerate
sampling?). Root-cause and fix -- the fix must not change output on non-hanging seeds
(byte-identity gate below). A hang usually means an unbounded loop: bound it the way the
REFERENCE implementation bounds it (check the sgd2 reference package's semantics), not
with an arbitrary cap; if the reference would also hang there, document that astonishing
fact instead.

THEN CLOSE: run the R2 paired runner to full 100 seeds both rows; verdict: identical /
equivalent / named divergence.

GATES (before commit): hang fixed (seed 113 completes in sane time); byte-identity on 10
non-hanging seeds pre/post fix across both rows + 3 other sgd2_multi combos; paired
100-seed verdict produced; pytest -k "sgd2" green; ruff clean. KNOWN pre-existing failures
(must not block): the standard 6-item list. COMMITS ON r78/sgd2 AUTHORIZED AND REQUIRED on
gate pass.

DELIVERABLES: append "## S3: sgd2 hang + closure" to r78_RESIDUAL_MOP.md (root cause w/
file:line, fix, byte gates, the 100-seed verdict tables, commit shas). ASCII. NO AI
attribution. No push/merge.
</task>
<completeness_contract>
Done = hang root-caused + fixed + byte-identity held + 100-seed paired verdict for both
rows, committed. The hang is a defect: "could not reproduce" requires showing the exact
R2 command completing.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r78/sgd2 only. Never touch other engines/eval scoring/
runners. No runtime reference imports in dagua/layout.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices.
</default_follow_through_policy>

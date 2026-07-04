<task>
r77-A9b: close the A9 guard hole -- ONE row, ONE job. A9 (HEAD ecd85be in this worktree)
improved 5/6 cluster rows but REGRESSED small_label_storm::classic_sugiyama_graphviz_
fidelity from d_R 0.006039 (near tier) to 0.045864. The sprint's no-regression invariant is
absolute: no row leaves the bit-exact/near tier. A9's own notes say mixed cluster+edge-label
DOT was "guarded", yet small_label_storm (mixed) still changed -- find the guard hole.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-sugiyama-final (branch r77/sugiyama-final).
PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

WORK: reproduce small_label_storm's layout pre/post A9 (git stash or detached 0c2dee7 run
vs HEAD); binary-search WHICH A9 change reaches it despite the guard (cluster metadata
plumbing? asymmetric lw/rw metadata? the wrapper's cluster-only DOT classification
misclassifying this graph?). Tighten the gating so small_label_storm (and any graph in its
class) is BYTE-IDENTICAL to pre-A9 output, while the 5 improved cluster rows keep their A9
gains (re-verify their d_R after the guard fix).

GATES: small_label_storm byte-identical to 0c2dee7 output (3 seeds); the 5 improved rows
retain their A9 d_R values; byte-identity samples from A9's notes still hold (plain
graphviz 5, label-only 5, igraph 5); pytest tests/ -k "sugiyama or mincross or dot_rank"
-x -q green; ruff clean. KNOWN pre-existing failures (must not block): test_bench_large;
classic_fcose; double-border smoke; test_50_node_dag; graphopt seed-matrix;
test_classify_early_exit. Commit on r77/sugiyama-final; append the guard-hole explanation
to the A9 dossier section.
</task>
<completeness_contract>
Done = guard hole NAMED + fixed + committed with all gates green. This is a surgical fixup;
scope creep is failure.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r77/sugiyama-final only. Touch only the gating logic + tests +
dossier. No benches.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices.
</default_follow_through_policy>

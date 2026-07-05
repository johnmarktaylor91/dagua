<task>
r78-H1: kill the recursion-depth crash on huge graphs -- JMT has authorized verifying the
2000-5000-node monsters, and classic_sugiyama_graphviz_fidelity currently dies with
"maximum recursion depth exceeded" on them (seen on ba_2000/ba_5000/er_2000/powerlaw_2000/
rgg_2000/rgg_500/sbm_8x100 in /tmp/r77_cx_a5.log's bench; also 200 watchdog rows in
/tmp/r77_cx_a7.log's bench). A7 already added a recursion-limit guard for igraph
compaction (see commit 922464a) -- the graphviz-fidelity path needs the equivalent
treatment, preferably ITERATIVE rewrites of the recursive walks rather than
setrecursionlimit bumps.

WORKTREE: create `git -C /home/jtaylor/projects/dagua worktree add
~/.claude/worktrees/dagua-recursion -b r78/recursion develop`. PYTHONPATH=$PWD;
MPLCONFIGDIR=/tmp/mpl.

METHOD: reproduce on ba_2000 x classic_sugiyama_graphviz_fidelity (1 seed) with
faulthandler/full traceback to find the recursive call site(s) (candidates: decompose DFS
from A1, cluster walks, aux-graph builders, network-simplex tree walks). Rewrite each
iteratively (explicit stack), byte-identical output gate on small graphs.

GATES (before commit): ba_2000 + rgg_2000 x graphviz_fidelity complete 3 seeds each
without crash (any generous time is fine -- correctness first); byte-identical positions on
binary_tree/dense_pair_50/weighted_karate_34 x graphviz_fidelity pre/post (3 seeds); pytest
tests/ -k "sugiyama or mincross or dot_rank" -x -q green; ruff clean. KNOWN pre-existing
failures (must not block): the standard 6-item list. Commit on r78/recursion.

DELIVERABLES: append "## H1: iterative rewrites for huge graphs" to
r76_IMPL_mincross_NOTES.md (call sites, rewrite approach, crash-free evidence, byte gates,
commit sha). ASCII. NO AI attribution. No push/merge. No full benches (the big-graph
campaign handles those).
</task>
<completeness_contract>
Done = named call sites rewritten, monsters complete without crash, byte-identity held,
committed.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r78/recursion only. Never touch igraph paths, eval scoring,
reference runners.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices.
</default_follow_through_policy>

<task>
RETRY (attempt 2 of 2) of the graphviz mincross phase-1 port for dagua sugiyama graphviz-fidelity.
Attempt 1 (uncommitted changes sitting in this worktree -- inspect via `git diff` and
`git status`) FAILED its ladder: crossings moved AWAY on 3/4 targets (dense_pair_50 391->400 vs
ref 331; hub_skip_superfan 3->5 vs 2; heavy_tail_weights_50 70->90 vs 67; weighted_karate_34
overshot 111->76 vs 108) and ba_500 timed out in the Python transpose loop. Read its full notes
FIRST: .project-context/research/sprint_rng_matching/r75_findings/r75_IMPL_mincross_NOTES.md
(includes a source-fidelity correction: 7.0.5 crossing counts use ED_xpenalty, NOT
omega/virtual_weight -- keep that correction).

Repo/worktree: /home/jtaylor/.claude/worktrees/dagua-mincross (branch r75/mincross). You may
keep, amend, or discard attempt 1's uncommitted changes -- your call after reading them.
Conventional commit when (and ONLY when) the ladder passes. No push, no merge.
PYTHONPATH=$PWD for all benchmark/test runs. Pre-commit ruff-format may reformat: re-add and
re-commit until it lands.

THE NEW DISCRIMINATOR (use it -- this is what attempt 1 lacked):
The installed `dot` binary (graphviz 7.0.5) prints its mincross result at -v:
  `echo '<dotfile>' | dot -v 2>&1 | grep mincross`
emits "mincross <g>: N crossings" -- the ORDERING-stage crossing count for the exact graph.
Build the DOT input for each ladder graph exactly the way the benchmark adapter does
(dagua/eval/competitors/graphviz_competitor.py -- same node order, same attrs) so the comparison
is apples-to-apples. Now you can iterate: your port's post-ordering weighted crossing count must
CONVERGE to dot -v's number per graph BEFORE looking at rendered-position crossings at all.
Iterate on small ladder graphs (binary_tree, bipartite_4_3_4, hub_skip_superfan,
weighted_karate_34) until the ordering-stage counts match or you can name the exact unported rule
(with 7.0.5 file:line).

LIKELY CULPRITS for attempt 1's regression (verify against 7.0.5 source, do not guess):
1. INIT ORDER: attempt 1 approximated GD_nlist with "reverse expanded-node creation order".
   Port the REAL install order: decompose/install_in_rank/build_ranks interplay
   (7.0.5:lib/dotgen/mincross.c:1212-1480, lib/dotgen/fastgr.c GD_nlist maintenance :205-264,
   lib/common/utils.c if referenced). The node list order graphviz iterates matters for BFS
   seeding and tie behavior.
2. TRANSPOSE semantics: exact left2right gating (even without flat/cluster constraints,
   :557-579 has ordering rules), the reverse-direction alternation, and exchange conditions.
3. Pass-2 convergence bookkeeping (best-order save/restore on non-improvement -- graphviz
   restores the BEST order seen, not the last).
PERF for ba_500: the transpose inner loop needs an incremental crossing-delta computation
(graphviz computes in_cross/out_cross deltas for the exchanged pair only, :632-688) -- port that
instead of recounting. Target: ba_500 1 seed completes < 240s on an otherwise-loaded machine.

LADDER (same as before, all must pass before commit):
a. Stage-A stress no-regression: binary_tree, bipartite_4_3_4, org_chart_1_5_4_8.
b. Ordering-stage convergence: your mincross crossing count matches `dot -v` on >=3 of the 4
   small ladder graphs (document any residual with the exact rule).
c. Rendered crossings move TOWARD reference on >=3/4 of: dense_pair_50, weighted_karate_34,
   hub_skip_superfan, heavy_tail_weights_50.
d. ba_500 1 seed < 240s AND crossings improve >2x from the 22344 baseline.
e. igraph/default paths byte-identical (5 seeds x default,tight on binary_tree+densenet_block);
   pytest tests/ -k sugiyama and tests/test_layout/test_dot_mincross.py green.
If after a genuine effort the ladder still fails, DO NOT COMMIT: write the failure analysis to
the NOTES file (r75_IMPL_mincross_NOTES.md, append a "Attempt 2" section) naming the exact
blocking rule with 7.0.5 citations, and leave the worktree changes uncommitted. This is the
final attempt this sprint; an honest documented failure routes it to the next sprint.
Update NOTES with everything either way.
</task>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices in NOTES. Stop only for
correctness-critical ambiguity.
</default_follow_through_policy>

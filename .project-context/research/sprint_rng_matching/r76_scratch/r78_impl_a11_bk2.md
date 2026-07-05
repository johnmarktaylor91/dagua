<task>
r78-A11: igraph BK SECOND-ORDER bisection -- finish what A7 started. A7 (READ FIRST:
.project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_igraph_NOTES.md, A6/A7
sections) ported the first BK divergence (flag-driven 4 runs over original
vertex_to_the_left + min-width anchor + median_4) and the GLPK rank dep landed; the A6
probe reached 8/10 under d_R 0.01. 145 igraph-family rows remain divergent
(per eval_output/fidelity_definitive_r77/OFFICIAL_R77_LEDGER.md registry) with the residual
NEVER bisected: representative classes hexagonal_lattice_42 and width_skew_late_merge
(x-stage), plus whatever the 145 cluster into. This gap is UNFINISHED WORK, not a floor --
finish it.

WORKTREE: create `git -C /home/jtaylor/projects/dagua worktree add
~/.claude/worktrees/dagua-bk2 -b r78/bk2 develop`. PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.
swiglpk installed (GLPK path active).

METHOD (the campaign-proven loop): (1) cluster the 145 rows by graph class (read the r77
wired rescore eval_output/fidelity_definitive/r77_sugiyama_wired.jsonl, igraph rows with
d_R>=0.01); (2) instrumented python-igraph build in a /tmp venv (NEVER the env's igraph)
dumping BK internals per node: Type-1/Type-2 conflict marks, per-direction alignment roots/
blocks, per-direction candidate x, the balancing combination, final x; (3) mirror dumps
from dagua; (4) on hexagonal_lattice_42 + width_skew_late_merge + 1 more class rep, find
the FIRST differing quantity; (5) port it gated to fidelity_mode="igraph"; (6) iterate
until the probe classes collapse or a quantity is proven non-reproducible (show the dump).
NOTE: batch reference igraph calls in fresh subprocesses (~75-call segfault known).

GATES (before commit): d_R<0.01 on >=8 of a 12-row probe spanning the residual classes; NO
row leaves bit-exact/near (the 141 bit-exact are sacred -- byte-identity sample 10 rows x 3
seeds); graphviz-fidelity byte-identical (5-row sample); no-swiglpk fallback tests green;
pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q green; ruff clean. KNOWN
pre-existing failures (must not block): test_bench_large; classic_fcose; double-border
smoke; test_50_node_dag; graphopt seed-matrix; test_classify_early_exit. Commit on r78/bk2;
then family bench --engines classic_sugiyama --variants --max-nodes 300 --seeds 100
--seed-start 100 --workers 5 --timeout 3600 --watchdog-timeout 7200 --output-dir
/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_bk2 (0 errors).

DELIVERABLES: append "## A11: BK second-order" to r76_IMPL_igraph_NOTES.md (cluster table
of the 145, per-quantity bisection, ports w/ source cites, before/after d_R, gate evidence,
commit shas, bench line). ASCII. NO AI attribution. No push/merge. Clean /tmp scratch.
</task>
<completeness_contract>
Done = the 145 rows clustered AND each cluster either PORTED (gates green) or
non-reproducibility PROVEN with the instrumented dump. "Another detail remains" without a
named quantity is NOT an acceptable endpoint this round.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r78/bk2 only. NEVER modify installed igraph; never touch
graphviz-fidelity paths, eval scoring, reference runners. Bench write to
benchmark_100seed_r78_bk2 only. NO runtime igraph imports in dagua/layout.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>

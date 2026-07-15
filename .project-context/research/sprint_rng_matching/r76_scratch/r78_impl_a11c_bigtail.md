<task>
r78-A11c: the igraph BIG-GRAPH tail (52 far + 18 close rows -- ALL on >300-node graphs:
ba_500/2000, er_500/2000, powerlaw_500/2000, rgg_500/2000, small_world_500, dependency_500,
sbm_8x100; list in .project-context/research/sprint_rng_matching/r76_scratch/
r78_igraph_tail.txt). CRITICAL CONTEXT: every family bench capped at <=300 nodes, so these
rows' ledger positions predate ALL the r77/r78 igraph fixes (GLPK, BK alignment, chain
incidence order) -- the ledger numbers are stale-code artifacts until proven otherwise.
Also: dagua's igraph fidelity has a documented >1000-node fallback ("Eades layering above
GLPK's node gate", tests/test_layout/test_sugiyama_fidelity.py) whose faithfulness to
igraph's ACTUAL behavior at scale has never been audited.

WORKTREE: create `git -C /home/jtaylor/projects/dagua worktree add
~/.claude/worktrees/dagua-bigtail -b r78/bigtail develop` (develop has ALL fixes: GLPK,
BK, chain order, iterative walks). PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

WORK:
1. GATE AUDIT: what does INSTALLED igraph actually do on 500- and 2000-node sugiyama
   (does its GLPK usage have a size gate? read the igraph source /tmp READ-ONLY + verify
   empirically: does layout("sugiyama") on er_2000 use GLPK-rank behavior)? If dagua's
   >1000 fallback diverges from igraph's real behavior, FIX the gate to mirror igraph
   exactly (maybe igraph has no gate and dagua needs GLPK at 2000 nodes -- measure GLPK
   runtime there; if minutes, that is acceptable).
2. CURRENT-CODE PROBES (same-process, both sides, 1 seed then 3): ba_500, er_2000,
   dependency_500, sbm_8x100 x default variant -- d_R vs installed igraph with CURRENT
   develop code. Expect many to already be near/bit-exact post-A11b (stale-artifact
   confirmation). For any still far: bisect with the instrumented igraph build (the
   standard /tmp venv pattern; subprocess-batch to dodge the segfault) and port the named
   law.
3. Report the stale-vs-real split for all 70 rows.

GATES (before commit, if code changes): probe d_R <0.01 on the fixed classes (3 seeds);
zero regressions (the 257 bit-exact sample 10 rows x 3 seeds byte-identical; no-swiglpk
fallback green; graphviz rows byte-identical 5-row sample); pytest tests/ -k "sugiyama or
mincross or dot_rank" -x -q green; ruff clean. KNOWN pre-existing failures (must not
block): the standard 6-item list. COMMITS ON r78/bigtail AUTHORIZED AND REQUIRED on gate
pass. Then bench the 11 tail graphs x classic_sugiyama --variants --max-nodes 0 --seeds
100 --seed-start 100 --workers 4 --timeout 21600 --watchdog-timeout 28800 --output-dir
/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_bigtail (0 errors; generous
timeouts are JMT-authorized; the H1 iterative rewrites removed the old crash class).

DELIVERABLES: append "## A11c: big-graph tail" to r75_findings/r76_IMPL_igraph_NOTES.md
(gate audit verdict w/ source cite, current-code probe table, stale-vs-real split, any
ports w/ cites, gate evidence, commit shas, bench line). ASCII. NO AI attribution. No
push/merge. Clean /tmp scratch.
</task>
<completeness_contract>
Done = gate audited (mirror-or-justified), current-code d_R measured for all tail classes,
every still-far class either ported (gates green) or instrument-dossiered, and the fresh
big-graph bench clean. Stale ledger numbers are NOT evidence -- measure with current code.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r78/bigtail only. NEVER modify installed igraph; no runtime
igraph imports in dagua/layout. Bench write to benchmark_100seed_r78_bigtail only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>

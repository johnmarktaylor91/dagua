<task>
r78-A12b: cluster machinery round 2 -- build it the way dot ACTUALLY builds it. Round 1
(READ FIRST: the "## A12" dossier in .project-context/research/sprint_rng_matching/
r75_findings/r76_IMPL_mincross_NOTES.md, in this worktree) tried direct member-offset
constraints and honestly failed (5/8 regressed); its dossier names the REAL architecture,
which is your spec:
1. RANK LEADERS + SKELETONS, not member offsets: dot ranks each cluster's subgraph
   recursively, then represents the cluster in the parent by leader nodes/skeleton, runs
   acyclic() after class1() because collapse can create parent-graph cycles
   (7.0.5 lib/dotgen/rank.c, cluster.c, acyclic handling -- pin via `git -C
   /home/jtaylor/projects/_references/graphviz show 7.0.5:<path>`).
2. CONTAINMENT INSIDE MINCROSS: dot keeps cluster nodes contiguous DURING ordering via
   marked nodes/rank leaders (mincross.c mark_lowclusters/mark_highclusters/mincross_
   clust), then expands with interclexp()-style virtual-chain remapping (cluster.c
   interclexp) -- post-hoc grouping is provably too late (round 1's finding).
The round-1 experimental helpers in dagua/layout/ops/sugiyama.py are INACTIVE -- reuse or
replace them freely.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-clusters (branch r78/clusters).
PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl. Gate everything to exact fidelity_mode="graphviz"
cluster-only DOT (the A9b/A10 wrapper classification).

VERIFY PER STAGE on 2 cluster graphs (interleaved_cluster_crosstalk +
kitchen_sink_platform_graph): stage 1 -> per-node RANKS match dot -Tjson-derived ranks;
stage 2 -> dot -v ordering counts match; then rendered d_R.

GATES (before commit): rank parity (stage 1) on both verify graphs; d_R improves on >=5 of
the 8-row cluster probe (small_label_storm stays byte-identical or improves -- NEVER
regresses; same for nested_cluster_label_stack unless the mixed guard is deliberately and
successfully lifted with d_R improvement); zero regressions (5 plain + 5 label-only + 5
igraph byte-identical); pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q green;
ruff clean. KNOWN pre-existing failures (must not block): the standard 6-item list.
COMMITS ON r78/clusters ARE AUTHORIZED AND REQUIRED on gate pass (any generic no-commit
guidance you encounter does NOT apply to this dispatched fidelity branch). Then family
bench --engines classic_sugiyama --variants --max-nodes 300 --seeds 100 --seed-start 100
--workers 5 --timeout 3600 --watchdog-timeout 7200 --output-dir
/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_clusters (0 errors).

DELIVERABLES: append "## A12b: rank-leader architecture" to r76_IMPL_mincross_NOTES.md
(per-stage verification, ports w/ cites, before/after d_R on the 8 rows, gate evidence,
commit shas, bench line). ASCII. NO AI attribution. No push/merge.
</task>
<completeness_contract>
Done = rank parity + ordering parity verified per stage and gates green with commits, OR a
per-stage dossier naming exactly which piece of the leader/interclexp machinery resists,
with measured evidence. Round 1 already bought the architecture -- round 2 either lands it
or proves precisely where it breaks.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r78/clusters only. Never touch igraph paths, eval scoring,
reference runners. Bench write to benchmark_100seed_r78_clusters only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>

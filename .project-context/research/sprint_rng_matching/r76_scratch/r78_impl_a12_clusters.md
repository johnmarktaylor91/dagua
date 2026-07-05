<task>
r78-A12: port dot's RECURSIVE CLUSTER RANK-COLLAPSE -- JMT has authorized the last big
subsystem ("TRULY FINISH"). 20 graphviz-sugiyama rows on cluster graphs remain divergent
because dagua lacks dot's cluster ranking pipeline. A9 already ported cluster x
slot/boundary constraints (READ: .project-context/research/sprint_rng_matching/
r75_findings/r76_IMPL_mincross_NOTES.md sections A8/A9/A9b/A10 -- the DOT-content audit,
the guard rules, what is already in). Your job is the remaining machinery.

WORKTREE: create `git -C /home/jtaylor/projects/dagua worktree add
~/.claude/worktrees/dagua-clusters -b r78/clusters develop`. PYTHONPATH=$PWD;
MPLCONFIGDIR=/tmp/mpl. VERSION PIN: `git -C /home/jtaylor/projects/_references/graphviz
show 7.0.5:<path>`.

THE MACHINERY (7.0.5, port in this order, verifying each stage against `dot -v`/`-Tjson`
on 2 representative cluster graphs before the next):
1. RANK COLLAPSE (lib/dotgen/rank.c + cluster.c collapse_cluster/expand_cluster): each
   cluster ranks its internal subgraph recursively, is collapsed to a single node for the
   parent graph's ranking, then expanded with rank remapping (GD_minrank/GD_maxrank,
   set_minmax). VERIFY: dagua's per-node ranks match dot -Tjson-derived ranks on cluster
   graphs (they currently should NOT; after this stage they must).
2. CLUSTER CONTAINMENT IN ORDERING (mincross.c: mark_lowclusters/mark_highclusters,
   cluster-local mincross passes, keeping cluster nodes contiguous within ranks). VERIFY:
   dot -v ordering counts on the cluster calibration graphs.
3. Interaction with the ALREADY-PORTED x constraints (A9) + edge-label machinery (A8):
   after stages 1-2, REMOVE the A9b mixed-DOT guard for cluster+label graphs IF the
   combined pipeline now matches (test on small_label_storm + nested_cluster_label_stack:
   byte-safety is superseded by correctness ONLY when d_R improves and does not regress).
Gate every change to exact fidelity_mode="graphviz".

GATES (before commit): per-node rank parity on 2 cluster graphs (stage 1); ordering counts
(stage 2); d_R improves materially on >=5 of the 8-row cluster probe set (the 20 rows'
graph classes incl small_label_storm/nested if the guard lifts; NO row leaves
bit-exact/near); zero regressions (5 plain + 5 label-only + 5 igraph rows byte-identical);
pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q green; ruff clean. KNOWN
pre-existing failures (must not block): test_bench_large; classic_fcose; double-border
smoke; test_50_node_dag; graphopt seed-matrix; test_classify_early_exit. Commit per stage
on r78/clusters; then family bench --engines classic_sugiyama --variants --max-nodes 300
--seeds 100 --seed-start 100 --workers 5 --timeout 3600 --watchdog-timeout 7200
--output-dir /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_clusters
(0 errors).

DELIVERABLES: append "## A12: recursive cluster rank-collapse" to r76_IMPL_mincross_
NOTES.md (per-stage verification tables, ports w/ 7.0.5 cites, before/after d_R on the 20
rows' classes, gate evidence, commit shas, bench line). ASCII. NO AI attribution. No
push/merge.
</task>
<completeness_contract>
Done = stages 1-2 ported with per-stage verification and gates green, OR an honest
per-stage dossier: exactly which stage resists, its cite, its measured residual. Partial
stage-1-only commits are acceptable if gates hold. This is the FINAL subsystem of the
campaign.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r78/clusters only. Never touch igraph-mode paths, eval
scoring, reference runners. Bench write to benchmark_100seed_r78_clusters only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>

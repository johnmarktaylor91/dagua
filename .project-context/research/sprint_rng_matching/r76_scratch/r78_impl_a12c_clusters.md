<task>
r78-A12c: cluster machinery round 3 -- STAGE 2 ONLY. Round 2 (READ: "## A12b" in
.project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_mincross_NOTES.md, this
worktree) LANDED stage 1: recursive rank collapse + reciprocal collapsed edges give EXACT
per-node rank parity with dot on both verify graphs (interleaved_cluster_crosstalk,
kitchen_sink_platform_graph). That uncommitted stage-1 code is your foundation -- do not
regress it. Stage 2 failed with the architecture named: dot's ordering containment uses
FIRST-CLASS CLUSTER SKELETON/RANK-LEADER NODES PARTICIPATING INSIDE MINCROSS, not post-hoc
containment.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-clusters (branch r78/clusters).
PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

THE SPEC (7.0.5, pin via `git -C /home/jtaylor/projects/_references/graphviz show
7.0.5:lib/dotgen/<f>`): mincross.c mincross_clust() + cluster.c: each cluster is
represented in the ROOT ordering by its skeleton (leader virtual nodes per rank spanning
GD_minrank..GD_maxrank); the root mincross orders leaders like nodes (keeping members
implicitly contiguous); each cluster runs its OWN LOCAL mincross over its members
(recursively for nested clusters); expand/install via interclexp() + install_cluster()
semantics; the final counts printed by dot -v are the sum over root+cluster passes.
VERIFY: dot -v ordering count on interleaved_cluster_crosstalk (target: dot reports 5;
round-2 candidate produced 2 with order mismatch -- match the COUNT and the per-rank order
extracted from -Tjson x-order).

GATES (before commit): stage-1 rank parity PRESERVED; stage-2 ordering parity (count +
per-rank order) on both verify graphs; rendered d_R improves on >=5/8 of the cluster probe
set, NO regressions (small_label_storm + nested byte-identical under the guard; 5 plain +
5 label-only + 5 igraph byte-identical); pytest tests/ -k "sugiyama or mincross or
dot_rank" -x -q green; ruff clean. KNOWN pre-existing failures (must not block): the
standard 6-item list. COMMITS AUTHORIZED AND REQUIRED on gate pass (stage-1 + stage-2 as a
clean series on r78/clusters). Then family bench --engines classic_sugiyama --variants
--max-nodes 300 --seeds 100 --seed-start 100 --workers 5 --timeout 3600
--watchdog-timeout 7200 --output-dir
/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_clusters (0 errors).

DELIVERABLES: append "## A12c: skeleton mincross" to r76_IMPL_mincross_NOTES.md (ordering
parity tables, ports w/ cites, before/after d_R, gate evidence, commit shas, bench line).
ASCII. NO AI attribution. No push/merge.
</task>
<completeness_contract>
Done = ordering parity achieved + gates green + committed, OR a dossier naming exactly
which skeleton/local-mincross element resists with dot -v/-Tjson evidence. Stage 1 is
already proven -- protect it either way (if stage 2 fails, commit stage 1 alone behind an
inactive flag with its parity evidence so the work is not lost).
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r78/clusters only. Never touch igraph paths, eval scoring,
reference runners. Bench write to benchmark_100seed_r78_clusters only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>

<task>
r78-A12d: cluster machinery round 4 -- X-STAGE INTEGRATION ONLY. Rounds 2-3 (READ "## A12b"
+ "## A12c" in .project-context/research/sprint_rng_matching/r75_findings/
r76_IMPL_mincross_NOTES.md, this worktree; code UNCOMMITTED here) landed VERIFIED rank
parity AND ordering parity (dot count 5 exact on interleaved_cluster_crosstalk). The last
failing stage: final x coordinates. Round 3's `_apply_graphviz_leaf_cluster_x_ties`
post-pass regressed 4 rows (edge_label_braid, clustered_longlabel_handoffs,
kitchen_sink_hybrid_net, small_label_storm) -- it leaks beyond cluster-only rows and/or
double-applies against the A9 cluster x-constraints already in the x network simplex.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-clusters (branch r78/clusters).
PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

THE JOB:
1. REMOVE the ad-hoc x-ties post-pass. Integrate clusters into the x stage the way dot
   does (7.0.5 lib/dotgen/position.c: cluster boxes enter the aux x graph as margin/border
   constraints around member ranges) -- RECONCILING with A9's existing cluster x
   constraints (audit for double-application; one mechanism, dot's, should remain).
2. Re-verify the full chain per graph: ranks (A12b tables) -> ordering (A12c counts) ->
   rendered d_R.
3. Regression roots: edge_label_braid + small_label_storm are LABEL rows -- the cluster
   x path must be unreachable for them (wrapper classification: label-only and mixed rows
   opt OUT; assert it in tests).

GATES (before commit): rank + ordering parity PRESERVED (re-run both verifications);
rendered d_R improves on >=5/8 of the cluster probe set with ZERO regressions
(edge_label_braid, small_label_storm, nested byte-identical to develop's A10 output; 5
plain + 5 label-only + 5 igraph byte-identical); pytest tests/ -k "sugiyama or mincross or
dot_rank" -x -q green; ruff clean. KNOWN pre-existing failures (must not block): the
standard 6-item list. COMMITS ON r78/clusters AUTHORIZED AND REQUIRED on gate pass (clean
series: stage1, stage2, x-integration, tests). Then family bench --engines
classic_sugiyama --variants --max-nodes 300 --seeds 100 --seed-start 100 --workers 5
--timeout 3600 --watchdog-timeout 7200 --output-dir
/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_clusters (0 errors).
FALLBACK if x-integration cannot pass gates: commit stages 1-2 BEHIND AN INACTIVE FLAG
with their parity evidence (they are verified fact; do not lose them) + dossier the
x-stage residual precisely.

DELIVERABLES: append "## A12d: x-stage integration" to r76_IMPL_mincross_NOTES.md
(reconciliation audit, per-stage re-verification, before/after d_R, gate evidence, commit
shas, bench line). ASCII. NO AI attribution. No push/merge.
</task>
<completeness_contract>
Done = full-chain parity + gates green + committed + clean bench, OR stages 1-2 preserved
behind a flag + an x-stage dossier naming the exact constraint mismatch. Losing the
verified rank/ordering work is not an option.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r78/clusters only. Never touch igraph paths, eval scoring,
runners. Bench write to benchmark_100seed_r78_clusters only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>

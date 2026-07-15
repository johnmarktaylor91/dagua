<task>
r77-A9: graphviz dot CLUSTER MACHINERY -- the FINAL port of the fidelity campaign. Every
other graphviz sugiyama stage is landed (ordering exact; box/units; aux graph; virtual
half-width seed; edge-label structure per A8). The remaining graphviz-fidelity far-tail
rows are exactly the graphs whose benchmark DOT emits `subgraph cluster_*`
(interleaved_cluster_crosstalk, clustered_longlabel_handoffs, kitchen_sink_platform_graph,
kitchen_sink_hybrid_net, nested_cluster_label_stack, ...). A8's dossier (READ FIRST:
.project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_mincross_NOTES.md,
"## A8") names the requirement: dot cluster handling -- rank/collapse, intercluster
remapping, cluster boundary x-constraints. Also carry A8's flagged detail: label virtual
nodes need ASYMMETRIC ND_lw/ND_rw in the x-solver.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-sugiyama-final (branch r77/sugiyama-final,
HEAD 0c2dee7 -- A5+A8 committed). PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl. VERSION PIN:
`git -C /home/jtaylor/projects/_references/graphviz show 7.0.5:<path>` ONLY.

THE MACHINERY (7.0.5): lib/dotgen/cluster.c (collapse_cluster/expand_cluster/mark_clusters),
lib/dotgen/rank.c (cluster-aware ranking: clusters rank recursively, then collapse to a
single node for the parent ranking, then expand + remap), lib/dotgen/mincross.c (cluster
containment during ordering: mark_lowclusters/mark_highclusters, cluster-local crossing
passes), lib/dotgen/position.c (cluster bounding boxes + margins as x-constraints).
METHOD: DOT-content audit first (which benchmark graphs emit clusters, their nesting);
then the discriminator loop per stage: dot -v ordering counts + -Tjson rank/coords on 2
representative cluster graphs; port stage by stage (rank-collapse first -- it changes
RANKS, the upstream-most divergence); verify each stage against the reference before the
next. Gate every change to exact fidelity_mode="graphviz".

GATES (before commit):
a. Ordering discriminator >=5/6 preserved on the (non-cluster) calibration set; A5 minlen
   parity preserved.
b. d_R improves materially on >=4 of the 6 cluster far-tail rows above (benchmark path,
   values from eval_output/fidelity_definitive/r76_sugiyama_rescore.jsonl baseline, MAIN
   repo read-only); NO row leaves bit-exact/near.
c. Zero regressions: 5 previously-identical sugiyama rows + 5 label-only rows (A8 gains)
   byte-identical; igraph-mode rows byte-identical (5-row sample).
d. pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q green; ruff clean. KNOWN
   pre-existing failures (must not block): test_bench_large; classic_fcose; double-border
   smoke; test_50_node_dag; graphopt seed-matrix; test_classify_early_exit.
e. Commit on r77/sugiyama-final. No full bench (orchestrator handles it post-merge).

DELIVERABLES: append "## A9: cluster machinery" to r76_IMPL_mincross_NOTES.md (DOT cluster
audit, per-stage port log w/ 7.0.5 cites, before/after d_R on the 6 rows, gate evidence,
commit shas). ASCII. NO AI attribution. No push/merge.
</task>
<completeness_contract>
Done = gated port committed with gates green, OR an honest per-stage dossier: which cluster
stages landed, which remain, each with its 7.0.5 cite and measured residual -- that dossier
becomes the final port-in-progress disposition for cluster rows. A partial that passes
gates a/c/d with SOME cluster rows improved is COMMIT-worthy; never weaken a gate to claim
more.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r77/sugiyama-final only. Never touch igraph-mode paths, eval
scoring, reference runners. No benches.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>

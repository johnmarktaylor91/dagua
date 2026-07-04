<task>
r77-A8: graphviz sugiyama STAGES B-D -- edge labels and clusters, the LAST graphviz-family
work item. With A1 (ordering exact), A4b (box/units), A4c (aux graph), and A5 (virtual
half-width seed -- minlen parity exact, d_R improved 10/10) all landed on develop, the
remaining graphviz-fidelity far-tail rows concentrate on LABEL- and CLUSTER-heavy graphs
(from the r76 rescore: edge_label_braid d_R 0.60, moe_router_sparse 0.36,
clustered_longlabel_handoffs 0.24, interleaved_cluster_crosstalk 0.63, kitchen_sink_* --
plus the earlier per-combo table in eval_output/fidelity_definitive/r76_sugiyama_rescore
.jsonl, MAIN repo read-only). Port the missing stages.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-sugiyama-final (branch r77/sugiyama-final,
HEAD 1dbcd80 -- already MERGED to develop; keep committing here). PYTHONPATH=$PWD;
MPLCONFIGDIR=/tmp/mpl. READ FIRST: r75_findings/r76_IMPL_mincross_NOTES.md (the whole A
saga -- what is already ported).

THE MISSING GRAPHVIZ MACHINERY (pin everything via `git -C
/home/jtaylor/projects/_references/graphviz show 7.0.5:<path>`):
1. EDGE LABELS: dot converts each labeled edge into a 2-hop virtual chain through a LABEL
   NODE with the label's box dimensions (lib/dotgen/class2.c label-edge handling +
   mklabel/edge-label placement in position.c; ranksep doubling via GD_ranksep when labels
   exist). If dagua's graphviz-fidelity path does not materialize label nodes, every
   labeled graph diverges at rank/order/x simultaneously -- check
   dagua/eval/competitors/graphviz_competitor.py:_graph_to_dot() for whether edge labels
   are emitted, and mirror what the reference DOT actually contains.
2. CLUSTERS: dot's cluster machinery (lib/dotgen/cluster.c, rank.c cluster ranking,
   mincross.c cluster containment constraints, position.c cluster boxes/margins). Port the
   pieces the benchmark graphs exercise (check which benchmark graphs declare clusters and
   HOW the DOT adapter emits them -- if the adapter emits NO subgraph clusters, dagua must
   also NOT apply cluster logic in fidelity mode; mirror the DOT, not the dagua graph
   metadata).
CRITICAL FRAMING: fidelity means matching what the reference dot binary SEES AND DOES with
the benchmark DOT input. First establish for 3 representative graphs what that DOT contains
(labels? clusters?), diff dot -v/-Tjson pipeline behavior vs dagua's, THEN port only what
is actually exercised.

GATES (before commit):
a. Ordering discriminator: >=5/6 exact preserved on the calibration set; minlen parity
   preserved (A5's tables).
b. d_R improves on >=6 of an 8-row probe drawn from the label/cluster far-tail rows above;
   NO row leaves bit-exact/near.
c. Zero regressions: 5 previously-identical sugiyama rows byte-identical; igraph-mode rows
   byte-identical (5-row sample).
d. pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q green; ruff clean. KNOWN
   pre-existing failures (must not block): test_bench_large; classic_fcose; double-border
   smoke; test_50_node_dag; graphopt seed-matrix; test_engine test_classify_early_exit.
e. Commit on r77/sugiyama-final. NO full bench (one is already running for A5; the
   orchestrator will bench after merge).

DELIVERABLES: append "## A8: stages B-D (labels/clusters)" to r76_IMPL_mincross_NOTES.md
(what the benchmark DOT actually contains per graph class, what was ported w/ 7.0.5 cites,
before/after d_R table, gate evidence, commit shas). ASCII. NO AI attribution. No
push/merge.
</task>
<completeness_contract>
Done = the DOT-content audit for label/cluster graphs exists AND (gated port committed, OR
a dossier naming exactly which stage remains with its 7.0.5 cite and measured residual).
This is the final graphviz-sugiyama work item of the campaign.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r77/sugiyama-final only. Never touch igraph-mode paths, eval
scoring, reference runners. No benches.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>

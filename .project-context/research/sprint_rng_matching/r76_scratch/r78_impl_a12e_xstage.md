<task>
r78-A12e: cluster x-stage, round 5 -- FULL-CHAIN instrumented completion. JMT has ordered
this closed within the sprint. Rounds 1-4 (READ "## A12"..."## A12d" in
.project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_mincross_NOTES.md)
achieved VERIFIED rank parity + ordering parity (dot count 5 exact), preserved behind
`graphviz_enable_cluster_skeleton=False` on develop. Every x-stage attempt so far was
piecewise (post-pass ties, partial aux constraints) and traded regressions. Round 5 does
it the way every winning port this campaign worked: INSTRUMENT THE REFERENCE END-TO-END,
then mirror.

WORKTREE: create `git -C /home/jtaylor/projects/dagua worktree add
~/.claude/worktrees/dagua-xstage -b r78/xstage develop` (develop has the flagged skeleton
machinery + all sugiyama fixes). PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

METHOD:
1. INSTRUMENTED DUMP of dot's x pass WITH clusters (/tmp/gv750-xstage archive build; the
   pack/aux instrumentation patterns from earlier rounds are described in the dossiers):
   for interleaved_cluster_crosstalk + kitchen_sink_platform_graph +
   clustered_longlabel_handoffs, dump the COMPLETE aux x graph dot builds
   (position.c/ns.c: every aux node incl cluster-border nodes, every edge w/
   weight/minlen, initial ranks, solved x) with the skeleton-ordered node order.
2. Enable `graphviz_enable_cluster_skeleton=True` in a probe and dump dagua's aux x graph
   the same way. DIFF STRUCTURALLY (node sets, constraint sets, weights) -- the rounds-3/4
   failures mean dagua's aux graph differs; find every difference, not the first one.
3. Port the aux-graph construction to match; solve; verify solved x per rank against the
   dot dump; then rendered d_R.
4. Reconcile: exactly ONE cluster-x mechanism ships (remove/supersede A9's partial
   constraints for skeleton-mode rows if they double-apply).

GATES (before commit): full-chain parity on the 3 dump graphs (ranks -> ordering -> aux-x
structural match -> solved x within float tolerance); rendered d_R improves on >=6/8 of the
cluster probe set with ZERO regressions (small_label_storm/nested byte-identical under the
guard; 5 plain + 5 label-only + 5 igraph byte-identical); pytest tests/ -k "sugiyama or
mincross or dot_rank" -x -q green; ruff clean; the skeleton flag becomes DEFAULT-ON for
cluster-only graphviz-fidelity rows via the wrapper. KNOWN pre-existing failures (must not
block): the standard 6-item list. COMMITS ON r78/xstage AUTHORIZED AND REQUIRED on gate
pass. Then family bench --engines classic_sugiyama --variants --max-nodes 300 --seeds 100
--seed-start 100 --workers 4 --timeout 3600 --watchdog-timeout 7200 --output-dir
/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_xstage (0 errors).

DELIVERABLES: append "## A12e: full-chain x" to r76_IMPL_mincross_NOTES.md (aux-graph
structural diff tables, ports w/ cites, per-stage verification, before/after d_R, gate
evidence, commit shas, bench line). ASCII. NO AI attribution. No push/merge. Clean /tmp.
</task>
<completeness_contract>
Done = aux-x structural parity demonstrated on the dump graphs + gates green + committed +
clean bench, OR a dossier listing EVERY structural difference found with the dumps attached
and which one resists porting and why. Piecewise x hacks are not acceptable; the aux-graph
diff is the deliverable either way.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r78/xstage only. Never touch igraph paths, eval scoring,
runners. Bench write to benchmark_100seed_r78_xstage only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>

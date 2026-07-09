<task>
Worktree: /home/jtaylor/.claude/worktrees/dagua-native-p1. FIRST:
`git checkout r79/native && git checkout -b r79/p5-clusters`
(branch from current r79/native -- contains native_stress, hybrid_v2, layered fixes incl.
ClusterContiguousOrder/ClusterAwareXCompaction, spacing guard). Python: .venv/bin/python.

GOAL: kill the native cluster fallback. Today `dagua.layout(g, cluster_aware=True)` with
the default algorithm warns "cluster_aware=True is not yet supported for
algorithm='dagua_native'; falling back to legacy flat placement" (engine.py ~1079-1092 warn
path; ClusterAwareDriver in dagua/layout/ops/cluster_driver.py works for FR/KK/FA2/SFDP but
not the native pipelines). This is a SPRINT STOP CRITERION: "native cluster path no longer
falls back to flat".

BACKGROUND (read first): dagua/layout/ops/cluster_driver.py (the recursive
place-children-then-parent driver; child cluster becomes a rigid placeholder box sized by
compute_cluster_placement_bbox), how engine.py dispatches _layout_cluster_aware_pipeline,
and .project-context/research/sprint_clusters/DEFERRED.md if present in-tree (the
historical blocker: recursive layered subproblems hit a layered-ordering precondition).
Also the durable design doc /home/jtaylor/.claude/research/dagua/r79-native/r79_DESIGN.md.

WORK:
1. WIRE ClusterAwareDriver FOR THE NATIVE PIPELINES: register the recursive driver support
   for algorithm="dagua_native" (and native_stress): each cluster's subgraph is laid out by
   the NATIVE router (recursively -- a cluster's internal graph may be a tree/DAG/undirected
   and should route accordingly), the cluster becomes a placeholder node with its bbox
   (plus label/padding extents), the parent level lays out placeholders + free nodes with
   the same router, then children are translated into their placeholder's final position.
   Diagnose and fix the historical "layered-ordering precondition" failure for recursive
   layered subproblems (reproduce it first on transformer_block-style corpus graphs; the
   P2c ops may have removed the blocker -- verify).
2. EDGE HANDLING: inter-cluster edges attach at cluster placeholder level during parent
   solve (the driver already models this -- verify) and render across levels afterwards;
   dummy-corridor machinery must not recurse into placeholders.
3. NESTED clusters: recursion depth >= 3 must work (r79_nested_clusters_3x2x10 has 2-3
   levels; construct one deeper synthetic case in tests).
4. REMOVE the fallback warning path for the native algorithm once the driver path works;
   keep the fallback for any algorithm that genuinely lacks driver support, but the default
   must never hit it.

GATE:
- Functional: zero containment violations (every node inside its cluster's final box; every
  child box inside its parent's) on ALL clustered corpus graphs <=500 nodes (enumerate by
  tag "clustered" + graph.clusters non-null); no fallback warning emitted for the default
  algorithm (assert via warnings capture in a test).
- Quality: full scripts/r79_baseline.py --dagua-only sweep (timeout 9000): total W/T/L not
  worse than the branch point (run once at branch point FIRST to record it -- the r79/native
  head may differ slightly from the last recorded 56/8/29+8/2/5); clustered graphs
  specifically: composite must not drop >0.5 on any; report cluster_separation subterm
  before/after for all clustered graphs.
- Jitter (sigma=0.5, 8 trials) on any claimed improvement; determinism two-run identical.
- Renders: 2-panel before/after for clustered_medium_5x20, r79_nested_clusters_3x2x10, and
  the 2 worst clustered graphs into .project-context/research/r79_native/gallery_p5/
  (<=2000px, true node sizes, committed).

CONSTRAINTS: decomposable ops; class predicates only; no external layout libs in
dagua/layout/; do not modify eval scripts/corpus or frozen external store rows (restore
churn); ASCII; conventional commits; no AI attribution; COMMITS REQUIRED on gate pass
(orchestrator-git notes in AGENTS.md do NOT apply). Evidence to
.project-context/research/r79_native/P5_EVIDENCE.md. If the recursive-layered blocker
proves genuinely deep after honest attempts, ship the driver for the STRESS-ROUTED cluster
internals only (undirected cluster contents), keep layered-internal clusters on the current
flat path WITHOUT the warning removal, and document exactly what remains -- partial honest
progress beats a fake pass.
</task>

<operational_rules>
1. Any assistant message WITHOUT a tool call TERMINATES your session; final no-tool-call
   message = report, only after commits verified. 2. stdin closed. 3. Long runs in ONE exec
   call with generous timeout; first corpus import takes minutes. 4. ENOSPC -> stop, report.
</operational_rules>

<default_follow_through_policy>
Most reasonable low-risk interpretation; keep going; note choices. Containment-violations-
zero and the no-regression sweep are non-negotiable for whatever scope ships.
</default_follow_through_policy>

<completeness_contract>
Done = native cluster-aware path works (full or documented-partial per the fallback
clause), containment gate proven, sweep not worse, renders + P5_EVIDENCE.md committed on
r79/p5-clusters.
</completeness_contract>

<verification_loop>
1) Fallback-warning repro before; absence after (warnings-capture test). 2) Containment
checker output for all clustered graphs. 3) Sweep W/T/L vs branch point. 4) Jitter +
determinism. 5) ruff/pytest green; git clean.
</verification_loop>

FINAL REPORT: scope shipped (full/partial); containment results; cluster_separation
before/after table; sweep W/T/L; commit shas; residuals.

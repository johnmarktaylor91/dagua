# r80 Visual Audit: portfolio flips + routing quality (Opus)

READ-heavy, render-and-inspect. Repo: /home/jtaylor/.claude/worktrees/dagua-native
(branch r79/native, venv .venv). You render images yourself, then audit them. Zero
source-code modifications; you may write scripts/renders under /tmp/r80_visual/.

## Why you
The r80 sprint flipped 11 benchmark graphs to wins via a portfolio route (dagua's own
sfdp/neato + convergent overlap cleanup, referee-selected) and upgraded edge routing
(node avoidance, port spread, label search). Metrics say better; your job is to catch
what metrics miss. Historical bar: a prior Sonnet audit said "indistinguishable" while
Opus found 19 real departures -- be that Opus. Composite-gaming pathologies to hunt:
degenerate compaction, lasso-loop edge curls on short edges, label-node collisions,
port fan-out artifacts, cluster box violations, aspect-ratio abuse.

## Graphs (the 11 counterfactual flips + 3 routing showcases)
Flips: petersen_10, regular_3_30, regular_4_40, real_karate_34, weighted_karate_34,
weighted_clusters_3x10, planar_60, random_bipartite_60, r79_undirected_sbm_high_mix_3x30,
plus 2 lattice/grid flips you find in P13_COUNTERFACTUAL.md.
Routing showcases: citation_dag_300, clustered_medium_5x20, heavy_tail_weights_50.

## Procedure
1. For each graph: render dagua's CURRENT full drawing (layout + routed edges + labels)
   via the standard draw path at default quality, PNG <= 950px per panel. Load positions
   fresh (dagua.layout) -- do NOT reuse frozen store positions for dagua (the store
   predates nothing, but fresh proves the shipping path).
2. Render the best-external comparison from the frozen store positions
   (eval_output/r79_baseline/results.json names the best external per graph; positions/
   <graph>__<engine>.pt) with dagua's router applied (that is the honest comparison of
   full drawings).
3. Compose 2-column panels (LEFT external, RIGHT dagua), each <= 2000px longest side
   (HARD LIMIT -- pre-crop; image reads FAIL above it).
4. Inspect with maximum pickiness: enumerate departures per category (node overlap,
   edge-node crossings, edge curls/loops, label collisions, port clutter, cluster
   containment, spacing regularity, symmetry). Minimum 2 observations per graph --
   fewer means you did not look hard enough.
5. Verdict per graph: CLEAN WIN (visually better or equal, no pathologies) /
   WIN WITH ARTIFACTS (list) / VISUAL REGRESSION (metrics say win, eyes say worse).

## Output contract
/tmp/r80_visual/VISUAL_AUDIT.md (structured: per-graph verdicts + enumerated
observations + pathology list) AND full text in your final message. Keep renders in
/tmp/r80_visual/ (do not commit). ASCII only. Disk: check df first; renders ~50MB max.

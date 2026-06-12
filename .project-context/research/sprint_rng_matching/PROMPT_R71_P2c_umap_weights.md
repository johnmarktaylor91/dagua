<task>
P2c diagnosis + fix for the umap Tier-4 cluster, r71 fidelity-completion plan
(/home/jtaylor/projects/dagua). Read FIRST: .project-context/research/sprint_rng_matching/
PLAN_r71_fidelity_completion.md sections 3 (diagnosis ladder) + Appendix B; the ladder
ORDER is mandatory -- do not jump to port surgery.

EVIDENCE (from the r71 cluster table, eval_output/fidelity_definitive/r71_cluster_table.json):
classic_umap_* has 34 Tier-4 (engine,graph) combos in Mode A: 16 on weighted/multiedge
graphs (median e_rel 0.62, median disp 0.62 -- reimpl seed-cloud much TIGHTER than the
reference's) and 18 on plain graphs (median e_rel 0.047 -- barely over the calibrated
margin). HYPOTHESIS (verified signal): dagua/layout/ops/pipelines/umap.py contains ZERO
edge-weight handling while the reference adapter (dagua/eval/competitors/ -- find the
umap_graph adapter) passes weights to umap-learn. If the reference layouts respond to
edge weights and the reimpl ignores them, the weighted-graph divergence and the tight
dispersion both follow.

DO, in ladder order:
1. PARAM_MISMATCH check: confirm the reference adapter's umap-learn params mirror the
   classic_umap_* variant params (n_neighbors, min_dist, spread, epochs, init). Record.
2. WEIGHT-PATH diagnosis: trace how edge weights flow (a) graph -> reference adapter ->
   umap-learn fuzzy-simplicial-set, vs (b) graph -> classic_umap pipeline ops. Identify
   the exact divergence (likely: pipeline builds its graph affinities unweighted).
3. FIX in the pipeline (dagua/layout/ops/pipelines/umap.py + at most one ops file):
   consume edge weights the same way umap-learn does for a weighted adjacency input
   (fidelity_mode path at minimum). NO runtime delegation to umap-learn for LAYOUT
   output (feedback_no_runtime_delegation_to_reference; importing umap-learn's
   fuzzy_simplicial_set primitive follows the tsnet/sklearn precedent ONLY if the
   pipeline already does so -- check; otherwise implement weighted affinities natively).
4. VERIFY per-seed: scripts/rng_match/check_engine.py for umap if fixtures exist, else a
   direct 5-seed Procrustes comparison on 2 weighted failing graphs
   (heavy_tail_weights_50 + one weighted_clusters graph; load via
   dagua.eval.graphs.get_test_graphs, run classic_umap_default fidelity path vs the
   reference adapter at matched seeds/params). Target: weighted-graph per-seed RMSD
   drops to the unweighted-graph level (~one order of magnitude). Print before/after.
5. REGRESSION LOCK: an unweighted failing graph's RMSD must NOT regress; run
   `python -m pytest tests/ -k umap -x -q` plus any umap pipeline tests; all green.
6. Report findings as: ladder step reached, root cause, files changed, before/after
   numbers per graph. If the diagnosis lands on CHAOTIC/irreducible instead of a code
   fix, STOP after step 2 and write the evidence -- do NOT force a fix.
</task>
<completeness_contract>
Done = diagnosis recorded (even if no fix warranted); if fixed: <=3 files changed
(pipeline + at most 1 ops file + 1 test), pytest green, before/after RMSD table printed.
No commits (CC commits after review). Plain-graph cluster (e_rel ~0.05) is OUT OF SCOPE
-- margin-edge cases, separate decision.
</completeness_contract>
<action_safety>
Never invoke the reference to PRODUCE pipeline output. Touch only pipeline/ops/test
files. No benchmark reruns (CC does those).
</action_safety>
<default_follow_through_policy>
Reasonable low-risk interpretation; stop for genuine walls or if the ladder says
no-fix.
</default_follow_through_policy>

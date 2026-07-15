<task>
Diagnostic probe: why does dagua's NEW disconnected classical_mds path (per-component MDS + DLA
merge, landed today: commits ec24b05 + 8266c77 on develop) still produce far worse stress than
igraph on disconnected graphs? RESEARCH ONLY: no repo modifications, no commits; scratch in /tmp;
write only the results file below.

Repo: /home/jtaylor/projects/dagua (develop @ 288d0ca or later, read-only).
igraph source: /home/jtaylor/projects/_references/igraph (unpinned tree -- verify behavioral
claims against INSTALLED python-igraph 1.0.0 at runtime).

THE ANOMALY (from eval_output/fidelity_definitive/r75_mds_rescore.jsonl):
battery stress D vs R after the DLA port:
  parallel_cycles_4x5:            D=0.55-0.64  R=0.011   (4 disjoint 5-cycles!)
  disconnected_label_cycle_collage: D=0.58-0.60  R~=1e-5
  multi_component_80:             D=0.35-0.36  R=0.084
  disconnected_encoder_residual:  D=0.20-0.23  R=0.025
A 5-cycle embeds near-perfectly with classical MDS; igraph's post-DLA full-graph stress is 0.011
while dagua's is 0.55 -- the DLA port improved things (old D was 1.0) but something is still
structurally wrong. Note battery stress = scale-fitted normalized stress over FINITE-distance
pairs only (cross-component pairs are excluded), computed by
scripts/definitive_fidelity_analysis.py quality_metric_samples -- so PACKING placement should
barely matter; WITHIN-component geometry dominates. That makes D=0.55 on clean cycles bizarre.

DIAGNOSE on parallel_cycles_4x5 (benchmark path, seed 100 to match the bench: use
get_competitor('classic_classical_mds').layout_with_variant(graph, timeout=120, seed=100,
variant_params={'igraph_fidelity': True}) and the graph from dagua.eval.graphs get_test_graphs):
1. Load dagua's produced layout; split by component; compute PER-COMPONENT normalized stress
   (each 5-cycle against its own graph distances). Are the individual cycles laid out well
   (stress < 0.05) or badly? Also render/print coordinates of one cycle.
2. Compare against installed igraph: ig.Graph + set_random_number_generator(random.Random(100))
   + layout('mds') on the same disconnected graph; same per-component stress computation.
3. If dagua's per-component embeddings are BAD: trace where -- does the new disconnected branch
   feed each component the correct SUBMATRIX of graph distances (check
   dagua/layout/ops/pipelines/classical_mds.py, the component slicing + the sub-MDS + the
   vertex-order row mapping)? A row-order/indexing bug would scramble coordinates exactly like
   this. Check whether the final row reordering maps component-local rows back to the right
   global vertex ids (construct a 2-component toy graph with distinguishable shapes -- a
   triangle + a path -- and check node-by-node).
4. If per-component embeddings are GOOD but full-graph finite-pair stress is bad: then the
   finite-pair set must include cross-component pairs somehow (check whether the benchmark graph
   has stray edges making it CONNECTED-ish, or the distance matrix fill makes cross-pairs
   finite: graph_utils.py:347 fills unreachable with max+1 -- does the ANALYSIS distance
   computation do the same fill? If the analysis treats filled distances as finite, packing
   geometry enters the stress and igraph's DLA placement genuinely matters -> compare dagua's
   DLA placement scale vs igraph's).
5. VERDICT: name the defect (indexing bug / wrong submatrix / scaling bug / analysis-side
   finite-fill / genuine DLA placement mismatch), CONFIRMED with the decisive evidence, plus a
   minimal gated fix sketch.

OUTPUT: .project-context/research/sprint_rng_matching/r75_findings/r75_PROBE_dla_quality_RESULTS.md
-- commands, numbers, verdict, fix sketch. ASCII only. Budget ~30 min.
</task>
<default_follow_through_policy>
Most reasonable low-risk interpretation; if one path blocks, document and continue.
</default_follow_through_policy>

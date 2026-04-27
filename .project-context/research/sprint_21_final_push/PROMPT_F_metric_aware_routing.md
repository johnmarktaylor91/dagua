# Area F — Metric-aware adaptive routing

## Question

Today's dispatcher (`_choose_native_pipeline` in dagua_native.py)
picks ONE pipeline per graph based on classifier topology. The polish
op then improves the result. But the picker only tries variants of ONE
primitive (edge-equalize).

What if the dispatcher picked from the FULL set of pipelines + polish
combinations per graph, scored each by composite(full()), and returned
the best? Like multi_start_k but at the pipeline level, not just the
seed level.

## Specific evidence

For petersen_10:
- force=tree:           63.57
- force=layered_dag:    70.69
- force=hybrid:         70.69
- force=force_directed: 28.86
- force=legacy_monolith: 70.69
- auto + polish:        74.64

For hex_lattice_42:
- force=layered_dag:    85.45
- force=force_directed: 37.44
- force=planar:         42.36

The picker would always pick the right pipeline for each graph IF it
tried more than one. Currently it commits to ONE pipeline per graph
based on classifier heuristics that might be slightly wrong.

For some loss-bucket graphs, a different pipeline might score better.
disconnected_label_cycle_collage was tied at 74.41 across all
pipelines pre-polish — but maybe with a custom config (different
weights or per-component handling), one pipeline could pull ahead.

## Research targets

1. **Cost-benefit of pipeline-level multi-start**: if we ran 2-3
   pipelines per layout instead of 1, how much would total layout
   cost increase? Current baseline is ~200ms-2s per layout. Running
   2-3 pipelines + polish on each = 3-6x cost. Is this acceptable?

2. **Smart subset**: not all 6 pipelines need to run. For each
   topology class, which 2 pipelines have the highest probability of
   producing the winning layout? E.g.:
   - tree-like graphs: native_tree + native_layered_dag
   - lattices: native_layered_dag + native_force_directed (with
     post-scale)
   - multi-component cyclic: per-component + force_directed
     (proposed combination)

3. **Per-graph pipeline-class mapping**: empirically which pipeline
   wins on each of the 93 benchmark graphs (with polish enabled)?
   Construct a confusion matrix: classifier_says vs which_pipeline_wins.

4. **Failed-classifier safety**: when the classifier is wrong (e.g.
   classifies a near-tree as GENERAL because of one back-edge),
   pipeline-level multi-start would catch it.

5. **RNG-deterministic scoring concern**: the picker uses seeded
   composite(full()). Pipeline output IS deterministic. So the
   pipeline-level multi-start is fully deterministic — no flakiness.

## Output format

`.project-context/research/sprint_21_final_push/F_metric_aware_routing__<your_agent_name>.md`

Include:
- TL;DR with the recommended adaptive-routing strategy
- Cost-benefit analysis (compute overhead vs composite gain)
- Empirical pipeline-vs-graph mapping (sample at least 20 graphs across
  topology classes)
- Recommended subset of pipelines to consider per topology class
- Implementation sketch (where in the code, gate conditions)
- Risk: cases where multi-pipeline picker regresses vs current
  classifier-only routing
- Bonus: what if we adapt the polish settings per pipeline output?

## Constraints

- READ-ONLY. Findings file only.
- Read CONTEXT.md first.
- Reference `dagua_native.py:_choose_native_pipeline` and
  `build_dagua_pipeline` for the current dispatch.
- Budget: 1500-2500 words.

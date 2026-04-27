# Sprint 31 -- Hierarchical DAGs with Skip Edges

## Problem class

Honest h2h after sprint-30 cleanup shows 8 moderate/big losses. Four
share a structural pattern:

| graph | dagua | best | delta | best_engine |
|---|---:|---:|---:|---|
| mixed_width_labels | 77.58 | 84.52 | -6.94 | elk_layered |
| unet_small | 70.79 | 77.04 | -6.25 | elk_layered |
| extreme_mixed_width_transformer | 74.46 | 77.99 | -3.53 | graphviz_dot |
| hierarchical_residual_stage | 82.29 | 84.71 | -2.42 | dagre |

All four are layered DAGs where:
- Layers vary in width (narrow input / output, wide middle)
- Skip edges span multiple layers (residual connections in ML
  architectures)

dagua's gradient pipeline produces layouts where these skip edges
pass through intermediate-layer nodes' x-positions OR overlap with
shorter parallel edges. Under the fixed segments_intersect, those
geometric configurations now score as crossings (correctly). The
competitors (mostly elk_layered) avoid this via Sugiyama's
dummy-node expansion + Brandes-Koepf horizontal coordinate
refinement that explicitly minimizes long-edge horizontal extent.

## What's already in dagua

The native pipeline already has:
- Dummy-node insertion for long edges (``insert_dummy_nodes`` config,
  default True for layered DAGs)
- Brandes-Koepf horizontal coordinate refinement
  (``brandes_koepf_refine`` config, default True)
- Median-sweep + transpose ordering (``use_native_median_transpose``)

So the problem isn't missing infrastructure -- it's that the
gradient pipeline overrides or undoes the dummy-node corridor
positioning before it lands in final coordinates.

## Anti-gaming guards (MANDATORY for sprint-31 research)

This sprint follows the sprint-30 cleanup that removed 17 fixture
polishes. The user has been explicit that future polish work must
generalize. Every research proposal must satisfy:

1. **No exact-N+E signature gates.** Gate by topology *class*
   (e.g. "layered DAG with edge-span variance > X", "wide-narrow-wide
   layer pattern with hub-ratio < Y") -- never one specific graph's
   N + E + edge set.

2. **No hardcoded position/offset/rank/gap tables.** The polish
   must compute its output from ``pos`` + ``edge_index`` + a small
   set of named hyperparameters with documented meaning.

3. **No fudge constants tuned per benchmark graph.** Constants like
   ``9.5`` (densenet's slot table), ``5120`` (compound_dag's wave
   amplitude), ``5000.0`` (recurrent_feedback's pitch) are
   forbidden. If a constant comes from local optimization on one
   graph, that's a fixture not an algorithm.

4. **Validate generalization.** Provide >= 3 graphs from the same
   structural class that benefit. If you can only find one, mark
   the proposal as "do not ship; insufficient evidence of
   generalization."

5. **Jitter-validate every claimed lift.** Sigma=0.5 Gaussian on
   positions, 8+ trials. Lift that evaporates under jitter is a
   metric artifact.

6. **Honest failure recommendation accepted.** If the only
   mechanism you find is metric exploitation or you cannot identify
   a principled fix, say so directly. "No principled win" is the
   correct conclusion if it's true.

## Constraints

- Branch HEAD: ``702d7f5`` (post-cleanup)
- READ-ONLY on dagua/
- Use ``dagua.metrics.composite(dagua.metrics.full(...))`` with
  default node_sizes ``[[40, 20]] * N``
- The metric now correctly counts collinear-overlap crossings
  (segments_intersect was fixed in fe17460); any layout that scored
  high pre-fix via vertical-spine collapse will not score high now

## Output spec

Each agent writes a structured report:
- TL;DR (5 bullets max): ship/don't-ship + measured deltas
- Per-metric diagnosis on at least 2 of the 4 target graphs (not
  cherry-picked; representative)
- Algorithm sketch as Python pseudocode (~50-150 LOC)
- Empirical validation table including 3+ graphs from the class
  AND 5 protected wins (must not regress)
- Gate predicate (must be class-based, not graph-based)
- Recommended action with explicit justification

If the conclusion is "no principled fix found," that's a valid
report. Don't invent overfit polishes to fill a sprint.

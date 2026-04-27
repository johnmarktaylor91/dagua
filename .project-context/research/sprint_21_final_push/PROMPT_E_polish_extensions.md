# Area E — Polish-op extensions / new projection primitives

## Question

The sprint-20k `_best_of_polish` op (in `dagua/layout/ops/pipelines/
dagua_native.py`) tries 7 candidates of one primitive (edge-equalize:
nudge each edge endpoint toward mean edge length) and picks the best
by composite(full()). It delivered +94 net composite and 0 regressions.

What OTHER projection primitives could be added to the picker to lift
the remaining graphs?

## Specific evidence

The picker has a 0.5-margin gate, so adding new primitives is
**strictly upside** (worse picks are filtered out). Extending the
candidate set is low-risk, high-leverage IF a new primitive is found
that uniquely helps a graph the existing primitives can't.

What graphs the current polish helped (sprint-20l):
- ragged_feature_pyramid: +5.81 (edge-equalize, iters=10/0.1)
- residual_block: +5.49 (edge-equalize)
- sierpinski_42: +3.57 (edge-equalize, iters=10/0.1)
- petersen_10: +3.95 (aggressive variant 50/0.05)
- disconnected_label_cycle_collage: +2.96 (50/0.05 on tile path)
- 40+ other graphs improved by smaller amounts

What graphs polish CANNOT help (kept baseline):
- hexagonal_lattice_42 (already-uniform structure, polish breaks straightness)
- triangular_lattice_36 (similar reason)
- transformer_layer (-1.91 close-loss)
- dependency_500 (-2.90 moderate-loss)
- small_world_500 (-2.0)

## Research targets

Propose new projection primitives that could be added as polish
candidates. Each should be:

1. **Direct projection (not gradient)** — gradient is saturated,
   that's why simple constraint projection works.
2. **Single-pass or few-iteration** — the polish budget is small.
3. **Targets a specific metric component** that current edge-equalize
   doesn't touch.

Candidate primitive types to consider:

- **Layer-internal x-equalize**: within each y-layer, redistribute
  x-coordinates to be evenly spaced. Targets `edge_length_cv` and
  `crossing_rate` together.

- **Grid-snap projection**: snap to nearest integer grid (with
  optimal grid-size fit). Targets `edge_length_cv` to near-zero
  for lattice-like graphs. Risk: breaks `edge_straightness`.

- **Aspect-ratio-preserving stress polish**: like our edge-equalize
  but constraining aspect not to drift. May help dependency_500
  where the layout's aspect is locked but edges aren't uniform.

- **Crossings-aware swap**: detect overlapping edges, swap pairs of
  nodes if it reduces crossings. Targets `crossing_rate`.

- **Backbone-then-leaf smoothing**: identify the longest path
  through the graph, line up backbone nodes; then redistribute
  leaves around it. Targets `edge_straightness` AND `edge_length_cv`.

- **Force-directed-on-residual**: keep dagua's positions, but run
  10 steps of FR/SFDP on a copy and BLEND the results. Picker
  decides if blend is better.

- **Manhattan-axis snap**: snap each edge to be axis-aligned (only
  90-degree turns). Targets dag_consistency + straightness for
  layered DAGs.

## Output format

`.project-context/research/sprint_21_final_push/E_polish_extensions__<your_agent_name>.md`

Include:
- TL;DR with the 3 most promising primitives
- Per-primitive analysis:
  - Pseudocode
  - Which target graphs benefit (with quantified expected delta)
  - Cost (iter count, complexity)
  - Risk (which existing wins might regress)
- Recommended order to add to polish settings list
- A combined picker variant that uses different primitives for
  different topology classes (lattice, large-DAG, multi-component)

## Constraints

- READ-ONLY. Findings file only.
- Read CONTEXT.md first.
- Reference `dagua/layout/ops/pipelines/dagua_native.py:_equalize_edges`
  and `_best_of_polish` for the existing primitive shape.
- Budget: 2000-3000 words.

# Area D — dependency_500 escape from gradient saturation

## Question

`dependency_500` (N=500, E=1471, family=GENERAL, max_degree=53) loses by -2.90 to elk_layered. Score breakdown: edge_length_cv 0.95 vs elk 0.79 is the dominant gap (~3.2 weighted points). Gradient is saturated — w_length_variance sweep 0..200 produces identical layouts. multi_start_k=20 produces identical output. Polish primitives don't beat baseline (margin gate rejects).

Two algorithmic proposals from sprint-21 D research:

**Codex D #3 (gap-constrained layered local search):** identify long edges, run adjacent same-layer x-swaps that lower a weighted objective `crossings + alpha * gaps + beta * edge_span_cv`. 120-200 LOC. Reuse existing `dagua/layout/ops/crossing_swap.py`.

**Codex E (aspect_preserving_equalize):** locks bounding box during projection. Expected +0.5..+1.5 on dependency_500. ~30 LOC.

## Research targets

1. **Implement aspect_preserving_equalize first** (lower effort). The current `_equalize_edges` (in `dagua/layout/ops/pipelines/dagua_native.py:_equalize_edges`) lets the bounding box drift. A version that re-scales after each iter to preserve the original bbox should let the optimizer reduce CV without losing the aspect that the layered_dag pipeline carefully constructed.

2. **If (1) doesn't close enough of the gap, implement gap-constrained search.** For each layer with high local CV, run adjacent x-swaps that minimize a weighted objective. Score each candidate via composite(full(...)) and accept if better.

3. **Test empirically** on dependency_500 (target) plus protected wins:
   - random_dag_200 (was protected, now strict win)
   - org_chart_deep
   - hub_fanout_label_skew

4. **Predict realistic delta**. Even with both fixes, dependency_500 might only close to -1.0 or -1.5 (close-loss bucket) — not all the way to a win. That's still progress.

## Output

`.project-context/research/sprint_22_algo_bets/D_dependency_500_escape__<your_agent>.md`

- TL;DR
- Aspect-preserving equalize pseudocode + measured delta
- Gap-constrained search pseudocode + measured delta (if needed)
- Combined effect prediction
- Risk: large-DAG protected wins (random_dag_200, org_chart_deep)

## Constraints

- READ-ONLY in dagua/. /tmp/ scripts allowed.
- Read CONTEXT.md first.
- 2000-3500 words.
- Empirical measurements required, not estimates.

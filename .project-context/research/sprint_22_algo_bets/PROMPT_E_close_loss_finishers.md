# Area E — Close-loss tail finishers

## Question

After sprint-21b, 8 graphs sit in close-loss bucket [-2..-0.5]:

| Graph | dagua | best | comp | delta |
|---|---|---|---|---|
| small_world_500 | 52.19 | elk_layered | 54.15 | -1.96 |
| disconnected_encoder_residual | 84.01 | elk_layered | 85.63 | -1.62 |
| triangular_lattice_36 | 85.48 | graphviz_dot | 87.09 | -1.61 |
| clustered_medium_5x20 | 69.78 | graphviz_dot | 71.20 | -1.41 |
| outerplanar_dag_20 | 72.42 | igraph_sugiyama | 73.16 | -0.74 |
| multi_component_80 | 74.46 | graphviz_dot | 75.10 | -0.64 |
| hexagonal_lattice_42 | 88.35 | graphviz_dot | 88.99 | -0.63 |
| parallel_cycles_4x5 | 62.11 | graphviz_sfdp | 62.73 | -0.62 |

Three are owned by other sprint-22 areas: hex_lattice (A/B), tri_lattice (A/B), disconnected_encoder_residual (C), parallel_cycles_4x5 (A maybe), multi_component_80 (C maybe), small_world_500 (its own area), dependency_500 (D).

Remaining for this area's investigation: **clustered_medium_5x20, outerplanar_dag_20, recurrent_feedback_cell** (in TIE bucket at -0.39, very flippable).

## Research targets

For each of clustered_medium_5x20, outerplanar_dag_20, recurrent_feedback_cell:

1. Run `composite(full(...))` breakdown vs best competitor. Identify the dominant losing metric component.

2. Identify the structural reason the competitor wins. Read the cached competitor positions, look at the layout shape.

3. Propose a targeted fix:
   - Cluster-aware compression for clustered_medium_5x20 (C Codex #3).
   - Fanout angle polish for outerplanar_dag_20 (C Codex #3 secondary).
   - Back-edge micro-polish for recurrent_feedback_cell (C Codex #5).

4. Implement each in /tmp/ and measure real composite delta. Quote numbers.

5. Cluster the three — if a single new polish primitive flips multiple, that's higher leverage than three separate fixes.

## Output

`.project-context/research/sprint_22_algo_bets/E_close_loss_finishers__<your_agent>.md`

- TL;DR with three lifts and their predicted composite gain
- Per-graph: dominant losing metric, competitor strategy, fix sketch, /tmp/ pseudocode, measured delta
- Cluster recommendations (single polish op covering multiple graphs is preferred)
- Risk: protected wins to verify

## Constraints

- READ-ONLY in dagua/. /tmp/ scripts allowed.
- Read CONTEXT.md first.
- Per-graph empirical evidence required.
- 2000-4000 words for the three-graph survey.

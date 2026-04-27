# Area F — petersen_10 + small_world_500 algorithm-ceiling investigation

## Question

Two graphs in moderate-or-close-loss bucket may be at structural algorithm ceiling:

**petersen_10** (-2.72): non-planar 3-regular, 10 nodes, 15 edges. igraph_sugiyama wins by 77.36 vs dagua 74.64. Sprint-21 B Claude said "petersen +3.42 fresh measure at sprint-20l HEAD" — i.e., petersen MAY have already moved to win bucket. Verify at sprint-21b HEAD.

**small_world_500** (-1.96): elk_layered 54.15. dagua wins edge_length_cv (+15.45) but loses dag_consistency (-12.35) and edge_straightness (-6.12). The gradient pipeline imposed enough hierarchy to bring small_world_100 to a win, but at N=500 the same approach loses more on hierarchy than it gains on geometry.

## Research targets

### Petersen verification (5-min task)

Run at HEAD `c821eb6`:
```python
torch.manual_seed(0)
score = float(composite(full(layout(petersen_10, LayoutConfig(seed=42)), ...)))
```
If score > 77.36 (sugiyama), petersen is not a loss. Done; close this branch.

If score < 77.36, propose:
- B Codex #1 (exact per-layer permutation search for N <= 12) — feasible at N=10 (10!/2 = 1.8M permutations).
- B Codex #2 (spectral init using Laplacian eigenvectors {3, 1^5, -2^4}).

### small_world_500 algorithmic options

**Option A: graduated stress route.** small_world_100 wins via stress route (sprint-20i). small_world_500 currently uses layered_dag. Test forcing it through stress route and measure.

**Option B: per-layer-cap.** Hierarchical layering with a cap on layer width forces multi-row layouts on small_world graphs (which currently get one node per layer).

**Option C: hybrid — stress for x, layered for y.** Use longest-path layering for y (hierarchy preserved), stress-SGD for x (better neighborhood preservation). Should improve dag_consistency without sacrificing edge length uniformity.

For each option, implement in /tmp/, measure delta on small_world_500 + small_world_100 (verify no regression).

## Output

`.project-context/research/sprint_22_algo_bets/F_petersen_smallworld_ceiling__<your_agent>.md`

- Petersen status (loss / not loss) — quick check first
- For small_world_500: implementation sketch + measured delta for each option
- Recommended approach (single best, with predicted delta)

## Constraints

- READ-ONLY in dagua/. /tmp/ scripts allowed.
- Read CONTEXT.md first.
- Petersen verification is mandatory — if not a loss, skip the rest of B.
- 1500-3000 words.

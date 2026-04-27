# Sprint 19 Improvement Scan -- Synthesis & Action Queue

Date: 2026-04-24
Commits on branch: sprint-19a (56c3b93), sprint-19b (64731dd)
Baseline at sprint-19b: dagua mean composite 77.29 on 93 graphs (benchmark_full, n<=500).

## Scope & Process

Five areas investigated by two agents each (Claude subagent + Codex CLI).
9 of 10 reports landed; Codex B produced log output but not a file (Claude B
covered the same scope empirically so no net loss).

| Area | Theme | Claude | Codex |
|------|-------|--------|-------|
| A | Algorithm core gaps vs competitors | [area_A_algorithm_core__claude.md] | [area_A_algorithm_core__codex.md] |
| B | Per-graph loss diagnosis (10 worst) | [area_B_loss_buckets__claude.md] | (log only) |
| C | Latent-bug hunt  | [area_C_bug_hunt__claude.md] | [area_C_bug_hunt__codex.md] |
| D | Runtime profiling | [area_D_runtime__claude.md] | [area_D_runtime__codex.md] |
| E | Per-metric gap + opportunity cost | [area_E_metric_gaps__claude.md] | [area_E_metric_gaps__codex.md] |

## Convergent Findings (both agents in an area, or two areas, agree)

### Structural gap: existing ops not wired into the default pipeline

A (Claude, Codex) + E (Claude, Codex) independently identified that dagua
already implements the classic Sugiyama phases in `dagua/layout/ops/` but
the default `dagua_native.py` pipeline does not call them. Specifically:

- `InsertDummyNodes` (long-edge splitting, Sugiyama Phase 1.5) -- only used
  by the `sugiyama` pipeline.
- `BrandesKopf4Pass` coordinate assignment (at `ops/coordinate.py:1249`
  and `ops/sugiyama.py:1733`) -- unused by default; default does iterative
  gradient + barycenter sort for x.
- `TransposeHeuristic` -- registered op, never invoked.
- `LayerPromotion` -- exists, never invoked.
- `DetectComponents` -- exists in `ops/preprocess.py`, never invoked.

Impact (Claude E quantified opportunity cost across 93 graphs):
- overlap_count gap: 210 composite pts (10 iters of `OverlapProjection`
  doesn't converge on dense n>=500).
- edge_length_cv gap: 187 pts (no dummy-node splitting).
- crossing_rate gap: 142 pts (no transpose + barycenter on dummy-expanded).
- edge_straightness gap: 71 pts (no BK coordinate assignment).

Codex E projected +0.8 to +1.2 composite on dummy-node work alone.

### Topology-insensitive AspectRatioFit (Claude B, Codex E)

`AspectRatioFit` uses target=0.25 regardless of graph topology. Planar
lattices get stretched w=456 h=2640 (hex_42) or w=1320 h=5280
(sierpinski_42). Blows `edge_length_cv` by 0.17-0.48 and crushes
`angular_resolution` by 20+ deg. Affects 6+ graphs, summed opportunity
+15-25 pts. Fix: topology-aware target (e.g. 0.5-1.0 for planar, 0.3-0.5
for wide-parallel, keep 0.2-0.25 for deep chains).

### Cyclic-flat graph degeneration (Claude B, sprint-19a/b context)

`small_world_100` ends with 100 unique y-levels (one per node); the
sprint-19a gate `relayered_max <= max_layer_count` accepts effectively-
chain relayerings. Net dag_consistency 0.49-0.52. Two graphs: +10-16 pts.
Fix: tighten gate to also reject when max relayer layer count is close to
num_nodes (chain detection).

### Runtime cliff at N>100 (Claude D, Codex D)

`_project_exact` in `OverlapProjection` is O(N^2). At n=300 sparse graph,
takes 14.9s of 27.5s total (54%). `_project_sweep` already exists but is
not routed. This cliff explains why `dependency_500` didn't complete in
22 min.

Additional runtime fires:
- Exact repulsion + overlap thresholds default to 2000; on cycle_100 these
  consume 94% of runtime. Dropping to 200 gives 25-35% speedup without
  quality loss (sample_k=128 already covers pairs at N<=128).
- Crossing-loss fallback dispatch: `num_edges < 20` triggers the fast
  path. Raising to ~200 recovers 50-70% on small layered graphs.
- Tree fast path: `use_tree_fast_path` config defaults `False` but
  pipeline `getattr` assumes `True`. Never fires. 95% speedup on trees.
- `crossing_loss` rebuilds segment expansions and pair indices every
  iteration even though `edge_index` and `layers` are stable.

### Verified correctness bugs (Claude C, Codex C)

Ranked by blast radius. Each has a runnable reproducer in the source reports.

| Bug | File:line | Severity | Impact |
|-----|-----------|----------|--------|
| `dag_consistency` strict `>` counts ties and self-loops as violations | metrics.py:291-303 | HIGH | up to 25/100 per graph on ties |
| `edge_direction_straightness` asymmetric TB vs LR zero-length handling | metrics.py:480-487 | HIGH | up to 10/100 on LR graphs |
| `angular_resolution` uses global `torch.randperm`, non-deterministic | metrics.py:757 | HIGH | benchmark noise masks improvements |
| `segments_intersect` determinant sign undercounts crossings | metrics.py (count_crossings path) | HIGH | crossing_rate under-reports |
| `count_overlaps_detailed` misses adjacent-cell overlaps for N>2000 | metrics.py | HIGH | overlap_count under-reports on big graphs |
| `make_acyclic_robust` can still return cyclic on self-loops | cycle.py:196-221 | HIGH | sprint-19a edge case (partial fix applied) |
| `composite()` CPython-specific NaN handling (`max(0, nan) == 0`) | metrics.py:1166-1206 | MED | future-fragile |
| Spectral initializer non-deterministic despite docstring | init_placement.py | MED | seed-breaker |
| Vectorized x-centering off by half-slot | init_placement.py | MED | tiny positional error |
| `CrossingSwapPolish` filters edges too aggressively (disabled today) | ops/crossing_swap.py:175-194 | LOW | latent trap |
| `benchmark.py` omits `cluster_ids` when calling `full()` | eval/benchmark.py | LOW | cluster_separation scored neutral |

## Cross-Metric Trade-offs (Claude E, Codex E)

- `edge_straightness` vs `angular_resolution`: r = -0.55. Rank-alignment
  squishes angles. Brandes-Köpf alone without horizontal spread will
  regress angular.
- `edge_straightness` vs `overlap_count`: r = +0.37. Aligning stacks nodes
  into narrow bands, creating overlaps.
- `cluster_separation` vs `edge_length_cv`: r = -0.46. Spreading clusters
  lengthens inter-cluster bridges.
- `crossing_rate` vs `angular_resolution`: r = +0.54. Improving ordering
  usually helps both simultaneously.

Implication: wire `InsertDummyNodes` + `BrandesKopf4Pass` together, paired
with a post-BK overlap projection. Ship as a unit; splitting regresses.

## Action Queue (ranked by impact / effort)

### Phase 1: metric bug fixes (1-5 LOC each, ~2-4 hr total)
1. `dag_consistency` strict `>` -> `>=` and exclude self-loops (HIGH)
2. `angular_resolution` accept `seed` kwarg; pass benchmark seed through
3. `edge_direction_straightness` symmetric TB/LR zero-length clamp
4. `segments_intersect` use abs(determinant) or correct sign handling
5. `count_overlaps_detailed` expand adjacent-cell lookup
6. `benchmark.py` pass `cluster_ids` to `full()`
7. `composite()` explicit NaN -> 0 guard

### Phase 2: runtime cliffs (small code change, big impact, ~3-6 hr)
8. `OverlapProjection` route to `_project_sweep` for N>100 (40-55% speedup)
9. Exact repulsion + overlap thresholds 2000 -> 200 (25-35% on small)
10. `crossing_loss` layered fallback threshold 20 -> 200 (50-70% on small layered)
11. Tree fast path default-value bug fix
12. Cache `crossing_loss` segment expansions across iterations

### Phase 3: wire existing ops into default (~6-10 hr)
13. Insert `InsertDummyNodes` before barycenter/coordinate phase
14. Replace iterative x-gradient with `BrandesKopf4Pass` for layered graphs
15. Add `TransposeHeuristic` iteration after barycenter
16. Wire `DetectComponents` + per-component layout + tiling

### Phase 4: new logic (~6-12 hr)
17. Topology-aware AR target (hook `classify_graph()` output)
18. Family dispatch: DAG -> upgraded Sugiyama; planar -> planar-aware init
19. More `OverlapProjection` iterations + convergence check
20. Tighten cycle-reversal gate to reject near-chain relayerings
21. Strongly-connected init for flat cyclic graphs (circular instead of 2D random)

## Sprint 19c-onward plan

Sprint 19c: Phase 1 bug fixes (one commit per bug, run h2h after each)
Sprint 19d: Phase 2 runtime (unblocks dependency_500)
Sprint 19e: Phase 3 wiring (biggest metric gains)
Sprint 19f: Phase 4 topology-aware logic

Expected cumulative gain: ~+5 to +10 mean composite on top of 77.29.
Target: pull mean over 80 and eliminate all losses vs any competitor on >5 graphs.

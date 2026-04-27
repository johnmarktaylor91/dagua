# Area B -- Non-planar regular graphs (petersen_10) -- codex

Agent: codex
Date: 2026-04-25
Scope: research only; no source changes.

## TL;DR

- The remaining Petersen gap is not edge-length quality. Dagua is much better
  than `igraph_sugiyama` on edge-length CV (`0.213` vs `0.490`) and better on
  straightness, but loses the whole matchup on crossings: `crossing_rate=0.108`
  vs `0.027`. Under `composite()`, that is roughly a 7.3-point crossing swing
  against Dagua.
- The best algorithmic answer is a small-graph exact layered solver: enumerate
  or branch-and-bound cycle-removal/layer/order candidates for `N <= 12`, score
  exact crossings plus the existing composite proxy, assign coordinates with
  Brandes-Kopf-style horizontal compaction, then run the existing edge-equalize
  polish as a final candidate.
- This is better targeted than generic stress. Petersen is non-planar, so the
  optimum is not "no crossings"; it is "few unavoidable crossings while keeping
  the metric's y-axis semantics." Sugiyama wins because it constructs a layered
  DAG and minimizes crossings before compaction.
- Expected petersen_10 delta: `+3` to `+5` composite, enough to close the
  `-2.72` gap and likely land around `78-80`, depending on how much edge-length
  uniformity is lost while reducing crossings.
- Fallbacks: spectral-initialized stress for `N > 12` regular/non-planar graphs,
  and a local 2-opt anti-crossing swap polish candidate for layouts that are
  already layered but have crossing score clamped to zero.
- Generality is narrow but useful: exact enumeration should fire only on
  `petersen_10`-class graphs. The looser spectral/anti-crossing family can also
  help `regular_3_30`, `regular_4_40`, `small_world_100`, `small_world_500`,
  and possibly `real_karate_34` / `weighted_karate_34`, but those need picker
  gating because several are already wins.

## Evidence

I ran a Petersen-only version of `/tmp/score_breakdown.py` because the script's
checked-in target list does not include `petersen_10`. The result matches the
prompt:

| metric | dagua | igraph_sugiyama | direction |
|---|---:|---:|---|
| composite | 74.64 | 77.36 | dagua `-2.72` |
| dag_consistency | 1.0000 | 1.0000 | tie |
| edge_length_cv | 0.2129 | 0.4898 | dagua wins strongly |
| depth_spearman_rho | 0.9387 | 0.9813 | sugiyama small win |
| overlap_count | 0 | 0 | tie |
| edge_straightness_below_15 | 0.6000 | 0.4000 | dagua wins |
| crossing_rate | 0.1081 | 0.0270 | sugiyama wins decisively |

The composite formula gives crossing density 10 points with
`max(0, 1 - crossing_rate * 10)`. Dagua's `0.1081` is above the clamp point, so
it receives zero crossing points. Sugiyama's `0.0270` receives about `7.3`
points. Dagua claws back about `5.5` points from edge-length uniformity and some
straightness, but not enough to offset the crossing loss.

The cached igraph layout is a compact layered drawing:

```text
0: ( 50,   0)
1: (  0,  50)
2: (100, 100)
3: (100, 150)
4: (100, 200)
5: ( 50,  50)
6: (  0, 150)
7: (200, 150)
8: ( 50, 200)
9: ( 75, 250)
```

That is not a symmetric Petersen drawing. It is metric-aware: every edge points
downward enough to score `dag_consistency=1.0`, layers are monotone enough for
high depth Spearman, and the within-layer ordering keeps crossings low. This
confirms the prompt's hypothesis: the important "secret" is not global force
layout quality, it is the discrete Sugiyama phase that chooses a favorable
acyclic orientation, layering, and per-layer order.

External source check: igraph documents `layout_sugiyama` as a layered DAG
algorithm whose within-layer x coordinates are chosen by a Sugiyama crossing
minimization heuristic; for cyclic or undirected inputs, igraph first tries to
eliminate cycles and assign layers automatically, with no quality guarantee
([igraph C manual](https://igraph.org/c/html/main/igraph-Layout.html#igraph_layout_sugiyama)).
The current igraph source comments list the same high-level phases: approximate
feedback arc set, layer assignment, dummy nodes, Sugiyama ordering, and
Brandes-Kopf horizontal coordinate assignment
([source](https://github.com/igraph/igraph/blob/main/src/layout/sugiyama.c)).

## Recommended algorithm

Primary recommendation: add an exact small-graph crossing-aware layered solver,
used as a candidate in the native/polish picker for `N <= 12` regular
non-planar graphs.

The solver should be deterministic and deliberately metric-aware:

1. Build a small set of acyclic orientations. For `N <= 12`, brute force is
   plausible, but full `10!` vertex-order enumeration is unnecessary. Start
   with Eades-style feedback arc candidates, reverse-order candidates, BFS
   roots, eccentricity roots, and all automorphism-safe rotations if available.
2. For each orientation, assign layers with longest-path or Coffman-Graham-like
   width controls. Keep layer counts that preserve `dag_consistency=1.0` and
   high depth correlation.
3. Insert dummy nodes only for crossing estimation and coordinate assignment,
   not in the final graph. This mirrors why Sugiyama works on long edges:
   ordering sees the segment structure instead of treating every multi-layer
   edge as a single diagonal.
4. Enumerate per-layer permutations exactly. On Petersen's observed igraph
   split, the hard layers are small; the product of factorials is tiny compared
   with `10!`. Where a layer is wide, use branch-and-bound with exact crossing
   count as the bound.
5. Score candidates by an internal objective aligned to `composite()`:
   crossings first, then edge-length CV, then straightness/depth. The final
   selection can call `composite(full(...))` just like `_best_of_polish`.
6. Assign x coordinates with the existing Brandes-Kopf-style coordinate
   machinery if available, or a compact median/barycenter coordinate pass.
7. Feed the result into the existing edge-equalize polish settings and keep the
   best by the same `+0.5` margin gate.

This is not a general-purpose replacement for native layout. It is an exact
rescue route for the one class where Dagua's continuous optimizer is saturated:
small non-planar regular graphs where a few discrete x-order swaps dominate the
score.

Expected delta on `petersen_10`: reducing crossing rate from `0.108` to
`0.027` is worth about `+7.3` raw composite points before tradeoffs. If the
solver loses half of Dagua's edge-length advantage (`-2.5` to `-3`) and a small
amount of straightness/depth (`-0.5` to `-1`), the net is still `+3` to `+5`.
That closes the `-2.72` gap with margin.

## Fallbacks

Fallback 1: spectral-initialized stress for larger regular/non-planar graphs.
For `N > 12`, exact enumeration becomes the wrong tool. Use normalized
Laplacian spectral coordinates, but treat them as initialization only. Petersen
and related regular graphs have highly degenerate eigenspaces, so ARPACK's
arbitrary basis is not reliable by itself; rotate/sample bases in the eigenspace
and choose the one with best edge-length CV and crossing proxy. Then run a short
stress/edge-equalize refinement and let the picker compare it to the native
layout.

Expected effect: low confidence on Petersen under the directed composite,
because a symmetric circular drawing may lose dag/depth points even if it looks
canonical. More useful for `regular_3_30` and `regular_4_40`, where exact
layer-order search is infeasible and preserving graph symmetry can improve
edge-length and angular components.

Fallback 2: local 2-opt anti-crossing swap polish. For layouts that already
have `dag_consistency >= 0.98`, `overlap_count = 0`, and crossing score clamped
to zero, collect crossing edge pairs and try same-layer or near-layer adjacent
x-order swaps. Accept only swaps that improve `composite(full(...))`. This is
cheaper and less ambitious than the exact solver; it will not discover a new
layer assignment, but it can recover the final 1-3 points when the layering is
good and the x order is the only bad part.

## Detection gate

Exact solver gate:

```text
N <= 12
E <= 3N
connected
no clusters
degree_cv == 0 or tags contains "regular"
min_degree >= 3
planarity check is false, or graph is tagged famous/non-planar
current composite < 78 OR current crossing_rate >= 0.06
current dag_consistency >= 0.95
```

The `current dag_consistency` condition matters. If the graph is already being
scored as a layered/directed object, the rescue should preserve that semantic
axis. A canonical circular Petersen drawing might be prettier to a human, but
it can throw away the 25-point DAG consistency term and lose under the current
benchmark.

Looser fallback gate for spectral/stress:

```text
12 < N <= 100
degree_cv <= 0.20
edge_to_node_ratio between 1.3 and 2.5
not tree-like
not DAG-heavy
crossing_rate or edge_length_cv is the dominant loss metric
```

Looser fallback gate for 2-opt anti-crossing:

```text
N <= 200
E <= 400
overlap_count == 0
dag_consistency >= 0.95
crossing_rate >= 0.06 or crossing score == 0
```

All three should be candidates, not unconditional routes. The final arbiter
should remain `composite(full(...))` with the existing positive-margin picker.

## Generality

Direct exact-solver beneficiaries:

| graph | tags observed locally | current status |
|---|---|---|
| `petersen_10` | `famous`, `regular`, `small` | direct target; `-2.72` vs `igraph_sugiyama`; crossing dominated |

Related regular/non-planar graphs:

| graph | tags observed locally | expected effect |
|---|---|---|
| `regular_3_30` | `regular`, `sparse` | exact solver too expensive; spectral/stress or anti-crossing only. Prior sprint notes indicate Dagua is already ahead, so gate carefully. |
| `regular_4_40` | `regular` | same as above; likely already a Dagua win, so picker-only. |

Adjacent cyclic/symmetric graphs that may benefit from the fallbacks:

| graph | tags observed locally | expected effect |
|---|---|---|
| `small_world_100` | `cyclic`, `small-world` | not regular, but crossing/order dominated; stress route already helped in sprint-20i. Anti-crossing swaps may add a small lift. |
| `small_world_500` | `cyclic`, `small-world` | close-loss class per context; exact search impossible, spectral/stress and local crossing polish are the plausible tools. |
| `real_karate_34` / `weighted_karate_34` | `community`, `real-world`, `social` | not a regular detector target; spectral/stress can help symmetry/community separation but risks hurting DAG metrics. |

I would not make "non-planar regular" a broad dispatcher family. The exact
algorithm is a small-graph exception. The scalable idea is "crossing-aware
candidate in the picker," not "all regular graphs go to Sugiyama."

## Risk

The main regression risk is metric mismatch. Symmetric graph drawings often
look good but score poorly under `composite()` because `dag_consistency`,
`depth_spearman`, and edge straightness encode a vertical hierarchy. A spectral
Petersen drawing can improve visual symmetry while losing the benchmark. That
is why the primary solver remains layered and metric-aware.

The second risk is runtime creep. `10! / 2` is feasible for one graph, but
normalizing this into a general path would be a mistake. Keep the exact solver
at `N <= 12`; use branch-and-bound and stop once a candidate beats the current
layout by a safe margin.

The third risk is overfitting to Petersen. The mitigation is simple: expose the
exact solver only as a scored candidate behind a narrow structural gate. If it
does not beat the current layout by `+0.5`, return the current layout unchanged.

## Implementation order

1. Add a small exact crossing counter and per-layer permutation search for
   `N <= 12`. Test only on `petersen_10` and one planar small regular control
   where it must not fire.
2. Wire it as a named candidate in the post-layout picker, not as a replacement
   for `dagua_native`.
3. Add the cheaper 2-opt anti-crossing candidate for `N <= 200, E <= 400`.
4. Only after those are measured, try spectral/stress as a fallback for
   `regular_3_30`, `regular_4_40`, and small-world graphs.

## Assumptions

- I treated the prompt's `dagua=74.64` as authoritative for the sprint-20l
  target. My local Petersen-only breakdown reproduced that exact number.
- I did not assume competitor availability at runtime. The `score < 0.95 *
  best competitor` idea is useful for benchmark analysis, but production
  dispatch needs topology and self-metric gates.

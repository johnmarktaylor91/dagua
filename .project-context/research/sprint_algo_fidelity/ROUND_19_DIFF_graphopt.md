# Round 19 GraphOpt Adversarial Diff

Scope: diagnosis only. Compared Dagua GraphOpt against
`/home/jtaylor/projects/_references/igraph/src/layout/graphopt.c` line by line, plus
igraph random initialization in `layout_random.c`.

## 1. Overall Flow

igraph `igraph_layout_graphopt()`:

- Allocates pending force vectors once: `pending_forces_x`, `pending_forces_y`
  at `graphopt.c:349-361`.
- If `use_seed` is false, calls `igraph_layout_random(graph, res)` at
  `graphopt.c:369-370`; if `use_seed` is true but the matrix shape is invalid,
  warns and also randomizes at `graphopt.c:363-368`.
- Runs exactly `niter` iterations with `for (i = niter; i > 0; i--)` at
  `graphopt.c:373-428`.
- Every iteration zeroes force vectors at `graphopt.c:380-382`, optionally
  applies all-pairs electrical repulsion at `graphopt.c:384-413`, applies every
  edge spring at `graphopt.c:415-423`, then moves nodes at `graphopt.c:425-427`.

Dagua:

- `build_graphopt_pipeline()` creates `FixedSteps`, `ValidateGraphOptInputs`,
  `GraphOptInitializePositions`, `GraphOptPrepareState`, `Repeat(n=niter,
  [ZeroForces(), GraphOptIteration()])`, and `GraphOptFinalizePositions` at
  `dagua/layout/ops/pipelines/graphopt.py:82-96`.
- `GraphOptPrepareState` precomputes edge and pair buffers once at
  `dagua/layout/ops/force.py:1295-1329`.
- `GraphOptIteration` internally starts with `forces = torch.zeros_like(positions)`
  at `dagua/layout/ops/force.py:1435-1437`, so the pipeline's preceding
  `ZeroForces()` is behaviorally redundant for this specific op.

Verdict: the iteration ordering matches igraph. Dagua adds preprocessing and
finalization around the loop. `ZeroForces()` is dead/redundant for GraphOpt
because `GraphOptIteration` ignores prior `state.forces`.

## 2. Coulomb Repulsion Formula

igraph:

- Constant: `#define COULOMBS_CONSTANT 8987500000.0` at `graphopt.c:29`.
- Directed magnitude: `COULOMBS_CONSTANT * ((node_charge * node_charge) /
  (distance * distance))` at `graphopt.c:144-145`.
- Applied only when `distance != 0.0` and `distance < 500.0` at
  `graphopt.c:392-401`.

Dagua:

- Constant: `_GRAPHOPT_COULOMBS_CONSTANT = 8_987_500_000.0` at
  `dagua/layout/ops/force.py:27`.
- Main GraphOpt path computes `magnitude = _GRAPHOPT_COULOMBS_CONSTANT *
  (node_charge * node_charge) / pair_distance_sq` at
  `dagua/layout/ops/force.py:1452-1456`.
- Uses the cached cutoff from `GraphOptPrepareState`, default `500.0`, squared at
  `dagua/layout/ops/force.py:1234-1244` and `dagua/layout/ops/force.py:1326-1329`.
- Applies when `(distance_sq > _GRAPHOPT_MIN_DISTANCE) & (distance_sq <
  max_repulsion_distance_sq)` at `dagua/layout/ops/force.py:1442-1446`, with
  `_GRAPHOPT_MIN_DISTANCE = 1.0e-12` at `dagua/layout/ops/force.py:28`.

Verdict: formula and cutoff are aligned except for the zero-distance predicate.
igraph excludes only exactly zero distance; Dagua excludes distances with
`distance_sq <= 1e-12`, i.e. distances up to `1e-6`. This is tiny but real.

## 3. Spring / Hooke Attraction Formula

igraph:

- Distance is Euclidean at `graphopt.c:77-82` and used for each edge at
  `graphopt.c:209-210`.
- Exact zero-distance edges return without spring force at `graphopt.c:216-218`.
- Displacement is `abs(distance - spring_length)` at `graphopt.c:220-223`.
- Directed force is `-1 * spring_constant * displacement` at `graphopt.c:224`.
- Direction/sign is delegated to `igraph_i_determine_spring_axal_forces()` at
  `graphopt.c:230-232`; that helper flips sign for `distance < spring_length` at
  `graphopt.c:186-189` and halves each component at `graphopt.c:190-191`.

Dagua:

- Main GraphOpt spring path computes edge distance at
  `dagua/layout/ops/force.py:1464-1465`.
- Masks out `distance <= _GRAPHOPT_MIN_DISTANCE` at
  `dagua/layout/ops/force.py:1466`.
- Computes `stretch = abs(distance - spring_length)` and
  `magnitude = 0.5 * spring_constant * stretch` at
  `dagua/layout/ops/force.py:1470-1471`.
- Uses `source_sign = -1` when `distance > spring_length`, else `+1`, at
  `dagua/layout/ops/force.py:1479-1483`.
- Applies contribution to source and negative contribution to target at
  `dagua/layout/ops/force.py:1484-1486`.

Verdict: formula and half-force behavior are aligned. The same tiny
zero-distance predicate divergence exists: igraph skips only exact zero;
Dagua skips `distance <= 1e-12`.

## 4. Force Vector Direction Conventions

igraph:

- `igraph_i_determine_electric_axal_forces()` treats the force on `this_node` as
  away from `other_node`; components are initially negative under the assumption
  that `other_node` is above/right of `this_node` at `graphopt.c:93-124`, then
  polarity flips when the assumption is false at `graphopt.c:126-133`.
- Electrical application adds this force to `this_node` and subtracts it from
  `other_node` at `graphopt.c:152-155`.

Dagua:

- Repulsion uses `delta = positions[pair_source] - positions[pair_target]` at
  `dagua/layout/ops/force.py:1442`, `direction = delta / distance` at
  `dagua/layout/ops/force.py:1451`, then adds to source and subtracts from target
  at `dagua/layout/ops/force.py:1457-1459`.
- For pair `(this_node, other_node)` from igraph's loop, Dagua's `source` equals
  `this_node`, so `delta` is the vector away from `other_node`, matching igraph.
- Spring direction uses the same source-minus-target delta at
  `dagua/layout/ops/force.py:1464`, then sign inversion for stretched springs at
  `dagua/layout/ops/force.py:1479-1484`, matching igraph's negative directed
  force and half-force convention.

Verdict: direction conventions match.

## 5. Newton's Second Law Step and `max_sa_movement`

igraph:

- The movement model is explicitly `displacement = force / mass` at
  `graphopt.c:247-259`.
- Each axis is clamped independently to `[-max_sa_movement, max_sa_movement]` at
  `graphopt.c:267-279`.
- Positions are updated in-place by x/y movement at `graphopt.c:281-282`.

Dagua:

- `GraphOptIteration` computes `movement = torch.clamp(forces / node_mass,
  min=-max_sa_movement, max=max_sa_movement)` at
  `dagua/layout/ops/force.py:1488-1492`.
- Positions are updated with `positions + movement` at
  `dagua/layout/ops/force.py:1493-1494`.
- The standalone `GraphOptApplyDisplacement` does the same at
  `dagua/layout/ops/force.py:1223-1230`, but the graphopt pipeline does not use
  it; it uses the fused `GraphOptIteration`.

Verdict: movement and per-axis clamping match.

## 6. Edge Weight Handling

igraph:

- `igraph_layout_graphopt()` accepts no weights parameter at `graphopt.c:341-347`.
- Spring force is called with only endpoint ids, `spring_length`, and
  `spring_constant` at `graphopt.c:416-422`.
- There is no weight lookup or multiplier anywhere in `graphopt.c`.

Dagua:

- Public pipeline accepts optional `edge_weights` at
  `dagua/layout/ops/pipelines/graphopt.py:112` and validates/stores it at
  `dagua/layout/ops/pipelines/graphopt.py:162-175`.
- `ValidateGraphOptInputs` also validates optional weights at
  `dagua/layout/ops/init.py:293-303`.
- `GraphOptPrepareState` filters and stores `problem.edge_weights` as
  `graphopt_spring_weights` at `dagua/layout/ops/force.py:1309-1317`.
- `GraphOptIteration` multiplies spring magnitude by those weights at
  `dagua/layout/ops/force.py:1472-1478`.

Verdict: divergence. igraph GraphOpt has no per-edge weight semantics; Dagua
does if weights are supplied. With `edge_weights is None`, behavior matches.
For fidelity mode, GraphOpt should ignore weights or the public wrapper should
reject them.

## 7. RNG Semantics for Initial Layout

igraph:

- GraphOpt calls `igraph_layout_random()` when no valid seed matrix is supplied
  at `graphopt.c:363-370`.
- `igraph_layout_random()` resizes to `[vcount, 2]` at `layout_random.c:44-48`.
- It fills by column first, then vertex: `for (j = 0; j < 2; j++)` and
  `for (i = 0; i < vcount; i++)` at `layout_random.c:50-53`.
- Coordinates are sampled from `RNG_UNIF(-1, 1)` at `layout_random.c:52`.

Dagua:

- `GraphOptInitializePositions` seeds Python's `random.Random(problem.seed)` at
  `dagua/layout/ops/init.py:515`.
- It fills row-major, per vertex then dimension, at
  `dagua/layout/ops/init.py:516-519`.
- Coordinates are sampled with `rng.random()`, i.e. `[0, 1)`, and converted to
  float64 at `dagua/layout/ops/init.py:520-522`.
- Pipeline docstring says `seed` is for `random.Random` at
  `dagua/layout/ops/pipelines/graphopt.py:125-126`.

Verdict: divergence. Dagua currently uses Python MT19937, row-major draw order,
and `[0, 1)` range. igraph uses igraph's active RNG, column-major draw order,
and `[-1, 1]` range. Round 16 may have aligned defaults, but this source still
does not match igraph's random initializer. Even if both systems receive the
same numeric seed, Python `random.Random`, torch, NumPy, and igraph RNG streams
should not be expected to produce identical sequences.

## 8. Self-Loops Handling

igraph:

- The edge loop processes every edge from `edge = 0` to `no_of_edges - 1` at
  `graphopt.c:416-423`.
- For a self-loop, `this_node == other_node`, so `igraph_i_apply_spring_force()`
  computes `distance == 0.0` and returns at `graphopt.c:209-218`.
- Self-loops therefore do not affect force, but they are still visited.

Dagua:

- `GraphOptPrepareState` explicitly filters self-loops with
  `non_self = edges[0] != edges[1]` at `dagua/layout/ops/force.py:1299-1300`.
- If all edges are self-loops, it stores an empty spring edge tensor at
  `dagua/layout/ops/force.py:1301-1303`.
- Comment says self-loops are ignored while duplicate and reciprocal edges are
  preserved at `dagua/layout/ops/force.py:1305`.

Verdict: force result matches for exact self-loops because igraph returns early.
Dagua avoids visiting them, so only tracing/performance differs.

## 9. Hyperparameter Alignment Table

Round 16 already verified defaults match; source confirmation:

| Parameter | igraph source | Dagua source | Status |
| --- | --- | --- | --- |
| `niter` | Original default documented as 500 at `graphopt.c:317-321` | `niter: int = 500` at `graphopt.py:27-28` and `graphopt.py:105-106` | Aligned |
| `node_charge` | Original default documented as 0.001 at `graphopt.c:322-323` | `node_charge: float = 0.001` at `graphopt.py:28-29` and `graphopt.py:106-107` | Aligned |
| `node_mass` | Original default documented as 30 at `graphopt.c:324-325` | `node_mass: float = 30.0` at `graphopt.py:29-30` and `graphopt.py:107-108` | Aligned |
| `spring_length` | Original default documented as zero at `graphopt.c:326-327` | `spring_length: float = 0.0` at `graphopt.py:30-31` and `graphopt.py:108-109` | Aligned |
| `spring_constant` | Original default documented as one at `graphopt.c:328-329` | `spring_constant: float = 1.0` at `graphopt.py:31-32` and `graphopt.py:109-110` | Aligned |
| `max_sa_movement` | Original default documented as 5 at `graphopt.c:330-332` | `max_sa_movement: float = 5.0` at `graphopt.py:32-33` and `graphopt.py:110-111` | Aligned |
| Coulomb constant | `8987500000.0` at `graphopt.c:29` | `8_987_500_000.0` at `force.py:27` | Aligned |
| Repulsion cutoff | `< 500.0` at `graphopt.c:401` | `500.0`, squared at `force.py:1244` and `force.py:1326-1329` | Aligned, aside from zero-distance epsilon |

## 10. Ranked Fix List

1. Initial layout range and draw order: change GraphOpt initialization to match
   igraph's `[-1, 1]` and column-major fill (`layout_random.c:50-53` vs
   `init.py:516-519`). This is the largest unavoidable coordinate divergence.
2. RNG engine semantics: decide whether fidelity requires igraph-like RNG
   streams or just deterministic Dagua starts. Same seed will not match while
   Dagua uses Python `random.Random` (`init.py:515`) and igraph uses `RNG_UNIF`
   (`layout_random.c:52`).
3. Edge weights: igraph GraphOpt does not support weights (`graphopt.c:341-347`,
   `graphopt.c:416-422`), while Dagua multiplies spring force by weights
   (`force.py:1472-1478`). Ignore or reject weights in the fidelity path.
4. Zero-distance predicates: igraph skips only exact zero for repulsion and
   spring (`graphopt.c:401`, `graphopt.c:216-218`); Dagua skips a small epsilon
   range (`force.py:1444-1446`, `force.py:1466`). This is small but line-level
   divergent.
5. Final dtype: igraph stores `igraph_real_t` matrix values, typically double;
   Dagua finalizes to float32 at `postprocess.py:456`. This can create tiny
   output differences after otherwise matching float64 iteration.
6. Remove redundant GraphOpt `ZeroForces()` from the pipeline or document it as
   no-op for `GraphOptIteration`; the fused op zeroes its own force tensor at
   `force.py:1435-1437`.

## 11. Recommended Round 20 Fix Scope

Recommended Round 20 should be narrow and testable:

- Update `GraphOptInitializePositions` to sample `[-1, 1]` and fill positions in
  igraph's column-major order.
- Add regression tests for first initialized coordinates under a fixed seed,
  making the expectation explicit for Dagua's selected RNG engine.
- Decide and encode edge-weight fidelity: reject `edge_weights` for GraphOpt or
  ignore them in `GraphOptPrepareState`.
- Change GraphOpt's exact-fidelity distance masks from epsilon checks to exact
  nonzero checks if parity is the priority over numerical guardrails.
- Consider preserving float64 in `GraphOptFinalizePositions` for fidelity tests,
  or explicitly accept float32 finalization as Dagua API behavior.

Do not retune the confirmed hyperparameters.

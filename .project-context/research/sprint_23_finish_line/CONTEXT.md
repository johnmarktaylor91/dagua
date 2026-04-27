# Sprint 23 -- Squeeze Every Drop

## Mandate

JMT directive: "Plz do another full sprint addressing everything you just
mentioned. Don't leave a single drop unsqueezed or anything on the
table."

## State at HEAD = sprint-22e (`fd1f200`)

Bucket distribution (deterministic seed=0 scoring, 93 graphs):

```
WIN strong (>+5):        40  (43%)
WIN modest (+0.5..+5):   41  (44%)
TIE (-0.5..+0.5):         6  (6%)
close LOSS (-2..-0.5):    5  (5%)
moderate LOSS (-5..-2):   1  (1%)
big LOSS (<-5):           0  (0%)

best-or-tied: 87/93 = 94%
competitive:  92/93 = 99%
```

**The single non-competitive graph remaining:**

| Graph | dagua | best | comp | delta |
|---|---|---|---|---|
| petersen_10 | 74.64 | igraph_sugiyama | 77.36 | -2.72 |

**The 5 close-losses (-2..-0.5):**

To be re-measured at sprint-22e HEAD; sprint-22b CONTEXT estimates:

| Graph | likely delta | known target |
|---|---|---|
| small_world_500 | -1.96 | elk_layered 54.15 |
| triangular_lattice_36 | ~-0.48 (was -1.61, +1.13 from sprint-22c) | graphviz_dot 87.09 |
| clustered_medium_5x20 | -1.41 | graphviz_dot 71.20 |
| outerplanar_dag_20 | -0.74 | igraph_sugiyama 73.16 |
| multi_component_80 | -0.64 | graphviz_dot 75.10 |
| hexagonal_lattice_42 | -0.63 | graphviz_dot 88.99 |

## Sprint-22 inventory (what landed)

- 22a (1ee12d7): back-edge-aware relayer (5-graph multi-flip on
  cyclic targets; small_world_100/500, recurrent_feedback_cell,
  braided_feedback_tails, parallel_cycles_4x5)
- 22b (83fdd51): global-depth align polish for multi-component DAGs
  (disconnected_encoder_residual -1.62 -> +0.56 strict win)
- 22b (da58b14): deterministic composite() metric (seed=0 fixed)
- 22c (205ce1b): dot-mimic LP polish (triangular_lattice_36 -1.61
  -> -0.48; close-loss but didn't quite flip)
- 22d (88c7343): tutte_cyclic_planar polish (parallel_cycles_4x5
  -0.62 -> +2.63 strict win vs sfdp)
- 22e (fd1f200): gap-validated layer swaps (dependency_500 -2.91
  -> -1.92; closed but still close-loss)

## Bigger algorithmic bets for sprint-23

These are the structural fixes that sprint-22 polish couldn't reach.
Each bet is paired with explicit empirical predictions; ship only
candidates that survive contact with the picker margin gate.

### Bet A: Network-simplex x-coordinate for non-planar 3-regular

**Target:** petersen_10 -2.72 (the single non-competitive graph).

**Algorithm:** GKNV93 IEEE TSE 19(3) section 4.2 -- a network simplex
that minimizes weighted sum of edge x-spans subject to per-layer
adjacency separations. This is exactly what graphviz_dot uses
internally and what igraph_sugiyama implements. Our sprint-22c
dot_lattice_lp uses HiGHS LP for the x-step but only on DAGs;
generalizing to non-planar 3-regular requires (a) feedback arc set
removal, (b) layer assignment by Coffman-Graham or longest-path on
the residual DAG, (c) median crossing reduction on the dummy-expanded
graph, (d) network-simplex x with NSE-specific weights for the
back-edge dummies.

**Prediction:** +3..+5 composite (flips petersen_10).

**LOC budget:** ~250-350. 80 of those overlap with sprint-22c; the
real new code is FAS removal + non-planar dummy expansion.

**Risk:** the gate (non-planar 3-regular) is narrow but petersen
isn't the only graph in this class. If sprint-23 adds heawood_14,
moebius_kantor_16, mcgee_24 etc. to the suite, we want them to
benefit too. Should the gate fire on any 3-regular graph? Need
empirical envelope.

### Bet B: Lattice-aware grid-snap for hex/tri/square

**Target:** hexagonal_lattice_42 -0.63, triangular_lattice_36 -0.48.

**Diagnosis:** sprint-22c dot_lattice_lp uses default scipy linprog
HiGHS tolerance + minimization objective. graphviz_dot tightens this
with (a) integer-grid x positions via the network-simplex branch-and-
bound step (not pure LP relaxation), (b) explicit per-layer "tight"
constraints that pull adjacent nodes to integer-grid lines, (c) edge-
length CV is implicitly minimized via the integer-grid quantization,
not as a primary LP objective.

**Prediction:** +0.5..+1.5 composite each on hex_42, tri_36.
Possibly +2..+3 on grid_5x5 if its current win margin is small.

**LOC budget:** ~150 if implemented as a tightening of sprint-22c's
existing LP; ~300 if implemented as a separate quantization pass.

**Risk:** generic LP tightening might regress on graphs where the
HiGHS-default LP currently scores well (the gate already filters
non-DAG; lattice-specific tightening should be opt-in).

### Bet C: Long-edge-aware Sugiyama ordering for dense DAGs

**Target:** dependency_500 -1.92, possibly clustered_medium_5x20
-1.41, outerplanar_dag_20 -0.74.

**Diagnosis:** sprint-22e gap_validated_layer_swaps was a tactical
patch over a structural gap. D Codex's research note explicitly
flagged the deeper bet: dummy-node expansion for long edges, then
barycenter / median ordering with transpose phase, then weighted
Sugiyama coordinate assignment with gap penalty. Our current
layered_dag pipeline does dummy expansion but its ordering pass is
weaker than dot's median-with-transpose.

**Prediction:** +1.5..+2.5 on dependency_500 (closes it),
incidental +0.5..+1.5 on clustered_medium_5x20 and other dense
DAGs.

**LOC budget:** ~400-500 if implemented as a new ordering op +
ordering-aware coordinate phase. ~150 if implemented as an additional
median-transpose pass on the existing dummy-expanded graph (lower
expected lift but cheap).

**Risk:** ordering changes can affect edge crossings on graphs that
currently win on crossing rate. The picker margin gate handles
regression risk if the new ordering is added as a polish candidate
rather than a forced override.

### Bet D: Spectral-x + depth-y for non-planar lattices and 3-regular

**Target:** small_world_500 -1.96, possibly hex_42 if Bet B doesn't
close it.

**Diagnosis:** B Codex's sprint-22 note included a spectral_x_depth_y
sketch (only partially implemented as sprint-22c). The eigen-vectors
of the unsigned Laplacian give a 1D embedding that minimizes total
edge-span on connected graphs; combined with longest_path_layering for
y, this can outperform the gradient pipeline on graphs where the
Laplacian's dominant eigenvector is well-aligned with the natural
horizontal axis.

**Prediction:** +1..+2 on small_world_500. Possibly +0.5..+1 on
clustered_medium_5x20.

**LOC budget:** ~120 (one eigsh call + depth-warp + tiebreak).

**Risk:** spectral methods can produce weird outputs on disconnected
graphs (eigenvalues degenerate). Gate must check connectedness.

### Bet E: Outerplanar / multi-component finishers

**Target:** outerplanar_dag_20 -0.74, multi_component_80 -0.64.

**Diagnosis:** these are tiny graphs where the gradient pipeline is
already close to ceiling but missing the last ~1 point because the
outer-face / component-tile-permutation isn't optimal. sprint-21a
overlap_jitter and sprint-22b global_depth_align both touched this
area; remaining gap is small enough that targeted picker-safe
permutation search should close them.

**Prediction:** +0.5..+1 each (flips both close-losses to ties).

**LOC budget:** ~80 each, just narrowly-gated permutation candidates.

**Risk:** very low; tiny graphs with small search space.

### Bet F: Metric-noise residual cleanup

**Target:** all close-losses where the delta is within composite
metric noise (sampled_crossing_rate, neighborhood_preservation).

**Diagnosis:** sprint-22b da58b14 fixed the seed but some metrics
still have small variance from sampling. Increasing crossing_samples
from 1M to 5M (or switching to exact crossing count for N <= 200)
might reveal that some close-losses are actually ties.

**Prediction:** unmeasured; could re-classify 1-2 close-losses as
ties without any algorithm change.

**LOC budget:** ~30.

**Risk:** breaks scoring reproducibility against the historical
benchmark unless the new scoring runs as a separate metric.

## Research questions per area

Each area gets dispatched to BOTH a Codex agent AND a Claude
sub-agent (per global CLAUDE.md dual-dispatch rule for research).
Areas A and C are the highest-leverage; B and D are tactical
finishers; E and F are insurance.

Per-area prompts assembled in PROMPT_*.md files in this directory.

## Timeline

- Round 1 (research, parallel x6 areas x 2 agents = 12 dispatches):
  ~30-60 min wall clock.
- Round 2 (synthesis, single CC pass): ~15 min.
- Round 3 (implementation, per-bet committed individually):
  ~2-3 hours wall clock.
- Round 4 (final h2h + test suite): ~30 min.

## Success criteria

- Best-or-tied: 94% -> >= 96% (88/93+ graphs win or tie).
- Competitive: 99% -> 100% (all 93 within 2 points of best).
- Petersen_10 specifically flipped to win or tie.
- Test suite green; no regressions on sprint-22 wins.

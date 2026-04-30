# Round 21 Diff: classic_stress_sgd vs sgd2

Diagnosis-only adversarial diff for the dagua `classic_stress_sgd` family against
the reference `s_gd2` package (`sgd2` competitor). No source changes were made.

## 1. Files read

### Dagua implementation and wiring

- `dagua/layout/ops/stress_sgd.py`
  - Constants and state keys: lines 23-39.
  - Config dataclasses: lines 41-118.
  - Distance helpers and connectivity: lines 120-211.
  - Schedule helpers: lines 214-313.
  - Pivot/approximation helpers: lines 316-449.
  - Exact term builder and pair sampler: lines 452-537.
  - Pair update and disconnected fallback: lines 540-625.
  - `InitializeStressSGDState`: lines 628-719.
  - `PrepareStressSGDTerms`: lines 722-803.
  - `RunStressSGDExactSchedule`: lines 806-913.
  - `RunStressSGDApproximateSchedule`: lines 916-1069.
- `dagua/layout/ops/pipelines/stress_sgd.py`
  - Pipeline constructor: lines 28-84.
  - Public pipeline entrypoint: lines 87-180.
- `dagua/eval/variants.py`
  - stress_sgd variants and reference param mapping: lines 737-779.
  - stochastic/heavy metadata: lines 1820-1888.
- `dagua/eval/competitors/classic_competitor.py`
  - `_ClassicBase` variant dispatch and seed default: lines 26-97.
  - `classic_stress_sgd` spec default `steps=300`: lines 169-173.
  - `ClassicStressSGD.layout` direct wrapper: lines 852-911.
  - `_quick_classic` generic wrapper, including edge-weight forwarding: lines 1570-1627.
- `dagua/eval/competitors/sgd2_competitor.py`
  - Availability import: lines 18-31.
  - Reference edge preprocessing: lines 34-80.
  - MDS helper distances: lines 83-131.
  - `SGD2` adapter: lines 134-242.
  - `SGD2MDS` adapter: lines 245-319.
- `dagua/eval/competitors/base.py`
  - Competitor seed contract: lines 26-91.
  - Runtime seed env helper: lines 96-119.
- `tests/test_edge_weights_adapters.py`
  - Current tests documenting sgd2 symmetrized edge-weight summation: lines 161-173.
  - Current tests documenting weighted condensed distances: lines 175-182.

### Existing analysis

- `eval_output/fidelity_report/report.md`
  - Stress-SGD current verdicts: lines 88-91.
  - Report methodology summary: lines 109-120.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md`
  - Sprint context and stochastic-floor lesson: lines 17-25 and 111-130.
  - Accepted residual context: lines 157-198.

### Reference implementation

- Installed package metadata:
  - `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/s_gd2-1.8.1.dist-info/METADATA`
    lines 1-9: package name/version/homepage.
  - `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/s_gd2-1.8.1.dist-info/RECORD`
    lines 1-17: installed Python wrapper and native `_layout` extension.
- Installed Python wrapper:
  - `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/s_gd2/__init__.py`
    lines 1-2.
  - `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/s_gd2/s_gd2.py`
    lines 15-31 (`layout`), 69-89 (`layout_sparse`), 92-123 (`mds_direct` and
    default schedule), 129-162 (seed and initialization).
  - `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/s_gd2/swig/layout.py`
    lines 138-188: SWIG-exposed native calls.
- Reference source cloned read-only to `/tmp/s_gd2_src` from the package homepage:
  - `/tmp/s_gd2_src/README.md` lines 27-43: API semantics.
  - `/tmp/s_gd2_src/cpp/s_gd2/layout.hpp` lines 11-19 and 25-60: public C++
    entrypoints, `term` representation, and helper declarations.
  - `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp` lines 17-90: SGD update, shuffle, stress.
  - `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp` lines 92-165: unweighted graph build and BFS terms.
  - `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp` lines 168-265: weighted graph build and Dijkstra terms.
  - `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp` lines 268-325: schedules.
  - `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp` lines 327-340: convergent SGD start.
  - `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp` lines 427-483: public native entrypoints.
  - `/tmp/s_gd2_src/cpp/s_gd2/sparse.cpp` lines 17-74: sparse SGD update and shuffle.
  - `/tmp/s_gd2_src/cpp/s_gd2/sparse.cpp` lines 527-572: sparse schedule and native entrypoints.
- Graphviz lineage cross-check:
  - `/home/jtaylor/projects/_references/graphviz/lib/neatogen/sgd.c` lines 17-26
    (stress), 29-36 (shuffle), 39-132 (adjacency), 142-257 (main SGD).
  - `/home/jtaylor/projects/_references/graphviz/lib/neatogen/sgd.h` lines 11-23
    (`term_sgd` and `graph_sgd` storage).

## 2. Overall pipeline structure

### Dagua

Dagua exposes `layout_stress_sgd_pipeline(...)` in
`dagua/layout/ops/pipelines/stress_sgd.py:87-180`. It constructs a `LayoutProblem`
with `edge_index`, `num_nodes`, optional `node_sizes`, optional `edge_weights`, and
`seed` at lines 156-162, applies the pipeline at lines 166-172, and returns either
the final tensor or `(tensor, traces)` at lines 174-180.

The actual pipeline is assembled in `build_stress_sgd_pipeline` at
`dagua/layout/ops/pipelines/stress_sgd.py:28-84`:

1. `BuildAdjacency(weighted=True, dedup="min", format="list", directed=False)` at
   lines 65-72.
2. `InitializeStressSGDState` at line 73.
3. `PrepareStressSGDTerms` at line 74.
4. `RunStressSGDExactSchedule` at line 75.
5. `RunStressSGDApproximateSchedule` at lines 76-81.

The exact/approximate switch is dagua-only for this competitor: `PrepareStressSGDTerms`
chooses exact terms when `num_nodes <= max_exact_nodes` at
`dagua/layout/ops/stress_sgd.py:777-787`; otherwise it builds pivot distances at
lines 789-803. The public default cutoff is 10,000 nodes
(`dagua/layout/ops/pipelines/stress_sgd.py:24-31`, `:91-97`).

### Reference `s_gd2`

The benchmark adapter imports `s_gd2` and calls `s_gd2.layout(...)` at
`dagua/eval/competitors/sgd2_competitor.py:193-220`. The installed Python wrapper
defines `layout(I, J, V=None, t_max=30, eps=0.01, random_seed=None, init=None)` at
`.../site-packages/s_gd2/s_gd2.py:15-31`. Its flow is:

1. Reject empty or mismatched edge lists at lines 19-20.
2. Resolve seed via `_check_random_seed` at lines 22-23.
3. Initialize `X` via `random_init` at line 25.
4. Call native `layout_unweighted` when `V is None` at lines 27-28, or
   `layout_weighted` when weights are supplied at lines 29-30.
5. Return the mutated NumPy array at line 31.

The native source mirrors this: `layout_unweighted` builds BFS stress terms,
builds the schedule, and calls `sgd` in `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:427-431`;
`layout_weighted` builds Dijkstra stress terms, schedule, and calls `sgd` at
lines 434-438.

### Adapter-level structure difference

The dagua-vs-reference comparison is not a direct call with identical input:

- Dagua `classic_stress_sgd` receives the original `graph.edge_index` and optional
  `graph.edge_weights` through `_quick_classic` at
  `dagua/eval/competitors/classic_competitor.py:1604-1619`.
- Reference `sgd2` first expands every edge into both directions, removes self-loops,
  unique-sorts directed pairs, and sums weights across duplicate directed pairs in
  `_symmetrized_unique_edges` at `dagua/eval/competitors/sgd2_competitor.py:53-80`,
  then calls `s_gd2.layout` and multiplies coordinates by 100.0 at lines 220-221.

That adapter preprocessing is a real semantic layer and explains several residuals
below.

## 3. Energy / loss / objective

### Shared exact objective

Both exact implementations target stress over all unordered node pairs:

`E(X) = sum_{i<j} w_ij * (||x_i - x_j|| - d_ij)^2`, with `w_ij = 1 / d_ij^2`.

Reference source:

- `calculate_stress` computes `stretch = d_ij - sqrt(dx*dx + dy*dy)` and accumulates
  `w_ij*stretch*stretch` in `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:74-89`.
- The same stress formula appears in the Graphviz lineage implementation at
  `/home/jtaylor/projects/_references/graphviz/lib/neatogen/sgd.c:17-26`.
- The C++ term stores `double d, w` in `/tmp/s_gd2_src/cpp/s_gd2/layout.hpp:25-30`.

Dagua source:

- Exact terms are built for upper-triangle pairs in `_build_exact_terms` at
  `dagua/layout/ops/stress_sgd.py:486-507`.
- For each pair, dagua sets `distances[write_index] = graph_distance` and
  `weights[write_index] = 1.0 / (graph_distance * graph_distance)` at
  `dagua/layout/ops/stress_sgd.py:501-505`.
- The pair update uses the same stress-SGD closed-form move as reference, not a
  PyTorch autograd loss; see `dagua/layout/ops/stress_sgd.py:584-595`.

### Weighted vs unweighted distance objective

Reference `s_gd2.layout` has two modes:

- `V is None`: native `layout_unweighted`, which builds graph distances by BFS in
  `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:117-165`; terms have `d_ij` as hop count and
  `w_ij = 1/(d_ij*d_ij)` at lines 148-154.
- `V is not None`: native `layout_weighted`, which builds graph distances by Dijkstra
  in `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:204-265`; terms have weighted shortest
  path `d_ij` and `w_ij = 1/(d_ij*d_ij)` at lines 238-241.

Dagua chooses the same weighted-vs-unweighted distance family based on whether
`problem.edge_weights is not None` at `dagua/layout/ops/stress_sgd.py:681-685`.
`_graph_distances` dispatches BFS vs Dijkstra at lines 162-191. The exact term
builder then uses those distances at lines 493-505.

### Objective divergences

1. Dagua exact terms store `distances` and `weights` in `float32` arrays at
   `dagua/layout/ops/stress_sgd.py:488-491`; reference stores terms as doubles in
   `/tmp/s_gd2_src/cpp/s_gd2/layout.hpp:25-30`. The update casts the `float32`
   values to Python float at `dagua/layout/ops/stress_sgd.py:902-904`, so the
   optimization math is double-ish but the objective parameters have already been
   rounded.
2. Dagua adapter-side weighted multiedges are deduped by minimum weight because the
   pipeline builds adjacency with `dedup="min"` at
   `dagua/layout/ops/pipelines/stress_sgd.py:65-72`. Reference adapter sums duplicate
   symmetrized weights at `dagua/eval/competitors/sgd2_competitor.py:72-78`, and
   tests lock in that summing behavior at `tests/test_edge_weights_adapters.py:161-173`.
   The native reference itself rejects asymmetric duplicate weights at
   `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:184-197`, but the dagua adapter hides that by
   pre-summing.
3. Dagua large-graph approximate mode is not the same objective as reference
   `s_gd2.layout`. Dagua switches to pivot-distance approximation at
   `dagua/layout/ops/stress_sgd.py:789-803`, whereas the reference adapter calls
   exact `s_gd2.layout`, not `layout_sparse`, at
   `dagua/eval/competitors/sgd2_competitor.py:220`. The reference package does have
   sparse APIs (`s_gd2.layout_sparse`) at `.../site-packages/s_gd2/s_gd2.py:69-89`
   and `/tmp/s_gd2_src/cpp/s_gd2/sparse.cpp:555-572`, but they are not used by the
   `sgd2` competitor.

## 4. Force / gradient computation

Neither side uses a global force accumulation in exact mode. Both perform sequential
pair updates after shuffling terms each epoch.

Reference pair update in `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:17-62`:

- Seed RandomKit at lines 23-24.
- For each eta, shuffle terms at lines 26-33.
- For each term, compute `mu = min(eta * w_ij, 1)` at lines 43-46.
- Compute displacement vector and magnitude at lines 48-49.
- Compute `r = (mu * (mag-d_ij)) / (2*mag)` at line 52.
- Move `i` by `-r * delta` and `j` by `+r * delta` at lines 56-59.

Dagua pair update in `dagua/layout/ops/stress_sgd.py:559-596`:

- Compute `mu = min(eta * weight, 1.0)` at line 584.
- Compute `dx`, `dy`, and `math.hypot` magnitude at lines 585-587.
- Return early on zero magnitude at lines 588-589.
- Compute `ratio = mu * (magnitude - target_distance) / (2.0 * magnitude)` at line 591.
- Move source and target exactly like reference at lines 592-595.

The sign is aligned: reference uses `dx = X[i]-X[j]`, `r=(mag-d)/(2mag)`, then
`X[i]-=r*dx`, `X[j]+=r*dx` at `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:48-59`; dagua uses
the same `dx`, ratio, and subtraction/addition at `dagua/layout/ops/stress_sgd.py:585-595`.

Residual divergence: reference does not guard `mag == 0` in exact `sgd`; division by
zero would produce non-finite values if two initialized points coincide exactly
(`/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:48-56`). Dagua skips zero-distance terms at
`dagua/layout/ops/stress_sgd.py:587-589`. With random continuous initialization this
is almost never hit, but it is a behavioral difference for user-supplied identical
initial positions if dagua later adds `init`.

## 5. Initialization

### Reference

`s_gd2.layout` calls `random_init(I, J, random_seed, init)` at
`.../site-packages/s_gd2/s_gd2.py:25`. `random_init` computes `n = max(max(I), max(J)) + 1`
at lines 156-162 and delegates to `_random_init`. `_random_init`:

- Accepts `init` directly after shape checks at lines 135-145.
- Otherwise calls `np.random.seed(random_seed)` at line 146.
- Draws `np.random.rand(n, 2)` for 2D at lines 147-148.

The wrapper resolves `random_seed=None` by drawing from global NumPy via
`np.random.randint(65536)` at `.../site-packages/s_gd2/s_gd2.py:129-132`.

### Dagua

Dagua exact mode seeds the module-level NumPy RNG in `InitializeStressSGDState`:
`np.random.seed(problem.seed)` at `dagua/layout/ops/stress_sgd.py:713-717`. Exact mode
then draws `positions = np.random.rand(num_nodes, 2)` at
`dagua/layout/ops/stress_sgd.py:876-879`.

The public pipeline default seed is 42 at
`dagua/layout/ops/pipelines/stress_sgd.py:87-97`. The classic competitor default also
resolves `seed=None` to 42 through `_ClassicBase._layout_seed` at
`dagua/eval/competitors/classic_competitor.py:29-42`, and forwards it to the layout
function through `_quick_classic` at lines 1613-1619.

### Initialization divergences

1. Dagua has no `init` parameter in `layout_stress_sgd_pipeline`
   (`dagua/layout/ops/pipelines/stress_sgd.py:87-98`), while reference supports
   user-supplied `init` (`.../site-packages/s_gd2/s_gd2.py:15`, `:135-145`).
2. Reference infers `n` from max edge endpoint in `random_init` at
   `.../site-packages/s_gd2/s_gd2.py:156-162`. The dagua adapter passes an explicit
   `graph.num_nodes` at `dagua/eval/competitors/classic_competitor.py:1613-1618`.
   The reference adapter avoids missing isolated-node issues only indirectly; it
   returns zeros when the edge list is empty at `dagua/eval/competitors/sgd2_competitor.py:207-211`,
   but if a graph has isolated nodes plus some edges, `s_gd2.random_init` creates
   coordinates only through max endpoint. Dagua always returns shape `[graph.num_nodes, 2]`.
3. Dagua disconnected fallback uses `torch.randn` scaled by 10.0 at
   `dagua/layout/ops/stress_sgd.py:598-625` and lines 699-705. Reference errors for
   disconnected graphs in native BFS/Dijkstra when term count is incomplete
   (`/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:159-162`, `:259-262`), but the adapter catches
   exceptions and returns `pos=None` at `dagua/eval/competitors/sgd2_competitor.py:225-232`.

## 6. Iteration / convergence

### Fixed schedule

Reference `layout` uses a fixed number of epochs `t_max`, default 30:

- Python default: `t_max=30, eps=0.01` at
  `.../site-packages/s_gd2/s_gd2.py:15`.
- README documents the same API at `/tmp/s_gd2_src/README.md:27-31`.
- Native schedule computes `eta_max = 1.0 / w_min`, `eta_min = eps / w_max`,
  `lambda = log(eta_max/eta_min)/(t_max-1)`, and
  `eta_t = eta_max * exp(-lambda*t)` in
  `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:268-288`.

Dagua exact mode:

- Pipeline function default is `steps=30, eps=0.01` at
  `dagua/layout/ops/pipelines/stress_sgd.py:87-97`.
- Classic competitor spec overrides the default to `steps=300` at
  `dagua/eval/competitors/classic_competitor.py:169-173`.
- Variants align dagua `steps` to reference `t_max` and `eps` at
  `dagua/eval/variants.py:737-779`.
- `_schedule_from_weights` implements the same `1/w_min`, `eps/w_max`, exponential
  schedule at `dagua/layout/ops/stress_sgd.py:275-313`.

### Convergence

Reference `s_gd2.layout` does not run a convergence test; it always consumes `t_max`
etas. Reference `layout_convergent` exists at `.../site-packages/s_gd2/s_gd2.py:34-66`
and uses native `layout_*_convergent`, but the dagua `sgd2` adapter never calls it
(`dagua/eval/competitors/sgd2_competitor.py:220`).

Dagua exact mode also has no stress-based convergence test; it runs the full schedule
and marks `state.converged = True` at `dagua/layout/ops/stress_sgd.py:891-913`.
The `state.converged` checks at lines 858-861 are pipeline-control guards, not
optimization convergence criteria.

### Iteration divergences

1. Reference native shuffle mutates the `terms` vector itself each epoch
   (`/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:63-72`). Dagua shuffles an index vector
   over fixed term arrays (`dagua/layout/ops/stress_sgd.py:886-895`). For a given
   sequence of random swap indices, these are equivalent in traversal order. They
   are not bit-identical because the RNGs differ.
2. Reference RandomKit shuffle uses `rk_interval(i, &rstate)` at
   `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:63-72`. Dagua uses NumPy
   `RandomState.shuffle` on an integer array at `dagua/layout/ops/stress_sgd.py:891-895`.
3. Reference native code does not special-case `t_max <= 1`; schedule divides by
   `t_max-1` at `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:280`. Dagua handles `steps == 1`
   in `_schedule_from_weights` at `dagua/layout/ops/stress_sgd.py:299-313`.

## 7. Hyperparameter alignment table

| Parameter / behavior | Dagua default / value | Reference default / value | Match? | Evidence |
|---|---:|---:|:---:|---|
| Public algorithm | `layout_stress_sgd_pipeline` | `s_gd2.layout` | Y | Dagua entrypoint `dagua/layout/ops/pipelines/stress_sgd.py:87-180`; reference wrapper `.../s_gd2/s_gd2.py:15-31`. |
| Default epochs in bare API | `steps=30` | `t_max=30` | Y | Dagua `dagua/layout/ops/pipelines/stress_sgd.py:91`; reference `.../s_gd2/s_gd2.py:15`. |
| Default epochs in benchmark base `classic_stress_sgd` | `steps=300` | `t_max=30` unless variant overrides | N for base, Y for paired variants | Dagua spec `dagua/eval/competitors/classic_competitor.py:169-173`; variants map `steps` to `t_max` at `dagua/eval/variants.py:737-779`. |
| `eps` default | `0.01` | `0.01` | Y | Dagua `dagua/layout/ops/pipelines/stress_sgd.py:24,95`; reference `.../s_gd2/s_gd2.py:15`. |
| Schedule formula | `1/w_min` to `eps/w_max` exponential over `steps` | same over `t_max` | Y | Dagua `dagua/layout/ops/stress_sgd.py:299-313`; reference `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:268-288`. |
| Stress weight | `1/d^2` | `1/d^2` | Y | Dagua `dagua/layout/ops/stress_sgd.py:503-505`; reference `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:151-154`, `:238-241`. |
| Unweighted distances | BFS | BFS | Y | Dagua `dagua/layout/ops/stress_sgd.py:120-138`, `:184-185`; reference `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:117-165`. |
| Weighted distances | Dijkstra | Dijkstra | Y | Dagua `dagua/layout/ops/stress_sgd.py:141-159`, `:187-191`; reference `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:204-265`. |
| Pair set | all `i<j` connected pairs | all `i<j` connected pairs | Y exact | Dagua `dagua/layout/ops/stress_sgd.py:493-505`; reference `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:127-154`, `:214-241`. |
| Pair order per epoch | shuffled | shuffled | Y conceptually, N bitwise | Dagua `dagua/layout/ops/stress_sgd.py:886-895`; reference `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:30-33`, `:63-72`. |
| Shuffle RNG | NumPy global `np.random` MT19937 | RandomKit `rk_state` | N | Dagua `dagua/layout/ops/stress_sgd.py:713-717`, `:891-895`; reference `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:23-24`, `:68`. |
| Initial position RNG | NumPy global `np.random.rand` | NumPy global `np.random.rand` | Y | Dagua `dagua/layout/ops/stress_sgd.py:876-879`; reference `.../s_gd2/s_gd2.py:146-148`. |
| Initial coordinate scale | `[0,1)` | `[0,1)` | Y | Same refs as previous row. |
| Output scale in benchmark | raw dagua coordinates | adapter multiplies by `100.0` | N | Dagua returns final tensor at `dagua/layout/ops/pipelines/stress_sgd.py:174-180`; reference adapter scales at `dagua/eval/competitors/sgd2_competitor.py:220-221`. Procrustes largely removes scale, but non-Procrustes metrics may not. |
| Edge preprocessing | `BuildAdjacency(... dedup="min", directed=False)` | adapter symmetrizes, unique-sorts, sums duplicate weights | N | Dagua `dagua/layout/ops/pipelines/stress_sgd.py:65-72`; reference adapter `dagua/eval/competitors/sgd2_competitor.py:53-80`. |
| Self-loops | ignored by adjacency builder behavior implied by no distance term; exact builder never creates `i==j` | adapter removes before native call; native ignores self-loops | Y in effect | Reference adapter `dagua/eval/competitors/sgd2_competitor.py:57-63`; native `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:104-110`, `:184-192`. |
| Empty graph | zeros `[0,2]` / `[N,2]` depending `num_nodes <= 1` and disconnected fallback | adapter zeros when `num_nodes` 0/1 or no edges | Mostly N for edgeless `N>1` | Dagua `dagua/layout/ops/stress_sgd.py:686-711`; reference adapter `dagua/eval/competitors/sgd2_competitor.py:197-211`. |
| Disconnected graph | random normal fallback, marked converged | native errors; adapter returns error/None | N | Dagua `dagua/layout/ops/stress_sgd.py:696-711`; reference native `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:159-162`, `:259-262`; adapter catch `dagua/eval/competitors/sgd2_competitor.py:225-232`. |
| Large graph exactness | exact only up to 10,000 nodes, then pivot approximation | exact `layout` for all adapter calls up to max_nodes | N | Dagua cutoff `dagua/layout/ops/stress_sgd.py:777-803`; reference adapter max_nodes 50,000 and calls exact at `dagua/eval/competitors/sgd2_competitor.py:138-140`, `:220`. |
| Trace support | optional `trace_every` | no trace in `layout` | N (extra feature) | Dagua traces at `dagua/layout/ops/stress_sgd.py:540-557`, `:907-911`; reference API `.../s_gd2/s_gd2.py:15-31`. |
| `init` support | none | supported | N | Dagua signature `dagua/layout/ops/pipelines/stress_sgd.py:87-98`; reference `.../s_gd2/s_gd2.py:15`, `:135-145`. |
| Convergent variant | none in this pipeline | `layout_convergent` available but unused | N/A for benchmark | Reference `.../s_gd2/s_gd2.py:34-66`; adapter uses `layout` at `dagua/eval/competitors/sgd2_competitor.py:220`. |

## 8. Edge cases

### Self-loops

Reference adapter removes self-loops after symmetrization:
`non_self_mask = sources != targets` at `dagua/eval/competitors/sgd2_competitor.py:57-63`.
Native reference also skips `i == j` in unweighted and weighted graph builders at
`/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:104-110` and `:184-192`.

Dagua exact term builder only creates pairs with `target_index > source_index` at
`dagua/layout/ops/stress_sgd.py:493-505`, so no self-pair stress term exists. Any
self-loop only matters if `BuildAdjacency` lets it affect connectivity; the stress
pipeline requests undirected list adjacency at
`dagua/layout/ops/pipelines/stress_sgd.py:65-72`. I did not read `BuildAdjacency`
internals in this round, so self-loop parity is inferred from the pipeline request
and pair-term construction, not proven down to adjacency insertion.

### Multi-edges

This is the highest-risk semantic mismatch.

Reference adapter:

- Symmetrizes all edges at `dagua/eval/competitors/sgd2_competitor.py:57-58`.
- Keeps weights aligned at lines 65-70.
- Unique-sorts directed pairs at line 73.
- Sums duplicate weights via `np.add.at` at lines 75-78.
- Test coverage expects reverse duplicate weights 2.0 and 3.0 to become weight 5.0
  in both directions at `tests/test_edge_weights_adapters.py:161-173`.

Native reference expects symmetric duplicate weights to agree; otherwise it throws
`"graph edge lengths not symmetric"` at `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:184-197`.
The adapter's summing strategy is therefore an adapter-specific compatibility layer.

Dagua pipeline asks `BuildAdjacency` to deduplicate with `"min"` at
`dagua/layout/ops/pipelines/stress_sgd.py:65-72`. If a graph has parallel weighted
edges or asymmetric reverse weights, dagua will use the minimum edge length for
shortest paths, while the reference adapter will sum directed duplicates before
passing `V`. Because `V` is interpreted by reference as edge length
(`/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:180-192`, `:251-254`), summed duplicate weights
can substantially increase shortest-path distances relative to dagua's min.

### Disconnected components

Dagua checks connectivity using BFS from node 0 at `dagua/layout/ops/stress_sgd.py:194-211`
and returns a deterministic Gaussian fallback when disconnected at lines 696-711.
That fallback uses `torch.Generator(device="cpu")`, `manual_seed(seed)`, and
`torch.randn(...)*10.0` at lines 622-625.

Reference native exact mode requires every source to reach all later nodes; otherwise
it throws at `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:159-162` or `:259-262`. The adapter
catches and returns an error result at `dagua/eval/competitors/sgd2_competitor.py:225-232`.
For edgeless graphs, the adapter special-cases zeros at lines 197-211.

This means disconnected benchmark rows may compare a dagua random fallback against a
missing reference result, or may be filtered by the harness. If cached reference
positions exist for disconnected graphs, the semantics are not native `s_gd2.layout`
semantics.

### Weighted edges

Dagua and reference both treat edge weights as graph distances for shortest paths in
the stress target metric. Dagua sets `weighted = problem.edge_weights is not None` at
`dagua/layout/ops/stress_sgd.py:681-685` and dispatches Dijkstra at lines 162-191.
Reference adapter passes `V=edge_weights.tolist()` at
`dagua/eval/competitors/sgd2_competitor.py:218-220`, and native `layout_weighted`
uses Dijkstra at `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:434-438`.

Mismatch: Dagua's adapter path forwards raw `graph.edge_weights` through
`_quick_classic` at `dagua/eval/competitors/classic_competitor.py:1607-1619`, while
reference adapter preprocesses and sums symmetrized weights at
`dagua/eval/competitors/sgd2_competitor.py:53-80`.

### Empty graph

Reference adapter returns zeros for `num_nodes == 0`, `num_nodes == 1`, or no
non-self edges at `dagua/eval/competitors/sgd2_competitor.py:197-211`.

Dagua returns zeros for `num_nodes <= 1` at `dagua/layout/ops/stress_sgd.py:686-692`.
For `num_nodes > 1` and no usable edges, `_is_connected` will be false and dagua
returns random Gaussian fallback at lines 696-711. Therefore edgeless `N>1` is a
clear edge-case mismatch: reference zeros, dagua random normal scaled by 10.

### Isolated nodes with some edges

Reference `s_gd2.random_init` determines `n` from max edge endpoint at
`.../site-packages/s_gd2/s_gd2.py:156-162`, so isolated high-index nodes are not
represented unless the adapter injects edges. The adapter does not inject edges; it
passes `sources.tolist(), targets.tolist()` at
`dagua/eval/competitors/sgd2_competitor.py:220`. Dagua uses explicit `graph.num_nodes`
at `dagua/eval/competitors/classic_competitor.py:1613-1618` and
`dagua/layout/ops/pipelines/stress_sgd.py:156-162`. This can produce shape or
missing-node divergence on graphs with isolated trailing nodes.

## 9. Numerical precision

Reference:

- Python initialization produces NumPy `float64` by default with `np.random.rand`
  at `.../site-packages/s_gd2/s_gd2.py:146-148`.
- Native coordinates are `double* X` in `/tmp/s_gd2_src/cpp/s_gd2/layout.hpp:11-19`.
- Native terms store `double d, w` in `/tmp/s_gd2_src/cpp/s_gd2/layout.hpp:25-30`.
- Native BFS and Dijkstra compute `w_ij` as double at
  `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:151-154` and `:238-241`.
- Native updates use `double` for eta, mu, dx, dy, magnitude, and displacement at
  `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:29-59`.
- The adapter converts final coordinates to `torch.float32` and scales by 100 at
  `dagua/eval/competitors/sgd2_competitor.py:220-221`.

Dagua:

- Initial positions are NumPy `float64` from `np.random.rand` at
  `dagua/layout/ops/stress_sgd.py:876-879`.
- Exact term arrays store sources/targets as `int32`, distances and weights as
  `float32` at `dagua/layout/ops/stress_sgd.py:486-491`.
- `_apply_pair_update` computes Python floats using `math.hypot` at
  `dagua/layout/ops/stress_sgd.py:584-595`.
- Final positions are cast to `float32` at `dagua/layout/ops/stress_sgd.py:910`.
- Approximate pivot distances are `float32` at lines 403-415, converted back to
  `float64` before approximate use at lines 991-1001.

Residual numerical differences:

1. `float32` term storage in dagua exact mode can perturb `d_ij`, `w_ij`, `w_min`,
   `w_max`, and all etas. This is zero for small unweighted integer hop distances
   that fit exactly, but nonzero for weighted shortest paths.
2. Summation order is minimal in exact SGD because there is no global sum in the
   update, but shortest-path tie order may differ between dagua Python Dijkstra and
   reference C++ priority queue. Reference uses `std::priority_queue<edge,...>` at
   `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:219-255`; dagua uses its `dijkstra_distances`
   helper via `dagua/layout/ops/stress_sgd.py:141-159`, read only at the call site
   this round.
3. Final benchmark comparison sees both sides as `torch.float32`, but reference has
   already multiplied coordinates by 100.0 at
   `dagua/eval/competitors/sgd2_competitor.py:220-221`.

## 10. RNG semantics

The answer to the specific question: no, dagua's torch seed does not produce the
same sequence as reference's RNG. In exact connected mode dagua does not use torch
for the main layout RNG; it seeds NumPy, while reference uses NumPy for initialization
and RandomKit for epoch shuffles.

Detailed sequence:

Reference:

1. Python wrapper resolves seed at `.../site-packages/s_gd2/s_gd2.py:129-132`.
2. Python wrapper seeds NumPy and draws initial positions at lines 146-148.
3. Native `sgd` seeds RandomKit with the same integer at
   `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:23-24`.
4. Native shuffle uses `rk_interval` at lines 63-72.

Dagua exact mode:

1. `_ClassicBase._layout_seed` resolves `None` to 42 at
   `dagua/eval/competitors/classic_competitor.py:29-42`.
2. `InitializeStressSGDState` calls `np.random.seed(problem.seed)` and stores
   `np.random` at `dagua/layout/ops/stress_sgd.py:713-717`.
3. `RunStressSGDExactSchedule` draws initial positions using the module-level NumPy
   RNG at `dagua/layout/ops/stress_sgd.py:876-879`.
4. It shuffles pair indices with `rng.shuffle(order)` at lines 891-895.

Dagua disconnected fallback is a separate torch RNG path:
`torch.Generator(device="cpu")`, `manual_seed(seed)`, `torch.randn` at
`dagua/layout/ops/stress_sgd.py:622-625`. Reference has no equivalent disconnected
fallback. Therefore, even where both receive seed 42, exact connected initial
positions align with reference NumPy, but pair-shuffle order does not align because
NumPy shuffle and RandomKit `rk_interval` are different random streams.

## 11. Edge-case bugs

Ranked by confidence that the behavior is incorrect relative to the benchmark target:

1. **Weighted multi-edge preprocessing mismatch.** Dagua uses `dedup="min"` in
   `dagua/layout/ops/pipelines/stress_sgd.py:65-72`; reference adapter sums duplicate
   symmetrized directed weights at `dagua/eval/competitors/sgd2_competitor.py:72-78`.
   For graphs such as `parallel_multiedge_bundle` and weighted fixtures, this changes
   shortest-path distances, objective weights, schedule bounds, and final layout.
2. **Edgeless `N>1` mismatch.** Dagua returns random fallback for disconnected
   graphs at `dagua/layout/ops/stress_sgd.py:696-711`; reference adapter returns zeros
   when no non-self edges survive at `dagua/eval/competitors/sgd2_competitor.py:207-211`.
3. **Trailing isolated-node mismatch in reference adapter.** Reference `s_gd2` infers
   `n` from edge endpoints at `.../site-packages/s_gd2/s_gd2.py:156-162`; the adapter
   does not pass `num_nodes` or pad coordinates at
   `dagua/eval/competitors/sgd2_competitor.py:220-221`. Dagua always uses
   `graph.num_nodes` at `dagua/layout/ops/pipelines/stress_sgd.py:156-162`.
4. **Exact-vs-approx cutoff mismatch.** Dagua switches to approximation over 10,000
   nodes at `dagua/layout/ops/stress_sgd.py:777-803`; reference adapter calls exact
   `s_gd2.layout` up to `max_nodes=50_000` at
   `dagua/eval/competitors/sgd2_competitor.py:138-140`, `:220`.
5. **RNG shuffle non-equivalence.** Dagua comments claim "bit-for-bit parity" with
   classic NumPy state at `dagua/layout/ops/stress_sgd.py:713-717`, but the actual
   reference uses RandomKit for shuffling at `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:23-33`,
   `:63-72`. This is not a correctness bug for distributional equivalence, but it is
   a false parity claim and a reproducibility gap.
6. **Zero magnitude behavior differs.** Dagua silently skips zero-distance update at
   `dagua/layout/ops/stress_sgd.py:587-589`; reference divides by `mag` without a
   guard at `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:48-56`. Dagua's behavior is safer,
   but it is not bit-compatible.
7. **`steps == 1` schedule behavior differs.** Dagua returns `[eta_max]` at
   `dagua/layout/ops/stress_sgd.py:299-313`; reference divides by `t_max-1` at
   `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:280`. The standard variants avoid this, so
   impact is low.
8. **Reference output scale is adapter-only.** Reference adapter multiplies by 100.0
   at `dagua/eval/competitors/sgd2_competitor.py:220-221`; dagua returns raw units at
   `dagua/layout/ops/pipelines/stress_sgd.py:174-180`. Procrustes RMSD with scale
   alignment tolerates this; raw layout metrics may not.

## 12. Ranked fix list

1. **Align weighted edge preprocessing for `classic_stress_sgd`.**
   - Evidence: dagua `dedup="min"` at `dagua/layout/ops/pipelines/stress_sgd.py:65-72`;
     reference adapter sum behavior at `dagua/eval/competitors/sgd2_competitor.py:72-78`;
     tests document summing at `tests/test_edge_weights_adapters.py:161-173`.
   - Expected RMSD impact: high on weighted and multiedge graphs, likely low on simple
     unweighted graphs.
   - Proposed fix: add a fidelity-mode preprocessing path for stress_sgd that mirrors
     `_symmetrized_unique_edges` before `BuildAdjacency`, or change only the classic
     competitor wrapper to pass preprocessed edges/weights.
   - Size estimate: M (40-90 LOC plus focused tests).

2. **Add a reference-compatible edgeless/disconnected policy toggle.**
   - Evidence: dagua fallback random positions at `dagua/layout/ops/stress_sgd.py:696-711`;
     reference zeros for empty surviving edge list at
     `dagua/eval/competitors/sgd2_competitor.py:207-211`; reference native errors for
     disconnected graphs at `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:159-162`, `:259-262`.
   - Expected RMSD impact: high on edgeless/disconnected benchmark cases, zero on
     connected cases.
   - Proposed fix: for fidelity variants, return zeros for no usable edges and return
     an error/filtered result for disconnected graphs, or make reference adapter and
     dagua use the same component fallback before comparison.
   - Size estimate: S-M (20-60 LOC depending on harness policy).

3. **Use RandomKit-equivalent shuffle or call a tiny local RandomKit port for exact mode.**
   - Evidence: reference `rk_seed` and `rk_interval` at
     `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:23-24`, `:63-72`; dagua `np.random.seed` and
     `rng.shuffle` at `dagua/layout/ops/stress_sgd.py:713-717`, `:891-895`.
   - Expected RMSD impact: medium for same-seed Procrustes, probably low for
     distributional TOST.
   - Proposed fix: implement a small RandomKit-compatible `rk_interval` shuffle or use
     the package's native shuffle only in a fidelity oracle. Keep default NumPy path
     if performance/maintenance risk is too high.
   - Size estimate: M-L (80-180 LOC plus parity tests against captured shuffle orders).

4. **Store exact distances/weights as float64 until final tensor conversion.**
   - Evidence: dagua exact arrays are `float32` at
     `dagua/layout/ops/stress_sgd.py:488-491`; reference term fields are double at
     `/tmp/s_gd2_src/cpp/s_gd2/layout.hpp:25-30`.
   - Expected RMSD impact: low on unweighted graphs, medium on weighted graphs with
     non-integer shortest paths.
   - Proposed fix: change `distances` and `weights` arrays in `_build_exact_terms` to
     `np.float64`; keep final `torch.float32` output unchanged.
   - Size estimate: S (5-15 LOC plus weighted regression).

5. **Avoid approximate mode for the `classic_stress_sgd` vs `sgd2` fidelity family.**
   - Evidence: dagua switches beyond 10,000 nodes at
     `dagua/layout/ops/stress_sgd.py:777-803`; reference adapter exact call at
     `dagua/eval/competitors/sgd2_competitor.py:220`; reference sparse API is separate
     at `.../s_gd2/s_gd2.py:69-89`.
   - Expected RMSD impact: high for `N > 10_000`, none for current small graphs.
   - Proposed fix: set `max_exact_nodes` at least to `SGD2.max_nodes` for fidelity
     variants, or compare dagua approximate mode against `s_gd2.layout_sparse` instead
     of `s_gd2.layout`.
   - Size estimate: S (variant/config change) if memory acceptable; M if adding a
     separate sparse-reference family.

6. **Pad or reject isolated-node reference outputs consistently.**
   - Evidence: reference `random_init` infers `n` from edges at
     `.../s_gd2/s_gd2.py:156-162`; adapter does not pad after `s_gd2.layout` at
     `dagua/eval/competitors/sgd2_competitor.py:220-221`; dagua explicit `num_nodes`
     at `dagua/layout/ops/pipelines/stress_sgd.py:156-162`.
   - Expected RMSD impact: high only on graphs with isolated trailing nodes.
   - Proposed fix: in the reference adapter, detect `coordinates.shape[0] != graph.num_nodes`
     and either pad deterministic zeros/fallback or return a clear error so the harness
     does not compare mismatched node sets.
   - Size estimate: S (10-30 LOC plus adapter test).

7. **Expose `init` in dagua stress_sgd pipeline for direct parity testing.**
   - Evidence: reference `init` support at `.../s_gd2/s_gd2.py:15`, `:135-145`; dagua
     signature lacks it at `dagua/layout/ops/pipelines/stress_sgd.py:87-98`.
   - Expected RMSD impact: medium for diagnosis, low for benchmark default because
     both default to NumPy `[0,1)`.
   - Proposed fix: optional `init_pos` parameter that bypasses `np.random.rand` when
     provided.
   - Size estimate: M (40-80 LOC plus tests).

8. **Remove or reword the "bit-for-bit parity" comment.**
   - Evidence: comment at `dagua/layout/ops/stress_sgd.py:713-715`; actual reference
     shuffling uses RandomKit at `/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:23-33`, `:63-72`.
   - Expected RMSD impact: none, but prevents future incorrect assumptions.
   - Proposed fix: say initialization matches `s_gd2`'s NumPy initialization, while
     shuffle semantics are only distributionally similar unless RandomKit is ported.
   - Size estimate: XS (comment-only).

## 13. Recommended Round 22+ fix scope

Recommended one-round bundle:

1. **Adapter/preprocessing parity first:** align the dagua classic path with
   `_symmetrized_unique_edges` for weighted and multiedge inputs, or add a dedicated
   fidelity-mode preprocessing function. This attacks the most concrete objective
   mismatch with file-local evidence (`dagua/layout/ops/pipelines/stress_sgd.py:65-72`
   vs `dagua/eval/competitors/sgd2_competitor.py:53-80`).
2. **Float64 exact terms:** change exact `distances` and `weights` to `float64` in
   `dagua/layout/ops/stress_sgd.py:488-491`. This is small and reduces weighted
   numerical drift.
3. **Edge-case policy tests:** add tests for edgeless `N>1`, disconnected, self-loop,
   reverse weighted multiedge, and trailing isolated node behavior. The current tests
   already cover reference weighted adapter expectations at
   `tests/test_edge_weights_adapters.py:161-182`; extend from there.
4. **Do not port RandomKit in the same round unless the above produces no movement.**
   RNG shuffle parity is real (`/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:63-72` vs
   `dagua/layout/ops/stress_sgd.py:891-895`), but it is more maintenance-heavy and
   mainly affects same-seed reproducibility rather than distributional equivalence.

Expected result: this bundle should reduce residual divergences on weighted/multiedge
and edge-case graphs without changing the core exact stress-SGD update, which is
already formula-aligned (`dagua/layout/ops/stress_sgd.py:584-595` vs
`/tmp/s_gd2_src/cpp/s_gd2/layout.cpp:43-59`).

## Verification

- Report target created: `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_stress_sgd.md`.
- Diagnosis-only source policy followed: no dagua source files were edited.
- Note: `/tmp/s_gd2_src` is a temporary read-only clone used to inspect the reference
  C++ source corresponding to installed `s_gd2` metadata
  (`.../site-packages/s_gd2-1.8.1.dist-info/METADATA:1-9`).

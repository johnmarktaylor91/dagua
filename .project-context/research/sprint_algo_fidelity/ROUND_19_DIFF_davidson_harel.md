# Davidson-Harel: line-by-line dagua-vs-igraph diff

## 1. Overall structure

Reference files read in full:

- igraph C reference: `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c`, 454 lines.
- Dagua op implementation: `dagua/layout/ops/davidson_harel.py`, 469 lines.
- Dagua pipeline wrapper: `dagua/layout/ops/pipelines/davidson_harel.py`, 145 lines.
- Prior notes: `.project-context/research/sprint_algo_fidelity/ROUND_12_BLOCKED.md` and `eval_output/algo_fidelity/round_13/SUMMARY.md`.

The igraph implementation is a single stateful function, `igraph_layout_davidson_harel`, at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:141`. It allocates permutation and trial-direction vectors at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:190-196`, optionally seeds the output matrix at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:198-231`, precomputes 30 circular unit directions at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:233-237`, then runs one combined loop over annealing plus fine-tuning rounds at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:239-444`. Energy is never recomputed as a full scalar in the main loop. Instead, each candidate move computes a local `diff_energy` from only terms affected by the moved vertex at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:255-420`, and acceptance uses that delta directly at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:422-436`.

Dagua decomposes the algorithm into pipeline ops. `build_davidson_harel_pipeline` creates `FixedSteps`, `InitializeDHPositions`, `PrepareDHState`, a `Repeat` of `DHAnnealingRound` plus `DHCool`, and `FinalizeDHPositions` at `dagua/layout/ops/pipelines/davidson_harel.py:50-63`. This is structurally different from igraph in four important ways:

1. Dagua has no fine-tuning loop. The pipeline repeats exactly `rounds` annealing rounds at `dagua/layout/ops/pipelines/davidson_harel.py:55-61`; igraph runs `maxiter + fineiter` rounds and changes behavior when `round >= maxiter` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:239-249`.
2. Dagua caches a full scalar energy in `PrepareDHState` at `dagua/layout/ops/davidson_harel.py:312-321` and updates it by recomputing the entire energy for each candidate at `dagua/layout/ops/davidson_harel.py:385-390`. igraph computes incremental deltas and never stores a global energy scalar in the main loop.
3. Dagua performs final centering and scaling at `dagua/layout/ops/davidson_harel.py:461-468`. igraph returns the matrix after the last accepted move and does no final recenter/rescale at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:442-453`.
4. Dagua converts the input graph to sorted unique undirected edges at `dagua/layout/ops/davidson_harel.py:39-70` and stores those at `dagua/layout/ops/davidson_harel.py:304-310`. igraph keeps the original graph edge count at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:149-150` and uses `igraph_neighbors`, `igraph_incident`, `IGRAPH_FROM`, and `IGRAPH_TO` against the original graph at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:327-419`.

The width/extent baseline mostly matches only when `node_sizes is None`: igraph sets `width = sqrt(no_nodes) * 10` and `move_radius = width / 2` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:151-157`; Dagua's shared `layout_extent` returns `sqrt(num_nodes) * 5.0` with a floor of 1.0 at `dagua/layout/ops/graph_utils.py:194-213`, then uses that half-width at `dagua/layout/ops/davidson_harel.py:273-275` and `dagua/layout/ops/davidson_harel.py:327-329`. With `node_sizes` present, Dagua changes the extent based on node dimensions at `dagua/layout/ops/graph_utils.py:212-213`; igraph has no node-size concept in this function.

The previous Round 13 report says energy weights and move schedule improved median RMSD from `0.361980` to `0.237719` at `eval_output/algo_fidelity/round_13/SUMMARY.md:65-82`, but also flags residual incremental-energy and edge-multiplicity gaps at `eval_output/algo_fidelity/round_13/SUMMARY.md:113-118`. This diagnosis confirms those residuals and adds the fine-tuning, RNG stream, node-edge gating, boundary, and final-normalization divergences below.

## 2. Energy function

Important structural note: the C source does not define a standalone `dh_energy` in this checked-out reference. The effective energy is represented by the five per-move delta blocks in `igraph_layout_davidson_harel`: node-node distance, border, edge length, edge crossings, and node-edge distance. Dagua has a standalone full-energy function `_energy` at `dagua/layout/ops/davidson_harel.py:119-207`.

- Term: node-node distance
- igraph: `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:278-291`, especially `diff_energy += w_node_dist / dist2 - w_node_dist / odist2` at line 290.
- dagua: `dagua/layout/ops/davidson_harel.py:145-153`, especially `distribution = squared_distances.reciprocal().sum()` at line 152 and `_NODE_DIST_WEIGHT * distribution` at `dagua/layout/ops/davidson_harel.py:201-203`.
- Mathematically identical? Mostly Y for simple graphs and nonzero distances. Dagua full energy sums each unordered pair once via `torch.triu_indices` at `dagua/layout/ops/davidson_harel.py:148`; igraph's local delta loops all `u != v` for the moved vertex at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:279-291`, which is equivalent to changing the unordered-pair full sum.
- Numerically identical? N. Dagua clamps squared distance to `_MIN_DISTANCE = 1.0e-3` at `dagua/layout/ops/davidson_harel.py:17` and `dagua/layout/ops/davidson_harel.py:149-151`; igraph divides by raw `dist2` and `odist2` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:288-290`. Dagua uses torch float32 positions by default from `dagua/layout/ops/davidson_harel.py:82` and final output line `dagua/layout/ops/davidson_harel.py:468`; igraph uses `igraph_real_t`, normally double. If two vertices coincide, igraph can produce infinities, while Dagua floors the penalty.
- Severity: MEDIUM. The formula is the same away from degeneracy, but the clamp and dtype can alter early moves, especially because the term is singular.

- Term: border distance
- igraph: `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:294-325`, with old/new four-border reciprocal-square deltas at lines 319-324. The term is gated by `if (w_borderlines != 0)` at line 294, and default weight is copied from `weight_border` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:162-164`.
- dagua: `dagua/layout/ops/davidson_harel.py:154-163` computes `border_distances.reciprocal().square().sum()`, and it is multiplied by `_BORDER_WEIGHT` at `dagua/layout/ops/davidson_harel.py:201-204`. `_BORDER_WEIGHT = 0.0` at `dagua/layout/ops/davidson_harel.py:18-20`.
- Mathematically identical? Mostly Y only when border weight is enabled and coordinates are inside bounds. Both use four distances to square borders. However, igraph repairs negative old/new distances to `2` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:299-318`; Dagua clamps every border distance to `_MIN_DISTANCE` at `dagua/layout/ops/davidson_harel.py:154-163`.
- Numerically identical? N. Current default weight zero means neither contributes to default layouts, so the numerical mismatch is latent unless a future API exposes nonzero border weight. Boundary clamping differs too: igraph can set the lower bound to `-width / 2 - 1e-6` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:265-267`; Dagua clamps exactly into `[-extent, extent]` at `dagua/layout/ops/davidson_harel.py:385-386`.
- Severity: LOW for current defaults; MEDIUM if nonzero border weight is used.

- Term: edge length
- igraph: `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:327-341`, with neighbor retrieval using `IGRAPH_ALL, IGRAPH_NO_LOOPS, IGRAPH_MULTIPLE` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:327-330` and `diff_energy += w_edge_lengths * (dist2 - odist2)` at line 340.
- dagua: `_unique_edges` drops self-loops and aggregates duplicates at `dagua/layout/ops/davidson_harel.py:55-69`; full edge length sums `norm(...).square() * edge_weight_tensor[index]` at `dagua/layout/ops/davidson_harel.py:165-177`, with weight applied at `dagua/layout/ops/davidson_harel.py:201-205`.
- Mathematically identical? Y for simple unweighted loop-free graphs. For multigraphs, Dagua's duplicate aggregation can match the edge-length multiplicity if all duplicate edge weights are 1.0, because a duplicate pair gets aggregated at `dagua/layout/ops/davidson_harel.py:62-68`. With explicit `edge_weights`, Dagua can represent weighted edge lengths, but igraph's C API here has no per-edge weight argument, so weighted Dagua layouts intentionally diverge.
- Numerically identical? N. Dagua recomputes full edge length with torch and sorted unique edges; igraph computes only moved incident edges in graph API order. Dagua edge list ordering does not change the scalar sum, but floating-point accumulation order can differ. More importantly, Dagua uses full-energy acceptance, so this term is combined with other full terms that are not igraph-equivalent.
- Severity: LOW for simple unweighted graphs; MEDIUM for multiedges and weighted Dagua calls.

- Term: edge crossing count
- igraph: `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:344-374`. For every neighbor edge incident to moved vertex `v`, igraph loops over every graph edge `e`, skips edges sharing `v` or that neighbor `u` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:356-362`, subtracts old intersections at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:367-368`, adds new intersections at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:369-370`, and adds `w_edge_crossings * no` at line 373. Segment intersection uses parametric division and returns false for parallel/collinear segments at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:33-55`, especially `if (s2 == 0) return false` at lines 45-48.
- dagua: full crossing energy loops over pairs of Dagua unique edges at `dagua/layout/ops/davidson_harel.py:179-186`, skips pairs sharing any endpoint at `dagua/layout/ops/davidson_harel.py:181-183`, uses `_segments_intersect` at `dagua/layout/ops/davidson_harel.py:90-103`, and multiplies by `_CROSSING_WEIGHT` at `dagua/layout/ops/davidson_harel.py:201-206`.
- Mathematically identical? N. Dagua counts total crossings among unique undirected edge pairs. igraph's delta counts crossings involving moved incident edges, with graph multiplicity retained. For simple graphs, a full crossing count delta is conceptually equivalent to igraph's local delta only if the segment-intersection predicate is identical and each moved edge is represented once. For multiedges, Dagua collapses duplicates, while igraph can count each duplicate incident neighbor and each duplicate crossed edge through `IGRAPH_MULTIPLE` and `no_edges`.
- Numerically identical? N. The intersection predicates differ. igraph returns false for collinear parallel segments at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:45-48`; Dagua treats near-collinear orientations as intersecting because `_segments_intersect` returns true if either relevant orientation magnitude is below `_COLLINEAR_EPSILON` at `dagua/layout/ops/davidson_harel.py:97-103`, with `_COLLINEAR_EPSILON = 1.0e-10` at `dagua/layout/ops/davidson_harel.py:23`. Dagua therefore penalizes touching/collinear cases that igraph often ignores.
- Severity: HIGH. Crossing terms can dominate discrete move acceptance and are discontinuous; predicate and multiplicity mismatches can flip accept/reject decisions.

- Term: node-edge distance
- igraph: `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:376-420`. It is gated by both `w_node_edge_dist != 0` and `fine_tuning` at line 376. First it evaluates the moved node against all non-incident original graph edges at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:377-395`; then it evaluates every other node against every incident edge of the moved vertex at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:397-419`. Distances are squared distances from `igraph_i_layout_point_segment_dist2` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:58-78`, and the energy uses `w_node_edge_dist / d_ev` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:391-394` and `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:412-417`.
- dagua: full node-edge penalties are always computed when `_NODE_EDGE_WEIGHT` is nonzero at `dagua/layout/ops/davidson_harel.py:188-199`, skipping only incident unique edges at `dagua/layout/ops/davidson_harel.py:190-193`. `_NODE_EDGE_WEIGHT = 0.2` at `dagua/layout/ops/davidson_harel.py:21-22`, and it contributes in every annealing round at `dagua/layout/ops/davidson_harel.py:201-207`.
- Mathematically identical? N. This is the biggest energy-term mismatch. igraph excludes node-edge distance from the annealing phase entirely and only uses it in fine tuning. Dagua includes it from the first candidate move because `_energy` always includes node-edge when `_NODE_EDGE_WEIGHT` is nonzero.
- Numerically identical? N. Dagua computes Euclidean distance in `_point_segment_distance` at `dagua/layout/ops/davidson_harel.py:106-116`, then squares the reciprocal via `distance.clamp(...).reciprocal().square()` at `dagua/layout/ops/davidson_harel.py:194-197`, which equals `1 / squared_distance` away from the clamp. igraph returns squared distance directly at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:58-78` and divides by it. The formula is equivalent away from zero, but Dagua clamps the segment denominator and final distance at `dagua/layout/ops/davidson_harel.py:112-116` and `dagua/layout/ops/davidson_harel.py:197`; igraph only special-cases zero-length segments at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:66-68`.
- Severity: HIGH. Dagua is optimizing a different objective during the entire annealing phase, while igraph delays this term until a short deterministic fine-tuning phase.

- Term: terms in one side only
- igraph only: no full-energy cache; no final normalization energy effect. Fine-tuning changes the objective by adding node-edge distance and disabling uphill acceptance at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:376-423`.
- dagua only: final centering and scaling at `dagua/layout/ops/davidson_harel.py:461-468`; edge weights API at `dagua/layout/ops/pipelines/davidson_harel.py:74` and validation at `dagua/layout/ops/pipelines/davidson_harel.py:110-116`; duplicate aggregation weights at `dagua/layout/ops/davidson_harel.py:62-69`.
- Mathematically identical? N.
- Numerically identical? N.
- Severity: HIGH for fine-tuning and final normalization; LOW for optional edge weights unless used.

## 3. Move-acceptance loop

- Outer loop count
  - igraph: one loop over `round < maxiter + fineiter` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:239`, with `fine_tuning = round >= maxiter` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:244`.
  - dagua: `Repeat(n=rounds)` at `dagua/layout/ops/pipelines/davidson_harel.py:55-61`, default `rounds=100` at `dagua/layout/ops/pipelines/davidson_harel.py:26` and `dagua/layout/ops/pipelines/davidson_harel.py:72`. There is no `fineiter` parameter in `layout_davidson_harel_pipeline` at `dagua/layout/ops/pipelines/davidson_harel.py:68-75`.
  - Divergence: HIGH. Dagua cannot represent igraph's second phase. The benchmark variants align `rounds` to igraph `maxiter` only at `dagua/eval/variants.py:1089-1118`, leaving fineiter to the igraph default.

- Per-node inner loop
  - igraph: shuffles `perm` each round at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:242`, then visits each vertex via `v = VECTOR(perm)[p]` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:251-253`.
  - dagua: generates `node_order = torch.randperm(problem.num_nodes, generator=generator)` at `dagua/layout/ops/davidson_harel.py:371`, then iterates `for node in node_order.tolist()` at `dagua/layout/ops/davidson_harel.py:373`.
  - Divergence: MEDIUM. Structure matches, but RNG stream and permutation algorithm differ.

- Candidate direction generation
  - igraph: `no_tries = 30` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:161`; circular directions are `cos(phi), sin(phi)` for `phi = 2 * M_PI / no_tries * i` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:233-237`; `try_idx` is shuffled per vertex at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:253`.
  - dagua: `_MOVE_TRIES = 30` and `_MOVE_DIRECTIONS` are the same circular grid at `dagua/layout/ops/davidson_harel.py:24-28`; `direction_order = torch.randperm(_MOVE_TRIES, generator=generator)` at `dagua/layout/ops/davidson_harel.py:375`.
  - Divergence: LOW structurally, MEDIUM numerically. Same set of directions, different shuffle/RNG implementation.

- Move radius schedule
  - igraph: starts with `move_radius = width / 2` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:156`, decays after every round by `move_radius *= cool_fact` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:442`, and during fine-tuning resets per round to `min(0.01 * (max_x - min_x), 0.01 * (max_y - min_y))` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:245-249`.
  - dagua: initializes `state.temperature` to extent at `dagua/layout/ops/davidson_harel.py:323-329`, uses `move_scale = min(max(temperature, _MIN_DISTANCE), extent)` at `dagua/layout/ops/davidson_harel.py:374`, and cools by `state.temperature = state.temperature * self.config.cooling_factor` at `dagua/layout/ops/davidson_harel.py:431-434`.
  - Divergence: MEDIUM in annealing because Dagua caps at `extent`, although the temperature starts at extent and decays so the cap is usually inert. HIGH for fine-tuning because Dagua has no bounding-box based 0.01 radius.

- Boundary handling
  - igraph: lower x/y underflow clamps to `-width / 2 - 1e-6` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:265-267` and `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:271-273`; upper overflow clamps to `width / 2 - 1e-6` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:268-275`.
  - dagua: `candidate[node] = (candidate[node] + delta).clamp(min=-extent, max=extent)` at `dagua/layout/ops/davidson_harel.py:385-386`.
  - Divergence: LOW to MEDIUM. The lower-bound `-width / 2 - 1e-6` in igraph appears asymmetric relative to its own comment and can put a coordinate just outside the rectangle. Dagua clamps exactly inside.

- Acceptance criterion
  - igraph: accepts when `diff_energy < 0` or, only outside fine-tuning, `RNG_UNIF01() < exp(-diff_energy / move_radius)` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:422-423`.
  - dagua: accepts downhill when `delta_energy <= 0` at `dagua/layout/ops/davidson_harel.py:392-396`; uphill acceptance is always available and uses `torch.exp(-delta_energy / max(temperature, _MIN_DISTANCE))` at `dagua/layout/ops/davidson_harel.py:398-405`.
  - Divergence: HIGH. During annealing the only formula difference is `< 0` versus `<= 0` and denominator floor. During igraph fine-tuning uphill moves are forbidden; Dagua never enters that mode and would still accept uphill moves if extra rounds are requested.

- Fine-tuning phase
  - igraph: documented as separate at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:98-100`; parameter `fineiter` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:141-147`; round mode switch at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:244-249`; node-edge energy only active in this phase at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:376`; uphill acceptance disabled by `!fine_tuning` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:422-423`.
  - dagua: no `fineiter`, no `fine_tuning` flag, no min-bounding-box fine radius, no uphill-disable phase in `dagua/layout/ops/pipelines/davidson_harel.py:50-63` or `dagua/layout/ops/davidson_harel.py:351-410`.
  - Divergence: HIGH.

## 4. RNG / determinism

- Initial layout RNG
  - igraph: if `use_seed` is false, every vertex consumes two draws from the active igraph RNG with `RNG_UNIF(-width / 2, width / 2)` and `RNG_UNIF(-height / 2, height / 2)` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:198-213`. If `use_seed` is true, igraph uses caller-provided coordinates after validating matrix shape at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:178-184` and scanning bounds at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:214-231`.
  - dagua: `_initialize_positions` creates a new CPU `torch.Generator`, seeds it with `problem.seed`, and draws `torch.rand((num_nodes, 2))` at `dagua/layout/ops/davidson_harel.py:73-82`. There is no `use_seed` or initial-position matrix parameter in `layout_davidson_harel_pipeline` at `dagua/layout/ops/pipelines/davidson_harel.py:68-75`.
  - Divergence: HIGH. The RNG engine differs, and Dagua cannot consume caller-supplied starting coordinates like the C function.

- Move-direction RNG
  - igraph: one global RNG stream is used for initialization, `igraph_vector_int_shuffle(&perm)` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:242`, and `igraph_vector_int_shuffle(&try_idx)` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:253`.
  - dagua: `PrepareDHState` creates a second new CPU `torch.Generator` and seeds it again with the same `problem.seed` at `dagua/layout/ops/davidson_harel.py:331-333`; node and direction orders use that generator at `dagua/layout/ops/davidson_harel.py:371` and `dagua/layout/ops/davidson_harel.py:375`.
  - Divergence: HIGH. Dagua resets the move RNG instead of continuing after the initial `2N` coordinate draws. Even if PyTorch used igraph's RNG algorithm, the stream position would be different.

- Move-acceptance RNG
  - igraph: uphill threshold uses `RNG_UNIF01()` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:422-423` and consumes draws only for uphill annealing candidates that reach the acceptance expression.
  - dagua: uphill threshold uses `torch.rand((1,), generator=generator).item()` at `dagua/layout/ops/davidson_harel.py:403`; draws are also conditional on uphill candidate energy at `dagua/layout/ops/davidson_harel.py:392-404`.
  - Divergence: MEDIUM to HIGH. Conditional draw structure is similar during annealing, but because energy deltas, fine-tuning behavior, and prior shuffles differ, draw counts diverge quickly.

- Numpy versus torch semantics
  - Dagua's implementation here is torch-only for positions, energy, permutations, and acceptance at `dagua/layout/ops/davidson_harel.py:9`, `dagua/layout/ops/davidson_harel.py:73-82`, and `dagua/layout/ops/davidson_harel.py:119-207`.
  - igraph uses C doubles and igraph RNG macros at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:24-25` and `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:198-203`.
  - Divergence: MEDIUM. This family is stochastic, so bit identity is unrealistic without reusing igraph RNG semantics or accepting seeded-layout comparison instead of seeded-RNG comparison.

## 5. Edge cases

- Disconnected components handling
  - igraph: no component separation. All vertices share one rectangle from `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:151`, and node-node repulsion applies to every `u != v` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:278-291`.
  - dagua: no component separation. `_energy` computes all unordered node pairs at `dagua/layout/ops/davidson_harel.py:145-153`.
  - Match? Mostly Y. Residual differences are RNG, final normalization, and node-edge/fine-tuning, not explicit component logic.

- Self-loops handling
  - igraph: edge length and crossing neighbor calls pass `IGRAPH_NO_LOOPS` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:327-330` and `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:347-349`; incident edges for node-edge pass `IGRAPH_NO_LOOPS` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:397-399`. The all-edge loops for crossings and moved-node-to-edge checks iterate `no_edges` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:356-371` and `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:378-395`, but self-loops incident to the moved vertex are skipped by endpoint checks at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:360-362` and `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:382-383`; self-loops not incident to the moved vertex become zero-length segments in node-edge fine-tuning.
  - dagua: `_unique_edges` drops `source == target` unconditionally at `dagua/layout/ops/davidson_harel.py:60-61`, so self-loops never appear in any term.
  - Match? N for fine-tuning node-edge with self-loops on other vertices; likely LOW in common graphs, MEDIUM if self-loops exist.

- Multi-edges handling
  - igraph: graph edge count is original `igraph_ecount(graph)` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:149-150`; edge-length and crossing neighbor lists request `IGRAPH_MULTIPLE` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:327-330` and `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:347-349`; crossing and node-edge loops iterate every original edge at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:356-371` and `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:378-395`.
  - dagua: duplicate undirected pairs are collapsed and sorted at `dagua/layout/ops/davidson_harel.py:62-69`.
  - Match? N. Dagua can preserve multiplicity in edge-length via aggregated weights, but not in crossing or node-edge counts. HIGH for graphs like `parallel_multiedge_bundle`.

- Empty graph
  - igraph: if `no_nodes == 0`, returns success at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:186-188`.
  - dagua: if `num_nodes == 0`, returns an empty `[0, 2]` float32 tensor at `dagua/layout/ops/pipelines/davidson_harel.py:125-126`.
  - Match? Y behaviorally.

- One-node graph
  - igraph: not special-cased after empty. Width is 10 at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:151`, initial random x/y are drawn at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:198-213`, and the loop runs but no node-node or edge terms apply.
  - dagua: returns exactly `[[0, 0]]` at `dagua/layout/ops/pipelines/davidson_harel.py:127-128`.
  - Match? N. Usually LOW for RMSD after centering, but it is a direct reference divergence.

## 6. Hyperparameter alignment table

| Param | igraph default/source | dagua default/source | Match? | Notes |
|---|---|---|---|---|
| rectangle width | `sqrt(no_nodes) * 10` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:151` | half extent `sqrt(num_nodes) * 5.0` when no sizes at `dagua/layout/ops/graph_utils.py:209-210` | Y for no sizes | Dagua differs when `node_sizes` is provided at `dagua/layout/ops/graph_utils.py:212-213`. |
| initial move radius | `width / 2` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:156` | extent at `dagua/layout/ops/davidson_harel.py:327-329` | Y for no sizes | Same nominal half-width. |
| maxiter / rounds | C parameter `maxiter` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:141-143`; docs call 10 reasonable at lines 112-113 | `rounds=100` at `dagua/layout/ops/pipelines/davidson_harel.py:26` and line 72 | Partial | Eval wrapper calls `rounds=100` at `dagua/eval/competitors/classic_competitor.py:1450-1455`; variants align maxiter at `dagua/eval/variants.py:1089-1118`. |
| fineiter | C parameter `fineiter` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:141-143`; docs reasonable `max(10, log2(n))` at lines 114-116 | absent | N | Highest-impact missing phase. |
| cool_fact | C parameter, valid `(0,1)` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:175-177`; copied at line 143 | `DHCoolConfig.cooling_factor = 0.75` at `dagua/layout/ops/davidson_harel.py:243-253` | Y if caller uses 0.75 | Dagua does not validate custom config range in op. |
| no_tries | `30` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:161` | `_MOVE_TRIES = 30` at `dagua/layout/ops/davidson_harel.py:24` | Y | Direction set matches. |
| fine_tuning_factor | `0.01` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:157` | absent | N | Needed for fine-tuning radius. |
| weight_node_dist | copied from parameter at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:162`; docs reasonable 1.0 at line 119 | `_NODE_DIST_WEIGHT = 1.0` at `dagua/layout/ops/davidson_harel.py:18` | Y | Clamp/dtype still differs. |
| weight_border | copied at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:163`; docs allow zero at lines 120-122 | `_BORDER_WEIGHT = 0.0` at `dagua/layout/ops/davidson_harel.py:19` | Y | Latent formula mismatch if nonzero. |
| weight_edge_lengths | copied at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:164`; comment `0.0001` | `_EDGE_LENGTH_WEIGHT = 0.0001` at `dagua/layout/ops/davidson_harel.py:20` | Y | Multiplicity semantics partly differ. |
| weight_edge_crossings | copied at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:165`; comment `1.0` | `_CROSSING_WEIGHT = 1.0` at `dagua/layout/ops/davidson_harel.py:21` | Y | Predicate and multiplicity differ. |
| weight_node_edge_dist | copied at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:166`; comment `0.2` | `_NODE_EDGE_WEIGHT = 0.2` at `dagua/layout/ops/davidson_harel.py:22` | Weight Y, phase N | Dagua applies during annealing; igraph only fine-tuning at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:376`. |
| use_seed | supported at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:178-184` and `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:214-231` | absent from `layout_davidson_harel_pipeline` at `dagua/layout/ops/pipelines/davidson_harel.py:68-75` | N | Important for deterministic reference matching. |
| edge weights | no per-edge weight parameter in C signature at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:141-147` | optional `edge_weights` at `dagua/layout/ops/pipelines/davidson_harel.py:74` | N | Dagua extension, not igraph parity. |
| final normalization | none before return at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:442-453` | centroid/scale at `dagua/layout/ops/davidson_harel.py:461-468` | N | Directly changes output coordinates after optimization. |

## 7. Ranked fix list

1. Add igraph fine-tuning phase and gate node-edge energy to that phase. Dagua lines: `dagua/layout/ops/pipelines/davidson_harel.py:55-61`, `dagua/layout/ops/davidson_harel.py:188-207`, `dagua/layout/ops/davidson_harel.py:392-405`. igraph lines: `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:239-249`, `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:376-423`. Proposed fix: add `fineiter` to the pipeline, add a fine-tuning op or phase flag, use bounding-box `0.01 * min(span_x, span_y)` radius, include node-edge distance only in that phase, and disable uphill acceptance in that phase. Expected median delta: high, plausibly `0.04-0.10`, because Dagua currently optimizes the node-edge term during all annealing moves while igraph does not.

2. Remove or make optional final centering/scaling for igraph-fidelity mode. Dagua lines: `dagua/layout/ops/pipelines/davidson_harel.py:62`, `dagua/layout/ops/davidson_harel.py:461-468`. igraph lines: `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:442-453`. Proposed fix: skip `FinalizeDHPositions` for the igraph-comparison pipeline or replace it with a no-op. Expected median delta: high in raw coordinate RMSD if comparator already Procrustes-aligns imperfectly; likely `0.02-0.08`. This is a pure postprocess divergence and should be easy to isolate.

3. Replace full-energy recomputation with igraph-style incremental delta blocks. Dagua lines: full `_energy` at `dagua/layout/ops/davidson_harel.py:119-207` and candidate recomputation at `dagua/layout/ops/davidson_harel.py:385-390`. igraph lines: `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:255-420`. Proposed fix: implement `_move_delta_energy(...)` that exactly mirrors the five C blocks, with a `fine_tuning` flag. Expected median delta: high, `0.03-0.08`, because it naturally fixes phase gating, multiplicity count opportunities, and accumulation order for moved terms.

4. Preserve original edge multiplicity and graph edge order for crossing and node-edge terms. Dagua lines: `_unique_edges` at `dagua/layout/ops/davidson_harel.py:39-70`, edge cache at `dagua/layout/ops/davidson_harel.py:304-310`, crossing loop at `dagua/layout/ops/davidson_harel.py:179-186`, node-edge loop at `dagua/layout/ops/davidson_harel.py:188-199`. igraph lines: `no_edges` at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:149-150`, neighbor/edge loops at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:327-419`. Proposed fix: cache original directed endpoint arrays plus adjacency/incident lists with loop and multiple-edge policies matching `IGRAPH_NO_LOOPS` and `IGRAPH_MULTIPLE`; keep unique weighted edges only for optional Dagua extensions if needed. Expected median delta: medium overall, high on multiedge graphs, likely `0.01-0.06`.

5. Align segment intersection exactly with igraph's parametric predicate. Dagua lines: `_orientation` and `_segments_intersect` at `dagua/layout/ops/davidson_harel.py:85-103`. igraph lines: `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:33-55`. Proposed fix: port the C `s1/s2/t1/t2` predicate, including returning false when `s2 == 0`. Expected median delta: medium, `0.005-0.03`, with larger effects on collinear/touching cases.

6. Use a single RNG stream or support seeded initial positions for fidelity comparisons. Dagua lines: initial generator at `dagua/layout/ops/davidson_harel.py:79-82`, reset move generator at `dagua/layout/ops/davidson_harel.py:331-333`, node/direction/acceptance draws at `dagua/layout/ops/davidson_harel.py:371-404`. igraph lines: initial draws at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:198-203`, shuffles at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:242-253`, acceptance draw at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:422-423`. Proposed fix: do not reset the Dagua move generator after initialization, or better, compare with `use_seed`-style starting matrices so the stochastic stream mismatch is reduced to moves. Expected median delta: medium but hard to predict; RNG differences are chaotic and may change seed-level comparability more than aggregate quality.

7. Match boundary clamp and singular-distance behavior. Dagua lines: `_MIN_DISTANCE` at `dagua/layout/ops/davidson_harel.py:17`, candidate clamp at `dagua/layout/ops/davidson_harel.py:385-386`, distance clamps at `dagua/layout/ops/davidson_harel.py:149-163` and `dagua/layout/ops/davidson_harel.py:197`. igraph lines: boundary clamp at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:265-276`, raw reciprocal divisions at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:288-290`, `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:391-394`, and `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:414-417`. Proposed fix: for fidelity mode, use igraph's exact clamp constants and remove broad floors except where C special-cases zero-length segments. Expected median delta: low to medium, `0.002-0.02`, but can flip moves near collisions/borders.

8. Match one-node and `use_seed` behavior. Dagua lines: one-node shortcut at `dagua/layout/ops/pipelines/davidson_harel.py:127-128`, no seed-matrix parameter at `dagua/layout/ops/pipelines/davidson_harel.py:68-75`. igraph lines: empty-only shortcut at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:186-188`, seed validation/use at `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:178-231`. Proposed fix: only special-case empty graphs and optionally accept `initial_pos`. Expected median delta: low for benchmark suites but important for line-by-line parity.

## 8. Recommended Round 20 fix scope

Round 20 should not try to tune weights again; the headline weights are already aligned at `dagua/layout/ops/davidson_harel.py:18-22` versus `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:162-166`, and Round 13 measured that improvement at `eval_output/algo_fidelity/round_13/SUMMARY.md:65-82`.

Recommended Round 20 prompt:

1. Implement an igraph-fidelity move delta path in `dagua/layout/ops/davidson_harel.py` that mirrors `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:255-420`.
2. Add `fineiter` support to `layout_davidson_harel_pipeline` and `build_davidson_harel_pipeline`, defaulting to the igraph/Python binding behavior used by the comparator if confirmed, and at minimum allowing explicit variant alignment.
3. During annealing, include node-node, border, edge-length, and crossing deltas only; during fine-tuning, add node-edge deltas, use `0.01 * min(current_span_x, current_span_y)` move radius, and disable uphill acceptance.
4. Preserve original edge arrays and adjacency/incident lists for `IGRAPH_NO_LOOPS` and `IGRAPH_MULTIPLE` semantics instead of using sorted unique edges for all terms.
5. Remove `FinalizeDHPositions` from the igraph-fidelity pipeline or make it opt-in, then rerun the same Round 13 small comparison before touching lower-impact RNG and clamp details.

Expected best first bundle: items 1, 2, and 5 from the ranked list. That bundle addresses the largest known objective and output divergences while keeping the patch contained to `dagua/layout/ops/davidson_harel.py`, `dagua/layout/ops/pipelines/davidson_harel.py`, and tests. Multiplicity and exact RNG should be the next pass if the family remains partial-match after fine-tuning and finalization are corrected.

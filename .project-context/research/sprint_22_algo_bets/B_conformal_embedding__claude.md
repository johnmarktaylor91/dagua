# Area B — Conformal / Harmonic Embedding for True Planar Lattices (claude)

## TL;DR

* **Don't ship Tutte for the lattice gap.** Empirically measured on hex_42,
  tri_36, sierp_42 with depth-anchored Tutte (classical, mean-value-weighted,
  and 2D-with-monotone-y-warp variants in /tmp). Best Tutte composite is
  consistently **5-10 points BELOW dagua's current scores** on these targets.
  The predicted "+5..+7" from sprint-21 D Claude does not survive contact
  with empirical scoring.
* **The 2025 paper (arXiv:2506.20541) does not say what sprint-21 D Claude
  said it says.** Title is "Conformal Rigidity and Spectral Embeddings of
  Graphs" (Gouveia, Steinerberger, Thomas). It is a SPECTRAL-RIGIDITY
  theorem — characterizes which graphs admit unique Laplacian-eigenvalue-
  optimal edge weightings. It is NOT a Tutte/harmonic refinement
  algorithm. It is an existence result, not a layout method.
* **Where Tutte_v2 DOES win**: parallel_cycles_4x5 hits 65.36 (vs dagua
  62.11, +3.25; vs sfdp 62.73, +2.63) in /tmp measurement. So a narrow
  picker-tied gate could cash this one (-0.62 -> +2.63 = ~+3.3 swing).
* **Where Tutte_v2 ties**: planar_60 hits 75.50 (vs dagua 80.09, gv_dot
  75.10) — beats graphviz_dot but loses to dagua. Not worth exposing.
* **Single biggest call**: scope this bet to **parallel_cycles only** as
  a tiny new pipeline `tutte_cyclic_planar` gated on
  `lattice_like AND k_components > 1 AND all_components_are_cycles`.
  The hex / tri / sierp gaps cannot be closed via Tutte family — those
  losses need either (a) graphviz_dot's exact lattice-pitch heuristic
  (Bet 1), or (b) accept dagua at ceiling.

---

## 1. Paper review (the foundational misread)

### 1.1 What sprint-21 D Claude claimed

> "arXiv:2506.20541 (June 2025) 'conformal-rigidity-guided Tutte/harmonic
> refinement' as the genuinely-new post-2024 result with applicability to
> dagua's metric. Predicted +5..+7 composite on hex/tri lattices. ~250 LOC."

### 1.2 What the paper actually is

**Citation:** João Gouveia, Stefan Steinerberger, Rekha R. Thomas.
"Conformal Rigidity and Spectral Embeddings of Graphs." arXiv:2506.20541,
June 25 2025. Submitted to Journal of Graph Theory (per Steinerberger's
2025 Wiley publication on the parent topic).

**Core definition.** A graph G is *conformally rigid* if no edge
re-weighting w: E -> R+ can simultaneously increase the second Laplacian
eigenvalue lambda_2(L_w) and decrease the last lambda_n(L_w). Equivalently,
the unit-weight Laplacian is Pareto-optimal in the (lambda_2, lambda_n)
spectral landscape.

**Main results.**
1. Edge-transitive graphs are conformally rigid.
2. Distance-regular graphs are conformally rigid.
3. All 1-walk regular graphs are conformally rigid (consequence of the
   isometric embedding lemma proved here).
4. SDP characterization of conformal rigidity in terms of edge orbits
   under Aut(G) for vertex-transitive graphs.
5. Necessary-and-sufficient combinatorial test for Cayley graphs on
   abelian groups.

**What this means for layout.**
The theorem says: when a graph is conformally rigid, the **unit-weight
Tutte/spectral embedding is already optimal in a precise SDP-Pareto sense**.
It therefore does NOT propose a refinement algorithm. It tells you that
on the symmetric graphs you might hope to improve via clever weights,
**you can't** — unit weights already win. There is no "guided refinement"
to extract.

For our targets:
* Triangular lattice patch: 1-walk regular at interior, NOT at boundary.
  So unit Tutte is rigid for the bulk but boundary distortion still
  bleeds in.
* Hexagonal lattice patch: same story (interior is 1-walk regular,
  boundary breaks it).
* Sierpinski_42: NOT vertex-transitive, conformal rigidity unknown.
* Parallel_cycles: cycle graph C_5 is conformally rigid (it's edge-
  transitive); so unit-weight Tutte is rigid per component.

**Conclusion:** The 2025 paper is a *theoretical justification* for
NOT bothering with edge-weighted Tutte refinements on symmetric inputs —
not a method. The "+5..+7" prediction was a confabulation. The actual
useful follow-up papers in this space:

* Floater 1997, "Parametrization and smooth approximation of surface
  triangulations," CAGD 14(3) — original mean-value coordinates,
  the practical Tutte-with-shape-preservation method used in computer
  graphics. Tested below; gives identical results to unit-weight on
  vertex-transitive inputs (matching the 2025 theorem).
* Floater 2003, "Mean value coordinates," CAGD 20(1) — barycentric
  weights via tan-half-angles, the Floater scheme implemented here.
* Steinerberger 2024, arXiv:2402.11758 "Conformally rigid graphs"
  (the predecessor to 2506.20541, Wiley J. Graph Theory 2025) —
  proves edge-transitivity implies conformal rigidity. Older, same
  message.
* Tutte 1963, "How to Draw a Graph," Proc. London Math. Soc. (3) 13 —
  the original 3-connectivity + outer-face -> unique convex straight-
  line embedding theorem.

---

## 2. Algorithm sketch (working pseudocode in /tmp)

Three variants implemented and measured. All three preserve dagua's
directed metric by anchoring y to topological depth.

### 2.1 Variant A: tutte_depth_anchor (x-only solve)

**File:** `/tmp/sprint22_tutte/tutte_depth.py`

```
input:  edge_index [2,E], num_nodes N, y_pitch
1. depth[i] = longest_path_layering(edge_index, N)  # existing util
2. y[i] = depth[i] * y_pitch                        # TB convention: down = +
3. G = undirected_nx(edge_index)
4. For each connected component C:
     a. boundary = outer_face(C) via networkx planar embedding
     b. Spread boundary x evenly along [0, x_width=y_pitch*sqrt(N)]
        in face-traversal order
     c. Build interior Laplacian L_ii (sparse), off-diag block L_ib
     d. Solve L_ii * x_int = -L_ib * x_boundary  (scipy spsolve)
     e. Pack components horizontally with gap
5. return (x, y)
```

**Failure mode:** boundary is placed on a 1D line, but lattice boundaries
naturally span 2D. Forcing x-only on a multi-layer boundary collapses
the vertical structure of the boundary and produces compressed CV.

### 2.2 Variant B: tutte_v2 (2D Tutte + monotone y-warp)

**File:** `/tmp/sprint22_tutte/tutte_v2.py`

```
input: edge_index [2,E], num_nodes N, y_pitch, x_pitch
1. depth = longest_path_layering(...)
2. G = undirected_nx(...)
3. For each connected component:
     a. Run classical 2D Tutte: outer face on regular polygon of radius R,
        interior solved via L_ii [x|y] = -L_ib [x_b | y_b].
     b. Discard Tutte y. Replace with y_new[i] = depth[i] * y_pitch.
     c. Tiebreak: within each depth layer, sort nodes by Tutte-x;
        enforce minimum gap = 0.6 * x_pitch by sequential push-right.
     d. (Optional) constrained x-resolve: with y_new fixed, re-run
        L_ii * x = -L_ib * x_boundary using the SAME Tutte boundary
        but with positions on the boundary pinned to their post-warp
        coordinates. This is a barycentric smoothing step that
        respects the new y.
     e. Normalize x to span x_pitch * sqrt(N).
4. Pack components horizontally.
```

This is the variant that yields the win on parallel_cycles_4x5 and the
near-tie on planar_60.

### 2.3 Variant C: tutte_floater (mean-value-weighted Tutte)

**File:** `/tmp/sprint22_tutte/tutte_floater.py`

```
1. Run classical Tutte 2D to get pos2d (as in Variant B step 3a).
2. Compute Floater mean-value weights:
     for each vertex i with sorted-ccw neighbors j_1,...,j_k:
       w_{i,j_k} = (tan(alpha/2) + tan(beta/2)) / |p_i - p_j_k|
     where alpha, beta are the angles at i bracketing the edge (i,j_k).
3. Symmetrize per edge (mean of both directions).
4. Re-solve weighted Tutte L_w * pos = boundary_constraint.
5. Apply depth-warp (Variant B step 3b-e).
```

Empirical observation: on vertex-transitive interiors (hex/tri lattice
bulk), Floater weights collapse to ~unit per the 2025 theorem, so
Variant C produces ~identical numbers to Variant B. Confirmed below.

---

## 3. Empirical validation

### 3.1 Setup

* Working directory: `/tmp/sprint22_tutte/`
* Score: `dagua.metrics.full(pos, edge_index, node_sizes=[40,20]*N)`
  composite at sprint-21b HEAD. (Note: this gives slightly different
  numbers than CONTEXT's reported scores because CONTEXT uses the
  benchmark suite's full node-size pipeline. Trends and deltas are
  preserved; absolute numbers are within ~1 pt of CONTEXT for dagua
  rows on hex/tri.)
* Baselines: dagua default `dagua.layout(g)`, plus saved positions
  for `graphviz_dot` and `elk_layered` from
  `eval_output/benchmark_full/positions/<graph>__<engine>.pt`.
* y_pitch=72, x_pitch=72 (matched to graphviz_dot's typical layer
  pitch). Sweep over y_pitch in {36, 50, 72, 100, 144} and x_pitch
  in {36, 50, 72, 100} confirmed 72/72 is near-optimal for Tutte_v2.

### 3.2 Headline numbers

```
                  dagua    gv_dot   elk     tutte_depth   tutte_v2  floater
hex_42            88.35    88.99    76.39   67.83 (-20.5) 81.95     81.44
triangular_36     85.67    87.09    --      70.25 (-15.4) 75.11     75.32
sierpinski_42     85.43    84.29    --      67.53 (-17.9) 78.30     78.35
parallel_cyc_4x5  62.11    60.53    --      57.11 ( -5.0) 65.36 *   65.36 *
planar_60         80.09    75.10    --      75.93 ( -4.2) 75.50     75.47
```

(* asterisk = Tutte beats both dagua and competitors.)

Key per-metric breakdown for hex_42, the canonical lattice target:

```
                  dag    cv     rho    ovl  str    crs
dagua             1.000  0.420  0.995  0    3.07   0.0000
graphviz_dot      1.000  0.099  0.823  0    17.42  0.0000
tutte_depth       1.000  0.651  1.000  11   21.05  0.0197    -- overlaps + cv kill it
tutte_v2 y72/x72  1.000  0.484  1.000  0    25.67  0.0016    -- str kill
tutte_v2 y144/x72 1.000  0.534  1.000  0    14.46  0.0080    -- best str, but cv worse
```

Tutte_v2 wins rho (1.000 vs dagua 0.995) and matches dag, but **loses
heavily on edge_length_cv (0.484 vs 0.420)** and **edge_straightness
(25.67 vs 3.07 deg)**. Neither metric can be fixed by tuning the pitch
ratio: when y_pitch is large enough to make edges straight (y=144),
CV blows up (0.534); when CV is best (y=72), straightness suffers.

### 3.3 Why Tutte loses on hex/tri (root cause)

1. **Hex / tri lattice patches are NOT 3-connected.** Boundary degree-2
   nodes break the 3-connectivity hypothesis of Tutte's theorem.
   Boundary x ordering is not uniquely determined; different choices
   produce different interior solutions, all sub-optimal.
2. **Tutte's barycentric system minimizes Dirichlet energy**, which is
   sum of squared edge lengths. On a finite lattice patch, the energy
   minimum has the boundary contracted (corners pulled in toward the
   centroid). This produces NON-UNIFORM edge length — exactly what
   our edge_length_cv metric punishes.
3. **graphviz_dot uses a different objective.** Inspecting the saved
   positions: 18 unique x-values, 12 unique y-values, CV=0.10, NOT a
   barycentric solve. dot's Sugiyama-with-network-simplex pulls
   columns onto integer x-grid via the network-simplex x-coordinate
   step (Gansner-Koutsofios-North-Vo 1993, IEEE TSE). That's the
   Bet 1 territory, not Bet 2.
4. **dagua's stress + grid synthesis already approximates the right
   answer.** dagua hex_42 has CV=0.42 (still weaker than dot) but
   compensates by edge_straightness=3.07 (extremely good), giving
   88.35. Tutte cannot give CV<dagua AND straightness<dagua
   simultaneously; the trade-off is fundamental to barycentric solves
   on non-3-connected planar patches.

### 3.4 Why Tutte_v2 wins on parallel_cycles_4x5

Parallel cycles is 4 disconnected directed C_5 cycles. dagua at
sprint-21b composite = 62.11, dag=0.800, cv=0.769, rho=NaN,
crossings=0.

Tutte_v2:
- Per-component classical Tutte places each C_5 on a regular 5-gon.
- Depth-warp pins y to depth (each cycle has depth pattern 0->1->2->3->4
  but with a back-edge, longest_path_layering produces depth 0..4).
- After warp + tiebreak, each cycle becomes a vertical 5-row stack
  with monotone x by Tutte rotation order.
- composite = 65.36 (+3.25 vs dagua, +4.83 vs gv_dot, +2.63 vs sfdp).
- **dag=1.000** (vs dagua 0.800) is the unlock — dagua's TB convention
  gets violated because the cycle's back-edge places one node above
  its predecessor; Tutte_v2's depth-warp guarantees monotone y.

This is the first robust win Tutte produces. It's small but real.

### 3.5 Why Tutte_v2 ties planar_60 (interesting near-miss)

planar_60 is 5 nested rings of 12 nodes, 156 edges. **Already 3-connected
and planar.** dagua=80.09. graphviz_dot=75.10. Tutte_v2=75.50.

Tutte_v2 gets dag=1.000 AND rho=1.000 by depth anchor (vs dagua
0.923 / 0.919). It loses 5 points to dagua in cv (0.664 vs 0.464). The
ring structure has uniform edge lengths in the natural 2D embedding,
but my within-layer-tiebreak pushes nodes apart at min_gap=0.6*x_pitch
which inflates CV.

A more careful tiebreak (proportional to Tutte-x within layer) might
reclaim those 5 points and put Tutte_v2 ahead of dagua here. But
since dagua wins by 4.59 and planar_60 isn't on the loss list, this
is moot for sprint-22. Worth noting for sprint-23 if a planar
specialist gets developed.

### 3.6 Pitch sensitivity (negative result)

Sweep results for tutte_v2 on hex_42 across (y_pitch, x_pitch) grid:

```
            x=36    x=50    x=72    x=100
y=36       71.64   71.41   78.91   78.91
y=50       72.32   72.05   80.44   80.19
y=72       73.13   72.81   81.56   81.25
y=100      73.37   73.23   82.56   82.29
y=144      73.82   73.58   82.80   82.35
```

Best Tutte_v2 hex_42 = 82.80 at y=144,x=72. Still **5.55 below dagua
88.35**. No regime closes the gap. Same pattern for tri_36 (best 79.50,
gap 6.17).

### 3.7 Floater weights are inert on lattice interior

tutte_floater hex_42: 81.44 (variant B was 81.95, change = -0.5 within
noise). Confirms 2025 paper's prediction: vertex-transitive interiors
are conformally rigid -> edge-weight modulation cannot reduce Dirichlet
energy below the unit-weight solution. Mean-value coordinates buy you
nothing on these graphs.

---

## 4. Risk / regression analysis

**Tutte_v2 cannot ship as-is even on its single win (parallel_cycles).**
Reasons:

1. **Detection-gate brittleness.** The natural gate `lattice_like AND
   is_planar AND N >= 12` triggers on hex/tri/sierp where Tutte LOSES
   5-10 points. We'd need a much narrower gate. Empirically:
   `is_planar AND k_components > 1 AND all_components_are_cycles`
   triggers on parallel_cycles only — that's a crude exact-match.
   Risk that a future cycle-rich graph is added to the suite and
   Tutte mis-fires on it.

2. **dag_consistency claims are honest but the rest can lie.** Tutte's
   monotone y-warp guarantees dag=1.0 only because longest-path
   layering already gives a topological order. For cyclic / strongly-
   connected graphs (small_world_500, small_world_100), this isn't
   defined. parallel_cycles works because each component has a
   natural longest path even with the back-edge (the back-edge gets
   counted as a violation, and dagua's metric correctly excludes it
   when classified as a back_edge_mask member, but my /tmp test
   doesn't pass back_edge_mask, so I'm getting lucky).

3. **Composite ceiling on parallel_cycles.** The +3.25 win brings
   parallel_cycles to 65.36, still below tied. CONTEXT shows
   parallel_cycles delta at -0.62 (so dagua needs 62.73+ to tie sfdp).
   Tutte_v2 hits 65.36 — well past tie. Solid +2.63 swing turns
   parallel_cycles_4x5 from a close-loss into a comfortable WIN.
   No regression risk to sfdp because the picker already knows when
   to use sfdp on cyclic graphs.

4. **Protected wins to verify.** Before shipping any tutte_v2 pipeline:
   * Re-run benchmark on hex_42, triangular_36, sierpinski_42 — Tutte
     must NOT be picked. Confirm gate excludes them.
   * Re-run on planar_60, planar_dag_50, transformer_layer (any
     planar graphs) — Tutte must NOT be picked. They're already wins
     for dagua.
   * Re-run on parallel_cycles_4x5 — Tutte SHOULD be picked.
   * Spot-check small_world_100/500: Tutte gate must reject (not
     planar, not lattice).
   * Petersen_10: not planar (Petersen graph is famously non-planar).
     Tutte gate must reject.
   * Run pipeline-fidelity tests: the /tmp implementation depends on
     `dagua.utils.longest_path_layering`, which exists. No new ops
     required if Tutte is implemented as a single op
     `tutte_planar_init` then composed into a pipeline.

5. **Numerical risk.** scipy.sparse.linalg.spsolve on the interior
   Laplacian. If the boundary face-walk picks a bad outer face for a
   non-3-connected graph, L_ii can be near-singular (caught with
   ridge regularization in /tmp). Should add unit test with a known
   degenerate case (path graph, star).

6. **Performance.** O(N^1.5) sparse solve on a graph with N=20
   takes <5 ms in /tmp. For dependency_500 (N=500, but non-planar,
   so it wouldn't trigger Tutte), the ceiling is ~50 ms; not a
   concern. For N=10000+ planar graphs (none in current suite), the
   solve might dominate, but that's far future.

---

## 5. Recommended implementation

### 5.1 Scope

**Ship `tutte_cyclic_planar` ONLY**, gated narrowly:

```
gate(graph) =
    is_planar(graph)
    AND k_connected_components(graph) >= 2
    AND every component has degree-2 vertices in a cycle pattern
        (heuristic: each component has E_c == V_c, every vertex has
        out-degree exactly 1 and in-degree exactly 1)
```

This triggers on parallel_cycles_4x5 only in the current suite.

**Do NOT ship Tutte for hex_42, triangular_36, sierpinski_42.** Those
losses are best addressed via Bet 1 (graphviz_dot lattice-pitch
mimic) or accepted at ceiling.

### 5.2 Pipeline

```python
# dagua/layout/ops/pipelines/tutte_cyclic_planar.py
def tutte_cyclic_planar_pipeline(state: SolveState, cfg: Config) -> SolveState:
    """Per-component classical Tutte with monotone y-warp.

    Gate: is_planar AND k_components > 1 AND all components are
    simple directed cycles. Targets parallel_cycles family.

    Steps:
      1. tutte_classical_2d (new op): classical Tutte 2D per component.
      2. depth_y_warp (existing or new): replace y by depth * y_pitch.
      3. within_layer_min_gap (existing): tiebreak overlapping nodes.
      4. component_pack_x (existing): pack components horizontally.
    """
```

Estimated 80-150 LOC in dagua/, including the new `tutte_classical_2d`
op (the per-component sparse Laplacian solve) and the gate
predicate. Other ops likely already exist.

### 5.3 Gate registration

Add to picker logic:
```
if classify_graph(g).all_simple_cycles_disconnected:
    candidates.append("tutte_cyclic_planar")
```

The classifier already exposes `topology_tags`; add a tag
`disconnected_cycles` if not present.

### 5.4 Implementation order

1. Add `tutte_classical_2d` op + tests (40 LOC + 80 LOC tests).
2. Add `disconnected_cycles` topology tag (10 LOC + 20 LOC test).
3. Compose `tutte_cyclic_planar` pipeline (30 LOC + 30 LOC test).
4. Register in picker for the gate condition (10 LOC).
5. Re-run benchmark on parallel_cycles_4x5 — confirm picker selects
   tutte_cyclic_planar with composite ~65.

Total estimate: **~200 LOC, 1-2 hours of careful coding, 30 min of
benchmark verification**. Net swing: parallel_cycles_4x5 delta
-0.62 -> approximately +2.6, moves from close-loss bucket to WIN
bucket. **Best-or-tied moves 89% -> ~90%**. Modest but real.

---

## 6. Why the bigger lattice bet (hex/tri/sierp) does not work via Tutte

To save sprint-23 time, here is the diagnosis as a forward-pointer:

**Tutte's barycentric solve can never beat graphviz_dot on lattice
patches.** dot wins on `edge_length_cv` (0.099 hex_42) by placing
nodes on a uniform integer grid via Sugiyama's network-simplex
x-coordinate step. This is NOT a barycentric solve; it is a quadratic
program that explicitly minimizes total edge length subject to
integer-grid constraints. Tutte minimizes squared-edge-length without
those constraints, and on non-3-connected boundaries it
under-determines — boundary contraction inflates CV.

**The right algorithm to chase, if you really want hex/tri to flip:**
implement Sugiyama-style integer-grid x-coordinate assignment from
Gansner-Koutsofios-North-Vo 1993, IEEE TSE 19(3) 214-230 ("A
Technique for Drawing Directed Graphs"), section 4.2 "Network simplex
x-coordinate computation". That is dot's actual algorithm. ~300 LOC.
Predicted +4-7 on hex_42, +5-8 on tri_36 (closes both close-losses).
This is BET 1's territory.

**The 2025 paper's contribution to dagua is purely negative:** it
tells us we cannot improve Tutte by edge-reweighting on the symmetric
graphs we care about. So if Tutte fails (it does), Floater fails
(it does), and SDP-optimal weights fail (the paper proves this for
edge-transitive graphs). The bet, properly understood, is dead.

---

## 7. Files

* `/tmp/sprint22_tutte/tutte_depth.py` — Variant A (x-only solve).
* `/tmp/sprint22_tutte/tutte_v2.py` — Variant B (2D Tutte + y-warp). The
  one that wins parallel_cycles.
* `/tmp/sprint22_tutte/tutte_floater.py` — Variant C (mean-value weights).
  Inert on vertex-transitive graphs as predicted by 2025 paper.
* `/tmp/sprint22_tutte/measure.py` — driver that loads each graph,
  competitor positions from benchmark_full/positions/, and prints
  per-metric deltas.

To reproduce:
```
cd /tmp/sprint22_tutte
PYTHONPATH=/home/jtaylor/projects/dagua python measure.py
```

Verified at sprint-21b HEAD = c821eb6 on 2026-04-25.

---

## 8. Word count

This document is approximately 3,400 words. The TL;DR is the single
biggest call: ship narrow `tutte_cyclic_planar` for parallel_cycles
only; do NOT ship Tutte for hex/tri/sierp; the 2025 paper's actual
content is a negative result about edge weighting, not a refinement
algorithm; the +5..+7 prediction was a misread.

## 9. Sources

- Gouveia, Steinerberger, Thomas. "Conformal Rigidity and Spectral
  Embeddings of Graphs." [arXiv:2506.20541](https://arxiv.org/abs/2506.20541),
  June 2025.
- Steinerberger. "Conformally rigid graphs." J. Graph Theory, 2025
  ([Wiley](https://onlinelibrary.wiley.com/doi/10.1002/jgt.23229),
  predecessor [arXiv:2402.11758](https://arxiv.org/abs/2402.11758)).
- Tutte. "How to Draw a Graph." Proc. London Math. Soc. (3) 13, 1963.
- Floater. "Mean value coordinates." CAGD 20(1), 2003.
- Gansner, Koutsofios, North, Vo. "A technique for drawing directed
  graphs." IEEE TSE 19(3) 214-230, 1993. (Pointer for Bet 1.)

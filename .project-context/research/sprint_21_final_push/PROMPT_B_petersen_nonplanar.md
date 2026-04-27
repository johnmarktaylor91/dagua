# Area B — Non-planar regular graphs (petersen_10)

## Question

petersen_10 dagua=74.64 vs igraph_sugiyama=77.36, delta=-2.72. After
sprint-20l aggressive polish variants the gap closed from -5.07 to
-2.72, but the algorithm's ceiling is reached. The Petersen graph is
the canonical non-planar 3-regular graph (10 nodes, 15 edges, no
planar embedding). What algorithm closes the remaining gap?

## Specific evidence

igraph_sugiyama achieves 77.36 on petersen by:
- Cycle removal -> layered DAG -> per-layer crossing min -> compact
- Score breakdown vs dagua TBD (run /tmp/score_breakdown.py for petersen)

dagua's full pipeline sweep on petersen:
- force=tree:           63.57
- force=layered_dag:    70.69 (= force=hybrid = force=legacy_monolith)
- force=force_directed: 28.86 (FD pipeline broken on small graphs)
- force=planar:         FAIL (graph IS non-planar)
- auto (with polish):   74.64

multi_start_k=20 → identical output. Dispatcher saturated.

## Research targets

1. **Why does igraph_sugiyama win on petersen specifically?** It's a
   barycenter/median Sugiyama implementation. Their secret has to be
   the layer assignment (which 5+5 split) and the per-layer x ordering.
   Read igraph's sugiyama source if findable, or the underlying
   Eades/Sugiyama 1990 paper for the canonical algorithm.

2. **Could a small N=10 brute-force enumeration help?** With 10!/2 =
   ~1.8M layer permutations, a deterministic best-of-permutation
   solver is feasible at small N. Gate to N <= 12 and dispatch only
   when dagua's score < 0.95 * best competitor.

3. **Stress-majorization with carefully chosen pivots**: petersen
   has high symmetry (automorphism group of order 120). A stress
   solver that respects symmetry (initialize by spectral coords on
   the dominant eigenvectors) might match sugiyama's quality.

4. **Gomory-Hu / spectral layout for highly symmetric graphs**: the
   Petersen graph has eigenvalues {3, 1^5, -2^4}. The spectral
   embedding using the second/third smallest eigenvectors of the
   Laplacian gives a natural circular embedding. Worth measuring.

5. **What other graphs would benefit from a non-planar-regular
   detector?** ~5 graphs in benchmark have similar structure. List
   them with their current dagua deltas.

## Output format

`.project-context/research/sprint_21_final_push/B_petersen_nonplanar__<your_agent_name>.md`

Include:
- TL;DR (4-6 bullets)
- Recommended algorithm (single best, plus 1-2 fallbacks)
- Detection gate
- Expected delta (quantified)
- Generality: which other benchmark graphs would benefit
- Risk: where this could regress

## Constraints

- READ-ONLY. Do NOT write code or commit. Findings file only.
- Read `.project-context/research/sprint_21_final_push/CONTEXT.md` first.
- Budget: 1500-2500 words.

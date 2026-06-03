Close drl + davidson_harel diverging cases (igraph anneal). Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md. You OWN
dagua/layout/ops/_igraph_rng.py, dagua/layout/ops/pipelines/drl.py, davidson_harel.py. drl ~1.0 but
35/42 fixtures bit-exact; davidson_harel ~0.36. The RNG is MOSTLY matched -- the diverging cases are
a specific RNG-draw-order or arithmetic-order mismatch (or genuine chaotic-anneal basin divergence).
For the failing fixtures (drl: the 7 non-exact; davidson: path8/grid3x3/complete5 seeds), trace
igraph's exact draw sequence (igraph C source layout_drl / layout_davidson_harel) step-by-step vs
dagua and find/fix the first divergence. NO replay/delegation. Verify check_engine.py
classic_drl_default + classic_davidson_harel_rounds100. Drive <1e-7; if a case is genuine chaotic
basin divergence (anneal legitimately lands in a different valid minimum from a 1-ULP early diff),
document it precisely as irreducible.

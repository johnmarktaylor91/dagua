Close sugiyama to bit-exact (deterministic tie-break, tractable). Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md. Pipeline:
dagua/layout/ops/pipelines/sugiyama.py (do NOT touch _igraph_rng.py; sugiyama is deterministic).
Current 0.93 on SYMMETRIC graphs (complete5/petersen/wheel/two_triangles). The gap is igraph's
within-layer ORDERING tie-break + coordinate assignment: on symmetric graphs igraph picks a specific
valid ordering; dagua picks another. Read igraph's layout_sugiyama / layered C source for the EXACT
tie-break comparator + node iteration order + barycenter/median rounding, and replicate it. Verify
check_engine.py classic_sugiyama_default (+ wide/tight/passes4/passes48). Drive <1e-7; if it is a
genuine equally-valid-ordering ambiguity that cannot be pinned, document precisely.

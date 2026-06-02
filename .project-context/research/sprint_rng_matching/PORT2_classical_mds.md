PORT classical_mds to bit-exact. Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md. Pipeline: dagua/layout/ops/pipelines/classical_mds.py.
At 1.09. Reference: igraph_mds. classical MDS is DETERMINISTIC (no RNG) -- this is a pure
arithmetic/eigensolver-order match: distance matrix -> double-centering -> eigendecomposition
(match igraph's eigensolver: LAPACK routine, eigenvector sign/order conventions) -> scaling.
Read igraph's layout_mds C source. Do NOT touch _igraph_rng.py (deterministic, no RNG).
Verify check_engine.py classic_classical_mds_default. Drive <1e-7 (deterministic -> ~1e-13).

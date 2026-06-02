FINAL attempt: classical_mds bit-exact. Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md. Pipeline: dagua/layout/ops/pipelines/classical_mds.py.
Current 0.77; bit-exact on non-degenerate fixtures but STUCK on igraph's vendored-LAPACK dsyevr eigenvector
basis for REPEATED eigenvalues (degenerate-eigenvector convention). Try to match igraph's LAPACK dsyevr/MRRR
eigenvector selection for degenerate eigenvalues (read igraph's vendored LAPACK + how it orders/signs
degenerate eigenvectors). DETERMINISTIC, no RNG. If this LAPACK degenerate-basis convention genuinely cannot
be reproduced in pure python/torch, DOCUMENT it precisely as the irreducible wall and STOP (don't flail).
Verify check_engine.py classic_classical_mds_default.

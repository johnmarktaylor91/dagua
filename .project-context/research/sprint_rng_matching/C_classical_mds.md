Cheap attempt: classical_mds via scipy LAPACK. Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md. Pipeline:
dagua/layout/ops/pipelines/classical_mds.py. Current 0.77 (DEGENERATE eigenvalue basis differs from
igraph's vendored LAPACK 3.4.2 dsyevr). TRY: call scipy.linalg.lapack.dsyevr (a common Python lib --
allowed, it is a numerical primitive not delegation) with igraph's EXACT params (range='I', uplo='U',
abstol=1e-14, the same eigenvalue index range) to see if scipy's LAPACK reproduces igraph's
degenerate-eigenvector basis. If it MATCHES -> bit-exact, done. If scipy's LAPACK != igraph's
vendored 3.4.2 (likely), STOP and document: the degenerate-subspace basis is implementation-specific;
note that dagua's output is GEOMETRICALLY EQUIVALENT (a rotation within the degenerate eigenspace) --
a metric artifact, not a fidelity error. Do NOT port LAPACK from scratch (out of scope). Verify
check_engine.py classic_classical_mds_default. NO delegation to igraph.

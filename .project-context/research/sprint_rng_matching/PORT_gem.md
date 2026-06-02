Port dagua's GEM fidelity path to bit-match OGDF gem. Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md FULLY first.
- Pipeline: dagua/layout/ops/pipelines/gem.py + dagua/layout/ops/gem.py (has _glibc_rand_values --
  prior init port; extend to full match).
- Reference: ogdf_gem (OGDF GEMLayout). OGDF uses glibc rand() for init + node permutation.
  Prior work aligned INIT; the residual is post-init (node-permutation RNG, packing). Match the
  full RNG draw order + algorithm from OGDF source.
- Baseline max RMSD ~1.15 (gem_iters100). Verify: python scripts/rng_match/check_engine.py
  classic_gem_iters100 (also iters500/2000). Drive < 1e-7, ideally ~1e-15. NO delegation/replay.

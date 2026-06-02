Port dagua's UMAP fidelity path to bit-match umap-learn. Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md FULLY first.
- Pipeline: dagua/layout/ops/pipelines/umap_layout.py.
- Reference: umap_graph (umap-learn, installed -- read its Python/numba source). umap uses numpy
  RandomState for init + the optimize_layout SGD with a sampled negative-edge schedule (numba).
  Match numpy RandomState seeding + the exact draw sequence (spectral/random init, then the
  epoch/negative-sampling RNG). NO delegation/replay.
- Baseline max RMSD ~1.12 (umap_default). Verify: python scripts/rng_match/check_engine.py
  classic_umap_default (also the umap_* variants). Drive < 1e-7, ideally ~1e-15.

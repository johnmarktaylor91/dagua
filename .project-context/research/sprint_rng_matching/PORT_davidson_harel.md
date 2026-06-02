Port dagua's DAVIDSON_HAREL fidelity path to bit-match igraph davidson_harel. Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md FULLY first.
- Pipeline: dagua/layout/ops/pipelines/davidson_harel.py (+ _igraph_rng.py).
- Reference: igraph_davidson_harel. Davidson-Harel is simulated annealing -> heavy RNG use
  (random moves, acceptance). Match igraph's RNG generator + EXACT draw sequence (move proposals,
  acceptance draws) from igraph source.
- Baseline max RMSD ~0.36 (CLOSEST divergent -- good proof case). Verify:
  python scripts/rng_match/check_engine.py classic_davidson_harel_rounds100 (also rounds50/200).
  Drive < 1e-7, ideally ~1e-15. NO delegation/replay.

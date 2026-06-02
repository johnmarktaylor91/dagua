Port dagua's DRL fidelity path to bit-match igraph drl. Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md FULLY first.
- Pipeline: dagua/layout/ops/pipelines/drl.py (+ dagua/layout/ops/_igraph_rng.py exists -- reuse/extend).
- Reference: igraph_drl (python-igraph). Read igraph's DrL source + its RNG (igraph uses its own
  RNG; see _igraph_rng.py for the existing port). Reimplement generator + draw order.
- Baseline max RMSD ~1.0 (drl_default). Verify: python scripts/rng_match/check_engine.py classic_drl_default
  (also drl_coarsen/coarsest/refine/final). Drive < 1e-7, ideally ~1e-15. NO delegation/replay.

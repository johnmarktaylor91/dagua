Port dagua's SFDP fidelity path to bit-match graphviz 7.0.5 sfdp. Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md FULLY first.
- Pipeline: dagua/layout/ops/pipelines/sfdp.py (it already has GraphvizRandom/gv_random -- make
  it actually match).
- Reference: graphviz_sfdp (-Gstart=seed). Graphviz 7.0.5 source ~/tools/graphviz-7.0.5-src/
  (sfdp = lib/sfdpgen, spring_electrical.c). Use instrumented build + GV_TRACE=1 to trace.
- RNG: graphviz sfdp multilevel init RNG. Reimplement generator + draw order from source.
- Baseline max RMSD ~1.07. Verify: python scripts/rng_match/check_engine.py classic_sfdp_default
  (also check the other classic_sfdp_* variants). Drive < 1e-7, ideally ~1e-15.

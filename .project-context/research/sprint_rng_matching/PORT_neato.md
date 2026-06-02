Port dagua's NEATO fidelity path to bit-match graphviz 7.0.5 neato. Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md FULLY first.
- Pipeline: dagua/layout/ops/pipelines/neato.py (+ any shared graphviz RNG helper it uses).
- Reference: graphviz_neato (adapter dagua/eval/competitors/graphviz_competitor.py runs
  graphviz with -Gstart=seed -> random init). Graphviz 7.0.5 source: ~/tools/graphviz-7.0.5-src/
  (neato = lib/neatogen). Use instrumented build ~/tools/graphviz-7.0.5-instr/bin/dot with
  GV_TRACE=1 GV_TRACE_FILE=/tmp/neato_trace.txt to dump internal per-iteration state, trace
  step-by-step.
- RNG: graphviz's init RNG (drand48-family) driven by -Gstart=seed. Reimplement the generator+
  draw order from source. NO delegation, NO replay (see protocol).
- Baseline max RMSD ~1.23. Verify: python scripts/rng_match/check_engine.py classic_neato
- Drive < 1e-7, ideally ~1e-15.

Finish sfdp to bit-exact. Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md. Pipeline: dagua/layout/ops/pipelines/sfdp.py + dagua/layout/ops/sfdp.py.
Current 0.44 (multilevel spring-electrical, partial). Use instrumented build ~/tools/graphviz-7.0.5-instr/
bin/dot (GV_TRACE=1) + graphviz src ~/tools/graphviz-7.0.5-src/ (lib/sfdpgen) to trace coarsening levels +
per-level init RNG + force iterations, match step-by-step. Verify check_engine.py classic_sfdp_default.
Drive <1e-7; document the first irreducible divergence if stuck.

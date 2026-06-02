ITERATE neato to bit-exact. Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md. Pipeline: dagua/layout/ops/pipelines/neato.py.
neato is at 7.3e-3 (matched params/seed). RNG init already matched (srand48/drand48). Residual:
graphviz stops at CONVERGENCE (e.g. ~152 iters on path8 seed3) but dagua runs full 200. Match
graphviz neato's convergence/termination test (lib/neatogen/stress.c convergence check) + final
iteration count exactly, using the instrumented build (~/tools/graphviz-7.0.5-instr/bin/dot,
GV_TRACE=1) to find the exact stop. Verify: python scripts/rng_match/check_engine.py classic_neato. Drive <1e-7 (~1e-13).

FIX sgd2_multi UNAVAILABLE (8 variants). Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md for context. s_gd2 IS installed and the
`sgd2` competitor works, but `sgd2_multi_ref.available()` returns False. Diagnose WHY (read
dagua/eval/competitors/sgd2_multi_competitor.py available() + its deps) and FIX so it RUNS on
small graphs (install any missing user-space dep, or fix the availability gate). Goal: make it
produce real comparisons via check_engine.py classic_sgd2_multi_default. You may edit
dagua/eval/competitors/sgd2_multi_competitor.py (NOT variants.py, NOT dagua/layout/ops/). Report
the root cause + fix + the resulting RMSD. (Bit-exact port can follow; first make it RUN.)

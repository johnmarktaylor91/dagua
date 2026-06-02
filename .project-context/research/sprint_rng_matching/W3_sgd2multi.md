Port sgd2_multi to bit-exact (it now RUNS after the dep-fix). Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md. Pipeline:
dagua/layout/ops/pipelines/sgd2_multi.py. Reference: sgd2_multi_ref (uses s_gd2 / S_GD2 stochastic
SGD). Current: batch128 CLOSE 9.86e-6, lr001 0.02, others 0.13-1.18. Match s_gd2's RNG (it shuffles
term order each epoch via a PRNG) + the SGD schedule + arithmetic order from s_gd2 source. NO delegation/
replay. Verify python scripts/rng_match/check_engine.py classic_sgd2_multi_default (+ batch128/lr001/etc).
Drive <1e-7.
